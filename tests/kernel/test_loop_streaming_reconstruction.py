# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Streaming reconstruction: the final streamed response reuses the loop's assembled messages."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.kernel import tool
from chrys.kernel.loop import (
    ToolLoopLayer,
)
from chrys.kernel.middleware import (
    ChatMiddlewareLayer,
)
from chrys.kernel.types import (
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)
from tests.kernel._fakes import (
    _call_contents,
    _call_update,
    _CarriageFunction,
    _make_tool,
    _provenance_tool,
    _result_contents,
    _ScriptedClient,
    _stack,
    _TerminateFunction,
    _text_update,
    _user,
)
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer

# ---------------------------------------------------------------------------
# Streaming reconstruction (single reconstruction authority)
# ---------------------------------------------------------------------------


class _StructuredAnswer(BaseModel):
    answer: int


class TestStreamingReconstruction:
    """The final streamed response reuses the loop's assembled messages, never a re-merge."""

    @pytest.mark.asyncio
    async def test_streamed_double_echo_of_landed_call_does_not_reexecute(self) -> None:
        """Assembly must not launder an echoed call's identity past the memo.

        Consecutive same-call fragments merge into a NEW Content object, so
        an already-landed call object echoed twice adjacently would dodge the
        identity memo. Echo fragments are stripped from assembly input, so
        the merge copy never exists and nothing re-executes.
        """
        events: list[str] = []
        c1_obj = Content.from_function_call("c1", "echo", arguments={"text": "once"})
        layer, _wire = _stack(
            [
                [ChatResponseUpdate(contents=[c1_obj], role="assistant")],
                [
                    ChatResponseUpdate(contents=[c1_obj], role="assistant"),
                    ChatResponseUpdate(contents=[c1_obj], role="assistant"),
                ],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:once"], "the landed call executes exactly once"
        assert [c.call_id for c in _call_contents(final)] == ["c1"]
        assert [r.call_id for r in _result_contents(final)] == ["c1"]

    @pytest.mark.asyncio
    async def test_streamed_same_object_twice_nonadjacent_executes_once(self) -> None:
        """A non-adjacent within-response duplicate collapses to one copy too.

        With text between the two emissions there is no self-merge hazard;
        assembly-pass dedup (and, failing that, landing) must still keep a
        single copy — parity with the blocking path.
        """
        events: list[str] = []
        call = Content.from_function_call(call_id="c1", name="echo", arguments='{"text": "once"}')
        layer, _wire = _stack(
            [
                [
                    ChatResponseUpdate(contents=[call], role="assistant"),
                    ChatResponseUpdate(contents=[Content.from_text("thinking")], role="assistant"),
                    ChatResponseUpdate(contents=[call], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:once"]
        assert [c.call_id for c in _call_contents(final)] == ["c1"]
        assert [r.call_id for r in _result_contents(final)] == ["c1"]

    @pytest.mark.asyncio
    async def test_streamed_fragment_replay_after_merge_does_not_reexecute(self) -> None:
        """Raw stream fragments are conversation history too.

        A multi-fragment call assembles into a NEW merged Content, so only
        the merged copy reaches the transcript — if the memo held nothing
        else, the raw fragment objects would stay unknown and a stateful
        client replaying its own buffered fragments next turn would
        re-execute the call. Accepted update-content identities are
        recorded after landing, so a replayed fragment strips as an echo
        while genuinely fresh work still lands.
        """
        events: list[str] = []
        frag_a = Content.from_function_call(call_id="c1", name="echo", arguments='{"text": "on')
        frag_b = Content.from_function_call(call_id="c1", name="echo", arguments='ce"}')
        fresh = Content.from_function_call("c2", "echo", arguments={"text": "fresh"})
        layer, _wire = _stack(
            [
                [
                    ChatResponseUpdate(contents=[frag_a], role="assistant"),
                    ChatResponseUpdate(contents=[frag_b], role="assistant"),
                ],
                [
                    ChatResponseUpdate(contents=[frag_a], role="assistant"),
                    ChatResponseUpdate(contents=[frag_b], role="assistant"),
                    ChatResponseUpdate(contents=[fresh], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:once", "tool:echo:fresh"], "replayed fragments must not re-merge and re-execute"
        assert [c.call_id for c in _call_contents(final)] == ["c1", "c2"]
        assert [r.call_id for r in _result_contents(final)] == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_echo_stripping_survives_response_validation_proxy(self) -> None:
        """The strip filter must reach BENEATH a draining stream proxy.

        ``ResponseValidationMiddleware`` returns its own ``ResponseStream``
        whose generator drains the provider stream, preview-finalizes, and
        replays updates — ``with_update_filter`` push-down cannot cross that
        semantic boundary (the proxy's source is a plain generator), so the
        filter is delivered on the request path and the middleware pipeline
        attaches it to the stream its final handler resolves. Without that,
        the proxy's cached final response is assembled UNFILTERED and the
        laundering P1 returns in the production topology.
        """
        from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware

        events: list[str] = []
        landed = Content.from_function_call("c1", "echo", arguments={"text": "old"})
        fresh_same_id = Content.from_function_call("c1", "echo", arguments={"text": "new"})
        wire = _ScriptedClient(
            [
                [ChatResponseUpdate(contents=[landed], role="assistant")],
                [
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[Content.from_text("thinking")], role="assistant"),
                    ChatResponseUpdate(contents=[fresh_same_id], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )
        layer = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(wire, middleware=[ResponseValidationMiddleware(backoff_schedule=[0.0])])
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:old", "tool:echo:new"], "no re-execution, no swallowed fresh call"
        assert [c.arguments["text"] for c in _call_contents(final)] == ["old", "new"]

    @pytest.mark.asyncio
    async def test_echo_stripping_reattaches_on_validation_retry(self) -> None:
        """Every validation retry's fresh provider stream gets the filter.

        Attempt one of the second loop turn returns whitespace-only text
        (invalid, retried inside the validation proxy); the retry attempt
        carries the echo-laundering shape. The filter must cover the RETRY
        attempt's inner stream, not just the first.
        """
        from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware

        events: list[str] = []
        landed = Content.from_function_call("c1", "echo", arguments={"text": "old"})
        fresh_same_id = Content.from_function_call("c1", "echo", arguments={"text": "new"})
        wire = _ScriptedClient(
            [
                [ChatResponseUpdate(contents=[landed], role="assistant")],
                [_text_update("   \n  ")],
                [
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[Content.from_text("thinking")], role="assistant"),
                    ChatResponseUpdate(contents=[fresh_same_id], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )
        layer = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(wire, middleware=[ResponseValidationMiddleware(backoff_schedule=[0.0])])
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:old", "tool:echo:new"]
        assert [c.arguments["text"] for c in _call_contents(final)] == ["old", "new"]

    @pytest.mark.asyncio
    async def test_echo_stripping_drops_only_the_message_shell_it_emptied(self) -> None:
        """An echo in one message must not erase a different empty message.

        A genuine empty shell precedes an echo-only shell and fresh content.
        Streaming echo removal must discard only the middle shell; a
        call-wide "anything was stripped" flag would erase both empty
        messages.
        """
        historical = Content.from_text("historical")
        layer, _wire = _stack(
            [
                [
                    ChatResponseUpdate(contents=[], role="assistant", message_id="legitimate-empty"),
                    ChatResponseUpdate(contents=[historical], role="assistant", message_id="echo-only"),
                    ChatResponseUpdate(
                        contents=[Content.from_text("fresh")],
                        role="assistant",
                        message_id="fresh",
                    ),
                ]
            ]
        )

        stream = layer.get_response([Message("user", [historical])], stream=True)
        _ = [update async for update in stream]
        final = await stream.get_final_response()

        assert [(message.message_id, message.text) for message in final.messages] == [
            ("legitimate-empty", ""),
            ("fresh", "fresh"),
        ]

    @pytest.mark.asyncio
    async def test_echo_emptied_marker_isolated_from_validation_retry_message_id_collision(self) -> None:
        """A rejected attempt cannot mark a same-id accepted shell as echo.

        Providers may restart positional message IDs on every response.
        The rejected attempt strips an echo from ``msg_0``; the accepted
        attempt legitimately emits an empty ``msg_0`` tool message. Cleanup
        must follow accepted update identity, not an ID set shared across
        validation retries.
        """
        from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware

        historical = Content.from_text("historical")
        wire = _ScriptedClient(
            [
                [ChatResponseUpdate(contents=[historical], role="assistant", message_id="msg_0")],
                [
                    ChatResponseUpdate(contents=[], role="tool", message_id="msg_0"),
                    ChatResponseUpdate(
                        contents=[Content.from_text("fresh")],
                        role="assistant",
                        message_id="msg_1",
                    ),
                ],
            ]
        )
        layer = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(wire, middleware=[ResponseValidationMiddleware(backoff_schedule=[0.0])])
        )

        stream = layer.get_response([Message("user", [historical])], stream=True)
        _ = [update async for update in stream]
        final = await stream.get_final_response()

        assert len(wire.calls) == 2
        assert [(message.message_id, message.role, message.text) for message in final.messages] == [
            ("msg_0", "tool", ""),
            ("msg_1", "assistant", "fresh"),
        ]

    @pytest.mark.asyncio
    async def test_blocking_and_streaming_echo_cleanup_preserve_the_same_message_shape(self) -> None:
        """Blocking and streaming converge for empty, echo, fresh messages."""
        blocking_historical = Content.from_text("historical")
        blocking_layer, _wire = _stack(
            [
                ChatResponse(
                    messages=[
                        Message("assistant", [], message_id="legitimate-empty"),
                        Message("assistant", [blocking_historical], message_id="echo-only"),
                        Message("assistant", ["fresh"], message_id="fresh"),
                    ]
                )
            ]
        )
        blocking = await blocking_layer.get_response([Message("user", [blocking_historical])])

        streaming_historical = Content.from_text("historical")
        streaming_layer, _wire = _stack(
            [
                [
                    ChatResponseUpdate(contents=[], role="assistant", message_id="legitimate-empty"),
                    ChatResponseUpdate(contents=[streaming_historical], role="assistant", message_id="echo-only"),
                    ChatResponseUpdate(
                        contents=[Content.from_text("fresh")],
                        role="assistant",
                        message_id="fresh",
                    ),
                ]
            ]
        )
        stream = streaming_layer.get_response([Message("user", [streaming_historical])], stream=True)
        _ = [update async for update in stream]
        streaming = await stream.get_final_response()

        def shape(response: ChatResponse) -> list[tuple[str | None, str, str]]:
            return [(message.message_id, message.role, message.text) for message in response.messages]

        assert (
            shape(streaming)
            == shape(blocking)
            == [
                ("legitimate-empty", "assistant", ""),
                ("fresh", "assistant", "fresh"),
            ]
        )

    @pytest.mark.asyncio
    async def test_echo_stripping_does_not_resurrect_hook_removed_calls(self) -> None:
        """Echo handling must compose with result hooks, never race them.

        ``get_final_response`` runs the provider finalizer and result hooks;
        a hook deleting a call from the assembled response is a safety gate
        the loop promises runs before tool extraction. When the same turn
        also carries echo fragments, echo handling must not re-assemble the
        response behind the hook's back and resurrect the deleted call —
        echo fragments are stripped BEFORE assembly (update filter), so the
        hook operates on the one and only assembly.
        """
        events: list[str] = []
        landed = Content.from_function_call("c1", "echo", arguments={"text": "old"})

        def drop_danger(response: ChatResponse) -> ChatResponse:
            for message in response.messages:
                message.contents = [
                    c for c in message.contents if not (c.type == "function_call" and c.name == "danger")
                ]
            return response

        layer, _wire = _stack(
            [
                [ChatResponseUpdate(contents=[landed], role="assistant")],
                [
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(
                        contents=[Content.from_function_call("c9", "danger", arguments={"text": "boom"})],
                        role="assistant",
                    ),
                    ChatResponseUpdate(
                        contents=[Content.from_function_call("c2", "echo", arguments={"text": "new"})],
                        role="assistant",
                    ),
                ],
                [_text_update("done")],
            ],
            result_hook=drop_danger,
        )
        tools = [_make_tool(events), _make_tool(events, name="danger")]

        stream = layer.get_response([_user()], stream=True, options={"tools": tools})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:old", "tool:echo:new"], "hook-removed danger never runs; echo never re-runs"
        assert [c.call_id for c in _call_contents(final)] == ["c1", "c2"]
        assert [r.call_id for r in _result_contents(final)] == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_streamed_double_echo_merge_beside_same_id_fresh_call_executes_fresh_only(self) -> None:
        """A laundered echo merge must not shadow fresh same-id work.

        The poison combo: an already-landed call object echoed twice
        adjacently (assembly merges the fragments into a NEW object) AND a
        fresh call reusing the same call id later in the same stream turn.
        Response-level per-id provenance cannot express this — the fresh
        fragment would exonerate the id, letting the merged echo copy land,
        execute the OLD arguments again, and shadow the fresh call via
        duplicate-id dispatch filtering. Echo fragments must be removed from
        assembly input instead, so the merge copy never exists.
        """
        events: list[str] = []
        landed = Content.from_function_call("c1", "echo", arguments={"text": "old"})
        fresh_same_id = Content.from_function_call("c1", "echo", arguments={"text": "new"})
        layer, _wire = _stack(
            [
                [ChatResponseUpdate(contents=[landed], role="assistant")],
                [
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[Content.from_text("thinking")], role="assistant"),
                    ChatResponseUpdate(contents=[fresh_same_id], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:old", "tool:echo:new"], "old ran once in turn 1; the fresh call still runs"
        assert [c.arguments["text"] for c in _call_contents(final)] == ["old", "new"]

    @pytest.mark.asyncio
    async def test_streamed_fresh_same_id_call_beside_echo_stays_fresh_work(self) -> None:
        """Echo removal must not over-remove: id reuse by a fresh object stays fresh.

        One echoed landed object plus a DIFFERENT fresh call object reusing
        the same call id in the same stream turn: the echo is stripped from
        assembly input, and the fresh call must still land and execute —
        providers legitimately reuse ids across responses.
        """
        events: list[str] = []
        landed = Content.from_function_call("c1", "echo", arguments={"text": "old"})
        fresh_same_id = Content.from_function_call("c1", "echo", arguments={"text": "new"})
        layer, _wire = _stack(
            [
                [ChatResponseUpdate(contents=[landed], role="assistant")],
                [
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[Content.from_text("thinking")], role="assistant"),
                    ChatResponseUpdate(contents=[fresh_same_id], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:old", "tool:echo:new"], "fresh same-id work still executes"
        assert [c.arguments["text"] for c in _call_contents(final)] == ["old", "new"]
        assert all(not message._chrys_echo_content_stripped for message in final.messages)

    @pytest.mark.asyncio
    async def test_fragmented_call_keeps_stamps_and_metadata_through_final_response(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        frag1 = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c1", name="echo", arguments='{"text": "he')],
            role="assistant",
        )
        frag2 = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c1", name="echo", arguments='llo"}')],
            role="assistant",
        )
        tool_obj = _provenance_tool(static={"server_name": "static"})
        carrier = _CarriageFunction(result_metadata={"shell_exit_code": 0}, tool_context={"built": "filled"})
        layer, _wire = _stack([[frag1, frag2], [_text_update("done")]])

        stream = layer.get_response([_user()], stream=True, options={"tools": [tool_obj]}, middleware=[carrier])
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        calls = _call_contents(final)
        assert len(calls) == 1
        merged_call = calls[0]
        assert merged_call.parse_arguments() == {"text": "hello"}
        assert merged_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert merged_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        assert merged_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] == {
            "server_name": "static",
            "built": "filled",
        }
        result = _result_contents(final)[0]
        assert str(result.result) == "echo:hello"
        assert result.additional_properties[TOOL_RESULT_METADATA_KEY] == {"shell_exit_code": 0}

    @pytest.mark.asyncio
    async def test_final_response_snapshots_wrappers_and_reuses_content_objects(self) -> None:
        # The result hook runs before landing rebinds ``response.messages``,
        # so copying the list here captures the assembled pre-landing wrappers.
        assembled_wrappers: list[list[Message]] = []

        def capture(response: ChatResponse) -> ChatResponse:
            assembled_wrappers.append(list(response.messages))
            return response

        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [_text_update("final")]], result_hook=capture
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert len(assembled_wrappers) == 2
        # Landing retains loop-owned Message snapshots (a client mutating a
        # wrapper it returned must not rewrite landed history); the CONTENT
        # objects are reused by identity — the echo memo depends on it.
        assert final.messages[0] is not assembled_wrappers[0][0]
        assert final.messages[0].contents[0] is assembled_wrappers[0][0].contents[0]
        assert final.messages[-1] is not assembled_wrappers[1][-1]
        assert final.messages[-1].contents[0] is assembled_wrappers[1][-1].contents[0]
        assert [m.role for m in final.messages] == ["assistant", "tool", "assistant"]

    @pytest.mark.asyncio
    async def test_no_call_single_turn_snapshots_wrapper_and_reuses_content(self) -> None:
        assembled_wrappers: list[list[Message]] = []

        def capture(response: ChatResponse) -> ChatResponse:
            assembled_wrappers.append(list(response.messages))
            return response

        layer, _wire = _stack([[_text_update("hello")]], result_hook=capture)
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert final.messages[0] is not assembled_wrappers[0][0]
        assert final.messages[0].contents[0] is assembled_wrappers[0][0].contents[0]

    @pytest.mark.asyncio
    async def test_termination_exit_keeps_accumulated_transcript_with_stamps(self) -> None:
        layer, _wire = _stack([[_call_update("c1", "echo", {"text": "a"})], [_text_update("never")]])
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()]},
            middleware=[_TerminateFunction(result="stopped")],
        )
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        # Streaming termination keeps the accumulated transcript (deliberate
        # asymmetry with the non-streaming termination return).
        kinds = [item.type for msg in final.messages for item in msg.contents]
        assert "function_call" in kinds
        assert "function_result" in kinds
        assert _call_contents(final)[0].additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert str(_result_contents(final)[0].result) == "stopped"

    @pytest.mark.asyncio
    async def test_error_cap_exit_keeps_stamps(self) -> None:
        @tool(name="boom")
        async def boom(text: str) -> str:
            raise RuntimeError("kaboom")

        layer, wire = _stack(
            [[_call_update("c1", "boom", {"text": "a"})], [_text_update("forced")]],
            max_consecutive_errors=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [boom]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert wire.calls[1]["options"].get("tool_choice") == "none"
        assert _call_contents(final)[0].additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert final.messages[-1].contents[-1].text == "forced"

    @pytest.mark.asyncio
    async def test_exhaustion_tail_keeps_stamps(self) -> None:
        layer, wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [_text_update("forced")]],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert wire.calls[1]["options"].get("tool_choice") == "none"
        assert _call_contents(final)[0].additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert final.messages[-1].contents[-1].text == "forced"

    @pytest.mark.asyncio
    async def test_degenerate_stream_falls_back_to_from_updates(self) -> None:
        """No loop exit completed: finalize_stream keeps the raw-update merge as a fallback."""

        class _RaisingClient:
            def get_response(self, messages: Any, *, stream: bool = False, options: Any = None, **kwargs: Any) -> Any:
                async def _gen() -> Any:
                    yield _call_update("c1", "echo", {"text": "a"})
                    raise RuntimeError("wire died")

                return ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        # Bare ToolLoopLayer on purpose (hygiene-allowlisted): the wire dies
        # mid-stream, no loop iteration ever lands, and the fallback re-merge
        # below is NOT a loop-landed transcript — the invariant oracle's
        # precondition does not hold for it.
        layer = ToolLoopLayer(ChatMiddlewareLayer(_RaisingClient()))
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        with pytest.raises(RuntimeError, match="wire died"):
            async for _ in stream:
                pass
        final = await stream.get_final_response()

        # Fallback reconstruction: raw updates re-merged (no loop stamps —
        # the iteration never landed), but the transcript is not lost.
        calls = _call_contents(final)
        assert len(calls) == 1
        assert TOOL_INVOCATION_ORDER_KEY not in calls[0].additional_properties

    @pytest.mark.asyncio
    async def test_boundary_delta_starts_fresh_assistant_message(self) -> None:
        """A first final-turn delta with no role and no message_id must not glue onto the tool message."""
        call_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c1", name="echo", arguments={"text": "a"})],
            role="assistant",
            message_id="m1",
        )
        boundary_delta = ChatResponseUpdate(
            contents=[Content.from_text("final")],
            role=None,
            message_id=None,
        )
        usage_update = ChatResponseUpdate(
            contents=[
                Content.from_usage(
                    usage_details={"input_token_count": 1, "output_token_count": 2, "total_token_count": 3}
                )
            ],
            role=None,
            message_id=None,
        )
        layer, _wire = _stack([[call_update], [boundary_delta, usage_update]])
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert [m.role for m in final.messages] == ["assistant", "tool", "assistant"]
        assert final.messages[0].message_id == "m1"
        assert final.messages[-1].text == "final"
        assert all(item.type != "usage" for msg in final.messages for item in msg.contents)
        assert final.usage_details == {"input_token_count": 1, "output_token_count": 2, "total_token_count": 3}

    @pytest.mark.asyncio
    async def test_response_format_parses_structured_value_from_swapped_messages(self) -> None:
        layer, _wire = _stack([[_call_update("c1", "echo", {"text": "a"})], [_text_update('{"answer": 42}')]])
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()], "response_format": _StructuredAnswer},
        )
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert _call_contents(final)[0].additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert isinstance(final.value, _StructuredAnswer)
        assert final.value.answer == 42
