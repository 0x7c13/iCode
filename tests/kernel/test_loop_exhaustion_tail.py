# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The exhaustion-tail strip: the final tool_choice="none" turn never persists unexecutable calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from chrys.foundation.trajectory.context import TRAJECTORY_CONTEXT_KWARG
from chrys.foundation.trajectory.event_types import EventType as TrajectoryEventType
from chrys.foundation.trajectory.event_types import ToolOutcome
from chrys.foundation.trajectory.metadata import (
    OPERATION_ID_KEY,
)
from chrys.kernel import AgentSession, FunctionTool, LoopRecorder, tool
from chrys.kernel.loop import (
    _CONSECUTIVE_ERRORS_FALLBACK_TEXT,
    _MAX_FUNCTION_CALLS_FALLBACK_TEXT,
    _MAX_ITERATIONS_FALLBACK_TEXT,
    _strip_unexecutable_calls_from_update,
)
from chrys.kernel.middleware import (
    FunctionInvocationContext,
    FunctionMiddleware,
    MiddlewareTermination,
)
from chrys.kernel.types import (
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
)
from tests.kernel._fakes import (
    _call_response,
    _call_update,
    _final_response,
    _make_tool,
    _stack,
    _text_update,
    _user,
)
from tests.service.trajectory._fakes import FakeSink, make_context

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


# ---------------------------------------------------------------------------
# Exhaustion-tail strip of unexecutable function calls
# ---------------------------------------------------------------------------


class TestExhaustionTailStrip:
    """The final ``tool_choice="none"`` turn never persists unexecutable calls.

    A provider that ignores the disabled tools would otherwise leave a
    dangling function call on the success path — a shape no later repair
    revisits and that rejects every subsequent turn of a stored conversation.
    """

    @pytest.mark.asyncio
    async def test_call_only_final_is_stripped_with_fallback(self) -> None:
        recorder = LoopRecorder()
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "tool_choice ignored"})),
            ],
            max_iterations=1,
        )
        response = await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()]},
            client_kwargs={"loop_recorder": recorder},
        )
        assert wire.calls[1]["options"].get("tool_choice") == "none"
        # The executed exchange survives; the misbehaving final call is gone
        # and the fallback text takes its place.
        assert [m.role for m in response.messages] == ["assistant", "tool", "assistant"]
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT
        call_ids = [c.call_id for m in response.messages for c in m.contents if c.type == "function_call"]
        assert call_ids == ["c1"]
        recorded_call_ids = [
            content.call_id
            for message in recorder.loop_messages or []
            for content in message.contents
            if content.type == "function_call"
        ]
        assert "c2" not in recorded_call_ids
        assert response._chrys_service_state_invalidated is False

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_a_stripped_call_closes_its_operation_as_filtered(self, stream: bool) -> None:
        """The strip takes the call off the transcript; the operation it opened still has to end."""
        ignored = Content.from_function_call("c2", "echo", arguments={"text": "tool_choice ignored"})
        if stream:
            turns: list[Any] = [
                [_call_update("c1", "echo", {"text": "a"})],
                [ChatResponseUpdate(contents=[ignored], role="assistant", finish_reason="tool_calls")],
            ]
        else:
            turns = [
                _call_response(("c1", "echo", {"text": "a"})),
                ChatResponse(messages=[Message("assistant", [ignored])]),
            ]
        layer, _wire = _stack(turns, max_iterations=1)
        sink = FakeSink()

        await _final_response(
            layer,
            [_user()],
            stream=stream,
            options={"tools": [_make_tool()]},
            client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
        )

        filtered = [
            event
            for event in sink.of_type(TrajectoryEventType.TOOL_OPERATION_FINISHED)
            if event.payload["outcome"] == ToolOutcome.FILTERED
        ]
        assert len(filtered) == 1
        assert filtered[0].operation_id == ignored.additional_properties[OPERATION_ID_KEY]

    @pytest.mark.asyncio
    async def test_informational_call_survives_the_strip(self) -> None:
        informational = Content.from_function_call(
            "hosted-1",
            "remote_search",
            arguments={},
            informational_only=True,
        )
        final = ChatResponse(messages=[Message("assistant", [informational, Content.from_text("summarized")])])
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), final],
            max_iterations=1,
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        final_contents = response.messages[-1].contents
        assert [c.type for c in final_contents] == ["function_call", "text"]
        assert final_contents[0] is informational
        assert final_contents[1].text == "summarized"

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_mixed_text_and_call_strip_normalizes_finish_reason(self, stream: bool) -> None:
        # Visible text survives the strip (no fallback), but the response
        # must stop advertising tool work that no longer exists.
        mixed_contents = [
            Content.from_text("half an answer"),
            Content.from_function_call("c2", "echo", arguments={"text": "b"}),
        ]
        if stream:
            mixed_update = ChatResponseUpdate(contents=mixed_contents, role="assistant", finish_reason="tool_calls")
            layer, _wire = _stack([[_call_update("c1", "echo", {"text": "a"})], [mixed_update]], max_iterations=1)
            response_stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
            updates = [u async for u in response_stream]
            response = await response_stream.get_final_response()
            # The corrective metadata update carries the normalized reason so the
            # latest-non-null assembly and streamed consumers both see "stop".
            assert updates[-1].contents == []
            assert updates[-1].finish_reason == "stop"
        else:
            mixed_final = ChatResponse(messages=[Message("assistant", mixed_contents)], finish_reason="tool_calls")
            layer, _wire = _stack([_call_response(("c1", "echo", {"text": "a"})), mixed_final], max_iterations=1)
            response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert response.messages[-1].text == "half an answer"
        assert _MAX_ITERATIONS_FALLBACK_TEXT not in (response.text or "")
        call_ids = [c.call_id for m in response.messages for c in m.contents if c.type == "function_call"]
        assert call_ids == ["c1"]
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_blank_final_synthesizes_fallback(self) -> None:
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), ChatResponse(messages=[])],
            max_iterations=1,
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert response.messages[-1].role == "assistant"
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT

    @pytest.mark.asyncio
    async def test_reasoning_only_final_synthesizes_fallback(self) -> None:
        reasoning_final = ChatResponse(messages=[Message("assistant", [Content.from_text_reasoning(text="pondering")])])
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), reasoning_final],
            max_iterations=1,
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        # Nothing was stripped; the reasoning survives with the fallback beside it.
        assert any(c.type == "text_reasoning" for m in response.messages for c in m.contents)
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT

    @pytest.mark.asyncio
    async def test_stream_call_only_final_strips_updates_and_yields_fallback(self) -> None:
        exhaustion_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})],
            role="assistant",
            finish_reason="tool_calls",
        )
        layer, wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [exhaustion_update]],
            max_iterations=1,
        )
        sink = FakeSink()
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()]},
            client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
        )
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        assert wire.calls[1]["options"].get("tool_choice") == "none"
        # The call the tail asked not to receive is stripped from the updates
        # but still on the response the cycle is counted from, so the log says
        # the provider ignored ``tool_choice="none"`` rather than complied.
        counted = [
            event.payload["function_call_count"] for event in sink.of_type(TrajectoryEventType.MODEL_CYCLE_FINISHED)
        ]
        assert counted == [1, 1]
        streamed_call_ids = [c.call_id for u in updates for c in u.contents if c.type == "function_call"]
        assert streamed_call_ids == ["c1"], "the exhaustion turn's call must not reach the stream"
        assert updates[-1].contents[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT
        # The fallback ends the run: its "stop" must override the stripped
        # update's "tool_calls" in the latest-non-null assembly.
        assert updates[-1].finish_reason == "stop"
        assert response.finish_reason == "stop"
        assert [m.role for m in response.messages] == ["assistant", "tool", "assistant"]
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT
        call_ids = [c.call_id for m in response.messages for c in m.contents if c.type == "function_call"]
        assert call_ids == ["c1"]

    @pytest.mark.asyncio
    async def test_stream_emptied_update_with_metadata_still_yields(self) -> None:
        final_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})],
            role="assistant",
            response_id="resp-2",
            finish_reason="stop",
        )
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [final_update]],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        emptied = [u for u in updates if u.response_id == "resp-2" and not u.contents]
        assert len(emptied) == 1
        assert emptied[0] is not final_update, "yielded update is a copy"
        # The stripped copy must not carry the provider's terminal reason:
        # the tail re-emits the corrected one after every stripped update,
        # and a consumer keying on the first terminal reason would otherwise
        # treat the run as complete on withheld calls.
        assert emptied[0].finish_reason is None
        # Copy-on-write: the retry stream's recorded update keeps the call
        # and its reason, so the post-assembly strip still observes both.
        assert final_update.contents[0].call_id == "c2"
        assert final_update.finish_reason == "stop"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_terminal_reason_arrives_once_and_last(self) -> None:
        exhaustion_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})],
            role="assistant",
            finish_reason="tool_calls",
        )
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [exhaustion_update]],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        # A reason-only stripped update has nothing left once the terminal
        # reason is cleared, so it drops from the stream entirely; the
        # provider's "tool_calls" verdict on withheld work must never reach
        # a streaming consumer.
        assert all(u.finish_reason != "tool_calls" for u in updates)
        terminal = [i for i, u in enumerate(updates) if u.finish_reason in ("tool_calls", "stop")]
        assert len(terminal) == 1, "exactly one terminal reason crosses the stream"
        assert terminal[0] == len(updates) - 1, "the corrected terminal reason arrives last"
        assert updates[-1].finish_reason == "stop"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_separate_terminal_chunk_suppressed_after_strip(self) -> None:
        # Providers commonly emit the call delta and the finish-reason chunk
        # separately. The stateless per-update strip cannot see across
        # chunks, so the tail must remember the strip and suppress the later
        # reason-only chunk — otherwise it crosses untouched and a
        # first-terminal consumer stops before the corrective update.
        tail_turn = [
            _call_update("c2", "echo", {"text": "b"}),
            ChatResponseUpdate(contents=[], role="assistant", finish_reason="tool_calls"),
        ]
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], tail_turn],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        reasons = [u.finish_reason for u in updates if u.finish_reason is not None]
        assert reasons == ["stop"], "only the corrective terminal reason may cross the stream"
        assert updates[-1].finish_reason == "stop"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_stripped_length_reason_waits_for_corrective_update(self) -> None:
        # "length" on a stripped update is the provider's verdict on a turn
        # that ended in withheld calls; letting it cross ahead of the
        # corrective update makes a first-terminal consumer stop before the
        # fallback text arrives.
        exhaustion_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})],
            role="assistant",
            finish_reason="length",
        )
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [exhaustion_update]],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        await stream.get_final_response()
        reasons = [u.finish_reason for u in updates if u.finish_reason is not None]
        assert reasons == ["stop"], "the fallback's terminal reason must be the only one crossing"
        assert updates[-1].contents and updates[-1].contents[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT

    @pytest.mark.asyncio
    async def test_stream_corrective_reemits_non_stop_reason_after_suppression(self) -> None:
        # Visible text in the tail means no fallback update; the suppressed
        # provider reason ("length" here, via a separate reason-only chunk)
        # must still reach the consumer once — re-emitted by the corrective
        # update after every stripped update, matching the assembled
        # response's reason.
        tail_turn = [
            _text_update("partial answer"),
            _call_update("c2", "echo", {"text": "b"}),
            ChatResponseUpdate(contents=[], role="assistant", finish_reason="length"),
        ]
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], tail_turn],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        reasons = [u.finish_reason for u in updates if u.finish_reason is not None]
        assert reasons == ["length"], "the corrective update re-emits the assembled response's reason"
        assert updates[-1].finish_reason == "length"
        assert updates[-1].contents == []
        assert response.finish_reason == "length"
        assert any(c.text == "partial answer" for m in response.messages for c in m.contents if c.type == "text")

    def test_strip_update_helper_metadata_exemptions(self) -> None:
        def call_update(**kwargs: Any) -> ChatResponseUpdate:
            return ChatResponseUpdate(
                contents=[Content.from_function_call(call_id="c9", name="echo", arguments={})],
                role="assistant",
                **kwargs,
            )

        assert _strip_unexecutable_calls_from_update(call_update()) is None
        for exempting in (
            {"response_id": "resp-1"},
            {"conversation_id": "conv-1"},
            {"continuation_token": {"response_id": "bg-1"}},
            {"model": "m-1"},
            {"created_at": "2026-01-01"},
            {"message_id": "msg-1"},
            {"author_name": "agent"},
            {"additional_properties": {"provider": "meta"}},
        ):
            kept = _strip_unexecutable_calls_from_update(call_update(**exempting))
            assert kept is not None, f"metadata {exempting} must exempt the emptied update"
            assert kept.contents == []
        # finish_reason never exempts and every stripped copy sheds it — the
        # provider concluded the turn on the withheld calls, so any reason it
        # attached (terminal or "length") must wait for the corrective final
        # update. The original stays untouched for the post-assembly strip.
        for reason in ("stop", "tool_calls", "length"):
            assert _strip_unexecutable_calls_from_update(call_update(finish_reason=reason)) is None
            original = call_update(response_id="resp-1", finish_reason=reason)
            kept = _strip_unexecutable_calls_from_update(original)
            assert kept is not None
            assert kept.finish_reason is None
            assert original.finish_reason == reason
        passthrough = _text_update("hello")
        assert _strip_unexecutable_calls_from_update(passthrough) is passthrough

    def test_strip_update_helper_suppresses_later_reason_only_chunks(self) -> None:
        # Providers commonly emit the call delta and the finish-reason chunk
        # separately; once the tail has stripped a call, the caller raises
        # suppress_finish_reason and the later chunk must not carry the
        # provider's verdict across the stream.
        reason_only = ChatResponseUpdate(contents=[], role="assistant", finish_reason="tool_calls")
        assert _strip_unexecutable_calls_from_update(reason_only) is reason_only, (
            "without the flag a call-less update passes through untouched"
        )
        assert _strip_unexecutable_calls_from_update(reason_only, suppress_finish_reason=True) is None
        assert reason_only.finish_reason == "tool_calls", "the recorded original stays untouched"
        with_metadata = ChatResponseUpdate(contents=[], role="assistant", response_id="resp-1", finish_reason="length")
        kept = _strip_unexecutable_calls_from_update(with_metadata, suppress_finish_reason=True)
        assert kept is not None
        assert kept.finish_reason is None
        assert kept.response_id == "resp-1"
        reasonless = _text_update("hello")
        assert _strip_unexecutable_calls_from_update(reasonless, suppress_finish_reason=True) is reasonless

    @pytest.mark.asyncio
    async def test_strip_invalidates_service_continuation_blocking(self) -> None:
        session = AgentSession()
        final = ChatResponse(
            messages=[Message("assistant", [Content.from_function_call("c2", "echo", arguments={"text": "b"})])],
            conversation_id="conv-svc",
            response_id="resp-svc",
        )
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), final],
            max_iterations=1,
        )
        response = await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()], "store": True},
            client_kwargs={"session": session},
        )
        # The service-side transcript still holds the stripped call unanswered;
        # its continuation handle must not survive anywhere.
        assert response.conversation_id is None
        assert response.response_id is None
        assert response._chrys_service_state_invalidated is True
        assert session.service_session_id is None

    @pytest.mark.asyncio
    async def test_strip_without_service_storage_preserves_response_metadata(self) -> None:
        # Client-side storage: the service holds nothing, so the strip must
        # not erase ids a local-storage client attached as plain metadata.
        final = ChatResponse(
            messages=[Message("assistant", [Content.from_function_call("c2", "echo", arguments={"text": "b"})])],
            conversation_id="conv-local",
            response_id="resp-local",
        )
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), final],
            max_iterations=1,
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert response.conversation_id == "conv-local"
        assert response.response_id == "resp-local"
        assert response._chrys_service_state_invalidated is False
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT

    @pytest.mark.asyncio
    async def test_stream_strip_invalidates_service_continuation(self) -> None:
        session = AgentSession()
        final_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})],
            role="assistant",
            conversation_id="conv-svc",
            response_id="resp-svc",
        )
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [final_update]],
            max_iterations=1,
        )
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()], "store": True},
            client_kwargs={"session": session},
        )
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        # The emptied copy still yields (it carries conversation metadata) …
        emptied = [u for u in updates if u.conversation_id == "conv-svc"]
        assert len(emptied) == 1
        assert emptied[0].contents == []
        # … but the finalized response withholds the stale handles even though
        # ``from_updates`` would restore them from that update.
        assert response.conversation_id is None
        assert response.response_id is None
        assert response._chrys_service_state_invalidated is True
        assert session.service_session_id is None
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT

    @pytest.mark.asyncio
    async def test_stream_stopped_at_the_fallback_has_already_recorded_the_withheld_handles(self) -> None:
        # The fallback update is a suspension point: a consumer that stops
        # there never resumes the tail, so the handles the strip withholds
        # must already be recorded on the session by then.
        session = AgentSession()
        final_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})],
            role="assistant",
            conversation_id="conv-svc",
            response_id="resp-svc",
        )
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [final_update]],
            max_iterations=1,
        )
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()], "store": True},
            client_kwargs={"session": session},
        )
        stopped_at_fallback = False
        async for update in stream:
            if update.text == _MAX_ITERATIONS_FALLBACK_TEXT:
                stopped_at_fallback = True
                break
        await stream.aclose()
        assert stopped_at_fallback
        assert {"conv-svc", "resp-svc"} <= session.invalidated_service_session_ids
        assert session.service_session_id is None

    @pytest.mark.asyncio
    async def test_service_tail_restores_fresh_handle_after_finalization(self) -> None:
        # Entering the service-stored tail clears the consumed handle; a
        # successfully finalized, non-invalidated tail restores the fresh one.
        session = AgentSession()
        session.service_session_id = "conv-old"
        tail_updates = [
            ChatResponseUpdate(contents=[Content.from_text("done")], role="assistant"),
            ChatResponseUpdate(contents=[], role="assistant", conversation_id="conv-new"),
        ]
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], tail_updates],
            max_iterations=1,
        )
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()], "store": True},
            client_kwargs={"session": session},
        )
        [u async for u in stream]
        response = await stream.get_final_response()
        assert response.messages[-1].text == "done"
        assert response._chrys_service_state_invalidated is False
        assert session.service_session_id == "conv-new"

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_middleware_termination_invalidates_service_state(self, stream: bool) -> None:
        # Blocking: the termination return issues no further request, so the
        # service transcript behind the mirrored handle keeps this batch's
        # calls unanswered — the early return must carry the same invalidation
        # verdict as the exhaustion strip.
        # Streaming: termination on a NON-last iteration — the verdict must not
        # lean on the last-iteration pre-clear, and ``finalize_stream`` must withhold
        # the handles ``from_updates`` would restore from the raw updates.
        class _Terminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                raise MiddlewareTermination("stop")

        session = AgentSession()
        if stream:
            metadata_update = ChatResponseUpdate(
                contents=[],
                role="assistant",
                conversation_id="conv-old",
                response_id="resp-old",
            )
            turns: list[Any] = [[_call_update("c1", "echo", {"text": "a"}), metadata_update]]
        else:
            turns = [_call_response(("c1", "echo", {"text": "a"}), conversation_id="conv-old", response_id="resp-old")]
        layer, _wire = _stack(turns, max_iterations=3)
        response = await _final_response(
            layer,
            [_user()],
            stream=stream,
            options={"tools": [_make_tool()], "store": True},
            middleware=[_Terminate()],
            client_kwargs={"session": session},
        )
        assert response._chrys_service_state_invalidated is True
        assert response.conversation_id is None
        assert response.response_id is None
        assert session.service_session_id is None
        assert {"conv-old", "resp-old"} <= session.invalidated_service_session_ids

    @pytest.mark.asyncio
    async def test_stream_last_iteration_termination_still_records_session_handle(self) -> None:
        # On the LAST iteration the pre-clear ahead of the results yield
        # would drop the session's mirrored handle unrecorded; the
        # termination verdict must win that ordering. The round response
        # carries no ids, so the session field is the only handle source.
        class _Terminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                raise MiddlewareTermination("stop")

        session = AgentSession()
        session.service_session_id = "conv-old"
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})]],
            max_iterations=1,
        )
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()], "store": True},
            middleware=[_Terminate()],
            client_kwargs={"session": session},
        )
        [u async for u in stream]
        response = await stream.get_final_response()
        assert response._chrys_service_state_invalidated is True
        assert session.service_session_id is None
        assert "conv-old" in session.invalidated_service_session_ids


# ---------------------------------------------------------------------------
# A tool limit ends the loop through the same final turn
# ---------------------------------------------------------------------------


def _failing_tool(runs: list[str]) -> FunctionTool:
    @tool(name="boom")
    async def boom(text: str) -> str:
        runs.append(text)
        raise RuntimeError("kaput")

    return boom


class TestToolLimitEndsTheLoop:
    """Once a tool limit is hit, a provider ignoring ``tool_choice="none"`` gets no further call run."""

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    @pytest.mark.parametrize("limit", ["consecutive-errors", "max-function-calls"])
    async def test_calls_after_the_limit_are_stripped_not_run(self, stream: bool, limit: str) -> None:
        runs: list[str] = []
        if limit == "consecutive-errors":
            tool_name, tools, limits = "boom", [_failing_tool(runs)], {"max_consecutive_errors": 1}
            expected_text = _CONSECUTIVE_ERRORS_FALLBACK_TEXT
        else:
            tool_name, limits = "echo", {"max_function_calls": 1}
            tools = [_make_tool(runs)]
            expected_text = _MAX_FUNCTION_CALLS_FALLBACK_TEXT
        if stream:
            turns: list[Any] = [
                [_call_update("c1", tool_name, {"text": "a"})],
                [_call_update("c2", tool_name, {"text": "tool_choice ignored"})],
            ]
        else:
            turns = [
                _call_response(("c1", tool_name, {"text": "a"})),
                _call_response(("c2", tool_name, {"text": "tool_choice ignored"})),
            ]
        layer, wire = _stack(turns, max_iterations=5, **limits)
        sink = FakeSink()

        response = await _final_response(
            layer,
            [_user()],
            stream=stream,
            options={"tools": tools},
            client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
        )

        assert len(runs) == 1, "the call returned after the limit must not run"
        assert len(wire.calls) == 2
        assert wire.calls[1]["options"].get("tool_choice") == "none"
        assert [m.role for m in response.messages] == ["assistant", "tool", "assistant"]
        assert response.messages[-1].text == expected_text
        call_ids = [c.call_id for m in response.messages for c in m.contents if c.type == "function_call"]
        assert call_ids == ["c1"]
        cycle_indexes = [
            event.payload["cycle_index"] for event in sink.of_type(TrajectoryEventType.MODEL_CYCLE_STARTED)
        ]
        assert cycle_indexes == [0, 1]
