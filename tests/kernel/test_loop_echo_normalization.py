# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Landing normalization against echoing or mutating clients, plus its seeded adversarial transcript fuzz."""

from __future__ import annotations

import gc
import json
import random
from typing import TYPE_CHECKING, Any

import pytest

from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.foundation.trajectory.context import TRAJECTORY_CONTEXT_KWARG
from chrys.foundation.trajectory.event_types import EventType as TrajectoryEventType
from chrys.foundation.trajectory.metadata import (
    OPERATION_ID_KEY,
)
from chrys.kernel.identity import WeakIdentityRegistry
from chrys.kernel.loop import (
    ConsumedInjectionMessageProbe,
    _record_wire_request_content_identities,
)
from chrys.kernel.middleware import (
    ChatContext,
    ChatMiddleware,
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
    _call_response,
    _call_update,
    _final_response,
    _make_tool,
    _ordinals,
    _result_contents,
    _ScriptedClient,
    _stack,
    _text_response,
    _text_update,
    _user,
)
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


# ---------------------------------------------------------------------------
# Landing normalization: echoed / mutated client output vs. landed history
# ---------------------------------------------------------------------------


class TestEchoNormalization:
    """Echoed or mutated client output can neither rewrite landed history nor re-execute work."""

    @pytest.mark.asyncio
    async def test_client_mutating_its_retained_message_cannot_rewrite_landed_history(self) -> None:
        """A stateful client appends a fresh call to a Message it already returned.

        Landing retains loop-owned snapshots, so the in-place append changes
        only the client's object: the first exchange keeps exactly the calls
        it landed with, and the late call lands once, in the later response.
        """
        first_message = Message("assistant", [Content.from_function_call("c1", "echo", arguments={"text": "a"})])
        served = {"count": 0}

        class _MutatingStatefulClient:
            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                served["count"] += 1
                if served["count"] == 1:
                    turn = ChatResponse(messages=[first_message])
                elif served["count"] == 2:
                    first_message.contents.append(Content.from_function_call("c2", "echo", arguments={"text": "b"}))
                    turn = ChatResponse(messages=[first_message])
                else:
                    turn = _text_response()

                async def _resolve() -> ChatResponse:
                    return turn

                return _resolve()

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_MutatingStatefulClient()))
        events: list[str] = []
        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a", "tool:echo:b"], "each call executes exactly once"
        assert _ordinals(response) == [0, 1]
        # The first exchange's landed snapshot still holds ONLY c1; c2 lives
        # in the client's object and in the second exchange it landed with.
        assert [c.call_id for c in response.messages[0].contents] == ["c1"]
        assert all(m is not first_message for m in response.messages)

    @pytest.mark.asyncio
    async def test_client_mutating_next_turn_input_cannot_rewrite_landed_history(self) -> None:
        """The reverse aliasing direction: the client mutates a message it RECEIVED.

        The wire call hands the client per-call snapshot views, never the
        loop's retained transcript wrappers — so appending a call to an input
        history message (and re-sending that object as the response) changes
        only the throwaway view: the first exchange keeps exactly the calls
        it landed with, and the appended call lands once, in its own response.
        """
        served = {"count": 0}

        class _InputMutatingClient:
            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                served["count"] += 1
                if served["count"] == 1:
                    turn = ChatResponse(
                        messages=[
                            Message("assistant", [Content.from_function_call("c1", "echo", arguments={"text": "a"})])
                        ]
                    )
                elif served["count"] == 2:
                    victim = next(
                        m
                        for m in messages
                        if m.role == "assistant" and any(c.type == "function_call" for c in m.contents)
                    )
                    victim.contents.append(Content.from_function_call("c2", "echo", arguments={"text": "b"}))
                    turn = ChatResponse(messages=[victim])
                else:
                    turn = _text_response()

                async def _resolve() -> ChatResponse:
                    return turn

                return _resolve()

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_InputMutatingClient()))
        events: list[str] = []
        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a", "tool:echo:b"], "each call executes exactly once"
        assert _ordinals(response) == [0, 1]
        assert [c.call_id for c in response.messages[0].contents] == ["c1"]

    @pytest.mark.asyncio
    async def test_client_echoing_historical_call_object_does_not_reexecute(self) -> None:
        """The echo memo is seeded from caller history.

        A client returning the SAME Content object as a historical function
        call (a bridged or echoing client replaying the conversation) must
        not re-execute the past side-effecting call: the memo covers every
        content the conversation holds, not just what this run landed.
        """
        events: list[str] = []
        hist_call = Content.from_function_call("h1", "echo", arguments={"text": "once"})
        history = [
            _user("do it"),
            Message("assistant", [hist_call]),
            Message("tool", [Content.from_function_result("h1", result="echo:once")]),
            _user("again?"),
        ]
        served = {"count": 0}

        class _HistoryEchoingClient:
            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                served["count"] += 1
                turn = (
                    ChatResponse(messages=[Message("assistant", [hist_call])])
                    if served["count"] == 1
                    else _text_response()
                )

                async def _resolve() -> ChatResponse:
                    return turn

                return _resolve()

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_HistoryEchoingClient()))
        response = await layer.get_response(history, options={"tools": [_make_tool(events)]})

        assert events == [], "historical call must not re-execute"
        assert TOOL_INVOCATION_ORDER_KEY not in hist_call.additional_properties
        assert _call_contents(response) == []

    def test_wire_request_identity_registration_registers_contents_and_skips_usage(self) -> None:
        """Wire-only contents enter the echo registry; usage stays out via the
        registry's construction-time policy. No retention side: weakref
        liveness makes a collected wire-only content's entry evict, so its
        recycled id can never strip a genuinely fresh content."""
        injected = Content.from_text("wire-only")
        usage = Content.from_usage(
            usage_details={"input_token_count": 1, "output_token_count": 0, "total_token_count": 1}
        )
        echo_registry = WeakIdentityRegistry(ignore_usage=True)

        _record_wire_request_content_identities(
            [Message("user", [injected, usage])],
            echo_registry,
        )

        assert injected in echo_registry
        assert usage not in echo_registry
        assert len(echo_registry) == 1

        del injected
        gc.collect()
        assert len(echo_registry) == 0, "no retention: a dead wire-only content's entry evicts"

    @pytest.mark.asyncio
    async def test_blocking_usage_identity_can_repeat_across_history_and_model_calls(self) -> None:
        """Request-scoped usage never enters the blocking echo memo."""
        usage = Content.from_usage(
            usage_details={"input_token_count": 3, "output_token_count": 1, "total_token_count": 4}
        )
        history = [
            _user("old"),
            Message("assistant", [usage]),
            _user("continue"),
        ]
        first = ChatResponse(
            messages=[
                Message(
                    "assistant",
                    [
                        Content.from_function_call("c1", "echo", arguments={"text": "a"}),
                        usage,
                    ],
                )
            ]
        )
        final = ChatResponse(messages=[Message("assistant", [Content.from_text("done"), usage])])
        layer, wire = _stack([first, final])

        response = await layer.get_response(history, options={"tools": [_make_tool()]})
        landed_usage = [
            content for message in response.messages for content in message.contents if content.type == "usage"
        ]

        assert len(wire.calls) == 2
        assert landed_usage == [usage, usage]
        assert all(content is usage for content in landed_usage)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_post_middleware_request_content_echo_is_removed(self, stream: bool) -> None:
        """A wire-only injected object cannot land back as assistant text."""
        injected_message = Message("user", ["INJECTED"])
        injected_content = injected_message.contents[0]

        class _InjectWireOnlyMessage(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                context.messages = [*context.messages, injected_message]
                await call_next()

        class _EchoInjectedContentClient:
            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                assert any(item is injected_content for message in messages for item in message.contents)
                fresh = Content.from_text("done")
                if not stream:

                    async def _resolve() -> ChatResponse:
                        return ChatResponse(messages=[Message("assistant", [injected_content, fresh])])

                    return _resolve()

                async def _gen() -> Any:
                    yield ChatResponseUpdate(contents=[injected_content], role="assistant")
                    yield ChatResponseUpdate(contents=[fresh], role="assistant")

                return ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        layer = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(_EchoInjectedContentClient(), middleware=[_InjectWireOnlyMessage()])
        )

        if stream:
            response_stream = layer.get_response([_user()], stream=True)
            yielded = [update async for update in response_stream]
            response = await response_stream.get_final_response()
            assert [content.text for update in yielded for content in update.contents] == ["done"]
        else:
            response = await layer.get_response([_user()])

        assert response.text == "done"
        assert all(item is not injected_content for message in response.messages for item in message.contents)

    @pytest.mark.asyncio
    async def test_client_mutating_received_history_wrapper_cannot_corrupt_session_state(self) -> None:
        """Caller-history wrappers reach the client as per-call views too.

        A client appending a call to a received historical assistant message
        (and re-sending that object as its response) mutates only the
        throwaway view: the caller's live history object is untouched, the
        echoed historical call does not re-execute, and only the genuinely
        fresh call lands and runs.
        """
        events: list[str] = []
        hist_asst = Message("assistant", [Content.from_function_call("h1", "echo", arguments={"text": "old"})])
        history = [
            _user("do it"),
            hist_asst,
            Message("tool", [Content.from_function_result("h1", result="echo:old")]),
            _user("again?"),
        ]
        served = {"count": 0}

        class _HistoryMutatingClient:
            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                served["count"] += 1
                if served["count"] == 1:
                    victim = next(
                        m
                        for m in messages
                        if m.role == "assistant" and any(c.type == "function_call" for c in m.contents)
                    )
                    victim.contents.append(Content.from_function_call("c9", "echo", arguments={"text": "new"}))
                    turn = ChatResponse(messages=[victim])
                else:
                    turn = _text_response()

                async def _resolve() -> ChatResponse:
                    return turn

                return _resolve()

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_HistoryMutatingClient()))
        response = await layer.get_response(history, options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:new"], "only the fresh call executes"
        assert [c.call_id for c in hist_asst.contents] == ["h1"], "live session-state history stays intact"
        assert [c.call_id for c in _call_contents(response)] == ["c9"]

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_client_mutating_received_injection_cannot_corrupt_retained_copy(self, stream: bool) -> None:
        """Consumed injection wrappers the loop retains go out as views too.

        The probe hands the loop a middleware-owned user message between
        iterations, and the loop re-sends it on every later call. A client
        appending to the received wrapper corrupts only its per-call view:
        the retained wrapper — what later calls re-send and what persistence
        captures — keeps exactly its original content.
        """
        injected = Message("user", ["injected note"])
        injected.additional_properties["_injection_id"] = "inj-1"
        pending = [injected]
        probe = ConsumedInjectionMessageProbe(lambda: [pending.pop()] if pending else [])
        received: list[Message] = []

        class _InjectionMutatingClient(_ScriptedClient):
            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                if len(self.calls) == 1:
                    victim = next(m for m in messages if m.role == "user" and m.contents[0].text == "injected note")
                    received.append(victim)
                    victim.contents.append(Content.from_text("CORRUPTED"))
                return super().get_response(messages, stream=stream, **kwargs)

        if stream:
            turns: list[Any] = [
                [_call_update("c1", "echo", {"text": "a"})],
                [_call_update("c2", "echo", {"text": "b"})],
                [_text_update("final")],
            ]
        else:
            turns = [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "b"})),
                _text_response(),
            ]
        wire = _InjectionMutatingClient(turns)
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
        await _final_response(
            layer,
            [_user()],
            stream=stream,
            options={"tools": [_make_tool()]},
            client_kwargs={"consumed_injection_message_probe": probe},
        )

        assert [c.text for c in injected.contents] == ["injected note"]
        view = received[0]
        assert view is not injected
        assert view.contents[0] is injected.contents[0]
        assert view.additional_properties is injected.additional_properties
        # The third call's wire input carries the clean injection, not the mutation.
        final_wire = wire.calls[2]["messages"]
        third_call_copies = [m for m in final_wire if m.role == "user" and m.contents[0].text == "injected note"]
        assert len(third_call_copies) == 1
        assert [c.text for c in third_call_copies[0].contents] == ["injected note"]

    @pytest.mark.asyncio
    async def test_duplicate_and_already_answered_calls_are_stamped_inert(self) -> None:
        """Calls the dispatch filter drops still consume ordinals but never execute."""
        events: list[str] = []
        answered_call = Content.from_function_call("done-1", "echo", arguments={"text": "past"})
        answered_result = Content.from_function_result("done-1", result="already")
        live_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        duplicate_call = Content.from_function_call("c1", "echo", arguments={"text": "b"})
        first_turn = ChatResponse(
            messages=[
                Message("assistant", [answered_call, live_call, duplicate_call]),
                Message("tool", [answered_result]),
            ]
        )
        layer, _wire = _stack([first_turn, _text_response("done")])
        sink = FakeSink()

        response = await layer.get_response(
            [_user()],
            options={"tools": [_make_tool(events)]},
            client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
        )

        assert events == ["tool:echo:a"], "answered + duplicate calls must not execute"
        assert answered_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert live_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 1
        assert duplicate_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 2
        assert OPERATION_ID_KEY not in answered_call.additional_properties
        assert (
            duplicate_call.additional_properties[OPERATION_ID_KEY] == live_call.additional_properties[OPERATION_ID_KEY]
        )
        # Neither dropped call opens an operation of its own: the answered one
        # is provider-owned history and the duplicate rides the dispatched
        # call's operation, so there is nothing here for a terminal to close.
        # (The executed call's own pair comes from the tool-events middleware,
        # which this kernel-only stack does not install.)
        assert sink.of_type(TrajectoryEventType.TOOL_OPERATION_STARTED) == []
        assert sink.of_type(TrajectoryEventType.TOOL_OPERATION_FINISHED) == []
        new_results = [r for r in _result_contents(response) if str(r.result) != "already"]
        assert [r.call_id for r in new_results] == ["c1"]

    @pytest.mark.asyncio
    async def test_fresh_call_reusing_an_echoed_call_id_still_executes(self) -> None:
        """Echo removal is by object identity; a fresh same-id call is new work.

        Providers mint per-response counter ids, so a later response can
        legitimately pair an echoed old object with a NEW content reusing the
        same call id. The fresh call must land, stamp, and execute.
        """
        events: list[str] = []
        old_call = Content.from_function_call("call_00", "echo", arguments={"text": "old"})
        fresh_call = Content.from_function_call("call_00", "echo", arguments={"text": "new"})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [old_call])]),
                ChatResponse(messages=[Message("assistant", [old_call, fresh_call])]),
                _text_response("done"),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:old", "tool:echo:new"]
        assert old_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert fresh_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 1
        assert [r.call_id for r in _result_contents(response)] == ["call_00", "call_00"]

    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_same_object_twice_in_one_response_executes_once(self, stream: bool) -> None:
        """An identity duplicate within one response is normalized to one copy.

        Streaming counterpart: the echo filter only knows PREVIOUSLY landed
        identities, so a fresh object emitted twice within one streamed
        response reaches assembly twice — and adjacent copies would
        SELF-MERGE, concatenating the call's own argument string into
        unparseable JSON and turning a valid call into an argument-parsing
        failure. Assembly skips a content object it already incorporated in
        the same pass, matching the blocking path's one-copy-one-execution
        collapse at landing.
        """
        events: list[str] = []
        if stream:
            call = Content.from_function_call(call_id="c1", name="echo", arguments='{"text": "once"}')
            turns: list[Any] = [
                [
                    ChatResponseUpdate(contents=[call], role="assistant"),
                    ChatResponseUpdate(contents=[call], role="assistant"),
                ],
                [_text_update("done")],
            ]
        else:
            call = Content.from_function_call("c1", "echo", arguments={"text": "once"})
            turns = [ChatResponse(messages=[Message("assistant", [call, call])]), _text_response("done")]
        layer, _wire = _stack(turns)

        response = await _final_response(layer, [_user()], stream=stream, options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:once"], "the duplicated object must execute exactly once"
        assert call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert [c.call_id for c in _call_contents(response)] == ["c1"]
        assert [r.call_id for r in _result_contents(response)] == ["c1"]

    @pytest.mark.asyncio
    async def test_echoed_message_object_preserves_prior_transcript(self) -> None:
        """Echo removal must not mutate a Message the transcript already holds.

        A client can echo the whole prior Message object, not just its
        contents; clearing that message in place would also clear the
        accumulated transcript's copy, leaving an orphaned result whose call
        vanished from history.
        """
        events: list[str] = []
        call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        shared_message = Message("assistant", [call])
        layer, _wire = _stack(
            [
                ChatResponse(messages=[shared_message]),
                ChatResponse(messages=[shared_message]),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a"]
        assert shared_message.contents == [call], "the shared message must keep its call"
        assert [c.call_id for c in _call_contents(response)] == ["c1"]
        assert [r.call_id for r in _result_contents(response)] == ["c1"]
        assert response.messages[-1].role == "tool"

    @pytest.mark.asyncio
    async def test_full_exchange_echo_is_removed(self) -> None:
        """Echoing a completed call/result exchange lands as pure history.

        The echoed result must vanish with its call: removing only the call
        would leave one call answered by two results — an invalid transcript.
        """
        events: list[str] = []
        call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        first_response = ChatResponse(messages=[Message("assistant", [call])])

        class _EchoingClient(_ScriptedClient):
            """Second turn echoes the first exchange — including the loop-made result — in new Message wrappers."""

            def get_response(self, messages: Any, **kwargs: Any) -> Any:
                if not self.turns:
                    self.turns.append(
                        ChatResponse(messages=[Message(m.role, list(m.contents)) for m in first_response.messages])
                    )
                return super().get_response(messages, **kwargs)

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_EchoingClient([first_response])))

        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a"]
        assert [c.call_id for c in _call_contents(response)] == ["c1"]
        assert [r.call_id for r in _result_contents(response)] == ["c1"], "the echoed result must not duplicate"
        assert [m.role for m in response.messages] == ["assistant", "tool"]

    @pytest.mark.asyncio
    async def test_streaming_echoed_result_content_is_removed(self) -> None:
        """A streamed echo of an already-landed result does not duplicate it."""
        events: list[str] = []
        answered_call = Content.from_function_call("done-1", "echo", arguments={"text": "past"})
        answered_result = Content.from_function_result("done-1", result="already")
        live_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack(
            [
                [
                    ChatResponseUpdate(contents=[answered_call], role="assistant"),
                    ChatResponseUpdate(contents=[answered_result], role="tool"),
                    ChatResponseUpdate(contents=[live_call], role="assistant"),
                ],
                [
                    ChatResponseUpdate(contents=[answered_call], role="assistant"),
                    ChatResponseUpdate(contents=[answered_result], role="tool"),
                ],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:a"]
        assert [c.call_id for c in _call_contents(final)].count("done-1") == 1
        assert [r.call_id for r in _result_contents(final)].count("done-1") == 1

    @pytest.mark.asyncio
    async def test_revisited_content_is_removed_from_streaming_final_response(self) -> None:
        """A streamed echo is removed from the assembled final response."""
        events: list[str] = []
        first_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack(
            [
                [ChatResponseUpdate(contents=[first_call], role="assistant")],
                [ChatResponseUpdate(contents=[first_call], role="assistant")],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:a"]
        assert len(_call_contents(final)) == 1, "the echo must not survive as a dangling duplicate call"
        assert [r.call_id for r in _result_contents(final)] == ["c1"]


# ---------------------------------------------------------------------------
# Seeded adversarial transcript fuzz
# ---------------------------------------------------------------------------


class _AdversarialEchoClient:
    """Wire double that mixes fresh work with every known echo shape.

    Each tool turn emits 1-2 FRESH calls (unique ids within the turn; ids may
    deliberately REUSE a previous turn's id — providers mint per-response
    counter ids, so reuse is legitimate new work; arguments are randomly
    dict- or string-typed — the string form is the wire representation,
    where a duplicate object self-merging in assembly corrupts by
    concatenation instead of merging silently) and then, driven by the
    seeded rng, injects adversarial shapes the landing normalization must
    neutralize:

    - an echo of a prior function_call content (same object, new wrapper);
    - an echo of a prior function_result content (same object);
    - a whole prior Message object re-sent verbatim (assistant or tool);
    - one of this turn's fresh call objects duplicated within the response;
    - a previously returned assistant Message mutated IN PLACE (a fresh call
      appended) and re-sent — the stateful-client shape (non-streamed turns
      only: streaming assembly mints fresh containers, so the client's
      wrappers never enter the transcript there);
    - an assistant Message from this call's INPUT mutated in place and
      re-sent — the reverse aliasing direction (the wire hands the client
      per-call snapshot views, so the mutation must change only the view;
      non-streamed turns only, same reason as above);
    - the same landed call object twice ADJACENTLY in a streamed turn —
      assembly would merge the pair into a NEW object, so the echo
      fragments must be stripped from assembly input before a laundered
      copy can be minted;
    - that merged double echo followed, behind a text separator, by a
      FRESH call reusing the same call id — the poison combo: the
      laundered copy must not re-execute the old arguments, and the fresh
      call must not be swallowed by duplicate-id dispatch filtering;
    - a fresh call split into raw string-argument fragments across
      adjacent updates (streamed turns) — assembly merges them into ONE
      call that executes once;
    - a raw fragment from a COMPLETED turn replayed verbatim on a later
      streamed turn — the merge minted a NEW assembled object, so only
      post-landing fragment-identity recording can recognize the replay;
      it must not re-execute and must not corrupt adjacent fresh work.

    Echo candidates are harvested from the conversation the loop sends back
    to the wire (``prepped_messages`` carries the full exchange, including
    loop-produced result messages), exactly where a real echoing or bridged
    client would get them — including CALLER HISTORY when the run is seeded
    with one, whose objects the memo must treat as landed from the start.
    After ``tool_turns`` turns it returns a plain text response so the loop
    terminates.

    The oracle contract for tests: every fresh call executes exactly once and
    yields exactly one result; echoes execute nothing and leave no trace in
    the final transcript (enforced by ``assert_transcript_invariants`` via
    ``InvariantCheckedToolLoopLayer``).
    """

    def __init__(self, rng: random.Random, tool_turns: int) -> None:
        self.rng = rng
        self.remaining_tool_turns = tool_turns
        self.fresh_calls: list[Content] = []
        self.minted_ids: list[str] = []
        self.returned_messages: list[Message] = []
        self._serial = 0
        # Raw fragments emitted on the turn in flight vs. fragments whose
        # turn has landed — only the latter are fair game for replay (a
        # same-turn duplicate object is a different shape: within one
        # response, assembly merges it instead of the memo stripping it).
        self._pending_fragments: list[Content] = []
        self._replayable_fragments: list[Content] = []

    def _fresh_call(self, response_ids: set[str]) -> Content:
        reusable = [i for i in self.minted_ids if i not in response_ids]
        if reusable and self.rng.random() < 0.4:
            call_id = self.rng.choice(reusable)
        else:
            call_id = f"call_{self._serial}"
            self.minted_ids.append(call_id)
        arguments: Any = {"text": f"t{self._serial}"}
        if self.rng.random() < 0.3:
            # String-typed arguments (the wire representation): a repeated
            # object corrupts by self-concatenation if assembly ever
            # merges it with itself, where dict arguments merge silently.
            arguments = json.dumps(arguments)
        call = Content.from_function_call(call_id, "echo", arguments=arguments)
        self._serial += 1
        self.fresh_calls.append(call)
        response_ids.add(call_id)
        return call

    def _build_turn_messages(self, conversation: list[Message], *, distinct_call_ids: bool) -> list[Message]:
        # ``distinct_call_ids`` keeps a streamed turn's function-call ids
        # unique: streaming assembly coalesces same-id call contents within
        # one response as fragments of a single call, so an echoed call
        # sharing a fresh call's id would be MERGED with it — a shape real
        # providers never emit inside one response (ids are unique per
        # response; reuse happens across responses, which both fuzz modes
        # still exercise). The non-streaming mode leaves collisions in to
        # also cover echo-beside-same-id-fresh-call within one response.
        prior_messages = [m for m in conversation if m.role in ("assistant", "tool")]
        prior_calls = [c for m in prior_messages for c in m.contents if c.type == "function_call"]
        prior_results = [c for m in prior_messages for c in m.contents if c.type == "function_result"]

        response_ids: set[str] = set()
        contents: list[Content] = [self._fresh_call(response_ids) for _ in range(self.rng.randint(1, 2))]
        if prior_calls and self.rng.random() < 0.6:
            echoed_call = self.rng.choice(prior_calls)
            if not (distinct_call_ids and echoed_call.call_id in response_ids):
                response_ids.add(echoed_call.call_id or "")
                contents.insert(self.rng.randrange(len(contents) + 1), echoed_call)
        if prior_calls and distinct_call_ids and self.rng.random() < 0.35:
            # Laundering shape (streamed turns): the SAME landed call object
            # twice ADJACENTLY — assembly would merge the pair into a NEW
            # object, so the echo fragments must be stripped before assembly.
            echoed_call = self.rng.choice(prior_calls)
            if echoed_call.call_id not in response_ids:
                response_ids.add(echoed_call.call_id or "")
                contents.extend([echoed_call, echoed_call])
                if self.rng.random() < 0.5:
                    # Poison combo: a FRESH call reusing the laundered id
                    # later in the same turn (text keeps it out of the
                    # merge). The old arguments must not run again, and the
                    # fresh call must not fall to duplicate-id dispatch.
                    contents.append(Content.from_text("separator"))
                    fresh_reuse = Content.from_function_call(
                        echoed_call.call_id, "echo", arguments={"text": f"t{self._serial}"}
                    )
                    self._serial += 1
                    self.fresh_calls.append(fresh_reuse)
                    contents.append(fresh_reuse)
        if self.rng.random() < 0.4:
            # Same fresh object twice in one response — regardless of how the
            # landing/assembly path normalizes the duplicate, the oracle
            # holds it to one execution and one transcript copy.
            contents.append(self.rng.choice([c for c in contents if c.type == "function_call"]))

        hosted_continuation: Content | None = None
        if self.rng.random() < 0.35:
            # Hosted-tool shapes widen the fuzz beyond function calls:
            # provider-executed calls arrive answered within the same
            # response — embedded in the call-carrying message itself
            # (either content order: exchange pairing is message-phased) or
            # riding a result-only assistant continuation message. Ids are
            # minted unique, hosted calls are never dispatched or stamped,
            # and nothing here enters the echo pools, which harvest
            # function shapes only.
            hosted_id = f"hosted_{self._serial}"
            self._serial += 1
            if self.rng.random() < 0.5:
                hosted_call = Content.from_mcp_server_tool_call(hosted_id, "search")
                hosted_result = Content.from_mcp_server_tool_result(hosted_id, output="ok")
            else:
                hosted_call = Content.from_image_generation_tool_call(image_id=hosted_id)
                hosted_result = Content.from_image_generation_tool_result(image_id=hosted_id)
            placement = self.rng.random()
            if placement < 0.4:
                contents.extend([hosted_call, hosted_result])
            elif placement < 0.7:
                contents.extend([hosted_result, hosted_call])
            else:
                contents.append(hosted_call)
                hosted_continuation = hosted_result

        minted = Message("assistant", contents)
        self.returned_messages.append(minted)
        messages = [minted]
        if prior_results and self.rng.random() < 0.6:
            messages.append(Message("tool", [self.rng.choice(prior_results)]))
        if prior_messages and self.rng.random() < 0.5:
            echoed_message = self.rng.choice(prior_messages)
            echoed_ids = {c.call_id for c in echoed_message.contents if c.type == "function_call" and c.call_id}
            if not (distinct_call_ids and echoed_ids & response_ids):
                response_ids.update(echoed_ids)
                messages.insert(self.rng.randrange(len(messages) + 1), echoed_message)
        input_assistants = [m for m in prior_messages if m.role == "assistant"]
        if input_assistants and not distinct_call_ids and self.rng.random() < 0.3:
            # Reverse aliasing: mutate a history message received as INPUT
            # and re-send that same object. The wire view makes the mutation
            # land on a throwaway wrapper; at landing the old contents drop
            # as echoes and only the appended call lands. Always re-sent so
            # the execution oracle's fresh-call count stays exact.
            victim = self.rng.choice(input_assistants)
            victim.contents.append(self._fresh_call(response_ids))
            if all(m is not victim for m in messages):
                messages.insert(self.rng.randrange(len(messages) + 1), victim)
        earlier_returned = [m for m in self.returned_messages if m is not minted]
        if earlier_returned and not distinct_call_ids and self.rng.random() < 0.35:
            # Stateful-client mutation: append a fresh call INTO a message
            # this client returned on an earlier turn, then re-send that same
            # object. The append must change nothing the loop already landed
            # (landing snapshots wrappers); at landing the old contents drop
            # as echoes and only the appended call lands. Always re-sent so
            # the execution oracle's fresh-call count stays exact.
            victim = self.rng.choice(earlier_returned)
            victim.contents.append(self._fresh_call(response_ids))
            if all(m is not victim for m in messages):
                messages.insert(self.rng.randrange(len(messages) + 1), victim)
        if hosted_continuation is not None:
            # The continuation goes LAST: a call-carrying assistant behind a
            # result-only assistant would start a new exchange, while the
            # loop folds every executed result after the whole response —
            # real hosted results either share the call's message or close
            # the turn, so the fuzz keeps them turn-closing too.
            messages.append(Message("assistant", [hosted_continuation]))
        return messages

    def get_response(
        self,
        messages: Any,
        *,
        stream: bool = False,
        options: Any = None,
        **kwargs: Any,
    ) -> Any:
        conversation = list(messages)
        # The previous turn has landed by the time the loop calls again, so
        # its fragments graduate to the replayable pool.
        self._replayable_fragments.extend(self._pending_fragments)
        self._pending_fragments = []
        fresh_before = len(self.fresh_calls)
        if self.remaining_tool_turns > 0:
            self.remaining_tool_turns -= 1
            turn_messages = self._build_turn_messages(conversation, distinct_call_ids=stream)
        else:
            turn_messages = [Message("assistant", [Content.from_text("done")])]
        turn_fresh_ids = {id(c) for c in self.fresh_calls[fresh_before:]}

        if not stream:
            turn_response = ChatResponse(messages=turn_messages)

            async def _resolve() -> ChatResponse:
                return turn_response

            return _resolve()

        async def _gen() -> Any:
            replay = (
                self.rng.choice(self._replayable_fragments)
                if self._replayable_fragments and self.rng.random() < 0.35
                else None
            )
            if replay is not None and self.rng.random() < 0.5:
                yield ChatResponseUpdate(contents=[replay], role="assistant")
                replay = None
            all_contents = [c for m in turn_messages for c in m.contents]
            for message in turn_messages:
                whole: list[Content] = []
                for content in message.contents:
                    # Only calls minted THIS turn and appearing once may be
                    # fragmented: fragmenting an echoed prior-turn object (or
                    # one leg of a duplicated pair) would mint new objects
                    # carrying old data — a data copy, which the identity
                    # model deliberately treats as fresh work.
                    fragment = (
                        content.type == "function_call"
                        and id(content) in turn_fresh_ids
                        and sum(1 for c in all_contents if c is content) == 1
                        and self.rng.random() < 0.35
                    )
                    if not fragment:
                        whole.append(content)
                        continue
                    if whole:
                        yield ChatResponseUpdate(contents=whole, role=message.role)
                        whole = []
                    args_json = (
                        content.arguments if isinstance(content.arguments, str) else json.dumps(content.arguments)
                    )
                    split_at = self.rng.randint(1, len(args_json) - 1)
                    frag_pair = [
                        Content.from_function_call(content.call_id, content.name, arguments=args_json[:split_at]),
                        Content.from_function_call(content.call_id, content.name, arguments=args_json[split_at:]),
                    ]
                    self._pending_fragments.extend(frag_pair)
                    for frag in frag_pair:
                        yield ChatResponseUpdate(contents=[frag], role=message.role)
                if whole:
                    yield ChatResponseUpdate(contents=whole, role=message.role)
            if replay is not None:
                yield ChatResponseUpdate(contents=[replay], role="assistant")

        rs: ResponseStream[ChatResponseUpdate, ChatResponse] = ResponseStream(
            _gen(), finalizer=ChatResponse.from_updates
        )
        return rs


def _fresh_call_text(call: Content) -> str:
    """The ``text`` argument of a fuzz fresh call, dict- or string-typed."""
    args = call.arguments
    if isinstance(args, str):
        args = json.loads(args)
    return args["text"]


def _fuzz_history(rng: random.Random) -> list[Message]:
    """Seeded caller history: 0-2 past tool exchanges the client can echo.

    Historical call/result objects reach the client through the wire views
    (shared content objects), so every echo branch of the adversarial client
    naturally exercises the history-seeded memo as well.
    """
    history: list[Message] = [_user()]
    for i in range(rng.randint(0, 2)):
        history.append(
            Message("assistant", [Content.from_function_call(f"hist_{i}", "echo", arguments={"text": f"h{i}"})])
        )
        history.append(Message("tool", [Content.from_function_result(f"hist_{i}", result=f"echo:h{i}")]))
    history.append(_user("again"))
    return history


class TestTranscriptInvariantFuzz:
    """Seeded adversarial fuzz of the landing/dispatch/fold transcript contract.

    Rationale: the echo-normalization defects were each found one
    counterexample at a time (dangling duplicate call, suppressed same-id
    fresh call, in-place mutation of a shared message, double-recorded
    result). This harness generates the whole input space those shapes came
    from — random compositions of fresh work, id reuse, and object echoes —
    and holds the loop to the structural oracle plus an exact execution
    count. A regression in ANY of the normalization rules surfaces here
    without anyone having to imagine the specific failing shape first.

    Seeds are fixed and each seed is its own parametrized case, so any
    failure — including one raised by the structural oracle inside the
    checked layer — names its seed in the failing test id; rerun exactly that
    case with ``-k`` while debugging. Do not replace the seeded rng with
    entropy — flaky guards get deleted, deterministic ones get fixed.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", range(30))
    async def test_adversarial_echo_clients_never_break_transcript_invariants(self, seed: int) -> None:
        rng = random.Random(seed)
        wire = _AdversarialEchoClient(rng, tool_turns=rng.randint(1, 3))
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
        events: list[str] = []
        history = _fuzz_history(rng)
        history_shapes = [[c.call_id for c in m.contents] for m in history]

        response = await layer.get_response(history, options={"tools": [_make_tool(events)]})

        # The structural invariants already ran inside the checked layer.
        # The execution oracle pins behavior the structure alone cannot:
        # every fresh call ran exactly once, echoes ran zero times, and the
        # caller's history objects came through structurally untouched.
        expected = sorted(f"tool:echo:{_fresh_call_text(c)}" for c in wire.fresh_calls)
        assert sorted(events) == expected, "fresh-call executions diverged"
        assert len(_call_contents(response)) == len(wire.fresh_calls), "echoed call leaked"
        assert len(_result_contents(response)) == len(wire.fresh_calls), "result count diverged"
        assert [[c.call_id for c in m.contents] for m in history] == history_shapes, "caller history corrupted"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", range(20))
    async def test_adversarial_echo_clients_never_break_streaming_invariants(self, seed: int) -> None:
        # Streaming assembly preserves object identity for whole-content
        # updates, so the same echo shapes must normalize identically; whole
        # echoed Messages degrade to their contents (assembly builds fresh
        # containers), which the content-identity memo still covers — except
        # the adjacent double echo, stripped from assembly input before its
        # merge can launder the identity, and replayed raw fragments, caught
        # by post-landing fragment-identity recording.
        rng = random.Random(seed)
        wire = _AdversarialEchoClient(rng, tool_turns=rng.randint(1, 3))
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
        events: list[str] = []
        history = _fuzz_history(rng)
        history_shapes = [[c.call_id for c in m.contents] for m in history]

        stream = layer.get_response(history, stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        expected = sorted(f"tool:echo:{_fresh_call_text(c)}" for c in wire.fresh_calls)
        assert sorted(events) == expected, "fresh-call executions diverged"
        assert len(_call_contents(final)) == len(wire.fresh_calls), "echoed call leaked"
        assert len(_result_contents(final)) == len(wire.fresh_calls), "result count diverged"
        assert [[c.call_id for c in m.contents] for m in history] == history_shapes, "caller history corrupted"
