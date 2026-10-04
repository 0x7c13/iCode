# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Characterization of the Chrys-owned run-pipeline invariants.

These tests used to pin undocumented upstream behavior. Step 2 of the
full decoupling plan moves the live wire base client into Chrys, so the same
contracts now pin the owned stack directly:
``ToolLoopLayer(ChatMiddlewareLayer(BaseChatClient))``.

1. **Tool-loop accumulation**: the loop keeps ONE ``prepped_messages`` list and
   grows it via ``extend(response.messages)`` each iteration
   (``kernel/loop.py::_LoopRun._queue_batch``, consumed injections via
   ``_LoopRun._queue_consumed_injections``), so per-call middleware snapshots grow
   strictly by appending — earlier snapshots are identity-prefixes of later ones.
   ``LoopRecorder`` derives interrupt-recovery messages from its
   ``initial_count`` arithmetic and suppresses crash-recovery checkpoints via a
   ``(len, last-message-hash)`` key
   (``kernel/_loop_recorder.py::LoopRecorder._should_suppress_checkpoint``);
   both break if the loop ever rebuilds instead of appends. Service-side continuations
   (``conversation_id`` set) instead RESET the list to just the newest tool-result
   message (the same two steps, shared by the blocking and streaming drivers) — the
   reason ``capture_service_loop_messages`` must retain earlier iterations.

2. **Per-call shallow copy** (``ChatMiddlewareLayer``): every model call builds
   ``ChatContext(messages=list(messages))`` over that call's wire view (``kernel/middleware.py``) and
   the inner request sends ``context.messages`` as observed AFTER the pipeline
   (``ChatMiddlewareLayer.get_response``). SystemReminderMiddleware swaps enriched copies into
   that per-call list (system_reminder.py:249-264) and InjectionMiddleware replaces
   the whole list (injection.py:226-237); both rely on (a) the mutation reaching
   the wire and (b) never leaking back into the loop's accumulated list.

3. **ResponseStream laziness + hooks**: construction schedules nothing and
   ``stream=True`` returns the stream synchronously — the wire client's ``_inner_get_response`` must
   add hooks WITHOUT wrap-and-await or the pipeline would run eagerly.
   ``with_transform_hook`` observes every yielded update (usage.py:92),
   ``with_result_hook`` fires exactly once on finalization (wire-client result hooks),
   and ``_run_cleanup_hooks`` is a private idempotent coroutine so
   ``ResponseStream`` can safely clean up after partial consumption or a stall
   (responses.py:34-42).

4. **Session message aliasing** (``SessionContext.extend_messages``): Chrys stores
   the same ``Message`` objects without shallow-copying them, so
   intra-run annotations land directly on stored history.
"""

from __future__ import annotations

import inspect
import itertools
from typing import TYPE_CHECKING

from chrys.kernel import (
    BaseChatClient,
    ChatMiddleware,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionTool,
    Message,
    ResponseStream,
    SessionContext,
    ToolLoopLayer,
)
from chrys.kernel.compaction import EXCLUDED_KEY
from chrys.kernel.middleware import ChatContext
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence


# --------------------------------------------------------------------------- #
# Harness: a minimal client composed exactly like the real ones
# --------------------------------------------------------------------------- #


def _echo(x: str) -> str:
    return f"echo:{x}"


def _echo_tool() -> FunctionTool:
    return FunctionTool(name="echo", description="Echo back", func=_echo)


def _tool_call_response(call_id: str, *, conversation_id: str | None = None) -> ChatResponse:
    return ChatResponse(
        conversation_id=conversation_id,
        messages=[
            Message(
                role="assistant",
                contents=[{"type": "function_call", "call_id": call_id, "name": "echo", "arguments": '{"x": "a"}'}],
            )
        ],
    )


def _text_response(text: str = "done", *, conversation_id: str | None = None) -> ChatResponse:
    return ChatResponse(conversation_id=conversation_id, messages=[Message(role="assistant", contents=[text])])


def _function_call_update(call_id: str, *, conversation_id: str | None = None) -> ChatResponseUpdate:
    return ChatResponseUpdate(
        role="assistant",
        conversation_id=conversation_id,
        contents=[{"type": "function_call", "call_id": call_id, "name": "echo", "arguments": '{"x": "a"}'}],
    )


def _text_update(text: str, *, conversation_id: str | None = None) -> ChatResponseUpdate:
    return ChatResponseUpdate(
        role="assistant", conversation_id=conversation_id, contents=[{"type": "text", "text": text}]
    )


class _RecordingChatMiddleware(ChatMiddleware):
    """Snapshots ``context.messages`` per model call — the loop-recorder viewpoint."""

    def __init__(self) -> None:
        self.snapshots: list[list[Message]] = []

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        self.snapshots.append(list(context.messages))
        await call_next()


class _ScriptedWireClient(BaseChatClient):
    """Loop-free fake wire client used inside the owned production composition."""

    def __init__(
        self,
        responses: Sequence[ChatResponse] = (),
        *,
        update_rounds: Sequence[Sequence[ChatResponseUpdate]] = (),
    ) -> None:
        super().__init__()
        self._responses = list(responses)
        self._update_rounds = [list(round_updates) for round_updates in update_rounds]
        self.wire_messages: list[list[Message]] = []

    def _inner_get_response(  # type: ignore[override]
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: object,
        **kwargs: object,
    ) -> object:
        self.wire_messages.append(list(messages))
        if stream:
            # Per the BaseChatClient contract, stream=True must
            # return a ResponseStream SYNCHRONOUSLY, not a coroutine — the laziness
            # chain that the wire client preserves by never wrap-and-awaiting.
            updates = self._update_rounds.pop(0)

            async def _gen():
                for update in updates:
                    yield update

            return ResponseStream(_gen(), finalizer=ChatResponse.from_updates)
        response = self._responses.pop(0)

        async def _respond() -> ChatResponse:
            return response

        return _respond()


class _ScriptedChatClient(InvariantCheckedToolLoopLayer):
    """Production-shaped owned stack: loop outside chat middleware, raw wire inside.

    Derives from the invariant-checked layer so every final response these
    contract tests obtain also passes the transcript oracle.
    """

    def __init__(
        self,
        responses: Sequence[ChatResponse] = (),
        *,
        update_rounds: Sequence[Sequence[ChatResponseUpdate]] = (),
        middleware: Sequence[ChatMiddleware] | None = None,
        **kwargs: object,
    ) -> None:
        self.wire = _ScriptedWireClient(responses=responses, update_rounds=update_rounds)
        super().__init__(ChatMiddlewareLayer(self.wire, middleware=middleware), **kwargs)

    @property
    def wire_messages(self) -> list[list[Message]]:
        return self.wire.wire_messages


# --------------------------------------------------------------------------- #
# 0. Owned stack keeps the production layer order
# --------------------------------------------------------------------------- #


class TestOwnedClientLayerOrder:
    def test_tool_loop_wraps_chat_middleware_and_raw_wire_client(self) -> None:
        client = _ScriptedChatClient()

        assert isinstance(client, ToolLoopLayer)
        assert isinstance(client.inner, ChatMiddlewareLayer)
        assert isinstance(client.inner.inner, BaseChatClient)


# --------------------------------------------------------------------------- #
# 1. Tool loop: prepped_messages accumulates; snapshots grow by identity-prefix
# --------------------------------------------------------------------------- #


class TestToolLoopAccumulation:
    async def test_local_loop_grows_one_list_with_identity_prefixes(self) -> None:
        recorder = _RecordingChatMiddleware()
        client = _ScriptedChatClient(
            responses=[_tool_call_response("call-1"), _tool_call_response("call-2"), _text_response()],
            middleware=[recorder],
        )
        user = Message(role="user", contents=["go"])

        response = await client.get_response([user], options={"tools": [_echo_tool()]})

        assert response.text == "done"
        # One middleware pass per loop iteration; each iteration appended exactly
        # the assistant tool-call message plus the tool-result message.
        assert [len(snapshot) for snapshot in recorder.snapshots] == [1, 3, 5]
        # Below the wire boundary EVERY message arrives as a per-call VIEW —
        # caller history included: fresh wrapper + fresh contents list, SAME
        # content objects, SHARED props dict — so a client mutating a received
        # message in place can corrupt neither landed history nor caller
        # session state, while message metadata still writes through. The
        # canonical accumulated list itself still grows with strict identity
        # prefixes; LoopRecorder observes it via record_pre_call ABOVE the
        # view, so its initial_count slicing and checkpoint-suppression key
        # keep their assumption.
        for earlier, later in itertools.pairwise(recorder.snapshots):
            for a, b in zip(earlier, later, strict=False):
                assert a is not b, "every outgoing wrapper is a per-call view"
                assert a.role == b.role
                assert all(x is y for x, y in zip(a.contents, b.contents, strict=True))
        delta = recorder.snapshots[1][len(recorder.snapshots[0]) :]
        assert [message.role for message in delta] == ["assistant", "tool"]
        assert [content.type for message in delta for content in message.contents] == [
            "function_call",
            "function_result",
        ]
        # The caller's message reaches the wire as a view too: fresh wrapper,
        # same content objects, SHARED props dict (metadata write-through).
        assert recorder.snapshots[-1][0] is not user
        assert recorder.snapshots[-1][0].additional_properties is user.additional_properties
        assert all(x is y for x, y in zip(recorder.snapshots[-1][0].contents, user.contents, strict=True))

    async def test_streaming_loop_accumulates_identically(self) -> None:
        recorder = _RecordingChatMiddleware()
        client = _ScriptedChatClient(
            update_rounds=[[_function_call_update("call-1")], [_text_update("all done")]],
            middleware=[recorder],
        )
        user = Message(role="user", contents=["go"])

        stream = client.get_response([user], stream=True, options={"tools": [_echo_tool()]})
        async for _ in stream:
            pass
        final = await stream.get_final_response()

        assert final.text == "all done"
        assert [len(snapshot) for snapshot in recorder.snapshots] == [1, 3]
        first, second = recorder.snapshots
        for a, b in zip(first, second, strict=False):
            assert a is not b, "every outgoing wrapper is a per-call view"
            assert a.role == b.role
            assert all(x is y for x, y in zip(a.contents, b.contents, strict=True))
        assert second[0] is not user
        assert second[0].additional_properties is user.additional_properties
        # Streaming messages are rebuilt by the finalizer, but the appended round
        # still has the same [assistant tool-call, tool result] shape.
        assert [message.role for message in second[1:]] == ["assistant", "tool"]
        assert [content.type for message in second[1:] for content in message.contents] == [
            "function_call",
            "function_result",
        ]

    async def test_service_continuation_resets_prepped_to_latest_tool_result(self) -> None:
        """``conversation_id`` flips accumulate→reset: the next call sends ONLY the new
        tool-result message (``kernel/loop.py::_LoopRun._queue_batch``).
        ``capture_service_loop_messages`` exists to reconstruct the dropped history for
        pause-time local replay.
        """
        recorder = _RecordingChatMiddleware()
        client = _ScriptedChatClient(
            responses=[
                _tool_call_response("call-1", conversation_id="conv-1"),
                _text_response(conversation_id="conv-1"),
            ],
            middleware=[recorder],
        )

        await client.get_response([Message(role="user", contents=["go"])], options={"tools": [_echo_tool()]})

        assert [len(snapshot) for snapshot in recorder.snapshots] == [1, 1]
        (only,) = recorder.snapshots[1]
        assert only.role == "tool"
        assert [content.type for content in only.contents] == ["function_result"]

    async def test_streaming_service_continuation_resets_identically(self) -> None:
        """The streaming driver reaches the reset through its own iterations
        (``kernel/loop.py::_LoopRun.stream``); Chrys runs service-side continuations
        through the streaming path in production, so pin it separately from the
        blocking driver above.
        """
        recorder = _RecordingChatMiddleware()
        client = _ScriptedChatClient(
            update_rounds=[
                [_function_call_update("call-1", conversation_id="conv-1")],
                [_text_update("done", conversation_id="conv-1")],
            ],
            middleware=[recorder],
        )

        stream = client.get_response(
            [Message(role="user", contents=["go"])], stream=True, options={"tools": [_echo_tool()]}
        )
        async for _ in stream:
            pass
        final = await stream.get_final_response()

        assert final.text == "done"
        assert [len(snapshot) for snapshot in recorder.snapshots] == [1, 1]
        (only,) = recorder.snapshots[1]
        assert only.role == "tool"
        assert [content.type for content in only.contents] == ["function_result"]


# --------------------------------------------------------------------------- #
# 2. ChatContext.messages: fresh per-call list; provider gets final wrapper views
# --------------------------------------------------------------------------- #


class TestChatContextShallowCopy:
    async def test_context_messages_is_a_fresh_list_across_loop_iterations(self) -> None:
        seen_contexts: list[ChatContext] = []

        class Probe(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                seen_contexts.append(context)
                await call_next()

        client = _ScriptedChatClient(
            responses=[_tool_call_response("call-1"), _text_response()],
            middleware=[Probe()],
        )
        caller_messages = [Message(role="user", contents=["go"])]

        await client.get_response(caller_messages, options={"tools": [_echo_tool()]})

        first, second = seen_contexts
        # A NEW list object on EVERY call, never the caller's. Through the loop
        # this holds even without the layer's own copy, since every request is a
        # fresh wire view (``_WireCaller._call``); the layer's copy of the exact
        # list it is handed is pinned in ``test_middleware.py::TestOwnChatLayer``.
        assert first.messages is not caller_messages
        assert first.messages is not second.messages
        # Per-call views: a fresh wrapper each call, but the content objects
        # and the props dict are the caller's — mutation isolation with
        # metadata write-through.
        for ctx in (first, second):
            assert ctx.messages[0] is not caller_messages[0]
            assert ctx.messages[0].additional_properties is caller_messages[0].additional_properties
            assert all(x is y for x, y in zip(ctx.messages[0].contents, caller_messages[0].contents, strict=True))
        assert first.messages[0] is not second.messages[0]

    async def test_element_replacement_reaches_the_wire_and_resets_next_call(self) -> None:
        """The system-reminder pattern: swap an enriched copy into ``context.messages``.

        Pins both halves: (a) the wire sees the swap for THAT call, and (b) the loop's
        own list is untouched, so the next iteration presents the ORIGINAL again —
        re-enrichment per call is both required and safe.
        """
        original = Message(role="user", contents=["go"])
        replacement = Message(role="user", contents=["go [enriched]"])
        first_elements: list[Message] = []

        class Reminder(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                first_elements.append(context.messages[0])
                context.messages[0] = replacement
                await call_next()

        client = _ScriptedChatClient(
            responses=[_tool_call_response("call-1"), _text_response()],
            middleware=[Reminder()],
        )

        await client.get_response([original], options={"tools": [_echo_tool()]})

        assert len(client.wire_messages) == 2
        for wire in client.wire_messages:
            provider_view = wire[0]
            assert provider_view is not replacement
            assert provider_view.additional_properties is replacement.additional_properties
            assert all(x is y for x, y in zip(provider_view.contents, replacement.contents, strict=True))
        # Identity on purpose (Message.__eq__ is object identity today; a future
        # value-based __eq__ must not weaken this contract silently). Each call
        # presents a FRESH view of the loop's original element — the swap never
        # reaches the loop's own list, and the caller's object itself is never
        # exposed.
        assert len(first_elements) == 2
        assert first_elements[0] is not first_elements[1]
        for element in first_elements:
            assert element is not original
            assert element.additional_properties is original.additional_properties
            assert all(x is y for x, y in zip(element.contents, original.contents, strict=True))

    async def test_whole_list_replacement_reaches_the_wire_without_polluting_the_loop(self) -> None:
        """The injection pattern: ``context.messages = [*context.messages, injected]``.

        The injected message must reach the model for that call only. If the owned
        chat layer ever wrote ``context.messages`` back into ``prepped_messages``, queued
        injections would duplicate (Chrys persists them separately via on_consumed).
        """
        injected = Message(role="user", contents=["injected"])
        fired = False

        class Injector(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                nonlocal fired
                if not fired:
                    fired = True
                    context.messages = [*context.messages, injected]
                await call_next()

        recorder = _RecordingChatMiddleware()
        client = _ScriptedChatClient(
            responses=[_tool_call_response("call-1"), _text_response()],
            middleware=[Injector(), recorder],
        )

        await client.get_response([Message(role="user", contents=["go"])], options={"tools": [_echo_tool()]})

        # Call 1: middleware downstream of the injector AND the wire saw the message.
        assert recorder.snapshots[0][-1] is injected
        provider_view = client.wire_messages[0][-1]
        assert provider_view is not injected
        assert provider_view.additional_properties is injected.additional_properties
        assert all(x is y for x, y in zip(provider_view.contents, injected.contents, strict=True))
        # Call 2: the loop re-prepped from its own accumulated list — injection gone.
        assert all(message is not injected for message in recorder.snapshots[1])
        assert all(message is not injected for message in client.wire_messages[1])

    async def test_post_compaction_provider_boundary_observes_and_views_inserted_summary(self) -> None:
        """Compaction additions below chat middleware still get both guards."""

        async def run(*, stream: bool) -> None:
            summary_content = Content.from_text("COMPACTED")
            summary = Message(
                role="assistant",
                contents=[summary_content],
                additional_properties={"summary": True},
            )
            seen_provider_messages: list[Message] = []

            class EchoCompactedClient(BaseChatClient):
                def _inner_get_response(  # type: ignore[override]
                    self,
                    *,
                    messages: Sequence[Message],
                    stream: bool,
                    options: object,
                    **kwargs: object,
                ) -> object:
                    del options, kwargs
                    provider_summary = messages[0]
                    seen_provider_messages.append(provider_summary)
                    assert provider_summary is not summary
                    assert provider_summary.contents is not summary.contents
                    assert provider_summary.additional_properties is summary.additional_properties
                    assert provider_summary.contents[0] is summary_content
                    provider_summary.contents.append(Content.from_text("WIRE MUTATION"))
                    fresh = Content.from_text("done")
                    if stream:

                        async def updates():
                            yield ChatResponseUpdate(contents=[summary_content], role="assistant")
                            yield ChatResponseUpdate(contents=[fresh], role="assistant")

                        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)

                    async def response() -> ChatResponse:
                        return ChatResponse(messages=[Message("assistant", [summary_content, fresh])])

                    return response()

            async def insert_summary(messages: list[Message], _context: object) -> None:
                messages.insert(0, summary)

            layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(EchoCompactedClient()))
            result = layer.get_response(
                [Message("user", ["hi"])],
                stream=stream,
                options={},
                compaction_strategy=insert_summary,
            )
            if stream:
                async for _ in result:
                    pass
                response = await result.get_final_response()
            else:
                response = await result

            assert response.text == "done", "the identity-echoed summary must not land"
            assert summary.contents == [summary_content], "provider list mutation must not corrupt the source summary"
            assert len(seen_provider_messages) == 1

        await run(stream=False)
        await run(stream=True)


# --------------------------------------------------------------------------- #
# 3. ResponseStream: lazy resolution, hook semantics, private cleanup
# --------------------------------------------------------------------------- #


class TestResponseStreamContract:
    async def test_layered_stream_true_returns_synchronously_and_runs_nothing(self) -> None:
        recorder = _RecordingChatMiddleware()
        client = _ScriptedChatClient(update_rounds=[[_text_update("hi")]], middleware=[recorder])

        stream = client.get_response([Message(role="user", contents=["go"])], stream=True, options={})

        # A ResponseStream, synchronously — NOT a coroutine. The wire client adds its
        # result hook directly to this object; wrap-and-awaiting here would run the
        # middleware pipeline and open the wire call eagerly.
        assert isinstance(stream, ResponseStream)
        assert recorder.snapshots == []
        assert client.wire_messages == []

        async for _ in stream:
            pass
        final = await stream.get_final_response()

        assert final.text == "hi"
        assert len(recorder.snapshots) == 1
        assert len(client.wire_messages) == 1

    async def test_awaiting_resolves_the_source_without_consuming(self) -> None:
        started = False

        async def _source():
            nonlocal started
            started = True

            async def _gen():
                yield "u1"
                yield "u2"

            return _gen()

        stream: ResponseStream[str, list[str]] = ResponseStream(_source(), finalizer=list)
        assert started is False  # construction schedules nothing

        resolved = await stream

        assert resolved is stream  # ``__await__`` resolves the source and returns self
        assert started is True
        assert list(stream.updates) == []  # ...without consuming a single update
        assert [update async for update in stream] == ["u1", "u2"]

    async def test_observer_hooks_see_everything_and_result_hook_fires_once(self) -> None:
        async def _gen():
            yield 1
            yield 2

        stream: ResponseStream[int, int] = ResponseStream(_gen(), finalizer=sum)
        seen_updates: list[int] = []
        seen_results: list[int] = []
        # Observer hooks return None → ResponseStream keeps the original value
        # (the usage.py / loop-recorder / wire-client pattern).
        stream.with_transform_hook(seen_updates.append)  # type: ignore[arg-type]
        stream.with_result_hook(seen_results.append)  # type: ignore[arg-type]

        yielded = [update async for update in stream]
        final = await stream.get_final_response()

        assert yielded == [1, 2]
        assert seen_updates == [1, 2]
        assert final == 3
        # Exactly once: auto-finalization at exhaustion fired it; the explicit
        # get_final_response() above returned the cached result.
        assert seen_results == [3]

    async def test_update_filter_applies_before_accumulation_for_every_observer(self) -> None:
        async def _gen():
            yield 1
            yield 2

        stream: ResponseStream[int, int] = ResponseStream(_gen(), finalizer=sum)
        hook_saw: list[int] = []
        stream.with_update_filter(lambda update: update * 10)
        stream.with_transform_hook(hook_saw.append)  # type: ignore[arg-type]

        yielded = [update async for update in stream]
        final = await stream.get_final_response()

        # One filtered sequence everywhere: accumulation (finalizer input),
        # transform hooks, and iteration — unlike transform hooks, which run
        # after the update already joined the accumulated list.
        assert yielded == [10, 20]
        assert hook_saw == [10, 20]
        assert list(stream.updates) == [10, 20]
        assert final == 30

    async def test_update_filter_pushes_down_to_the_innermost_stream_exactly_once(self) -> None:
        # The wire shape: ChatMiddlewareLayer and BaseChatClient both wrap
        # with from_awaitable, so the finalizer that builds the final
        # response lives two levels down. A filter registered on the
        # OUTERMOST stream must still shape that innermost assembly — and
        # each update must pass through the filter exactly once.
        applications: list[int] = []

        def counting_filter(update: int) -> int:
            applications.append(update)
            return update * 10

        async def _gen():
            yield 1
            yield 2

        inner: ResponseStream[int, int] = ResponseStream(_gen(), finalizer=sum)
        inner_hook_saw: list[int] = []
        inner.with_result_hook(inner_hook_saw.append)  # type: ignore[arg-type]

        async def _resolve_inner() -> ResponseStream[int, int]:
            return inner

        async def _resolve_mid() -> ResponseStream[int, int]:
            return ResponseStream.from_awaitable(_resolve_inner())

        outer: ResponseStream[int, int] = ResponseStream.from_awaitable(_resolve_mid())
        outer.with_update_filter(counting_filter)

        yielded = [update async for update in outer]
        final = await outer.get_final_response()

        assert yielded == [10, 20]
        assert applications == [1, 2], "each update filtered exactly once, at the innermost level"
        assert final == 30, "the innermost finalizer assembled the filtered updates"
        assert inner_hook_saw == [30], "result hooks observe the filtered assembly"

    async def test_update_filter_registered_after_resolution_still_reaches_the_source(self) -> None:
        async def _gen():
            yield 1
            yield 2

        inner: ResponseStream[int, int] = ResponseStream(_gen(), finalizer=sum)

        async def _resolve_inner() -> ResponseStream[int, int]:
            return inner

        outer: ResponseStream[int, int] = ResponseStream.from_awaitable(_resolve_inner())
        await outer  # resolves the source before any filter exists
        outer.with_update_filter(lambda update: update * 10)

        yielded = [update async for update in outer]

        assert yielded == [10, 20]
        assert await outer.get_final_response() == 30

    async def test_update_filter_on_mapped_stream_applies_after_the_map(self) -> None:
        async def _gen():
            yield 1
            yield 2

        inner: ResponseStream[int, int] = ResponseStream(_gen(), finalizer=sum)
        mapped = inner.map(str, finalizer=",".join)
        # The mapped stream's filter expects the POST-map type; it must stay
        # local instead of being pushed down to the int-typed inner stream.
        mapped.with_update_filter(lambda update: f"<{update}>")

        yielded = [update async for update in mapped]

        assert yielded == ["<1>", "<2>"]
        assert await mapped.get_final_response() == "<1>,<2>"

    async def test_cleanup_hooks_auto_run_once_on_exhaustion(self) -> None:
        runs: list[str] = []

        async def _gen():
            yield 1

        stream: ResponseStream[int, list[int]] = ResponseStream(_gen(), finalizer=list)
        stream.with_cleanup_hook(lambda: runs.append("ran"))

        async for _ in stream:
            pass

        assert runs == ["ran"]

    async def test_run_cleanup_hooks_is_private_idempotent_and_safe_mid_stream(self) -> None:
        """ResponseStream cleanup must remain awaitable and idempotent mid-stream,
        so closing after a stall is safe even if cleanup already ran (responses.py:34-42).
        """
        assert inspect.iscoroutinefunction(ResponseStream._run_cleanup_hooks)

        runs: list[str] = []

        async def _gen():
            yield 1
            yield 2

        stream: ResponseStream[int, list[int]] = ResponseStream(_gen(), finalizer=list)
        stream.with_cleanup_hook(lambda: runs.append("ran"))
        cleanup = getattr(stream, "_run_cleanup_hooks", None)
        assert cleanup is not None

        assert await anext(stream) == 1  # stall scenario: partially consumed
        await cleanup()
        assert runs == ["ran"]

        async for _ in stream:  # natural exhaustion must not re-run the hooks
            pass
        await stream._run_cleanup_hooks()
        assert runs == ["ran"]


# --------------------------------------------------------------------------- #
# 4. SessionContext.extend_messages: stored history aliases the input messages
# --------------------------------------------------------------------------- #


class TestSessionExtendMessagesAliases:
    def test_stored_history_keeps_the_same_message_object(self) -> None:
        original = Message(role="assistant", contents=["hello"])
        original.additional_properties["preexisting"] = "kept"
        context = SessionContext(input_messages=[])

        context.extend_messages("some_provider", [original])

        (stored,) = context.context_messages["some_provider"]
        assert stored is original
        assert stored.contents is original.contents
        assert stored.additional_properties is original.additional_properties
        assert stored.additional_properties["preexisting"] == "kept"

    def test_flags_set_after_extend_reach_the_original(self) -> None:
        """Chrys intentionally aliases messages so compaction annotations land directly."""
        original = Message(role="assistant", contents=["hello"])
        context = SessionContext(input_messages=[])
        context.extend_messages("some_provider", [original])
        (stored,) = context.context_messages["some_provider"]

        stored.additional_properties[EXCLUDED_KEY] = True

        assert original.additional_properties[EXCLUDED_KEY] is True
