# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A consumer closing the loop stream mid-response closes the provider stream.

The provider stream's own cleanup hooks (usage, telemetry) run only when that
stream is closed; garbage collection would close just the bare generator under
it. So the close has to reach the provider stream before it returns, through
both the run and the logical call it was reading. Finalization, which closes
each abandoned generator in a task of its own, must not have one loop
generator close another.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any

import pytest

import chrys.kernel.loop as loop_module
from chrys.foundation.trajectory.context import TRAJECTORY_CONTEXT_KWARG
from chrys.foundation.trajectory.event_types import EventType, ExchangeOutcome
from chrys.kernel import (
    ChatContext,
    ChatMiddleware,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
    tool,
)
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer


def _text(text: str, *, continuation_token: dict[str, str] | None = None) -> ChatResponseUpdate:
    return ChatResponseUpdate(
        role="assistant",
        contents=[Content.from_text(text)],
        continuation_token=continuation_token,
    )


def _call(call_id: str) -> ChatResponseUpdate:
    return ChatResponseUpdate(
        role="assistant",
        contents=[Content.from_function_call(call_id, "echo", arguments={"text": "x"})],
    )


@tool(name="echo")
async def _echo(text: str) -> str:
    return f"echo:{text}"


class _StreamEndsClient:
    """Client streaming one scripted round per request, recording how each stream ends."""

    def __init__(self, rounds: list[list[ChatResponseUpdate]]) -> None:
        self._rounds = list(rounds)
        self.ends: list[list[str]] = []

    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool,
        options: Mapping[str, Any] | None,
        function_invocation_kwargs: Mapping[str, Any] | None,
        compaction_strategy: object,
        tokenizer: object,
        client_kwargs: Mapping[str, Any] | None,
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
        del messages, options, function_invocation_kwargs, compaction_strategy, tokenizer, client_kwargs
        assert stream is True
        updates = self._rounds.pop(0)
        ends: list[str] = []
        self.ends.append(ends)

        async def _provider() -> AsyncIterator[ChatResponseUpdate]:
            try:
                for update in updates:
                    yield update
            finally:
                # A real provider's teardown suspends: it closes a connection.
                await asyncio.sleep(0)
                ends.append("provider closed")

        def _finalize(collected: Sequence[ChatResponseUpdate]) -> ChatResponse:
            ends.append("finalized")
            return ChatResponse.from_updates(collected)

        return ResponseStream(_provider(), finalizer=_finalize).with_cleanup_hook(lambda: ends.append("cleanup"))


class _CleanupCountingChat(ChatMiddleware):
    """Pass-through chat middleware counting the cleanups of the stream it hands on."""

    def __init__(self) -> None:
        self.cleanups = 0

    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        await call_next()
        if isinstance(context.result, ResponseStream):
            context.result.with_cleanup_hook(self._count_cleanup)

    def _count_cleanup(self) -> None:
        self.cleanups += 1


@pytest.mark.asyncio
@pytest.mark.parametrize("with_middleware", [False, True], ids=["bare", "chat_middleware"])
async def test_closing_mid_response_closes_the_provider_stream_before_returning(with_middleware: bool) -> None:
    # Production always runs chat middleware, whose result stream wraps the
    # provider stream: the close has to pass through it too.
    client = _StreamEndsClient([[_text("one"), _text("two")]])
    middleware = _CleanupCountingChat()
    layer = InvariantCheckedToolLoopLayer(
        ChatMiddlewareLayer(client, middleware=[middleware] if with_middleware else None)
    )
    sink = FakeSink()

    stream = layer.get_response(
        [Message("user", ["hi"])],
        stream=True,
        client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
    )
    async for update in stream:
        assert update.text == "one"
        break
    await stream.aclose()

    assert client.ends == [["provider closed", "cleanup"]]
    assert middleware.cleanups == (1 if with_middleware else 0)
    assert sink.only(EventType.MODEL_CYCLE_FINISHED).payload["outcome"] == ExchangeOutcome.ABANDONED
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_closing_mid_continuation_poll_closes_the_poll_stream_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loop_module, "CONTINUATION_POLL_INTERVAL_SECONDS", 0.0)
    client = _StreamEndsClient(
        [[_text("started", continuation_token={"id": "resp_1"})], [_text("poll one"), _text("poll two")]]
    )
    layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(client))

    stream = layer.get_response([Message("user", ["hi"])], stream=True)
    async for update in stream:
        if update.text == "poll one":
            break
    await stream.aclose()

    # One logical call, two provider streams: the one that started the
    # background response ran to its end; the poll the consumer left is
    # closed and cleaned up, never finalized.
    assert client.ends == [
        ["provider closed", "cleanup", "finalized"],
        ["provider closed", "cleanup"],
    ]


@pytest.mark.asyncio
async def test_closing_mid_exhaustion_tail_closes_the_tail_provider_stream_before_returning() -> None:
    client = _StreamEndsClient([[_call("c1")], [_text("tail one"), _text("tail two")]])
    layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(client), max_iterations=1)

    stream = layer.get_response([Message("user", ["hi"])], stream=True, options={"tools": [_echo]})
    async for update in stream:
        if update.text == "tail one":
            break
    await stream.aclose()

    # The first stream ran to its end, so it was finalized and cleaned up once;
    # the tail stream the consumer left is closed and cleaned up, never finalized.
    assert client.ends == [
        ["provider closed", "cleanup", "finalized"],
        ["provider closed", "cleanup"],
    ]


def test_a_stream_left_open_at_loop_shutdown_closes_each_generator_on_its_own() -> None:
    # ``loop.shutdown_asyncgens`` (like garbage collection) closes every
    # abandoned generator in a task of its own. A generator whose exit closed
    # another one would collide with that one's own close once a provider
    # teardown suspends ("aclose(): asynchronous generator is already
    # running"), or run the provider stream's cleanup before its generator
    # finished. Only an explicit close of the stream crosses generators.
    client = _StreamEndsClient([[_text("one"), _text("two")]])
    layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(client))
    sink = FakeSink()
    reported: list[dict[str, Any]] = []
    left_open: list[object] = []

    async def _read_one_update_and_leave() -> None:
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: reported.append(context))
        stream = layer.get_response(
            [Message("user", ["hi"])],
            stream=True,
            client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
        )
        # Kept alive, so the shutdown closes its generators, not the GC.
        left_open.append(stream)
        async for update in stream:
            assert update.text == "one"
            break

    asyncio.run(_read_one_update_and_leave())

    assert reported == []
    assert client.ends == [["provider closed"]]
    assert sink.only(EventType.MODEL_CYCLE_FINISHED).payload["outcome"] == ExchangeOutcome.ABANDONED
