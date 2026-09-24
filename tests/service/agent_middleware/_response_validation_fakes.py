# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Scripted call_next, message builders, and context factories shared by the response-validation middleware tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

from chrys.foundation.hosted_tools import HostedToolPhase
from chrys.foundation.trajectory.context import TRAJECTORY_EXCHANGE_KWARG, ExchangeTrace
from chrys.kernel import (
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)
from chrys.kernel.middleware import ChatContext


def _assistant(contents: list[Any]) -> ChatResponse:
    """Build a minimal ChatResponse with a single assistant message."""
    return ChatResponse(messages=[Message(role="assistant", contents=contents)], finish_reason="stop")


def _assistant_truncated(contents: list[Any]) -> ChatResponse:
    """Assistant response that hit the output token limit (finish_reason='length')."""
    return ChatResponse(messages=[Message(role="assistant", contents=contents)], finish_reason="length")


# The three distinct invalid shapes the default validator catches.  Used
# to construct exhaustion sequences that walk through different
# ``ValidationResult.reason`` strings on consecutive attempts so the
# middleware's fail-fast (identical-reason short-circuit) does not fire
# and the loop actually reaches MAX_RETRIES.
def _bad_empty() -> ChatResponse:
    return _assistant([])  # reason: "empty contents"


def _bad_whitespace() -> ChatResponse:
    return _assistant([Content.from_text("\n\n  \t\n")])  # reason: "empty or whitespace-only text response"


def _bad_leaked() -> ChatResponse:
    return _assistant(
        [Content.from_text("minimax:tool_call {} </minimax:tool_call>")]
    )  # reason: "leaked tool-call marker..."


def _search_without_final_text(*, intermediate_text: str = "Checking sources.") -> ChatResponse:
    return _assistant(
        [
            Content.from_text(intermediate_text),
            Content.from_search_tool_call(
                "ws_1",
                tool_name="web_search",
                status="completed",
                provider_phase=HostedToolPhase.TERMINAL,
                provider_status="completed",
            ),
            Content.from_search_tool_result(
                "ws_1",
                tool_name="web_search",
                status="completed",
                provider_phase=HostedToolPhase.TERMINAL,
                provider_status="completed",
                result={"query": "Chrys"},
            ),
        ]
    )


def _varied_bads(n: int, *, last: ChatResponse | None = None) -> list[ChatResponse]:
    """Build n bad responses cycling through distinct reasons.

    Each consecutive pair has a different ``ValidationResult.reason`` so
    fail-fast does not short-circuit the retry loop.  When ``last`` is
    given, it overrides the final response — useful for tests that need
    the give-up path to scrub a specific shape (e.g. drop an empty
    assistant message).
    """
    factories = [_bad_empty, _bad_whitespace, _bad_leaked]
    out = [factories[i % len(factories)]() for i in range(n)]
    if last is not None and out:
        out[-1] = last
    return out


def _make_context(stream: bool = False) -> ChatContext:
    return ChatContext(
        client=None,  # type: ignore[arg-type]
        messages=[Message(role="user", contents=[Content.from_text("hi")])],
        options=None,
        stream=stream,
    )


def _response_stream_from(response: ChatResponse) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
    """Build a fake ResponseStream that yields one update per content and finalizes to *response*."""
    msg = response.messages[-1] if response.messages else Message(role="assistant", contents=[])
    updates = [ChatResponseUpdate(contents=msg.contents or [], role="assistant")]

    async def _gen() -> AsyncIterator[ChatResponseUpdate]:
        for u in updates:
            yield u

    return ResponseStream(_gen(), finalizer=lambda _u: response)


def _semantic_updates(updates: list[ChatResponseUpdate]) -> list[ChatResponseUpdate]:
    """Exclude raw-only transport heartbeats from response-content assertions."""
    return [
        update
        for update in updates
        if update.contents
        or update.role is not None
        or update.author_name is not None
        or update.response_id is not None
        or update.message_id is not None
        or update.conversation_id is not None
        or update.model is not None
        or update.created_at is not None
        or update.finish_reason is not None
        or update.continuation_token is not None
        or update.additional_properties
    ]


class _FakeCallNext:
    """Simulates a middleware chain where each ``call_next()`` sets ``context.result``
    to the next scripted response — either a ``ChatResponse`` (non-stream) or
    a fresh ``ResponseStream`` (stream).
    """

    def __init__(self, responses: list[ChatResponse], *, stream: bool) -> None:
        self._responses = list(responses)
        self._stream = stream
        self._call_count = 0
        self.context: ChatContext | None = None

    @property
    def call_count(self) -> int:
        return self._call_count

    def bind(self, context: ChatContext) -> None:
        self.context = context

    async def __call__(self) -> None:
        assert self.context is not None
        idx = min(self._call_count, len(self._responses) - 1)
        resp = self._responses[idx]
        self._call_count += 1
        if self._stream:
            self.context.result = _response_stream_from(resp)
        else:
            self.context.result = resp


class _ObservationHook:
    """Records the optional validation observation contract in call order."""

    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    async def begin_response(self, *, response_index: int | None = None, batch_id: int | None = None) -> None:
        self.events.append(("response", response_index, batch_id))

    async def observe_contents(self, contents: list[Any], *, is_final: bool = False) -> None:
        self.events.append(("contents", is_final, tuple(content.type for content in contents)))

    async def attempt_started(self, *, continuation: bool = False) -> None:
        self.events.append(("started", continuation))

    async def attempt_rejected(self, reason: str = "") -> None:
        self.events.append(("rejected", reason))

    async def attempt_accepted(self, messages: Sequence[Message]) -> None:
        self.events.append(
            (
                "accepted",
                tuple(tuple(content.type for content in message.contents) for message in messages),
            )
        )


def _bad_hosted_mcp() -> ChatResponse:
    return _assistant(
        [
            Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github"),
            Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("created #42")]),
            Content.from_text("<tool_use>malformed</tool_use>"),
        ]
    )


def _service_context(stream: bool = False, exchange: ExchangeTrace | None = None) -> ChatContext:
    """Service-storage ``ChatContext``: a non-storing client asked to ``store`` via ``extra_body``.

    ``exchange`` threads a trajectory exchange through ``client_kwargs`` the
    way the executor does, so verdict recording can be asserted on a sink.
    """
    client_kwargs: dict[str, Any] = {} if exchange is None else {TRAJECTORY_EXCHANGE_KWARG: exchange}
    return ChatContext(
        client=SimpleNamespace(STORES_BY_DEFAULT=False),
        messages=[Message("user", ["hi"])],
        options={"extra_body": {"store": True}},
        stream=stream,
        kwargs={"client_kwargs": client_kwargs},
    )


async def _final_response(ctx: ChatContext, *, stream: bool) -> ChatResponse:
    """Resolve ``ctx.result`` to the final ``ChatResponse`` in either mode.

    Streaming results are finalized through the validating proxy (which
    consumes any updates not yet pulled); blocking results are returned as is.
    """
    if stream:
        assert isinstance(ctx.result, ResponseStream)
        return await ctx.result.get_final_response()
    assert isinstance(ctx.result, ChatResponse)
    return ctx.result
