# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared wire doubles, builders and middleware probes for the kernel loop tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from chrys.foundation.retry import StreamStall
from chrys.foundation.tool_result_metadata import TOOL_ERROR_KIND_METADATA_KEY, TOOL_ERROR_MESSAGE_METADATA_KEY
from chrys.kernel import FunctionTool, StallExhaustedAction, tool
from chrys.kernel.loop import ToolLoopLayer
from chrys.kernel.middleware import (
    ChatMiddleware,
    ChatMiddlewareLayer,
    FunctionInvocationContext,
    FunctionMiddleware,
    MiddlewareTermination,
)
from chrys.kernel.types import ChatResponse, ChatResponseUpdate, Content, Message, ResponseStream
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _user(text: str = "hi") -> Message:
    return Message(role="user", contents=[text])


def _text_response(text: str = "done", **kwargs: Any) -> ChatResponse:
    return ChatResponse(messages=[Message(role="assistant", contents=[text])], **kwargs)


def _call_response(*calls: tuple[str, str, dict[str, Any]], **kwargs: Any) -> ChatResponse:
    """Assistant response containing function_call contents (call_id, name, args)."""
    contents = [Content.from_function_call(call_id=cid, name=name, arguments=args) for cid, name, args in calls]
    return ChatResponse(messages=[Message(role="assistant", contents=contents)], **kwargs)


def _text_update(text: str) -> ChatResponseUpdate:
    return ChatResponseUpdate(contents=[{"type": "text", "text": text}], role="assistant")


def _call_update(cid: str, name: str, args: dict[str, Any]) -> ChatResponseUpdate:
    return ChatResponseUpdate(
        contents=[Content.from_function_call(call_id=cid, name=name, arguments=args)], role="assistant"
    )


class _ScriptedClient:
    """Innermost fake wire client.

    Non-stream turns are ``ChatResponse`` objects; stream turns are lists of
    ``ChatResponseUpdate``. An exception as a turn fails the call, and one in
    a stream turn fails the stream there. Records every call's
    messages/options/kwargs.
    """

    def __init__(self, turns: list[Any], *, result_hook: Callable[[ChatResponse], ChatResponse] | None = None) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []
        self.result_hook = result_hook

    def get_response(
        self,
        messages: Any,
        *,
        stream: bool = False,
        options: Any = None,
        function_invocation_kwargs: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            {
                "messages": list(messages),
                "options": dict(options or {}),
                "kwargs": dict(kwargs),
                "stream": stream,
            }
        )
        turn = self.turns.pop(0)
        if not stream:

            async def _resolve() -> ChatResponse:
                if isinstance(turn, BaseException):
                    raise turn
                return turn

            return _resolve()

        async def _gen() -> Any:
            for update in [turn] if isinstance(turn, BaseException) else turn:
                if isinstance(update, BaseException):
                    raise update
                yield update

        rs: ResponseStream[ChatResponseUpdate, ChatResponse] = ResponseStream(
            _gen(), finalizer=ChatResponse.from_updates
        )
        if self.result_hook is not None:
            rs.with_result_hook(self.result_hook)
        return rs


class _OverflowSink:
    """A compaction strategy reduced to the context-overflow note; it records each note."""

    def __init__(self, *, resend_helps: bool = True) -> None:
        self.resend_helps = resend_helps
        self.notes: list[BaseException | None] = []

    async def __call__(self, messages: list[Message], context: Any = None) -> bool:
        return False

    def note_context_overflow(self, exc: BaseException | None = None) -> bool:
        self.notes.append(exc)
        return self.resend_helps


@dataclass
class _WireRetryPolicy:
    """Local-storage wire retry: retries connection drops and stalls at once, never a provider rejection."""

    max_retries: int = 2
    stall_timeout_seconds: float | None = None
    stall_max_retries: int = 0
    stall_exhausted_action: StallExhaustedAction = StallExhaustedAction.BLOCKING_FALLBACK
    retries: list[tuple[int, int, int, BaseException]] = field(default_factory=list)
    """``(attempt, max_attempts, delay_seconds, exc)`` per announced retry."""
    before_retry_calls: int = 0
    hosted_in_flight: tuple[str, ...] = ()
    """Hosted tool calls the provider already ran in the failed attempt."""

    def backoff_seconds(self, _attempt: int) -> int:
        return 0

    def hosted_commits_in_flight(self) -> tuple[str, ...]:
        return self.hosted_in_flight

    def is_retryable(self, exc: BaseException) -> bool:
        return isinstance(exc, ConnectionError | StreamStall)

    def is_interrupted(self) -> bool:
        return False

    async def sleep(self, seconds: int) -> bool:
        return False

    async def on_retry(self, message: str, attempt: int, max_attempts: int, delay: int, exc: BaseException) -> None:
        self.retries.append((attempt, max_attempts, delay, exc))

    def before_retry(self) -> None:
        self.before_retry_calls += 1


def _stack(
    turns: list[Any],
    *,
    result_hook: Callable[[ChatResponse], ChatResponse] | None = None,
    **layer_kwargs: Any,
) -> tuple[ToolLoopLayer, _ScriptedClient]:
    """The production composition: ToolLoopLayer(ChatMiddlewareLayer(wire)).

    The returned layer transparently checks the transcript invariants on
    every final response (see ``InvariantCheckedToolLoopLayer``).
    """
    wire = _ScriptedClient(turns, result_hook=result_hook)
    return InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire), **layer_kwargs), wire


def _make_tool(events: list[str] | None = None, *, name: str = "echo") -> FunctionTool:
    @tool(name=name)
    async def echo(text: str) -> str:
        if events is not None:
            events.append(f"tool:{name}:{text}")
        return f"echo:{text}"

    return echo


async def _final_response(
    layer: ToolLoopLayer, messages: list[Message], *, stream: bool, **kwargs: Any
) -> ChatResponse:
    """Drive ``layer.get_response`` in either mode and return the loop's final response."""
    if not stream:
        return await layer.get_response(messages, **kwargs)
    response_stream = layer.get_response(messages, stream=True, **kwargs)
    async for _update in response_stream:
        pass
    return await response_stream.get_final_response()


class _NullableValue(BaseModel):
    value: str | None


class _DivergentDefaults(BaseModel):
    count: int = 7


class _AliasedNullable(BaseModel):
    """``serialize_by_alias``: dumps are keyed by the wire alias, not the field name."""

    model_config = ConfigDict(serialize_by_alias=True)

    value: str | None = Field(alias="wire")


def _result_contents(response: ChatResponse) -> list[Content]:
    return [item for msg in response.messages for item in msg.contents if item.type == "function_result"]


def _assert_actionable_argument_error(result: Content, tool_name: str, *expected_parts: str) -> str:
    result_text = str(result.result)
    assert result_text.startswith(f"Error: Invalid arguments for '{tool_name}':")
    assert "Expected:" in result_text
    for expected in expected_parts:
        assert expected in result_text
    assert result.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_parsing"
    assert result.additional_properties[TOOL_ERROR_MESSAGE_METADATA_KEY] == result_text.removeprefix("Error: ")
    return result_text


class _ProbeFunction(FunctionMiddleware):
    def __init__(self) -> None:
        self.contexts: list[FunctionInvocationContext] = []

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        self.contexts.append(context)
        await call_next()


class _ProbeChat(ChatMiddleware):
    def __init__(self) -> None:
        self.calls = 0

    async def process(self, context: Any, call_next: Callable[[], Awaitable[None]]) -> None:
        self.calls += 1
        await call_next()


class _TerminateFunction(FunctionMiddleware):
    def __init__(self, *, result: Any = None, exc_result: Any = None) -> None:
        self._result = result
        self._exc_result = exc_result

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        if self._result is not None:
            context.result = self._result
        raise MiddlewareTermination("stop", result=self._exc_result)


def _provenance_tool(*, name: str = "echo", kind: str = "probe.kind", static: dict | None = None) -> FunctionTool:
    from chrys.foundation.tool_call_context import set_tool_context
    from chrys.foundation.tool_kinds import set_tool_kind

    t = _make_tool(name=name)
    set_tool_kind(t, kind)
    if static is not None:
        set_tool_context(t, static)
    return t


def _call_contents(response: ChatResponse) -> list[Content]:
    return [item for msg in response.messages for item in msg.contents if item.type == "function_call"]


def _ordinals(response: ChatResponse) -> list[int | None]:
    from chrys.foundation.tool_invocation_order import read_tool_invocation_order

    return [read_tool_invocation_order(c.additional_properties) for c in _call_contents(response)]


class _CarriageFunction(FunctionMiddleware):
    """Writes result metadata + a builder context subkey, like the event middleware."""

    def __init__(self, *, result_metadata: dict[str, Any] | None = None, tool_context: dict[str, Any] | None = None):
        self._result_metadata = result_metadata
        self._tool_context = tool_context

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        await call_next()
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        if self._result_metadata is not None:
            context.metadata[TOOL_RESULT_METADATA_KEY] = dict(self._result_metadata)
        if self._tool_context is not None:
            context.metadata[TOOL_CALL_CONTEXT_METADATA_KEY] = dict(self._tool_context)
