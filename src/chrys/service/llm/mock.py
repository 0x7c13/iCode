# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""MockChatClient — scriptable chat client for replay and testing.

Supports both scripted responses (from a list of canned replies) and
streaming simulation with configurable delays. Useful for:
- Replaying saved sessions without hitting a real LLM
- Testing agent pipelines end-to-end
- UI development without API keys

Usage::

    from chrys.service.llm.mock import MockChatClient, MockResponse

    client = MockChatClient(responses=[
        MockResponse(text="Hello! How can I help?"),
        MockResponse(tool_calls=[("read_file", "call_1", {"path": "foo.py"})]),
        MockResponse(text="Here's the file content."),
    ])
    # Use with Agent like any other client
    agent = Agent(client=client, ...)
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from chrys.foundation.trajectory.context import TRAJECTORY_EXCHANGE_KWARG
from chrys.kernel import (
    BaseChatClient,
    ChatMiddlewareLayer,
    FinishReason,
    FinishReasonLiteral,
    ToolLoopLayer,
    in_internal_side_call,
    split_middleware,
)
from chrys.kernel.client import _prepare_provider_request_messages, _PreparedRequestObserverClient
from chrys.kernel.types import ChatResponse, ChatResponseUpdate, Content, Message, ResponseStream, UsageDetails
from chrys.service.llm.instrumented import (
    _count_function_calls,
    _ExchangeObserver,
    _extract_intermediate_text,
    resolve_exchange_trace,
)
from chrys.service.trajectory.revisions import record_context_revision

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Awaitable, Callable, Sequence

    from chrys.kernel.compaction import TokenizerProtocol


@dataclass
class MockResponse:
    """A single scripted response the mock client will return.

    Specify either ``text`` for a plain text reply, or ``tool_calls`` for
    function call responses, or both.

    Attributes:
        text: Plain text response content.
        reasoning_text: Hidden reasoning content (``text_reasoning``) emitted
            before the visible text — set it alone to script a reasoning-only
            response.
        tool_calls: List of (function_name, call_id, arguments) tuples.
        finish_reason: Why generation stopped ("stop", "tool_calls", etc.).
        chunk_size: Characters per streaming chunk (0 = single chunk).
        chunk_delay: Seconds between streaming chunks.
        delay: Seconds to wait before returning (non-streaming path).
        model_id: Model ID to report in the response.
        conversation_id: Optional provider-side conversation ID to report.
        usage_details: Optional token usage payload to attach to the response.
    """

    text: str = ""
    reasoning_text: str = ""
    tool_calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    finish_reason: str = ""
    chunk_size: int = 0
    chunk_delay: float = 0.02
    delay: float = 0
    model_id: str = "mock-model"
    conversation_id: str = ""
    usage_details: UsageDetails | None = None


class _LoopInnerAdapter(_PreparedRequestObserverClient):
    """Present the mock's wire surface as the tool loop's inner client.

    ``MockChatClient.get_response`` routes through an internal
    ``ToolLoopLayer → ChatMiddlewareLayer`` stack; this adapter closes the cycle
    back to the mock's own wire method.
    """

    def __init__(self, mock: MockChatClient) -> None:
        self._mock = mock

    async def aclose(self) -> None:
        """Nothing to release: the adapter closes the mock's own internal cycle."""

    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        compaction_overrides = self._mock._resolve_compaction_overrides(
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
        )
        merged_client_kwargs = dict(kwargs.pop("client_kwargs", None) or {})
        merged_client_kwargs.update(kwargs)
        options_snapshot, sanitized_client_kwargs, kwargs_cap_ceiling = self._mock._normalize_wire_inputs(
            options,
            merged_client_kwargs,
        )

        if not compaction_overrides or not isinstance(options_snapshot, Mapping):
            wire_messages = _prepare_provider_request_messages(messages, request_message_observer)
            return self._mock._inner_get_response(
                messages=wire_messages,
                stream=stream,
                options=options_snapshot,
                **sanitized_client_kwargs,
            )

        # Same per-call context as BaseChatClient.get_response, so mock-stack
        # Phase 4 exercises the cache-safe completer path (with the same
        # store-mode gate) instead of silently diverging to the fallback.
        call_context = self._mock._build_compaction_call_context(
            stream=stream,
            options=options_snapshot,
            client_kwargs=sanitized_client_kwargs,
            tokenizer=compaction_overrides.get("tokenizer"),
            kwargs_cap_ceiling=kwargs_cap_ceiling,
        )

        if stream:

            async def _get_stream() -> ResponseStream[ChatResponseUpdate, ChatResponse[Any]]:
                prepared_messages, wire_options = await self._mock._prepare_wire_call(
                    messages,
                    call_context=call_context,
                    options_snapshot=options_snapshot,
                    sanitized_client_kwargs=sanitized_client_kwargs,
                    compaction_overrides=compaction_overrides,
                    request_message_observer=request_message_observer,
                )
                stream_response = self._mock._inner_get_response(
                    messages=prepared_messages,
                    stream=True,
                    options=wire_options,
                    **sanitized_client_kwargs,
                )
                if isinstance(stream_response, ResponseStream):
                    # The response-model parameter is erased at runtime; every
                    # streaming mock response uses this same ResponseStream class.
                    return cast("ResponseStream[ChatResponseUpdate, ChatResponse[Any]]", stream_response)
                awaited_stream_response = await stream_response
                if isinstance(awaited_stream_response, ResponseStream):
                    return awaited_stream_response
                raise ValueError("Streaming responses must return a ResponseStream.")

            return ResponseStream.from_awaitable(_get_stream())

        async def _get_response() -> ChatResponse[Any]:
            prepared_messages, wire_options = await self._mock._prepare_wire_call(
                messages,
                call_context=call_context,
                options_snapshot=options_snapshot,
                sanitized_client_kwargs=sanitized_client_kwargs,
                compaction_overrides=compaction_overrides,
                request_message_observer=request_message_observer,
            )
            return cast(
                "ChatResponse[Any]",
                await self._mock._inner_get_response(
                    messages=prepared_messages,
                    stream=False,
                    options=wire_options,
                    **sanitized_client_kwargs,
                ),
            )

        return _get_response()


class MockChatClient(BaseChatClient):
    """Scriptable chat client that returns canned responses in order.

    Drives the chrys-owned tool loop internally: ``get_response`` routes
    through ``ToolLoopLayer(ChatMiddlewareLayer(...))`` — the same stack shape
    as real clients built by ``chrys.service.llm.instrumented`` — so agent tool loops
    and chat middleware work automatically.

    When all scripted responses are exhausted, returns a default
    "No more scripted responses" message.

    Args:
        responses: Ordered list of responses to return.
        default_model_id: Model ID reported when none specified in response.
        middleware: Chrys chat/function middleware wired into the internal
            stack (chat → ``ChatMiddlewareLayer``, function → ``ToolLoopLayer``).
    """

    def __init__(
        self,
        responses: Sequence[MockResponse] | None = None,
        default_model_id: str = "mock-model",
        middleware: Sequence[Any] | None = None,
        on_intermediate_text_async: Callable[[str], Awaitable[None]] | None = None,
        on_intermediate_text_sync: Callable[[str], None] | None = None,
        tool_result_ceiling_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self._responses: list[MockResponse] = list(responses) if responses else []
        self._default_model_id = default_model_id
        self._call_index: int = 0
        self._call_history: list[tuple[Sequence[Message], Mapping[str, Any]]] = []
        self._on_intermediate_text_async = on_intermediate_text_async
        self._on_intermediate_text_sync = on_intermediate_text_sync
        split = split_middleware(list(middleware) if middleware else None)
        self._tool_loop = ToolLoopLayer(
            ChatMiddlewareLayer(_LoopInnerAdapter(self), middleware=split.chat),
            middleware=split.function,
            tool_result_ceiling_tokens=tool_result_ceiling_tokens,
        )

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = False,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
    ) -> Awaitable[ChatResponse[Any]]: ...

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse[Any]]: ...

    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
    ) -> Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]:
        """Route through the internal tool-loop stack (run-contract §3 shapes)."""
        result = self._tool_loop.get_response(
            messages,
            stream=stream,
            options=options,
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
            function_invocation_kwargs=function_invocation_kwargs,
            client_kwargs=client_kwargs,
            request_message_observer=request_message_observer,
        )
        # ``ChatResponse``'s response-model parameter is erased at runtime;
        # the mock forwards the same ToolLoop result for every option shape.
        return cast(
            "Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]",
            result,
        )

    @property
    def call_count(self) -> int:
        """Number of get_response calls made."""
        return self._call_index

    @property
    def call_history(self) -> list[tuple[Sequence[Message], Mapping[str, Any]]]:
        """List of (messages, options) from each call, for test assertions."""
        return self._call_history

    def _finalize_response_updates(
        self,
        updates: Sequence[ChatResponseUpdate],
        *,
        response_format: Any | None = None,
    ) -> ChatResponse[Any]:
        """Finalize response updates into a Chrys ``ChatResponse``."""
        return ChatResponse.from_updates(
            updates,
            output_format_type=response_format,
        )

    def _build_response_stream(
        self,
        stream: AsyncIterable[ChatResponseUpdate] | Awaitable[AsyncIterable[ChatResponseUpdate]],
        *,
        response_format: Any | None = None,
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
        """Create a Chrys ``ResponseStream`` with the standard finalizer."""
        return ResponseStream(
            stream,
            finalizer=lambda updates: self._finalize_response_updates(updates, response_format=response_format),
        )

    def add_response(self, response: MockResponse) -> None:
        """Append a response to the queue."""
        self._responses.append(response)

    def reset(self) -> None:
        """Reset call index and history."""
        self._call_index = 0
        self._call_history.clear()

    def _next_response(self) -> MockResponse:
        """Get the next scripted response, or a default fallback."""
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
        else:
            resp = MockResponse(text="[No more scripted responses]")
        self._call_index += 1
        return resp

    def _build_messages(self, resp: MockResponse) -> list[Message]:
        """Build Message list from a MockResponse."""
        contents: list[Content] = []

        for name, call_id, args in resp.tool_calls:
            contents.append(Content.from_function_call(call_id, name, arguments=args))

        if resp.reasoning_text:
            contents.append(Content.from_text_reasoning(text=resp.reasoning_text))

        if resp.text:
            contents.append(Content.from_text(resp.text))

        return [Message(role="assistant", contents=contents)]

    def _finish_reason(self, resp: MockResponse) -> FinishReasonLiteral | FinishReason:
        if resp.finish_reason:
            return FinishReason(resp.finish_reason)
        return "tool_calls" if resp.tool_calls else "stop"

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]:
        # Loop-to-client trajectory handle; mirror _IntermediateTextMixin so
        # mock-driven sessions record the same exchange facts as real ones.
        forwarded_trace = kwargs.pop(TRAJECTORY_EXCHANGE_KWARG, None)
        self._call_history.append((list(messages), dict(options)))

        resp = self._next_response()
        model_id = resp.model_id or self._default_model_id
        finish_reason = self._finish_reason(resp)

        # Internal side calls (e.g. the Phase-4 last-words completer) never
        # join the conversation — mirror _IntermediateTextMixin and keep the
        # intermediate-text callbacks silent for them.
        suppress_intermediate = in_internal_side_call()
        trace = resolve_exchange_trace(forwarded_trace, internal_side_call=suppress_intermediate)
        # Mirrors _IntermediateTextMixin: a trace the resolver minted here
        # belongs to this client, one the loop handed down does not.
        owns_trace = trace is not None and trace is not forwarded_trace
        observer = _ExchangeObserver(trace, owned=owns_trace) if trace is not None else None
        if trace is not None and not suppress_intermediate:
            # Mirrors _IntermediateTextMixin: the exact request this
            # acquisition sends, as a revision of the actor's context chain,
            # named before the start marker that carries it is written.
            trace.set_context_revision(record_context_revision(trace.context, messages))
        if observer is not None:
            observer.started(options, stream=stream)

        if stream:
            result = self._build_response_stream(self._stream_updates(resp, model_id, finish_reason))
            if observer is not None:
                observer.attach_stream(result)
            # Attach result_hook for intermediate text (streaming path).
            # Mirrors _IntermediateTextMixin._wrap_stream_intermediate.
            sync_cb = None if suppress_intermediate else self._on_intermediate_text_sync
            if sync_cb is not None:

                def _on_finalized(response: ChatResponse) -> ChatResponse:
                    text = _extract_intermediate_text(response)
                    if text:
                        sync_cb(text)
                    elif _count_function_calls(response):
                        sync_cb("")  # Batch boundary signal
                    return response

                result.with_result_hook(_on_finalized)
            return result

        async_cb = None if suppress_intermediate else self._on_intermediate_text_async
        if observer is not None:
            return observer.wrap_awaitable(self._mock_response(resp, model_id, finish_reason, async_cb))
        return self._mock_response(resp, model_id, finish_reason, async_cb)

    async def _mock_response(
        self,
        resp: MockResponse,
        model_id: str,
        finish_reason: FinishReasonLiteral | FinishReason,
        async_cb: Callable[[str], Awaitable[None]] | None,
    ) -> ChatResponse[Any]:
        if resp.delay > 0:
            await asyncio.sleep(resp.delay)
        response = ChatResponse(
            messages=self._build_messages(resp),
            response_id=f"mock-{self._call_index}",
            conversation_id=resp.conversation_id or None,
            model=model_id,
            finish_reason=finish_reason,
        )
        if resp.usage_details is not None:
            response.usage_details = resp.usage_details.copy()
        # Fire intermediate text callback (non-streaming path).
        # Mirrors _IntermediateTextMixin._intercept.
        if async_cb is not None:
            text = _extract_intermediate_text(response)
            if text:
                await async_cb(text)
            elif _count_function_calls(response):
                await async_cb("")  # Batch boundary signal
        return response

    async def _stream_updates(
        self,
        resp: MockResponse,
        model_id: str,
        finish_reason: FinishReasonLiteral | FinishReason,
    ) -> AsyncIterable[ChatResponseUpdate]:
        """Yield streaming chunks for a MockResponse."""
        # Stream tool calls first (as a single chunk)
        if resp.tool_calls:
            tool_contents = [
                Content.from_function_call(call_id, name, arguments=args) for name, call_id, args in resp.tool_calls
            ]
            yield ChatResponseUpdate(
                contents=tool_contents,
                role="assistant",
                conversation_id=resp.conversation_id or None,
                model=model_id,
            )

        # Stream reasoning as a single chunk before the visible text.
        if resp.reasoning_text:
            yield ChatResponseUpdate(
                contents=[Content.from_text_reasoning(text=resp.reasoning_text)],
                role="assistant",
                conversation_id=resp.conversation_id or None,
                model=model_id,
            )

        # Stream text in chunks
        if resp.text:
            text = resp.text
            chunk_size = resp.chunk_size if resp.chunk_size > 0 else len(text)
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                yield ChatResponseUpdate(
                    contents=[Content.from_text(chunk)],
                    role="assistant",
                    conversation_id=resp.conversation_id or None,
                    model=model_id,
                )
                if resp.chunk_delay > 0 and i + chunk_size < len(text):
                    await asyncio.sleep(resp.chunk_delay)

        if resp.usage_details is not None:
            yield ChatResponseUpdate(
                contents=[Content.from_usage(usage_details=resp.usage_details.copy())],
                role="assistant",
                conversation_id=resp.conversation_id or None,
                model=model_id,
            )

        # Final chunk with finish_reason
        yield ChatResponseUpdate(
            contents=[],
            role="assistant",
            conversation_id=resp.conversation_id or None,
            model=model_id,
            finish_reason=finish_reason,
        )
