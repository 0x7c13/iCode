# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""The Messages API client: one request out, one message or event stream back.

It runs over a configured Anthropic SDK client (direct, Bedrock, Foundry or
Vertex) and owns closing it. :mod:`.request` builds the request,
:mod:`.decode` and :mod:`.stream` read the answer.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, Awaitable, Mapping, Sequence
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

from chrys.foundation.util.once_close import OnceClose
from chrys.kernel import ChatResponse, ChatResponseUpdate, Message, ResponseStream
from chrys.service.llm.wire_client import RequestHeaders, WireClient

from .decode import decode_message
from .request import build_request
from .stream import StreamState

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicFoundry, AsyncAnthropicVertex

    from chrys.kernel.compaction import CompactionStrategy, TokenizerProtocol
    from chrys.service.llm.observer import WireCallObserver

    type AnthropicSdkClient = AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicFoundry | AsyncAnthropicVertex

logger = logging.getLogger(__name__)


class AnthropicMessagesClient(WireClient):
    """Messages API wire client over a configured Anthropic SDK client.

    The tool loop and chat middleware wrap it in the stack the client factory
    builds.
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        *,
        sdk_client: AnthropicSdkClient | None = None,
        observer: WireCallObserver | None = None,
        request_headers: RequestHeaders | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        additional_properties: dict[str, Any] | None = None,
    ) -> None:
        if sdk_client is None:
            raise ValueError("AnthropicMessagesClient requires a pre-configured sdk_client.")
        super().__init__(
            observer=observer,
            request_headers=request_headers,
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
            additional_properties=additional_properties,
        )
        self.sdk_client = sdk_client
        self.model = model or ""
        self._close_sdk = OnceClose(self._close_sdk_client)

    @classmethod
    @override
    def from_sdk_client(
        cls,
        sdk_client: AnthropicSdkClient,
        *,
        model: str,
        observer: WireCallObserver | None = None,
        request_headers: RequestHeaders | None = None,
    ) -> Self:
        return cls(model=model, sdk_client=sdk_client, observer=observer, request_headers=request_headers)

    @override
    def service_url(self) -> str:
        return str(self.sdk_client.base_url)

    async def aclose(self) -> None:
        """Close the SDK client and its HTTP pool; concurrent callers share one close."""
        await self._close_sdk()

    async def _close_sdk_client(self) -> None:
        await self.sdk_client.close()

    def _build_request(
        self,
        messages: Sequence[Message],
        options: Mapping[str, Any],
        call_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = build_request(messages, options, call_kwargs, model=self.model)
        self._stamp_request_headers(request)
        return request

    @override
    def _send(
        self,
        *,
        messages: Sequence[Message],
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse]:
        request = self._build_request(messages, options, kwargs)

        async def response() -> ChatResponse:
            message = await self.sdk_client.beta.messages.create(**request, stream=False)  # type: ignore[misc]
            return decode_message(message, response_format=options.get("response_format"))

        return response()

    @override
    def _open_stream(
        self,
        *,
        messages: Sequence[Message],
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
        request = self._build_request(messages, options, kwargs)

        async def updates() -> AsyncIterable[ChatResponseUpdate]:
            state = StreamState()
            events: Any = None
            try:
                events = await self.sdk_client.beta.messages.create(**request, stream=True)  # type: ignore[misc]
                async for event in events:
                    for update in state.updates_for(event):
                        yield update
                state.finish()
            finally:
                if events is not None:
                    await _close_event_stream(events)

        return self._build_response_stream(updates(), response_format=options.get("response_format"))


async def _close_event_stream(events: Any) -> None:
    """Close the SDK event stream; a failure to close is only logged."""
    try:
        close = getattr(events, "close", None) or getattr(events, "aclose", None)
        if close is not None and isawaitable(closing := close()):
            await closing
    except Exception:
        logger.debug("Failed to close Anthropic message stream", exc_info=True)
