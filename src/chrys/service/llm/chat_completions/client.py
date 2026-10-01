# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""The Chat Completions clients: one request out, one completion or chunk stream back.

They run over a configured ``AsyncOpenAI`` client and own closing it.
:mod:`.request` builds the request, :mod:`.decode` and :mod:`.stream` read
the answer. OpenAI-compatible endpoints differ only in what a
:class:`ChatCompletionsVariant` describes, so the DeepSeek and GLM clients
are the same client with another variant.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterable, Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, override

from openai import BadRequestError

from chrys.foundation.util.once_close import OnceClose
from chrys.kernel import ChatResponse, ChatResponseUpdate, Message, ResponseStream
from chrys.kernel.exceptions import ChatClientException
from chrys.service.llm.openai_exceptions import OpenAIContentFilterException
from chrys.service.llm.providers import CHAT_COMPLETIONS_TOKEN_LIMIT_PARAMS
from chrys.service.llm.wire_client import RequestHeaders, WireClient

from .decode import decode_completion
from .request import build_request
from .stream import StreamState
from .validation import (
    bounded_body_preview,
    parse_completion,
    raise_invalid_response,
    validate_stream_response,
    zero_event_message,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from chrys.kernel.compaction import CompactionStrategy, TokenizerProtocol
    from chrys.service.llm.observer import WireCallObserver

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatCompletionsVariant:
    """What differs between endpoints that speak the Chat Completions protocol."""

    # The wire name of the output-token cap.
    max_output_param: str
    # Reasoning replays only on a request that sends tools or follows a tool
    # interaction, and then every assistant message carries a
    # ``reasoning_content``, empty where it has none.
    reasoning_with_tools: bool
    # Fragments of one message merge as far as their roles allow, and content
    # is a plain string where it can be.
    strict_messages: bool
    # Usage reports DeepSeek's prompt-cache hits.
    reports_prompt_cache_hits: bool


OPENAI = ChatCompletionsVariant(
    max_output_param=CHAT_COMPLETIONS_TOKEN_LIMIT_PARAMS["openai"],
    reasoning_with_tools=False,
    strict_messages=False,
    reports_prompt_cache_hits=False,
)
# DeepSeek takes the legacy ``max_tokens``
# (https://api-docs.deepseek.com/api/create-chat-completion). In thinking
# mode (https://api-docs.deepseek.com/guides/thinking_mode) a request without
# tools ignores historical ``reasoning_content``, while one that sends tools
# must replay all of it, turns without a call included, or get HTTP 400; the
# live API was seen accepting such requests anyway, but the client follows
# the documented contract. Its schema wants a message's same-role fragments
# merged, plain-string content and ``content: ""`` beside ``tool_calls``.
# Usage reports cache reads as ``prompt_cache_hit_tokens``.
DEEPSEEK = ChatCompletionsVariant(
    max_output_param=CHAT_COMPLETIONS_TOKEN_LIMIT_PARAMS["deepseek-openai"],
    reasoning_with_tools=True,
    strict_messages=True,
    reports_prompt_cache_hits=True,
)
# GLM (Zhipu AI / z.ai) documents only the legacy ``max_tokens``. Its
# preserved thinking needs historical ``reasoning_content`` on every
# multi-turn request, turns without a call included, as OpenAI's default
# replay sends it: https://docs.z.ai/guides/llm/glm-4.7 (interleaved and
# preserved thinking) and https://docs.bigmodel.cn/api-reference.
GLM = ChatCompletionsVariant(
    max_output_param=CHAT_COMPLETIONS_TOKEN_LIMIT_PARAMS["glm-openai"],
    reasoning_with_tools=False,
    strict_messages=False,
    reports_prompt_cache_hits=False,
)


class ChatCompletionsClient(WireClient):
    """Chat Completions wire client over a configured ``AsyncOpenAI`` client.

    The tool loop and chat middleware wrap it in the stack the client factory
    builds.
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "openai"
    INJECTABLE: ClassVar[set[str]] = {"sdk_client"}
    VARIANT: ClassVar[ChatCompletionsVariant] = OPENAI

    def __init__(
        self,
        model: str | None = None,
        *,
        sdk_client: AsyncOpenAI | None = None,
        observer: WireCallObserver | None = None,
        request_headers: RequestHeaders | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        additional_properties: dict[str, Any] | None = None,
    ) -> None:
        if sdk_client is None:
            raise ValueError(f"{type(self).__name__} requires a pre-configured sdk_client.")
        self.sdk_client = sdk_client
        self._close_sdk = OnceClose(self._close_sdk_client)
        self.model = model or ""
        super().__init__(
            observer=observer,
            request_headers=request_headers,
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
            additional_properties=additional_properties,
        )

    @classmethod
    @override
    def from_sdk_client(
        cls,
        sdk_client: AsyncOpenAI,
        *,
        model: str,
        observer: WireCallObserver | None = None,
        request_headers: RequestHeaders | None = None,
    ) -> Self:
        return cls(model=model, sdk_client=sdk_client, observer=observer, request_headers=request_headers)

    async def aclose(self) -> None:
        """Close the SDK client and its HTTP pool; concurrent callers share one close."""
        await self._close_sdk()

    async def _close_sdk_client(self) -> None:
        await self.sdk_client.close()

    @override
    def service_url(self) -> str:
        if not self.sdk_client:
            return "Unknown"
        return str(self.sdk_client.base_url)

    def _build_request(self, messages: Sequence[Message], options: Mapping[str, Any]) -> dict[str, Any]:
        request = build_request(messages, options, model=self.model, variant=self.VARIANT)
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
        # Built now, so a request that cannot be built fails the call itself.
        request = self._build_request(messages, options)
        return self._complete(request, options)

    async def _complete(self, request: dict[str, Any], options: Mapping[str, Any]) -> ChatResponse:
        try:
            # The raw wrapper (it adds ``X-Stainless-Raw-Response: true``)
            # keeps the status, headers and body the diagnostics quote.
            raw = await self.sdk_client.chat.completions.with_raw_response.create(stream=False, **request)
            return decode_completion(parse_completion(raw), options, variant=self.VARIANT)
        except ChatClientException:
            raise
        except Exception as ex:
            raise _service_error(type(self), ex) from ex

    @override
    def _open_stream(
        self,
        *,
        messages: Sequence[Message],
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
        request = self._build_request(messages, options)
        request["stream_options"] = {"include_usage": True}

        async def updates() -> AsyncIterable[ChatResponseUpdate]:
            state = StreamState(self.VARIANT)
            sdk_stream: Any = None
            try:
                # The same raw wrapper as the blocking path, for the same reason.
                raw: Any = await self.sdk_client.chat.completions.with_raw_response.create(stream=True, **request)
                replay = await validate_stream_response(raw)
                sdk_stream = raw.parse()
                received = False
                async for chunk in sdk_stream:
                    received = True
                    for update in state.updates_for(chunk):
                        yield update
                if not received:
                    # Valid framing with no event after it (comments only, or
                    # ``[DONE]`` alone) is never a usable completion.
                    raise_invalid_response(zero_event_message(replay))
                # Some gateways end the stream without a finish reason.
                if (calls := state.finish()) is not None:
                    yield calls
            except json.JSONDecodeError as ex:
                raise_invalid_response(
                    f"Chat Completions API returned invalid stream event JSON ({ex}). "
                    f"Event data: {bounded_body_preview(ex.doc)}"
                )
            except ChatClientException:
                raise
            except Exception as ex:
                raise _service_error(type(self), ex) from ex
            finally:
                if sdk_stream is not None:
                    try:
                        await sdk_stream.close()
                    except Exception:
                        logger.debug("Failed to close OpenAI chat-completion stream", exc_info=True)

        return self._build_response_stream(updates(), response_format=options.get("response_format"))


class DeepSeekChatCompletionsClient(ChatCompletionsClient):
    """DeepSeek's Chat Completions endpoint, thinking mode included."""

    VARIANT: ClassVar[ChatCompletionsVariant] = DEEPSEEK


class GlmChatCompletionsClient(ChatCompletionsClient):
    """GLM's Chat Completions endpoint, with preserved thinking."""

    VARIANT: ClassVar[ChatCompletionsVariant] = GLM


def _service_error(client_type: type, error: Exception) -> ChatClientException:
    if isinstance(error, BadRequestError) and error.code == "content_filter":
        return OpenAIContentFilterException(
            f"{client_type} service encountered a content error: {error}", inner_exception=error
        )
    return ChatClientException(f"{client_type} service failed to complete the prompt: {error}", inner_exception=error)
