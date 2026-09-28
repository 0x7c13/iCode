# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chrys-owned OpenAI Chat Completions raw wire client.

Parser and serializer logic, with provider
settings removed: callers must inject a pre-configured ``AsyncOpenAI`` client.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from inspect import isawaitable
from itertools import chain
from typing import Any, ClassVar, Generic, Literal, TypeGuard, TypeVar, cast, overload, override

from openai import AsyncOpenAI, BadRequestError
from openai.lib._parsing._completions import type_to_response_format_param
from openai.types import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_message_custom_tool_call import ChatCompletionMessageCustomToolCall
from openai.types.chat.completion_create_params import WebSearchOptions
from pydantic import BaseModel
from typing_extensions import TypedDict

from chrys.foundation.util.once_close import OnceClose
from chrys.kernel import (
    TOOL_CALL_CONTENT_TYPES,
    BaseChatClient,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FinishReason,
    FunctionTool,
    Message,
    ResponseStream,
    ToolTypes,
    UsageDetails,
    is_image_content,
    normalize_tools,
    prepend_instructions_to_messages,
    validate_tool_mode,
)
from chrys.kernel._content import _ANTHROPIC_REDACTED_THINKING_KEY
from chrys.kernel.compaction import CompactionStrategy, TokenizerProtocol
from chrys.kernel.exceptions import (
    ChatClientException,
    ChatClientInvalidRequestException,
    ChatClientInvalidResponseException,
)
from chrys.service.agent_middleware.events.hosted_tools import cross_provider_hosted_degradations
from chrys.service.llm._chat_stream_validation import (
    _bounded_response_body_preview,
    _parse_chat_completion_raw_response,
    _raise_invalid_chat_completion_response,
    _raise_invalid_chat_completion_stream_event,
    _ReplayByteStream,
    _validate_chat_completion_stream_response,
    _zero_event_stream_message,
)
from chrys.service.llm._structured_outputs import (
    _materialize_json_structure,
    _sanitize_response_format_name,
    _strictify_response_schema,
)
from chrys.service.llm.openai_exceptions import OpenAIContentFilterException
from chrys.service.llm.openai_timestamps import openai_created_at_iso
from chrys.service.text_blocks import join_text_blocks, reconstruct_text_blocks, text_block_id

logger = logging.getLogger(__name__)

_PENDING_IMAGE_PARTS_KEY = "_chrys_pending_image_parts"
_HOSTED_CONTEXT_SUMMARY_KEY = "_chrys_hosted_context_summary"

# OpenAI validates the message ``name`` field as ``^[^\s<|\\/>]+$`` (max 64
# chars) — CJK and hyphens are allowed, so strip only the characters the rule
# actually forbids rather than whitelisting ASCII.
_INVALID_AUTHOR_NAME_CHARS_RE = re.compile(r"[\s<|\\/>]+")


def _sanitize_author_name(author_name: str | None) -> str | None:
    """Sanitize an author name for the Chat Completions ``name`` field.

    Returns ``None`` when nothing survives so callers omit the key entirely
    instead of sending ``""``.
    """
    if not author_name:
        return None
    sanitized = _INVALID_AUTHOR_NAME_CHARS_RE.sub("", author_name)[:64]
    return sanitized or None


def _sanitize_enveloped_response_format(response_format: dict[str, Any]) -> dict[str, Any]:
    """Return a ``json_schema`` envelope with its ``name`` sanitized.

    Copies only when the name actually changes (or is missing) so valid
    user-supplied dicts pass through by identity; non-envelope shapes are
    returned untouched.
    """
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, Mapping):
        return response_format
    json_schema_typed = cast("Mapping[str, Any]", json_schema)
    name = json_schema_typed.get("name")
    sanitized = _sanitize_response_format_name(name)
    if name == sanitized:
        return response_format
    return {**response_format, "json_schema": {**json_schema_typed, "name": sanitized}}


REASONING_DETAILS_FIELD = "reasoning_details"
REASONING_CONTENT_FIELD = "reasoning_content"
REASONING_FIELD = "reasoning"
REASONING_FORMAT_KEY = "openai_reasoning_format"

_REASONING_FIELDS: tuple[str, ...] = (REASONING_DETAILS_FIELD, REASONING_CONTENT_FIELD, REASONING_FIELD)


def _reasoning_fields_from_openai_message(message: Any) -> dict[str, Any]:
    """Return supported reasoning fields with vLLM's ``reasoning`` as a fallback.

    ``reasoning_details`` and ``reasoning_content`` are established provider
    dialects that may carry richer or provider-required replay state. Some
    compatible gateways also mirror their plaintext into vLLM's newer
    canonical ``reasoning`` field. Preserve the established fields without
    duplicating that mirror; use ``reasoning`` only when neither established
    field is present on this message or delta.
    """
    fields: dict[str, Any] = {}
    reasoning_details = getattr(message, REASONING_DETAILS_FIELD, None)
    if reasoning_details is not None:
        fields[REASONING_DETAILS_FIELD] = reasoning_details
    reasoning_content = getattr(message, REASONING_CONTENT_FIELD, None)
    if reasoning_content is not None:
        fields[REASONING_CONTENT_FIELD] = reasoning_content
    if fields:
        return fields
    reasoning = getattr(message, REASONING_FIELD, None)
    if reasoning is not None:
        fields[REASONING_FIELD] = reasoning
    return fields


@dataclass
class _PendingOpenAIToolCall:
    """One streamed Chat Completions function call, assembled before kernel exposure."""

    choice_index: int
    tool_index: int | None
    first_seen_order: int
    call_id: str | None = None
    name: str | None = None
    argument_deltas: list[str] = field(default_factory=list)
    raw_representations: list[Any] = field(default_factory=list)

    def to_content(self) -> Content:
        """Create the single function-call Content emitted for this logical call."""
        raw_representation: Any = self.raw_representations
        if len(self.raw_representations) == 1:
            raw_representation = self.raw_representations[0]
        return Content.from_function_call(
            call_id=self.call_id or "",
            name=self.name or "",
            arguments="".join(self.argument_deltas),
            raw_representation=raw_representation,
        )


@dataclass
class _OpenAIChoiceStreamState:
    """Request-local assembly state for one Chat Completions choice."""

    pending_calls: list[_PendingOpenAIToolCall] = field(default_factory=list)
    calls_by_index: dict[int, _PendingOpenAIToolCall] = field(default_factory=dict)
    calls_by_id: dict[str, _PendingOpenAIToolCall] = field(default_factory=dict)
    empty_reasoning_emitted: bool = False
    saw_nonempty_reasoning: bool = False
    finished: bool = False


@dataclass
class _OpenAIChatStreamState:
    """Per-wire-request state for streamed Chat Completions normalization."""

    choices: dict[int, _OpenAIChoiceStreamState] = field(default_factory=dict)
    next_first_seen_order: int = 0

    def choice(self, choice_index: int) -> _OpenAIChoiceStreamState:
        """Return the state for *choice_index*, creating it on first use."""
        if choice_index not in self.choices:
            self.choices[choice_index] = _OpenAIChoiceStreamState()
        return self.choices[choice_index]

    def new_pending_call(self, choice_index: int, tool_index: int | None) -> _PendingOpenAIToolCall:
        """Register a new logical call in first-seen order."""
        choice_state = self.choice(choice_index)
        pending = _PendingOpenAIToolCall(
            choice_index=choice_index,
            tool_index=tool_index,
            first_seen_order=self.next_first_seen_order,
        )
        self.next_first_seen_order += 1
        choice_state.pending_calls.append(pending)
        if tool_index is not None:
            choice_state.calls_by_index[tool_index] = pending
        return pending


def _streamed_tool_index(tool: Any) -> int | None:
    """Return a usable SDK tool-call index without trusting construct-time validation."""
    value = getattr(tool, "index", None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _nonempty_stream_string(value: Any) -> str | None:
    """Narrow a provider identifier/name to a non-empty string."""
    return value if isinstance(value, str) and value else None


def _nonempty_stream_call_id(value: Any) -> str | None:
    """Narrow a provider call id, rejecting a stringified JSON null."""
    return value if isinstance(value, str) and value and value != "null" else None


def _accumulate_openai_tool_calls(
    choice: ChunkChoice,
    stream_state: _OpenAIChatStreamState,
) -> None:
    """Accumulate indexed Chat Completions tool deltas without exposing fragments.

    Non-empty call ids are authoritative because some compatible gateways emit
    multiple complete calls with the same nominal index. Id-less continuations
    then use the latest index binding, or the sole pending call when unambiguous.
    """
    if choice.delta is None or not choice.delta.tool_calls:  # pyright: ignore[reportUnnecessaryComparison]
        return

    choice_index = choice.index
    choice_state = stream_state.choice(choice_index)
    if choice_state.finished:
        logger.warning(
            "Ignoring streamed tool-call fragment received after the terminal update for choice %d",
            choice_index,
        )
        return
    for tool in choice.delta.tool_calls:
        if isinstance(tool, ChatCompletionMessageCustomToolCall):
            continue

        tool_index = _streamed_tool_index(tool)
        call_id = _nonempty_stream_call_id(getattr(tool, "id", None))
        function = getattr(tool, "function", None)
        name = _nonempty_stream_string(getattr(function, "name", None)) if function is not None else None
        arguments = getattr(function, "arguments", None) if function is not None else None

        pending: _PendingOpenAIToolCall | None = None
        if call_id is not None:
            pending = choice_state.calls_by_id.get(call_id)
            if pending is None:
                indexed_pending = choice_state.calls_by_index.get(tool_index) if tool_index is not None else None
                if indexed_pending is not None and indexed_pending.call_id is None:
                    # Some compatible endpoints omit the id on the opening
                    # fragment and supply it later for the same index.
                    pending = indexed_pending
                elif (
                    tool_index is None
                    and len(choice_state.pending_calls) == 1
                    and choice_state.pending_calls[0].call_id is None
                ):
                    # Preserve the sole-pending fallback when both the opening
                    # fragment and the late-id fragment omit an index.
                    pending = choice_state.pending_calls[0]
                else:
                    pending = stream_state.new_pending_call(choice_index, tool_index)
                pending.call_id = call_id
                choice_state.calls_by_id[call_id] = pending
            if tool_index is not None:
                # Id wins over index. Gateways such as Gemini-compatible Chat
                # Completions may reuse index=0 for distinct complete calls.
                choice_state.calls_by_index[tool_index] = pending
                if pending.tool_index is None:
                    pending.tool_index = tool_index
        else:
            if tool_index is not None:
                pending = choice_state.calls_by_index.get(tool_index)
            if pending is None:
                sole = choice_state.pending_calls[0] if len(choice_state.pending_calls) == 1 else None
                if sole is not None and (tool_index is None or sole.tool_index is None):
                    pending = sole
                elif tool_index is not None or not choice_state.pending_calls:
                    pending = stream_state.new_pending_call(choice_index, tool_index)
                else:
                    raise ChatClientInvalidResponseException(
                        "Ambiguous streamed tool-call fragment without a call id "
                        f"for choice {choice_index}, index {tool_index!r}."
                    )
                if tool_index is not None:
                    choice_state.calls_by_index[tool_index] = pending
                    if pending.tool_index is None:
                        pending.tool_index = tool_index

        if name is not None:
            if pending.name is not None and pending.name != name:
                raise ChatClientInvalidResponseException(
                    f"Conflicting streamed tool-call names for choice {choice_index}, index {tool_index!r}."
                )
            pending.name = name
        if arguments is not None:
            if not isinstance(arguments, str):
                raise ChatClientInvalidResponseException(
                    f"Non-string streamed tool-call arguments for choice {choice_index}, index {tool_index!r}."
                )
            pending.argument_deltas.append(arguments)
        pending.raw_representations.append(function if function is not None else tool)


def _drain_openai_tool_calls(
    stream_state: _OpenAIChatStreamState,
    *,
    finish_reasons: Mapping[int, str | None] | None = None,
) -> ChatResponseUpdate | None:
    """Emit complete calls once, before their terminal update.

    A missing id retains the kernel's legacy id-less-call behavior. A missing
    name is unsafe except when the response was explicitly truncated before
    the model could finish the call, in which case validators own the terminal
    length diagnosis and the unusable fragment is discarded.
    """
    choice_indices = sorted(finish_reasons) if finish_reasons is not None else sorted(stream_state.choices)
    contents: list[Content] = []
    for choice_index in choice_indices:
        choice_state = stream_state.choices.get(choice_index)
        if choice_state is None:
            if finish_reasons is not None:
                # Azure-compatible streams may send a terminal choice with
                # delta=null before any content. Keep a tombstone so a
                # protocol-violating late tool delta cannot start fresh state
                # and escape through the EOF drain.
                stream_state.choice(choice_index).finished = True
            continue
        finish_reason = finish_reasons.get(choice_index) if finish_reasons is not None else None
        pending_calls = sorted(
            choice_state.pending_calls,
            key=lambda pending: (
                pending.tool_index if pending.tool_index is not None else pending.first_seen_order,
                pending.first_seen_order,
            ),
        )
        if len(pending_calls) > 1:
            # Adjacent empty-id calls are indistinguishable to the kernel's
            # fragment merger. Assign request-local ids only when necessary
            # to preserve distinct parallel calls and their result pairing.
            used_call_ids = {pending.call_id for pending in pending_calls if pending.call_id}
            for pending in pending_calls:
                if pending.call_id:
                    continue
                base_call_id = f"call_chrys_{choice_index}_{pending.first_seen_order}"
                call_id = base_call_id
                suffix = 1
                while call_id in used_call_ids:
                    call_id = f"{base_call_id}_{suffix}"
                    suffix += 1
                pending.call_id = call_id
                used_call_ids.add(call_id)
                logger.debug(
                    "Assigned a local id to an id-less parallel tool call for choice %d, index %r",
                    choice_index,
                    pending.tool_index,
                )
        for pending in pending_calls:
            if not pending.name:
                if finish_reason == "length":
                    logger.warning(
                        "Discarding truncated streamed tool call without a name for choice %d, index %r",
                        choice_index,
                        pending.tool_index,
                    )
                    continue
                raise ChatClientInvalidResponseException(
                    "Streamed tool call ended without a function name "
                    f"for choice {choice_index}, index {pending.tool_index!r}."
                )
            if not pending.call_id:
                logger.debug(
                    "OpenAI-compatible stream emitted a tool call without an id "
                    "for choice %d, index %r; preserving legacy id-less behavior",
                    choice_index,
                    pending.tool_index,
                )
            contents.append(pending.to_content())
        choice_state.pending_calls.clear()
        choice_state.calls_by_index.clear()
        choice_state.calls_by_id.clear()
        if finish_reasons is not None:
            choice_state.finished = True

    if not contents:
        return None
    return ChatResponseUpdate(contents=contents)


def _parse_reasoning_details_payload(protected_data: str) -> Any | None:
    """Decode a chat-completion client's own JSON reasoning payload.

    Other providers store opaque payloads in ``protected_data`` too (Responses
    encrypted reasoning, Anthropic thinking signatures); those never decode as
    JSON and must be treated as foreign, not replayed.
    """
    try:
        return json.loads(protected_data)
    except ValueError:
        return None


ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel | None, default=None)

# region OpenAI Chat Options TypedDict


class PredictionTextContent(TypedDict, total=False):
    """Prediction text content options for OpenAI Chat completions."""

    type: Literal["text"]
    text: str


class Prediction(TypedDict, total=False):
    """Prediction options for OpenAI Chat completions."""

    type: Literal["content"]
    content: str | list[PredictionTextContent]


class OpenAIChatCompletionOptions(ChatOptions[ResponseModelT], Generic[ResponseModelT], total=False):
    """OpenAI-specific chat options dict.

    Extends ChatOptions with options specific to OpenAI's Chat Completions API.

    Keys:
        model: The model to use for the request,
            translates to ``model`` in OpenAI API.
        temperature: Sampling temperature between 0 and 2.
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum number of tokens to generate,
            translates to ``max_completion_tokens`` in OpenAI API.
        stop: Stop sequences.
        seed: Random seed for reproducibility.
        frequency_penalty: Frequency penalty between -2.0 and 2.0.
        presence_penalty: Presence penalty between -2.0 and 2.0.
        tools: List of tools (functions) available to the model.
        tool_choice: How the model should use tools.
        allow_multiple_tool_calls: Whether to allow parallel tool calls,
            translates to ``parallel_tool_calls`` in OpenAI API.
        response_format: Structured output schema.
        metadata: Request metadata for tracking.
        user: End-user identifier for abuse monitoring.
        store: Whether to store the conversation.
        instructions: System instructions for the model (prepended as system message).
        # OpenAI-specific options (supported by all models):
        logit_bias: Token bias values (-100 to 100).
        logprobs: Whether to return log probabilities.
        top_logprobs: Number of top log probabilities to return (0-20).
        prediction: Whether to use predicted return tokens.
    """

    # OpenAI-specific generation parameters (supported by all models)
    logit_bias: dict[str | int, float]  # type: ignore[misc]
    logprobs: bool
    top_logprobs: int
    prediction: Prediction
    verbosity: Literal["low", "medium", "high"]
    """Output verbosity for GPT-5 family models. Lower values yield shorter responses.
    See: https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_new_params_and_tools#1-verbosity-parameter"""


OPTION_TRANSLATIONS: dict[str, str] = {
    "allow_multiple_tool_calls": "parallel_tool_calls",
    "max_tokens": "max_completion_tokens",
}


def _is_string_keyed_dict(value: object) -> TypeGuard[dict[str, Any]]:
    """Narrow response-format dicts whose public schema requires string keys."""
    return isinstance(value, dict)


# region Base Client
class RawOpenAIChatCompletionClient(BaseChatClient):
    """Raw OpenAI Chat completion class without middleware, telemetry, or function invocation.

    Warning:
        **This class should not normally be used directly.** It does not include middleware,
        telemetry, or function invocation support that you most likely need. If you do use it,
        you should consider which additional layers to apply. There is a defined ordering that
        you should follow:

        1. **ToolLoopLayer** - Owns the tool/function calling loop and routes function middleware
        2. **ChatMiddlewareLayer** - Applies chat middleware per model call and stays outside telemetry
        3. **ChatTelemetryLayer** - Must stay inside chat middleware for correct per-call telemetry

        Use ``create_instrumented_openai_client`` for the production stack with all layers applied.
    """

    INJECTABLE: ClassVar[set[str]] = {"client"}

    TOKEN_LIMIT_PARAM: ClassVar[str] = "max_completion_tokens"
    """Wire name of the output-token cap.

    Real OpenAI hard-rejects the legacy ``max_tokens`` spelling on current
    models, while some OpenAI-compatible endpoints (DeepSeek, GLM) only honor
    the legacy name. Subclasses override this to rename the translated
    ``max_completion_tokens`` option just before the request is built.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        async_client: AsyncOpenAI | None = None,
        instruction_role: str | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        additional_properties: dict[str, Any] | None = None,
    ) -> None:
        if async_client is None:
            raise ValueError("RawOpenAIChatCompletionClient requires a pre-configured async_client.")
        self.client = async_client
        self._close_sdk = OnceClose(self._close_sdk_client)
        self.model = model or ""
        self.org_id = None
        self.base_url = str(getattr(async_client, "base_url", "") or "") or None
        self.azure_endpoint = None
        self.api_version = None
        self.default_headers = None
        self.instruction_role = instruction_role
        super().__init__(
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
            additional_properties=additional_properties,
        )

    async def aclose(self) -> None:
        """Close the provider SDK client and its HTTP pool; concurrent callers share one close."""
        await self._close_sdk()

    async def _close_sdk_client(self) -> None:
        await self.client.close()

    # region Hosted Tool Factory Methods

    @staticmethod
    def get_web_search_tool(
        *,
        web_search_options: WebSearchOptions | None = None,
    ) -> dict[str, Any]:
        """Create a web search tool configuration for the Chat Completions API.

        Note: For the Chat Completions API, web search is passed via the `web_search_options`
        parameter rather than in the `tools` array. This method returns a dict that can be
        passed as a tool to ChatAgent, which will handle it appropriately.

        Keyword Args:
            web_search_options: The full WebSearchOptions configuration. This TypedDict includes:
                - user_location: Location context with "type" and "approximate" containing
                  "city", "country", "region", "timezone".
                - search_context_size: One of "low", "medium", "high".

        Returns:
            A dict configuration that enables web search when passed to ChatAgent.

        Examples:
            .. code-block:: python

                from chrys.service.llm.openai_chat_completion import RawOpenAIChatCompletionClient

                # Basic web search
                tool = RawOpenAIChatCompletionClient.get_web_search_tool()

                # With location context
                tool = RawOpenAIChatCompletionClient.get_web_search_tool(
                    web_search_options={
                        "user_location": {
                            "type": "approximate",
                            "approximate": {"city": "Seattle", "country": "US"},
                        },
                        "search_context_size": "medium",
                    }
                )

                agent = ChatAgent(client, tools=[tool])
        """
        tool: dict[str, Any] = {"type": "web_search"}

        if web_search_options:
            tool.update(web_search_options)

        return tool

    # endregion

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = ...,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: CompactionStrategy | None = None,
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
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse[Any]]: ...

    @override
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
    ) -> Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]:
        """Get a response from the raw OpenAI chat client."""
        super_get_response = cast(
            "Callable[..., Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]]",
            super().get_response,  # type: ignore[misc]
        )
        return super_get_response(  # type: ignore[no-any-return]
            messages=messages,
            stream=stream,
            options=options,
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
            function_invocation_kwargs=function_invocation_kwargs,
            client_kwargs=client_kwargs,
            request_message_observer=request_message_observer,
        )

    @override
    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        options: Mapping[str, Any],
        stream: bool = False,
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        # prepare
        options_dict = self._prepare_options(messages, options)

        if stream:
            # Streaming mode
            options_dict["stream_options"] = {"include_usage": True}

            async def _stream() -> AsyncIterable[ChatResponseUpdate]:
                client = self.client
                stream_state = _OpenAIChatStreamState()
                sdk_stream: Any = None
                replay: _ReplayByteStream | None = None
                try:
                    if isinstance(client, AsyncOpenAI):
                        # The SDK raw wrapper adds ``X-Stainless-Raw-Response: true``;
                        # that wire change is required to retain status, headers, and
                        # body for deterministic diagnostics.
                        raw_response: Any = await client.chat.completions.with_raw_response.create(
                            stream=True, **options_dict
                        )
                        replay = await _validate_chat_completion_stream_response(raw_response)
                        sdk_stream = raw_response.parse()
                    else:
                        sdk_stream = await client.chat.completions.create(stream=True, **options_dict)
                    received_event = False
                    async for chunk in sdk_stream:
                        received_event = True
                        choices = getattr(chunk, "choices", None)
                        if not isinstance(choices, list):
                            _raise_invalid_chat_completion_stream_event(chunk, choices)
                        if len(choices) == 0 and chunk.usage is None:
                            continue
                        parsed_chunk = self._parse_response_update_from_openai(chunk, stream_state=stream_state)
                        finish_reasons = {
                            choice.index: choice.finish_reason
                            for choice in chunk.choices
                            if choice.finish_reason is not None
                        }
                        if finish_reasons and (
                            boundary_update := _drain_openai_tool_calls(
                                stream_state,
                                finish_reasons=finish_reasons,
                            )
                        ):
                            # Some consumers stop on a terminal finish reason;
                            # expose complete calls before that update.
                            yield boundary_update
                        yield parsed_chunk
                    if replay is not None and not received_event:
                        # Valid framing but no event followed (comment-only body,
                        # ``[DONE]`` alone): never a usable completion.
                        _raise_invalid_chat_completion_response(_zero_event_stream_message(replay))
                    if boundary_update := _drain_openai_tool_calls(stream_state):
                        # Compatible gateways occasionally close normally
                        # without a terminal finish chunk.
                        yield boundary_update
                except json.JSONDecodeError as ex:
                    _raise_invalid_chat_completion_response(
                        f"Chat Completions API returned invalid stream event JSON ({ex}). "
                        f"Event data: {_bounded_response_body_preview(ex.doc)}"
                    )
                except BadRequestError as ex:
                    if ex.code == "content_filter":
                        raise OpenAIContentFilterException(
                            f"{type(self)} service encountered a content error: {ex}",
                            inner_exception=ex,
                        ) from ex
                    raise ChatClientException(
                        f"{type(self)} service failed to complete the prompt: {ex}",
                        inner_exception=ex,
                    ) from ex
                except ChatClientException:
                    raise
                except Exception as ex:
                    raise ChatClientException(
                        f"{type(self)} service failed to complete the prompt: {ex}",
                        inner_exception=ex,
                    ) from ex
                finally:
                    if sdk_stream is not None:
                        try:
                            close = getattr(sdk_stream, "close", None) or getattr(sdk_stream, "aclose", None)
                            if close is not None:
                                close_result = close()
                                if isawaitable(close_result):
                                    await close_result
                        except Exception:
                            logger.debug("Failed to close OpenAI chat-completion stream", exc_info=True)

            return self._build_response_stream(_stream(), response_format=options.get("response_format"))

        # Non-streaming mode
        async def _get_response() -> ChatResponse:
            client = self.client
            try:
                if isinstance(client, AsyncOpenAI):
                    # Keep the raw wrapper here in sync with the streaming path;
                    # it intentionally adds ``X-Stainless-Raw-Response: true``.
                    raw_response: Any = await client.chat.completions.with_raw_response.create(
                        stream=False, **options_dict
                    )
                    response = _parse_chat_completion_raw_response(raw_response)
                else:
                    response = await client.chat.completions.create(stream=False, **options_dict)
                return self._parse_response_from_openai(response, options)
            except BadRequestError as ex:
                if ex.code == "content_filter":
                    raise OpenAIContentFilterException(
                        f"{type(self)} service encountered a content error: {ex}",
                        inner_exception=ex,
                    ) from ex
                raise ChatClientException(
                    f"{type(self)} service failed to complete the prompt: {ex}",
                    inner_exception=ex,
                ) from ex
            except ChatClientException:
                raise
            except Exception as ex:
                raise ChatClientException(
                    f"{type(self)} service failed to complete the prompt: {ex}",
                    inner_exception=ex,
                ) from ex

        return _get_response()

    # region content creation

    def _prepare_tools_for_openai(
        self,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None,
    ) -> dict[str, Any]:
        """Prepare tools for the OpenAI Chat Completions API.

        Converts FunctionTool to JSON schema format. Web search tools are routed
        to web_search_options parameter. All other tools pass through unchanged.

        Args:
            tools: Tool(s) to prepare.

        Returns:
            Dict containing tools and optionally web_search_options.
        """
        chat_tools: list[Any] = []
        web_search_options: dict[str, Any] | None = None
        for tool in normalize_tools(tools):
            if isinstance(tool, FunctionTool):
                chat_tools.append(tool.to_json_schema_spec())
            elif isinstance(tool, MutableMapping):
                typed_tool = cast(MutableMapping[str, Any], tool)
                if typed_tool.get("type") == "web_search":
                    # Web search is handled via web_search_options, not tools array
                    web_search_options = {k: v for k, v in typed_tool.items() if k != "type"}
                else:
                    # Pass through all other dict-based tools unchanged
                    chat_tools.append(typed_tool)
            else:
                # Pass through all other tools (SDK types) unchanged
                chat_tools.append(tool)
        result: dict[str, Any] = {}
        if chat_tools:
            result["tools"] = chat_tools
        if web_search_options is not None:
            result["web_search_options"] = web_search_options
        return result

    def _prepare_options(self, messages: Sequence[Message], options: Mapping[str, Any]) -> dict[str, Any]:
        # The kernel consumes one conversation, not alternative choices. Check
        # extra_body too: the SDK merges it over named request parameters.
        extra_body = options.get("extra_body")
        for source in (options, extra_body if isinstance(extra_body, Mapping) else {}):
            choice_count = source.get("n")
            if choice_count is not None and (type(choice_count) is not int or choice_count != 1):
                raise ChatClientInvalidRequestException("The Chat Completions client supports only n=1")
        # Prepend instructions from options if they exist
        if instructions := options.get("instructions"):
            messages = prepend_instructions_to_messages(list(messages), instructions, role="system")

        # Start with a copy of options
        run_options = {
            k: v for k, v in options.items() if v is not None and k not in {"instructions", "tools", "conversation_id"}
        }

        # messages
        if messages and "messages" not in run_options:
            run_options["messages"] = self._prepare_messages_for_openai(messages)
        if "messages" not in run_options:
            raise ChatClientInvalidRequestException("Messages are required for chat completions")

        # Translation between options keys and Chat Completion API
        for old_key, new_key in OPTION_TRANSLATIONS.items():
            if old_key in run_options and old_key != new_key:
                run_options[new_key] = run_options.pop(old_key)
        if (
            self.TOKEN_LIMIT_PARAM != "max_completion_tokens"
            and "max_completion_tokens" in run_options
            and self.TOKEN_LIMIT_PARAM not in run_options
        ):
            run_options[self.TOKEN_LIMIT_PARAM] = run_options.pop("max_completion_tokens")

        # model id
        if not run_options.get("model"):
            if not self.model:
                raise ValueError("model must be a non-empty string")
            run_options["model"] = self.model

        # tools
        tools = options.get("tools")
        if tools is not None:
            run_options.update(self._prepare_tools_for_openai(tools))
        # Only include tool_choice and parallel_tool_calls if tools are present
        if not run_options.get("tools"):
            run_options.pop("parallel_tool_calls", None)
            run_options.pop("tool_choice", None)
        elif tool_choice := run_options.pop("tool_choice", None):
            tool_mode = validate_tool_mode(tool_choice)
            if tool_mode is not None:
                if (mode := tool_mode.get("mode")) == "required" and (
                    func_name := tool_mode.get("required_function_name")
                ) is not None:
                    run_options["tool_choice"] = {
                        "type": "function",
                        "function": {"name": func_name},
                    }
                elif mode in ("auto", "required") and tool_mode.get("allowed_tools") is not None:
                    logger.warning(
                        "allowed_tools is not supported by the Chat Completions API; "
                        "the setting will be ignored. Use the OpenAI Responses API client instead."
                    )
                    run_options["tool_choice"] = mode
                else:
                    run_options["tool_choice"] = mode

        # response format; gate on presence, not truthiness — an empty
        # mapping is a valid type-less JSON Schema and must still be
        # normalized instead of riding the seeded copy unwrapped.
        response_format = options.get("response_format")
        if response_format is not None:
            if isinstance(response_format, Mapping):
                run_options["response_format"] = self._normalize_response_format_dict(
                    cast("Mapping[str, Any]", response_format),
                    model=run_options.get("model"),
                )
            else:
                # The SDK helper copies a model class's __name__ into
                # json_schema.name unchecked; sanitize the envelope it built.
                run_options["response_format"] = _sanitize_enveloped_response_format(
                    cast("dict[str, Any]", type_to_response_format_param(response_format))
                )
        return run_options

    @staticmethod
    def _normalize_response_format_dict(response_format: Mapping[str, Any], *, model: Any = None) -> dict[str, Any]:
        """Wrap raw JSON schemas (e.g. ``{"type": "object", ...}``) in the json_schema envelope.

        Mirrors the Responses client's ``_convert_response_format`` handling of
        raw schemas so both clients accept the same inputs. Enveloped
        ``json_schema`` dicts keep their shape but get their name sanitized
        (copy-on-write); ``json_object``/``text`` pass through unchanged; any
        other mapping is a raw schema — JSON Schema admits array-valued and
        absent ``type`` (e.g. enum-only schemas), so classifying by primitive
        type names or keyword sniffing under-matches.
        """
        if not _is_string_keyed_dict(response_format):
            # Materialize read-only views (e.g. ``MappingProxyType``) once;
            # plain dicts keep identity for the passthrough branches below.
            response_format = dict(response_format)
        format_type = response_format.get("type")
        if format_type == "json_schema":
            return _sanitize_enveloped_response_format(response_format)
        # Tuple membership compares by equality, so an array-valued ``type``
        # (unhashable) classifies as a raw schema instead of raising.
        if format_type in ("json_object", "text"):
            return response_format
        # Materialized copy: the strictifier mutates nested mappings in
        # place, and the caller's schema may nest read-only views.
        schema = _materialize_json_structure(response_format)
        # Pop title from schema since OpenAI strict mode rejects unknown keys;
        # use it (sanitized) as the schema name in the envelope instead.
        name = _sanitize_response_format_name(schema.pop("title", None))
        schema, strict = _strictify_response_schema(schema, model=model)
        envelope: dict[str, Any] = {"name": name, "schema": schema}
        if strict:
            envelope["strict"] = True
        return {"type": "json_schema", "json_schema": envelope}

    def _parse_response_from_openai(self, response: ChatCompletion, options: Mapping[str, Any]) -> ChatResponse:
        """Parse a response from OpenAI into a ChatResponse."""
        response_metadata = self._get_metadata_from_chat_response(response)
        messages: list[Message] = []
        finish_reason: FinishReason | None = None
        for choice in response.choices:
            response_metadata.update(self._get_metadata_from_chat_choice(choice))
            if choice.finish_reason:
                finish_reason = FinishReason(choice.finish_reason)
            contents: list[Content] = []
            # Keep the historical text/tool/reasoning order for complete
            # responses. Unlike streamed deltas, these fields have no chunk
            # boundary that can split visible text; its reconstruction is unchanged.
            contents.extend(self._parse_text_contents_from_openai(choice))
            if parsed_tool_calls := list(self._parse_tool_calls_from_openai(choice)):
                contents.extend(parsed_tool_calls)
            contents.extend(self._parse_reasoning_from_choice_message(choice))
            messages.append(
                Message(
                    role="assistant",
                    contents=contents,
                    additional_properties=self._reasoning_additional_properties_from_choice_message(choice),
                )
            )
        return ChatResponse(
            response_id=response.id,
            created_at=openai_created_at_iso(response.created),
            usage_details=self._parse_usage_from_openai(response.usage) if response.usage else None,
            messages=messages,
            model=response.model,
            additional_properties=response_metadata,
            finish_reason=finish_reason,
            response_format=options.get("response_format"),
        )

    def _parse_response_update_from_openai(
        self,
        chunk: ChatCompletionChunk,
        *,
        stream_state: _OpenAIChatStreamState | None = None,
    ) -> ChatResponseUpdate:
        """Parse a streaming response update from OpenAI."""
        chunk_metadata = self._get_metadata_from_streaming_chat_response(chunk)
        contents: list[Content] = []
        finish_reason: FinishReason | None = None

        # Gemini can send usage alongside text or tool content. Process both
        # parts of the chunk so usage handling never drops the response content.
        if chunk.usage:
            contents.append(
                Content.from_usage(usage_details=self._parse_usage_from_openai(chunk.usage), raw_representation=chunk)
            )

        for choice in chunk.choices:
            chunk_metadata.update(self._get_metadata_from_chat_choice(choice))
            if choice.finish_reason:
                finish_reason = FinishReason(choice.finish_reason)

            # Some OpenAI-compatible providers (e.g. Azure) send `"delta": null`
            # on finish chunks instead of the spec-compliant `"delta": {}`.
            # Guard here so all content-parsing below can assume delta is present.
            if choice.delta is None:  # pyright: ignore[reportUnnecessaryComparison]
                continue

            include_plain_reasoning = True
            if stream_state is None:
                contents.extend(self._parse_tool_calls_from_openai(choice))
            else:
                choice_state = stream_state.choice(choice.index)
                reasoning_fields = _reasoning_fields_from_openai_message(choice.delta)
                plain_reasoning = next(
                    ((field, value) for field, value in reasoning_fields.items() if field != REASONING_DETAILS_FIELD),
                    None,
                )
                if plain_reasoning is not None and plain_reasoning[1] == "":
                    include_plain_reasoning = False
                    if not choice_state.empty_reasoning_emitted and not choice_state.saw_nonempty_reasoning:
                        # Preserve field presence once, before visible content
                        # and before the terminal function-call boundary. This
                        # avoids fragmenting text and keeps a length-truncated
                        # function call as the response's final content.
                        contents.append(
                            Content.from_text_reasoning(
                                text="",
                                additional_properties={REASONING_FORMAT_KEY: plain_reasoning[0]},
                                raw_representation=choice.delta,
                            )
                        )
                        choice_state.empty_reasoning_emitted = True
                elif plain_reasoning is not None:
                    choice_state.saw_nonempty_reasoning = True
                _accumulate_openai_tool_calls(choice, stream_state)
            # These fields carry no relative order. Interpret a mixed delta as
            # the conventional end-of-reasoning / start-of-answer boundary,
            # so chunking alone cannot insert reasoning into the answer text.
            # Across deltas, preserve the provider's original order.
            contents.extend(
                self._parse_reasoning_from_choice_delta(
                    choice,
                    include_plain_reasoning=include_plain_reasoning,
                )
            )
            contents.extend(self._parse_text_contents_from_openai(choice))
        return ChatResponseUpdate(
            created_at=openai_created_at_iso(chunk.created),
            contents=contents,
            role="assistant",
            model=chunk.model,
            additional_properties=chunk_metadata,
            finish_reason=finish_reason,
            raw_representation=chunk,
            response_id=chunk.id,
            message_id=chunk.id,
        )

    def _parse_usage_from_openai(self, usage: CompletionUsage) -> UsageDetails:
        details = UsageDetails(
            input_token_count=usage.prompt_tokens,
            output_token_count=usage.completion_tokens,
            total_token_count=usage.total_tokens,
        )
        if usage.completion_tokens_details:
            if tokens := usage.completion_tokens_details.accepted_prediction_tokens:
                details["completion/accepted_prediction_tokens"] = tokens  # type: ignore[typeddict-unknown-key]
            if tokens := usage.completion_tokens_details.audio_tokens:
                details["completion/audio_tokens"] = tokens  # type: ignore[typeddict-unknown-key]
            if (tokens := usage.completion_tokens_details.reasoning_tokens) is not None:
                details["completion/reasoning_tokens"] = tokens  # type: ignore[typeddict-unknown-key]
                details["reasoning_output_token_count"] = tokens
            if tokens := usage.completion_tokens_details.rejected_prediction_tokens:
                details["completion/rejected_prediction_tokens"] = tokens  # type: ignore[typeddict-unknown-key]
        if usage.prompt_tokens_details:
            if tokens := usage.prompt_tokens_details.audio_tokens:
                details["prompt/audio_tokens"] = tokens  # type: ignore[typeddict-unknown-key]
            if (tokens := usage.prompt_tokens_details.cached_tokens) is not None:
                details["prompt/cached_tokens"] = tokens  # type: ignore[typeddict-unknown-key]
                details["cache_read_input_token_count"] = tokens
            # Untyped in the SDK (extra='allow' retains it): reported for
            # billed explicit prompt caching. Explicit 0 is preserved.
            if (tokens := getattr(usage.prompt_tokens_details, "cache_write_tokens", None)) is not None:
                details["prompt/cache_write_tokens"] = tokens  # type: ignore[typeddict-unknown-key]
                details["cache_creation_input_token_count"] = tokens
        return details

    def _parse_text_contents_from_openai(self, choice: Choice | ChunkChoice) -> list[Content]:
        """Parse content and refusal independently; reject non-string values in either field."""
        contents: list[Content] = []
        if text_content := self._parse_text_from_openai(choice):
            contents.append(text_content)
        message = choice.message if isinstance(choice, Choice) else choice.delta
        if isinstance(refusal := message.refusal, str) and refusal:
            contents.append(Content.from_text(text=refusal, raw_representation=choice))
        return contents

    def _parse_text_from_openai(self, choice: Choice | ChunkChoice) -> Content | None:
        """Parse only the ordinary content field, without interpreting structured payloads."""
        message = choice.message if isinstance(choice, Choice) else choice.delta
        if message.content:
            if not isinstance(message.content, str):
                logger.debug("Ignoring non-string Chat Completions content of type %s", type(message.content).__name__)
                return None
            return Content.from_text(text=message.content, raw_representation=choice)
        return None

    def _parse_reasoning_from_choice_message(self, choice: Choice) -> list[Content]:
        """Capture raw reasoning fields from a non-streaming choice.

        Established dialects are captured losslessly when present —
        ``reasoning_details`` (OpenRouter style) first, then
        ``reasoning_content`` (DeepSeek/GLM style). vLLM's ``reasoning`` is a
        fallback only when neither is present, preventing mirrored aliases from
        duplicating the reasoning transcript. Every captured value is a
        protected JSON payload stamped with its wire field name so replay can
        route it back to the exact field it came from.
        """
        return [
            Content.from_text_reasoning(
                protected_data=json.dumps(value),
                additional_properties={REASONING_FORMAT_KEY: field},
            )
            for field, value in _reasoning_fields_from_openai_message(choice.message).items()
        ]

    def _parse_reasoning_from_choice_delta(
        self,
        choice: ChunkChoice,
        *,
        include_plain_reasoning: bool = True,
    ) -> list[Content]:
        """Capture raw reasoning fields from a streaming delta.

        ``reasoning_details`` deltas stay protected JSON. Plaintext
        ``reasoning_content`` or fallback ``reasoning`` deltas ride as visible
        reasoning text (the payload IS the display text), stamped with their
        exact wire field name.
        """
        contents: list[Content] = []
        for reasoning_field, value in _reasoning_fields_from_openai_message(choice.delta).items():
            if reasoning_field == REASONING_DETAILS_FIELD:
                contents.append(
                    Content.from_text_reasoning(
                        protected_data=json.dumps(value),
                        additional_properties={REASONING_FORMAT_KEY: reasoning_field},
                    )
                )
                continue
            if not include_plain_reasoning:
                continue
            if not isinstance(value, str):
                logger.debug(
                    "Ignoring non-string Chat Completions reasoning delta of type %s",
                    type(value).__name__,
                )
                continue
            contents.append(
                Content.from_text_reasoning(
                    text=value,
                    additional_properties={REASONING_FORMAT_KEY: reasoning_field},
                )
            )
        return contents

    def _reasoning_additional_properties_from_choice_message(self, choice: Choice) -> dict[str, Any]:
        """Stash the choice's raw reasoning fields on the kernel message.

        Message-level props are a duplicate representation of the captured
        contents (replay prefers content-level per field); the format key
        records the leading dialect, details-first.
        """
        props = _reasoning_fields_from_openai_message(choice.message)
        if props:
            props[REASONING_FORMAT_KEY] = next(iter(props))
        return props

    def _get_metadata_from_chat_response(self, response: ChatCompletion) -> dict[str, Any]:
        """Get metadata from a chat response."""
        return {
            "system_fingerprint": getattr(response, "system_fingerprint", None),
        }

    def _get_metadata_from_streaming_chat_response(self, response: ChatCompletionChunk) -> dict[str, Any]:
        """Get metadata from a streaming chat response."""
        return {
            "system_fingerprint": getattr(response, "system_fingerprint", None),
        }

    def _get_metadata_from_chat_choice(self, choice: Choice | ChunkChoice) -> dict[str, Any]:
        """Get metadata from a chat choice."""
        return {
            "logprobs": getattr(choice, "logprobs", None),
        }

    def _parse_tool_calls_from_openai(self, choice: Choice | ChunkChoice) -> list[Content]:
        """Parse tool calls from an OpenAI response choice."""
        resp: list[Content] = []
        content = choice.message if isinstance(choice, Choice) else choice.delta
        if content and content.tool_calls:
            for tool in content.tool_calls:
                if not isinstance(tool, ChatCompletionMessageCustomToolCall) and tool.function:
                    # ignoring tool.custom
                    fcc = Content.from_function_call(
                        call_id=tool.id or "",
                        name=tool.function.name or "",
                        arguments=tool.function.arguments or "",
                        raw_representation=tool.function,
                    )
                    resp.append(fcc)

        # When you enable asynchronous content filtering in Azure OpenAI, you may receive empty deltas
        return resp

    def _degrade_cross_provider_hosted_history(self, chat_messages: Sequence[Message]) -> Sequence[Message]:
        """Replace provider-hosted history with neutral assistant context.

        The Chat Completions wire has no hosted item types: a replayed raw
        hosted content reaches the endpoint as an unknown content part, and
        an informational hosted call becomes a dangling ``tool_calls`` entry
        with no tool response — strict endpoints reject both. Rewriting the
        messages before serialization covers every prep path, including the
        reasoning-coalescer delegation. Request-local summaries retain a
        private text boundary marker that is removed during serialization.
        """
        # Chat Completions dialects host no tools, so every provider-marked
        # hosted content in history is foreign to this wire.
        degradations = cross_provider_hosted_degradations(chat_messages, target_provider="")
        if not degradations:
            return chat_messages
        prepared: list[Message] = []
        for message in chat_messages:
            if not any(id(content) in degradations for content in message.contents):
                prepared.append(message)
                continue
            contents: list[Content] = []
            for content in message.contents:
                if id(content) in degradations:
                    if summary := degradations[id(content)]:
                        contents.append(
                            Content.from_text(text=summary, additional_properties={_HOSTED_CONTEXT_SUMMARY_KEY: True})
                        )
                    continue
                contents.append(content)
            prepared.append(
                Message(
                    message.role,
                    contents,
                    author_name=message.author_name,
                    message_id=message.message_id,
                    additional_properties=message.additional_properties,
                    raw_representation=message.raw_representation,
                )
            )
        return prepared

    def _prepare_messages_for_openai(
        self,
        chat_messages: Sequence[Message],
        role_key: str = "role",
        content_key: str = "content",
    ) -> list[dict[str, Any]]:
        """Prepare the chat history for an OpenAI request.

        Allowing customization of the key names for role/author, and optionally overriding the role.

        "tool" messages need to be formatted different than system/user/assistant messages:
            They require a "tool_call_id" and (function) "name" key, and the "metadata" key should
            be removed. The "encoding" key should also be removed.

        Override this method to customize the formatting of the chat history for a request.

        Args:
            chat_messages: The chat history to prepare.
            role_key: The key name for the role/author.
            content_key: The key name for the content/message.

        Returns:
            prepared_chat_history (Any): The prepared chat history for a request.
        """
        chat_messages = self._degrade_cross_provider_hosted_history(chat_messages)
        replay_reasoning = self._should_replay_reasoning(chat_messages)
        list_of_list = [
            self._prepare_message_for_openai(message, replay_reasoning=replay_reasoning) for message in chat_messages
        ]
        # Flatten the list of lists into a single list
        return self._insert_synthetic_image_messages(list(chain.from_iterable(list_of_list)))

    def _should_replay_reasoning(self, chat_messages: Sequence[Message]) -> bool:
        """Whether historical raw reasoning fields are re-sent on this request.

        The base replays unconditionally: endpoints that document preserved
        thinking (GLM) require historical reasoning on every multi-turn
        request, and Kimi — which runs on the plain ``openai`` provider —
        documents that each historical assistant message must keep its
        ``reasoning_content``. Subclasses narrow this (DeepSeek replays only
        once a tool interaction exists).

        Replay cannot be scoped by provider (Kimi and native OpenAI share
        the ``openai`` provider, differing only in base URL) and does not
        need to be: OpenAI's Chat Completions API silently drops unknown
        message-level fields before tokenization — verified live, no 400
        and zero prompt-token cost — so cross-provider histories replay
        safely. Its strict validation applies to top-level request
        parameters only; do not conflate the two layers.
        """
        return True

    # region Parsers

    def _lower_function_result_to_openai(self, content: Content) -> tuple[str, list[dict[str, Any]]]:
        """Lower image tool-result items for Chat Completions-compatible APIs."""
        if not content.items:
            return content.result if content.result is not None else "", []

        text_parts: list[str] = []
        pending_image_parts: list[dict[str, Any]] = []
        omitted_rich_content = False
        for item_index, item in enumerate(content.items, start=1):
            if item.type == "text":
                text_parts.append(item.text or "")
            elif item.type in ("data", "uri"):
                if not is_image_content(item) or not isinstance(item.uri, str):
                    omitted_rich_content = True
                    continue
                image_part = self._prepare_image_content_for_openai(item)
                pending_image_parts.append(
                    {
                        "type": "text",
                        "text": f"Image from tool call {content.call_id}, item {item_index}:",
                    }
                )
                pending_image_parts.append(image_part)

        if omitted_rich_content:
            logger.warning(
                "OpenAI Chat Completions API does not support non-image rich content "
                "in tool results. Rich content items will be omitted. "
                "Use the Responses API client for rich tool results."
            )
        if pending_image_parts:
            text_parts.append("(see following user message for image)")
        return "\n".join(text_parts) if text_parts else "", pending_image_parts

    def _insert_synthetic_image_messages(self, all_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Insert one synthetic user image message after each run of tool messages."""
        prepared: list[dict[str, Any]] = []
        pending_image_parts: list[dict[str, Any]] = []
        for message in all_messages:
            image_parts = message.pop(_PENDING_IMAGE_PARTS_KEY, None)
            if message.get("role") == "tool":
                if isinstance(image_parts, list):
                    pending_image_parts.extend(cast(list[dict[str, Any]], image_parts))
                prepared.append(message)
                continue
            if pending_image_parts:
                prepared.append({"role": "user", "content": pending_image_parts})
                pending_image_parts = []
            prepared.append(message)
        if pending_image_parts:
            prepared.append({"role": "user", "content": pending_image_parts})
        return prepared

    def _prepare_image_content_for_openai(self, content: Content) -> dict[str, Any]:
        image_url_obj: dict[str, Any] = {"url": content.uri}
        detail = content.additional_properties.get("detail")
        if isinstance(detail, str):
            image_url_obj["detail"] = detail
        return {
            "type": "image_url",
            "image_url": image_url_obj,
        }

    def _aggregate_reasoning_payload(self, fields: dict[str, Any], field: str, payload: Any) -> None:
        """Fold one reasoning contribution into the per-field aggregate.

        Plaintext ``reasoning_content`` / ``reasoning`` values concatenate
        when both are strings;
        ``reasoning_details`` values extend when both are lists; any other
        shape pairing is last-wins (never a cross-type crash or concat).
        """
        if field not in fields:
            fields[field] = payload
            return
        existing = fields[field]
        if (
            field in (REASONING_CONTENT_FIELD, REASONING_FIELD)
            and isinstance(existing, str)
            and isinstance(payload, str)
        ):
            fields[field] = existing + payload
        elif field == REASONING_DETAILS_FIELD and isinstance(existing, list) and isinstance(payload, list):
            fields[field] = [*cast(list[Any], existing), *cast(list[Any], payload)]
        else:
            fields[field] = payload

    def _reasoning_contribution(self, content: Content) -> tuple[str, Any] | None:
        """The (field, payload) one reasoning content contributes at replay, or ``None``.

        The single definition of replayability: the format stamp is
        authoritative (a stamp we don't recognize marks another dialect's
        state, and replaying it under a field it wasn't captured from would
        forge that dialect's payload); marked contents contribute their
        protected JSON payload when present (a payload that fails to decode —
        or decodes to ``null`` — is another provider's opaque state and
        contributes nothing, suppressing the content's text), else their
        non-``None`` text; unmarked contents contribute to
        ``reasoning_details`` only through a decodable protected payload
        (legacy pre-marker histories).
        """
        if content.additional_properties.get(_ANTHROPIC_REDACTED_THINKING_KEY):
            return None
        marker = content.additional_properties.get(REASONING_FORMAT_KEY)
        field = marker if marker in _REASONING_FIELDS else None
        if field is None and REASONING_FORMAT_KEY in content.additional_properties:
            return None
        if content.protected_data is not None:
            payload = _parse_reasoning_details_payload(content.protected_data)
            if payload is None:
                return None
            return (field if field is not None else REASONING_DETAILS_FIELD, payload)
        if field is None:
            return None
        if content.text is not None:
            return (field, content.text)
        return None

    def _replayable_reasoning_fields(self, message: Message) -> dict[str, Any]:
        """Aggregate a message's replayable raw reasoning fields, in capture order.

        Contribution rules live in ``_reasoning_contribution``; message-level
        props are a duplicate representation and fill in only fields no
        content supplied.
        """
        fields: dict[str, Any] = {}
        content_fields: set[str] = set()
        for content in message.contents:
            if content.type != "text_reasoning":
                continue
            contribution = self._reasoning_contribution(content)
            if contribution is None:
                continue
            field, payload = contribution
            self._aggregate_reasoning_payload(fields, field, payload)
            content_fields.add(field)
        for field in _REASONING_FIELDS:
            if field in content_fields:
                continue
            value = message.additional_properties.get(field)
            if value is not None:
                fields[field] = value
        return fields

    def _prepare_reasoning_assistant_message(self, message: Message) -> list[dict[str, Any]]:
        """Assemble a reasoning-bearing assistant message around coalesced aggregates.

        Text and ``function_call`` fragments coalesce into one aggregate wire
        message per run, so a text/tool_calls pair never reaches the wire as a
        split the downstream canonicalizer would have to refuse. The aggregate
        content stays a plain string (assistant content arrays only admit
        text/refusal parts, and GLM's documented replay shape is a string);
        non-text fragments keep their standalone emission ahead of their run's
        aggregate. Tool results end the current run and stay standalone
        ``role: "tool"`` records in source order (the only valid wire shape
        for a ``tool_call_id`` record), preserving call→result adjacency.
        Within each run, tool calls separate text reconstruction segments;
        degraded hosted summaries occupy their own segments and reasoning
        does not split text. These segments share the same wire aggregate.
        Reasoning ownership is per run: each run's contributions ride that
        run's aggregate, next to the tool calls that thinking produced, and
        are never reassigned backward across a result boundary. A run holding
        ONLY reasoning defers past the contiguous result block (a carrier
        inside the block would split tool_calls from their results) and joins
        the next run's aggregate, or emits as a trailing empty-content
        carrier; message-level props fill in once, message-wide, for fields
        no content supplied.
        """
        all_messages: list[dict[str, Any]] = []
        text_segments: list[list[tuple[str, str | None]]] = []
        tool_calls: list[dict[str, Any]] = []
        run_reasoning: dict[str, Any] = {}
        content_fields: set[str] = set()

        def end_text_segment() -> None:
            if text_segments and text_segments[-1]:
                text_segments.append([])

        def flush_run(*, force_reasoning_carrier: bool = False) -> None:
            if not text_segments and not tool_calls and not (force_reasoning_carrier and run_reasoning):
                return
            aggregate: dict[str, Any] = {"role": message.role}
            if author_name := _sanitize_author_name(message.author_name):
                aggregate["name"] = author_name
            aggregate["content"] = join_text_blocks(
                block for segment in text_segments for block in reconstruct_text_blocks(segment)
            )
            aggregate.update(run_reasoning)
            if tool_calls:
                aggregate["tool_calls"] = list(tool_calls)
            all_messages.append(aggregate)
            text_segments.clear()
            tool_calls.clear()
            run_reasoning.clear()

        for content in message.contents:
            match content.type:
                case "text_reasoning":
                    contribution = self._reasoning_contribution(content)
                    if contribution is None:
                        continue
                    field, payload = contribution
                    self._aggregate_reasoning_payload(run_reasoning, field, payload)
                    content_fields.add(field)
                case "function_call":
                    tool_calls.append(self._prepare_content_for_openai(content))
                    end_text_segment()
                case "text":
                    if content.text is not None:
                        is_hosted_summary = content.additional_properties.get(_HOSTED_CONTEXT_SUMMARY_KEY) is True
                        if is_hosted_summary:
                            end_text_segment()
                        if not text_segments:
                            text_segments.append([])
                        text_segments[-1].append((content.text, text_block_id(content.additional_properties)))
                        if is_hosted_summary:
                            end_text_segment()
                case "function_result":
                    # A reasoning-only run does NOT flush here: it defers past
                    # the result block and joins the next run instead.
                    flush_run()
                    result_args: dict[str, Any] = {"role": "tool", "tool_call_id": content.call_id}
                    result_args["content"], pending_image_parts = self._lower_function_result_to_openai(content)
                    if pending_image_parts:
                        result_args[_PENDING_IMAGE_PARTS_KEY] = pending_image_parts
                    all_messages.append(result_args)
                case _:
                    if content.type in TOOL_CALL_CONTENT_TYPES:
                        end_text_segment()
                    excluded_args: dict[str, Any] = {"role": message.role}
                    if author_name := _sanitize_author_name(message.author_name):
                        excluded_args["name"] = author_name
                    excluded_args["content"] = [self._prepare_content_for_openai(content)]
                    all_messages.append(excluded_args)
        flush_run(force_reasoning_carrier=True)
        props_fields: dict[str, Any] = {}
        for field in _REASONING_FIELDS:
            if field in content_fields:
                continue
            value = message.additional_properties.get(field)
            if value is not None:
                props_fields[field] = value
        self._attach_reasoning_fields_to_messages(all_messages, message, props_fields)
        return all_messages

    def _attach_reasoning_fields_to_messages(
        self,
        all_messages: list[dict[str, Any]],
        message: Message,
        reasoning_fields: dict[str, Any],
        *,
        start: int = 0,
    ) -> None:
        """Attach a reasoning aggregate to its wire carrier exactly once.

        The carrier is the last message with ``tool_calls`` (DeepSeek's and
        GLM's documented replay shape puts reasoning on the tool-calls
        assistant message), else the last same-role TEXTUAL message that is
        not a tool-result record (reasoning rides string-content messages; a
        multimodal content array must never carry it), else — for assistant
        messages only — a synthesized empty-content message appended at the
        end of ``all_messages``. ``start`` bounds the carrier search to the
        messages a single run emitted, giving per-run reasoning ownership at
        result boundaries. Reasoning that finds no carrier on a non-assistant
        message is dropped rather than emitted as an invalid wire message.
        """
        if not reasoning_fields:
            return
        candidates = all_messages[start:]
        for msg in reversed(candidates):
            if "tool_calls" in msg:
                msg.update(reasoning_fields)
                return
        for msg in reversed(candidates):
            if msg.get("role") != message.role or "tool_call_id" in msg:
                continue
            content_value = msg.get("content")
            if isinstance(content_value, str) or (
                isinstance(content_value, list)
                and all(
                    isinstance(item, Mapping) and cast(Mapping[str, Any], item).get("type") == "text"
                    for item in cast(list[object], content_value)
                )
            ):
                msg.update(reasoning_fields)
                return
        if message.role != "assistant":
            return
        pending_args: dict[str, Any] = {"role": message.role, "content": "", **reasoning_fields}
        if author_name := _sanitize_author_name(message.author_name):
            pending_args["name"] = author_name
        all_messages.append(pending_args)

    def _prepare_message_for_openai(
        self,
        message: Message,
        *,
        replay_reasoning: bool = True,
    ) -> list[dict[str, Any]]:
        """Prepare a chat message for OpenAI."""
        # System/developer messages must use plain string content because some
        # OpenAI-compatible endpoints reject list content for non-user roles.
        # Reasoning contents are never replayed on these roles.
        if message.role in ("system", "developer"):
            texts = [content.text for content in message.contents if content.type == "text" and content.text]
            if texts:
                sys_args: dict[str, Any] = {"role": message.role, "content": "\n".join(texts)}
                if author_name := _sanitize_author_name(message.author_name):
                    sys_args["name"] = author_name
                return [sys_args]
            return []

        if replay_reasoning and message.role == "assistant" and self._replayable_reasoning_fields(message):
            return self._prepare_reasoning_assistant_message(message)

        all_messages: list[dict[str, Any]] = []
        pending_reasoning: dict[str, Any] = {}
        for content in message.contents:
            args: dict[str, Any] = {
                "role": message.role,
            }
            if message.role != "tool" and (author_name := _sanitize_author_name(message.author_name)):
                args["name"] = author_name
            if replay_reasoning:
                for field in _REASONING_FIELDS:
                    value = message.additional_properties.get(field)
                    if value is not None:
                        args[field] = value
            match content.type:
                case "function_call":
                    if all_messages and "tool_calls" in all_messages[-1]:
                        # If the last message already has tool calls, append to it
                        all_messages[-1]["tool_calls"].append(self._prepare_content_for_openai(content))
                    else:
                        args["tool_calls"] = [self._prepare_content_for_openai(content)]
                case "function_result":
                    args["tool_call_id"] = content.call_id
                    args["content"], pending_image_parts = self._lower_function_result_to_openai(content)
                    if pending_image_parts:
                        args[_PENDING_IMAGE_PARTS_KEY] = pending_image_parts
                    all_messages.append(args)
                    continue
                case "text_reasoning":
                    if not replay_reasoning:
                        continue
                    contribution = self._reasoning_contribution(content)
                    if contribution is not None:
                        # Buffer reasoning to attach to the next message with content/tool_calls
                        reasoning_field, reasoning_payload = contribution
                        self._aggregate_reasoning_payload(pending_reasoning, reasoning_field, reasoning_payload)
                    continue
                case _:
                    if "content" not in args:
                        args["content"] = []
                    # this is a list to allow multi-modal content
                    args["content"].append(self._prepare_content_for_openai(content))
            if "content" in args or "tool_calls" in args:
                if pending_reasoning:
                    args.update(pending_reasoning)
                    pending_reasoning = {}
                all_messages.append(args)

        # Only assistants may synthesize an empty-content reasoning carrier.
        if pending_reasoning:
            if all_messages:
                all_messages[-1].update(pending_reasoning)
            elif message.role == "assistant":
                pending_args: dict[str, Any] = {
                    "role": message.role,
                    "content": "",
                    **pending_reasoning,
                }
                if message.role != "tool" and (author_name := _sanitize_author_name(message.author_name)):
                    pending_args["name"] = author_name
                all_messages.append(pending_args)

        # Endpoints such as Foundry Local require plain strings for text-only
        # messages. Flatten those lists while retaining multimodal content arrays.
        for msg in all_messages:
            msg_content: Any = msg.get("content")
            if isinstance(msg_content, list):
                typed_msg_content = cast(list[object], msg_content)
                text_items: list[Mapping[str, Any]] = []
                for item in typed_msg_content:
                    if not isinstance(item, Mapping):
                        break
                    text_item = cast(Mapping[str, Any], item)
                    if text_item.get("type") != "text":
                        break
                    text_items.append(text_item)
                else:
                    msg["content"] = "\n".join(
                        text_item.get("text", "") if isinstance(text_item.get("text", ""), str) else ""
                        for text_item in text_items
                    )

        return all_messages

    def _prepare_content_for_openai(self, content: Content) -> dict[str, Any]:
        """Prepare content for OpenAI."""
        match content.type:
            case "text" if _HOSTED_CONTEXT_SUMMARY_KEY in content.additional_properties:
                # Multimodal messages can retain text-part metadata on the wire.
                # Remove only the request-local marker, without changing history.
                prepared = content.to_dict(exclude_none=True)
                properties = dict(prepared["additional_properties"])
                properties.pop(_HOSTED_CONTEXT_SUMMARY_KEY)
                if properties:
                    prepared["additional_properties"] = properties
                else:
                    prepared.pop("additional_properties")
                return prepared
            case "function_call":
                args = json.dumps(content.arguments) if isinstance(content.arguments, Mapping) else content.arguments
                return {
                    "id": content.call_id,
                    "type": "function",
                    "function": {"name": content.name, "arguments": args},
                }
            case "function_result":
                return {
                    "tool_call_id": content.call_id,
                    "content": content.result if content.result is not None else "",
                }
            case "data" | "uri" if content.has_top_level_media_type("image"):
                return self._prepare_image_content_for_openai(content)
            case "data" | "uri" if content.has_top_level_media_type("audio"):
                if content.media_type and "wav" in content.media_type:
                    audio_format = "wav"
                elif content.media_type and "mp3" in content.media_type:
                    audio_format = "mp3"
                else:
                    # Fallback to default to_dict for unsupported audio formats
                    return content.to_dict(exclude_none=True)

                # Extract base64 data from data URI
                audio_data = content.uri
                if audio_data is None:
                    return content.to_dict(exclude_none=True)
                if audio_data.startswith("data:"):
                    # Extract just the base64 part after "data:audio/format;base64,"
                    audio_data = audio_data.split(",", 1)[-1]

                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_data,
                        "format": audio_format,
                    },
                }
            case "data" | "uri" if (
                content.has_top_level_media_type("application")
                and content.uri is not None
                and content.uri.startswith("data:")
            ):
                # All application/* media types should be treated as files for OpenAI
                filename = getattr(content, "filename", None) or (
                    content.additional_properties.get("filename") if content.additional_properties else None
                )
                file_obj = {"file_data": content.uri}
                if filename:
                    file_obj["filename"] = filename
                return {
                    "type": "file",
                    "file": file_obj,
                }
            case _:
                # Default fallback for all other content types
                return content.to_dict(exclude_none=True)

    @override
    def service_url(self) -> str:
        """Get the URL of the service.

        Override this in the subclass to return the proper URL.
        If the service does not have a URL, return None.
        """
        return str(self.client.base_url) if self.client else "Unknown"
