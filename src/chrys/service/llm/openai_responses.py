# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chrys-owned OpenAI Responses raw wire client.

Parser and serializer logic, with provider
settings removed: callers must inject a pre-configured ``AsyncOpenAI`` client.
"""

from __future__ import annotations

import json
import logging
import shlex
from collections import Counter, deque
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Generic, Literal, NoReturn, TypeVar, cast, override

from openai import AsyncOpenAI, AsyncStream, BadRequestError
from openai.types.responses import FunctionShellTool
from openai.types.responses.file_search_tool_param import FileSearchToolParam
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.parsed_response import ParsedResponse
from openai.types.responses.response import Response as OpenAIResponse
from openai.types.responses.response_stream_event import ResponseStreamEvent as OpenAIResponseStreamEvent
from openai.types.responses.response_usage import ResponseUsage
from openai.types.responses.tool_param import (
    CodeInterpreter,
    CodeInterpreterContainerCodeInterpreterToolAuto,
    ImageGeneration,
    Mcp,
)
from openai.types.responses.web_search_tool_param import Filters, WebSearchToolParam
from pydantic import BaseModel
from typing_extensions import TypedDict

from chrys.foundation.hosted_tools import (
    OPENAI_HOSTED_WIRE_ITEM_KEY,
    PRESENTATION_TEXT_SEGMENT_ID_KEY,
    HostedRetrySafety,
    HostedToolFamily,
    HostedToolPhase,
    HostedToolStatus,
    normalize_hosted_tool_status,
)
from chrys.foundation.tool_kinds import KIND_SHELL as SHELL_TOOL_KIND_VALUE
from chrys.kernel import (
    OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY,
    Annotation,
    BaseChatClient,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    ContinuationToken,
    FunctionTool,
    Message,
    ResponseStream,
    Role,
    TextSpanRegion,
    ToolTypes,
    UsageDetails,
    detect_media_type_from_base64,
    normalize_tools,
    tool,
    validate_tool_mode,
)
from chrys.kernel.compaction import (
    GROUP_ANNOTATION_KEY,
    GROUP_HAS_REASONING_KEY,
    GROUP_KIND_KEY,
    CompactionStrategy,
    TokenizerProtocol,
    group_annotation_signature,
)
from chrys.kernel.exceptions import ChatClientException, ChatClientInvalidRequestException
from chrys.kernel.exchanges import (
    TOOL_CALL_CONTENT_TYPES,
    TOOL_RESULT_CONTENT_TYPES,
    EmptyIdPolicy,
    LiveAccessor,
    NoneIdPolicy,
    PairingKey,
    PairingPolicy,
    iter_exchanges,
    namespaced_pairing_key,
    pair_results,
)
from chrys.service.agent_middleware.events.hosted_tools import cross_provider_hosted_degradations
from chrys.service.llm._structured_outputs import (
    _materialize_json_structure,
    _sanitize_response_format_name,
    _strictify_response_schema,
)
from chrys.service.llm.openai_exceptions import OpenAIContentFilterException
from chrys.service.profiles.models.options import effective_store_option

logger = logging.getLogger(__name__)

OPENAI_SHELL_ENVIRONMENT_KEY = "openai.responses.shell.environment"
OPENAI_SHELL_OUTPUT_TYPE_KEY = "openai.responses.shell.output_type"
OPENAI_LOCAL_SHELL_CALL_ITEM_ID_KEY = "openai.responses.local_shell.call_item_id"
OPENAI_LOCAL_SHELL_COMMAND_PARTS_KEY = "openai.local_shell_command_parts"
OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL = "shell_call_output"
OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL = "local_shell_call_output"
OPENAI_HOSTED_REPLAY_SHADOW_KEY = "openai.responses.replay_shadow"
_OUTPUT_MESSAGE_ENVELOPE_FIELDS = ("id", "status", "phase")

# Internal marker emitted by `_prepare_content_for_openai` for an
# `mcp_server_tool_result` Content. The Responses API expects an `mcp_call`
# input item to carry both arguments and output as one item, so result
# Contents cannot be serialized standalone. `_prepare_messages_for_openai`
# coalesces these markers into the most recent matching `mcp_call` input
# item before returning, dropping any that are unmatched.
_CHRYS_MCP_PENDING_OUTPUT_KEY = "__chrys_pending_mcp_result__"
_CHRYS_IMAGE_GENERATION_RESULT_KEY = "__chrys_image_generation_result__"
_CHRYS_HOSTED_REPLAY_SHADOW_KEY = "__chrys_hosted_replay_shadow__"
_TOOL_CONTENT_TYPES = TOOL_CALL_CONTENT_TYPES | TOOL_RESULT_CONTENT_TYPES

_LIVE_ACCESSOR = LiveAccessor()
_REPLAY_PAIRING_POLICY = PairingPolicy(
    call_types=TOOL_CALL_CONTENT_TYPES,
    include_informational_calls=True,
    result_types=TOOL_RESULT_CONTENT_TYPES,
    none_id=NoneIdPolicy.POSITIONAL,
    empty_id=EmptyIdPolicy.POSITIONAL,
    malformed_id="stringify",
)


def _presentation_text_segment_id(event: Any) -> str:
    """Return a response-local identity for one streamed output-text part."""
    item_id = getattr(event, "item_id", None)
    output_index = getattr(event, "output_index", None)
    content_index = getattr(event, "content_index", None)
    if isinstance(item_id, str) and item_id:
        base = f"item:{item_id}"
    elif isinstance(output_index, int) and not isinstance(output_index, bool):
        base = f"output:{output_index}"
    else:
        return ""
    if isinstance(content_index, int) and not isinstance(content_index, bool):
        return f"{base}:content:{content_index}"
    return base


def _presentation_text_properties(event: Any) -> dict[str, str] | None:
    """Build private streamed-text provenance without leaking provider SDK objects."""
    segment_id = _presentation_text_segment_id(event)
    return {PRESENTATION_TEXT_SEGMENT_ID_KEY: segment_id} if segment_id else None


def _output_message_envelope(item: Any) -> dict[str, str]:
    """Capture replayable fields from an SDK response output message."""
    envelope: dict[str, str] = {}
    for field_name in _OUTPUT_MESSAGE_ENVELOPE_FIELDS:
        value = getattr(item, field_name, None)
        if isinstance(value, str) and value:
            envelope[field_name] = value
    return envelope


def _output_message_content_properties(item: Any) -> dict[str, Any] | None:
    """Wrap an output-message envelope for storage on parsed text content."""
    envelope = _output_message_envelope(item)
    return {OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY: envelope} if envelope else None


def _content_output_message_envelope(content: Content) -> dict[str, str]:
    """Return validated output-message replay fields stored on content."""
    raw_envelope = content.additional_properties.get(OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY)
    if not isinstance(raw_envelope, Mapping):
        return {}
    return {
        field_name: value
        for field_name in _OUTPUT_MESSAGE_ENVELOPE_FIELDS
        if isinstance((value := raw_envelope.get(field_name)), str) and value
    }


def _sanitize_text_format_model(model: type[BaseModel]) -> type[BaseModel]:
    """Return ``model`` or a renamed subclass whose wire schema name is valid.

    The SDK derives the ``text.format`` name from the class ``__name__``
    unchanged, so a Unicode or over-long class name — both legal Python —
    would 400 the request. The renamed subclass keeps the original model as
    the parse target: same fields and validators, and every parsed instance
    is an instance of the original class.
    """
    sanitized = _sanitize_response_format_name(model.__name__)
    if sanitized == model.__name__:
        return model
    return cast("type[BaseModel]", type(sanitized, (model,), {}))


def _sanitize_text_format_config(text_config: dict[str, Any]) -> dict[str, Any]:
    """Return a ``text`` config with a json_schema ``format.name`` sanitized.

    Single choke point for every name-producing form: the converted
    ``response_format`` branches all land in ``text.format`` before the wire,
    and a caller-supplied ``text`` mapping passes through here too. Copies only
    on change; non-json_schema ``format`` mappings stay byte-identical.
    """
    format_config = text_config.get("format")
    if not isinstance(format_config, Mapping):
        return text_config
    format_typed = cast("Mapping[str, Any]", format_config)
    if format_typed.get("type") != "json_schema":
        return text_config
    name = format_typed.get("name")
    sanitized = _sanitize_response_format_name(name)
    if name == sanitized:
        return text_config
    return {**text_config, "format": {**format_typed, "name": sanitized}}


@dataclass(slots=True, frozen=True)
class _ReplayMessageGroup:
    """One positional request group."""

    position: int
    messages: tuple[Message, ...]
    annotated_reasoning_tool_call: bool
    exchange_ordinal: int | None


@dataclass(slots=True)
class _ReasoningReplayPlan:
    """Serialization decisions for one positional request group."""

    group: _ReplayMessageGroup
    reasoning_items: dict[int, dict[str, Any]] = field(default_factory=dict)
    drop_content_ids: set[int] = field(default_factory=set)
    omit_function_call_id_content_ids: set[int] = field(default_factory=set)
    reasoning_ids: list[str] = field(default_factory=list)
    call_ids: list[str] = field(default_factory=list)
    wire_ids: list[str] = field(default_factory=list)
    mcp_wire_ids: list[str] = field(default_factory=list)
    mcp_result_ids: list[str] = field(default_factory=list)
    degraded: bool = False


def _is_reasoning_only_assistant(message: Message) -> bool:
    return (
        message.role == "assistant"
        and bool(message.contents)
        and all(content.type == "text_reasoning" for content in message.contents)
    )


def _is_tool_call_assistant(message: Message) -> bool:
    return message.role == "assistant" and any(content.type in TOOL_CALL_CONTENT_TYPES for content in message.contents)


def _annotation_marks_reasoning_tool_call(message: Message) -> bool:
    annotation = message.additional_properties.get(GROUP_ANNOTATION_KEY)
    return (
        isinstance(annotation, Mapping)
        and annotation.get(GROUP_KIND_KEY) == "tool_call"
        and annotation.get(GROUP_HAS_REASONING_KEY) is True
    )


def _is_tool_result_follower(message: Message) -> bool:
    """Whether a message only carries results for a preceding tool-call block."""
    if message.role == "tool":
        return True
    return (
        message.role == "assistant"
        and not _is_tool_call_assistant(message)
        and any(content.type in TOOL_RESULT_CONTENT_TYPES for content in message.contents)
    )


def _joins_annotation_run(
    run_signature: list[tuple[str, str, int, bool] | None],
    signature: tuple[str, str, int, bool] | None,
) -> bool:
    """Extend a group's annotation run; conflicting non-None signatures split."""
    if run_signature[0] is not None and signature is not None and signature != run_signature[0]:
        return False
    if run_signature[0] is None:
        run_signature[0] = signature
    return True


def _tool_call_keys(message: Message) -> set[PairingKey]:
    """Namespaced pairing keys announced by a message's tool-call contents."""
    return {
        key
        for content in message.contents
        if content.type in TOOL_CALL_CONTENT_TYPES
        and (key := namespaced_pairing_key(content.type, _LIVE_ACCESSOR.raw_id(content))) is not None
    }


def _tool_result_keys(message: Message) -> set[PairingKey]:
    """Namespaced pairing keys answered by a message's tool-result contents."""
    return {
        key
        for content in message.contents
        if content.type in TOOL_RESULT_CONTENT_TYPES
        and (key := namespaced_pairing_key(content.type, _LIVE_ACCESSOR.raw_id(content))) is not None
    }


def _result_follower_joins(
    message: Message,
    run_signature: list[tuple[str, str, int, bool] | None],
    signature: tuple[str, str, int, bool] | None,
    group_call_keys: set[PairingKey],
) -> bool:
    """Assistant-role results answering this group's calls join structurally:
    the annotator assigns them a span of their own, so a conflicting signature
    is expected, not a boundary. Results answering calls made elsewhere follow
    the annotation rule instead, so a stray orphan cannot attach to a valid
    group and degrade it."""
    if message.role == "assistant":
        result_keys = _tool_result_keys(message)
        if result_keys and result_keys <= group_call_keys:
            return True
    return _joins_annotation_run(run_signature, signature)


def _partition_replay_message_groups(messages: Sequence[Message]) -> list[_ReplayMessageGroup]:
    """Partition request history by contiguous assistant/tool structure.

    Which output run answers which call run — with history markers as hard
    boundaries — comes from the shared exchange grammar: a group only
    consumes result followers its own exchange owns, and a marker message
    is exchange chrome that always stands alone. Within an exchange the
    partition stays local: each call-carrying sibling opens its own group.
    Compaction annotations only ever split groups (a candidate whose
    signature conflicts with the group's stays out); joining remains purely
    structural so unannotated histories partition unchanged.
    """

    signatures = [group_annotation_signature(message) for message in messages]
    output_owner: dict[int, int] = {}
    call_owner: dict[int, int] = {}
    for ordinal, exchange in enumerate(iter_exchanges(messages, _LIVE_ACCESSOR)):
        for response_index in exchange.response_indices:
            call_owner[response_index] = ordinal
        for output_index in exchange.output_indices:
            output_owner[output_index] = ordinal

    def _follower_joins_group(index: int, owner: int | None) -> bool:
        """Whether the message may extend the group's output run.

        A group that consumed a call sibling owns exactly its exchange's
        output run; a call-less group has no exchange, so its orphan
        followers keep today's structural join (minus marker chrome) and
        degrade with the group.
        """
        if owner is not None:
            return output_owner.get(index) == owner
        return _is_tool_result_follower(messages[index]) and not _LIVE_ACCESSOR.has_marker(messages[index])

    def _message_exchange_owner(index: int) -> int | None:
        response_owner = call_owner.get(index)
        return response_owner if response_owner is not None else output_owner.get(index)

    groups: list[_ReplayMessageGroup] = []
    total = len(messages)
    index = 0
    while index < total:
        start = index
        run_signature: list[tuple[str, str, int, bool] | None] = [signatures[index]]
        group_call_keys: set[PairingKey] = set()
        owner: int | None = None
        message = messages[index]

        if _LIVE_ACCESSOR.has_marker(message):
            index += 1
        elif _is_reasoning_only_assistant(message):
            index += 1
            while (
                index < total
                and _is_reasoning_only_assistant(messages[index])
                and _joins_annotation_run(run_signature, signatures[index])
            ):
                index += 1
            if (
                index < total
                and _is_tool_call_assistant(messages[index])
                and _joins_annotation_run(run_signature, signatures[index])
            ):
                group_call_keys |= _tool_call_keys(messages[index])
                owner = call_owner.get(index)
                index += 1
                while (
                    index < total
                    and _is_reasoning_only_assistant(messages[index])
                    and _joins_annotation_run(run_signature, signatures[index])
                ):
                    index += 1
            while (
                index < total
                and _follower_joins_group(index, owner)
                and _result_follower_joins(messages[index], run_signature, signatures[index], group_call_keys)
            ):
                index += 1
        elif _is_tool_call_assistant(message):
            group_call_keys |= _tool_call_keys(message)
            owner = call_owner.get(index)
            index += 1
            while (
                index < total
                and _is_reasoning_only_assistant(messages[index])
                and _joins_annotation_run(run_signature, signatures[index])
            ):
                index += 1
            while (
                index < total
                and _follower_joins_group(index, owner)
                and _result_follower_joins(messages[index], run_signature, signatures[index], group_call_keys)
            ):
                index += 1
        elif _is_tool_result_follower(message):
            index += 1
            while (
                index < total
                and _is_tool_result_follower(messages[index])
                and not _LIVE_ACCESSOR.has_marker(messages[index])
                and _result_follower_joins(messages[index], run_signature, signatures[index], group_call_keys)
            ):
                index += 1
        else:
            index += 1

        group_messages = tuple(messages[start:index])
        exchange_owners = {
            exchange_owner
            for message_index in range(start, index)
            if (exchange_owner := _message_exchange_owner(message_index)) is not None
        }
        groups.append(
            _ReplayMessageGroup(
                position=len(groups),
                messages=group_messages,
                annotated_reasoning_tool_call=any(
                    _annotation_marks_reasoning_tool_call(item) for item in group_messages
                ),
                # The exchange grammar keeps a replay group within at most
                # one exchange. Fail closed for malformed histories rather
                # than coalescing hosted results across ambiguous ownership.
                exchange_ordinal=next(iter(exchange_owners)) if len(exchange_owners) == 1 else None,
            )
        )
    return groups


class OpenAIContinuationToken(ContinuationToken):
    """Continuation token for OpenAI Responses API background operations."""

    response_id: str
    """OpenAI Responses API response ID."""


# region OpenAI Responses Options TypedDict


class ReasoningOptions(TypedDict, total=False):
    """Configuration options for reasoning models (gpt-5, o-series).

    See: https://platform.openai.com/docs/guides/reasoning
    """

    effort: Literal["none", "low", "medium", "high", "xhigh"]
    """The effort level for reasoning. Higher effort means more reasoning tokens."""

    summary: Literal["auto", "concise", "detailed"]
    """How to summarize reasoning in the response."""


class StreamOptions(TypedDict, total=False):
    """Options for streaming responses."""

    include_usage: bool
    """Whether to include usage statistics in stream events."""


ResponseFormatT = TypeVar("ResponseFormatT", bound=BaseModel | None, default=None)


class OpenAIChatOptions(ChatOptions[ResponseFormatT], Generic[ResponseFormatT], total=False):
    """OpenAI Responses API-specific chat options.

    Extends ChatOptions with options specific to OpenAI's Responses API.
    These options provide fine-grained control over response generation,
    reasoning, and API behavior.

    See: https://platform.openai.com/docs/api-reference/responses/create
    """

    # Responses API-specific parameters

    include: list[str]
    """Additional output data to include in the response.
    Supported values include:
    - 'web_search_call.action.sources'
    - 'code_interpreter_call.outputs'
    - 'file_search_call.results'
    - 'message.input_image.image_url'
    - 'message.output_text.logprobs'
    - 'reasoning.encrypted_content'
    """

    max_tool_calls: int
    """Maximum number of total calls to built-in tools in a response."""

    prompt: dict[str, Any]
    """Reference to a prompt template and its variables.
    Learn more: https://platform.openai.com/docs/guides/text#reusable-prompts"""

    prompt_cache_key: str
    """Used by OpenAI to cache responses for similar requests.
    Replaces the deprecated 'user' field for caching purposes."""

    prompt_cache_retention: Literal["24h"]
    """Retention policy for prompt cache. Set to '24h' for extended caching."""

    reasoning: ReasoningOptions
    """Configuration for reasoning models (gpt-5, o-series).
    See: https://platform.openai.com/docs/guides/reasoning"""

    verbosity: Literal["low", "medium", "high"]
    """Output verbosity for GPT-5 family models. Lower values yield shorter responses.
    Translated to ``text.verbosity`` when sent to the Responses API.
    See: https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_new_params_and_tools#1-verbosity-parameter"""

    safety_identifier: str
    """A stable identifier for detecting policy violations.
    Recommend hashing username/email to avoid sending identifying info."""

    service_tier: Literal["auto", "default", "flex", "priority"]
    """Processing type for serving the request.
    - 'auto': Use project settings
    - 'default': Standard pricing/performance
    - 'flex': Flexible processing
    - 'priority': Priority processing"""

    stream_options: StreamOptions
    """Options for streaming responses. Only set when stream=True."""

    top_logprobs: int
    """Number of most likely tokens (0-20) to return at each position."""

    truncation: Literal["auto", "disabled"]
    """Truncation strategy for model response.
    - 'auto': Truncate from beginning if exceeds context
    - 'disabled': Fail with 400 error if exceeds context"""

    background: bool
    """Whether to run the model response in the background.
    When True, the response returns immediately with a continuation token
    that can be used to poll for the result.
    See: https://platform.openai.com/docs/guides/background"""

    continuation_token: OpenAIContinuationToken
    """Token for resuming or polling a long-running background operation.
    Pass the ``continuation_token`` from a previous response to poll for
    completion or resume a streaming response."""


# endregion


# region Helpers


def _annotations_to_output_text(annotations: Sequence[Annotation] | None) -> list[dict[str, Any]]:
    """Convert Chrys `Annotation` objects to Responses API `output_text` annotation dicts.

    Citations from `file_search`, `code_interpreter` file paths, and url citations all collapse
    to Chrys `Annotation(type="citation", ...)`. The original API form is recovered
    here so assistant messages roundtrip cleanly through history forwarding.

    Each Responses API annotation dict carries at most one `start_index`/`end_index` pair, so an
    `Annotation` with multiple `annotated_regions` is fanned out into one entry per region.
    Regions missing valid integer span bounds are skipped.
    """
    if not annotations:
        return []
    out: list[dict[str, Any]] = []
    for annotation in annotations:
        if annotation.get("type") != "citation":
            continue
        props = annotation.get("additional_properties") or {}
        regions = annotation.get("annotated_regions") or []
        file_id = annotation.get("file_id")
        url = annotation.get("url")
        title = annotation.get("title")
        container_id = props.get("container_id")

        if container_id and file_id:
            for region in regions:
                start = region.get("start_index")
                end = region.get("end_index")
                if not (isinstance(start, int) and isinstance(end, int)):
                    continue
                entry: dict[str, Any] = {
                    "type": "container_file_citation",
                    "container_id": container_id,
                    "file_id": file_id,
                    "start_index": start,
                    "end_index": end,
                }
                if url:
                    entry["filename"] = url
                out.append(entry)
        elif url and not file_id and regions:
            for region in regions:
                start = region.get("start_index")
                end = region.get("end_index")
                if not (isinstance(start, int) and isinstance(end, int)):
                    continue
                out.append(
                    {
                        "type": "url_citation",
                        "url": url,
                        "title": title or "",
                        "start_index": start,
                        "end_index": end,
                    }
                )
        elif file_id and url:
            entry = {
                "type": "file_citation",
                "file_id": file_id,
                "filename": url,
            }
            if (idx := props.get("index")) is not None:
                entry["index"] = idx
            out.append(entry)
        elif file_id:
            entry = {
                "type": "file_path",
                "file_id": file_id,
            }
            if (idx := props.get("index")) is not None:
                entry["index"] = idx
            out.append(entry)
    return out


# endregion


# region ResponsesClient


class RawOpenAIChatClient(BaseChatClient):
    """Raw OpenAI Responses client without middleware, telemetry, or function invocation.

    Warning:
        **This class should not normally be used directly.** It does not include middleware,
        telemetry, or function invocation support that you most likely need. If you do use it,
        you should consider which additional layers to apply. There is a defined ordering that
        you should follow:

        1. **ToolLoopLayer** - Owns the tool/function calling loop and routes function middleware
        2. **ChatMiddlewareLayer** - Applies chat middleware per model call and stays outside telemetry
        3. **ChatTelemetryLayer** - Must stay inside chat middleware for correct per-call telemetry

        Use ``create_instrumented_openai_responses_client`` for the production stack with all layers applied.
    """

    INJECTABLE: ClassVar[set[str]] = {"client"}
    STORES_BY_DEFAULT: ClassVar[bool] = True  # type: ignore[reportIncompatibleVariableOverride, misc]
    MIN_OUTPUT_CAP_TOKENS: ClassVar[int] = 16
    SUPPORTS_RICH_FUNCTION_OUTPUT: ClassVar[bool] = True
    INJECT_ENCRYPTED_REASONING_INCLUDE: ClassVar[bool] = True
    MINTS_CONTINUATION_TOKENS: ClassVar[bool] = True
    HOSTED_PROVIDER: ClassVar[str] = "openai"

    # Azure OpenAI Responses API may include this header in responses naming the actual model that
    # served the request (e.g. ``gpt-5-nano-2025-08-07``), which can differ from the deployment alias
    # that the request was addressed to and that ``response.model`` reports. When present, we use it
    # as the value of ``ChatResponse.model`` / ``ChatResponseUpdate.model`` so telemetry and callers
    # see the actually served model. (Chat Completions API already returns the snapshot in
    # ``response.model``, so this header only matters for the Responses API.)
    SERVED_MODEL_HEADER: ClassVar[str] = "x-ms-served-model"

    FILE_SEARCH_MAX_RESULTS: int = 50

    def __init__(
        self,
        model: str | None = None,
        *,
        async_client: AsyncOpenAI | None = None,
        instruction_role: str | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        additional_properties: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> None:
        del timeout
        if async_client is None:
            raise ValueError("RawOpenAIChatClient requires a pre-configured async_client.")
        self.client = async_client
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

    # region Inner Methods

    async def _prepare_request(
        self,
        messages: Sequence[Message],
        options: Mapping[str, Any],
    ) -> tuple[AsyncOpenAI, dict[str, Any], dict[str, Any]]:
        """Validate options and prepare the request.

        Returns:
            Tuple of (client, run_options, validated_options).
        """
        client = self.client
        validated_options = await self._validate_options(options)
        run_options = await self._prepare_options(messages, validated_options)
        return client, run_options, validated_options

    def _handle_request_error(self, ex: Exception) -> NoReturn:
        """Convert exceptions to appropriate service exceptions. Always raises."""
        if isinstance(ex, BadRequestError) and ex.code == "content_filter":
            raise OpenAIContentFilterException(
                f"{type(self)} service encountered a content error: {ex}",
                inner_exception=ex,
            ) from ex
        raise ChatClientException(
            f"{type(self)} service failed to complete the prompt: {ex}",
            inner_exception=ex,
        ) from ex

    @override
    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        options: Mapping[str, Any],
        stream: bool = False,
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        continuation_token: OpenAIContinuationToken | None = options.get("continuation_token")  # type: ignore[assignment]

        if stream:
            function_call_ids: dict[int, tuple[str, str]] = {}
            seen_reasoning_delta_item_ids: set[str] = set()
            hosted_call_contents: dict[int, Content] = {}
            hosted_result_contents: dict[int, Content] = {}
            output_message_envelopes: dict[int, dict[str, str]] = {}
            output_message_contents: dict[int, list[Content]] = {}
            validated_options: dict[str, Any] | None = None
            # Captured once request options are validated/prepared so the streaming finalizer can
            # still parse the aggregated response into structured output after the stream completes.
            response_format: Any | None = None

            def _finalize_with_captured_format(updates: Sequence[ChatResponseUpdate]) -> ChatResponse[Any]:
                # ResponseStream only calls the finalizer after iterating or draining `_stream()`,
                # so `response_format` has already been populated from the validated request state
                # unless request setup failed before streaming began.
                return self._finalize_response_updates(updates, response_format=response_format)

            async def _stream() -> AsyncIterable[ChatResponseUpdate]:
                nonlocal response_format, validated_options
                if continuation_token is not None:
                    # Resume a background streaming response by retrieving with stream=True
                    client = self.client
                    validated_options = await self._validate_options(options)
                    response_format = validated_options.get("response_format")
                    try:
                        raw_stream_response = await client.responses.with_raw_response.retrieve(
                            continuation_token["response_id"],
                            stream=True,
                        )
                        # Read headers defensively: telemetry instrumentors can wrap streaming
                        # responses without proxying .headers. Missing served-model metadata
                        # must not prevent the response from being consumed.
                        served_model = self._extract_served_model(getattr(raw_stream_response, "headers", None))
                        # ``stream=True`` guarantees the raw wrapper parses to
                        # ``AsyncStream``; the SDK's raw-response decorator loses
                        # the underlying overload in its generic return type.
                        parsed_stream = cast("AsyncStream[OpenAIResponseStreamEvent]", raw_stream_response.parse())
                        async with parsed_stream as stream_response:
                            async for chunk in stream_response:
                                update = self._parse_chunk_from_openai(
                                    chunk,
                                    options=validated_options,
                                    function_call_ids=function_call_ids,
                                    seen_reasoning_delta_item_ids=seen_reasoning_delta_item_ids,
                                    hosted_call_contents=hosted_call_contents,
                                    hosted_result_contents=hosted_result_contents,
                                    output_message_envelopes=output_message_envelopes,
                                    output_message_contents=output_message_contents,
                                )
                                if served_model is not None:
                                    update.model = served_model
                                yield update
                    except Exception as ex:
                        self._handle_request_error(ex)
                else:
                    (
                        client,
                        run_options,
                        validated_options,
                    ) = await self._prepare_request(messages, options)
                    response_format = validated_options.get("response_format")
                    try:
                        if "text_format" in run_options:
                            # The SDK's ``responses.stream(text_format=...)`` helper preserves
                            # client-side ``output_parsed`` partial parsing for structured outputs,
                            # but it does not expose the raw HTTP response (no ``x-ms-served-model``
                            # access). We accept that trade-off: this single streaming path keeps
                            # the deployment alias as the reported model name. All other paths
                            # surface the served-model header.
                            async with client.responses.stream(**run_options) as response:
                                async for chunk in response:
                                    yield self._parse_chunk_from_openai(
                                        chunk,
                                        options=validated_options,
                                        function_call_ids=function_call_ids,
                                        seen_reasoning_delta_item_ids=seen_reasoning_delta_item_ids,
                                        hosted_call_contents=hosted_call_contents,
                                        hosted_result_contents=hosted_result_contents,
                                        output_message_envelopes=output_message_envelopes,
                                        output_message_contents=output_message_contents,
                                    )
                        else:
                            raw_create_response = await client.responses.with_raw_response.create(
                                stream=True, **run_options
                            )
                            # Read headers defensively: telemetry wrappers can omit .headers.
                            # Missing served-model metadata must not fail the response.
                            served_model = self._extract_served_model(getattr(raw_create_response, "headers", None))
                            # Same raw-wrapper overload loss as the retrieve path.
                            parsed_stream = cast("AsyncStream[OpenAIResponseStreamEvent]", raw_create_response.parse())
                            async with parsed_stream as stream_response:
                                async for chunk in stream_response:
                                    update = self._parse_chunk_from_openai(
                                        chunk,
                                        options=validated_options,
                                        function_call_ids=function_call_ids,
                                        seen_reasoning_delta_item_ids=seen_reasoning_delta_item_ids,
                                        hosted_call_contents=hosted_call_contents,
                                        hosted_result_contents=hosted_result_contents,
                                        output_message_envelopes=output_message_envelopes,
                                        output_message_contents=output_message_contents,
                                    )
                                    if served_model is not None:
                                        update.model = served_model
                                    yield update
                    except Exception as ex:
                        self._handle_request_error(ex)

            return ResponseStream(_stream(), finalizer=_finalize_with_captured_format)

        # Non-streaming
        async def _get_response() -> ChatResponse:
            if continuation_token is not None:
                # Poll a background response by retrieving without stream
                client = self.client
                validated_options = await self._validate_options(options)
                try:
                    raw_response = await client.responses.with_raw_response.retrieve(continuation_token["response_id"])
                    # Retrieval without streaming always parses to a Response.
                    response = cast("OpenAIResponse", raw_response.parse())
                except Exception as ex:
                    self._handle_request_error(ex)
                chat_response = self._parse_response_from_openai(response, options=validated_options)
                # Read headers defensively: telemetry wrappers can omit .headers.
                # Missing served-model metadata must not fail the response.
                served_model = self._extract_served_model(getattr(raw_response, "headers", None))
                if served_model is not None:
                    chat_response.model = served_model
                # Once the background response completes, drop the continuation_token from
                # the caller's options dict. ToolLoopLayer reuses the same dict
                # across tool-loop iterations, so leaving it in place makes the next iteration
                # retrieve the same completed response instead of POSTing tool results.
                # Keep `background` so subsequent iterations still create
                # background responses.
                if chat_response.continuation_token is None and isinstance(options, dict):
                    options.pop("continuation_token", None)
                return chat_response
            client, run_options, validated_options = await self._prepare_request(messages, options)
            try:
                if "text_format" in run_options:
                    raw_response = await client.responses.with_raw_response.parse(stream=False, **run_options)
                else:
                    raw_response = await client.responses.with_raw_response.create(stream=False, **run_options)
                # ``stream=False`` guarantees a foreground Response (parsed
                # structured calls return the ParsedResponse subtype).
                response = cast("OpenAIResponse | ParsedResponse[BaseModel]", raw_response.parse())
            except Exception as ex:
                self._handle_request_error(ex)
            chat_response = self._parse_response_from_openai(response, options=validated_options)
            # Read headers defensively: telemetry wrappers can omit .headers.
            # Missing served-model metadata must not fail the response.
            served_model = self._extract_served_model(getattr(raw_response, "headers", None))
            if served_model is not None:
                chat_response.model = served_model
            return chat_response

        return _get_response()

    @classmethod
    def _extract_served_model(cls, headers: Any) -> str | None:
        """Return the Azure OpenAI ``x-ms-served-model`` response header value when present.

        Azure OpenAI Responses API returns the deployment alias in ``response.model`` but the actual
        snapshot served via the ``x-ms-served-model`` response header (e.g. ``gpt-5-nano-2025-08-07``
        vs deployment alias ``gpt-5-nano``). When present, the served snapshot is the source of truth
        for observability and downstream callers. Empty/whitespace-only header values are rejected
        here so every caller can simply check ``if served_model is not None``.
        """
        if headers is None:
            return None
        served_model = headers.get(cls.SERVED_MODEL_HEADER)
        if isinstance(served_model, str):
            stripped = served_model.strip()
            if stripped:
                return stripped
        return None

    def _prepare_response_and_text_format(
        self,
        *,
        response_format: Any,
        text_config: MutableMapping[str, Any] | None,
        model: Any = None,
    ) -> tuple[type[BaseModel] | None, dict[str, Any] | None]:
        """Normalize response_format into Responses text configuration and parse target."""
        if text_config is not None and not isinstance(text_config, MutableMapping):
            raise ChatClientInvalidRequestException("text must be a mapping when provided.")
        text_config = cast(dict[str, Any], text_config) if isinstance(text_config, MutableMapping) else None

        if response_format is None:
            return None, text_config

        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            if text_config and "format" in text_config:
                raise ChatClientInvalidRequestException("response_format cannot be combined with explicit text.format.")
            return response_format, text_config

        if isinstance(response_format, Mapping):
            format_config = self._convert_response_format(cast("Mapping[str, Any]", response_format), model=model)
            if text_config is None:
                text_config = {"format": format_config}
            elif "format" in text_config and text_config["format"] != format_config:
                raise ChatClientInvalidRequestException("Conflicting response_format definitions detected.")
            else:
                # Copy before writing: the caller's mapping is a reused profile
                # option object; writing the converted format into it would
                # leak this turn's format into every later request.
                text_config = {**text_config, "format": format_config}
            return None, text_config

        raise ChatClientInvalidRequestException("response_format must be a Pydantic model or mapping.")

    def _convert_response_format(self, response_format: Mapping[str, Any], *, model: Any = None) -> dict[str, Any]:
        """Convert Chat style response_format into Responses text format config."""
        if "format" in response_format and isinstance(response_format["format"], Mapping):
            return dict(cast("Mapping[str, Any]", response_format["format"]))

        format_type = response_format.get("type")
        if format_type == "json_schema":
            schema_section = response_format.get("json_schema", response_format)
            if not isinstance(schema_section, Mapping):
                raise ChatClientInvalidRequestException("json_schema response_format must be a mapping.")
            schema_section_typed = cast("Mapping[str, Any]", schema_section)
            schema: Any = schema_section_typed.get("schema")
            if schema is None:
                raise ChatClientInvalidRequestException("json_schema response_format requires a schema.")
            name: str = str(
                schema_section_typed.get("name")
                or schema_section_typed.get("title")
                or (cast("Mapping[str, Any]", schema).get("title") if isinstance(schema, Mapping) else None)
                or "response"
            )
            format_config: dict[str, Any] = {
                "type": "json_schema",
                "name": name,
                "schema": schema,
            }
            if "strict" in schema_section:
                format_config["strict"] = schema_section["strict"]
            if "description" in schema_section and schema_section["description"] is not None:
                format_config["description"] = schema_section["description"]
            return format_config

        # Tuple membership compares by equality, so an array-valued ``type``
        # (unhashable) classifies as a raw schema instead of raising.
        if format_type in ("json_object", "text"):
            return {"type": format_type}

        # Anything else is a raw JSON Schema (e.g. {"type": "object", ...})
        # to wrap in the expected json_schema envelope: JSON Schema admits
        # array-valued and absent ``type`` (e.g. enum-only schemas), so
        # classifying by primitive type names or keyword sniffing
        # under-matches.
        # Materialized copy: the Mapping contract admits read-only views
        # (e.g. ``MappingProxyType``) at any nesting depth, and the
        # strictifier mutates nested mappings in place.
        schema = _materialize_json_structure(response_format)
        # Pop title from schema since OpenAI strict mode rejects unknown keys;
        # use it as the schema name in the envelope instead.
        name = str(schema.pop("title", None) or "response")
        schema, strict = _strictify_response_schema(schema, model=model)
        format_config = {
            "type": "json_schema",
            "name": name,
            "schema": schema,
        }
        if strict:
            format_config["strict"] = True
        return format_config

    def _get_conversation_id(
        self, response: OpenAIResponse | ParsedResponse[BaseModel], store: bool | None
    ) -> str | None:
        """Get the conversation ID from the response if store is True."""
        if store is False:
            return None
        # If conversation ID exists, it means that we operate with conversation
        # so we use conversation ID as input and output.
        if response.conversation and response.conversation.id:
            return response.conversation.id
        # If conversation ID doesn't exist, we operate with responses
        # so we use response ID as input and output.
        return response.id

    # region Prep methods

    def _prepare_tools_for_openai(
        self,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None,
    ) -> list[Any]:
        """Prepare tools for the OpenAI Responses API.

        Converts FunctionTool to Responses API format. Shell-enabled FunctionTools
        with explicit shell environment metadata are mapped to OpenAI shell tools.
        All other tools pass through unchanged.

        Args:
            tools: A single tool or sequence of tools to prepare.

        Returns:
            List of tool parameters ready for the OpenAI API.
        """
        tools_list = normalize_tools(tools)
        if not tools_list:
            return []
        response_tools: list[Any] = []
        for tool_item in tools_list:
            if isinstance(tool_item, FunctionTool) and tool_item.kind == SHELL_TOOL_KIND_VALUE:
                shell_env = (tool_item.additional_properties or {}).get(OPENAI_SHELL_ENVIRONMENT_KEY)
                response_tools.append(
                    FunctionShellTool(
                        type="shell",
                        environment=shell_env,  # type: ignore[typeddict-item]
                    )
                )
                continue
            if isinstance(tool_item, FunctionTool):
                params = tool_item.parameters()
                params["additionalProperties"] = False
                response_tools.append(
                    FunctionToolParam(
                        name=tool_item.name,
                        parameters=params,
                        strict=False,
                        type="function",
                        description=tool_item.description,
                    )
                )
            else:
                # Pass through all other tools (dicts, SDK types) unchanged
                response_tools.append(tool_item)
        return response_tools

    def _get_local_shell_tool_name(
        self,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None,
    ) -> str | None:
        """Return the name of the configured local shell tool function, if any."""
        for tool_item in normalize_tools(tools):
            if not isinstance(tool_item, FunctionTool):
                continue
            if tool_item.kind != SHELL_TOOL_KIND_VALUE:
                continue
            shell_env = (tool_item.additional_properties or {}).get(OPENAI_SHELL_ENVIRONMENT_KEY)
            if isinstance(shell_env, Mapping) and shell_env.get("type") == "local":  # type: ignore[typeddict-item]
                return tool_item.name
        return None

    # region Hosted Tool Factory Methods

    @staticmethod
    def get_code_interpreter_tool(
        *,
        file_ids: list[str] | None = None,
        container: Literal["auto"] | CodeInterpreterContainerCodeInterpreterToolAuto = "auto",
    ) -> Any:
        """Create a code interpreter tool configuration for the Responses API.

        Keyword Args:
            file_ids: List of file IDs to make available to the code interpreter.
            container: Container configuration. Use "auto" for automatic container management,
                or provide a TypedDict with custom container settings.

        Returns:
            A CodeInterpreter tool parameter ready to pass to ChatAgent.

        Examples:
            .. code-block:: python

                from chrys.service.llm.openai_responses import RawOpenAIChatClient

                # Basic code interpreter
                tool = RawOpenAIChatClient.get_code_interpreter_tool()

                # With file access
                tool = RawOpenAIChatClient.get_code_interpreter_tool(file_ids=["file-abc123"])

                # Use with agent
                agent = ChatAgent(client, tools=[tool])
        """
        container_config: CodeInterpreterContainerCodeInterpreterToolAuto = (
            container if isinstance(container, dict) else {"type": "auto"}
        )

        if file_ids:
            container_config["file_ids"] = file_ids

        return CodeInterpreter(type="code_interpreter", container=container_config)

    @staticmethod
    def get_web_search_tool(
        *,
        user_location: dict[str, str] | None = None,
        search_context_size: Literal["low", "medium", "high"] | None = None,
        filters: Filters | None = None,
    ) -> Any:
        """Create a web search tool configuration for the Responses API.

        Keyword Args:
            user_location: Location context for search results. Dict with keys like
                "city", "country", "region", "timezone".
            search_context_size: Amount of context to include from search results.
                One of "low", "medium", or "high".
            filters: Additional search filters.

        Returns:
            A WebSearchToolParam dict ready to pass to ChatAgent.

        Examples:
            .. code-block:: python

                from chrys.service.llm.openai_responses import RawOpenAIChatClient

                # Basic web search
                tool = RawOpenAIChatClient.get_web_search_tool()

                # With location context
                tool = RawOpenAIChatClient.get_web_search_tool(
                    user_location={"city": "Seattle", "country": "US"},
                    search_context_size="medium",
                )

                agent = ChatAgent(client, tools=[tool])
        """
        web_search_tool = WebSearchToolParam(type="web_search")

        if user_location:
            web_search_tool["user_location"] = {
                "type": "approximate",
                "city": user_location.get("city"),
                "country": user_location.get("country"),
                "region": user_location.get("region"),
                "timezone": user_location.get("timezone"),
            }

        if search_context_size:
            web_search_tool["search_context_size"] = search_context_size

        if filters:
            web_search_tool["filters"] = filters

        return web_search_tool

    @staticmethod
    def get_image_generation_tool(
        *,
        size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] | None = None,
        output_format: Literal["png", "jpeg", "webp"] | None = None,
        model: Literal["gpt-image-1", "gpt-image-1-mini"] | str | None = None,
        quality: Literal["low", "medium", "high", "auto"] | None = None,
        partial_images: int | None = None,
        background: Literal["transparent", "opaque", "auto"] | None = None,
        moderation: Literal["auto", "low"] | None = None,
        output_compression: int | None = None,
    ) -> Any:
        """Create an image generation tool configuration for the Responses API.

        Keyword Args:
            size: Image dimensions. One of "1024x1024", "1024x1536", "1536x1024", or "auto".
            output_format: Output image format. One of "png", "jpeg", or "webp".
            model: Model to use for image generation. One of "gpt-image-1" or "gpt-image-1-mini".
            quality: Image quality level. One of "low", "medium", "high", or "auto".
            partial_images: Number of partial images to stream during generation.
            background: Background type. One of "transparent", "opaque", or "auto".
            moderation: Moderation level. One of "auto" or "low".
            output_compression: Compression level for output (0-100).

        Returns:
            An ImageGeneration tool parameter dict ready to pass to ChatAgent.

        Examples:
            .. code-block:: python

                from chrys.service.llm.openai_responses import RawOpenAIChatClient

                # Basic image generation
                tool = RawOpenAIChatClient.get_image_generation_tool()

                # High quality large image
                tool = RawOpenAIChatClient.get_image_generation_tool(
                    size="1536x1024",
                    quality="high",
                    output_format="png",
                )

                agent = ChatAgent(client, tools=[tool])
        """
        tool: ImageGeneration = {"type": "image_generation"}

        if size:
            tool["size"] = size
        if output_format:
            tool["output_format"] = output_format
        if model:
            tool["model"] = model
        if quality:
            tool["quality"] = quality
        if partial_images is not None:
            tool["partial_images"] = partial_images
        if background:
            tool["background"] = background
        if moderation:
            tool["moderation"] = moderation
        if output_compression is not None:
            tool["output_compression"] = output_compression

        return tool

    @staticmethod
    def get_shell_tool(
        *,
        func: Callable[..., Any] | FunctionTool | None = None,
        environment: Literal["auto"] | dict[str, Any] | None = "auto",
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Create a shell tool for the Responses API.

        - When ``func`` is ``None`` (default), returns an OpenAI hosted shell
          tool declaration.
        - When ``func`` is provided, returns a local FunctionTool that is
          declared to OpenAI as a local shell tool and executed via the function
          invocation layer.

        Keyword Args:
            func: Optional local shell function or ``FunctionTool``.
            environment: Container environment configuration.
                Used only when ``func`` is ``None``.
                Use ``"auto"`` (default) for managed containers, or provide a
                dict with explicit hosted container settings.
            name: Optional local tool name when ``func`` is provided.
            description: Optional local tool description when ``func`` is provided.

        Returns:
            A hosted shell declaration or a local shell FunctionTool.

        Examples:
            .. code-block:: python

                from chrys.service.llm.openai_responses import RawOpenAIChatClient

                # Hosted shell (OpenAI container)
                tool = RawOpenAIChatClient.get_shell_tool()

                # Hosted shell with custom environment
                tool = RawOpenAIChatClient.get_shell_tool(environment={"type": "container_auto", "file_ids": ["file-abc"]})

                # Local shell execution
                tool = RawOpenAIChatClient.get_shell_tool(
                    func=my_shell_func,
                )
        """
        if func is None:
            env_config: dict[str, Any] = (
                dict(environment) if isinstance(environment, dict) else {"type": "container_auto"}
            )
            if env_config.get("type") == "local":
                raise ValueError("Local shell requires func. Provide func for local execution.")
            return FunctionShellTool(type="shell", environment=env_config)  # type: ignore[typeddict-item]

        if isinstance(environment, dict):
            raise ValueError("When func is provided, environment config is not supported.")
        local_env = {"type": "local"}

        base_tool: FunctionTool
        if isinstance(func, FunctionTool):
            base_tool = func
            if name is not None:
                base_tool.name = name
            if description is not None:
                base_tool.description = description
        else:
            base_tool = tool(
                func=func,
                name=name,
                description=description,
            )

        if base_tool.func is None:
            raise ValueError("Shell tool requires an executable function.")

        additional_properties = dict(base_tool.additional_properties or {})
        additional_properties[OPENAI_SHELL_ENVIRONMENT_KEY] = local_env
        base_tool.additional_properties = additional_properties
        base_tool.kind = SHELL_TOOL_KIND_VALUE
        return base_tool

    @staticmethod
    def get_mcp_tool(
        *,
        name: str,
        url: str,
        description: str | None = None,
        allowed_tools: list[str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Create a hosted MCP (Model Context Protocol) tool configuration for the Responses API.

        This configures an MCP server that will be called by OpenAI's service.
        The tools from this MCP server are executed remotely by OpenAI,
        not locally by your application.

        Note:
            For local MCP execution where your application calls the MCP server
            directly, use the MCP client tools instead of this method.

        Note:
            The tool is configured with ``require_approval: "never"``. Chrys
            does not implement the Responses approval round-trip
            (``mcp_approval_request`` → ``mcp_approval_response``); with the
            platform default of ``"always"`` every hosted tool call would
            stall waiting for an approval chrys never sends. Restrict what
            the server may do via ``allowed_tools`` instead.

        Keyword Args:
            name: A label/name for the MCP server.
            url: The URL of the MCP server.
            description: A description of what the MCP server provides.
            allowed_tools: List of tool names that are allowed to be used from this MCP server.
            headers: HTTP headers to include in requests to the MCP server.

        Returns:
            An Mcp tool parameter dict ready to pass to ChatAgent.

        Examples:
            .. code-block:: python

                from chrys.service.llm.openai_responses import RawOpenAIChatClient

                # Basic MCP tool
                tool = RawOpenAIChatClient.get_mcp_tool(
                    name="my_mcp",
                    url="https://mcp.example.com",
                )

                # With headers
                tool = RawOpenAIChatClient.get_mcp_tool(
                    name="github_mcp",
                    url="https://mcp.github.com",
                    description="GitHub MCP server",
                    headers={"Authorization": "Bearer token"},
                )

                agent = ChatAgent(client, tools=[tool])
        """
        mcp: Mcp = {
            "type": "mcp",
            "server_label": name.replace(" ", "_"),
            "server_url": url,
            # The platform default is "always", but the tool loop cannot replay
            # ``mcp_approval_request`` output: it drops that unparsed item and
            # would leave the hosted call unanswered. Hosted-MCP trust is
            # decided at configuration time (choosing the server +
            # ``allowed_tools``), not per call.
            "require_approval": "never",
        }

        if description:
            mcp["server_description"] = description

        if headers:
            mcp["headers"] = headers

        # ``is not None``: an explicit empty allow-list means "expose no
        # tools" (matching the local MCP semantics in ``service/mcp/owned``),
        # which matters more than ever with approval pinned to "never".
        if allowed_tools is not None:
            mcp["allowed_tools"] = allowed_tools

        return mcp

    @staticmethod
    def get_file_search_tool(
        *,
        vector_store_ids: list[str],
        max_num_results: int | None = None,
    ) -> Any:
        """Create a file search tool configuration for the Responses API.

        Keyword Args:
            vector_store_ids: List of vector store IDs to search within.
            max_num_results: Maximum number of results to return. Defaults to 50 if not specified.

        Returns:
            A FileSearchToolParam dict ready to pass to ChatAgent.

        Examples:
            .. code-block:: python

                from chrys.service.llm.openai_responses import RawOpenAIChatClient

                # Basic file search
                tool = RawOpenAIChatClient.get_file_search_tool(
                    vector_store_ids=["vs_abc123"],
                )

                # With result limit
                tool = RawOpenAIChatClient.get_file_search_tool(
                    vector_store_ids=["vs_abc123", "vs_def456"],
                    max_num_results=10,
                )

                agent = ChatAgent(client, tools=[tool])
        """
        tool = FileSearchToolParam(
            type="file_search",
            vector_store_ids=vector_store_ids,
        )

        if max_num_results is not None:
            tool["max_num_results"] = max_num_results

        return tool

    # endregion

    async def _prepare_options(
        self,
        messages: Sequence[Message],
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Take options dict and create the specific options for Responses API."""
        # An explicit empty ``include`` is the caller's opt-out from the
        # encrypted-reasoning include (some Responses gateways reject the
        # value); a non-empty caller list still gains it, so a profile asking
        # for logprobs or hosted-tool outputs keeps multi-turn reasoning replay.
        caller_include = options.get("include")
        caller_opted_out_of_encrypted_reasoning = isinstance(caller_include, list) and not caller_include
        # Exclude keys that are not supported or handled separately
        exclude_keys = {
            "type",
            "presence_penalty",  # not supported
            "frequency_penalty",  # not supported
            "logit_bias",  # not supported
            "seed",  # not supported
            "stop",  # not supported
            "response_format",  # handled separately
            "conversation_id",  # handled separately
            "tool_choice",  # handled separately
            "continuation_token",  # handled separately in _inner_get_response
        }
        # ``instructions`` is NOT excluded: the Responses API accepts it as a
        # first-class request param on every create, which keeps instruction
        # changes effective on continuation turns too (a prepended system
        # message would be skipped there and changes silently dropped).
        run_options: dict[str, Any] = {k: v for k, v in options.items() if k not in exclude_keys and v is not None}

        # messages
        request_uses_service_side_storage = False
        for key in ("conversation_id", "previous_response_id", "conversation"):
            value = options.get(key)
            mapping_id = value.get("id") if isinstance(value, Mapping) else None
            if (isinstance(value, str) and value) or (isinstance(mapping_id, str) and mapping_id):
                request_uses_service_side_storage = True
                break
        if (
            not request_uses_service_side_storage
            and type(self).INJECT_ENCRYPTED_REASONING_INCLUDE
            and not caller_opted_out_of_encrypted_reasoning
        ):
            include = list(run_options.get("include", []))
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            run_options["include"] = include
        if not run_options.get("include"):
            run_options.pop("include", None)
        request_input = self._prepare_messages_for_openai(
            messages,
            request_uses_service_side_storage=request_uses_service_side_storage,
        )
        if not request_input:
            raise ChatClientInvalidRequestException("Messages are required for chat completions")
        conversation_id = options.get("conversation_id")
        run_options["input"] = request_input

        # model id
        self._check_model_presence(run_options)

        # translations between options and Responses API
        translations = {
            "allow_multiple_tool_calls": "parallel_tool_calls",
            "conversation_id": "previous_response_id",
            "max_tokens": "max_output_tokens",
        }
        for old_key, new_key in translations.items():
            if old_key in run_options and old_key != new_key:
                run_options[new_key] = run_options.pop(old_key)

        # Handle different conversation ID formats
        if conversation_id := options.get("conversation_id"):
            if conversation_id.startswith("resp_"):
                # For response IDs, set previous_response_id and remove conversation property
                run_options["previous_response_id"] = conversation_id
            elif conversation_id.startswith("conv_"):
                # For conversation IDs, set conversation and remove previous_response_id property
                run_options["conversation"] = conversation_id
            else:
                # If the format is unrecognized, default to previous_response_id
                run_options["previous_response_id"] = conversation_id

        # tools
        if tools := self._prepare_tools_for_openai(options.get("tools")):
            run_options["tools"] = tools
            # tool_choice: convert ToolMode to appropriate format
            if tool_choice := options.get("tool_choice"):
                tool_mode = validate_tool_mode(tool_choice)
                if tool_mode is not None:
                    if (mode := tool_mode.get("mode")) == "required" and (
                        func_name := tool_mode.get("required_function_name")
                    ) is not None:
                        run_options["tool_choice"] = {
                            "type": "function",
                            "name": func_name,
                        }
                    elif mode in ("auto", "required") and (allowed := tool_mode.get("allowed_tools")) is not None:
                        run_options["tool_choice"] = {
                            "type": "allowed_tools",
                            "mode": mode,
                            "tools": [{"type": "function", "name": name} for name in allowed],
                        }
                    else:
                        run_options["tool_choice"] = mode
        else:
            run_options.pop("parallel_tool_calls", None)
            run_options.pop("tool_choice", None)

        # response format and text config
        response_format = options.get("response_format")
        text_config = run_options.pop("text", None)
        response_format, text_config = self._prepare_response_and_text_format(
            response_format=response_format,
            text_config=text_config,
            model=run_options.get("model"),
        )
        # The Responses API nests verbosity under ``text.verbosity``; surface it as a
        # top-level option for parity with ``reasoning`` and translate here.
        if (verbosity := run_options.pop("verbosity", None)) is not None:
            text_config = dict(text_config) if text_config else {}
            text_config["verbosity"] = verbosity
        if text_config:
            run_options["text"] = _sanitize_text_format_config(text_config)
        if response_format:
            run_options["text_format"] = _sanitize_text_format_model(response_format)

        return run_options

    def _check_model_presence(self, options: dict[str, Any]) -> None:
        """Check if the 'model' param is present, and if not raise a Error.

        Subclasses can override this when they populate the model through a different option field.
        """
        if not options.get("model"):
            if not self.model:
                raise ValueError("model must be a non-empty string")
            options["model"] = self.model

    def _prepare_messages_for_openai(
        self,
        chat_messages: Sequence[Message],
        *,
        request_uses_service_side_storage: bool = True,
    ) -> list[dict[str, Any]]:
        """Prepare the chat messages for a request.

        Allowing customization of the key names for role/author, and optionally overriding the role.

        "tool" messages need to be formatted different than system/user/assistant messages:
            They require a "tool_call_id" and (function) "name" key, and the "metadata" key should
            be removed. The "encoding" key should also be removed.

        Override this method to customize the formatting of the chat history for a request.

        Args:
            chat_messages: The chat history to prepare.
            request_uses_service_side_storage: Whether this request continues a service-managed
                response/conversation and can safely reference service-scoped response items.

        Returns:
            The prepared chat messages for a request.
        """
        groups = _partition_replay_message_groups(chat_messages)
        hosted_degradations = cross_provider_hosted_degradations(
            chat_messages,
            target_provider=type(self).HOSTED_PROVIDER,
            unsafe_same_provider_families=((HostedToolFamily.IMAGE,) if not request_uses_service_side_storage else ()),
        )
        if request_uses_service_side_storage:
            plans = [_ReasoningReplayPlan(group=group) for group in groups]
        else:
            plans = self._classify_reasoning_replay_groups(groups)

        prepared: list[dict[str, Any]] = []
        active_exchange_ordinal: int | None = None
        active_exchange_items: list[dict[str, Any]] = []

        def flush_active_exchange() -> None:
            nonlocal active_exchange_ordinal, active_exchange_items
            if active_exchange_items:
                coalesced = self._coalesce_pending_hosted_results(active_exchange_items)
                prepared.extend(self._complete_unanswered_apply_patch_calls(coalesced))
            active_exchange_ordinal = None
            active_exchange_items = []

        for plan in plans:
            group_items: list[dict[str, Any]] = []
            for message in plan.group.messages:
                group_items.extend(
                    self._prepare_message_for_openai(
                        message,
                        request_uses_service_side_storage=request_uses_service_side_storage,
                        reasoning_items=plan.reasoning_items,
                        drop_content_ids=plan.drop_content_ids,
                        omit_function_call_id_content_ids=plan.omit_function_call_id_content_ids,
                        hosted_degradations=hosted_degradations,
                    )
                )
            exchange_ordinal = plan.group.exchange_ordinal
            if exchange_ordinal is None:
                flush_active_exchange()
                coalesced = self._coalesce_pending_hosted_results(group_items)
                prepared.extend(self._complete_unanswered_apply_patch_calls(coalesced))
                continue
            if active_exchange_ordinal != exchange_ordinal:
                flush_active_exchange()
                active_exchange_ordinal = exchange_ordinal
            active_exchange_items.extend(group_items)
        flush_active_exchange()
        return prepared

    @staticmethod
    def _complete_unanswered_apply_patch_calls(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Answer replayed apply-patch calls within one owning exchange.

        Call IDs may repeat in later exchanges; an answer here must never
        suppress completion elsewhere in the conversation.

        The Responses API rejects a history that contains an
        ``apply_patch_call`` without a matching output item, and chrys has no
        client-side executor for the tool, so replay completes each unanswered
        call with an explicit failure instead of a guaranteed request error.

        If chrys ever executes apply_patch for real: parse the call's
        ``operation`` (create_file/update_file/delete_file with a V4A-format
        diff, not unified diff) and route it through the local file-mutation
        pipeline (approval, snapshot/provenance), then serialize the execution
        result as an ``apply_patch_call_output`` — not a plain
        ``function_call_output`` — with ``status`` "completed"/"failed". Calls
        answered that way are already in ``answered`` here, so this post-pass
        degrades to a no-op for them and stays correct for histories recorded
        before the executor existed.
        """
        answered = {
            item.get("call_id")
            for item in items
            if item.get("type") == "apply_patch_call_output" and item.get("call_id")
        }
        if not any(item.get("type") == "apply_patch_call" and item.get("call_id") not in answered for item in items):
            return items
        completed: list[dict[str, Any]] = []
        for item in items:
            completed.append(item)
            if item.get("type") != "apply_patch_call":
                continue
            call_id = item.get("call_id")
            if not call_id or call_id in answered:
                continue
            answered.add(call_id)
            completed.append(
                {
                    "type": "apply_patch_call_output",
                    "call_id": call_id,
                    "status": "failed",
                    "output": "The client did not execute this apply_patch call.",
                }
            )
        return completed

    @staticmethod
    def _logical_reasoning_occurrences(group: _ReplayMessageGroup) -> list[list[Content]]:
        """Collect contiguous provider reasoning-item occurrences within a group."""

        occurrences: list[list[Content]] = []
        current: list[Content] = []
        for message in group.messages:
            for content in message.contents:
                if content.type != "text_reasoning":
                    if current:
                        occurrences.append(current)
                        current = []
                    continue
                if current and current[-1].id != content.id:
                    occurrences.append(current)
                    current = []
                if any(existing is content for existing in current):
                    continue
                current.append(content)
        if current:
            occurrences.append(current)
        return occurrences

    def _prepare_reasoning_items_for_openai(
        self,
        occurrences: Sequence[Sequence[Content]],
    ) -> dict[int, dict[str, Any]]:
        """Reconstruct one provider reasoning item per logical occurrence."""

        reasoning_items: dict[int, dict[str, Any]] = {}
        for contents in occurrences:
            if not contents:
                continue
            reasoning_id = contents[0].id
            # Later payloads refine earlier ones for a single reasoning id:
            # the added-event snapshot is stamped on every sibling content but
            # the done-event terminal payload lands only on the last, so the
            # LAST non-empty payload is the authoritative one to replay.
            encrypted_content = next(
                (
                    content.protected_data or content.additional_properties.get("encrypted_content")
                    for content in reversed(contents)
                    if content.protected_data or content.additional_properties.get("encrypted_content")
                ),
                None,
            )
            if not reasoning_id or not encrypted_content:
                continue

            item: dict[str, Any] = {
                "type": "reasoning",
                "id": reasoning_id,
                "summary": [],
                "encrypted_content": encrypted_content,
            }
            reasoning_texts: list[dict[str, str]] = []
            for content in contents:
                properties = content.additional_properties
                if status := properties.get("status"):
                    item["status"] = status
                reasoning_text_marker = properties.get("reasoning_text")
                if reasoning_text_marker:
                    reasoning_text = content.text if reasoning_text_marker is True else reasoning_text_marker
                    if isinstance(reasoning_text, str) and reasoning_text:
                        reasoning_texts.append({"type": "reasoning_text", "text": reasoning_text})
                elif content.text:
                    item["summary"].append({"type": "summary_text", "text": content.text})
            if reasoning_texts:
                item["content"] = reasoning_texts
            reasoning_items[id(contents[0])] = item
        return reasoning_items

    def _reasoning_occurrences_are_valid(
        self,
        occurrences: Sequence[Sequence[Content]],
        reasoning_items: Mapping[int, dict[str, Any]],
    ) -> bool:
        """Return whether every occurrence has a safe replay representation."""
        return len(reasoning_items) == len(occurrences)

    @staticmethod
    def _result_owner_group_positions(groups: Sequence[_ReplayMessageGroup]) -> dict[int, int]:
        """Map paired result Content identities to their call's replay group.

        The map is frame-local and occurrence-derived: ``pair_results`` owns
        duplicate-id cursor semantics, while Content identity lets the later
        group-local serializer preserve a result physically carried by a
        different sibling group. Ambiguous repeated object identities fail
        closed by receiving no owner.
        """
        messages: list[Message] = []
        group_positions: list[int] = []
        for group in groups:
            messages.extend(group.messages)
            group_positions.extend([group.position] * len(group.messages))

        owners: dict[int, int] = {}
        ambiguous_content_ids: set[int] = set()
        for exchange in iter_exchanges(messages, _LIVE_ACCESSOR):
            pairing = pair_results(messages, exchange, _LIVE_ACCESSOR, _REPLAY_PAIRING_POLICY)
            assignment_runs = (*pairing.truthy_assignments.values(), *pairing.falsy_assignments.values())
            for assignments in assignment_runs:
                for call, result in assignments:
                    if result is None:
                        continue
                    content = messages[result.message_index].contents[result.content_index]
                    content_identity = id(content)
                    if content_identity in ambiguous_content_ids:
                        continue
                    owner = group_positions[call.message_index]
                    existing_owner = owners.get(content_identity)
                    if existing_owner is not None and existing_owner != owner:
                        owners.pop(content_identity, None)
                        ambiguous_content_ids.add(content_identity)
                    else:
                        owners[content_identity] = owner
        return owners

    def _classify_reasoning_replay_groups(
        self,
        groups: Sequence[_ReplayMessageGroup],
    ) -> list[_ReasoningReplayPlan]:
        """Build a single positional replay/degradation plan for the request."""

        result_owner_group_positions = self._result_owner_group_positions(groups)
        plans: list[_ReasoningReplayPlan] = []
        for group in groups:
            occurrences = self._logical_reasoning_occurrences(group)
            reasoning_items = self._prepare_reasoning_items_for_openai(occurrences)
            plan = _ReasoningReplayPlan(group=group, reasoning_items=reasoning_items)
            plan.reasoning_ids = [occurrence[0].id or "<missing>" for occurrence in occurrences if occurrence]

            followers: list[tuple[Message, Content]] = [
                (message, content)
                for message in group.messages
                for content in message.contents
                if content.type in _TOOL_CONTENT_TYPES
            ]
            for _message, content in followers:
                identifier = content.call_id or content.image_id
                if isinstance(identifier, str) and identifier:
                    plan.call_ids.append(identifier)

            has_reasoning = bool(occurrences) or group.annotated_reasoning_tool_call
            invalid_reasoning = bool(occurrences) and not self._reasoning_occurrences_are_valid(
                occurrences,
                reasoning_items,
            )
            if group.annotated_reasoning_tool_call and followers and not occurrences:
                invalid_reasoning = True
                plan.reasoning_ids.append("<missing>")
            if group.annotated_reasoning_tool_call and occurrences and not followers:
                invalid_reasoning = True

            unsafe_content_ids: set[int] = set()
            if has_reasoning:
                for message, content in followers:
                    properties = content.additional_properties
                    marker_bearing_shell = content.type in {"function_call", "function_result"} and properties.get(
                        OPENAI_SHELL_OUTPUT_TYPE_KEY
                    ) in {
                        OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL,
                        OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL,
                    }
                    informational_function = content.type == "function_call" and content.informational_only
                    prepared = self._prepare_content_for_openai(
                        message.role,
                        content,
                        replays_local_storage="_attribution" in message.additional_properties,
                    )
                    if marker_bearing_shell or informational_function or not prepared:
                        unsafe_content_ids.add(id(content))
                    elif content.type == "function_call":
                        function_call_wire_id = prepared.get("id")
                        if isinstance(function_call_wire_id, str) and function_call_wire_id:
                            plan.wire_ids.append(function_call_wire_id)

                function_call_ids = {
                    content.call_id
                    for _message, content in followers
                    if content.type == "function_call" and id(content) not in unsafe_content_ids and content.call_id
                }
                mcp_call_ids = {
                    content.call_id
                    for _message, content in followers
                    if content.type == "mcp_server_tool_call"
                    and id(content) not in unsafe_content_ids
                    and content.call_id
                }
                for _message, content in followers:
                    if content.type not in {"function_result", "mcp_server_tool_result"}:
                        continue
                    result_owner = result_owner_group_positions.get(id(content))
                    if result_owner is not None and result_owner != group.position:
                        # A shared output message is physically carried by
                        # this group but the exchange cursor assigned this
                        # occurrence to another sibling's call.
                        continue
                    local_call_ids = function_call_ids if content.type == "function_result" else mcp_call_ids
                    if result_owner is None or not content.call_id or content.call_id not in local_call_ids:
                        unsafe_content_ids.add(id(content))
            else:
                for message, content in followers:
                    if content.type != "function_call" or not content.call_id:
                        continue
                    prepared = self._prepare_content_for_openai(
                        message.role,
                        content,
                        replays_local_storage="_attribution" in message.additional_properties,
                    )
                    function_call_wire_id = prepared.get("id") if prepared else None
                    if isinstance(function_call_wire_id, str) and function_call_wire_id:
                        plan.wire_ids.append(function_call_wire_id)

            plan.wire_ids.extend(
                item["id"] for item in reasoning_items.values() if isinstance(item.get("id"), str) and item["id"]
            )
            plan.mcp_wire_ids.extend(
                content.call_id
                for _message, content in followers
                if content.type == "mcp_server_tool_call" and content.call_id
            )
            plan.mcp_result_ids.extend(
                content.call_id
                for _message, content in followers
                if content.type == "mcp_server_tool_result" and content.call_id
            )
            plan.wire_ids.extend(plan.mcp_wire_ids)

            if invalid_reasoning or unsafe_content_ids:
                plan.drop_content_ids.update(unsafe_content_ids)
                self._degrade_reasoning_replay_plan(plan)
            plans.append(plan)

        seen_wire_ids: set[str] = set()
        function_only_wire_ids: set[str] = set()
        degradable_plans: list[_ReasoningReplayPlan] = []
        for plan in plans:
            if plan.degraded:
                continue
            emits_mcp_items = any(
                content.type == "mcp_server_tool_call" and content.call_id
                for message in plan.group.messages
                for content in message.contents
            )
            # Function-only groups never yield: their duplicate-id replay
            # predates this classifier and stays untouched, so their wire ids
            # are pre-reserved against every reasoning/MCP-emitting group
            # regardless of position.
            if plan.reasoning_items or emits_mcp_items:
                degradable_plans.append(plan)
            else:
                function_only_wire_ids.update(plan.wire_ids)

        for plan in degradable_plans:
            group_wire_ids = plan.wire_ids
            group_wire_id_counts = Counter(group_wire_ids)
            mcp_wire_id_counts = Counter(plan.mcp_wire_ids)
            mcp_result_id_counts = Counter(plan.mcp_result_ids)
            collides_with_group = any(
                count > 1 and not (mcp_wire_id_counts[wire_id] == count and mcp_result_id_counts[wire_id] == count)
                for wire_id, count in group_wire_id_counts.items()
            )
            collides_with_request = any(
                wire_id in seen_wire_ids or wire_id in function_only_wire_ids for wire_id in group_wire_ids
            )
            # Fully answered repeated MCP ids inside one exchange are distinct
            # occurrences and coalesce FIFO. Call-only/incomplete repeats,
            # other duplicate top-level ids, and ids reused by another
            # positional group keep the fail-closed path.
            if collides_with_group or collides_with_request:
                self._degrade_reasoning_replay_plan(plan)
                continue
            seen_wire_ids.update(group_wire_ids)

        self._drop_mcp_results_owned_by_degraded_plans(
            plans,
            result_owner_group_positions,
        )
        degraded_plans = [plan for plan in plans if plan.degraded]
        if degraded_plans:
            logger.warning(
                "Degraded stateless reasoning replay: groups=%s reasoning_ids=%s call_ids=%s",
                [plan.group.position for plan in degraded_plans],
                self._ordered_unique(identifier for plan in degraded_plans for identifier in plan.reasoning_ids),
                self._ordered_unique(identifier for plan in degraded_plans for identifier in plan.call_ids),
            )
        return plans

    @staticmethod
    def _degrade_reasoning_replay_plan(plan: _ReasoningReplayPlan) -> None:
        """Apply the uniform safe serialization form to one group."""

        plan.degraded = True
        plan.reasoning_items.clear()
        for message in plan.group.messages:
            for content in message.contents:
                content_identity = id(content)
                if content.type in {"text_reasoning", "mcp_server_tool_call"}:
                    plan.drop_content_ids.add(content_identity)
                elif content.type == "function_call" and content_identity not in plan.drop_content_ids:
                    plan.omit_function_call_id_content_ids.add(content_identity)

    @staticmethod
    def _drop_mcp_results_owned_by_degraded_plans(
        plans: Sequence[_ReasoningReplayPlan],
        result_owner_group_positions: Mapping[int, int],
    ) -> None:
        """Drop MCP result occurrences whose paired owning calls cannot replay.

        Owner-less orphan markers stay until coalescing: although coalescing
        drops them, their serialized position still separates adjacent
        message runs.
        """
        degraded_positions = {plan.group.position for plan in plans if plan.degraded}
        for plan in plans:
            for message in plan.group.messages:
                for content in message.contents:
                    if content.type != "mcp_server_tool_result":
                        continue
                    owner = result_owner_group_positions.get(id(content))
                    if owner is not None and owner in degraded_positions:
                        plan.drop_content_ids.add(id(content))

    @staticmethod
    def _ordered_unique(values: Iterable[str]) -> list[str]:
        """Return values once in encounter order."""

        return list(dict.fromkeys(values))

    def _prepare_message_for_openai(
        self,
        message: Message,
        *,
        request_uses_service_side_storage: bool = True,
        reasoning_items: dict[int, dict[str, Any]] | None = None,
        drop_content_ids: set[int] | None = None,
        omit_function_call_id_content_ids: set[int] | None = None,
        hosted_degradations: Mapping[int, str | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Prepare a chat message for the OpenAI Responses API format."""
        all_messages: list[dict[str, Any]] = []
        args: dict[str, Any] = {
            "type": "message",
            "role": message.role,
        }
        additional_properties = message.additional_properties
        replays_local_storage = "_attribution" in additional_properties
        # Service-managed continuation already owns provider item identities, while
        # client-managed history uses the positional replay plan supplied here.
        reasoning_items = reasoning_items or {}
        drop_content_ids = drop_content_ids or set()
        omit_function_call_id_content_ids = omit_function_call_id_content_ids or set()
        if hosted_degradations is None:
            hosted_degradations = cross_provider_hosted_degradations(
                [message],
                target_provider=type(self).HOSTED_PROVIDER,
                unsafe_same_provider_families=(
                    (HostedToolFamily.IMAGE,) if not request_uses_service_side_storage else ()
                ),
            )

        def flush_message_run() -> None:
            nonlocal args
            if "content" in args or "tool_calls" in args:
                all_messages.append(args)
                args = dict[str, Any](type="message", role=message.role)

        for content in message.contents:
            content_identity = id(content)
            if content_identity in hosted_degradations:
                summary = hosted_degradations[content_identity]
                if summary:
                    flush_message_run()
                    all_messages.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": summary,
                                    "annotations": [],
                                }
                            ],
                        }
                    )
                continue
            match content.type:
                case "text_reasoning":
                    if request_uses_service_side_storage or content_identity in drop_content_ids:
                        continue
                    if reasoning_item := reasoning_items.pop(content_identity, None):
                        flush_message_run()
                        all_messages.append(reasoning_item)
                    continue
                case "function_result":
                    if content_identity in drop_content_ids:
                        continue
                    new_args: dict[str, Any] = {}
                    new_args.update(
                        self._prepare_content_for_openai(
                            message.role,
                            content,
                            replays_local_storage=replays_local_storage,
                        )
                    )
                    if new_args:
                        flush_message_run()
                        all_messages.append(new_args)
                case "function_call":
                    if request_uses_service_side_storage or content_identity in drop_content_ids:
                        continue
                    function_call = self._prepare_content_for_openai(
                        message.role,
                        content,
                        replays_local_storage=replays_local_storage,
                    )
                    if function_call:
                        if content_identity in omit_function_call_id_content_ids:
                            function_call.pop("id", None)
                        flush_message_run()
                        all_messages.append(function_call)
                case (
                    "search_tool_call"
                    | "search_tool_result"
                    | "mcp_server_tool_call"
                    | "mcp_server_tool_result"
                    | "code_interpreter_tool_call"
                    | "code_interpreter_tool_result"
                    | "image_generation_tool_call"
                    | "image_generation_tool_result"
                    | "shell_tool_call"
                    | "shell_tool_result"
                    | "hosted_tool_call"
                    | "hosted_tool_result"
                ):
                    # Provider-hosted items are top-level Responses input items,
                    # never nested message content. Parser-created contents keep
                    # their replay payload; manually-created MCP contents retain
                    # the legacy call/result coalescing path.
                    if request_uses_service_side_storage or content_identity in drop_content_ids:
                        continue
                    prepared_hosted = self._prepare_content_for_openai(
                        message.role,
                        content,
                        replays_local_storage=replays_local_storage,
                    )
                    if prepared_hosted:
                        flush_message_run()
                        all_messages.append(prepared_hosted)
                case _:
                    if content_identity in drop_content_ids:
                        continue
                    prepared_content = self._prepare_content_for_openai(
                        message.role,
                        content,
                        replays_local_storage=replays_local_storage,
                    )
                    if prepared_content:
                        if message.role == "assistant":
                            replay_envelope = _content_output_message_envelope(content)
                            current_envelope = {
                                field_name: args[field_name]
                                for field_name in _OUTPUT_MESSAGE_ENVELOPE_FIELDS
                                if field_name in args
                            }
                            if "content" in args and replay_envelope != current_envelope:
                                flush_message_run()
                            args.update(replay_envelope)
                        if "content" not in args:
                            args["content"] = []
                        args["content"].append(prepared_content)
        if "content" in args or "tool_calls" in args:
            all_messages.append(args)
        return all_messages

    def _prepare_content_for_openai(
        self,
        role: Role | str,
        content: Content,
        *,
        replays_local_storage: bool = False,
    ) -> dict[str, Any]:
        """Prepare content for the OpenAI Responses API format."""
        role = Role(role)
        replay_item = content.additional_properties.get(OPENAI_HOSTED_WIRE_ITEM_KEY)
        if isinstance(replay_item, Mapping) and (
            content.hosted_provider == type(self).HOSTED_PROVIDER or content.type == "function_call"
        ):
            prepared_replay_item = dict(replay_item)
            if prepared_replay_item.get("type") in {"tool_search_call", "tool_search_output"}:
                # `created_by` is response-only metadata and is not accepted on replay.
                prepared_replay_item.pop("created_by", None)
            if prepared_replay_item.get("type") in {"shell_call", "shell_call_output"}:
                prepared_replay_item.pop("created_by", None)
            if prepared_replay_item.get("type") == "shell_call_output":
                prepared_replay_item.pop("status", None)
            if prepared_replay_item.get("type") == "image_generation_call":
                # Under local storage the server-issued image item id is not
                # persisted and replaying it returns 404. The message-level
                # planner normally replaces the paired call/result with a
                # neutral summary; fail closed if this lower-level serializer
                # is reached without that plan.
                return {}
            return prepared_replay_item
        if content.additional_properties.get(OPENAI_HOSTED_REPLAY_SHADOW_KEY) is True:
            if content.type == "image_generation_tool_result":
                image_result = self._image_generation_replay_result(content)
                if image_result is not None:
                    return {
                        _CHRYS_IMAGE_GENERATION_RESULT_KEY: True,
                        "image_id": content.image_id,
                        "result": image_result,
                    }
            return {_CHRYS_HOSTED_REPLAY_SHADOW_KEY: True}
        match content.type:
            case "text":
                if role == "assistant":
                    # Assistant history is represented as output text items; Azure validation
                    # requires `annotations` to be present for this type.
                    return {
                        "type": "output_text",
                        "text": content.text,
                        "annotations": _annotations_to_output_text(content.annotations),
                    }
                return {
                    "type": "input_text",
                    "text": content.text,
                }
            case "text_reasoning":
                ret: dict[str, Any] = {"type": "reasoning", "summary": []}
                if content.id:
                    ret["id"] = content.id
                props = content.additional_properties
                reasoning_text_marker: Any = None
                if props:
                    if status := props.get("status"):
                        ret["status"] = status
                    if reasoning_text_marker := props.get("reasoning_text"):
                        reasoning_text = content.text if reasoning_text_marker is True else reasoning_text_marker
                        if isinstance(reasoning_text, str) and reasoning_text:
                            ret["content"] = [{"type": "reasoning_text", "text": reasoning_text}]
                    if encrypted_content := props.get("encrypted_content"):
                        ret["encrypted_content"] = encrypted_content
                if content.text and reasoning_text_marker is not True:
                    ret["summary"].append({"type": "summary_text", "text": content.text})
                return ret
            case "data" | "uri":
                if content.has_top_level_media_type("image"):
                    result: dict[str, Any] = {
                        "type": "input_image",
                        "image_url": content.uri,
                        "detail": content.additional_properties.get("detail", "auto")
                        if content.additional_properties
                        else "auto",
                    }
                    file_id = content.additional_properties.get("file_id") if content.additional_properties else None
                    if file_id is not None:
                        result["file_id"] = file_id
                    return result
                if content.has_top_level_media_type("audio"):
                    if content.media_type and "wav" in content.media_type:
                        format = "wav"
                    elif content.media_type and "mp3" in content.media_type:
                        format = "mp3"
                    else:
                        logger.warning("Unsupported audio media type: %s", content.media_type)
                        return {}
                    return {
                        "type": "input_audio",
                        "input_audio": {
                            "data": content.uri,
                            "format": format,
                        },
                    }
                if content.has_top_level_media_type("application"):
                    filename = getattr(content, "filename", None) or (
                        content.additional_properties.get("filename") if content.additional_properties else None
                    )
                    file_obj = {
                        "type": "input_file",
                        "file_data": content.uri,
                    }
                    if filename:
                        file_obj["filename"] = filename
                    return file_obj
                return {}
            case "function_call":
                if not content.call_id:
                    logger.warning(f"FunctionCallContent missing call_id for function '{content.name}'")
                    return {}
                fc_id = content.call_id
                if not replays_local_storage and content.additional_properties:
                    live_fc_id = content.additional_properties.get("fc_id")
                    if isinstance(live_fc_id, str) and live_fc_id:
                        fc_id = live_fc_id
                # OpenAI Responses API requires IDs to start with `fc_`
                if not fc_id.startswith("fc_"):
                    fc_id = f"fc_{fc_id}"

                function_call_obj = {
                    "call_id": content.call_id,
                    "id": fc_id,
                    "type": "function_call",
                    "name": content.name,
                    "arguments": content.arguments,
                }
                if status := content.additional_properties.get("status"):
                    function_call_obj["status"] = status
                return function_call_obj
            case "function_result":
                shell_output_type = (
                    content.additional_properties.get(OPENAI_SHELL_OUTPUT_TYPE_KEY)
                    if content.additional_properties
                    else None
                )
                if shell_output_type == OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL:
                    return {
                        "call_id": content.call_id,
                        "type": OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL,
                        "output": self._to_shell_call_output_payload(content),
                    }
                if shell_output_type == OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL:
                    return {
                        # openai-python names this field `id`, but the value is the model-generated
                        # local shell call reference, not the server-issued `local_shell_call` item id.
                        "id": content.call_id,
                        "type": OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL,
                        "output": self._to_local_shell_output_payload(content),
                    }
                # call_id for the result needs to be the same as the call_id for the function call
                output: str | list[dict[str, Any]] = content.result or ""
                if (
                    self.SUPPORTS_RICH_FUNCTION_OUTPUT
                    and content.items
                    and any(item.type in ("data", "uri") for item in content.items)
                ):
                    output_parts: list[dict[str, Any]] = []
                    for item in content.items:
                        if item.type == "text":
                            output_parts.append({"type": "input_text", "text": item.text or ""})
                        else:
                            part = self._prepare_content_for_openai("user", item)
                            if part:
                                output_parts.append(part)
                    if output_parts:
                        output = output_parts
                return {
                    "call_id": content.call_id,
                    "type": "function_call_output",
                    "output": output,
                }
            case "mcp_server_tool_call":
                if not content.call_id:
                    return {}
                return {
                    "type": "mcp_call",
                    "id": content.call_id,
                    "server_label": content.server_name or "",
                    "name": content.tool_name or "",
                    "arguments": self._stringify_mcp_arguments(content.arguments),
                }
            case "mcp_server_tool_result":
                if not content.call_id:
                    return {}
                return {
                    _CHRYS_MCP_PENDING_OUTPUT_KEY: True,
                    "call_id": content.call_id,
                    "output": self._stringify_mcp_output(content.output),
                }
            case "hosted_file":
                # `input_file` is an input-only content type in the Responses API and is rejected
                # inside an assistant message. Hosted-file content on an assistant message
                # represents a citation produced by a hosted tool (e.g., file_search) and cannot be
                # meaningfully replayed as input — drop it. The accompanying text annotations carry
                # the citation context for round-tripping.
                if role == "assistant":
                    return {}
                return {
                    "type": "input_file",
                    "file_id": content.file_id,
                }
            case _:  # should catch UsageDetails and ErrorContent and HostedVectorStoreContent
                logger.debug("Unsupported content type passed (type: %s)", content.type)
                return {}

    @staticmethod
    def _to_local_shell_output_payload(content: Content) -> str:
        """Convert function tool output to the local shell JSON payload format."""
        payload: dict[str, Any]
        if isinstance(content.result, Mapping):
            payload = dict(content.result)  # type: ignore[assignment]
        else:
            payload = {
                "stdout": "" if content.result is None else str(content.result),
            }
        # A failed call's ``exception`` is a record for people, never for the
        # model: stdout already carries the error result the model reads.
        if "exit_code" not in payload:
            payload["exit_code"] = 1 if content.exception is not None else 0
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _to_shell_call_output_payload(content: Content) -> list[dict[str, Any]]:
        """Convert function tool output to shell_call_output payload format."""
        payload: dict[str, Any]
        if isinstance(content.result, Mapping):
            payload = dict(content.result)  # type: ignore[assignment]
        else:
            payload = {
                "stdout": "" if content.result is None else str(content.result),
            }
        # Pass through native payload shape when tool already returns shell output entries.
        direct_output = payload.get("output")
        if isinstance(direct_output, list) and all(isinstance(item, Mapping) for item in direct_output):  # type: ignore[reportUnknownMemberType]
            return [dict(item) for item in direct_output]  # type: ignore[reportUnknownMemberType]

        stdout = str(payload.get("stdout", ""))
        # Only what the tool returned: a failed call's ``exception`` is a record
        # for people, never for the model, whose error result is in stdout.
        stderr = str(payload.get("stderr", ""))
        timed_out = bool(payload.get("timed_out", False))
        if timed_out:
            outcome: dict[str, Any] = {"type": "timeout"}
        else:
            exit_code_raw = payload.get("exit_code")
            try:
                exit_code = (
                    int(exit_code_raw) if exit_code_raw is not None else (1 if content.exception is not None else 0)
                )
            except TypeError, ValueError:
                exit_code = 1 if content.exception is not None else 0
            outcome = {"type": "exit", "exit_code": exit_code}
        return [
            {
                "stdout": stdout,
                "stderr": stderr,
                "outcome": outcome,
            }
        ]

    @staticmethod
    def _join_shell_commands(commands: Sequence[str]) -> str:
        """Join shell commands into a single executable command string."""
        return "\n".join(command for command in commands if command).strip()

    def _shell_item_to_contents(self, item: Any, local_shell_tool_name: str | None) -> list[Content]:
        """Convert a shell output item into Chrys ``Content`` objects."""
        contents: list[Content] = []
        item_type = getattr(item, "type", None)
        if item_type == "shell_call":
            shell_call_id = getattr(item, "call_id", None) or ""
            shell_commands: list[str] = []
            shell_timeout_ms: int | None = None
            shell_max_output: int | None = None
            if action := getattr(item, "action", None):
                shell_commands = list(getattr(action, "commands", []) or [])
                shell_timeout_ms = getattr(action, "timeout_ms", None)
                shell_max_output = getattr(action, "max_output_length", None)
            if local_shell_tool_name:
                command_text = self._join_shell_commands(shell_commands)
                contents.append(
                    Content.from_function_call(
                        call_id=shell_call_id,
                        name=local_shell_tool_name,
                        arguments=json.dumps({"command": command_text}),
                        additional_properties={
                            OPENAI_SHELL_OUTPUT_TYPE_KEY: OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL,
                            OPENAI_LOCAL_SHELL_COMMAND_PARTS_KEY: shell_commands,
                        },
                        raw_representation=item,
                    )
                )
            else:
                contents.append(
                    Content.from_shell_tool_call(
                        call_id=shell_call_id,
                        commands=shell_commands,
                        timeout_ms=shell_timeout_ms,
                        max_output_length=shell_max_output,
                        status=getattr(item, "status", None),
                        hosted_provider=type(self).HOSTED_PROVIDER,
                        provider_item_type=item_type,
                        provider_item_id=getattr(item, "id", None),
                        provider_phase=self._hosted_item_phase(item),
                        provider_status=getattr(item, "status", None),
                        retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
                        additional_properties=self._hosted_item_properties(item),
                        raw_representation=item,
                    )
                )
        elif item_type == "local_shell_call":
            local_call_id = getattr(item, "call_id", None) or ""
            local_command_parts = list(getattr(getattr(item, "action", None), "command", []) or [])
            local_command = shlex.join(local_command_parts) if local_command_parts else ""
            if local_shell_tool_name:
                contents.append(
                    Content.from_function_call(
                        call_id=local_call_id,
                        name=local_shell_tool_name,
                        arguments=json.dumps({"command": local_command}),
                        additional_properties={
                            OPENAI_SHELL_OUTPUT_TYPE_KEY: OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL,
                            OPENAI_LOCAL_SHELL_CALL_ITEM_ID_KEY: getattr(item, "id", None),
                            OPENAI_LOCAL_SHELL_COMMAND_PARTS_KEY: local_command_parts,
                        },
                        raw_representation=item,
                    )
                )
            else:
                contents.append(
                    Content.from_shell_tool_call(
                        call_id=local_call_id,
                        commands=[local_command] if local_command else [],
                        timeout_ms=getattr(getattr(item, "action", None), "timeout_ms", None),
                        status=getattr(item, "status", None),
                        hosted_provider=type(self).HOSTED_PROVIDER,
                        provider_item_type=item_type,
                        provider_item_id=getattr(item, "id", None),
                        provider_phase=self._hosted_item_phase(item),
                        provider_status=getattr(item, "status", None),
                        retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
                        additional_properties=self._hosted_item_properties(item),
                        raw_representation=item,
                    )
                )
        elif item_type == "shell_call_output":
            shell_output_call_id = getattr(item, "call_id", None) or ""
            shell_outputs: list[Content] = []
            for shell_out in getattr(item, "output", []) or []:
                s_exit_code: int | None = None
                s_timed_out: bool | None = None
                if outcome := getattr(shell_out, "outcome", None):
                    if getattr(outcome, "type", None) == "exit":
                        s_exit_code = getattr(outcome, "exit_code", None)
                        s_timed_out = False
                    elif getattr(outcome, "type", None) == "timeout":
                        s_timed_out = True
                shell_outputs.append(
                    Content.from_shell_command_output(
                        stdout=getattr(shell_out, "stdout", None),
                        stderr=getattr(shell_out, "stderr", None),
                        exit_code=s_exit_code,
                        timed_out=s_timed_out,
                        raw_representation=shell_out,
                    )
                )
            contents.append(
                Content.from_shell_tool_result(
                    call_id=shell_output_call_id,
                    outputs=shell_outputs,
                    max_output_length=getattr(item, "max_output_length", None),
                    hosted_provider=type(self).HOSTED_PROVIDER,
                    provider_item_type=item_type,
                    provider_item_id=getattr(item, "id", None),
                    provider_phase=self._hosted_item_phase(item),
                    provider_status=getattr(item, "status", None),
                    retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
                    additional_properties=self._hosted_item_properties(
                        item,
                        error=next(
                            (
                                (
                                    "shell command timed out"
                                    if output.timed_out
                                    else output.stderr or f"shell command exited with code {output.exit_code}"
                                )
                                for output in shell_outputs
                                if output.timed_out or (output.exit_code is not None and output.exit_code != 0)
                            ),
                            None,
                        ),
                    ),
                    raw_representation=item,
                )
            )
        return contents

    @staticmethod
    def _stringify_mcp_arguments(arguments: Any) -> str:
        """Render hosted-MCP tool-call arguments as a JSON string for the Responses API."""
        if arguments is None:
            return ""
        if isinstance(arguments, str):
            return arguments
        try:
            return json.dumps(arguments)
        except TypeError, ValueError:
            return str(arguments)

    @staticmethod
    def _stringify_mcp_output(output: Any) -> str:
        """Render a hosted-MCP tool-call result into the string `mcp_call.output` field.

        Accepts a string, a list of text-bearing Content objects (the form
        the chat client produces when parsing an `mcp_call` Responses item),
        or any other value. List entries that are dicts with the canonical
        MCP text-content shape (`{"text": "..."}`) are unwrapped to their
        text. Anything else falls back to JSON encoding rather than Python
        `repr`, so the wire payload stays parseable for downstream callers.
        """
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        if isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
            # cast is for pyright (reportUnknownVariableType); mypy considers
            # it redundant after the isinstance narrowing.
            entries = cast(Sequence[Any], output)  # type: ignore[redundant-cast]
            parts: list[str] = []
            for entry in entries:
                if isinstance(entry, str):
                    parts.append(entry)
                    continue
                text = getattr(entry, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if isinstance(entry, Mapping):
                    mapping_text = cast(Any, entry).get("text")
                    if isinstance(mapping_text, str):
                        parts.append(mapping_text)
                        continue
                parts.append(json.dumps(entry, default=str))
            return "".join(parts)
        return json.dumps(output, default=str)

    @staticmethod
    def _coalesce_pending_hosted_results(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge canonical result shadows into their provider replay items.

        Hosted MCP needs one ``mcp_call`` carrying both arguments and output.
        Image generation similarly reattaches the canonical data URI payload
        to the stripped ``image_generation_call`` only while preparing the
        next request, keeping persisted history to one copy of the base64.
        Unmatched markers are dropped because neither is a valid standalone
        Responses input item.
        """
        out: list[dict[str, Any]] = []
        calls_by_key: dict[PairingKey, deque[dict[str, Any]]] = {}
        for item in items:
            if item.get(_CHRYS_HOSTED_REPLAY_SHADOW_KEY):
                continue
            if item.get(_CHRYS_MCP_PENDING_OUTPUT_KEY):
                target_call_id = item.get("call_id")
                key = namespaced_pairing_key("mcp_server_tool_result", target_call_id)
                queue = calls_by_key.get(key) if key is not None else None
                target = queue.popleft() if queue else None
                if target is not None:
                    if target.get("output") is None:
                        target["output"] = item.get("output")
                else:
                    logger.debug(
                        "Dropping orphan mcp_server_tool_result for call_id=%s; "
                        "no matching mcp_call appeared in input.",
                        target_call_id,
                    )
                continue
            if item.get(_CHRYS_IMAGE_GENERATION_RESULT_KEY):
                image_id = item.get("image_id")
                key = namespaced_pairing_key("image_generation_tool_result", image_id)
                queue = calls_by_key.get(key) if key is not None else None
                target = queue.popleft() if queue else None
                result = item.get("result")
                if target is not None and isinstance(result, str) and target.get("result") is None:
                    target["result"] = result
                    if target.get("status") in {"in_progress", "generating"}:
                        target["status"] = "completed"
                elif target is None:
                    logger.debug(
                        "Dropping orphan image_generation_tool_result for image_id=%s; "
                        "no matching image_generation_call appeared in input.",
                        image_id,
                    )
                continue
            out.append(item)
            item_id = item.get("id")
            item_type = item.get("type")
            content_type = (
                "mcp_server_tool_call"
                if item_type == "mcp_call"
                else "image_generation_tool_call"
                if item_type == "image_generation_call"
                else None
            )
            if content_type is not None:
                key = namespaced_pairing_key(content_type, item_id)
                if key is not None:
                    calls_by_key.setdefault(key, deque()).append(item)
        return out

    @staticmethod
    def _image_generation_replay_result(content: Content) -> str | None:
        """Extract one non-partial base64 payload from canonical image output."""
        outputs = content.outputs if isinstance(content.outputs, list) else [content.outputs]
        for output in reversed(outputs):
            if not isinstance(output, Content) or output.additional_properties.get("is_partial_image") is True:
                continue
            uri = output.uri or ""
            marker = ";base64,"
            if marker in uri:
                return uri.split(marker, 1)[1]
        return None

    @staticmethod
    def _serialize_provider_payload(value: Any) -> Any:
        """Convert OpenAI SDK objects into JSON-serializable Python values."""
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", exclude_none=True, exclude_unset=True)
        if isinstance(value, Mapping):
            return {str(key): RawOpenAIChatClient._serialize_provider_payload(item) for key, item in value.items()}  # type: ignore[reportUnknownVariableType]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [RawOpenAIChatClient._serialize_provider_payload(item) for item in value]  # type: ignore[reportUnknownVariableType]
        if hasattr(value, "__dict__"):
            return {
                str(key): RawOpenAIChatClient._serialize_provider_payload(item)
                for key, item in vars(value).items()
                if not key.startswith("_") and item is not None
            }
        return value

    def _hosted_item_properties(
        self,
        item: Any,
        *,
        replay_shadow: bool = False,
        error: Any = None,
        wire_exclude: set[str] | None = None,
    ) -> dict[str, Any]:
        """Build persisted provider replay and failure metadata for one output item."""
        properties: dict[str, Any] = {}
        if replay_shadow:
            properties[OPENAI_HOSTED_REPLAY_SHADOW_KEY] = True
        else:
            wire_item = self._serialize_provider_payload(item)
            if isinstance(wire_item, dict):
                for key in wire_exclude or ():
                    wire_item.pop(key, None)
            properties[OPENAI_HOSTED_WIRE_ITEM_KEY] = wire_item
        if error is not None:
            properties["is_error"] = True
            properties["error"] = self._serialize_provider_payload(error)
        elif str(getattr(item, "status", "")).lower() in {"failed", "incomplete", "cancelled", "canceled"}:
            properties["is_error"] = True
        return properties

    @staticmethod
    def _hosted_item_phase(item: Any) -> str:
        """Phase for an assembled output item: terminal unless still running.

        Blocking snapshots of an in-progress response (background mode,
        continuation polls) carry items that are still ``in_progress`` or
        ``queued``; stamping those terminal publishes a premature result and
        the real one is then discarded as late. Absent or unrecognized
        statuses keep the terminal reading — completed responses routinely
        omit item status.
        """
        status = getattr(item, "status", None)
        normalized = normalize_hosted_tool_status(status if isinstance(status, str) else None)
        if normalized in (HostedToolStatus.PENDING, HostedToolStatus.RUNNING):
            return HostedToolPhase.SNAPSHOT
        return HostedToolPhase.TERMINAL

    @staticmethod
    def _get_search_tool_name(item_type: str) -> str:
        """Map OpenAI search output item types to unified content tool names."""
        return "web_search" if item_type == "web_search_call" else "file_search"

    def _parse_search_tool_call_content(
        self,
        item: Any,
        *,
        phase: str | None = None,
    ) -> Content:
        """Create unified search tool call content from an OpenAI search output item."""
        item_type = getattr(item, "type", "")
        call_id = getattr(item, "id", None) or getattr(item, "call_id", None) or ""
        if item_type == "web_search_call":
            arguments = self._serialize_provider_payload(getattr(item, "action", None))
        else:
            arguments = {"queries": list(getattr(item, "queries", []) or [])}
        return Content.from_search_tool_call(
            call_id=call_id,
            tool_name=self._get_search_tool_name(item_type),
            arguments=arguments,
            status=getattr(item, "status", None),
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type=item_type,
            provider_item_id=getattr(item, "id", None),
            provider_phase=phase if phase is not None else self._hosted_item_phase(item),
            provider_status=getattr(item, "status", None),
            retry_safety=HostedRetrySafety.READ_ONLY,
            additional_properties=self._hosted_item_properties(item),
            raw_representation=item,
        )

    def _parse_search_tool_result_content(
        self,
        item: Any,
        *,
        replay_shadow: bool = True,
    ) -> Content:
        """Create unified search tool result content from an OpenAI search output item."""
        item_type = getattr(item, "type", "")
        call_id = getattr(item, "id", None) or getattr(item, "call_id", None) or ""
        if item_type == "web_search_call":
            result = {"action": self._serialize_provider_payload(getattr(item, "action", None))}
        else:
            result = {"results": self._serialize_provider_payload(getattr(item, "results", None))}
        return Content.from_search_tool_result(
            call_id=call_id,
            tool_name=self._get_search_tool_name(item_type),
            result=result,
            status=getattr(item, "status", None),
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type=item_type,
            provider_item_id=getattr(item, "id", None),
            provider_phase=self._hosted_item_phase(item),
            provider_status=getattr(item, "status", None),
            retry_safety=HostedRetrySafety.READ_ONLY,
            additional_properties=self._hosted_item_properties(item, replay_shadow=replay_shadow),
            raw_representation=item,
        )

    def _parse_hosted_function_call_content(
        self,
        item: Any,
        *,
        name: str,
        arguments: Any = None,
    ) -> Content:
        """Preserve a client-executed custom call without making it locally executable."""
        item_type = str(getattr(item, "type", ""))
        additional_properties: dict[str, Any] = {
            "item_type": item_type,
            OPENAI_HOSTED_WIRE_ITEM_KEY: self._serialize_provider_payload(item),
        }
        if item_type == "custom_tool_call":
            call_id = getattr(item, "call_id", "") or ""
            if item_id := getattr(item, "id", None):
                additional_properties["item_id"] = item_id
            if namespace := getattr(item, "namespace", None):
                additional_properties["namespace"] = namespace
        else:
            item_id = getattr(item, "id", "") or ""
            call_id = getattr(item, "call_id", None) or item_id
            additional_properties.update(
                {
                    "item_id": item_id,
                    "status": getattr(item, "status", None),
                    "execution": self._serialize_provider_payload(getattr(item, "execution", None)),
                }
            )
            if created_by := getattr(item, "created_by", None):
                additional_properties["created_by"] = created_by
        return Content.from_function_call(
            call_id=call_id,
            name=name,
            arguments=self._serialize_provider_payload(arguments),
            informational_only=True,
            additional_properties=additional_properties,
            raw_representation=item,
        )

    def _parse_tool_search_call_content(
        self,
        item: Any,
        *,
        phase: str | None = None,
    ) -> Content:
        """Normalize a Responses tool-search call as hosted tool discovery."""
        item_id = getattr(item, "id", None)
        call_id = getattr(item, "call_id", None) or item_id
        return Content.from_hosted_tool_call(
            call_id=call_id,
            tool_name="tool_search",
            arguments=self._serialize_provider_payload(getattr(item, "arguments", None)),
            status=getattr(item, "status", None),
            hosted_family=HostedToolFamily.TOOL_DISCOVERY,
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type=str(getattr(item, "type", "tool_search_call")),
            provider_item_id=item_id,
            provider_phase=phase if phase is not None else self._hosted_item_phase(item),
            provider_status=getattr(item, "status", None),
            retry_safety=HostedRetrySafety.READ_ONLY,
            additional_properties=self._hosted_item_properties(item),
            raw_representation=item,
        )

    def _parse_tool_search_result_content(self, item: Any) -> Content:
        """Normalize a distinct Responses tool-search output item."""
        item_id = getattr(item, "id", None)
        call_id = getattr(item, "call_id", None) or item_id
        status = getattr(item, "status", None)
        return Content.from_hosted_tool_result(
            call_id=call_id,
            tool_name="tool_search",
            result={"tools": self._serialize_provider_payload(getattr(item, "tools", []))},
            status=status,
            hosted_family=HostedToolFamily.TOOL_DISCOVERY,
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type=str(getattr(item, "type", "tool_search_output")),
            provider_item_id=item_id,
            provider_phase=self._hosted_item_phase(item),
            provider_status=status,
            retry_safety=HostedRetrySafety.READ_ONLY,
            additional_properties=self._hosted_item_properties(item),
            raw_representation=item,
        )

    @staticmethod
    def _has_provider_execution_evidence(item: Any) -> bool:
        """Return whether an unrecognized output item explicitly records server execution."""
        if getattr(item, "execution", None) == "server":
            return True
        return getattr(item, "server_execution", None) is not None

    def _parse_generic_hosted_item_contents(
        self,
        item: Any,
        *,
        phase: str | None = None,
    ) -> list[Content]:
        """Normalize an unrecognized server-executed Responses item fail-closed."""
        item_type = str(getattr(item, "type", "unknown_server_item"))
        item_id = getattr(item, "id", None)
        call_id = getattr(item, "call_id", None) or item_id
        status = getattr(item, "status", None)
        tool_name = str(getattr(item, "name", None) or item_type)
        arguments = getattr(item, "arguments", None)
        if arguments is None:
            arguments = getattr(item, "input", None)
        call = Content.from_hosted_tool_call(
            call_id=call_id,
            tool_name=tool_name,
            arguments=self._serialize_provider_payload(arguments),
            status=status,
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type=item_type,
            provider_item_id=item_id,
            provider_phase=phase if phase is not None else self._hosted_item_phase(item),
            provider_status=status,
            retry_safety=HostedRetrySafety.UNKNOWN,
            additional_properties=self._hosted_item_properties(item),
            raw_representation=item,
        )
        result_payload = next(
            (
                self._serialize_provider_payload(value)
                for value in (
                    getattr(item, "output", None),
                    getattr(item, "result", None),
                    getattr(item, "outputs", None),
                )
                if value is not None
            ),
            None,
        )
        if result_payload is None:
            return [call]
        result = Content.from_hosted_tool_result(
            call_id=call_id,
            tool_name=tool_name,
            result=result_payload,
            status=status,
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type=item_type,
            provider_item_id=item_id,
            provider_phase=self._hosted_item_phase(item),
            provider_status=status,
            retry_safety=HostedRetrySafety.UNKNOWN,
            additional_properties=self._hosted_item_properties(item, replay_shadow=True),
            raw_representation=item,
        )
        return [call, result]

    def _parse_mcp_item_contents(
        self,
        item: Any,
        *,
        phase: str | None = None,
    ) -> list[Content]:
        """Normalize one hosted MCP output item."""
        call_id = getattr(item, "id", None) or getattr(item, "call_id", None) or ""
        status = getattr(item, "status", None)
        item_id = getattr(item, "id", None)
        call = Content.from_mcp_server_tool_call(
            call_id=call_id,
            tool_name=getattr(item, "name", "") or "",
            server_name=getattr(item, "server_label", None),
            arguments=getattr(item, "arguments", None),
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type="mcp_call",
            provider_item_id=item_id,
            provider_phase=phase if phase is not None else self._hosted_item_phase(item),
            provider_status=status,
            retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
            additional_properties=self._hosted_item_properties(item),
            raw_representation=item,
        )
        output = getattr(item, "output", None)
        error = getattr(item, "error", None)
        if output is None and error is None:
            return [call]
        parsed_output = (
            [Content.from_error(message=str(error), raw_representation=item)]
            if error is not None
            else [Content.from_text(text=str(output), raw_representation=item)]
        )
        result = Content.from_mcp_server_tool_result(
            call_id=call_id,
            output=parsed_output,
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type="mcp_call",
            provider_item_id=item_id,
            provider_phase=self._hosted_item_phase(item),
            provider_status=status,
            retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
            additional_properties=self._hosted_item_properties(item, replay_shadow=True, error=error),
            raw_representation=item,
        )
        return [call, result]

    def _parse_code_outputs(self, item: Any) -> list[Content]:
        """Normalize code-interpreter outputs, including hosted file artifacts."""
        outputs: list[Content] = []
        for code_output in getattr(item, "outputs", None) or []:
            match getattr(code_output, "type", None):
                case "logs":
                    outputs.append(Content.from_text(text=code_output.logs, raw_representation=code_output))
                case "image":
                    outputs.append(
                        Content.from_uri(
                            uri=code_output.url,
                            raw_representation=code_output,
                            media_type="image",
                        )
                    )
                case "file" | "hosted_file":
                    if file_id := getattr(code_output, "file_id", None):
                        outputs.append(
                            Content.from_hosted_file(
                                file_id=file_id,
                                name=getattr(code_output, "filename", None),
                                raw_representation=code_output,
                            )
                        )
                case _:
                    logger.debug(
                        "responses_parser_unrecognized_code_output provider=%s output_type=%s",
                        type(self).HOSTED_PROVIDER,
                        getattr(code_output, "type", None),
                    )
        return outputs

    def _parse_code_item_contents(
        self,
        item: Any,
        *,
        phase: str | None = None,
    ) -> list[Content]:
        """Normalize one code-interpreter item into a call and optional result."""
        call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
        item_id = getattr(item, "id", None)
        status = getattr(item, "status", None)
        code = getattr(item, "code", None)
        call = Content.from_code_interpreter_tool_call(
            call_id=call_id,
            inputs=[Content.from_text(text=code, raw_representation=item)] if code else [],
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type="code_interpreter_call",
            provider_item_id=item_id,
            provider_phase=phase if phase is not None else self._hosted_item_phase(item),
            provider_status=status,
            retry_safety=HostedRetrySafety.SANDBOXED,
            additional_properties=self._hosted_item_properties(item),
            raw_representation=item,
        )
        outputs = self._parse_code_outputs(item)
        if not outputs:
            return [call]
        result = Content.from_code_interpreter_tool_result(
            call_id=call_id,
            outputs=outputs,
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type="code_interpreter_call",
            provider_item_id=item_id,
            provider_phase=self._hosted_item_phase(item),
            provider_status=status,
            retry_safety=HostedRetrySafety.SANDBOXED,
            additional_properties=self._hosted_item_properties(item, replay_shadow=True),
            raw_representation=item,
        )
        return [call, result]

    def _parse_image_item_contents(
        self,
        item: Any,
        *,
        phase: str | None = None,
    ) -> list[Content]:
        """Normalize one image-generation item into a call and optional result."""
        item_id = getattr(item, "id", None)
        status = getattr(item, "status", None)
        image_base64 = getattr(item, "result", None)
        # OpenAI can retain ``status="generating"`` on a completed response;
        # the full result payload is stronger terminal evidence than that stale status.
        item_phase = (
            phase
            if phase is not None
            else HostedToolPhase.TERMINAL
            if image_base64 is not None
            else self._hosted_item_phase(item)
        )
        call = Content.from_image_generation_tool_call(
            image_id=item_id,
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type="image_generation_call",
            provider_item_id=item_id,
            provider_phase=item_phase,
            provider_status=status,
            retry_safety=HostedRetrySafety.SANDBOXED,
            additional_properties=self._hosted_item_properties(item, wire_exclude={"result"}),
            raw_representation=item,
        )
        if image_base64 is None:
            return [call]
        image_output = Content.from_uri(
            uri=f"data:{detect_media_type_from_base64(data_str=image_base64) or 'image/png'};base64,{image_base64}",
            raw_representation=image_base64,
        )
        result = Content.from_image_generation_tool_result(
            image_id=item_id,
            outputs=[image_output],
            hosted_provider=type(self).HOSTED_PROVIDER,
            provider_item_type="image_generation_call",
            provider_item_id=item_id,
            provider_phase=item_phase,
            provider_status=status,
            retry_safety=HostedRetrySafety.SANDBOXED,
            additional_properties=self._hosted_item_properties(item, replay_shadow=True),
            raw_representation=item,
        )
        return [call, result]

    @staticmethod
    def _refresh_streamed_hosted_content(target: Content, snapshot: Content) -> None:
        """Refresh one indexed streamed occurrence without replacing its identity."""
        target.call_id = snapshot.call_id
        target.image_id = snapshot.image_id
        target.name = snapshot.name
        target.tool_name = snapshot.tool_name
        target.server_name = snapshot.server_name
        target.arguments = snapshot.arguments
        target.inputs = snapshot.inputs
        target.outputs = snapshot.outputs
        target.output = snapshot.output
        target.result = snapshot.result
        target.items = snapshot.items
        target.commands = snapshot.commands
        target.timeout_ms = snapshot.timeout_ms
        target.max_output_length = snapshot.max_output_length
        target.status = snapshot.status
        target.additional_properties = snapshot.additional_properties
        target.raw_representation = snapshot.raw_representation
        target.provider_item_type = snapshot.provider_item_type
        target.provider_item_id = snapshot.provider_item_id
        target.provider_phase = snapshot.provider_phase
        target.provider_status = snapshot.provider_status
        target.retry_safety = snapshot.retry_safety

    def _merge_streamed_hosted_content(
        self,
        contents_by_index: dict[int, Content],
        output_index: int,
        snapshot: Content,
    ) -> Content:
        """Fold a streamed snapshot into its indexed occurrence carrier."""
        existing = contents_by_index.get(output_index)
        if existing is None:
            contents_by_index[output_index] = snapshot
            return snapshot
        self._refresh_streamed_hosted_content(existing, snapshot)
        return existing

    # region Parse methods
    def _parse_response_from_openai(
        self,
        response: OpenAIResponse | ParsedResponse[BaseModel],
        options: dict[str, Any],
    ) -> ChatResponse:
        """Parse an OpenAI Responses API response into a ChatResponse."""
        # The SDK's ``ParsedResponse`` generic is erased by ``isinstance``;
        # this client only requests Pydantic response formats.
        structured_response = (
            cast("BaseModel | None", response.output_parsed) if isinstance(response, ParsedResponse) else None
        )

        metadata: dict[str, Any] = response.metadata or {}
        contents: list[Content] = []
        local_shell_tool_name = self._get_local_shell_tool_name(options.get("tools"))
        try:
            response_outputs = response.output  # type: ignore[reportUnknownMemberType]
        except AttributeError:
            response_outputs = []
        for item in response_outputs:  # type: ignore[reportUnknownVariableType]
            match item.type:
                # types:
                # ParsedResponseOutputMessage[Unknown] |
                # ParsedResponseFunctionToolCall |
                # ResponseFileSearchToolCall |
                # ResponseFunctionWebSearch |
                # ResponseComputerToolCall |
                # ResponseReasoningItem |
                # MCPCall |
                # MCPApprovalRequest |
                # ImageGenerationCall |
                # LocalShellCall |
                # LocalShellCallAction |
                # MCPListTools |
                # ResponseCodeInterpreterToolCall |
                # ResponseCustomToolCall |
                # ParsedResponseOutputMessage[BaseModel] |
                # ResponseOutputMessage |
                # ResponseFunctionToolCall
                case "message":  # ResponseOutputMessage
                    message_properties = _output_message_content_properties(item)
                    for message_content in item.content:  # type: ignore[reportMissingTypeArgument]
                        match message_content.type:
                            case "output_text":
                                text_content = Content.from_text(
                                    text=message_content.text,
                                    raw_representation=message_content,  # type: ignore[reportUnknownArgumentType]
                                    additional_properties=message_properties,
                                )
                                metadata.update(self._get_metadata_from_response(message_content))
                                if message_content.annotations:
                                    text_content.annotations = []
                                    for annotation in message_content.annotations:
                                        match annotation.type:
                                            case "file_path":
                                                text_content.annotations.append(  # pyright: ignore[reportUnknownMemberType]
                                                    Annotation(
                                                        type="citation",
                                                        file_id=annotation.file_id,
                                                        additional_properties={
                                                            "index": annotation.index,
                                                        },
                                                        raw_representation=annotation,
                                                    )
                                                )
                                            case "file_citation":
                                                text_content.annotations.append(  # pyright: ignore[reportUnknownMemberType]
                                                    Annotation(
                                                        type="citation",
                                                        url=annotation.filename,
                                                        file_id=annotation.file_id,
                                                        raw_representation=annotation,
                                                        additional_properties={
                                                            "index": annotation.index,
                                                        },
                                                    )
                                                )
                                            case "url_citation":
                                                text_content.annotations.append(  # pyright: ignore[reportUnknownMemberType]
                                                    Annotation(
                                                        type="citation",
                                                        title=annotation.title,
                                                        url=annotation.url,
                                                        annotated_regions=[
                                                            TextSpanRegion(
                                                                type="text_span",
                                                                start_index=annotation.start_index,
                                                                end_index=annotation.end_index,
                                                            )
                                                        ],
                                                        raw_representation=annotation,
                                                    )
                                                )
                                            case "container_file_citation":
                                                text_content.annotations.append(  # pyright: ignore[reportUnknownMemberType]
                                                    Annotation(
                                                        type="citation",
                                                        file_id=annotation.file_id,
                                                        url=annotation.filename,
                                                        additional_properties={
                                                            "container_id": annotation.container_id,
                                                        },
                                                        annotated_regions=[
                                                            TextSpanRegion(
                                                                type="text_span",
                                                                start_index=annotation.start_index,
                                                                end_index=annotation.end_index,
                                                            )
                                                        ],
                                                        raw_representation=annotation,
                                                    )
                                                )
                                            case _:
                                                logger.debug(
                                                    "Unparsed annotation type: %s",
                                                    annotation.type,
                                                )
                                contents.append(text_content)
                            case "refusal":
                                contents.append(
                                    Content.from_text(
                                        text=message_content.refusal,
                                        raw_representation=message_content,
                                        additional_properties=message_properties,
                                    )
                                )
                case "reasoning":  # ResponseOutputReasoning
                    added_reasoning = False
                    encrypted_content = getattr(item, "encrypted_content", None)
                    if item_content := getattr(item, "content", None):
                        for index, reasoning_content in enumerate(item_content):
                            additional_properties: dict[str, Any] = {"reasoning_text": True}
                            if hasattr(item, "summary") and item.summary and index < len(item.summary):
                                additional_properties["summary"] = item.summary[index]
                            contents.append(
                                Content.from_text_reasoning(
                                    id=item.id,
                                    text=reasoning_content.text,
                                    protected_data=encrypted_content if not added_reasoning else None,
                                    raw_representation=reasoning_content,
                                    additional_properties=additional_properties or None,
                                )
                            )
                            added_reasoning = True
                    if item_summary := getattr(item, "summary", None):
                        for summary in item_summary:
                            contents.append(
                                Content.from_text_reasoning(
                                    id=item.id,
                                    text=summary.text,
                                    protected_data=encrypted_content if not added_reasoning else None,
                                    raw_representation=summary,  # type: ignore[arg-type]
                                )
                            )
                            added_reasoning = True
                    if not added_reasoning:
                        # Reasoning item with no visible text (e.g. encrypted reasoning).
                        # Always emit an empty marker so co-occurrence detection can be done
                        additional_properties_empty: dict[str, Any] = {}
                        contents.append(
                            Content.from_text_reasoning(
                                id=item.id,
                                text="",
                                protected_data=encrypted_content,
                                raw_representation=item,
                                additional_properties=additional_properties_empty or None,
                            )
                        )
                case "code_interpreter_call":  # ResponseOutputCodeInterpreterCall
                    contents.extend(self._parse_code_item_contents(item))
                case "function_call":  # ResponseOutputFunctionCall
                    contents.append(
                        Content.from_function_call(
                            call_id=item.call_id,
                            name=item.name,
                            arguments=item.arguments,
                            additional_properties={"fc_id": item.id, "status": item.status},
                            raw_representation=item,
                        )
                    )
                case "custom_tool_call":
                    contents.append(
                        self._parse_hosted_function_call_content(item, name=item.name, arguments=item.input)
                    )
                case "apply_patch_call":
                    contents.append(
                        self._parse_hosted_function_call_content(
                            item, name="apply_patch", arguments=getattr(item, "operation", None)
                        )
                    )
                case "tool_search_call":
                    contents.append(self._parse_tool_search_call_content(item))
                case "tool_search_output":
                    contents.append(self._parse_tool_search_result_content(item))
                case "web_search_call" | "file_search_call":
                    contents.append(self._parse_search_tool_call_content(item))
                    contents.append(self._parse_search_tool_result_content(item))
                case "mcp_call":
                    contents.extend(self._parse_mcp_item_contents(item))
                case "image_generation_call":  # ResponseOutputImageGenerationCall
                    contents.extend(self._parse_image_item_contents(item))
                case "shell_call" | "local_shell_call" | "shell_call_output":
                    contents.extend(self._shell_item_to_contents(item, local_shell_tool_name))
                case _:
                    if self._has_provider_execution_evidence(item):
                        contents.extend(self._parse_generic_hosted_item_contents(item))
                    else:
                        logger.debug(
                            "responses_parser_unrecognized_server_item provider=%s item_type=%s "
                            "execution_evidence=false",
                            type(self).HOSTED_PROVIDER,
                            item.type,
                        )
        response_message = Message(role="assistant", contents=contents)
        args: dict[str, Any] = {
            "response_id": response.id,
            "created_at": datetime.fromtimestamp(response.created_at, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "messages": response_message,
            "model": response.model,
            "additional_properties": metadata,
            "raw_representation": response,
        }

        if conversation_id := self._get_conversation_id(response, effective_store_option(options)):
            args["conversation_id"] = conversation_id
        if response.usage and (usage_details := self._parse_usage_from_openai(response.usage)):
            args["usage_details"] = usage_details
        if structured_response:
            args["value"] = structured_response
        elif response_format := options.get("response_format"):
            args["response_format"] = response_format
        # Set continuation_token when the operation is still in progress and the
        # response is actually retrievable: an unstored response (store false on
        # the wire) 404s on retrieval, so a token would turn every
        # disconnect-retry into a guaranteed failure instead of a fresh create.
        if (
            type(self).MINTS_CONTINUATION_TOKENS
            and response.status
            and response.status in ("in_progress", "queued")
            and effective_store_option(options) is not False
        ):
            args["continuation_token"] = OpenAIContinuationToken(response_id=response.id)
        # Map an output-token cutoff (status "incomplete" / reason
        # "max_output_tokens") to the canonical "length" finish_reason so
        # downstream truncation handling (response validation, tool-arg parsing)
        # treats it like Chat Completions / Anthropic instead of an opaque empty
        # response that gets retried.
        incomplete_details = getattr(response, "incomplete_details", None)
        if response.status == "incomplete" and getattr(incomplete_details, "reason", None) == "max_output_tokens":
            args["finish_reason"] = "length"
        return ChatResponse(**args)

    def _parse_chunk_from_openai(
        self,
        event: OpenAIResponseStreamEvent,
        options: dict[str, Any],
        function_call_ids: dict[int, tuple[str, str]],
        seen_reasoning_delta_item_ids: set[str] | None = None,
        hosted_call_contents: dict[int, Content] | None = None,
        hosted_result_contents: dict[int, Content] | None = None,
        output_message_envelopes: dict[int, dict[str, str]] | None = None,
        output_message_contents: dict[int, list[Content]] | None = None,
    ) -> ChatResponseUpdate:
        """Parse an OpenAI Responses API streaming event into a ChatResponseUpdate."""
        hosted_call_contents = hosted_call_contents if hosted_call_contents is not None else {}
        hosted_result_contents = hosted_result_contents if hosted_result_contents is not None else {}
        output_message_envelopes = output_message_envelopes if output_message_envelopes is not None else {}
        output_message_contents = output_message_contents if output_message_contents is not None else {}

        def streamed_message_properties(stream_event: Any) -> dict[str, Any] | None:
            properties: dict[str, Any] = dict(_presentation_text_properties(stream_event) or {})
            output_index = getattr(stream_event, "output_index", None)
            if isinstance(output_index, int) and not isinstance(output_index, bool):
                envelope = output_message_envelopes.get(output_index)
                if envelope:
                    properties[OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY] = dict(envelope)
            return properties or None

        def track_message_content(stream_event: Any, content: Content) -> Content:
            output_index = getattr(stream_event, "output_index", None)
            if isinstance(output_index, int) and not isinstance(output_index, bool):
                output_message_contents.setdefault(output_index, []).append(content)
            return content

        metadata: dict[str, Any] = {}
        contents: list[Content] = []
        local_shell_tool_name = self._get_local_shell_tool_name(options.get("tools"))
        conversation_id: str | None = None
        response_id: str | None = None
        created_at: str | None = None
        continuation_token: OpenAIContinuationToken | None = None
        finish_reason: str | None = None
        model = self.model
        match event.type:
            # types:
            # ResponseAudioDeltaEvent,
            # ResponseAudioDoneEvent,
            # ResponseAudioTranscriptDeltaEvent,
            # ResponseAudioTranscriptDoneEvent,
            # ResponseCodeInterpreterCallCodeDeltaEvent,
            # ResponseCodeInterpreterCallCodeDoneEvent,
            # ResponseCodeInterpreterCallCompletedEvent,
            # ResponseCodeInterpreterCallInProgressEvent,
            # ResponseCodeInterpreterCallInterpretingEvent,
            # ResponseCompletedEvent,
            # ResponseContentPartAddedEvent,
            # ResponseContentPartDoneEvent,
            # ResponseCreatedEvent,
            # ResponseErrorEvent,
            # ResponseFileSearchCallCompletedEvent,
            # ResponseFileSearchCallInProgressEvent,
            # ResponseFileSearchCallSearchingEvent,
            # ResponseFunctionCallArgumentsDeltaEvent,
            # ResponseFunctionCallArgumentsDoneEvent,
            # ResponseInProgressEvent,
            # ResponseFailedEvent,
            # ResponseIncompleteEvent,
            # ResponseOutputItemAddedEvent,
            # ResponseOutputItemDoneEvent,
            # ResponseReasoningSummaryPartAddedEvent,
            # ResponseReasoningSummaryPartDoneEvent,
            # ResponseReasoningSummaryTextDeltaEvent,
            # ResponseReasoningSummaryTextDoneEvent,
            # ResponseReasoningTextDeltaEvent,
            # ResponseReasoningTextDoneEvent,
            # ResponseRefusalDeltaEvent,
            # ResponseRefusalDoneEvent,
            # ResponseTextDeltaEvent,
            # ResponseTextDoneEvent,
            # ResponseWebSearchCallCompletedEvent,
            # ResponseWebSearchCallInProgressEvent,
            # ResponseWebSearchCallSearchingEvent,
            # ResponseImageGenCallCompletedEvent,
            # ResponseImageGenCallGeneratingEvent,
            # ResponseImageGenCallInProgressEvent,
            # ResponseImageGenCallPartialImageEvent,
            # ResponseMcpCallArgumentsDeltaEvent,
            # ResponseMcpCallArgumentsDoneEvent,
            # ResponseMcpCallCompletedEvent,
            # ResponseMcpCallFailedEvent,
            # ResponseMcpCallInProgressEvent,
            # ResponseMcpListToolsCompletedEvent,
            # ResponseMcpListToolsFailedEvent,
            # ResponseMcpListToolsInProgressEvent,
            # ResponseOutputTextAnnotationAddedEvent,
            # ResponseQueuedEvent,
            # ResponseCustomToolCallInputDeltaEvent,
            # ResponseCustomToolCallInputDoneEvent,
            case "response.content_part.added":
                event_part = event.part
                match event_part.type:
                    case "output_text":
                        contents.append(
                            track_message_content(
                                event,
                                Content.from_text(
                                    text=event_part.text,
                                    raw_representation=event,
                                    additional_properties=streamed_message_properties(event),
                                ),
                            )
                        )
                        metadata.update(self._get_metadata_from_response(event_part))
                    case "refusal":
                        contents.append(
                            track_message_content(
                                event,
                                Content.from_text(
                                    text=event_part.refusal,
                                    raw_representation=event,
                                    additional_properties=streamed_message_properties(event),
                                ),
                            )
                        )
                    case _:
                        pass
            case "response.output_text.delta" | "response.refusal.delta":
                contents.append(
                    track_message_content(
                        event,
                        Content.from_text(
                            text=event.delta,
                            raw_representation=event,
                            additional_properties=streamed_message_properties(event),
                        ),
                    )
                )
                metadata.update(self._get_metadata_from_response(event))
            case "response.reasoning_text.delta":
                if seen_reasoning_delta_item_ids is not None:
                    seen_reasoning_delta_item_ids.add(event.item_id)
                contents.append(
                    Content.from_text_reasoning(
                        id=event.item_id,
                        text=event.delta,
                        raw_representation=event,
                        additional_properties={"reasoning_text": True},
                    )
                )
                metadata.update(self._get_metadata_from_response(event))
            case "response.reasoning_text.done":
                # Done event carries the full accumulated text. Emit it only as a
                # fallback when no delta was already received for this item_id, to
                # avoid duplicating content in downstream accumulators (e.g. ag-ui).
                if seen_reasoning_delta_item_ids is None or event.item_id not in seen_reasoning_delta_item_ids:
                    contents.append(
                        Content.from_text_reasoning(
                            id=event.item_id,
                            text=event.text,
                            raw_representation=event,
                            additional_properties={"reasoning_text": True},
                        )
                    )
                metadata.update(self._get_metadata_from_response(event))
            case "response.reasoning_summary_text.delta":
                if seen_reasoning_delta_item_ids is not None:
                    seen_reasoning_delta_item_ids.add(event.item_id)
                contents.append(
                    Content.from_text_reasoning(
                        id=event.item_id,
                        text=event.delta,
                        raw_representation=event,
                    )
                )
                metadata.update(self._get_metadata_from_response(event))
            case "response.reasoning_summary_text.done":
                # Done event carries the full accumulated text. Emit it only as a
                # fallback when no delta was already received for this item_id, to
                # avoid duplicating content in downstream accumulators (e.g. ag-ui).
                if seen_reasoning_delta_item_ids is None or event.item_id not in seen_reasoning_delta_item_ids:
                    contents.append(
                        Content.from_text_reasoning(
                            id=event.item_id,
                            text=event.text,
                            raw_representation=event,
                        )
                    )
                metadata.update(self._get_metadata_from_response(event))
            case "response.code_interpreter_call_code.delta":
                call_id = getattr(event, "call_id", None) or getattr(event, "id", None) or event.item_id
                ci_additional_properties = {
                    "output_index": event.output_index,
                    "sequence_number": event.sequence_number,
                    "item_id": event.item_id,
                }
                code_call = hosted_call_contents.get(event.output_index)
                if code_call is None:
                    code_call = Content.from_code_interpreter_tool_call(
                        call_id=call_id,
                        inputs=[],
                        hosted_provider=type(self).HOSTED_PROVIDER,
                        provider_item_type="code_interpreter_call",
                        provider_item_id=event.item_id,
                        retry_safety=HostedRetrySafety.SANDBOXED,
                    )
                    hosted_call_contents[event.output_index] = code_call
                    contents.append(code_call)
                if not code_call.inputs:
                    code_call.inputs = [Content.from_text(text="")]
                code_call.inputs[0].text = (code_call.inputs[0].text or "") + event.delta
                code_call.inputs[0].raw_representation = event
                code_call.inputs[0].additional_properties = ci_additional_properties
                code_call.provider_phase = HostedToolPhase.DELTA
                code_call.raw_representation = event
                if code_call not in contents:
                    contents.append(code_call)
                metadata.update(self._get_metadata_from_response(event))
                # NOTE: Unlike reasoning done events, code_interpreter done events always
                # emit content because downstream consumers do not accumulate
                # code_interpreter deltas the same way.
            case "response.code_interpreter_call_code.done":
                call_id = getattr(event, "call_id", None) or getattr(event, "id", None) or event.item_id
                ci_additional_properties = {
                    "output_index": event.output_index,
                    "sequence_number": event.sequence_number,
                    "item_id": event.item_id,
                }
                code_call = hosted_call_contents.get(event.output_index)
                if code_call is None:
                    code_call = Content.from_code_interpreter_tool_call(
                        call_id=call_id,
                        inputs=[],
                        hosted_provider=type(self).HOSTED_PROVIDER,
                        provider_item_type="code_interpreter_call",
                        provider_item_id=event.item_id,
                        retry_safety=HostedRetrySafety.SANDBOXED,
                    )
                    hosted_call_contents[event.output_index] = code_call
                code_call.inputs = [
                    Content.from_text(
                        text=event.code,
                        raw_representation=event,
                        additional_properties=ci_additional_properties,
                    )
                ]
                code_call.provider_phase = HostedToolPhase.SNAPSHOT
                code_call.raw_representation = event
                contents.append(code_call)
                metadata.update(self._get_metadata_from_response(event))
            case "response.created":
                response_id = event.response.id
                conversation_id = self._get_conversation_id(event.response, effective_store_option(options))
                if (
                    type(self).MINTS_CONTINUATION_TOKENS
                    and event.response.status
                    and event.response.status
                    in (
                        "in_progress",
                        "queued",
                    )
                    and effective_store_option(options) is not False
                ):
                    continuation_token = OpenAIContinuationToken(response_id=event.response.id)
            case "response.in_progress":
                response_id = event.response.id
                conversation_id = self._get_conversation_id(event.response, effective_store_option(options))
                if type(self).MINTS_CONTINUATION_TOKENS and effective_store_option(options) is not False:
                    continuation_token = OpenAIContinuationToken(response_id=event.response.id)
            case "response.completed" | "response.incomplete":
                response_id = event.response.id
                conversation_id = self._get_conversation_id(event.response, effective_store_option(options))
                model = event.response.model
                created_at = datetime.fromtimestamp(event.response.created_at, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                if event.response.usage:
                    usage = self._parse_usage_from_openai(event.response.usage)
                    if usage:
                        contents.append(Content.from_usage(usage_details=usage, raw_representation=event))
                # An output-token cutoff (status "incomplete" / reason
                # "max_output_tokens") maps to the canonical "length"
                # finish_reason so downstream truncation handling matches Chat
                # Completions / Anthropic instead of an opaque empty response.
                incomplete_details = getattr(event.response, "incomplete_details", None)
                if (
                    event.response.status == "incomplete"
                    and getattr(incomplete_details, "reason", None) == "max_output_tokens"
                ):
                    finish_reason = "length"
            case "response.output_item.added":
                event_item = event.item
                output_index = getattr(event, "output_index", -1)
                match event_item.type:
                    # types:
                    # ResponseOutputMessage,
                    # ResponseFileSearchToolCall,
                    # ResponseFunctionToolCall,
                    # ResponseFunctionWebSearch,
                    # ResponseComputerToolCall,
                    # ResponseReasoningItem,
                    # ImageGenerationCall,
                    # ResponseCodeInterpreterToolCall,
                    # LocalShellCall,
                    # McpCall,
                    # McpListTools,
                    # McpApprovalRequest,
                    # ResponseCustomToolCall,
                    case "message":
                        envelope = _output_message_envelope(event_item)
                        if envelope:
                            output_message_envelopes[output_index] = envelope
                    case "function_call":
                        function_call_ids[output_index] = (
                            event_item.call_id,
                            event_item.name,
                        )
                    case "mcp_call":
                        call = self._parse_mcp_item_contents(event_item, phase=HostedToolPhase.START)[0]
                        hosted_call_contents[output_index] = call
                        contents.append(call)
                    case "code_interpreter_call":  # ResponseOutputCodeInterpreterCall
                        call = self._parse_code_item_contents(event_item, phase=HostedToolPhase.START)[0]
                        hosted_call_contents[output_index] = call
                        contents.append(call)
                    case "image_generation_call":
                        call = self._parse_image_item_contents(event_item, phase=HostedToolPhase.START)[0]
                        hosted_call_contents[output_index] = call
                        contents.append(call)
                    case "shell_call" | "local_shell_call" | "shell_call_output":
                        if local_shell_tool_name or event_item.type == "shell_call_output":
                            # Preserve the existing client-executed local-shell path;
                            # executable function calls are emitted from the done item.
                            pass
                        else:
                            shell_contents = self._shell_item_to_contents(event_item, local_shell_tool_name)
                            if shell_contents:
                                call = shell_contents[0]
                                call.provider_phase = HostedToolPhase.START
                                hosted_call_contents[output_index] = call
                                contents.append(call)
                    case "reasoning":  # ResponseOutputReasoning
                        reasoning_id = getattr(event_item, "id", None)
                        added_reasoning = False
                        encrypted_content = getattr(event_item, "encrypted_content", None)
                        if hasattr(event_item, "content") and event_item.content:
                            for reasoning_content in event_item.content:
                                contents.append(
                                    Content.from_text_reasoning(
                                        id=reasoning_id or None,
                                        text=reasoning_content.text,
                                        protected_data=encrypted_content,
                                        raw_representation=reasoning_content,
                                        additional_properties={"reasoning_text": True},
                                    )
                                )
                                added_reasoning = True
                        if item_summary := getattr(event_item, "summary", None):
                            for summary in item_summary:
                                contents.append(
                                    Content.from_text_reasoning(
                                        id=reasoning_id or None,
                                        text=summary.text,
                                        protected_data=encrypted_content,
                                        raw_representation=summary,
                                    )
                                )
                                added_reasoning = True
                        if not added_reasoning:
                            # Reasoning item with no visible text (e.g. encrypted reasoning).
                            # Always emit an empty marker so co-occurrence detection can occur.
                            contents.append(
                                Content.from_text_reasoning(
                                    id=reasoning_id or None,
                                    text="",
                                    protected_data=encrypted_content,
                                    raw_representation=event_item,
                                )
                            )
                    case "web_search_call" | "file_search_call":
                        call = self._parse_search_tool_call_content(event_item, phase=HostedToolPhase.START)
                        hosted_call_contents[output_index] = call
                        contents.append(call)
                    case "tool_search_call":
                        call = self._parse_tool_search_call_content(event_item, phase=HostedToolPhase.START)
                        hosted_call_contents[output_index] = call
                        contents.append(call)
                    case "tool_search_output":
                        pass
                    case _:
                        if self._has_provider_execution_evidence(event_item):
                            call = self._parse_generic_hosted_item_contents(
                                event_item,
                                phase=HostedToolPhase.START,
                            )[0]
                            hosted_call_contents[output_index] = call
                            contents.append(call)
                        else:
                            logger.debug(
                                "responses_parser_unrecognized_server_item provider=%s item_type=%s "
                                "execution_evidence=false",
                                type(self).HOSTED_PROVIDER,
                                event_item.type,
                            )
            case (
                "response.web_search_call.in_progress"
                | "response.web_search_call.searching"
                | "response.web_search_call.completed"
                | "response.file_search_call.in_progress"
                | "response.file_search_call.searching"
                | "response.file_search_call.completed"
                | "response.mcp_call.in_progress"
                | "response.mcp_call.completed"
                | "response.mcp_call.failed"
                | "response.code_interpreter_call.in_progress"
                | "response.code_interpreter_call.interpreting"
                | "response.code_interpreter_call.completed"
                | "response.image_generation_call.in_progress"
                | "response.image_generation_call.generating"
                | "response.image_generation_call.completed"
            ):
                output_index = getattr(event, "output_index", -1)
                if call := hosted_call_contents.get(output_index):
                    provider_status = event.type.rsplit(".", 1)[-1]
                    call.status = provider_status
                    call.provider_status = provider_status
                    call.provider_phase = HostedToolPhase.SNAPSHOT
                    call.raw_representation = event
                    contents.append(call)
            case "response.function_call_arguments.delta":
                call_id, name = function_call_ids.get(event.output_index, (None, None))
                if call_id and name:
                    contents.append(
                        Content.from_function_call(
                            call_id=call_id,
                            name=name,
                            arguments=event.delta,
                            additional_properties={
                                "output_index": event.output_index,
                                "fc_id": event.item_id,
                            },
                            raw_representation=event,
                        )
                    )
            case "response.image_generation_call.partial_image":
                image_base64 = event.partial_image_b64
                partial_index = event.partial_image_index
                image_output = Content.from_uri(
                    uri=f"data:{detect_media_type_from_base64(data_str=image_base64) or 'image/png'}"
                    f";base64,{image_base64}",
                    additional_properties={
                        "partial_image_index": partial_index,
                        "is_partial_image": True,
                    },
                    raw_representation=event,
                )

                image_id = getattr(event, "item_id", None)
                output_index = getattr(event, "output_index", -1)
                image_call = hosted_call_contents.get(output_index)
                if image_call is None:
                    image_call = Content.from_image_generation_tool_call(
                        image_id=image_id,
                        hosted_provider=type(self).HOSTED_PROVIDER,
                        provider_item_type="image_generation_call",
                        provider_item_id=image_id,
                        provider_phase=HostedToolPhase.START,
                        retry_safety=HostedRetrySafety.SANDBOXED,
                        raw_representation=event,
                    )
                    hosted_call_contents[output_index] = image_call
                    contents.append(image_call)
                image_result = hosted_result_contents.get(output_index)
                if image_result is None:
                    image_result = Content.from_image_generation_tool_result(
                        image_id=image_id,
                        outputs=[],
                        hosted_provider=type(self).HOSTED_PROVIDER,
                        provider_item_type="image_generation_call",
                        provider_item_id=image_id,
                        provider_phase=HostedToolPhase.SNAPSHOT,
                        provider_status="generating",
                        retry_safety=HostedRetrySafety.SANDBOXED,
                        raw_representation=event,
                    )
                    hosted_result_contents[output_index] = image_result
                if not isinstance(image_result.outputs, list):
                    image_result.outputs = []
                image_result.outputs.append(image_output)
                image_result.provider_phase = HostedToolPhase.SNAPSHOT
                image_result.provider_status = "generating"
                image_result.raw_representation = event
                contents.append(image_result)
            case "response.output_text.annotation.added":
                # Handle streaming text annotations (file citations, file paths, etc.)
                annotation: Any = event.annotation

                def _get_ann_value(key: str) -> Any:
                    """Extract value from annotation (dict or object)."""
                    if isinstance(annotation, dict):
                        return cast("dict[str, Any]", annotation).get(key)
                    return getattr(annotation, key, None)

                ann_type = _get_ann_value("type")
                ann_file_id = _get_ann_value("file_id")
                # Hosted-file citations attach as text annotations (matching the non-streaming path)
                # so they don't roundtrip as standalone `input_file` items in assistant history.
                if ann_type == "file_path":
                    if ann_file_id:
                        annotation_obj = Annotation(
                            type="citation",
                            file_id=str(ann_file_id),
                            additional_properties={
                                "annotation_index": event.annotation_index,
                                "index": _get_ann_value("index"),
                            },
                            raw_representation=annotation,
                        )
                        contents.append(
                            track_message_content(
                                event,
                                Content.from_text(
                                    text="",
                                    annotations=[annotation_obj],
                                    raw_representation=event,
                                    additional_properties=streamed_message_properties(event),
                                ),
                            )
                        )
                elif ann_type == "file_citation":
                    if ann_file_id:
                        ann_filename = _get_ann_value("filename")
                        annotation_obj = Annotation(
                            type="citation",
                            file_id=str(ann_file_id),
                            url=ann_filename,
                            additional_properties={
                                "annotation_index": event.annotation_index,
                                "index": _get_ann_value("index"),
                            },
                            raw_representation=annotation,
                        )
                        contents.append(
                            track_message_content(
                                event,
                                Content.from_text(
                                    text="",
                                    annotations=[annotation_obj],
                                    raw_representation=event,
                                    additional_properties=streamed_message_properties(event),
                                ),
                            )
                        )
                elif ann_type == "container_file_citation":
                    if ann_file_id:
                        ann_filename = _get_ann_value("filename")
                        ann_start = _get_ann_value("start_index")
                        ann_end = _get_ann_value("end_index")
                        annotation_obj = Annotation(
                            type="citation",
                            file_id=str(ann_file_id),
                            url=ann_filename,
                            additional_properties={
                                "annotation_index": event.annotation_index,
                                "container_id": _get_ann_value("container_id"),
                            },
                            raw_representation=annotation,
                        )
                        if ann_start is not None and ann_end is not None:
                            annotation_obj["annotated_regions"] = [
                                TextSpanRegion(
                                    type="text_span",
                                    start_index=ann_start,
                                    end_index=ann_end,
                                )
                            ]
                        contents.append(
                            track_message_content(
                                event,
                                Content.from_text(
                                    text="",
                                    annotations=[annotation_obj],
                                    raw_representation=event,
                                    additional_properties=streamed_message_properties(event),
                                ),
                            )
                        )
                elif ann_type == "url_citation":
                    ann_url = _get_ann_value("url")
                    if ann_url:
                        ann_start = _get_ann_value("start_index")
                        ann_end = _get_ann_value("end_index")
                        annotation_properties: dict[str, Any] = {"annotation_index": event.annotation_index}
                        ann_get_url = _get_ann_value("get_url")
                        if ann_get_url is not None:
                            annotation_properties["get_url"] = ann_get_url
                        annotation_obj = Annotation(
                            type="citation",
                            title=_get_ann_value("title") or "",
                            url=str(ann_url),
                            additional_properties=annotation_properties,
                            raw_representation=annotation,
                        )
                        if ann_start is not None and ann_end is not None:
                            annotation_obj["annotated_regions"] = [
                                TextSpanRegion(
                                    type="text_span",
                                    start_index=ann_start,
                                    end_index=ann_end,
                                )
                            ]
                        contents.append(
                            track_message_content(
                                event,
                                Content.from_text(
                                    text="",
                                    annotations=[annotation_obj],
                                    raw_representation=event,
                                    additional_properties=streamed_message_properties(event),
                                ),
                            )
                        )
                else:
                    logger.debug("Unparsed annotation type in streaming: %s", ann_type)
            case "response.output_item.done":
                done_item = event.item
                output_index = getattr(event, "output_index", -1)
                if getattr(done_item, "type", None) == "message":
                    envelope = _output_message_envelope(done_item)
                    if envelope:
                        output_message_envelopes[output_index] = envelope
                    for content in output_message_contents.get(output_index, []):
                        if envelope:
                            content.additional_properties[OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY] = dict(envelope)
                        else:
                            content.additional_properties.pop(OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY, None)
                elif getattr(done_item, "type", None) == "reasoning":
                    encrypted_content = getattr(done_item, "encrypted_content", None)
                    if encrypted_content:
                        contents.append(
                            Content.from_text_reasoning(
                                id=getattr(done_item, "id", None),
                                text="",
                                protected_data=encrypted_content,
                                raw_representation=done_item,
                            )
                        )
                elif getattr(done_item, "type", None) == "mcp_call":
                    snapshots = self._parse_mcp_item_contents(done_item)
                    contents.append(
                        self._merge_streamed_hosted_content(
                            hosted_call_contents,
                            output_index,
                            snapshots[0],
                        )
                    )
                    if len(snapshots) > 1:
                        contents.append(
                            self._merge_streamed_hosted_content(
                                hosted_result_contents,
                                output_index,
                                snapshots[1],
                            )
                        )
                elif getattr(done_item, "type", None) in ("web_search_call", "file_search_call"):
                    contents.append(
                        self._merge_streamed_hosted_content(
                            hosted_call_contents,
                            output_index,
                            self._parse_search_tool_call_content(done_item),
                        )
                    )
                    contents.append(
                        self._merge_streamed_hosted_content(
                            hosted_result_contents,
                            output_index,
                            self._parse_search_tool_result_content(done_item),
                        )
                    )
                elif getattr(done_item, "type", None) == "code_interpreter_call":
                    snapshots = self._parse_code_item_contents(done_item)
                    contents.append(
                        self._merge_streamed_hosted_content(
                            hosted_call_contents,
                            output_index,
                            snapshots[0],
                        )
                    )
                    if len(snapshots) > 1:
                        contents.append(
                            self._merge_streamed_hosted_content(
                                hosted_result_contents,
                                output_index,
                                snapshots[1],
                            )
                        )
                elif getattr(done_item, "type", None) == "image_generation_call":
                    snapshots = self._parse_image_item_contents(done_item)
                    contents.append(
                        self._merge_streamed_hosted_content(
                            hosted_call_contents,
                            output_index,
                            snapshots[0],
                        )
                    )
                    if len(snapshots) > 1:
                        contents.append(
                            self._merge_streamed_hosted_content(
                                hosted_result_contents,
                                output_index,
                                snapshots[1],
                            )
                        )
                    elif image_result := hosted_result_contents.get(output_index):
                        status = getattr(done_item, "status", None)
                        image_result.status = status
                        image_result.provider_status = status
                        image_result.provider_phase = HostedToolPhase.TERMINAL
                        image_result.additional_properties = self._hosted_item_properties(
                            done_item,
                            replay_shadow=True,
                        )
                        image_result.raw_representation = done_item
                        contents.append(image_result)
                elif getattr(done_item, "type", None) in ("shell_call", "local_shell_call", "shell_call_output"):
                    shell_contents = self._shell_item_to_contents(done_item, local_shell_tool_name)
                    if local_shell_tool_name:
                        contents.extend(shell_contents)
                    elif shell_contents:
                        store = (
                            hosted_result_contents
                            if shell_contents[0].type == "shell_tool_result"
                            else hosted_call_contents
                        )
                        contents.append(
                            self._merge_streamed_hosted_content(
                                store,
                                output_index,
                                shell_contents[0],
                            )
                        )
                elif getattr(done_item, "type", None) == "custom_tool_call":
                    contents.append(
                        self._parse_hosted_function_call_content(
                            done_item,
                            name=getattr(done_item, "name", "") or "",
                            arguments=getattr(done_item, "input", None),
                        )
                    )
                elif getattr(done_item, "type", None) == "apply_patch_call":
                    contents.append(
                        self._parse_hosted_function_call_content(
                            done_item,
                            name="apply_patch",
                            arguments=getattr(done_item, "operation", None),
                        )
                    )
                elif getattr(done_item, "type", None) == "tool_search_call":
                    contents.append(
                        self._merge_streamed_hosted_content(
                            hosted_call_contents,
                            output_index,
                            self._parse_tool_search_call_content(done_item),
                        )
                    )
                elif getattr(done_item, "type", None) == "tool_search_output":
                    contents.append(
                        self._merge_streamed_hosted_content(
                            hosted_result_contents,
                            output_index,
                            self._parse_tool_search_result_content(done_item),
                        )
                    )
                elif self._has_provider_execution_evidence(done_item):
                    snapshots = self._parse_generic_hosted_item_contents(done_item)
                    contents.append(
                        self._merge_streamed_hosted_content(
                            hosted_call_contents,
                            output_index,
                            snapshots[0],
                        )
                    )
                    if len(snapshots) > 1:
                        contents.append(
                            self._merge_streamed_hosted_content(
                                hosted_result_contents,
                                output_index,
                                snapshots[1],
                            )
                        )
                else:
                    logger.debug(
                        "responses_parser_unrecognized_server_item provider=%s item_type=%s execution_evidence=false",
                        type(self).HOSTED_PROVIDER,
                        getattr(done_item, "type", None),
                    )
            case _:
                logger.debug("Unparsed event of type: %s: %s", event.type, event)

        return ChatResponseUpdate(
            contents=contents,
            conversation_id=conversation_id,
            response_id=response_id,
            role="assistant",
            model=model,
            created_at=created_at,
            continuation_token=continuation_token,
            finish_reason=finish_reason,
            additional_properties=metadata,
            raw_representation=event,
        )

    def _parse_usage_from_openai(self, usage: ResponseUsage) -> UsageDetails | None:
        details = UsageDetails(
            input_token_count=usage.input_tokens,
            output_token_count=usage.output_tokens,
            total_token_count=usage.total_tokens,
        )
        if usage.input_tokens_details:
            cached_tokens = getattr(usage.input_tokens_details, "cached_tokens", None)
            if cached_tokens is not None:
                details["openai.cached_input_tokens"] = cached_tokens  # type: ignore[typeddict-unknown-key]
                details["cache_read_input_token_count"] = cached_tokens
            # Untyped in the SDK (extra='allow' retains it): reported for
            # billed explicit prompt caching. Explicit 0 is preserved.
            cache_write_tokens = getattr(usage.input_tokens_details, "cache_write_tokens", None)
            if cache_write_tokens is not None:
                details["openai.cache_write_tokens"] = cache_write_tokens  # type: ignore[typeddict-unknown-key]
                details["cache_creation_input_token_count"] = cache_write_tokens
        if usage.output_tokens_details:
            reasoning_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", None)
            if reasoning_tokens is not None:
                details["openai.reasoning_tokens"] = reasoning_tokens  # type: ignore[typeddict-unknown-key]
                details["reasoning_output_token_count"] = reasoning_tokens
        return details

    def _get_metadata_from_response(self, output: Any) -> dict[str, Any]:
        """Get metadata from a chat choice."""
        if logprobs := getattr(output, "logprobs", None):
            return {
                "logprobs": logprobs,
            }
        return {}
