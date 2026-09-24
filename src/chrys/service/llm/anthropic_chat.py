# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chrys-owned Anthropic raw wire client.

Parser and serializer logic, with provider
settings removed: callers must inject a pre-configured ``AsyncAnthropic``-style client.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, ClassVar, Final, Generic, Literal, TypeVar, override

from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicFoundry, AsyncAnthropicVertex
from anthropic.types.beta import (
    BetaContentBlock,
    BetaMessage,
    BetaMessageDeltaUsage,
    BetaRawContentBlockDelta,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawMessageStreamEvent,
    BetaTextBlock,
    BetaUsage,
)
from anthropic.types.beta.beta_bash_code_execution_tool_result_error import BetaBashCodeExecutionToolResultError
from anthropic.types.beta.beta_code_execution_result_block import BetaCodeExecutionResultBlock
from anthropic.types.beta.beta_code_execution_tool_result_error import BetaCodeExecutionToolResultError
from anthropic.types.beta.beta_encrypted_code_execution_result_block import BetaEncryptedCodeExecutionResultBlock
from pydantic import BaseModel
from typing_extensions import TypedDict

from chrys.foundation.hosted_tools import (
    ANTHROPIC_HOSTED_WIRE_BLOCK_KEY,
    HostedRetrySafety,
    HostedToolFamily,
    HostedToolPhase,
)
from chrys.foundation.tool_kinds import KIND_SHELL as SHELL_TOOL_KIND_VALUE
from chrys.kernel import (
    Annotation,
    BaseChatClient,
    ChatClientInvalidRequestException,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FinishReason,
    FinishReasonLiteral,
    FunctionTool,
    Message,
    ResponseStream,
    TextSpanRegion,
    UsageDetails,
    normalize_tools,
    prepend_instructions_to_messages,
    tool,
    validate_tool_mode,
)
from chrys.kernel._content import _ANTHROPIC_REDACTED_THINKING_KEY, _get_data_bytes_as_str
from chrys.kernel.compaction import CompactionStrategy, TokenizerProtocol
from chrys.service.agent_middleware.events.hosted_tools import cross_provider_hosted_degradations
from chrys.service.llm.defaults import apply_default_max_tokens

logger = logging.getLogger(__name__)

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel | None, default=None)
AnthropicAsyncClient = AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicFoundry | AsyncAnthropicVertex

BETA_FLAGS: Final[list[str]] = ["mcp-client-2025-04-04", "code-execution-2025-08-25"]
_HOSTED_CONTEXT_ROLE_KEY = "_chrys_hosted_context"


# region Anthropic Chat Options TypedDict


class ThinkingConfig(TypedDict, total=False):
    """Configuration for enabling Claude's extended thinking.

    When enabled, responses include ``thinking`` content blocks showing Claude's
    thinking process before the final answer. Requires a minimum budget of 1,024
    tokens and counts towards your ``max_tokens`` limit.

    See https://docs.claude.com/en/docs/build-with-claude/extended-thinking for details.

    Keys:
        type: "enabled" to enable extended thinking, "disabled" to disable.
        budget_tokens: The token budget for thinking (minimum 1024, required when type="enabled").
    """

    type: Literal["enabled", "disabled"]
    budget_tokens: int


class AnthropicChatOptions(ChatOptions[ResponseModelT], Generic[ResponseModelT], total=False):
    """Anthropic-specific chat options.

    Extends ChatOptions with options specific to Anthropic's Messages API.
    Options that Anthropic doesn't support are typed as None to indicate they're unavailable.

    Note:
        Anthropic requires max_tokens to be specified. If not provided,
        a default output cap will be used.

    Keys:
        temperature: Sampling temperature between 0 and 1.
        top_p: Nucleus sampling parameter.
        max_tokens: Maximum number of tokens to generate (REQUIRED).
        stop: Stop sequences,
            translates to ``stop_sequences`` in Anthropic API.
        tools: List of tools (functions) available to the model.
        tool_choice: How the model should use tools.
        response_format: Structured output schema.
        output_config: Anthropic output configuration. When ``response_format``
            is also supplied, fields such as ``effort`` are preserved but an
            explicit ``format`` conflicts and is rejected.
        metadata: Request metadata with user_id for tracking.
        user: User identifier, translates to ``metadata.user_id`` in Anthropic API.
        instructions: System instructions for the model,
            translates to ``system`` in Anthropic API.
        top_k: Number of top tokens to consider for sampling.
        service_tier: Service tier ("auto" or "standard_only").
        thinking: Extended thinking configuration for Claude models.
            When enabled, responses include ``thinking`` content blocks showing Claude's
            thinking process before the final answer. Requires a minimum budget of 1,024
            tokens and counts towards your ``max_tokens`` limit.
            See https://docs.claude.com/en/docs/build-with-claude/extended-thinking for details.
        container: Container configuration for skills.
        additional_beta_flags: Additional beta flags to enable on the request.
    """

    # Anthropic-specific generation parameters (supported by all models)
    top_k: int
    service_tier: Literal["auto", "standard_only"]

    # Extended thinking (Claude models)
    thinking: ThinkingConfig

    # GA output configuration (structured format is normally derived from response_format)
    output_config: dict[str, Any]

    # Skills
    container: dict[str, Any]

    # Beta features
    additional_beta_flags: list[str]

    # Unsupported base options (override with None to indicate not supported)
    logit_bias: None
    seed: None
    frequency_penalty: None
    presence_penalty: None
    store: None
    conversation_id: None


# Translation between framework options keys and Anthropic Messages API
OPTION_TRANSLATIONS: dict[str, str] = {
    "stop": "stop_sequences",
    "instructions": "system",
}


def _apply_option_translations(options: dict[str, Any]) -> None:
    """Translate framework option keys to Anthropic request keys in-place.

    When both the old and new key are present, the new key wins and the old key
    is discarded to preserve explicit overrides.
    """
    for old_key, new_key in OPTION_TRANSLATIONS.items():
        if old_key not in options or old_key == new_key:
            continue
        old_value = options.pop(old_key)
        options.setdefault(new_key, old_value)


# region Role and Finish Reason Maps


ROLE_MAP: dict[str, str] = {
    "user": "user",
    "assistant": "assistant",
    "system": "user",
    "tool": "user",
}

FINISH_REASON_MAP: dict[str, FinishReasonLiteral] = {
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "end_turn": "stop",
    "refusal": "content_filter",
    "pause_turn": "stop",
}


def _map_finish_reason(stop_reason: str | None) -> FinishReason | None:
    """Keep provider stop reasons available even before Chrys maps them."""
    return FinishReason(FINISH_REASON_MAP.get(stop_reason, stop_reason)) if stop_reason else None


@dataclass
class _PendingFunctionCall:
    """A streamed local tool call being assembled by Anthropic content-block index."""

    call_id: str
    name: str
    initial_arguments: str | Mapping[str, Any] | None
    raw_representations: list[Any]
    argument_deltas: list[str]

    def append_arguments(self, partial_json: str, raw_representation: Any) -> None:
        """Append one indexed input delta without exposing an orphan function call."""
        self.argument_deltas.append(partial_json)
        self.raw_representations.append(raw_representation)

    def to_content(self) -> Content:
        """Build the single complete function-call content emitted for this block."""
        arguments: str | Mapping[str, Any] | None = (
            "".join(self.argument_deltas) if self.argument_deltas else self.initial_arguments
        )
        raw_representation: Any = self.raw_representations
        if len(self.raw_representations) == 1:
            raw_representation = self.raw_representations[0]
        return Content.from_function_call(
            call_id=self.call_id,
            name=self.name,
            arguments=arguments,
            raw_representation=raw_representation,
        )


@dataclass
class _AnthropicStreamState:
    """Per-response state for indexed Anthropic content blocks."""

    pending_function_calls: dict[int, _PendingFunctionCall]
    hosted_tool_indices: set[int]
    hosted_tool_calls: dict[int, Content]
    hosted_argument_deltas: dict[int, list[str]]
    deferred_updates: dict[int, list[ChatResponseUpdate]]
    defer_from_index: int | None
    initial_cache_read_input_tokens: int | None = None


def _drain_deferred_content_update(stream_state: _AnthropicStreamState) -> ChatResponseUpdate | None:
    """Remove deferred content as one ordered tool-response-boundary update."""
    contents: list[Content] = []
    raw_representations: list[Any] = []
    indices = sorted(stream_state.pending_function_calls.keys() | stream_state.deferred_updates.keys())
    for index in indices:
        if function_call := stream_state.pending_function_calls.pop(index, None):
            contents.append(function_call.to_content())
        for update in stream_state.deferred_updates.pop(index, []):
            contents.extend(update.contents)
            if update.raw_representation is not None:
                raw_representations.append(update.raw_representation)
    stream_state.defer_from_index = None
    if not contents:
        return None
    return ChatResponseUpdate(
        contents=contents,
        raw_representation=raw_representations or None,
    )


class RawAnthropicClient(BaseChatClient):
    """Raw Anthropic chat client without middleware, telemetry, or function invocation support.

    Warning:
        **This class should not normally be used directly.** It does not include middleware,
        telemetry, or function invocation support that you most likely need. If you do use it,
        you should consider which additional layers to apply. There is a defined ordering that
        you should follow:

        1. **ToolLoopLayer** - Owns the tool/function calling loop and routes function middleware
        2. **ChatMiddlewareLayer** - Applies chat middleware per model call and stays outside telemetry
        3. **ChatTelemetryLayer** - Must stay inside chat middleware for correct per-call telemetry

        Use ``create_instrumented_anthropic_client`` for the production stack with all layers applied.
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "anthropic"  # type: ignore[reportIncompatibleVariableOverride, misc]

    def __init__(
        self,
        model: str | None = None,
        *,
        anthropic_client: AnthropicAsyncClient | None = None,
        additional_beta_flags: list[str] | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        additional_properties: dict[str, Any] | None = None,
    ) -> None:
        if anthropic_client is None:
            raise ValueError("RawAnthropicClient requires a pre-configured anthropic_client.")
        super().__init__(
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
            additional_properties=additional_properties,
        )
        self.anthropic_client = anthropic_client
        self.additional_beta_flags = additional_beta_flags or []
        self.model = model or ""
        self._tool_name_aliases: dict[str, str] = {}

    # region Static factory methods for hosted tools

    @staticmethod
    def get_code_interpreter_tool(
        *,
        type_name: str | None = None,
        name: str = "code_execution",
    ) -> dict[str, Any]:
        """Create a code interpreter tool configuration for Anthropic.

        Keyword Args:
            type_name: Override the tool type name. Defaults to "code_execution_20250825".
            name: The name for this tool. Defaults to "code_execution".

        Returns:
            A dict-based tool configuration ready to pass to ChatAgent.

        Examples:
            .. code-block:: python

                from chrys.service.llm.anthropic_chat import RawAnthropicClient

                tool = RawAnthropicClient.get_code_interpreter_tool()
                tools = [tool]
        """
        return {"type": type_name or "code_execution_20250825", "name": name}

    @staticmethod
    def get_web_search_tool(
        *,
        type_name: str | None = None,
        name: str = "web_search",
    ) -> dict[str, Any]:
        """Create a web search tool configuration for Anthropic.

        Keyword Args:
            type_name: Override the tool type name. Defaults to "web_search_20250305".
            name: The name for this tool. Defaults to "web_search".

        Returns:
            A dict-based tool configuration ready to pass to ChatAgent.

        Examples:
            .. code-block:: python

                from chrys.service.llm.anthropic_chat import RawAnthropicClient

                tool = RawAnthropicClient.get_web_search_tool()
                tools = [tool]
        """
        return {"type": type_name or "web_search_20250305", "name": name}

    @staticmethod
    def get_shell_tool(
        *,
        func: Callable[..., Any] | FunctionTool,
        description: str | None = None,
        type_name: str | None = None,
    ) -> FunctionTool:
        """Create a local shell FunctionTool for Anthropic.

        This helper wraps ``func`` as a shell-enabled ``FunctionTool`` for local
        execution and configures Anthropic API declaration details via metadata.

        Anthropic always exposes this tool to the model as ``name="bash"`` and
        executes it using a ``bash_*`` tool type.

        Keyword Args:
            func: Python callable or ``FunctionTool`` that executes the requested shell command.
            description: Optional tool description shown to the model.
            type_name: Optional Anthropic shell tool type override.
                Defaults to ``"bash_20250124"`` when omitted.

        Returns:
            A shell-enabled ``FunctionTool`` suitable for ``ChatOptions.tools``.
        """
        base_tool: FunctionTool
        if isinstance(func, FunctionTool):
            base_tool = func
            if description is not None:
                base_tool.description = description
        else:
            base_tool = tool(
                func=func,
                description=description,
            )

        additional_properties: dict[str, Any] = dict(base_tool.additional_properties or {})
        if type_name:
            additional_properties["type"] = type_name

        if base_tool.func is None:
            raise ValueError("Shell tool requires an executable function.")

        base_tool.additional_properties = additional_properties
        base_tool.kind = SHELL_TOOL_KIND_VALUE
        return base_tool

    @staticmethod
    def get_mcp_tool(
        *,
        name: str,
        url: str,
        allowed_tools: list[str] | None = None,
        authorization_token: str | None = None,
    ) -> dict[str, Any]:
        """Create a hosted MCP tool configuration for Anthropic.

        This configures an MCP (Model Context Protocol) server that will be called
        by Anthropic's service. The tools from this MCP server are executed remotely
        by Anthropic, not locally by your application.

        Note:
            For local MCP execution where your application calls the MCP server
            directly, use the MCP client tools instead of this method.

        Keyword Args:
            name: A label/name for the MCP server.
            url: The URL of the MCP server.
            allowed_tools: List of tool names that are allowed to be used from this MCP server.
            authorization_token: Authorization token for the MCP server (e.g., Bearer token).

        Returns:
            A dict-based tool configuration ready to pass to ChatAgent.

        Examples:
            .. code-block:: python

                from chrys.service.llm.anthropic_chat import RawAnthropicClient

                tool = RawAnthropicClient.get_mcp_tool(
                    name="GitHub",
                    url="https://api.githubcopilot.com/mcp/",
                    authorization_token="Bearer ghp_xxx",
                )
                tools = [tool]
        """
        result: dict[str, Any] = {
            "type": "mcp",
            "server_label": name.replace(" ", "_"),
            "server_url": url,
        }

        if allowed_tools:
            result["allowed_tools"] = allowed_tools

        if authorization_token:
            result["headers"] = {"authorization": authorization_token}

        return result

    # endregion

    # region Get response methods

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
        run_options = self._prepare_options(messages, options, **kwargs)

        if stream:
            # Streaming mode
            async def _stream() -> AsyncIterable[ChatResponseUpdate]:
                stream_state = _AnthropicStreamState(
                    pending_function_calls={},
                    hosted_tool_indices=set(),
                    hosted_tool_calls={},
                    hosted_argument_deltas={},
                    deferred_updates={},
                    defer_from_index=None,
                )
                sdk_stream: Any = None
                try:
                    sdk_stream = await self.anthropic_client.beta.messages.create(**run_options, stream=True)  # type: ignore[misc]
                    async for chunk in sdk_stream:
                        if (
                            chunk.type in ("message_delta", "message_stop")
                            and (stream_state.pending_function_calls or stream_state.deferred_updates)
                            and (deferred_update := _drain_deferred_content_update(stream_state))
                        ):
                            # A finish reason is terminal to some stream consumers. Emit complete calls first,
                            # together with any later blocks held behind them in canonical block order.
                            yield deferred_update
                        parsed_chunk = self._process_stream_event(chunk, stream_state)
                        if parsed_chunk:
                            if not parsed_chunk.contents:
                                # Raw-only heartbeats keep the outer stream-idle watchdog aligned with
                                # healthy provider traffic while content remains buffered by block index.
                                yield parsed_chunk
                            elif (
                                isinstance(chunk, BetaRawContentBlockStartEvent | BetaRawContentBlockDeltaEvent)
                                and stream_state.defer_from_index is not None
                                and chunk.index >= stream_state.defer_from_index
                            ):
                                stream_state.deferred_updates.setdefault(chunk.index, []).append(parsed_chunk)
                            else:
                                yield parsed_chunk
                    if (stream_state.pending_function_calls or stream_state.deferred_updates) and (
                        deferred_update := _drain_deferred_content_update(stream_state)
                    ):
                        yield deferred_update
                finally:
                    if sdk_stream is not None:
                        try:
                            close = getattr(sdk_stream, "close", None) or getattr(sdk_stream, "aclose", None)
                            if close is not None:
                                close_result = close()
                                if isawaitable(close_result):
                                    await close_result
                        except Exception:
                            logger.debug("Failed to close Anthropic message stream", exc_info=True)

            return self._build_response_stream(_stream(), response_format=options.get("response_format"))

        # Non-streaming mode
        async def _get_response() -> ChatResponse:
            message = await self.anthropic_client.beta.messages.create(**run_options, stream=False)  # type: ignore[misc]
            return self._process_message(message, options)

        return _get_response()

    # region Prep methods

    def _prepare_options(
        self,
        messages: Sequence[Message],
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create run options for the Anthropic client based on messages and options.

        Args:
            messages: The list of chat messages.
            options: The options dict.
            kwargs: Additional keyword arguments.

        Returns:
            A dictionary of run options for the Anthropic client.
        """
        # Prepend instructions from options if they exist
        instructions = options.get("instructions")
        if instructions:
            messages = prepend_instructions_to_messages(list(messages), instructions, role="system")

        # Start with a copy of options, excluding keys we handle separately
        run_options: dict[str, Any] = {
            k: v
            for k, v in options.items()
            if v is not None and k not in {"instructions", "response_format", "additional_beta_flags"}
        }
        # Framework-level options handled elsewhere; do not forward as raw Anthropic request kwargs.
        run_options.pop("allow_multiple_tool_calls", None)
        # Stream mode is controlled explicitly at call sites.
        run_options.pop("stream", None)

        _apply_option_translations(run_options)

        # Filter out framework kwargs that should not be passed to the Anthropic API.
        # This includes underscore-prefixed internal objects (like _function_middleware_pipeline)
        # and framework kwargs like 'thread' and 'middleware'.
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if not k.startswith("_") and k not in {"thread", "middleware", "additional_beta_flags"}
        }
        _apply_option_translations(filtered_kwargs)
        run_options.update(filtered_kwargs)

        # model
        if not run_options.get("model"):
            if not self.model:
                raise ValueError("model must be a non-empty string")
            run_options["model"] = self.model

        # max_tokens - Anthropic requires this, default if not provided
        apply_default_max_tokens(run_options)

        # messages
        run_options["messages"] = self._prepare_messages_for_anthropic(messages)

        # system message - first system message is passed as instructions
        if messages and isinstance(messages[0], Message) and messages[0].role == "system":
            run_options["system"] = messages[0].text

        # betas
        run_options["betas"] = self._prepare_betas(options)

        # extra headers
        run_options.setdefault("extra_headers", {})

        # Handle user option -> metadata.user_id (Anthropic uses metadata.user_id instead of user)
        if user := run_options.pop("user", None):
            metadata = run_options.get("metadata", {})
            if "user_id" not in metadata:
                metadata["user_id"] = user
            run_options["metadata"] = metadata

        # tools, mcp servers and tool choice
        if tools_config := self._prepare_tools_for_anthropic(options):
            run_options.update(tools_config)

        # response_format - emit Anthropic's GA output_config.format shape.
        response_format = options.get("response_format")
        if response_format is not None:
            existing_output_config = run_options.get("output_config")
            if existing_output_config is None:
                output_config: dict[str, Any] = {}
            elif isinstance(existing_output_config, Mapping):
                output_config = dict(existing_output_config)
            else:
                raise ChatClientInvalidRequestException("output_config must be a mapping.")
            if output_config.get("format") is not None:
                raise ChatClientInvalidRequestException(
                    "response_format cannot be combined with explicit output_config.format."
                )
            output_config["format"] = self._prepare_response_format(response_format)
            run_options["output_config"] = output_config

        return run_options

    def _prepare_betas(self, options: Mapping[str, Any]) -> set[str]:
        """Prepare the beta flags for the Anthropic API request.

        Args:
            options: The options dict that may contain additional beta flags.

        Returns:
            A set of beta flag strings to include in the request.
        """
        return {
            *BETA_FLAGS,
            *self.additional_beta_flags,
            *options.get("additional_beta_flags", []),
        }

    def _prepare_response_format(self, response_format: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
        """Build the ``output_config.format`` payload for structured output.

        Args:
            response_format: Either a Pydantic model class or a dict with the schema specification.
                If a dict, it can be in OpenAI-style format with "json_schema" key,
                or direct format with "schema" key, or the raw schema dict itself.

        Returns:
            The JSON-schema format value nested under Anthropic's GA ``output_config``.
        """
        if isinstance(response_format, dict):
            if "json_schema" in response_format:
                schema = response_format["json_schema"].get("schema", {})
            elif "schema" in response_format:
                schema = response_format["schema"]
            else:
                schema = response_format

            if isinstance(schema, dict):
                schema = {**schema, "additionalProperties": False}

            return {
                "type": "json_schema",
                "schema": schema,
            }

        schema = response_format.model_json_schema()
        schema["additionalProperties"] = False

        return {
            "type": "json_schema",
            "schema": schema,
        }

    def _prepare_messages_for_anthropic(self, messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Prepare a list of ChatMessages for the Anthropic client.

        This skips the first message if it is a system message,
        as Anthropic expects system instructions as a separate parameter.
        """
        # first system message is passed as instructions
        source_messages = messages[1:] if messages and messages[0].role == "system" else messages
        hosted_degradations = cross_provider_hosted_degradations(source_messages, target_provider="anthropic")
        prepared: list[dict[str, Any]] = []
        for message in source_messages:
            prepared.extend(
                self._prepare_message_groups_for_anthropic(
                    message,
                    hosted_degradations=hosted_degradations,
                )
            )
        return prepared

    def _prepare_message_groups_for_anthropic(
        self,
        message: Message,
        *,
        hosted_degradations: Mapping[int, str | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Split mixed content into the message roles Anthropic requires."""
        prepared_message = self._prepare_message_for_anthropic(
            message,
            hosted_degradations=hosted_degradations,
        )
        content = prepared_message.get("content")
        if not isinstance(content, list):
            return [prepared_message]
        if not content and hosted_degradations and all(id(item) in hosted_degradations for item in message.contents):
            return []

        default_role = str(prepared_message.get("role", "user"))
        groups: list[dict[str, Any]] = []
        current_role: str | None = None
        current_content: list[dict[str, Any]] = []

        for content_block in content:
            block_role = self._role_for_anthropic_content_block(content_block, default_role)
            if current_content and current_role != block_role:
                groups.append({"role": current_role, "content": current_content})
                current_content = []
            current_role = block_role
            clean_block = dict(content_block)
            clean_block.pop(_HOSTED_CONTEXT_ROLE_KEY, None)
            current_content.append(clean_block)

        if current_content:
            groups.append({"role": current_role or default_role, "content": current_content})
        return groups or [prepared_message]

    @staticmethod
    def _role_for_anthropic_content_block(content_block: Mapping[str, Any], default_role: str) -> str:
        """Return the Anthropic message role required by a content block."""
        if content_block.get(_HOSTED_CONTEXT_ROLE_KEY) is True:
            return "assistant"
        match content_block.get("type"):
            case "tool_use" | "mcp_tool_use" | "server_tool_use":
                return "assistant"
            case "tool_result":
                return "user"
            case _:
                return default_role

    def _prepare_message_for_anthropic(
        self,
        message: Message,
        *,
        hosted_degradations: Mapping[int, str | None] | None = None,
    ) -> dict[str, Any]:
        """Prepare a Message for the Anthropic client.

        Args:
            message: The Message to convert.

        Returns:
            A dictionary representing the message in Anthropic format.
        """
        if hosted_degradations is None:
            hosted_degradations = cross_provider_hosted_degradations([message], target_provider="anthropic")
        a_content: list[dict[str, Any]] = []
        for content in message.contents:
            if id(content) in hosted_degradations:
                summary = hosted_degradations[id(content)]
                if summary:
                    a_content.append({"type": "text", "text": summary, _HOSTED_CONTEXT_ROLE_KEY: True})
                continue
            replay_block = content.additional_properties.get(ANTHROPIC_HOSTED_WIRE_BLOCK_KEY)
            if content.hosted_provider == "anthropic" and isinstance(replay_block, Mapping):
                a_content.append(dict(replay_block))
                continue
            match content.type:
                case "text":
                    # Skip empty text content blocks - Anthropic API rejects them
                    if content.text:
                        a_content.append({"type": "text", "text": content.text})
                case "data":
                    if content.has_top_level_media_type("image"):
                        a_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "data": _get_data_bytes_as_str(content),  # type: ignore[attr-defined]
                                    "media_type": content.media_type,
                                    "type": "base64",
                                },
                            }
                        )
                    else:
                        logger.debug(f"Ignoring unsupported data content media type: {content.media_type} for now")
                case "uri":
                    if content.has_top_level_media_type("image"):
                        a_content.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": content.uri},
                            }
                        )
                    else:
                        logger.debug(f"Ignoring unsupported data content media type: {content.media_type} for now")
                case "function_call":
                    a_content.append(
                        {
                            "type": "tool_use",
                            "id": content.call_id,
                            "name": content.name,
                            "input": content.parse_arguments(),
                        }
                    )
                case "function_result":
                    if content.items:
                        tool_content: list[dict[str, Any]] = []
                        for item in content.items:
                            if item.type == "text":
                                tool_content.append({"type": "text", "text": item.text or ""})
                            elif item.type == "data" and item.has_top_level_media_type("image"):
                                tool_content.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "data": _get_data_bytes_as_str(item),  # type: ignore[attr-defined]
                                            "media_type": item.media_type,
                                            "type": "base64",
                                        },
                                    }
                                )
                            elif item.type == "uri" and item.has_top_level_media_type("image"):
                                tool_content.append(
                                    {
                                        "type": "image",
                                        "source": {"type": "url", "url": item.uri},
                                    }
                                )
                            else:
                                logger.debug(
                                    "Ignoring unsupported rich content media type in tool result: %s",
                                    item.media_type,
                                )
                        tool_result_content = tool_content or (content.result if content.result is not None else "")
                        a_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content.call_id,
                                "content": tool_result_content,
                                "is_error": content.exception is not None,
                            }
                        )
                    else:
                        a_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content.call_id,
                                "content": content.result if content.result is not None else "",
                                "is_error": content.exception is not None,
                            }
                        )
                case "mcp_server_tool_call":
                    mcp_call: dict[str, Any] = {
                        "type": "mcp_tool_use",
                        "id": content.call_id,
                        "name": content.tool_name,
                        "server_name": content.server_name or "",
                        "input": content.parse_arguments() or {},
                    }
                    a_content.append(mcp_call)
                case "mcp_server_tool_result":
                    mcp_result: dict[str, Any] = {
                        "type": "mcp_tool_result",
                        "tool_use_id": content.call_id,
                        "content": content.output if content.output is not None else "",
                    }
                    a_content.append(mcp_result)
                case "text_reasoning":
                    if content.additional_properties.get(_ANTHROPIC_REDACTED_THINKING_KEY):
                        a_content.append({"type": "redacted_thinking", "data": content.protected_data})
                        continue
                    if (
                        content.id
                        or content.additional_properties.get("reasoning_text")
                        or content.additional_properties.get("openai_reasoning_format")
                    ):
                        # Reasoning owned by another provider's serializer
                        # (Responses reasoning is id-keyed and marker-stamped;
                        # chat-completions dialects stamp their wire format);
                        # replaying it here would forge a thinking signature.
                        continue
                    if content.text is None:
                        if (
                            content.protected_data
                            and a_content
                            and a_content[-1].get("type") == "thinking"
                            and "signature" not in a_content[-1]
                        ):
                            a_content[-1]["signature"] = content.protected_data
                        continue
                    thinking_block: dict[str, Any] = {"type": "thinking", "thinking": content.text}
                    if content.protected_data:
                        thinking_block["signature"] = content.protected_data
                    a_content.append(thinking_block)
                case _:
                    logger.debug(f"Ignoring unsupported content type: {content.type} for now")

        return {
            "role": ROLE_MAP.get(message.role, "user"),
            "content": a_content,
        }

    def _prepare_tools_for_anthropic(self, options: Mapping[str, Any]) -> dict[str, Any] | None:
        """Prepare tools and tool choice configuration for the Anthropic API request.

        Converts FunctionTool to Anthropic format. MCP tools are routed to separate
        mcp_servers parameter. All other tools pass through unchanged.

        Args:
            options: The options dict containing tools and tool choice settings.

        Returns:
            A dictionary with tools, mcp_servers, and tool_choice configuration, or None if empty.
        """
        result: dict[str, Any] = {}
        tools = options.get("tools")

        # Process tools
        if tools:
            tool_list: list[Any] = []
            mcp_server_list: list[Any] = []
            tool_name_aliases: dict[str, str] = {}
            for tool in normalize_tools(tools):
                if isinstance(tool, FunctionTool) and tool.kind == SHELL_TOOL_KIND_VALUE:
                    api_type = (tool.additional_properties or {}).get("type", "bash_20250124")
                    tool_name_aliases["bash"] = tool.name
                    tool_list.append(
                        {
                            "type": api_type,
                            "name": "bash",
                        }
                    )
                elif isinstance(tool, FunctionTool):
                    tool_list.append(
                        {
                            "type": "custom",
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.parameters(),
                        }
                    )
                elif isinstance(tool, Mapping) and tool.get("type") == "mcp":  # type: ignore[reportUnknownMemberType]
                    # MCP servers must be routed to separate mcp_servers parameter
                    server_def: dict[str, Any] = {
                        "type": "url",
                        "name": tool.get("server_label", ""),  # type: ignore[reportUnknownMemberType]
                        "url": tool.get("server_url", ""),  # type: ignore[reportUnknownMemberType]
                    }
                    allowed_tools = tool.get("allowed_tools")  # type: ignore[reportUnknownMemberType]
                    if isinstance(allowed_tools, Sequence) and not isinstance(allowed_tools, str):
                        server_def["tool_configuration"] = {
                            "allowed_tools": [str(item) for item in allowed_tools]  # pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType]
                        }
                    headers = tool.get("headers")  # type: ignore[reportUnknownMemberType]
                    authorization = headers.get("authorization") if isinstance(headers, Mapping) else None  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
                    if isinstance(authorization, str):
                        server_def["authorization_token"] = authorization
                    mcp_server_list.append(server_def)
                else:
                    # Pass through all other tools (dicts, SDK types) unchanged
                    tool_list.append(tool)

            if tool_list:
                result["tools"] = tool_list
            if mcp_server_list:
                result["mcp_servers"] = mcp_server_list
            self._tool_name_aliases = tool_name_aliases
        else:
            self._tool_name_aliases = {}

        # Process tool choice
        if options.get("tool_choice") is None:
            return result or None
        tool_mode = validate_tool_mode(options.get("tool_choice"))
        if tool_mode is None:
            return result or None
        if "allowed_tools" in tool_mode:
            logger.warning("allowed_tools is not supported by Anthropic; the setting will be ignored")
        allow_multiple = options.get("allow_multiple_tool_calls")
        tool_choice: dict[str, Any]
        match tool_mode.get("mode"):
            case "auto":
                tool_choice = {"type": "auto"}
                if allow_multiple is not None:
                    tool_choice["disable_parallel_tool_use"] = not allow_multiple
                result["tool_choice"] = tool_choice
            case "required":
                if "required_function_name" in tool_mode:
                    required_name = tool_mode["required_function_name"]
                    api_tool_name = next(
                        (
                            api_name
                            for api_name, local_name in self._tool_name_aliases.items()
                            if local_name == required_name
                        ),
                        required_name,
                    )
                    tool_choice = {
                        "type": "tool",
                        "name": api_tool_name,
                    }
                else:
                    tool_choice = {"type": "any"}
                if allow_multiple is not None:
                    tool_choice["disable_parallel_tool_use"] = not allow_multiple
                result["tool_choice"] = tool_choice
            case "none":
                result["tool_choice"] = {"type": "none"}
            case _:
                logger.debug(f"Ignoring unsupported tool choice mode: {tool_mode} for now")

        return result or None

    # region Response Processing Methods

    def _process_message(self, message: BetaMessage, options: Mapping[str, Any]) -> ChatResponse:
        """Process the response from the Anthropic client.

        Args:
            message: The message returned by the Anthropic client.
            options: The options dict used for the request.

        Returns:
            A ChatResponse object containing the processed response.
        """
        usage_details = self._parse_usage_from_anthropic(message.usage)
        if (
            usage_details is not None
            and (context_estimate := self._anthropic_blocking_context_input_estimate(message.usage, message.content))
            is not None
        ):
            usage_details["context_input_token_estimate"] = context_estimate
        return ChatResponse(
            response_id=message.id,
            messages=[
                Message(
                    role="assistant",
                    contents=self._parse_contents_from_anthropic(message.content),
                    raw_representation=message,
                )
            ],
            usage_details=usage_details,
            model=message.model,
            finish_reason=_map_finish_reason(message.stop_reason),
            response_format=options.get("response_format"),
            raw_representation=message,
        )

    def _process_stream_event(
        self,
        event: BetaRawMessageStreamEvent,
        stream_state: _AnthropicStreamState,
    ) -> ChatResponseUpdate | None:
        """Process a streaming event from the Anthropic client.

        Args:
            event: The streaming event returned by the Anthropic client.
            stream_state: Request-local indexed tool-call assembly state.

        Returns:
            A ChatResponseUpdate object containing the processed update.
        """
        match event.type:
            case "message_start":
                usage_details: list[Content] = []
                stream_state.initial_cache_read_input_tokens = self._non_negative_usage_value(
                    event.message.usage.cache_read_input_tokens if event.message.usage else None
                )
                if event.message.usage and (details := self._parse_usage_from_anthropic(event.message.usage)):
                    usage_details.append(Content.from_usage(usage_details=details))

                return ChatResponseUpdate(
                    role="assistant",
                    response_id=event.message.id,
                    contents=[
                        *self._parse_contents_from_anthropic(event.message.content),
                        *usage_details,
                    ],
                    model=event.message.model,
                    finish_reason=_map_finish_reason(event.message.stop_reason),
                    raw_representation=event,
                )
            case "message_delta":
                usage = self._parse_usage_from_anthropic(event.usage)
                if (
                    usage is not None
                    and (
                        context_input := self._anthropic_stream_context_input_tokens(
                            event.usage,
                            initial_cache_read_input_tokens=stream_state.initial_cache_read_input_tokens,
                        )
                    )
                    is not None
                ):
                    usage["context_input_token_count"] = context_input
                return ChatResponseUpdate(
                    contents=[Content.from_usage(usage_details=usage, raw_representation=event.usage)] if usage else [],
                    finish_reason=_map_finish_reason(event.delta.stop_reason),
                    raw_representation=event,
                )
            case "message_stop":
                stream_state.hosted_tool_indices.clear()
                stream_state.hosted_tool_calls.clear()
                stream_state.hosted_argument_deltas.clear()
                logger.debug("Received message_stop event; no content to process.")
            case "content_block_start":
                if event.content_block.type == "tool_use":
                    if stream_state.defer_from_index is None or event.index < stream_state.defer_from_index:
                        stream_state.defer_from_index = event.index
                    stream_state.pending_function_calls[event.index] = _PendingFunctionCall(
                        call_id=event.content_block.id,
                        name=self._tool_name_aliases.get(event.content_block.name, event.content_block.name),
                        initial_arguments=event.content_block.input,
                        raw_representations=[event.content_block],
                        argument_deltas=[],
                    )
                    return ChatResponseUpdate(contents=[], raw_representation=event)
                if event.content_block.type in ("mcp_tool_use", "server_tool_use"):
                    stream_state.hosted_tool_indices.add(event.index)
                contents = self._parse_contents_from_anthropic([event.content_block])
                if event.index in stream_state.hosted_tool_indices:
                    hosted_call = next(
                        (content for content in contents if content.provider_hosted and content.call_id),
                        None,
                    )
                    if hosted_call is not None:
                        stream_state.hosted_tool_calls[event.index] = hosted_call
                        stream_state.hosted_argument_deltas[event.index] = []
                return ChatResponseUpdate(
                    contents=contents,
                    raw_representation=event,
                )
            case "content_block_delta":
                if event.delta.type == "input_json_delta":
                    if event.index in stream_state.hosted_tool_indices:
                        argument_deltas = stream_state.hosted_argument_deltas.setdefault(event.index, [])
                        argument_deltas.append(event.delta.partial_json)
                        hosted_call = stream_state.hosted_tool_calls.get(event.index)
                        if hosted_call is not None:
                            raw_arguments = "".join(argument_deltas)
                            try:
                                arguments = json.loads(raw_arguments)
                            except json.JSONDecodeError:
                                arguments = raw_arguments
                            self._apply_anthropic_streamed_arguments(hosted_call, arguments)
                            hosted_call.provider_phase = HostedToolPhase.DELTA
                            hosted_call.raw_representation = event
                            return ChatResponseUpdate(contents=[hosted_call], raw_representation=event)
                    else:
                        pending_function_call = stream_state.pending_function_calls.get(event.index)
                        if pending_function_call is None:
                            logger.warning(
                                "Ignoring Anthropic input_json_delta without a matching local tool-use block at index %d",
                                event.index,
                            )
                        else:
                            pending_function_call.append_arguments(event.delta.partial_json, event.delta)
                    return ChatResponseUpdate(contents=[], raw_representation=event)
                contents = self._parse_contents_from_anthropic([event.delta])
                return ChatResponseUpdate(
                    contents=contents,
                    raw_representation=event,
                )
            case "content_block_stop":
                stream_state.hosted_tool_indices.discard(event.index)
                hosted_call = stream_state.hosted_tool_calls.pop(event.index, None)
                if hosted_call is not None:
                    hosted_call.provider_phase = HostedToolPhase.START
                stream_state.hosted_argument_deltas.pop(event.index, None)
                if event.index in stream_state.pending_function_calls:
                    # Stop events may themselves be interleaved. Keep completed local calls buffered so
                    # the pre-terminal drain can emit them in canonical content-block order.
                    return ChatResponseUpdate(contents=[], raw_representation=event)
                logger.debug("Received content_block_stop event; no content to process.")
            case _:
                logger.debug(f"Ignoring unsupported event type: {event.type}")
        return None

    @staticmethod
    def _non_negative_usage_value(value: Any) -> int | None:
        """Return a non-negative provider usage integer without accepting booleans."""
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None

    @classmethod
    def _anthropic_stream_context_input_tokens(
        cls,
        usage: BetaMessageDeltaUsage | None,
        *,
        initial_cache_read_input_tokens: int | None,
    ) -> int | None:
        """Recover final context occupancy from Anthropic's hosted-loop aggregate usage.

        Anthropic's terminal server-tool snapshot reports cache reads accumulated
        across every internal sampling pass. Only the initial request's cache read
        belongs to the final context; the terminal cache creation and uncached input
        describe the retained remainder.
        """
        if initial_cache_read_input_tokens is None:
            return None
        context_floor = cls._anthropic_context_input_floor(usage)
        if context_floor is None:
            return None
        return initial_cache_read_input_tokens + context_floor

    @classmethod
    def _anthropic_context_input_floor(
        cls,
        usage: BetaUsage | BetaMessageDeltaUsage | None,
    ) -> int | None:
        """Return context tokens known to remain without counting aggregate cache reads."""
        if usage is None:
            return None
        uncached_input = cls._non_negative_usage_value(usage.input_tokens)
        cache_creation = cls._non_negative_usage_value(usage.cache_creation_input_tokens)
        if uncached_input is None or cache_creation is None:
            return None
        return cache_creation + uncached_input

    @classmethod
    def _anthropic_blocking_context_input_estimate(
        cls,
        usage: BetaUsage | None,
        content: Sequence[Any],
    ) -> int | None:
        """Estimate blocking hosted occupancy from aggregate cache reads.

        Blocking Anthropic responses expose only the aggregate cache read over
        every internal server-tool sampling pass. The average read estimates
        a retained cached prefix when cache creation no longer dominates the
        request; nested tools executed inside a code container do not add a
        sampling pass of their own.
        """
        context_floor = cls._anthropic_context_input_floor(usage)
        if usage is None or context_floor is None:
            return None
        cache_read = cls._non_negative_usage_value(usage.cache_read_input_tokens)
        cache_creation = cls._non_negative_usage_value(usage.cache_creation_input_tokens)
        sampling_passes = cls._anthropic_blocking_sampling_passes(content)
        if cache_read is None or cache_creation is None or sampling_passes is None:
            return context_floor
        average_cache_read = (cache_read + sampling_passes - 1) // sampling_passes
        if average_cache_read <= cache_creation:
            return context_floor
        return context_floor + average_cache_read

    @staticmethod
    def _anthropic_blocking_sampling_passes(content: Sequence[Any]) -> int | None:
        """Count top-level server-tool continuations in one blocking response."""
        container_names = {"code_execution", "bash_code_execution", "text_editor_code_execution"}
        open_container_ids: list[str] = []
        top_level_results = 0
        for block in content:
            block_type = str(getattr(block, "type", ""))
            if block_type == "server_tool_use":
                block_id = getattr(block, "id", None)
                if getattr(block, "name", None) in container_names and isinstance(block_id, str):
                    open_container_ids.append(block_id)
                continue
            if block_type == "tool_result" or not block_type.endswith("_tool_result"):
                continue
            tool_use_id = getattr(block, "tool_use_id", None)
            if isinstance(tool_use_id, str) and tool_use_id in open_container_ids:
                container_index = open_container_ids.index(tool_use_id)
                if container_index == 0:
                    top_level_results += 1
                del open_container_ids[container_index]
            elif not open_container_ids:
                top_level_results += 1
        return top_level_results + 1 if top_level_results else None

    def _parse_usage_from_anthropic(self, usage: BetaUsage | BetaMessageDeltaUsage | None) -> UsageDetails | None:
        """Parse usage details from the Anthropic message usage."""
        if not usage:
            return None
        usage_details = UsageDetails(output_token_count=usage.output_tokens)
        if usage.input_tokens is not None:
            usage_details["input_token_count"] = usage.input_tokens
        if usage.cache_creation_input_tokens is not None:
            usage_details["anthropic.cache_creation_input_tokens"] = usage.cache_creation_input_tokens  # type: ignore[typeddict-unknown-key]
            usage_details["cache_creation_input_token_count"] = usage.cache_creation_input_tokens
        if usage.cache_read_input_tokens is not None:
            usage_details["anthropic.cache_read_input_tokens"] = usage.cache_read_input_tokens  # type: ignore[typeddict-unknown-key]
            usage_details["cache_read_input_token_count"] = usage.cache_read_input_tokens
        if (context_floor := self._anthropic_context_input_floor(usage)) is not None:
            usage_details["context_input_token_floor"] = context_floor
        return usage_details

    @staticmethod
    def _serialize_anthropic_payload(value: Any) -> Any:
        """Convert Anthropic SDK blocks into persisted JSON-safe payloads."""
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", exclude_none=True, exclude_unset=True)
        if isinstance(value, Mapping):
            return {str(key): RawAnthropicClient._serialize_anthropic_payload(item) for key, item in value.items()}  # type: ignore[reportUnknownVariableType]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [RawAnthropicClient._serialize_anthropic_payload(item) for item in value]  # type: ignore[reportUnknownVariableType]
        if hasattr(value, "__dict__"):
            return {
                str(key): RawAnthropicClient._serialize_anthropic_payload(item)
                for key, item in vars(value).items()
                if not key.startswith("_") and item is not None
            }
        return value

    def _anthropic_hosted_properties(self, block: Any, *, error: Any = None) -> dict[str, Any]:
        """Build persisted Anthropic replay and failure metadata."""
        properties: dict[str, Any] = {ANTHROPIC_HOSTED_WIRE_BLOCK_KEY: self._serialize_anthropic_payload(block)}
        if error is not None:
            properties["is_error"] = True
            properties["error"] = self._serialize_anthropic_payload(error)
        return properties

    @staticmethod
    def _anthropic_result_is_error(result: Any) -> bool:
        """Return whether an Anthropic server-tool result payload is an error block."""
        result_type = str(getattr(result, "type", ""))
        return result_type.endswith(("_error", "_error_block"))

    @staticmethod
    def _anthropic_shell_commands(arguments: Any) -> list[str]:
        """Extract command text from a server-side bash execution input."""
        command = arguments.get("command") if isinstance(arguments, Mapping) else arguments
        if isinstance(command, str):
            return [command]
        if isinstance(command, Sequence) and not isinstance(command, (str, bytes, bytearray)):
            return [str(part) for part in command]
        return []

    @staticmethod
    def _anthropic_code_input(arguments: Any) -> str:
        """Extract canonical source text from Anthropic code-execution input."""
        code = arguments.get("code") if isinstance(arguments, Mapping) else None
        return code if isinstance(code, str) else json.dumps(arguments, ensure_ascii=False)

    def _parse_anthropic_server_tool_call(self, content_block: Any) -> Content:
        """Map an exact Anthropic server-tool name to its canonical hosted family."""
        item_id = content_block.id
        name = content_block.name or ""
        arguments = self._serialize_anthropic_payload(content_block.input)
        properties = self._anthropic_hosted_properties(content_block)
        match name:
            case "web_search":
                return Content.from_search_tool_call(
                    call_id=item_id,
                    tool_name="web_search",
                    arguments=arguments,
                    status="running",
                    retry_safety=HostedRetrySafety.READ_ONLY,
                    hosted_provider="anthropic",
                    provider_item_type="server_tool_use",
                    provider_item_id=item_id,
                    provider_phase=HostedToolPhase.START,
                    provider_status="running",
                    additional_properties=properties,
                    raw_representation=content_block,
                )
            case "web_fetch":
                return Content.from_search_tool_call(
                    call_id=item_id,
                    tool_name="web_fetch",
                    arguments=arguments,
                    status="running",
                    hosted_family=HostedToolFamily.FETCH,
                    retry_safety=HostedRetrySafety.READ_ONLY,
                    hosted_provider="anthropic",
                    provider_item_type="server_tool_use",
                    provider_item_id=item_id,
                    provider_phase=HostedToolPhase.START,
                    provider_status="running",
                    additional_properties=properties,
                    raw_representation=content_block,
                )
            case "code_execution":
                return Content.from_code_interpreter_tool_call(
                    call_id=item_id,
                    inputs=[Content.from_text(self._anthropic_code_input(arguments))],
                    retry_safety=HostedRetrySafety.SANDBOXED,
                    hosted_provider="anthropic",
                    provider_item_type="server_tool_use",
                    provider_item_id=item_id,
                    provider_phase=HostedToolPhase.START,
                    provider_status="running",
                    additional_properties=properties,
                    raw_representation=content_block,
                )
            case "bash_code_execution":
                return Content.from_shell_tool_call(
                    call_id=item_id,
                    commands=self._anthropic_shell_commands(arguments),
                    status="running",
                    retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
                    hosted_provider="anthropic",
                    provider_item_type="server_tool_use",
                    provider_item_id=item_id,
                    provider_phase=HostedToolPhase.START,
                    provider_status="running",
                    additional_properties=properties,
                    raw_representation=content_block,
                )
            case "text_editor_code_execution":
                return Content.from_hosted_tool_call(
                    call_id=item_id,
                    tool_name=name,
                    arguments=arguments,
                    status="running",
                    hosted_family=HostedToolFamily.FILE_OPERATION,
                    retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
                    hosted_provider="anthropic",
                    provider_item_type="server_tool_use",
                    provider_item_id=item_id,
                    provider_phase=HostedToolPhase.START,
                    provider_status="running",
                    additional_properties=properties,
                    raw_representation=content_block,
                )
            case "tool_search" | "tool_search_tool_regex" | "tool_search_tool_bm25":
                return Content.from_hosted_tool_call(
                    call_id=item_id,
                    tool_name="tool_search",
                    arguments=arguments,
                    status="running",
                    hosted_family=HostedToolFamily.TOOL_DISCOVERY,
                    retry_safety=HostedRetrySafety.READ_ONLY,
                    hosted_provider="anthropic",
                    provider_item_type="server_tool_use",
                    provider_item_id=item_id,
                    provider_phase=HostedToolPhase.START,
                    provider_status="running",
                    additional_properties=properties,
                    raw_representation=content_block,
                )
            case _:
                return Content.from_hosted_tool_call(
                    call_id=item_id,
                    tool_name=name or "anthropic_server_tool",
                    arguments=arguments,
                    status="running",
                    retry_safety=HostedRetrySafety.UNKNOWN,
                    hosted_provider="anthropic",
                    provider_item_type="server_tool_use",
                    provider_item_id=item_id,
                    provider_phase=HostedToolPhase.START,
                    provider_status="running",
                    additional_properties=properties,
                    raw_representation=content_block,
                )

    @staticmethod
    def _apply_anthropic_streamed_arguments(content: Content, arguments: Any) -> None:
        """Refresh one streamed hosted call from its complete indexed JSON input."""
        if content.type == "code_interpreter_tool_call":
            content.inputs = [Content.from_text(RawAnthropicClient._anthropic_code_input(arguments))]
        elif content.type == "shell_tool_call":
            content.commands = RawAnthropicClient._anthropic_shell_commands(arguments)
        else:
            content.arguments = arguments
        replay_block = content.additional_properties.get(ANTHROPIC_HOSTED_WIRE_BLOCK_KEY)
        if isinstance(replay_block, dict):
            replay_block["input"] = arguments

    def _parse_contents_from_anthropic(
        self,
        content: Sequence[BetaContentBlock | BetaRawContentBlockDelta | BetaTextBlock],
    ) -> list[Content]:
        """Parse contents from the Anthropic message."""
        contents: list[Content] = []
        for content_block in content:
            match content_block.type:
                case "text" | "text_delta":
                    contents.append(
                        Content.from_text(
                            text=content_block.text,
                            raw_representation=content_block,
                            annotations=self._parse_citations_from_anthropic(content_block),
                        )
                    )
                case "tool_use" | "mcp_tool_use" | "server_tool_use":
                    if content_block.type == "mcp_tool_use":
                        contents.append(
                            Content.from_mcp_server_tool_call(
                                call_id=content_block.id,
                                tool_name=content_block.name,
                                server_name=content_block.server_name,
                                arguments=content_block.input,
                                hosted_provider="anthropic",
                                provider_item_type="mcp_tool_use",
                                provider_item_id=content_block.id,
                                provider_phase=HostedToolPhase.START,
                                provider_status="running",
                                retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
                                additional_properties=self._anthropic_hosted_properties(content_block),
                                raw_representation=content_block,
                            )
                        )
                    elif content_block.type == "server_tool_use":
                        contents.append(self._parse_anthropic_server_tool_call(content_block))
                    else:
                        resolved_tool_name = self._tool_name_aliases.get(content_block.name, content_block.name)
                        contents.append(
                            Content.from_function_call(
                                call_id=content_block.id,
                                name=resolved_tool_name,
                                arguments=content_block.input,
                                raw_representation=content_block,
                            )
                        )
                case "mcp_tool_result":
                    parsed_output: list[Content] | None = None
                    if content_block.content:
                        if isinstance(content_block.content, list):
                            parsed_output = self._parse_contents_from_anthropic(content_block.content)
                        elif isinstance(content_block.content, (str, bytes)):
                            parsed_output = [
                                Content.from_text(
                                    text=str(content_block.content),
                                    raw_representation=content_block,
                                )
                            ]
                        else:
                            parsed_output = self._parse_contents_from_anthropic([content_block.content])
                    mcp_is_error = bool(getattr(content_block, "is_error", False))
                    mcp_error = content_block.content if mcp_is_error else None
                    contents.append(
                        Content.from_mcp_server_tool_result(
                            call_id=content_block.tool_use_id,
                            output=parsed_output,
                            hosted_provider="anthropic",
                            provider_item_type="mcp_tool_result",
                            provider_item_id=content_block.tool_use_id,
                            provider_phase=HostedToolPhase.TERMINAL,
                            provider_status="failed" if mcp_is_error else "completed",
                            retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
                            additional_properties=self._anthropic_hosted_properties(
                                content_block,
                                error=mcp_error,
                            ),
                            raw_representation=content_block,
                        )
                    )
                case "web_search_tool_result" | "web_fetch_tool_result":
                    tool_name = "web_search" if content_block.type == "web_search_tool_result" else "web_fetch"
                    family = (
                        HostedToolFamily.SEARCH
                        if content_block.type == "web_search_tool_result"
                        else HostedToolFamily.FETCH
                    )
                    is_error = self._anthropic_result_is_error(content_block.content)
                    contents.append(
                        Content.from_search_tool_result(
                            call_id=content_block.tool_use_id,
                            tool_name=tool_name,
                            result=self._serialize_anthropic_payload(content_block.content),
                            status="failed" if is_error else "completed",
                            hosted_family=family,
                            hosted_provider="anthropic",
                            provider_item_type=content_block.type,
                            provider_item_id=content_block.tool_use_id,
                            provider_phase=HostedToolPhase.TERMINAL,
                            provider_status="failed" if is_error else "completed",
                            retry_safety=HostedRetrySafety.READ_ONLY,
                            additional_properties=self._anthropic_hosted_properties(
                                content_block,
                                error=content_block.content if is_error else None,
                            ),
                            raw_representation=content_block,
                        )
                    )
                case "code_execution_tool_result":
                    code_outputs: list[Content] = []
                    code_error = (
                        content_block.content if self._anthropic_result_is_error(content_block.content) else None
                    )
                    if content_block.content:
                        if isinstance(content_block.content, BetaCodeExecutionToolResultError):
                            code_outputs.append(
                                Content.from_error(
                                    message=content_block.content.error_code,
                                    raw_representation=content_block.content,
                                )
                            )
                        else:
                            if (
                                isinstance(content_block.content, BetaCodeExecutionResultBlock)
                                and content_block.content.stdout
                            ):
                                code_outputs.append(
                                    Content.from_text(
                                        text=content_block.content.stdout,
                                        raw_representation=content_block.content,
                                    )
                                )
                            if (
                                isinstance(content_block.content, BetaEncryptedCodeExecutionResultBlock)
                                and content_block.content.encrypted_stdout
                            ):
                                code_outputs.append(
                                    Content.from_text(
                                        text=content_block.content.encrypted_stdout,
                                        raw_representation=content_block.content,
                                    )
                                )
                            if content_block.content.stderr:
                                code_outputs.append(
                                    Content.from_error(
                                        message=content_block.content.stderr,
                                        raw_representation=content_block.content,
                                    )
                                )
                            code_outputs.extend(
                                Content.from_hosted_file(
                                    file_id=code_file_content.file_id,
                                    raw_representation=code_file_content,
                                )
                                for code_file_content in content_block.content.content
                            )
                    contents.append(
                        Content.from_code_interpreter_tool_result(
                            call_id=content_block.tool_use_id,
                            raw_representation=content_block,
                            outputs=code_outputs,
                            hosted_provider="anthropic",
                            provider_item_type="code_execution_tool_result",
                            provider_item_id=content_block.tool_use_id,
                            provider_phase=HostedToolPhase.TERMINAL,
                            provider_status="failed" if code_error is not None else "completed",
                            retry_safety=HostedRetrySafety.SANDBOXED,
                            additional_properties=self._anthropic_hosted_properties(
                                content_block,
                                error=code_error,
                            ),
                        )
                    )
                case "bash_code_execution_tool_result":
                    shell_outputs: list[Content] = []
                    bash_error = (
                        content_block.content if self._anthropic_result_is_error(content_block.content) else None
                    )
                    if content_block.content:
                        if isinstance(
                            content_block.content,
                            BetaBashCodeExecutionToolResultError,
                        ):
                            shell_outputs.append(
                                Content.from_shell_command_output(
                                    stderr=content_block.content.error_code,
                                    timed_out=content_block.content.error_code == "execution_time_exceeded",
                                    raw_representation=content_block.content,
                                )
                            )
                        else:
                            shell_outputs.append(
                                Content.from_shell_command_output(
                                    stdout=content_block.content.stdout or None,
                                    stderr=content_block.content.stderr or None,
                                    exit_code=int(content_block.content.return_code),
                                    timed_out=False,
                                    raw_representation=content_block.content,
                                )
                            )
                            shell_outputs.extend(
                                Content.from_hosted_file(
                                    file_id=bash_file_content.file_id,
                                    raw_representation=bash_file_content,
                                )
                                for bash_file_content in content_block.content.content
                            )
                    contents.append(
                        Content.from_shell_tool_result(
                            call_id=content_block.tool_use_id,
                            outputs=shell_outputs,
                            hosted_provider="anthropic",
                            provider_item_type="bash_code_execution_tool_result",
                            provider_item_id=content_block.tool_use_id,
                            provider_phase=HostedToolPhase.TERMINAL,
                            provider_status="failed" if bash_error is not None else "completed",
                            retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
                            additional_properties=self._anthropic_hosted_properties(
                                content_block,
                                error=bash_error,
                            ),
                            raw_representation=content_block,
                        )
                    )
                case "text_editor_code_execution_tool_result":
                    text_editor_outputs: list[Content] = []
                    text_editor_error = (
                        content_block.content if self._anthropic_result_is_error(content_block.content) else None
                    )
                    match content_block.content.type:
                        case "text_editor_code_execution_tool_result_error":
                            text_editor_outputs.append(
                                Content.from_error(
                                    message=content_block.content.error_message or content_block.content.error_code,
                                    error_code=content_block.content.error_code,
                                    raw_representation=content_block.content,
                                )
                            )
                        case "text_editor_code_execution_view_result":
                            annotations = (
                                [
                                    Annotation(
                                        type="citation",
                                        raw_representation=content_block.content,
                                        annotated_regions=[
                                            TextSpanRegion(
                                                type="text_span",
                                                start_index=content_block.content.start_line,
                                                end_index=content_block.content.start_line
                                                + (content_block.content.num_lines or 0),
                                            )
                                        ],
                                    )
                                ]
                                if content_block.content.num_lines is not None
                                and content_block.content.start_line is not None
                                else None
                            )
                            text_editor_outputs.append(
                                Content.from_text(
                                    text=content_block.content.content,
                                    annotations=annotations,
                                    raw_representation=content_block.content,
                                )
                            )
                        case "text_editor_code_execution_str_replace_result":
                            old_annotation = (
                                Annotation(
                                    type="citation",
                                    raw_representation=content_block.content,
                                    annotated_regions=[
                                        TextSpanRegion(
                                            type="text_span",
                                            start_index=content_block.content.old_start or 0,
                                            end_index=(
                                                (content_block.content.old_start or 0)
                                                + (content_block.content.old_lines or 0)
                                            ),
                                        )
                                    ],
                                )
                                if content_block.content.old_lines is not None
                                and content_block.content.old_start is not None
                                else None
                            )
                            new_annotation = (
                                Annotation(
                                    type="citation",
                                    raw_representation=content_block.content,
                                    snippet=(
                                        "\n".join(content_block.content.lines) if content_block.content.lines else None
                                    ),
                                    annotated_regions=[
                                        TextSpanRegion(
                                            type="text_span",
                                            start_index=content_block.content.new_start or 0,
                                            end_index=(
                                                (content_block.content.new_start or 0)
                                                + (content_block.content.new_lines or 0)
                                            ),
                                        )
                                    ],
                                )
                                if content_block.content.new_lines is not None
                                and content_block.content.new_start is not None
                                else None
                            )
                            annotations = [ann for ann in [old_annotation, new_annotation] if ann is not None]

                            text_editor_outputs.append(
                                Content.from_text(
                                    text=(
                                        "\n".join(content_block.content.lines) if content_block.content.lines else ""
                                    ),
                                    annotations=annotations or None,
                                    raw_representation=content_block.content,
                                )
                            )
                        case "text_editor_code_execution_create_result":
                            text_editor_outputs.append(
                                Content.from_text(
                                    text=f"File update: {content_block.content.is_file_update}",
                                    raw_representation=content_block.content,
                                )
                            )
                    contents.append(
                        Content.from_hosted_tool_result(
                            call_id=content_block.tool_use_id,
                            tool_name="text_editor_code_execution",
                            result=self._serialize_anthropic_payload(content_block.content),
                            items=text_editor_outputs,
                            status="failed" if text_editor_error is not None else "completed",
                            hosted_family=HostedToolFamily.FILE_OPERATION,
                            hosted_provider="anthropic",
                            provider_item_type="text_editor_code_execution_tool_result",
                            provider_item_id=content_block.tool_use_id,
                            provider_phase=HostedToolPhase.TERMINAL,
                            provider_status="failed" if text_editor_error is not None else "completed",
                            retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
                            additional_properties=self._anthropic_hosted_properties(
                                content_block,
                                error=text_editor_error,
                            ),
                            raw_representation=content_block,
                        )
                    )
                case "tool_search_tool_result":
                    tool_search_error = (
                        content_block.content if self._anthropic_result_is_error(content_block.content) else None
                    )
                    contents.append(
                        Content.from_hosted_tool_result(
                            call_id=content_block.tool_use_id,
                            tool_name="tool_search",
                            result=self._serialize_anthropic_payload(content_block.content),
                            status="failed" if tool_search_error is not None else "completed",
                            hosted_family=HostedToolFamily.TOOL_DISCOVERY,
                            hosted_provider="anthropic",
                            provider_item_type="tool_search_tool_result",
                            provider_item_id=content_block.tool_use_id,
                            provider_phase=HostedToolPhase.TERMINAL,
                            provider_status="failed" if tool_search_error is not None else "completed",
                            retry_safety=HostedRetrySafety.READ_ONLY,
                            additional_properties=self._anthropic_hosted_properties(
                                content_block,
                                error=tool_search_error,
                            ),
                            raw_representation=content_block,
                        )
                    )
                case "input_json_delta":
                    # Stream-level parsing owns these because only the event carries the block index.
                    logger.debug("Ignoring Anthropic input_json_delta without stream event context")
                case "thinking" | "thinking_delta":
                    contents.append(
                        Content.from_text_reasoning(
                            text=content_block.thinking,
                            protected_data=getattr(content_block, "signature", None),
                            raw_representation=content_block,
                        )
                    )
                case "redacted_thinking":
                    contents.append(
                        Content.from_text_reasoning(
                            text=None,
                            protected_data=content_block.data,
                            additional_properties={_ANTHROPIC_REDACTED_THINKING_KEY: True},
                            raw_representation=content_block,
                        )
                    )
                case "signature_delta":
                    contents.append(
                        Content.from_text_reasoning(
                            text=None,
                            protected_data=content_block.signature,
                            raw_representation=content_block,
                        )
                    )
                case _:
                    if content_block.type != "tool_result" and content_block.type.endswith("_tool_result"):
                        unknown_block: Any = content_block
                        unknown_error = (
                            unknown_block.content if self._anthropic_result_is_error(unknown_block.content) else None
                        )
                        contents.append(
                            Content.from_hosted_tool_result(
                                call_id=unknown_block.tool_use_id,
                                tool_name=content_block.type.removesuffix("_tool_result"),
                                result=self._serialize_anthropic_payload(unknown_block.content),
                                status="failed" if unknown_error is not None else "completed",
                                hosted_provider="anthropic",
                                provider_item_type=content_block.type,
                                provider_item_id=unknown_block.tool_use_id,
                                provider_phase=HostedToolPhase.TERMINAL,
                                provider_status="failed" if unknown_error is not None else "completed",
                                retry_safety=HostedRetrySafety.UNKNOWN,
                                additional_properties=self._anthropic_hosted_properties(
                                    content_block,
                                    error=unknown_error,
                                ),
                                raw_representation=content_block,
                            )
                        )
                    else:
                        logger.debug(f"Ignoring unsupported content type: {content_block.type} for now")
        return contents

    def _parse_citations_from_anthropic(
        self, content_block: BetaContentBlock | BetaRawContentBlockDelta | BetaTextBlock
    ) -> list[Annotation] | None:
        content_blocks = getattr(content_block, "citations", None)
        if not content_blocks:
            return None
        annotations: list[Annotation] = []
        for citation in content_blocks:
            cit = Annotation(type="citation", raw_representation=citation)
            match citation.type:
                case "char_location":
                    cit["title"] = citation.title
                    cit["snippet"] = citation.cited_text
                    if citation.file_id:
                        cit["file_id"] = citation.file_id
                    cit["annotated_regions"] = [
                        TextSpanRegion(
                            type="text_span",
                            start_index=citation.start_char_index,
                            end_index=citation.end_char_index,
                        )
                    ]
                case "page_location":
                    cit["title"] = citation.document_title
                    cit["snippet"] = citation.cited_text
                    if citation.file_id:
                        cit["file_id"] = citation.file_id
                    cit["annotated_regions"] = [
                        TextSpanRegion(
                            type="text_span",
                            start_index=citation.start_page_number,
                            end_index=citation.end_page_number,
                        )
                    ]
                case "content_block_location":
                    cit["title"] = citation.document_title
                    cit["snippet"] = citation.cited_text
                    if citation.file_id:
                        cit["file_id"] = citation.file_id
                    cit["annotated_regions"] = [
                        TextSpanRegion(
                            type="text_span",
                            start_index=citation.start_block_index,
                            end_index=citation.end_block_index,
                        )
                    ]
                case "web_search_result_location":
                    cit["title"] = citation.title
                    cit["snippet"] = citation.cited_text
                    cit["url"] = citation.url
                case "search_result_location":
                    cit["title"] = citation.title
                    cit["snippet"] = citation.cited_text
                    cit["url"] = citation.source
                    cit["annotated_regions"] = [
                        TextSpanRegion(
                            type="text_span",
                            start_index=citation.start_block_index,
                            end_index=citation.end_block_index,
                        )
                    ]
                case _:
                    logger.debug(f"Unknown citation type encountered: {citation.type}")
            annotations.append(cit)
        return annotations or None

    def service_url(self) -> str:
        """Get the service URL for the chat client.

        Returns:
            The service URL for the chat client, or None if not set.
        """
        return str(self.anthropic_client.base_url)
