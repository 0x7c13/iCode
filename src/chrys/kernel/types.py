# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Public kernel type façade.

Step 1 of the final decoupling plan moves the chat/agent type system and
serialization primitives to Chrys-owned modules. The façade remains the stable
import chokepoint for the rest of the codebase: consumers still import from
``chrys.kernel`` or ``chrys.kernel.types`` while this module chooses
the owned implementation.

``FunctionTool``, ``tool`` and ``normalize_tools`` are sourced from the
sibling ``tools`` module. That module owns tool construction, schema
generation, serialization, result parsing, and normalization.

Deliberately NOT re-exported:

- ``MCPStdioTool`` / ``MCPStreamableHTTPTool`` — consumed only inside the
  MCP red-line domain (``service/mcp/adapter.py``); keeping them out of the
  façade keeps MCP coupling visibly confined to that domain.
- Raw wire clients (``chrys.service.llm.chat_completions``,
  ``chrys.service.llm.openai_responses``, ``chrys.service.llm.anthropic_messages``) — permanent
  red line, never façade material.

HARD RULE: kernel modules may import only the stdlib, intra-package modules,
allowed third-party packages, and downward ``chrys.foundation.*`` modules.
Sibling or upward ``chrys.*`` imports remain forbidden. Intra-package sibling
imports use relative imports.
"""

from ._content import (
    OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY,
    Annotation,
    Content,
    TextSpanRegion,
    UsageDetails,
    add_usage_details,
    detect_media_type_from_base64,
    normalize_stream_usage,
)
from ._types import (
    RETRY_BOUNDARY_UPDATE_KEY,
    AgentResponse,
    AgentResponseUpdate,
    AgentRunInputs,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    ContinuationToken,
    FinishReason,
    FinishReasonLiteral,
    Message,
    ResponseStream,
    Role,
    is_retry_boundary_update,
    map_chat_to_agent_update,
    merge_chat_options,
    normalize_messages,
    prepend_instructions_to_messages,
    validate_chat_options,
    validate_tool_mode,
)
from .client import BaseChatClient
from .exceptions import (
    AdditionItemMismatch,
    AgentInvalidResponseException,
    ChatClientContentFilterException,
    ChatClientException,
    ChatClientInvalidRequestException,
    ContentError,
)
from .tools import (
    SKIP_PARSING,
    FunctionInvocationConfiguration,
    FunctionTool,
    ToolException,
    ToolTypes,
    normalize_tools,
    tool,
)

__all__ = [
    "OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY",
    "RETRY_BOUNDARY_UPDATE_KEY",
    "SKIP_PARSING",
    "AdditionItemMismatch",
    "AgentInvalidResponseException",
    "AgentResponse",
    "AgentResponseUpdate",
    "AgentRunInputs",
    "Annotation",
    "BaseChatClient",
    "ChatClientContentFilterException",
    "ChatClientException",
    "ChatClientInvalidRequestException",
    "ChatOptions",
    "ChatResponse",
    "ChatResponseUpdate",
    "Content",
    "ContentError",
    "ContinuationToken",
    "FinishReason",
    "FinishReasonLiteral",
    "FunctionInvocationConfiguration",
    "FunctionTool",
    "Message",
    "ResponseStream",
    "Role",
    "TextSpanRegion",
    "ToolException",
    "ToolTypes",
    "UsageDetails",
    "add_usage_details",
    "detect_media_type_from_base64",
    "is_retry_boundary_update",
    "map_chat_to_agent_update",
    "merge_chat_options",
    "normalize_messages",
    "normalize_stream_usage",
    "normalize_tools",
    "prepend_instructions_to_messages",
    "tool",
    "validate_chat_options",
    "validate_tool_mode",
]
