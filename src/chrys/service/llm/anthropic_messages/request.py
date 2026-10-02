# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Build the ``messages.create`` arguments of one call.

:func:`build_request` decides every request field: the renamed chat options,
the default output cap, the encoded history and system prompt, the beta set,
the user id, tool declarations and structured output. The client stamps its
Chrys headers on the result.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from chrys.kernel import (
    ChatClientInvalidRequestException,
    FunctionTool,
    Message,
    normalize_tools,
    prepend_instructions_to_messages,
    validate_tool_mode,
)

from .history import encode_messages

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_BETAS: Final = ("mcp-client-2025-04-04", "code-execution-2025-08-25")
"""Betas every request enables; ``additional_beta_flags`` adds to them."""

FALLBACK_MAX_OUTPUT_TOKENS: Final = 16 * 1024
"""Output cap of a call that sets none: the Messages API requires one.

Profile-driven calls never use it: ``effective_chat_options`` gives every
live request the profile's ``max_output_tokens``, whose default
(``DEFAULT_MAX_OUTPUT_TOKENS`` in ``chrys.service.profiles.models.schema``)
is a deliberately separate constant.
"""

# (chat option, Messages API field) pairs; an explicit field wins over its option.
_RENAMED_OPTIONS: Final = (("stop", "stop_sequences"), ("instructions", "system"))

# Chat options build_request reads itself instead of copying them, and ``stream``,
# which the call site sets.
_OPTIONS_NOT_COPIED: Final = frozenset(
    {"instructions", "response_format", "additional_beta_flags", "allow_multiple_tool_calls", "stream"}
)

# Call keywords that configure the call and never become request fields.
_CALL_SETTINGS: Final = frozenset({"thread", "middleware", "additional_beta_flags"})


def build_request(
    messages: Sequence[Message],
    options: Mapping[str, Any],
    call_kwargs: Mapping[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    """Return the request for *messages* under *options*, without Chrys headers.

    Options set to None are left out. Call keywords become request fields too,
    except private (underscore) names and :data:`_CALL_SETTINGS`. *model* is
    used when neither sets one.
    """
    if instructions := options.get("instructions"):
        messages = prepend_instructions_to_messages(list(messages), instructions, role="system")

    request = {key: value for key, value in options.items() if value is not None and key not in _OPTIONS_NOT_COPIED}
    _rename_options(request)
    call_fields = {
        key: value for key, value in call_kwargs.items() if not key.startswith("_") and key not in _CALL_SETTINGS
    }
    _rename_options(call_fields)
    request.update(call_fields)

    if not request.get("model"):
        if not model:
            raise ValueError("model must be a non-empty string")
        request["model"] = model
    if not request.get("max_tokens"):
        request["max_tokens"] = FALLBACK_MAX_OUTPUT_TOKENS
    request["messages"] = encode_messages(messages)
    if messages and isinstance(messages[0], Message) and messages[0].role == "system":
        request["system"] = messages[0].text
    request["betas"] = {*DEFAULT_BETAS, *options.get("additional_beta_flags", [])}
    request.setdefault("extra_headers", {})
    if user := request.pop("user", None):
        # The Messages API takes the end-user id as ``metadata.user_id``.
        metadata = dict(request.get("metadata") or {})
        if "user_id" not in metadata:
            metadata["user_id"] = user
        request["metadata"] = metadata
    if tool_fields := encode_tools(options):
        request.update(tool_fields)
    if (response_format := options.get("response_format")) is not None:
        request["output_config"] = _output_config_with_format(request.get("output_config"), response_format)
    return request


def _rename_options(fields: dict[str, Any]) -> None:
    for option, field in _RENAMED_OPTIONS:
        if option in fields:
            fields.setdefault(field, fields.pop(option))


def encode_tools(options: Mapping[str, Any]) -> dict[str, Any] | None:
    """Encode the call's tools and tool choice; None when there are neither.

    A function tool becomes a ``custom`` declaration, an ``mcp`` mapping an
    ``mcp_servers`` entry; any other tool is a provider declaration and is
    sent as given.
    """
    fields: dict[str, Any] = {}
    if tools := options.get("tools"):
        declarations: list[Any] = []
        mcp_servers: list[dict[str, Any]] = []
        for tool in normalize_tools(tools):
            if isinstance(tool, FunctionTool):
                declarations.append(
                    {
                        "type": "custom",
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters(),
                    }
                )
            elif isinstance(tool, Mapping) and tool.get("type") == "mcp":
                mcp_servers.append(_mcp_server(tool))
            else:
                declarations.append(tool)
        if declarations:
            fields["tools"] = declarations
        if mcp_servers:
            fields["mcp_servers"] = mcp_servers
    if (tool_choice := _tool_choice(options)) is not None:
        fields["tool_choice"] = tool_choice
    return fields or None


def _mcp_server(tool: Mapping[str, Any]) -> dict[str, Any]:
    server: dict[str, Any] = {"type": "url", "name": tool.get("server_label", ""), "url": tool.get("server_url", "")}
    allowed_tools = tool.get("allowed_tools")
    if isinstance(allowed_tools, Sequence) and not isinstance(allowed_tools, str):
        server["tool_configuration"] = {"allowed_tools": [str(name) for name in allowed_tools]}
    headers = tool.get("headers")
    authorization = headers.get("authorization") if isinstance(headers, Mapping) else None
    if isinstance(authorization, str):
        server["authorization_token"] = authorization
    return server


def _tool_choice(options: Mapping[str, Any]) -> dict[str, Any] | None:
    if options.get("tool_choice") is None:
        return None
    tool_mode = validate_tool_mode(options.get("tool_choice"))
    if tool_mode is None:
        return None
    if "allowed_tools" in tool_mode:
        logger.warning("allowed_tools is not supported by Anthropic; the setting will be ignored")
    choice: dict[str, Any]
    match tool_mode.get("mode"):
        case "none":
            return {"type": "none"}
        case "auto":
            choice = {"type": "auto"}
        case "required" if "required_function_name" in tool_mode:
            choice = {"type": "tool", "name": tool_mode["required_function_name"]}
        case "required":
            choice = {"type": "any"}
        case _:
            logger.debug("Ignoring unsupported tool choice mode: %s", tool_mode)
            return None
    # The choice carries the parallel-call setting; it is no request field of its own.
    if (allow_multiple := options.get("allow_multiple_tool_calls")) is not None:
        choice["disable_parallel_tool_use"] = not allow_multiple
    return choice


def _output_config_with_format(output_config: Any, response_format: Any) -> dict[str, Any]:
    if output_config is None:
        merged: dict[str, Any] = {}
    elif isinstance(output_config, Mapping):
        merged = dict(output_config)
    else:
        raise ChatClientInvalidRequestException("output_config must be a mapping.")
    if merged.get("format") is not None:
        raise ChatClientInvalidRequestException(
            "response_format cannot be combined with explicit output_config.format."
        )
    merged["format"] = encode_output_format(response_format)
    return merged


def encode_output_format(response_format: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    """The ``output_config.format`` value for a Pydantic model or a JSON-schema mapping.

    A mapping may carry its schema under ``json_schema.schema`` or ``schema``,
    or be the schema itself. The schema is sent closed to extra properties.
    """
    schema = _json_schema(response_format)
    if isinstance(schema, dict):
        schema = {**schema, "additionalProperties": False}
    return {"type": "json_schema", "schema": schema}


def _json_schema(response_format: type[BaseModel] | dict[str, Any]) -> Any:
    if not isinstance(response_format, dict):
        return response_format.model_json_schema()
    if "json_schema" in response_format:
        return response_format["json_schema"].get("schema", {})
    return response_format.get("schema", response_format)
