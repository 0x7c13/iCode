# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Encode chat history as Messages API messages.

A chat message becomes one wire message per run of blocks that share a wire
role, in their original order: tool calls go out as ``assistant`` and local
tool results as ``user`` whatever message carries them; other blocks, hosted
results included, keep their message's role. Two repairs keep the request
valid: an assistant message's thinking without a signature is dropped (the API
rejects it), and so is a wire message left with no blocks.

Hosted-tool history from another provider is replaced by the neutral summary
:func:`cross_provider_hosted_degradations` writes, sent as assistant context.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from chrys.foundation.hosted_tools import ANTHROPIC_HOSTED_WIRE_BLOCK_KEY
from chrys.kernel._content import _ANTHROPIC_REDACTED_THINKING_KEY, _get_data_bytes_as_str
from chrys.service.agent_middleware.events.hosted_tools import cross_provider_hosted_degradations

if TYPE_CHECKING:
    from chrys.kernel import Content, Message

logger = logging.getLogger(__name__)

# Wire roles fixed by a block's type; other blocks take their message's role.
_ROLE_OF_BLOCK_TYPE: Final[Mapping[str, str]] = {
    "tool_use": "assistant",
    "mcp_tool_use": "assistant",
    "server_tool_use": "assistant",
    "tool_result": "user",
}


@dataclass(frozen=True, slots=True)
class _Block:
    """One encoded content block."""

    wire: dict[str, Any]
    assistant_context: bool = False
    """Sent as ``assistant`` whatever its type: a summary of foreign hosted-tool history."""

    def role(self, message_role: str) -> str:
        if self.assistant_context:
            return "assistant"
        return _ROLE_OF_BLOCK_TYPE.get(self.wire.get("type"), message_role)


def encode_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Encode *messages* after a leading system message, which the request sends as ``system``."""
    history = messages[1:] if messages and messages[0].role == "system" else messages
    summaries = cross_provider_hosted_degradations(history, target_provider="anthropic")
    wire_messages: list[dict[str, Any]] = []
    for message in history:
        role = "assistant" if message.role == "assistant" else "user"
        wire_messages.extend(_by_role(_encode_blocks(message, role, summaries), role))
    return wire_messages


def _by_role(blocks: list[_Block], message_role: str) -> list[dict[str, Any]]:
    """Group consecutive blocks with the same wire role into one wire message each."""
    grouped: list[dict[str, Any]] = []
    for block in blocks:
        role = block.role(message_role)
        if grouped and grouped[-1]["role"] == role:
            grouped[-1]["content"].append(block.wire)
        else:
            grouped.append({"role": role, "content": [block.wire]})
    return grouped


def _encode_blocks(message: Message, role: str, summaries: Mapping[int, str | None]) -> list[_Block]:
    blocks: list[_Block] = []
    for content in message.contents:
        if id(content) in summaries:
            if summary := summaries[id(content)]:
                blocks.append(_Block({"type": "text", "text": summary}, assistant_context=True))
            continue
        # Anthropic's own hosted blocks replay exactly as they were received.
        received = content.additional_properties.get(ANTHROPIC_HOSTED_WIRE_BLOCK_KEY)
        if content.hosted_provider == "anthropic" and isinstance(received, Mapping):
            blocks.append(_Block(dict(received)))
        elif content.type == "text_reasoning":
            _add_thinking(content, blocks)
        elif (wire := _encode_content(content)) is not None:
            blocks.append(_Block(wire))
    if role == "assistant":
        blocks = [block for block in blocks if not _is_unsigned_thinking(block.wire)]
    return blocks


def _is_unsigned_thinking(wire: Mapping[str, Any]) -> bool:
    # Some gateways omit thinking signatures, and replayed thinking without one
    # fails the request ("signature: Field required"). The reasoning is dropped
    # rather than replayed as visible text.
    return wire.get("type") == "thinking" and not wire.get("signature")


def _add_thinking(content: Content, blocks: list[_Block]) -> None:
    properties = content.additional_properties
    if properties.get(_ANTHROPIC_REDACTED_THINKING_KEY):
        blocks.append(_Block({"type": "redacted_thinking", "data": content.protected_data}))
        return
    if content.id or properties.get("reasoning_text") or properties.get("openai_reasoning_format"):
        # Another protocol's reasoning (Responses items carry an id and a
        # marker, Chat Completions dialects stamp their format): sending it as
        # thinking would forge a signature.
        return
    if content.text is None:
        # A streamed signature arrives as its own fragment and signs the
        # thinking block before it.
        previous = blocks[-1].wire if blocks else None
        if (
            content.protected_data
            and previous is not None
            and previous.get("type") == "thinking"
            and "signature" not in previous
        ):
            previous["signature"] = content.protected_data
        return
    thinking: dict[str, Any] = {"type": "thinking", "thinking": content.text}
    if content.protected_data:
        thinking["signature"] = content.protected_data
    blocks.append(_Block(thinking))


def _encode_content(content: Content) -> dict[str, Any] | None:
    match content.type:
        case "text":
            # The API rejects empty text blocks.
            return {"type": "text", "text": content.text} if content.text else None
        case "data" | "uri":
            if (image := _image_block(content)) is None:
                logger.debug("Ignoring unsupported data content media type: %s", content.media_type)
            return image
        case "function_call":
            return {"type": "tool_use", "id": content.call_id, "name": content.name, "input": content.parse_arguments()}
        case "function_result":
            return {
                "type": "tool_result",
                "tool_use_id": content.call_id,
                "content": _tool_result_blocks(content) or (content.result if content.result is not None else ""),
                "is_error": content.exception is not None,
            }
        case "mcp_server_tool_call":
            return {
                "type": "mcp_tool_use",
                "id": content.call_id,
                "name": content.tool_name,
                "server_name": content.server_name or "",
                "input": content.parse_arguments() or {},
            }
        case "mcp_server_tool_result":
            return {
                "type": "mcp_tool_result",
                "tool_use_id": content.call_id,
                "content": content.output if content.output is not None else "",
            }
        case _:
            logger.debug("Ignoring unsupported content type: %s", content.type)
            return None


def _tool_result_blocks(result: Content) -> list[dict[str, Any]]:
    """The text and image items of a tool result; other items are not sent."""
    blocks: list[dict[str, Any]] = []
    for item in result.items or ():
        if item.type == "text":
            blocks.append({"type": "text", "text": item.text or ""})
        elif item.type in ("data", "uri") and (image := _image_block(item)) is not None:
            blocks.append(image)
        else:
            logger.debug("Ignoring unsupported rich content media type in tool result: %s", item.media_type)
    return blocks


def _image_block(content: Content) -> dict[str, Any] | None:
    """A ``data`` or ``uri`` content as an image block; None when it is no image."""
    if not content.has_top_level_media_type("image"):
        return None
    if content.type == "data":
        return {
            "type": "image",
            "source": {"data": _get_data_bytes_as_str(content), "media_type": content.media_type, "type": "base64"},
        }
    return {"type": "image", "source": {"type": "url", "url": content.uri}}
