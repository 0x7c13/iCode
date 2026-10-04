# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Encode chat history as Messages API messages.

A chat message becomes one wire message per run of blocks that share a wire
role, in their original order: tool calls go out as ``assistant`` and local
tool results as ``user`` whatever message carries them; other blocks, hosted
results included, keep their message's role. Repairs keep the request
valid: thinking without a signature is dropped (the API rejects it), and so
are blank text and a wire message left with no blocks; a tool-call id another
provider minted (``functions.read_file:0``) is sent as an id the API accepts,
the same for the call and its result. Thinking another endpoint issued is
left out: only its issuer can check the signature, and so is thinking a
request was accepted without after the service refused it as bound to a
different conversation (:data:`THINKING_STRIPPED_KEY`).

Hosted-tool history from another provider is replaced by the neutral summary
:func:`cross_provider_hosted_degradations` writes, sent as assistant context.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

from chrys.foundation.hosted_tools import ANTHROPIC_HOSTED_WIRE_BLOCK_KEY
from chrys.foundation.models.history_markers import ANTHROPIC_THINKING_STRIPPED_KEY as THINKING_STRIPPED_KEY
from chrys.foundation.reasoning_origin import replays_to
from chrys.kernel._content import _ANTHROPIC_REDACTED_THINKING_KEY
from chrys.service.agent_middleware.events.hosted_tools import cross_provider_hosted_degradations
from chrys.service.llm.images import UNSUPPORTED_IMAGE_TEXT, wire_image

if TYPE_CHECKING:
    from chrys.foundation.reasoning_origin import ReasoningOrigin
    from chrys.kernel import Content, Message

logger = logging.getLogger(__name__)

# Wire roles fixed by a block's type; other blocks take their message's role.
_ROLE_OF_BLOCK_TYPE: Final[Mapping[str, str]] = {
    "tool_use": "assistant",
    "mcp_tool_use": "assistant",
    "server_tool_use": "assistant",
    "tool_result": "user",
}


# The tool-use ids the API accepts.
_TOOL_ID: Final = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class _Block:
    """One encoded content block."""

    wire: dict[str, Any]
    assistant_context: bool = False
    """Sent as ``assistant`` whatever its type: a summary of foreign hosted-tool history."""
    thinking: tuple[Content, ...] = ()
    """The reasoning contents a thinking or redacted thinking block is encoded from, a signature it took included."""

    def role(self, message_role: str) -> str:
        if self.assistant_context:
            return "assistant"
        return _ROLE_OF_BLOCK_TYPE.get(self.wire.get("type"), message_role)


@dataclass(frozen=True, slots=True)
class EncodedHistory:
    """The wire messages of a history and the reasoning contents their thinking blocks replay."""

    messages: list[dict[str, Any]]
    thinking: tuple[Content, ...]
    """Every content a sent thinking or redacted thinking block is encoded from, each once, in order."""


def encode_messages(messages: Sequence[Message], *, origin: ReasoningOrigin | None = None) -> list[dict[str, Any]]:
    """Encode *messages* after a leading system message, which the request sends as ``system``.

    *origin* is the endpoint the request goes to: thinking, signatures and
    redacted thinking another endpoint issued are left out.
    """
    return encode_history(messages, origin=origin).messages


def encode_history(
    messages: Sequence[Message], *, origin: ReasoningOrigin | None = None, skip: Collection[Content] = ()
) -> EncodedHistory:
    """Encode *messages* like :func:`encode_messages`, leaving out the thinking of the contents in *skip*.

    A thinking block is left out when any content it is encoded from is in
    *skip* (by identity) or carries :data:`THINKING_STRIPPED_KEY`. Blocks are
    matched with their signatures first and left out after, so a signature
    never moves to the thinking before the block left out.
    """
    history = messages[1:] if messages and messages[0].role == "system" else messages
    summaries = cross_provider_hosted_degradations(history, target_provider="anthropic")
    skipped = {id(content) for content in skip}
    wire_messages: list[dict[str, Any]] = []
    thinking: dict[int, Content] = {}
    for message in history:
        role = "assistant" if message.role == "assistant" else "user"
        blocks = [
            block
            for block in _encode_blocks(message, summaries, origin)
            if not any(id(content) in skipped or _was_stripped(content) for content in block.thinking)
        ]
        for block in blocks:
            thinking.update((id(content), content) for content in block.thinking)
        wire_messages.extend(_by_role(blocks, role))
    return EncodedHistory(wire_messages, tuple(thinking.values()))


def _was_stripped(content: Content) -> bool:
    return content.additional_properties.get(THINKING_STRIPPED_KEY) is True


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


def _encode_blocks(
    message: Message, summaries: Mapping[int, str | None], origin: ReasoningOrigin | None
) -> list[_Block]:
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
            if replays_to(content.additional_properties, origin):
                _add_thinking(content, blocks)
        elif (wire := _encode_content(content)) is not None:
            blocks.append(_Block(wire))
    return [block for block in blocks if not _is_unsigned_thinking(block.wire)]


def _is_unsigned_thinking(wire: Mapping[str, Any]) -> bool:
    # Some gateways omit thinking signatures, and replayed thinking without one
    # fails the request ("signature: Field required"). The reasoning is dropped
    # rather than replayed as visible text.
    return wire.get("type") == "thinking" and not wire.get("signature")


def _add_thinking(content: Content, blocks: list[_Block]) -> None:
    properties = content.additional_properties
    if properties.get(_ANTHROPIC_REDACTED_THINKING_KEY):
        blocks.append(_Block({"type": "redacted_thinking", "data": content.protected_data}, thinking=(content,)))
        return
    if content.id or properties.get("reasoning_text") or properties.get("openai_reasoning_format"):
        # Another protocol's reasoning (Responses items carry an id and a
        # marker, Chat Completions dialects stamp their format): sending it as
        # thinking would forge a signature.
        return
    if content.text is None and content.protected_data:
        # A streamed signature arrives as its own fragment and signs the
        # thinking block before it.
        previous = blocks[-1] if blocks else None
        if previous is not None and previous.wire.get("type") == "thinking" and "signature" not in previous.wire:
            previous.wire["signature"] = content.protected_data
            blocks[-1] = replace(previous, thinking=(*previous.thinking, content))
        return
    # Thinking streamed without its text (an empty start and an empty delta)
    # assembles with no text and the start's empty signature: it is an empty
    # block, which the signature fragment after it signs; without one, the
    # block is unsigned and dropped.
    thinking: dict[str, Any] = {"type": "thinking", "thinking": content.text or ""}
    if content.protected_data:
        thinking["signature"] = content.protected_data
    blocks.append(_Block(thinking, thinking=(content,)))


def _encode_content(content: Content) -> dict[str, Any] | None:
    match content.type:
        case "text":
            # The API rejects blank text blocks.
            return {"type": "text", "text": content.text} if content.text and not content.text.isspace() else None
        case "data" | "uri":
            if (image := _image_block(content)) is None:
                logger.debug("Ignoring unsupported data content media type: %s", content.media_type)
            return image
        case "function_call":
            return {
                "type": "tool_use",
                "id": _wire_tool_id(content.call_id),
                "name": content.name,
                "input": content.parse_arguments(),
            }
        case "function_result":
            return {
                "type": "tool_result",
                "tool_use_id": _wire_tool_id(content.call_id),
                "content": _tool_result_blocks(content) or _tool_result_text(content.result),
                "is_error": content.exception is not None,
            }
        case "mcp_server_tool_call":
            return {
                "type": "mcp_tool_use",
                "id": _wire_tool_id(content.call_id),
                "name": content.tool_name,
                "server_name": content.server_name or "",
                "input": content.parse_arguments() or {},
            }
        case "mcp_server_tool_result":
            return {
                "type": "mcp_tool_result",
                "tool_use_id": _wire_tool_id(content.call_id),
                "content": content.output if content.output is not None else "",
            }
        case _:
            logger.debug("Ignoring unsupported content type: %s", content.type)
            return None


def _wire_tool_id(call_id: str | None) -> str | None:
    """*call_id* as an id the API accepts.

    An id it rejects maps to one derived from its value, so a call and its
    result keep matching. An empty id is sent as it is.
    """
    if not call_id or _TOOL_ID.fullmatch(call_id):
        return call_id
    digest = hashlib.sha256(call_id.encode("utf-8", "surrogatepass")).hexdigest()
    return f"toolu_chrys_{digest[:32]}"


def _tool_result_text(result: Any) -> Any:
    """A tool result's own value, or ``""`` for none or blank text, which the API rejects."""
    if result is None or (isinstance(result, str) and result.isspace()):
        return ""
    return result


def _tool_result_blocks(result: Content) -> list[dict[str, Any]]:
    """The text and image items of a tool result; blank text and other items are not sent."""
    blocks: list[dict[str, Any]] = []
    for item in result.items or ():
        if item.type == "text":
            if item.text and not item.text.isspace():
                blocks.append({"type": "text", "text": item.text})
        elif item.type in ("data", "uri") and (image := _image_block(item)) is not None:
            blocks.append(image)
        else:
            logger.debug("Ignoring unsupported rich content media type in tool result: %s", item.media_type)
    return blocks


def _image_block(content: Content) -> dict[str, Any] | None:
    """A ``data`` or ``uri`` content as an image block; None when it is no image.

    An image the API can't read goes out as a text block saying so; a link
    saved without a type (older sessions kept MCP links that way) is no image.
    """
    if content.media_type is None or not content.has_top_level_media_type("image"):
        return None
    if (image := wire_image(content)) is None:
        return {"type": "text", "text": UNSUPPORTED_IMAGE_TEXT}
    if image.data is not None:
        return {"type": "image", "source": {"data": image.data, "media_type": image.media_type, "type": "base64"}}
    return {"type": "image", "source": {"type": "url", "url": image.uri}}
