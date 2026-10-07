# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Chat history as Responses input items.

:func:`encode_message` turns one message into input items. Text, media and
file parts gather into ``message`` items; reasoning, function calls and
their outputs, and hosted tool contents each go out as a top-level item that
closes the message run before it. :func:`encode_content` encodes a single
content; one decoded from this provider's hosted item sends that item back
as it arrived (:mod:`.hosted`).

A request that continues a stored response leaves out what the service
already holds: reasoning, function calls and hosted items. Otherwise they
replay from local history the way :mod:`.replay` planned.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from chrys.foundation.hosted_tools import OPENAI_HOSTED_WIRE_ITEM_KEY, HostedToolFamily
from chrys.foundation.text.model_json import model_json
from chrys.kernel import OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY, Content
from chrys.kernel.exchanges import TOOL_CALL_CONTENT_TYPES, TOOL_RESULT_CONTENT_TYPES
from chrys.service.agent_middleware.events.hosted_tools import cross_provider_hosted_degradations
from chrys.service.llm.images import UNSUPPORTED_IMAGE_TEXT, wire_image

from .decode import ENVELOPE_FIELDS
from .hosted import (
    OPENAI_HOSTED_REPLAY_SHADOW_KEY,
    PENDING_IMAGE_RESULT_KEY,
    PENDING_MCP_OUTPUT_KEY,
    SHADOW_PLACEHOLDER_KEY,
)

if TYPE_CHECKING:
    from chrys.kernel import Annotation, Message, TextSpanRegion

logger = logging.getLogger(__name__)

# Nothing writes the shell output markers any more; saved history that has
# them still replays as shell output items.
OPENAI_SHELL_OUTPUT_TYPE_KEY = "openai.responses.shell.output_type"
OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL = "shell_call_output"
OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL = "local_shell_call_output"

# Every tool content is a top-level input item, never a message part.
_TOP_LEVEL_TYPES = TOOL_CALL_CONTENT_TYPES | TOOL_RESULT_CONTENT_TYPES
# What a stored response already holds; function outputs are always sent.
_HELD_BY_SERVICE = (_TOP_LEVEL_TYPES | {"text_reasoning"}) - {"function_result"}
# Response-only metadata the API refuses on input.
_DROPS_CREATED_BY = frozenset({"tool_search_call", "tool_search_output", "shell_call", "shell_call_output"})


def hosted_degradations(messages: Sequence[Message], *, provider: str, service_side: bool) -> Mapping[int, str | None]:
    """Hosted contents that cannot replay here, each with its stand-in summary.

    Another provider's hosted items never replay. Under local storage neither
    do this provider's image items: the server does not keep their ids.
    """
    return cross_provider_hosted_degradations(
        messages,
        target_provider=provider,
        unsafe_same_provider_families=() if service_side else (HostedToolFamily.IMAGE,),
    )


def encode_message(
    message: Message,
    *,
    provider: str,
    service_side: bool = True,
    reasoning_items: dict[int, dict[str, Any]] | None = None,
    dropped: Collection[int] = frozenset(),
    calls_without_id: Collection[int] = frozenset(),
    degradations: Mapping[int, str | None] | None = None,
) -> list[dict[str, Any]]:
    """The input items for *message*.

    The replay plan supplies, by content identity, the rebuilt *reasoning_items*
    (keyed by an occurrence's first content and consumed here), the contents
    to leave out and the function calls to send without their item id.
    *degradations* defaults to the ones this message has on its own.
    """
    if degradations is None:
        degradations = hosted_degradations([message], provider=provider, service_side=service_side)
    pending_reasoning = reasoning_items or {}
    replays_local_storage = "_attribution" in message.additional_properties
    items: list[dict[str, Any]] = []
    run = _message_item(message.role)

    def close_run() -> None:
        nonlocal run
        if "content" in run:
            items.append(run)
            run = _message_item(message.role)

    def add_item(item: dict[str, Any]) -> None:
        close_run()
        items.append(item)

    for content in message.contents:
        key = id(content)
        if key in degradations:
            if summary := degradations[key]:
                add_item(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": summary, "annotations": []}],
                    }
                )
            continue
        if key in dropped or (service_side and content.type in _HELD_BY_SERVICE):
            continue
        if content.type == "text_reasoning":
            if reasoning_item := pending_reasoning.pop(key, None):
                add_item(reasoning_item)
            continue
        encoded = encode_content(message.role, content, provider=provider, replays_local_storage=replays_local_storage)
        if not encoded:
            continue
        if content.type in _TOP_LEVEL_TYPES:
            if key in calls_without_id:
                encoded.pop("id", None)
            add_item(encoded)
            continue
        if message.role == "assistant":
            # Parts from different output messages must not share one item.
            envelope = _stored_envelope(content)
            if "content" in run and envelope != {name: run[name] for name in ENVELOPE_FIELDS if name in run}:
                close_run()
            run.update(envelope)
        run.setdefault("content", []).append(encoded)
    close_run()
    return items


def _message_item(role: str) -> dict[str, Any]:
    """An empty message item, which consecutive parts of one message fill."""
    return {"type": "message", "role": role}


def encode_content(
    role: str, content: Content, *, provider: str, replays_local_storage: bool = False
) -> dict[str, Any]:
    """*content* as an input item or a message part; ``{}`` when it has no wire form.

    *replays_local_storage* marks a message saved under local storage, whose
    function calls go back under their call id instead of the server's item id.
    """
    stored_item = content.additional_properties.get(OPENAI_HOSTED_WIRE_ITEM_KEY)
    if isinstance(stored_item, Mapping) and (content.hosted_provider == provider or content.type == "function_call"):
        return _replayed_item(stored_item)
    if content.additional_properties.get(OPENAI_HOSTED_REPLAY_SHADOW_KEY) is True:
        if content.type == "image_generation_tool_result" and (payload := image_replay_base64(content)) is not None:
            return {PENDING_IMAGE_RESULT_KEY: True, "image_id": content.image_id, "result": payload}
        return {SHADOW_PLACEHOLDER_KEY: True}
    match content.type:
        case "text":
            if role == "assistant":
                # Strict validators refuse an output_text part without annotations.
                return {
                    "type": "output_text",
                    "text": content.text,
                    "annotations": encode_annotations(content.annotations),
                }
            return {"type": "input_text", "text": content.text}
        case "text_reasoning":
            return _reasoning_item(content)
        case "data" | "uri":
            return _media_part(content)
        case "function_call":
            return _function_call_item(content, replays_local_storage)
        case "function_result":
            return _function_output_item(content, provider)
        case "mcp_server_tool_call" | "mcp_server_tool_result" if not content.call_id:
            return {}
        case "mcp_server_tool_call":
            server, tool = content.server_name or "", content.tool_name or ""
            arguments = arguments_text(content.arguments)
            return {
                "type": "mcp_call",
                "id": content.call_id,
                "server_label": server,
                "name": tool,
                "arguments": arguments,
            }
        case "mcp_server_tool_result":
            return {PENDING_MCP_OUTPUT_KEY: True, "call_id": content.call_id, "output": mcp_output_text(content.output)}
        case "hosted_file":
            # input_file is input-only and refused inside an assistant message.
            # There a hosted file is a hosted tool's citation, which the text
            # annotations already carry.
            if role == "assistant":
                return {}
            return {"type": "input_file", "file_id": content.file_id}
        case other:
            logger.debug("Unsupported content type passed (type: %s)", other)
            return {}


def _replayed_item(stored: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(stored)
    item_type = item.get("type")
    if item_type == "image_generation_call":
        # The server does not keep image item ids under local storage, so a
        # replayed one fails; degradation swaps the pair for a summary first,
        # and this fails closed when it did not.
        return {}
    if item_type in _DROPS_CREATED_BY:
        item.pop("created_by", None)
    if item_type == "shell_call_output":
        item.pop("status", None)
    return item


def marked_reasoning_text(content: Content) -> str | None:
    """The ``reasoning_text`` part of a reasoning content, if it has one.

    Its ``reasoning_text`` marker is True when the content's own text is
    that part, or holds the text itself.
    """
    marker = content.additional_properties.get("reasoning_text")
    if not marker:
        return None
    text = content.text if marker is True else marker
    return text if isinstance(text, str) and text else None


def _reasoning_item(content: Content) -> dict[str, Any]:
    item: dict[str, Any] = {"type": "reasoning", "summary": []}
    if content.id:
        item["id"] = content.id
    properties = content.additional_properties
    if status := properties.get("status"):
        item["status"] = status
    if text := marked_reasoning_text(content):
        item["content"] = [{"type": "reasoning_text", "text": text}]
    if encrypted := properties.get("encrypted_content"):
        item["encrypted_content"] = encrypted
    if content.text and properties.get("reasoning_text") is not True:
        item["summary"].append({"type": "summary_text", "text": content.text})
    return item


def _media_part(content: Content) -> dict[str, Any]:
    # Older sessions kept MCP links as their URL, some without a type: only
    # an image is sent by URL, audio and files go inline or not at all.
    if content.media_type is None:
        return {}
    properties = content.additional_properties
    if content.has_top_level_media_type("image"):
        if (image := wire_image(content)) is None:
            return {"type": "input_text", "text": UNSUPPORTED_IMAGE_TEXT}
        part: dict[str, Any] = {
            "type": "input_image",
            "image_url": image.uri,
            "detail": properties.get("detail", "auto"),
        }
        if (file_id := properties.get("file_id")) is not None:
            part["file_id"] = file_id
        return part
    if not (content.uri or "").startswith("data:"):
        return {}
    if content.has_top_level_media_type("audio"):
        media_type = content.media_type or ""
        audio_format = "wav" if "wav" in media_type else "mp3" if "mp3" in media_type else None
        if audio_format is None:
            logger.warning("Unsupported audio media type: %s", content.media_type)
            return {}
        return {"type": "input_audio", "input_audio": {"data": content.uri, "format": audio_format}}
    if content.has_top_level_media_type("application"):
        part = {"type": "input_file", "file_data": content.uri}
        if filename := properties.get("filename"):
            part["filename"] = filename
        return part
    return {}


def _function_call_item(content: Content, replays_local_storage: bool) -> dict[str, Any]:
    if not content.call_id:
        logger.warning("FunctionCallContent missing call_id for function '%s'", content.name)
        return {}
    item_id = content.call_id
    live_id = content.additional_properties.get("fc_id")
    if not replays_local_storage and isinstance(live_id, str) and live_id:
        item_id = live_id
    # The API wants function-call item ids to start with "fc_".
    if not item_id.startswith("fc_"):
        item_id = f"fc_{item_id}"
    item = {
        "call_id": content.call_id,
        "id": item_id,
        "type": "function_call",
        "name": content.name,
        # History can hold dict arguments (another provider's tool input, an
        # edited approval); the API takes only a string.
        "arguments": arguments_text(content.arguments),
    }
    if status := content.additional_properties.get("status"):
        item["status"] = status
    return item


def _function_output_item(content: Content, provider: str) -> dict[str, Any]:
    shell_output_type = content.additional_properties.get(OPENAI_SHELL_OUTPUT_TYPE_KEY)
    if shell_output_type == OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL:
        return {
            "call_id": content.call_id,
            "type": OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL,
            "output": _shell_call_output(content),
        }
    if shell_output_type == OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL:
        # The SDK names the field ``id``, but it holds the model's call
        # reference, not a server item id.
        return {
            "id": content.call_id,
            "type": OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL,
            "output": _local_shell_output(content),
        }
    output: str | list[dict[str, Any]] = content.result or ""
    if content.items and any(item.type in ("data", "uri") for item in content.items):
        parts = [
            {"type": "input_text", "text": item.text or ""}
            if item.type == "text"
            else encode_content("user", item, provider=provider)
            for item in content.items
        ]
        if parts := [part for part in parts if part]:
            output = parts
    return {"call_id": content.call_id, "type": "function_call_output", "output": output}


def _shell_result_payload(content: Content) -> dict[str, Any]:
    if isinstance(content.result, Mapping):
        return dict(content.result)
    return {"stdout": "" if content.result is None else str(content.result)}


def _local_shell_output(content: Content) -> str:
    payload = _shell_result_payload(content)
    # A failed call's exception is a record for people, never for the model:
    # stdout already holds the error result the model reads.
    payload.setdefault("exit_code", 1 if content.exception is not None else 0)
    return model_json(payload)


def _shell_call_output(content: Content) -> list[dict[str, Any]]:
    payload = _shell_result_payload(content)
    native = payload.get("output")
    # A tool that already returns shell output entries passes them through.
    if isinstance(native, list) and all(isinstance(entry, Mapping) for entry in native):
        return [dict(entry) for entry in native]
    # Only what the tool returned; the exception is a record for people.
    stdout = str(payload.get("stdout", ""))
    stderr = str(payload.get("stderr", ""))
    failed = 1 if content.exception is not None else 0
    if payload.get("timed_out", False):
        outcome: dict[str, Any] = {"type": "timeout"}
    else:
        outcome = {"type": "exit", "exit_code": _exit_code(payload.get("exit_code"), failed)}
    return [{"stdout": stdout, "stderr": stderr, "outcome": outcome}]


def _exit_code(raw: Any, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except TypeError, ValueError:
        return default


def arguments_text(arguments: Any) -> str:
    """Call arguments as the JSON string the API takes."""
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    try:
        return model_json(arguments)
    except TypeError, ValueError:
        return str(arguments)


def mcp_output_text(output: Any) -> str:
    """The ``output`` string of a replayed ``mcp_call``.

    Text entries (strings, text contents, ``{"text": ...}`` mappings) are
    joined; anything else becomes JSON, never a Python repr.
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, Sequence) and not isinstance(output, (bytes, bytearray)):
        return "".join(_entry_text(entry) for entry in output)
    return model_json(output, default=str)


def _entry_text(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(text := getattr(entry, "text", None), str):
        return text
    if isinstance(entry, Mapping) and isinstance(text := entry.get("text"), str):
        return text
    return model_json(entry, default=str)


def image_replay_base64(content: Content) -> str | None:
    """The base64 payload of the last complete image in an image result."""
    outputs = content.outputs if isinstance(content.outputs, list) else [content.outputs]
    for output in reversed(outputs):
        if isinstance(output, Content) and output.additional_properties.get("is_partial_image") is not True:
            _, marker, payload = (output.uri or "").partition(";base64,")
            if marker:
                return payload
    return None


def _stored_envelope(content: Content) -> dict[str, str]:
    """The output-message fields a decoded text part kept."""
    stored = content.additional_properties.get(OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY)
    if not isinstance(stored, Mapping):
        return {}
    return {name: value for name in ENVELOPE_FIELDS if isinstance(value := stored.get(name), str) and value}


def encode_annotations(annotations: Sequence[Annotation] | None) -> list[dict[str, Any]]:
    """Citations as ``output_text`` annotations again.

    Decoding folds file paths, file and container-file citations and URL
    citations into one citation shape; which fields are set tells them apart
    here. A wire annotation holds one span, so a citation over several
    regions becomes several entries; regions without integer bounds are
    skipped.
    """
    encoded: list[dict[str, Any]] = []
    for annotation in annotations or ():
        if annotation.get("type") == "citation":
            encoded.extend(_citation_entries(annotation))
    return encoded


def _citation_entries(annotation: Annotation) -> list[dict[str, Any]]:
    properties = annotation.get("additional_properties") or {}
    regions = annotation.get("annotated_regions") or []
    file_id = annotation.get("file_id")
    url = annotation.get("url")
    container_id = properties.get("container_id")
    if container_id and file_id:
        cited = {"type": "container_file_citation", "container_id": container_id, "file_id": file_id}
        named = {"filename": url} if url else {}
        return [{**cited, "start_index": start, "end_index": end, **named} for start, end in _spans(regions)]
    if url and not file_id and regions:
        title = annotation.get("title") or ""
        return [
            {"type": "url_citation", "url": url, "title": title, "start_index": start, "end_index": end}
            for start, end in _spans(regions)
        ]
    if not file_id:
        return []
    entry = (
        {"type": "file_citation", "file_id": file_id, "filename": url}
        if url
        else {"type": "file_path", "file_id": file_id}
    )
    if (index := properties.get("index")) is not None:
        entry["index"] = index
    return [entry]


def _spans(regions: Sequence[TextSpanRegion]) -> Iterator[tuple[int, int]]:
    for region in regions:
        start = region.get("start_index")
        end = region.get("end_index")
        if isinstance(start, int) and isinstance(end, int):
            yield start, end


def answer_apply_patch_calls(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Follow each unanswered ``apply_patch_call`` with a failed output.

    The API refuses a history holding an apply-patch call without its output,
    and Chrys does not execute the tool, so replay answers each such call
    with an explicit failure. *items* is one exchange: a call id can repeat
    in a later exchange, and an answer here must not cover a call there.

    An executor would apply the call's ``operation`` (create, update or
    delete a file from a V4A diff, not a unified diff) through the local
    file-mutation pipeline and answer with a real ``apply_patch_call_output``,
    which this then leaves alone.
    """
    answered = {
        item.get("call_id") for item in items if item.get("type") == "apply_patch_call_output" and item.get("call_id")
    }
    if not any(item.get("type") == "apply_patch_call" and item.get("call_id") not in answered for item in items):
        return items
    completed: list[dict[str, Any]] = []
    for item in items:
        completed.append(item)
        call_id = item.get("call_id")
        if item.get("type") == "apply_patch_call" and call_id and call_id not in answered:
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
