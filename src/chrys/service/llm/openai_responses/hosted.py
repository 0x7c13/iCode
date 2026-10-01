# Copyright (c) Microsoft. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Microsoft Agent Framework (MIT License; see NOTICE).

"""Hosted-tool items: decoded as contents, merged back when history replays.

A tool the server runs (search, tool search, MCP, code, image generation,
shell, or an unknown item the server says it executed) arrives as one output
item. :func:`decode_hosted_item` turns it into a call content plus, when the
item already holds its outcome, a result content. The blocking parse and both
stream events that carry whole items (``output_item.added`` and ``.done``)
decode through it.

A call keeps its wire item under ``OPENAI_HOSTED_WIRE_ITEM_KEY`` so history
can send it back unchanged; a result decoded from the same item only carries
``OPENAI_HOSTED_REPLAY_SHADOW_KEY``, because the call already replays it.
MCP output and generated images travel back inside their call item, so
history writes them as request-local markers that
:func:`coalesce_pending_results` folds into the call before the request
leaves.
"""

from __future__ import annotations

import logging
import shlex
from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from chrys.foundation.hosted_tools import (
    OPENAI_HOSTED_WIRE_ITEM_KEY,
    HostedRetrySafety,
    HostedToolFamily,
    HostedToolPhase,
    HostedToolStatus,
    normalize_hosted_tool_status,
)
from chrys.kernel import Content, detect_media_type_from_base64
from chrys.kernel.exchanges import namespaced_pairing_key

if TYPE_CHECKING:
    from chrys.kernel.exchanges import PairingKey

logger = logging.getLogger(__name__)

OPENAI_HOSTED_REPLAY_SHADOW_KEY = "openai.responses.replay_shadow"

# Request-local markers: history writes them, coalesce_pending_results
# consumes them, and none ever reaches the wire.
PENDING_MCP_OUTPUT_KEY = "__chrys_pending_mcp_result__"
PENDING_IMAGE_RESULT_KEY = "__chrys_image_generation_result__"
SHADOW_PLACEHOLDER_KEY = "__chrys_hosted_replay_shadow__"

_FAILED_STATUSES = frozenset({"failed", "incomplete", "cancelled", "canceled"})
# Replayed call items a pending result marker can complete, by wire type.
_COMPLETABLE_CALLS = {"mcp_call": "mcp_server_tool_call", "image_generation_call": "image_generation_tool_call"}


def to_payload(value: Any) -> Any:
    """*value* as plain JSON data; SDK models leave out unset and None fields."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    if isinstance(value, Mapping):
        return {str(key): to_payload(entry) for key, entry in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_payload(entry) for entry in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): to_payload(entry)
            for key, entry in vars(value).items()
            if not key.startswith("_") and entry is not None
        }
    return value


def item_phase(item: Any) -> str:
    """SNAPSHOT while *item* is pending or running, else TERMINAL.

    Background and polled responses carry items that are still running;
    reading those as terminal would publish a result early and then drop the
    real one as late. A missing or unknown status reads as terminal, since
    finished responses often leave item status out.
    """
    status = getattr(item, "status", None)
    normalized = normalize_hosted_tool_status(status if isinstance(status, str) else None)
    if normalized in (HostedToolStatus.PENDING, HostedToolStatus.RUNNING):
        return HostedToolPhase.SNAPSHOT
    return HostedToolPhase.TERMINAL


def item_properties(
    item: Any, *, shadow: bool = False, error: Any = None, omit: tuple[str, ...] = ()
) -> dict[str, Any]:
    """What a decoded content persists about *item*.

    The wire item without the *omit* keys (or only the shadow flag), then the
    failure flag with the error payload, or the flag alone for a failed status.
    """
    properties: dict[str, Any] = {}
    if shadow:
        properties[OPENAI_HOSTED_REPLAY_SHADOW_KEY] = True
    else:
        wire_item = to_payload(item)
        if isinstance(wire_item, dict):
            for key in omit:
                wire_item.pop(key, None)
        properties[OPENAI_HOSTED_WIRE_ITEM_KEY] = wire_item
    if error is not None:
        properties["is_error"] = True
        properties["error"] = to_payload(error)
    elif str(getattr(item, "status", "")).lower() in _FAILED_STATUSES:
        properties["is_error"] = True
    return properties


def image_data_uri(image_base64: str) -> str:
    """A data URI for a generated image, PNG unless the payload says otherwise."""
    media_type = detect_media_type_from_base64(data_str=image_base64) or "image/png"
    return f"data:{media_type};base64,{image_base64}"


def _provider_fields(
    item: Any,
    provider: str,
    *,
    item_type: str | None,
    item_id: Any,
    phase: str,
    safety: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    """Keywords every hosted content decoded from *item* passes to its factory."""
    return {
        "hosted_provider": provider,
        "provider_item_type": item_type,
        "provider_item_id": item_id,
        "provider_phase": phase,
        "provider_status": getattr(item, "status", None),
        "retry_safety": safety,
        "additional_properties": properties,
        "raw_representation": item,
    }


def decode_hosted_item(item: Any, provider: str, *, phase: str | None = None) -> list[Content] | None:
    """The call content of a hosted *item*, then its result when it has one.

    *phase* replaces the item's own phase on the call (the stream marks calls
    it has only seen start). Returns None, after a debug log, for an unknown
    item the server does not say it executed.
    """
    match getattr(item, "type", None):
        case "web_search_call" | "file_search_call":
            return _search(item, provider, phase)
        case "tool_search_call":
            return [_tool_search_call(item, provider, phase)]
        case "tool_search_output":
            return [_tool_search_output(item, provider)]
        case "mcp_call":
            return _mcp(item, provider, phase)
        case "code_interpreter_call":
            return _code(item, provider, phase)
        case "image_generation_call":
            return _image(item, provider, phase)
        case "shell_call" | "local_shell_call" | "shell_call_output":
            return [_shell(item, provider, phase)]
        case item_type:
            if getattr(item, "execution", None) == "server" or getattr(item, "server_execution", None) is not None:
                return _server_executed(item, provider, phase)
            logger.debug(
                "responses_parser_unrecognized_server_item provider=%s item_type=%s execution_evidence=false",
                provider,
                item_type,
            )
            return None


def _search(item: Any, provider: str, phase: str | None) -> list[Content]:
    item_type = getattr(item, "type", "")
    item_id = getattr(item, "id", None)
    call_id = item_id or getattr(item, "call_id", None) or ""
    tool_name = "web_search" if item_type == "web_search_call" else "file_search"
    # Each side serializes on its own: the call and its result never share a dict.
    if item_type == "web_search_call":
        arguments = to_payload(getattr(item, "action", None))
        outcome = {"action": to_payload(getattr(item, "action", None))}
    else:
        arguments = {"queries": list(getattr(item, "queries", []) or [])}
        outcome = {"results": to_payload(getattr(item, "results", None))}
    status = getattr(item, "status", None)
    call = Content.from_search_tool_call(
        call_id=call_id,
        tool_name=tool_name,
        arguments=arguments,
        status=status,
        **_provider_fields(
            item,
            provider,
            item_type=item_type,
            item_id=item_id,
            phase=phase if phase is not None else item_phase(item),
            safety=HostedRetrySafety.READ_ONLY,
            properties=item_properties(item),
        ),
    )
    result = Content.from_search_tool_result(
        call_id=call_id,
        tool_name=tool_name,
        result=outcome,
        status=status,
        **_provider_fields(
            item,
            provider,
            item_type=item_type,
            item_id=item_id,
            phase=item_phase(item),
            safety=HostedRetrySafety.READ_ONLY,
            properties=item_properties(item, shadow=True),
        ),
    )
    return [call, result]


def _tool_search_call(item: Any, provider: str, phase: str | None) -> Content:
    item_id = getattr(item, "id", None)
    return Content.from_hosted_tool_call(
        call_id=getattr(item, "call_id", None) or item_id,
        tool_name="tool_search",
        arguments=to_payload(getattr(item, "arguments", None)),
        status=getattr(item, "status", None),
        hosted_family=HostedToolFamily.TOOL_DISCOVERY,
        **_provider_fields(
            item,
            provider,
            item_type=str(getattr(item, "type", "tool_search_call")),
            item_id=item_id,
            phase=phase if phase is not None else item_phase(item),
            safety=HostedRetrySafety.READ_ONLY,
            properties=item_properties(item),
        ),
    )


def _tool_search_output(item: Any, provider: str) -> Content:
    # Its own output item, so unlike other results it replays the wire item.
    item_id = getattr(item, "id", None)
    return Content.from_hosted_tool_result(
        call_id=getattr(item, "call_id", None) or item_id,
        tool_name="tool_search",
        result={"tools": to_payload(getattr(item, "tools", []))},
        status=getattr(item, "status", None),
        hosted_family=HostedToolFamily.TOOL_DISCOVERY,
        **_provider_fields(
            item,
            provider,
            item_type=str(getattr(item, "type", "tool_search_output")),
            item_id=item_id,
            phase=item_phase(item),
            safety=HostedRetrySafety.READ_ONLY,
            properties=item_properties(item),
        ),
    )


def _server_executed(item: Any, provider: str, phase: str | None) -> list[Content]:
    """An unknown item the server ran: kept with unknown retry safety."""
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
        arguments=to_payload(arguments),
        status=status,
        **_provider_fields(
            item,
            provider,
            item_type=item_type,
            item_id=item_id,
            phase=phase if phase is not None else item_phase(item),
            safety=HostedRetrySafety.UNKNOWN,
            properties=item_properties(item),
        ),
    )
    outcomes = (getattr(item, "output", None), getattr(item, "result", None), getattr(item, "outputs", None))
    outcome = next((to_payload(value) for value in outcomes if value is not None), None)
    if outcome is None:
        return [call]
    result = Content.from_hosted_tool_result(
        call_id=call_id,
        tool_name=tool_name,
        result=outcome,
        status=status,
        **_provider_fields(
            item,
            provider,
            item_type=item_type,
            item_id=item_id,
            phase=item_phase(item),
            safety=HostedRetrySafety.UNKNOWN,
            properties=item_properties(item, shadow=True),
        ),
    )
    return [call, result]


def _mcp(item: Any, provider: str, phase: str | None) -> list[Content]:
    item_id = getattr(item, "id", None)
    call_id = item_id or getattr(item, "call_id", None) or ""
    call = Content.from_mcp_server_tool_call(
        call_id=call_id,
        tool_name=getattr(item, "name", "") or "",
        server_name=getattr(item, "server_label", None),
        arguments=getattr(item, "arguments", None),
        **_provider_fields(
            item,
            provider,
            item_type="mcp_call",
            item_id=item_id,
            phase=phase if phase is not None else item_phase(item),
            safety=HostedRetrySafety.SIDE_EFFECTFUL,
            properties=item_properties(item),
        ),
    )
    output = getattr(item, "output", None)
    error = getattr(item, "error", None)
    if output is None and error is None:
        return [call]
    if error is not None:
        outcome = Content.from_error(message=str(error), raw_representation=item)
    else:
        outcome = Content.from_text(text=str(output), raw_representation=item)
    result = Content.from_mcp_server_tool_result(
        call_id=call_id,
        output=[outcome],
        **_provider_fields(
            item,
            provider,
            item_type="mcp_call",
            item_id=item_id,
            phase=item_phase(item),
            safety=HostedRetrySafety.SIDE_EFFECTFUL,
            properties=item_properties(item, shadow=True, error=error),
        ),
    )
    return [call, result]


def _code(item: Any, provider: str, phase: str | None) -> list[Content]:
    item_id = getattr(item, "id", None)
    call_id = getattr(item, "call_id", None) or item_id
    code = getattr(item, "code", None)
    call = Content.from_code_interpreter_tool_call(
        call_id=call_id,
        inputs=[Content.from_text(text=code, raw_representation=item)] if code else [],
        **_provider_fields(
            item,
            provider,
            item_type="code_interpreter_call",
            item_id=item_id,
            phase=phase if phase is not None else item_phase(item),
            safety=HostedRetrySafety.SANDBOXED,
            properties=item_properties(item),
        ),
    )
    outputs = _code_outputs(item, provider)
    if not outputs:
        return [call]
    result = Content.from_code_interpreter_tool_result(
        call_id=call_id,
        outputs=outputs,
        **_provider_fields(
            item,
            provider,
            item_type="code_interpreter_call",
            item_id=item_id,
            phase=item_phase(item),
            safety=HostedRetrySafety.SANDBOXED,
            properties=item_properties(item, shadow=True),
        ),
    )
    return [call, result]


def _code_outputs(item: Any, provider: str) -> list[Content]:
    """Logs, images and files a code-interpreter run produced."""
    outputs: list[Content] = []
    for entry in getattr(item, "outputs", None) or []:
        match getattr(entry, "type", None):
            case "logs":
                outputs.append(Content.from_text(text=entry.logs, raw_representation=entry))
            case "image":
                outputs.append(Content.from_uri(uri=entry.url, raw_representation=entry, media_type="image"))
            case "file" | "hosted_file":
                if file_id := getattr(entry, "file_id", None):
                    outputs.append(
                        Content.from_hosted_file(
                            file_id=file_id, name=getattr(entry, "filename", None), raw_representation=entry
                        )
                    )
            case output_type:
                logger.debug(
                    "responses_parser_unrecognized_code_output provider=%s output_type=%s", provider, output_type
                )
    return outputs


def _image(item: Any, provider: str, phase: str | None) -> list[Content]:
    item_id = getattr(item, "id", None)
    image_base64 = getattr(item, "result", None)
    if phase is None:
        # A finished image can still report status "generating": the
        # payload is the stronger evidence that it is done.
        phase = HostedToolPhase.TERMINAL if image_base64 is not None else item_phase(item)
    # The call's wire item leaves the payload out; the result holds the one copy.
    call = Content.from_image_generation_tool_call(
        image_id=item_id,
        **_provider_fields(
            item,
            provider,
            item_type="image_generation_call",
            item_id=item_id,
            phase=phase,
            safety=HostedRetrySafety.SANDBOXED,
            properties=item_properties(item, omit=("result",)),
        ),
    )
    if image_base64 is None:
        return [call]
    result = Content.from_image_generation_tool_result(
        image_id=item_id,
        outputs=[Content.from_uri(uri=image_data_uri(image_base64), raw_representation=image_base64)],
        **_provider_fields(
            item,
            provider,
            item_type="image_generation_call",
            item_id=item_id,
            phase=phase,
            safety=HostedRetrySafety.SANDBOXED,
            properties=item_properties(item, shadow=True),
        ),
    )
    return [call, result]


def _shell(item: Any, provider: str, phase: str | None) -> Content:
    """A shell call, a local shell call (argv joined into one command), or a shell result."""
    item_type = getattr(item, "type", None)
    call_id = getattr(item, "call_id", None) or ""
    status = getattr(item, "status", None)
    action = getattr(item, "action", None)
    if item_type == "shell_call_output":
        outputs = [_shell_command_output(entry) for entry in getattr(item, "output", []) or []]
        return Content.from_shell_tool_result(
            call_id=call_id,
            outputs=outputs,
            max_output_length=getattr(item, "max_output_length", None),
            **_provider_fields(
                item,
                provider,
                item_type=item_type,
                item_id=getattr(item, "id", None),
                phase=item_phase(item),
                safety=HostedRetrySafety.SIDE_EFFECTFUL,
                properties=item_properties(item, error=_shell_failure(outputs)),
            ),
        )
    fields = _provider_fields(
        item,
        provider,
        item_type=item_type,
        item_id=getattr(item, "id", None),
        phase=phase if phase is not None else item_phase(item),
        safety=HostedRetrySafety.SIDE_EFFECTFUL,
        properties=item_properties(item),
    )
    if item_type == "local_shell_call":
        argv = list(getattr(action, "command", []) or [])
        command = shlex.join(argv) if argv else ""
        return Content.from_shell_tool_call(
            call_id=call_id,
            commands=[command] if command else [],
            timeout_ms=getattr(action, "timeout_ms", None),
            status=status,
            **fields,
        )
    commands: list[str] = []
    timeout_ms: int | None = None
    max_output_length: int | None = None
    if action:
        commands = list(getattr(action, "commands", []) or [])
        timeout_ms = getattr(action, "timeout_ms", None)
        max_output_length = getattr(action, "max_output_length", None)
    return Content.from_shell_tool_call(
        call_id=call_id,
        commands=commands,
        timeout_ms=timeout_ms,
        max_output_length=max_output_length,
        status=status,
        **fields,
    )


def _shell_command_output(entry: Any) -> Content:
    exit_code: int | None = None
    timed_out: bool | None = None
    if outcome := getattr(entry, "outcome", None):
        match getattr(outcome, "type", None):
            case "exit":
                exit_code = getattr(outcome, "exit_code", None)
                timed_out = False
            case "timeout":
                timed_out = True
            case _:
                pass
    return Content.from_shell_command_output(
        stdout=getattr(entry, "stdout", None),
        stderr=getattr(entry, "stderr", None),
        exit_code=exit_code,
        timed_out=timed_out,
        raw_representation=entry,
    )


def _shell_failure(outputs: list[Content]) -> str | None:
    """Why the first failing command failed, or None when all succeeded."""
    for output in outputs:
        if output.timed_out:
            return "shell command timed out"
        if output.exit_code is not None and output.exit_code != 0:
            return output.stderr or f"shell command exited with code {output.exit_code}"
    return None


def refresh_in_place(target: Content, snapshot: Content) -> None:
    """Copy *snapshot* onto the already-emitted *target*.

    The stream re-emits a hosted call or result each time it learns more
    about it; the kernel assembles the response by content identity, so the
    first object must carry every later snapshot.
    """
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


def coalesce_pending_results(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold the request-local result markers into their call items.

    A hosted MCP call goes back as one ``mcp_call`` holding arguments and
    output, and an image call gets its payload back for this request only
    (persisted history keeps one copy, on the result). Each marker completes
    the earliest open call with its key. Markers without a call and shadow
    placeholders are dropped: none of them is a valid input item on its own.
    """
    kept: list[dict[str, Any]] = []
    open_calls: dict[PairingKey, deque[dict[str, Any]]] = {}

    def claim(result_type: str, call_id: Any) -> dict[str, Any] | None:
        key = namespaced_pairing_key(result_type, call_id)
        waiting = open_calls.get(key) if key is not None else None
        return waiting.popleft() if waiting else None

    for item in items:
        if item.get(SHADOW_PLACEHOLDER_KEY):
            continue
        if item.get(PENDING_MCP_OUTPUT_KEY):
            call_id = item.get("call_id")
            target = claim("mcp_server_tool_result", call_id)
            if target is None:
                logger.debug(
                    "Dropping orphan mcp_server_tool_result for call_id=%s; no matching mcp_call appeared in input.",
                    call_id,
                )
            elif target.get("output") is None:
                target["output"] = item.get("output")
            continue
        if item.get(PENDING_IMAGE_RESULT_KEY):
            image_id = item.get("image_id")
            target = claim("image_generation_tool_result", image_id)
            payload = item.get("result")
            if target is None:
                logger.debug(
                    "Dropping orphan image_generation_tool_result for image_id=%s; "
                    "no matching image_generation_call appeared in input.",
                    image_id,
                )
            elif isinstance(payload, str) and target.get("result") is None:
                target["result"] = payload
                if target.get("status") in {"in_progress", "generating"}:
                    target["status"] = "completed"
            continue
        kept.append(item)
        call_type = _COMPLETABLE_CALLS.get(item.get("type"))
        key = namespaced_pairing_key(call_type, item.get("id")) if call_type is not None else None
        if key is not None:
            open_calls.setdefault(key, deque()).append(item)
    return kept
