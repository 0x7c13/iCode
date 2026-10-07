# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Provider-neutral planning and live presentation for hosted tools."""

from __future__ import annotations

import inspect
import itertools
import json
import logging
from collections import Counter, defaultdict, deque
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from typing import Any

from chrys.foundation.hosted_tools import (
    HOSTED_TOOL_DEFAULT_RETRY_SAFETY_BY_FAMILY,
    HOSTED_TOOL_DEFAULT_TITLE_BY_FAMILY,
    PRESENTATION_TEXT_SEGMENT_ID_KEY,
    HostedRetrySafety,
    HostedToolFamily,
    HostedToolPhase,
    HostedToolStatus,
    normalize_hosted_tool_status,
)
from chrys.foundation.io.result_content import extract_result_images, extract_result_text
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.tool_result_metadata import (
    TOOL_ERROR_CODE_METADATA_KEY,
    TOOL_ERROR_MESSAGE_METADATA_KEY,
    TOOL_ERRORED_METADATA_KEY,
    TOOL_FAILURE_TEXT_SYNTHESIZED_METADATA_KEY,
)
from chrys.kernel import AgentResponse, Content, Message
from chrys.kernel.exchanges import (
    TOOL_CALL_CONTENT_TYPES,
    TOOL_RESULT_CONTENT_TYPES,
    EmptyIdPolicy,
    LiveAccessor,
    NoneIdPolicy,
    Occurrence,
    PairingPolicy,
    iter_exchanges,
    pair_results,
)
from chrys.kernel.identity import WeakIdentityRegistry
from chrys.kernel.images import is_image_data_uri, is_image_media_type
from chrys.service.agent_middleware.events.intermediate_text import intermediate_text_contents

type ContentOrder = tuple[int, int, int]

_HOSTED_CALL_TYPES = TOOL_CALL_CONTENT_TYPES - {"function_call"}
_HOSTED_RESULT_TYPES = TOOL_RESULT_CONTENT_TYPES - {"function_result"}
_TERMINAL_STATUSES = frozenset({HostedToolStatus.COMPLETED, HostedToolStatus.FAILED, HostedToolStatus.INTERRUPTED})
_PRESENTATION_PAIRING_POLICY = PairingPolicy(
    call_types=TOOL_CALL_CONTENT_TYPES,
    include_informational_calls=True,
    result_types=TOOL_RESULT_CONTENT_TYPES,
    none_id=NoneIdPolicy.POSITIONAL,
    empty_id=EmptyIdPolicy.POSITIONAL,
    malformed_id="stringify",
)


@dataclass(frozen=True)
class HostedToolView:
    """Safe, provider-neutral presentation fields for one hosted occurrence."""

    family: str
    provider: str
    provider_item_type: str
    provider_call_id: str
    tool_name: str
    display_title: str
    arguments: dict[str, Any]
    result_text: str
    image_contents: list[Content]
    artifacts: list[Content]
    provider_status: str
    status: str
    phase: str
    retry_safety: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class IntermediateTextOp:
    """Text that precedes or separates tool activity."""

    text: str
    order: ContentOrder
    attempt_id: str = ""
    segment_ids: tuple[str, ...] = ()
    provisional: bool = False
    source_contents: tuple[Content, ...] = field(default=(), compare=False, repr=False)


@dataclass(frozen=True)
class PresentationAttemptAcceptedOp:
    """Commit the canonical subset of provisional text from one attempt."""

    attempt_id: str
    segments: tuple[IntermediateTextOp, ...]


@dataclass(frozen=True)
class PresentationAttemptRejectedOp:
    """Retract every provisional text segment published by one attempt."""

    attempt_id: str


@dataclass(frozen=True)
class HostedToolStartOp:
    """Start one hosted presentation occurrence."""

    occurrence_ordinal: int
    presentation_id: str
    view: HostedToolView
    order: ContentOrder


@dataclass(frozen=True)
class HostedToolArgsOp:
    """Publish a hosted occurrence's current normalized arguments."""

    occurrence_ordinal: int
    presentation_id: str
    view: HostedToolView
    order: ContentOrder


@dataclass(frozen=True)
class HostedToolProgressOp:
    """Publish a non-terminal hosted progress snapshot."""

    occurrence_ordinal: int
    presentation_id: str
    view: HostedToolView
    order: ContentOrder


@dataclass(frozen=True)
class HostedToolStatusOp:
    """Publish a hosted lifecycle status without fabricating a result."""

    occurrence_ordinal: int
    presentation_id: str
    view: HostedToolView
    order: ContentOrder


@dataclass(frozen=True)
class HostedToolResultOp:
    """Publish a terminal hosted result."""

    occurrence_ordinal: int
    presentation_id: str
    view: HostedToolView
    order: ContentOrder


@dataclass(frozen=True)
class FinalTextOp:
    """Sentinel for the final agent message; empty text remains meaningful."""

    text: str
    structured_output_completed: bool = False


type HostedToolOperation = (
    HostedToolStartOp | HostedToolArgsOp | HostedToolProgressOp | HostedToolStatusOp | HostedToolResultOp
)
type PresentationOperation = IntermediateTextOp | HostedToolOperation
type PresentationSinkOperation = (
    PresentationOperation | FinalTextOp | PresentationAttemptAcceptedOp | PresentationAttemptRejectedOp
)
type PresentationSink = Callable[[PresentationSinkOperation], Awaitable[None] | None]

logger = logging.getLogger(__name__)

_CONTEXT_ARGUMENTS_LIMIT = 1_200
_CONTEXT_RESULT_LIMIT = 2_000


def _family_for(content: Content | None) -> str:
    if content is None:
        return HostedToolFamily.GENERIC
    if content.hosted_family:
        return content.hosted_family
    if content.type.startswith("search_"):
        return HostedToolFamily.SEARCH
    if content.type.startswith("mcp_server_"):
        return HostedToolFamily.MCP
    if content.type.startswith("code_interpreter_"):
        return HostedToolFamily.CODE
    if content.type.startswith("image_generation_"):
        return HostedToolFamily.IMAGE
    if content.type.startswith("shell_"):
        return HostedToolFamily.SHELL
    return HostedToolFamily.GENERIC


def _provider_call_id(content: Content | None) -> str:
    if content is None:
        return ""
    raw_id = content.image_id if content.type.startswith("image_generation_") else content.call_id
    return raw_id if isinstance(raw_id, str) else ""


def _presentation_text_segment_ids(content: Content) -> tuple[str, ...]:
    raw_ids = content.additional_properties.get(PRESENTATION_TEXT_SEGMENT_ID_KEY)
    if isinstance(raw_ids, str):
        return (raw_ids,) if raw_ids else ()
    if isinstance(raw_ids, Sequence):
        return tuple(dict.fromkeys(segment_id for segment_id in raw_ids if isinstance(segment_id, str) and segment_id))
    return ()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]
    if isinstance(value, Content):
        return value.to_dict()
    return str(value)


def _arguments_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {"input": value}
        if isinstance(decoded, Mapping):
            return {str(key): _json_safe(item) for key, item in decoded.items()}
        return {"input": _json_safe(decoded)}
    return {"input": _json_safe(value)}


def _content_values(content: Content | None) -> list[Content]:
    if content is None:
        return []
    candidates: list[Any] = []
    if content.items is not None:
        candidates.extend(content.items)
    if isinstance(content.outputs, Sequence) and not isinstance(content.outputs, str | bytes | bytearray):
        candidates.extend(content.outputs)
    if isinstance(content.output, Sequence) and not isinstance(content.output, str | bytes | bytearray):
        candidates.extend(content.output)
    if isinstance(content.result, Sequence) and not isinstance(content.result, str | bytes | bytearray):
        candidates.extend(content.result)
    return [item for item in candidates if isinstance(item, Content)]


def _is_image(content: Content) -> bool:
    # Share the extraction-side predicate: OpenAI code outputs carry the bare
    # "image" media type, which must not double as an artifact.
    return is_image_media_type(content.media_type) or is_image_data_uri(content.uri)


def _is_artifact(content: Content) -> bool:
    if content.type in {"hosted_file", "hosted_vector_store"}:
        return True
    return content.type in {"data", "uri"} and not _is_image(content)


def _result_payload(content: Content | None) -> Any:
    if content is None:
        return None
    if content.type == "mcp_server_tool_result":
        return content.output
    if content.type in {"code_interpreter_tool_result", "image_generation_tool_result", "shell_tool_result"}:
        return content.outputs
    if content.result is not None:
        return content.result
    if content.items is not None:
        return content.items
    if content.message is not None:
        return content.message
    return None


def _error_code(content: Content | None) -> str | int | None:
    if content is None:
        return None
    if content.error_code:
        return content.error_code
    error = content.additional_properties.get("error")
    if isinstance(error, Mapping):
        code = error.get("error_code", error.get("code"))
        if isinstance(code, str | int) and not isinstance(code, bool):
            return code
    code = content.additional_properties.get("error_code")
    if isinstance(code, str | int) and not isinstance(code, bool):
        return code
    return None


def _is_failed(content: Content | None, provider_status: str) -> bool:
    if normalize_hosted_tool_status(provider_status) is HostedToolStatus.FAILED:
        return True
    return bool(content is not None and content.additional_properties.get("is_error") is True)


def _failure_text(text: str, provider_status: str) -> tuple[str, bool]:
    message = text.strip() or provider_status.strip()
    synthesized = not message
    message = message or "Provider-hosted tool failed."
    return (message if message.startswith("Error: ") else f"Error: {message}", synthesized)


class GenericHostedAdapter:
    """Adapt any canonical hosted pair without provider-specific branches."""

    def adapt(self, call: Content | None, result: Content | None = None) -> HostedToolView:
        family = _family_for(call or result)
        tool_name = ""
        if call is not None:
            tool_name = call.tool_name or call.name or ""
        if not tool_name and result is not None:
            tool_name = result.tool_name or result.name or ""
        if not tool_name:
            tool_name = family

        arguments = self._arguments(call)
        payload = _result_payload(result)
        values = _content_values(result)
        image_contents = extract_result_images(values)
        artifacts = [content for content in values if _is_artifact(content)]
        textual_values = [content for content in values if not _is_image(content) and not _is_artifact(content)]
        if textual_values:
            result_text = extract_result_text(textual_values)
        elif values:
            result_text = ""
        else:
            result_text = extract_result_text(payload) if payload is not None else ""
        provider_status = ""
        for candidate in (result, call):
            if candidate is not None and (candidate.provider_status or candidate.status):
                provider_status = candidate.provider_status or candidate.status or ""
                break
        status = normalize_hosted_tool_status(provider_status)
        failed = _is_failed(result or call, provider_status)
        failure_text_synthesized = False
        if failed:
            status = HostedToolStatus.FAILED
            result_text, failure_text_synthesized = _failure_text(result_text, provider_status)

        phase = ""
        for candidate in (result, call):
            if candidate is not None and candidate.provider_phase:
                phase = candidate.provider_phase
                break
        if not phase:
            phase = HostedToolPhase.TERMINAL if result is not None else HostedToolPhase.START

        retry_safety = ""
        for candidate in (result, call):
            if candidate is not None and candidate.retry_safety:
                retry_safety = candidate.retry_safety
                break
        retry_safety = retry_safety or HOSTED_TOOL_DEFAULT_RETRY_SAFETY_BY_FAMILY.get(family, HostedRetrySafety.UNKNOWN)

        provider = ""
        provider_item_type = ""
        provider_item_id = ""
        for candidate in (call, result):
            if candidate is None:
                continue
            provider = provider or candidate.hosted_provider or ""
            provider_item_type = provider_item_type or candidate.provider_item_type or ""
            provider_item_id = provider_item_id or candidate.provider_item_id or ""

        metadata: dict[str, Any] = {}
        if provider_item_id:
            metadata["provider_item_id"] = provider_item_id
        if phase:
            metadata["provider_phase"] = phase
        if failed:
            metadata[TOOL_ERRORED_METADATA_KEY] = True
            metadata[TOOL_ERROR_MESSAGE_METADATA_KEY] = result_text.removeprefix("Error: ").strip()
            if failure_text_synthesized:
                metadata[TOOL_FAILURE_TEXT_SYNTHESIZED_METADATA_KEY] = True
            metadata["provider_status"] = provider_status
            error_code = _error_code(result or call)
            if error_code is not None:
                metadata[TOOL_ERROR_CODE_METADATA_KEY] = error_code

        return HostedToolView(
            family=family,
            provider=provider,
            provider_item_type=provider_item_type,
            provider_call_id=_provider_call_id(call or result),
            tool_name=tool_name,
            display_title=HOSTED_TOOL_DEFAULT_TITLE_BY_FAMILY.get(family, tool_name),
            arguments=arguments,
            result_text=result_text,
            image_contents=image_contents,
            artifacts=artifacts,
            provider_status=provider_status,
            status=status,
            phase=phase,
            retry_safety=retry_safety,
            metadata=metadata,
        )

    def _arguments(self, call: Content | None) -> dict[str, Any]:
        if call is None:
            return {}
        return _arguments_dict(call.arguments)


class SearchHostedAdapter(GenericHostedAdapter):
    """Adapt hosted search and fetch pairs."""


class McpHostedAdapter(GenericHostedAdapter):
    """Adapt provider-managed MCP pairs."""

    def _arguments(self, call: Content | None) -> dict[str, Any]:
        arguments = super()._arguments(call)
        if call is not None and call.server_name:
            return {"server": call.server_name, **arguments}
        return arguments


class CodeHostedAdapter(GenericHostedAdapter):
    """Adapt code-interpreter calls, logs, images, and artifacts."""

    def _arguments(self, call: Content | None) -> dict[str, Any]:
        if call is None:
            return {}
        inputs = call.inputs or []
        code = "".join(content.text or "" for content in inputs if content.type == "text")
        # Anthropic's first canonical representation stored the complete
        # ``{"code": ...}`` input object as text. Unwrap that representation
        # so live and replayed cards expose one provider-neutral code string.
        nested_code = _arguments_dict(code).get("code")
        if isinstance(nested_code, str):
            code = nested_code
        return {"code": code} if code else super()._arguments(call)


class ImageHostedAdapter(GenericHostedAdapter):
    """Adapt image generation snapshots and terminal outputs."""


class ShellHostedAdapter(GenericHostedAdapter):
    """Adapt provider-hosted shell calls and command outputs."""

    def adapt(self, call: Content | None, result: Content | None = None) -> HostedToolView:
        if result is None or result.type != "shell_tool_result" or result.outputs is None:
            return super().adapt(call, result)

        normalized_outputs: list[Content] = []
        command_details: list[dict[str, Any]] = []
        for output in result.outputs:
            if output.type != "shell_command_output":
                normalized_outputs.append(output)
                continue
            details = {
                key: value
                for key, value in {
                    "stdout": output.stdout,
                    "stderr": output.stderr,
                    "exit_code": output.exit_code,
                    "timed_out": output.timed_out,
                }.items()
                if value is not None
            }
            command_details.append(details)
            text = "\n".join(value for value in (output.stdout, output.stderr) if isinstance(value, str) and value)
            if text:
                normalized_outputs.append(Content.from_text(text))

        normalized_result = Content.from_shell_tool_result(
            call_id=result.call_id,
            outputs=normalized_outputs,
            max_output_length=result.max_output_length,
            hosted_provider=result.hosted_provider,
            provider_item_type=result.provider_item_type,
            provider_item_id=result.provider_item_id,
            provider_phase=result.provider_phase,
            provider_status=result.provider_status,
            retry_safety=result.retry_safety,
            annotations=result.annotations,
            additional_properties=dict(result.additional_properties),
            raw_representation=result.raw_representation,
        )
        view = super().adapt(call, normalized_result)
        if not command_details:
            return view

        metadata = dict(view.metadata)
        metadata["outputs"] = command_details
        stdout = "\n".join(
            str(details["stdout"]) for details in command_details if details.get("stdout") not in (None, "")
        )
        stderr = "\n".join(
            str(details["stderr"]) for details in command_details if details.get("stderr") not in (None, "")
        )
        if stdout:
            metadata["stdout"] = stdout
        if stderr:
            metadata["stderr"] = stderr
        exit_codes = [
            details["exit_code"]
            for details in command_details
            if isinstance(details.get("exit_code"), int) and not isinstance(details.get("exit_code"), bool)
        ]
        if exit_codes:
            metadata["exit_code"] = next((code for code in exit_codes if code != 0), exit_codes[-1])
        if any(details.get("timed_out") is True for details in command_details):
            metadata["timed_out"] = True
        elif any("timed_out" in details for details in command_details):
            metadata["timed_out"] = False
        return replace(view, metadata=metadata)

    def _arguments(self, call: Content | None) -> dict[str, Any]:
        if call is None:
            return {}
        arguments: dict[str, Any] = {}
        if call.commands is not None:
            arguments["commands"] = list(call.commands)
        if call.timeout_ms is not None:
            arguments["timeout_ms"] = call.timeout_ms
        if call.max_output_length is not None:
            arguments["max_output_length"] = call.max_output_length
        return arguments or super()._arguments(call)


_SEARCH_ADAPTER = SearchHostedAdapter()
_MCP_ADAPTER = McpHostedAdapter()
_CODE_ADAPTER = CodeHostedAdapter()
_IMAGE_ADAPTER = ImageHostedAdapter()
_SHELL_ADAPTER = ShellHostedAdapter()
_GENERIC_ADAPTER = GenericHostedAdapter()


def get_hosted_adapter(family: str) -> GenericHostedAdapter:
    """Return the six-family adapter selected by canonical hosted family."""
    if family in {HostedToolFamily.SEARCH, HostedToolFamily.FETCH}:
        return _SEARCH_ADAPTER
    if family == HostedToolFamily.MCP:
        return _MCP_ADAPTER
    if family == HostedToolFamily.CODE:
        return _CODE_ADAPTER
    if family == HostedToolFamily.IMAGE:
        return _IMAGE_ADAPTER
    if family == HostedToolFamily.SHELL:
        return _SHELL_ADAPTER
    return _GENERIC_ADAPTER


def adapt_hosted_tool(call: Content | None, result: Content | None = None) -> HostedToolView:
    """Adapt a canonical hosted call and optional paired result."""
    return get_hosted_adapter(_family_for(call or result)).adapt(call, result)


def hosted_replay_status(view: HostedToolView, *, has_result: bool) -> str:
    """Return the terminal status used when restoring a persisted occurrence."""
    status = normalize_hosted_tool_status(view.status)
    if status in _TERMINAL_STATUSES:
        return status.value
    if view.phase == HostedToolPhase.TERMINAL:
        return HostedToolStatus.COMPLETED
    # Persisted history cannot keep a live card alive. A pending/running call,
    # or a partial result snapshot, was interrupted when the session stopped.
    del has_result
    return HostedToolStatus.INTERRUPTED


def _is_successful_structured_output(view: HostedToolView, *, has_result: bool) -> bool:
    """Return whether a hosted occurrence can stand as the turn's visible output."""
    return hosted_replay_status(view, has_result=has_result) == HostedToolStatus.COMPLETED and view.family not in {
        HostedToolFamily.SEARCH,
        HostedToolFamily.FETCH,
        HostedToolFamily.TOOL_DISCOVERY,
    }


def _bounded_context_value(value: Any, limit: int) -> str:
    try:
        rendered = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    except TypeError, ValueError:
        rendered = str(value)
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[: limit - 1]}…"


def hosted_context_summary(call: Content | None, result: Content | None = None) -> str:
    """Build a bounded provider-neutral assistant summary for one occurrence."""
    view = adapt_hosted_tool(call, result)
    status = hosted_replay_status(view, has_result=result is not None)
    lines = [
        "[Provider-hosted tool context]",
        f"Tool: {view.tool_name or view.display_title}",
        f"Family: {view.family}",
        f"Status: {status}",
    ]
    if view.arguments:
        lines.append(f"Arguments: {_bounded_context_value(view.arguments, _CONTEXT_ARGUMENTS_LIMIT)}")
    if view.result_text:
        lines.append(f"Result: {_bounded_context_value(view.result_text, _CONTEXT_RESULT_LIMIT)}")
    if view.image_contents:
        lines.append(f"Images: {len(view.image_contents)}")
    if view.artifacts:
        lines.append(f"Artifacts: {len(view.artifacts)}")
    return "\n".join(lines)


def _is_hosted_call(content: Content) -> bool:
    if content.type in _HOSTED_CALL_TYPES:
        return True
    return content.type == "function_call" and content.informational_only and content.provider_hosted


def _is_hosted_result(content: Content) -> bool:
    return content.type in _HOSTED_RESULT_TYPES or (content.type == "function_result" and content.provider_hosted)


def is_legacy_server_executed_informational_call(content: Content | Mapping[str, Any]) -> bool:
    """Recognize old OpenAI informational calls with explicit server execution proof."""
    if isinstance(content, Content):
        if content.type != "function_call" or not content.informational_only:
            return False
        extra: Mapping[str, Any] = content.additional_properties
    else:
        if content.get("type") != "function_call" or content.get("informational_only") is not True:
            return False
        raw_extra = content.get("additional_properties")
        if not isinstance(raw_extra, Mapping):
            return False
        extra = raw_extra
    item_type = extra.get("item_type")
    if item_type not in {"tool_search_call", "custom_tool_call"}:
        return False
    return extra.get("execution") == "server" or (
        "server_execution" in extra and extra.get("server_execution") is not None
    )


def normalize_hosted_replay_content(content: Content) -> Content:
    """Promote supported legacy server-executed calls to canonical hosted content."""
    if not is_legacy_server_executed_informational_call(content):
        return content
    promoted = copy(content)
    extra = promoted.additional_properties
    item_type = str(extra.get("item_type") or "")
    promoted.provider_hosted = True
    promoted.hosted_family = (
        HostedToolFamily.TOOL_DISCOVERY if item_type == "tool_search_call" else HostedToolFamily.GENERIC
    )
    promoted.hosted_provider = "openai"
    promoted.provider_item_type = item_type
    promoted.provider_item_id = str(extra.get("item_id") or "") or None
    promoted.provider_status = str(extra.get("status") or "") or None
    promoted.provider_phase = HostedToolPhase.TERMINAL
    promoted.tool_name = promoted.name or "tool"
    return promoted


def _coord(occurrence: Occurrence) -> tuple[int, int]:
    return occurrence.message_index, occurrence.content_index


@dataclass(frozen=True)
class _Pair:
    call: tuple[int, int] | None
    result: tuple[int, int] | None


def _presentation_pairs(messages: Sequence[Message]) -> dict[tuple[int, int], _Pair]:
    accessor = LiveAccessor()
    pairs: dict[tuple[int, int], _Pair] = {}
    for exchange in iter_exchanges(messages, accessor):
        pairing = pair_results(messages, exchange, accessor, _PRESENTATION_PAIRING_POLICY)
        assignments = [assignment for entries in pairing.truthy_assignments.values() for assignment in entries] + [
            assignment for entries in pairing.falsy_assignments.values() for assignment in entries
        ]
        for call_occurrence, result_occurrence in assignments:
            call_coord = _coord(call_occurrence)
            result_coord = _coord(result_occurrence) if result_occurrence is not None else None
            pair = _Pair(call_coord, result_coord)
            pairs[call_coord] = pair
            if result_coord is not None:
                pairs[result_coord] = pair
        unconsumed = [occurrence for entries in pairing.unconsumed_results.values() for occurrence in entries] + [
            occurrence for entries in pairing.falsy_unconsumed_results.values() for occurrence in entries
        ]
        for occurrence in [*unconsumed, *pairing.unpairable_results]:
            result_coord = _coord(occurrence)
            pairs.setdefault(result_coord, _Pair(None, result_coord))
        for occurrence in pairing.unpairable_calls:
            call_coord = _coord(occurrence)
            pairs.setdefault(call_coord, _Pair(call_coord, None))

    # ``iter_exchanges`` starts at a call-bearing assistant run, so a fully
    # standalone hosted result belongs to no exchange. Inventory those
    # occurrences without attempting any global ID matching; each result is
    # its own presentation occurrence and degradation anchor.
    for message_index, message in enumerate(messages):
        if message.role not in {"assistant", "tool"} or HistoryMarkerKind.KEY in message.additional_properties:
            continue
        for content_index, content in enumerate(message.contents):
            coordinate = (message_index, content_index)
            if coordinate not in pairs and _is_hosted_result(content):
                pairs[coordinate] = _Pair(None, coordinate)
    return pairs


def _content_at(messages: Sequence[Message], coordinate: tuple[int, int] | None) -> Content | None:
    if coordinate is None:
        return None
    message_index, content_index = coordinate
    return messages[message_index].contents[content_index]


def cross_provider_hosted_degradations(
    messages: Sequence[Message],
    *,
    target_provider: str,
    unsafe_same_provider_families: Collection[str] = (),
) -> dict[int, str | None]:
    """Map hosted contents that are unsafe to replay to one neutral summary each.

    A provider switch always makes a hosted item unsafe. Callers may also
    identify same-provider families whose server-issued item identities cannot
    be replayed through a locally managed transcript.
    """
    pairs = _presentation_pairs(messages)
    degradations: dict[int, str | None] = {}
    handled_pairs: set[_Pair] = set()
    for message_index, message in enumerate(messages):
        for content_index, content in enumerate(message.contents):
            # Hosted-tool persistence shipped together with this marker, so
            # pre-marker histories cannot contain supported hosted records.
            # Current producers must use the canonical factories, which always
            # set it; marker-less hand-built payloads are outside compatibility.
            if not content.provider_hosted:
                continue
            source_provider = content.hosted_provider or ""
            family = _family_for(content)
            # Chat Completions passes an empty target because that wire has no
            # hosted item types: every provider-hosted item is unsafe there,
            # including factory-created contents with unknown provenance. A
            # hosted-capable target may still receive unstamped contents built
            # for its current dialect, so only an explicit mismatch is foreign.
            cross_provider = not target_provider or bool(source_provider and source_provider != target_provider)
            unsafe_same_provider = source_provider == target_provider and family in unsafe_same_provider_families
            if not cross_provider and not unsafe_same_provider:
                continue
            pair = pairs.get((message_index, content_index), _Pair((message_index, content_index), None))
            if pair in handled_pairs:
                continue
            handled_pairs.add(pair)
            call = _content_at(messages, pair.call)
            result = _content_at(messages, pair.result)
            members = [candidate for candidate in (call, result) if candidate is not None and candidate.provider_hosted]
            if not members:
                continue
            anchor = call if call is not None and call.provider_hosted else members[0]
            for member in members:
                degradations[id(member)] = None
            degradations[id(anchor)] = hosted_context_summary(call, result)
            representative = call or result
            if representative is None:
                raise RuntimeError("A hosted exchange requires a call or a result.")
            logger.debug(
                "Degrading provider-hosted history to assistant context: source=%s target=%s family=%s item_type=%s",
                source_provider,
                target_provider,
                _family_for(call or result),
                representative.provider_item_type,
            )
    return degradations


def _hosted_ops(
    *,
    occurrence: int,
    view: HostedToolView,
    order: ContentOrder,
    start: bool,
    result_content: bool,
) -> list[HostedToolOperation]:
    operations: list[HostedToolOperation] = []
    suborder = order[2]
    if start:
        operations.append(HostedToolStartOp(occurrence, "", view, (order[0], order[1], suborder)))
        suborder += 1
    if view.arguments and (start or result_content):
        # Some providers attach arguments only to the terminal item (OpenAI
        # web_search action), so result observations re-announce them; the
        # publish-side signature dedupes unchanged payloads.
        operations.append(HostedToolArgsOp(occurrence, "", view, (order[0], order[1], suborder)))
        suborder += 1
    status = normalize_hosted_tool_status(view.status)
    nonterminal_snapshot = (
        view.phase in {HostedToolPhase.DELTA, HostedToolPhase.SNAPSHOT} and status not in _TERMINAL_STATUSES
    )
    if result_content and nonterminal_snapshot:
        operations.append(HostedToolProgressOp(occurrence, "", view, (order[0], order[1], suborder)))
    elif result_content:
        operations.append(HostedToolResultOp(occurrence, "", view, (order[0], order[1], suborder)))
    elif view.phase in {HostedToolPhase.DELTA, HostedToolPhase.SNAPSHOT} and (view.result_text or view.image_contents):
        operations.append(HostedToolProgressOp(occurrence, "", view, (order[0], order[1], suborder)))
    elif status is not HostedToolStatus.UNKNOWN:
        operations.append(HostedToolStatusOp(occurrence, "", view, (order[0], order[1], suborder)))
    return operations


@dataclass(frozen=True)
class ResponsePresentationPlan:
    """Pure, exchange-aware presentation plan for an assembled response."""

    operations: list[PresentationOperation]
    final_text: str
    structured_output_completed: bool = False

    @classmethod
    def from_messages(cls, messages: Sequence[Message]) -> ResponsePresentationPlan:
        """Plan text and hosted operations without rebuilding any content."""
        has_hosted = any(
            _is_hosted_call(content) or _is_hosted_result(content)
            for message in messages
            for content in message.contents
        )
        if not has_hosted:
            final_text = ""
            if messages:
                final_text = "".join(
                    content.text or "" for content in messages[-1].contents if content.type == "text" and content.text
                )
            return cls([], final_text)

        pairs = _presentation_pairs(messages)
        operations: list[PresentationOperation] = []
        occurrence_by_pair: dict[_Pair, int] = {}
        emitted_starts: set[int] = set()
        terminal_occurrences: set[int] = set()
        successful_structured_occurrences: set[int] = set()
        pending_text: list[Content] = []
        trailing_text_final = False
        next_occurrence = 0

        def flush_intermediate(order: ContentOrder) -> None:
            if not pending_text:
                return
            segment_ids = tuple(
                dict.fromkeys(
                    segment_id for content in pending_text for segment_id in _presentation_text_segment_ids(content)
                )
            )
            operations.append(
                IntermediateTextOp(
                    "".join(content.text or "" for content in pending_text),
                    order,
                    segment_ids=segment_ids,
                    source_contents=tuple(pending_text),
                )
            )
            pending_text.clear()

        for message_index, message in enumerate(messages):
            for content_index, content in enumerate(message.contents):
                coordinate = (message_index, content_index)
                order = (message_index, content_index, 0)
                if content.type == "text":
                    if content.text:
                        pending_text.append(content)
                    continue
                if content.type == "function_call" and not content.informational_only:
                    flush_intermediate((message_index, content_index, -1))
                    trailing_text_final = False
                    continue
                if content.type == "function_result" and not _is_hosted_result(content):
                    # A full-run transcript interleaves local results between
                    # responses; the loop resumed past this call, so text that
                    # follows is the model's post-tool answer, not preamble.
                    flush_intermediate((message_index, content_index, -1))
                    trailing_text_final = True
                    continue

                pair = pairs.get(coordinate)
                if pair is None:
                    continue
                call = _content_at(messages, pair.call)
                result = _content_at(messages, pair.result)
                if call is not None and not _is_hosted_call(call):
                    continue
                if call is None and (result is None or not _is_hosted_result(result)):
                    continue
                occurrence = occurrence_by_pair.get(pair)
                if occurrence is None:
                    occurrence = next_occurrence
                    next_occurrence += 1
                    occurrence_by_pair[pair] = occurrence

                if _is_hosted_result(content):
                    flush_intermediate((message_index, content_index, -1))
                    start = occurrence not in emitted_starts
                    view = adapt_hosted_tool(call, content)
                    operations.extend(
                        _hosted_ops(
                            occurrence=occurrence,
                            view=view,
                            order=order,
                            start=start,
                            result_content=True,
                        )
                    )
                    emitted_starts.add(occurrence)
                    status = normalize_hosted_tool_status(view.status)
                    if view.phase == HostedToolPhase.TERMINAL or status in _TERMINAL_STATUSES:
                        terminal_occurrences.add(occurrence)
                        if _is_successful_structured_output(view, has_result=True):
                            successful_structured_occurrences.add(occurrence)
                        trailing_text_final = True
                    else:
                        trailing_text_final = False
                    continue

                if _is_hosted_call(content) and occurrence not in emitted_starts:
                    flush_intermediate((message_index, content_index, -1))
                    view = adapt_hosted_tool(content)
                    operations.extend(
                        _hosted_ops(
                            occurrence=occurrence,
                            view=view,
                            order=order,
                            start=True,
                            result_content=False,
                        )
                    )
                    emitted_starts.add(occurrence)
                    has_later_result = pair.result is not None and pair.result != coordinate
                    status = normalize_hosted_tool_status(view.status)
                    terminal_call = not has_later_result and (
                        view.phase == HostedToolPhase.TERMINAL or status in _TERMINAL_STATUSES
                    )
                    if terminal_call:
                        terminal_occurrences.add(occurrence)
                        if _is_successful_structured_output(view, has_result=False):
                            successful_structured_occurrences.add(occurrence)
                        trailing_text_final = True
                    else:
                        trailing_text_final = False

        if pending_text and trailing_text_final:
            final_text = "".join(content.text or "" for content in pending_text)
        else:
            flush_intermediate((len(messages), 0, 0))
            final_text = ""
        return cls(operations, final_text, bool(successful_structured_occurrences))


@dataclass(frozen=True, slots=True)
class FinalSegment:
    """A delegated agent's answer: the caller's result and the transcript's last text.

    The caller receives the response's final segment, as a main turn's answer
    is its last message: text written between tool calls and hosted outputs
    is published live as intermediate transcript text. A response whose final
    segment is empty (it ends on a tool result or hosted output) gives the
    caller everything it said instead; all of that is intermediate text
    already, so ``transcript`` stays empty.
    """

    result: str
    transcript: str

    @classmethod
    def of(cls, response: AgentResponse[Any]) -> FinalSegment:
        final_text = ResponsePresentationPlan.from_messages(response.messages).final_text
        return cls(final_text or response.text, final_text)


@dataclass
class _LiveRecord:
    operation: PresentationOperation | None = None
    local_call_id: str = ""
    local_call_content: Content | None = None
    text_segment_id: str = ""
    released: bool = False


@dataclass
class _LiveTextSegment:
    record: _LiveRecord
    order: ContentOrder
    parts: list[str] = field(default_factory=list)


@dataclass
class _HostedOccurrence:
    ordinal: int
    presentation_id: str
    pairing_key: tuple[str, str]
    provider_item_id: str
    call_seen: bool = False
    result_seen: bool = False
    start_published: bool = False
    terminal: bool = False
    last_view: HostedToolView | None = None


@dataclass
class _PlanBarrier:
    call_id: str
    order: ContentOrder
    content: Content
    released: bool = False


_hosted_run_generations = itertools.count(1)


def next_hosted_run_generation() -> int:
    """Return a hosted pass generation that no other pass in this process has used.

    Presentation ids embed it, and frontends key tool cards by those ids. A
    counter per bindings object restarts on every engine rebuild and would
    hand a new card the id of an earlier one.
    """
    return next(_hosted_run_generations)


@dataclass
class HostedPresentationBridge:
    """Stateful streaming/blocking bridge that emits presentation operations."""

    publish: PresentationSink
    run_generation: int = 0
    response_index: int = 0
    batch_id: int = 0
    # Awaited as each logical response begins, before anything it produces
    # is presented; it publishes text an earlier response left buffered.
    before_response: Callable[[], Awaitable[None]] | None = None
    attempt_index: int = field(default=-1, init=False)
    next_occurrence_ordinal: int = field(default=0, init=False)
    published_terminal_signatures: set[tuple[Any, ...]] = field(default_factory=set, init=False)
    _published_signatures: set[tuple[Any, ...]] = field(default_factory=set, init=False, repr=False)
    _live_records: list[_LiveRecord] = field(default_factory=list, init=False, repr=False)
    _live_cursor: int = field(default=0, init=False, repr=False)
    _occurrences: list[_HostedOccurrence] = field(default_factory=list, init=False, repr=False)
    _accepted_occurrences: list[_HostedOccurrence] = field(default_factory=list, init=False, repr=False)
    _open_occurrences: dict[tuple[str, str], deque[_HostedOccurrence]] = field(
        default_factory=lambda: defaultdict(deque), init=False, repr=False
    )
    _provider_item_candidates: dict[str, list[_HostedOccurrence]] = field(
        default_factory=lambda: defaultdict(list), init=False, repr=False
    )
    _live_text_segments: dict[str, _LiveTextSegment] = field(default_factory=dict, init=False, repr=False)
    _published_provisional_segments: dict[str, IntermediateTextOp] = field(default_factory=dict, init=False, repr=False)
    _published_canonical_text_contents: WeakIdentityRegistry = field(
        default_factory=WeakIdentityRegistry, init=False, repr=False
    )
    _started_local_call_contents: WeakIdentityRegistry = field(
        default_factory=WeakIdentityRegistry, init=False, repr=False
    )
    _accepted_barriers: list[_PlanBarrier] = field(default_factory=list, init=False, repr=False)
    _unbound_local_start_counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)
    _pending_reconciliation: deque[PresentationSinkOperation] = field(default_factory=deque, init=False, repr=False)
    _finalized_response_keys: set[tuple[int, int]] = field(default_factory=set, init=False, repr=False)
    _response_started: bool = field(default=False, init=False, repr=False)
    _saw_incremental_contents: bool = field(default=False, init=False, repr=False)
    _saw_hosted_contents: bool = field(default=False, init=False, repr=False)

    async def begin_response(self, *, response_index: int | None = None, batch_id: int | None = None) -> None:
        """Begin another logical response without resetting run-monotonic ordinals."""
        if self.before_response is not None:
            await self.before_response()
        if response_index is not None:
            self.response_index = response_index
        elif self._response_started:
            self.response_index += 1
        self._response_started = True
        if batch_id is not None:
            self.batch_id = batch_id
        # Provider-controlled call ids may repeat in a later response. Only
        # signals that could not yet bind to a concrete Content are scoped to
        # the current response; identity-bound starts survive for cumulative
        # final reconciliation. Continuation attempts do not call this method.
        self._unbound_local_start_counts.clear()
        self.attempt_index = -1
        self._clear_attempt_state()

    async def attempt_started(self, *, continuation: bool = False) -> None:
        """Open a validation attempt; continuation resumes the current occurrence set."""
        self._response_started = True
        if continuation and self.attempt_index >= 0:
            return
        self.attempt_index += 1
        self._clear_attempt_state()

    async def observe_contents(self, contents: Sequence[Content], *, is_final: bool = False) -> None:
        """Observe one raw provider update or final content batch in content order."""
        if self.attempt_index < 0:
            await self.attempt_started()
        if not is_final and any(content.type != "usage" for content in contents):
            self._saw_incremental_contents = True
        blocking_final_snapshot = is_final and not self._saw_incremental_contents
        for content in contents:
            if content.type == "text":
                if not content.text:
                    continue
                if blocking_final_snapshot:
                    self._live_records.append(
                        _LiveRecord(
                            operation=IntermediateTextOp(
                                content.text,
                                (0, len(self._live_records), 0),
                                source_contents=(content,),
                            )
                        )
                    )
                    continue
                if is_final:
                    continue
                segment_ids = _presentation_text_segment_ids(content)
                if len(segment_ids) == 1:
                    segment_id = segment_ids[0]
                    segment = self._live_text_segments.get(segment_id)
                    if segment is None:
                        record = _LiveRecord(text_segment_id=segment_id)
                        self._live_records.append(record)
                        segment = _LiveTextSegment(record, (0, len(self._live_records) - 1, 0))
                        self._live_text_segments[segment_id] = segment
                    segment.parts.append(content.text)
                else:
                    self._live_records.append(
                        _LiveRecord(operation=IntermediateTextOp(content.text, (0, len(self._live_records), 0)))
                    )
                continue
            if content.type == "function_call" and not content.informational_only:
                if not is_final and self._saw_hosted_contents:
                    await self._seal_live_text_segments()
                call_id = _provider_call_id(content)
                record = _LiveRecord(local_call_id=call_id, local_call_content=content)
                if self._unbound_local_start_counts[call_id] > 0:
                    record.released = True
                    self._started_local_call_contents.register(content)
                    self._unbound_local_start_counts[call_id] -= 1
                    if self._unbound_local_start_counts[call_id] == 0:
                        del self._unbound_local_start_counts[call_id]
                self._live_records.append(record)
                await self._drain_live_prefix()
                continue
            if _is_hosted_call(content):
                self._saw_hosted_contents = True
                if not is_final:
                    await self._seal_live_text_segments()
                await self._observe_call(content)
            elif _is_hosted_result(content):
                self._saw_hosted_contents = True
                if not is_final:
                    await self._seal_live_text_segments()
                await self._observe_result(content)

    async def local_call_start_published(self, provider_call_id: str) -> None:
        """Release the next local-call ordering barrier for a provider call id."""
        bound = False
        for record in self._live_records:
            if record.local_call_id == provider_call_id and not record.released:
                record.released = True
                if record.local_call_content is not None:
                    self._started_local_call_contents.register(record.local_call_content)
                bound = True
                break
        for barrier in self._accepted_barriers:
            if barrier.call_id == provider_call_id and not barrier.released:
                barrier.released = True
                self._started_local_call_contents.register(barrier.content)
                bound = True
                break
        if not bound:
            self._unbound_local_start_counts[provider_call_id] += 1
        await self._drain_live_prefix()
        await self._drain_reconciliation()

    async def attempt_rejected(
        self,
        reason: str = "",
        *,
        status: str = HostedToolStatus.FAILED,
        preserve_provisional: bool = False,
    ) -> None:
        """Terminalize visible cards and conclude all rejected-attempt buffers.

        Validation failures and retries retract presentation-only text.  A
        user interrupt may instead preserve text the user has already read;
        publishing it as accepted also commits its reserved persistence batch.
        """
        if self._published_provisional_segments:
            if preserve_provisional:
                segments = tuple(dict.fromkeys(self._published_provisional_segments.values()))
                await self._publish_once(PresentationAttemptAcceptedOp(self._attempt_id(), segments))
            else:
                await self._publish_once(PresentationAttemptRejectedOp(self._attempt_id()))
        terminal_status = (
            HostedToolStatus.INTERRUPTED if status == HostedToolStatus.INTERRUPTED else HostedToolStatus.FAILED
        )
        for occurrence in self._occurrences:
            if not occurrence.start_published or occurrence.terminal or occurrence.last_view is None:
                continue
            metadata = dict(occurrence.last_view.metadata)
            result_text = occurrence.last_view.result_text
            if terminal_status is HostedToolStatus.FAILED:
                metadata[TOOL_ERRORED_METADATA_KEY] = True
                metadata[TOOL_ERROR_MESSAGE_METADATA_KEY] = reason or "Response attempt rejected."
                result_text, failure_text_synthesized = _failure_text(reason, terminal_status)
                if failure_text_synthesized:
                    metadata[TOOL_FAILURE_TEXT_SYNTHESIZED_METADATA_KEY] = True
            terminal_view = replace(
                occurrence.last_view,
                result_text=result_text,
                status=terminal_status,
                provider_status=terminal_status,
                phase=HostedToolPhase.TERMINAL,
                metadata=metadata,
            )
            operation = HostedToolStatusOp(
                occurrence.ordinal,
                occurrence.presentation_id,
                terminal_view,
                (0, len(self._live_records), 0),
            )
            await self._publish_once(operation)
            occurrence.terminal = True
        self._clear_attempt_state()

    async def attempt_accepted(self, messages: Sequence[Message] | None = None) -> None:
        """Mark validation accepted and reconcile its live-visible operations."""
        for occurrence in self._occurrences:
            if occurrence not in self._accepted_occurrences:
                self._accepted_occurrences.append(occurrence)
        if messages is not None:
            # The client's intermediate-text callback already owns this
            # response's narration, and a final reconcile over a run with
            # hosted output would publish it a second time.
            for content in intermediate_text_contents(messages):
                self._published_canonical_text_contents.register(content)
            # A logical response inside agent.run() is not necessarily the
            # turn's final response: local function calls can continue the
            # loop. Publish its canonical text/tool ordering now, but leave
            # the terminal FinalTextOp to the run-completion reconciliation.
            await self._reconcile_accepted(
                messages,
                final=False,
                publish_final=False,
                occurrence_candidates=tuple(self._occurrences),
            )

    async def reconcile_accepted(
        self,
        messages: Sequence[Message],
        *,
        final: bool = False,
        publish_final: bool = True,
    ) -> None:
        """Reconcile live events against assembled content and emit final text last.

        ``final`` marks the run-completion reconcile: no further local tool
        starts can arrive after it. ``publish_final`` is false while accepting
        one non-terminal response inside the tool loop.
        """
        accepted_ordinals = {occurrence.ordinal for occurrence in self._accepted_occurrences}
        occurrence_candidates = (
            *self._accepted_occurrences,
            *(occurrence for occurrence in self._occurrences if occurrence.ordinal not in accepted_ordinals),
        )
        await self._reconcile_accepted(
            messages,
            final=final,
            publish_final=publish_final,
            occurrence_candidates=occurrence_candidates,
        )

    async def _reconcile_accepted(
        self,
        messages: Sequence[Message],
        *,
        final: bool,
        publish_final: bool,
        occurrence_candidates: Sequence[_HostedOccurrence],
    ) -> None:
        response_key = (self.attempt_index, self.response_index)
        if publish_final and response_key in self._finalized_response_keys:
            return
        plan = ResponsePresentationPlan.from_messages(messages)
        mapped_occurrences: dict[int, _HostedOccurrence] = {}
        used_ordinals: set[int] = set()
        reconciled: list[PresentationSinkOperation] = []
        for operation in plan.operations:
            if isinstance(operation, IntermediateTextOp):
                reconciled.append(replace(operation, attempt_id=self._attempt_id()))
                continue
            occurrence = mapped_occurrences.get(operation.occurrence_ordinal)
            if occurrence is None:
                occurrence = self._match_reconciled_occurrence(
                    operation.view,
                    used_ordinals,
                    occurrence_candidates,
                )
                mapped_occurrences[operation.occurrence_ordinal] = occurrence
                used_ordinals.add(occurrence.ordinal)
            reconciled.append(
                replace(
                    operation,
                    occurrence_ordinal=occurrence.ordinal,
                    presentation_id=occurrence.presentation_id,
                )
            )

        accepted_segments, reconciled = self._accept_provisional_text(reconciled)
        reconciled = [
            operation
            for operation in reconciled
            if not (
                isinstance(operation, IntermediateTextOp)
                and operation.source_contents
                and all(content in self._published_canonical_text_contents for content in operation.source_contents)
            )
        ]
        if self._published_provisional_segments:
            await self._publish_once(
                PresentationAttemptAcceptedOp(
                    self._attempt_id(),
                    tuple(accepted_segments),
                )
            )

        self._accepted_barriers = self._plan_barriers(messages)
        for barrier in self._accepted_barriers:
            if barrier.content in self._started_local_call_contents:
                barrier.released = True
                continue
            if self._unbound_local_start_counts[barrier.call_id] > 0:
                barrier.released = True
                self._started_local_call_contents.register(barrier.content)
                self._unbound_local_start_counts[barrier.call_id] -= 1
                if self._unbound_local_start_counts[barrier.call_id] == 0:
                    del self._unbound_local_start_counts[barrier.call_id]
        if final:
            # Calls that never entered the tool pipeline (unknown tool,
            # pre-pipeline argument rejection) publish no start; once the run
            # is over their barriers can only gate FinalTextOp forever.
            for barrier in self._accepted_barriers:
                barrier.released = True
        pending: list[PresentationSinkOperation] = list(reconciled)
        if publish_final:
            pending.append(
                FinalTextOp(
                    plan.final_text,
                    structured_output_completed=not plan.final_text.strip() and plan.structured_output_completed,
                )
            )
            self._finalized_response_keys.add(response_key)
        self._pending_reconciliation = deque(pending)
        await self._drain_reconciliation()

    async def publish_blocking(self, messages: Sequence[Message]) -> None:
        """Plan an assembled blocking response and publish its final sentinel last."""
        if self.attempt_index < 0:
            await self.attempt_started()
        plan = ResponsePresentationPlan.from_messages(messages)
        mapped: dict[int, _HostedOccurrence] = {}
        for operation in plan.operations:
            outbound: PresentationOperation = operation
            if not isinstance(operation, IntermediateTextOp):
                occurrence = mapped.get(operation.occurrence_ordinal)
                if occurrence is None:
                    occurrence = self._allocate_occurrence(operation.view)
                    mapped[operation.occurrence_ordinal] = occurrence
                outbound = replace(
                    operation,
                    occurrence_ordinal=occurrence.ordinal,
                    presentation_id=occurrence.presentation_id,
                )
            await self._publish_once(outbound)
        await self._publish_sink(
            FinalTextOp(
                plan.final_text,
                structured_output_completed=not plan.final_text.strip() and plan.structured_output_completed,
            )
        )

    async def _observe_call(self, content: Content) -> None:
        view = adapt_hosted_tool(content)
        occurrence = self._call_occurrence(content, view)
        if occurrence.terminal and not occurrence.call_seen:
            occurrence.call_seen = True
            if occurrence.last_view is not None:
                occurrence.last_view = replace(
                    occurrence.last_view,
                    tool_name=view.tool_name,
                    display_title=view.display_title,
                    arguments=view.arguments,
                )
            return
        occurrence.call_seen = True
        occurrence.last_view = view
        operations = _hosted_ops(
            occurrence=occurrence.ordinal,
            view=view,
            order=(0, len(self._live_records), 0),
            start=not self._has_start_record(occurrence),
            result_content=False,
        )
        for operation in operations:
            await self._append_live(replace(operation, presentation_id=occurrence.presentation_id))
        status = normalize_hosted_tool_status(view.status)
        if view.phase == HostedToolPhase.TERMINAL or status in _TERMINAL_STATUSES:
            occurrence.terminal = True

    async def _observe_result(self, content: Content) -> None:
        key = self._pairing_key(content)
        provider_item_id = content.provider_item_id or ""
        occurrence = (
            self._provider_item_candidates[provider_item_id][-1]
            if provider_item_id and self._provider_item_candidates[provider_item_id]
            else next(
                (candidate for candidate in reversed(self._open_occurrences[key]) if not candidate.result_seen),
                None,
            )
        )
        if occurrence is None:
            view = adapt_hosted_tool(None, content)
            occurrence = self._allocate_occurrence(view, pairing_key=key)
        view = adapt_hosted_tool(None, content)
        if occurrence.call_seen:
            view = replace(
                view,
                tool_name=occurrence.last_view.tool_name if occurrence.last_view is not None else view.tool_name,
                display_title=(
                    occurrence.last_view.display_title if occurrence.last_view is not None else view.display_title
                ),
                arguments=occurrence.last_view.arguments if occurrence.last_view is not None else view.arguments,
            )
        occurrence.result_seen = True
        occurrence.last_view = view
        operations = _hosted_ops(
            occurrence=occurrence.ordinal,
            view=view,
            order=(0, len(self._live_records), 0),
            start=not self._has_start_record(occurrence),
            result_content=True,
        )
        for operation in operations:
            await self._append_live(replace(operation, presentation_id=occurrence.presentation_id))
        status = normalize_hosted_tool_status(view.status)
        if view.phase == HostedToolPhase.TERMINAL or status in _TERMINAL_STATUSES:
            occurrence.terminal = True

    def _call_occurrence(self, content: Content, view: HostedToolView) -> _HostedOccurrence:
        provider_item_id = content.provider_item_id or ""
        if provider_item_id and self._provider_item_candidates[provider_item_id]:
            return self._provider_item_candidates[provider_item_id][-1]
        key = self._pairing_key(content)
        for occurrence in reversed(self._open_occurrences[key]):
            if not occurrence.call_seen or not occurrence.result_seen:
                return occurrence
        return self._allocate_occurrence(view, pairing_key=key)

    def _allocate_occurrence(
        self,
        view: HostedToolView,
        *,
        pairing_key: tuple[str, str] | None = None,
    ) -> _HostedOccurrence:
        ordinal = self.next_occurrence_ordinal
        self.next_occurrence_ordinal += 1
        occurrence = _HostedOccurrence(
            ordinal=ordinal,
            presentation_id=(
                f"hosted:{self.run_generation}:{max(self.attempt_index, 0)}:{self.response_index}:{ordinal}"
            ),
            pairing_key=pairing_key or ("call_id", view.provider_call_id),
            provider_item_id=str(view.metadata.get("provider_item_id", "")),
            last_view=view,
        )
        self._occurrences.append(occurrence)
        self._open_occurrences[occurrence.pairing_key].append(occurrence)
        if occurrence.provider_item_id:
            self._provider_item_candidates[occurrence.provider_item_id].append(occurrence)
        return occurrence

    def _match_reconciled_occurrence(
        self,
        view: HostedToolView,
        used_ordinals: set[int],
        candidates: Sequence[_HostedOccurrence],
    ) -> _HostedOccurrence:
        provider_item_id = str(view.metadata.get("provider_item_id", ""))
        if provider_item_id:
            for occurrence in candidates:
                if occurrence.provider_item_id != provider_item_id:
                    continue
                if occurrence.ordinal not in used_ordinals:
                    return occurrence
        for occurrence in candidates:
            if occurrence.ordinal in used_ordinals:
                continue
            if occurrence.pairing_key == ("call_id", view.provider_call_id):
                return occurrence
            if occurrence.last_view is not None and (
                occurrence.last_view.family,
                occurrence.last_view.tool_name,
            ) == (view.family, view.tool_name):
                return occurrence
        return self._allocate_occurrence(view)

    def _pairing_key(self, content: Content) -> tuple[str, str]:
        if content.type.startswith("image_generation_"):
            return "image_id", content.image_id or ""
        return "call_id", content.call_id or ""

    def _has_start_record(self, occurrence: _HostedOccurrence) -> bool:
        return any(
            isinstance(record.operation, HostedToolStartOp)
            and record.operation.occurrence_ordinal == occurrence.ordinal
            for record in self._live_records
        )

    async def _append_live(self, operation: HostedToolOperation) -> None:
        self._live_records.append(_LiveRecord(operation=operation))
        await self._drain_live_prefix()

    async def _seal_live_text_segments(self) -> None:
        """Publish completed text only once later output proves it intermediate."""
        if not self._live_text_segments:
            return
        for segment_id, segment in self._live_text_segments.items():
            segment.record.operation = IntermediateTextOp(
                "".join(segment.parts),
                segment.order,
                attempt_id=self._attempt_id(),
                segment_ids=(segment_id,),
                provisional=True,
            )
            segment.record.text_segment_id = ""
        self._live_text_segments.clear()
        await self._drain_live_prefix()

    async def _drain_live_prefix(self) -> None:
        while self._live_cursor < len(self._live_records):
            record = self._live_records[self._live_cursor]
            if record.local_call_id:
                if not record.released:
                    return
                self._live_cursor += 1
                continue
            if record.text_segment_id:
                return
            if isinstance(record.operation, IntermediateTextOp):
                if not record.operation.provisional:
                    return
                await self._publish_once(record.operation)
                for segment_id in record.operation.segment_ids:
                    self._published_provisional_segments[segment_id] = record.operation
                self._live_cursor += 1
                continue
            if record.operation is None:
                self._live_cursor += 1
                continue
            await self._publish_once(record.operation)
            if isinstance(record.operation, HostedToolStartOp):
                occurrence = self._occurrence_by_ordinal(record.operation.occurrence_ordinal)
                occurrence.start_published = True
            self._live_cursor += 1

    async def _drain_reconciliation(self) -> None:
        while self._pending_reconciliation:
            operation = self._pending_reconciliation[0]
            if isinstance(operation, PresentationAttemptAcceptedOp | PresentationAttemptRejectedOp):
                self._pending_reconciliation.popleft()
                await self._publish_once(operation)
                continue
            if isinstance(operation, FinalTextOp):
                if any(not barrier.released for barrier in self._accepted_barriers):
                    return
            elif any(not barrier.released and barrier.order < operation.order for barrier in self._accepted_barriers):
                return
            self._pending_reconciliation.popleft()
            await self._publish_once(operation)

    def _plan_barriers(self, messages: Sequence[Message]) -> list[_PlanBarrier]:
        barriers: list[_PlanBarrier] = []
        for message_index, message in enumerate(messages):
            for content_index, content in enumerate(message.contents):
                if content.type == "function_call" and not content.informational_only:
                    barriers.append(
                        _PlanBarrier(
                            _provider_call_id(content),
                            (message_index, content_index, 0),
                            content,
                        )
                    )
        return barriers

    def _occurrence_by_ordinal(self, ordinal: int) -> _HostedOccurrence:
        return next(occurrence for occurrence in self._occurrences if occurrence.ordinal == ordinal)

    async def _publish_once(self, operation: PresentationSinkOperation) -> None:
        signature = self._signature(operation)
        if signature in self._published_signatures:
            return
        self._published_signatures.add(signature)
        if isinstance(operation, HostedToolResultOp) or (
            isinstance(operation, HostedToolStatusOp)
            and normalize_hosted_tool_status(operation.view.status) in _TERMINAL_STATUSES
        ):
            self.published_terminal_signatures.add(signature)
        if isinstance(operation, IntermediateTextOp) and not operation.provisional:
            for content in operation.source_contents:
                self._published_canonical_text_contents.register(content)
        await self._publish_sink(operation)

    async def _publish_sink(self, operation: PresentationSinkOperation) -> None:
        outcome = self.publish(operation)
        if inspect.isawaitable(outcome):
            await outcome

    def _signature(self, operation: PresentationSinkOperation) -> tuple[Any, ...]:
        if isinstance(operation, FinalTextOp):
            return (
                "final",
                self.attempt_index,
                self.response_index,
                operation.text,
                operation.structured_output_completed,
            )
        if isinstance(operation, PresentationAttemptAcceptedOp):
            return ("presentation_accepted", operation.attempt_id)
        if isinstance(operation, PresentationAttemptRejectedOp):
            return ("presentation_rejected", operation.attempt_id)
        if isinstance(operation, IntermediateTextOp):
            if operation.segment_ids:
                return ("text", operation.attempt_id, operation.segment_ids, operation.text)
            return ("text", self.attempt_index, self.response_index, operation.order, operation.text)
        view = operation.view
        base = (type(operation).__name__, operation.presentation_id)
        if isinstance(operation, HostedToolStartOp):
            return base
        if isinstance(operation, HostedToolArgsOp):
            return (*base, json.dumps(_json_safe(view.arguments), sort_keys=True, ensure_ascii=False))
        if isinstance(operation, HostedToolProgressOp):
            images = tuple((content.uri, content.media_type, content.file_id) for content in view.image_contents)
            return (*base, view.phase, view.provider_status, view.result_text, images)
        if isinstance(operation, HostedToolStatusOp):
            return (*base, view.status, view.provider_status)
        artifacts = tuple((content.file_id, content.uri, content.media_type) for content in view.artifacts)
        images = tuple((content.uri, content.media_type, content.file_id) for content in view.image_contents)
        return (*base, view.status, view.provider_status, view.result_text, images, artifacts)

    def _clear_attempt_state(self) -> None:
        self._live_records.clear()
        self._live_cursor = 0
        self._occurrences.clear()
        self._open_occurrences.clear()
        self._provider_item_candidates.clear()
        self._live_text_segments.clear()
        self._published_provisional_segments.clear()
        self._accepted_barriers.clear()
        self._pending_reconciliation.clear()
        self._saw_incremental_contents = False
        self._saw_hosted_contents = False

    def _attempt_id(self) -> str:
        return f"presentation:{self.run_generation}:{self.response_index}:{max(self.attempt_index, 0)}"

    def _accept_provisional_text(
        self,
        operations: list[PresentationSinkOperation],
    ) -> tuple[list[IntermediateTextOp], list[PresentationSinkOperation]]:
        """Match live provisional text to canonical intermediate operations."""
        if not self._published_provisional_segments:
            return [], operations
        accepted: list[IntermediateTextOp] = []
        retained: list[PresentationSinkOperation] = []
        accepted_ids: set[str] = set()
        published_by_text: dict[str, list[IntermediateTextOp]] = defaultdict(list)
        for operation in dict.fromkeys(self._published_provisional_segments.values()):
            published_by_text[operation.text].append(operation)

        for operation in operations:
            if not isinstance(operation, IntermediateTextOp):
                retained.append(operation)
                continue
            matched: list[IntermediateTextOp] = []
            if operation.segment_ids and all(
                segment_id in self._published_provisional_segments for segment_id in operation.segment_ids
            ):
                matched = list(
                    dict.fromkeys(
                        self._published_provisional_segments[segment_id] for segment_id in operation.segment_ids
                    )
                )
            elif candidates := published_by_text.get(operation.text):
                matched = [
                    candidate for candidate in candidates if not accepted_ids.intersection(candidate.segment_ids)
                ][:1]
            if not matched:
                retained.append(operation)
                continue
            # The streamed fragments are presentation-only objects, while
            # ``operation.source_contents`` belongs to the assembled response
            # that later becomes the final run transcript. Remember that
            # canonical identity now so run-completion reconciliation cannot
            # re-publish an already accepted provisional segment.
            for content in operation.source_contents:
                self._published_canonical_text_contents.register(content)
            for candidate in matched:
                if accepted_ids.intersection(candidate.segment_ids):
                    continue
                accepted.append(candidate)
                accepted_ids.update(candidate.segment_ids)
        return accepted, retained


__all__ = [
    "CodeHostedAdapter",
    "FinalSegment",
    "FinalTextOp",
    "GenericHostedAdapter",
    "HostedPresentationBridge",
    "HostedToolArgsOp",
    "HostedToolProgressOp",
    "HostedToolResultOp",
    "HostedToolStartOp",
    "HostedToolStatusOp",
    "HostedToolView",
    "ImageHostedAdapter",
    "IntermediateTextOp",
    "McpHostedAdapter",
    "PresentationAttemptAcceptedOp",
    "PresentationAttemptRejectedOp",
    "PresentationOperation",
    "PresentationSinkOperation",
    "ResponsePresentationPlan",
    "SearchHostedAdapter",
    "ShellHostedAdapter",
    "adapt_hosted_tool",
    "cross_provider_hosted_degradations",
    "get_hosted_adapter",
    "hosted_context_summary",
    "hosted_replay_status",
    "is_legacy_server_executed_informational_call",
    "next_hosted_run_generation",
    "normalize_hosted_replay_content",
]
