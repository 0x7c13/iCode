# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Independent reference implementation of the transcript exchange grammar.

This module deliberately imports NOTHING from ``chrys.kernel.exchanges``:
it is the differential oracle's other half, written directly from the
grammar's prose rules with its own literal type sets, so a defect in the
production iterator cannot hide by being mirrored here (a separate parity
pin compares the literal sets against the production constants, making any
divergence a conscious decision).

Unlike the invariant oracle (``tests.support.transcript_invariants``),
which only asserts, this state machine REPORTS boundaries and pairing so
tests can diff its trace against the production iterator's output over the
grammar corpus — live objects and their ``to_dict`` serializations alike.

The trace pairs under ONE canonical policy: all call and result types,
informational calls included, ``None``/``""`` ids as two separate
positional queues, non-string ids stringified (``str(v) if v else ""``).
Consumer-specific policy rows are covered by their own pins; the
differential guards the GRAMMAR — boundaries, sibling runs, output runs,
marker behavior, embedded results, and canonical pairing order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Test-owned LITERAL sets — never imported from production code.
REFERENCE_CALL_TYPES = frozenset(
    {
        "function_call",
        "hosted_tool_call",
        "code_interpreter_tool_call",
        "image_generation_tool_call",
        "mcp_server_tool_call",
        "search_tool_call",
        "shell_tool_call",
    }
)
REFERENCE_RESULT_TYPES = frozenset(
    {
        "function_result",
        "hosted_tool_result",
        "code_interpreter_tool_result",
        "image_generation_tool_result",
        "mcp_server_tool_result",
        "search_tool_result",
        "shell_tool_result",
    }
)
_REFERENCE_LEGACY_CALL_TYPE = "legacy"
_REFERENCE_IMAGE_TYPES = frozenset({"image_generation_tool_call", "image_generation_tool_result"})
_REFERENCE_MARKER_KEY = "_chrys_kind"


@dataclass(frozen=True)
class ReferenceTrace:
    """One exchange's boundaries plus canonical pairing, as coordinates."""

    response_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    # (call (message_index, content_index), result coords or None), in
    # registration order across the whole exchange; falsy streams appended
    # after all truthy keys, "none" queue before "empty" queue.
    assignments: tuple[tuple[tuple[int, int], tuple[int, int] | None], ...]
    orphan_results: tuple[tuple[int, int], ...]


def _read(obj: Any, attr: str, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, attr, None)


def _role(message: Any) -> str:
    role = _read(message, "role", "role")
    return role if isinstance(role, str) else ""


def _contents(message: Any) -> Sequence[Any]:
    contents = _read(message, "contents", "contents")
    return contents if isinstance(contents, list) else []


def _content_type(content: Any) -> str | None:
    content_type = _read(content, "type", "type")
    return content_type if isinstance(content_type, str) else None


def _has_marker(message: Any) -> bool:
    properties = _read(message, "additional_properties", "additional_properties")
    if isinstance(properties, dict):
        # Key presence: the value is the marker kind and may be anything.
        return _REFERENCE_MARKER_KEY in properties
    return False


def _is_call(content: Any, *, dict_transcript: bool) -> bool:
    content_type = _content_type(content)
    if content_type in REFERENCE_CALL_TYPES:
        return True
    return dict_transcript and content_type == _REFERENCE_LEGACY_CALL_TYPE


def _is_result(content: Any) -> bool:
    return _content_type(content) in REFERENCE_RESULT_TYPES


def _raw_id(content: Any) -> str | None:
    if _content_type(content) in _REFERENCE_IMAGE_TYPES:
        value = _read(content, "image_id", "image_id")
    else:
        value = _read(content, "call_id", "call_id")
    if value is None or isinstance(value, str):
        return value
    return str(value) if value else ""


def _pairing_key(content: Any) -> tuple[str, str] | None:
    raw = _raw_id(content)
    if not raw:
        return None
    namespace = "image_id" if _content_type(content) in _REFERENCE_IMAGE_TYPES else "call_id"
    return (namespace, raw)


def _is_result_only_assistant(message: Any, *, dict_transcript: bool) -> bool:
    if _role(message) != "assistant":
        return False
    saw_result = False
    for content in _contents(message):
        if _is_call(content, dict_transcript=dict_transcript):
            return False
        if _is_result(content):
            saw_result = True
    return saw_result


@dataclass
class _Pairing:
    truthy: dict[tuple[str, str], list[list[Any]]] = field(default_factory=dict)
    none_queue: list[list[Any]] = field(default_factory=list)
    empty_queue: list[list[Any]] = field(default_factory=list)
    orphans: list[tuple[int, int]] = field(default_factory=list)
    order: list[list[Any]] = field(default_factory=list)

    def register_call(self, coord: tuple[int, int], content: Any) -> None:
        slot = [coord, None]
        key = _pairing_key(content)
        if key is not None:
            self.truthy.setdefault(key, []).append(slot)
        elif _raw_id(content) is None:
            self.none_queue.append(slot)
        else:
            self.empty_queue.append(slot)
        self.order.append(slot)

    def consume(self, coord: tuple[int, int], content: Any) -> None:
        key = _pairing_key(content)
        if key is not None:
            slots = self.truthy.get(key, [])
        elif _raw_id(content) is None:
            slots = self.none_queue
        else:
            slots = self.empty_queue
        for slot in slots:
            if slot[1] is None:
                slot[1] = coord
                return
        self.orphans.append(coord)


def reference_exchange_trace(messages: Sequence[Any]) -> list[ReferenceTrace]:
    """Report every exchange of *messages* under the reference grammar.

    Accepts live messages or ``to_dict`` mappings; a transcript is treated
    as dict-typed (enabling the ``legacy`` call type) when its first
    message is a mapping.
    """
    dict_transcript = bool(messages) and isinstance(messages[0], dict)
    traces: list[ReferenceTrace] = []
    total = len(messages)
    index = 0
    while index < total:
        message = messages[index]
        if (
            _has_marker(message)
            or _role(message) != "assistant"
            or _is_result_only_assistant(message, dict_transcript=dict_transcript)
        ):
            index += 1
            continue

        run_start = index
        while (
            index < total
            and _role(messages[index]) == "assistant"
            and not _has_marker(messages[index])
            and not _is_result_only_assistant(messages[index], dict_transcript=dict_transcript)
        ):
            index += 1
        response_indices = tuple(range(run_start, index))

        if not any(
            _is_call(content, dict_transcript=dict_transcript)
            for i in response_indices
            for content in _contents(messages[i])
        ):
            continue

        output_start = index
        while index < total and not _has_marker(messages[index]):
            candidate = messages[index]
            if _role(candidate) == "tool" or _is_result_only_assistant(candidate, dict_transcript=dict_transcript):
                index += 1
                continue
            break
        output_indices = tuple(range(output_start, index))

        pairing = _Pairing()
        for message_index in response_indices:
            contents = _contents(messages[message_index])
            for content_index, content in enumerate(contents):
                if _is_call(content, dict_transcript=dict_transcript):
                    pairing.register_call((message_index, content_index), content)
            for content_index, content in enumerate(contents):
                if _is_result(content):
                    pairing.consume((message_index, content_index), content)
        for message_index in output_indices:
            for content_index, content in enumerate(_contents(messages[message_index])):
                if _is_result(content):
                    pairing.consume((message_index, content_index), content)

        traces.append(
            ReferenceTrace(
                response_indices=response_indices,
                output_indices=output_indices,
                assignments=tuple((slot[0], slot[1]) for slot in pairing.order),
                orphan_results=tuple(pairing.orphans),
            )
        )
    return traces
