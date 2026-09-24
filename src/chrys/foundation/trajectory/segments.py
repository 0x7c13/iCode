# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Segmentation of unbounded event fields into ``event.segment`` lines.

Fields that can grow without bound (context membership refs, large provider
usage blobs, compaction group lists) never degrade to a hash: the base event
declares ``segmented_fields[]`` and the field's content continues in one or
more ``event.segment`` events that each fit the line budget. Two encodings:

* ``array_slice`` — ``entries[]`` is an ordered slice of the original array;
  concatenating segments by ``segment_index`` restores it.
* ``object_entries`` — ``entries[]`` holds ``{key, value}`` pairs; a duplicate
  key within one group is corruption, so writers de-duplicate before packing.

An entry that still exceeds the budget alone is marked ``entry_oversized`` and
its value is replaced by an ``{omitted, byte_length, value_hash}`` sentinel
(keyed HMAC) — the only sanctioned "segmented yet degraded" path. Without a
fingerprint key the sentinel keeps the length and drops the hash: an unkeyed
digest of a short value is guessable, and identifying the value is not worth
handing out a way to recover it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from chrys.foundation.trajectory.envelope import LINE_BUDGET_BYTES, SegmentedField, encode_json_value
from chrys.foundation.trajectory.ids import new_analytics_id

ENCODING_ARRAY_SLICE: Final = "array_slice"
ENCODING_OBJECT_ENTRIES: Final = "object_entries"

_INDEX_PLACEHOLDER: Final = 10**9


class SegmentationError(ValueError):
    """Raised when a field cannot be segmented under the contract."""


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """Result of segmenting one field: its declaration plus segment payloads."""

    declaration: SegmentedField
    encoding: str
    segment_payloads: tuple[dict[str, Any], ...]
    oversized_entry_count: int

    @property
    def segment_count(self) -> int:
        return len(self.segment_payloads)


LineMeasure = Callable[[Mapping[str, Any]], int]
"""Return the encoded line length of a full event carrying *payload*."""

ValueHasher = Callable[[bytes], str]
"""Return the keyed fingerprint of an omitted value's canonical bytes."""


def _entry_bytes(entry: Any) -> bytes:
    return encode_json_value(entry)


def _segment_payload(
    *,
    segment_group_id: str,
    parent_event_id: str,
    field_pointer: str,
    encoding: str,
    segment_index: int,
    segment_count: int,
    entries: Sequence[Any],
    entry_oversized: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "segment_group_id": segment_group_id,
        "parent_event_id": parent_event_id,
        "field_pointer": field_pointer,
        "segment_index": segment_index,
        "segment_count": segment_count,
        "encoding": encoding,
        "entries": list(entries),
    }
    if entry_oversized:
        payload["entry_oversized"] = True
    return payload


def _oversized_sentinel(raw: bytes, hasher: ValueHasher | None) -> dict[str, Any]:
    sentinel: dict[str, Any] = {"omitted": True, "byte_length": len(raw)}
    if hasher is not None:
        sentinel["value_hash"] = hasher(raw)
    return sentinel


def plan_segments(
    *,
    parent_event_id: str,
    field_pointer: str,
    encoding: str,
    entries: Sequence[Any],
    measure: LineMeasure,
    hasher: ValueHasher | None,
    budget: int = LINE_BUDGET_BYTES,
) -> SegmentPlan:
    """Pack *entries* into segment payloads that each encode under *budget* bytes.

    For ``object_entries`` every entry must already be a ``{key, value}`` mapping
    with string keys; duplicate keys raise :class:`SegmentationError` because the
    reader treats them as corruption rather than merging.
    """
    if encoding not in {ENCODING_ARRAY_SLICE, ENCODING_OBJECT_ENTRIES}:
        raise SegmentationError(f"Unknown segment encoding: {encoding!r}")
    if encoding == ENCODING_OBJECT_ENTRIES:
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {"key", "value"} or not isinstance(entry["key"], str):
                raise SegmentationError("object_entries segments require {key, value} entries with string keys.")
            if entry["key"] in seen:
                raise SegmentationError(f"Duplicate object_entries key before segmentation: {entry['key']!r}")
            seen.add(entry["key"])
    segment_group_id = new_analytics_id()
    overhead = measure(
        _segment_payload(
            segment_group_id=segment_group_id,
            parent_event_id=parent_event_id,
            field_pointer=field_pointer,
            encoding=encoding,
            segment_index=_INDEX_PLACEHOLDER,
            segment_count=_INDEX_PLACEHOLDER,
            entries=(),
            entry_oversized=True,
        )
    )
    if overhead >= budget:
        raise SegmentationError("Segment envelope alone exceeds the line budget.")
    capacity = budget - overhead

    groups: list[tuple[list[Any], bool]] = []
    current: list[Any] = []
    current_size = 0
    oversized_count = 0
    for entry in entries:
        raw = _entry_bytes(entry)
        cost = len(raw) + (1 if current else 0)
        if len(raw) > capacity:
            if current:
                groups.append((current, False))
                current, current_size = [], 0
            if encoding == ENCODING_OBJECT_ENTRIES:
                value_raw = _entry_bytes(entry["value"])
                sentinel: Any = {"key": entry["key"], "value": _oversized_sentinel(value_raw, hasher)}
            else:
                sentinel = _oversized_sentinel(raw, hasher)
            if len(_entry_bytes(sentinel)) > capacity:
                raise SegmentationError("Oversized-entry sentinel does not fit the line budget.")
            groups.append(([sentinel], True))
            oversized_count += 1
            continue
        if current_size + cost > capacity:
            groups.append((current, False))
            current, current_size = [], 0
            cost = len(raw)
        current.append(entry)
        current_size += cost
    if current or not groups:
        groups.append((current, False))

    segment_count = len(groups)
    payloads = tuple(
        _segment_payload(
            segment_group_id=segment_group_id,
            parent_event_id=parent_event_id,
            field_pointer=field_pointer,
            encoding=encoding,
            segment_index=index,
            segment_count=segment_count,
            entries=group,
            entry_oversized=oversized,
        )
        for index, (group, oversized) in enumerate(groups)
    )
    for payload in payloads:
        if measure(payload) > budget:  # pragma: no cover - defensive: sizing above is exact
            raise SegmentationError("Segment packing exceeded the line budget.")
    declaration = SegmentedField(
        field_pointer=field_pointer,
        segment_group_id=segment_group_id,
        segment_count=segment_count,
    )
    return SegmentPlan(
        declaration=declaration,
        encoding=encoding,
        segment_payloads=payloads,
        oversized_entry_count=oversized_count,
    )


def object_entries(mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert a mapping into ``object_entries`` form (insertion order kept)."""
    return [{"key": key, "value": value} for key, value in mapping.items()]


def reassemble_array_slice(segment_payloads: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Concatenate ``array_slice`` segments in ``segment_index`` order."""
    ordered = sorted(segment_payloads, key=lambda payload: int(payload["segment_index"]))
    result: list[Any] = []
    for payload in ordered:
        result.extend(payload["entries"])
    return result


def reassemble_object_entries(segment_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge ``object_entries`` segments; duplicate keys raise (corruption)."""
    result: dict[str, Any] = {}
    for payload in sorted(segment_payloads, key=lambda payload: int(payload["segment_index"])):
        for entry in payload["entries"]:
            key = entry["key"]
            if key in result:
                raise SegmentationError(f"Duplicate object_entries key across segments: {key!r}")
            result[key] = entry["value"]
    return result
