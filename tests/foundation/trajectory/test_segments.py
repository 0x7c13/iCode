# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from chrys.foundation.trajectory.envelope import EventDraft, build_event, encode_event_line
from chrys.foundation.trajectory.ids import is_valid_analytics_id, new_analytics_id
from chrys.foundation.trajectory.segments import (
    ENCODING_ARRAY_SLICE,
    ENCODING_OBJECT_ENTRIES,
    SegmentationError,
    object_entries,
    plan_segments,
    reassemble_array_slice,
    reassemble_object_entries,
)

SESSION_ID = "12345678-1234-1234-1234-123456789abc"
RUNTIME_ID = new_analytics_id()
COVERAGE_ID = new_analytics_id()
BRANCH_ID = new_analytics_id()


def _measure(payload: dict[str, Any]) -> int:
    event = build_event(
        EventDraft(event_type="event.segment", payload=payload),
        sequence=10**9,
        runtime_id=RUNTIME_ID,
        coverage_id=COVERAGE_ID,
        session_id=SESSION_ID,
        branch_id=BRANCH_ID,
    )
    return len(encode_event_line(event))


def _hasher(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_array_slice_segments_are_contiguous_under_budget_and_reassemble() -> None:
    """Acceptance 18: indexes run 0..count-1 without duplicates; every line fits."""
    entries = [{"item_id": new_analytics_id(), "action": "add", "occurrence": i, "position": i} for i in range(300)]
    parent = new_analytics_id()
    plan = plan_segments(
        parent_event_id=parent,
        field_pointer="/payload/entries",
        encoding=ENCODING_ARRAY_SLICE,
        entries=entries,
        measure=_measure,
        hasher=_hasher,
    )
    assert plan.segment_count > 1
    assert plan.oversized_entry_count == 0
    assert plan.declaration.field_pointer == "/payload/entries"
    assert plan.declaration.segment_count == plan.segment_count
    assert is_valid_analytics_id(plan.declaration.segment_group_id)
    indexes = [payload["segment_index"] for payload in plan.segment_payloads]
    assert indexes == list(range(plan.segment_count))
    for payload in plan.segment_payloads:
        assert payload["segment_group_id"] == plan.declaration.segment_group_id
        assert payload["parent_event_id"] == parent
        assert payload["field_pointer"] == "/payload/entries"
        assert payload["segment_count"] == plan.segment_count
        assert payload["encoding"] == ENCODING_ARRAY_SLICE
        assert "entry_oversized" not in payload
        assert _measure(payload) <= 4096
    assert reassemble_array_slice(plan.segment_payloads) == entries
    # Order of delivery does not matter for reassembly.
    assert reassemble_array_slice(list(reversed(plan.segment_payloads))) == entries


def test_object_entries_segments_reassemble_into_mapping() -> None:
    mapping = {f"key_{i}": {"count": i, "tokens": i * 7} for i in range(200)}
    plan = plan_segments(
        parent_event_id=new_analytics_id(),
        field_pointer="/payload/buckets",
        encoding=ENCODING_OBJECT_ENTRIES,
        entries=object_entries(mapping),
        measure=_measure,
        hasher=_hasher,
    )
    assert plan.segment_count > 1
    assert reassemble_object_entries(plan.segment_payloads) == mapping


def test_object_entries_reject_duplicate_keys_and_bad_shapes() -> None:
    with pytest.raises(SegmentationError):
        plan_segments(
            parent_event_id=new_analytics_id(),
            field_pointer="/p",
            encoding=ENCODING_OBJECT_ENTRIES,
            entries=[{"key": "a", "value": 1}, {"key": "a", "value": 2}],
            measure=_measure,
            hasher=_hasher,
        )
    with pytest.raises(SegmentationError):
        plan_segments(
            parent_event_id=new_analytics_id(),
            field_pointer="/p",
            encoding=ENCODING_OBJECT_ENTRIES,
            entries=[{"k": "a", "v": 1}],
            measure=_measure,
            hasher=_hasher,
        )
    with pytest.raises(SegmentationError):
        reassemble_object_entries(
            [
                {"segment_index": 0, "entries": [{"key": "a", "value": 1}]},
                {"segment_index": 1, "entries": [{"key": "a", "value": 2}]},
            ]
        )


def test_unknown_encoding_is_rejected() -> None:
    with pytest.raises(SegmentationError):
        plan_segments(
            parent_event_id=new_analytics_id(),
            field_pointer="/p",
            encoding="blob",
            entries=[1],
            measure=_measure,
            hasher=_hasher,
        )


def test_oversized_entry_becomes_a_hashed_sentinel_not_a_dropped_field() -> None:
    big = "x" * 6000
    entries = [{"n": 1}, {"blob": big}, {"n": 3}]
    plan = plan_segments(
        parent_event_id=new_analytics_id(),
        field_pointer="/payload/entries",
        encoding=ENCODING_ARRAY_SLICE,
        entries=entries,
        measure=_measure,
        hasher=_hasher,
    )
    assert plan.oversized_entry_count == 1
    oversized = [payload for payload in plan.segment_payloads if payload.get("entry_oversized")]
    assert len(oversized) == 1
    sentinel = oversized[0]["entries"][0]
    assert sentinel["omitted"] is True
    assert sentinel["byte_length"] > 6000
    assert sentinel["value_hash"] == _hasher(b'{"blob":"' + big.encode() + b'"}')
    assert big not in str(plan.segment_payloads)
    rebuilt = reassemble_array_slice(plan.segment_payloads)
    assert rebuilt[0] == {"n": 1}
    assert rebuilt[2] == {"n": 3}
    assert rebuilt[1]["omitted"] is True


def test_an_oversized_entry_without_a_key_records_its_size_and_no_digest() -> None:
    """An unkeyed digest of a short value is guessable, so none is written."""
    plan = plan_segments(
        parent_event_id=new_analytics_id(),
        field_pointer="/payload/entries",
        encoding=ENCODING_ARRAY_SLICE,
        entries=[{"blob": "x" * 6000}],
        measure=_measure,
        hasher=None,
    )
    sentinel = plan.segment_payloads[0]["entries"][0]
    assert sentinel["omitted"] is True
    assert sentinel["byte_length"] > 6000
    assert "value_hash" not in sentinel


def test_oversized_object_entry_keeps_its_key() -> None:
    plan = plan_segments(
        parent_event_id=new_analytics_id(),
        field_pointer="/payload/buckets",
        encoding=ENCODING_OBJECT_ENTRIES,
        entries=object_entries({"small": 1, "huge": "y" * 5000}),
        measure=_measure,
        hasher=_hasher,
    )
    rebuilt = reassemble_object_entries(plan.segment_payloads)
    assert rebuilt["small"] == 1
    assert rebuilt["huge"]["omitted"] is True
    assert rebuilt["huge"]["byte_length"] == 5002


def test_empty_entries_still_produce_one_segment() -> None:
    plan = plan_segments(
        parent_event_id=new_analytics_id(),
        field_pointer="/p",
        encoding=ENCODING_ARRAY_SLICE,
        entries=[],
        measure=_measure,
        hasher=_hasher,
    )
    assert plan.segment_count == 1
    assert plan.segment_payloads[0]["entries"] == []
    assert reassemble_array_slice(plan.segment_payloads) == []


def test_tiny_budget_is_rejected_when_envelope_does_not_fit() -> None:
    with pytest.raises(SegmentationError):
        plan_segments(
            parent_event_id=new_analytics_id(),
            field_pointer="/p",
            encoding=ENCODING_ARRAY_SLICE,
            entries=[1],
            measure=_measure,
            hasher=_hasher,
            budget=100,
        )
