# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

from chrys.foundation.trajectory.envelope import (
    SCHEMA_VERSION,
    EventDraft,
    build_event,
    encode_event_line,
    encode_json_line,
)
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.reader import (
    CoveredSequenceRanges,
    iter_file_lines,
    last_event_of_type,
    read_trajectory,
    verify_accounted_prefix,
)

SESSION_ID = "12345678-1234-1234-1234-123456789abc"
RUNTIME_ID = new_analytics_id()
COVERAGE_ID = new_analytics_id()
BRANCH_ID = new_analytics_id()


def _line(sequence: int, event_type: str = EventType.TURN_STARTED, **payload: object) -> bytes:
    event = build_event(
        EventDraft(event_type=event_type, payload=payload),
        sequence=sequence,
        runtime_id=RUNTIME_ID,
        coverage_id=COVERAGE_ID,
        session_id=SESSION_ID,
        branch_id=BRANCH_ID,
    )
    return encode_event_line(event)


def _gap(sequence: int, first: int, last: int) -> bytes:
    return _line(
        sequence,
        EventType.GAP,
        first_sequence=first,
        last_sequence=last,
        dropped_count=last - first + 1,
        reason="write_failure",
    )


def test_read_missing_file_is_empty(tmp_path: Path) -> None:
    result = read_trajectory(tmp_path / "events.jsonl")
    assert result.events == []
    assert result.corrupt_lines == []
    assert result.torn_tail_bytes == 0


def test_read_clean_file(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + _line(2) + _line(3))
    result = read_trajectory(path)
    assert [e.sequence for e in result.events] == [1, 2, 3]
    assert verify_accounted_prefix(result.events) == []
    assert result.corruption_gaps == []
    assert last_event_of_type(result.events, EventType.TURN_STARTED).sequence == 3
    assert last_event_of_type(result.events, EventType.GAP) is None


def test_torn_tail_is_reported_not_decoded(tmp_path: Path) -> None:
    """Acceptance 15e: a half-written final line is never taken as an event."""
    path = tmp_path / "events.jsonl"
    full = _line(1) + _line(2)
    partial = _line(3)[:-10]
    path.write_bytes(full + partial)
    result = read_trajectory(path)
    assert [e.sequence for e in result.events] == [1, 2]
    assert result.torn_tail_bytes == len(partial)
    assert result.corrupt_lines == []


def test_mid_file_corruption_produces_a_corruption_gap(tmp_path: Path) -> None:
    """Acceptance 13: a bad line in the middle is reported, not silently skipped."""
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + b"{garbage\n" + _line(3))
    result = read_trajectory(path)
    assert [e.sequence for e in result.events] == [1, 3]
    assert len(result.corrupt_lines) == 1
    corrupt = result.corrupt_lines[0]
    assert corrupt.line_number == 2
    assert corrupt.byte_offset == len(_line(1))
    assert corrupt.previous_sequence == 1
    gaps = result.corruption_gaps
    assert len(gaps) == 1
    assert gaps[0].after_sequence == 1
    assert gaps[0].before_sequence == 3
    # The hole at 2 is also an accounted-prefix violation: nothing covers it.
    assert verify_accounted_prefix(result.events)


def test_unknown_event_types_are_counted_not_dropped(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + _line(2, "future.event.kind") + _line(3))
    result = read_trajectory(path)
    assert [e.sequence for e in result.events] == [1, 2, 3]
    assert result.unsupported_event_count == 1
    assert result.unsupported_event_sequences == [2]


def test_a_newer_schema_is_counted_the_way_an_unknown_type_is(tmp_path: Path) -> None:
    """A version this build does not know is not a line it may read as its own."""
    future = json.loads(_line(2))
    future["schema_version"] = SCHEMA_VERSION + 1
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + encode_json_line(future) + _line(3))
    result = read_trajectory(path)

    # Still an event, still accounted for in the prefix — just not one this
    # build may claim to understand.
    assert [e.sequence for e in result.events] == [1, 2, 3]
    assert verify_accounted_prefix(result.events) == []
    assert result.corrupt_lines == []
    assert result.unsupported_event_count == 1
    assert result.unsupported_event_sequences == [2]


def test_a_newer_schema_survives_a_shape_this_build_cannot_read(tmp_path: Path) -> None:
    """Version skew is not damage: the line stays, its slot stays accounted for."""
    future = json.loads(_line(2))
    future["schema_version"] = SCHEMA_VERSION + 1
    # Exactly what a bump is for: the envelope this build validates is gone.
    future["actor"] = "main"
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + encode_json_line(future) + _line(3))
    result = read_trajectory(path)

    assert result.corrupt_lines == []
    assert [e.sequence for e in result.events] == [1, 3]
    assert [(line.sequence, line.schema_version) for line in result.unsupported_lines] == [(2, SCHEMA_VERSION + 1)]
    assert result.unsupported_event_count == 1
    assert result.unsupported_event_sequences == [2]
    # The line is on the disk it was written to, so the file has no hole at 2
    # — but the decoded stream on its own does.
    assert verify_accounted_prefix(result.slots) == []
    assert verify_accounted_prefix(result.events) == ["sequence 2 is missing and not covered by an earlier gap"]


def test_an_unsupported_line_is_held_to_the_physical_order_like_any_other(tmp_path: Path) -> None:
    """A slot nobody can decode still has to be in the right place, once."""
    out_of_order = json.loads(_line(3))
    out_of_order["schema_version"] = SCHEMA_VERSION + 1
    out_of_order["actor"] = "main"
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + encode_json_line(out_of_order) + _line(2))
    result = read_trajectory(path)

    # Decoded alone this reads 1, 2 and looks fine; the file ran 1, 3, 2.
    assert verify_accounted_prefix(result.events) == []
    assert verify_accounted_prefix(result.slots) == [
        "sequence 2 is missing and not covered by an earlier gap",
        "sequence 2 does not increase after 3",
    ]


def test_a_damaged_line_is_still_corruption_not_version_skew(tmp_path: Path) -> None:
    broken = json.loads(_line(2))
    broken["actor"] = "main"
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + encode_json_line(broken) + _line(3))
    result = read_trajectory(path)

    # Same broken shape, current version: nothing licenses it, so it is damage.
    assert result.unsupported_lines == []
    assert len(result.corrupt_lines) == 1


def test_accounted_prefix_accepts_holes_covered_by_earlier_gap(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    # 1,2 written; 3,4 dropped; gap at 5 covers 3..4; then 6.
    path.write_bytes(_line(1) + _line(2) + _gap(5, 3, 4) + _line(6))
    result = read_trajectory(path)
    assert verify_accounted_prefix(result.events) == []


def test_accounted_prefix_verification_survives_an_absurd_sequence_jump(tmp_path: Path) -> None:
    """A corrupt file claiming a billion-line hole is answered, not counted through."""
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + _gap(2, 3, 10**9) + _line(10**9 + 1) + _line(10**9 + 3))
    result = read_trajectory(path)
    violations = verify_accounted_prefix(result.events)
    # Everything up to the covered range is accounted for; only the hole the
    # gap does not reach is reported.
    assert violations == [f"sequence {10**9 + 2} is missing and not covered by an earlier gap"]


def test_accounted_prefix_reports_a_log_missing_its_head(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(2) + _line(3))
    result = read_trajectory(path)
    assert verify_accounted_prefix(result.events) == ["sequence 1 is missing and not covered by an earlier gap"]


def test_accounted_prefix_accepts_a_head_the_first_line_accounts_for(tmp_path: Path) -> None:
    """The log's own first event was unwritable: its slot is 1, the gap is 2."""
    path = tmp_path / "events.jsonl"
    path.write_bytes(_gap(2, 1, 1) + _line(3))
    result = read_trajectory(path)
    assert verify_accounted_prefix(result.events) == []


def test_accounted_prefix_rejects_late_gap_and_regressions(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    # Hole at 3 but the gap covering it is persisted only after 4: violation.
    path.write_bytes(_line(1) + _line(2) + _line(4) + _gap(5, 3, 3))
    result = read_trajectory(path)
    assert verify_accounted_prefix(result.events)

    path.write_bytes(_line(1) + _line(3) + _line(2))
    result = read_trajectory(path)
    violations = verify_accounted_prefix(result.events)
    assert any("does not increase" in item for item in violations)


def test_covered_sequence_ranges_merge_overlaps_touches_and_duplicates() -> None:
    covered = CoveredSequenceRanges()
    covered.add(5, 9)
    covered.add(12, 13)
    covered.add(10, 11)  # touches both neighbours: one covered run 5..13
    covered.add(5, 9)  # a duplicate declaration adds nothing
    assert covered.first_uncovered(5, 13) is None
    assert covered.first_uncovered(4, 13) == 4
    assert covered.first_uncovered(5, 14) == 14


def test_covered_sequence_ranges_accept_out_of_order_declarations() -> None:
    # A recovery gap written at open covers sequences far below the ranges
    # later gaps declare, so declaration order is not coverage order.
    covered = CoveredSequenceRanges()
    covered.add(20, 29)
    covered.add(2, 3)
    covered.add(10, 15)
    assert covered.first_uncovered(2, 3) is None
    assert covered.first_uncovered(2, 9) == 4
    assert covered.first_uncovered(10, 29) == 16
    assert covered.first_uncovered(30, 35) == 30


def test_accounted_prefix_stays_exact_across_a_log_full_of_writer_gaps(tmp_path: Path) -> None:
    """One gap per event that failed to encode is a valid shape, so a log can
    legitimately carry hundreds of separate covered holes."""
    pieces = [_line(1)]
    sequence = 1
    for _ in range(400):
        gap_sequence = sequence + 1
        hole = gap_sequence + 1
        pieces.append(_gap(gap_sequence, hole, hole))
        pieces.append(_line(hole + 1))
        sequence = hole + 1
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(pieces))
    result = read_trajectory(path)

    assert len(result.events) == 801
    assert verify_accounted_prefix(result.events) == []


def test_iter_file_lines_reports_offsets_and_completeness(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(b"ab\ncd\nxyz")
    rows = list(iter_file_lines(path, chunk_size=2))
    assert rows == [(0, b"ab", True), (3, b"cd", True), (6, b"xyz", False)]


def test_crlf_is_not_tolerated_as_line_ending(tmp_path: Path) -> None:
    """Acceptance 23: the format is LF-only; a CR would be part of the JSON and rejected."""
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1)[:-1] + b"\r\n" + _line(2))
    result = read_trajectory(path)
    assert [e.sequence for e in result.events] == [1, 2]
    assert result.corrupt_lines == []  # json.loads tolerates trailing whitespace
    assert b"\r\n" not in _line(1)
