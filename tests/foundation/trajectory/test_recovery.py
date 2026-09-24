# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chrys.foundation.trajectory.envelope import (
    SCHEMA_VERSION,
    EventDraft,
    build_event,
    encode_event_line,
    encode_json_line,
)
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.reader import read_trajectory
from chrys.foundation.trajectory.recovery import (
    _INITIAL_TAIL_BYTES,
    scan_for_recovery,
    scan_open_file,
    truncate_torn_tail,
)

SESSION_ID = "12345678-1234-1234-1234-123456789abc"
RUNTIME_ID = new_analytics_id()
COVERAGE_ID = new_analytics_id()
BRANCH_ID = new_analytics_id()


def _line(sequence: int, **payload: object) -> bytes:
    event = build_event(
        EventDraft(event_type=EventType.TURN_STARTED, payload=payload),
        sequence=sequence,
        runtime_id=RUNTIME_ID,
        coverage_id=COVERAGE_ID,
        session_id=SESSION_ID,
        branch_id=BRANCH_ID,
    )
    return encode_event_line(event)


def _newer_schema_line(sequence: int) -> bytes:
    """A line from a build that changed what a line means."""
    data = json.loads(_line(sequence))
    data["schema_version"] = SCHEMA_VERSION + 1
    data["actor"] = "main"
    return encode_json_line(data)


def _damaged_line(sequence: int) -> bytes:
    """A complete line whose header still reads but whose body no longer decodes."""
    data = json.loads(_line(sequence))
    data["actor"] = {"kind": "not-an-actor"}
    return encode_json_line(data)


def test_scan_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"")
    scan = scan_for_recovery(path)
    assert scan.file_size == 0
    assert scan.complete_offset == 0
    assert scan.last_sequence == 0
    assert scan.last_event is None
    assert scan.had_valid_events is False
    assert scan.truncated_bytes == 0


def test_scan_clean_file_recovers_last_sequence(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    data = _line(1) + _line(2) + _line(3)
    path.write_bytes(data)
    scan = scan_for_recovery(path)
    assert scan.complete_offset == len(data)
    assert scan.last_sequence == 3
    assert scan.last_event is not None
    assert scan.last_event.sequence == 3
    assert scan.had_valid_events
    assert scan.truncated_bytes == 0


def test_scan_and_truncate_torn_tail(tmp_path: Path) -> None:
    """Acceptance 5: the torn tail is truncated and sequence resumes from the last valid event."""
    path = tmp_path / "events.jsonl"
    complete = _line(1) + _line(2)
    torn = _line(3)[:-7]
    path.write_bytes(complete + torn)
    scan = scan_for_recovery(path)
    assert scan.complete_offset == len(complete)
    assert scan.truncated_bytes == len(torn)
    assert scan.last_sequence == 2
    fd = os.open(path, os.O_RDWR)
    try:
        assert truncate_torn_tail(fd, scan) == len(torn)
        assert truncate_torn_tail(fd, scan_for_recovery(path)) == 0
    finally:
        os.close(fd)
    assert path.read_bytes() == complete
    read = read_trajectory(path)
    assert [e.sequence for e in read.events] == [1, 2]
    assert read.torn_tail_bytes == 0


def test_scan_whole_file_single_torn_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1)[:-1])
    scan = scan_for_recovery(path)
    assert scan.complete_offset == 0
    assert scan.last_sequence == 0
    assert scan.had_valid_events is False
    assert scan.truncated_bytes == path.stat().st_size


def test_a_trailing_line_that_names_no_slot_leaves_nothing_to_resume_from(tmp_path: Path) -> None:
    """A complete-but-corrupt last line keeps its bytes, and its slot stays unknowable."""
    path = tmp_path / "events.jsonl"
    data = _line(1) + _line(2) + b"{not json\n"
    path.write_bytes(data)

    scan = scan_for_recovery(path)

    assert scan.complete_offset == len(data)
    # The last readable event is 2, but the line after it took some slot no
    # header names: resuming at 3 would hand out a slot that may be spent.
    assert scan.last_sequence == 2
    assert scan.unreadable_tail is True


def test_a_line_that_names_no_slot_behind_a_readable_event_is_not_a_tail(tmp_path: Path) -> None:
    """Only damage past the last readable event can hide a spent slot."""
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + b"{not json\n" + _line(3))

    scan = scan_for_recovery(path)

    # Slot 3 is on disk and readable, so what the line before it took changes
    # nothing about which sequence comes next.
    assert scan.last_sequence == 3
    assert scan.unreadable_tail is False


def test_scan_walks_back_past_the_initial_tail_window(tmp_path: Path) -> None:
    """A final valid event buried under a huge torn tail is still found."""
    path = tmp_path / "events.jsonl"
    complete = _line(1) + _line(2, marker="last-valid")
    torn = b'{"huge": "' + b"x" * (200 * 1024) + b'"'  # no newline: torn
    path.write_bytes(complete + torn)
    scan = scan_for_recovery(path)
    assert scan.complete_offset == len(complete)
    assert scan.last_sequence == 2
    assert scan.last_event is not None
    assert scan.last_event.payload["marker"] == "last-valid"
    assert scan.truncated_bytes == len(torn)


def test_scan_handles_many_small_lines_beyond_window(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    lines = b"".join(_line(n) for n in range(1, 400))  # well beyond 64 KiB
    path.write_bytes(lines + _line(400)[:-3])
    scan = scan_for_recovery(path)
    assert scan.last_sequence == 399
    assert scan.complete_offset == len(lines)


def test_a_newer_schema_tail_blocks_the_scan_instead_of_rewinding(tmp_path: Path) -> None:
    """The slot a newer line took is unknown, so there is nothing to resume from."""
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + _newer_schema_line(2))

    scan = scan_for_recovery(path)

    assert scan.newer_schema_version == SCHEMA_VERSION + 1
    # Emphatically not sequence 1: resuming there would hand slot 2 out twice.
    assert scan.last_sequence == 0
    assert scan.had_valid_events is False


def test_a_newer_schema_line_earlier_in_the_file_does_not_block_the_tail(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + _newer_schema_line(2) + _line(3))

    scan = scan_for_recovery(path)

    # The last line is one this build wrote and can read: its slot is known,
    # so appending after it takes nothing that is already used.
    assert scan.newer_schema_version is None
    assert scan.last_sequence == 3


def test_a_damaged_line_still_spends_the_slot_its_header_names(tmp_path: Path) -> None:
    """The bytes stay on disk, so the sequence they claim is not free to reuse."""
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + _line(2) + _damaged_line(3))

    scan = scan_for_recovery(path)

    assert scan.last_sequence == 3  # not 2: writing 3 again would double up on that slot
    assert scan.last_event is not None
    assert scan.last_event.sequence == 2  # the runtime facts still come from the last readable event
    assert scan.had_valid_events
    # Slot 3 is spent but will never reach a reader: appending past it leaves a
    # hole, and the resuming writer is told which one to explain.
    assert scan.unreadable_slots == (3, 3)


def test_a_file_of_nothing_but_damaged_lines_still_reports_their_slots(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_damaged_line(1) + _damaged_line(2))

    scan = scan_for_recovery(path)

    assert scan.last_sequence == 2
    assert scan.last_event is None
    assert scan.had_valid_events is False
    assert scan.unreadable_slots == (1, 2)  # nothing in this file reads, so no slot in it is accounted for


def test_a_slot_too_large_to_encode_is_not_a_slot_the_scan_counts(tmp_path: Path) -> None:
    """A header naming a sequence past int64 names nothing this build can resume from."""
    path = tmp_path / "events.jsonl"
    data = json.loads(_line(3))
    data["sequence"] = 2**63  # parses as JSON, rejected by the decoder as out of range
    path.write_bytes(_line(1) + _line(2) + encode_json_line(data))

    scan = scan_for_recovery(path)

    # Emphatically not 2**63: resuming there would append lines every reader
    # rejects, in place of the events that should have followed.
    assert scan.last_sequence == 2
    assert scan.unreadable_tail is True


def test_a_clean_file_owes_no_gap(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + _line(2))

    assert scan_for_recovery(path).unreadable_slots is None


def test_damage_past_the_first_window_still_finds_the_events_behind_it(tmp_path: Path) -> None:
    """A tail window is where the scan starts, not how far back a valid event counts."""
    path = tmp_path / "events.jsonl"
    padding = b"".join(_line(n) for n in range(1, 400))
    path.write_bytes(padding + _damaged_line(400))
    assert len(padding) > _INITIAL_TAIL_BYTES  # the last valid event is outside the first read

    scan = scan_for_recovery(path)

    assert scan.had_valid_events  # widened until it found one, rather than calling the file eventless
    assert scan.last_event is not None
    assert scan.last_event.sequence == 399
    assert scan.last_sequence == 400
    assert scan.unreadable_slots == (400, 400)


@pytest.mark.skipif(os.name == "nt", reason="swapping the file a handle holds open is POSIX-only")
def test_scanning_a_descriptor_reads_the_file_it_holds_not_the_path(tmp_path: Path) -> None:
    """A pathname can come to name a different file; the descriptor cannot."""
    path = tmp_path / "events.jsonl"
    path.write_bytes(_line(1) + _line(2) + _line(3))
    decoy = tmp_path / "decoy.jsonl"
    decoy.write_bytes(_line(1))

    fd = os.open(path, os.O_RDONLY)
    try:
        os.replace(decoy, path)
        scan = scan_open_file(fd)
    finally:
        os.close(fd)

    assert scan.last_sequence == 3
    assert scan_for_recovery(path).last_sequence == 1  # what reopening the name would have said
