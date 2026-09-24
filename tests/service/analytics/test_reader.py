# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Streaming diagnostics and live-tail behavior for trajectory analysis."""

from __future__ import annotations

import json
import os
import tracemalloc
from hashlib import blake2b
from threading import Event

import pytest

from chrys.foundation.trajectory.envelope import SCHEMA_VERSION
from chrys.foundation.trajectory.event_types import EventType
from chrys.service.analytics import (
    AnalysisAvailability,
    TrajectoryAnalyzer,
    TrajectoryScanCancelled,
    analyze_trajectory,
)
from chrys.service.analytics import _turns as turns_module
from chrys.service.analytics import aggregation as aggregation_module
from chrys.service.analytics.reader import scan_trajectory_batch
from tests.service.analytics._events import EventLog


def test_missing_trajectory_is_explicitly_unavailable(tmp_path) -> None:
    analysis = TrajectoryAnalyzer().load(tmp_path / "missing.jsonl")

    assert analysis.availability is AnalysisAvailability.UNAVAILABLE
    assert analysis.overview is None
    assert analysis.turns == ()


def test_live_tail_reuses_prefix_and_advances_generation(tmp_path) -> None:
    first = EventLog()
    first.coverage()
    path = tmp_path / "events.jsonl"
    first.write(path)
    analyzer = TrajectoryAnalyzer()
    initial = analyzer.load(path)
    appended = EventLog()
    appended.turn(1, 2)
    append_bytes = path.read_bytes()
    appended.write(tmp_path / "append.jsonl", start_sequence=2)
    path.write_bytes(append_bytes + (tmp_path / "append.jsonl").read_bytes())

    refreshed = analyzer.refresh()

    assert refreshed.generation == initial.generation + 1
    assert len(refreshed.turns) == 1


def test_same_size_rewrite_invalidates_the_cached_prefix(tmp_path) -> None:
    initial_log = EventLog()
    initial_log.coverage()
    initial_log.turn(1, 2)
    path = tmp_path / "events.jsonl"
    initial_log.write(path)
    analyzer = TrajectoryAnalyzer()
    initial = analyzer.load(path)
    initial_stat = path.stat()

    replacement = EventLog()
    replacement.coverage()
    replacement.turn(3, 5)
    replacement.write(path)
    replacement_stat = path.stat()
    assert replacement_stat.st_size == initial_stat.st_size
    # Windows can preserve the previous mtime across two rapid same-size
    # writes. Advance it explicitly so this test deterministically exercises
    # the analyzer's same-size replacement path instead of its unchanged-file
    # cache hit.
    os.utime(
        path,
        ns=(replacement_stat.st_atime_ns, initial_stat.st_mtime_ns + 1_000_000_000),
    )
    assert path.stat().st_mtime_ns != initial_stat.st_mtime_ns

    refreshed = analyzer.refresh()

    assert refreshed.generation == initial.generation + 1
    assert refreshed.turns[0].elapsed_ns.value == 2


def test_same_inode_larger_rewrite_invalidates_changed_consumed_prefix(tmp_path, monkeypatch) -> None:
    initial_log = EventLog()
    initial_log.coverage()
    initial_log.turn(0, 1)
    path = tmp_path / "events.jsonl"
    initial_log.write(path)
    analyzer = TrajectoryAnalyzer()
    analyzer.load(path)
    identity = (path.stat().st_dev, path.stat().st_ino)
    # The growth is far below the amortized replay threshold, so only the
    # boundary probe of the retained-digest fast path can catch this; a full
    # replay would mask a hole in that branch.
    monkeypatch.setattr(aggregation_module, "verify_prefix", _fail_verify)

    replacement = EventLog()
    replacement.coverage()
    replacement.add("turn.started", 0, payload={"turn_number": 1})
    replacement.add("turn.finished", 9, payload={"end_reason": "cancelled", "duration_ms": 0})
    replacement.add("turn.started", 10, turn_id="5" * 32, payload={"turn_number": 2})
    replacement.add(
        "turn.finished",
        12,
        turn_id="5" * 32,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    replacement.write(path)

    refreshed = analyzer.refresh()

    assert (path.stat().st_dev, path.stat().st_ino) == identity
    assert [turn.elapsed_ns.value for turn in refreshed.turns] == [9, 2]


def _fail_verify(*args, **kwargs):
    raise AssertionError("the retained-digest fast path must not fall back to a prefix replay")


def _turn_batch(tmp_path, name: str, start_sequence: int, entries: list[tuple[str, int, int]]) -> bytes:
    log = EventLog()
    for turn_id, turn_number, start_ns in entries:
        log.add("turn.started", start_ns, turn_id=turn_id, payload={"turn_number": turn_number})
        log.add(
            "turn.finished",
            start_ns + 2,
            turn_id=turn_id,
            payload={"end_reason": "cancelled", "duration_ms": 0},
        )
    log.write(tmp_path / name, start_sequence=start_sequence)
    return (tmp_path / name).read_bytes()


def test_append_refreshes_continue_the_digest_state_and_replay_only_at_doubling(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(aggregation_module, "_PREFIX_REVERIFY_FLOOR_BYTES", 1)
    replays: list[bool] = []
    original_verify = aggregation_module.verify_prefix

    def recording_verify(handle, **kwargs):
        result = original_verify(handle, **kwargs)
        replays.append(result is not None)
        return result

    monkeypatch.setattr(aggregation_module, "verify_prefix", recording_verify)
    log = EventLog()
    log.coverage()
    log.turn(1, 2)
    path = tmp_path / "events.jsonl"
    log.write(path)
    analyzer = TrajectoryAnalyzer()
    assert len(analyzer.load(path).turns) == 1
    initial_size = path.stat().st_size

    first = _turn_batch(tmp_path, "a.jsonl", 4, [("5" * 32, 2, 10), ("6" * 32, 3, 20), ("7" * 32, 4, 30)])
    assert len(first) >= initial_size  # after this batch the replay is due
    with path.open("ab") as handle:
        handle.write(first)
    assert len(analyzer.refresh().turns) == 4
    assert replays == []  # the retained digest state continued; no replay

    second = _turn_batch(tmp_path, "b.jsonl", 10, [("8" * 32, 5, 40)])
    assert len(second) < initial_size + len(first)  # next batch is not due again
    with path.open("ab") as handle:
        handle.write(second)
    assert len(analyzer.refresh().turns) == 5
    assert replays == [True]  # the due replay revalidated the continued digest

    with path.open("ab") as handle:
        handle.write(_turn_batch(tmp_path, "c.jsonl", 12, [("9" * 32, 6, 50)]))
    assert len(analyzer.refresh().turns) == 6
    assert replays == [True]  # the verified length doubled; nothing new is due


def test_due_prefix_replay_detects_an_out_of_contract_prefix_rewrite(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(aggregation_module, "_PREFIX_REVERIFY_FLOOR_BYTES", 1)
    # Shrink the boundary probe so the rewrite below lands outside its window
    # and detection falls to the amortized replay, not the probe.
    monkeypatch.setattr(aggregation_module, "_PREFIX_PROBE_BYTES", 64)
    log = EventLog()
    log.coverage()
    log.turn(1, 2)
    path = tmp_path / "events.jsonl"
    log.write(path)
    analyzer = TrajectoryAnalyzer()
    analyzer.load(path)
    initial_size = path.stat().st_size
    first = _turn_batch(tmp_path, "a.jsonl", 4, [("5" * 32, 2, 10), ("6" * 32, 3, 20), ("7" * 32, 4, 30)])
    assert len(first) >= max(initial_size, 64)  # replay due next batch; rewrite stays outside the probe
    with path.open("ab") as handle:
        handle.write(first)
    stale = _elapsed(analyzer.refresh())

    lines = path.read_bytes().split(b"\n")
    lines[2] = b"x" * len(lines[2])  # rewrite the first turn's terminal line in place
    path.write_bytes(b"\n".join(lines) + _turn_batch(tmp_path, "b.jsonl", 10, [("8" * 32, 5, 40)]))

    refreshed = analyzer.refresh()

    assert _elapsed(refreshed) == _elapsed(analyze_trajectory(path))
    assert _elapsed(refreshed) != [*stale, 2]  # the stale cache did not survive


def _elapsed(analysis) -> list[int | None]:
    return [None if turn.elapsed_ns is None else turn.elapsed_ns.value for turn in analysis.turns]


def test_cancelled_scan_drops_partial_intermediate_state(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.turn(0, 1)
    path = tmp_path / "events.jsonl"
    log.write(path)
    cancel_event = Event()
    cancel_event.set()
    analyzer = TrajectoryAnalyzer()

    with pytest.raises(TrajectoryScanCancelled):
        analyzer.load(path, cancel_event=cancel_event)

    assert analyzer._intermediate is None


def test_cancellation_during_eof_resolve_drops_partial_intermediate_state(tmp_path, monkeypatch) -> None:
    log = EventLog()
    log.coverage()
    log.turn(0, 1)
    log.add("turn.started", 2, turn_id="5" * 32, payload={"turn_number": 2})
    log.add("turn.finished", 3, turn_id="5" * 32, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "resolve-cancel.jsonl"
    log.write(path)
    cancel_event = Event()
    analyzer = TrajectoryAnalyzer()
    original = turns_module._resolve_turn
    resolved_turn_count = 0

    def cancel_after_first_turn(*args, **kwargs):
        nonlocal resolved_turn_count
        result = original(*args, **kwargs)
        resolved_turn_count += 1
        cancel_event.set()
        return result

    monkeypatch.setattr(turns_module, "_resolve_turn", cancel_after_first_turn)

    with pytest.raises(TrajectoryScanCancelled):
        analyzer.load(path, cancel_event=cancel_event)

    assert resolved_turn_count == 1
    assert analyzer._intermediate is None


def test_over_budget_lines_are_corrupt_without_retention_and_the_scan_continues(tmp_path) -> None:
    prefix_log = EventLog()
    prefix_log.coverage()
    path = tmp_path / "events.jsonl"
    prefix_log.write(path)
    prefix = path.read_bytes()
    streamed = b"{" + b"x" * 200_000 + b"}\n"  # crosses several read chunks
    retained = b"{" + b"y" * 8_000 + b"}\n"  # over budget within one chunk
    suffix_log = EventLog()
    suffix_log.turn(1, 2)
    suffix_log.write(tmp_path / "suffix.jsonl", start_sequence=2)
    suffix = (tmp_path / "suffix.jsonl").read_bytes()
    content = prefix + streamed + retained + suffix
    path.write_bytes(content)

    batch = scan_trajectory_batch(path, lambda *args: None)

    assert [(line.line_number, line.byte_length) for line in batch.corrupt_lines] == [
        (2, len(streamed)),
        (3, len(retained)),
    ]
    assert all("line budget" in line.reason for line in batch.corrupt_lines)
    assert batch.event_count == 3
    assert batch.prefix_violations == []
    assert batch.torn_tail_bytes == 0
    assert batch.cursor.byte_offset == len(content)
    assert batch.prefix_digest == blake2b(content, digest_size=16).digest()


def test_torn_over_budget_tail_keeps_digest_and_cursor_at_consumed_lines(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    path = tmp_path / "events.jsonl"
    log.write(path)
    consumed = path.read_bytes()
    path.write_bytes(consumed + b"{" + b"x" * 150_000)

    batch = scan_trajectory_batch(path, lambda *args: None)

    assert batch.torn_tail_bytes == 150_001
    assert batch.corrupt_lines == []
    assert batch.event_count == 1
    assert batch.cursor.byte_offset == len(consumed)
    assert batch.prefix_digest == blake2b(consumed, digest_size=16).digest()
    assert batch.prefix_hasher is not None
    assert batch.prefix_hasher.digest() == batch.prefix_digest


def test_scan_memory_stays_bounded_for_a_single_giant_line(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    path = tmp_path / "events.jsonl"
    log.write(path)
    piece = b"x" * (1 << 16)
    with path.open("ab") as handle:
        handle.write(b"{")
        for _ in range(32):  # one ~2 MiB line
            handle.write(piece)
        handle.write(b"}\n")

    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        batch = scan_trajectory_batch(path, lambda *args: None)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert batch.corrupt_lines[0].byte_length == 2 + 32 * (1 << 16) + 1
    assert peak - before < (1 << 20)


def test_analysis_retains_unsupported_line_sequence_and_location(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("profile.switched", 1, payload={"profile": "Code"})
    log.add("turn.finished", 2, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "unsupported-detail.jsonl"
    log.write(path)
    lines = path.read_bytes().splitlines()
    unsupported = json.loads(lines[2])
    unsupported["event_type"] = "future.event"
    lines[2] = json.dumps(unsupported).encode()
    path.write_bytes(b"\n".join(lines) + b"\n")

    diagnostic = analyze_trajectory(path).diagnostics.unsupported_lines[0]

    assert diagnostic.line_number == 3
    assert diagnostic.sequence == 3
    assert diagnostic.byte_offset > 0


def test_gap_coverage_from_an_earlier_batch_covers_a_hole_a_refresh_appends(tmp_path) -> None:
    """The cursor's covered ranges persist across append batches: a gap event
    consumed by the initial load accounts for a hole that only arrives later."""
    log = EventLog()
    log.coverage()
    log.add(
        EventType.GAP,
        0,
        turn_id=None,
        payload={"first_sequence": 3, "last_sequence": 4, "dropped_count": 2, "reason": "write_failure"},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)
    analyzer = TrajectoryAnalyzer()
    loaded = analyzer.load(path)
    assert loaded.diagnostics.accounted_prefix_violations == ()

    with path.open("ab") as handle:
        handle.write(_turn_batch(tmp_path, "gap-suffix.jsonl", 5, [("a" * 32, 1, 10)]))
    refreshed = analyzer.refresh()

    assert refreshed.diagnostics.accounted_prefix_violations == ()


def test_future_schema_gap_accounts_for_its_slot_without_covering_or_reaching_consumer(tmp_path) -> None:
    prefix = EventLog()
    prefix.coverage()
    path = tmp_path / "future-gap.jsonl"
    prefix.write(path)
    suffix = EventLog()
    suffix.add(
        EventType.GAP,
        1,
        turn_id=None,
        payload={"first_sequence": 2, "last_sequence": 2, "dropped_count": 1, "reason": "write_failure"},
    )
    suffix_path = tmp_path / "suffix.jsonl"
    suffix.write(suffix_path, start_sequence=3)
    future_gap = json.loads(suffix_path.read_bytes())
    future_gap["schema_version"] = SCHEMA_VERSION + 1
    path.write_bytes(path.read_bytes() + json.dumps(future_gap).encode() + b"\n")
    consumed: list[str] = []

    batch = scan_trajectory_batch(path, lambda event, *_args: consumed.append(event.event_type))

    assert consumed == [EventType.COVERAGE_STARTED]
    assert batch.event_count == 2
    assert batch.unsupported_event_sequences == [3]
    assert [(item.first_sequence, item.last_sequence) for item in batch.prefix_violations] == [(2, 2)]
