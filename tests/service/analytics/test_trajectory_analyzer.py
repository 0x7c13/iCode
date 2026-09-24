# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""TrajectoryAnalyzer load, refresh, cache invalidation, session span, fingerprint-key discovery, and counter samples."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.foundation.trajectory.event_types import EventType
from chrys.service.analytics import (
    Precision,
    SessionSpan,
    TrajectoryAnalyzer,
)
from chrys.service.analytics import _turns as turns_module
from chrys.service.analytics import aggregation as aggregation_module
from tests.service.analytics._events import NS, EventLog


def test_session_span_keeps_first_turn_start_last_turn_finish_and_runtime_count(tmp_path) -> None:
    """Wall-clock anchors stay the producer's strings; runtimes count once per start."""
    later_runtime = "7" * 32
    log = EventLog()
    log.coverage()
    log.add(EventType.RUNTIME_STARTED, 0, turn_id=None)
    log.add("turn.started", 0, payload={"turn_number": 1}, occurred_at="2026-08-01T10:00:00.000000Z")
    log.add(
        "turn.finished",
        NS,
        payload={"end_reason": "cancelled", "duration_ms": 0},
        occurred_at="2026-08-01T10:00:01.000000Z",
    )
    log.add(EventType.RUNTIME_STARTED, 2 * NS, turn_id=None, runtime_id=later_runtime)
    log.add(
        "turn.started",
        2 * NS,
        turn_id="9" * 32,
        payload={"turn_number": 2},
        runtime_id=later_runtime,
        occurred_at="2026-08-02T10:00:00.000000Z",
    )
    log.add(
        "turn.finished",
        3 * NS,
        turn_id="9" * 32,
        payload={"end_reason": "cancelled", "duration_ms": 0},
        runtime_id=later_runtime,
        occurred_at="2026-08-02T10:00:05.000000Z",
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    first = analyzer.load(path).session_span

    assert first == SessionSpan(
        first_turn_started_at="2026-08-01T10:00:00.000000Z",
        last_turn_finished_at="2026-08-02T10:00:05.000000Z",
        runtime_count=2,
    )

    log.add("turn.started", 4 * NS, turn_id="8" * 32, payload={"turn_number": 3}, occurred_at="2026-08-03T00:00:00Z")
    log.add(
        "turn.finished",
        5 * NS,
        turn_id="8" * 32,
        payload={"end_reason": "cancelled", "duration_ms": 0},
        occurred_at="2026-08-03T00:00:09Z",
    )
    log.write(path)

    refreshed = analyzer.refresh().session_span

    assert refreshed == SessionSpan(
        first_turn_started_at="2026-08-01T10:00:00.000000Z",
        last_turn_finished_at="2026-08-03T00:00:09Z",
        runtime_count=2,
    )


def test_refresh_surfaces_in_flight_work_added_to_an_already_cached_turn(tmp_path) -> None:
    """Adding a projected node re-resolves the turn even without a terminal event.

    One appended event per refresh: a single dirty flag re-resolves the whole
    turn, so batching the events would let any one working path mask the rest.
    """
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    assert analyzer.load(path).turns[0].operations == ()

    def visible() -> set[tuple[str, str]]:
        return {(operation.family, operation.operation_id) for operation in analyzer.refresh().turns[0].operations}

    log.add("model.exchange.started", 1 * NS, operation_id="a" * 32)
    log.write(path)
    assert ("model.exchange", "a" * 32) in visible()

    log.add("approval.requested", 2 * NS, operation_id="b" * 32, payload={"approval_request_id": "b" * 32})
    log.write(path)
    assert ("approval", "b" * 32) in visible()

    log.add(
        "retry.scheduled",
        3 * NS,
        operation_id=None,
        payload={
            "retry_mode": "run",
            "previous_operation_id": "a" * 32,
            "next_operation_id": "c" * 32,
            "delay_ms": 1000,
        },
    )
    log.write(path)
    assert ("retry", "c" * 32) in visible()

    log.add(
        "compaction.phase.finished",
        4 * NS,
        operation_id="d" * 32,
        payload={
            "compaction_run_id": "e" * 32,
            "phase": "summaries",
            "groups_compacted": 1,
            "duration_ms": 5,
            "tokens_before": 100,
            "tokens_after": 50,
            "last_words_generated": False,
        },
    )
    log.write(path)
    refreshed = visible()
    assert any(family == "compaction.phase" for family, _ in refreshed)

    fresh = {
        (operation.family, operation.operation_id) for operation in TrajectoryAnalyzer().load(path).turns[0].operations
    }
    assert refreshed == fresh


def test_refresh_revalidates_counter_axis_only_for_dirty_folded_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_turn_id = "4" * 32
    retry_turn_id = "5" * 32
    log = EventLog()
    log.coverage()
    log.add(
        EventType.TURN_STARTED,
        0,
        turn_id=first_turn_id,
        payload={"turn_number": 1, "is_retry": False},
    )
    log.add("model.exchange.started", 0, turn_id=first_turn_id, operation_id="a" * 32)
    log.add(
        "model.exchange.finished",
        10 * NS,
        turn_id=first_turn_id,
        operation_id="a" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 10_000,
            "usage": {"normalized": {"input_total": 5, "output_total": 5}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add(
        EventType.TURN_FINISHED,
        NS,
        turn_id=first_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    log.add(
        EventType.TURN_STARTED,
        20 * NS,
        turn_id=retry_turn_id,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        EventType.TURN_FINISHED,
        21 * NS,
        turn_id=retry_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    loaded = analyzer.load(path).turns[0]
    assert len(loaded.attempts) == 2
    assert "usage sample lies outside its owning attempt axis" in loaded.diagnostics
    calls: list[tuple[str, bool]] = []
    real_counter_axis_diagnostics = turns_module._counter_axis_diagnostics

    def tracked_counter_axis_diagnostics(*args, **kwargs):
        calls.append((args[1], kwargs["refresh"]))
        return real_counter_axis_diagnostics(*args, **kwargs)

    def unexpected_full_projection(*_args, **_kwargs):
        raise AssertionError("live refresh rebuilt the full counter sample projection")

    monkeypatch.setattr(turns_module, "_counter_axis_diagnostics", tracked_counter_axis_diagnostics)
    monkeypatch.setattr(turns_module, "_usage_samples_by_turn", unexpected_full_projection)
    monkeypatch.setattr(turns_module, "_context_samples_by_turn", unexpected_full_projection)
    revision_id = "6" * 32
    log.add(
        EventType.CONTEXT_REVISION_RECORDED,
        22 * NS,
        turn_id=retry_turn_id,
        operation_id=revision_id,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 1},
    )
    log.write(path)

    refreshed = analyzer.refresh()

    assert sorted(calls) == [(first_turn_id, False), (retry_turn_id, True)]
    assert "usage sample lies outside its owning attempt axis" in refreshed.turns[0].diagnostics
    assert "context sample lies outside its owning attempt axis" in refreshed.turns[0].diagnostics
    with pytest.raises(AssertionError, match="live refresh rebuilt the full counter sample projection"):
        analyzer.counter_samples()


def test_refresh_duplicate_revision_invalidates_every_affected_turn(tmp_path: Path) -> None:
    first_turn_id = "4" * 32
    second_turn_id = "5" * 32
    revision_id = "6" * 32
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 0, turn_id=first_turn_id, payload={"turn_number": 1})
    log.add(
        EventType.CONTEXT_REVISION_RECORDED,
        10 * NS,
        turn_id=first_turn_id,
        operation_id=revision_id,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 1},
    )
    log.add(
        EventType.TURN_FINISHED,
        5 * NS,
        turn_id=first_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 5_000},
    )
    log.add(EventType.TURN_STARTED, 20 * NS, turn_id=second_turn_id, payload={"turn_number": 2})
    log.add(
        EventType.TURN_FINISHED,
        21 * NS,
        turn_id=second_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    initial = analyzer.load(path).turn(first_turn_id)
    assert initial is not None
    assert "context sample lies outside its owning attempt axis" in initial.diagnostics

    log.add(
        EventType.CONTEXT_REVISION_RECORDED,
        22 * NS,
        turn_id=second_turn_id,
        operation_id="7" * 32,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 1},
    )
    log.write(path)

    refreshed = analyzer.refresh().turn(first_turn_id)
    cold = TrajectoryAnalyzer().load(path).turn(first_turn_id)

    assert refreshed is not None
    assert cold is not None
    assert refreshed == cold
    assert "context sample lies outside its owning attempt axis" not in refreshed.diagnostics


def test_installed_fingerprint_key_falls_back_to_the_recorder_config_directory(tmp_path, monkeypatch) -> None:
    """A custom session root stores sessions away from the config directory
    that holds the recorder's key; the reader must still find it there."""
    import chrys.foundation.platform as platform_module
    from chrys.foundation.trajectory.keys import load_or_create_fingerprint_key

    events = tmp_path / "custom-root" / "sessions" / "abc123" / "trajectory" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.touch()
    config_dir = tmp_path / "config"
    recorder_key = load_or_create_fingerprint_key(config_dir)

    real_platform = platform_module.get_platform()

    class _RecorderPlatform:
        def __getattr__(self, name: str) -> object:
            return getattr(real_platform, name)

    installed = _RecorderPlatform()
    installed.__dict__["config_dir"] = config_dir
    monkeypatch.setattr(platform_module, "get_platform", lambda: installed)

    assert aggregation_module._read_installed_fingerprint_key(events) == recorder_key

    # A key beside the session root (the default layout, or a tree copied
    # along with its key) still outranks the local installation's key.
    beside_key = load_or_create_fingerprint_key(tmp_path / "custom-root")
    assert aggregation_module._read_installed_fingerprint_key(events) == beside_key


def test_refresh_surfaces_a_duplicate_turn_start_on_an_already_cached_turn(tmp_path) -> None:
    """A duplicate lifecycle start must evict the cached analysis of its turn."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("turn.finished", NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    assert analyzer.load(path).turns[0].turn_number == 1

    log.add("turn.started", 2 * NS, payload={"turn_number": 1})
    log.write(path)

    refreshed = analyzer.refresh().turns[0]
    fresh = TrajectoryAnalyzer().load(path).turns[0]

    assert refreshed.turn_number == 1
    assert refreshed.elapsed_ns.precision is Precision.UNRESOLVED
    assert refreshed.elapsed_ns == fresh.elapsed_ns


def test_counter_samples_come_from_the_analyzer_on_demand(tmp_path) -> None:
    """Timestamped counter samples join exchange finishes without per-turn retention."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.exchange_usage(
        "a" * 32,
        0,
        NS,
        {"input_total": 100, "output_total": 20, "reasoning": 7, "cache_read": 60},
    )
    log.add(
        "context.revision.recorded",
        NS,
        operation_id="5" * 32,
        payload={"revision_id": "5" * 32, "is_checkpoint": True, "item_count": 3, "unidentified_item_count": 0},
    )
    log.add("turn.finished", NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    analysis = analyzer.load(path)
    turn_id = analysis.turns[0].turn_id
    samples = analyzer.counter_samples()

    usage = samples.usage_by_turn[turn_id]
    assert len(usage) == 1
    assert usage[0].end_ns == NS
    assert (usage[0].input_tokens, usage[0].output_tokens) == (100, 20)
    assert (usage[0].reasoning_tokens, usage[0].cache_read_tokens, usage[0].cache_creation_tokens) == (7, 60, None)
    context = samples.context_by_turn[turn_id]
    assert [(sample.ns, sample.item_count) for sample in context] == [(NS, 3)]


def test_counter_samples_omit_negative_context_item_counts(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "context.revision.recorded",
        NS,
        operation_id="5" * 32,
        payload={"revision_id": "5" * 32, "is_checkpoint": True, "item_count": -1, "unidentified_item_count": 0},
    )
    log.add("turn.finished", NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    analysis = analyzer.load(path)

    assert analyzer.counter_samples().context_by_turn.get(analysis.turns[0].turn_id, ()) == ()
