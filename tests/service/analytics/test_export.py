# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Counter-examples for the transient export serializers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import fields, replace
from itertools import pairwise

from chrys.foundation.trajectory.envelope import SegmentedField
from chrys.service.analytics import (
    FLOW_TERMINAL_INDEX,
    ChangeVerification,
    ChangeVerificationRow,
    ChangeVerificationState,
    Metric,
    Precision,
    TrajectoryAnalyzer,
    TrajectoryDiagnostics,
)
from chrys.service.analytics.export import analysis_json, findings_csv, perfetto_trace, turns_csv
from tests.service.analytics._events import RUNTIME_ID, EventLog, caused_by

_NS = 1_000_000_000


def _rich_log() -> EventLog:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "1" * 32,
        0,
        0,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "2" * 32},
    )
    log.span("model.exchange", "2" * 32, 0, 0, parent_operation_id="1" * 32)
    log.span(
        "preparation",
        "e" * 32,
        0,
        0,
        parent_operation_id="2" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "c" * 32},
    )
    log.span(
        "tool.operation",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="2" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "batch_index": 0,
            "parent_model_operation_id": "2" * 32,
            "call_item_id": "7" * 32,
        },
        finish_payload={"result_item_id": "9" * 32},
        links=caused_by("e" * 32),
    )
    revision = log.add(
        "context.revision.recorded",
        10 * _NS,
        operation_id="5" * 32,
        parent_operation_id="4" * 32,
        payload={"revision_id": "5" * 32, "is_checkpoint": True, "item_count": 2, "unidentified_item_count": 0},
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id="6" * 32, segment_count=1),),
    )
    log.add(
        "event.segment",
        10 * _NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": "6" * 32,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": item_id, "occurrence": 0, "position": index, "action": "add"}
                for index, item_id in enumerate(("7" * 32, "9" * 32))
            ],
        },
    )
    log.span(
        "model.cycle",
        "3" * 32,
        10 * _NS,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "4" * 32},
    )
    log.span(
        "model.exchange",
        "4" * 32,
        10 * _NS,
        10 * _NS,
        parent_operation_id="3" * 32,
        start_payload={"context_revision_id": "5" * 32},
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    return log


def _analyzed(tmp_path, log: EventLog):
    path = tmp_path / "events.jsonl"
    log.write(path)
    analyzer = TrajectoryAnalyzer()
    analysis = analyzer.load(path)
    return analysis, analyzer.counter_samples()


def test_perfetto_slices_are_lossless_and_never_overlap_within_a_lane(tmp_path) -> None:
    analysis, samples = _analyzed(tmp_path, _rich_log())
    turn = analysis.turns[0]

    trace = perfetto_trace(analysis, samples)

    events = trace["traceEvents"]
    slices = [event for event in events if event["ph"] == "X"]
    timeable = [operation for operation in turn.operations if operation.start_ns is not None]
    assert len(slices) == len(timeable) + 1  # + the synthetic response terminal
    by_tid = defaultdict(list)
    for item in slices:
        by_tid[item["tid"]].append((item["ts"], item["ts"] + item["dur"]))
    for intervals in by_tid.values():
        intervals.sort()
        for (_, first_end), (second_start, _) in pairwise(intervals):
            assert second_start >= first_end


def test_perfetto_flow_arrows_come_only_from_causal_edges(tmp_path) -> None:
    analysis, samples = _analyzed(tmp_path, _rich_log())
    flow = analysis.turns[0].flow
    assert flow is not None

    trace = perfetto_trace(analysis, samples)

    events = trace["traceEvents"]
    starts = [event for event in events if event["ph"] == "s"]
    finishes = [event for event in events if event["ph"] == "f"]
    assert len(starts) == len(flow.causal_edges())
    assert sorted(event["id"] for event in starts) == sorted(event["id"] for event in finishes)
    assert all(event["bp"] == "e" for event in finishes)
    turn = analysis.turns[0]
    origin = turn.axis_start_ns
    for start, finish, (source, target) in zip(starts, finishes, flow.causal_edges(), strict=True):
        source_operation = turn.operations[source]
        assert source_operation.end_ns is not None
        assert start["ts"] == (source_operation.end_ns - origin) / 1000.0
        target_ns = turn.axis_end_ns if target == FLOW_TERMINAL_INDEX else turn.operations[target].start_ns
        assert target_ns is not None
        assert finish["ts"] == (target_ns - origin) / 1000.0


def test_perfetto_never_fabricates_flow_from_adjacency(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "tool.operation",
        "a" * 32,
        0,
        _NS,
        start_payload={"tool_name": "first", "tool_kind": "filesystem.read"},
    )
    log.span(
        "tool.operation",
        "b" * 32,
        _NS,
        2 * _NS,
        start_payload={"tool_name": "second", "tool_kind": "filesystem.read"},
    )
    log.add("turn.finished", 2 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    analysis, samples = _analyzed(tmp_path, log)

    trace = perfetto_trace(analysis, samples)

    assert [event for event in trace["traceEvents"] if event["ph"] in {"s", "f"}] == []


def test_perfetto_emits_token_context_and_concurrency_counters(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("model.exchange.started", 0, operation_id="a" * 32)
    log.add(
        "model.exchange.finished",
        _NS,
        operation_id="a" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 1,
            "usage": {"normalized": {"input_total": 100, "output_total": 20}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add(
        "context.revision.recorded",
        _NS,
        operation_id="5" * 32,
        payload={"revision_id": "5" * 32, "is_checkpoint": True, "item_count": 3, "unidentified_item_count": 0},
    )
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    analysis, samples = _analyzed(tmp_path, log)

    trace = perfetto_trace(analysis, samples)

    counters = [event for event in trace["traceEvents"] if event["ph"] == "C"]
    tokens = [event for event in counters if event["name"] == "tokens"]
    assert [event["args"] for event in tokens] == [{"input": 100, "output": 20}]
    context = [event for event in counters if event["name"] == "context items"]
    assert [event["args"] for event in context] == [{"items": 3}]
    assert [event for event in counters if event["name"] == "active work"]


def test_perfetto_gives_each_runtime_its_own_process_and_origin(tmp_path) -> None:
    resumed_runtime = "9" * 32
    resumed_turn = "b" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 50 * _NS, payload={"turn_number": 1})
    log.span(
        "tool.operation",
        "a" * 32,
        50 * _NS,
        51 * _NS,
        start_payload={"tool_name": "read", "tool_kind": "filesystem.read"},
    )
    log.add("turn.finished", 51 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    # The resumed process restarts the monotonic clock, so its raw timestamps
    # precede the first run's despite happening later.
    log.add(
        "trajectory.coverage.started",
        0,
        turn_id=None,
        payload={"coverage_reason": "runtime_resumed"},
        runtime_id=resumed_runtime,
    )
    log.add("turn.started", 2 * _NS, turn_id=resumed_turn, runtime_id=resumed_runtime, payload={"turn_number": 2})
    log.add("model.exchange.started", 2 * _NS, turn_id=resumed_turn, operation_id="c" * 32, runtime_id=resumed_runtime)
    log.add(
        "model.exchange.finished",
        3 * _NS,
        turn_id=resumed_turn,
        operation_id="c" * 32,
        runtime_id=resumed_runtime,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {"normalized": {"input_total": 7, "output_total": 3}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add(
        "turn.finished",
        3 * _NS,
        turn_id=resumed_turn,
        runtime_id=resumed_runtime,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    analysis, samples = _analyzed(tmp_path, log)

    trace = perfetto_trace(analysis, samples)

    events = trace["traceEvents"]
    processes = [event for event in events if event["ph"] == "M" and event["name"] == "process_name"]
    assert [event["pid"] for event in processes] == [1, 2]
    assert processes[0]["args"]["name"].endswith(f" · run 1 ({RUNTIME_ID[:8]})")
    assert processes[1]["args"]["name"].endswith(f" · run 2 ({resumed_runtime[:8]})")
    threads = {
        (event["pid"], event["tid"]): event["args"]["name"]
        for event in events
        if event["ph"] == "M" and event["name"] == "thread_name"
    }
    assert threads[(1, 1)] == "turn 1"
    assert threads[(2, 1)] == "turn 2"
    slice_starts = defaultdict(list)
    for event in events:
        if event["ph"] == "X":
            slice_starts[event["pid"]].append(event["ts"])
    assert min(slice_starts[1]) == 0.0
    assert min(slice_starts[2]) == 0.0
    tokens = [event for event in events if event["ph"] == "C" and event["name"] == "tokens"]
    assert [(event["pid"], event["ts"], event["args"]) for event in tokens] == [
        (2, _NS / 1000.0, {"input": 7, "output": 3})
    ]


def test_folded_retry_keeps_physical_runtimes_and_maps_json_counters_to_logical_axis(tmp_path) -> None:
    resumed_runtime = "9" * 32
    resumed_turn = "b" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 50 * _NS, payload={"turn_number": 1, "is_retry": False})
    log.add("model.exchange.started", 50 * _NS, operation_id="a" * 32)
    log.add(
        "model.exchange.finished",
        51 * _NS,
        operation_id="a" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {"normalized": {"input_total": 5, "output_total": 5}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add(
        "context.revision.recorded",
        51 * _NS,
        operation_id="c" * 32,
        payload={"revision_id": "c" * 32, "is_checkpoint": True, "item_count": 1, "unidentified_item_count": 0},
    )
    log.add("turn.finished", 52 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.add(
        "trajectory.coverage.started",
        0,
        turn_id=None,
        payload={"coverage_reason": "runtime_resumed"},
        runtime_id=resumed_runtime,
    )
    log.add(
        "turn.started",
        2 * _NS,
        turn_id=resumed_turn,
        runtime_id=resumed_runtime,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        "model.exchange.started",
        2 * _NS,
        turn_id=resumed_turn,
        operation_id="d" * 32,
        runtime_id=resumed_runtime,
    )
    log.add(
        "model.exchange.finished",
        3 * _NS,
        turn_id=resumed_turn,
        operation_id="d" * 32,
        runtime_id=resumed_runtime,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {"normalized": {"input_total": 7, "output_total": 3}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add(
        "context.revision.recorded",
        3 * _NS,
        turn_id=resumed_turn,
        operation_id="e" * 32,
        runtime_id=resumed_runtime,
        payload={"revision_id": "e" * 32, "is_checkpoint": True, "item_count": 2, "unidentified_item_count": 0},
    )
    log.add(
        "turn.finished",
        4 * _NS,
        turn_id=resumed_turn,
        runtime_id=resumed_runtime,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    analysis, samples = _analyzed(tmp_path, log)

    assert len(analysis.turns) == 1
    turn = analysis.turns[0]
    assert [attempt.runtime_id for attempt in turn.attempts] == [RUNTIME_ID, resumed_runtime]
    assert "logical turn spans multiple trajectory runtimes" in turn.diagnostics

    exported_turn = analysis_json(analysis, samples)["turns"][0]
    assert [attempt["runtime_id"] for attempt in exported_turn["attempts"]] == [RUNTIME_ID, resumed_runtime]
    assert [attempt["is_retry"] for attempt in exported_turn["attempts"]] == [False, True]
    assert [operation["end_ns"] for operation in exported_turn["operations"]] == [51 * _NS, 53 * _NS]
    assert [sample["end_ns"] for sample in exported_turn["usage_samples"]] == [51 * _NS, 53 * _NS]
    assert [sample["ns"] for sample in exported_turn["context_samples"]] == [51 * _NS, 53 * _NS]

    events = perfetto_trace(analysis, samples)["traceEvents"]
    processes = [event for event in events if event["ph"] == "M" and event["name"] == "process_name"]
    assert [event["pid"] for event in processes] == [1, 2]
    assert processes[0]["args"]["name"].endswith(f" · run 1 ({RUNTIME_ID[:8]})")
    assert processes[1]["args"]["name"].endswith(f" · run 2 ({resumed_runtime[:8]})")
    slices = [event for event in events if event["ph"] == "X" and event["cat"] == "model.exchange"]
    assert sorted((event["pid"], event["ts"], event["dur"]) for event in slices) == [
        (1, 0.0, _NS / 1000.0),
        (2, 0.0, _NS / 1000.0),
    ]
    tokens = [event for event in events if event["ph"] == "C" and event["name"] == "tokens"]
    assert [(event["pid"], event["ts"], event["args"]) for event in tokens] == [
        (1, _NS / 1000.0, {"input": 5, "output": 5}),
        (2, _NS / 1000.0, {"input": 7, "output": 3}),
    ]
    active = [event for event in events if event["ph"] == "C" and event["name"] == "active work"]
    assert {event["pid"] for event in active} == {1, 2}
    assert max(event["args"]["count"] for event in active) == 1


def test_folded_attempts_in_one_runtime_share_a_single_turn_track(tmp_path) -> None:
    """Two attempts of one turn on one clock lane-pack together instead of naming two tracks alike."""
    retry_turn = "5" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1, "is_retry": False})
    log.span("model.exchange", "a" * 32, 0, 3 * _NS)
    log.add("turn.finished", 4 * _NS, payload={"end_reason": "cancelled", "duration_ms": 4_000})
    log.add("turn.started", 10 * _NS, turn_id=retry_turn, payload={"turn_number": 1, "is_retry": True})
    log.span("model.exchange", "b" * 32, 10 * _NS, 13 * _NS, turn_id=retry_turn)
    log.add(
        "turn.finished",
        14 * _NS,
        turn_id=retry_turn,
        payload={"end_reason": "completed", "duration_ms": 4_000},
    )
    analysis, samples = _analyzed(tmp_path, log)

    turn = analysis.turns[0]
    assert len(turn.attempts) == 2
    assert len({attempt.runtime_id for attempt in turn.attempts}) == 1

    events = perfetto_trace(analysis, samples)["traceEvents"]
    threads = [
        (event["pid"], event["tid"], event["args"]["name"])
        for event in events
        if event["ph"] == "M" and event["name"] == "thread_name"
    ]
    assert threads == [(1, 1, "turn 1")]
    slices = sorted(
        (event["ts"], event["dur"]) for event in events if event["ph"] == "X" and event["cat"] == "model.exchange"
    )
    assert slices == [(0.0, 3 * _NS / 1000.0), (10 * _NS / 1000.0, 3 * _NS / 1000.0)]


def test_folded_export_rejects_samples_outside_their_owning_attempt_axis(tmp_path) -> None:
    retry_turn = "b" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1, "is_retry": False})
    log.add("model.exchange.started", 0, operation_id="a" * 32)
    log.add(
        "model.exchange.finished",
        10 * _NS,
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
        "context.revision.recorded",
        11 * _NS,
        operation_id="c" * 32,
        payload={"revision_id": "c" * 32, "is_checkpoint": True, "item_count": 1, "unidentified_item_count": 0},
    )
    log.add("turn.finished", 5 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.add(
        "turn.started",
        20 * _NS,
        turn_id=retry_turn,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        "turn.finished",
        21 * _NS,
        turn_id=retry_turn,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    analysis, samples = _analyzed(tmp_path, log)

    turn = analysis.turns[0]
    assert "usage sample lies outside its owning attempt axis" in turn.diagnostics
    assert "context sample lies outside its owning attempt axis" in turn.diagnostics
    exported_turn = analysis_json(analysis, samples)["turns"][0]
    assert exported_turn["usage_samples"][0]["end_ns"] is None
    assert exported_turn["context_samples"][0]["ns"] is None
    events = perfetto_trace(analysis, samples)["traceEvents"]
    assert [event for event in events if event["ph"] == "C" and event["name"] == "tokens"] == []
    assert [event for event in events if event["ph"] == "C" and event["name"] == "context items"] == []


def test_single_attempt_export_diagnoses_counter_samples_outside_its_axis(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1, "is_retry": False})
    log.add("model.exchange.started", 0, operation_id="a" * 32)
    log.add(
        "model.exchange.finished",
        10 * _NS,
        operation_id="a" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 10_000,
            "usage": {"normalized": {"input_total": 11, "output_total": 3}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add(
        "context.revision.recorded",
        11 * _NS,
        operation_id="c" * 32,
        payload={"revision_id": "c" * 32, "is_checkpoint": True, "item_count": 1, "unidentified_item_count": 0},
    )
    log.add("turn.finished", 5 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    analysis, samples = _analyzed(tmp_path, log)

    turn = analysis.turns[0]
    assert len(turn.attempts) == 1
    assert "usage sample lies outside its owning attempt axis" in turn.diagnostics
    assert "context sample lies outside its owning attempt axis" in turn.diagnostics
    exported_turn = analysis_json(analysis, samples)["turns"][0]
    assert exported_turn["usage_samples"][0]["end_ns"] is None
    assert exported_turn["context_samples"][0]["ns"] is None
    events = perfetto_trace(analysis, samples)["traceEvents"]
    assert [event for event in events if event["ph"] == "C" and event["name"] == "tokens"] == []
    assert [event for event in events if event["ph"] == "C" and event["name"] == "context items"] == []


def test_empty_turn_export_diagnoses_counter_sample_outside_its_zero_width_axis(tmp_path) -> None:
    turn_id = "b" * 32
    log = EventLog()
    log.coverage()
    for monotonic_ns in (0, _NS):
        log.add(
            "turn.started",
            monotonic_ns,
            turn_id=turn_id,
            payload={"turn_number": 1, "is_retry": False},
        )
    log.add("model.exchange.started", 2 * _NS, turn_id=turn_id, operation_id="a" * 32)
    log.add(
        "model.exchange.finished",
        5 * _NS,
        turn_id=turn_id,
        operation_id="a" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 3_000,
            "usage": {"normalized": {"input_total": 11, "output_total": 3}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add(
        "turn.finished",
        6 * _NS,
        turn_id=turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    analysis, samples = _analyzed(tmp_path, log)

    turn = analysis.turns[0]
    assert turn.attempts[0].physical_axis_start_ns == turn.attempts[0].physical_axis_end_ns == 0
    assert "usage sample lies outside its owning attempt axis" in turn.diagnostics
    exported_turn = analysis_json(analysis, samples)["turns"][0]
    assert len(exported_turn["usage_samples"]) == 1
    assert exported_turn["usage_samples"][0]["end_ns"] is None
    assert exported_turn["usage_samples"][0]["input_tokens"] == 11
    events = perfetto_trace(analysis, samples)["traceEvents"]
    assert [event for event in events if event["ph"] == "C" and event["name"] == "tokens"] == []


def test_perfetto_never_sweeps_work_concurrency_across_runtimes(tmp_path) -> None:
    resumed_runtime = "9" * 32
    resumed_turn = "b" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", _NS, payload={"turn_number": 1})
    log.span(
        "tool.operation",
        "a" * 32,
        _NS,
        4 * _NS,
        start_payload={"tool_name": "read", "tool_kind": "filesystem.read"},
    )
    log.add("turn.finished", 4 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    # Raw intervals overlap across the runtimes; only a per-runtime sweep
    # keeps that clock artifact out of the concurrency counter.
    log.add(
        "trajectory.coverage.started",
        0,
        turn_id=None,
        payload={"coverage_reason": "runtime_resumed"},
        runtime_id=resumed_runtime,
    )
    log.add("turn.started", 2 * _NS, turn_id=resumed_turn, runtime_id=resumed_runtime, payload={"turn_number": 2})
    log.span(
        "tool.operation",
        "d" * 32,
        2 * _NS,
        5 * _NS,
        turn_id=resumed_turn,
        runtime_id=resumed_runtime,
        start_payload={"tool_name": "read", "tool_kind": "filesystem.read"},
    )
    log.add(
        "turn.finished",
        5 * _NS,
        turn_id=resumed_turn,
        runtime_id=resumed_runtime,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    analysis, samples = _analyzed(tmp_path, log)

    trace = perfetto_trace(analysis, samples)

    active = [event for event in trace["traceEvents"] if event["ph"] == "C" and event["name"] == "active work"]
    assert {event["pid"] for event in active} == {1, 2}
    assert max(event["args"]["count"] for event in active) == 1


def test_perfetto_single_runtime_stays_one_unsuffixed_process(tmp_path) -> None:
    analysis, samples = _analyzed(tmp_path, _rich_log())

    trace = perfetto_trace(analysis, samples)

    events = trace["traceEvents"]
    processes = [event for event in events if event["ph"] == "M" and event["name"] == "process_name"]
    assert len(processes) == 1
    assert " · run" not in processes[0]["args"]["name"]
    assert {event["pid"] for event in events} == {1}


def test_json_export_redacts_the_session_path_by_default(tmp_path) -> None:
    analysis, samples = _analyzed(tmp_path, _rich_log())

    redacted = analysis_json(analysis, samples)
    sensitive = analysis_json(analysis, samples, include_sensitive=True)

    assert redacted["schema"] == "chrys.trajectory.export/1"
    assert str(redacted["session"]["path"]).startswith("redacted:")
    assert sensitive["session"]["path"] == str(analysis.path)
    turn = redacted["turns"][0]
    assert turn["flow"]["causal_edges"]
    assert turn["usage_samples"] == []
    assert [sample["item_count"] for sample in turn["context_samples"]] == [2]


def test_json_export_includes_complete_session_diagnostics(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'not-json\n{"incomplete"')
    analyzer = TrajectoryAnalyzer()
    analysis = analyzer.load(path)

    exported = analysis_json(analysis, analyzer.counter_samples())

    diagnostics = exported["diagnostics"]
    assert set(diagnostics) == {field.name for field in fields(TrajectoryDiagnostics)}
    assert diagnostics["corrupt_line_count"] == 1
    (corrupt_line,) = diagnostics["corrupt_lines"]
    assert set(corrupt_line) == {"line_number", "byte_offset", "byte_length", "reason", "after_sequence"}
    assert corrupt_line["line_number"] == 1
    assert corrupt_line["byte_offset"] == 0
    assert corrupt_line["byte_length"] == 9
    assert corrupt_line["after_sequence"] == 0
    assert "not valid UTF-8 JSON" in corrupt_line["reason"]
    assert diagnostics["torn_tail_bytes"] == len(b'{"incomplete"')
    assert exported["overview"]["elapsed_ns"] == {
        "value": 0,
        "precision": Precision.UNRESOLVED,
        "reason": "session trajectory integrity is unresolved: corrupt lines, torn tail",
    }


def test_json_export_never_claims_exact_zero_for_an_unresolved_rollback_projection(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add(
        "branch.superseded",
        1,
        turn_id=None,
        payload={"branch_id": "3" * 32, "superseded_by": "6" * 32},
    )
    analysis, samples = _analyzed(tmp_path, log)

    exported = analysis_json(analysis, samples)

    reason = "session trajectory integrity is unresolved: unresolved rollback projection"
    unresolved_zero = {"value": 0, "precision": Precision.UNRESOLVED, "reason": reason}
    assert exported["diagnostics"]["rollback_projection_unresolved"] is True
    assert exported["overview"]["elapsed_ns"] == unresolved_zero
    assert exported["overview"]["usage_tokens"] == unresolved_zero
    assert exported["validation"]["tool_count"] == unresolved_zero
    assert exported["token_usage"]["input"] == unresolved_zero


def test_json_export_carries_validation_and_change_verification_with_redacted_row_paths(tmp_path) -> None:
    analysis, samples = _analyzed(tmp_path, _rich_log())
    exact_zero = Metric(0, Precision.EXACT)
    enriched = replace(
        analysis,
        change_verification=ChangeVerification(
            detail_available=True,
            detection_truncated=False,
            files_touched=Metric(1, Precision.EXACT),
            created=exact_zero,
            modified=Metric(1, Precision.EXACT),
            deleted=exact_zero,
            net_zero=exact_zero,
            rows=(
                ChangeVerificationRow(
                    path="secret/changed.py",
                    state=ChangeVerificationState.VERIFIED,
                    last_change_turn=1,
                    precision=Precision.ESTIMATED,
                    evidence=("classification:1",),
                ),
            ),
        ),
    )

    redacted = analysis_json(enriched, samples)
    sensitive = analysis_json(enriched, samples, include_sensitive=True)

    validation = redacted["validation"]
    assert validation is not None
    assert set(validation["funnel"]) == {"search", "read", "edit", "verify"}
    assert validation["tool_count"] == {"value": 1, "precision": Precision.EXACT, "reason": None}
    change = redacted["change_verification"]
    assert change is not None
    assert change["files_touched"]["value"] == 1
    (row,) = change["rows"]
    assert str(row["path"]).startswith("redacted:")
    assert row["state"] == ChangeVerificationState.VERIFIED
    assert row["precision"] == Precision.ESTIMATED
    assert row["evidence"] == ["classification:1"]
    (sensitive_row,) = sensitive["change_verification"]["rows"]
    assert sensitive_row["path"] == "secret/changed.py"


def test_csv_exports_render_one_row_per_turn_and_finding(tmp_path) -> None:
    analysis, _ = _analyzed(tmp_path, _rich_log())

    turns = turns_csv(analysis).splitlines()
    findings = findings_csv(analysis).splitlines()

    assert turns[0].startswith("turn_number,turn_id,start_sequence,elapsed_ns")
    assert len(turns) == 1 + len(analysis.turns)
    # Every exported metric family carries its precision column, so a
    # spreadsheet consumer can tell exact numbers from partial estimates.
    header = turns[0].split(",")
    for name in ("exclusive_work_precision", "parallelism_precision", "overlap_gain_precision"):
        assert name in header
    assert sum(name.startswith("wall_") and name.endswith("_precision") for name in header) == sum(
        name.startswith("wall_") and name.endswith("_ns") for name in header
    )
    assert sum(name.startswith("tokens_") and name.endswith("_precision") for name in header) == sum(
        name.startswith("tokens_") and not name.endswith("_precision") for name in header
    )
    assert all(len(row.split(",")) == len(header) for row in turns[1:])
    assert findings[0].startswith("rule_id,severity,turn_number")
    assert len(findings) == 1 + len(analysis.findings)
