# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Counter-examples for the frozen P0 timing and KPI formulas."""

from __future__ import annotations

import pytest

from chrys.foundation.trajectory.envelope import Actor, SegmentedField
from chrys.service.analytics import (
    FLOW_TERMINAL_INDEX,
    Metric,
    Precision,
    TimelineDiagnosticCode,
    TimelineOperationDetail,
    TrajectoryAnalyzer,
    WallBucket,
    analyze_trajectory,
)
from chrys.service.analytics._metric_ops import _sum_metrics
from tests.service.analytics._events import NS, EventLog, caused_by, operation_index


def _two_parallel_tools_log() -> EventLog:
    """The two-parallel-tools turn shared by the CP/parallelism and flow-edge pins."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "1" * 32,
        0,
        0,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "2" * 32},
    )
    log.span("model.exchange", "2" * 32, 0, 0, parent_operation_id="1" * 32)
    item_ids = (("7" * 32, "9" * 32), ("8" * 32, "0" * 32))
    for prefix, (call_item_id, result_item_id) in zip(("c", "d"), item_ids, strict=True):
        tool_id = prefix * 32
        preamble_id = ("e" if prefix == "c" else "f") * 32
        log.span(
            "preparation",
            preamble_id,
            0,
            0,
            parent_operation_id="2" * 32,
            start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": tool_id},
        )
        log.span(
            "tool.operation",
            tool_id,
            0,
            10 * NS,
            parent_operation_id="2" * 32,
            start_payload={
                "tool_name": prefix,
                "tool_kind": "filesystem.read",
                "batch_index": 0,
                "parent_model_operation_id": "2" * 32,
                "call_item_id": call_item_id,
            },
            finish_payload={"result_item_id": result_item_id},
            links=caused_by(preamble_id),
        )
    revision = log.add(
        "context.revision.recorded",
        10 * NS,
        operation_id="5" * 32,
        parent_operation_id="4" * 32,
        payload={"revision_id": "5" * 32, "is_checkpoint": True, "item_count": 4, "unidentified_item_count": 0},
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id="6" * 32, segment_count=1),),
    )
    log.add(
        "event.segment",
        10 * NS,
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
                for index, item_id in enumerate(item_id for pair in item_ids for item_id in pair)
            ],
        },
    )
    log.span(
        "model.cycle",
        "3" * 32,
        10 * NS,
        10 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "4" * 32},
    )
    log.span(
        "model.exchange",
        "4" * 32,
        10 * NS,
        10 * NS,
        parent_operation_id="3" * 32,
        start_payload={"context_revision_id": "5" * 32},
    )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    return log


@pytest.mark.parametrize(
    ("reason", "diagnostic_code"),
    [
        ("missing code", None),
        (None, TimelineDiagnosticCode.MISSING_START),
    ],
)
def test_timeline_operation_requires_reason_and_diagnostic_code_together(
    reason: str | None,
    diagnostic_code: TimelineDiagnosticCode | None,
) -> None:
    with pytest.raises(ValueError, match="reason and diagnostic code must be set together"):
        TimelineOperationDetail(
            reason=reason,
            diagnostic_code=diagnostic_code,
        )


def test_metric_sum_preserves_estimated_precision() -> None:
    total = _sum_metrics([Metric(2, Precision.EXACT), Metric(3, Precision.ESTIMATED)])

    assert (total.value, total.precision) == (5, Precision.ESTIMATED)


def test_two_parallel_tasks_have_cp_equal_elapsed_and_parallelism_two(tmp_path) -> None:
    """CP/elapsed=1 does not imply an absence of parallel work."""
    path = tmp_path / "events.jsonl"
    _two_parallel_tools_log().write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.value == pytest.approx(10 * NS)
    assert turn.response_cp_ns.value == pytest.approx(10 * NS)
    assert turn.parallelism.value == pytest.approx(2.0)
    assert turn.overlap_gain_ns.value == 10 * NS


def test_wall_partition_is_additive_but_utilization_is_not(tmp_path) -> None:
    """Concurrent model and hook work produces 100% wall time but 200% utilization."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * NS, links=caused_by("a" * 32))
    log.span(
        "hook.operation",
        "c" * 32,
        0,
        10 * NS,
        start_payload={"hook_event": "session_start", "execution_mode": "async", "scope": "session"},
    )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert sum(metric.value for metric in turn.wall_time_ns.values()) == 10 * NS
    assert turn.wall_time_ns[WallBucket.MODEL].value == 10 * NS
    assert turn.wall_time_ns[WallBucket.TOOLS].value == 0
    assert turn.utilization[WallBucket.MODEL].value == pytest.approx(1.0)
    assert turn.utilization[WallBucket.TOOLS].value == pytest.approx(1.0)


def test_approval_only_turn_has_wait_wall_time_but_zero_work(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "approval.requested",
        0,
        operation_id="a" * 32,
        payload={"approval_request_id": "a" * 32},
    )
    log.add(
        "approval.resolved",
        10 * NS,
        operation_id="a" * 32,
        payload={"approval_request_id": "a" * 32, "outcome": "approved", "wait_ms": 10_000},
        measurements={"/payload/wait_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.exclusive_work_ns.value == 0
    assert turn.parallelism.value == 0
    assert turn.wall_time_ns[WallBucket.WAIT].value == 10 * NS


def test_finalizer_tail_stays_idle_in_wall_partition(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * NS, links=caused_by("a" * 32))
    log.add("turn.finished", 10 * NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(20 * NS)
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.value == 20 * NS
    assert turn.wall_time_ns[WallBucket.MODEL].value == 10 * NS
    assert turn.wall_time_ns[WallBucket.IDLE].value == 10 * NS
    assert turn.wall_time_ns[WallBucket.TOOLS].value == 0
    assert turn.wall_time_ns[WallBucket.WAIT].value == 0


@pytest.mark.parametrize("missing", ["turn", "tool"])
def test_missing_required_caused_by_edge_makes_both_cp_families_unresolved(tmp_path, missing: str) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, NS, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, NS, 10 * NS, links=() if missing == "turn" else caused_by("a" * 32))
    log.span(
        "preparation",
        "c" * 32,
        2 * NS,
        3 * NS,
        parent_operation_id="b" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "d" * 32},
    )
    log.span(
        "tool.operation",
        "d" * 32,
        3 * NS,
        4 * NS,
        parent_operation_id="b" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "b" * 32,
        },
        links=() if missing == "tool" else caused_by("c" * 32),
    )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert turn.exclusive_work_ns.precision is Precision.EXACT


@pytest.mark.parametrize(
    ("outcome", "has_result", "expected"),
    [
        # Invalid-argument and unknown-tool closes still own a result item the
        # next exchange consumes; a filtered close never gets one.
        ("invalid_arguments", True, Precision.EXACT),
        ("unknown_tool", True, Precision.EXACT),
        ("filtered", False, Precision.EXACT),
        ("errored", True, Precision.UNRESOLVED),
    ],
)
def test_never_dispatched_tool_needs_no_preamble_pairing(
    tmp_path, outcome: str, has_result: bool, expected: Precision
) -> None:
    """The kernel closes undispatched calls without a ``tool_preamble``; only those outcomes are exempt."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 6 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        6 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "f" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, NS, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "1" * 32,
        NS,
        2 * NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "2" * 32},
    )
    log.span(
        "tool.operation",
        "2" * 32,
        2 * NS,
        4 * NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "d" * 32,
            "call_item_id": "5" * 32,
        },
        finish_payload={"result_item_id": "6" * 32},
        links=caused_by("1" * 32),
    )
    log.span(
        "tool.operation",
        "3" * 32,
        NS,
        NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "zsh",
            "tool_kind": "shell",
            "parent_model_operation_id": "d" * 32,
            "call_item_id": "7" * 32,
        },
        finish_payload={"outcome": outcome, "error_kind": outcome}
        | ({"result_item_id": "8" * 32} if has_result else {}),
    )
    member_ids = ("5" * 32, "6" * 32, "7" * 32, *(("8" * 32,) if has_result else ()))
    revision = log.add(
        "context.revision.recorded",
        4 * NS,
        operation_id="9" * 32,
        parent_operation_id="f" * 32,
        payload={
            "revision_id": "9" * 32,
            "is_checkpoint": True,
            "item_count": len(member_ids),
            "unidentified_item_count": 0,
        },
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id="0" * 32, segment_count=1),),
    )
    log.add(
        "event.segment",
        4 * NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": "0" * 32,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": item_id, "occurrence": 0, "position": index, "action": "add"}
                for index, item_id in enumerate(member_ids)
            ],
        },
    )
    log.span(
        "model.exchange",
        "f" * 32,
        4 * NS,
        6 * NS,
        parent_operation_id="c" * 32,
        start_payload={"context_revision_id": "9" * 32},
    )
    log.add("turn.finished", 6 * NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(6 * NS)
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is expected
    assert turn.response_cp_ns.precision is expected
    pairing_diagnostics = [diagnostic for diagnostic in turn.diagnostics if "tool_preamble" in diagnostic]
    if expected is Precision.EXACT:
        assert pairing_diagnostics == []
        assert not any("tool result fan-in" in diagnostic for diagnostic in turn.diagnostics)
        assert turn.compute_cp_ns.value == 6 * NS
    else:
        assert "tool_preamble to tool pairing is not one-to-one" in pairing_diagnostics
        assert "tool operation lacks caused_by link to its tool_preamble" in pairing_diagnostics


def test_continuation_poll_wait_connects_neighboring_exchanges_serially(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 4 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        4 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "e" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, NS, parent_operation_id="c" * 32)
    log.span(
        "model.exchange",
        "e" * 32,
        3 * NS,
        4 * NS,
        parent_operation_id="c" * 32,
        start_payload={"continuation_mode": "poll"},
    )
    log.add("turn.finished", 4 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "continuation-poll.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.value == 2 * NS
    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.value == 4 * NS
    assert turn.response_cp_ns.precision is Precision.EXACT


def test_retry_run_chain_uses_previous_run_pointer_without_time_adjacency(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * NS, links=caused_by("a" * 32))
    log.add(
        "retry.scheduled",
        10 * NS,
        operation_id=None,
        payload={
            "retry_mode": "run",
            "previous_operation_id": "b" * 32,
            "next_operation_id": "c" * 32,
            "delay_ms": 1000,
        },
    )
    log.add(
        "retry.started",
        11 * NS,
        operation_id="c" * 32,
        payload={
            "retry_mode": "run",
            "previous_operation_id": "b" * 32,
            "next_operation_id": "c" * 32,
        },
    )
    log.span(
        "model.run",
        "c" * 32,
        11 * NS,
        21 * NS,
        start_payload={"previous_run_operation_id": "b" * 32},
    )
    log.span(
        "model.cycle",
        "d" * 32,
        21 * NS,
        21 * NS,
        parent_operation_id="c" * 32,
        finish_payload={"final_exchange_operation_id": "e" * 32},
    )
    log.span("model.exchange", "e" * 32, 21 * NS, 21 * NS, parent_operation_id="d" * 32)
    log.add("turn.finished", 21 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    assert len(analysis.turns) == 1
    turn = analysis.turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.compute_cp_ns.value == 20 * NS
    assert turn.response_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.value == 21 * NS


def test_response_cp_without_typed_final_exchange_is_unresolved(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * NS, links=caused_by("a" * 32))
    log.span("model.cycle", "c" * 32, 0, 10 * NS, parent_operation_id="b" * 32)
    log.span("model.exchange", "d" * 32, 0, 10 * NS, parent_operation_id="c" * 32)
    log.add("turn.finished", 10 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.compute_cp_ns.value == 10 * NS
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.value is None


@pytest.mark.parametrize("typed_fork", [False, True])
def test_response_cp_requires_typed_root_to_fence_path_and_deduplicates_fork_overlap(
    tmp_path,
    typed_fork: bool,
) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 10 * NS, parent_operation_id="c" * 32)
    log.span(
        "hook.operation",
        "e" * 32,
        8 * NS,
        13 * NS,
        start_payload={
            "hook_event": "after_turn",
            "execution_mode": "async",
            "scope": "turn",
            **({"target_operation_id": "d" * 32} if typed_fork else {}),
        },
        links=caused_by("d" * 32) if typed_fork else (),
    )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(13 * NS, waited_hook_ids=["e" * 32])
    path = tmp_path / "events.jsonl"
    log.write(path)

    response_cp = analyze_trajectory(path).turns[0].response_cp_ns

    if typed_fork:
        assert response_cp.precision is Precision.EXACT
        assert response_cp.value == 13 * NS
    else:
        assert response_cp.precision is Precision.UNRESOLVED
        assert response_cp.value is None


def test_sub_agent_subgraph_nests_under_its_tool_and_counts_once(tmp_path) -> None:
    """An in-process sub-agent is the tool's displaced subgraph, not a side call."""
    sub_actor = Actor(kind="agent", role="sub_agent", actor_id="0" * 32)
    call_item_id = "9" * 32
    result_item_id = "a" * 32
    revision_id = "6" * 32
    segment_group_id = "5" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 20 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.exchange_usage("d" * 32, 0, 2 * NS, {"input_total": 10, "output_total": 1}, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "e" * 32,
        2 * NS,
        2 * NS,
        parent_operation_id="c" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        2 * NS,
        10 * NS,
        parent_operation_id="c" * 32,
        start_payload={
            "tool_name": "explore_agent",
            "tool_kind": "sub_agent",
            "batch_index": 0,
            "parent_model_operation_id": "d" * 32,
            "call_item_id": call_item_id,
        },
        finish_payload={"result_item_id": result_item_id},
        links=caused_by("e" * 32),
    )
    log.span(
        "sub_agent",
        "1" * 32,
        2 * NS,
        9 * NS,
        parent_operation_id="f" * 32,
        start_payload={"invocation_id": "deadbeef1234", "parent_tool_operation_id": "f" * 32},
    )
    log.span(
        "model.cycle",
        "2" * 32,
        3 * NS,
        8 * NS,
        parent_operation_id="1" * 32,
        actor=sub_actor,
        finish_payload={"final_exchange_operation_id": "3" * 32},
    )
    log.exchange_usage(
        "3" * 32,
        3 * NS,
        8 * NS,
        {"input_total": 100, "output_total": 20},
        parent_operation_id="2" * 32,
        actor=sub_actor,
    )
    revision = log.add(
        "context.revision.recorded",
        10 * NS,
        operation_id=revision_id,
        parent_operation_id="8" * 32,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 2, "unidentified_item_count": 0},
        segmented_fields=(
            SegmentedField(field_pointer="/payload/refs", segment_group_id=segment_group_id, segment_count=1),
        ),
    )
    log.add(
        "event.segment",
        10 * NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": segment_group_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": call_item_id, "occurrence": 0, "position": 0, "action": "add"},
                {"item_id": result_item_id, "occurrence": 0, "position": 1, "action": "add"},
            ],
        },
    )
    log.span(
        "model.cycle",
        "7" * 32,
        10 * NS,
        20 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "8" * 32},
    )
    log.exchange_usage(
        "8" * 32,
        10 * NS,
        20 * NS,
        {"input_total": 30, "output_total": 3},
        parent_operation_id="7" * 32,
        start_payload={"context_revision_id": revision_id},
    )
    log.add("turn.finished", 20 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    analysis = analyzer.load(path)
    turn = analysis.turns[0]

    assert turn.diagnostics == ()
    # The sub-agent's exchange is real usage of the turn.
    assert (turn.usage_tokens.value, turn.usage_tokens.precision) == (164, Precision.EXACT)
    # The consumer waited for the whole tool, sub-agent included.
    assert (turn.compute_cp_ns.value, turn.compute_cp_ns.precision) == (20 * NS, Precision.EXACT)
    # Displacement keeps the nested time counted exactly once.
    assert (turn.exclusive_work_ns.value, turn.exclusive_work_ns.precision) == (20 * NS, Precision.EXACT)
    assert turn.wall_time_ns[WallBucket.MODEL].value == 17 * NS
    assert turn.wall_time_ns[WallBucket.TOOLS].value == 3 * NS
    operation_ids = {operation.operation_id for operation in turn.operations}
    assert {"1" * 32, "2" * 32, "3" * 32} <= operation_ids
    assert analyzer.counter_samples().usage_by_turn[turn.turn_id][-1].input_tokens == 30


def test_response_fence_rejects_duplicate_hook_membership_ids(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 10 * NS, parent_operation_id="c" * 32)
    for hook_id, end_ns in (("e" * 32, 5 * NS), ("f" * 32, 15 * NS)):
        log.span(
            "hook.operation",
            hook_id,
            4 * NS,
            end_ns,
            start_payload={
                "hook_event": "after_turn",
                "execution_mode": "async",
                "scope": "turn",
                "target_operation_id": "d" * 32,
            },
            links=caused_by("d" * 32),
        )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(16 * NS, waited_hook_ids=["e" * 32, "e" * 32])
    path = tmp_path / "duplicate-fence-members.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any("hook membership is incomplete" in item for item in turn.diagnostics)


def test_response_fence_accepts_empty_drained_scopes_without_hooks(tmp_path) -> None:
    """No hook manager means there was no turn scope to drain, not a partial fence."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 10 * NS, parent_operation_id="c" * 32)
    log.add("turn.finished", 10 * NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(10 * NS, drained_scopes=[])
    path = tmp_path / "no-hooks-fence.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.precision is Precision.EXACT
    assert not any("drained_scopes" in item for item in turn.diagnostics)


def test_closed_producing_exchange_is_a_normal_tool_containment_shape(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 3 * NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        2 * NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * NS, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "e" * 32,
        2 * NS,
        2 * NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        2 * NS,
        3 * NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "d" * 32,
            "call_item_id": "1" * 32,
        },
        finish_payload={"result_item_id": "2" * 32},
        links=caused_by("e" * 32),
    )
    log.add("turn.finished", 3 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "closed-producing-exchange.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert analysis.diagnostics.containment_violation_count == 0


def test_duration_mismatch_threshold_is_diagnostic_only(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("wait.started", 0, operation_id="a" * 32, payload={"category": "user_input"})
    log.add(
        "wait.finished",
        10 * NS,
        operation_id="a" * 32,
        payload={"outcome": "completed", "duration_ms": 1},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert analysis.diagnostics.span_duration_mismatch_count == 1
    assert analysis.diagnostics.span_duration_mismatches[0].family == "wait"
    assert analysis.diagnostics.span_duration_mismatches[0].operation_id == "a" * 32
    assert analysis.turns[0].wall_time_ns[WallBucket.WAIT].value == 10 * NS


def test_containment_diagnostics_retain_family_and_operation_callsite(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("model.run", "a" * 32, 2 * NS, 8 * NS)
    log.span(
        "wait",
        "b" * 32,
        NS,
        9 * NS,
        parent_operation_id="a" * 32,
        start_payload={"category": "user_input"},
    )
    log.add("turn.finished", 10 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "containment-detail.jsonl"
    log.write(path)

    diagnostic = analyze_trajectory(path).diagnostics.containment_violations[0]

    assert diagnostic.family == "wait"
    assert diagnostic.operation_id == "b" * 32
    assert diagnostic.parent_family == "model.run"
    assert diagnostic.parent_operation_id == "a" * 32


def test_turn_flow_types_parent_and_causal_edges_against_operations(tmp_path) -> None:
    """Flow edges are index pairs into operations, split by displacing-vs-pointer proof."""
    path = tmp_path / "events.jsonl"
    _two_parallel_tools_log().write(path)

    turn = analyze_trajectory(path).turns[0]
    flow = turn.flow

    assert flow is not None
    assert flow.acyclic
    assert flow.has_terminal
    assert flow.root_index == operation_index(turn, "preparation", "a" * 32)
    parent_edges = set(flow.parent_edges())
    causal_edges = set(flow.causal_edges())
    run_to_cycle = (
        operation_index(turn, "model.run", "b" * 32),
        operation_index(turn, "model.cycle", "1" * 32),
    )
    exchange_to_tool = (
        operation_index(turn, "model.exchange", "2" * 32),
        operation_index(turn, "tool.operation", "c" * 32),
    )
    preamble_to_tool = (
        operation_index(turn, "preparation", "e" * 32),
        operation_index(turn, "tool.operation", "c" * 32),
    )
    preamble_to_run = (
        operation_index(turn, "preparation", "a" * 32),
        operation_index(turn, "model.run", "b" * 32),
    )
    assert run_to_cycle in parent_edges
    assert exchange_to_tool in parent_edges
    assert preamble_to_tool in causal_edges
    assert preamble_to_tool not in parent_edges
    assert preamble_to_run in causal_edges
    assert any(target == FLOW_TERMINAL_INDEX for _, target in causal_edges)
    assert all(target != FLOW_TERMINAL_INDEX for _, target in parent_edges)


def test_turn_flow_never_fabricates_edges_from_sequence_adjacency(tmp_path) -> None:
    """Back-to-back operations without declared pointers stay unconnected in the flow."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "tool.operation",
        "a" * 32,
        0,
        NS,
        start_payload={"tool_name": "first", "tool_kind": "filesystem.read"},
    )
    log.span(
        "tool.operation",
        "b" * 32,
        NS,
        2 * NS,
        start_payload={"tool_name": "second", "tool_kind": "filesystem.read"},
    )
    log.add("turn.finished", 2 * NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]
    flow = turn.flow

    assert flow is not None
    assert flow.parent_edges() == ()
    assert flow.causal_edges() == ()
    assert flow.root_index is None
    assert not flow.has_terminal
