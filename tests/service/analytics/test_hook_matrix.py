# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Executable coverage of the frozen hook ownership matrix."""

from __future__ import annotations

import pytest

from chrys.foundation.trajectory.envelope import Link, LinkRelation, SegmentedField, measurement
from chrys.service.analytics import (
    HookOwnership,
    Metric,
    Precision,
    WallBucket,
    analyze_trajectory,
    classify_hook_ownership,
)
from chrys.service.hooks.events import HookEvent
from tests.service.analytics._events import EventLog, caused_by

_NS = 1_000_000_000
_BLOCKING_OWNERS = {
    HookEvent.SESSION_START: HookOwnership.SESSION_ROOT,
    HookEvent.WORKFLOW_RUN_START: HookOwnership.WORKFLOW_ROOT,
    HookEvent.WORKFLOW_RUN_END: HookOwnership.WORKFLOW_ROOT,
    HookEvent.SESSION_RESTORED: HookOwnership.SESSION_ROOT,
    HookEvent.SESSION_END: HookOwnership.SESSION_ROOT,
    HookEvent.BEFORE_TURN: HookOwnership.TURN_PREAMBLE,
    HookEvent.AFTER_TURN: HookOwnership.TURN_TAIL,
    HookEvent.USER_PROMPT_SUBMIT: HookOwnership.PRE_TURN,
    HookEvent.BEFORE_TOOL_CALL: HookOwnership.TOOL_PREAMBLE,
    HookEvent.AFTER_TOOL_CALL: HookOwnership.TOOL_TAIL,
    HookEvent.TOOL_ERROR: HookOwnership.TOOL_TAIL,
    HookEvent.APPROVAL_REQUESTED: HookOwnership.APPROVAL,
    HookEvent.APPROVAL_RESOLVED: HookOwnership.APPROVAL,
    HookEvent.SUB_AGENT_START: HookOwnership.SUB_AGENT,
    HookEvent.SUB_AGENT_END: HookOwnership.SUB_AGENT,
    HookEvent.PRE_COMPACT: HookOwnership.COMPACTION,
    HookEvent.USER_INTERRUPT: HookOwnership.CONCURRENT,
}
_EXPECTED_METRICS = {
    HookOwnership.CONCURRENT: (10, 10, 10, 11, 1.1, 1, (8, 2, 0, 0), (0.8, 0.3)),
    HookOwnership.SESSION_ROOT: (10, 10, 10, 11, 1.1, 1, (8, 2, 0, 0), (0.8, 0.3)),
    HookOwnership.WORKFLOW_ROOT: (10, 10, 10, 11, 1.1, 1, (8, 2, 0, 0), (0.8, 0.3)),
    HookOwnership.TURN_PREAMBLE: (10, 10, 10, 10, 1.0, 0, (8, 2, 0, 0), (0.8, 0.2)),
    HookOwnership.PRE_TURN: (10, 10, 10, 10, 1.0, 0, (8, 2, 0, 0), (0.8, 0.2)),
    HookOwnership.TURN_TAIL: (12, 12, 12, 12, 1.0, 0, (8, 4, 0, 0), (2 / 3, 1 / 3)),
    HookOwnership.TOOL_PREAMBLE: (10, 10, 10, 10, 1.0, 0, (3, 7, 0, 0), (0.3, 0.7)),
    HookOwnership.TOOL_TAIL: (10, 10, 10, 10, 1.0, 0, (3, 7, 0, 0), (0.3, 0.7)),
    HookOwnership.APPROVAL: (10, 8, 10, 8, 0.8, 0, (3, 5, 2, 0), (0.3, 0.5)),
    HookOwnership.SUB_AGENT: (10, 10, 10, 10, 1.0, 0, (3, 7, 0, 0), (0.3, 0.7)),
    HookOwnership.COMPACTION: (10, 10, 10, 10, 1.0, 0, (7, 3, 0, 0), (0.7, 0.3)),
}


@pytest.mark.parametrize("hook_event", list(HookEvent))
@pytest.mark.parametrize("execution_mode", ["blocking", "async", "fire_and_forget"])
def test_full_hook_event_by_execution_mode_matrix_projects_all_metric_families(
    tmp_path,
    hook_event: HookEvent,
    execution_mode: str,
) -> None:
    expected = _BLOCKING_OWNERS[hook_event] if execution_mode == "blocking" else HookOwnership.CONCURRENT

    assert classify_hook_ownership(hook_event, execution_mode) is expected
    turn = _project_hook_sample(tmp_path, hook_event, execution_mode)
    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.precision is Precision.EXACT
    assert turn.elapsed_ns.precision is Precision.EXACT
    assert all(metric.precision is Precision.EXACT for metric in turn.wall_time_ns.values())
    assert turn.parallelism.precision is Precision.EXACT
    assert all(metric.precision is Precision.EXACT for metric in turn.utilization.values())
    assert turn.overlap_gain_ns.precision is Precision.EXACT
    elapsed, compute_cp, response_cp, work, parallelism, overlap, wall, utilization = _EXPECTED_METRICS[expected]
    assert turn.elapsed_ns.value == elapsed * _NS
    assert turn.compute_cp_ns.value == compute_cp * _NS
    assert turn.response_cp_ns.value == response_cp * _NS
    assert turn.exclusive_work_ns.value == work * _NS
    assert turn.parallelism.value == pytest.approx(parallelism)
    assert turn.overlap_gain_ns.value == overlap * _NS
    assert tuple(turn.wall_time_ns[bucket].value for bucket in WallBucket) == tuple(value * _NS for value in wall)
    assert tuple(turn.utilization[bucket].value for bucket in (WallBucket.MODEL, WallBucket.TOOLS)) == pytest.approx(
        utilization
    )


def test_user_interrupt_is_concurrent_even_when_configured_blocking(tmp_path) -> None:
    assert classify_hook_ownership(HookEvent.USER_INTERRUPT, "blocking") is HookOwnership.CONCURRENT
    turn = _project_hook_sample(tmp_path, HookEvent.USER_INTERRUPT, "blocking")
    assert turn.exclusive_work_ns.precision is Precision.EXACT
    assert turn.overlap_gain_ns.value == _NS


def test_multiple_blocking_hooks_in_one_dispatch_form_one_serial_chain(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 2 * _NS, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    for operation_id, start_ns, end_ns in (("1" * 32, 0, _NS), ("2" * 32, _NS, 2 * _NS)):
        log.span(
            "hook.operation",
            operation_id,
            start_ns,
            end_ns,
            start_payload={
                "hook_event": HookEvent.BEFORE_TURN,
                "execution_mode": "blocking",
                "scope": "turn",
                "target_operation_id": "a" * 32,
            },
        )
    log.span("model.run", "b" * 32, 2 * _NS, 10 * _NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        2 * _NS,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 2 * _NS, 10 * _NS, parent_operation_id="c" * 32)
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "serial-blocking-hooks.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns == Metric(10 * _NS, Precision.EXACT)
    assert turn.response_cp_ns == Metric(10 * _NS, Precision.EXACT)


def test_serial_after_tool_hooks_with_same_target_are_not_absorbed_into_tool_preamble(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span(
        "model.run",
        "b" * 32,
        0,
        10 * _NS,
        links=(Link(relation=LinkRelation.CAUSED_BY, target_operation_id="a" * 32),),
    )
    log.span(
        "model.cycle",
        "1" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "2" * 32},
    )
    log.span("model.exchange", "2" * 32, 0, 10 * _NS, parent_operation_id="1" * 32)
    log.span(
        "preparation",
        "c" * 32,
        0,
        5 * _NS,
        parent_operation_id="2" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "d" * 32},
    )
    log.span(
        "tool.operation",
        "d" * 32,
        5 * _NS,
        10 * _NS,
        parent_operation_id="2" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "2" * 32,
            "call_item_id": "3" * 32,
        },
        finish_payload={"result_item_id": "4" * 32},
        links=(Link(relation=LinkRelation.CAUSED_BY, target_operation_id="c" * 32),),
    )
    log.span(
        "hook.operation",
        "e" * 32,
        8 * _NS,
        9 * _NS,
        parent_operation_id="d" * 32,
        start_payload={
            "hook_event": HookEvent.AFTER_TOOL_CALL,
            "execution_mode": "blocking",
            "scope": "turn",
            "target_operation_id": "d" * 32,
        },
    )
    log.span(
        "hook.operation",
        "f" * 32,
        9 * _NS,
        10 * _NS,
        parent_operation_id="d" * 32,
        start_payload={
            "hook_event": HookEvent.AFTER_TOOL_CALL,
            "execution_mode": "blocking",
            "scope": "turn",
            "target_operation_id": "d" * 32,
        },
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]
    tool_preamble = [item for item in turn.slices if item.operation_id == "c" * 32]

    assert sum(item.duration_ns for item in tool_preamble) == 5 * _NS
    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.precision is Precision.EXACT


def test_parallel_hook_closing_after_turn_is_not_absorbed(tmp_path) -> None:
    log = _base_log()
    log.span(
        "preparation",
        "1" * 32,
        5 * _NS,
        5 * _NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "2" * 32},
    )
    log.span(
        "tool.operation",
        "2" * 32,
        5 * _NS,
        10 * _NS,
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
        "hook.operation",
        "3" * 32,
        8 * _NS,
        12 * _NS,
        start_payload={
            "hook_event": HookEvent.AFTER_TOOL_CALL,
            "execution_mode": "fire_and_forget",
            "scope": "background",
            "target_operation_id": "2" * 32,
        },
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    tool = [item for item in turn.slices if item.operation_id == "2" * 32]
    hook = [item for item in turn.slices if item.operation_id == "3" * 32]
    assert sum(item.duration_ns for item in tool) == 5 * _NS
    assert sum(item.duration_ns for item in hook) == 2 * _NS
    assert turn.overlap_gain_ns.value >= 2 * _NS


def test_unplaceable_blocking_hook_uses_safe_over_degradation(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span(
        "model.run",
        "b" * 32,
        0,
        10 * _NS,
        links=(Link(relation=LinkRelation.CAUSED_BY, target_operation_id="a" * 32),),
    )
    log.span(
        "hook.operation",
        "c" * 32,
        2 * _NS,
        4 * _NS,
        start_payload={"hook_event": HookEvent.PRE_COMPACT, "execution_mode": "blocking", "scope": "turn"},
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.exclusive_work_ns.precision is Precision.UNRESOLVED
    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert turn.elapsed_ns.precision is Precision.UNRESOLVED
    assert all(metric.precision is Precision.UNRESOLVED for metric in turn.wall_time_ns.values())
    assert turn.parallelism.precision is Precision.UNRESOLVED
    assert all(metric.precision is Precision.UNRESOLVED for metric in turn.utilization.values())
    assert turn.overlap_gain_ns.precision is Precision.UNRESOLVED
    assert all(not item.counts_as_work for item in turn.slices if item.operation_id == "c" * 32)


@pytest.mark.parametrize("execution_mode", [None, "future_mode"])
def test_missing_or_unknown_execution_mode_is_malformed_and_over_degrades(tmp_path, execution_mode: str | None) -> None:
    log = _base_log()
    payload = {"hook_event": HookEvent.SESSION_START, "scope": "session"}
    if execution_mode is not None:
        payload["execution_mode"] = execution_mode
    log.span("hook.operation", "f" * 32, 3 * _NS, 4 * _NS, start_payload=payload)
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    turn = analysis.turns[0]

    assert classify_hook_ownership(HookEvent.SESSION_START, execution_mode) is HookOwnership.UNSAFE
    assert analysis.diagnostics.malformed_hook_execution_mode_count == 1
    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert all(metric.precision is Precision.UNRESOLVED for metric in turn.wall_time_ns.values())
    assert turn.parallelism.precision is Precision.UNRESOLVED
    assert all(metric.precision is Precision.UNRESOLVED for metric in turn.utilization.values())
    assert turn.overlap_gain_ns.precision is Precision.UNRESOLVED


def _base_log(*, revision_id: str | None = None) -> EventLog:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 2 * _NS, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 2 * _NS, 10 * _NS, links=caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        2 * _NS,
        (3 if revision_id is not None else 10) * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span(
        "model.exchange",
        "d" * 32,
        2 * _NS,
        (3 if revision_id is not None else 10) * _NS,
        parent_operation_id="c" * 32,
    )
    if revision_id is None:
        return log
    revision = log.add(
        "context.revision.recorded",
        8 * _NS,
        operation_id=revision_id,
        parent_operation_id="f" * 32,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 2, "unidentified_item_count": 0},
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id="8" * 32, segment_count=1),),
    )
    log.add(
        "event.segment",
        8 * _NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": "8" * 32,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": "5" * 32, "occurrence": 0, "position": 0, "action": "add"},
                {"item_id": "6" * 32, "occurrence": 0, "position": 1, "action": "add"},
            ],
        },
    )
    log.span(
        "model.cycle",
        "e" * 32,
        8 * _NS,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "f" * 32},
    )
    log.span(
        "model.exchange",
        "f" * 32,
        8 * _NS,
        10 * _NS,
        parent_operation_id="e" * 32,
        start_payload={"context_revision_id": revision_id} if revision_id is not None else None,
    )
    return log


def _project_hook_sample(tmp_path, hook_event: HookEvent, execution_mode: str):
    has_tool = execution_mode == "blocking" and hook_event in {
        HookEvent.BEFORE_TOOL_CALL,
        HookEvent.AFTER_TOOL_CALL,
        HookEvent.TOOL_ERROR,
        HookEvent.APPROVAL_REQUESTED,
        HookEvent.APPROVAL_RESOLVED,
    }
    log = _base_log(revision_id="7" * 32 if has_tool else None)
    hook_id = "9" * 32
    turn_id = None if hook_event == HookEvent.USER_PROMPT_SUBMIT and execution_mode == "blocking" else "4" * 32
    payload = {"hook_event": hook_event, "execution_mode": execution_mode, "scope": "session"}
    start_ns = 3 * _NS
    end_ns = 4 * _NS
    if execution_mode == "blocking":
        if hook_event == HookEvent.BEFORE_TURN:
            payload["target_operation_id"] = "a" * 32
            start_ns, end_ns = 0, _NS
        elif has_tool:
            log.span(
                "preparation",
                "1" * 32,
                3 * _NS,
                4 * _NS,
                parent_operation_id="d" * 32,
                start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "2" * 32},
            )
            log.span(
                "tool.operation",
                "2" * 32,
                4 * _NS,
                8 * _NS,
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
            payload["target_operation_id"] = "2" * 32
            start_ns, end_ns = (3 * _NS, 4 * _NS) if hook_event == HookEvent.BEFORE_TOOL_CALL else (7 * _NS, 8 * _NS)
            if hook_event in {HookEvent.APPROVAL_REQUESTED, HookEvent.APPROVAL_RESOLVED}:
                # Native approval spans share the target tool's operation ID.
                # The hook runs inside the approval interval, so its work must
                # displace that wait rather than overlap it or become tool tail.
                payload["target_operation_id"] = "6" * 32
                log.add(
                    "approval.requested",
                    4 * _NS,
                    operation_id="2" * 32,
                    parent_operation_id="d" * 32,
                    payload={"target_tool_operation_id": "2" * 32, "approval_request_id": "6" * 32},
                )
                log.add(
                    "approval.resolved",
                    7 * _NS,
                    operation_id="2" * 32,
                    parent_operation_id="d" * 32,
                    payload={"approval_request_id": "6" * 32, "wait_ms": 3000},
                    measurements={"/payload/wait_ms": measurement("monotonic_clock", method_version=1)},
                )
                start_ns, end_ns = (
                    (4 * _NS, 5 * _NS) if hook_event == HookEvent.APPROVAL_REQUESTED else (6 * _NS, 7 * _NS)
                )
        elif hook_event in {HookEvent.SUB_AGENT_START, HookEvent.SUB_AGENT_END}:
            log.span("sub_agent", "3" * 32, 3 * _NS, 8 * _NS, parent_operation_id="d" * 32)
            payload["target_operation_id"] = "3" * 32
        elif hook_event == HookEvent.PRE_COMPACT:
            log.span("compaction", "3" * 32, 3 * _NS, 8 * _NS, parent_operation_id="d" * 32)
            payload["target_operation_id"] = "3" * 32
        elif hook_event == HookEvent.AFTER_TURN:
            payload["scope"] = "turn"
            start_ns, end_ns = 10 * _NS, 12 * _NS
    else:
        payload["target_operation_id"] = "d" * 32
    log.span("hook.operation", hook_id, start_ns, end_ns, turn_id=turn_id, start_payload=payload)
    if hook_event == HookEvent.AFTER_TURN and execution_mode == "blocking":
        log.add("turn.finished", 10 * _NS, payload={"end_reason": "completed", "duration_ms": 0})
        log.settled(12 * _NS)
    else:
        log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / f"{execution_mode}-{hook_event}.jsonl"
    log.write(path)
    turn = analyze_trajectory(path).turns[0]
    assert sum(1 for item in turn.slices if item.operation_id == hook_id) == (0 if turn_id is None else 1)
    assert sum(int(metric.value or 0) for metric in turn.wall_time_ns.values()) == int(turn.elapsed_ns.value or 0)
    assert set(turn.utilization) == {WallBucket.MODEL, WallBucket.TOOLS}
    return turn
