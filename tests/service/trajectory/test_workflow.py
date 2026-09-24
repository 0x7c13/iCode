# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow attempt identity, retry settlement, recorder failure and cancellation contracts."""

from __future__ import annotations

import asyncio

import pytest

from chrys.foundation.trajectory.event_types import EventType, WorkflowNodeOutcome
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.service.trajectory.workflow import WorkflowTrace
from chrys.service.workflows.scheduler import ActivationState, AttemptRef, NodeStateChanged

from ._fakes import CancelAckSink, FakeSink, make_context


async def test_run_and_retried_node_have_distinct_attempt_spans() -> None:
    sink = FakeSink()
    context = make_context(sink).with_turn(None).with_run(None)
    run_id = new_analytics_id()
    trace = WorkflowTrace(context, run_id=run_id, workflow_id="example")
    await trace.started()
    for attempt, outcome in [(1, ActivationState.RETRYING), (2, ActivationState.COMPLETED)]:
        ref = AttemptRef(run_id, "review", "review@iter#1", attempt)
        await trace.node_state(NodeStateChanged(ref, ActivationState.RUNNING), kind="agent")
        await trace.node_state(NodeStateChanged(ref, outcome), kind="agent")
    skipped = AttemptRef(run_id, "unused", "unused@iter#1", 0)
    await trace.node_state(NodeStateChanged(skipped, ActivationState.SKIPPED), kind="python")
    trace.finished("completed")
    trace.finished("completed")
    starts = sink.of_type(EventType.WORKFLOW_NODE_STARTED)
    assert [e.payload["attempt"] for e in starts] == [1, 2, 0]
    assert len({e.operation_id for e in starts}) == 3
    assert {e.parent_operation_id for e in starts} == {run_id}
    assert [e.payload["outcome"] for e in sink.of_type(EventType.WORKFLOW_NODE_FINISHED)] == [
        WorkflowNodeOutcome.FAILED,
        WorkflowNodeOutcome.COMPLETED,
        WorkflowNodeOutcome.SKIPPED,
    ]
    assert all(e.payload["duration_ms"] >= 0 for e in sink.drafts if e.event_type.endswith(".finished"))
    sink.assert_operations_settled()


async def test_cancellation_waits_for_drain_and_does_not_finish_an_attempt_twice() -> None:
    sink = FakeSink()
    run_id = new_analytics_id()
    trace = WorkflowTrace(make_context(sink), run_id=run_id, workflow_id="cancel")
    await trace.started()
    ref = AttemptRef(run_id, "node", "node@iter#1", 1)
    await trace.node_state(NodeStateChanged(ref, ActivationState.RUNNING), kind="agent")
    await trace.node_state(NodeStateChanged(ref, ActivationState.CANCELLED), kind="agent")
    assert not sink.of_type(EventType.WORKFLOW_NODE_FINISHED)
    trace.finished("cancelled")
    assert sink.only(EventType.WORKFLOW_NODE_FINISHED).payload["outcome"] == WorkflowNodeOutcome.CANCELLED
    sink.assert_operations_settled()


@pytest.mark.parametrize("cancel", [False, True])
async def test_cancelled_ack_and_recorder_failure_never_leave_an_unowned_opening(cancel: bool) -> None:
    sink = CancelAckSink(at=1) if cancel else FakeSink()
    context = make_context(sink)
    trace = WorkflowTrace(context, run_id=new_analytics_id(), workflow_id="recording")
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await trace.started()
    else:
        sink.fail_next = True
        await trace.started()
    context.finalizers.close()
    trace.finished("cancelled")
    sink.assert_operations_settled()
    assert len(sink.drafts) == (2 if cancel else 0)


def test_node_outcomes_are_closed() -> None:
    assert {outcome.value for outcome in WorkflowNodeOutcome} == {
        "completed",
        "failed",
        "skipped",
        "cancelled",
        "abandoned",
    }
    with pytest.raises(ValueError):
        WorkflowNodeOutcome("retrying")


@pytest.mark.parametrize("close", ["run", "recorder"])
async def test_close_before_start_commits_refuses_a_late_opening(close: str) -> None:
    sink = FakeSink()
    context = make_context(sink)
    trace = WorkflowTrace(context, run_id=new_analytics_id(), workflow_id="late")
    if close == "run":
        trace.finished("cancelled")
    else:
        context.finalizers.close()
    await trace.started()
    trace.finished("cancelled")
    assert not sink.drafts
