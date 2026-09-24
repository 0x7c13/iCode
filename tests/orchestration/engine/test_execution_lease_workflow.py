# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The execution lease as one owner for a turn or a workflow run, and the rebuild permit's refusal while a run is held."""

from __future__ import annotations

import asyncio

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.models.execution import ExecutionSnapshot
from chrys.foundation.trajectory.context import TrajectoryContext, main_actor
from chrys.orchestration.engine.assembly import assemble_agent_engine
from chrys.orchestration.engine.execution import PreAdmissionPreparationEntry
from chrys.orchestration.engine.run.turn_state import TurnRuntimeState
from chrys.orchestration.engine.state.lifecycle_permits import RebuildPermit, RebuildPermitDenied
from chrys.service.trajectory.preparation import PreparationScope, PreparationTrace
from tests.service.trajectory._fakes import FakeSink


def test_a_free_lease_is_idle_and_admits_one_workflow_run() -> None:
    lease = TurnRuntimeState().lease
    assert lease.execution() == ExecutionSnapshot("idle")
    assert lease.workflow_start_allowed()
    assert not lease.execution_busy()

    execution = lease.begin_workflow("run-1", "req-1")

    assert lease.workflow is execution
    assert (execution.run_id, execution.request_id, execution.task) == ("run-1", "req-1", None)
    assert lease.execution() == ExecutionSnapshot("workflow", run_id="run-1", cancellable=True, request_id="req-1")
    assert lease.execution_busy()
    assert not lease.turn_busy()
    assert not lease.workflow_start_allowed()
    with pytest.raises(RuntimeError, match="not free"):
        lease.begin_workflow("run-2", "req-2")

    lease.end_workflow(execution)
    assert lease.workflow is None
    assert lease.execution() == ExecutionSnapshot("idle")


def test_a_stale_release_does_not_clear_a_newer_run() -> None:
    lease = TurnRuntimeState().lease
    first = lease.begin_workflow("run-1", "req-1")
    lease.end_workflow(first)
    second = lease.begin_workflow("run-2", "req-2")
    lease.end_workflow(first)
    assert lease.workflow is second


@pytest.mark.asyncio
async def test_a_live_turn_task_blocks_a_workflow_start_and_reads_as_a_turn() -> None:
    lease = TurnRuntimeState().lease
    lease.run_task = asyncio.create_task(asyncio.sleep(3600))
    try:
        assert lease.turn_busy()
        assert lease.execution() == ExecutionSnapshot("turn", cancellable=True)
        assert not lease.workflow_start_allowed()
        with pytest.raises(RuntimeError, match="not free"):
            lease.begin_workflow("run-1", "req-1")
    finally:
        lease.run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await lease.run_task
    assert not lease.turn_busy()
    assert lease.workflow_start_allowed()


def test_closed_prompt_admission_blocks_a_workflow_start() -> None:
    lease = TurnRuntimeState().lease
    lease.close_prompt_admission_for_rebuild()
    assert not lease.workflow_start_allowed()
    assert lease.execution() == ExecutionSnapshot("idle")


@pytest.mark.asyncio
async def test_the_rebuild_permit_is_denied_while_a_workflow_run_holds_the_lease() -> None:
    engine = assemble_agent_engine(EventBus(), settings=Settings())
    lease = engine.turns.turn_state.lease
    execution = lease.begin_workflow("run-1", "req-1")
    try:
        denied = await engine.permits.acquire_rebuild_permit(engine.permits.capture_control_token())
        assert isinstance(denied, RebuildPermitDenied)
        assert (denied.reason, denied.code) == ("busy", "runtime_mutation_busy")
        assert "workflow run" in denied.message
        assert not engine.permits.gate_lock.locked()
    finally:
        lease.end_workflow(execution)
        await lease.settle_notifications()
    permit = await engine.permits.acquire_rebuild_permit(engine.permits.capture_control_token())
    assert isinstance(permit, RebuildPermit)
    engine.permits.release_rebuild_permit(permit)


def test_a_registered_preparation_blocks_a_workflow_start() -> None:
    lease = TurnRuntimeState().lease
    context = TrajectoryContext(sink=FakeSink(), session_id="s", actor=main_actor("s"))
    preparation = PreparationTrace.open(scope=PreparationScope.PRE_TURN, phase="input_admission", context=context)
    assert preparation is not None
    entry = PreAdmissionPreparationEntry(preparation=preparation)
    lease.register_pre_admission_preparation(entry)
    assert not lease.workflow_start_allowed()
    assert lease.execution() == ExecutionSnapshot("idle")  # a preparation is not yet a turn
    lease.deregister_pre_admission_preparation(entry)
    assert lease.workflow_start_allowed()


def test_a_session_operation_in_flight_blocks_a_workflow_start() -> None:
    lease = TurnRuntimeState().lease
    with lease.session_operation():
        assert not lease.workflow_start_allowed()
        assert lease.execution() == ExecutionSnapshot("idle")  # a restore or a fork is not a turn either
        with lease.session_operation():  # a delete nested in a clear
            assert not lease.workflow_start_allowed()
        assert not lease.workflow_start_allowed()
    assert lease.workflow_start_allowed()


async def test_lease_publishes_admission_and_final_release_in_order() -> None:
    from chrys.foundation.events.types import ExecutionChanged, WorkflowRunFinished
    from chrys.orchestration.engine.execution import ExecutionLease

    bus = EventBus()
    lease = ExecutionLease(bus=bus)
    snapshots: list[ExecutionSnapshot] = []

    async def changed(event: ExecutionChanged) -> None:
        snapshots.append(event.snapshot)

    await bus.subscribe(ExecutionChanged, changed)
    try:
        execution = lease.begin_workflow("run", "request")
        await lease.settle_notifications()
        assert snapshots == [ExecutionSnapshot("workflow", "run", True, "request")]
        await bus.publish(WorkflowRunFinished(run_id="run", outcome="completed"))
        assert lease.execution_busy()
        assert len(snapshots) == 1
        lease.end_workflow(execution)
        lease.end_workflow(execution)
        await lease.settle_notifications()
        assert snapshots == [ExecutionSnapshot("workflow", "run", True, "request"), ExecutionSnapshot("idle")]

        complete = asyncio.Event()
        task = asyncio.create_task(complete.wait())
        lease.run_task = task
        await lease.settle_notifications()
        assert snapshots[-1] == ExecutionSnapshot("turn", cancellable=True)
        complete.set()
        await task
        await lease.settle_notifications()
        assert snapshots[-1] == ExecutionSnapshot("idle")
        count = len(snapshots)
        lease.release_run_task()
        await lease.settle_notifications()
        assert len(snapshots) == count
    finally:
        if lease.run_task is not None:
            lease.run_task.cancel()
            await asyncio.gather(lease.run_task, return_exceptions=True)
        lease.end_workflow(execution)
        await lease.settle_notifications()
        await bus.unsubscribe(ExecutionChanged, changed)
