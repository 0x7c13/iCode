# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the shared test waiting helpers."""

from __future__ import annotations

import asyncio

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.orchestration.engine.assembly import assemble_agent_engine
from chrys.orchestration.engine.execution import RunTaskDrainOutcome
from tests.support.waiting import (
    ENGINE_TURN_TIMEOUT,
    _timeout_diagnostic,
    _wait_window,
    await_run_task_chain,
    wait_for,
    wait_until,
    wait_until_quiet,
    with_wait_deadline,
)


@pytest.mark.asyncio
async def test_await_run_task_chain_returns_boundary_outcome() -> None:
    expected = RunTaskDrainOutcome(cancelled=True)

    class _TurnState:
        async def drain_for_boundary(self) -> RunTaskDrainOutcome:
            return expected

    class _Engine:
        turn_lifecycle_task: asyncio.Task[None] | None = None

        async def wait_for_run_task(self) -> None:
            pass

    assert await await_run_task_chain(_Engine(), turn_state=_TurnState()) is expected


@pytest.mark.asyncio
@with_wait_deadline(0.0)
async def test_await_run_task_chain_honours_shared_deadline() -> None:
    class _TurnState:
        async def drain_for_boundary(self) -> RunTaskDrainOutcome:
            await asyncio.Event().wait()
            return RunTaskDrainOutcome()

    class _Engine:
        turn_lifecycle_task: asyncio.Task[None] | None = None

        async def wait_for_run_task(self) -> None:
            await asyncio.Event().wait()

    with pytest.raises(AssertionError) as exc_info:
        await await_run_task_chain(_Engine(), turn_state=_TurnState())

    message = str(exc_info.value)
    assert message.startswith("engine run-task chain completion not met after ")
    assert "timeout=0.000s from shared deadline; requested=15.000s" in message


@pytest.mark.asyncio
async def test_await_run_task_chain_can_require_an_installed_task() -> None:
    engine = assemble_agent_engine(EventBus(), settings=Settings())

    with pytest.raises(AssertionError, match="engine run-task chain was not installed"):
        await await_run_task_chain(engine, turn_state=engine.turns.turn_state, expect_installed=True)


@pytest.mark.asyncio
async def test_await_run_task_chain_can_propagate_inner_cancellation() -> None:
    engine = assemble_agent_engine(EventBus(), settings=Settings())
    task = asyncio.create_task(asyncio.sleep(0))
    engine.turns.turn_state.lease.run_task = task
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await await_run_task_chain(engine, turn_state=engine.turns.turn_state, propagate_inner_cancel=True)

    assert task.cancelled()


@pytest.mark.asyncio
async def test_await_run_task_chain_timeout_does_not_cancel_the_run_task() -> None:
    engine = assemble_agent_engine(EventBus(), settings=Settings())
    started = asyncio.Event()
    release = asyncio.Event()

    async def hung_turn() -> None:
        started.set()
        await release.wait()

    task = asyncio.create_task(hung_turn())
    engine.turns.turn_state.lease.run_task = task
    await asyncio.wait_for(started.wait(), timeout=5.0)

    try:
        with pytest.raises(AssertionError, match="engine run-task chain completion"):
            await await_run_task_chain(engine, turn_state=engine.turns.turn_state, timeout=0.05)

        assert not task.cancelled()
        assert not task.done()
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_wait_for_failure_reports_description_elapsed_and_last_observation() -> None:
    class _PendingObservation:
        def __bool__(self) -> bool:
            return False

        def __repr__(self) -> str:
            return "PendingObservation(phase='persisting', saved=False)"

    observation = _PendingObservation()

    with pytest.raises(AssertionError) as exc_info:
        await wait_for(
            lambda: observation,
            timeout=0.0,
            description="session persistence",
        )

    message = str(exc_info.value)
    assert message.startswith("session persistence not met after ")
    assert "(timeout=0.000s)" in message
    assert "last observed=PendingObservation(phase='persisting', saved=False)" in message


@pytest.mark.asyncio
@with_wait_deadline(0.0)
async def test_wait_for_reports_when_shared_deadline_limits_requested_timeout() -> None:
    with pytest.raises(AssertionError) as exc_info:
        await wait_for(
            lambda: False,
            timeout=ENGINE_TURN_TIMEOUT,
            description="engine turn",
        )

    assert "timeout=0.000s from shared deadline; requested=15.000s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_decimal_timeout_diagnostic_does_not_infer_a_shared_limit_from_rounding() -> None:
    _loop, _started, _deadline, effective, shared_limited = _wait_window(0.1)

    assert shared_limited is False
    assert _timeout_diagnostic(0.1, effective, shared_limited=shared_limited) == "timeout=0.100s"


@pytest.mark.asyncio
@with_wait_deadline(0.0)
async def test_wait_until_ignores_exhausted_shared_deadline() -> None:
    observations = iter([False, True])

    class _Pilot:
        async def pause(self, _interval: float) -> None:
            pass

    assert await wait_until(
        lambda: next(observations),
        timeout=0.5,
        pilot=_Pilot(),
    )


@pytest.mark.asyncio
async def test_wait_until_quiet_failure_reports_description_elapsed_and_last_transition() -> None:
    observations = iter(["mounting", "reflowing"])

    class _Pilot:
        async def pause(self, _interval: float) -> None:
            pass

    with pytest.raises(AssertionError) as exc_info:
        await wait_until_quiet(
            lambda: next(observations),
            description="dialog geometry",
            timeout=0.0,
            pilot=_Pilot(),
        )

    message = str(exc_info.value)
    assert message.startswith("dialog geometry still changing after ")
    assert "(timeout=0.000s)" in message
    assert "previous='mounting'; current='reflowing'" in message
