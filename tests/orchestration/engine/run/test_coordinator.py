# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for engine-owned turn coordination."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Error, UserInterrupt, UserMessage, Warning
from chrys.orchestration.engine.state.machine import EngineStateMachine
from chrys.orchestration.engine.trajectory import TrajectoryRecorder
from chrys.service.session.history import SessionHistoryManager
from tests.support.components import make_current, make_permits, make_session, make_turn_state
from tests.support.loaded_agents import install_loaded_agent
from tests.support.turn_services import make_turn_coordinator


def _make_coordinator(bus: EventBus) -> SimpleNamespace:
    session = make_session()
    current = make_current()
    turn_state = make_turn_state()
    permits = make_permits(session=session, turn_state=turn_state)
    turns = make_turn_coordinator(
        session=session,
        current=current,
        turn_state=turn_state,
        permits=permits,
        bus=bus,
        fsm=EngineStateMachine(),
        history=SessionHistoryManager(),
        trajectory_recorder=TrajectoryRecorder(),
    )
    return SimpleNamespace(session=session, current=current, permits=permits, turns=turns)


def _assert_display_message(event: Error | Warning, key: str) -> None:
    reference = event.display_message
    assert reference is not None
    assert reference.definition.key == key
    assert reference.args == ()


async def _collect[T](events: list[T], event: T) -> None:
    events.append(event)


@pytest.mark.asyncio
async def test_interrupt_does_not_target_retry_task_replaced_during_sub_agent_cascade() -> None:
    """A Stop captured for task A must not interrupt replacement retry task B."""
    components = _make_coordinator(EventBus())
    release_a = asyncio.Event()
    release_b = asyncio.Event()
    task_b_started = asyncio.Event()

    async def run_a() -> None:
        await release_a.wait()

    async def run_b() -> None:
        task_b_started.set()
        await release_b.wait()

    task_a = asyncio.create_task(run_a())
    await asyncio.sleep(0)
    components.turns.turn_state.lease.run_task = task_a

    class ReplacementExecutor:
        running = False

        def __init__(self) -> None:
            self.interrupt_calls = 0

        async def interrupt(self) -> None:
            self.interrupt_calls += 1

        @property
        def backend(self):
            return self

        @property
        def inputs(self):
            return self

        @property
        def state(self):
            return self

        @property
        def approval(self):
            return self

        @property
        def tool_events(self):
            return self

    executor = ReplacementExecutor()
    install_loaded_agent(components, bindings=executor)  # type: ignore[assignment]

    class ReplacingSubAgents:
        async def cascade_abort_all(self) -> None:
            release_a.set()
            await task_a
            # Mirrors terminal finalization dispatching a pending retry by
            # replacing the visible run task with B on the same executor.
            components.turns.turn_state.lease.run_task = asyncio.create_task(run_b())
            executor.state.running = True
            await asyncio.sleep(0)

    install_loaded_agent(components, sub_agent_tools=ReplacingSubAgents())  # type: ignore[assignment]
    try:
        await components.turns.on_user_interrupt(UserInterrupt())

        assert components.turns.turn_state.lease.run_task is not task_a
        assert task_b_started.is_set()
        assert executor.interrupt_calls == 0
    finally:
        release_a.set()
        release_b.set()
        await task_a
        task_b = components.turns.turn_state.lease.run_task
        if task_b is not None and task_b is not task_a:
            await task_b


@pytest.mark.asyncio
async def test_user_message_without_executor_publishes_semantic_not_ready_error() -> None:
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    components = _make_coordinator(bus)
    components.session.session_id = "session-1"

    await components.turns.on_user_message(UserMessage(text="hello"))

    assert len(errors) == 1
    assert (errors[0].code, errors[0].message, errors[0].session_id) == (
        "not_ready",
        "Engine not started",
        "session-1",
    )
    _assert_display_message(errors[0], "coordinator.engine_not_started")


@pytest.mark.asyncio
async def test_interrupt_during_load_publishes_semantic_warning() -> None:
    bus = EventBus()
    warnings: list[Warning] = []
    await bus.subscribe(Warning, lambda event: _collect(warnings, event))
    components = _make_coordinator(bus)
    components.session.session_id = "session-1"
    components.permits.begin_agent_load()

    await components.turns.on_user_interrupt(UserInterrupt())

    assert len(warnings) == 1
    assert (warnings[0].code, warnings[0].message, warnings[0].session_id) == (
        "agent_loading_interrupt_ignored",
        "Interrupt ignored while agent infrastructure is loading.",
        "session-1",
    )
    _assert_display_message(warnings[0], "coordinator.interrupt_ignored_loading")


@pytest.mark.asyncio
async def test_prompt_admission_conflict_publishes_semantic_error() -> None:
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    components = _make_coordinator(bus)
    components.session.session_id = "session-1"

    await components.turns._publish_prompt_admission_conflict()

    assert len(errors) == 1
    assert (errors[0].code, errors[0].message, errors[0].session_id) == (
        "prompt_admission_conflict",
        "Prompt could not be admitted because another turn started.",
        "session-1",
    )
    _assert_display_message(errors[0], "coordinator.prompt_admission_conflict")
