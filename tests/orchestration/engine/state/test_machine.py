# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for EngineStateMachine — state transitions, guards, and error handling."""

from __future__ import annotations

import pytest

from chrys.orchestration.engine.state.machine import (
    EngineState,
    EngineStateMachine,
    InvalidTransition,
    Trigger,
)


class TestEngineStateMachine:
    """Core transition tests."""

    def test_initial_state(self) -> None:
        fsm = EngineStateMachine()
        assert fsm.state is EngineState.UNINITIALIZED

    def test_start_transition(self) -> None:
        fsm = EngineStateMachine()
        result = fsm.transition(Trigger.START)
        assert result is EngineState.IDLE
        assert fsm.state is EngineState.IDLE

    def test_full_successful_run_cycle(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        assert fsm.state is EngineState.RUNNING
        fsm.transition(Trigger.RUN_COMPLETED)
        assert fsm.state is EngineState.IDLE

    def test_interrupted_run_cycle(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RUN_INTERRUPTED)
        assert fsm.state is EngineState.INTERRUPTED

    def test_failed_run_cycle(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RUN_FAILED)
        assert fsm.state is EngineState.FAILED

    def test_message_after_interrupt(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RUN_INTERRUPTED)
        fsm.transition(Trigger.USER_MESSAGE)
        assert fsm.state is EngineState.RUNNING

    def test_message_after_failure(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RUN_FAILED)
        fsm.transition(Trigger.USER_MESSAGE)
        assert fsm.state is EngineState.RUNNING

    def test_retry_from_interrupted(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RUN_INTERRUPTED)
        fsm.transition(Trigger.RETRY_STARTED)
        assert fsm.state is EngineState.RUNNING

    def test_retry_from_failed(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RUN_FAILED)
        fsm.transition(Trigger.RETRY_STARTED)
        assert fsm.state is EngineState.RUNNING

    def test_restore_interrupted_from_idle(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.RESTORE_INTERRUPTED)
        assert fsm.state is EngineState.INTERRUPTED

    def test_restore_failed_from_idle(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.RESTORE_FAILED)
        assert fsm.state is EngineState.FAILED

    def test_restore_terminal_state_helper(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        result = fsm.restore_terminal_state(failed=False)
        assert result is EngineState.INTERRUPTED
        assert fsm.state is EngineState.INTERRUPTED

    def test_pending_retry_while_running(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RETRY_REQUESTED)
        assert fsm.state is EngineState.PENDING_RETRY

    def test_pending_retry_auto_starts_on_completed(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RETRY_REQUESTED)
        fsm.transition(Trigger.RUN_COMPLETED)
        assert fsm.state is EngineState.RUNNING

    def test_pending_retry_auto_starts_on_interrupted(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RETRY_REQUESTED)
        fsm.transition(Trigger.RUN_INTERRUPTED)
        assert fsm.state is EngineState.RUNNING

    def test_pending_retry_does_not_auto_start_on_failed(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RETRY_REQUESTED)
        fsm.transition(Trigger.RUN_FAILED)
        assert fsm.state is EngineState.FAILED

    def test_shutdown_from_idle(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.SHUTDOWN)
        assert fsm.state is EngineState.SHUTTING_DOWN

    def test_shutdown_from_interrupted(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RUN_INTERRUPTED)
        fsm.transition(Trigger.SHUTDOWN)
        assert fsm.state is EngineState.SHUTTING_DOWN

    def test_shutdown_from_uninitialized(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.SHUTDOWN)
        assert fsm.state is EngineState.SHUTTING_DOWN


class TestInvalidTransitions:
    """Verify that illegal transitions raise InvalidTransition."""

    def test_message_before_start(self) -> None:
        fsm = EngineStateMachine()
        with pytest.raises(InvalidTransition):
            fsm.transition(Trigger.USER_MESSAGE)

    def test_double_start(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        with pytest.raises(InvalidTransition):
            fsm.transition(Trigger.START)

    def test_run_completed_when_idle(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        with pytest.raises(InvalidTransition):
            fsm.transition(Trigger.RUN_COMPLETED)

    def test_message_while_running(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        with pytest.raises(InvalidTransition):
            fsm.transition(Trigger.USER_MESSAGE)

    def test_retry_from_idle(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        with pytest.raises(InvalidTransition):
            fsm.transition(Trigger.RETRY_STARTED)

    def test_shutdown_while_running(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        # SHUTDOWN is allowed from every state, including RUNNING
        fsm.transition(Trigger.SHUTDOWN)
        assert fsm.state == EngineState.SHUTTING_DOWN

    def test_error_contains_state_and_trigger(self) -> None:
        fsm = EngineStateMachine()
        with pytest.raises(InvalidTransition) as exc_info:
            fsm.transition(Trigger.USER_MESSAGE)
        assert exc_info.value.state is EngineState.UNINITIALIZED
        assert exc_info.value.trigger is Trigger.USER_MESSAGE


class TestTryTransition:
    """Tests for the non-raising try_transition method."""

    def test_valid_transition_returns_new_state(self) -> None:
        fsm = EngineStateMachine()
        result = fsm.try_transition(Trigger.START)
        assert result is EngineState.IDLE
        assert fsm.state is EngineState.IDLE

    def test_invalid_transition_returns_none(self) -> None:
        fsm = EngineStateMachine()
        result = fsm.try_transition(Trigger.USER_MESSAGE)
        assert result is None
        assert fsm.state is EngineState.UNINITIALIZED


class TestGuardMethods:
    """Tests for is_accepting_messages and is_running."""

    def test_accepting_when_idle(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        assert fsm.is_accepting_messages() is True
        assert fsm.is_running() is False

    def test_not_accepting_when_running(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        assert fsm.is_accepting_messages() is False
        assert fsm.is_running() is True

    def test_accepting_when_interrupted(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RUN_INTERRUPTED)
        assert fsm.is_accepting_messages() is True
        assert fsm.is_running() is False

    def test_accepting_when_failed(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RUN_FAILED)
        assert fsm.is_accepting_messages() is True
        assert fsm.is_running() is False

    def test_running_when_pending_retry(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.transition(Trigger.RETRY_REQUESTED)
        assert fsm.is_running() is True
        assert fsm.is_accepting_messages() is False

    def test_not_accepting_when_uninitialized(self) -> None:
        fsm = EngineStateMachine()
        assert fsm.is_accepting_messages() is False
        assert fsm.is_running() is False


class TestReset:
    """Tests for the reset method."""

    def test_reset_returns_to_uninitialized(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        fsm.reset()
        assert fsm.state is EngineState.UNINITIALIZED


class TestAwaitingSubAgents:
    """Transitions involving the AWAITING_SUB_AGENTS state."""

    def _running_fsm(self) -> EngineStateMachine:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        fsm.transition(Trigger.USER_MESSAGE)
        return fsm

    def test_pause_from_running(self) -> None:
        fsm = self._running_fsm()
        fsm.transition(Trigger.SUB_AGENT_PAUSED)
        assert fsm.state is EngineState.AWAITING_SUB_AGENTS
        assert fsm.is_running() is True
        assert fsm.is_accepting_messages() is False
        assert fsm.is_awaiting_sub_agents() is True

    def test_pause_from_pending_retry(self) -> None:
        fsm = self._running_fsm()
        fsm.transition(Trigger.RETRY_REQUESTED)
        assert fsm.state is EngineState.PENDING_RETRY
        fsm.transition(Trigger.SUB_AGENT_PAUSED)
        assert fsm.state is EngineState.AWAITING_SUB_AGENTS

    def test_resolve_returns_to_running(self) -> None:
        fsm = self._running_fsm()
        fsm.transition(Trigger.SUB_AGENT_PAUSED)
        fsm.transition(Trigger.SUB_AGENT_RESOLVED)
        assert fsm.state is EngineState.RUNNING
        assert fsm.is_awaiting_sub_agents() is False

    def test_awaiting_plus_interrupt_goes_to_interrupted(self) -> None:
        fsm = self._running_fsm()
        fsm.transition(Trigger.SUB_AGENT_PAUSED)
        fsm.transition(Trigger.RUN_INTERRUPTED)
        assert fsm.state is EngineState.INTERRUPTED

    def test_awaiting_plus_failure_goes_to_failed(self) -> None:
        fsm = self._running_fsm()
        fsm.transition(Trigger.SUB_AGENT_PAUSED)
        fsm.transition(Trigger.RUN_FAILED)
        assert fsm.state is EngineState.FAILED

    def test_awaiting_plus_completion_goes_to_idle(self) -> None:
        fsm = self._running_fsm()
        fsm.transition(Trigger.SUB_AGENT_PAUSED)
        fsm.transition(Trigger.RUN_COMPLETED)
        assert fsm.state is EngineState.IDLE

    def test_pause_not_valid_from_idle(self) -> None:
        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)
        with pytest.raises(InvalidTransition):
            fsm.transition(Trigger.SUB_AGENT_PAUSED)

    def test_resolve_not_valid_when_not_awaiting(self) -> None:
        fsm = self._running_fsm()
        with pytest.raises(InvalidTransition):
            fsm.transition(Trigger.SUB_AGENT_RESOLVED)

    def test_awaiting_is_not_accepting_messages(self) -> None:
        fsm = self._running_fsm()
        fsm.transition(Trigger.SUB_AGENT_PAUSED)
        assert fsm.is_accepting_messages() is False
