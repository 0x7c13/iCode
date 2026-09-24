# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""EngineStateMachine — explicit state tracking for AgentEngine lifecycle.

Replaces scattered boolean flags (is_running, was_interrupted, run_failed,
_pending_retry) with a single enum + transition table.  The transition
table makes legal state changes explicit and testable.
"""

from __future__ import annotations

import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class EngineState(Enum):
    """Possible lifecycle states of the AgentEngine."""

    UNINITIALIZED = auto()
    IDLE = auto()
    RUNNING = auto()
    AWAITING_SUB_AGENTS = auto()
    INTERRUPTED = auto()
    FAILED = auto()
    PENDING_RETRY = auto()
    SHUTTING_DOWN = auto()


class Trigger(Enum):
    """Events that cause state transitions."""

    START = auto()
    USER_MESSAGE = auto()
    RUN_COMPLETED = auto()
    RUN_INTERRUPTED = auto()
    RUN_FAILED = auto()
    RETRY_REQUESTED = auto()
    RETRY_STARTED = auto()
    RESTORE_INTERRUPTED = auto()
    RESTORE_FAILED = auto()
    SHUTDOWN = auto()
    SUB_AGENT_PAUSED = auto()
    SUB_AGENT_RESOLVED = auto()


# (current_state, trigger) → next_state
TRANSITIONS: dict[tuple[EngineState, Trigger], EngineState] = {
    # Initialization
    (EngineState.UNINITIALIZED, Trigger.START): EngineState.IDLE,
    # New user message starts a run
    (EngineState.IDLE, Trigger.USER_MESSAGE): EngineState.RUNNING,
    (EngineState.INTERRUPTED, Trigger.USER_MESSAGE): EngineState.RUNNING,
    (EngineState.FAILED, Trigger.USER_MESSAGE): EngineState.RUNNING,
    # Run outcomes
    (EngineState.RUNNING, Trigger.RUN_COMPLETED): EngineState.IDLE,
    (EngineState.RUNNING, Trigger.RUN_INTERRUPTED): EngineState.INTERRUPTED,
    (EngineState.RUNNING, Trigger.RUN_FAILED): EngineState.FAILED,
    # Retry queued while still running
    (EngineState.RUNNING, Trigger.RETRY_REQUESTED): EngineState.PENDING_RETRY,
    # New user message cancels pending retry and starts a new run
    (EngineState.PENDING_RETRY, Trigger.USER_MESSAGE): EngineState.RUNNING,
    # Pending retry auto-starts after current run completes or is interrupted.
    (EngineState.PENDING_RETRY, Trigger.RUN_COMPLETED): EngineState.RUNNING,
    (EngineState.PENDING_RETRY, Trigger.RUN_INTERRUPTED): EngineState.RUNNING,
    (EngineState.PENDING_RETRY, Trigger.RUN_FAILED): EngineState.FAILED,
    # Explicit retry from interrupted/failed
    (EngineState.INTERRUPTED, Trigger.RETRY_STARTED): EngineState.RUNNING,
    (EngineState.FAILED, Trigger.RETRY_STARTED): EngineState.RUNNING,
    # Restore persisted terminal history state after session load.  Normal
    # IDLE retry remains invalid; restore explicitly reconstructs the runtime
    # terminal state from saved markers before admitting retry.
    (EngineState.IDLE, Trigger.RESTORE_INTERRUPTED): EngineState.INTERRUPTED,
    (EngineState.IDLE, Trigger.RESTORE_FAILED): EngineState.FAILED,
    # Sub-agent pause / resolve.  The parent ``agent.run()`` is still in
    # flight when a sub-agent pauses, so these transition on top of the
    # "running-ish" states.  ``AWAITING_SUB_AGENTS`` is only driven by
    # the edge transitions (0 ↔ 1 paused); multi-pause / partial resolve
    # bookkeeping lives in ``AgentEngine`` via a paused-id set.
    (EngineState.RUNNING, Trigger.SUB_AGENT_PAUSED): EngineState.AWAITING_SUB_AGENTS,
    (EngineState.PENDING_RETRY, Trigger.SUB_AGENT_PAUSED): EngineState.AWAITING_SUB_AGENTS,
    (EngineState.AWAITING_SUB_AGENTS, Trigger.SUB_AGENT_RESOLVED): EngineState.RUNNING,
    # Deliberately no ``(AWAITING_SUB_AGENTS, USER_MESSAGE)`` entry: the
    # parent's tool call is still in flight, so a new user message is
    # injected into the running executor (see ``AgentEngine._on_user_message``
    # gated on ``EngineStateMachine.is_running``) rather than starting a
    # fresh run that would collide with the live parent task.
    # Parent run outcomes while awaiting sub-agents (e.g. user interrupt
    # cascades, or all sub-agents aborted and the parent's tool call
    # returned error → parent finished its run).
    (EngineState.AWAITING_SUB_AGENTS, Trigger.RUN_COMPLETED): EngineState.IDLE,
    (EngineState.AWAITING_SUB_AGENTS, Trigger.RUN_INTERRUPTED): EngineState.INTERRUPTED,
    (EngineState.AWAITING_SUB_AGENTS, Trigger.RUN_FAILED): EngineState.FAILED,
    # Shutdown is accepted from every state.  ``AgentEngine.shutdown``
    # cascade-aborts paused sub-agents and cancels the running executor
    # before the trigger fires, so in practice the FSM is usually already
    # in IDLE/INTERRUPTED/FAILED by the time SHUTDOWN arrives — but the
    # in-flight transitions below make the trigger a true no-refusal
    # operation rather than a silent no-op from RUNNING / PENDING_RETRY /
    # AWAITING_SUB_AGENTS.
    (EngineState.IDLE, Trigger.SHUTDOWN): EngineState.SHUTTING_DOWN,
    (EngineState.INTERRUPTED, Trigger.SHUTDOWN): EngineState.SHUTTING_DOWN,
    (EngineState.FAILED, Trigger.SHUTDOWN): EngineState.SHUTTING_DOWN,
    (EngineState.UNINITIALIZED, Trigger.SHUTDOWN): EngineState.SHUTTING_DOWN,
    (EngineState.RUNNING, Trigger.SHUTDOWN): EngineState.SHUTTING_DOWN,
    (EngineState.PENDING_RETRY, Trigger.SHUTDOWN): EngineState.SHUTTING_DOWN,
    (EngineState.AWAITING_SUB_AGENTS, Trigger.SHUTDOWN): EngineState.SHUTTING_DOWN,
}


class InvalidTransition(Exception):
    """Raised when a trigger is not valid for the current state."""

    def __init__(self, state: EngineState, trigger: Trigger) -> None:
        self.state = state
        self.trigger = trigger
        super().__init__(f"No transition from {state.name} via {trigger.name}")


class EngineStateMachine:
    """Lightweight state machine for AgentEngine lifecycle.

    Usage::

        fsm = EngineStateMachine()
        fsm.transition(Trigger.START)        # UNINITIALIZED → IDLE
        fsm.transition(Trigger.USER_MESSAGE)  # IDLE → RUNNING
    """

    def __init__(self) -> None:
        self._state = EngineState.UNINITIALIZED

    @property
    def state(self) -> EngineState:
        return self._state

    def transition(self, trigger: Trigger) -> EngineState:
        """Apply a trigger and return the new state.

        Raises ``InvalidTransition`` if the trigger is not valid for
        the current state.
        """
        key = (self._state, trigger)
        next_state = TRANSITIONS.get(key)
        if next_state is None:
            raise InvalidTransition(self._state, trigger)
        prev = self._state
        self._state = next_state
        logger.debug("FSM: %s -[%s]-> %s", prev.name, trigger.name, next_state.name)
        return next_state

    def try_transition(self, trigger: Trigger) -> EngineState | None:
        """Apply a trigger if valid, otherwise return ``None`` (no-op)."""
        key = (self._state, trigger)
        next_state = TRANSITIONS.get(key)
        if next_state is None:
            return None
        prev = self._state
        self._state = next_state
        logger.debug("FSM: %s -[%s]-> %s", prev.name, trigger.name, next_state.name)
        return next_state

    def is_accepting_messages(self) -> bool:
        """True when the engine can accept a new user message."""
        return self._state in (EngineState.IDLE, EngineState.INTERRUPTED, EngineState.FAILED)

    def is_running(self) -> bool:
        """True when the engine is executing.

        Includes ``PENDING_RETRY`` (retry queued while current run still
        finishing) and ``AWAITING_SUB_AGENTS`` (the parent's tool call is
        still awaiting a paused sub-agent's decision — the run is
        logically in-flight even though no tokens are streaming).
        """
        return self._state in (
            EngineState.RUNNING,
            EngineState.PENDING_RETRY,
            EngineState.AWAITING_SUB_AGENTS,
        )

    def is_awaiting_sub_agents(self) -> bool:
        """True iff at least one sub-agent is paused waiting on user input."""
        return self._state is EngineState.AWAITING_SUB_AGENTS

    def restore_terminal_state(self, *, failed: bool) -> EngineState | None:
        """Reconstruct a persisted terminal state after session restore."""
        trigger = Trigger.RESTORE_FAILED if failed else Trigger.RESTORE_INTERRUPTED
        return self.try_transition(trigger)

    def reset(self) -> None:
        """Reset to UNINITIALIZED (used when starting a fresh session)."""
        self._state = EngineState.UNINITIALIZED
