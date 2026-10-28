# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Turn-specific recovery input and injection state, borrowing an ExecutionLease."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from chrys.foundation.models.turns import UserMessageKind
from chrys.orchestration.engine.execution import ExecutionLease, RunTaskDrainOutcome

if TYPE_CHECKING:
    from chrys.orchestration.engine.execution import CurrentRunInjectionWindow, CurrentRunScope
    from chrys.orchestration.engine.run.bindings import TurnBindings
    from chrys.orchestration.invoker.kernel import KernelConversation
    from chrys.service.agent_middleware.system_reminder import CurrentRunReminderTarget, SystemReminderMiddleware
    from chrys.service.hooks.operation import HookOperation


@dataclass
class CurrentTurnInput:
    """Current prompt data used by crash-recovery checkpoint writes.

    ``kind`` records how a recovery checkpoint must re-create the message
    (``"opener"`` unflagged, ``"injected"`` flagged mid-turn input) — see
    :func:`chrys.foundation.models.turns.user_text_matches`. Empty-input
    resumes clear this field; opener replay registers the popped anchor.
    """

    text: str = ""
    contents: list[Any] | None = None
    created_at: datetime | str | None = None
    kind: UserMessageKind = "opener"


@dataclass(frozen=True)
class ActiveInjectionTarget:
    """Operation-local target captured before awaited active-injection work."""

    route: Literal["fsm_active", "executor_fallback"]
    session_id: str | None
    session_generation: int
    build_generation: int
    current_run_scope: CurrentRunScope
    run_task: asyncio.Task[None]
    bindings: TurnBindings
    conversation: KernelConversation
    reminder_middleware: SystemReminderMiddleware
    reminder_target: CurrentRunReminderTarget
    injection_window: CurrentRunInjectionWindow
    trajectory_turn_id: str | None = None


class PendingInputOperations:
    """The user inputs whose ``user_prompt_submit`` check is in flight, by the lifetimes that can end them.

    Every input entry (a new turn's message, an injected message, a retry
    note) tracks its operation here from its check to its commit, adding it
    before the check and discarding it where the entry ends, so each lifetime
    signal ends every input it outlives in one place: the user withdrawing a
    queued message or interrupting, and the session or build being replaced. A confirmation
    the input's hooks wait on ends with it.
    """

    def __init__(self) -> None:
        self._tracked: dict[str, tuple[HookOperation, str | None]] = {}

    def add(self, operation: HookOperation, *, injection_id: str | None = None) -> None:
        """Track *operation*; *injection_id* is the message the user can withdraw."""
        self._tracked[operation.operation_id] = (operation, injection_id)

    def discard(self, operation: HookOperation | None) -> None:
        """Stop tracking *operation* (None: nothing was tracked)."""
        if operation is not None:
            self._tracked.pop(operation.operation_id, None)

    def withdraw(self, injection_id: str) -> None:
        """The user withdrew message *injection_id*: end its input."""
        for operation, tracked_id in list(self._tracked.values()):
            if tracked_id == injection_id:
                operation.invalidate("withdrawn")

    def invalidate_all(self, reason: str) -> None:
        """End every tracked input: the user interrupted, or the session or build it belonged to is gone."""
        for operation, _ in list(self._tracked.values()):
            operation.invalidate(reason)


@dataclass
class TurnRuntimeState:
    """Turn product state; lease owns the unique task and all lifetime fences."""

    lease: ExecutionLease = field(default_factory=ExecutionLease)
    current_input: CurrentTurnInput = field(default_factory=CurrentTurnInput)
    pass_started_at: datetime | str | None = None
    """When the user asked for the running pass: sent its message, or pressed Continue or Retry.

    Its turn marker keeps it.
    """

    paused_sub_agents: set[str] = field(default_factory=set)
    shutdown_used_cancel_fallback: bool = False

    history_start_index: int = 0

    cancelled_injection_ids: set[str] = field(default_factory=set)

    inflight_injection_ids: dict[str, int] = field(default_factory=dict)

    input_operations: PendingInputOperations = field(default_factory=PendingInputOperations)

    async def drain_for_boundary(self) -> RunTaskDrainOutcome:
        """Observe the active run-task chain for a rebuild/session boundary."""
        return await self.lease.observe_run_task_chain(propagate_inner_cancel=False)

    def begin_inflight_injection(self, injection_id: str | None) -> None:
        """Track one in-flight admission of a frontend-identified injection.

        While an id is tracked, a user cancel for it is recorded as a mark
        (:meth:`mark_injection_cancelled`) that the admission observes at its
        cancellation checkpoints. No-op for id-less submits.
        """
        if not injection_id:
            return
        self.inflight_injection_ids[injection_id] = self.inflight_injection_ids.get(injection_id, 0) + 1

    def finish_inflight_injection(self, injection_id: str | None) -> None:
        """Stop tracking one in-flight admission for *injection_id*.

        When the last admission for the id exits, any cancel mark it did not
        observe (e.g. the admission aborted for an unrelated reason first) is
        dropped with it — nothing can observe the mark afterwards.
        """
        if not injection_id:
            return
        count = self.inflight_injection_ids.get(injection_id, 0)
        if count > 1:
            self.inflight_injection_ids[injection_id] = count - 1
            return
        self.inflight_injection_ids.pop(injection_id, None)
        self.cancelled_injection_ids.discard(injection_id)

    def is_injection_inflight(self, injection_id: str) -> bool:
        """Return whether an admission for *injection_id* is currently in flight."""
        return injection_id in self.inflight_injection_ids

    def mark_injection_cancelled(self, injection_id: str) -> None:
        """Record a user cancel for an injection whose admission is in flight.

        Callers must only mark ids that something can still observe (an
        in-flight admission — see :meth:`is_injection_inflight`); the mark is
        consumed at the admission's next cancellation checkpoint and dropped
        at admission end otherwise, so it can never outlive its injection.
        """
        self.cancelled_injection_ids.add(injection_id)
        self.input_operations.withdraw(injection_id)

    def is_injection_cancelled(self, injection_id: str | None) -> bool:
        """Return whether *injection_id* carries an unconsumed cancel mark."""
        return injection_id is not None and injection_id in self.cancelled_injection_ids

    def discard_injection_cancellation(self, injection_id: str | None) -> bool:
        """Consume the cancel mark for *injection_id*; returns whether it was set."""
        if injection_id is None or injection_id not in self.cancelled_injection_ids:
            return False
        self.cancelled_injection_ids.discard(injection_id)
        return True

    def set_current_input(
        self,
        text: str,
        contents: list[Any] | None,
        created_at: datetime | str | None,
        kind: UserMessageKind = "opener",
    ) -> None:
        """Set current prompt data for recovery checkpointing."""
        self.current_input = CurrentTurnInput(text=text, contents=contents, created_at=created_at, kind=kind)

    def clear_current_input(self) -> None:
        """Clear current prompt data after run finalization."""
        self.current_input = CurrentTurnInput()

    def reset_after_session_shutdown(self, *, prompt_admission_owner: str | None = None) -> CurrentRunScope | None:
        """Reset Turn-only data after the execution owner has drained and reset."""
        old_scope = self.lease.reset_after_session_shutdown(prompt_admission_owner=prompt_admission_owner)
        self.current_input = CurrentTurnInput()
        self.pass_started_at = None
        self.history_start_index = 0
        self.cancelled_injection_ids.clear()
        self.inflight_injection_ids.clear()
        self.input_operations.invalidate_all("session_ended")
        return old_scope
