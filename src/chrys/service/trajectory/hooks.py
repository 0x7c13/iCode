# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``hook.operation.started`` / ``hook.operation.finished`` recording for the hook manager.

A hook run is itself a wait node on the timeline (a blocking hook holds the
turn; an async one still occupies a subprocess slot), so it gets its own
lifecycle pair and never a ``wait.*`` twin.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from chrys.foundation.trajectory.context import TrajectoryContext, current_trajectory
from chrys.foundation.trajectory.envelope import MeasurementSource, measurement
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id

logger = logging.getLogger(__name__)

TrajectoryContextProvider = Callable[[], TrajectoryContext | None]
"""Resolves the recording scope for hooks fired outside a model run (session/turn hooks)."""


class HookOutcome:
    """``hook.operation.finished.outcome``."""

    SUCCESS = "success"
    FAILED = "failed"
    LAUNCH_ERROR = "launch_error"
    TIMED_OUT = "timed_out"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"
    DETACHED = "detached"
    SPAWN_FAILED = "spawn_failed"


class HookOperationTrace:
    """One hook subprocess run under the ambient (or provided) trajectory scope."""

    __slots__ = (
        "_context",
        "_finished",
        "_lock",
        "_operation_id",
        "_start_committed",
        "_started_ns",
        "_target_operation_id",
    )

    def __init__(self, context: TrajectoryContext, *, target_operation_id: str | None) -> None:
        self._context = context
        self._operation_id = new_analytics_id()
        self._target_operation_id = target_operation_id
        self._started_ns = time.monotonic_ns()
        self._finished = False
        self._start_committed = False
        self._lock = threading.Lock()

    @classmethod
    def open(
        cls,
        *,
        target_operation_id: str | None = None,
        context: TrajectoryContext | None = None,
        provider: TrajectoryContextProvider | None = None,
    ) -> HookOperationTrace | None:
        """Bind to explicit, ambient, then provider scope; return ``None`` when unrecorded."""
        resolved = context or current_trajectory()
        if resolved is None and provider is not None:
            resolved = provider()
        if resolved is None:
            return None
        return cls(resolved, target_operation_id=target_operation_id)

    @property
    def operation_id(self) -> str:
        return self._operation_id

    @property
    def start_committed(self) -> bool:
        """Whether ``hook.operation.started`` acquired a log sequence."""
        return self._start_committed

    def _parent(self) -> str | None:
        return self._target_operation_id or self._context.innermost_model_operation_id

    async def started(
        self,
        *,
        hook_id: str,
        hook_event: str,
        execution_mode: str,
        detach: bool,
        delivery: str,
        scope: str = "turn",
        drain_scope: str | None = "turn",
    ) -> None:
        payload: dict[str, Any] = {
            # Keep schema-v1 readers compatible: unknown ``*_id`` fields are
            # analytics IDs to them, while this user-configured key is opaque.
            "hook_key": hook_id,
            "hook_event": hook_event,
            "execution_mode": execution_mode,
            "detach": detach,
            "delivery": delivery,
            "scope": scope,
        }
        if drain_scope is not None:
            payload["drain_scope"] = drain_scope
        if self._target_operation_id is not None:
            payload["target_operation_id"] = self._target_operation_id

        def _commit(_sequence: int) -> dict[str, Any]:
            # Runs where the writer takes the sequence and queues the line in
            # one locked step: reaching here is what makes this hook real to
            # the log, and nothing below may close a span the log never opened.
            with self._lock:
                if self._finished:
                    raise RuntimeError("Hook observation ended before operation start")
                self._start_committed = True
                try:
                    self._context.finalizers.add(self._abandon)
                except RuntimeError:
                    # Close won the race with a lazy-activation submission.
                    # Refuse a start that would come after settlement.
                    self._start_committed = False
                    raise
            return payload

        try:
            await self._context.sink.emit(
                self._context.draft(
                    EventType.HOOK_OPERATION_STARTED,
                    operation_id=self._operation_id,
                    parent_operation_id=self._parent(),
                    payload=payload,
                ),
                payload_factory=_commit,
            )
        except Exception:
            logger.debug("Trajectory hook.operation.started emit failed", exc_info=True)

    def _finished_draft(
        self,
        *,
        outcome: str,
        arguments_modified: bool,
        exit_code: int | None,
        timed_out: bool,
    ) -> Any:
        duration_ms = max(0, (time.monotonic_ns() - self._started_ns) // 1_000_000)
        payload: dict[str, Any] = {
            "outcome": outcome,
            "arguments_modified": arguments_modified,
            "duration_ms": duration_ms,
        }
        if exit_code is not None:
            payload["exit_code"] = exit_code
        if timed_out:
            payload["timed_out"] = True
        return self._context.draft(
            EventType.HOOK_OPERATION_FINISHED,
            operation_id=self._operation_id,
            parent_operation_id=self._parent(),
            payload=payload,
            measurements={"/payload/duration_ms": measurement(MeasurementSource.MONOTONIC_CLOCK, method_version=1)},
        )

    def _abandon(self) -> None:
        self.finished_soon(outcome=HookOutcome.ABANDONED)

    def _claim_finish(self) -> bool:
        # The lazy-activation thread can still be committing the start while
        # the event-loop task is cancelled or the recorder is closing.
        with self._lock:
            if self._finished:
                return False
            self._finished = True
            self._context.finalizers.discard(self._abandon)
            return self._start_committed

    async def finished(
        self,
        *,
        outcome: str,
        arguments_modified: bool = False,
        exit_code: int | None = None,
        timed_out: bool = False,
    ) -> None:
        if not self._claim_finish():
            return
        draft = self._finished_draft(
            outcome=outcome, arguments_modified=arguments_modified, exit_code=exit_code, timed_out=timed_out
        )
        try:
            await self._context.sink.emit(draft)
        except Exception:
            logger.debug("Trajectory hook.operation.finished emit failed", exc_info=True)

    def finished_soon(self, *, outcome: str) -> None:
        """Close without awaiting the ack (cancellation paths)."""
        if not self._claim_finish():
            # Cancellation before sequence assignment drops the hook; a
            # delayed submission then refuses the start instead of leaving
            # a span with no observer to finish it.
            return
        draft = self._finished_draft(outcome=outcome, arguments_modified=False, exit_code=None, timed_out=False)
        try:
            self._context.sink.emit_soon(draft)
        except Exception:
            logger.debug("Trajectory hook.operation.finished emit failed", exc_info=True)
