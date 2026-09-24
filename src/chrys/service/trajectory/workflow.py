# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session trajectory spans for a workflow run and each activation attempt.

The runner supplies scheduler states, including failures entering retry waits.
Values, prompts, source paths and diagnostic prose stay in the workflow store.
Cancellation terminals wait for the runner's drain so child agent operations
finish inside their node span. Recorder failures never affect execution.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from chrys.foundation.trajectory.context import TrajectoryContext
from chrys.foundation.trajectory.envelope import EventDraft, MeasurementSource, measurement
from chrys.foundation.trajectory.event_types import EventType, WorkflowNodeOutcome
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.service.workflows.scheduler import ActivationState, NodeStateChanged

logger = logging.getLogger(__name__)


class _Span:
    """An opening owns its terminal only once the writer commits its sequence."""

    def __init__(
        self,
        context: TrajectoryContext,
        started_type: str,
        finished_type: str,
        payload: dict[str, Any],
        *,
        operation_id: str | None = None,
    ) -> None:
        self.context = context
        self.operation_id = operation_id or new_analytics_id()
        self._started_type = started_type
        self._finished_type = finished_type
        self._payload = payload
        self._started_ns = time.monotonic_ns()
        self._committed = False
        self._finished = False
        self._lock = threading.Lock()

    def _draft(self, event_type: str, payload: dict[str, Any]) -> EventDraft:
        return self.context.draft(
            event_type,
            operation_id=self.operation_id,
            parent_operation_id=self.context.innermost_model_operation_id,
            payload={**self._payload, **payload},
            measurements={"/payload/duration_ms": measurement(MeasurementSource.MONOTONIC_CLOCK, method_version=1)}
            if event_type == self._finished_type
            else {},
        )

    async def started(self) -> None:
        def commit(_sequence: int) -> dict[str, Any]:
            with self._lock:
                if self._finished:
                    raise RuntimeError("Workflow span closed before its start committed")
                self.context.finalizers.add(self.abandon)
                self._committed = True
            return self._payload

        try:
            await self.context.sink.emit(self._draft(self._started_type, {}), payload_factory=commit)
        except Exception:
            logger.debug("Trajectory workflow start emit failed", exc_info=True)

    def finished(self, outcome: str) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self.context.finalizers.discard(self.abandon)
            if not self._committed:
                return
        try:
            self.context.sink.emit_soon(
                self._draft(
                    self._finished_type,
                    {"outcome": outcome, "duration_ms": max(0, (time.monotonic_ns() - self._started_ns) // 1_000_000)},
                )
            )
        except Exception:
            logger.debug("Trajectory workflow finish emit failed", exc_info=True)

    def abandon(self) -> None:
        self.finished(WorkflowNodeOutcome.ABANDONED)


class WorkflowTrace:
    """One run's recording state; attempts are keyed by activation and attempt, never just node name."""

    def __init__(self, context: TrajectoryContext, *, run_id: str, workflow_id: str) -> None:
        self._run = _Span(
            context,
            EventType.WORKFLOW_RUN_STARTED,
            EventType.WORKFLOW_RUN_FINISHED,
            {"workflow": workflow_id, "run_id": run_id},
            operation_id=run_id,
        )
        self._nodes: dict[tuple[str, int], _Span] = {}
        self._cancelled: set[tuple[str, int]] = set()

    async def started(self) -> None:
        await self._run.started()

    async def node_state(self, decision: NodeStateChanged, *, kind: str) -> None:
        ref, state = decision.ref, decision.state
        key = (ref.activation_id, ref.attempt)
        span = self._nodes.get(key)
        if span is None:
            span = _Span(
                self._run.context.with_run(self._run.operation_id),
                EventType.WORKFLOW_NODE_STARTED,
                EventType.WORKFLOW_NODE_FINISHED,
                {
                    "run_id": ref.run_id,
                    "node": ref.node_id,
                    "activation": ref.activation_id,
                    "attempt": ref.attempt,
                    "kind": kind,
                },
            )
            self._nodes[key] = span
            await span.started()
        if state is ActivationState.CANCELLED:
            self._cancelled.add(key)
        elif state in (ActivationState.RETRYING, ActivationState.AWAITING_RETRY, ActivationState.FAILED):
            span.finished(WorkflowNodeOutcome.FAILED)
        elif state is ActivationState.COMPLETED:
            span.finished(WorkflowNodeOutcome.COMPLETED)
        elif state is ActivationState.SKIPPED:
            span.finished(WorkflowNodeOutcome.SKIPPED)

    def node_context(self, activation: str, attempt: int) -> TrajectoryContext | None:
        span = self._nodes.get((activation, attempt))
        return span.context.with_run(span.operation_id) if span is not None else None

    def finished(self, outcome: str) -> None:
        for key, span in self._nodes.items():
            span.finished(WorkflowNodeOutcome.CANCELLED if key in self._cancelled else WorkflowNodeOutcome.ABANDONED)
        self._run.finished(outcome)
