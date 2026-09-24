# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``sub_agent.started`` / ``sub_agent.finished`` boundary recording.

The parent always records the boundary — status, duration, the invocation
and actor ids — whatever the child's own trace looks like. An in-process
child inlines its events into the parent writer under the child's actor
(:meth:`SubAgentTrace.child_context`); an external ACP child is
``boundary_only``: the child process's own execution detail (model runs and
tools) is never written to the parent log, while parent-recorded control-plane
operations about the child (approvals, input waits, and connection retries)
still appear under the child actor.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from chrys.foundation.trajectory.context import (
    TrajectoryContext,
    current_tool_operation_id,
    current_trajectory,
    sub_agent_actor,
)
from chrys.foundation.trajectory.envelope import Link, LinkRelation, MeasurementSource, measurement
from chrys.foundation.trajectory.event_types import EventType, TraceCoverage
from chrys.foundation.trajectory.ids import new_analytics_id

logger = logging.getLogger(__name__)


class SubAgentStatus:
    """``sub_agent.finished.status``."""

    OK = "ok"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SETUP_FAILED = "setup_failed"


class SubAgentTrace:
    """One sub-agent invocation seen from the parent."""

    __slots__ = (
        "_child_context",
        "_context",
        "_finished",
        "_invocation_id",
        "_operation_id",
        "_parent_tool_operation_id",
        "_started_ns",
        "_trace_coverage",
    )

    def __init__(
        self,
        context: TrajectoryContext,
        *,
        invocation_id: str,
        trace_coverage: str,
        parent_tool_operation_id: str | None,
    ) -> None:
        self._context = context
        self._invocation_id = invocation_id
        self._trace_coverage = trace_coverage
        self._parent_tool_operation_id = parent_tool_operation_id
        self._operation_id = new_analytics_id()
        self._started_ns = time.monotonic_ns()
        self._finished = False
        actor = sub_agent_actor(context.session_id, invocation_id)
        # The child's model cycles hang directly under the boundary: a
        # sub-agent is single-turn and has no service-side run of its own.
        # The parent's profile facts do not describe the child's exchanges.
        self._child_context = context.with_actor(actor).with_run(self._operation_id).with_exchange_facts({})

    @classmethod
    def open(cls, *, invocation_id: str, trace_coverage: str) -> SubAgentTrace | None:
        """Bind to the ambient scope of the delegating tool call; ``None`` when unrecorded."""
        context = current_trajectory()
        if context is None:
            return None
        return cls(
            context,
            invocation_id=invocation_id,
            trace_coverage=trace_coverage,
            parent_tool_operation_id=current_tool_operation_id(),
        )

    @property
    def operation_id(self) -> str:
        return self._operation_id

    @property
    def child_context(self) -> TrajectoryContext:
        """The scope an in-process child records under (its own actor, boundary as run)."""
        return self._child_context

    def _parent(self) -> str | None:
        return self._parent_tool_operation_id or self._context.innermost_model_operation_id

    def _links(self) -> tuple[Link, ...]:
        if self._parent_tool_operation_id is None:
            return ()
        return (Link(relation=LinkRelation.BOUNDARY_OF, target_operation_id=self._parent_tool_operation_id),)

    async def started(self, *, tool_name: str, agent_profile: str) -> None:
        self._child_context = self._child_context.with_exchange_facts({"agent_profile_id": agent_profile})
        payload: dict[str, Any] = {
            "invocation_id": self._invocation_id,
            "actor_id": self._child_context.actor.actor_id,
            "tool_name": tool_name,
            "agent_profile": agent_profile,
            "trace_coverage": self._trace_coverage,
        }
        if self._parent_tool_operation_id is not None:
            payload["parent_tool_operation_id"] = self._parent_tool_operation_id
        try:
            await self._context.sink.emit(
                self._context.draft(
                    EventType.SUB_AGENT_STARTED,
                    operation_id=self._operation_id,
                    parent_operation_id=self._parent(),
                    payload=payload,
                    links=self._links(),
                )
            )
        except Exception:
            logger.debug("Trajectory sub_agent.started emit failed", exc_info=True)

    def _finished_draft(self, *, status: str, failure_reason_code: str | None) -> Any:
        payload: dict[str, Any] = {
            "invocation_id": self._invocation_id,
            "status": status,
            "duration_ms": max(0, (time.monotonic_ns() - self._started_ns) // 1_000_000),
        }
        if failure_reason_code is not None:
            payload["failure_reason_code"] = failure_reason_code
        return self._context.draft(
            EventType.SUB_AGENT_FINISHED,
            operation_id=self._operation_id,
            parent_operation_id=self._parent(),
            payload=payload,
            links=self._links(),
            measurements={"/payload/duration_ms": measurement(MeasurementSource.MONOTONIC_CLOCK, method_version=1)},
        )

    async def finished(self, *, status: str, failure_reason_code: str | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            await self._context.sink.emit(self._finished_draft(status=status, failure_reason_code=failure_reason_code))
        except Exception:
            logger.debug("Trajectory sub_agent.finished emit failed", exc_info=True)

    def finished_soon(self, *, status: str, failure_reason_code: str | None = None) -> None:
        """Close without awaiting the ack (cancellation paths)."""
        if self._finished:
            return
        self._finished = True
        try:
            self._context.sink.emit_soon(self._finished_draft(status=status, failure_reason_code=failure_reason_code))
        except Exception:
            logger.debug("Trajectory sub_agent.finished emit failed", exc_info=True)


__all__ = ["SubAgentStatus", "SubAgentTrace", "TraceCoverage"]
