# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Best-effort trajectory markers around service-owned retry backoff."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from chrys.foundation.trajectory.context import TrajectoryContext, current_trajectory
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id

logger = logging.getLogger(__name__)


class RetryBackoffTrace:
    """One ``retry.scheduled`` → ``retry.started`` backoff interval."""

    __slots__ = ("_context", "_finished", "_operation_id", "_parent_operation_id", "_retry_mode", "_scheduled")

    def __init__(self, context: TrajectoryContext, *, parent_operation_id: str | None, retry_mode: str) -> None:
        self._context = context
        self._parent_operation_id = parent_operation_id
        self._retry_mode = retry_mode
        self._operation_id = new_analytics_id()
        self._scheduled = False
        self._finished = False

    @classmethod
    def open(
        cls,
        *,
        parent_operation_id: str | None,
        retry_mode: str,
        context: TrajectoryContext | None = None,
    ) -> RetryBackoffTrace | None:
        resolved = context or current_trajectory()
        if resolved is None:
            return None
        return cls(resolved, parent_operation_id=parent_operation_id, retry_mode=retry_mode)

    async def scheduled(
        self,
        *,
        reason_code: str,
        delay_seconds: float,
        committed_work_present: bool = False,
    ) -> None:
        payload = {
            "reason_code": reason_code,
            "delay_ms": max(0, int(delay_seconds * 1000)),
            "retry_mode": self._retry_mode,
            "committed_work_present": committed_work_present,
        }

        def _commit(_sequence: int) -> Mapping[str, Any]:
            self._scheduled = True
            return payload

        try:
            await self._context.sink.emit(
                self._context.draft(
                    EventType.RETRY_SCHEDULED,
                    operation_id=self._operation_id,
                    parent_operation_id=self._parent_operation_id,
                    payload=payload,
                ),
                payload_factory=_commit,
            )
        except Exception:
            logger.debug("Trajectory retry.scheduled emit failed", exc_info=True)

    def _started_draft(self) -> Any:
        return self._context.draft(
            EventType.RETRY_STARTED,
            operation_id=self._operation_id,
            parent_operation_id=self._parent_operation_id,
            payload={"retry_mode": self._retry_mode},
        )

    async def started(self) -> None:
        if self._finished:
            return
        self._finished = True
        if not self._scheduled:
            return
        try:
            await self._context.sink.emit(self._started_draft())
        except Exception:
            logger.debug("Trajectory retry.started emit failed", exc_info=True)


__all__ = ["RetryBackoffTrace"]
