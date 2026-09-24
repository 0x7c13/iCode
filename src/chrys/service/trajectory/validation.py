# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``model.validation.finished`` and validation-retry recording for the response validator.

The validation middleware runs beneath the kernel loop, so the kernel's
:class:`ExchangeTrace` is the only handle it has on the exchange it judged.
An in-place re-send rolls that handle onto a new exchange operation
(:meth:`ExchangeTrace.reissue`) so every wire acquisition stays its own
exchange while the kernel keeps stamping the final, accepted one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from chrys.foundation.trajectory.context import TRAJECTORY_EXCHANGE_KWARG, ExchangeTrace
from chrys.foundation.trajectory.envelope import Link, LinkRelation
from chrys.foundation.trajectory.event_types import EventType, RetryMode, RetryReason, ValidationOutcome
from chrys.foundation.trajectory.ids import new_analytics_id

logger = logging.getLogger(__name__)


class ValidationTrace:
    """Validation facts for the exchanges one ``call_next`` cycle acquires."""

    __slots__ = ("_attempt_index", "_pending_next_id", "_trace")

    def __init__(self, trace: ExchangeTrace) -> None:
        self._trace = trace
        self._attempt_index = 0
        self._pending_next_id: str | None = None

    @classmethod
    def open(cls, kwargs: Mapping[str, Any]) -> ValidationTrace | None:
        """Pick the kernel's exchange trace out of the middleware context kwargs."""
        nested = kwargs.get("client_kwargs")
        trace = nested.get(TRAJECTORY_EXCHANGE_KWARG) if isinstance(nested, Mapping) else None
        if not isinstance(trace, ExchangeTrace):
            return None
        return cls(trace)

    @property
    def exchange_operation_id(self) -> str:
        return self._trace.operation_id

    async def finished(
        self,
        *,
        accepted: bool,
        reason_code: str | None = None,
        retryable: bool | None = None,
        gave_up: bool = False,
    ) -> None:
        """Record the verdict on the current exchange."""
        context = self._trace.context
        exchange_id = self._trace.operation_id
        payload: dict[str, Any] = {
            "outcome": ValidationOutcome.ACCEPTED if accepted else ValidationOutcome.REJECTED,
            "exchange_operation_id": exchange_id,
            "attempt_index": self._attempt_index,
        }
        if reason_code is not None:
            payload["reason_code"] = reason_code
        if retryable is not None:
            payload["retryable"] = retryable
        if gave_up:
            payload["gave_up"] = True
        try:
            await context.sink.emit(
                context.draft(
                    EventType.MODEL_VALIDATION_FINISHED,
                    operation_id=new_analytics_id(),
                    parent_operation_id=exchange_id,
                    payload=payload,
                    links=(Link(relation=LinkRelation.VALIDATES, target_operation_id=exchange_id),),
                )
            )
        except Exception:
            logger.debug("Trajectory model.validation.finished emit failed", exc_info=True)

    async def retry_scheduled(self, *, delay_seconds: float) -> None:
        """Record that the request will be re-sent in place after ``delay_seconds``."""
        context = self._trace.context
        next_id = new_analytics_id()
        self._pending_next_id = next_id
        try:
            await context.sink.emit(
                context.draft(
                    EventType.RETRY_SCHEDULED,
                    operation_id=next_id,
                    parent_operation_id=context.cycle_operation_id or context.run_operation_id,
                    payload={
                        "reason_code": RetryReason.VALIDATION_REJECTED,
                        "delay_ms": max(0, int(delay_seconds * 1000)),
                        "retry_mode": RetryMode.VALIDATION,
                        "previous_operation_id": self._trace.operation_id,
                        "committed_work_present": False,
                        "fallback_to_blocking": False,
                    },
                )
            )
        except Exception:
            logger.debug("Trajectory retry.scheduled emit failed", exc_info=True)

    async def retry_started(self) -> None:
        """Record the re-send and roll the exchange trace onto the new operation."""
        context = self._trace.context
        previous = self._trace.operation_id
        next_id = self._pending_next_id or new_analytics_id()
        self._pending_next_id = None
        try:
            await context.sink.emit(
                context.draft(
                    EventType.RETRY_STARTED,
                    operation_id=next_id,
                    parent_operation_id=context.cycle_operation_id or context.run_operation_id,
                    payload={
                        "retry_mode": RetryMode.VALIDATION,
                        "next_operation_id": next_id,
                        "previous_operation_id": previous,
                    },
                )
            )
        except Exception:
            logger.debug("Trajectory retry.started emit failed", exc_info=True)
        self._trace.reissue(next_id)
        self._attempt_index += 1
