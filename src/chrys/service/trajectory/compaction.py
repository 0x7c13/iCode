# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``compaction.started`` / ``compaction.phase.finished`` / ``compaction.finished`` recording.

``compaction.skipped`` is the marker for usage past the trigger that started no
pass, so a context that fills up without compacting says why.

One run covers one triggered compaction pass (or one forced compression);
its phases report what they freed, with ``turn_numbers``, ``tool_names``, and
Phase-4 ``consumed_item_ids`` always continued as ``event.segment`` lines so
no phase line can outgrow the budget however many groups it touched. The same
Phase-4 event names the already-reserved successor exchange.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from contextvars import ContextVar, Token
from typing import Any

from chrys.foundation.trajectory.context import TrajectoryContext, current_trajectory
from chrys.foundation.trajectory.envelope import MeasurementSource, measurement
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.service.trajectory.segmented import emit_segmented, plan_segmented

logger = logging.getLogger(__name__)

TOKEN_MEASUREMENT_SOURCE = MeasurementSource.LOCAL_TOKENIZER
"""Every token figure here is the local included-token estimate, never a provider count."""

_CURRENT_COMPACTION_OPERATION: ContextVar[str | None] = ContextVar(
    "chrys_trajectory_compaction_operation",
    default=None,
)


def bind_compaction_operation(operation_id: str | None) -> Token[str | None]:
    """Bind the compaction container for nested LAST_WORDS retry markers."""
    return _CURRENT_COMPACTION_OPERATION.set(operation_id)


def reset_compaction_operation(token: Token[str | None]) -> None:
    _CURRENT_COMPACTION_OPERATION.reset(token)


def current_compaction_operation_id() -> str | None:
    return _CURRENT_COMPACTION_OPERATION.get()


def _token_measurements(*pointers: str, source: str = TOKEN_MEASUREMENT_SOURCE) -> dict[str, Any]:
    token = measurement(source, method_version=1)
    return dict.fromkeys(pointers, token)


def _before_after_measurements(source: str) -> dict[str, Any]:
    """Provenance for one phase's token pair.

    Only the before-figure can come from the provider. Every after-figure is
    the local estimate of what the fold removed subtracted from it, so it is
    labelled for the estimator it depends on however the run started.
    """
    return {
        **_token_measurements("/payload/tokens_before", source=source),
        **_token_measurements("/payload/tokens_after", source=TOKEN_MEASUREMENT_SOURCE),
    }


def record_compaction_skipped(
    *,
    reason_code: str,
    estimated_input_tokens: int,
    trigger_tokens: int,
    max_context_tokens: int,
) -> None:
    """Queue ``compaction.skipped`` under the ambient scope; recording never blocks the request."""
    context = current_trajectory()
    if context is None:
        return
    try:
        context.sink.emit_soon(
            context.draft(
                EventType.COMPACTION_SKIPPED,
                parent_operation_id=context.innermost_model_operation_id,
                payload={
                    "reason_code": reason_code,
                    "estimated_input_tokens": estimated_input_tokens,
                    "trigger_tokens": trigger_tokens,
                    "max_context_tokens": max_context_tokens,
                },
                measurements=_token_measurements("/payload/estimated_input_tokens"),
            )
        )
    except Exception:
        logger.debug("Trajectory compaction.skipped emit failed", exc_info=True)


class CompactionRunTrace:
    """One compaction run under the ambient trajectory scope."""

    __slots__ = (
        "_context",
        "_finished",
        "_last_words_generated",
        "_next_exchange_operation_id",
        "_phase_started_ns",
        "_run_id",
        "_started_ns",
        "_token_source",
        "_tokens_after",
        "_tokens_before",
    )

    def __init__(
        self,
        context: TrajectoryContext,
        *,
        token_source: str = TOKEN_MEASUREMENT_SOURCE,
        next_exchange_operation_id: str | None = None,
    ) -> None:
        self._context = context
        self._run_id = new_analytics_id()
        self._started_ns = time.monotonic_ns()
        self._phase_started_ns = self._started_ns
        self._last_words_generated = False
        self._next_exchange_operation_id = next_exchange_operation_id
        self._finished = False
        self._token_source = token_source
        self._tokens_before = 0
        self._tokens_after = 0

    @classmethod
    def open(
        cls,
        *,
        token_source: str = TOKEN_MEASUREMENT_SOURCE,
        next_exchange_operation_id: str | None = None,
    ) -> CompactionRunTrace | None:
        """Bind to the ambient scope; *token_source* labels every token figure of the run."""
        context = current_trajectory()
        if context is None:
            return None
        return cls(
            context,
            token_source=token_source,
            next_exchange_operation_id=next_exchange_operation_id,
        )

    @classmethod
    async def record_compression(
        cls,
        *,
        trigger: str,
        compressed_context_id: str,
        messages_freed: int,
        tokens_before: int,
        tokens_after: int,
        token_source: str = TOKEN_MEASUREMENT_SOURCE,
    ) -> None:
        """Record a standalone cross-turn fold (agent-requested or forced) as one run."""
        run = cls.open(token_source=token_source)
        if run is None:
            return
        try:
            await run.started(trigger=trigger, tokens_before=tokens_before)
            await run.finished(
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                compressed_context_id=compressed_context_id,
                messages_freed=messages_freed,
            )
        except BaseException:
            # Interrupted between the two markers: the start line is already
            # committed, so the terminal one has to follow it — with the fold's
            # real figures, which the caller already knows.
            run.finished_soon(
                tokens_after=tokens_after,
                compressed_context_id=compressed_context_id,
                messages_freed=messages_freed,
            )
            raise

    @property
    def run_id(self) -> str:
        return self._run_id

    def _parent(self) -> str | None:
        return self._context.innermost_model_operation_id

    async def started(self, *, trigger: str, tokens_before: int) -> None:
        self._tokens_before = tokens_before
        self._tokens_after = tokens_before
        try:
            await self._context.sink.emit(
                self._context.draft(
                    EventType.COMPACTION_STARTED,
                    operation_id=self._run_id,
                    parent_operation_id=self._parent(),
                    payload={
                        "compaction_run_id": self._run_id,
                        "trigger": trigger,
                        "tokens_before": tokens_before,
                        "token_measurement_source": self._token_source,
                    },
                    measurements=_token_measurements("/payload/tokens_before", source=self._token_source),
                )
            )
        except Exception:
            logger.debug("Trajectory compaction.started emit failed", exc_info=True)

    def _phase_emit_args(
        self,
        *,
        phase: str,
        groups_compacted: int,
        turn_numbers: Sequence[int],
        tool_names: Sequence[str],
        tokens_before: int,
        tokens_after: int,
        last_words_generated: bool,
        consumed_item_ids: Sequence[str] | None,
        messages_freed: int | None,
        compressed_context_id: str | None,
    ) -> dict[str, Any]:
        """The segmented-emit arguments for one finished phase.

        Closing the phase is what advances the run's phase clock and its
        running after-figure, so this is called once per phase whichever of
        the two emit paths carries it.
        """
        now = time.monotonic_ns()
        duration_ms = max(0, (now - self._phase_started_ns) // 1_000_000)
        self._phase_started_ns = now
        self._tokens_after = tokens_after
        self._last_words_generated = self._last_words_generated or last_words_generated
        measurements = _before_after_measurements(self._token_source)
        measurements["/payload/duration_ms"] = measurement(MeasurementSource.MONOTONIC_CLOCK, method_version=1)
        payload: dict[str, Any] = {
            "compaction_run_id": self._run_id,
            "phase": phase,
            "groups_compacted": groups_compacted,
            "duration_ms": duration_ms,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "last_words_generated": last_words_generated,
        }
        if messages_freed is not None:
            payload["messages_freed"] = messages_freed
        if compressed_context_id is not None:
            payload["compressed_context_id"] = compressed_context_id
        if phase == "phase4" and self._next_exchange_operation_id is not None:
            payload["next_exchange_operation_id"] = self._next_exchange_operation_id
        array_fields = {
            "/payload/turn_numbers": list(turn_numbers),
            "/payload/tool_names": list(tool_names),
        }
        if phase == "phase4":
            array_fields["/payload/consumed_item_ids"] = list(consumed_item_ids or ())
        return {
            "event_type": EventType.COMPACTION_PHASE_FINISHED,
            "operation_id": new_analytics_id(),
            "parent_operation_id": self._run_id,
            "payload": payload,
            "array_fields": array_fields,
            "measurements": measurements,
        }

    async def phase_finished(
        self,
        *,
        phase: str,
        groups_compacted: int,
        turn_numbers: Sequence[int],
        tool_names: Sequence[str],
        tokens_before: int,
        tokens_after: int,
        last_words_generated: bool = False,
        consumed_item_ids: Sequence[str] | None = None,
        messages_freed: int | None = None,
        compressed_context_id: str | None = None,
    ) -> None:
        """Record one finished phase; folds also name the block they produced."""
        await emit_segmented(
            self._context,
            **self._phase_emit_args(
                phase=phase,
                groups_compacted=groups_compacted,
                turn_numbers=turn_numbers,
                tool_names=tool_names,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                last_words_generated=last_words_generated,
                consumed_item_ids=consumed_item_ids,
                messages_freed=messages_freed,
                compressed_context_id=compressed_context_id,
            ),
        )

    def phase_finished_soon(
        self,
        *,
        phase: str,
        groups_compacted: int,
        turn_numbers: Sequence[int],
        tool_names: Sequence[str],
        tokens_before: int,
        tokens_after: int,
        last_words_generated: bool = False,
        consumed_item_ids: Sequence[str] | None = None,
        messages_freed: int | None = None,
        compressed_context_id: str | None = None,
    ) -> None:
        """Record a finished phase without awaiting the acks (cancellation paths).

        A phase whose delivery is deferred to a task would be written after
        the interrupted pass had already closed its run.
        """
        try:
            base, segments = plan_segmented(
                self._context,
                **self._phase_emit_args(
                    phase=phase,
                    groups_compacted=groups_compacted,
                    turn_numbers=turn_numbers,
                    tool_names=tool_names,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    last_words_generated=last_words_generated,
                    consumed_item_ids=consumed_item_ids,
                    messages_freed=messages_freed,
                    compressed_context_id=compressed_context_id,
                ),
            )
            for draft in (base, *segments):
                self._context.sink.emit_soon(draft)
        except Exception:
            logger.debug("Trajectory compaction.phase.finished emit failed", exc_info=True)

    def _finished_draft(
        self,
        *,
        tokens_before: int,
        tokens_after: int,
        compressed_context_id: str | None,
        messages_freed: int | None,
    ) -> Any:
        payload: dict[str, Any] = {
            "compaction_run_id": self._run_id,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "duration_ms": max(0, (time.monotonic_ns() - self._started_ns) // 1_000_000),
            "last_words_generated": self._last_words_generated,
        }
        if compressed_context_id is not None:
            payload["compressed_context_id"] = compressed_context_id
        if messages_freed is not None:
            payload["messages_freed"] = messages_freed
        measurements = _before_after_measurements(self._token_source)
        measurements["/payload/duration_ms"] = measurement(MeasurementSource.MONOTONIC_CLOCK, method_version=1)
        return self._context.draft(
            EventType.COMPACTION_FINISHED,
            operation_id=self._run_id,
            parent_operation_id=self._parent(),
            payload=payload,
            measurements=measurements,
        )

    async def finished(
        self,
        *,
        tokens_before: int,
        tokens_after: int,
        compressed_context_id: str | None = None,
        messages_freed: int | None = None,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            await self._context.sink.emit(
                self._finished_draft(
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    compressed_context_id=compressed_context_id,
                    messages_freed=messages_freed,
                )
            )
        except Exception:
            logger.debug("Trajectory compaction.finished emit failed", exc_info=True)

    def finished_soon(
        self,
        *,
        tokens_after: int | None = None,
        compressed_context_id: str | None = None,
        messages_freed: int | None = None,
    ) -> None:
        """Close an interrupted run without awaiting the ack (cancellation paths).

        Left to itself the run reports what it measured — its own before-figure
        and the last phase's after-figure — so a pass cut short names the
        reduction it actually reached instead of a figure from another pass. A
        caller that already knows the outcome passes it instead.
        """
        if self._finished:
            return
        self._finished = True
        try:
            self._context.sink.emit_soon(
                self._finished_draft(
                    tokens_before=self._tokens_before,
                    tokens_after=self._tokens_after if tokens_after is None else tokens_after,
                    compressed_context_id=compressed_context_id,
                    messages_freed=messages_freed,
                )
            )
        except Exception:
            logger.debug("Trajectory compaction.finished emit failed", exc_info=True)
