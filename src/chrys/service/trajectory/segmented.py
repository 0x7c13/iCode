# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Emit one base event whose unbounded fields continue in ``event.segment`` lines.

The base event is written first carrying ``segmented_fields[]`` (group id and
count per field), then every segment of every field in index order. The
segment lines are measured against the same envelope the writer produces, so
each one fits the line budget exactly as the writer will see it.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from chrys.foundation.trajectory.context import TrajectoryContext
from chrys.foundation.trajectory.envelope import (
    SEGMENT_EVENT_TYPE,
    EventDraft,
    Link,
    build_event,
    encode_event_line,
)
from chrys.foundation.trajectory.fingerprint import DOMAIN_SEGMENT_VALUE, keyed_fingerprint
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.segments import ENCODING_ARRAY_SLICE, SegmentPlan, plan_segments

logger = logging.getLogger(__name__)

# Writer-owned ids are all fixed-width, so a placeholder envelope measures
# exactly what the writer will encode (the sequence placeholder matches the
# writer's own ten-digit measuring slot).
_MEASURE_ID = "0" * 32
_MEASURE_SEQUENCE = 10**9


def measure_line(draft: EventDraft, *, session_id: str) -> int:
    """The encoded on-disk length *draft* will have once the writer addresses it."""
    event = build_event(
        draft,
        sequence=_MEASURE_SEQUENCE,
        runtime_id=_MEASURE_ID,
        coverage_id=_MEASURE_ID,
        session_id=session_id,
        branch_id=_MEASURE_ID,
    )
    return len(encode_event_line(event))


def _value_hasher(key: bytes | None):
    # No key (the writer has not activated yet) means no fingerprint: an
    # unkeyed digest of a short value can be guessed back, so the oversized
    # sentinel records the length alone.
    if key is None:
        return None
    return lambda raw: keyed_fingerprint(key, DOMAIN_SEGMENT_VALUE, raw)


def plan_segmented(
    context: TrajectoryContext,
    *,
    event_type: str,
    payload: Mapping[str, Any],
    array_fields: Mapping[str, Sequence[Any]],
    operation_id: str | None = None,
    parent_operation_id: str | None = None,
    measurements: Mapping[str, Mapping[str, Any]] | None = None,
    links: tuple[Link, ...] = (),
) -> tuple[EventDraft, list[EventDraft]]:
    """Build the base draft of *event_type* plus the ``event.segment`` drafts continuing it.

    ``array_fields`` maps a JSON pointer into the base event (``/payload/turn_numbers``)
    to the ordered entries that field would hold; the base payload never
    carries them inline. The base must be queued before its segments.
    """
    base = context.draft(
        event_type,
        operation_id=operation_id,
        parent_operation_id=parent_operation_id,
        payload=dict(payload),
        measurements=dict(measurements or {}),
        links=links,
    )
    session_id = context.session_id
    hasher = _value_hasher(context.sink.fingerprint_key)
    plans: list[SegmentPlan] = []
    for pointer, entries in array_fields.items():
        plans.append(
            plan_segments(
                parent_event_id=base.event_id,
                field_pointer=pointer,
                encoding=ENCODING_ARRAY_SLICE,
                entries=list(entries),
                measure=lambda seg_payload: measure_line(
                    # The same shape the segments below are emitted with: a
                    # probe missing the turn its event belongs to measures a
                    # shorter line than the writer will encode, and the packer
                    # fills the difference with entries that push it over.
                    EventDraft(
                        event_type=SEGMENT_EVENT_TYPE,
                        actor=context.actor,
                        turn_id=context.turn_id,
                        payload=seg_payload,
                    ),
                    session_id=session_id,
                ),
                hasher=hasher,
            )
        )
    base = replace(base, segmented_fields=tuple(plan.declaration for plan in plans))
    segments = [
        EventDraft(
            event_type=SEGMENT_EVENT_TYPE,
            event_id=new_analytics_id(),
            actor=context.actor,
            turn_id=context.turn_id,
            payload=seg_payload,
            occurred_at=base.occurred_at,
            monotonic_ns=base.monotonic_ns,
        )
        for plan in plans
        for seg_payload in plan.segment_payloads
    ]
    return base, segments


async def emit_segmented(context: TrajectoryContext, **kwargs: Any) -> bool:
    """Emit a segmented event (see :func:`plan_segmented`), awaiting each ack; ``False`` on failure."""
    pending: list[EventDraft] = []
    try:
        base, segments = plan_segmented(context, **kwargs)
        pending = [base, *segments]
        while pending:
            await context.sink.emit(pending[0])
            pending.pop(0)
    except Exception:
        logger.debug("Trajectory segmented emit failed for %s", kwargs.get("event_type"), exc_info=True)
        return False
    except BaseException:
        # Cancelled mid-group: the write being awaited is committed already,
        # so the rest has to follow it. A group that stops short reassembles
        # as a shorter field with nothing on the line to say it was cut.
        for draft in pending[1:]:
            with contextlib.suppress(Exception):
                context.sink.emit_soon(draft)
        raise
    return True


def emit_segmented_soon(context: TrajectoryContext, **kwargs: Any) -> bool:
    """Queue a segmented event in order without awaiting acks (synchronous producers)."""
    try:
        base, segments = plan_segmented(context, **kwargs)
        context.sink.emit_soon(base)
        for segment in segments:
            context.sink.emit_soon(segment)
    except Exception:
        logger.debug("Trajectory segmented emit failed for %s", kwargs.get("event_type"), exc_info=True)
        return False
    return True
