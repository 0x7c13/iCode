# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``context.revision.recorded`` for every wire request of a recorded actor.

The wire client calls :func:`record_context_revision` with the exact message
list it is about to send. Membership (ordered item ids with repeats) is
computed against the actor's chain and written as a checkpoint or a delta,
with the entries continued as ``event.segment`` lines under
``/payload/refs`` (see :mod:`chrys.foundation.trajectory.revisions`).
Token buckets come from the compaction annotations already on the
messages — no tokenizer runs on the request path; items without an
annotation are counted in ``untokenized_item_count``. An item that
carries no analytics id cannot enter the membership at all; it is
counted in ``unidentified_item_count`` so a reader never mistakes a
partial membership for the whole request.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.trajectory.context import TrajectoryContext
from chrys.foundation.trajectory.envelope import MeasurementSource, measurement
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.metadata import read_analytics_item_id
from chrys.foundation.trajectory.revisions import membership_hash, membership_of
from chrys.kernel import Message
from chrys.kernel.compaction import TOKEN_ESTIMATOR_VERSION, _token_count
from chrys.service.trajectory.segmented import emit_segmented_soon

logger = logging.getLogger(__name__)

REFS_POINTER = "/payload/refs"
TOKENIZER_FINGERPRINT = f"chrys-estimator-v{TOKEN_ESTIMATOR_VERSION}"
"""Identifies the local estimator the token buckets come from."""

BUCKET_SYSTEM = "system"
BUCKET_LIVE_HISTORY = "live_history"
BUCKET_COMPRESSED_SUMMARIES = "compressed_summaries"
BUCKET_TOOL_RESULTS = "tool_results"
BUCKET_CURRENT_USER = "current_user"
_BUCKETS = (BUCKET_SYSTEM, BUCKET_LIVE_HISTORY, BUCKET_COMPRESSED_SUMMARIES, BUCKET_TOOL_RESULTS, BUCKET_CURRENT_USER)


def _bucket_of(message: Message, *, is_current_user: bool) -> str:
    if message.role == "system":
        return BUCKET_SYSTEM
    if message.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY:
        return BUCKET_COMPRESSED_SUMMARIES
    if any(content.type == "function_result" for content in message.contents):
        return BUCKET_TOOL_RESULTS
    if is_current_user:
        return BUCKET_CURRENT_USER
    return BUCKET_LIVE_HISTORY


def _current_user_message(messages: Sequence[Message]) -> Message | None:
    """The last user message that is not a tool-result carrier — the turn's own input."""
    for message in reversed(messages):
        if message.role == "user" and not any(content.type == "function_result" for content in message.contents):
            return message
    return None


def record_context_revision(context: TrajectoryContext, messages: Sequence[Message]) -> str | None:
    """Queue the revision *messages* form for *context*'s actor; return its id (``None`` if nothing recorded)."""
    try:
        return _record(context, messages)
    except Exception:
        logger.debug("Trajectory context.revision.recorded failed", exc_info=True)
        return None


def _record(context: TrajectoryContext, messages: Sequence[Message]) -> str | None:
    actor_id = context.actor.actor_id
    if actor_id is None:
        return None
    item_ids: list[str] = []
    buckets = dict.fromkeys(_BUCKETS, 0)
    untokenized = 0
    unidentified = 0
    current_user = _current_user_message(messages)
    for message in messages:
        item_id = read_analytics_item_id(message.additional_properties)
        if item_id is not None:
            item_ids.append(item_id)
        else:
            # Its tokens still went on the wire, so the buckets count it; the
            # membership cannot name it, and says so rather than pretending
            # the request held one item fewer.
            unidentified += 1
        tokens = _token_count(message)
        if tokens is None:
            untokenized += 1
        else:
            buckets[_bucket_of(message, is_current_user=message is current_user)] += tokens
    membership = membership_of(item_ids)
    chain = context.revisions.chain(actor_id)
    plan = chain.plan(membership)
    payload: dict[str, Any] = {
        "revision_id": plan.revision_id,
        "parent_revision_id": plan.parent_revision_id,
        "membership_hash": membership_hash(context.sink.fingerprint_key, membership),
        "token_buckets": buckets,
        "tokenizer_fingerprint": TOKENIZER_FINGERPRINT,
        "item_count": plan.item_count,
        "untokenized_item_count": untokenized,
        "unidentified_item_count": unidentified,
        "is_checkpoint": plan.is_checkpoint,
    }
    queued = emit_segmented_soon(
        context,
        event_type=EventType.CONTEXT_REVISION_RECORDED,
        operation_id=plan.revision_id,
        parent_operation_id=context.innermost_model_operation_id,
        payload=payload,
        array_fields={REFS_POINTER: list(plan.entries)},
        measurements={"/payload/token_buckets": measurement(MeasurementSource.LOCAL_TOKENIZER, method_version=1)},
    )
    if not queued:
        return None
    chain.commit(plan)
    return plan.revision_id


__all__ = ["REFS_POINTER", "TOKENIZER_FINGERPRINT", "record_context_revision"]
