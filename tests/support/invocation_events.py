# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Inputs shared by the frontend projection contract tests."""

from __future__ import annotations

from chrys.foundation.events.types import (
    InvocationAborted,
    InvocationCascadeAborted,
    InvocationContextPressure,
    InvocationEvent,
    InvocationMessage,
    InvocationPaused,
    InvocationPresentationAttemptAccepted,
    InvocationPresentationAttemptRejected,
    InvocationResumed,
    InvocationRetryAttempt,
    InvocationToolCallArgsUpdated,
    InvocationToolCallStart,
    ProvisionalPresentation,
)
from chrys.foundation.models.invocations import InvocationOrigin

PHASES = (
    "intermediate",
    "final",
    "provisional",
    "accepted",
    "rejected",
    "wire",
    "run",
    "compaction",
    "connection",
    "paused",
    "resumed",
    "aborted",
    "cascade",
    "tool",
    "pressure",
    "args",
)
PROSE_PHASES = frozenset(PHASES[:5])
ORIGINS = (
    InvocationOrigin("turn", "s1", "main", None),
    InvocationOrigin("sub_agent", "s1", "child", InvocationOrigin("turn", "s1", "main", None)),
    InvocationOrigin("workflow_node", "s1", "node", None),
    InvocationOrigin("sub_agent", "s1", "node-child", InvocationOrigin("workflow_node", "s1", "node", None)),
)
ORIGIN_IDS = ("turn", "chat_child", "workflow_node", "workflow_child")
"""Parametrize ids for ``ORIGINS``: two of them share the ``sub_agent`` kind."""


def has_chat_card(origin: InvocationOrigin) -> bool:
    """A chat turn's sub-agent has a nested card in every frontend; a workflow node's children have none."""
    return origin.kind == "sub_agent" and origin.root.kind != "workflow_node"


def projection_event(origin: InvocationOrigin, phase: str) -> InvocationEvent:
    """Produce a fact without changing the wire-facing text or control fields."""
    if phase in ("intermediate", "final", "provisional"):
        return InvocationMessage(
            origin=origin,
            session_id=origin.session_id,
            text="private child prose" if origin.kind == "sub_agent" else "main text",
            is_final=phase == "final",
            is_intermediate=phase != "final",
            presentation=ProvisionalPresentation("attempt", "segment") if phase == "provisional" else None,
        )
    if phase == "accepted":
        return InvocationPresentationAttemptAccepted(origin=origin, attempt_id="attempt", segment_ids=("segment",))
    if phase == "rejected":
        return InvocationPresentationAttemptRejected(origin=origin, attempt_id="attempt")
    if phase in ("wire", "run", "compaction", "connection"):
        return InvocationRetryAttempt(origin=origin, scope=phase, message="retry", attempt=1, max_attempts=3)
    if phase == "paused":
        return InvocationPaused(origin=origin, reason="framework_exc", last_error="failure")
    if phase == "resumed":
        return InvocationResumed(origin=origin)
    if phase == "aborted":
        return InvocationAborted(origin=origin, last_error="failure")
    if phase == "cascade":
        return InvocationCascadeAborted(origin=origin)
    if phase == "tool":
        return InvocationToolCallStart(origin=origin, call_id="tool", tool_name="read_file")
    if phase == "pressure":
        return InvocationContextPressure(
            origin=origin, reason="round_limit", source="sub_agent" if origin.kind == "sub_agent" else "main"
        )
    if phase == "args":
        return InvocationToolCallArgsUpdated(
            origin=origin, tool_name="read_file", call_id="tool", args={"file_path": "edited"}
        )
    raise AssertionError(phase)
