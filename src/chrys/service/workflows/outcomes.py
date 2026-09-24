# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Run outcomes, cancellation reasons and their presentation status."""

from __future__ import annotations

from enum import Enum
from typing import Final, Literal


class RunOutcome(Enum):
    """Terminal outcomes, including restore-only orphan reconciliation."""

    COMPLETED = "completed"
    NODE_FAILED = "node_failed"
    LOOP_EXHAUSTED = "loop_exhausted"
    CANCELLED = "cancelled"
    WORKER_LOST = "worker_lost"
    STORAGE_FAILED = "storage_failed"
    ORPHANED = "orphaned"


REASON_DEADLINE_EXCEEDED: Final = "deadline_exceeded"
"""``WorkflowRunFinished.reason`` when the run-level timeout cancelled the run."""
REASON_SHUTDOWN: Final = "shutdown"
"""``WorkflowRunFinished.reason`` when the owning engine shut down under the run."""
REASON_INTERNAL_ERROR: Final = "internal_error"
"""``WorkflowRunFinished.reason`` when the runner itself failed and cancelled the run to converge it."""
ORPHAN_REASON_PROCESS_TERMINATED: Final = "process_terminated"

type RunStatus = Literal["running", "completed", "failed", "cancelled", "interrupted"]


def run_status(outcome: str | None, reason: str | None, *, active: bool) -> RunStatus:
    """Project stored or live outcome facts to the status shown in history."""
    if outcome == RunOutcome.COMPLETED.value:
        return "completed"
    if outcome == RunOutcome.CANCELLED.value:
        if reason in {REASON_DEADLINE_EXCEEDED, REASON_INTERNAL_ERROR}:
            return "failed"
        return "interrupted" if reason == REASON_SHUTDOWN else "cancelled"
    if outcome == RunOutcome.ORPHANED.value:
        return "interrupted"
    if outcome:
        return "failed"
    return "running" if active else "interrupted"
