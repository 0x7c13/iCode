# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cancellation-safe ownership of one workflow activation's per-attempt archives."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chrys.foundation.util.once_close import finish_close
from chrys.service.session.sub_agent_logs import (
    SUB_AGENT_TRANSCRIPT_OCCURRENCE_IDENTITY,
    SubAgentLogStats,
    serialized_history_has_occurrence_ids,
)
from chrys.service.state.serializers import serialize_state
from chrys.service.workflows.store import (
    NODE_RECORD_SESSION,
    NODE_RECORD_USAGE,
    WorkflowRunStore,
    WorkflowStorageFailed,
    node_stderr_path,
)

logger = logging.getLogger(__name__)


class AgentNodeArchive:
    """Keep snapshots separate from the live conversation and from sibling attempts."""

    def __init__(self, store: WorkflowRunStore, *, activation_id: str, invocation_id: str, profile_name: str) -> None:
        self._store = store
        self._activation_id = activation_id
        self._invocation_id = invocation_id
        self._profile_name = profile_name
        self._lock = asyncio.Lock()

    @property
    def stderr_path(self) -> Path:
        return node_stderr_path(self._store.run_dir, self._activation_id)

    @property
    def log_dir(self) -> Path:
        """Keep context and last-words diagnostics with this activation's artifacts."""
        return self.stderr_path.parent / "debug"

    async def write(
        self,
        *,
        attempt: int,
        status: str,
        error: str,
        state: dict[str, Any] | None,
        acp_state: dict[str, Any] | None,
        stats: SubAgentLogStats,
    ) -> None:
        """Snapshot state before handing it to the atomic, owner-only writer.

        The pass owns this write through cancellation: a delayed checkpoint
        cannot overwrite its terminal snapshot or spill into the next attempt.
        This archive is for history viewing, not a durable continuation ticket.
        Running checkpoints are best-effort; a terminal failure must reach the
        run owner so it cannot report durable success with missing history.
        """
        async with self._lock:
            meta = {
                "schema_version": 1,
                "record_type": "workflow_node_session",
                "runner": "acp" if acp_state is not None else "kernel",
                "activation_id": self._activation_id,
                "attempt": attempt,
                "invocation_id": self._invocation_id,
                "agent_display_name": self._profile_name,
                "status": status,
                "last_error": error,
                **asdict(stats),
            }
            try:
                if acp_state is not None:
                    envelope = {"meta": meta, "acp_state": acp_state}
                else:
                    serialized = serialize_state(state or {})
                    if serialized_history_has_occurrence_ids(serialized):
                        meta["transcript_occurrence_identity"] = SUB_AGENT_TRANSCRIPT_OCCURRENCE_IDENTITY
                    envelope = {"meta": meta, "state": serialized}
                await finish_close(asyncio.create_task(self._write_record(attempt, envelope)))
            except (WorkflowStorageFailed, TypeError, ValueError) as exc:
                if status != "running":
                    raise WorkflowStorageFailed(
                        f"Agent archive for {self._activation_id} attempt {attempt} could not be saved: {exc}"
                    ) from exc
                logger.warning(
                    "Workflow transcript for %s attempt %s could not be saved",
                    self._activation_id,
                    attempt,
                    exc_info=True,
                )

    async def _write_record(self, attempt: int, envelope: dict[str, Any]) -> None:
        await asyncio.to_thread(self._write_snapshot, attempt, envelope)

    def _write_snapshot(self, attempt: int, envelope: dict[str, Any]) -> None:
        self._store.write_node_value(self._activation_id, attempt, NODE_RECORD_SESSION, envelope)
        # Keep graph restoration bounded by node count, not transcript size.
        meta = envelope["meta"]
        self._store.write_node_value(
            self._activation_id,
            attempt,
            NODE_RECORD_USAGE,
            {key: meta[key] for key in ("tool_call_count", "total_usage_tokens", "usage_unreported_attempts")},
        )


class CoalescedCheckpoint:
    """At most one snapshot per second, with a single writer and an explicit terminal barrier.

    Changes during a write schedule the next window. Closing wakes a pending
    timer immediately and drains an in-flight write; the caller then writes the
    final snapshot, which includes any still-dirty updates. No timer or writer
    survives into the next attempt.
    """

    def __init__(self, write: Callable[[], Awaitable[None]], *, interval: float = 1.0) -> None:
        self._write = write
        self._interval = interval
        self._dirty = asyncio.Event()
        self._closed = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def changed(self) -> None:
        if self._closed.is_set():
            return
        self._dirty.set()
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="chrys.workflow.agent.checkpoint")

    async def _run(self) -> None:
        while not self._closed.is_set():
            await self._dirty.wait()
            self._dirty.clear()
            try:
                await asyncio.wait_for(self._closed.wait(), timeout=self._interval)
                return
            except TimeoutError:
                pass
            await self._write()

    async def close(self) -> None:
        self._closed.set()
        self._dirty.set()
        if self._task is not None:
            await finish_close(self._task)
