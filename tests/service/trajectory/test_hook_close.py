# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hook observation ending while a first event is still being submitted by a worker."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from chrys.foundation.trajectory.envelope import EventDraft
from chrys.foundation.trajectory.event_types import EventType, RuntimeFinishReason
from chrys.foundation.trajectory.writer import EmitResult
from chrys.service.trajectory.hooks import HookOperationTrace, HookOutcome
from chrys.service.trajectory.session import SessionStartInfo, SessionTrajectory, trajectory_events_path
from tests.support.trajectory_invariants import assert_trajectory_file_accounted, assert_trajectory_operation_settlement


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["cancel", "close"])
async def test_late_first_hook_submission_cannot_open_an_unobserved_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    trajectory = SessionTrajectory(
        session_id=str(uuid4()),
        session_dir=tmp_path,
        session_start_info=lambda: SessionStartInfo(str(tmp_path), "a" * 64, "b" * 64),
    )
    loop = asyncio.get_running_loop()
    activated = asyncio.Event()
    submitted = asyncio.Event()
    release = threading.Event()
    activate = trajectory._activate
    emit_blocking = trajectory.emit_blocking

    def held_activate() -> None:
        activate()
        loop.call_soon_threadsafe(activated.set)
        assert release.wait(10.0), "test did not release first hook submission"

    def observed_emit(
        draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult:
        try:
            return emit_blocking(draft, payload_factory=payload_factory)
        finally:
            loop.call_soon_threadsafe(submitted.set)

    monkeypatch.setattr(trajectory, "_activate", held_activate)
    monkeypatch.setattr(trajectory, "emit_blocking", observed_emit)
    trace = HookOperationTrace(trajectory.context(), target_operation_id=None)
    starting = asyncio.create_task(
        trace.started(
            hook_id="late",
            hook_event="session_start",
            execution_mode="fire_and_forget",
            detach=False,
            delivery="best_effort",
        )
    )
    try:
        await asyncio.wait_for(activated.wait(), timeout=5.0)
        assert not trace.start_committed
        if boundary == "cancel":
            starting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await starting
            trace.finished_soon(outcome=HookOutcome.CANCELLED)
        else:
            await trajectory.close(reason=RuntimeFinishReason.GRACEFUL_SHUTDOWN)

        release.set()
        await asyncio.wait_for(submitted.wait(), timeout=5.0)
        if boundary == "close":
            await asyncio.wait_for(starting, timeout=5.0)
        await trajectory.close(reason=RuntimeFinishReason.GRACEFUL_SHUTDOWN)
        recorded = assert_trajectory_file_accounted(trajectory_events_path(tmp_path))
        assert_trajectory_operation_settlement(recorded.events)
        assert not [
            event
            for event in recorded.events
            if event.event_type in {EventType.HOOK_OPERATION_STARTED, EventType.HOOK_OPERATION_FINISHED}
        ]
        assert not trace.start_committed
    finally:
        release.set()
        try:
            await asyncio.wait_for(asyncio.gather(starting, return_exceptions=True), timeout=5.0)
            # Cancellation does not join asyncio.to_thread's worker. Join its
            # completion signal before monkeypatch restores the recorder.
            await asyncio.wait_for(submitted.wait(), timeout=5.0)
        finally:
            await trajectory.close(reason=RuntimeFinishReason.GRACEFUL_SHUTDOWN)
