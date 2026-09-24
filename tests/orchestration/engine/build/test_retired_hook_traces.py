# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A replaced hook manager keeps reporting to its original, still-open recorder."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from chrys.foundation.trajectory.event_types import EventType
from chrys.orchestration.engine.loader import _close_replaced_hook_manager
from chrys.orchestration.engine.trajectory import TrajectoryRecorder
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.manager import HookManager
from chrys.service.hooks.outbox import DONE, FAILED
from chrys.service.hooks.runner import HookResult
from chrys.service.hooks.schema import HookConfig, HookExecution, HookRun, HooksFile
from chrys.service.trajectory import hooks as hook_traces
from chrys.service.trajectory.hooks import HookOutcome
from chrys.service.trajectory.session import SessionStartInfo, trajectory_events_path
from tests.support.trajectory_invariants import assert_trajectory_file_accounted, assert_trajectory_operation_settlement


@pytest.mark.asyncio
@pytest.mark.parametrize("close_before_result", [False, True], ids=["real-result", "recorder-closes-first"])
@pytest.mark.parametrize("exit_code", [0, 7], ids=["success", "failure"])
@pytest.mark.parametrize("mode", ["fire_and_forget", "async"])
async def test_replaced_managers_report_results_until_the_original_recorder_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, close_before_result: bool, exit_code: int, mode: str
) -> None:
    recorder = TrajectoryRecorder()
    session_id = str(uuid4())
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    info = SessionStartInfo(str(tmp_path), "a" * 64, "b" * 64)
    trajectory = recorder.bind_session(
        session_id=session_id, session_dir=session_dir, write_lock_path=None, session_start_info=lambda: info
    )
    release = asyncio.Event()
    clock_ns = 0
    monkeypatch.setattr(hook_traces, "time", SimpleNamespace(monotonic_ns=lambda: clock_ns))
    tasks: list[asyncio.Task[HookResult]] = []
    managers: list[HookManager] = []

    async def held_result(hook: HookConfig, _payload: dict[str, Any]) -> HookResult:
        await release.wait()
        return HookResult(hook_id=hook.id, exit_code=exit_code)

    try:
        # Multiple replacements must retain every old trace, even when the new
        # executor has no hook manager. Derived contexts share that ownership.
        for index in range(2):
            manager = HookManager(
                file=HooksFile(
                    hooks=[
                        HookConfig(
                            id=f"retired-{index}",
                            event=HookEvent.USER_INTERRUPT,
                            run=HookRun(type="command", argv=[sys.executable, "-c", "pass"]),
                            execution=HookExecution(mode=mode, delivery="durable"),
                        )
                    ]
                ),
                hooks_dir=tmp_path / f"hooks-{index}",
            )
            managers.append(manager)
            monkeypatch.setattr(manager._runner, "run_and_wait", held_result)
            context = trajectory.context().with_actor(trajectory.sub_agent_actor("c" * 12)).with_turn("d" * 32)
            manager.trajectory_context_provider = lambda context=context: context
            await manager.fire(HookEvent.USER_INTERRUPT, {}, scope="detached")
            tasks.append(manager._inflight[0].task)
            await _close_replaced_hook_manager(manager)
            assert manager._closed
            assert not tasks[-1].done()
            assert (
                recorder.bind_session(
                    session_id=session_id,
                    session_dir=session_dir,
                    write_lock_path=None,
                    session_start_info=lambda: info,
                )
                is trajectory
            )

        before = assert_trajectory_file_accounted(trajectory_events_path(session_dir))
        assert len([event for event in before.events if event.event_type == EventType.HOOK_OPERATION_STARTED]) == 2
        assert not [event for event in before.events if event.event_type == EventType.HOOK_OPERATION_FINISHED]

        if close_before_result:
            clock_ns = 2_000_000_000
            await recorder.close()
            closed_bytes = trajectory_events_path(session_dir).read_bytes()
            # A successor session cannot inherit late terminals from old hooks.
            next_dir = tmp_path / "next-session"
            next_dir.mkdir()
            recorder.bind_session(
                session_id=str(uuid4()), session_dir=next_dir, write_lock_path=None, session_start_info=lambda: info
            )
        clock_ns = 5_000_000_000
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
        await recorder.close()

        if close_before_result:
            assert trajectory_events_path(session_dir).read_bytes() == closed_bytes
            assert not trajectory_events_path(next_dir).exists()
        recorded = assert_trajectory_file_accounted(trajectory_events_path(session_dir))
        assert_trajectory_operation_settlement(recorded.events)
        starts = [event for event in recorded.events if event.event_type == EventType.HOOK_OPERATION_STARTED]
        terminals = [event for event in recorded.events if event.event_type == EventType.HOOK_OPERATION_FINISHED]
        assert len(terminals) == 2
        assert {event.operation_id for event in terminals} == {event.operation_id for event in starts}
        expected = (
            HookOutcome.ABANDONED
            if close_before_result
            else HookOutcome.SUCCESS
            if exit_code == 0
            else HookOutcome.FAILED
        )
        for event in terminals:
            assert event.payload["outcome"] == expected
            assert event.payload.get("exit_code") == (None if close_before_result else exit_code)
            assert event.payload["duration_ms"] == (2000 if close_before_result else 5000)
            assert event.actor == starts[0].actor
            assert event.turn_id == starts[0].turn_id
        for index in range(2):
            outbox = tmp_path / f"hooks-{index}" / "outbox"
            assert len(list((outbox / (DONE if exit_code == 0 else FAILED)).glob("*.json"))) == 1
            assert not list((outbox / "pending").glob("*.json"))
    finally:
        release.set()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
        finally:
            for manager in managers:
                await manager.drain_session()
            await recorder.close()
