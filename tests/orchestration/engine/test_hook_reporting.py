# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hook runs reach the bus as snapshots for the TUI; failures publish nothing (logs only, as before).

Hooks are real ``python -c`` subprocesses; a held hook polls for a release
file the test creates.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import HookExecutionsChanged, Warning
from chrys.foundation.models.hook_executions import HookExecutionView
from chrys.orchestration.engine.hook_reporting import wire_hook_reporting
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.manager import HookManager
from chrys.service.hooks.schema import HookConfig, HookExecution, HookMatch, HookRun, HookSettings, HooksFile
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

_SESSION = "s1"


def _held(release: Path) -> str:
    return textwrap.dedent(
        f"""
        import os, time
        while not os.path.exists({str(release)!r}):
            time.sleep(0.01)
        """
    )


def _hook(
    hook_id: str,
    event: HookEvent,
    *,
    script: str = "pass",
    argv: list[str] | None = None,
    timeout: float = 30.0,
    tool_name: str | None = None,
) -> HookConfig:
    return HookConfig(
        id=hook_id,
        event=event,
        run=HookRun(type="command", argv=argv or [sys.executable, "-c", script]),
        execution=HookExecution(mode="blocking", timeout_seconds=timeout),
        match=HookMatch(tool_name=tool_name),
    )


def _manager(tmp_path: Path, *hooks: HookConfig, **settings: int) -> HookManager:
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    return HookManager(file=HooksFile(settings=HookSettings(**settings), hooks=list(hooks)), hooks_dir=hooks_dir)


async def _capture(bus: EventBus) -> tuple[list[HookExecutionsChanged], list[Warning]]:
    snapshots: list[HookExecutionsChanged] = []
    warnings: list[Warning] = []

    async def on_snapshot(event: HookExecutionsChanged) -> None:
        snapshots.append(event)

    async def on_warning(event: Warning) -> None:
        warnings.append(event)

    await bus.subscribe(HookExecutionsChanged, on_snapshot)
    await bus.subscribe(Warning, on_warning)
    return snapshots, warnings


def _payload(name: str = "shell") -> dict[str, object]:
    return {"tool": {"name": name, "kind": "shell", "call_id": "c1", "args": {}}}


async def test_snapshots_carry_the_session_and_the_managers_state(tmp_path: Path) -> None:
    release = tmp_path / "release"
    manager = _manager(tmp_path, _hook("guard", HookEvent.BEFORE_TOOL_CALL, script=_held(release)))
    bus = EventBus()
    snapshots, warnings = await _capture(bus)
    wire_hook_reporting(manager, bus, session_id=_SESSION)
    fire = asyncio.create_task(manager.fire(HookEvent.BEFORE_TOOL_CALL, _payload("zsh")))
    try:
        await wait_for(
            lambda: any(not view.queued for event in snapshots for view in event.executions),
            timeout=ENGINE_TURN_TIMEOUT,
            description="the running snapshot",
        )
        assert {event.session_id for event in snapshots} == {_SESSION}
        assert snapshots[-1].executions == manager.executions()
        (view,) = snapshots[-1].executions
        assert (view.hook_id, view.waited_at) == ("guard", "call")
    finally:
        release.touch()
    await fire
    assert snapshots[-1].executions == ()
    assert warnings == []


async def test_the_last_snapshot_received_matches_the_manager_under_parallel_runs(tmp_path: Path) -> None:
    """No sequence numbers needed: the last snapshot a synchronous subscriber keeps is the current one."""
    releases = [tmp_path / f"release-{i}" for i in range(3)]
    manager = _manager(
        tmp_path,
        *(
            _hook(f"h{i}", HookEvent.BEFORE_TOOL_CALL, script=_held(r), tool_name=f"t{i}")
            for i, r in enumerate(releases)
        ),
        max_parallel_hooks=2,
    )
    bus = EventBus()
    kept: list[tuple[HookExecutionView, ...]] = []
    mismatches: list[str] = []

    async def keep(event: HookExecutionsChanged) -> None:
        kept[:] = [event.executions]
        if event.executions != manager.executions():
            mismatches.append(repr(event.executions))

    await bus.subscribe(HookExecutionsChanged, keep)
    wire_hook_reporting(manager, bus, session_id=_SESSION)
    fires = [asyncio.create_task(manager.fire(HookEvent.BEFORE_TOOL_CALL, _payload(f"t{i}"))) for i in range(3)]
    try:
        await wait_for(
            lambda: len(manager.executions()) == 3 and sum(not v.queued for v in manager.executions()) == 2,
            timeout=ENGINE_TURN_TIMEOUT,
            description="two running, one queued",
        )
        assert kept == [manager.executions()]
        releases[0].touch()
        releases[1].touch()
        await wait_for(
            lambda: [v.hook_id for v in manager.executions() if not v.queued] == ["h2"],
            timeout=ENGINE_TURN_TIMEOUT,
            description="only the third left",
        )
        assert kept == [manager.executions()]
    finally:
        for release in releases:
            release.touch()
    await asyncio.gather(*fires)
    assert kept == [()]
    assert mismatches == []


async def test_failing_blocking_hooks_publish_no_warning(tmp_path: Path) -> None:
    """No failure is told: a failed, timed-out or unstartable hook still only logs."""
    hooks = (
        _hook("exits", HookEvent.BEFORE_TOOL_CALL, script="import sys; sys.exit(3)", tool_name="exit"),
        _hook("slow", HookEvent.AFTER_TOOL_CALL, script="import time; time.sleep(30)", timeout=0.5),
        _hook("gone", HookEvent.AFTER_TURN, argv=["/definitely/not/a/real/chrys-hook-command"]),
    )
    bus = EventBus()
    snapshots, warnings = await _capture(bus)
    manager = _manager(tmp_path, *hooks)
    wire_hook_reporting(manager, bus, session_id=_SESSION)
    for _ in range(2):
        decision = await manager.fire(HookEvent.BEFORE_TOOL_CALL, _payload("exit"))
        assert decision.blocked is False
    await asyncio.wait_for(manager.fire(HookEvent.AFTER_TOOL_CALL, _payload()), ENGINE_TURN_TIMEOUT)
    await manager.fire(HookEvent.AFTER_TURN, {"turn": 1})
    assert warnings == []
    # Their runs were still reported while they ran.
    assert {view.hook_id for event in snapshots for view in event.executions} >= {"exits", "slow"}
    assert snapshots[-1].executions == ()


async def test_the_snapshot_includes_runs_nobody_waits_for(tmp_path: Path) -> None:
    release = tmp_path / "release"
    observer = HookConfig(
        id="observer",
        event=HookEvent.AFTER_TOOL_CALL,
        run=HookRun(type="command", argv=[sys.executable, "-c", _held(release)]),
        execution=HookExecution(mode="fire_and_forget", timeout_seconds=30.0),
    )
    manager = _manager(tmp_path, observer)
    bus = EventBus()
    snapshots, _warnings = await _capture(bus)
    wire_hook_reporting(manager, bus, session_id=_SESSION)
    await manager.fire(HookEvent.AFTER_TOOL_CALL, _payload())
    try:
        await wait_for(
            lambda: any(not v.queued for e in snapshots for v in e.executions),
            timeout=ENGINE_TURN_TIMEOUT,
            description="the fire-and-forget run reported",
        )
        assert snapshots[-1].executions[0].waited_at == "none"
    finally:
        release.touch()
    await wait_for(lambda: snapshots[-1].executions == (), timeout=ENGINE_TURN_TIMEOUT, description="ended")
