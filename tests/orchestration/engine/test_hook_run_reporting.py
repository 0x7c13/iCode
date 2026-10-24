# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A session's hook manager writes the engine's run log and tells the TUI with ``HookRunsChanged``.

Covers the publication rules (5.2) and the runs that end late (5.1): cancelled
by a drain, or ending after the manager closed. Hooks are real ``python -c``
subprocesses; a held hook polls for a release file the test creates.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Event, HookExecutionsChanged, HookRunsChanged
from chrys.foundation.models.hook_runs import HookRunRecord
from chrys.orchestration import session_host
from chrys.orchestration.engine.hook_reporting import wire_hook_reporting
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.manager import HookManager
from chrys.service.hooks.run_log import HookRunLog
from chrys.service.hooks.schema import HookConfig, HookExecution, HookMatch, HookRun, HookSettings, HooksFile
from chrys.service.trajectory.hooks import HookOutcome
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


def _hook(hook_id: str, event: HookEvent, *, mode: str = "blocking", script: str = "pass") -> HookConfig:
    return HookConfig(
        id=hook_id,
        event=event,
        run=HookRun(type="command", argv=[sys.executable, "-c", script]),
        execution=HookExecution(mode=mode, timeout_seconds=30.0),
        match=HookMatch(),
    )


def _manager(tmp_path: Path, *hooks: HookConfig, **settings: float) -> HookManager:
    hooks_dir = tmp_path / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    return HookManager(file=HooksFile(settings=HookSettings(**settings), hooks=list(hooks)), hooks_dir=hooks_dir)


class _Seen:
    """Every snapshot and run-log notice in bus order, with the log as each notice found it."""

    def __init__(self, log: HookRunLog) -> None:
        self.log = log
        self.events: list[Event] = []
        self.log_at_notice: list[tuple[HookRunRecord, ...]] = []

    async def on_runs(self, event: HookRunsChanged) -> None:
        self.events.append(event)
        self.log_at_notice.append(self.log.records())

    async def on_snapshot(self, event: HookExecutionsChanged) -> None:
        self.events.append(event)

    async def attach(self, bus: EventBus) -> None:
        await bus.subscribe(HookRunsChanged, self.on_runs)
        await bus.subscribe(HookExecutionsChanged, self.on_snapshot)

    def notices(self, since: int = 0) -> list[HookRunsChanged]:
        return [e for e in self.events[since:] if isinstance(e, HookRunsChanged)]

    def snapshots(self, since: int = 0) -> list[HookExecutionsChanged]:
        return [e for e in self.events[since:] if isinstance(e, HookExecutionsChanged)]


def _states(seen: _Seen, hook_id: str, since: int = 0) -> list[str]:
    start = len([e for e in seen.events[:since] if isinstance(e, HookRunsChanged)])
    states: list[str] = []
    for records in seen.log_at_notice[start:]:
        state = next((r.state for r in records if r.hook_id == hook_id), "absent")
        if not states or states[-1] != state:
            states.append(state)
    return states


async def test_each_write_is_followed_by_a_notice_carrying_the_session(tmp_path: Path) -> None:
    release = tmp_path / "release"
    manager = _manager(tmp_path, _hook("guard", HookEvent.BEFORE_TOOL_CALL, script=_held(release)))
    bus, log = EventBus(), HookRunLog()
    seen = _Seen(log)
    await seen.attach(bus)
    wire_hook_reporting(manager, bus, session_id=_SESSION, run_log=log)
    fire = asyncio.create_task(manager.fire(HookEvent.BEFORE_TOOL_CALL, {"tool": {"name": "zsh", "args": {}}}))
    try:
        await wait_for(
            lambda: any(r.state == "running" for records in seen.log_at_notice for r in records),
            timeout=ENGINE_TURN_TIMEOUT,
            description="a notice after the run took its slot",
        )
    finally:
        release.touch()
    await fire
    assert {event.session_id for event in seen.notices()} == {_SESSION}
    # Written before the notice: queued, running, ended.
    assert _states(seen, "guard") == ["queued", "running", "ended"]
    assert seen.log_at_notice[-1] == log.records()
    # In one change the snapshot goes first, then the notice.
    last_snapshot = max(i for i, e in enumerate(seen.events) if isinstance(e, HookExecutionsChanged))
    last_notice = max(i for i, e in enumerate(seen.events) if isinstance(e, HookRunsChanged))
    assert last_snapshot < last_notice


async def test_without_a_run_log_no_notice_is_published(tmp_path: Path) -> None:
    manager = _manager(tmp_path, _hook("quick", HookEvent.AFTER_TURN))
    bus, log = EventBus(), HookRunLog()
    seen = _Seen(log)
    await seen.attach(bus)
    wire_hook_reporting(manager, bus, session_id=_SESSION)
    await manager.fire(HookEvent.AFTER_TURN, {"turn": 1})
    assert seen.snapshots(), "the snapshots still go out"
    assert seen.notices() == []
    assert log.records() == ()


async def test_a_run_a_non_closing_drain_cancels_is_recorded_cancelled_and_told(tmp_path: Path) -> None:
    release = tmp_path / "release"
    manager = _manager(
        tmp_path,
        _hook("lingering", HookEvent.SESSION_END, mode="async", script=_held(release)),
        shutdown_grace_seconds=0.3,
    )
    bus, log = EventBus(), HookRunLog()
    seen = _Seen(log)
    await seen.attach(bus)
    wire_hook_reporting(manager, bus, session_id=_SESSION, run_log=log)
    try:
        await manager.fire(HookEvent.SESSION_END, {}, scope="session")
        await wait_for(
            lambda: any(r.state == "running" for r in log.records()),
            timeout=ENGINE_TURN_TIMEOUT,
            description="running",
        )
        await asyncio.wait_for(manager.drain_session(close=False), ENGINE_TURN_TIMEOUT)
        (record,) = log.records()
        assert (record.state, record.outcome) == ("ended", HookOutcome.CANCELLED)
        assert seen.log_at_notice[-1] == log.records()
    finally:
        release.touch()
    await manager.drain_session()


async def test_a_background_run_ending_after_close_is_still_recorded_and_told_without_a_snapshot(
    tmp_path: Path,
) -> None:
    release_holder = tmp_path / "release-holder"
    ran = tmp_path / "queued-ran"
    manager = _manager(
        tmp_path,
        HookConfig(
            id="holder",
            event=HookEvent.AFTER_TOOL_CALL,
            run=HookRun(type="command", argv=[sys.executable, "-c", _held(release_holder)]),
            execution=HookExecution(mode="fire_and_forget", timeout_seconds=30.0),
            match=HookMatch(tool_name="a"),
        ),
        HookConfig(
            id="late",
            event=HookEvent.AFTER_TOOL_CALL,
            run=HookRun(type="command", argv=[sys.executable, "-c", f"open({str(ran)!r}, 'w').write('ran')"]),
            execution=HookExecution(mode="fire_and_forget", timeout_seconds=30.0),
            match=HookMatch(tool_name="b"),
        ),
        max_parallel_hooks=1,
    )
    bus, log = EventBus(), HookRunLog()
    seen = _Seen(log)
    await seen.attach(bus)
    wire_hook_reporting(manager, bus, session_id=_SESSION, run_log=log)
    try:
        await manager.fire(HookEvent.AFTER_TOOL_CALL, {"tool": {"name": "a", "args": {}}})
        await wait_for(
            lambda: any(r.hook_id == "holder" and r.state == "running" for r in log.records()),
            timeout=ENGINE_TURN_TIMEOUT,
            description="the holder running",
        )
        await manager.fire(HookEvent.AFTER_TOOL_CALL, {"tool": {"name": "b", "args": {}}})
        await wait_for(
            lambda: any(r.hook_id == "late" and r.state == "queued" for r in log.records()),
            timeout=ENGINE_TURN_TIMEOUT,
            description="the second queued",
        )
        await asyncio.wait_for(manager.drain_session(close=True), ENGINE_TURN_TIMEOUT)
        closed_at = len(seen.events)
    finally:
        release_holder.touch()
    await wait_for(
        lambda: all(r.state == "ended" for r in log.records()) and ran.exists(),
        timeout=ENGINE_TURN_TIMEOUT,
        description="both runs ended after the close",
    )
    await wait_for(
        lambda: bool(seen.log_at_notice) and seen.log_at_notice[-1] == log.records(),
        timeout=ENGINE_TURN_TIMEOUT,
        description="the last write told",
    )
    by_id = {record.hook_id: record for record in log.records()}
    assert {record.session_id for record in by_id.values()} == {_SESSION}
    assert by_id["late"].outcome == HookOutcome.SUCCESS
    assert by_id["late"].started_at is not None
    # Told when it took its slot and when it ended, after the close; no snapshot after it.
    assert _states(seen, "late", since=closed_at)[-2:] == ["running", "ended"]
    assert seen.snapshots(closed_at) == []
    assert all(notice.session_id == _SESSION for notice in seen.notices(closed_at))


async def test_a_late_run_is_told_even_when_a_newer_session_reports_on_the_same_bus(tmp_path: Path) -> None:
    """Section 5.2: not filtered by session; the TUI rereads every session's runs."""
    release = tmp_path / "release"
    old = _manager(tmp_path, _hook("ff", HookEvent.AFTER_TOOL_CALL, mode="fire_and_forget", script=_held(release)))
    new = _manager(tmp_path, _hook("next", HookEvent.AFTER_TURN))
    bus, log = EventBus(), HookRunLog()
    seen = _Seen(log)
    await seen.attach(bus)
    wire_hook_reporting(old, bus, session_id="old", run_log=log)
    try:
        await old.fire(HookEvent.AFTER_TOOL_CALL, {"tool": {"name": "x", "args": {}}})
        await wait_for(
            lambda: any(r.state == "running" for r in log.records()), timeout=ENGINE_TURN_TIMEOUT, description="ff"
        )
        await asyncio.wait_for(old.drain_session(close=True), ENGINE_TURN_TIMEOUT)
        wire_hook_reporting(new, bus, session_id="new", run_log=log)
        await new.fire(HookEvent.AFTER_TURN, {"turn": 1})
    finally:
        release.touch()
    await wait_for(
        lambda: all(r.state == "ended" for r in log.records()) and seen.log_at_notice[-1] == log.records(),
        timeout=ENGINE_TURN_TIMEOUT,
        description="the old session's run ended and told",
    )
    sessions = {record.hook_id: record.session_id for record in log.records()}
    assert sessions == {"ff": "old", "next": "new"}
    assert {notice.session_id for notice in seen.notices()} == {"old", "new"}
    await new.drain_session()


async def test_a_raising_bus_subscriber_does_not_stop_the_hook(tmp_path: Path) -> None:
    script = (
        "import json, os\njson.dump({'action': 'block', 'reason': 'no'}, open(os.environ['CHRYS_HOOK_RESULT'], 'w'))"
    )
    manager = _manager(tmp_path, _hook("guard", HookEvent.BEFORE_TOOL_CALL, script=script))
    bus, log = EventBus(), HookRunLog()

    async def explode(_event: HookRunsChanged) -> None:
        raise RuntimeError("subscriber exploded")

    await bus.subscribe(HookRunsChanged, explode)
    wire_hook_reporting(manager, bus, session_id=_SESSION, run_log=log)
    decision = await manager.fire(HookEvent.BEFORE_TOOL_CALL, {"tool": {"name": "zsh", "args": {}}})
    assert decision.blocked is True
    (record,) = log.records()
    assert (record.state, record.effect) == ("ended", "blocked")


def test_the_notice_does_not_reach_headless_or_acp_output() -> None:
    assert not issubclass(HookRunsChanged, session_host._HEADLESS_RUN_EVENT_TYPES)
    assert HookRunsChanged(session_id="s").session_id == "s"
