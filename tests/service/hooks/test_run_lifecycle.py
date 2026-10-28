# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""When a recorded run ends: once, at how its process ended, before anything is awaited.

A run's end is settled the moment its outcome is known, and only then reported
and written to the trajectory: an interrupt landing on those awaits never logs a
finished run as cancelled, and still closes its span. Whatever the run is
handed, it fails as a run, never as the caller, and lets go of its payload once
it ended. Hooks are real ``python -c`` subprocesses.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.models.hook_runs import HookRunRecord
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.foundation.trajectory.context import trajectory_scope
from chrys.foundation.trajectory.envelope import EventDraft
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.writer import EmitResult
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.manager import HookManager
from chrys.service.hooks.run_log import HookRunLog
from chrys.service.hooks.schema import HookConfig, HookExecution, HookMatch, HookRun, HooksFile
from chrys.service.trajectory.hooks import HookOutcome
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

_PRINTS = "import sys\nsys.stdout.write('printed')\n"


def _hook(hook_id: str, event: HookEvent, *, mode: str, script: str = _PRINTS) -> HookConfig:
    return HookConfig(
        id=hook_id,
        event=event,
        run=HookRun(type="command", argv=[sys.executable, "-c", script]),
        execution=HookExecution(mode=mode, timeout_seconds=30.0),
        match=HookMatch(),
    )


def _recorded(
    hooks_dir: Path, *hooks: HookConfig, on_change: Callable[[], Any] | None = None
) -> tuple[HookManager, HookRunLog]:
    manager = HookManager(file=HooksFile(hooks=list(hooks)), hooks_dir=hooks_dir)
    log = HookRunLog()

    async def changed() -> None:
        if on_change is not None:
            on_change()

    manager.record_runs(log, session_id="s1", on_change=changed)
    return manager, log


def _only(log: HookRunLog) -> HookRunRecord:
    (record,) = log.records()
    return record


class _CancelOnFinishedSink(FakeSink):
    """A sink whose terminal hook write is interrupted, the way an interrupt during its ack would."""

    async def emit(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult:
        if draft.event_type == EventType.HOOK_OPERATION_FINISHED:
            raise asyncio.CancelledError
        return await super().emit(draft, payload_factory=payload_factory)


@pytest.fixture
def hooks_dir(tmp_path: Path) -> Path:
    path = tmp_path / "hooks"
    path.mkdir()
    return path


# --- Interrupted after the run ended ---------------------------------------------


@pytest.mark.parametrize("mode", ["blocking", "async"])
async def test_a_run_interrupted_on_its_span_marker_keeps_how_it_ended(hooks_dir: Path, mode: str) -> None:
    manager, log = _recorded(hooks_dir, _hook("lint", HookEvent.AFTER_TURN, mode=mode))

    with trajectory_scope(make_context(_CancelOnFinishedSink())):
        if mode == "blocking":
            with pytest.raises(asyncio.CancelledError):
                await manager.fire(HookEvent.AFTER_TURN, {"turn": 1})
        else:
            await manager.fire(HookEvent.AFTER_TURN, {"turn": 1})
            await asyncio.wait_for(manager.drain_turn(), ENGINE_TURN_TIMEOUT)

    record = _only(log)
    assert (record.state, record.outcome, record.exit_code) == ("ended", HookOutcome.SUCCESS, 0)
    detail = log.detail(record.run_id)
    assert detail is not None and detail.io is not None
    assert detail.io.stdout == "printed"
    assert manager.executions() == ()


@pytest.mark.parametrize("mode", ["blocking", "async"])
async def test_an_interrupt_while_reporting_the_end_still_closes_the_span_as_it_ended(
    hooks_dir: Path, mode: str
) -> None:
    interrupted: list[str] = []
    log_box: list[HookRunLog] = []

    def on_change() -> None:
        # The end's report: the run log already says how it ended.
        if not interrupted and any(record.state == "ended" for record in log_box[0].records()):
            interrupted.append("once")
            raise asyncio.CancelledError

    manager, log = _recorded(hooks_dir, _hook("lint", HookEvent.AFTER_TURN, mode=mode), on_change=on_change)
    log_box.append(log)
    sink = FakeSink()

    with trajectory_scope(make_context(sink)):
        if mode == "blocking":
            with pytest.raises(asyncio.CancelledError):
                await manager.fire(HookEvent.AFTER_TURN, {"turn": 1})
        else:
            await manager.fire(HookEvent.AFTER_TURN, {"turn": 1})
            await asyncio.wait_for(manager.drain_turn(), ENGINE_TURN_TIMEOUT)
        await wait_for(
            lambda: sink.of_type(EventType.HOOK_OPERATION_FINISHED),
            timeout=ENGINE_TURN_TIMEOUT,
            description="the span closed",
        )

    assert interrupted == ["once"]
    (finished,) = sink.of_type(EventType.HOOK_OPERATION_FINISHED)
    assert finished.payload["outcome"] == HookOutcome.SUCCESS
    assert _only(log).outcome == HookOutcome.SUCCESS
    assert manager.executions() == ()


async def test_a_detached_run_interrupted_while_handed_off_is_recorded_as_cancelled(
    hooks_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changes: list[None] = []
    hook = _hook("worker", HookEvent.AFTER_TURN, mode="fire_and_forget")
    hook = dataclasses.replace(hook, execution=dataclasses.replace(hook.execution, detach=True))
    manager, log = _recorded(hooks_dir, hook, on_change=lambda: changes.append(None))
    handing_off = asyncio.Event()

    async def spawn_detached(
        hook: HookConfig, payload: dict[str, Any], *, job: object = None, payload_text: str | None = None
    ) -> Path:
        handing_off.set()
        await asyncio.Event().wait()
        raise AssertionError("never handed off")

    monkeypatch.setattr(manager._runner, "spawn_detached", spawn_detached)
    sink = FakeSink()
    with trajectory_scope(make_context(sink)):
        fire = asyncio.create_task(manager.fire(HookEvent.AFTER_TURN, {"turn": 1}))
        await wait_for(lambda: handing_off.is_set() or fire.done(), description="the hand-off started")
        assert handing_off.is_set() and not fire.done()
        fire.cancel()
        with pytest.raises(asyncio.CancelledError):
            await fire
        await wait_for(
            lambda: sink.of_type(EventType.HOOK_OPERATION_FINISHED),
            timeout=ENGINE_TURN_TIMEOUT,
            description="the span closed",
        )

    (finished,) = sink.of_type(EventType.HOOK_OPERATION_FINISHED)
    assert finished.payload["outcome"] == HookOutcome.CANCELLED
    record = _only(log)
    assert (record.state, record.outcome, record.detach) == ("ended", HookOutcome.CANCELLED, True)
    assert changes, "the Hooks tab is told"
    assert manager.executions() == ()


# --- What a run holds ------------------------------------------------------------


async def test_a_payload_that_cannot_be_serialized_fails_the_run_not_the_caller(hooks_dir: Path) -> None:
    manager, log = _recorded(hooks_dir, _hook("gate", HookEvent.BEFORE_TOOL_CALL, mode="blocking"))

    class _NoTextForm:
        def __str__(self) -> str:
            raise TypeError("no text form")

    # A value with no text form: JSON cannot write it, whatever ``default=`` says.
    payload = {"tool": {"name": "zsh", "call_id": "c1", "args": {"value": _NoTextForm()}}}

    decision = await manager.fire(HookEvent.BEFORE_TOOL_CALL, payload)

    assert decision.blocked is False
    record = _only(log)
    assert (record.state, record.outcome, record.tool_name) == ("ended", HookOutcome.LAUNCH_ERROR, "zsh")
    detail = log.detail(record.run_id)
    assert detail is not None and detail.io is not None
    assert "no text form" in detail.io.stderr
    assert manager.executions() == ()


async def test_an_ended_run_awaiting_its_turn_drain_no_longer_holds_its_payload(hooks_dir: Path) -> None:
    manager, log = _recorded(hooks_dir, _hook("lint", HookEvent.AFTER_TURN, mode="async"))

    await manager.fire(HookEvent.AFTER_TURN, {"turn": 1, "transcript": "t" * 100_000})
    await wait_for(
        lambda: _only(log).state == "ended", timeout=ENGINE_TURN_TIMEOUT, description="the run ended undrained"
    )
    # Kept until the turn's drain, as the drain reports what it waited on.
    (tracked,) = manager._inflight
    assert tracked.task.done()
    assert tracked.run.payload_text is None
    detail = log.detail(_only(log).run_id)
    assert detail is not None and detail.io is not None and "t" * 1000 in detail.io.payload, "the log keeps its copy"
    await asyncio.wait_for(manager.drain_turn(), ENGINE_TURN_TIMEOUT)


# --- Who fired it ----------------------------------------------------------------


async def test_a_sub_agent_has_one_name_on_every_event_it_fires(hooks_dir: Path) -> None:
    manager, log = _recorded(
        hooks_dir,
        _hook("gate", HookEvent.BEFORE_TOOL_CALL, mode="blocking", script="pass"),
        _hook("compact", HookEvent.PRE_COMPACT, mode="blocking", script="pass"),
    )
    turn = InvocationOrigin(kind="turn", session_id="s1", invocation_id="turn-1", parent=None)
    child = InvocationOrigin(kind="sub_agent", session_id="s1", invocation_id="child-1", parent=turn)

    # A tool payload carries the sub-agent's profile name; pre_compact its display name too.
    await manager.fire(
        HookEvent.BEFORE_TOOL_CALL,
        {"profile": "Explore", "tool": {"name": "read_file", "call_id": "r1", "args": {}}},
        origin=child,
    )
    await manager.fire(
        HookEvent.PRE_COMPACT,
        {"profile": "Explore", "sub_agent": {"name": "Explore Agent", "tool_name": "Explore"}},
        origin=child,
    )
    await manager.fire(HookEvent.BEFORE_TOOL_CALL, {"profile": "Code", "tool": {"name": "Explore"}}, origin=turn)

    assert [(record.hook_id, record.sub_agent) for record in log.records()] == [
        ("gate", "Explore"),
        ("compact", "Explore"),
        ("gate", ""),
    ]
