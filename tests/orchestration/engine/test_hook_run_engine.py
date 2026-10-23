# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Engine-driven: the engine's hook run log, who each run is recorded for, and what outlives a session.

The hooks file sits in the per-test config dir, as a user's would; the hooks
are real subprocesses, held by a release file the test creates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    ApprovalRequest,
    ApprovalResponse,
    HookRunsChanged,
    SessionNew,
    SessionRestore,
    SettingsReload,
    UserInterrupt,
    UserMessage,
    WorkspaceChange,
)
from chrys.foundation.models.hook_runs import HookRunRecord
from chrys.foundation.platform import get_platform
from chrys.orchestration.session_host import ChrysSessionHost
from chrys.service.hooks.run_log import HookRunLog
from chrys.service.llm.mock import MockResponse
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import AgentProfile
from chrys.service.state.store import JsonFileStateStore
from chrys.service.trajectory.hooks import HookOutcome
from tests.orchestration.engine.test_hook_reporting_engine import _build_compacting_child_engine
from tests.orchestration.invoker._build_fixtures import build_recipe_engine
from tests.support.waiting import (
    ENGINE_TEST_WAIT_TIMEOUT,
    ENGINE_TURN_TIMEOUT,
    await_run_task_chain,
    wait_for,
    with_wait_deadline,
)


def _held(release: Path) -> list[str]:
    script = textwrap.dedent(
        f"""
        import os, time
        while not os.path.exists({str(release)!r}):
            time.sleep(0.01)
        """
    )
    return [sys.executable, "-c", script]


def _entry(
    hook_id: str, event: str, argv: list[str] | None = None, *, mode: str = "blocking", tool_name: str | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": hook_id,
        "event": event,
        "run": {"type": "command", "argv": argv or [sys.executable, "-c", "pass"]},
        "execution": {"mode": mode, "timeout_seconds": 30},
    }
    if tool_name is not None:
        entry["match"] = {"tool_name": tool_name}
    return entry


def _write_hooks(*entries: dict[str, Any]) -> None:
    hooks_dir = get_platform().config_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    document = {"version": 1, "settings": {"shutdown_grace_seconds": 30}, "hooks": list(entries)}
    (hooks_dir / "hooks.json").write_text(json.dumps(document), encoding="utf-8")


def _recording(agent_engine: Any) -> Any:
    def factory(*args: Any, **kwargs: Any) -> Any:
        return agent_engine(*args, record_hook_runs=True, **kwargs)

    return factory


def _log(engine: Any) -> HookRunLog:
    log = engine.hook_runs
    assert isinstance(log, HookRunLog)
    return log


def _runs(engine: Any, hook_id: str) -> list[HookRunRecord]:
    return [record for record in _log(engine).records() if record.hook_id == hook_id]


@pytest.fixture
def release(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "release"
    yield path
    path.touch()


# --- Which engines keep a log (5.1) ---------------------------------------------


async def test_only_an_engine_assembled_to_record_has_a_log(agent_engine: Any, tmp_path: Path) -> None:
    plain = agent_engine(EventBus(), state_store=JsonFileStateStore(tmp_path / "a"))
    assert plain.hook_runs is None
    recording = agent_engine(EventBus(), state_store=JsonFileStateStore(tmp_path / "b"), record_hook_runs=True)
    assert isinstance(recording.hook_runs, HookRunLog)
    assert recording.hook_runs.records() == ()


async def test_a_headless_session_host_keeps_no_log(tmp_path: Path) -> None:
    agents = AgentProfileRegistry()
    agents.register(AgentProfile(name="Code", instructions="x"))
    host = ChrysSessionHost(profile_name="Code", agent_registry=agents, state_store=JsonFileStateStore(tmp_path))
    try:
        assert host.engine.hook_runs is None
    finally:
        await host.shutdown()


# --- Attribution (5.4) ----------------------------------------------------------


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_sub_agents_tool_and_approval_hooks_are_recorded_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: Any
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("notes", encoding="utf-8")
    _write_hooks(
        _entry("main-gate", "before_tool_call", tool_name="Explore"),
        _entry("child-gate", "before_tool_call", tool_name="read_file"),
        _entry("child-check", "after_tool_call", tool_name="read_file"),
        _entry("on-approval", "approval_requested"),
        _entry("on-resolved", "approval_resolved"),
        _entry("after-turn", "after_turn"),
    )
    engine, _main, _child = await build_recipe_engine(
        _recording(agent_engine),
        monkeypatch,
        tmp_path,
        main=[MockResponse(tool_calls=[("Explore", "call-1", {"prompt": "look"})]), MockResponse(text="done")],
        child=[MockResponse(tool_calls=[("read_file", "read-1", {"path": str(target)})]), MockResponse(text="read")],
    )

    answers: list[asyncio.Task[None]] = []

    async def answer(event: ApprovalRequest) -> None:
        # A human answers later: approval_requested has run by then.
        await wait_for(
            lambda: bool(_runs(engine, "on-approval")), timeout=ENGINE_TURN_TIMEOUT, description="approval hook ran"
        )
        # Edited args: the approval re-runs before_tool_call with them.
        await engine.event_bus.publish(
            ApprovalResponse(request_id=event.request_id, approved=True, modified_args={"path": str(target)})
        )

    async def approve(event: ApprovalRequest) -> None:
        answers.append(asyncio.create_task(answer(event)))

    await engine.event_bus.subscribe(ApprovalRequest, approve)
    await engine.event_bus.publish(UserMessage(text="delegate"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
    await asyncio.gather(*answers)

    child_gate = _runs(engine, "child-gate")
    assert len(child_gate) == 2, "the call and its re-check after the edited approval"
    for hook_id in ("child-gate", "child-check", "on-approval", "on-resolved"):
        runs = _runs(engine, hook_id)
        assert runs, hook_id
        assert {record.sub_agent for record in runs} == {"Explore"}, (hook_id, runs)
    assert {record.tool_name for record in child_gate} == {"read_file"}
    for hook_id in ("main-gate", "after-turn"):
        (record,) = _runs(engine, hook_id)
        assert record.sub_agent == "", hook_id
    assert _runs(engine, "main-gate")[0].tool_name == "Explore"


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_sub_agents_real_pre_compact_is_recorded_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: Any
) -> None:
    big = tmp_path / "big.txt"
    big.write_text("\n".join(f"line {i}: " + "lorem ipsum dolor sit amet " * 8 for i in range(3000)), encoding="utf-8")
    _write_hooks(_entry("on-compact", "pre_compact"))
    note = (
        "## Task\nThe user asked the explorer to read big.txt in the workspace and report what it holds, "
        "briefly and without reading any other file.\n"
        "## Progress\nbig.txt was read once in full with read_file. It holds three thousand numbered lines of "
        "placeholder lorem ipsum text and nothing else of interest: no code, no configuration and no data.\n"
        "## Next\nReply that big.txt is filler text only, without reading it again, and mention that its "
        "lines are numbered from zero so the caller can cite them.\n"
    )
    engine = await _build_compacting_child_engine(
        _recording(agent_engine),
        monkeypatch,
        tmp_path,
        main=[MockResponse(tool_calls=[("Explore", "call-1", {"prompt": "read big.txt"})]), MockResponse(text="done")],
        child=[
            MockResponse(tool_calls=[("read_file", "read-1", {"path": str(big)})]),
            MockResponse(text=note),
            MockResponse(text="read"),
        ],
    )
    await engine.event_bus.publish(UserMessage(text="delegate"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
    runs = _runs(engine, "on-compact")
    assert runs, "the child never compacted"
    assert {record.sub_agent for record in runs} == {"Explore"}


# --- Lifecycle (5.1, 5.2) -------------------------------------------------------


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_an_interrupted_hook_is_recorded_cancelled_and_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: Any, release: Path
) -> None:
    _write_hooks(_entry("gate", "before_tool_call", _held(release), tool_name="Explore"))
    engine, _main, _child = await build_recipe_engine(
        _recording(agent_engine),
        monkeypatch,
        tmp_path,
        main=[MockResponse(tool_calls=[("Explore", "call-1", {"prompt": "look"})]), MockResponse(text="done")],
        child=[MockResponse(text="looked")],
    )
    told: list[tuple[HookRunRecord, ...]] = []

    async def on_runs(event: HookRunsChanged) -> None:
        assert event.session_id
        told.append(_log(engine).records())

    await engine.event_bus.subscribe(HookRunsChanged, on_runs)
    await engine.event_bus.publish(UserMessage(text="delegate"))
    await wait_for(
        lambda: any(r.hook_id == "gate" and r.state == "running" for r in _log(engine).records()),
        timeout=ENGINE_TURN_TIMEOUT,
        description="the gate running",
    )
    await engine.event_bus.publish(UserInterrupt())
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
    (record,) = _runs(engine, "gate")
    assert (record.state, record.outcome) == ("ended", HookOutcome.CANCELLED)
    # Told by the time the turn has unwound; no idle sweep needed.
    assert told and told[-1] == _log(engine).records()


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_runs_survive_a_settings_reload_a_workspace_change_and_session_switches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: Any
) -> None:
    _write_hooks(_entry("after-turn", "after_turn"))
    engine, _main, _child = await build_recipe_engine(
        _recording(agent_engine),
        monkeypatch,
        tmp_path,
        main=[MockResponse(text=f"answer {n}") for n in range(4)],
        child=[],
    )
    log = _log(engine)

    async def turn(text: str) -> None:
        await engine.event_bus.publish(UserMessage(text=text))
        await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    first_session = engine.session_id
    await turn("one")
    managers = {id(engine.session.hook_manager)}
    await engine.event_bus.publish(SettingsReload(), raise_handler_errors=True)
    managers.add(id(engine.session.hook_manager))
    await turn("two")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    await engine.event_bus.publish(WorkspaceChange(primary_cwd=str(elsewhere)), raise_handler_errors=True)
    await turn("three")
    managers.add(id(engine.session.hook_manager))
    assert len(managers) > 1, "a reload or a working folder change replaced the manager"
    assert [r.session_id for r in _runs(engine, "after-turn")] == [first_session] * 3

    await engine.event_bus.publish(SessionNew(), raise_handler_errors=True)
    assert engine.session_id != first_session
    second_session = engine.session_id
    await turn("four")
    assert engine.hook_runs is log
    assert [r.session_id for r in _runs(engine, "after-turn")] == [first_session] * 3 + [second_session]

    before_restore = log.records()
    await engine.event_bus.publish(SessionRestore(session_id=first_session), raise_handler_errors=True)
    await wait_for(lambda: engine.session_id == first_session, timeout=ENGINE_TURN_TIMEOUT, description="restored")
    assert engine.hook_runs is log
    kept = {r.run_id for r in log.records()}
    assert {r.run_id for r in before_restore} <= kept


# --- Sensitive data (5.5) -------------------------------------------------------


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_the_details_are_not_written_to_session_files_trajectory_or_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: Any, caplog: pytest.LogCaptureFixture
) -> None:
    stdout_secret, stderr_secret, result_secret = "STDOUT-SECRET-7d1e", "STDERR-SECRET-41aa", "RESULT-SECRET-90bc"

    # Each secret is assembled by the hook, so the hooks file itself never holds it.
    def assembled(secret: str) -> str:
        head, tail = secret.split("-", 1)
        return f"{head!r} + '-' + {tail!r}"

    script = textwrap.dedent(
        f"""
        import json, os, sys
        print({assembled(stdout_secret)})
        sys.stderr.write({assembled(stderr_secret)})
        json.dump({{"note": {assembled(result_secret)}}}, open(os.environ["CHRYS_HOOK_RESULT"], "w"))
        """
    )
    _write_hooks(_entry("talkative", "after_turn", [sys.executable, "-c", script]))
    caplog.set_level(logging.DEBUG)
    engine, _main, _child = await build_recipe_engine(
        _recording(agent_engine), monkeypatch, tmp_path, main=[MockResponse(text="done")], child=[]
    )
    await engine.event_bus.publish(UserMessage(text="hello"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
    (record,) = _runs(engine, "talkative")
    detail = _log(engine).detail(record.run_id)
    assert detail is not None and detail.io is not None
    assert stdout_secret in detail.io.stdout, "kept in memory for the panel"
    # Shutdown writes the last of the session and trajectory, and releases the
    # session's active lock and trajectory writer lease, which on Windows are
    # byte-range locks that refuse every other handle's read while held.
    await engine.shutdown()

    leaks: list[str] = []
    for root in (tmp_path, get_platform().config_dir):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix == ".py":
                continue
            data = path.read_bytes()
            secrets = (stdout_secret, stderr_secret, result_secret)
            leaks.extend(f"{secret} in {path}" for secret in secrets if secret.encode() in data)
    assert leaks == []
    for secret in (stdout_secret, stderr_secret, result_secret):
        assert secret not in caplog.text
