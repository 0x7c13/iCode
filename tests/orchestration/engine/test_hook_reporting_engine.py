# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Engine-driven: a session's hook manager reports its runs and failures with no extra wiring.

The hooks file sits in the per-test config dir, as a user's would; the hooks
are real subprocesses, held by a release file the test creates.
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from chrys.app.tui.widgets.hook_activity import indicator_activity
from chrys.foundation.events.types import (
    ApprovalRequest,
    ApprovalResponse,
    Event,
    HookExecutionsChanged,
    InvocationToolCallStart,
    SessionNew,
    UserMessage,
    Warning,
)
from chrys.foundation.platform import get_platform
from chrys.service.llm.mock import MockResponse
from tests.orchestration.invoker._build_fixtures import build_recipe_engine
from tests.support.waiting import (
    ENGINE_TEST_WAIT_TIMEOUT,
    ENGINE_TURN_TIMEOUT,
    await_run_task_chain,
    wait_for,
    with_wait_deadline,
)


def _held(release: Path, *, exit_code: int = 0) -> list[str]:
    script = textwrap.dedent(
        f"""
        import os, sys, time
        while not os.path.exists({str(release)!r}):
            time.sleep(0.01)
        sys.exit({exit_code})
        """
    )
    return [sys.executable, "-c", script]


def _entry(
    hook_id: str, event: str, argv: list[str], *, mode: str = "blocking", tool_name: str | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": hook_id,
        "event": event,
        "run": {"type": "command", "argv": argv},
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


class _Recorder:
    """Keeps snapshots, tool starts and warnings in the order the bus delivered them."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def record(self, event: Event) -> None:
        self.events.append(event)

    async def attach(self, engine: Any) -> None:
        for kind in (HookExecutionsChanged, InvocationToolCallStart, Warning):
            await engine.event_bus.subscribe(kind, self.record)

    def snapshots(self) -> list[HookExecutionsChanged]:
        return [event for event in self.events if isinstance(event, HookExecutionsChanged)]

    def running(self, hook_id: str) -> list[Any]:
        return [
            view
            for event in self.snapshots()
            for view in event.executions
            if view.hook_id == hook_id and not view.queued
        ]

    def index_of(self, predicate: Any) -> int:
        return next(index for index, event in enumerate(self.events) if predicate(event))


@pytest.fixture
def release(tmp_path: Path):
    path = tmp_path / "release"
    yield path
    path.touch()


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_main_tool_hook_is_reported_before_its_tool_card_exists(
    tmp_path, monkeypatch, agent_engine, release: Path
) -> None:
    _write_hooks(_entry("gate-explore", "before_tool_call", _held(release), tool_name="Explore"))
    engine, _main, _child = await build_recipe_engine(
        agent_engine,
        monkeypatch,
        tmp_path,
        main=[MockResponse(tool_calls=[("Explore", "call-1", {"prompt": "look"})]), MockResponse(text="done")],
        child=[MockResponse(text="looked")],
    )
    recorder = _Recorder()
    await recorder.attach(engine)
    session_id = engine.session_id
    await engine.event_bus.publish(UserMessage(text="delegate"))
    try:
        await wait_for(
            lambda: recorder.running("gate-explore"), timeout=ENGINE_TURN_TIMEOUT, description="the hook running"
        )
        view = recorder.running("gate-explore")[0]
        assert (view.hook_id, view.waited_at) == ("gate-explore", "call")
        # before_tool_call runs before the tool card's start event.
        assert not any(
            isinstance(event, InvocationToolCallStart) and event.tool_name == "Explore" for event in recorder.events
        )
    finally:
        release.touch()
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

    snapshots = recorder.snapshots()
    assert {event.session_id for event in snapshots} == {session_id}
    first_running = recorder.index_of(
        lambda e: isinstance(e, HookExecutionsChanged) and any(v.hook_id == "gate-explore" for v in e.executions)
    )
    tool_start = recorder.index_of(lambda e: isinstance(e, InvocationToolCallStart) and e.tool_name == "Explore")
    assert first_running < tool_start
    assert snapshots[-1].executions == ()
    assert not [event for event in recorder.events if isinstance(event, Warning)]


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_sub_agents_hooks_reach_the_snapshot_and_its_failures_tell_nothing(
    tmp_path, monkeypatch, agent_engine, release: Path
) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("notes", encoding="utf-8")
    _write_hooks(
        _entry("child-gate", "before_tool_call", _held(release), tool_name="read_file"),
        _entry(
            "child-check", "after_tool_call", [sys.executable, "-c", "import sys; sys.exit(3)"], tool_name="read_file"
        ),
        _entry("on-start", "sub_agent_start", [sys.executable, "-c", "pass"]),
        _entry("on-approval", "approval_requested", [sys.executable, "-c", "pass"]),
        _entry("on-resolved", "approval_resolved", [sys.executable, "-c", "pass"]),
    )
    engine, _main, _child = await build_recipe_engine(
        agent_engine,
        monkeypatch,
        tmp_path,
        main=[MockResponse(tool_calls=[("Explore", "call-1", {"prompt": "look"})]), MockResponse(text="done")],
        child=[
            MockResponse(
                tool_calls=[
                    ("read_file", "read-1", {"path": str(target)}),
                ]
            ),
            MockResponse(tool_calls=[("read_file", "read-2", {"path": str(target)})]),
            MockResponse(text="read twice"),
        ],
    )
    recorder = _Recorder()
    await recorder.attach(engine)

    answers: list[asyncio.Task[None]] = []

    async def answer(event: ApprovalRequest, seen: int) -> None:
        # A human answer comes later: approval_requested runs while the user decides.
        await wait_for(
            lambda: len(recorder.running("on-approval")) > seen,
            timeout=ENGINE_TURN_TIMEOUT,
            description="approval_requested hook running",
        )
        # Edited args: the approval re-runs before_tool_call with them.
        await engine.event_bus.publish(
            ApprovalResponse(request_id=event.request_id, approved=True, modified_args={"path": str(target)})
        )

    async def approve(event: ApprovalRequest) -> None:
        answers.append(asyncio.create_task(answer(event, len(recorder.running("on-approval")))))

    await engine.event_bus.subscribe(ApprovalRequest, approve)
    await engine.event_bus.publish(UserMessage(text="delegate"))
    try:
        await wait_for(
            lambda: recorder.running("child-gate"), timeout=ENGINE_TURN_TIMEOUT, description="the child's hook running"
        )
    finally:
        release.touch()
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
    await asyncio.gather(*answers)

    child_starts = [
        event
        for event in recorder.events
        if isinstance(event, InvocationToolCallStart) and event.origin.kind == "sub_agent"
    ]
    assert child_starts, "the child's read_file calls reached their card"
    # Only the child calls read_file and waits on approvals here, so these runs were fired from inside it;
    # through the shared manager they land in the same snapshots as the main agent's.
    assert {view.waited_at for view in recorder.running("child-gate")} == {"call"}
    # Two calls, each re-checked after its edited approval: four distinct runs.
    assert len({view.started_at for view in recorder.running("child-gate")}) == 4
    for approval_hook in ("on-approval", "on-resolved"):
        assert recorder.running(approval_hook), approval_hook
    # sub_agent_start is fired by the main agent around its child.
    assert recorder.running("on-start")

    # The child's failing hook was in the snapshot while it ran, but no failure is told (section 7: logs only).
    assert recorder.running("child-check")
    assert [event for event in recorder.events if isinstance(event, Warning)] == []
    assert recorder.snapshots()[-1].executions == ()


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_new_session_reports_session_end_then_the_session_drain(tmp_path, monkeypatch, agent_engine) -> None:
    end_release, start_release = tmp_path / "end-release", tmp_path / "start-release"
    _write_hooks(
        _entry("on-start", "session_start", _held(start_release), mode="async"),
        _entry("on-end", "session_end", _held(end_release)),
    )
    engine, _main, _child = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, main=[], child=[])
    recorder = _Recorder()
    await recorder.attach(engine)
    old_session = engine.session_id
    try:
        await wait_for(
            lambda: (
                (engine.session.hook_manager is not None)
                and any(v.hook_id == "on-start" for v in engine.session.hook_manager.executions())
            ),
            timeout=ENGINE_TURN_TIMEOUT,
            description="the session_start hook running",
        )
        view = next(v for v in engine.session.hook_manager.executions() if v.hook_id == "on-start")
        assert view.waited_at == "session"

        switching = asyncio.create_task(engine.event_bus.publish(SessionNew()))
        await wait_for(
            lambda: recorder.running("on-end") or switching.done(),
            timeout=ENGINE_TURN_TIMEOUT,
            description="session_end running",
        )
        assert not switching.done()
        latest = recorder.snapshots()[-1]
        assert latest.session_id == old_session
        by_id = {v.hook_id: v for v in latest.executions}
        assert by_id["on-end"].waited_at == "call"
        assert by_id["on-start"].waited_at == "session"

        end_release.touch()
        await wait_for(
            lambda: (
                [[v.hook_id for v in e.executions] for e in recorder.snapshots() if e.session_id == old_session][-1]
                == ["on-start"]
                or switching.done()
            ),
            timeout=ENGINE_TURN_TIMEOUT,
            description="only the session drain left",
        )
        assert not switching.done()
        start_release.touch()
        await asyncio.wait_for(switching, ENGINE_TURN_TIMEOUT)
    finally:
        end_release.touch()
        start_release.touch()
    old = [event for event in recorder.snapshots() if event.session_id == old_session]
    assert old[-1].executions == ()
    assert engine.session_id != old_session


async def _build_compacting_child_engine(
    agent_engine: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    main: list[MockResponse],
    child: list[MockResponse],
) -> Any:
    """Like ``build_recipe_engine``, but the child compacts: a small window, compaction on."""
    from unittest.mock import create_autospec

    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.models.workspace import Workspace
    from chrys.orchestration.engine.build import builder
    from chrys.orchestration.sub_agents import tools as child_tools
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.agents.schema import (
        AgentProfile,
        ApprovalConfig,
        CompactionConfig,
        ModelConfig,
        SkillsConfig,
        SubAgentRef,
        SubAgentsConfig,
        ToolsConfig,
    )
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.schema import ModelProfile
    from chrys.service.state.store import JsonFileStateStore
    from tests.orchestration.invoker._build_fixtures import CountedClient

    models = ModelProfileRegistry()
    models.register(ModelProfile(id="root-model", name="root-model", provider="mock", model_id="root", vision=False))
    models.register(
        ModelProfile(
            id="child-model",
            name="child-model",
            provider="mock",
            model_id="child",
            vision=False,
            # Holds the child's tool schemas, its first request and its last-words call, not the file it reads.
            max_context_tokens=9000,
            max_output_tokens=1000,
        )
    )
    skills = SkillsConfig(auto_load_user_agents_skills=False, auto_load_cwd_agents_skills=False)
    root = AgentProfile(
        name="Root",
        instructions="ROOT",
        model=ModelConfig(profile_id="root-model"),
        tools=ToolsConfig(builtins=[]),
        skills=skills,
        approval=ApprovalConfig(default="skip"),
        compaction=CompactionConfig(enabled=False),
        sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Explore", tool_name="Explore")]),
    )
    explore = AgentProfile(
        name="Explore",
        instructions="CHILD",
        model=ModelConfig(profile_id="child-model"),
        tools=ToolsConfig(builtins=["filesystem.read"]),
        skills=skills,
        approval=ApprovalConfig(default="skip"),
        compaction=CompactionConfig(enabled=True),
    )
    profiles = AgentProfileRegistry()
    profiles.register(root)
    profiles.register(explore)
    main_client, child_client = CountedClient(main), CountedClient(child)

    def client_for(profile: ModelProfile, **kwargs: object) -> CountedClient:
        client = main_client if profile.id == "root-model" else child_client
        client._on_intermediate_text = kwargs.get("on_intermediate_text")
        return client

    monkeypatch.setattr(builder, "create_client", create_autospec(builder.create_client, side_effect=client_for))
    monkeypatch.setattr(
        child_tools, "create_client", create_autospec(child_tools.create_client, side_effect=client_for)
    )
    engine = agent_engine(
        EventBus(),
        settings=Settings(
            model_profile_override="root-model",
            max_transient_retries=1,
            mutation_coordination=False,
            workspace_change_notice=False,
        ),
        model_registry=models,
        agent_registry=profiles,
        state_store=JsonFileStateStore(tmp_path / "sessions"),
    )
    await engine.start(root, workspace=Workspace.from_cwd(str(tmp_path)))
    return engine


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_sub_agents_real_pre_compact_hook_reaches_the_snapshot(
    tmp_path, monkeypatch, agent_engine, release: Path
) -> None:
    big = tmp_path / "big.txt"
    big.write_text("\n".join(f"line {i}: " + "lorem ipsum dolor sit amet " * 8 for i in range(3000)), encoding="utf-8")
    _write_hooks(_entry("on-compact", "pre_compact", _held(release)))
    note = (
        "## Task\nThe user asked the explorer to read big.txt in the workspace and report what it holds, "
        "briefly and without reading any other file.\n"
        "## Progress\nbig.txt was read once in full with read_file. It holds three thousand numbered lines of "
        "placeholder lorem ipsum text and nothing else of interest: no code, no configuration and no data.\n"
        "## Next\nReply that big.txt is filler text only, without reading it again, and mention that its "
        "lines are numbered from zero so the caller can cite them.\n"
    )
    engine = await _build_compacting_child_engine(
        agent_engine,
        monkeypatch,
        tmp_path,
        main=[MockResponse(tool_calls=[("Explore", "call-1", {"prompt": "read big.txt"})]), MockResponse(text="done")],
        child=[
            MockResponse(tool_calls=[("read_file", "read-1", {"path": str(big)})]),
            MockResponse(text=note),
            MockResponse(text="read"),
        ],
    )
    recorder = _Recorder()
    await recorder.attach(engine)
    await engine.event_bus.publish(UserMessage(text="delegate"))
    try:
        await wait_for(
            lambda: recorder.running("on-compact") or engine.turns.turn_state.lease.run_task is None,
            timeout=ENGINE_TURN_TIMEOUT,
            description="the child's pre_compact hook running",
        )
        views = recorder.running("on-compact")
        assert views, "the child never compacted"
        child_starts = [
            event
            for event in recorder.events
            if isinstance(event, InvocationToolCallStart) and event.origin.kind == "sub_agent"
        ]
        assert child_starts, "the child ran its tool before compacting"
        # Only the child compacts (the root has compaction off), so this run came from inside it.
        assert {view.waited_at for view in views} == {"call"}
        # What the indicator makes of that snapshot once it has lasted a second.
        activity = indicator_activity(recorder.snapshots()[-1].executions, now=time.monotonic() + 2)
        assert activity is not None
        assert activity.hook_id == "on-compact"
    finally:
        release.touch()
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
    assert recorder.snapshots()[-1].executions == ()
