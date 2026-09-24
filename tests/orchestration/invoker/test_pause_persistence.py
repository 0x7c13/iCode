# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Read durable pause artifacts inside the event's publication boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.types import InvocationPaused, UserInterrupt, UserMessage
from chrys.foundation.platform.files import read_owner_verified_bounded
from chrys.service.approval.policy import ApprovalMode
from chrys.service.llm.mock import MockResponse
from chrys.service.session.sub_agent_logs import MAX_SUB_AGENT_AUDIT_BYTES
from tests.orchestration.sub_agents.test_acp_engine import _start_acp_engine
from tests.orchestration.sub_agents.test_integration import _make_ctx, _sub_tool_call
from tests.support.engines import AgentEngineFactory
from tests.support.scripted_clients import FrameworkBoom
from tests.support.waiting import (
    ENGINE_TEST_WAIT_TIMEOUT,
    ENGINE_TURN_TIMEOUT,
    await_run_task_chain,
    wait_for,
    with_wait_deadline,
)


@pytest.fixture(autouse=True)
def _isolated_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise real workspace scans without scanning the checkout or audit files."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


@pytest.mark.parametrize("backend", ["kernel", "acp"])
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_pause_publication_already_has_pending_and_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory, backend: str
) -> None:
    ctx = None
    if backend == "kernel":
        ctx = await _make_ctx(
            tmp_path,
            main_outcomes=[_sub_tool_call("work"), MockResponse(text="parent")],
            sub_outcomes=[FrameworkBoom("pause")],
            agent_engine=agent_engine,
        )
        engine, bus = ctx.engine, ctx.bus
    else:
        engine, bus, _, _ = await _start_acp_engine(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            agent_engine=agent_engine,
            scenario="retry_reused_tool_usage",
            flag_path=tmp_path / "wire.jsonl",
        )
    snapshots: list[tuple[InvocationPaused, list[dict], list[dict]]] = []
    errors: list[Exception] = []

    async def at_publish(event: InvocationPaused) -> None:
        # EventBus swallows subscriber exceptions by default: propagate observations
        # and errors to the test body, never assert only inside this callback.
        try:
            root = engine.session.session_dir / "sub_agents"
            pending = [json.loads(p.read_text()) for p in (root / "pending").glob("*.json")]
            audits = [json.loads(p.read_text()) for p in (root / "sessions").glob("*.json")]
            snapshots.append((event, pending, audits))
        except Exception as exc:
            errors.append(exc)

    await bus.subscribe(InvocationPaused, at_publish)
    try:
        await bus.publish(UserMessage(text="delegate"))
        # This includes parent preparation, real ACP startup/handshake, and
        # durable pause writes; the generic five-second event budget is too short.
        await wait_for(
            lambda: snapshots or errors,
            timeout=ENGINE_TURN_TIMEOUT,
            description="pause publication snapshot",
        )
        assert errors == []
        event, pending, audits = snapshots[0]
        assert len(pending) == len(audits) == 1
        assert pending[0]["invocation_id"] == event.origin.invocation_id
        assert audits[0]["meta"]["status"] == "paused"
        assert audits[0]["meta"]["invocation_id"] == event.origin.invocation_id
    finally:
        await bus.unsubscribe(InvocationPaused, at_publish)
        try:
            await bus.publish(UserInterrupt())
            await await_run_task_chain(
                engine,
                turn_state=engine.turns.turn_state,
                timeout=ENGINE_TURN_TIMEOUT,
                propagate_inner_cancel=True,
            )
        finally:
            if ctx is not None:
                await ctx.cleanup()


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_session_new_permission_and_update_survive_in_owned_pause_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory
) -> None:
    trace = tmp_path / "opening.json"
    engine, bus, _, captured = await _start_acp_engine(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        agent_engine=agent_engine,
        scenario="new_session_permission_update_failure",
        flag_path=trace,
    )
    tools = engine.current.loaded.sub_agent_tools
    assert tools is not None
    tools.set_approval_mode(ApprovalMode.BYPASS)
    from chrys.orchestration.sub_agents import tools as tools_module

    original_spec = tools_module.resolve_acp_spec

    def with_mode(*args, **kwargs):
        return replace(original_spec(*args, **kwargs), session_mode="review")

    monkeypatch.setattr(tools_module, "resolve_acp_spec", create_autospec(original_spec, side_effect=with_mode))
    try:
        await bus.publish(UserMessage(text="delegate"))
        await wait_for(
            lambda: captured.get(InvocationPaused),
            timeout=ENGINE_TURN_TIMEOUT,
            description="session/new failure pause",
        )
        pause = captured[InvocationPaused][0]
        assert trace.exists()
        remote = json.loads(trace.read_text())
        assert remote["pid"] > 0
        assert remote["outcome"]["outcome"]["outcome"] == "selected"
        assert remote["outcome"]["outcome"]["optionId"] == "allow"
        root = engine.session.session_dir
        assert root is not None
        (path,) = (root / "sub_agents" / "sessions").glob("*.json")
        envelope = json.loads(read_owner_verified_bounded(path, max_bytes=MAX_SUB_AGENT_AUDIT_BYTES))
        assert envelope is not None
        assert envelope["meta"]["invocation_id"] == pause.origin.invocation_id
        assert envelope["meta"]["status"] == "paused"
        state = envelope["acp_state"]
        assert state["attempts"] == []  # open_session never returned its handshake to the controller.
        rows = state["translated_updates"]
        permission = [row for row in rows if row["update"]["sessionUpdate"] == "permission_request"]
        assert len(permission) == 1
        assert permission[0]["attempt"] == 1
        assert permission[0]["update"]["rawInput"] == {"phase": "new"}
        assert permission[0]["update"]["outcome"] == "allowed"
        prose = [row for row in rows if row["update"]["sessionUpdate"] == "agent_message_chunk"]
        assert len(prose) == 1
        assert prose[0]["attempt"] == 1
        assert prose[0]["update"]["content"]["text"] == "before open_session returned"
    finally:
        await bus.publish(UserInterrupt())
        await await_run_task_chain(
            engine,
            turn_state=engine.turns.turn_state,
            timeout=ENGINE_TURN_TIMEOUT,
            propagate_inner_cancel=True,
        )
