# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real cancellation commits and durable writes gate parent pending cleanup."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Event as ThreadEvent
from unittest.mock import create_autospec

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationPaused, UserInterrupt, UserMessage
from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.kernel import Agent, AgentSession, Message
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.invoker.resources import Conversation
from chrys.orchestration.sub_agents import tools as tools_module
from chrys.orchestration.sub_agents.kernel_policy import KernelSubAgentPolicy
from chrys.orchestration.sub_agents.tools import SubAgentTools
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.agent_middleware.control.ask_user import AskUserMiddleware
from chrys.service.agent_middleware.injection import InjectionMiddleware
from chrys.service.approval.policy import ApprovalPolicy
from chrys.service.llm.mock import MockChatClient
from chrys.service.profiles.agents.schema import ApprovalConfig
from chrys.service.session import sub_agent_logs
from chrys.service.session.message_metadata import TOOL_RESULT_METADATA_KEY
from chrys.service.session.sub_agent_logs import SubAgentSessionLogWriter
from chrys.service.state.store import JsonFileStateStore
from tests.orchestration.sub_agents.test_integration import _make_ctx, _sub_tool_call
from tests.support.engines import AgentEngineFactory
from tests.support.loaded_agents import install_loaded_agent
from tests.support.scripted_clients import FrameworkBoom
from tests.support.waiting import (
    ENGINE_TEST_WAIT_TIMEOUT,
    ENGINE_TURN_TIMEOUT,
    await_run_task_chain,
    wait_for,
    with_wait_deadline,
)


@pytest.mark.parametrize("write_ok", [True, False])
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_cascade_commits_before_repair_and_terminal_writer_upgrades_real_raw_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory, write_ok: bool
) -> None:
    # Keep real parent preparation independent of checkout size and audit writes.
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    ctx = await _make_ctx(
        tmp_path,
        main_outcomes=[_sub_tool_call("work")],
        sub_outcomes=[FrameworkBoom("pause")],
        agent_engine=agent_engine,
    )
    engine = ctx.engine
    order: list[str] = []
    snapshots: list[dict] = []
    original_repair = KernelSubAgentPolicy._repair_paused_history
    original_commit = tools_module._commit_interrupted_parent_result
    original_write = SubAgentSessionLogWriter.write
    original_upgrade = tools_module._upgrade_interrupted_parent_result
    original_atomic = sub_agent_logs._atomic_write_json

    def atomic(path, envelope):
        if not write_ok and envelope["meta"]["status"] == "cascade_aborted":
            raise OSError("terminal audit disk failure")
        return original_atomic(path, envelope)

    monkeypatch.setattr(sub_agent_logs, "_atomic_write_json", create_autospec(original_atomic, side_effect=atomic))

    def commit(metadata):
        was_committed = metadata.interrupted_result_committed if metadata is not None else False
        original_commit(metadata)
        if not was_committed and metadata is not None and metadata.interrupted_result_committed:
            order.append("parent-commit")

    def repair(controller):
        if controller._shell.cascade_requested:
            order.append("repair")
        return original_repair(controller)

    async def write(*args, **kwargs):
        if kwargs["status"] == "cascade_aborted":
            order.append("writer-enter")
            result = await original_write(*args, **kwargs)
            order.append("writer-success" if result else "writer-failed")
            return result
        return await original_write(*args, **kwargs)

    def upgrade(metadata):
        original_upgrade(metadata)
        order.append("upgrade")

    monkeypatch.setattr(
        tools_module, "_commit_interrupted_parent_result", create_autospec(original_commit, side_effect=commit)
    )
    monkeypatch.setattr(
        tools_module, "_upgrade_interrupted_parent_result", create_autospec(original_upgrade, side_effect=upgrade)
    )
    monkeypatch.setattr(
        KernelSubAgentPolicy, "_repair_paused_history", create_autospec(original_repair, side_effect=repair)
    )
    monkeypatch.setattr(SubAgentSessionLogWriter, "write", create_autospec(original_write, side_effect=write))
    # Observe the actual recorder's raw result as it is committed by the kernel,
    # before final history repair can hide an omitted terminal upgrade.
    from chrys.kernel import LoopRecorder

    original_record = LoopRecorder._interrupt_slot

    def record(*args, **kwargs):
        result = original_record(*args, **kwargs)
        # Signature binding keeps this tied to the production result argument.
        from inspect import signature

        bound = signature(original_record).bind(*args, **kwargs).arguments
        content = bound["slot"].result
        assert content is not None
        if content.call_id == "call-1":
            snapshots.append(dict(content.additional_properties.get(TOOL_RESULT_METADATA_KEY, {})))
        return result

    monkeypatch.setattr(LoopRecorder, "_interrupt_slot", create_autospec(original_record, side_effect=record))
    try:
        await ctx.bus.publish(UserMessage(text="delegate"))
        await wait_for(
            lambda: any((isinstance(e, InvocationPaused) and e.origin.kind == "sub_agent") for e in ctx.events),
            timeout=ENGINE_TURN_TIMEOUT,
            description="child paused before cascade",
        )
        order.clear()
        await ctx.bus.publish(UserInterrupt())
        await await_run_task_chain(
            engine,
            turn_state=engine.turns.turn_state,
            timeout=ENGINE_TURN_TIMEOUT,
            propagate_inner_cancel=True,
        )
        assert order.index("parent-commit") < order.index("repair") < order.index("writer-enter")
        assert bool("upgrade" in order) is write_ok
        if write_ok:
            assert order.index("writer-success") < order.index("upgrade")
        assert snapshots
        assert snapshots[0].get("sub_agent_audit_complete") is not True
        assert (snapshots[-1].get("sub_agent_audit_complete") is True) is write_ok
        assert snapshots[-1]["sub_agent_invocation_id"]
        if write_ok:
            assert snapshots[-1]["sub_agent_log_file"]
    finally:
        try:
            # A failed pause wait must still release the child's decision future.
            await ctx.bus.publish(UserInterrupt())
            await await_run_task_chain(
                engine,
                turn_state=engine.turns.turn_state,
                timeout=ENGINE_TURN_TIMEOUT,
                propagate_inner_cancel=True,
            )
        finally:
            await ctx.cleanup()


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_pending_survives_real_store_write_until_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory, outcome: str
) -> None:
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    engine = agent_engine(bus, settings=Settings(), state_store=store)
    engine.session.session_id = "pending-save"
    session = AgentSession()
    session.state["chrys_history"] = {
        "messages": [Message("user", ["saved"])],
        "compressed_msgs": [],
        "turn_counter": 1,
    }
    executor = TurnBindings(
        conversation=Conversation(),
        agent=Agent(client=MockChatClient()),
        session=session,
        event_bus=bus,
        approval_middleware=ApprovalMiddleware(ApprovalPolicy(ApprovalConfig()), bus),
        ask_user_middleware=AskUserMiddleware(bus),
        injection_middleware=InjectionMiddleware(),
    )
    executor.resource_scope.own(executor.approval.close)
    install_loaded_agent(engine, bindings=executor)
    root = store.session_dir("pending-save")
    tools = SubAgentTools(event_bus=bus, session_id="pending-save", session_dir=root)
    install_loaded_agent(engine, sub_agent_tools=tools)
    pending = root / "sub_agents" / "pending" / "done.json"
    pending.parent.mkdir(parents=True)
    atomic_write_owner_only_text(pending, "{}")
    tools.queue_pending_cleanup(pending)
    entered, release, finished = ThreadEvent(), ThreadEvent(), ThreadEvent()
    original = store._save_session_sync

    def write(*args, **kwargs):
        entered.set()
        try:
            if not release.wait(10):
                raise TimeoutError("test did not release persistence write")
            if outcome == "failure":
                raise OSError("injected store write failure")
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(store, "_save_session_sync", create_autospec(original, side_effect=write))
    task = asyncio.create_task(engine.writer.save_current_session())
    try:
        await wait_for(entered.is_set, description="real persistence thread reached write latch")
        assert pending.exists()
        assert not (root / "session.json").exists()
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert pending.exists()
        release.set()
        await wait_for(finished.is_set, description="persistence thread drained")
        if outcome != "cancel":
            assert await task is (outcome == "success")
        assert pending.exists() is (outcome != "success")
        if outcome == "success":
            assert json.loads((root / "session.json").read_text())["state"]["messages"]
    finally:
        release.set()
        await wait_for(finished.is_set, description="persistence thread teardown")
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        install_loaded_agent(engine, loaded=None)
        await executor.resource_scope.aclose()
