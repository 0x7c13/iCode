# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real build replacement and turn drainage retain their resource authority."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.types import UserMessage
from chrys.kernel import Agent
from chrys.orchestration.engine.build import builder
from chrys.orchestration.invoker.resources import TurnTaskBinding
from chrys.orchestration.sub_agents import tools as child_tools
from chrys.service.llm.clients import create_client
from chrys.service.llm.mock import MockResponse
from tests.orchestration.invoker._build_fixtures import CountedClient, build_recipe_engine
from tests.support.close_races import assert_entered_before_completion
from tests.support.engines import AgentEngineFactory
from tests.support.waiting import await_run_task_chain


async def test_real_rebuild_cancel_after_install_still_releases_displaced_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory
) -> None:
    engine, old_main, old_child = await build_recipe_engine(
        agent_engine, monkeypatch, tmp_path, main=[MockResponse(text="old answer")], child=[]
    )
    await engine.event_bus.publish(UserMessage(text="preserved question"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state, expect_installed=True)
    old_messages = [
        (message.role, message.text) for message in engine.current.loaded.bindings.backend.history_state["messages"]
    ]
    old_prepared, old_executor, old_agent = (
        engine.current.loaded.prepared,
        engine.current.loaded.bindings,
        engine.current.loaded.agent,
    )
    old_tools, old_conversation = engine.current.loaded.sub_agent_tools, engine.current.loaded.conversation
    old_main.exit_release = asyncio.Event()
    new_main, new_child = CountedClient([]), CountedClient([])

    def replacement(profile, **kwargs):
        return new_main if profile.id == "root-model" else new_child

    monkeypatch.setattr(builder, "create_client", create_autospec(create_client, side_effect=replacement))
    monkeypatch.setattr(child_tools, "create_client", create_autospec(create_client, side_effect=replacement))
    rebuild = asyncio.create_task(engine.loader.reload(engine.session.agent_profile))
    try:
        await asyncio.wait_for(old_main.exit_entered.wait(), 5)
        assert engine.current.loaded.prepared is not old_prepared
        assert engine.current.loaded.bindings is not old_executor
        assert engine.current.loaded.agent is not old_agent
        assert engine.current.loaded.sub_agent_tools is not old_tools
        assert engine.current.loaded.conversation is not old_conversation
        assert [
            (message.role, message.text)
            for message in engine.current.loaded.bindings.backend.history_state["messages"][: len(old_messages)]
        ] == old_messages
        assert old_main.exits == old_child.exits == 0
        assert new_main.enters == new_child.enters == 1
        rebuild.cancel()
        rebuild.cancel()
        old_main.exit_release.set()
        with pytest.raises(asyncio.CancelledError):
            await rebuild
        assert old_main.exits == old_child.exits == 1
        assert new_main.exits == new_child.exits == 0
        await old_prepared.aclose()
        assert old_main.exits == old_child.exits == 1
    finally:
        old_main.exit_release.set()
        await asyncio.gather(rebuild, return_exceptions=True)
        await engine.shutdown()
    assert new_main.exits == new_child.exits == 1


async def test_turn_binding_drains_final_save_before_shared_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory
) -> None:
    engine, main, child = await build_recipe_engine(
        agent_engine, monkeypatch, tmp_path, main=[MockResponse(text="done")], child=[]
    )
    saving, cancelled_save, finish_save = asyncio.Event(), asyncio.Event(), asyncio.Event()
    drain_entered = asyncio.Event()
    original_drain = TurnTaskBinding._drain

    async def observed_drain(self: TurnTaskBinding) -> None:
        drain_entered.set()
        await original_drain(self)

    monkeypatch.setattr(TurnTaskBinding, "_drain", observed_drain)
    original_save = engine.writer.save_current_session
    save_exit_counts: list[tuple[int, int]] = []

    async def save() -> bool:
        saving.set()
        try:
            await finish_save.wait()
        except asyncio.CancelledError:
            cancelled_save.set()
            await finish_save.wait()
        result = await original_save()
        save_exit_counts.append((main.exits, child.exits))
        return result

    monkeypatch.setattr(engine.writer, "save_current_session", create_autospec(original_save, side_effect=save))
    await engine.event_bus.publish(UserMessage(text="work"))
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(saving.wait(), 5)
        operation = engine._turns.run_task
        assert operation is not None
        closing = asyncio.create_task(engine.current.loaded.prepared.aclose())
        await asyncio.wait_for(cancelled_save.wait(), 5)
        await assert_entered_before_completion(drain_entered, closing)
        assert main.exits == child.exits == 0
        assert not closing.done()
        finish_save.set()
        await asyncio.wait_for(closing, 5)
        assert operation.done()
        assert save_exit_counts == [(0, 0)]
        await engine.wait_for_run_task()
        assert main.exits == child.exits == 1
    finally:
        finish_save.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await engine.shutdown()
    assert main.exits == child.exits == 1


async def test_shutdown_closes_main_and_child_agent_methods_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory
) -> None:
    exits: list[Agent] = []
    original_exit = Agent.__aexit__

    async def exit_agent(self: Agent, *args: object) -> None:
        exits.append(self)
        await original_exit(self, *args)

    # Install before build captures the release methods. Count method entry,
    # since AsyncExitStack's idempotency otherwise conceals extra authorities.
    monkeypatch.setattr(Agent, "__aexit__", create_autospec(original_exit, side_effect=exit_agent))
    engine, main, child = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, main=[], child=[])
    main_agent = engine.current.loaded.agent
    child_agent = engine.current.loaded.sub_agent_tools._agents["Explore"]
    await engine.shutdown()
    assert sum(agent is main_agent for agent in exits) == 1
    assert sum(agent is child_agent for agent in exits) == 1
    assert len(exits) == 2
    assert main.exits == child.exits == 1


async def test_prepared_close_during_initial_child_audit_drains_pre_controller_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory
) -> None:
    from chrys.service.session.sub_agent_logs import SubAgentSessionLogWriter

    engine, main, child = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, main=[], child=[])
    entered, release, close_requested = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cancellation_counts = []
    from chrys.orchestration.sub_agents.shell import SubAgentToolShell

    original_request = SubAgentToolShell.request_close

    def request(shell, cause):
        original_request(shell, cause)
        cancellation_counts.append(operation.cancelling())
        close_requested.set()

    monkeypatch.setattr(SubAgentToolShell, "request_close", create_autospec(original_request, side_effect=request))
    original_write = SubAgentSessionLogWriter.write

    async def write(*args, **kwargs):
        if kwargs.get("status") == "running":
            entered.set()
            await release.wait()
        return await original_write(*args, **kwargs)

    monkeypatch.setattr(SubAgentSessionLogWriter, "write", create_autospec(original_write, side_effect=write))
    tools = engine.current.loaded.sub_agent_tools
    operation = asyncio.create_task(tools.get_tools()[0].func(prompt="work"))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert tools._controllers == {}
        closing = asyncio.create_task(tools._prepared_by_tool["Explore"].aclose())
        await asyncio.wait_for(close_requested.wait(), 5)
        assert cancellation_counts == [1]
        await asyncio.wait_for(closing, 5)
        assert operation.done()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert tools._total_active == 0
        assert child.call_count == 0
        assert child.exits == 1
        assert main.exits == 0
    finally:
        operation.cancel()
        release.set()
        await asyncio.gather(operation, *([closing] if closing is not None else []), return_exceptions=True)
        await engine.shutdown()
