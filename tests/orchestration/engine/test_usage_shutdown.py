# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Main-agent usage subscribers may request shutdown without waiting on their own publication."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import create_autospec

import pytest

import chrys.orchestration.engine.build.builder as builder_module
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import UsageUpdate, UserMessage
from chrys.foundation.models.workspace import Workspace
from chrys.kernel import UsageDetails
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import AgentProfile, CompactionConfig, ToolsConfig
from chrys.service.state.store import JsonFileStateStore
from tests.support.engines import AgentEngineFactory
from tests.support.pipeline_helpers import make_mock_settings_and_registry


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_main_usage_inline_shutdown_returns_but_external_shutdown_waits_for_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory, stream: bool
) -> None:
    bus = EventBus()
    settings, models = make_mock_settings_and_registry(stream=stream)
    profile = AgentProfile(
        name="Usage",
        instructions="Answer briefly.",
        tools=ToolsConfig(builtins=[]),
        compaction=CompactionConfig(enabled=False),
    )
    agents = AgentProfileRegistry()
    agents.register(profile)
    client = MockChatClient(
        responses=[MockResponse(text="done", usage_details=UsageDetails(input_token_count=10, output_token_count=5))]
    )
    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=client)
    )
    engine = agent_engine(
        bus,
        settings=settings,
        model_registry=models,
        agent_registry=agents,
        state_store=JsonFileStateStore(tmp_path / "sessions"),
        initial_workspace=Workspace.from_cwd(str(tmp_path)),
    )
    returned = asyncio.Event()
    release_handler = asyncio.Event()
    external_entered = asyncio.Event()
    delivered: list[UsageUpdate] = []
    external: asyncio.Task[None] | None = None

    async def shutdown_inline(event: UsageUpdate) -> None:
        if event.total_tokens != 15 or event.usage_source_id != engine.session.session_id:
            return
        assert engine.workflows.active_run_id is None
        assert asyncio.current_task() in engine._usage_publisher.tasks
        await engine.shutdown()
        returned.set()
        await release_handler.wait()
        delivered.append(event)

    async def shutdown_external() -> None:
        external_entered.set()
        await engine.shutdown()

    await bus.subscribe(UsageUpdate, shutdown_inline)
    try:
        await engine.start(profile)
        await bus.publish(UserMessage(text="Report usage.", session_id=engine.session.session_id))
        await asyncio.wait_for(returned.wait(), 5)
        owned_release = engine._release_task
        assert owned_release is not None and not owned_release.done()
        external = asyncio.create_task(shutdown_external())
        await asyncio.wait_for(external_entered.wait(), 5)
        assert not external.done()
        assert engine._release_task is owned_release
        release_handler.set()
        await asyncio.wait_for(external, 10)
        assert len(delivered) == 1
        assert owned_release.done() and not owned_release.cancelled()
        assert owned_release.exception() is None
        assert engine._usage_publisher.tasks == set()
        assert engine._usage_publisher.tail is None
        assert engine.current.loaded is None
        assert engine.turns.turn_state.lease.run_task is None
    finally:
        release_handler.set()
        if not returned.is_set():
            # Let a failing pre-fix self-wait unwind so the fixture can still close the real engine.
            for task in tuple(engine._usage_publisher.tasks):
                task.cancel()
        await asyncio.wait_for(engine.shutdown(), 10)
        if external is not None:
            await asyncio.gather(external, return_exceptions=True)
