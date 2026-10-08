# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A built main agent remembers an approval the person chose to keep."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import ApprovalRequest, ApprovalResponse, UserMessage
from chrys.foundation.models.approval_reuse import ApprovalReuseOffer
from chrys.foundation.models.workspace import Workspace
from chrys.orchestration.engine.build import builder
from chrys.service.llm.mock import MockResponse
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import (
    AgentProfile,
    ApprovalConfig,
    CompactionConfig,
    ModelConfig,
    SkillsConfig,
    ToolsConfig,
)
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.state.store import JsonFileStateStore
from tests.support.engines import AgentEngineFactory
from tests.support.scripted_clients import ErrorMockChatClient
from tests.support.waiting import await_run_task_chain


def _write(call_id: str, path: Path, content: str) -> MockResponse:
    args = {"path": str(path), "content": content, "overwrite": True}
    return MockResponse(tool_calls=[("write_file", call_id, args)])


async def test_remembered_write_skips_the_next_turns_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory
) -> None:
    target = tmp_path / "notes.txt"
    client = ErrorMockChatClient(
        [
            _write("call-1", target, "first"),
            MockResponse(text="one"),
            _write("call-2", target, "second"),
            MockResponse(text="two"),
        ]
    )
    monkeypatch.setattr(builder, "create_client", create_autospec(builder.create_client, return_value=client))
    models = ModelProfileRegistry()
    models.register(ModelProfile(id="model", name="model", provider="mock", model_id="model", vision=False))
    profiles = AgentProfileRegistry()
    profile = AgentProfile(
        name="Writer",
        instructions="Write files.",
        model=ModelConfig(profile_id="model"),
        tools=ToolsConfig(builtins=["filesystem.write"]),
        skills=SkillsConfig(auto_load_user_agents_skills=False, auto_load_cwd_agents_skills=False),
        approval=ApprovalConfig(default="require"),
        compaction=CompactionConfig(enabled=False),
    )
    profiles.register(profile)
    engine = agent_engine(
        EventBus(),
        settings=Settings(
            model_profile_override="model",
            mutation_coordination=False,
            workspace_change_notice=False,
        ),
        model_registry=models,
        agent_registry=profiles,
        state_store=JsonFileStateStore(tmp_path / "sessions"),
    )
    await engine.start(profile, workspace=Workspace.from_cwd(str(tmp_path)))
    requests: list[ApprovalRequest] = []

    async def approve(event: ApprovalRequest) -> None:
        requests.append(event)
        await engine.event_bus.publish(
            ApprovalResponse(request_id=event.request_id, approved=True, remember_choice="EXACT_SESSION")
        )

    await engine.event_bus.subscribe(ApprovalRequest, approve)
    try:
        for text in ("write it", "write it again"):
            await engine.event_bus.publish(UserMessage(text=text))
            await await_run_task_chain(engine, turn_state=engine.turns.turn_state, expect_installed=True)
        assert target.read_text(encoding="utf-8") == "second"
        assert client.call_count == 4
        offer = ApprovalReuseOffer("files", (str(target.resolve()),))
        assert [event.reuse_offer for event in requests] == [offer]
    finally:
        await engine.event_bus.unsubscribe(ApprovalRequest, approve)
