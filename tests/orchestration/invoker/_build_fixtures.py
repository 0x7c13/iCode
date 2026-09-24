# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real composition-root builds with counted, offline main and child clients."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.models.workspace import Workspace
from chrys.orchestration.engine.build import builder
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.sub_agents import tools as child_tools
from chrys.service.llm.mock import MockResponse
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import (
    AgentProfile,
    ApprovalConfig,
    CompactionConfig,
    MemoryConfig,
    ModelConfig,
    SkillsConfig,
    SubAgentRef,
    SubAgentsConfig,
    ToolsConfig,
)
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.state.store import JsonFileStateStore
from tests.support.engines import AgentEngineFactory
from tests.support.scripted_clients import ErrorMockChatClient


class CountedClient(ErrorMockChatClient):
    """Count real context entries/exits and expose an exit barrier."""

    def __init__(self, outcomes: list[MockResponse | BaseException]) -> None:
        super().__init__(outcomes)
        self.enters = 0
        self.exits = 0
        self.exit_entered = asyncio.Event()
        self.exit_release: asyncio.Event | None = None

    async def __aenter__(self) -> CountedClient:
        self.enters += 1
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exit_entered.set()
        if self.exit_release is not None:
            await self.exit_release.wait()
        self.exits += 1


async def build_recipe_engine(
    agent_engine: AgentEngineFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    main: list[MockResponse | BaseException],
    child: list[MockResponse | BaseException],
    stream: bool = False,
) -> tuple[AgentEngine, CountedClient, CountedClient]:
    """Keep model selection, tools, approvals, context and runtime factories real."""
    (tmp_path / "ROOT.md").write_text("ROOT-MEMORY-SENTINEL", encoding="utf-8")
    models = ModelProfileRegistry()
    for name in ("root-model", "child-model"):
        models.register(ModelProfile(id=name, name=name, provider="mock", model_id=name, stream=stream, vision=False))
    skills = SkillsConfig(auto_load_user_agents_skills=False, auto_load_cwd_agents_skills=False)
    root = AgentProfile(
        name="Root",
        instructions="ROOT-INSTRUCTION",
        model=ModelConfig(profile_id="child-model"),
        memory=MemoryConfig(files=["ROOT.md"]),
        tools=ToolsConfig(builtins=[]),
        skills=skills,
        approval=ApprovalConfig(default="skip", overrides={"read_file": "require"}),
        compaction=CompactionConfig(enabled=False),
        sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Explore", tool_name="Explore", max_concurrency=2)]),
    )
    child_profile = AgentProfile(
        name="Explore",
        instructions="CHILD-INSTRUCTION",
        model=ModelConfig(profile_id="child-model"),
        tools=ToolsConfig(builtins=["filesystem.read"]),
        skills=skills,
        approval=ApprovalConfig(default="skip"),
        compaction=CompactionConfig(enabled=False),
    )
    profiles = AgentProfileRegistry()
    profiles.register(root)
    profiles.register(child_profile)
    main_client, child_client = CountedClient(main), CountedClient(child)

    def client_for(profile: ModelProfile, **kwargs: object) -> CountedClient:
        client = main_client if profile.id == "root-model" else child_client
        client._on_intermediate_text_async = kwargs["on_intermediate_text_async"]
        client._on_intermediate_text_sync = kwargs["on_intermediate_text_sync"]
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
    return engine, main_client, child_client
