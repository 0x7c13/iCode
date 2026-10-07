# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""An engine whose main agent, sub-agent and judge build real OpenAI client stacks.

Every model profile points at a loopback address (by default one nothing
listens on): builds never send a request, so the HTTP pools
:mod:`tests.support.llm_http_clients` records are exactly the ones each owner
opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import ModelProfileSwitched, SetModelProfile
from chrys.foundation.models.workspace import Workspace
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
from tests.support.event_capture import capture_events
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

if TYPE_CHECKING:
    from pathlib import Path

    from chrys.orchestration.engine.engine import AgentEngine
    from tests.support.engines import AgentEngineFactory

MAIN_MODEL = "main-model"
ALT_MODEL = "alt-model"
SUB_MODEL = "sub-model"
SUB_AGENT = "Explore"
_NO_SKILLS = SkillsConfig(auto_load_user_agents_skills=False, auto_load_cwd_agents_skills=False)


REFUSED_BASE_URL = "http://127.0.0.1:9/v1"


def loopback_model(profile_id: str, *, base_url: str = REFUSED_BASE_URL) -> ModelProfile:
    """A non-streaming OpenAI profile on a loopback endpoint; building it sends nothing."""
    return ModelProfile(
        id=profile_id,
        name=profile_id,
        provider="openai",
        model_id=f"{profile_id}-id",
        api_key="sk-test",
        base_url=base_url,
        http_max_retries=0,
        stream=False,
    )


def parent_profile(*, sub_agent: bool = True, compaction: bool = False) -> AgentProfile:
    return AgentProfile(
        name="Parent",
        instructions="Reply briefly.",
        tools=ToolsConfig(builtins=[]),
        skills=_NO_SKILLS,
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=compaction),
        sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile=SUB_AGENT, tool_name=SUB_AGENT)] if sub_agent else []),
    )


def _sub_agent_profile() -> AgentProfile:
    return AgentProfile(
        name=SUB_AGENT,
        instructions="Explore.",
        tools=ToolsConfig(builtins=[]),
        skills=_NO_SKILLS,
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
        model=ModelConfig(profile_id=SUB_MODEL),
    )


@dataclass(frozen=True, slots=True)
class ClientEngine:
    engine: AgentEngine
    bus: EventBus
    profile: AgentProfile


async def start_client_engine(
    agent_engine: AgentEngineFactory,
    tmp_path: Path,
    *,
    sub_agent: bool = True,
    base_url: str = REFUSED_BASE_URL,
    compaction: bool = False,
) -> ClientEngine:
    """Start an engine over loopback OpenAI profiles; the fixture shuts it down."""
    model_registry = ModelProfileRegistry()
    for profile_id in (MAIN_MODEL, ALT_MODEL, SUB_MODEL):
        model_registry.register(loopback_model(profile_id, base_url=base_url))
    profile = parent_profile(sub_agent=sub_agent, compaction=compaction)
    agent_registry = AgentProfileRegistry()
    agent_registry.register(profile)
    agent_registry.register(_sub_agent_profile())
    bus = EventBus()
    engine = agent_engine(
        bus,
        settings=Settings(model_profile=MAIN_MODEL, workspace_change_notice=False),
        agent_registry=agent_registry,
        model_registry=model_registry,
        initial_workspace=Workspace.from_cwd(str(tmp_path)),
    )
    await engine.start(profile)
    return ClientEngine(engine=engine, bus=bus, profile=profile)


async def switch_model(started: ClientEngine, profile_id: str) -> None:
    """Rebuild the agent on *profile_id* and wait until the engine reports the switch."""
    switched = await capture_events(started.bus, ModelProfileSwitched)
    await started.bus.publish(SetModelProfile(profile_id=profile_id))
    await wait_for(
        lambda: any(event.model_profile_id == profile_id for event in switched),
        timeout=ENGINE_TURN_TIMEOUT,
        description=f"switch to {profile_id}",
    )


async def open_judge_client(started: ClientEngine) -> None:
    """Make the current build's approval judge create its client, as its first evaluation does."""
    await started.engine.current.require_loaded().approval_judge._get_client()
