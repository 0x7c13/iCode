# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Build installation helpers for engine and narrow host fixtures."""

from __future__ import annotations

from dataclasses import fields, replace
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock

from chrys.orchestration.engine.build.loaded import AgentManifest, LoadedAgent
from chrys.orchestration.engine.loader import AgentLoader
from chrys.orchestration.invoker.resources import Conversation, PreparedAgent
from chrys.service.agent_middleware.injection import InjectionMiddleware
from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware


def make_loaded_agent(**overrides) -> LoadedAgent:
    """Create a build record with independently owned test resources."""
    values = {
        "prepared": PreparedAgent(),
        "conversation": Conversation(),
        "agent": MagicMock(),
        "bindings": MagicMock(state=SimpleNamespace(running=False, run_failed=False, was_interrupted=False)),
        "runtime": MagicMock(),
        "injection": InjectionMiddleware(),
        "consumed_injections": [],
        "intermediate_texts": {},
        "loop_recorder": MagicMock(),
        "reminder_middleware": SystemReminderMiddleware(),
        "approval_judge": MagicMock(),
        "sub_agent_tools": None,
        "skills_provider": None,
        "mcp_adapter": None,
    }
    values.update(overrides)
    return LoadedAgent(**values)


def make_manifest(**overrides) -> AgentManifest:
    """Create the last-build manifest for a test."""
    for name in ("tool_names", "skill_names", "memory_files"):
        if name in overrides:
            overrides[name] = tuple(overrides[name])
    if "tool_kinds" in overrides:
        overrides["tool_kinds"] = MappingProxyType(dict(overrides["tool_kinds"]))
    return replace(AgentManifest.empty(), **overrides)


def install_loaded_agent(engine, *, loaded=..., manifest=..., **overrides) -> None:
    """Replace resource or manifest records while preserving unrelated fixture values."""
    current = engine.current
    resource_names = {field.name for field in fields(LoadedAgent)}
    resources = {name: value for name, value in overrides.items() if name in resource_names}
    details = {name: value for name, value in overrides.items() if name not in resource_names}
    if loaded is not ...:
        current.loaded = loaded
    elif resources:
        if isinstance(current.loaded, SimpleNamespace):
            current.loaded = SimpleNamespace(**(vars(current.loaded) | resources))
        elif current.loaded is None:
            current.loaded = make_loaded_agent(**resources)
        else:
            current.loaded = replace(current.loaded, **resources)
    if manifest is not ...:
        current.manifest = manifest
    elif details:
        if isinstance(current.manifest, SimpleNamespace):
            current.manifest = SimpleNamespace(**(vars(current.manifest) | details))
        else:
            current.manifest = make_manifest(
                **(
                    {
                        "tool_names": current.manifest.tool_names,
                        "tool_kinds": current.manifest.tool_kinds,
                        "skill_names": current.manifest.skill_names,
                        "memory_files": current.manifest.memory_files,
                        "agent_profile_fingerprint": current.manifest.agent_profile_fingerprint,
                        "model_profile_fingerprint": current.manifest.model_profile_fingerprint,
                        "runtime_details": current.manifest.runtime_details,
                        "active_profile": current.manifest.active_profile,
                    }
                    | details
                )
            )


class SkillRefreshLoader:
    """The manifest-refresh capability used by narrow turn-host fixtures."""

    apply_skill_refresh = AgentLoader.apply_skill_refresh

    def __init__(self, current) -> None:
        self._current = current
