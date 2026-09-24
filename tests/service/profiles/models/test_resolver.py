# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for model profile resolution."""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from chrys.foundation.config.settings import Settings
from chrys.service.profiles.agents.schema import AgentProfile, ModelConfig
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.resolver import (
    ModelSelection,
    resolve_active_profile,
    resolve_for_agent,
    resolve_selectable_profile,
    resolve_selection_for_agent,
    settings_with_active_model_profile,
)
from chrys.service.profiles.models.schema import UNCONFIGURED_MODEL_ID, ModelProfile


def _make_profile(pid: str, name: str, model_id: str = "gpt-test") -> ModelProfile:
    return ModelProfile(id=pid, name=name, provider="openai", model_id=model_id)


@pytest.fixture
def registry() -> ModelProfileRegistry:
    r = ModelProfileRegistry()
    r.register(_make_profile("active-id", "Active Profile"))
    r.register(_make_profile("override-id", "Override Profile"))
    return r


@pytest.fixture
def settings() -> Settings:
    return Settings(model_profile="active-id")


def test_no_override_returns_active(registry: ModelProfileRegistry, settings: Settings) -> None:
    agent = AgentProfile(name="a")
    profile = resolve_for_agent(registry, settings, agent)
    assert profile.id == "active-id"


def test_active_profile_accepts_unique_profile_name(registry: ModelProfileRegistry) -> None:
    profile = resolve_active_profile(registry, Settings(model_profile="Active Profile"))

    assert profile.id == "active-id"


def test_agent_model_binding_wins_over_active_profile(registry: ModelProfileRegistry, settings: Settings) -> None:
    agent = AgentProfile(name="a", model=ModelConfig(profile_id="override-id"))
    profile = resolve_for_agent(registry, settings, agent)
    assert profile.id == "override-id"


def test_cli_active_model_selection_does_not_override_agent_binding(
    registry: ModelProfileRegistry,
    settings: Settings,
) -> None:
    selected = registry.get("active-id")
    assert selected is not None
    selected_settings = settings_with_active_model_profile(settings, selected)
    agent = AgentProfile(name="a", model=ModelConfig(profile_id="override-id"))

    profile = resolve_for_agent(registry, selected_settings, agent)

    assert profile.id == "override-id"


def test_explicit_settings_override_wins_over_agent_binding(registry: ModelProfileRegistry) -> None:
    settings = Settings(model_profile="override-id", model_profile_override="active-id")
    agent = AgentProfile(name="a", model=ModelConfig(profile_id="override-id"))

    profile = resolve_for_agent(registry, settings, agent)

    assert profile.id == "active-id"


def test_explicit_settings_override_accepts_unique_profile_name(registry: ModelProfileRegistry) -> None:
    settings = Settings(model_profile_override="override profile")
    agent = AgentProfile(name="a")

    profile = resolve_for_agent(registry, settings, agent)

    assert profile.id == "override-id"


def test_explicit_settings_override_does_not_replace_sub_agent_binding_by_default(
    registry: ModelProfileRegistry,
) -> None:
    parent = _make_profile("parent-id", "Parent Profile")
    settings = Settings(model_profile_override="active-id")
    agent = AgentProfile(name="sub", model=ModelConfig(profile_id="override-id"))

    profile = resolve_for_agent(registry, settings, agent, fallback=parent)

    assert profile.id == "override-id"


def test_explicit_settings_override_can_force_sub_agent_model(registry: ModelProfileRegistry) -> None:
    parent = _make_profile("parent-id", "Parent Profile")
    settings = Settings(model_profile_override="active-id", model_profile_override_sub_agents=True)
    agent = AgentProfile(name="sub", model=ModelConfig(profile_id="override-id"))

    profile = resolve_for_agent(registry, settings, agent, fallback=parent)

    assert profile.id == "active-id"


def test_active_model_selection_does_not_override_sub_agent_binding(registry: ModelProfileRegistry) -> None:
    parent = _make_profile("parent-id", "Parent Profile")
    settings = Settings(model_profile="active-id")
    agent = AgentProfile(name="sub", model=ModelConfig(profile_id="override-id"))

    profile = resolve_for_agent(registry, settings, agent, fallback=parent)

    assert profile.id == "override-id"


def test_override_missing_falls_back_to_active(
    registry: ModelProfileRegistry, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Deleted override id: fall back to active profile, no throw, warn once."""
    agent = AgentProfile(name="a", model=ModelConfig(profile_id="deleted-id"))
    with caplog.at_level(logging.WARNING, logger="chrys.service.profiles.models.resolver"):
        profile = resolve_for_agent(registry, settings, agent)
    assert profile.id == "active-id"
    assert any("deleted-id" in r.message for r in caplog.records)


def test_override_missing_falls_back_to_fallback(registry: ModelProfileRegistry, settings: Settings) -> None:
    """Sub-agent path: missing override → fallback (parent), not active."""
    parent = _make_profile("parent-id", "Parent Profile")
    agent = AgentProfile(name="sub", model=ModelConfig(profile_id="deleted-id"))
    profile = resolve_for_agent(registry, settings, agent, fallback=parent)
    assert profile.id == "parent-id"


def test_fallback_used_when_no_override(registry: ModelProfileRegistry, settings: Settings) -> None:
    """Sub-agent without override inherits fallback, not active."""
    parent = _make_profile("parent-id", "Parent Profile")
    agent = AgentProfile(name="sub")
    profile = resolve_for_agent(registry, settings, agent, fallback=parent)
    assert profile.id == "parent-id"


def test_no_registry_returns_default(settings: Settings) -> None:
    """Without a registry, resolution falls to the hardcoded default profile."""
    agent = AgentProfile(name="a", model=ModelConfig(profile_id="override-id"))
    profile = resolve_for_agent(None, settings, agent)
    # Mirrors resolve_active_profile's no-registry behaviour.
    assert profile == resolve_active_profile(None, settings)


def test_resolve_selectable_profile_filters_hollow_and_dangling(registry: ModelProfileRegistry) -> None:
    """Only a pointer that lands on a structurally complete profile counts."""
    registry.register(ModelProfile(id="hollow-id", name="Hollow"))

    assert resolve_selectable_profile(registry, "active-id").id == "active-id"
    assert resolve_selectable_profile(registry, "Active Profile").id == "active-id"
    assert resolve_selectable_profile(registry, "hollow-id") is None
    assert resolve_selectable_profile(registry, "ghost-id") is None
    assert resolve_selectable_profile(registry, "") is None
    assert resolve_selectable_profile(None, "active-id") is None


def _assert_selection(
    registry: ModelProfileRegistry | None,
    settings: Settings,
    agent: AgentProfile,
    *,
    expected_profile_id: str,
    expected_source: str,
    fallback: ModelProfile | None = None,
) -> ModelSelection:
    selection = resolve_selection_for_agent(registry, settings, agent, fallback=fallback)

    assert selection.profile.id == expected_profile_id
    assert selection.source == expected_source
    assert resolve_for_agent(registry, settings, agent, fallback=fallback) == selection.profile
    return selection


def test_selection_source_is_override_for_registered_override(registry: ModelProfileRegistry) -> None:
    selection = _assert_selection(
        registry,
        Settings(model_profile="active-id", model_profile_override="override-id"),
        AgentProfile(name="a", model=ModelConfig(profile_id="active-id")),
        expected_profile_id="override-id",
        expected_source="override",
    )

    assert selection.profile is registry.get("override-id")


def test_selection_source_stays_override_when_override_profile_was_deleted(
    registry: ModelProfileRegistry,
) -> None:
    selection = _assert_selection(
        registry,
        Settings(model_profile="active-id", model_profile_override="deleted-id"),
        AgentProfile(name="a", model=ModelConfig(profile_id="override-id")),
        expected_profile_id="",
        expected_source="override",
    )

    assert selection.profile.model_id == UNCONFIGURED_MODEL_ID


def test_selection_source_is_agent_for_registered_binding(
    registry: ModelProfileRegistry,
    settings: Settings,
) -> None:
    _assert_selection(
        registry,
        settings,
        AgentProfile(name="a", model=ModelConfig(profile_id="override-id")),
        expected_profile_id="override-id",
        expected_source="agent",
    )


def test_deleted_agent_binding_falls_through_to_active_source(
    registry: ModelProfileRegistry,
    settings: Settings,
) -> None:
    _assert_selection(
        registry,
        settings,
        AgentProfile(name="a", model=ModelConfig(profile_id="deleted-id")),
        expected_profile_id="active-id",
        expected_source="active",
    )


def test_selection_source_is_inherited_for_sub_agent_fallback(
    registry: ModelProfileRegistry,
    settings: Settings,
) -> None:
    parent = _make_profile("parent-id", "Parent Profile")

    selection = _assert_selection(
        registry,
        settings,
        AgentProfile(name="sub"),
        expected_profile_id="parent-id",
        expected_source="inherited",
        fallback=parent,
    )

    assert selection.profile is parent


def test_selection_source_is_inherited_when_override_skips_sub_agents(
    registry: ModelProfileRegistry,
) -> None:
    parent = _make_profile("parent-id", "Parent Profile")

    selection = _assert_selection(
        registry,
        Settings(model_profile="active-id", model_profile_override="override-id"),
        AgentProfile(name="sub"),
        expected_profile_id="parent-id",
        expected_source="inherited",
        fallback=parent,
    )

    assert selection.profile is parent


def test_selection_source_is_active_for_registered_active_profile(
    registry: ModelProfileRegistry,
    settings: Settings,
) -> None:
    _assert_selection(
        registry,
        settings,
        AgentProfile(name="a"),
        expected_profile_id="active-id",
        expected_source="active",
    )


def test_selection_source_is_default_when_active_profile_is_missing(registry: ModelProfileRegistry) -> None:
    selection = _assert_selection(
        registry,
        Settings(model_profile="deleted-id"),
        AgentProfile(name="a"),
        expected_profile_id="",
        expected_source="default",
    )

    assert selection.profile.model_id == UNCONFIGURED_MODEL_ID


def test_selection_source_is_default_when_active_profile_is_hollow(registry: ModelProfileRegistry) -> None:
    """An active pointer at a stored-but-incomplete profile must not confirm
    a blank model_id — that would slip past the send guard whenever another
    selectable profile exists in the registry."""
    registry.register(ModelProfile(id="hollow-id", name="Hollow"))

    selection = _assert_selection(
        registry,
        Settings(model_profile="hollow-id"),
        AgentProfile(name="a"),
        expected_profile_id="",
        expected_source="default",
    )

    assert selection.profile.model_id == UNCONFIGURED_MODEL_ID


# ─── resolve_session_title_profile ──────────────────────────────────


def test_session_title_profile_falls_back_when_unset(registry: ModelProfileRegistry, settings: Settings) -> None:
    from chrys.service.profiles.models.resolver import resolve_session_title_profile

    fallback = _make_profile("agent-1", "Agent")
    assert resolve_session_title_profile(registry, settings, fallback) is fallback


def test_session_title_profile_prefers_env_selector(registry: ModelProfileRegistry, settings: Settings) -> None:
    from chrys.service.profiles.models.resolver import resolve_session_title_profile

    settings = replace(settings, session_title_model_profile="override-id")
    fallback = _make_profile("agent-1", "Agent")
    assert resolve_session_title_profile(registry, settings, fallback).id == "override-id"


def test_session_title_profile_accepts_unique_name(registry: ModelProfileRegistry, settings: Settings) -> None:
    from chrys.service.profiles.models.resolver import resolve_session_title_profile

    settings = replace(settings, session_title_model_profile="Override Profile")
    fallback = _make_profile("agent-1", "Agent")
    assert resolve_session_title_profile(registry, settings, fallback).id == "override-id"


def test_session_title_profile_missing_selector_falls_back(registry: ModelProfileRegistry, settings: Settings) -> None:
    from chrys.service.profiles.models.resolver import resolve_session_title_profile

    settings = replace(settings, session_title_model_profile="does-not-exist")
    fallback = _make_profile("agent-1", "Agent")
    assert resolve_session_title_profile(registry, settings, fallback) is fallback


def test_session_title_profile_no_registry_falls_back(settings: Settings) -> None:
    from chrys.service.profiles.models.resolver import resolve_session_title_profile

    settings = replace(settings, session_title_model_profile="override-id")
    fallback = _make_profile("agent-1", "Agent")
    assert resolve_session_title_profile(None, settings, fallback) is fallback
