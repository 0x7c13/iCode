# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Resource lifetime and manifest lifetime of an installed build."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace

import pytest

from chrys.orchestration.engine.build.loaded import LoadedAgent
from tests.support.components import make_current
from tests.support.loaded_agents import install_loaded_agent, make_loaded_agent


@pytest.mark.parametrize("field", [field.name for field in fields(LoadedAgent)])
def test_loaded_resource_fields_are_frozen(field: str) -> None:
    loaded = make_loaded_agent()
    with pytest.raises(FrozenInstanceError):
        setattr(loaded, field, None)


def test_empty_current_records_have_independent_manifests() -> None:
    left = make_current()
    right = make_current()
    assert left.loaded is None
    assert right.loaded is None
    assert left.manifest is not right.manifest
    assert left.manifest.runtime_details is not right.manifest.runtime_details


async def test_loaded_close_delegates_to_its_resource_owner() -> None:
    loaded = make_loaded_agent()
    closed: list[str] = []

    async def close_resource() -> None:
        closed.append("owned resource")

    loaded.prepared.own(close_resource)
    await loaded.aclose()
    assert closed == ["owned resource"]


async def test_shutdown_releases_resources_and_retains_the_installed_manifest() -> None:
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.orchestration.engine.assembly import assemble_agent_engine
    from tests.support.loaded_agents import install_loaded_agent, make_manifest

    engine = assemble_agent_engine(EventBus(), settings=Settings())
    manifest = make_manifest(skill_names=["retained"])
    install_loaded_agent(engine, loaded=make_loaded_agent(), manifest=manifest)
    await engine.shutdown()
    assert engine.current.loaded is None
    assert engine.current.manifest is manifest
    assert list(engine.current.manifest.skill_names) == ["retained"]
    engine.lifecycle.reset_for_restart(None)
    assert engine.current.loaded is None
    assert engine.current.manifest is manifest


async def test_required_resources_follow_replacement_and_unload() -> None:
    current = make_current()
    with pytest.raises(RuntimeError, match="has not been loaded"):
        current.require_loaded()
    owner = SimpleNamespace(current=current)
    first, second = make_loaded_agent(), make_loaded_agent()
    try:
        install_loaded_agent(owner, loaded=first)
        assert current.require_loaded() is first
        install_loaded_agent(owner, loaded=second)
        await first.aclose()
        assert current.require_loaded() is second
        install_loaded_agent(owner, loaded=None)
        with pytest.raises(RuntimeError, match="has not been loaded"):
            current.require_loaded()
    finally:
        await first.aclose()
        await second.aclose()
