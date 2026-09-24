# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Candidate construction and subscription ordering before installation."""

from __future__ import annotations

import copy
from unittest.mock import PropertyMock, create_autospec

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.orchestration.engine.assembly import assemble_agent_engine
from chrys.orchestration.engine.build import construction
from chrys.orchestration.engine.build.builder import AgentBuildResult
from chrys.service.profiles.agents.schema import AgentProfile
from tests.orchestration.engine.build.test_lifecycle_close import (
    _build_through_loader,
    _FakeExecutor,
    _fresh_approval,
    _LoaderFixture,
    _make_build_result,
    _stage,
)
from tests.support.engines import AgentEngineFactory


@pytest.mark.parametrize("failure", ["builder", "candidate_wiring"])
async def test_candidate_failure_preserves_live_objects(failure: str, monkeypatch: pytest.MonkeyPatch) -> None:
    bus = EventBus()
    engine = _LoaderFixture(bus, _FakeExecutor(_fresh_approval(bus)))
    engine.current.loaded.intermediate_texts[3] = "previous build"
    original_fields = vars(engine).copy()
    original_session = vars(engine.session).copy()
    original_texts = copy.copy(engine.current.loaded.intermediate_texts)
    candidate = _make_build_result(_fresh_approval(bus))
    if failure == "candidate_wiring":
        monkeypatch.setattr(
            type(candidate.loop_recorder),
            "on_pre_wire_barrier",
            PropertyMock(side_effect=RuntimeError("candidate wiring failed")),
            raising=False,
        )

    async def make_candidate(**_kwargs: object) -> AgentBuildResult:
        if failure == "builder":
            raise RuntimeError("builder failed")
        return candidate

    try:
        with pytest.raises(RuntimeError, match="failed"):
            await _build_through_loader(
                engine, AgentProfile(name="Code"), staged=_stage(engine), build_agent_fn=make_candidate
            )
        assert vars(engine) == original_fields
        assert vars(engine.session) == original_session
        assert engine.current.loaded.intermediate_texts == original_texts
        assert engine.current.loaded.bindings.closed is False
    finally:
        await candidate.prepared.aclose()
        await engine.current.loaded.prepared.aclose()


async def test_staging_failure_preserves_live_build(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = assemble_agent_engine(EventBus(), settings=Settings())
    engine.session.agent_profile = AgentProfile(name="Code")
    original_session = vars(engine.session).copy()
    original_settings = engine.loaded_settings
    original_generation = engine.build_generation
    failure = create_autospec(engine.loader.stage, side_effect=RuntimeError("stage failed"))
    monkeypatch.setattr(engine.loader, "stage", failure)
    with pytest.raises(RuntimeError, match="stage failed"):
        await engine.loader.reload(engine.session.agent_profile)
    assert engine.loaded_settings is original_settings
    assert vars(engine.session) == original_session
    assert engine.build_generation == original_generation
    assert engine.current.loaded is None
    assert engine.permits.agent_loading is False


async def test_start_subscribes_before_rearming_session_identity(
    monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory
) -> None:
    engine = agent_engine(EventBus(), settings=Settings())
    old_profile = AgentProfile(name="old")
    engine.session.agent_profile = old_profile
    engine.session.session_end_fired = True
    observations: list[tuple[object, bool]] = []
    subscribe = engine.event_bus.subscribe

    async def observe_subscription(event_type, handler) -> None:
        if not observations:
            observations.append((engine.session.agent_profile, engine.session.session_end_fired))
        await subscribe(event_type, handler)

    async def fail_build(profile: AgentProfile, staged: construction.StagedBuild) -> None:
        raise RuntimeError("candidate failed")

    monkeypatch.setattr(engine.event_bus, "subscribe", observe_subscription)
    monkeypatch.setattr(engine.loader, "build", fail_build)
    try:
        with pytest.raises(RuntimeError, match="candidate failed"):
            await engine.start(AgentProfile(name="new"))
        assert observations == [(old_profile, True)]
        assert engine.session.session_end_fired is False
        assert engine.session.agent_profile.name == "new"
    finally:
        engine.session.guard.release()
