# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which model the approval judge of a built agent is bound to."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import create_autospec

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.service.llm.mock import MockChatClient
from chrys.service.profiles.agents.schema import ModelConfig
from chrys.service.profiles.models.schema import ModelProfile
from tests.orchestration.engine._recovery_helpers import _model_registry, _profile, _registry

if TYPE_CHECKING:
    from pathlib import Path

_PINNED = ModelProfile(id="pinned-model", name="Pinned", model_id="gpt-pinned")
_OTHER = ModelProfile(id="other-model", name="Other", model_id="gpt-other")
_JUDGE = ModelProfile(id="judge-model", name="Judge", model_id="gpt-judge")


@pytest.fixture(autouse=True)
def _isolated_workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real workspace scans independent of the checkout."""
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


# "Use the active model" is what an empty judge selector means. What the agent runs on is decided
# by more than ``settings.model_profile``, and that pointer may name nothing usable at all.
@pytest.mark.parametrize(
    ("pinned_on_agent", "settings", "expected"),
    [
        pytest.param(True, Settings(), _PINNED, id="agent-model-without-an-active-pointer"),
        pytest.param(True, Settings(model_profile=_OTHER.id), _PINNED, id="agent-model-over-the-active-pointer"),
        pytest.param(
            False,
            Settings(model_profile=_OTHER.id, model_profile_override=_PINNED.id),
            _PINNED,
            id="session-override-over-the-active-pointer",
        ),
        pytest.param(False, Settings(model_profile=_OTHER.id), _OTHER, id="active-pointer"),
    ],
)
async def test_a_judge_without_a_model_of_its_own_runs_on_the_agents_model(
    agent_engine, monkeypatch: pytest.MonkeyPatch, pinned_on_agent: bool, settings: Settings, expected: ModelProfile
) -> None:
    import chrys.orchestration.engine.build.builder as builder_module

    monkeypatch.setattr(
        builder_module,
        "create_client",
        create_autospec(builder_module.create_client, return_value=MockChatClient(responses=[])),
    )
    profile = replace(_profile("Code"), model=ModelConfig(profile_id=_PINNED.id)) if pinned_on_agent else _profile()
    engine = agent_engine(
        EventBus(),
        settings=settings,
        agent_registry=_registry(profile),
        model_registry=_model_registry(_PINNED, _OTHER, _JUDGE),
    )

    await engine.start(profile)

    assert engine.current.loaded is not None
    assert engine.active_model_profile == expected
    assert engine.current.loaded.approval_judge.profile == expected


async def test_a_judge_with_a_model_of_its_own_keeps_it(agent_engine, monkeypatch: pytest.MonkeyPatch) -> None:
    import chrys.orchestration.engine.build.builder as builder_module

    monkeypatch.setattr(
        builder_module,
        "create_client",
        create_autospec(builder_module.create_client, return_value=MockChatClient(responses=[])),
    )
    profile = replace(_profile("Code"), model=ModelConfig(profile_id=_PINNED.id))
    judge_profile = _JUDGE
    engine = agent_engine(
        EventBus(),
        settings=Settings(approval_judge_model_profile=_JUDGE.id),
        agent_registry=_registry(profile),
        model_registry=_model_registry(_PINNED, _OTHER, judge_profile),
    )

    await engine.start(profile)

    assert engine.current.loaded is not None
    assert engine.active_model_profile == _PINNED
    judge = engine.current.loaded.approval_judge
    assert judge.profile == judge_profile
    assert judge._reasoning_judge.profile == _PINNED
