# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Spec tests for the one place the main screen reads which models the session has."""

from __future__ import annotations

import pytest

from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState, SessionModels, session_models
from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import LoadedSettings, SettingsHandle
from chrys.foundation.config.spec import Source
from chrys.foundation.events.bus import EventBus


def _models(*, confirmed: str | None, active: str, pointer: str, pin: str = "", handle: bool = True) -> SessionModels:
    state = MainScreenState()
    if confirmed is not None:
        state.runtime.details_confirmed = True
        state.runtime.details.model.profile_id = confirmed
    loaded = LoadedSettings(settings=Settings(), provenance={}).overlay(Source.PROCESS_RUNTIME, model_profile=pointer)
    if pin:
        loaded = loaded.overlay(Source.SESSION, model_profile=pin)
    services = MainScreenServices(
        bus=EventBus(), settings_handle=SettingsHandle(loaded) if handle else None, active_model_profile_id=active
    )
    return session_models(state, services)


@pytest.mark.parametrize(
    ("confirmed", "pin", "handle", "expected"),
    [
        # Before the runtime names a model, both read the launch setting.
        (None, "", True, SessionModels(running="model-p", selected="model-p")),
        # A prompt shortcut's session-only switch the runtime took up.
        ("model-s", "model-s", True, SessionModels(running="model-s", selected="model-s")),
        # An agent's own model runs while the session's own choice waits underneath.
        ("model-a", "model-s", True, SessionModels(running="model-a", selected="model-s")),
        ("model-a", "", True, SessionModels(running="model-a", selected="model-p")),
        # A runtime answer that names no model says nothing: the launch setting stands.
        ("", "", True, SessionModels(running="model-p", selected="model-p")),
        # Without the settings the engine shares, the launch setting is the best known choice.
        ("model-a", "", False, SessionModels(running="model-a", selected="model-p")),
    ],
    ids=["before-runtime", "shortcut-switch", "agent-own-over-switch", "agent-own", "blank-answer", "no-settings"],
)
def test_session_models_reads_what_runs_and_what_the_session_chose(
    confirmed: str | None, pin: str, handle: bool, expected: SessionModels
) -> None:
    assert _models(confirmed=confirmed, active="model-p", pointer="model-p", pin=pin, handle=handle) == expected
