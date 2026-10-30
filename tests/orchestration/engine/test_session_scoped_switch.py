# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session-scoped model and agent switches: what each session transition keeps and releases.

A model switch made with ``session_scoped=True`` (a prompt shortcut
card) and an agent switch made with ``session_scoped=True`` belong to the
session they were made in: a new or cleared session, and another restored
session, start as if they never happened, while reloads and restoring the same
session keep them. A caller-owned pin (headless ``--model``) is never released.
Every test drives a real engine build over mock model clients.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

import chrys.orchestration.engine.build.builder as builder_module
import chrys.orchestration.engine.session_lifecycle as session_lifecycle_module
from chrys.foundation.config.runtime_pointer import get_model_pointer
from chrys.foundation.config.settings_store import load_settings
from chrys.foundation.config.spec import Source
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentLoadFailed,
    AgentProfileSwitch,
    Event,
    SessionClear,
    SessionNew,
    SessionRestore,
    SessionRestored,
    SetModelProfile,
    SettingsReload,
    UserMessage,
)
from chrys.foundation.platform import get_platform
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.engine.state.active_session import session_pin_overrides
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.profiles.models.resolver import loaded_with_active_model_profile
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.state.store import JsonFileStateStore
from tests.orchestration.engine._recovery_helpers import (
    _model_registry,
    _profile,
    _registry,
    _seed_restorable_session,
)
from tests.support.event_capture import capture_event_sequence, collect_events
from tests.support.waiting import (
    ENGINE_TEST_WAIT_TIMEOUT,
    ENGINE_TURN_TIMEOUT,
    await_run_task_chain,
    with_wait_deadline,
)

DEFAULT_MODEL = "model-a"
SWITCHED_MODEL = "model-b"
SAVED_MODEL = "model-c"

_CODE = _profile("Code", "Code Agent")
_QA = _profile("QA", "QA Agent")
_DOCS = _profile("Docs", "Docs Agent")


def _mock_model(profile_id: str) -> ModelProfile:
    return ModelProfile(
        id=profile_id,
        name=profile_id,
        provider="mock",
        model_id=profile_id,
        max_context_tokens=100_000,
        stream=False,
    )


@pytest.fixture(autouse=True)
def _default_model_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The saved default is the process pointer; teardown gives back whatever a restore wrote to it."""
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", DEFAULT_MODEL)
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    monkeypatch.chdir(workdir)


class _Clients:
    """Hand every build a fresh mock client; ``fail_builds`` makes the next builds fail."""

    def __init__(self) -> None:
        self.fail_builds = 0
        self.created = 0

    async def create(self, _settings: Any = None, **_kwargs: Any) -> MockChatClient:
        if self.fail_builds:
            self.fail_builds -= 1
            raise RuntimeError("model client unavailable")
        self.created += 1
        return MockChatClient(responses=[MockResponse(text=f"reply {self.created}") for _ in range(4)])


async def _start_engine(
    agent_engine: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    loaded: Any = None,
    agent: Any = _CODE,
) -> tuple[AgentEngine, _Clients, JsonFileStateStore]:
    clients = _Clients()
    monkeypatch.setattr(builder_module, "create_client", clients.create)
    store = JsonFileStateStore(tmp_path / "sessions")
    engine: AgentEngine = agent_engine(
        EventBus(),
        loaded_settings=loaded if loaded is not None else load_settings(),
        agent_registry=_registry(_CODE, _QA, _DOCS),
        model_registry=_model_registry(
            _mock_model(DEFAULT_MODEL), _mock_model(SWITCHED_MODEL), _mock_model(SAVED_MODEL)
        ),
        state_store=store,
    )
    await engine.start(agent)
    assert engine.current.loaded is not None
    return engine, clients, store


async def _publish(engine: AgentEngine, event: Event) -> None:
    await engine.event_bus.publish(event, raise_handler_errors=True)


def _model(engine: AgentEngine) -> tuple[str, str]:
    details = engine.current.manifest.runtime_details.model
    return details.profile_id, details.selection_source


def _agent(engine: AgentEngine) -> str:
    profile = engine.session.agent_profile
    assert profile is not None
    return profile.name


async def _complete_turn(engine: AgentEngine, text: str = "hello") -> None:
    await engine.event_bus.publish(UserMessage(text=text))
    await await_run_task_chain(
        engine,
        turn_state=engine.turns.turn_state,
        timeout=ENGINE_TURN_TIMEOUT,
        propagate_inner_cancel=True,
        expect_installed=True,
    )


async def _restore(engine: AgentEngine, session_id: str, **kwargs: Any) -> None:
    async with capture_event_sequence(engine.event_bus, SessionRestored) as restored:
        await _publish(engine, SessionRestore(session_id=session_id, **kwargs))
    assert [e.session_id for e in restored if isinstance(e, SessionRestored)] == [session_id]


async def _scoped_model_switch(engine: AgentEngine, profile_id: str = SWITCHED_MODEL) -> None:
    await _publish(engine, SetModelProfile(profile_id=profile_id, session_scoped=True))
    assert _model(engine) == (profile_id, "active")


# ---------------------------------------------------------------------------
# 1. A session-scoped model switch
# ---------------------------------------------------------------------------


async def test_scoped_model_switch_is_active_selection_and_leaves_the_default(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    assert _model(engine) == (DEFAULT_MODEL, "active")

    await _publish(engine, SetModelProfile(profile_id=SWITCHED_MODEL, session_scoped=True))

    assert _model(engine) == (SWITCHED_MODEL, "active")
    assert engine.settings.model_profile == SWITCHED_MODEL
    assert engine.settings.model_profile_override == ""
    # The saved default is untouched: process pointer, user settings file, a fresh load.
    assert get_model_pointer()[0] == DEFAULT_MODEL
    assert not (get_platform().config_dir / "settings.yaml").exists()
    assert load_settings().settings.model_profile == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# 2. Reloads
# ---------------------------------------------------------------------------


async def test_reload_releasing_the_pin_follows_the_reloaded_default(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _scoped_model_switch(engine)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", SAVED_MODEL)

    await _publish(engine, SettingsReload(release_session_scoped_model=True))

    assert _model(engine) == (SAVED_MODEL, "active")
    assert engine.settings.model_profile == SAVED_MODEL


async def test_reload_releasing_the_pin_returns_to_an_unchanged_default(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _scoped_model_switch(engine)

    await _publish(engine, SettingsReload(release_session_scoped_model=True))

    assert _model(engine) == (DEFAULT_MODEL, "active")
    assert engine.session.session_switch.model is False
    assert engine.session.model_profile_pinned is False


async def test_plain_reload_keeps_the_session_model(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _scoped_model_switch(engine)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", SAVED_MODEL)

    await _publish(engine, SettingsReload())

    assert _model(engine) == (SWITCHED_MODEL, "active")
    # Still the session's own pin: a new session leaves it behind.
    await _publish(engine, SessionNew())
    assert _model(engine) == (SAVED_MODEL, "active")


# ---------------------------------------------------------------------------
# 3. New and cleared sessions
# ---------------------------------------------------------------------------


async def test_new_session_starts_on_the_default_model(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    old_session = engine.session.session_id
    await _scoped_model_switch(engine)

    await _publish(engine, SessionNew())

    assert engine.session.session_id != old_session
    assert _model(engine) == (DEFAULT_MODEL, "active")
    assert engine.settings.model_profile == DEFAULT_MODEL
    # The release is complete: a later default change is followed by a plain reload.
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", SAVED_MODEL)
    await _publish(engine, SettingsReload())
    assert _model(engine) == (SAVED_MODEL, "active")


async def test_cleared_session_starts_on_the_default_model(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _complete_turn(engine)
    old_session = engine.session.session_id
    await _scoped_model_switch(engine)

    await _publish(engine, SessionClear(session_id=old_session))

    assert engine.session.session_id != old_session
    assert _model(engine) == (DEFAULT_MODEL, "active")
    assert engine.settings.model_profile == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# 4. Restores
# ---------------------------------------------------------------------------


async def test_restoring_another_session_releases_the_pin_to_the_default(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _seed_restorable_session(store, agent_profile="Code")
    await _scoped_model_switch(engine)

    await _restore(engine, "restore_me")

    assert _model(engine) == (DEFAULT_MODEL, "active")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", SAVED_MODEL)
    await _publish(engine, SettingsReload())
    assert _model(engine) == (SAVED_MODEL, "active")


async def test_restoring_another_session_uses_its_saved_model(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _seed_restorable_session(store, agent_profile="Code", model_profile_id=SAVED_MODEL)
    await _scoped_model_switch(engine)

    await _restore(engine, "restore_me", apply_saved_model=True)

    assert _model(engine)[0] == SAVED_MODEL
    assert engine.settings.model_profile_override == ""


async def test_restoring_the_current_session_keeps_the_pin(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _scoped_model_switch(engine)
    await _complete_turn(engine)
    current = engine.session.session_id
    assert current is not None

    await _restore(engine, current)

    assert engine.session.session_id == current
    assert _model(engine) == (SWITCHED_MODEL, "active")
    # Still the session's: a new session leaves it behind.
    await _publish(engine, SessionNew())
    assert _model(engine) == (DEFAULT_MODEL, "active")


# ---------------------------------------------------------------------------
# 5. Caller-owned pin (headless --model)
# ---------------------------------------------------------------------------


async def _headless_engine(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[AgentEngine, _Clients, JsonFileStateStore]:
    loaded = loaded_with_active_model_profile(load_settings(), _mock_model(SWITCHED_MODEL), Source.CLI)
    engine, clients, store = await _start_engine(agent_engine, monkeypatch, tmp_path, loaded=loaded)
    engine.pin_model_profile()
    assert _model(engine)[0] == SWITCHED_MODEL
    return engine, clients, store


async def test_headless_model_pin_survives_a_new_session(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _headless_engine(agent_engine, monkeypatch, tmp_path)

    await _publish(engine, SessionNew())

    assert _model(engine)[0] == SWITCHED_MODEL
    assert engine.settings.model_profile == SWITCHED_MODEL


async def test_headless_model_pin_survives_a_releasing_reload(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _headless_engine(agent_engine, monkeypatch, tmp_path)

    await _publish(engine, SettingsReload(release_session_scoped_model=True))

    assert _model(engine)[0] == SWITCHED_MODEL
    # And it is still owned: neither a later new session nor another release drops it.
    await _publish(engine, SessionNew())
    await _publish(engine, SettingsReload(release_session_scoped_model=True))
    assert _model(engine)[0] == SWITCHED_MODEL


# ---------------------------------------------------------------------------
# 6. Failed rebuild keeps the pin
# ---------------------------------------------------------------------------


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_failed_releasing_reload_keeps_the_pinned_model(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _scoped_model_switch(engine)
    failures: list[AgentLoadFailed] = []
    await engine.event_bus.subscribe(AgentLoadFailed, lambda e: collect_events(failures, e))

    clients.fail_builds = 1
    # Default delivery: the bus logs and swallows the handler's failure, as in the TUI.
    await engine.event_bus.publish(SettingsReload(release_session_scoped_model=True))

    assert failures, "the rebuild was expected to fail"
    assert engine.settings.model_profile == SWITCHED_MODEL
    assert _model(engine) == (SWITCHED_MODEL, "active")
    # The release is rolled back with it: the pin is still recorded as the session's.
    assert engine.session.model_profile_pinned is True
    assert engine.session.session_switch.model is True
    # The pin is intact: a plain reload that succeeds still keeps the session model.
    await _publish(engine, SettingsReload())
    assert _model(engine) == (SWITCHED_MODEL, "active")
    # And it is still the session's: a new session leaves it behind.
    await _publish(engine, SessionNew())
    assert _model(engine) == (DEFAULT_MODEL, "active")


async def test_new_session_whose_settings_cannot_be_rederived_keeps_the_scoped_model(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    old_session = engine.session.session_id
    await _scoped_model_switch(engine)
    attempts: list[dict[str, Any]] = []

    def unreadable(**kwargs: Any) -> Any:
        attempts.append(kwargs)
        raise OSError("settings unreadable")

    monkeypatch.setattr(session_lifecycle_module, "load_settings", unreadable)

    await _publish(engine, SessionNew())

    # The re-derivation was tried and failed; /new still started a session.
    assert len(attempts) == 1
    assert engine.session.session_id != old_session
    assert engine.current.loaded is not None
    # On the live settings, scoped model pin included, and the pin is not released.
    assert _model(engine) == (SWITCHED_MODEL, "active")
    assert engine.settings.model_profile == SWITCHED_MODEL
    assert engine.session.model_profile_pinned is True
    assert engine.session.session_switch.model is True

    # Still the session's own: once settings read again, the next new session leaves it behind.
    monkeypatch.setattr(session_lifecycle_module, "load_settings", load_settings)
    await _publish(engine, SessionNew())
    assert _model(engine) == (DEFAULT_MODEL, "active")
    assert engine.session.model_profile_pinned is False
    assert engine.session.session_switch.model is False


# ---------------------------------------------------------------------------
# 7. Session-scoped agent switches
# ---------------------------------------------------------------------------


async def test_scoped_agent_switch_is_undone_by_a_new_session(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)

    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))
    assert _agent(engine) == "QA"

    await _publish(engine, SessionNew())
    assert _agent(engine) == "Code"
    # The record is consumed: the next new session stays on Code too.
    await _publish(engine, SessionNew())
    assert _agent(engine) == "Code"


async def test_scoped_agent_switch_is_undone_by_a_cleared_session(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _complete_turn(engine)

    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))
    assert _agent(engine) == "QA"

    await _publish(engine, SessionClear(session_id=engine.session.session_id))
    assert _agent(engine) == "Code"


async def test_two_scoped_agent_switches_return_to_the_first_agent(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)

    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))
    await _publish(engine, AgentProfileSwitch(profile_name="Docs", session_scoped=True))
    assert _agent(engine) == "Docs"

    await _publish(engine, SessionNew())
    assert _agent(engine) == "Code"


async def test_user_agent_switch_after_a_scoped_one_sticks(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)

    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))
    await _publish(engine, AgentProfileSwitch(profile_name="Docs"))
    assert _agent(engine) == "Docs"

    await _publish(engine, SessionNew())
    assert _agent(engine) == "Docs"


async def test_user_switch_to_the_scoped_agent_itself_sticks(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Choosing the agent already in use is still the user's choice."""
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)

    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))
    await _publish(engine, AgentProfileSwitch(profile_name="QA"))

    await _publish(engine, SessionNew())
    assert _agent(engine) == "QA"


async def test_restoring_another_session_forgets_the_scoped_agent_record(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _seed_restorable_session(store, agent_profile="Docs")
    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))

    await _restore(engine, "restore_me")
    assert _agent(engine) == "Docs"

    await _publish(engine, SessionNew())
    assert _agent(engine) == "Docs"


async def test_failed_scoped_agent_switch_leaves_no_record(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    failures: list[AgentLoadFailed] = []
    await engine.event_bus.subscribe(AgentLoadFailed, lambda e: collect_events(failures, e))

    clients.fail_builds = 1
    await engine.event_bus.publish(AgentProfileSwitch(profile_name="QA", session_scoped=True))

    assert failures, "the switch's build was expected to fail"
    assert _agent(engine) == "Code"
    assert engine.session.session_switch.agent_before is None
    # Later the user picks QA themselves; a new session keeps their choice.
    await _publish(engine, AgentProfileSwitch(profile_name="QA"))
    await _publish(engine, SessionNew())
    assert _agent(engine) == "QA"


async def test_failed_scoped_agent_switch_does_not_undo_a_later_user_switch_from_another_agent(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After a failed scoped switch, a later scoped switch records the agent it really replaced."""
    engine, clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)

    clients.fail_builds = 1
    await engine.event_bus.publish(AgentProfileSwitch(profile_name="QA", session_scoped=True))
    assert _agent(engine) == "Code"
    # A leftover Code record would send the new session back to Code instead of QA.
    await _publish(engine, AgentProfileSwitch(profile_name="QA"))
    await _publish(engine, AgentProfileSwitch(profile_name="Docs", session_scoped=True))

    await _publish(engine, SessionNew())
    assert _agent(engine) == "QA"


# ---------------------------------------------------------------------------
# 8. ACP-style model switch
# ---------------------------------------------------------------------------


async def test_acp_model_switch_is_an_override(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)

    await _publish(engine, SetModelProfile(profile_id=SWITCHED_MODEL))

    assert _model(engine) == (SWITCHED_MODEL, "override")
    assert engine.settings.model_profile == SWITCHED_MODEL
    assert engine.settings.model_profile_override == SWITCHED_MODEL
    assert get_model_pointer()[0] == DEFAULT_MODEL
    # A plain reload keeps it.
    await _publish(engine, SettingsReload())
    assert _model(engine) == (SWITCHED_MODEL, "override")
    assert engine.session.session_switch.model is False


async def test_acp_model_switch_survives_a_new_session(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    old_session = engine.session.session_id

    await _publish(engine, SetModelProfile(profile_id=SWITCHED_MODEL, session_scoped=False))
    await _publish(engine, SessionNew())

    assert engine.session.session_id != old_session
    assert _model(engine) == (SWITCHED_MODEL, "override")
    assert engine.settings.model_profile == SWITCHED_MODEL
    assert engine.settings.model_profile_override == SWITCHED_MODEL
    assert engine.session.session_switch.model is False
    # Not a default change either: a changed saved default does not take over on a plain reload.
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", SAVED_MODEL)
    await _publish(engine, SettingsReload())
    assert _model(engine) == (SWITCHED_MODEL, "override")


async def test_acp_model_switch_is_carried_into_another_restored_session(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _seed_restorable_session(store, agent_profile="Code")
    await _publish(engine, SetModelProfile(profile_id=SWITCHED_MODEL))

    await _restore(engine, "restore_me")

    assert engine.session.session_id == "restore_me"
    assert _model(engine) == (SWITCHED_MODEL, "override")
    assert engine.settings.model_profile_override == SWITCHED_MODEL
    assert engine.session.session_switch.model is False


# ---------------------------------------------------------------------------
# 9. The session-scoped record itself
# ---------------------------------------------------------------------------


async def test_scoped_model_switch_is_recorded_and_a_new_session_clears_the_record(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    assert engine.session.session_switch.model is False

    await _scoped_model_switch(engine)

    assert engine.session.session_switch.model is True
    assert engine.session.model_profile_pinned is True
    await _publish(engine, SessionNew())
    assert engine.session.session_switch.model is False
    assert engine.session.session_switch.active is False


async def test_plain_model_switch_after_a_scoped_one_replaces_it(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _scoped_model_switch(engine)

    await _publish(engine, SetModelProfile(profile_id=SAVED_MODEL, session_scoped=False))

    assert engine.session.session_switch.model is False
    assert _model(engine) == (SAVED_MODEL, "override")
    # No longer the session's own: a new session keeps it.
    await _publish(engine, SessionNew())
    assert _model(engine) == (SAVED_MODEL, "override")


async def test_scoped_model_switch_over_a_caller_owned_pin_is_not_recorded(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _headless_engine(agent_engine, monkeypatch, tmp_path)

    await _publish(engine, SetModelProfile(profile_id=SAVED_MODEL, session_scoped=True))

    assert engine.session.session_switch.model is False


async def test_scoped_agent_switch_records_the_agent_it_replaced(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)

    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))

    before = engine.session.session_switch.agent_before
    assert before is not None
    assert before.name == "Code"
    # The user's own switch ends that half of the record.
    await _publish(engine, AgentProfileSwitch(profile_name="Docs"))
    assert engine.session.session_switch.agent_before is None


async def test_restoring_the_current_session_keeps_both_scoped_switches(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))
    await _scoped_model_switch(engine)
    await _complete_turn(engine)
    current = engine.session.session_id
    assert current is not None

    await _restore(engine, current)

    before = engine.session.session_switch.agent_before
    assert before is not None
    assert before.name == "Code"
    assert engine.session.session_switch.model is True
    assert _agent(engine) == "QA"
    assert _model(engine) == (SWITCHED_MODEL, "active")
    # Both are still the session's: a new session undoes both.
    await _publish(engine, SessionNew())
    assert _agent(engine) == "Code"
    assert _model(engine) == (DEFAULT_MODEL, "active")


async def test_restoring_another_session_clears_both_scoped_switches(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _seed_restorable_session(store, agent_profile="Docs")
    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))
    await _scoped_model_switch(engine)

    await _restore(engine, "restore_me")

    assert engine.session.session_switch.agent_before is None
    assert engine.session.session_switch.model is False
    assert _model(engine) == (DEFAULT_MODEL, "active")


# ---------------------------------------------------------------------------
# 9b. Switches cut short, and transitions that fail after leaving the session
# ---------------------------------------------------------------------------


def _cancel_the_next_rebuild(engine: AgentEngine, monkeypatch: pytest.MonkeyPatch, *, after_it_commits: bool) -> None:
    """Make the next switch's rebuild end in a cancel (Esc), before it starts or once it went live."""
    lifecycle = engine.lifecycle
    original: Callable[..., Awaitable[None]] = lifecycle.reload_with_rebuild_permit
    pending = [True]

    async def rebuild(*args: Any, **kwargs: Any) -> None:
        if not pending:
            await original(*args, **kwargs)
            return
        pending.clear()
        if after_it_commits:
            await original(*args, **kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr(lifecycle, "reload_with_rebuild_permit", rebuild)


async def test_a_scoped_agent_switch_cancelled_once_live_still_ends_with_its_session(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    _cancel_the_next_rebuild(engine, monkeypatch, after_it_commits=True)

    with pytest.raises(asyncio.CancelledError):
        await engine.event_bus.publish(AgentProfileSwitch(profile_name="QA", session_scoped=True))

    assert _agent(engine) == "QA"
    await _publish(engine, SessionNew())
    assert _agent(engine) == "Code"


async def test_a_scoped_model_switch_cancelled_before_its_rebuild_leaves_the_session_model(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    _cancel_the_next_rebuild(engine, monkeypatch, after_it_commits=False)

    with pytest.raises(asyncio.CancelledError):
        await engine.event_bus.publish(SetModelProfile(profile_id=SWITCHED_MODEL, session_scoped=True))

    assert _model(engine) == (DEFAULT_MODEL, "active")
    assert engine.settings.model_profile == DEFAULT_MODEL
    assert engine.session.model_profile_pinned is False
    assert engine.session.session_switch.model is False
    # Nothing left over for a reload to pick up.
    await _publish(engine, SettingsReload())
    assert _model(engine) == (DEFAULT_MODEL, "active")


async def test_a_scoped_model_switch_cancelled_once_live_keeps_it_for_this_session_only(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    _cancel_the_next_rebuild(engine, monkeypatch, after_it_commits=True)

    with pytest.raises(asyncio.CancelledError):
        await engine.event_bus.publish(SetModelProfile(profile_id=SWITCHED_MODEL, session_scoped=True))

    # The runtime was built on the new model: the settings and the pin say so too.
    assert _model(engine) == (SWITCHED_MODEL, "active")
    assert engine.settings.model_profile == SWITCHED_MODEL
    assert engine.session.session_switch.model is True
    await _publish(engine, SessionNew())
    assert _model(engine) == (DEFAULT_MODEL, "active")


def _assert_the_switches_ended(engine: AgentEngine) -> None:
    """The closed session's switches are gone from the record, the pins, the settings and the agent alike."""
    assert engine.session.session_switch.agent_before is None
    assert engine.session.session_switch.model is False
    assert engine.session.model_profile_pinned is False
    assert "model_profile" not in session_pin_overrides(engine.session, engine.settings)
    assert engine.settings.model_profile == DEFAULT_MODEL
    assert _agent(engine) == "Code"


async def _next_session_starts_clean(engine: AgentEngine) -> None:
    await _publish(engine, SessionNew())
    assert _model(engine) == (DEFAULT_MODEL, "active")
    assert engine.settings.model_profile == DEFAULT_MODEL
    assert _agent(engine) == "Code"
    assert engine.session.session_switch.active is False


async def _fail_once(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("start failed")


async def _cancel_once(*_args: Any, **_kwargs: Any) -> None:
    raise asyncio.CancelledError


@pytest.mark.parametrize("ending", ["raises", "cancelled"])
async def test_a_new_session_that_never_starts_still_ends_the_scoped_switches(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ending: str
) -> None:
    engine, clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))
    await _scoped_model_switch(engine)
    lifecycle = engine.lifecycle
    original_start = lifecycle.start
    started: list[str] = []

    async def start(profile: Any, **kwargs: Any) -> None:
        if started:
            await original_start(profile, **kwargs)
            return
        started.append(profile.name)
        if ending == "raises":
            clients.fail_builds = 1
            await original_start(profile, **kwargs)
        else:
            await _cancel_once()

    monkeypatch.setattr(lifecycle, "start", start)
    if ending == "raises":
        failures: list[AgentLoadFailed] = []
        await engine.event_bus.subscribe(AgentLoadFailed, lambda e: collect_events(failures, e))
        # Default delivery: the bus logs and swallows the handler's failure, as in the TUI.
        await engine.event_bus.publish(SessionNew())
        assert failures, "the new session's build was expected to fail"
    else:
        with pytest.raises(asyncio.CancelledError):
            await engine.event_bus.publish(SessionNew())

    assert started == ["Code"]
    assert engine.current.loaded is None
    _assert_the_switches_ended(engine)
    await _next_session_starts_clean(engine)


@pytest.mark.parametrize("ending", ["raises", "cancelled"])
async def test_a_restore_of_another_session_that_never_starts_still_ends_the_scoped_switches(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ending: str
) -> None:
    engine, _clients, store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _seed_restorable_session(store, agent_profile="Docs")
    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))
    await _scoped_model_switch(engine)
    hydrate = _fail_once if ending == "raises" else _cancel_once

    monkeypatch.setattr(engine.lifecycle, "_hydrate_restored_session", hydrate)
    if ending == "raises":
        await engine.event_bus.publish(SessionRestore(session_id="restore_me"))
    else:
        with pytest.raises(asyncio.CancelledError):
            await engine.event_bus.publish(SessionRestore(session_id="restore_me"))

    assert engine.current.loaded is None
    _assert_the_switches_ended(engine)
    await _next_session_starts_clean(engine)


async def test_a_restore_failing_before_it_enters_the_target_folder_keeps_the_pin_with_its_settings(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The settings staged for the target's folder never go in force outside it: the pin and its settings stay together."""
    engine, _clients, store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _seed_restorable_session(store, agent_profile="Docs")
    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))
    await _scoped_model_switch(engine)
    workspace_before = engine.session.workspace

    real_cleanup = engine.lifecycle.cleanup_empty_session_dir
    failed: list[None] = []

    def cleanup_fails_once_the_session_is_closed() -> None:
        if engine.current.loaded is None and not failed:
            failed.append(None)
            raise RuntimeError("cleanup failed")
        real_cleanup()

    monkeypatch.setattr(engine.lifecycle, "cleanup_empty_session_dir", cleanup_fails_once_the_session_is_closed)
    await engine.event_bus.publish(SessionRestore(session_id="restore_me"))

    assert failed
    assert engine.current.loaded is None
    assert engine.session.workspace is workspace_before
    assert engine.session.session_switch.model is True
    assert engine.session.model_profile_pinned is True
    assert engine.settings.model_profile == SWITCHED_MODEL
    assert engine.session.session_switch.agent_before is None
    assert _agent(engine) == "Code"
    # The next session re-derives its settings without the switch's model.
    await _next_session_starts_clean(engine)


async def test_a_restore_that_fails_ends_a_scoped_agent_switch_without_a_model_one(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, _clients, store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    await _seed_restorable_session(store, agent_profile="Docs")
    await _publish(engine, AgentProfileSwitch(profile_name="QA", session_scoped=True))

    monkeypatch.setattr(engine.lifecycle, "_hydrate_restored_session", _fail_once)
    await engine.event_bus.publish(SessionRestore(session_id="restore_me"))

    _assert_the_switches_ended(engine)


# ---------------------------------------------------------------------------
# 10. session_pin_overrides
# ---------------------------------------------------------------------------

_MODEL_PIN_KEYS = {"model_profile", "model_profile_override", "model_profile_override_sub_agents"}


@pytest.mark.parametrize("scoped", [True, False], ids=["card", "acp"])
async def test_session_pin_overrides_leaves_out_only_a_scoped_model_pin(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scoped: bool
) -> None:
    engine, _clients, _store = await _start_engine(agent_engine, monkeypatch, tmp_path)
    assert session_pin_overrides(engine.session, engine.settings) == {}
    await _publish(engine, SetModelProfile(profile_id=SWITCHED_MODEL, session_scoped=scoped))
    engine.session.ask_user_timeout_pinned = True

    with_model = session_pin_overrides(engine.session, engine.settings, with_scoped_model=True)
    without = session_pin_overrides(engine.session, engine.settings, with_scoped_model=False)

    assert set(with_model) == _MODEL_PIN_KEYS | {"ask_user_timeout_seconds"}
    assert with_model["model_profile"] == SWITCHED_MODEL
    assert with_model["ask_user_timeout_seconds"] == engine.settings.ask_user_timeout_seconds
    if scoped:
        assert without == {"ask_user_timeout_seconds": engine.settings.ask_user_timeout_seconds}
    else:
        assert without == with_model
