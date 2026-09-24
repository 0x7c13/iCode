# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the sub-agents configuration panel and its cards."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from rich.text import Text
from textual.css.query import NoMatches
from textual.pilot import Pilot
from textual.widgets import (
    Button,
    Input,
    Select,
    TextArea,
)
from textual.widgets._select import SelectCurrent, SelectOverlay

from chrys.app.tui.screens.agents.panels.subagents import SubAgentCard, SubAgentsConfigPanel
from chrys.service.profiles.agents.schema import (
    SubAgentRef,
    SubAgentsConfig,
)
from tests.app.tui.screens._agent_config_support import (
    _DEFAULT_WAIT_TIMEOUT,
    _registry,
    _wait_for_selectors,
    open_agent_config,
)
from tests.support.tui_helpers import PlaceholderHostApp
from tests.support.waiting import wait_for

# The polling helpers in _agent_config_support share one 45s ceiling, below
# the repository's 60s default per-test timeout. This module retains a 120s
# override for expensive screen mounting before the first wait; once polling
# starts, the helper deadline still produces a clean AssertionError before the
# thread timeout can kill the xdist worker and surface only "worker gwN crashed".
# No ``isolated_chrys_config_dir``: nothing here reads the platform record, and
# the autouse ``_isolate_platform_config_dir`` in tests/conftest.py already pins
# the config directory for every test. The three agent-config modules opt into
# that fixture because they assert against ``tmp_path`` itself, which this one
# never touches.
pytestmark = [pytest.mark.timeout(120)]


async def _wait_for_card_count(panel: SubAgentsConfigPanel, pilot: Pilot, count: int, *, description: str) -> None:
    """Wait until *panel* holds exactly *count* mounted ``SubAgentCard`` widgets."""
    await wait_for(
        lambda: len(list(panel.query(SubAgentCard))) == count,
        pilot=pilot,
        timeout=_DEFAULT_WAIT_TIMEOUT,
        description=description,
    )


@contextlib.asynccontextmanager
async def _open_subagents_panel(*refs: SubAgentRef) -> AsyncIterator[tuple[SubAgentsConfigPanel, Pilot]]:
    """Mount a sub-agents panel seeded with *refs* and wait for one card per ref."""
    app = PlaceholderHostApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SubAgentsConfigPanel(
            sub_agents_config=SubAgentsConfig(agents=list(refs)),
            registry=_registry(),
            current_profile_name="Code",
        )
        await app.mount(panel)
        await _wait_for_card_count(panel, pilot, len(refs), description="initial sub-agent cards")
        yield panel, pilot


def _subagent_card(**overrides: object) -> SubAgentCard:
    """Build the seeded ``SubAgentCard`` the ``get_config`` unit tests share."""
    fields: dict[str, object] = {
        "profile_name": "Explore",
        "tool_name": "explore_custom",
        "tool_description": "Explore things",
        "max_concurrency": 3,
        "index": 0,
        "available_profiles": [(Text("Explore Agent"), "Explore")],
    }
    fields.update(overrides)
    return SubAgentCard(**fields)


async def test_agent_config_subagent_select_mount_race_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry()
    original_update = SelectCurrent.update
    tripped = False

    def flaky_update(self: SelectCurrent, label: object) -> None:
        nonlocal tripped
        parent = self.parent
        if parent is not None and parent.id == "sa-profile-0" and not tripped:
            tripped = True
            raise NoMatches("No nodes match '#label'")
        original_update(self, label)

    monkeypatch.setattr(SelectCurrent, "update", flaky_update)

    async with open_agent_config(registry, current_profile="Code", initial_tab="sub-agents", hydrated=False) as (
        screen,
        pilot,
    ):
        await wait_for(
            lambda: str(screen.query_one("#sa-profile-0", Select).query_one(SelectCurrent).label) == "Explore Agent",
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="sub-agent profile select visual sync",
        )

        select = screen.query_one("#sa-profile-0", Select)
        current = select.query_one(SelectCurrent)

    assert tripped is True
    assert select.value == "Explore"
    assert str(current.label) == "Explore Agent"


async def test_agent_config_subagent_select_overlay_mount_race_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    original_setup = Select._setup_options_renderables
    tripped = False

    def flaky_setup(self: Select) -> None:
        nonlocal tripped
        if self.id == "sa-profile-0" and not tripped:
            tripped = True
            raise NoMatches("No nodes match 'SelectOverlay'")
        original_setup(self)

    monkeypatch.setattr(Select, "_setup_options_renderables", flaky_setup)

    async with open_agent_config(registry, current_profile="Code", initial_tab="sub-agents", hydrated=False) as (
        screen,
        pilot,
    ):
        await pilot.pause()

        select = screen.query_one("#sa-profile-0", Select)
        current = select.query_one(SelectCurrent)
        overlay = select.query_one(SelectOverlay)

    assert tripped is True
    assert select.value == "Explore"
    assert str(current.label) == "Explore Agent"
    assert overlay.option_count > 1


async def test_agent_config_subagent_add_empty_card_uses_valid_blank_selection() -> None:
    registry = _registry()

    async with open_agent_config(registry, current_profile="Explore", initial_tab="sub-agents", hydrated=False) as (
        screen,
        pilot,
    ):
        screen.query_one("#sa-add-btn", Button).press()
        await pilot.pause()

        select = screen.query_one("#sa-profile-0", Select)

    assert select.value is Select.NULL


async def test_subagent_concurrency_placeholders_match_defaults() -> None:
    async with _open_subagents_panel(SubAgentRef(profile="Explore")) as (panel, _pilot):
        assert panel.query_one("#sa-max-total", Input).placeholder == "3"
        assert panel.query_one("#sa-max-conc-0", Input).placeholder == "3"


def test_subagents_get_config_falls_back_to_seed_when_cards_have_not_mounted() -> None:
    """Save fired before cards finish mounting must use the seed _config.

    Pins the Windows-CI fix where a single ``pilot.pause()`` after Clone
    returned before ``_rebuild_cards`` had pushed every ``SubAgentCard``
    into the DOM, so ``get_config`` saw zero cards and silent-drop
    persisted an empty sub_agents list over a profile with three refs.
    """
    seed = SubAgentsConfig(
        max_total_concurrency=4,
        agents=[
            SubAgentRef(profile="Explore", tool_name="t_explore", tool_description="d1", max_concurrency=2),
            SubAgentRef(profile="General", tool_name="t_general", tool_description="d2", max_concurrency=1),
            SubAgentRef(profile="", tool_name="blank", tool_description="d3", max_concurrency=1),
        ],
    )
    panel = SubAgentsConfigPanel(sub_agents_config=seed, registry=_registry(), current_profile_name="Code")

    # No cards mounted — simulates the post-Clone window before _rebuild_cards
    # has dispatched all SubAgentCard mounts.
    assert list(panel.query(SubAgentCard)) == []

    cfg = panel.get_config()
    assert [a.profile for a in cfg.agents] == ["Explore", "General"]
    assert [a.tool_name for a in cfg.agents] == ["t_explore", "t_general"]
    assert cfg.max_total_concurrency == 4


async def test_add_subagent_inserts_new_card_before_existing_refs() -> None:
    async with _open_subagents_panel(SubAgentRef(profile="Explore")) as (panel, pilot):
        panel.query_one("#sa-add-btn", Button).press()
        await _wait_for_card_count(panel, pilot, 2, description="new sub-agent card")
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-profile-1")

        cards = list(panel.query(SubAgentCard))
        cards[0].query_one("#sa-profile-0", Select).value = "General"
        await pilot.pause()

        assert cards[1].query_one("#sa-profile-1", Select).value == "Explore"
        assert [ref.profile for ref in panel.get_config().agents] == ["General", "Explore"]


async def test_add_subagent_preserves_invalid_existing_max_concurrency() -> None:
    async with _open_subagents_panel(SubAgentRef(profile="Explore", max_concurrency=2)) as (panel, pilot):
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-max-conc-0")
        await wait_for(
            lambda: next(iter(panel.query(SubAgentCard))).has_profile,
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="initial sub-agent profile selection",
        )

        cards = list(panel.query(SubAgentCard))
        cards[0].query_one("#sa-max-conc-0", Input).value = "abc"
        assert any("max concurrency must be a valid integer" in error for error in cards[0].validate())

        panel.query_one("#sa-add-btn", Button).press()

        await _wait_for_card_count(panel, pilot, 2, description="new sub-agent card")
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-profile-1", "#sa-max-conc-1")
        await wait_for(
            lambda: len(list(panel.query(SubAgentCard))) == 2 and list(panel.query(SubAgentCard))[1].has_profile,
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="existing sub-agent profile selection after add",
        )

        cards = list(panel.query(SubAgentCard))
        assert cards[1].query_one("#sa-profile-1", Select).value == "Explore"
        assert cards[1].query_one("#sa-max-conc-1", Input).value == "abc"
        assert any("max concurrency must be a valid integer" in error for error in panel.validate())


async def test_remove_subagent_preserves_edits_in_remaining_refs() -> None:
    async with _open_subagents_panel(SubAgentRef(profile="Explore"), SubAgentRef(profile="General")) as (panel, pilot):
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-profile-1")
        await wait_for(
            lambda: list(panel.query(SubAgentCard))[1].query_one("#sa-profile-1", Select).value == "General",
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="surviving sub-agent profile selection",
        )

        cards = list(panel.query(SubAgentCard))
        cards[1].query_one("#sa-tool-name-1", Input).value = "general_custom"
        cards[1].query_one("#sa-tool-desc-1", TextArea).text = "edited general description"
        cards[1].query_one("#sa-max-conc-1", Input).value = "5"
        cards[0].query_one("#sa-delete-btn-0", Button).press()

        await _wait_for_card_count(panel, pilot, 1, description="remaining sub-agent card")
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-tool-name-0", "#sa-tool-desc-0")
        await wait_for(
            lambda: next(iter(panel.query(SubAgentCard))).query_one("#sa-profile-0", Select).value == "General",
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="remaining sub-agent profile selection after remove",
        )

        remaining = next(iter(panel.query(SubAgentCard)))
        assert remaining.query_one("#sa-profile-0", Select).value == "General"
        assert remaining.query_one("#sa-tool-name-0", Input).value == "general_custom"
        assert remaining.query_one("#sa-tool-desc-0", TextArea).text == "edited general description"
        assert remaining.query_one("#sa-max-conc-0", Input).value == "5"
        assert [
            (ref.profile, ref.tool_name, ref.tool_description, ref.max_concurrency) for ref in panel.get_config().agents
        ] == [("General", "general_custom", "edited general description", 5)]


async def test_remove_subagent_preserves_invalid_existing_max_concurrency() -> None:
    async with _open_subagents_panel(
        SubAgentRef(profile="Explore"),
        SubAgentRef(profile="General", max_concurrency=2),
    ) as (panel, pilot):
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-profile-1", "#sa-max-conc-1")
        await wait_for(
            lambda: list(panel.query(SubAgentCard))[1].has_profile,
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="surviving sub-agent profile selection",
        )

        cards = list(panel.query(SubAgentCard))
        cards[1].query_one("#sa-max-conc-1", Input).value = "abc"
        assert any("max concurrency must be a valid integer" in error for error in cards[1].validate())

        cards[0].query_one("#sa-delete-btn-0", Button).press()

        await _wait_for_card_count(panel, pilot, 1, description="remaining sub-agent card")
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-max-conc-0")
        await wait_for(
            lambda: next(iter(panel.query(SubAgentCard))).has_profile,
            pilot=pilot,
            timeout=_DEFAULT_WAIT_TIMEOUT,
            description="remaining sub-agent profile selection after remove",
        )

        remaining = next(iter(panel.query(SubAgentCard)))
        assert remaining.query_one("#sa-profile-0", Select).value == "General"
        assert remaining.query_one("#sa-max-conc-0", Input).value == "abc"
        assert any("max concurrency must be a valid integer" in error for error in panel.validate())


def test_subagent_card_get_config_preserves_seed_before_children_mount() -> None:
    card = _subagent_card()

    cfg = card.get_config()

    assert cfg == {
        "profile": "Explore",
        "tool_name": "explore_custom",
        "tool_description": "Explore things",
        "max_concurrency": 3,
    }


def test_subagent_card_get_config_allows_cleared_mounted_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _subagent_card()
    card._profile_user_modified = True

    def query_one(selector: str, *_args: object, **_kwargs: object) -> object:
        if selector == "#sa-profile-0":
            return SimpleNamespace(value=Select.NULL)
        raise NoMatches(f"No nodes match {selector!r}")

    monkeypatch.setattr(card, "query_one", query_one)

    cfg = card.get_config()

    assert cfg["profile"] == ""
    assert cfg["tool_name"] == "explore_custom"


def test_subagent_card_get_config_preserves_seed_for_unmodified_mounting_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _subagent_card()

    def query_one(selector: str, *_args: object, **_kwargs: object) -> object:
        if selector == "#sa-profile-0":
            return SimpleNamespace(value=Select.NULL)
        raise NoMatches(f"No nodes match {selector!r}")

    monkeypatch.setattr(card, "query_one", query_one)

    cfg = card.get_config()

    assert cfg["profile"] == "Explore"
    assert cfg["tool_name"] == "explore_custom"
