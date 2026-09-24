# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Skills configuration panel."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from chrys.app.tui.screens.agents.panels import skills as skills_module
from chrys.app.tui.screens.agents.panels.skills import SkillDirCard, SkillsConfigPanel
from chrys.app.tui.screens.agents.skill_paths import skill_path_duplicate_match_description
from chrys.foundation.i18n import Localizer
from chrys.service.profiles.agents.schema import SkillsConfig
from tests.support.waiting import wait_for


class _SkillsPanelApp(App):
    def compose(self) -> ComposeResult:
        yield Static("placeholder")


async def _wait_for_skill_cards(panel: SkillsConfigPanel, pilot, count: int) -> list[SkillDirCard]:
    def cards_ready() -> bool:
        cards = list(panel.query(SkillDirCard))
        return len(cards) == count and all(
            cards[index].is_mounted and cards[index].query(f"#sk-path-{index}") for index in range(count)
        )

    await wait_for(cards_ready, pilot=pilot, description=f"{count} skill directory cards and path inputs are mounted")
    return list(panel.query(SkillDirCard))


async def _add_skill_dir(panel: SkillsConfigPanel, pilot, *, count: int = 1) -> SkillDirCard:
    panel.query_one("#sk-add-btn").press()
    cards = await _wait_for_skill_cards(panel, pilot, count)
    return cards[count - 1]


def _set_fake_platform(monkeypatch: pytest.MonkeyPatch, os_name: str) -> None:
    fake_platform = SimpleNamespace(
        is_macos=os_name == "macos",
        is_windows=os_name == "windows",
        is_linux=os_name == "linux",
    )
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)


def test_skill_path_normalization_hint_keeps_english_and_localizes_chinese(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_fake_platform(monkeypatch, "linux")
    assert skill_path_duplicate_match_description() == "paths are matched after normalization"
    assert skill_path_duplicate_match_description(Localizer("zh-Hans").render) == "路径规范化后进行匹配"


@pytest.mark.asyncio
async def test_add_skill_directory_inserts_new_card_before_existing_paths() -> None:
    """Adding another directory should put the blank row first without wiping existing input."""

    app = _SkillsPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig())
        await app.mount(panel)
        await pilot.pause()

        first = await _add_skill_dir(panel, pilot)
        first.query_one("#sk-path-0", Input).value = "/tmp/skills-one"

        await _add_skill_dir(panel, pilot, count=2)

        cards = list(panel.query(SkillDirCard))
        assert cards[0].query_one("#sk-path-0", Input).value == ""
        assert cards[1].query_one("#sk-path-1", Input).value == "/tmp/skills-one"
        assert panel.get_config().paths == ["/tmp/skills-one"]


@pytest.mark.asyncio
async def test_remove_skill_directory_preserves_edited_remaining_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing a row should snapshot live edits before rebuilding the remaining cards."""

    unmount_started, release_unmount = asyncio.Event(), asyncio.Event()
    release_unmount.set()

    class HeldSkillDirCard(SkillDirCard):
        async def on_unmount(self) -> None:
            if self._index == 1:
                unmount_started.set()
                await release_unmount.wait()

    monkeypatch.setattr(skills_module, "SkillDirCard", HeldSkillDirCard)
    app = _SkillsPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig(paths=["/tmp/old-one", "/tmp/old-two"]))
        await app.mount(panel)
        await pilot.pause()

        cards = await _wait_for_skill_cards(panel, pilot, 2)
        cards[1].query_one("#sk-path-1", Input).value = "/tmp/new-two"
        container = panel.query_one("#sk-dirs")
        with patch.object(container, "mount", autospec=True, side_effect=container.mount) as mount:
            try:
                release_unmount.clear()
                cards[0].query_one("#sk-delete-btn-0").press()
                await wait_for(unmount_started.is_set, description="old skill cards started unmounting")
                await wait_for(
                    lambda: list(panel.query(SkillDirCard)) == [cards[1]],
                    description="only the held old skill card remains",
                )
                # The old card's children are gone and its original path is stale.
                # Its count now matches the new list, so count alone is not readiness.
                assert cards[1].get_path() == "/tmp/old-two"
                mount.assert_not_called()
                assert panel.get_config().paths == ["/tmp/new-two"]
            finally:
                release_unmount.set()

        cards = await _wait_for_skill_cards(panel, pilot, 1)
        assert len(cards) == 1
        assert cards[0].query_one("#sk-path-0", Input).value == "/tmp/new-two"
        assert panel.get_config().paths == ["/tmp/new-two"]
        cards[0].query_one("#sk-path-0", Input).value = "/tmp/edited-again"
        assert panel.get_config().paths == ["/tmp/edited-again"]


async def test_remove_skill_directory_preserves_scroll_position() -> None:
    app = _SkillsPanelApp()
    async with app.run_test(size=(120, 24)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig(paths=[f"/tmp/skills-{index}" for index in range(15)]))
        await app.mount(panel)
        cards = await _wait_for_skill_cards(panel, pilot, 15)
        await wait_for(lambda: panel.max_scroll_y > 100, pilot=pilot)
        panel.scroll_to(y=100, animate=False, immediate=True)
        await wait_for(lambda: panel.scroll_y == 100, pilot=pilot)
        previous_max = panel.max_scroll_y

        container = panel.query_one("#sk-dirs")
        remove_children = container.remove_children

        async def remove_before_refresh() -> None:
            await remove_children()
            # Schedule the normal refresh boundary while the list is empty.
            # batch_update must defer this until replacement cards are mounted.
            app.screen.refresh(layout=True)
            app.screen._on_timer_update()

        with patch.object(container, "remove_children", autospec=True, side_effect=remove_before_refresh):
            cards[-1].query_one("#sk-delete-btn-14").press()
            await _wait_for_skill_cards(panel, pilot, 14)
        await wait_for(
            lambda: 100 < panel.max_scroll_y < previous_max,
            pilot=pilot,
            description="remaining skill cards have been laid out",
        )
        assert panel.scroll_y == 100


@pytest.mark.parametrize(
    "paths",
    [
        ["/tmp/skills", "/tmp/skills/"],
        ["/tmp/skills", "  /tmp/skills  "],
        ["/tmp/skills", "\\tmp\\skills"],
        [str(Path.home() / "skills"), "~/skills"],
        ["/tmp/skills", "/TMP/Skills"],
    ],
)
@pytest.mark.parametrize("os_name", ["macos", "windows"])
@pytest.mark.asyncio
async def test_validate_rejects_duplicate_skill_directory_paths_on_case_insensitive_platforms(
    paths: list[str],
    os_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate skill directories should block Save instead of silently repeating discovery."""

    _set_fake_platform(monkeypatch, os_name)
    app = _SkillsPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig(paths=paths))
        await app.mount(panel)
        await pilot.pause()

        errors = panel.validate()

    assert (
        f"Skill Directory 2: '{paths[1].strip()}' duplicates Skill Directory 1 "
        "(paths are matched case-insensitively after normalization)."
    ) in errors


@pytest.mark.asyncio
async def test_validate_allows_case_differing_skill_directory_paths_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux skill directories should remain case-sensitive during duplicate validation."""

    _set_fake_platform(monkeypatch, "linux")
    app = _SkillsPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig(paths=["/tmp/skills", "/TMP/Skills"]))
        await app.mount(panel)
        await pilot.pause()

        errors = panel.validate()

    assert errors == []


@pytest.mark.asyncio
async def test_validate_still_normalizes_duplicate_skill_directory_paths_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux duplicate validation should normalize paths without case folding."""

    _set_fake_platform(monkeypatch, "linux")
    app = _SkillsPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig(paths=["/tmp/skills", "/tmp/skills/"]))
        await app.mount(panel)
        await pilot.pause()

        errors = panel.validate()

    assert (
        "Skill Directory 2: '/tmp/skills/' duplicates Skill Directory 1 (paths are matched after normalization)."
    ) in errors
