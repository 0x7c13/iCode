# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Spec tests for what the Settings → Shortcuts folder rows save while one of them is invalid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from textual.widgets import Button, Checkbox, Input

from chrys.app.tui.screens.settings import SettingsDialog
from chrys.app.tui.screens.settings.layout import PROMPT_SHORTCUTS_TAB_ID
from chrys.app.tui.screens.settings.panes.prompt_shortcuts import PATHS_KEY, PromptDirCard, PromptShortcutsPane
from tests.app.tui.screens.settings.support import Host, StubPorts, wait_for_every_tab
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from textual.pilot import Pilot


def _folders(tmp_path: Path, *names: str) -> list[str]:
    paths = []
    for name in names:
        folder = tmp_path / name
        folder.mkdir()
        paths.append(str(folder))
    return paths


def _saved(ports: StubPorts) -> list[list[str]]:
    return [json.loads(value) for key, value in ports.live if key == PATHS_KEY]


def _cards(pane: PromptShortcutsPane) -> list[PromptDirCard]:
    return list(pane.query(PromptDirCard))


def _field(card: PromptDirCard) -> Input:
    return card.query_one(".sk-path-row Input", Input)


async def _pane_with(pilot: Pilot, ports: StubPorts, saved: list[str]) -> tuple[SettingsDialog, PromptShortcutsPane]:
    ports.desired[PATHS_KEY] = json.dumps(saved)
    dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID)
    await pilot.app.push_screen(dialog)
    await wait_for_every_tab(dialog, pilot)
    pane = dialog.query_one(PromptShortcutsPane)
    await wait_for(
        lambda: [_field(card).value for card in _cards(pane)] == saved, pilot=pilot, description="saved rows shown"
    )
    return dialog, pane


async def _make_invalid(pilot: Pilot, card: PromptDirCard) -> None:
    """Turn "Workspace relative" on for the row's absolute folder: the row now says why it cannot be used."""
    card.query_one(".sk-rel-cb", Checkbox).value = True
    await wait_for(lambda: card.entry() is None, pilot=pilot, description="row invalid")


async def test_closing_saves_the_valid_rows_while_another_row_is_invalid(tmp_path: Path) -> None:
    one, two, three = _folders(tmp_path, "one", "two", "three")
    ports = StubPorts(root=tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        _dialog, pane = await _pane_with(pilot, ports, [one, two])
        first, second = _cards(pane)
        await _make_invalid(pilot, second)
        # The first row's new folder is typed but never left: only closing saves it.
        field = _field(first)
        field.focus()
        await wait_for(lambda: field.has_focus, pilot=pilot, description="first row focused")
        field.value = three

        await pilot.press("escape")
        await wait_for(lambda: ports.closed == 1, pilot=pilot, description="dialog closed")

        # The edit lands; the invalid row keeps the folder saved for it.
        assert _saved(ports)[-1:] == [[three, two]]


async def test_an_invalid_row_never_drops_the_folder_saved_for_it(tmp_path: Path) -> None:
    (one,) = _folders(tmp_path, "one")
    ports = StubPorts(root=tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        _dialog, pane = await _pane_with(pilot, ports, [one])
        (card,) = _cards(pane)
        await _make_invalid(pilot, card)

        await pilot.press("escape")
        await wait_for(lambda: ports.closed == 1, pilot=pilot, description="dialog closed")

        assert all(one in saved for saved in _saved(ports))


async def test_removing_a_row_is_saved_while_another_row_is_invalid(tmp_path: Path) -> None:
    one, two = _folders(tmp_path, "one", "two")
    ports = StubPorts(root=tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        _dialog, pane = await _pane_with(pilot, ports, [one, two])
        first, second = _cards(pane)
        await _make_invalid(pilot, second)

        first.query_one(".config-card-delete-btn", Button).press()

        await wait_for(lambda: bool(_saved(ports)), pilot=pilot, description="removal saved")
        assert all(saved == [two] for saved in _saved(ports))
