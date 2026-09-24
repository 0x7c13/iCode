# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Screen-stack exposure must not release tests before modal controls mount."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

import pytest
from textual.widgets import OptionList

from chrys.app.tui.screens.settings.dialog import SettingsDialog
from chrys.app.tui.screens.themes.picker import ThemesScreen
from chrys.app.tui.widgets.select import Select
from tests.app.tui.screens.settings.support import StubPorts
from tests.support.paths import SRC_ROOT
from tests.support.waiting import wait_for

from .helpers import make_app, wait_for_settings_dialog, wait_for_themes


@pytest.mark.parametrize("kind", ["themes", "settings"])
async def test_dialog_readiness_waits_for_controls_before_keyboard_use(tmp_path: Path, kind: str) -> None:
    composing, release = asyncio.Event(), asyncio.Event()

    class DelayedThemes(ThemesScreen):
        async def _compose(self) -> None:
            composing.set()
            await release.wait()
            await super()._compose()

    class DelayedSettings(SettingsDialog):
        CSS_PATH: ClassVar[Path] = SRC_ROOT / "chrys/app/tui/screens/settings/settings.tcss"

        async def _compose(self) -> None:
            composing.set()
            await release.wait()
            await super()._compose()

    app = make_app(tmp_path, component=True)
    wait_ready = wait_for_themes if kind == "themes" else wait_for_settings_dialog
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = DelayedThemes(app.theme) if kind == "themes" else DelayedSettings(StubPorts())
        original = app.screen

        async def open_dialog() -> None:
            await app.push_screen(dialog)

        opening = asyncio.create_task(open_dialog())
        ready = None
        try:
            await wait_for(lambda: composing.is_set() or opening.done())
            if opening.done():
                await opening
            assert composing.is_set() and app.screen is dialog
            assert not dialog.is_mounted and not dialog.children
            ready = asyncio.create_task(wait_ready(pilot))
            # Exercise the exposed-but-uncomposed state for one scheduler turn.
            await asyncio.sleep(0)
            assert not ready.done()
        finally:
            release.set()
            try:
                # Await the real mount independently of the waiter under test,
                # even when a broken predicate returned before composition.
                await opening
            finally:
                if ready is not None:
                    await ready

        if isinstance(dialog, ThemesScreen):
            options = dialog.query_one(OptionList)
            options.highlighted = options.get_option_index("dracula")
            await wait_for(lambda: app.theme == "dracula", pilot=pilot)
            await pilot.press("enter")
            await wait_for(lambda: app.screen is original, pilot=pilot)
            assert app.theme == "dracula"
        else:
            theme = next(row for row in dialog.rows() if row.spec.key == "ui.theme").query_one(Select)
            assert theme.has_focus
            await pilot.press("enter")
            await wait_for(lambda: theme.expanded, pilot=pilot)
            await pilot.press("escape")
            await wait_for(lambda: not theme.expanded, pilot=pilot)
