# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for shared config Add button styling hooks."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.color import Color
from textual.widgets import Input

from chrys.app.tui.theme import TUI_VARIABLE_DEFAULTS
from chrys.app.tui.widgets.buttons import ConfigActionButton, ConfigAddButton
from tests.support.paths import SRC_ROOT
from tests.support.waiting import wait_for

_CHRYS_CSS = SRC_ROOT / "chrys" / "app" / "tui" / "chrys.tcss"


class _ConfigAddButtonStyleApp(App):
    CSS_PATH = str(_CHRYS_CSS)

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return dict(TUI_VARIABLE_DEFAULTS)

    def on_mount(self) -> None:
        self.theme = "textual-light"

    def compose(self) -> ComposeResult:
        yield ConfigAddButton("+ Add", id="add-btn")


class _FakeClick:
    def __init__(self) -> None:
        self.stopped = False
        self.prevented = False

    def stop(self) -> None:
        self.stopped = True

    def prevent_default(self) -> None:
        self.prevented = True


class _ConfigAddButtonFocusApp(App):
    def get_theme_variable_defaults(self) -> dict[str, str]:
        return dict(TUI_VARIABLE_DEFAULTS)

    def compose(self) -> ComposeResult:
        yield Input(id="focus-target")
        yield ConfigActionButton("+ Add", id="action-btn")


@pytest.mark.asyncio
async def test_config_action_button_mouse_click_drops_focus_after_press() -> None:
    app = _ConfigAddButtonFocusApp()
    async with app.run_test() as pilot:
        button = app.query_one("#action-btn", ConfigActionButton)
        button.focus()
        await wait_for(lambda: button.has_focus, pilot=pilot, description="control focus before interaction")
        button.mouse_hover = True
        await pilot.pause()

        assert button.has_focus is True
        assert button.mouse_hover is True

        click = _FakeClick()
        await button._on_click(click)
        await pilot.pause()

        assert click.stopped is True
        assert click.prevented is True
        assert button.has_focus is False
        assert app.screen.focused is None
        assert button.mouse_hover is False


@pytest.mark.asyncio
async def test_config_action_button_programmatic_press_preserves_focus() -> None:
    app = _ConfigAddButtonFocusApp()
    async with app.run_test() as pilot:
        button = app.query_one("#action-btn", ConfigActionButton)
        button.focus()
        await wait_for(lambda: button.has_focus, pilot=pilot, description="control focus before interaction")
        button.mouse_hover = True
        await pilot.pause()

        button.press()
        await pilot.pause()

        assert button.has_focus is True
        assert app.screen.focused is button
        assert button.mouse_hover is True


@pytest.mark.asyncio
async def test_config_add_button_uses_theme_secondary_foreground_on_light_theme() -> None:
    app = _ConfigAddButtonStyleApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        button = app.query_one("#add-btn", ConfigAddButton)

        assert button.styles.color == Color.parse(app.current_theme.secondary)
