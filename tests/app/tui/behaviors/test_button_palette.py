# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Button palette overrides survive renaming and cover the rendered states."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.color import Color
from rich.text import Text
from textual.app import App, ComposeResult
from textual.color import Color as TextualColor
from textual.theme import Theme
from textual.widgets import Button

from chrys.app.tui.theme import (
    CHRYS_ANSI_THEME,
    CHRYS_LEGACY_THEME,
    CHRYS_THEME,
    TuiVariableDefaultsMixin,
)
from chrys.app.tui.themes.document import copy_theme
from chrys.app.tui.themes.preview import theme_variables
from chrys.app.tui.themes.store import UserThemeStore
from chrys.app.tui.widgets.chrome.input_bar import InputBar
from tests.support.paths import SRC_ROOT
from tests.support.waiting import wait_for


class ButtonPaletteApp(TuiVariableDefaultsMixin, App):
    CSS_PATH = SRC_ROOT / "chrys" / "app" / "tui" / "chrys.tcss"

    def compose(self) -> ComposeResult:
        for variant in ("default", "primary", "success", "warning", "error"):
            yield Button(Text(variant), variant=variant, flat=True, id=variant)


@pytest.mark.parametrize("source", [CHRYS_THEME, CHRYS_LEGACY_THEME, CHRYS_ANSI_THEME], ids=lambda theme: theme.name)
async def test_saved_theme_copy_preserves_flat_button_states_without_chrys_name(tmp_path: Path, source: Theme) -> None:
    store = UserThemeStore(tmp_path / "themes")
    store.save(copy_theme(source, name="custom"), None)
    saved, _revision = store.load("custom")
    app = ButtonPaletteApp()
    app.register_theme(source)
    app.register_theme(saved)
    app.theme = source.name
    snapshots = []
    async with app.run_test(size=(100, 30)) as pilot:
        buttons = list(app.query(Button))
        for name in (source.name, saved.name):
            app.theme = name
            app.screen.set_focus(None)
            await pilot.hover(offset=(90, 29))
            await wait_for(lambda: app.focused is None, pilot=pilot)
            states = [tuple(button.rich_style for button in buttons)]
            assert all(style.color == Color.parse("#000000") for style in states[0])
            await pilot.hover("#primary")
            states.append(buttons[1].rich_style)
            buttons[1].focus()
            await wait_for(lambda: buttons[1].has_focus, pilot=pilot)
            states.append(buttons[1].rich_style)
            buttons[1].add_class("-active")
            states.append((buttons[1].rich_style, buttons[1].styles.tint))
            buttons[1].remove_class("-active")
            for button in buttons:
                button.disabled = True
            states.append(tuple(button.rich_style for button in buttons))
            snapshots.append(states)
            for button in buttons:
                button.disabled = False
        assert snapshots[0] == snapshots[1]
        assert not app.has_class("-chrys")


async def test_flat_button_overrides_apply_to_variants_and_hover_disabled_pressed_states() -> None:
    theme = Theme(
        name="custom",
        primary="#8060C0",
        variables={
            "button-flat-foreground": "#112233",
            "button-background": "#CCDDEE",
            "button-primary-background": "#AABBCC",
            "button-primary-foreground": "#334455",
            "button-success-background": "#ABCDEF",
            "button-warning-background": "#FEDCBA",
            "button-error-background": "#FEDCDE",
            "button-hover-foreground": "#123456",
            "button-hover-background": "#DDDDDD",
            "button-disabled-foreground": "#808080",
            "button-disabled-background": "#303030",
            "button-active-tint": "#FF0000 20%",
        },
    )
    app = ButtonPaletteApp()
    app.register_theme(theme)
    app.theme = theme.name
    async with app.run_test(size=(100, 30)) as pilot:
        app.screen.set_focus(None)
        await wait_for(lambda: app.focused is None, pilot=pilot)
        buttons = list(app.query(Button))
        for button, fill in zip(buttons, ("#CCDDEE", "#AABBCC", "#ABCDEF", "#FEDCBA", "#FEDCDE"), strict=True):
            assert button.rich_style.bgcolor == Color.parse(fill)
            assert button.rich_style.color == Color.parse("#334455" if button.variant == "primary" else "#112233")
        await pilot.hover("#primary")
        assert buttons[1].rich_style.color == Color.parse("#123456")
        assert buttons[1].rich_style.bgcolor == Color.parse("#DDDDDD")
        buttons[1].add_class("-active")
        assert buttons[1].styles.tint.a == pytest.approx(0.2)
        buttons[1].remove_class("-active")
        for button in buttons:
            button.disabled = True
            assert button.rich_style.color == Color.parse("#808080")
            # Textual also fades disabled controls with widget opacity.
            assert button.styles.background == TextualColor.parse("#303030")
        assert app.get_css_variables() == theme_variables(theme, app.get_theme_variable_defaults())


@pytest.mark.parametrize("source", [CHRYS_THEME, CHRYS_ANSI_THEME], ids=lambda theme: theme.name)
async def test_new_session_button_uses_hover_palette_without_expanding_the_input_bar(source: Theme) -> None:
    class InputApp(TuiVariableDefaultsMixin, App):
        CSS_PATH = SRC_ROOT / "chrys" / "app" / "tui" / "chrys.tcss"

        def compose(self) -> ComposeResult:
            yield InputBar()

    theme = copy_theme(source)
    theme.variables.update({"button-hover-background": "#224466", "button-hover-foreground": "#FFFFFF"})
    app = InputApp()
    app.register_theme(theme)
    app.theme = theme.name
    async with app.run_test(size=(100, 30)) as pilot:
        bar = app.query_one(InputBar)
        bar.has_messages = True
        button = bar.query_one("#new-btn", Button)
        await wait_for(lambda: button.content_size.width > 0, pilot=pilot)
        assert button.region.height == 1
        assert button.variant == "primary"
        normal = button.rich_style.bgcolor
        await pilot.hover(button)
        assert button.rich_style.bgcolor == Color.parse("#224466")
        assert button.rich_style.color == Color.parse("#FFFFFF")
        assert button.rich_style.bgcolor != normal
        assert button.region.height == 1
