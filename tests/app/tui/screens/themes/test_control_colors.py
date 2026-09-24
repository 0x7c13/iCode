# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Disabled controls and Rich/CSS hatches share editable theme colors."""

from __future__ import annotations

import pytest
from rich.color import Color as RichColor
from textual.app import App, ComposeResult
from textual.css.stylesheet import Stylesheet
from textual.widgets import Button, Input, Static, TextArea
from textual.widgets._select import SelectCurrent

from chrys.app.tui.screens.agents.panels.basic import BasicConfigPanel
from chrys.app.tui.theme import (
    CHRYS_ANSI_THEME,
    CHRYS_THEME,
    TuiVariableDefaultsMixin,
    concrete_theme_color,
    with_tui_css_variables,
)
from chrys.app.tui.themes.document import copy_theme
from chrys.app.tui.widgets.chat.session_json import SessionJsonPanel
from chrys.app.tui.widgets.diff_view import DiffView
from chrys.app.tui.widgets.diff_view.code import CodeColumn
from chrys.app.tui.widgets.diff_view.gutter import GutterColumn
from chrys.app.tui.widgets.hatch import HatchedEmptyState, hatch_text_style
from chrys.app.tui.widgets.select import Select
from chrys.service.profiles.agents.schema import AgentProfile
from tests.support.paths import SRC_ROOT
from tests.support.waiting import wait_for


class ControlApp(TuiVariableDefaultsMixin, App):
    CSS_PATH = SRC_ROOT / "chrys" / "app" / "tui" / "chrys.tcss"

    def compose(self) -> ComposeResult:
        yield BasicConfigPanel(AgentProfile(name="Code"), is_builtin=True)
        yield TextArea("disabled", disabled=True, id="disabled-area")
        yield HatchedEmptyState("", id="hatch")
        yield SessionJsonPanel()
        yield DiffView("old.py", "new.py", "a\nb\n", "a\nb\nc\n")


@pytest.mark.parametrize("ansi", [False, True])
async def test_disabled_controls_and_hatches_follow_user_colors(ansi: bool) -> None:
    app = ControlApp()
    theme = copy_theme(CHRYS_ANSI_THEME if ansi else CHRYS_THEME, name="custom-controls")
    theme.variables.update(
        {"text-disabled": "#AABBCC", "control-disabled-background": "#123456", "hatch-color": "#FF0000 50%"}
    )
    app.register_theme(theme)
    app.theme = theme.name
    async with app.run_test(size=(100, 80)) as pilot:
        await pilot.pause()
        widgets = [app.query_one("#bc-name", Input), app.query_one("#disabled-area", TextArea)]
        selector = app.query_one("#bc-agent-type", Select)
        widgets.append(selector.query_one(SelectCurrent))
        for widget in widgets:
            assert widget.is_disabled
            assert widget.styles.background.hex == "#123456"
            assert widget.styles.color.hex == "#AABBCC"
            assert widget.styles.opacity == 1
        assert selector.styles.opacity == 1
        assert selector.query_one("#label").styles.color.hex == "#AABBCC"
        hatch = app.query_one(HatchedEmptyState)
        rich_host = app.query_one(SessionJsonPanel)
        assert hatch_text_style(rich_host).color == next(iter(hatch.render_line(0))).style.color
        diff = app.query_one(DiffView)
        columns = [column for column in diff.query("GutterColumn, CodeColumn") if "╲" in column.render_line(2).text]
        assert len(columns) == 3
        assert sum(isinstance(column, GutterColumn) for column in columns) == 2
        assert sum(isinstance(column, CodeColumn) for column in columns) == 1
        initial = hatch_text_style(rich_host)
        assert initial.color is not None and initial.color != RichColor.parse("#FF0000")
        # Both render paths resolve native palette slots and alpha through CSS.
        other = copy_theme(theme, name="other-controls")
        other.variables["hatch-color"] = "ansi_cyan"
        app.register_theme(other)
        app.theme = other.name
        await wait_for(lambda: hatch_text_style(rich_host).color == RichColor.from_ansi(6), pilot=pilot)
        assert next(iter(hatch.render_line(0))).style.color == RichColor.from_ansi(6)
        # Existing gutter/code widgets must repaint without recomposing.
        for column in columns:
            assert column.is_mounted
            for segment in column.render_line(2):
                if "╲" in segment.text:
                    assert segment.style.color == RichColor.from_ansi(6)


@pytest.mark.parametrize(
    ("value", "background", "dark", "expected"),
    [
        ("auto 50%", "#202020", True, "#8F8F8F"),
        ("ansi_white 40%", "ansi_default", True, "ansi_white"),
        ("ansi_blue 50%", "#FFFFFF", False, "ansi_blue"),
        ("ansi_default 50%", "#000000", True, "ansi_default"),
        ("auto 50%", "ansi_default", False, "#7F7F7F"),
        ("red 25%", "#000000", True, "#3F0000"),
        ("ansi_cyan", "#000000", True, "ansi_cyan"),
        ("transparent", "#000000", True, "transparent"),
        ("not-a-color 50%", "#000000", True, "not-a-color 50%"),
        ("80%", "#000000", True, "80%"),
    ],
)
def test_legacy_expressions_become_concrete_theme_colors(value, background, dark, expected) -> None:
    assert concrete_theme_color(value, background, dark) == expected


@pytest.mark.parametrize("name", ["ansi-dark", "ansi-light", "chrys-ansi"])
async def test_ansi_expression_colors_keep_terminal_slots_in_rendered_controls(name: str) -> None:
    class AnsiApp(TuiVariableDefaultsMixin, App):
        CSS_PATH = ControlApp.CSS_PATH
        CSS = "#muted { color: $text-muted; }"

        def compose(self) -> ComposeResult:
            yield Static("Muted text", id="muted")
            for variant in ("primary", "error", "success"):
                yield Button(variant, variant=variant, flat=True, id=variant)

    app = AnsiApp()
    app.register_theme(CHRYS_ANSI_THEME)
    app.theme = name
    async with app.run_test() as pilot:
        app.set_focus(None)
        await pilot.pause()
        muted = RichColor.from_ansi(7) if name == "chrys-ansi" else RichColor.parse("default")
        assert app.query_one("#muted").rich_style.color == muted
        if name != "chrys-ansi":
            for variant, slot in (("primary", 4), ("error", 1), ("success", 2)):
                button = app.query_one(f"#{variant}", Button)
                assert button.rich_style.bgcolor == RichColor.from_ansi(slot)
                assert button.styles.border_top[1].ansi == slot


async def test_internal_contrast_roles_still_follow_the_control_fill() -> None:
    class ContrastApp(TuiVariableDefaultsMixin, App):
        CSS_PATH = ControlApp.CSS_PATH

        def compose(self) -> ComposeResult:
            yield Static("Auto", id="approval-badge", classes="approval-auto")
            yield Button("Raised", variant="warning")

    app = ContrastApp()
    app.theme = "textual-dark"
    async with app.run_test() as pilot:
        await pilot.pause()
        for widget in (app.query_one("#approval-badge"), app.query_one(Button)):
            assert widget.rich_style.color == RichColor.parse("#000000")


def test_dependent_button_expressions_resolve_to_the_same_color_as_their_swatches() -> None:
    theme = copy_theme(CHRYS_THEME)
    theme.variables.update({"button-hover-background": "red 25%", "button-hover-foreground": "auto 90%"})
    variables = with_tui_css_variables(theme, {**theme.to_color_system().generate(), **theme.variables})
    assert variables["button-hover-background"] == "#541515"
    assert variables["button-hover-foreground"].startswith("#")
    for variant in ("default", "primary", "success", "warning", "error"):
        assert variables[f"tui-button-{variant}-hover-background"] == variables["button-hover-background"]
        assert variables[f"tui-button-{variant}-hover-foreground"] == variables["button-hover-foreground"]


@pytest.mark.parametrize("root", ["foreground", "background"])
def test_legacy_root_expressions_keep_derived_hatches_and_tints_valid(root: str) -> None:
    theme = copy_theme(CHRYS_THEME)
    theme.variables.pop("button-active-tint")
    theme.variables[root] = "red 40%"
    variables = with_tui_css_variables(theme, {**theme.to_color_system().generate(), **theme.variables})
    assert variables[root].startswith("#")
    assert variables["hatch-color"].startswith("#")
    assert variables["button-active-tint"] == f"{variables['background']} 30%"
    sheet = Stylesheet(variables=variables)
    sheet.add_source(
        "Widget { color: $foreground; background: $background; hatch: left $hatch-color; tint: $button-active-tint; }"
    )
    sheet.parse()


@pytest.mark.parametrize("state", ["flat", "hover", "disabled"])
def test_shared_button_ink_expression_matches_its_swatch_without_a_background_override(state: str) -> None:
    theme = copy_theme(CHRYS_THEME)
    key = f"button-{state}-foreground"
    theme.variables[key] = "auto 90%"
    variables = with_tui_css_variables(theme, {**theme.to_color_system().generate(), **theme.variables})
    assert variables[key].startswith("#")
    for variant in ("default", "primary", "success", "warning", "error"):
        suffix = "" if state == "flat" else f"-{state}"
        assert variables[f"tui-button-{variant}{suffix}-foreground"] == variables[key]
