# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The editor exposes useful colors without losing advanced YAML data."""

from pathlib import Path

import pytest
from textual.theme import Theme
from textual.widgets import Input

from chrys.app.tui.screens.themes.palette import _ANSI_ONLY_EDITOR_VARIABLES
from chrys.app.tui.theme import CHRYS_ANSI_THEME, CHRYS_THEME
from chrys.app.tui.themes.document import copy_theme
from chrys.app.tui.themes.store import UserThemeStore
from chrys.app.tui.widgets.select import Select
from tests.support.waiting import wait_for

from .helpers import make_app, open_editor, press_button, wait_for_save_dialog


@pytest.mark.parametrize("ansi", [False, True])
async def test_curated_fields_follow_color_mode_and_keep_hidden_overrides_on_save(tmp_path: Path, ansi: bool) -> None:
    source = copy_theme(CHRYS_ANSI_THEME if ansi else CHRYS_THEME)
    hidden = {
        "footer-key-background": "#112233",
        "footer-description-background": "#223344",
        "footer-item-background": "#334455",
        "footer-foreground": "#EEEEEE",
        "scrollbar-background-hover": "#223344",
        "scrollbar-background-active": "#112233",
        "button-primary-foreground": "#FFFFFF",
        "unused-user-token": "#ABCDEF",
        "button-focus-text-style": "bold",
    }
    source.variables.update(hidden)
    app = make_app(tmp_path, source, component=True)
    async with app.run_test(size=(140, 50)) as pilot:
        editor = await open_editor(app, pilot, tmp_path)
        visible = editor._variable_swatches.keys()
        assert not hidden.keys() & visible
        if ansi:
            assert not {"surface", "panel", "boost"} & editor._color_buttons.keys()
        else:
            assert {"surface", "panel", "boost"} <= editor._color_buttons.keys()
        assert (visible >= _ANSI_ONLY_EDITOR_VARIABLES) is ansi
        if not ansi:
            assert not _ANSI_ONLY_EDITOR_VARIABLES & visible
        assert {
            "footer-background",
            "footer-key-foreground",
            "footer-description-foreground",
            "scrollbar-background",
            "border-color",
            "control-disabled-background",
            "text-disabled",
            "hatch-color",
            "tool-group-title-color",
            "markdown-h1-color",
            "markdown-h2-color",
            "markdown-h3-color",
            "markdown-h6-color",
        } <= visible
        headings = {f"markdown-h{level}-color": f"#{level}23456" for level in (range(1, 7) if ansi else (1, 2, 3, 6))}
        assert {name for name in visible if name.startswith("markdown-h")} == headings.keys()
        for name, value in headings.items():
            token = editor.document.begin(f"var:{name}")
            editor.preview_edit(token, value)
            assert editor.commit_edit(token)
        token = editor.document.begin("var:footer-background")
        editor.preview_edit(token, "#123456")
        assert editor.commit_edit(token)
        await editor.open_save()
        dialog = await wait_for_save_dialog(pilot)
        dialog.query_one(Input).value = "curated"
        await press_button(pilot, "#save-theme-confirm")
        await wait_for(lambda: app.screen is editor.screen and not editor.document.unsaved, pilot=pilot)
        saved, _revision = editor.store.load("curated")
        assert {key: saved.variables[key] for key in hidden} == hidden
        assert saved.variables["footer-background"] == "#123456"
        assert {name: saved.variables[name] for name in headings} == headings
        selector = editor.query_one("#theme-select", Select)
        other = "chrys" if ansi else "chrys-ansi"
        selector.value = other
        await wait_for(lambda: editor.document.draft.name == other and not editor._switching, pilot=pilot)
        if ansi:
            assert not _ANSI_ONLY_EDITOR_VARIABLES & editor._variable_swatches.keys()
            assert {"surface", "panel", "boost"} <= editor._color_buttons.keys()
        else:
            assert editor._variable_swatches.keys() >= _ANSI_ONLY_EDITOR_VARIABLES
            assert not {"surface", "panel", "boost"} & editor._color_buttons.keys()


async def test_minimal_user_ansi_theme_can_load_and_reset_terminal_color_overrides(tmp_path: Path) -> None:
    theme = Theme(name="custom-ansi", primary="ansi_magenta", ansi=True)
    store = UserThemeStore(tmp_path / "themes")
    store.save(theme, None)
    app = make_app(tmp_path, theme, component=True)
    app.theme = theme.name
    async with app.run_test(size=(140, 50)) as pilot:
        editor = await open_editor(app, pilot, tmp_path)
        assert app.current_theme.name == theme.name
        for name in ("ansi-background", "ansi-foreground"):
            assert name not in editor.document.draft.variables
            assert app.get_css_variables()[name] == "ansi_default"
            token = editor.document.begin(f"var:{name}")
            editor.preview_edit(token, "ansi_red")
            assert editor.commit_edit(token)
            assert app.get_css_variables()[name] == "ansi_red"
            editor._reset_buttons[f"var:{name}"].press()
            await wait_for(lambda name=name: name not in editor.document.draft.variables, pilot=pilot)
            assert app.get_css_variables()[name] == "ansi_default"
        assert app.theme_preview is not None
        assert not app.theme_preview.error
