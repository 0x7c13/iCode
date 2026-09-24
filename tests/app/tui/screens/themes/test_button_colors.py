# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Editing button colors uses the same preview, undo and save path as other colors."""

from __future__ import annotations

from pathlib import Path

from rich.color import Color
from textual.widgets import Button, Input, Label

from chrys.app.tui.theme import BUTTON_COLOR_VARIABLES
from chrys.app.tui.widgets.color_picker import ColorPicker
from tests.support.waiting import wait_for

from .helpers import make_app, open_editor, press_button, wait_for_editor, wait_for_picker, wait_for_save_dialog


async def test_button_ink_can_be_picked_undone_and_saved_under_a_new_name(tmp_path: Path) -> None:
    app = make_app(tmp_path, "chrys-legacy", locale="zh-Hans", component=True)
    async with app.run_test(size=(140, 50)) as pilot:
        editor = await open_editor(app, pilot, tmp_path)
        assert editor.query_one("#theme-save", Button).variant == "success"
        assert str(editor.query_one("#theme-group-button", Label).render()) == "Buttons"
        assert str(editor.query_one("#theme-group-markdown", Label).render()) == "Markdown headings"
        assert {name for name in editor._variable_swatches if name.startswith("button-")} == set(BUTTON_COLOR_VARIABLES)
        assert str(editor.query_one("#theme-label-button-flat-foreground", Label).render()) == "Text"
        app.locale_controller.switch_locale("en")
        await wait_for(lambda: str(editor.query_one("#theme-save", Button).label) == "Save", pilot=pilot)
        assert str(editor.query_one("#theme-label-button-flat-foreground", Label).render()) == "Text"
        assert str(editor.query_one("#theme-label-button-disabled-background", Label).render()) == "Disabled background"
        assert str(editor.query_one("#theme-group-markdown", Label).render()) == "Markdown headings"
        ink = editor._variable_swatches["button-flat-foreground"]
        ink.press()
        modal = await wait_for_picker(pilot, ColorPicker)
        modal.query_one("#color-expression", Input).value = "#FFFFFF"
        close = editor.query_one("#theme-close", Button)
        await wait_for(lambda: close.rich_style.color == Color.parse("#ffffff"), pilot=pilot)
        await press_button(pilot, "#picker-confirm")
        await wait_for(lambda: app.screen is editor.screen, pilot=pilot)
        editor.action_undo()
        assert close.rich_style.color == Color.parse("#000000")
        editor.action_redo()
        assert close.rich_style.color == Color.parse("#ffffff")
        await editor.open_save()
        dialog = await wait_for_save_dialog(pilot)
        assert dialog.query_one("#save-theme-confirm", Button).variant == "success"
        dialog.query_one(Input).value = "custom-buttons"
        await press_button(pilot, "#save-theme-confirm")
        await wait_for(lambda: app.screen is editor.screen and not editor.document.unsaved, pilot=pilot)
        await wait_for_editor(pilot)
        saved, _revision = editor.store.load("custom-buttons")
        assert saved.variables["button-flat-foreground"] == "#FFFFFF"
        # Hidden advanced fields remain theme data when common controls change.
        assert saved.variables["button-color-foreground"] == "#000000"
        assert saved.variables["button-active-tint"] == "transparent"
        await press_button(pilot, "#theme-close")
        await wait_for(lambda: app.theme_preview is None, pilot=pilot)
        assert app.theme == app._settings.theme == "custom-buttons"
        assert app.get_css_variables()["tui-button-warning-foreground"] == "#FFFFFF"
