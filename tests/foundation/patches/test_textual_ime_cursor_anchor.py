# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Textual IME cursor anchor patch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.events import Key
from textual.widgets import Input, Static, TextArea

from chrys.foundation.patches import textual_ime_cursor_anchor
from chrys.foundation.patches.textual_ime_cursor_anchor import apply_runtime_patch
from chrys.foundation.patches.textual_kitty_keyboard import apply_runtime_patch as apply_keyboard_runtime_patch


class TextAreaCursorLayoutApp(App[None]):
    CSS = "#top { height: 1; } TextArea { height: 3; }"

    def compose(self) -> ComposeResult:
        yield Static("top", id="top")
        yield TextArea("abc", id="text-area", compact=True, show_line_numbers=False)


class InputCursorLayoutApp(App[None]):
    CSS = "#top { height: 1; } Input { width: 20; }"

    def compose(self) -> ComposeResult:
        yield Static("top", id="top")
        yield Input("abc", id="input")


def test_file_patch_fragment_matches_installed_textual() -> None:
    spec = importlib.util.find_spec("textual")
    assert spec is not None
    assert spec.origin is not None
    source = (Path(spec.origin).parent / "app.py").read_text(encoding="utf-8")

    assert (
        textual_ime_cursor_anchor._OLD_BEGIN_UPDATE in source or textual_ime_cursor_anchor._NEW_BEGIN_UPDATE in source
    )


@pytest.mark.parametrize(
    ("app_type", "selector", "widget_type"),
    [
        (TextAreaCursorLayoutApp, "#text-area", TextArea),
        (InputCursorLayoutApp, "#input", Input),
    ],
)
async def test_runtime_patch_keeps_cursor_anchor_synced_after_layout_shift(
    app_type: type[App[None]],
    selector: str,
    widget_type: type[TextArea] | type[Input],
) -> None:
    apply_runtime_patch()
    app = app_type()

    async with app.run_test(size=(40, 10)) as pilot:
        widget = app.query_one(selector, widget_type)
        top = app.query_one("#top", Static)

        await pilot.pause()
        widget.focus()
        await pilot.pause()
        app._begin_update()
        assert app.cursor_position == widget.cursor_screen_offset

        top.styles.height = 3
        top.refresh(layout=True)
        await pilot.pause()
        app._begin_update()

        assert app.cursor_position == widget.cursor_screen_offset


async def test_runtime_patches_decode_ime_commit_and_anchor_cursor_after_layout_shift() -> None:
    apply_keyboard_runtime_patch()
    apply_runtime_patch()
    app = TextAreaCursorLayoutApp()

    async with app.run_test(size=(40, 10)) as pilot:
        text_area = app.query_one("#text-area", TextArea)
        top = app.query_one("#top", Static)

        await pilot.pause()
        text_area.focus()
        await pilot.pause()

        from textual._xterm_parser import XTermParser

        keys = [event for event in XTermParser().feed("\x1b[32;;24403:21069u") if isinstance(event, Key)]
        assert [(key.key, key.character) for key in keys] == [("当前", "当前")]

        await text_area._on_key(keys[0])
        await pilot.pause()

        assert text_area.text == "当前abc"
        assert text_area.cursor_location == (0, 2)

        top.styles.height = 3
        top.refresh(layout=True)
        await pilot.pause()
        app._begin_update()

        assert app.cursor_position == text_area.cursor_screen_offset
