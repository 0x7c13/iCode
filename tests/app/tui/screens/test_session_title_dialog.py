# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for SessionTitleDialog — submit, clear, and cancel semantics."""

from __future__ import annotations

import pytest
from textual.app import App

from chrys.app.tui.screens.dialogs.session_title import SessionTitleDialog
from chrys.app.tui.widgets import EnhancedInput

pytestmark = pytest.mark.asyncio


class _DialogApp(App):
    pass


async def test_enter_submits_stripped_custom_title() -> None:
    app = _DialogApp()
    async with app.run_test() as pilot:
        results: list[str | None] = []
        dialog = SessionTitleDialog(custom_title="", auto_title="Auto title")
        app.push_screen(dialog, results.append)
        await pilot.pause()

        field = dialog.query_one("#session-title-input", EnhancedInput)
        assert field.has_focus
        assert field.placeholder == "Auto title"
        field.value = "  My custom title  "
        await pilot.press("enter")
        await pilot.pause()

        assert results == ["My custom title"]


async def test_empty_submit_clears_custom_title() -> None:
    app = _DialogApp()
    async with app.run_test() as pilot:
        results: list[str | None] = []
        dialog = SessionTitleDialog(custom_title="Pinned", auto_title="Auto")
        app.push_screen(dialog, results.append)
        await pilot.pause()

        field = dialog.query_one("#session-title-input", EnhancedInput)
        assert field.value == "Pinned"
        field.value = ""
        await pilot.press("enter")
        await pilot.pause()

        assert results == [""]


async def test_escape_cancels_without_result() -> None:
    app = _DialogApp()
    async with app.run_test() as pilot:
        results: list[str | None] = []
        dialog = SessionTitleDialog(custom_title="Pinned", auto_title="")
        app.push_screen(dialog, results.append)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        assert results == [None]
