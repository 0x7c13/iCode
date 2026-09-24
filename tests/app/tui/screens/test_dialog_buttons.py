# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for shared dialog button widgets."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from chrys.app.tui.widgets import DialogButtonRow, DialogButtonSpec


class _DialogButtonRowApp(App):
    def compose(self) -> ComposeResult:
        yield DialogButtonRow(
            DialogButtonSpec("Apply", id="apply", variant="success"),
            DialogButtonSpec("Cancel", id="cancel", variant="warning", disabled=True),
            id="buttons",
        )


@pytest.mark.asyncio
async def test_dialog_button_row_composes_configured_buttons() -> None:
    async with _DialogButtonRowApp().run_test() as pilot:
        await pilot.pause()

        apply = pilot.app.query_one("#apply", Button)
        cancel = pilot.app.query_one("#cancel", Button)

        assert apply.label.plain == "Apply"
        assert apply.variant == "success"
        assert apply.flat is True
        assert cancel.label.plain == "Cancel"
        assert cancel.variant == "warning"
        assert cancel.disabled is True
