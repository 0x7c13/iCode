# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for Chrys' terminal-friendly checkbox widget."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Checkbox as TextualCheckbox

from chrys.app.tui.widgets.checkbox import CHECKED_MARKER, UNCHECKED_MARKER, Checkbox


class _CheckboxApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Checkbox("Enabled", value=True, id="enabled")
        yield Checkbox("Disabled", value=False, id="disabled")


@pytest.mark.asyncio
async def test_checkbox_renders_ascii_state_markers() -> None:
    async with _CheckboxApp().run_test() as pilot:
        await pilot.pause()

        enabled = pilot.app.query_one("#enabled", Checkbox)
        disabled = pilot.app.query_one("#disabled", Checkbox)

        assert isinstance(enabled, TextualCheckbox)
        assert enabled.render().plain.startswith(CHECKED_MARKER)
        assert disabled.render().plain.startswith(UNCHECKED_MARKER)

        disabled.value = True
        await pilot.pause()

        assert disabled.render().plain.startswith(CHECKED_MARKER)
