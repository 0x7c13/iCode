# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The settled click lands on its target while the layout around it is still moving."""

from __future__ import annotations

from textual import on
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for


class _Dialog(ModalScreen[None]):
    def compose(self) -> ComposeResult:
        yield Button("Close", id="close")

    @on(Button.Pressed, "#close")
    def close(self) -> None:
        self.dismiss()


class _ShiftingApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.pressed: list[str] = []

    def compose(self) -> ComposeResult:
        banner = Static("A row that appears above the buttons\nand takes the place of the first one", id="banner")
        banner.display = False
        yield banner
        yield Button("Stop", id="stop")
        yield Button("Start", id="start")

    @on(Button.Pressed)
    def record(self, event: Button.Pressed) -> None:
        self.pressed.append(event.button.id or "")


async def test_click_follows_a_reflow_that_was_still_pending() -> None:
    app = _ShiftingApp()
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        start = app.query_one("#start", Button)
        before = start.region
        # The button still reports the old region; where it stood is about to show another widget.
        app.query_one("#banner").display = True
        await click_when_settled(pilot, start)
        assert start.region != before
        await wait_for(lambda: app.pressed == ["start"], pilot=pilot, description="the press reached the App")


async def test_click_waits_for_a_modal_that_has_not_been_laid_out() -> None:
    app = _ShiftingApp()
    async with app.run_test(size=(60, 20)) as pilot:
        dialog = _Dialog()
        app.push_screen(dialog)
        await click_when_settled(pilot, "#close")
        await wait_for(lambda: app.pressed == ["close"], pilot=pilot, description="the press reached the App")
        assert app.screen is not dialog
