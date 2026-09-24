# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the patch that hands a removed screen's callbacks to the screen that replaced it."""

from __future__ import annotations

import asyncio
import inspect
from functools import partial
from typing import TYPE_CHECKING

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.message import Message
from textual.messages import Prune
from textual.screen import ModalScreen
from textual.widgets import Static

from chrys.foundation.patches.textual_removed_screen_callbacks import apply_runtime_patch
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from textual.pilot import Pilot


class Hold(Message):
    """Keeps the modal busy until released, so the removal queues up behind it."""


class Queued(Message):
    """Waits behind ``Prune`` so the modal does not idle between ``ScreenSuspend`` and ``Prune``."""


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("base", id="base")


class _Modal(ModalScreen[None]):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.holding = False
        self.posted: list[type[Message]] = []

    def compose(self) -> ComposeResult:
        yield Static("modal", id="modal")

    def post_message(self, message: Message) -> bool:
        self.posted.append(type(message))
        return super().post_message(message)

    async def on_hold(self, _message: Hold) -> None:
        self.holding = True
        await self.release.wait()


async def _pop_holding_callbacks(app: _Host, pilot: Pilot, ran: list[str]) -> _Modal:
    """Pop a busy modal that still holds a callback from the screen below and one from itself."""
    base = app.screen.query_one("#base", Static)
    modal = _Modal()
    await app.push_screen(modal)
    await pilot.pause()
    modal.post_message(Hold())
    try:
        await wait_for(lambda: modal.holding, description="the modal is busy")
        # What call_after_refresh leaves on the screen that was on top, before that screen idles.
        modal._callbacks.append((partial(ran.append, "base"), base))
        modal._callbacks.append((partial(ran.append, "modal"), modal.query_one("#modal", Static)))
        popped = app.pop_screen()
        await wait_for(
            lambda: {events.ScreenSuspend, Prune} <= set(modal.posted),
            description="the removal reaches the modal's queue",
        )
        modal.post_message(Queued())
    finally:
        modal.release.set()
    await popped
    await pilot.pause()
    return modal


async def test_upstream_drops_callbacks_a_removed_screen_still_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins that the patch is still needed; drop the patch once a Textual upgrade makes this fail."""
    apply_runtime_patch()
    monkeypatch.setattr(App, "_replace_screen", inspect.unwrap(App._replace_screen))
    ran: list[str] = []
    app = _Host()
    async with app.run_test() as pilot:
        modal = await _pop_holding_callbacks(app, pilot, ran)
        assert not modal.is_attached
        assert len(modal._callbacks) == 2
        assert ran == []


async def test_the_screen_below_runs_callbacks_its_widgets_left_on_a_removed_screen() -> None:
    apply_runtime_patch()
    ran: list[str] = []
    app = _Host()
    async with app.run_test() as pilot:
        modal = await _pop_holding_callbacks(app, pilot, ran)
        await wait_for(lambda: ran, pilot=pilot, description="the forwarded callback runs")
        assert ran == ["base"]
        assert not modal._callbacks


def test_runtime_patch_is_idempotent() -> None:
    apply_runtime_patch()
    patched = App._replace_screen

    apply_runtime_patch()

    assert App._replace_screen is patched
    assert inspect.unwrap(App._replace_screen) is not patched
