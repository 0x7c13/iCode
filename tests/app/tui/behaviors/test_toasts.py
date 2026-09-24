# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Toasts sit under the header at the top right, the newest on top."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual import __version__ as textual_version
from textual.screen import ModalScreen
from textual.widgets import Static
from textual.widgets._toast import Toast, ToastRack

from chrys.app.tui import toasts
from tests.support.tui_app_harness import make_chrys_app
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from pathlib import Path

    from textual.app import ComposeResult
    from textual.screen import Screen


def test_the_toast_rack_fork_matches_the_pinned_textual() -> None:
    """A Textual upgrade must explicitly re-audit the private toast rack fork."""
    assert textual_version == toasts.TEXTUAL_TOAST_RACK_FORK_VERSION


def _racked(screen: Screen[Any]) -> list[str]:
    """The messages on the screen's rack, top to bottom."""
    return [toast._notification.message for toast in screen.query_one(ToastRack).query(Toast)]


class _Dialog(ModalScreen[None]):
    def compose(self) -> ComposeResult:
        yield Static("dialog")


async def test_toasts_stack_downward_from_under_the_header(tmp_path: Path) -> None:
    app = make_chrys_app(tmp_path)
    async with app.run_test(size=(80, 24), notifications=True) as pilot:
        await pilot.pause()
        app.clear_notifications()
        app.notify("first")
        app.notify("second")
        await wait_for(lambda: _racked(app.screen) == ["second", "first"], pilot=pilot, description="two toasts racked")
        rack = app.screen.query_one(ToastRack)
        second, first = rack.query(Toast)
        await wait_for(
            lambda: first.region.area > 0 < second.region.area, pilot=pilot, description="the toasts laid out"
        )

        # The rack is never composited (it is visibility: hidden), so the toasts carry the geometry:
        # the newest sits one blank line under the header, the older one below it.
        header = app.screen.query_one("#app-header")
        assert header.region.y == 0
        assert second.region.y == header.region.bottom + 1
        assert first.region.y == second.region.bottom + 1
        # Right-aligned, short of the rack's always-shown scrollbar gutter.
        assert first.region.right == second.region.right == app.size.width - rack.styles.scrollbar_size_vertical

        # A later toast goes on top of the ones already racked, which move down by one toast.
        app.notify("third")
        await wait_for(
            lambda: _racked(app.screen) == ["third", "second", "first"], pilot=pilot, description="a third toast"
        )
        third = rack.query(Toast).first()
        await wait_for(
            lambda: third.region.y == header.region.bottom + 1 and second.region.y == third.region.bottom + 1,
            pilot=pilot,
            description="the third toast on top and the older ones below it",
        )

        # A screen pushed over the toasts racks the same live toasts, newest on top.
        dialog = _Dialog()
        await app.push_screen(dialog)
        await wait_for(
            lambda: _racked(dialog) == ["third", "second", "first"], pilot=pilot, description="the dialog's rack"
        )


async def test_more_toasts_than_fit_keep_the_newest_under_the_header(tmp_path: Path) -> None:
    """The stack overflows downward, so the oldest toasts are the ones clipped off the screen."""
    app = make_chrys_app(tmp_path)
    async with app.run_test(size=(80, 14), notifications=True) as pilot:
        await pilot.pause()
        app.clear_notifications()
        messages = [f"toast {index}" for index in range(6)]
        for message in messages:
            app.notify(message)
        await wait_for(lambda: _racked(app.screen) == messages[::-1], pilot=pilot, description="six toasts racked")
        rack = app.screen.query_one(ToastRack)
        newest = rack.query(Toast).first()
        oldest = rack.query(Toast).last()
        header = app.screen.query_one("#app-header")

        await wait_for(lambda: newest.region.y == header.region.bottom + 1, pilot=pilot, description="newest on top")
        assert oldest.region.y >= app.size.height
