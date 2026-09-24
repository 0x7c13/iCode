# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the command man-page dialog."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.content import Content
from textual.widgets import Static

from chrys.app.tui.screens.dialogs.man_page import ManPageDialog
from chrys.app.tui.widgets import StableAutoHeightScroll
from chrys.app.tui.widgets.chrome.commands import ManPageSpec, ManPageVerbatimBlock
from tests.support.waiting import wait_for


class _DialogHost(App):
    def compose(self) -> ComposeResult:
        yield Static("placeholder")


def _page(name: str, content: str) -> ManPageSpec:
    return ManPageSpec(name, (ManPageVerbatimBlock(content, indent=0),))


@pytest.mark.asyncio
async def test_man_page_scrollbar_does_not_change_content_width() -> None:
    dialog = ManPageDialog(
        [_page("demo", ("word " * 240).strip())],
    )
    app = _DialogHost()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        scroll = dialog.query_one("#man-scroll", StableAutoHeightScroll)
        content = dialog.query_one("#man-content", Static)
        assert scroll.styles.padding.left == 1
        assert scroll.styles.padding.right == 0
        assert scroll.styles.scrollbar_gutter == "stable"
        await wait_for(
            lambda: content.size.width > 0 and not scroll.show_vertical_scrollbar,
            pilot=pilot,
            description="man-page body to settle without a scrollbar",
        )
        assert scroll.show_vertical_scrollbar is False
        width_without_scrollbar = content.size.width

        await pilot.resize_terminal(100, 20)
        await wait_for(
            lambda: scroll.show_vertical_scrollbar,
            pilot=pilot,
            description="man-page body to show its vertical scrollbar",
        )

        assert scroll.show_vertical_scrollbar is True
        assert content.size.width == width_without_scrollbar


@pytest.mark.asyncio
async def test_man_page_scrolls_with_keyboard() -> None:
    dialog = ManPageDialog(
        [_page("demo", "\n".join(f"line {index}" for index in range(40)))],
    )
    app = _DialogHost()

    async with app.run_test(size=(100, 20)) as pilot:
        await app.push_screen(dialog)

        scroll = dialog.query_one("#man-scroll", StableAutoHeightScroll)
        await wait_for(
            lambda: scroll.has_focus and scroll.show_vertical_scrollbar,
            pilot=pilot,
            description="man-page scroller to receive focus",
        )
        assert scroll.scroll_y == 0

        await pilot.press("pagedown")
        await wait_for(
            lambda: scroll.scroll_y > 0,
            pilot=pilot,
            description="Page Down to scroll the man page",
        )

        await pilot.press("home")
        await wait_for(
            lambda: scroll.scroll_y == 0,
            pilot=pilot,
            description="Home to return the man page to the top",
        )

        await pilot.press("end")
        await wait_for(
            lambda: scroll.scroll_y > 0,
            pilot=pilot,
            description="End to scroll the man page to the bottom",
        )
        bottom_scroll_y = scroll.scroll_y

        await pilot.press("pageup")
        await wait_for(
            lambda: scroll.scroll_y < bottom_scroll_y,
            pilot=pilot,
            description="Page Up to scroll the man page toward the top",
        )


@pytest.mark.asyncio
async def test_man_page_title_uses_current_slash_command_keycap() -> None:
    dialog = ManPageDialog(
        [
            _page("new", "NAME\n    /new - Start a new session"),
            _page("exit", "NAME\n    /exit - Exit the app"),
        ],
    )
    app = _DialogHost()

    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        title = dialog.query_one("#man-title", Static)
        rendered = title.render()
        assert isinstance(rendered, Content)
        assert str(rendered) == " /new "
        assert any(span.style.reverse for span in rendered.spans)

        await pilot.press("down")
        await wait_for(
            lambda: str(title.render()) == " /exit ",
            pilot=pilot,
            description="man-page title to follow navigation",
        )

        rendered = title.render()
        assert isinstance(rendered, Content)
        assert any(span.style.reverse for span in rendered.spans)
