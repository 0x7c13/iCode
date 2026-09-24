# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Messages disabled for given widgets stay local to them, including while the block awaits."""

from __future__ import annotations

import asyncio

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Tab, Tabs

from chrys.app.tui.util.message_gate import messages_disabled
from tests.support.waiting import wait_for


class _TabsApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.activations: list[Tabs.TabActivated] = []

    def compose(self) -> ComposeResult:
        yield Tabs(id="first")
        yield Tabs(id="second")

    @on(Tabs.TabActivated)
    def record(self, event: Tabs.TabActivated) -> None:
        self.activations.append(event)


async def _activations_of(app: _TabsApp, tabs: Tabs) -> list[Tabs.TabActivated]:
    """Post a marker activation to *tabs* and return what reached the App from it up to the marker.

    Each hop handles messages in order, so every activation *tabs* posted before the marker arrives first.
    """
    marker = Tabs.TabActivated(tabs, tabs.query_one(Tab))
    assert tabs.post_message(marker)
    await wait_for(lambda: marker in app.activations, description="the marker activation reaches the App")
    return [event for event in app.activations if event.tabs is tabs and event is not marker]


@pytest.mark.asyncio
async def test_only_the_given_widgets_drop_the_messages_while_the_block_awaits() -> None:
    app = _TabsApp()
    async with app.run_test():
        first, second = app.query_one("#first", Tabs), app.query_one("#second", Tabs)
        with messages_disabled(Tabs.TabActivated, first):
            # A bar's first tab activates once it is mounted, before add_tab() returns.
            await first.add_tab(Tab("a", id="a"))
            await second.add_tab(Tab("b", id="b"))
            await wait_for(
                lambda: any(event.tabs is second for event in app.activations),
                description="the other bar's activation arrives while the block is open",
            )

        assert await _activations_of(app, first) == []


@pytest.mark.asyncio
async def test_a_cancelled_block_enables_the_messages_again() -> None:
    app = _TabsApp()
    async with app.run_test():
        first = app.query_one("#first", Tabs)
        await first.add_tab(Tab("a", id="a"))
        # Adding the tab activates it; drain those activations first.
        await _activations_of(app, first)
        app.activations.clear()
        entered, never = asyncio.Event(), asyncio.Event()

        async def update() -> None:
            with messages_disabled(Tabs.TabActivated, first):
                entered.set()
                await never.wait()

        task = asyncio.create_task(update())
        await wait_for(entered.is_set, description="the block is open")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        first.active = ""
        first.active = "a"
        activations = await _activations_of(app, first)
        assert [event.tab.id for event in activations] == ["a"]
