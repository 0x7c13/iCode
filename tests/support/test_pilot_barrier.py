# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A Pilot wait drains the messages that the messages it waited for posted, and names what held it."""

from __future__ import annotations

import asyncio

import pytest
import textual.pilot as textual_pilot
from textual import on
from textual.app import App, ComposeResult
from textual.message import Message
from textual.message_pump import MessagePump
from textual.pilot import Pilot, WaitForScreenTimeout
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from tests.support import pilot_barrier
from tests.support.waiting import wait_for

_APP_TURNS = 50
"""Loop turns the App spends in a handler: far more than the few a finished wait needs to resume its caller."""


class Relay(Message, bubble=False):
    """Handled by a widget, which posts the App the messages it carries, in order.

    It does not bubble: bubbling to the App would queue it there behind those messages, and
    a wait that only re-queues behind queued messages would then cover their handlers by luck.
    """

    def __init__(self, *messages: Message) -> None:
        super().__init__()
        self.messages = messages


class Slow(Message):
    """The App records it only after spending turns in its handler."""


class Record(Message):
    pass


class _Relaying(Static):
    def on_relay(self, event: Relay) -> None:
        for message in event.messages:
            self.app.post_message(message)


class _RelayApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.recorded: list[str] = []

    def compose(self) -> ComposeResult:
        yield _Relaying("relay", id="relay")

    async def on_slow(self) -> None:
        for _ in range(_APP_TURNS):
            await asyncio.sleep(0)
        self.recorded.append("slow")

    def on_record(self) -> None:
        self.recorded.append("record")


def test_the_settled_wait_is_installed_for_every_test() -> None:
    assert Pilot._wait_for_screen is pilot_barrier.settled_wait_for_screen


async def test_a_wait_drains_the_messages_a_handler_posted_while_it_waited(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _RelayApp()
    rounds: list[list[str]] = []
    barrier = pilot_barrier._barrier

    async def counted(waiting_app: App[object], pumps: list[MessagePump], deadline: float) -> bool | None:
        processed = await barrier(waiting_app, pumps, deadline)
        rounds.append(list(app.recorded))
        return processed

    monkeypatch.setattr(pilot_barrier, "_barrier", counted)
    async with app.run_test() as pilot:
        await pilot._wait_for_screen()
        rounds.clear()
        app.query_one(_Relaying).post_message(Relay(Slow(), Record()))

        assert await pilot._wait_for_screen()

        # The first round is Textual's own wait: every callback it queued has run, and the App
        # has not reached the message the relay posted it. The rounds after it drain that too.
        assert rounds[0] == []
        assert app.recorded == ["slow", "record"]
        assert len(rounds) > 1


async def test_a_wait_covers_the_handler_the_app_is_still_inside() -> None:
    """A pump takes a message off its queue before it runs the handler, so while the App awaits
    inside ``on_slow`` its queue reads empty; a round that only re-queued behind queued messages
    would end here with the App mid-handler."""
    app = _RelayApp()
    async with app.run_test() as pilot:
        await pilot._wait_for_screen()
        app.query_one(_Relaying).post_message(Relay(Slow()))

        assert await pilot._wait_for_screen()

        assert app.recorded == ["slow"]


class _Dialog(ModalScreen[None]):
    def compose(self) -> ComposeResult:
        yield Button("Close", id="close")

    @on(Button.Pressed, "#close")
    def close(self) -> None:
        self.dismiss()


class _DialogApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.pressed: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("underlay")

    @on(Button.Pressed)
    def record(self, event: Button.Pressed) -> None:
        self.pressed.append(event.button.id or "")


async def test_a_click_returns_after_the_press_it_set_off(monkeypatch: pytest.MonkeyPatch) -> None:
    async def starved_idle_probe(min_sleep: float = 0.02, max_sleep: float = 1) -> None:
        # A descheduled runner burns no CPU time in the probe's sleep, so the probe calls the
        # process idle after a single turn of the loop.
        await asyncio.sleep(0)

    monkeypatch.setattr(textual_pilot, "wait_for_idle", starved_idle_probe)
    app = _DialogApp()
    async with app.run_test(size=(60, 20)) as pilot:
        dialog = _Dialog()
        await app.push_screen(dialog)
        await wait_for(lambda: dialog.query_one("#close").region.area > 0, pilot=pilot)

        assert await pilot.click("#close")

        # The Pressed message bubbles to the App behind the dismissal it caused.
        assert app.pressed == ["close"]
        assert app.screen is not dialog


class BlockPump(Message):
    pass


class _Stuck(Static):
    def __init__(self) -> None:
        super().__init__("stuck", id="stuck")
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()

    async def on_block_pump(self) -> None:
        self.blocked.set()
        await self.release.wait()


class _StuckApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("free", id="free")
        yield _Stuck()


async def test_a_pilot_timeout_names_the_widget_whose_pump_is_blocked() -> None:
    app = _StuckApp()
    async with app.run_test() as pilot:
        assert await pilot._wait_for_screen(timeout=5.0)
        stuck = app.query_one(_Stuck)
        stuck.post_message(BlockPump())
        await wait_for(stuck.blocked.is_set, description="the widget's pump is held in its handler")

        try:
            with pytest.raises(WaitForScreenTimeout) as caught:
                await pilot._wait_for_screen(timeout=0.2)

            # The held widget is named with the handler its pump task is suspended in.
            message = str(caught.value)
            assert "_Stuck(id='stuck')" in message
            assert "on_block_pump" in message
        finally:
            stuck.release.set()  # A held pump cannot close, so a failed assertion would hang the App's exit.
        assert await pilot._wait_for_screen(timeout=5.0)


async def test_a_pump_inside_a_handler_holds_pending_work() -> None:
    app = _StuckApp()
    async with app.run_test() as pilot:
        assert await pilot._wait_for_screen(timeout=5.0)
        stuck = app.query_one(_Stuck)
        assert not pilot_barrier.holds_pending_work(stuck)
        # The Screen lays out from a timer, not a message, so the wait leaves that request behind.
        await wait_for(lambda: pilot_barrier.screen_is_settled(app, app.screen), description="the screen is settled")

        stuck.post_message(BlockPump())
        await wait_for(stuck.blocked.is_set, description="the widget's pump is held in its handler")

        try:
            # The message is in its handler: nothing is queued, kept aside, or waiting to run next.
            assert stuck._message_queue.empty()
            assert stuck._pending_message is None
            assert not stuck._next_callbacks
            assert pilot_barrier.holds_pending_work(stuck)
            assert not pilot_barrier.screen_is_settled(app, app.screen)
        finally:
            stuck.release.set()  # A held pump cannot close, so a failed assertion would hang the App's exit.
        assert await pilot._wait_for_screen(timeout=5.0)
        assert not pilot_barrier.holds_pending_work(stuck)
        await wait_for(lambda: pilot_barrier.screen_is_settled(app, app.screen), description="the screen is settled")
