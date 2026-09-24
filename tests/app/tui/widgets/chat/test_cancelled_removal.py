# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A transcript update that is cancelled while it removes widgets still finishes the removal.

Backend event handlers update the transcript inside the task that publishes the event, and an
interrupt cancels that task. Each test holds a child of a widget the update removes, so the removal
is still waiting for that child when the update is cancelled.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widget import Widget

from chrys.app.tui.widgets.chat.messages import AgentMessage, ErrorMessage, UserMessage
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.welcome import WelcomeWidget
from chrys.foundation.events.types import ProvisionalPresentation
from tests.support.tui_helpers import BusyWidget, assert_app_handles_messages, interrupt_removal


class _PanelApp(App):
    def compose(self) -> ComposeResult:
        yield ChatPanel()


async def _cancel_while_removing(busy: BusyWidget, update: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Start *update*, cancel it while its removal waits for *busy*, and wait for it to end."""
    tasks: list[asyncio.Task[None]] = []

    def start() -> None:
        tasks.append(asyncio.create_task(update()))

    def cancel() -> None:
        tasks[0].cancel()

    await interrupt_removal(busy, start, cancel)
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]


async def _busy_inside(widget: Widget) -> BusyWidget:
    busy = BusyWidget()
    await widget.mount(busy)
    return busy


@pytest.mark.asyncio
async def test_a_cancelled_message_update_still_removes_the_trailing_error() -> None:
    app = _PanelApp()
    async with app.run_test():
        panel = app.query_one(ChatPanel)
        await panel.add_error("boom")
        error = panel.query_one(ErrorMessage)
        busy = await _busy_inside(error)

        await _cancel_while_removing(busy, lambda: panel.add_agent_message("done"))

        assert not error.is_attached and not busy.is_attached
        await assert_app_handles_messages(app)


@pytest.mark.asyncio
async def test_a_cancelled_first_prompt_still_replaces_the_welcome_with_the_spacer() -> None:
    app = _PanelApp()
    async with app.run_test():
        panel = app.query_one(ChatPanel)
        welcome = panel.query_one(WelcomeWidget)
        busy = await _busy_inside(welcome)

        await _cancel_while_removing(busy, lambda: panel.add_user_message("first"))

        assert not welcome.is_attached
        spacer = panel.bottom_spacer()
        assert spacer is not None and spacer.display
        await assert_app_handles_messages(app)


@pytest.mark.asyncio
async def test_a_cancelled_new_prompt_still_removes_the_failed_prompt_and_its_contents_entry() -> None:
    app = _PanelApp()
    async with app.run_test():
        panel = app.query_one(ChatPanel)
        await panel.add_user_message("first")
        prompt = panel.query_one(UserMessage)
        await panel.add_error("boom")
        busy = await _busy_inside(prompt)

        await _cancel_while_removing(busy, lambda: panel.add_user_message("second"))

        assert not prompt.is_attached and not panel.query(ErrorMessage)
        assert [item.summary for item in panel.toc_items] == []
        await assert_app_handles_messages(app)


@pytest.mark.asyncio
async def test_a_cancelled_attempt_acceptance_still_removes_every_rejected_message() -> None:
    app = _PanelApp()
    async with app.run_test():
        panel = app.query_one(ChatPanel)
        for segment in ("first", "second", "kept"):
            await panel.add_agent_message(
                f"{segment} text", is_intermediate=True, presentation=ProvisionalPresentation("attempt", segment)
            )
        first, second, kept = panel.query(AgentMessage)
        busy = await _busy_inside(first)

        await _cancel_while_removing(busy, lambda: panel.accept_presentation_attempt("attempt", ("kept",)))

        assert not first.is_attached and not second.is_attached
        assert kept.is_attached
        await assert_app_handles_messages(app)


@pytest.mark.asyncio
async def test_a_cancelled_clear_still_rebuilds_the_panel() -> None:
    app = _PanelApp()
    async with app.run_test():
        panel = app.query_one(ChatPanel)
        welcome = panel.query_one(WelcomeWidget)
        await panel.add_error("boom")
        error = panel.query_one(ErrorMessage)
        busy = await _busy_inside(error)

        await _cancel_while_removing(busy, panel.clear)

        assert not error.is_attached and not welcome.is_attached
        assert panel.query_one(WelcomeWidget) is not welcome
        spacer = panel.bottom_spacer()
        assert spacer is not None and spacer.parent is panel
        await assert_app_handles_messages(app)
