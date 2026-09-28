# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Removing the inline ask-user prompt from its tool card."""

from __future__ import annotations

import gc
import weakref

import pytest
from textual.app import ComposeResult

from chrys.app.tui.widgets import AskUserPrompt
from chrys.app.tui.widgets.chat.renderers import ask_user as ask_user_renderer_module
from chrys.app.tui.widgets.chat.renderers.ask_user import AskUserToolCall
from chrys.foundation.models.ask_user import AskUserOption, AskUserQuestion
from tests.support.pilot_barrier import screen_is_settled
from tests.support.tui_helpers import LocalizedApp
from tests.support.waiting import wait_for


def _question(option: str) -> tuple[AskUserQuestion, ...]:
    return (AskUserQuestion("Pick?", options=(AskUserOption(option),)),)


class _ToolApp(LocalizedApp):
    def compose(self) -> ComposeResult:
        yield AskUserToolCall("c1", "ask_user", args={"question": "Pick?"})


async def test_a_cleared_inline_prompt_is_freed_although_the_card_looked_it_up() -> None:
    """``clear_inline_prompt`` looks the prompt up to remove it; that lookup must not keep it alive."""
    async with _ToolApp().run_test(size=(100, 30)) as pilot:
        tool = pilot.app.query_one(AskUserToolCall)
        assert tool.show_inline_prompt("req-1", _question("Python"))
        await wait_for(
            lambda: any(prompt.is_mounted for prompt in tool.query("#ask-inline")),
            pilot=pilot,
            description="inline prompt mounted",
        )
        prompt_ref = weakref.ref(tool.query("#ask-inline").first(AskUserPrompt))

        tool.clear_inline_prompt()
        await wait_for(lambda: not tool.query("#ask-inline"), pilot=pilot, description="inline prompt removed")
        await wait_for(lambda: screen_is_settled(pilot.app, pilot.app.screen), pilot=pilot, description="settled")
        gc.collect()

        assert prompt_ref() is None


@pytest.mark.parametrize("removed_by", ["clear", "card_removed"])
async def test_an_inline_prompt_removed_before_it_mounts_leaves_the_app_running(removed_by: str) -> None:
    """A question answered or cancelled at once, or its card cleared, removes the prompt before it mounts."""
    async with _ToolApp().run_test(size=(100, 30)) as pilot:
        tool = pilot.app.query_one(AskUserToolCall)
        assert tool.show_inline_prompt("req-1", _question("Python"))
        prompt = tool.query_one("#ask-inline", AskUserPrompt)
        assert not prompt.is_mounted

        if removed_by == "clear":
            tool.clear_inline_prompt()
        else:
            await tool.remove()
        await wait_for(lambda: not prompt.is_attached, pilot=pilot, description="the prompt is removed")
        await wait_for(lambda: screen_is_settled(pilot.app, pilot.app.screen), pilot=pilot, description="settled")

        assert pilot.app.is_running


async def test_a_focus_callback_that_lands_mid_removal_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prompt focuses its controls after a refresh; by then its removal may have pruned them."""
    landed_while_pruning: list[bool] = []

    class _LateFocusPrompt(AskUserPrompt):
        def on_mount(self) -> None:
            # Runs before the base handler, in the state a late refresh callback finds mid-removal.
            landed_while_pruning.append(self._pruning and self.is_attached and not self.children)
            self._focus_active()

    monkeypatch.setattr(ask_user_renderer_module, "AskUserPrompt", _LateFocusPrompt)
    async with _ToolApp().run_test(size=(100, 30)) as pilot:
        tool = pilot.app.query_one(AskUserToolCall)
        assert tool.show_inline_prompt("req-1", _question("Python"))
        prompt = tool.query_one("#ask-inline", AskUserPrompt)

        tool.clear_inline_prompt()
        await wait_for(lambda: not prompt.is_attached, pilot=pilot, description="the prompt is removed")
        await wait_for(lambda: screen_is_settled(pilot.app, pilot.app.screen), pilot=pilot, description="settled")

        assert landed_while_pruning == [True]
        assert pilot.app.is_running


def _inline_prompts(tool: AskUserToolCall) -> list[AskUserPrompt]:
    return list(tool.query("#ask-inline").results(AskUserPrompt))


@pytest.mark.parametrize("previous", ["mounting", "mounted", "being_removed"])
async def test_a_prompt_shown_while_the_previous_one_is_still_there_replaces_it(previous: str) -> None:
    """The next question can arrive before the previous prompt mounts, or while its removal still runs."""
    async with _ToolApp().run_test(size=(100, 30)) as pilot:
        tool = pilot.app.query_one(AskUserToolCall)
        assert tool.show_inline_prompt("req-1", _question("Python"))
        first = tool.query_one("#ask-inline", AskUserPrompt)
        if previous != "mounting":
            await wait_for(lambda: first.is_mounted, pilot=pilot, description="the first prompt mounted")
        if previous == "being_removed":
            tool.clear_inline_prompt()
            assert first.is_attached

        two_questions = (*_question("Rust"), AskUserQuestion("Why?", options=(AskUserOption("Speed"),)))
        assert tool.show_inline_prompt("req-2", two_questions)

        await wait_for(
            lambda: [(prompt.request_id, prompt.is_mounted) for prompt in _inline_prompts(tool)] == [("req-2", True)],
            pilot=pilot,
            description="only the second prompt is left, mounted",
        )
        await wait_for(lambda: screen_is_settled(pilot.app, pilot.app.screen), pilot=pilot, description="settled")

        (prompt,) = _inline_prompts(tool)
        assert prompt.request_id == "req-2"
        assert not first.is_attached
        assert tool.has_class("-inline")
        # The panel titles the new prompt's first question of two, not the removed prompt's single one.
        assert str(tool.query_one("#ask-panel").border_title) == str(prompt.active_title())
        assert "2" in str(prompt.active_title())
        assert pilot.app.is_running


@pytest.mark.parametrize("then", ["cleared", "shown_again"])
async def test_a_prompt_waiting_for_its_predecessor_yields_to_a_later_clear_or_show(then: str) -> None:
    """Only the newest show mounts once the previous prompt is gone; a clear meanwhile mounts none."""
    async with _ToolApp().run_test(size=(100, 30)) as pilot:
        tool = pilot.app.query_one(AskUserToolCall)
        assert tool.show_inline_prompt("req-1", _question("Python"))
        first = tool.query_one("#ask-inline", AskUserPrompt)
        await wait_for(lambda: first.is_mounted, pilot=pilot, description="the first prompt mounted")

        assert tool.show_inline_prompt("req-2", _question("Rust"))
        assert first.is_attached
        if then == "cleared":
            tool.clear_inline_prompt()
        else:
            assert tool.show_inline_prompt("req-3", _question("Go"))

        await wait_for(lambda: not first.is_attached, pilot=pilot, description="the first prompt is removed")
        await wait_for(lambda: screen_is_settled(pilot.app, pilot.app.screen), pilot=pilot, description="settled")

        expected = [] if then == "cleared" else [("req-3", True)]
        assert [(prompt.request_id, prompt.is_mounted) for prompt in _inline_prompts(tool)] == expected
        assert tool.has_class("-inline") is (then == "shown_again")
        assert pilot.app.is_running
