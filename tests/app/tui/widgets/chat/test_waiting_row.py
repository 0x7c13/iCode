# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A running turn shows a breathing dot under the transcript while it waits with nothing on screen moving."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from textual.screen import Screen

from chrys.app.tui.widgets.chat.chrome import ChatBottomGap
from chrys.app.tui.widgets.chat.compaction_card import CompactionCard
from chrys.app.tui.widgets.chat.context_fold import ContextFoldWidget
from chrys.app.tui.widgets.chat.messages import (
    AgentMessage,
    ErrorMessage,
    SystemMessage,
    ThinkingMessage,
    UserMessage,
)
from chrys.app.tui.widgets.chat.panel import _WAITING_ROW_DELAY_SECONDS, ChatPanel
from chrys.app.tui.widgets.chat.tool_call import ToolGroup
from chrys.app.tui.widgets.chat.waiting_row import WaitingRow, waits_after
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from chrys.foundation.tool_kinds import KIND_FILESYSTEM_READ
from tests.support.pilot_barrier import screen_is_settled
from tests.support.tui_helpers import ChatPanelApp
from tests.support.waiting import wait_for, wait_until

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.widget import Widget


def _compaction(status: str) -> CompactionCard:
    card = CompactionCard("c-1")
    card.status = status
    return card


@pytest.mark.parametrize(
    ("make", "waits"),
    [
        pytest.param(lambda: None, False, id="nothing"),
        pytest.param(lambda: UserMessage("prompt"), True, id="prompt"),
        pytest.param(lambda: SystemMessage("notice"), True, id="notice"),
        pytest.param(lambda: ContextFoldWidget("ctx-1", "Summary"), True, id="context-fold"),
        pytest.param(lambda: ThinkingMessage("Look.", is_final=False), False, id="thinking-streams"),
        pytest.param(lambda: ThinkingMessage("Look."), True, id="thinking-complete"),
        pytest.param(lambda: AgentMessage("Done.", profile_name="Agent"), False, id="answer"),
        pytest.param(lambda: AgentMessage("Next.", is_intermediate=True), True, id="text-between-tool-calls"),
        pytest.param(ToolGroup, False, id="tool-group-running"),
        pytest.param(lambda: ToolGroup(live=False), True, id="tool-group-done"),
        pytest.param(lambda: _compaction("running"), False, id="compaction-running"),
        pytest.param(lambda: _compaction("ok"), True, id="compaction-done"),
        pytest.param(lambda: ErrorMessage("failed"), False, id="error"),
    ],
)
def test_a_turn_waits_after_what_says_it_goes_on_and_not_after_what_still_moves_or_says_why(
    make: Callable[[], Widget | None], waits: bool
) -> None:
    assert waits_after(make()) is waits


def _row(panel: ChatPanel) -> WaitingRow:
    return panel.query_one(WaitingRow)


def _hidden_with_none_due(panel: ChatPanel) -> bool:
    return not _row(panel).display and panel._waiting_row_timer is None


async def test_the_row_shows_below_the_transcript_a_moment_after_the_turn_last_moved_and_hides_as_it_moves() -> None:
    async with ChatPanelApp(thinking_expanded=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        panel.agent_running = True
        await panel.add_user_message("prompt")
        # The moment keeps it from flashing up between one row ending and the next starting.
        assert not _row(panel).display and panel._waiting_row_timer is not None
        await wait_for(lambda: _row(panel).display, pilot=pilot, description="the waiting row shows")
        prompt = panel.query_one(UserMessage)
        children = list(panel.children)
        assert children.index(_row(panel)) == children.index(prompt) + 1
        assert isinstance(children[children.index(_row(panel)) + 1], ChatBottomGap)

        # New output takes its place at once.
        await panel.add_answer_delta("a1", "Let me look.")
        assert _hidden_with_none_due(panel)
        # An answer may end the turn: no row comes after it.
        panel.finish_answer_segment("a1", completed=True)
        assert _hidden_with_none_due(panel)

        await panel.add_tool_start("c1", "read_file", KIND_FILESYSTEM_READ, args={})
        await panel.add_tool_result("c1", "read_file", "ok")
        # The group's title still runs through its tool phase.
        assert _hidden_with_none_due(panel)

        await panel.add_thinking_delta("r1", "reasoning", "The file says yes.")
        assert _hidden_with_none_due(panel)
        panel.finish_thinking_segment("r1")
        await wait_for(lambda: _row(panel).display, pilot=pilot, description="the row after complete thinking")
        children = list(panel.children)
        assert children.index(_row(panel)) == children.index(panel.query(ThinkingMessage).last()) + 1

        panel.agent_running = False
        assert _hidden_with_none_due(panel)


async def test_a_turn_that_ends_before_the_moment_passes_never_shows_the_row() -> None:
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        panel.agent_running = True
        await panel.add_user_message("prompt")
        assert panel._waiting_row_timer is not None
        panel.agent_running = False
        assert _hidden_with_none_due(panel)
        # Past the moment it would have shown at.
        assert not await wait_until(lambda: _row(panel).display, timeout=_WAITING_ROW_DELAY_SECONDS * 2, pilot=pilot)


async def test_the_row_showing_lays_the_transcript_out_once() -> None:
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        for index in range(20):
            await panel.add_user_message(f"prompt {index}")
        panel.agent_running = True
        await panel.add_user_message("prompt")
        timer = panel._waiting_row_timer
        assert timer is not None
        timer.pause()
        screen = pilot.app.screen
        prompts = list(panel.query(UserMessage))
        await wait_for(
            lambda: (
                all(prompt.query_one(VirtualizedMarkdown).shown_source for prompt in prompts)
                and screen_is_settled(pilot.app, screen)
            ),
            pilot=pilot,
            description="the prompts are laid out",
        )
        # Taller than the view: the transcript's own rows, not the spacer under a short one, give the row its place.
        assert panel.max_scroll_y > 0
        with patch.object(screen, "_refresh_layout", autospec=True, side_effect=screen._refresh_layout) as layout:
            timer.resume()
            await wait_for(
                lambda: _row(panel).display and screen_is_settled(pilot.app, screen),
                pilot=pilot,
                description="the row shows",
            )
        # The layout that shows it gives it its rows: Textual's own size update would ask for another.
        assert layout.call_count == 1


async def test_a_turn_that_ends_as_the_row_shows_moves_the_view_in_the_frame_that_hides_the_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held a frame as the transcript shrinks, its rows would stay blank, and the view move down only after them."""
    async with ChatPanelApp(thinking_expanded=lambda: False).run_test(size=(80, 24)) as pilot:
        app = pilot.app
        panel = app.query_one(ChatPanel)
        for index in range(4):
            await panel.add_user_message(f"message {index}")
            await panel.add_agent_message("\n".join(f"line {line}" for line in range(8)))
        await panel.add_user_message("prompt")
        panel.agent_running = True
        await panel.add_thinking_delta("r1", "reasoning", "Think one.\n\nThink two.")
        panel.finish_thinking_segment("r1")
        thinking = panel.query_one(ThinkingMessage)
        row = _row(panel)
        await wait_for(
            lambda: row.display and panel.scroll_y == panel.max_scroll_y and screen_is_settled(app, panel.screen),
            pilot=pilot,
            description="the row shows at the bottom of a filled view",
        )
        painted: list[tuple[int | None, bool]] = []
        paint = Screen._compositor_refresh

        def record(screen: Screen) -> None:
            visible = screen._compositor.visible_widgets
            painted.append((visible[thinking][0].y if thinking in visible else None, row in visible))
            paint(screen)

        monkeypatch.setattr(Screen, "_compositor_refresh", record)
        panel.agent_running = False
        await wait_for(
            lambda: not row.display and screen_is_settled(app, panel.screen),
            pilot=pilot,
            description="the turn ended",
        )
        assert panel.scroll_y == panel.max_scroll_y
        assert set(painted) == {(thinking.region.y, False)}, painted


async def test_the_row_showing_under_a_tall_prompt_keeps_the_view_on_its_top() -> None:
    """The row says the turn goes on; it is no reply, which alone takes the view down from where the prompt begins."""
    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        app = pilot.app
        panel = app.query_one(ChatPanel)
        for index in range(8):
            await panel.add_user_message(f"message {index}")
        panel.agent_running = True
        await panel.add_user_message("\n".join(f"line {index}" for index in range(30)))
        prompt = panel.query(UserMessage).last()
        row = _row(panel)
        await wait_for(
            lambda: (
                prompt.outer_size
                and panel.scroll_y == prompt.virtual_region_with_margin.y
                and screen_is_settled(app, panel.screen)
            ),
            pilot=pilot,
            description="the view shows the prompt's top",
        )
        assert panel._anchor_released is True and not row.display

        await wait_for(
            lambda: row.display and screen_is_settled(app, panel.screen), pilot=pilot, description="the row shows"
        )
        assert panel.scroll_y == prompt.virtual_region_with_margin.y < panel.max_scroll_y
        assert panel._anchor_released is True

        await panel.add_answer_delta("a1", "\n\n".join(f"answer {index}" for index in range(10)))
        await wait_for(
            lambda: (
                panel.scroll_y == panel.max_scroll_y
                and panel._anchor_released is False
                and screen_is_settled(app, panel.screen)
            ),
            pilot=pilot,
            description="the reply takes the view down",
        )


async def test_thinking_in_the_rows_place_under_a_tall_prompt_takes_the_view_down() -> None:
    """Folded thinking takes the row's place at its height: the size stays, but the reply came, and the view follows."""
    async with ChatPanelApp(thinking_expanded=lambda: False).run_test(size=(80, 20)) as pilot:
        app = pilot.app
        panel = app.query_one(ChatPanel)
        for index in range(8):
            await panel.add_user_message(f"message {index}")
        panel.agent_running = True
        await panel.add_user_message("\n".join(f"line {index}" for index in range(30)))
        prompt = panel.query(UserMessage).last()
        row = _row(panel)
        await wait_for(
            lambda: (
                row.display
                and prompt.outer_size
                and panel.scroll_y == prompt.virtual_region_with_margin.y
                and screen_is_settled(app, panel.screen)
            ),
            pilot=pilot,
            description="the row shows under the prompt's top",
        )
        height = panel.virtual_size.height

        await panel.add_thinking_delta("r1", "reasoning", "Looking at the file.")
        thinking = panel.query_one(ThinkingMessage)
        await wait_for(
            lambda: thinking.outer_size and not row.display and screen_is_settled(app, panel.screen),
            pilot=pilot,
            description="thinking shows in the row's place",
        )
        assert panel.virtual_size.height == height
        assert panel._anchor_released is False
        assert panel.scroll_y == panel.max_scroll_y > prompt.virtual_region_with_margin.y
