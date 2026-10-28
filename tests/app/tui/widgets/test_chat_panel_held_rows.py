# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for a ChatPanel view that follows the bottom while its entries give rows up: a streamed answer's closing
markup, the folds its turn or the user makes, and an answer's final text."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest
from textual._arrange import DockArrangeResult
from textual.geometry import Region, Size
from textual.screen import Screen
from textual.widget import Widget

from chrys.app.tui.widgets.chat.compaction_card import CompactionCard, CompactionCardTitle
from chrys.app.tui.widgets.chat.fold_gap import FOLD_GAP_ROWS
from chrys.app.tui.widgets.chat.messages import (
    AgentMessage,
    MessageHeaderRow,
    StreamedMarkdownMessage,
    ThinkingMessage,
    _FoldRowHeader,
)
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chat.tool_call import ToolGroup, ToolGroupTitle
from chrys.app.tui.widgets.chat.waiting_row import WaitingRow
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.app.tui.widgets.markdown.blocks import MarkdownBlock
from chrys.foundation.tool_kinds import KIND_FILESYSTEM_READ
from tests.app.tui.widgets._chat_scroll import record_layouts, scrolled_to_bottom
from tests.support.pilot_barrier import screen_is_settled
from tests.support.tui_helpers import ChatPanelApp, click_when_settled, resize_when_settled
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for


def _shows(body: VirtualizedMarkdown, text: str) -> bool:
    """The markdown body was handed *text* and no parse is left to install it: its rows are the text's."""
    return body.source == text and body._pending_parse is None and not body.lock.is_locked


async def test_a_streamed_answer_keeps_the_row_its_closing_markup_gives_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Until its ``**`` closes, a line wraps that then fits again: the row stays blank until the text fills it.

    Given back, it moved a view that follows the bottom down a row and back up as the text grew.
    """
    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        await scrolled_to_bottom(cp, pilot)
        cp.agent_running = True
        await cp.add_answer_delta("segment-1", "start")
        answer = cp.mounted_agent_messages()[-1]
        body = answer.query_one(VirtualizedMarkdown)
        await wait_for(
            lambda: _shows(body, "start") and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the answer started",
        )
        # One word short of the width: the unclosed ``**b`` wraps, the closed bold ``b`` fits.
        text = "x" * (body.content_size.width - 3) + " **b"
        answer.stream_update(text)
        await wait_for(
            lambda: (
                body.virtual_size.height == 2 and cp.scroll_y == cp.max_scroll_y and screen_is_settled(app, cp.screen)
            ),
            pilot=pilot,
            description="the unclosed markup wrapped",
        )
        height, bottom = answer.outer_size.height, cp.virtual_size.height
        layouts = record_layouts(monkeypatch)

        answer.stream_update(text + "**")
        await wait_for(
            lambda: body.virtual_size.height == 1 and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the closed markup fits one row",
        )
        assert (answer.outer_size.height, cp.virtual_size.height) == (height, bottom)

        answer.stream_update(text + "** and more")
        await wait_for(
            lambda: body.virtual_size.height == 2 and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the text filled the row again",
        )
        assert layouts
        assert all(layout == (bottom, cp.max_scroll_y) for layout in layouts), layouts

        # Once the answer ends, it is as tall as its text.
        answer.stream_update(text + "**", is_final=True)
        await wait_for(
            lambda: answer.outer_size.height == height - 1 and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the finished answer gave its blank row back",
        )
        assert cp.scroll_y == cp.max_scroll_y


async def test_a_streamed_answer_gives_back_at_once_more_rows_than_a_third_of_the_view() -> None:
    """A closing fence can turn many rows of source into a short diagram: held blank, they would leave the text above the view."""
    async with ChatPanelApp().run_test(size=(80, 24)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        await scrolled_to_bottom(cp, pilot)
        cp.agent_running = True
        await cp.add_answer_delta("segment-1", "start")
        answer = cp.mounted_agent_messages()[-1]
        body = answer.query_one(VirtualizedMarkdown)
        tall = "\n\n".join(f"line {index}" for index in range(12))
        answer.stream_update(tall)
        await wait_for(
            lambda: _shows(body, tall) and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the answer streamed tall",
        )
        rows = answer.content_size.height

        answer.stream_update("short")
        await wait_for(
            lambda: _shows(body, "short") and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the answer's text took fewer rows",
        )
        assert rows - _text_rows(answer) > cp.size.height // 3
        assert answer.content_size.height == _text_rows(answer)
        assert cp.scroll_y == cp.max_scroll_y


def _text_rows(answer: AgentMessage) -> int:
    """The rows of the answer's header and text, and the blank row below text between tool calls."""
    body = answer.query_one(VirtualizedMarkdown)
    return answer.query_one(MessageHeaderRow).outer_size.height + body.outer_size.height + body.styles.margin.bottom


async def _following_a_tool_group(cp: ChatPanel, pilot: object, tools: int) -> ToolGroup:
    """Settle a view that follows the bottom below a running group of *tools* finished tool calls."""
    app = cp.app
    await scrolled_to_bottom(cp, pilot)
    cp.agent_running = True
    for index in range(tools):
        await cp.add_tool_start(
            f"c-{index}", "read_file", KIND_FILESYSTEM_READ, f"f{index}.py", {"path": f"f{index}.py"}
        )
        await cp.add_tool_result(f"c-{index}", "read_file", "line\n" * 3, 10)
    await wait_for(
        lambda: cp.scroll_y == cp.max_scroll_y and screen_is_settled(app, cp.screen),
        pilot=pilot,
        description="the tool calls laid out at the bottom",
    )
    return cp.query(ToolGroup).last()


def _padded(cp: ChatPanel) -> int:
    """The blank rows below the transcript's entries: those it gave up as its rows folded, while its turn runs."""
    gap = cp.bottom_gap()
    assert gap is not None
    return gap.outer_size.height


def _follows_the_bottom(cp: ChatPanel, layouts: list[tuple[int, float]], bottom: int) -> bool:
    """No layout showed the transcript shorter than *bottom*, or shorter than the one before, and each at its bottom.

    A view that follows the bottom moves the rows on screen down by the rows the transcript gives up.
    """
    heights = [height for height, _scroll_y in layouts]
    return (
        bool(layouts)
        and heights == sorted(heights)
        and heights[0] >= bottom
        and all(scroll_y == height - cp.size.height for height, scroll_y in layouts)
    )


async def test_an_answer_takes_back_the_rows_its_tool_group_gives_up_by_folding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The answer's start folds the tool group above it: the rows stay blank below the transcript until the text
    fills them, and go once the turn ends.

    Given up, they moved a view that follows the bottom down by them, and back up as the answer grew.
    """
    async with ChatPanelApp().run_test(size=(80, 24)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        group = await _following_a_tool_group(cp, pilot, tools=1)
        header = group.query_one("#tg-header")
        header_y, bottom = header.region.y, cp.virtual_size.height
        layouts = record_layouts(monkeypatch)

        await cp.add_answer_delta("segment-1", "The answer starts")
        answer = cp.mounted_agent_messages()[-1]
        body = answer.query_one(VirtualizedMarkdown)
        await wait_for(
            lambda: group.collapsed and _shows(body, "The answer starts") and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the answer started below the folded group",
        )
        padded = _padded(cp)
        assert padded > 0
        gap = cp.bottom_gap()
        assert gap is not None
        # Blank: nothing of the widget's own shows in its rows.
        assert [strip.text for strip in gap.render_lines(Region(0, 0, gap.outer_size.width, padded))] == [
            " " * gap.outer_size.width
        ] * padded
        text = "- The answer starts\n- and goes on"
        answer.stream_update(text)
        await wait_for(
            lambda: _shows(body, text) and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the answer grew",
        )
        assert _padded(cp) < padded
        assert header.region.y == header_y
        assert layouts
        assert all(layout == (bottom, cp.max_scroll_y) for layout in layouts), layouts

        # Once the turn ends, the transcript is as tall as its entries.
        answer.stream_update(text, is_final=True)
        await wait_for(lambda: screen_is_settled(app, cp.screen), pilot=pilot, description="the final text shows")
        paints = _record_paints(monkeypatch, cp, group, answer)
        cp.agent_running = False
        await wait_for(
            lambda: _padded(cp) == 0 and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the finished turn gave its blank rows back",
        )
        assert cp.virtual_size.height < bottom
        assert cp.scroll_y == cp.max_scroll_y
        # Held a frame, the view would show the rows given back below the answer first, and the answer moved up.
        assert {(region, end) for _folded, region, end in paints} == {(answer.region, True)}, paints


def _record_paints(
    monkeypatch: pytest.MonkeyPatch, cp: ChatPanel, folded: Widget, row: Widget
) -> list[tuple[bool, Region | None, bool]]:
    """For each frame the screen paints: whether *folded* was folded when the transcript was laid out as it paints, the
    screen region it paints *row* in, and whether the view shows the end of the transcript.

    A widget's repaint can reach the screen a frame before the layout another widget asked for: that frame paints the
    transcript as it was laid out before, so it is judged by the fold it was laid out with, not the one asked for.
    """
    paints: list[tuple[bool, Region | None, bool]] = []
    laid_out: dict[str, object] = {"arrangement": None, "collapsed": folded.collapsed}
    paint = Screen._compositor_refresh
    arrange = ChatPanel.arrange

    def arranged(panel: ChatPanel, size: Size, optimal: bool = False) -> DockArrangeResult:
        result = arrange(panel, size, optimal=optimal)
        # A cached arrangement comes back as the same result: only a new one reads the fold as it is now.
        if panel is cp and not optimal and result is not laid_out["arrangement"]:
            laid_out.update(arrangement=result, collapsed=folded.collapsed)
        return result

    def record(screen: Screen) -> None:
        painted = screen._compositor.visible_widgets.get(row)
        at_end = cp.scroll_y == cp.max_scroll_y
        paints.append((laid_out["collapsed"] is True, None if painted is None else painted[0], at_end))
        paint(screen)

    monkeypatch.setattr(ChatPanel, "arrange", arranged)
    monkeypatch.setattr(Screen, "_compositor_refresh", record)
    return paints


async def _foldable_entry(
    cp: ChatPanel, pilot: object, entry: str
) -> tuple[ToolGroup | StreamedMarkdownMessage | CompactionCard, Widget]:
    """Settle a view that follows the bottom of a running turn at *entry*; return it and the row a click folds it by."""
    if entry == "tool group":
        group = await _following_a_tool_group(cp, pilot, tools=1)
        return group, group.query_one(ToolGroupTitle)
    await scrolled_to_bottom(cp, pilot)
    cp.agent_running = True
    if entry == "thinking":
        await cp.add_thinking_delta("r1", "reasoning", "Think one.\n\nThink two.\n\nThink three.")
        cp.finish_thinking_segment("r1")
        message: StreamedMarkdownMessage = cp.query_one(ThinkingMessage)
        return message, message.query_one(_FoldRowHeader)
    if entry == "text between tool calls":
        await cp.add_answer_delta("a1", "Let me look.\n\nAt both files.")
        cp.finish_answer_segment("a1", completed=True)
        await cp.add_tool_start("c1", "read_file", KIND_FILESYSTEM_READ, "a.py", {"path": "a.py"})
        message = cp.mounted_agent_messages()[-1]
        return message, message.query_one(_FoldRowHeader)
    await cp.add_compaction_start("k1")
    cp.complete_compaction("k1", outcome="ok", last_words="A note.\n\nIt goes on.")
    card = cp.query_one(CompactionCard)
    return card, card.query_one(CompactionCardTitle)


@pytest.mark.parametrize(
    ("entry", "how"),
    [
        ("tool group", "click"),
        ("tool group", "fold all"),
        ("tool group", "click once the turn ended"),
        ("thinking", "click"),
        ("text between tool calls", "click"),
        ("compaction", "click"),
    ],
)
async def test_an_entry_the_user_folds_gives_its_rows_up_at_the_bottom_in_the_layout_that_folds_it(
    monkeypatch: pytest.MonkeyPatch, entry: str, how: str
) -> None:
    """A fold the user makes is theirs to see: no blank rows below the transcript hold its place, as they do for the
    turn's own folds, and a view that follows the bottom shows its end in every frame painted, the content below the
    fold coming up in none of them."""
    async with ChatPanelApp(thinking_expanded=lambda: True).run_test(size=(80, 24)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        folded, title = await _foldable_entry(cp, pilot, entry)
        if how == "click once the turn ended":
            cp.agent_running = False
        # Thinking and a compaction that ended wait for what comes next: the waiting row shows after a moment.
        waits = cp.agent_running and entry in {"thinking", "compaction"}

        def at_bottom(collapsed: bool) -> bool:
            return (
                folded.collapsed is collapsed
                and cp.query_one(WaitingRow).display is waits
                and all(
                    body._pending_parse is None and not body.lock.is_locked
                    for body in folded.query(VirtualizedMarkdown)
                )
                and cp.scroll_y == cp.max_scroll_y
                and screen_is_settled(app, cp.screen)
            )

        if folded.collapsed:
            # The turn folded it: the user opens it first.
            await click_when_settled(pilot, title)
        await wait_for(lambda: at_bottom(False), pilot=pilot, description="the entry shows open at the bottom")
        bottom = cp.virtual_size.height
        paints = _record_paints(monkeypatch, cp, folded, title)

        if how == "fold all":
            assert cp.toggle_fold_all()
        else:
            await click_when_settled(pilot, title)
        await wait_for(lambda: at_bottom(True), pilot=pilot, description="the user folded the entry")
        assert (_padded(cp), cp.virtual_size.height < bottom) == (0, True)
        folded_paints = [(row, at_end) for collapsed, row, at_end in paints if collapsed]
        assert folded_paints
        assert set(folded_paints) == {(title.region, True)}, paints


async def test_a_tool_group_folding_more_than_a_third_of_the_view_brings_the_content_above_into_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows that many would leave the answer above a blank third of the view: the view moves by the rest, in the
    frame that folds the group."""
    async with ChatPanelApp().run_test(size=(80, 24)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        group = await _following_a_tool_group(cp, pilot, tools=3)
        bottom = cp.virtual_size.height
        assert group.query_one("#tg-content").outer_size.height > cp.size.height // 3
        gap = cp.bottom_gap()
        assert gap is not None
        paints = _record_paints(monkeypatch, cp, group, gap)

        await cp.add_answer_delta("segment-1", "The answer starts")
        answer = cp.mounted_agent_messages()[-1]
        body = answer.query_one(VirtualizedMarkdown)
        await wait_for(
            lambda: group.collapsed and _shows(body, "The answer starts") and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the answer started below the folded group",
        )
        assert _padded(cp) == cp.size.height // 3
        assert cp.virtual_size.height < bottom
        assert cp.scroll_y == cp.max_scroll_y
        # Held a frame, the rows above would move up past the end of the transcript, and down to it in the next.
        assert {(gap and gap.bottom, end) for collapsed, gap, end in paints if collapsed} == {
            (cp.content_region.bottom, True)
        }, paints


async def test_text_and_the_tool_call_after_it_fold_as_what_follows_them_starts_and_the_view_stays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text a tool call follows folds to its header row as the call starts, and the call's group folds as the text
    after it starts: the rows each gives up stay blank below the transcript, so a view that follows the bottom never
    moves up."""
    async with ChatPanelApp().run_test(size=(80, 24)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        group = await _following_a_tool_group(cp, pilot, tools=1)
        await cp.add_answer_delta("segment-1", "A short note")
        note = cp.mounted_agent_messages()[-1]
        await wait_for(
            lambda: (
                group.collapsed
                and _shows(note.query_one(VirtualizedMarkdown), "A short note")
                and screen_is_settled(app, cp.screen)
            ),
            pilot=pilot,
            description="the note started below the folded group",
        )
        cp.finish_answer_segment("segment-1", completed=True)
        bottom = cp.virtual_size.height
        layouts = record_layouts(monkeypatch)

        await cp.add_tool_start("c-next", "read_file", KIND_FILESYSTEM_READ, "next.py", {"path": "next.py"})
        following = cp.query(ToolGroup).last()
        await wait_for(
            lambda: (
                following is not group
                and note.has_class("--intermediate")
                and note.collapsed
                and screen_is_settled(app, cp.screen)
            ),
            pilot=pilot,
            description="the next tool call folded the note",
        )
        # Its header, and the gap that parts it from the folded group above it.
        assert note.gap_above
        assert note.outer_size.height == 1 + FOLD_GAP_ROWS
        # Its running call may add rows of its own.
        assert _follows_the_bottom(cp, layouts, bottom), layouts

        # Its call's result is new content: the view follows it down.
        await cp.add_tool_result("c-next", "read_file", "line\n" * 3, 10)
        await wait_for(
            lambda: cp.scroll_y == cp.max_scroll_y and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the result laid out at the bottom",
        )
        bottom = cp.virtual_size.height
        layouts.clear()

        await cp.add_answer_delta("segment-2", "The answer")
        answer = cp.mounted_agent_messages()[-1]
        await wait_for(
            lambda: (
                following.collapsed
                and _shows(answer.query_one(VirtualizedMarkdown), "The answer")
                and screen_is_settled(app, cp.screen)
            ),
            pilot=pilot,
            description="the answer started below the folded group",
        )
        assert _padded(cp) > 0
        assert layouts
        assert all(layout == (bottom, cp.max_scroll_y) for layout in layouts), layouts


@pytest.mark.parametrize("thinking_expanded", [False, True], ids=["folded-thinking", "open-thinking"])
async def test_a_turn_with_its_tool_groups_folded_keeps_a_view_that_follows_the_bottom_where_it_was(
    monkeypatch: pytest.MonkeyPatch, thinking_expanded: bool
) -> None:
    """Long text, a tool call that folds it, its result, then thinking that ends the call's group: as the display
    settings have it by default, each step lays the transcript out no shorter than it was."""
    text = "\n\n".join(f"Paragraph {index} of the text between calls." for index in range(8))
    async with ChatPanelApp(thinking_expanded=lambda: thinking_expanded, tool_groups_expanded=lambda: False).run_test(
        size=(100, 60)
    ) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        await scrolled_to_bottom(cp, pilot)
        await cp.add_user_message("prompt")
        cp.agent_running = True
        await cp.add_answer_delta("segment-1", text)
        cp.finish_answer_segment("segment-1", completed=True)
        note = cp.mounted_agent_messages()[-1]
        await wait_for(
            lambda: (
                _shows(note.query_one(VirtualizedMarkdown), text)
                and cp.scroll_y == cp.max_scroll_y
                and screen_is_settled(app, cp.screen)
            ),
            pilot=pilot,
            description="the text laid out at the bottom",
        )
        bottom = cp.virtual_size.height
        layouts = record_layouts(monkeypatch)

        await cp.add_tool_start("c1", "read_file", KIND_FILESYSTEM_READ, "a.py", {"path": "a.py"})
        await cp.add_tool_result("c1", "read_file", "ok", 5)
        await cp.add_thinking_delta("r1", "reasoning", "Think one.\n\nThink two.")
        cp.finish_thinking_segment("r1")
        group = cp.query_one(ToolGroup)
        await wait_for(
            lambda: note.collapsed and cp.query(ThinkingMessage) and group.region and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the thinking started below the folded text and tool call",
        )
        assert _padded(cp) > 0
        assert _follows_the_bottom(cp, layouts, bottom), layouts


async def test_a_row_opened_again_takes_back_the_blank_rows_its_fold_left_below_the_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opened, the thinking takes its rows again: the blank rows below go, rather than adding to them."""
    text = "\n\n".join(f"Paragraph {index} of the thinking." for index in range(5))
    async with ChatPanelApp(thinking_expanded=lambda: True).run_test(size=(80, 40)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        await scrolled_to_bottom(cp, pilot)
        cp.agent_running = True
        await cp.add_thinking_delta("r1", "reasoning", text)
        cp.finish_thinking_segment("r1")
        thinking = cp.query_one(ThinkingMessage)
        await wait_for(
            lambda: (
                _shows(thinking.query_one(VirtualizedMarkdown), text)
                and cp.scroll_y == cp.max_scroll_y
                and screen_is_settled(app, cp.screen)
            ),
            pilot=pilot,
            description="the thinking laid out at the bottom",
        )
        await cp.add_answer_delta("segment-1", "The answer")
        answer = cp.mounted_agent_messages()[-1]
        await wait_for(
            lambda: (
                thinking.collapsed
                and _shows(answer.query_one(VirtualizedMarkdown), "The answer")
                and screen_is_settled(app, cp.screen)
            ),
            pilot=pilot,
            description="the answer folded the thinking",
        )
        assert _padded(cp) > 0
        bottom = cp.virtual_size.height
        layouts = record_layouts(monkeypatch)

        thinking.collapsed = False
        # Folded, it let its body go: it builds a new one.
        await wait_for(
            lambda: (
                any(_shows(body, text) for body in thinking.query(VirtualizedMarkdown))
                and screen_is_settled(app, cp.screen)
            ),
            pilot=pilot,
            description="the thinking opened again",
        )
        assert _padded(cp) == 0
        assert _follows_the_bottom(cp, layouts, bottom), layouts


async def test_a_resize_drops_the_rows_held_for_folds_in_the_frame_it_shows(monkeypatch: pytest.MonkeyPatch) -> None:
    """At another size its entries lay out anew: the view moves as it would, without the rows, as a fold moves it."""
    async with ChatPanelApp().run_test(size=(80, 24)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        answer, _body = await _an_answer_below_a_folded_group(cp, pilot)
        paints = _record_paints(monkeypatch, cp, answer, answer)

        await resize_when_settled(pilot, 80, 30)
        await wait_for(
            lambda: cp.size.height > 20 and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the chat laid out at the taller height",
        )
        assert _padded(cp) == 0
        assert {(region, end) for _collapsed, region, end in paints} == {(answer.region, True)}, paints


async def test_text_that_starts_while_the_turn_waits_keeps_the_view_as_the_thinking_folds_and_the_waiting_row_goes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The answer's start folds the thinking above it and hides the waiting row below it, in the same frame: the
    transcript keeps the rows of both, so a view that follows the bottom never moves up."""
    async with ChatPanelApp(thinking_expanded=lambda: True).run_test(size=(80, 24)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        await scrolled_to_bottom(cp, pilot)
        cp.agent_running = True
        await cp.add_thinking_delta("r1", "reasoning", "Line one.\n\nLine two.")
        cp.finish_thinking_segment("r1")
        thinking = cp.query_one(ThinkingMessage)
        row = cp.query_one(WaitingRow)
        await wait_for(
            lambda: (
                row.display
                and _shows(thinking.query_one(VirtualizedMarkdown), "Line one.\n\nLine two.")
                and cp.scroll_y == cp.max_scroll_y
                and screen_is_settled(app, cp.screen)
            ),
            pilot=pilot,
            description="the turn waits below its complete thinking",
        )
        bottom = cp.virtual_size.height
        layouts = record_layouts(monkeypatch)

        await cp.add_answer_delta("segment-1", "The answer")
        answer = cp.mounted_agent_messages()[-1]
        await wait_for(
            lambda: (
                thinking.collapsed
                and not row.display
                and _shows(answer.query_one(VirtualizedMarkdown), "The answer")
                and screen_is_settled(app, cp.screen)
            ),
            pilot=pilot,
            description="the answer started below the folded thinking",
        )
        assert _padded(cp) > 0
        assert layouts
        assert all(layout == (bottom, cp.max_scroll_y) for layout in layouts), layouts


async def test_a_tool_call_that_starts_while_the_turn_waits_keeps_the_view_as_the_thinking_folds_and_the_row_goes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its group mounts hidden until its first call shows: the waiting row stays until the group starts and folds
    the thinking above, so no layout shows the transcript shorter and a view that follows the bottom never moves up."""
    text = "\n\n".join(f"Line {index}." for index in range(5))
    async with ChatPanelApp(thinking_expanded=lambda: True).run_test(size=(80, 40)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        await scrolled_to_bottom(cp, pilot)
        cp.agent_running = True
        await cp.add_thinking_delta("r1", "reasoning", text)
        cp.finish_thinking_segment("r1")
        thinking = cp.query_one(ThinkingMessage)
        row = cp.query_one(WaitingRow)
        await wait_for(
            lambda: (
                row.display
                and _shows(thinking.query_one(VirtualizedMarkdown), text)
                and cp.scroll_y == cp.max_scroll_y
                and screen_is_settled(app, cp.screen)
            ),
            pilot=pilot,
            description="the turn waits below its complete thinking",
        )
        bottom = cp.virtual_size.height
        layouts = record_layouts(monkeypatch)

        await cp.add_tool_start("c-next", "read_file", KIND_FILESYSTEM_READ, "next.py", {"path": "next.py"})
        group = cp.query_one(ToolGroup)
        await wait_for(
            lambda: thinking.collapsed and not row.display and group.display and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the tool call started below the folded thinking",
        )
        assert _padded(cp) > 0
        assert layouts
        assert all(layout == (bottom, cp.max_scroll_y) for layout in layouts), layouts


async def _start(cp: ChatPanel, follows: str) -> Callable[[], bool]:
    """Start more text, or a tool call, as *follows* says; whether it shows."""
    if follows == "text":
        await cp.add_answer_delta("segment-2", "The answer")
        body = cp.mounted_agent_messages()[-1].query_one(VirtualizedMarkdown)
        return lambda: _shows(body, "The answer")
    await cp.add_tool_start("c-next", "read_file", KIND_FILESYSTEM_READ, "next.py", {"path": "next.py"})
    group = cp.query_one(ToolGroup)
    return lambda: group.display


@pytest.mark.parametrize("thinking_expanded", [False, True], ids=["folded-thinking", "open-thinking"])
@pytest.mark.parametrize("follows", ["tool-call", "text"])
async def test_text_folding_above_thinking_that_takes_a_gap_below_it_keeps_the_view_where_it_was(
    monkeypatch: pytest.MonkeyPatch, thinking_expanded: bool, follows: str
) -> None:
    """Text, then thinking, then a tool call or more text: the text folds above the thinking, which takes a gap below
    it, and a view that follows the bottom never moves."""
    async with ChatPanelApp(thinking_expanded=lambda: thinking_expanded, tool_groups_expanded=lambda: False).run_test(
        size=(80, 40)
    ) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        await scrolled_to_bottom(cp, pilot)
        cp.agent_running = True
        await cp.add_answer_delta("segment-1", "A note\n\nthat goes on\n\nand on")
        cp.finish_answer_segment("segment-1", completed=True)
        note = cp.mounted_agent_messages()[-1]
        await cp.add_thinking_delta("r1", "reasoning", "Think one.\n\nThink two.")
        cp.finish_thinking_segment("r1")
        thinking = cp.query_one(ThinkingMessage)
        await wait_for(
            lambda: cp.scroll_y == cp.max_scroll_y and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the thinking laid out below the note",
        )
        assert not thinking.gap_above
        bottom = cp.virtual_size.height
        layouts = record_layouts(monkeypatch)

        started = await _start(cp, follows)
        await wait_for(
            lambda: note.collapsed and thinking.collapsed and started() and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="what follows started below the folded note and thinking",
        )
        assert thinking.gap_above
        assert layouts
        assert all(layout == (bottom, cp.max_scroll_y) for layout in layouts), layouts


@contextmanager
def _held_parses(monkeypatch: pytest.MonkeyPatch, *sources: str) -> Iterator[dict[str, threading.Event]]:
    """Hold each markdown parse of one of *sources* until its event is set; leaving the block sets them all."""
    gates = {source: threading.Event() for source in sources}
    build_blocks = VirtualizedMarkdown._build_blocks

    def held_build_blocks(markdown: VirtualizedMarkdown, source: str) -> list[MarkdownBlock]:
        gate = gates.get(source)
        if gate is not None:
            gate.wait(timeout=ENGINE_TURN_TIMEOUT)
        return build_blocks(markdown, source)

    monkeypatch.setattr(VirtualizedMarkdown, "_build_blocks", held_build_blocks)
    try:
        yield gates
    finally:
        for gate in gates.values():
            gate.set()


async def _an_answer_below_a_folded_group(cp: ChatPanel, pilot: object) -> tuple[AgentMessage, VirtualizedMarkdown]:
    """Start an answer below the tool group it folds, in a view that follows the bottom: the rows the group gave up
    stay blank below it."""
    app = cp.app
    group = await _following_a_tool_group(cp, pilot, tools=1)
    await cp.add_answer_delta("segment-1", "The answer starts")
    answer = cp.mounted_agent_messages()[-1]
    body = answer.query_one(VirtualizedMarkdown)
    await wait_for(
        lambda: group.collapsed and _shows(body, "The answer starts") and screen_is_settled(app, cp.screen),
        pilot=pilot,
        description="the answer started below the folded group",
    )
    assert _padded(cp) > 0
    return answer, body


async def test_a_finished_answer_holds_its_rows_until_its_final_text_shows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The body parses the final text off the event loop: given back before, the rows showed an older text's height.

    The message grew and shrank as the text streamed before shrank and the final text took the rows again; so it did
    when the parse of the text streamed before landed after the final update.
    """
    tall = "- The answer starts\n- and fills\n- the rows"
    streamed = "The answer starts and"
    final = "- The answer starts\n- and fills\n- the rows at last"
    async with ChatPanelApp().run_test(size=(80, 24)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        answer, body = await _an_answer_below_a_folded_group(cp, pilot)
        answer.stream_update(tall)
        await wait_for(
            lambda: _shows(body, tall) and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the answer streamed tall",
        )
        held, bottom = answer.content_size.height, cp.virtual_size.height
        layouts = record_layouts(monkeypatch)
        with _held_parses(monkeypatch, streamed, final) as gates:
            answer.stream_update(streamed)
            await wait_for(
                lambda: body.source == streamed and body.lock.is_locked,
                pilot=pilot,
                description="the streamed text parses",
            )
            answer.stream_update(final, is_final=True)
            gates[streamed].set()
            await wait_for(
                lambda: (
                    body.shown_source == streamed and body._pending_parse is None and screen_is_settled(app, cp.screen)
                ),
                pilot=pilot,
                description="the streamed text showed while the final text parsed",
            )
            assert body.lock.is_locked
            assert _text_rows(answer) < held
            assert answer.content_size.height == held
        await wait_for(
            lambda: _shows(body, final) and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the final text showed",
        )
        assert answer.content_size.height == _text_rows(answer) == held
        assert all(height == bottom for height, _scroll_y in layouts), layouts
        assert cp.scroll_y == cp.max_scroll_y


async def test_a_retry_removes_an_answer_whose_final_text_still_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing the message runs waits for its body's parse: a wait there held off its removal until the parse ended."""
    final = "The answer starts\n\nand fills the rows"
    async with ChatPanelApp().run_test(size=(80, 24)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        answer, body = await _an_answer_below_a_folded_group(cp, pilot)
        with _held_parses(monkeypatch, final):
            answer.stream_update(final, is_final=True)
            await wait_for(lambda: body._pending_parse is None, description="the final text parses")
            removal = asyncio.create_task(cp.withdraw_answer(("segment-1",)))
            await wait_for(removal.done, description="the answer was removed")
            await removal
            assert body.lock.is_locked
        assert answer not in cp.mounted_agent_messages()
        assert not answer.is_attached


async def test_a_final_text_that_fails_to_parse_leaves_the_text_shown_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure is the body's: a wait for the parse in the message's handlers raised it there, and the App exited."""
    final = "The answer starts\n\nand cannot be parsed"
    build_blocks = VirtualizedMarkdown._build_blocks

    def failing_build_blocks(markdown: VirtualizedMarkdown, source: str) -> list[MarkdownBlock]:
        if source == final:
            raise RuntimeError("the parser failed")
        return build_blocks(markdown, source)

    async with ChatPanelApp().run_test(size=(80, 24)) as pilot:
        app = pilot.app
        cp = app.query_one(ChatPanel)
        answer, body = await _an_answer_below_a_folded_group(cp, pilot)
        monkeypatch.setattr(VirtualizedMarkdown, "_build_blocks", failing_build_blocks)
        answer.stream_update(final, is_final=True)
        await wait_for(
            lambda: body._pending_parse is None and not body.lock.is_locked and screen_is_settled(app, cp.screen),
            pilot=pilot,
            description="the final parse failed",
        )
        assert app.is_running
        assert body.shown_source == "The answer starts"
