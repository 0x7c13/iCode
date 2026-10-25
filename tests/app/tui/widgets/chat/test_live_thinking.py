# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The main chat shows the model's thinking as it streams, above what it led to.

Thinking is not part of the answer: an answer above it stays the answer,
copying leaves it out, and a retry that gives up its response takes it away.
A transcript whose thinking is not shown is as it would be without it.
"""

from __future__ import annotations

import asyncio

import pytest

from chrys.app.tui.widgets.chat.messages import (
    AgentMessage,
    ErrorMessage,
    InterruptedMessage,
    MessageCopyButton,
    ThinkingMessage,
    UserMessage,
    _AgentHeader,
)
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chat.tool_call import ToolGroup
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from tests.support.tui_helpers import ChatPanelApp, chat_content_children, click_when_settled
from tests.support.waiting import wait_for


def _shown() -> bool:
    return True


async def _think(cp: ChatPanel, segment_id: str, *texts: str, finish: bool = True) -> None:
    for text in texts:
        await cp.add_thinking_delta(segment_id, "reasoning", text)
    if finish:
        cp.finish_thinking_segment(segment_id)


async def _tool(cp: ChatPanel, call_id: str) -> None:
    await cp.add_tool_start(call_id, "read_file", "filesystem.read", f"path=/tmp/{call_id}")
    await cp.add_tool_result(call_id, "read_file", "tool output")


def _shape(cp: ChatPanel) -> list[tuple[str, str]]:
    """Each transcript entry as (kind, text); a tool group gives the number of its calls."""
    shape: list[tuple[str, str]] = []
    for child in chat_content_children(cp):
        if isinstance(child, ThinkingMessage):
            shape.append(("thinking", child.text))
        elif isinstance(child, AgentMessage):
            shape.append(("intermediate" if child._is_intermediate else "answer", child.text))
        elif isinstance(child, UserMessage):
            shape.append(("user", child.text))
        elif isinstance(child, ToolGroup):
            shape.append(("tools", str(len(child._tool_records))))
        else:
            shape.append((type(child).__name__, ""))
    return shape


async def _a_turn_with_thinking(cp: ChatPanel) -> None:
    await cp.add_user_message("prompt")
    await _think(cp, "r1", "Let me ", "look first.")
    await cp.add_answer_delta("a1", "Checking.")
    cp.finish_answer_segment("a1", completed=True)
    await _tool(cp, "read-1")
    await _think(cp, "r2", "The file says yes.")
    await cp.add_answer_delta("a2", "Yes.")
    cp.finish_answer_segment("a2", completed=True)
    await cp.add_agent_message("Yes.")


@pytest.mark.parametrize("show", [True, False], ids=["shown", "not-shown"])
async def test_thinking_streams_above_what_it_led_to_or_leaves_the_transcript_as_without_it(show: bool) -> None:
    async with ChatPanelApp(show_thinking=lambda: show).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await _a_turn_with_thinking(cp)
        await pilot.pause()

        without_thinking = [
            ("user", "prompt"),
            ("intermediate", "Checking."),
            ("tools", "1"),
            ("answer", "Yes."),
        ]
        if not show:
            assert _shape(cp) == without_thinking
            return
        assert _shape(cp) == [
            ("user", "prompt"),
            ("thinking", "Let me look first."),
            ("intermediate", "Checking."),
            ("tools", "1"),
            ("thinking", "The file says yes."),
            ("answer", "Yes."),
        ]
        thinking = list(cp.query(ThinkingMessage))
        assert [message._is_final for message in thinking] == [True, True]
        assert [message.collapsed for message in thinking] == [False, False], "it stays open once complete"
        await wait_for(
            lambda: (
                [message.query_one(VirtualizedMarkdown).shown_source for message in thinking]
                == ["Let me look first.", "The file says yes."]
            ),
            pilot=pilot,
            description="each thinking body shows its text",
        )
        # Copying the agent's replies, or the whole conversation, leaves the thinking out.
        assert cp.get_agent_responses() == [("Agent", "Checking."), ("Agent", "Yes.")]
        assert cp.get_all_messages() == [("You", "prompt"), ("Agent", "Checking."), ("Agent", "Yes.")]


async def test_thinking_ends_the_tool_group_above_it_once_its_text_is_not_blank() -> None:
    async with ChatPanelApp(show_thinking=_shown).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("prompt")
        await _tool(cp, "read-1")
        await _think(cp, "r-blank", " ", "\n")
        await _tool(cp, "read-2")
        group = cp.query_one(ToolGroup)
        assert _shape(cp) == [("user", "prompt"), ("tools", "2")], "blank thinking mounts nothing, the group stays open"

        await pilot.pause()
        assert not group.collapsed
        folded = group.rows_folding_gives_up()
        assert folded > 0
        await _think(cp, "r1", "Two reads were enough.", finish=False)
        thinking = cp.query_one(ThinkingMessage)
        # Measured before the fold hid the group's calls.
        assert thinking._folded_above == (folded, group.styles.margin.bottom)
        assert group.collapsed

        cp.finish_thinking_segment("r1")
        await _tool(cp, "read-3")
        assert _shape(cp) == [
            ("user", "prompt"),
            ("tools", "2"),
            ("thinking", "Two reads were enough."),
            ("tools", "1"),
        ]


async def test_thinking_not_shown_leaves_the_tool_group_as_it_was_for_the_calls_after_it() -> None:
    async with ChatPanelApp(show_thinking=lambda: False).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("prompt")
        await _tool(cp, "read-1")
        await pilot.pause()
        group = cp.query_one(ToolGroup)
        collapsed = group.collapsed

        await _think(cp, "r1", "Two reads were enough.")
        await _tool(cp, "read-2")

        assert _shape(cp) == [("user", "prompt"), ("tools", "2")]
        assert group.collapsed is collapsed


@pytest.mark.parametrize("answer_follows", [False, True], ids=["last-answer", "answer-follows"])
async def test_thinking_after_an_answer_leaves_it_the_answer_until_another_answer_follows(
    answer_follows: bool,
) -> None:
    async with ChatPanelApp(show_thinking=_shown).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("prompt")
        await cp.add_answer_delta("a1", "First.")
        cp.finish_answer_segment("a1", completed=True)
        await _think(cp, "r1", "Maybe more?")
        if answer_follows:
            await cp.add_answer_delta("a2", "Second.")
            cp.finish_answer_segment("a2", completed=True)
        await cp.add_agent_message("Second." if answer_follows else "First.")
        await pilot.pause()

        if answer_follows:
            assert _shape(cp) == [
                ("user", "prompt"),
                ("intermediate", "First."),
                ("thinking", "Maybe more?"),
                ("answer", "Second."),
            ]
        else:
            assert _shape(cp) == [("user", "prompt"), ("answer", "First."), ("thinking", "Maybe more?")]
            await wait_for(
                lambda: cp.query_one(AgentMessage).query_one(MessageCopyButton).display,
                pilot=pilot,
                description="the answer above the thinking offers its copy button",
            )


@pytest.mark.parametrize("ending", ["segment-failed", "error", "interrupted", "retry", "new-prompt"])
async def test_thinking_cut_off_keeps_the_text_it_showed(ending: str) -> None:
    async with ChatPanelApp(show_thinking=_shown).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("prompt")
        await _think(cp, "r1", "Half a ", "thought", finish=False)
        thinking = cp.query_one(ThinkingMessage)
        assert not thinking._is_final

        if ending == "segment-failed":
            cp.finish_thinking_segment("r1")
        elif ending == "error":
            await cp.add_error("failed")
        elif ending == "interrupted":
            await cp.add_interrupted()
        elif ending == "retry":
            await cp.prepare_retry()
        else:
            await cp.add_user_message("next prompt")

        assert thinking._is_final
        assert thinking.text == "Half a thought"
        # Whatever streams next starts its own message.
        await _think(cp, "r2", "Fresh start.")
        assert [message.text for message in cp.query(ThinkingMessage)] == ["Half a thought", "Fresh start."]


async def test_a_retry_removes_the_thinking_of_the_response_it_gave_up() -> None:
    async with ChatPanelApp(show_thinking=_shown).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("prompt")
        await _think(cp, "r-kept", "From a landed response.")
        await _tool(cp, "read-1")
        await _think(cp, "r-done", "Given up, though complete.")
        await _think(cp, "r-open", "Given up mid-", finish=False)

        await cp.abandon_thinking(("r-done", "r-open"))
        # The given-up stream's closing event comes after: it finds nothing to close.
        cp.finish_thinking_segment("r-open")
        await _think(cp, "r-retry", "Thinking again.")

        assert _shape(cp) == [
            ("user", "prompt"),
            ("thinking", "From a landed response."),
            ("tools", "1"),
            ("thinking", "Thinking again."),
        ]


async def test_an_interrupt_while_a_retry_removes_its_thinking_still_removes_all_of_it() -> None:
    """The removal runs in the engine task an interrupt cancels; the open block must not stay, never final."""
    async with ChatPanelApp(show_thinking=_shown).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("prompt")
        await _think(cp, "r-done", "Given up, though complete.")
        await _think(cp, "r-open", "Given up mid-", finish=False)

        removal = asyncio.create_task(cp.abandon_thinking(("r-done", "r-open")))
        # One turn of the loop: the removal of the first block has started, not finished.
        await asyncio.sleep(0)
        assert not removal.done()
        removal.cancel()
        with pytest.raises(asyncio.CancelledError):
            await removal

        assert _shape(cp) == [("user", "prompt")]


async def test_whether_thinking_shows_is_read_as_its_segment_opens() -> None:
    show = [False]
    async with ChatPanelApp(show_thinking=lambda: show[0]).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("prompt")
        await _think(cp, "r1", "Hidden ", finish=False)
        show[0] = True
        await _think(cp, "r1", "to its end.")
        await _think(cp, "r2", "Shown ", finish=False)
        show[0] = False
        await _think(cp, "r2", "to its end.")
        await _think(cp, "r3", "Hidden again.")

        assert _shape(cp) == [("user", "prompt"), ("thinking", "Shown to its end.")]


async def test_a_new_prompt_drops_a_failed_prompt_with_the_thinking_it_alone_got() -> None:
    async with ChatPanelApp(show_thinking=_shown).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("failed prompt")
        await _think(cp, "r1", "Weighing it ", finish=False)
        await _think(cp, "r2", "a summary")
        await cp.add_interrupted()

        await cp.add_user_message("replacement prompt")

        assert _shape(cp) == [("user", "replacement prompt")]
        assert [item.summary for item in cp.toc_items] == ["replacement prompt"]


@pytest.mark.parametrize("status_type", [ErrorMessage, InterruptedMessage])
async def test_a_new_prompt_keeps_a_failed_prompt_whose_turn_got_more_than_thinking(status_type: type) -> None:
    async with ChatPanelApp(show_thinking=_shown).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("failed prompt")
        await _think(cp, "r1", "First look.")
        await _tool(cp, "read-1")
        await _think(cp, "r2", "Half ", finish=False)
        if status_type is ErrorMessage:
            await cp.add_error("failed")
        else:
            await cp.add_interrupted()

        await cp.add_user_message("replacement prompt")

        assert _shape(cp) == [
            ("user", "failed prompt"),
            ("thinking", "First look."),
            ("tools", "1"),
            ("thinking", "Half"),
            ("user", "replacement prompt"),
        ]


async def test_thinking_names_what_it_shows_and_folds_from_its_header_once_complete() -> None:
    async with ChatPanelApp(show_thinking=_shown).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("prompt")
        await _think(cp, "r1", "Step one.")
        await cp.add_thinking_delta("s1", "reasoning_summary", "In short, ")
        thinking, summary = cp.query(ThinkingMessage)

        assert [message.query_one(_AgentHeader).render().plain for message in (thinking, summary)] == [
            "◇ Thinking",
            "◇ Thinking summary",
        ]
        assert not thinking.query(MessageCopyButton)
        # A selection over the header copies its label, as an answer's does.
        assert thinking.query_one(_AgentHeader).get_selection(None) == ("[Thinking]", "\n")  # type: ignore[arg-type]

        # A streaming message does not fold.
        await click_when_settled(pilot, summary.query_one(_AgentHeader))
        await pilot.pause()
        assert summary.collapsed is False

        await click_when_settled(pilot, thinking.query_one(_AgentHeader))
        await wait_for(lambda: thinking.collapsed, pilot=pilot, description="complete thinking folded")
        body = thinking.query_one(VirtualizedMarkdown)
        assert body.display is False
        assert thinking.query_one(_AgentHeader).render().plain == "▶ Thinking"
        await click_when_settled(pilot, thinking.query_one(_AgentHeader))
        await wait_for(lambda: not thinking.collapsed and body.display, pilot=pilot, description="unfolded")


async def test_fold_all_folds_complete_thinking_with_the_tool_groups_but_leaves_thinking_that_streams_open() -> None:
    async with ChatPanelApp(show_thinking=_shown).run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("prompt")
        await _tool(cp, "read-1")
        await _think(cp, "r1", "Step one.")
        await cp.add_thinking_delta("s1", "reasoning_summary", "In short, ")
        (group,) = cp.query(ToolGroup)
        thinking, streaming = cp.query(ThinkingMessage)
        group.collapsed = True

        # The open thinking is what is expanded, so fold-all folds.
        assert cp.toggle_fold_all() is True
        assert (group.collapsed, thinking.collapsed, streaming.collapsed) == (True, True, False)
        assert thinking.query_one(VirtualizedMarkdown).display is False

        assert cp.toggle_fold_all() is False
        assert (group.collapsed, thinking.collapsed, streaming.collapsed) == (False, False, False)
        assert thinking.query_one(VirtualizedMarkdown).display is True

        cp.finish_thinking_segment("s1")
        assert cp.toggle_fold_all() is True
        assert (group.collapsed, thinking.collapsed, streaming.collapsed) == (True, True, True)
