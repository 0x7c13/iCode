# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A folded thinking body goes by the same young/frozen removal contract as tool groups; text between tool calls keeps its hidden body unparsed."""

from __future__ import annotations

import asyncio
import threading

import pytest
from textual.screen import Screen

from chrys.app.tui.support import gc_freeze
from chrys.app.tui.support.gc_freeze import GcAbsorbReason, GcAbsorbRequested, GcReclaimRequested
from chrys.app.tui.widgets.chat import messages as messages_module
from chrys.app.tui.widgets.chat.messages import AgentMessage, ThinkingMessage, _AgentHeader
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from chrys.foundation.tool_kinds import KIND_FILESYSTEM_READ
from tests.support.pilot_barrier import screen_is_settled
from tests.support.tui_helpers import GcMessageChatPanelApp, click_when_settled
from tests.support.waiting import wait_for


@pytest.mark.parametrize("frozen", [False, True], ids=["young", "frozen"])
async def test_fold_releases_body_with_epoch_aware_reclaim(monkeypatch: pytest.MonkeyPatch, frozen: bool) -> None:
    app = GcMessageChatPanelApp(thinking_expanded=lambda: True)
    async with app.run_test() as pilot:
        panel = app.query_one(ChatPanel)
        await panel.add_user_message("prompt")
        await panel.add_thinking_delta("r1", "reasoning", "# Evidence\n\nLook at the file.")
        panel.finish_thinking_segment("r1")
        message = panel.query_one(ThinkingMessage)
        await wait_for(
            lambda: message.query_one(VirtualizedMarkdown).shown_source == message.text,
            pilot=pilot,
            description="thinking parse ready",
        )
        body = message.query_one(VirtualizedMarkdown)
        epoch = gc_freeze.current_gc_freeze_epoch()
        if frozen:
            monkeypatch.setattr(gc_freeze, "_freeze_epoch", epoch + 1)
        message.collapsed = True
        await wait_for(
            lambda: any(isinstance(event, GcReclaimRequested) for event in app.gc_messages),
            pilot=pilot,
            description="thinking reclaim request",
        )
        reclaim = next(event for event in app.gc_messages if isinstance(event, GcReclaimRequested))
        assert reclaim.prompt is False
        assert not message.query(VirtualizedMarkdown)
        assert not body.is_attached
        if frozen:
            assert reclaim.removed is None
        else:
            assert reclaim.removed is not None
            assert reclaim.removed.built_at_epoch == epoch
            assert body in [node() for node in reclaim.removed.nodes]
        message.collapsed = False
        await wait_for(lambda: bool(message.query(VirtualizedMarkdown)), pilot=pilot, description="thinking rebuilt")
        assert message.query_one(VirtualizedMarkdown) is not body
        await wait_for(
            lambda: message.query_one(VirtualizedMarkdown).shown_source == message.text,
            pilot=pilot,
            description="rebuilt thinking text ready",
        )


async def test_the_turn_ending_folds_its_thinking_and_fast_toggles_keep_one_body() -> None:
    async with GcMessageChatPanelApp(thinking_expanded=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_user_message("prompt")
        await panel.add_thinking_delta("r", "reasoning", "Reasoning.")
        panel.finish_thinking_segment("r")
        await panel.add_agent_message("Done.")
        message = panel.query_one(ThinkingMessage)
        # The turn went on past it: it folded.
        assert message.collapsed
        message.collapsed = False
        message.collapsed = True
        message.collapsed = False
        message.collapsed = True
        message.collapsed = False
        await wait_for(
            lambda: len(message.query(VirtualizedMarkdown)) == 1,
            pilot=pilot,
            description="one final body after rapid toggles",
        )
        assert not message.collapsed


async def test_text_a_tool_batch_follows_folds_as_it_starts_and_its_hidden_body_parses_its_final_text_only_if_it_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = GcMessageChatPanelApp(thinking_expanded=lambda: True)
    async with app.run_test() as pilot:
        panel = app.query_one(ChatPanel)
        await panel.add_user_message("prompt")
        await panel.add_answer_delta("a1", "Let me look at the file.")
        panel.finish_answer_segment("a1", completed=True)
        message = panel.query_one(AgentMessage)
        await wait_for(
            lambda: message.query_one(VirtualizedMarkdown).shown_source == message.text,
            pilot=pilot,
            description="the streamed text shows",
        )
        body = message.query_one(VirtualizedMarkdown)
        handed: list[str] = []
        update = body.update

        def recording_update(markdown: str) -> object:
            handed.append(markdown)
            return update(markdown)

        monkeypatch.setattr(body, "update", recording_update)

        await panel.add_tool_start("c1", "read_file", KIND_FILESYSTEM_READ, args={"path": "x.py"})
        assert (message.foldable, message.collapsed) == (True, True)
        await panel.add_tool_result("c1", "read_file", "ok", 3)
        await panel.add_answer_delta("a2", "More text.")
        # Removing the hidden body would lay the transcript out once more: it stays, handed nothing while folded.
        assert list(message.query(VirtualizedMarkdown)) == [body]
        assert (body.display, handed) == (False, [])

        blocks = body._blocks
        message.collapsed = False
        await wait_for(
            lambda: body.display and screen_is_settled(app, panel.screen),
            pilot=pilot,
            description="opened, it shows its body",
        )
        # It showed its final text before it folded: opened, it shows the blocks it had, not new ones parsed from the
        # same text that no GC absorb would take.
        assert (body.shown_source, handed) == (message.text, [])
        assert body._blocks is blocks


async def _text_folded_before_its_body_shows_its_final_text(
    panel: ChatPanel, pilot: object, monkeypatch: pytest.MonkeyPatch, *, shown: str = "First part.\n\nSecond part."
) -> tuple[AgentMessage, VirtualizedMarkdown]:
    """Text between tool calls whose body shows only what streamed of it as *shown*: the tool call after it folded
    it before its last parts reached the body."""
    # The body catches up with the stream no sooner than an hour after it last did.
    monkeypatch.setattr(messages_module, "_STREAMED_BODY_INTERVAL_SECONDS", 3600.0)
    await panel.add_user_message("prompt")
    panel.agent_running = True
    first, rest = shown.split("\n\n", 1)
    await panel.add_answer_delta("a1", first)
    message = panel.mounted_agent_messages()[-1]
    body = message.query_one(VirtualizedMarkdown)
    # The body is built with the text it starts from, and handed the next parts at once.
    await panel.add_answer_delta("a1", f"\n\n{rest}")
    await wait_for(lambda: body.shown_source == shown, pilot=pilot, description="the streamed text shows")
    await panel.add_answer_delta("a1", "\n\nThird part.\n\nFourth part.")
    panel.finish_answer_segment("a1", completed=True)
    await panel.add_tool_start("c1", "read_file", KIND_FILESYSTEM_READ, args={"path": "x.py"})
    assert (message.collapsed, body.display, body.shown_source) == (True, False, shown)
    return message, body


async def test_text_between_tool_calls_opened_shows_no_body_until_it_shows_the_final_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = GcMessageChatPanelApp()
    async with app.run_test(size=(80, 30)) as pilot:
        panel = app.query_one(ChatPanel)
        message, body = await _text_folded_before_its_body_shows_its_final_text(panel, pilot, monkeypatch)
        painted: list[str | None] = []
        paint = Screen._compositor_refresh

        def record(screen: Screen) -> None:
            painted.append(body.shown_source if body in screen._compositor.visible_widgets else None)
            paint(screen)

        def absorbs() -> list[GcAbsorbReason]:
            return [message.reason for message in app.gc_messages if isinstance(message, GcAbsorbRequested)]

        absorbs_before = len(absorbs())
        monkeypatch.setattr(Screen, "_compositor_refresh", record)
        await click_when_settled(pilot, message.query_one(_AgentHeader))
        await wait_for(
            lambda: painted[-1:] == [message.text], pilot=pilot, description="opened, the body shows the final text"
        )
        # Shown at once, the body would paint the text it had for a frame, and the final text's rows only after it.
        assert set(painted) == {None, message.text}
        # The kept body's parse of it is stable content, as a body's built on opening is.
        await wait_for(
            lambda: absorbs()[absorbs_before:],
            pilot=pilot,
            description="the opened body's parse is handed to the GC freeze",
        )
        assert absorbs()[absorbs_before:] == [GcAbsorbReason.STABLE_CONTENT_MOUNTED]


async def test_text_between_tool_calls_whose_final_text_fails_to_parse_shows_the_text_it_had_once_opened(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async with GcMessageChatPanelApp().run_test(size=(80, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        message, body = await _text_folded_before_its_body_shows_its_final_text(panel, pilot, monkeypatch)

        def failing_parse(markdown: str) -> list[object]:
            raise ValueError(markdown)

        monkeypatch.setattr(body, "_build_blocks", failing_parse)
        message.collapsed = False
        await wait_for(lambda: body.display, pilot=pilot, description="opened, the body shows")
        assert (message.collapsed, body.shown_source) == (False, "First part.\n\nSecond part.")
        assert "Parsing an opened message's text failed" in caplog.text


async def test_thinking_opened_as_it_streams_shows_its_final_text_once_that_parses_not_the_text_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its final text comes as the opening parse runs: the body shows once that text parses, not the opened one."""
    opened = "First part."
    final = "First part. And then\n\nanother paragraph\n\nand a third one."
    gates = {opened: threading.Event(), final: threading.Event()}
    build_blocks = VirtualizedMarkdown._build_blocks

    def gated_parse(body: VirtualizedMarkdown, markdown: str) -> list[object]:
        if gate := gates.get(markdown):
            gate.wait(10.0)
        return build_blocks(body, markdown)

    monkeypatch.setattr(VirtualizedMarkdown, "_build_blocks", gated_parse)
    async with GcMessageChatPanelApp(thinking_expanded=lambda: False).run_test(size=(80, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_user_message("prompt")
        panel.agent_running = True
        await panel.add_thinking_delta("r1", "reasoning", opened)
        message = panel.query_one(ThinkingMessage)
        await wait_for(lambda: screen_is_settled(pilot.app, panel.screen), pilot=pilot, description="it streams")
        assert message.collapsed
        painted: list[str | None] = []
        paint = Screen._compositor_refresh

        def record(screen: Screen) -> None:
            body = message._md_widget
            painted.append(body.shown_source if body in screen._compositor.visible_widgets else None)
            paint(screen)

        monkeypatch.setattr(Screen, "_compositor_refresh", record)
        message.collapsed = False
        await wait_for(lambda: message._md_widget is not None, pilot=pilot, description="opened, it builds a body")
        body = message._md_widget
        await panel.add_thinking_delta("r1", "reasoning", final.removeprefix(opened))
        panel.finish_thinking_segment("r1")
        gates[opened].set()
        await wait_for(lambda: body.shown_source == opened, pilot=pilot, description="the opened text parses")
        await wait_for(lambda: screen_is_settled(pilot.app, panel.screen), pilot=pilot, description="and lays out")
        # Shown now, it would take the opened text's rows, and the final text's once that parses.
        assert not body.display
        gates[final].set()
        await wait_for(lambda: painted[-1:] == [final], pilot=pilot, description="the body shows the final text")
        assert set(painted) == {None, final}


@pytest.mark.parametrize("entry", ["text between tool calls", "thinking"])
async def test_an_entry_opened_while_its_text_parses_clears_before_the_parse_ends(
    monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """Opening waits for the body's parse beside the message, not on its pump: the messages it handles go on."""
    async with GcMessageChatPanelApp(thinking_expanded=lambda: False).run_test(size=(80, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        message: AgentMessage | ThinkingMessage
        if entry == "thinking":
            await panel.add_user_message("prompt")
            await panel.add_thinking_delta("r1", "reasoning", "Reasoning.")
            panel.finish_thinking_segment("r1")
            message = panel.query_one(ThinkingMessage)
        else:
            message, _body = await _text_folded_before_its_body_shows_its_final_text(panel, pilot, monkeypatch)
        assert message.collapsed
        started = threading.Event()
        release = threading.Event()
        build_blocks = VirtualizedMarkdown._build_blocks

        def held_parse(body: VirtualizedMarkdown, markdown: str) -> list[object]:
            if markdown == message.text:
                started.set()
                release.wait(10.0)
            return build_blocks(body, markdown)

        monkeypatch.setattr(VirtualizedMarkdown, "_build_blocks", held_parse)
        message.collapsed = False
        # Not through the pilot: its pauses wait for the very messages a blocked pump would hold up.
        await wait_for(started.is_set, description="the opened body's parse started")
        clearing = asyncio.create_task(panel.clear())
        try:
            await wait_for(clearing.done, description="the transcript cleared as the parse runs")
        finally:
            release.set()
            await clearing
        assert not message.is_attached


@pytest.mark.parametrize("turn", ["running", "ended"])
async def test_text_between_tool_calls_opened_by_hand_moves_the_view_once_as_its_body_shows(
    monkeypatch: pytest.MonkeyPatch, turn: str
) -> None:
    """Until its body shows the final text, the opened text keeps the gap under it, and a running turn the rows it
    holds: the view would move up as they went, and down again as the body showed."""
    async with GcMessageChatPanelApp().run_test(size=(80, 24)) as pilot:
        app = pilot.app
        panel = app.query_one(ChatPanel)
        for index in range(6):
            await panel.add_user_message(f"message {index}")
            await panel.add_agent_message("\n".join(f"line {line}" for line in range(8)))
        shown = "\n\n".join(f"Part {index}." for index in range(6))
        message, body = await _text_folded_before_its_body_shows_its_final_text(panel, pilot, monkeypatch, shown=shown)
        gap = panel.bottom_gap()
        assert gap is not None
        if turn == "ended":
            await panel.add_tool_result("c1", "read_file", "ok", 3)
            panel.agent_running = False
        await wait_for(
            lambda: panel.scroll_y == panel.max_scroll_y and screen_is_settled(app, panel.screen),
            pilot=pilot,
            description="the view follows the bottom",
        )
        assert (gap.outer_size.height > 0) is (turn == "running")
        header = message.query_one(_AgentHeader)
        before = (header.region.y, panel.virtual_size.height)
        release = threading.Event()
        build_blocks = body._build_blocks

        def held_parse(markdown: str) -> list[object]:
            release.wait(10.0)
            return build_blocks(markdown)

        monkeypatch.setattr(body, "_build_blocks", held_parse)
        painted: list[tuple[int | None, int, str | None]] = []
        paint = Screen._compositor_refresh

        def record(screen: Screen) -> None:
            visible = screen._compositor.visible_widgets
            row = visible.get(header)
            shows = body.shown_source if body in visible else None
            painted.append((None if row is None else row[0].y, panel.virtual_size.height, shows))
            paint(screen)

        monkeypatch.setattr(Screen, "_compositor_refresh", record)
        try:
            await click_when_settled(pilot, header)
            if turn == "running":
                # What the turn lands while the text parses lays it out; the rows it holds would hide the gap going.
                await panel.add_tool_result("c1", "read_file", "ok", 3)
            else:
                # Whatever else in the transcript lays it out while the text parses.
                panel.mounted_agent_messages()[0].refresh(layout=True)
            await wait_for(
                lambda: not message.collapsed and screen_is_settled(app, panel.screen),
                pilot=pilot,
                description="a layout painted as the text parses",
            )
            assert painted and not body.display
        finally:
            release.set()
        await wait_for(
            lambda: painted[-1][2] == message.text and screen_is_settled(app, panel.screen),
            pilot=pilot,
            description="the body shows the final text",
        )
        assert {(row, height) for row, height, shows in painted if shows is None} == {before}, painted
        assert {shows for _row, _height, shows in painted if shows is not None} == {message.text}, painted
