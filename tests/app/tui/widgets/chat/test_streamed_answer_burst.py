# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A streamed answer burst costs one think-tag pass, one parse and a bounded number of layouts.

A synchronous burst of cumulative ``InvocationMessage`` snapshots, such as
the per-line replay of a buffered answer the main turn once published, must
not cost per event. Each event used to process think tags over the whole text
and queue a full markdown parse plus a transcript relayout, so an N-line
answer cost O(N^2) parsing after it had already arrived.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from textual.await_complete import AwaitComplete
from textual.geometry import Size

from chrys.app.tui.widgets.chat import messages as messages_module
from chrys.app.tui.widgets.chat.messages import AgentCopyButton, AgentMessage
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.markdown.blocks import MarkdownBlock
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from chrys.foundation.i18n import MessageRef
from chrys.foundation.i18n.formatting import format_message
from chrys.kernel import Content, Message
from tests.support.pilot_barrier import screen_is_settled
from tests.support.tui_helpers import ChatPanelApp, WidgetApp
from tests.support.waiting import ENGINE_TEST_WAIT_TIMEOUT, wait_for, with_wait_deadline

_ANSWER_LINES = 200
_SCREEN_LAYOUT_BOUND = 12
"""Screen layouts a burst may cost: one install plus cursor, copy button and scroll settling."""


def _answer(lines: int) -> str:
    parts = ["<think>outline the answer</think># Streaming pipeline report\n", "\n"]
    for index in range(lines - 2):
        if index % 20 == 0:
            parts.append(f"## Section {index // 20}\n")
        elif index % 20 == 10:
            parts.append(f"- item {index} with `inline_code_{index}()` and **bold** text\n")
        else:
            parts.append(f"Line {index} of the answer explains one more step of the pipeline.\n")
    return "".join(parts)


def _cumulative_lines(text: str) -> list[str]:
    """Cumulative snapshots of ``text``, one per line, as a line-by-line replay publishes them."""
    emitted = ""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        emitted += line
        out.append(emitted)
    return out


def _transcript(turns: int) -> list[dict[str, object]]:
    history: list[dict[str, object]] = []
    for index in range(turns):
        history.append(Message("user", [Content.from_text(f"Question {index}\nwith a second line")]).to_dict())
        history.append(
            Message(
                "assistant",
                [Content.from_text(f"## Answer {index}\n\nA paragraph with `code` and **emphasis**.\n\n- a\n- b")],
            ).to_dict()
        )
    return history


async def test_a_streamed_burst_processes_and_hands_the_body_its_text_once(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = _answer(_ANSWER_LINES)
    cumulative = _cumulative_lines(answer)
    message = AgentMessage(cumulative[0], is_final=False)

    async with WidgetApp(lambda: message).run_test() as pilot:
        body = message.query_one(VirtualizedMarkdown)
        await wait_for(lambda: body.source and body._blocks, pilot=pilot, description="first line parsed")
        await pilot.pause()

        handed: list[str] = []
        real_update = body.update

        def recording_update(markdown: str) -> AwaitComplete:
            handed.append(markdown)
            return real_update(markdown)

        processed: list[str] = []
        real_process = messages_module.process_think_tags

        def recording_process(
            text: str,
            *,
            intermediate: bool = False,
            render_message: Callable[[MessageRef], str] = format_message,
        ) -> str:
            processed.append(text)
            return real_process(text, intermediate=intermediate, render_message=render_message)

        monkeypatch.setattr(body, "update", recording_update)
        monkeypatch.setattr(messages_module, "process_think_tags", recording_process)

        for text in cumulative[1:]:
            message.stream_update(text, is_final=False)

        assert handed == []
        assert processed == []
        # Readers never see a stale answer, even before the body catches up.
        assert message.text == real_process(answer)
        assert message.format_agent_response_copy() == real_process(answer)
        assert processed == [answer]

        await wait_for(lambda: handed, pilot=pilot, description="body synced once per loop turn")
        assert handed == [real_process(answer)]
        assert processed == [answer]

        final = answer + "Done.\n"
        message.stream_update(final, is_final=True)

        # The final update reaches the body before the cursor and copy button change.
        assert handed == [real_process(answer), real_process(final)]
        await wait_for(
            lambda: not message.query(".agent-cursor") and message.query_one(AgentCopyButton).display,
            pilot=pilot,
            description="cursor removed and copy button shown",
        )
        await wait_for(
            lambda: body.source == real_process(final) and not body.lock.is_locked,
            pilot=pilot,
            description="final parse installed",
        )


@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_a_long_answer_burst_into_a_populated_transcript_parses_once_with_bounded_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer = _answer(_ANSWER_LINES)
    cumulative = _cumulative_lines(answer)
    final_body = messages_module.process_think_tags(answer)

    async with ChatPanelApp().run_test(size=(120, 36)) as pilot:
        app = pilot.app
        screen = app.screen
        panel = app.query_one(ChatPanel)
        await panel.replay_history(_transcript(24))
        await panel.add_user_message("Explain the streaming pipeline in depth.")
        await wait_for(
            lambda: (
                len(panel.query(AgentMessage)) == 24
                and all(markdown._blocks for markdown in panel.query(VirtualizedMarkdown))
            ),
            pilot=pilot,
            description="populated transcript parsed",
        )

        # The first event mounts the streamed message; its body parses the first line on mount.
        await panel.add_agent_message(cumulative[0], is_final=False)
        message = panel.query(AgentMessage).last()
        body = message.query_one(VirtualizedMarkdown)
        await wait_for(lambda: body._blocks, pilot=pilot, description="streamed message mounted")
        await wait_for(lambda: screen_is_settled(app, screen), pilot=pilot, description="settled before burst")

        parsed: list[str] = []
        real_build = VirtualizedMarkdown._build_blocks

        def recording_build(markdown_widget: VirtualizedMarkdown, markdown: str) -> list[MarkdownBlock]:
            if markdown_widget is body:
                parsed.append(markdown)
            return real_build(markdown_widget, markdown)

        layouts: list[None] = []
        real_refresh_layout = screen._refresh_layout

        def recording_refresh_layout(size: Size | None = None, scroll: bool = False) -> None:
            layouts.append(None)
            real_refresh_layout(size, scroll)

        monkeypatch.setattr(VirtualizedMarkdown, "_build_blocks", recording_build)
        monkeypatch.setattr(screen, "_refresh_layout", recording_refresh_layout)

        # MainScreen awaits each event's handler inline, so the whole burst runs in one loop step.
        for text in cumulative[1:]:
            await panel.add_agent_message(text, is_final=False)
        await panel.add_agent_message(answer, is_final=True)

        await wait_for(
            # The lock is held from the parse of the final text until its blocks are installed.
            lambda: parsed and parsed[-1] == final_body and not body.lock.is_locked,
            pilot=pilot,
            description="final answer installed",
        )
        await wait_for(lambda: screen_is_settled(app, screen), pilot=pilot, description="settled after burst")

        assert parsed == [final_body]
        assert body.source == final_body
        assert message.text == final_body
        assert len(layouts) <= _SCREEN_LAYOUT_BOUND, f"{len(layouts)} screen layouts for one streamed answer"
