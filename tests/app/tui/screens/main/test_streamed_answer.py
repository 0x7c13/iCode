# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A main turn's answer shows as its deltas arrive, and the final completion snapshot never repeats it."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from textual.widgets import Button

from chrys.app.tui.widgets.chat.messages import AgentMessage, InterruptedMessage
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationContentDelta, InvocationContentFinished, InvocationMessage
from chrys.foundation.models.invocations import InvocationOrigin
from tests.support.tui_app_harness import SessionGenerationEngine, make_chrys_app
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from pathlib import Path

_ORIGIN = InvocationOrigin("turn", "s1", "turn-1", None)


def _shown(panel: ChatPanel) -> list[str]:
    """The answer text each mounted agent message's markdown body currently renders."""
    return [message.query_one(VirtualizedMarkdown).source for message in panel.query(AgentMessage)]


async def test_streamed_answer_text_shows_before_the_final_message_and_once_after_it(tmp_path: Path) -> None:
    bus = EventBus()
    app = make_chrys_app(tmp_path, engine=SessionGenerationEngine(), event_bus=bus)
    async with app.run_test(size=(120, 36)) as pilot:
        main = app._main_screen
        assert main is not None
        panel = main.query_one(ChatPanel)
        main._set_agent_running(True)

        for text in ("Streaming ", "answer"):
            await bus.publish(
                InvocationContentDelta(
                    origin=_ORIGIN, session_id="s1", segment_id="seg-1", channel="answer", text=text
                ),
                raise_handler_errors=True,
            )
        await wait_for(
            lambda: _shown(panel) == ["Streaming answer"],
            pilot=pilot,
            description="streamed answer rendered before the final message",
        )
        (streaming,) = panel.query(AgentMessage)
        assert streaming.has_class("--streaming"), "muted until it is known to be the answer"
        assert main.agent_running_state is True

        await bus.publish(
            InvocationContentFinished(
                origin=_ORIGIN, session_id="s1", segment_id="seg-1", channel="answer", status="completed"
            ),
            raise_handler_errors=True,
        )
        await bus.publish(
            InvocationMessage(origin=_ORIGIN, session_id="s1", text="Streaming answer", is_final=True),
            raise_handler_errors=True,
        )
        await wait_for(
            lambda: not main.agent_running_state and not streaming.has_class("--streaming"),
            pilot=pilot,
            description="turn completed and the streamed message finalized",
        )
        assert list(panel.query(AgentMessage)) == [streaming]
        assert _shown(panel) == ["Streaming answer"]
        assert not streaming.has_class("--streaming") and not streaming.has_class("--intermediate")


def _delta(segment_id: str, text: str) -> InvocationContentDelta:
    return InvocationContentDelta(origin=_ORIGIN, session_id="s1", segment_id=segment_id, channel="answer", text=text)


async def test_a_bare_continue_removes_the_half_answer_it_asks_for_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare Continue drops the interrupted pass's answer from history and asks again; the transcript drops it too."""
    bus = EventBus()
    app = make_chrys_app(tmp_path, engine=SessionGenerationEngine(), event_bus=bus)
    async with app.run_test(size=(120, 36)) as pilot:
        main = app._main_screen
        assert main is not None
        # The harness configures no model; what matters here is what an admitted Continue does.
        monkeypatch.setattr(main, "_model_unconfigured", lambda: False)
        panel = main.query_one(ChatPanel)
        main._set_agent_running(True)
        await bus.publish(_delta("seg-1", "Half an ans"), raise_handler_errors=True)
        await bus.publish(
            InvocationContentFinished(
                origin=_ORIGIN, session_id="s1", segment_id="seg-1", channel="answer", status="interrupted"
            ),
            raise_handler_errors=True,
        )
        await main._input_flow.publish_interrupt()
        status = panel.query_one(InterruptedMessage)
        await wait_for(lambda: _shown(panel) == ["Half an ans"], pilot=pilot, description="half answer shown")

        await click_when_settled(pilot, status.query_one(Button))
        await wait_for(
            lambda: main.agent_running_state and not panel.query(AgentMessage),
            pilot=pilot,
            description="continue admitted and the half answer removed",
        )
        assert panel.query_one(InterruptedMessage) is status

        await bus.publish(_delta("seg-2", "The whole answer"), raise_handler_errors=True)
        await wait_for(lambda: _shown(panel) == ["The whole answer"], pilot=pilot, description="answer asked again")
