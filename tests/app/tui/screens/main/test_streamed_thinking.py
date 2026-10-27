# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A main turn's thinking shows as it streams, folded by default; the setting decides it for each new segment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chrys.app.tui.widgets.chat.messages import AgentMessage, ThinkingMessage
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    InvocationContentAbandoned,
    InvocationContentDelta,
    InvocationContentFinished,
    InvocationMessage,
)
from chrys.foundation.models.invocations import InvocationOrigin
from tests.support.tui_app_harness import SessionGenerationEngine, make_chrys_app
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from pathlib import Path

    from chrys.foundation.models.invocations import ContentChannel

_ORIGIN = InvocationOrigin("turn", "s1", "turn-1", None)


def _delta(segment_id: str, text: str, channel: ContentChannel = "reasoning") -> InvocationContentDelta:
    return InvocationContentDelta(origin=_ORIGIN, session_id="s1", segment_id=segment_id, channel=channel, text=text)


def _finished(segment_id: str, channel: ContentChannel = "reasoning") -> InvocationContentFinished:
    return InvocationContentFinished(
        origin=_ORIGIN, session_id="s1", segment_id=segment_id, channel=channel, status="completed"
    )


def _thinking(panel: ChatPanel) -> list[str]:
    return [message.text for message in panel.query(ThinkingMessage)]


async def test_thinking_streams_folded_above_the_answer_by_default_and_a_setting_change_applies_to_the_next_segment(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    app = make_chrys_app(tmp_path, engine=SessionGenerationEngine(), event_bus=bus)
    async with app.run_test(size=(120, 36)) as pilot:
        main = app._main_screen
        assert main is not None
        panel = main.query_one(ChatPanel)
        main._set_agent_running(True)

        for event in (_delta("r1", "Looking "), _delta("r1", "closer.")):
            await bus.publish(event, raise_handler_errors=True)
        await wait_for(lambda: _thinking(panel) == ["Looking closer."], pilot=pilot, description="thinking mounted")
        (first,) = panel.query(ThinkingMessage)
        assert first.collapsed is True

        for event in (
            _finished("r1"),
            _delta("a1", "Answer.", channel="answer"),
            _finished("a1", channel="answer"),
        ):
            await bus.publish(event, raise_handler_errors=True)
        await wait_for(lambda: len(panel.query(AgentMessage)) == 1, pilot=pilot, description="answer mounted")
        (answer,) = panel.query(AgentMessage)
        assert panel.children.index(first) < panel.children.index(answer)

        main._apply_thinking_expanded(True)
        assert app.settings_handle.settings.chat_thinking_expanded is True
        await bus.publish(_delta("r2", "Shown open."), raise_handler_errors=True)
        await wait_for(lambda: len(_thinking(panel)) == 2, pilot=pilot, description="second thinking mounted")
        second = panel.query(ThinkingMessage).last()
        assert (first.collapsed, second.collapsed) == (True, False)

        await bus.publish(_finished("r2"), raise_handler_errors=True)
        await bus.publish(
            InvocationMessage(origin=_ORIGIN, session_id="s1", text="Answer.", is_final=True),
            raise_handler_errors=True,
        )
        await wait_for(lambda: not main.agent_running_state, pilot=pilot, description="turn completed")
        assert _thinking(panel) == ["Looking closer.", "Shown open."]


async def test_a_retry_that_gives_up_thinking_while_the_prompt_mounts_removes_it_in_order(tmp_path: Path) -> None:
    """Events that wait for the user bubble replay in arrival order, so the removal follows the text it removes."""
    bus = EventBus()
    app = make_chrys_app(tmp_path, engine=SessionGenerationEngine(), event_bus=bus)
    async with app.run_test(size=(120, 36)):
        main = app._main_screen
        assert main is not None
        panel = main.query_one(ChatPanel)
        main._set_agent_running(True)
        main._state.render_gate.begin()

        for event in (
            _delta("r1", "Given up."),
            InvocationContentAbandoned(origin=_ORIGIN, session_id="s1", segment_ids=("r1",)),
            _delta("r2", "Thinking again."),
            _finished("r2"),
        ):
            await bus.publish(event, raise_handler_errors=True)
        assert _thinking(panel) == []

        await main._input_flow.flush_deferred_agent_messages()

        assert _thinking(panel) == ["Thinking again."]
