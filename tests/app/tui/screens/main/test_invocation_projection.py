# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Nested prose never terminalizes the main turn."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chrys.app.tui.screens.main.state import MainScreenState, RunState
from chrys.foundation.events.types import (
    InvocationContentDelta,
    InvocationContentFinished,
    InvocationMessage,
)
from chrys.foundation.models.invocations import ContentStatus, InvocationOrigin
from tests.support.invocation_events import ORIGIN_IDS, ORIGINS, PROSE_PHASES, has_chat_card, projection_event
from tests.support.tui_helpers import make_backend_handler


@pytest.mark.parametrize("origin", ORIGINS[1:], ids=ORIGIN_IDS[1:])
@pytest.mark.parametrize("phase", sorted(PROSE_PHASES))
async def test_nested_text_stays_on_the_owning_card(origin: InvocationOrigin, phase: str) -> None:
    calls = []

    class Panel:
        def add_sub_agent_answer(self, agent_name, invocation_id, segment_id, text):
            calls.append((invocation_id, segment_id, text))

    handler = make_backend_handler(
        SimpleNamespace(_state=MainScreenState(run=RunState(agent_running=True)), query_one=lambda _: Panel())
    )
    event = projection_event(origin, phase)
    if isinstance(event, InvocationMessage):
        await handler.on_agent_message(event)
    else:
        await handler.on_agent_content(event)
        if phase == "answer":
            # The card takes an answer segment whole, once it ends.
            assert calls == []
            await handler.on_agent_content(projection_event(origin, "segment_end"))
    assert handler.agent_running is True
    # Only a chat child's answer text reaches its card; its final result arrives with the tool result.
    expected = phase == "answer" and has_chat_card(origin)
    assert calls == ([(origin.invocation_id, "segment", "private child answer")] if expected else [])


@pytest.mark.parametrize("status", ["interrupted", "completed"])
async def test_a_child_answer_that_ends_after_its_run_stopped_shows_nothing_and_is_released(
    status: ContentStatus,
) -> None:
    calls: list[str] = []

    class Panel:
        def add_sub_agent_answer(self, agent_name: str, invocation_id: str, segment_id: str, text: str) -> None:
            calls.append(text)

    state = MainScreenState(run=RunState(agent_running=True))
    handler = make_backend_handler(SimpleNamespace(_state=state, query_one=lambda _: Panel()))
    origin = ORIGINS[1]
    await handler.on_agent_content(InvocationContentDelta(origin=origin, segment_id="s1", text="Half an"))
    # An interrupt stops the run before the child's cancellation ends its segment.
    state.run.agent_running = False
    await handler.on_agent_content(InvocationContentFinished(origin=origin, segment_id="s1", status=status))

    assert calls == []
    # The segment's end releases its text.
    assert handler._sub_agent_answers._open == {}
