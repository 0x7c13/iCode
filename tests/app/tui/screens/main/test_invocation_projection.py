# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Nested prose and provisional routing never terminalize the main turn."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chrys.app.tui.screens.main.state import MainScreenState, RunState
from chrys.foundation.events.types import (
    InvocationMessage,
    InvocationPresentationAttemptAccepted,
    InvocationPresentationAttemptRejected,
)
from chrys.foundation.models.invocations import InvocationOrigin
from tests.support.invocation_events import ORIGIN_IDS, ORIGINS, has_chat_card, projection_event
from tests.support.tui_helpers import make_backend_handler


@pytest.mark.parametrize("origin", ORIGINS[1:], ids=ORIGIN_IDS[1:])
@pytest.mark.parametrize("phase", ("intermediate", "final", "provisional", "accepted", "rejected"))
async def test_nested_text_and_retractions_stay_on_the_owning_card(origin: InvocationOrigin, phase: str) -> None:
    calls = []

    class Panel:
        def add_sub_agent_message(self, agent_name, invocation_id, text, *, presentation=None):
            calls.append(("message", invocation_id, text, presentation))

        def accept_sub_agent_presentation(self, invocation_id, attempt_id, segment_ids):
            calls.append(("accepted", invocation_id, attempt_id, segment_ids))

        def reject_sub_agent_presentation(self, invocation_id, attempt_id):
            calls.append(("rejected", invocation_id, attempt_id))

    panel = Panel()
    handler = make_backend_handler(
        SimpleNamespace(_state=MainScreenState(run=RunState(agent_running=True)), query_one=lambda _: panel)
    )
    event = projection_event(origin, phase)
    if isinstance(event, InvocationMessage):
        await handler.on_agent_message(event)
    elif isinstance(event, InvocationPresentationAttemptAccepted):
        await handler.on_presentation_attempt_accepted(event)
    else:
        assert isinstance(event, InvocationPresentationAttemptRejected)
        await handler.on_presentation_attempt_rejected(event)
    assert handler.agent_running is True
    assert len(calls) == (1 if has_chat_card(origin) else 0)
    if calls:
        assert calls[0][1] == origin.invocation_id
        assert calls[0][0] == (phase if phase in ("accepted", "rejected") else "message")
