# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chunks of one content segment share its ``messageId``; the next segment starts a new one."""

from __future__ import annotations

from typing import Any

import pytest
from acp.helpers import text_block

from chrys.app.acp.server import ChrysAcpServer
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    InvocationContentDelta,
    InvocationContentFinished,
    InvocationMessage,
    InvocationRetryAttempt,
    InvocationToolCallStart,
    TurnStarted,
)
from chrys.foundation.models.invocations import ContentChannel, ContentStatus, InvocationOrigin
from chrys.orchestration.session_host import Cancelled
from tests.app.acp._server_fakes import _FakeClient, _FakeHost, _FakeManager


def _origin() -> InvocationOrigin:
    return InvocationOrigin("turn", "s1", "turn-a", None)


def _turn_started() -> TurnStarted:
    return TurnStarted(session_id="s1", turn_id="turn-a", turn_number=1)


def _delta(segment_id: str, text: str, channel: ContentChannel = "answer") -> InvocationContentDelta:
    return InvocationContentDelta(origin=_origin(), session_id="s1", segment_id=segment_id, channel=channel, text=text)


def _finished(
    segment_id: str, channel: ContentChannel = "answer", status: ContentStatus = "completed"
) -> InvocationContentFinished:
    return InvocationContentFinished(
        origin=_origin(), session_id="s1", segment_id=segment_id, channel=channel, status=status
    )


def _server(events: list[Any]) -> tuple[ChrysAcpServer, _FakeClient]:
    host = _FakeHost(event_bus=EventBus(), events=events, outcome=Cancelled())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    return server, client


def _chunks(client: _FakeClient) -> list[tuple[str, str, str]]:
    return [
        (update.update.session_update, update.update.content.text, update.update.message_id)
        for update in client.updates
        if update.update.session_update in ("agent_message_chunk", "agent_thought_chunk")
    ]


@pytest.mark.anyio
async def test_each_streamed_segment_is_one_message_and_the_final_resends_nothing() -> None:
    """ACP groups chunks by ``messageId``: one segment, one id; a tool call starts the next segment."""
    server, client = _server(
        [
            _turn_started(),
            _delta("think", "Looking", "reasoning"),
            _finished("think", "reasoning"),
            _delta("intro", "Hel"),
            _delta("intro", "lo"),
            _finished("intro"),
            InvocationToolCallStart(tool_name="read_file", call_id="c1", origin=_origin()),
            _delta("answer", "Done"),
            _finished("answer"),
            InvocationMessage(text="Done", origin=_origin(), is_final=True),
        ]
    )

    await server.prompt([text_block("go")], session_id="s1")

    thought, first, second, answer = _chunks(client)
    assert [(kind, text) for kind, text, _id in (thought, first, second, answer)] == [
        ("agent_thought_chunk", "Looking"),
        ("agent_message_chunk", "Hel"),
        ("agent_message_chunk", "lo"),
        ("agent_message_chunk", "Done"),
    ]
    assert first[2] == second[2]
    assert len({thought[2], first[2], answer[2]}) == 3
    assert [
        update.update.field_meta["chrys"]
        for update in client.updates
        if update.update.session_update.startswith("agent_")
    ] == [
        {"segmentId": "think", "channel": "reasoning", "turnId": "turn-a", "turnNumber": 1},
        {"segmentId": "intro", "channel": "answer", "turnId": "turn-a", "turnNumber": 1},
        {"segmentId": "intro", "channel": "answer", "turnId": "turn-a", "turnNumber": 1},
        {"segmentId": "answer", "channel": "answer", "turnId": "turn-a", "turnNumber": 1},
    ]


@pytest.mark.anyio
async def test_text_before_a_retry_stays_and_the_next_text_is_a_new_message() -> None:
    """Shown text is never withdrawn: a failed segment ends, and what follows is a new message."""
    server, client = _server(
        [
            _turn_started(),
            _delta("draft", "draft"),
            _finished("draft", status="failed"),
            InvocationRetryAttempt(
                origin=_origin(), message="retrying", attempt=2, max_attempts=10, delay_seconds=3, scope="run"
            ),
            _delta("final", "final"),
            _finished("final"),
            InvocationMessage(text="final", origin=_origin(), is_final=True),
        ]
    )

    await server.prompt([text_block("go")], session_id="s1")

    (_kind, draft_text, draft_id), (_kind2, final_text, final_id) = _chunks(client)
    assert (draft_text, final_text) == ("draft", "final")
    assert draft_id != final_id
    assert [method for method, _params in client.ext_notifications] == ["chrys/turn_started", "chrys/retry_attempt"]
