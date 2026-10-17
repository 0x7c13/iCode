# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chunks of one agent message share its ``messageId``; a boundary starts a new one."""

from __future__ import annotations

from typing import Any

import pytest
from acp.helpers import text_block

from chrys.app.acp.server import ChrysAcpServer
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    InvocationMessage,
    InvocationPresentationAttemptRejected,
    InvocationRetryAttempt,
    InvocationToolCallStart,
    ProvisionalPresentation,
    TurnStarted,
)
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.orchestration.session_host import Cancelled
from tests.app.acp._server_fakes import _FakeClient, _FakeHost, _FakeManager


def _origin() -> InvocationOrigin:
    return InvocationOrigin("turn", "s1", "turn-a", None)


def _turn_started() -> TurnStarted:
    return TurnStarted(session_id="s1", turn_id="turn-a", turn_number=1)


def _server(events: list[Any]) -> tuple[ChrysAcpServer, _FakeClient]:
    host = _FakeHost(event_bus=EventBus(), events=events, outcome=Cancelled())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    return server, client


def _chunks(client: _FakeClient) -> list[str]:
    return [
        update.update.message_id for update in client.updates if update.update.session_update == "agent_message_chunk"
    ]


def _retractions(client: _FakeClient) -> list[list[str]]:
    return [params["messageIds"] for method, params in client.ext_notifications if method == "chrys/messages_retracted"]


@pytest.mark.anyio
async def test_a_streamed_answer_keeps_one_message_id_and_restarts_after_a_tool_call() -> None:
    """ACP groups chunks by ``messageId``: same answer, same id; a tool card splits it."""
    server, client = _server(
        [
            _turn_started(),
            InvocationMessage(text="Hel", origin=_origin()),
            InvocationMessage(text="Hello", origin=_origin(), is_final=True),
            InvocationToolCallStart(tool_name="read_file", call_id="c1", origin=_origin()),
            InvocationMessage(text="Done", origin=_origin(), is_final=True),
        ]
    )

    await server.prompt([text_block("go")], session_id="s1")

    first, second, third = _chunks(client)
    assert first == second
    assert third != second


@pytest.mark.anyio
async def test_a_segment_streamed_in_several_chunks_is_retracted_by_every_id_it_carried() -> None:
    """A withdrawn segment leaves nothing behind: every id it sent is named."""
    server, client = _server(
        [
            _turn_started(),
            InvocationMessage(
                text="part", origin=_origin(), presentation=ProvisionalPresentation("attempt-1", "seg-1")
            ),
            InvocationMessage(
                text="part one", origin=_origin(), presentation=ProvisionalPresentation("attempt-1", "seg-1")
            ),
            InvocationPresentationAttemptRejected(origin=_origin(), attempt_id="attempt-1"),
        ]
    )

    await server.prompt([text_block("go")], session_id="s1")

    sent = _chunks(client)
    assert len(sent) == 2
    assert sent[0] == sent[1]
    assert _retractions(client) == [[sent[0]]]


@pytest.mark.anyio
async def test_text_after_a_rejected_attempt_is_a_new_message() -> None:
    """The final answer never continues the message a retraction withdrew."""
    server, client = _server(
        [
            _turn_started(),
            InvocationMessage(
                text="draft", origin=_origin(), presentation=ProvisionalPresentation("attempt-1", "seg-1")
            ),
            InvocationPresentationAttemptRejected(origin=_origin(), attempt_id="attempt-1"),
            InvocationRetryAttempt(
                origin=_origin(), message="retrying", attempt=2, max_attempts=10, delay_seconds=3, scope="wire"
            ),
            InvocationMessage(text="final", origin=_origin(), is_final=True),
        ]
    )

    await server.prompt([text_block("go")], session_id="s1")

    draft_id, final_id = _chunks(client)
    assert _retractions(client) == [[draft_id]]
    assert final_id not in _retractions(client)[0]
