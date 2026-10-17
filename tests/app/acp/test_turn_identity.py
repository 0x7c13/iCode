# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Turn identity on the ACP surface: ``_meta.chrys``, ``messageId``, retry, retraction."""

from __future__ import annotations

from typing import Any

import pytest
from acp.helpers import text_block

from chrys.app.acp.server import ChrysAcpServer
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    InvocationMessage,
    InvocationPresentationAttemptAccepted,
    InvocationPresentationAttemptRejected,
    InvocationRetryAttempt,
    ProvisionalPresentation,
    TurnStarted,
)
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from chrys.orchestration.session_host import Cancelled
from chrys.service.state.store import SessionTranscript
from chrys.service.trajectory.state import TurnRecord
from tests.app.acp._server_fakes import _FakeClient, _FakeHost, _FakeManager

_CLIENT_MESSAGE_ID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
_NO_USAGE = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}


_ITEM_ID = "item-1"


class _SeededStore:
    """A state store whose single session carries the given turn registry."""

    def __init__(self, turns: list[TurnRecord]) -> None:
        self._turns = turns

    async def load_session_transcript(self, session_id: str, *, prefer_recovery: bool = False) -> SessionTranscript:
        _ = prefer_recovery
        return SessionTranscript(
            messages=[
                {
                    "role": "user",
                    "contents": [{"type": "text", "text": f"hello {session_id}"}],
                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: _ITEM_ID},
                }
            ],
            turns=list(self._turns),
        )


def _origin() -> InvocationOrigin:
    return InvocationOrigin("turn", "s1", "turn-a", None)


def _turn_started(**overrides: Any) -> TurnStarted:
    fields: dict[str, Any] = {"session_id": "s1", "turn_id": "turn-a", "turn_number": 1}
    fields.update(overrides)
    return TurnStarted(**fields)


def _server(events: list[Any], *, turns: list[TurnRecord] | None = None) -> tuple[ChrysAcpServer, _FakeClient]:
    host = _FakeHost(event_bus=EventBus(), events=events, outcome=Cancelled())
    manager = _FakeManager(host)
    if turns is not None:
        manager.state_store = _SeededStore(turns)  # type: ignore[assignment]
    client = _FakeClient()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    return server, client


def _retractions(client: _FakeClient) -> list[dict[str, Any]]:
    return [params for method, params in client.ext_notifications if method == "chrys/messages_retracted"]


def _retries(client: _FakeClient) -> list[dict[str, Any]]:
    return [params for method, params in client.ext_notifications if method == "chrys/retry_attempt"]


@pytest.mark.anyio
async def test_one_turn_id_reaches_the_response_live_and_history() -> None:
    """The client joins its own transcript on one id: the same value everywhere."""
    server, client = _server(
        [
            _turn_started(user_message_id=_CLIENT_MESSAGE_ID),
            InvocationMessage(text="answer", origin=_origin()),
        ],
        turns=[
            TurnRecord(turn_number=1, turn_id="turn-a", opening_item_id=_ITEM_ID, user_message_id=_CLIENT_MESSAGE_ID)
        ],
    )

    response = await server.prompt([text_block("go")], session_id="s1", message_id=_CLIENT_MESSAGE_ID)

    assert response.field_meta == {"chrys": {"turnId": "turn-a", "turnNumber": 1, "sessionUsage": _NO_USAGE}}
    assert client.ext_notifications == [
        (
            "chrys/turn_started",
            {
                "sessionId": "s1",
                "turnId": "turn-a",
                "turnNumber": 1,
                "isRetry": False,
                "userMessageId": _CLIENT_MESSAGE_ID,
            },
        )
    ]
    assert client.updates[0].update.field_meta == {"chrys": {"turnId": "turn-a", "turnNumber": 1}}

    history = await server.ext_method("session/history", {"sessionId": "s1"})
    assert history["turns"] == [{"turnId": "turn-a", "turnNumber": 1, "userMessageId": _CLIENT_MESSAGE_ID}]


@pytest.mark.anyio
async def test_a_run_with_no_turn_started_reports_no_turn_id() -> None:
    """Nothing opened a turn, so no id is invented and no notification claims one."""
    server, client = _server([])

    response = await server.prompt([text_block("go")], session_id="s1")

    # No turn opened, so no id is invented; the request still made no spend.
    assert response.field_meta == {"chrys": {"sessionUsage": _NO_USAGE}}
    assert client.ext_notifications == []


@pytest.mark.anyio
async def test_a_rejected_attempt_retracts_both_segments_and_the_answer_is_a_new_message() -> None:
    server, client = _server(
        [
            _turn_started(),
            InvocationMessage(
                text="first", origin=_origin(), presentation=ProvisionalPresentation("attempt-1", "seg-1")
            ),
            InvocationMessage(
                text="second", origin=_origin(), presentation=ProvisionalPresentation("attempt-1", "seg-2")
            ),
            InvocationPresentationAttemptRejected(origin=_origin(), attempt_id="attempt-1"),
            InvocationRetryAttempt(
                origin=_origin(), message="retrying", attempt=2, max_attempts=10, delay_seconds=3, scope="wire"
            ),
            InvocationMessage(text="final", origin=_origin()),
        ]
    )

    await server.prompt([text_block("go")], session_id="s1")

    chunk_ids = [update.update.message_id for update in client.updates]
    assert _retractions(client) == [{"sessionId": "s1", "turnId": "turn-a", "messageIds": [chunk_ids[0], chunk_ids[1]]}]
    # The final answer is a fresh message, not a continuation of the retracted ones.
    assert chunk_ids[2] not in _retractions(client)[0]["messageIds"]
    assert _retries(client) == [
        {
            "sessionId": "s1",
            "turnId": "turn-a",
            "attempt": 2,
            "maxAttempts": 10,
            "delaySeconds": 3,
            "scope": "wire",
            "message": "retrying",
        }
    ]


@pytest.mark.anyio
async def test_partial_accept_only_retracts_the_segments_the_backend_dropped() -> None:
    server, client = _server(
        [
            _turn_started(),
            InvocationMessage(
                text="keep", origin=_origin(), presentation=ProvisionalPresentation("attempt-1", "seg-1")
            ),
            InvocationMessage(
                text="drop", origin=_origin(), presentation=ProvisionalPresentation("attempt-1", "seg-2")
            ),
            InvocationPresentationAttemptAccepted(origin=_origin(), attempt_id="attempt-1", segment_ids=("seg-1",)),
        ]
    )

    await server.prompt([text_block("go")], session_id="s1")

    kept_id, dropped_id = (update.update.message_id for update in client.updates)
    assert _retractions(client) == [{"sessionId": "s1", "turnId": "turn-a", "messageIds": [dropped_id]}]
    assert kept_id not in _retractions(client)[0]["messageIds"]


@pytest.mark.anyio
async def test_an_attempt_with_nothing_dropped_is_not_a_retraction() -> None:
    server, client = _server(
        [
            _turn_started(),
            InvocationMessage(text="only", origin=_origin()),
            InvocationPresentationAttemptAccepted(origin=_origin(), attempt_id="attempt-1", segment_ids=()),
        ]
    )

    await server.prompt([text_block("go")], session_id="s1")

    assert _retractions(client) == []


@pytest.mark.anyio
async def test_a_sub_agent_retry_is_not_forwarded_as_the_turns_retry() -> None:
    """Only the main turn's retries carry the turn id; a child names its own invocation."""
    server, client = _server(
        [
            _turn_started(),
            InvocationRetryAttempt(
                origin=InvocationOrigin("sub_agent", "s1", "sub-1", None),
                message="retrying",
                attempt=1,
                max_attempts=3,
                scope="run",
            ),
        ]
    )

    await server.prompt([text_block("go")], session_id="s1")

    assert _retries(client) == []


@pytest.mark.anyio
async def test_a_compaction_retry_is_forwarded_with_its_scope() -> None:
    """Compaction retries ride the same notification; the scope is what tells them apart."""
    server, client = _server(
        [
            _turn_started(),
            InvocationRetryAttempt(
                origin=_origin(), message="compacting", attempt=1, max_attempts=10, scope="compaction"
            ),
        ]
    )

    await server.prompt([text_block("go")], session_id="s1")

    assert [params["scope"] for params in _retries(client)] == ["compaction"]
