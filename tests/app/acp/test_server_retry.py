# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
"""``session/retry`` — resuming a failed or interrupted turn over ACP.

An ACP retry is the existing TUI retry reached from another frontend: the same
``UserRetry`` event, the same engine admission. These tests pin the two things
the ACP surface adds — the eligibility refusal before anything runs, and the
response naming the turn that was resumed rather than a new one.
"""

from __future__ import annotations

import pytest
from acp import RequestError
from acp.helpers import text_block

from chrys.app.acp.server import ChrysAcpServer
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Error, InvocationMessage, TurnStarted
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.orchestration.session_host import Cancelled, EndTurn, Errored
from tests.app.acp._server_fakes import _FakeClient, _FakeEngine, _FakeHost, _FakeManager

_TURN_ID = "4" * 32


def _server(host: _FakeHost, *, client: _FakeClient | None = None) -> ChrysAcpServer:
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client if client is not None else _FakeClient())
    return server


def _retry_host(*, can_retry: bool, outcome: object) -> _FakeHost:
    """A host that answers a retry pass with one resumed turn."""
    engine = _FakeEngine()
    engine.can_retry = can_retry
    return _FakeHost(
        event_bus=EventBus(),
        engine=engine,
        retry_events=[
            TurnStarted(turn_id=_TURN_ID, turn_number=3, is_retry=True, session_id="s1"),
            InvocationMessage(
                text="answer",
                is_final=True,
                session_id="s1",
                origin=InvocationOrigin("turn", "s1", _TURN_ID, None),
            ),
        ],
        outcome=outcome,
    )


@pytest.mark.anyio
async def test_session_retry_resumes_the_same_turn_and_labels_it_a_retry() -> None:
    host = _retry_host(can_retry=True, outcome=EndTurn("answer"))
    client = _FakeClient()

    result = await _server(host, client=client).ext_method("session/retry", {"sessionId": "s1", "text": "try again"})

    assert host.retry_texts == ["try again"]
    turn_started = [params for method, params in client.ext_notifications if method == "chrys/turn_started"]
    assert len(turn_started) == 1
    assert turn_started[0]["isRetry"] is True
    assert turn_started[0]["turnId"] == _TURN_ID
    assert result["stopReason"] == "end_turn"
    assert result["_meta"]["chrys"]["turnId"] == _TURN_ID
    assert result["_meta"]["chrys"]["turnNumber"] == 3


@pytest.mark.anyio
async def test_session_retry_carries_no_user_message_id() -> None:
    """A retry opens no new user message, so it must not acknowledge one."""
    host = _retry_host(can_retry=True, outcome=EndTurn("answer"))

    result = await _server(host).ext_method("session/retry", {"sessionId": "s1", "text": "try again"})

    assert "userMessageId" not in result


@pytest.mark.anyio
async def test_session_retry_without_text_still_resumes_the_turn() -> None:
    host = _retry_host(can_retry=True, outcome=EndTurn("answer"))

    result = await _server(host).ext_method("session/retry", {"sessionId": "s1"})

    assert host.retry_texts == [None]
    assert result["_meta"]["chrys"]["turnId"] == _TURN_ID


@pytest.mark.anyio
async def test_session_retry_refuses_a_session_with_no_turn_to_resume() -> None:
    """The eligibility check runs before anything is published."""
    host = _retry_host(can_retry=False, outcome=EndTurn("answer"))

    with pytest.raises(RequestError) as exc_info:
        await _server(host).ext_method("session/retry", {"sessionId": "s1"})

    assert exc_info.value.data == {"details": "Session s1 has no turn that can be retried."}
    assert host.retry_texts == []


@pytest.mark.anyio
async def test_session_retry_refuses_a_session_with_a_turn_in_flight() -> None:
    """A retry arriving mid-turn is refused, never queued behind the running turn.

    Queued, it would wait for the running turn to finish and then start a paid
    retry pass long after the client gave up on the request. The session admits
    one turn at a time, and an admitted turn counts from the moment it is
    admitted — not from the moment its work acquires the lock.
    """
    host = _retry_host(can_retry=True, outcome=EndTurn("answer"))
    server = _server(host)
    session = server._manager.get("s1")

    with pytest.raises(RequestError) as exc_info:
        async with session.prompt_lock:
            _ = session.reserve_turn()
            try:
                await server.ext_method("session/retry", {"sessionId": "s1"})
            finally:
                session.release_turn()

    assert exc_info.value.code == -32602
    assert exc_info.value.data == {"details": "Session s1 is running a turn; cancel it before retrying."}
    assert host.retry_texts == []


@pytest.mark.anyio
async def test_session_retry_and_second_prompt_refuse_each_other_while_a_turn_is_admitted() -> None:
    """One turn per session: the second caller is told no, not made to wait.

    Both requests admit a turn before they touch the lock, so whichever arrives
    second fails immediately — including in the window where the first turn has
    the lock but the second has not yet asked for it.
    """
    host = _retry_host(can_retry=True, outcome=EndTurn("answer"))
    server = _server(host)
    session = server._manager.get("s1")

    async with session.prompt_lock:
        _ = session.reserve_turn()
        try:
            with pytest.raises(RequestError) as exc_info:
                await server.prompt([text_block("second")], session_id="s1")
        finally:
            session.release_turn()

    assert exc_info.value.code == -32602
    assert exc_info.value.data == {"details": "Session s1 is running a turn."}
    assert host.retry_texts == []


@pytest.mark.anyio
async def test_session_retry_reports_a_cancelled_retry_as_cancelled() -> None:
    host = _retry_host(can_retry=True, outcome=Cancelled(reason="stopped"))

    result = await _server(host).ext_method("session/retry", {"sessionId": "s1"})

    assert result["stopReason"] == "cancelled"
    assert result["_meta"]["chrys"]["turnId"] == _TURN_ID


@pytest.mark.anyio
async def test_session_retry_maps_a_failed_retry_to_a_request_error_with_its_turn() -> None:
    host = _retry_host(
        can_retry=True,
        outcome=Errored(error=Error(code="model_failed", message="provider stopped")),
    )

    with pytest.raises(RequestError) as exc_info:
        await _server(host).ext_method("session/retry", {"sessionId": "s1"})

    assert exc_info.value.data["code"] == "model_failed"
    assert exc_info.value.data["turnId"] == _TURN_ID
