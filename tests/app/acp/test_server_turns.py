# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
"""``session/turns`` — the ACP surface over the turn/request projection.

The projection itself is covered by ``tests/service/analytics/test_turn_trace.py``;
these tests pin how the method reads a session's registry and log, what it
answers for an unknown turn, and how it reports a session whose trajectory
never activated.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from acp import RequestError

from chrys.app.acp.server import ChrysAcpServer
from chrys.app.acp.session_manager import AcpSessionError
from chrys.foundation.events.bus import EventBus
from chrys.service.trajectory.state import TurnRecord
from tests.app.acp._server_fakes import _FakeHost, _FakeManager
from tests.service.analytics._events import EventLog

_NS = 1_000_000_000
_TURN_A = "a" * 32
_TURN_B = "b" * 32
_CALL_A = "1" * 32


def _write_turn_log(session_dir: Path) -> None:
    """Write one real turn with one model call under ``session_dir``."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, turn_id=_TURN_A, payload={"turn_number": 1, "logical_turn_id": _TURN_A})
    log.add("model.exchange.started", 0, turn_id=_TURN_A, operation_id=_CALL_A)
    log.add(
        "model.exchange.finished",
        _NS,
        turn_id=_TURN_A,
        operation_id=_CALL_A,
        payload={"outcome": "success", "duration_ms": 1, "usage": {"normalized": {"input_total": 10}}},
    )
    log.add("turn.finished", _NS, turn_id=_TURN_A, payload={"end_reason": "completed", "duration_ms": 0})
    path = session_dir / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    log.write(path)


def _server(manager: _FakeManager) -> ChrysAcpServer:
    return ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_session_turns_projects_the_log_and_merges_the_registry(tmp_path: Path) -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    manager.turn_context_dir = tmp_path
    manager.turn_records = [TurnRecord(turn_number=1, turn_id=_TURN_A, user_message_id="msg-1")]
    _write_turn_log(tmp_path)

    result = await _server(manager).ext_method("session/turns", {"sessionId": "s1", "cwd": "/tmp/project"})

    assert manager.turn_context_reads == [("/tmp/project", "s1")]
    assert result["sessionId"] == "s1"
    assert result["coverage"] == "exact"
    assert len(result["turns"]) == 1
    turn = result["turns"][0]
    assert turn["turnId"] == _TURN_A
    assert turn["turnNumber"] == 1
    assert turn["userMessageId"] == "msg-1"
    assert turn["inheritedFrom"] is None
    assert turn["traceAvailable"] is True
    assert turn["usage"] == {"inputTokens": 10}
    assert turn["modelCallCount"] == 1
    assert turn["requestAttemptCount"] == 0


@pytest.mark.anyio
async def test_session_turns_reports_a_registry_turn_the_log_never_ran(tmp_path: Path) -> None:
    """A fork-inherited turn is named with no local trace, not omitted."""
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    manager.turn_context_dir = tmp_path
    manager.turn_records = [TurnRecord(turn_number=1, turn_id=_TURN_B, origin_session_id="origin-session")]

    result = await _server(manager).ext_method("session/turns", {"sessionId": "s1"})

    assert result["coverage"] == "partial"
    assert [turn["turnId"] for turn in result["turns"]] == [_TURN_B]
    turn = result["turns"][0]
    assert turn["turnNumber"] == 1
    assert turn["inheritedFrom"] == "origin-session"
    assert turn["traceAvailable"] is False
    assert turn["modelCalls"] == []


@pytest.mark.anyio
async def test_session_turns_filters_to_one_turn_and_rejects_an_unknown_one(tmp_path: Path) -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    manager.turn_context_dir = tmp_path
    _write_turn_log(tmp_path)
    server = _server(manager)

    filtered = await server.ext_method("session/turns", {"sessionId": "s1", "turnId": _TURN_A})
    assert [turn["turnId"] for turn in filtered["turns"]] == [_TURN_A]

    with pytest.raises(RequestError) as exc_info:
        await server.ext_method("session/turns", {"sessionId": "s1", "turnId": _TURN_B})

    assert exc_info.value.data == {"details": f"No turn {_TURN_B!r} in session s1."}


@pytest.mark.anyio
async def test_session_turns_refuses_a_session_outside_the_requesting_workspace(tmp_path: Path) -> None:
    """The cwd scope is ``session/history``'s: another workspace's session is refused."""
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    manager.turn_context_dir = tmp_path
    _write_turn_log(tmp_path)

    async def _refuse(*, cwd: str | None, session_id: str) -> tuple[str, Path, list[TurnRecord]]:
        raise AcpSessionError(f"session {session_id} belongs to another workspace")

    manager.session_turn_context = _refuse  # type: ignore[method-assign]

    with pytest.raises(RequestError) as exc_info:
        await _server(manager).ext_method("session/turns", {"sessionId": "s1", "cwd": "/tmp/other"})

    assert exc_info.value.data == {"details": "session s1 belongs to another workspace"}


@pytest.mark.anyio
async def test_session_turns_reports_a_session_with_no_trajectory_log(tmp_path: Path) -> None:
    """Recording never activated: the registry is all there is, and it is reported."""
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    manager.turn_context_dir = tmp_path
    manager.turn_records = [TurnRecord(turn_number=3, turn_id=_TURN_B)]

    result = await _server(manager).ext_method("session/turns", {"sessionId": "s1"})

    assert result["coverage"] == "partial"
    assert [turn["turnId"] for turn in result["turns"]] == [_TURN_B]
    assert result["turns"][0]["traceAvailable"] is False
    assert result["sessionLevel"]["modelCallCount"] == 0
