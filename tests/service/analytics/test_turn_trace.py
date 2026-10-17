# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
"""Synthetic ``turn_trace`` projection tests.

Sequence shapes the real engine no longer writes — legacy logs without
``logical_turn_id``, reused ordinals, damaged records — are built through the
real writer and read back through the real reader, so the projection's
compatibility rules are pinned without a live engine.

Which events a live retry actually records needs a real engine and lives in
``tests/integration/engine/test_turn_trace_projection.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.trajectory.event_types import EventType
from chrys.service.analytics.turn_trace import scan_turn_trace, trace_payload_with_registry
from chrys.service.trajectory.state import TurnRecord, turn_records
from tests.service.analytics._events import SESSION_ID, EventLog

_NS = 1_000_000_000
_TURN_A = "a" * 32
_TURN_B = "b" * 32
_TURN_C = "c" * 32
_CALL_A = "1" * 32
_CALL_B = "2" * 32
_CALL_C = "3" * 32
"""A second call in the same pass: every wire exchange mints its own id."""
_CALL_D = "4" * 32
_RETRY_1 = "5" * 32
_ATTEMPT_1 = "6" * 32
"""A wire request attempt: ``*_id`` fields default to the 32-hex analytics class."""


def _turn_started(
    log: EventLog, pass_id: str, ns: int, *, logical: str | None = None, number: int = 1, is_retry: bool = False
) -> None:
    """One ``turn.started``; ``is_retry`` and no ``logical`` reproduce a legacy retry pass."""
    payload: dict[str, object] = {"turn_number": number, "is_retry": is_retry}
    if logical is not None:
        payload["logical_turn_id"] = logical
    log.add("turn.started", ns, turn_id=pass_id, payload=payload)


def _turn_finished(log: EventLog, pass_id: str, ns: int, *, end_reason: str = "completed") -> None:
    log.add("turn.finished", ns, turn_id=pass_id, payload={"end_reason": end_reason, "duration_ms": 0})


def _exchange(
    log: EventLog,
    operation_id: str,
    start_ns: int,
    *,
    pass_id: str | None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    outcome: str = "success",
) -> None:
    log.add("model.exchange.started", start_ns, turn_id=pass_id, operation_id=operation_id)
    payload: dict[str, object] = {"outcome": outcome, "duration_ms": 1}
    if input_tokens is not None or output_tokens is not None:
        normalized: dict[str, int] = {}
        if input_tokens is not None:
            normalized["input_total"] = input_tokens
        if output_tokens is not None:
            normalized["output_total"] = output_tokens
        payload["usage"] = {"normalized": normalized}
    log.add("model.exchange.finished", start_ns + _NS, turn_id=pass_id, operation_id=operation_id, payload=payload)


def _scan(log: EventLog, tmp_path: Path, *, records: list[TurnRecord] | None = None) -> dict[str, Any]:
    """Write the log through the real writer and project it, optionally with a registry."""
    path = tmp_path / "events.jsonl"
    log.write(path)
    trace = scan_turn_trace(path)
    if records is None:
        return trace.to_payload()
    payload = trace_payload_with_registry(trace, records)
    assert payload is not None
    return payload


def _turns(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {turn["turnId"]: turn for turn in payload["turns"]}


# ------------------------------------------------------------------- grouping


def test_legacy_retry_pass_groups_with_the_turn_it_resumed(tmp_path: Path) -> None:
    """A retry recorded before ``logical_turn_id`` existed joins the turn it resumed.

    The writer has always marked the retry, and the pass id it minted is its
    own; the branch cursor is what says which turn the retry belongs to.
    """
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS, end_reason="error")
    _turn_started(log, _TURN_B, 2 * _NS, is_retry=True)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=_TURN_B, input_tokens=20)
    _turn_finished(log, _TURN_B, 3 * _NS)

    payload = _scan(log, tmp_path)
    turn = _turns(payload)[_TURN_A]

    assert [turn_["turnId"] for turn_ in payload["turns"]] == [_TURN_A]
    assert [pass_["passId"] for pass_ in turn["passes"]] == [_TURN_A, _TURN_B]
    assert turn["usage"] == {"inputTokens": 30}
    assert turn["status"] == "completed"
    assert payload["coverage"] == "exact"
    assert payload["unassignedPasses"] == []


def test_legacy_retry_after_a_targetless_rollback_stands_alone(tmp_path: Path) -> None:
    """A rollback that names no target leaves no cursor, so the retry pass is its own turn."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    log.resolved_rollback(2, 2 * _NS)
    _turn_started(log, _TURN_B, 3 * _NS, is_retry=True)
    _exchange(log, _CALL_B, 3 * _NS, pass_id=_TURN_B, input_tokens=20)

    payload = _scan(log, tmp_path)

    assert [turn["turnId"] for turn in payload["turns"]] == [_TURN_B]
    assert [turn["turnId"] for turn in payload["supersededTurns"]] == [_TURN_A]
    assert _turns(payload)[_TURN_B]["passes"][0]["passId"] == _TURN_B


def test_legacy_retry_after_a_targeted_rollback_groups_with_the_target(tmp_path: Path) -> None:
    """A rollback naming ``target_turn_id`` restores the cursor to that turn."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    rollback_from = log.next_sequence
    _turn_started(log, _TURN_B, 2 * _NS)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=_TURN_B, input_tokens=20)
    _turn_finished(log, _TURN_B, 3 * _NS)
    log.resolved_rollback(rollback_from, 4 * _NS, target_turn_id=_TURN_A)
    _turn_started(log, _TURN_B, 5 * _NS, is_retry=True)
    _exchange(log, _CALL_C, 5 * _NS, pass_id=_TURN_B, input_tokens=30)

    payload = _scan(log, tmp_path)

    assert [turn["turnId"] for turn in payload["turns"]] == [_TURN_A]
    assert [turn["turnId"] for turn in payload["supersededTurns"]] == [_TURN_B]
    assert _turns(payload)[_TURN_A]["usage"] == {"inputTokens": 40}
    assert [pass_["passId"] for pass_ in _turns(payload)[_TURN_A]["passes"]] == [_TURN_A, _TURN_B]


def test_unfinished_turn_is_reported_as_unfinished(tmp_path: Path) -> None:
    """A pass with no ``turn.finished`` — an interrupted process — keeps its calls."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)

    payload = _scan(log, tmp_path)

    assert _turns(payload)[_TURN_A]["status"] == "unfinished"


# ----------------------------------------------------------------- supersession


def test_rollback_supersedes_turns_inside_its_recorded_range(tmp_path: Path) -> None:
    """Only the turns whose first pass falls in the rolled-back range move out of ``turns``."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    rollback_from = log.next_sequence
    _turn_started(log, _TURN_B, 2 * _NS)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=_TURN_B, input_tokens=20)
    _turn_finished(log, _TURN_B, 3 * _NS)
    log.resolved_rollback(rollback_from, 4 * _NS)

    payload = _scan(log, tmp_path)

    assert [turn["turnId"] for turn in payload["turns"]] == [_TURN_A]
    assert [turn["turnId"] for turn in payload["supersededTurns"]] == [_TURN_B]
    assert payload["coverage"] == "exact"


def test_unsaved_turn_after_a_rollback_is_superseded_too(tmp_path: Path) -> None:
    """A rollback range can reach a turn the process never saved."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_finished(log, _TURN_A, _NS)
    rollback_from = log.next_sequence
    _turn_started(log, _TURN_B, 2 * _NS)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=_TURN_B, input_tokens=20)
    log.resolved_rollback(rollback_from, 3 * _NS)

    payload = _scan(log, tmp_path)

    assert [turn["turnId"] for turn in payload["supersededTurns"]] == [_TURN_B]
    assert payload["coverage"] == "exact"


def test_retried_turn_keeps_the_retry_pass_and_the_superseded_one(tmp_path: Path) -> None:
    """A turn reopened after a rollback carries both passes; only the discarded turn moves out."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    rollback_from = log.next_sequence
    _turn_started(log, _TURN_B, 2 * _NS)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=_TURN_B, input_tokens=20)
    _turn_finished(log, _TURN_B, 3 * _NS)
    log.resolved_rollback(rollback_from, 4 * _NS, target_turn_id=_TURN_A)
    _turn_started(log, _TURN_B, 5 * _NS, logical=_TURN_A, is_retry=True)
    _exchange(log, _CALL_C, 5 * _NS, pass_id=_TURN_B, input_tokens=30)
    _turn_finished(log, _TURN_B, 6 * _NS)

    payload = _scan(log, tmp_path)
    live = _turns(payload)[_TURN_A]

    assert [pass_["passId"] for pass_ in live["passes"]] == [_TURN_A, _TURN_B]
    assert live["usage"] == {"inputTokens": 40}
    assert [turn["turnId"] for turn in payload["supersededTurns"]] == [_TURN_B]


# -------------------------------------------------------------------- registry


def test_registry_fills_names_and_reports_turns_the_log_never_ran(tmp_path: Path) -> None:
    """The registry names a turn's opener and is the only record of an inherited turn."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    records = [
        TurnRecord(turn_number=1, turn_id=_TURN_A, user_message_id="msg-1"),
        TurnRecord(turn_number=2, turn_id=_TURN_B, origin_session_id="origin-session"),
    ]

    payload = _scan(log, tmp_path, records=records)
    turns = _turns(payload)

    assert turns[_TURN_A]["turnNumber"] == 1
    assert turns[_TURN_A]["userMessageId"] == "msg-1"
    assert turns[_TURN_A]["traceAvailable"] is True
    assert turns[_TURN_A]["inheritedFrom"] is None
    assert turns[_TURN_B]["turnNumber"] == 2
    assert turns[_TURN_B]["traceAvailable"] is False
    assert turns[_TURN_B]["inheritedFrom"] == "origin-session"
    assert turns[_TURN_B]["modelCalls"] == []
    assert payload["unassignedPasses"] == []


def test_registry_turns_are_merged_in_start_order_with_the_log_turns(tmp_path: Path) -> None:
    """``turns`` is in start order: a fork-inherited turn is not pushed to the end.

    The registry's turns are appended after the log's, so a naive merge reports
    ``[B, A]`` for a fork that inherited the first turn and then ran the second.
    """
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_B, 0, number=2)
    _exchange(log, _CALL_B, 0, pass_id=_TURN_B, input_tokens=20)
    _turn_finished(log, _TURN_B, _NS)
    records = [
        TurnRecord(turn_number=1, turn_id=_TURN_A, origin_session_id="origin-session"),
        TurnRecord(turn_number=2, turn_id=_TURN_B),
    ]

    payload = _scan(log, tmp_path, records=records)

    assert [turn["turnId"] for turn in payload["turns"]] == [_TURN_A, _TURN_B]
    assert payload["turns"][0]["traceAvailable"] is False


def test_inherited_turn_carries_no_local_usage(tmp_path: Path) -> None:
    """An inherited turn reports the copy's own execution only, never the source session's."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    records = [TurnRecord(turn_number=1, turn_id=_TURN_B, origin_session_id="origin-session")]

    payload = _scan(log, tmp_path, records=records)
    inherited = _turns(payload)[_TURN_B]

    assert inherited["usage"] == {}
    assert inherited["modelCallCount"] == 0
    assert inherited["requestAttemptCount"] == 0
    assert inherited["retryCount"] == 0
    assert inherited["passes"] == []


def test_log_turn_the_registry_has_not_saved_reports_no_opener(tmp_path: Path) -> None:
    """The turn in flight is in the log before its record is written."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0, number=7)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)

    payload = _scan(log, tmp_path, records=[])
    turn = _turns(payload)[_TURN_A]

    assert turn["turnNumber"] == 7
    assert turn["userMessageId"] is None
    assert turn["traceAvailable"] is True


def test_legacy_registry_cannot_move_an_overwritten_turn_after_its_successor(tmp_path: Path) -> None:
    """The legacy registry lost B when C reused its number; the log still orders both."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0, number=1)
    _turn_finished(log, _TURN_A, _NS)
    _turn_started(log, _TURN_B, 2 * _NS, number=2)
    _turn_finished(log, _TURN_B, 3 * _NS, end_reason="error")
    sequence_c = log.next_sequence
    _turn_started(log, _TURN_C, 4 * _NS, number=2)
    _turn_finished(log, _TURN_C, 5 * _NS)
    records = turn_records(
        {
            "chrys_trajectory": {
                "turns": {
                    "1": {"turn_id": _TURN_A, "started_sequence": 2},
                    "2": {"turn_id": _TURN_C, "started_sequence": sequence_c},
                }
            }
        }
    )

    payload = _scan(log, tmp_path, records=records)

    assert [turn["turnId"] for turn in payload["turns"]] == [_TURN_A, _TURN_B, _TURN_C]
    assert [turn["turnNumber"] for turn in payload["turns"]] == [1, 2, 2]


def test_superseded_turn_keeps_its_logged_number_after_the_registry_rolls_back(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0, number=1)
    _turn_finished(log, _TURN_A, _NS)
    rollback_from = log.next_sequence
    _turn_started(log, _TURN_B, 2 * _NS, number=2)
    _turn_finished(log, _TURN_B, 3 * _NS)
    log.resolved_rollback(rollback_from, 4 * _NS, target_turn_id=_TURN_A)

    payload = _scan(log, tmp_path, records=[TurnRecord(turn_number=1, turn_id=_TURN_A)])

    assert payload["supersededTurns"][0]["turnId"] == _TURN_B
    assert payload["supersededTurns"][0]["turnNumber"] == 2


def test_retry_keeps_the_first_logged_turn_number_unless_the_registry_supplies_it(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0, number=2)
    _turn_finished(log, _TURN_A, _NS, end_reason="error")
    _turn_started(log, _TURN_B, 2 * _NS, logical=_TURN_A, number=3, is_retry=True)
    _turn_finished(log, _TURN_B, 3 * _NS)
    path = tmp_path / "events.jsonl"
    log.write(path)
    trace = scan_turn_trace(path)

    without_registry = trace_payload_with_registry(trace, [])
    with_registry = trace_payload_with_registry(trace, [TurnRecord(turn_number=1, turn_id=_TURN_A)])

    assert without_registry is not None
    assert without_registry["turns"][0]["turnNumber"] == 2
    assert with_registry is not None
    assert with_registry["turns"][0]["turnNumber"] == 1


def test_reused_turn_number_keeps_both_turns(tmp_path: Path) -> None:
    """Ordinal drift can leave two turns sharing a number; ``turnId`` decides."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    _turn_started(log, _TURN_B, 2 * _NS)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=_TURN_B, input_tokens=20)
    _turn_finished(log, _TURN_B, 3 * _NS)
    records = [
        TurnRecord(turn_number=1, turn_id=_TURN_A),
        TurnRecord(turn_number=1, turn_id=_TURN_B),
    ]

    payload = _scan(log, tmp_path, records=records)

    assert [turn["turnId"] for turn in payload["turns"]] == [_TURN_A, _TURN_B]
    assert [turn["turnNumber"] for turn in payload["turns"]] == [1, 1]


def test_turn_filter_returns_only_the_requested_turn(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_finished(log, _TURN_A, _NS)
    _turn_started(log, _TURN_B, 2 * _NS)
    _turn_finished(log, _TURN_B, 3 * _NS)
    path = tmp_path / "events.jsonl"
    log.write(path)
    trace = scan_turn_trace(path)

    payload = trace_payload_with_registry(trace, [], turn_id=_TURN_B)

    assert payload is not None
    assert [turn["turnId"] for turn in payload["turns"]] == [_TURN_B]
    assert payload["coverage"] == "exact"
    assert trace_payload_with_registry(trace, [], turn_id="c" * 32) is None


def test_turn_filter_finds_a_superseded_turn(tmp_path: Path) -> None:
    """The filter asks by id, so a rollback-discarded turn is not invisible to it."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_finished(log, _TURN_A, _NS)
    rollback_from = log.next_sequence
    _turn_started(log, _TURN_B, 2 * _NS)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=_TURN_B, input_tokens=20)
    _turn_finished(log, _TURN_B, 3 * _NS)
    log.resolved_rollback(rollback_from, 4 * _NS)
    path = tmp_path / "events.jsonl"
    log.write(path)
    trace = scan_turn_trace(path)

    payload = trace_payload_with_registry(trace, [], turn_id=_TURN_B)

    assert payload is not None
    # The filter names the turn, and it stays in the list the log put it in:
    # a superseded turn is not promoted back onto the current branch by being
    # asked about.
    assert payload["turns"] == []
    assert [turn["turnId"] for turn in payload["supersededTurns"]] == [_TURN_B]
    assert payload["supersededTurns"][0]["usage"] == {"inputTokens": 20}


def test_turn_filter_finds_an_inherited_turn(tmp_path: Path) -> None:
    """A fork-inherited turn has no local `turn.started`; the registry is its record."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_finished(log, _TURN_A, _NS)
    path = tmp_path / "events.jsonl"
    log.write(path)
    trace = scan_turn_trace(path)
    records = [TurnRecord(turn_number=2, turn_id=_TURN_B, origin_session_id="origin-session")]

    payload = trace_payload_with_registry(trace, records, turn_id=_TURN_B)

    assert payload is not None
    assert [turn["turnId"] for turn in payload["turns"]] == [_TURN_B]
    assert payload["turns"][0]["inheritedFrom"] == "origin-session"
    assert payload["turns"][0]["traceAvailable"] is False


@pytest.mark.parametrize("kind", ["current", "superseded", "inherited"])
def test_turn_filter_omits_unassigned_passes_but_keeps_session_coverage(tmp_path: Path, kind: str) -> None:
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_finished(log, _TURN_A, _NS)
    if kind == "superseded":
        log.resolved_rollback(2, 2 * _NS)
    _exchange(log, _CALL_B, 3 * _NS, pass_id=_TURN_B, input_tokens=10)
    path = tmp_path / "events.jsonl"
    log.write(path)
    trace = scan_turn_trace(path)
    unfiltered = trace.to_payload()
    records = [TurnRecord(turn_number=1, turn_id=_TURN_C, origin_session_id="origin-session")]
    selected_id = _TURN_C if kind == "inherited" else _TURN_A

    payload = trace_payload_with_registry(trace, records, turn_id=selected_id)

    assert payload is not None
    assert payload["unassignedPasses"] == []
    assert payload["coverage"] == unfiltered["coverage"] == "partial"
    assert payload["gapCount"] == unfiltered["gapCount"]
    selected = payload["supersededTurns"] if kind == "superseded" else payload["turns"]
    assert [turn["turnId"] for turn in selected] == [selected_id]
    assert selected[0]["coverage"] == "exact"
    assert [entry["passId"] for entry in unfiltered["unassignedPasses"]] == [_TURN_B]
    assert trace.to_payload() == unfiltered


# ------------------------------------------------------------------- integrity


@pytest.mark.parametrize("damage", ["corrupt", "torn", "gap"])
def test_damage_after_an_unfinished_retry_marks_its_turn_partial(tmp_path: Path, damage: str) -> None:
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_finished(log, _TURN_A, _NS, end_reason="error")
    _turn_started(log, _TURN_B, 2 * _NS, logical=_TURN_A, is_retry=True)
    _exchange(log, _CALL_B, 3 * _NS, pass_id=_TURN_B, input_tokens=10)
    if damage == "gap":
        missing = log.next_sequence
        log.add(
            EventType.GAP,
            5 * _NS,
            turn_id=None,
            payload={"first_sequence": missing, "last_sequence": missing, "dropped_count": 1},
        )
    path = tmp_path / "events.jsonl"
    log.write(path)
    if damage != "gap":
        with path.open("ab") as handle:
            handle.write(b"{not json}\n" if damage == "corrupt" else b'{"partial":')

    payload = scan_turn_trace(path).to_payload()

    assert payload["coverage"] == "partial"
    assert payload["turns"][0]["coverage"] == "partial"
    assert payload["turns"][0]["status"] == "unfinished"
    assert payload["turns"][0]["modelCalls"][0]["outcome"] == "success"


def test_gap_between_a_finished_pass_and_its_retry_does_not_damage_the_turn(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_finished(log, _TURN_A, _NS, end_reason="error")
    missing = log.next_sequence
    log.add(
        EventType.GAP,
        2 * _NS,
        turn_id=None,
        payload={"first_sequence": missing, "last_sequence": missing, "dropped_count": 1},
    )
    _turn_started(log, _TURN_B, 3 * _NS, logical=_TURN_A, is_retry=True)
    _turn_finished(log, _TURN_B, 4 * _NS)

    payload = _scan(log, tmp_path)

    assert payload["coverage"] == "partial"
    assert payload["turns"][0]["coverage"] == "exact"


def test_next_pass_bounds_an_unfinished_pass_for_coverage(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_started(log, _TURN_B, _NS)
    path = tmp_path / "events.jsonl"
    log.write(path)
    with path.open("ab") as handle:
        handle.write(b'{"partial":')

    payload = scan_turn_trace(path).to_payload()

    assert _turns(payload)[_TURN_A]["coverage"] == "exact"
    assert _turns(payload)[_TURN_B]["coverage"] == "partial"


def test_damage_inside_an_unfinished_turn_marks_only_that_turn_partial(tmp_path: Path) -> None:
    """A turn still open when the damage starts cannot be vouched for."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    _turn_started(log, _TURN_B, 2 * _NS)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=_TURN_B, input_tokens=20)
    path = tmp_path / "events.jsonl"
    log.write(path)
    with path.open("ab") as handle:
        handle.write(b"{not json}\n")

    payload = scan_turn_trace(path).to_payload()
    turns = _turns(payload)

    assert turns[_TURN_A]["coverage"] == "exact"
    assert turns[_TURN_B]["coverage"] == "partial"
    assert payload["coverage"] == "partial"


def test_calls_without_a_turn_started_stay_in_unassigned_passes(tmp_path: Path) -> None:
    """A readable call the projection cannot place is kept, never silently dropped."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=_TURN_B, input_tokens=20)

    payload = _scan(log, tmp_path)

    assert [turn["turnId"] for turn in payload["turns"]] == [_TURN_A]
    assert [entry["passId"] for entry in payload["unassignedPasses"]] == [_TURN_B]
    assert payload["unassignedPasses"][0]["coverage"] == "partial"
    assert payload["unassignedPasses"][0]["modelCallCount"] == 1
    assert payload["unassignedPasses"][0]["modelCalls"][0]["usage"] == {"inputTokens": 20}
    assert payload["coverage"] == "partial"


def test_call_without_a_pass_is_session_level(tmp_path: Path) -> None:
    """A call the log records outside every pass — a generated title — is not guessed at."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=None, input_tokens=5)

    payload = _scan(log, tmp_path)

    assert payload["sessionLevel"]["modelCallCount"] == 1
    assert payload["sessionLevel"]["usage"] == {"inputTokens": 5}
    assert payload["sessionLevel"]["modelCalls"][0]["passId"] is None
    assert payload["unassignedPasses"] == []


@pytest.mark.parametrize("pass_id", [_TURN_A, _TURN_B, None], ids=["turn", "unassigned", "session"])
@pytest.mark.parametrize(
    "request_events",
    [
        (EventType.MODEL_REQUEST_PREPARED,),
        (EventType.MODEL_REQUEST_HEADERS_RECEIVED,),
        (EventType.MODEL_REQUEST_PREPARED, EventType.MODEL_REQUEST_HEADERS_RECEIVED),
    ],
    ids=["prepared-only", "headers-only", "prepared-and-headers"],
)
def test_request_records_survive_without_either_exchange_marker(
    tmp_path: Path, pass_id: str | None, request_events: tuple[str, ...]
) -> None:
    log = EventLog()
    log.coverage()
    if pass_id == _TURN_A:
        _turn_started(log, _TURN_A, 0)
    # Repeated observations of the same attempt still describe only one request.
    for event_type in request_events * 2:
        fields: dict[str, object] = {"request_attempt_id": _ATTEMPT_1}
        if event_type == EventType.MODEL_REQUEST_HEADERS_RECEIVED:
            fields.update(status_code=200, provider_request_id="provider-1")
        log.add(event_type, _NS, turn_id=pass_id, operation_id=_CALL_A, payload=fields)
    if pass_id == _TURN_A:
        _turn_finished(log, _TURN_A, 2 * _NS)

    payload = _scan(log, tmp_path)

    assert payload["sessionId"] == SESSION_ID
    assert payload["coverage"] == "partial"
    if pass_id == _TURN_A:
        container = payload["turns"][0]
        assert container["coverage"] == "partial"
        assert container["usageUnreportedCalls"] == 1
    elif pass_id == _TURN_B:
        container = payload["unassignedPasses"][0]
        assert container["passId"] == _TURN_B
        assert container["coverage"] == "partial"
    else:
        container = payload["sessionLevel"]
    assert container["modelCallCount"] == 1
    assert container["requestAttemptCount"] == 1
    call = container["modelCalls"][0]
    assert call["modelCallId"] == _CALL_A
    assert call["passId"] == pass_id
    assert call["startedAt"] is None
    assert call["endedAt"] is None
    assert call["outcome"] is None
    assert call["usage"] == {}
    assert call["usageReported"] is False
    has_headers = EventType.MODEL_REQUEST_HEADERS_RECEIVED in request_events
    assert call["attempts"] == [
        {
            "requestAttemptId": _ATTEMPT_1,
            "httpStatus": 200 if has_headers else None,
            "providerRequestId": "provider-1" if has_headers else None,
        }
    ]


def test_gap_marks_the_turn_that_covers_its_range_partial(tmp_path: Path) -> None:
    """A gap is positioned by the range it names, not by the line that reports it.

    The line lands while the damaged turn is still open; a turn that already
    ended before the range is untouched.
    """
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_finished(log, _TURN_A, _NS)
    _turn_started(log, _TURN_B, 2 * _NS)
    log.add(
        EventType.GAP,
        3 * _NS,
        turn_id=None,
        payload={"first_sequence": 5, "last_sequence": 6, "dropped_count": 2, "reason": "write_failure"},
    )
    _turn_finished(log, _TURN_B, 4 * _NS)

    payload = _scan(log, tmp_path)
    turns = _turns(payload)

    assert payload["gapCount"] == 1
    assert turns[_TURN_A]["coverage"] == "exact"
    assert turns[_TURN_B]["coverage"] == "partial"
    assert payload["coverage"] == "partial"


def test_call_the_log_never_finished_is_kept_and_marks_the_turn_partial(tmp_path: Path) -> None:
    """A started call with no terminal is readable work with an unknown end.

    Dropping it would hide the request — and the attempt ids that join it to
    the wire — so it is reported with ``outcome: null`` and its turn is not
    vouched for.
    """
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    log.add("model.exchange.started", 0, turn_id=_TURN_A, operation_id=_CALL_A)

    payload = _scan(log, tmp_path)
    turn = _turns(payload)[_TURN_A]

    assert turn["coverage"] == "partial"
    assert turn["modelCallCount"] == 1
    assert turn["modelCalls"][0]["modelCallId"] == _CALL_A
    assert turn["modelCalls"][0]["outcome"] is None
    assert turn["modelCalls"][0]["startedAt"] is not None
    assert payload["coverage"] == "partial"


@pytest.mark.parametrize(
    "request_event", [None, EventType.MODEL_REQUEST_PREPARED, EventType.MODEL_REQUEST_HEADERS_RECEIVED]
)
def test_exchange_finished_without_its_started_is_still_reported(tmp_path: Path, request_event: str | None) -> None:
    """A terminal the log kept while the start was lost — a torn head — is not thrown away."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    if request_event is not None:
        log.add(
            request_event,
            0,
            turn_id=_TURN_A,
            operation_id=_CALL_A,
            payload={"request_attempt_id": _ATTEMPT_1, "status_code": 200, "provider_request_id": "provider-1"},
        )
    log.add(
        "model.exchange.finished",
        _NS,
        turn_id=_TURN_A,
        operation_id=_CALL_A,
        payload={
            "agent_profile_id": "Code",
            "provider": "openai",
            "request_model": "profile-model",
            "outcome": "success",
            "duration_ms": 1,
            "usage": {"normalized": {"input_total": 10}},
        },
    )
    _turn_finished(log, _TURN_A, 2 * _NS)

    payload = _scan(log, tmp_path)
    call = _turns(payload)[_TURN_A]["modelCalls"][0]

    assert call["modelCallId"] == _CALL_A
    assert call["startedAt"] is None
    assert call["endedAt"] is not None
    assert call["outcome"] == "success"
    assert call["usage"] == {"inputTokens": 10}
    assert call["agentProfile"] == "Code"
    assert call["provider"] == "openai"
    assert call["requestModel"] == "profile-model"
    has_headers = request_event == EventType.MODEL_REQUEST_HEADERS_RECEIVED
    assert call["attempts"] == (
        []
        if request_event is None
        else [
            {
                "requestAttemptId": _ATTEMPT_1,
                "httpStatus": 200 if has_headers else None,
                "providerRequestId": "provider-1" if has_headers else None,
            }
        ]
    )
    assert _turns(payload)[_TURN_A]["coverage"] == "partial"
    assert payload["coverage"] == "partial"


def test_exchange_terminal_does_not_overwrite_request_specific_facts(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    log.span(
        "model.exchange",
        _CALL_A,
        0,
        _NS,
        turn_id=_TURN_A,
        start_payload={"agent_profile_id": "Code", "provider": "openai", "request_model": "request-override"},
        finish_payload={"agent_profile_id": "Code", "provider": "openai", "request_model": "profile-model"},
    )
    _turn_finished(log, _TURN_A, 2 * _NS)

    call = _turns(_scan(log, tmp_path))[_TURN_A]["modelCalls"][0]

    assert call["requestModel"] == "request-override"


def test_active_retry_has_no_end_timestamp_until_its_own_pass_finishes(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_finished(log, _TURN_A, _NS, end_reason="error")
    _turn_started(log, _TURN_B, 2 * _NS, logical=_TURN_A, is_retry=True)

    active = _turns(_scan(log, tmp_path))[_TURN_A]

    assert active["status"] == "unfinished"
    assert active["passes"][0]["endedAt"] is not None
    assert active["endedAt"] is None

    _turn_finished(log, _TURN_B, 3 * _NS)
    completed = _turns(_scan(log, tmp_path))[_TURN_A]

    assert completed["status"] == "completed"
    assert completed["endedAt"] is not None
    assert completed["endedAt"] == completed["passes"][-1]["endedAt"]


def test_damage_in_a_later_turn_does_not_spare_it(tmp_path: Path) -> None:
    """A turn is judged against the damage inside it, not against the first damage found.

    The earlier turn here ends where the unreadable records begin, so it is
    untouched by them; the gap falls inside the *later* turn, which is the one
    the loss actually took something from.
    """
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_finished(log, _TURN_A, _NS)
    # The gap the reader would have written: records 4-5, dropped while the
    # second turn was open.
    log.add(
        EventType.GAP,
        _NS,
        turn_id=None,
        payload={"first_sequence": 4, "last_sequence": 5, "dropped_count": 2, "reason": "write_failure"},
    )
    _turn_started(log, _TURN_B, 2 * _NS)
    _turn_finished(log, _TURN_B, 3 * _NS)

    payload = _scan(log, tmp_path)
    turns = _turns(payload)

    assert payload["gapCount"] == 1
    assert turns[_TURN_A]["coverage"] == "exact"
    assert turns[_TURN_B]["coverage"] == "partial"


def test_a_torn_tail_reaches_the_turn_that_could_have_run_into_it(tmp_path: Path) -> None:
    """Bytes with no terminating newline are damage at the end of the log."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    _turn_started(log, _TURN_B, 2 * _NS)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=_TURN_B, input_tokens=20)
    path = tmp_path / "events.jsonl"
    log.write(path)
    with path.open("ab") as handle:
        handle.write(b'{"partial":')

    payload = scan_turn_trace(path).to_payload()
    turns = _turns(payload)

    assert turns[_TURN_A]["coverage"] == "exact"
    assert turns[_TURN_B]["coverage"] == "partial"
    assert payload["coverage"] == "partial"


def test_retry_that_never_started_is_pending_not_counted(tmp_path: Path) -> None:
    """A backoff the interrupt cancelled is a pending record, not a retry that ran.

    Counting the ``retry.scheduled`` alone would report a retry that never
    happened; it is kept instead, and the log it left behind is reported as
    still holding that backoff open.
    """
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    log.add(
        EventType.RETRY_SCHEDULED,
        _NS,
        turn_id=_TURN_A,
        operation_id=_RETRY_1,
        payload={"reason_code": "stream_stall", "delay_ms": 3000, "retry_mode": "wire"},
    )
    _turn_finished(log, _TURN_A, 2 * _NS)

    payload = _scan(log, tmp_path)
    turn = _turns(payload)[_TURN_A]

    assert turn["retryCount"] == 0
    assert [retry["retryId"] for retry in turn["retries"]] == [_RETRY_1]
    assert turn["retries"][0]["retryMode"] == "wire"
    # The backoff was still open when the log ended, so nothing here vouches
    # for the turn that was waiting it out.
    assert turn["coverage"] == "partial"
    assert payload["coverage"] == "partial"


@pytest.mark.parametrize("branch_change", ["none", "later-turn", "rollback", "rollback-then-turn"])
@pytest.mark.parametrize("started", [False, True], ids=["pending", "started"])
def test_passless_retry_keeps_its_scheduled_turn(tmp_path: Path, branch_change: str, started: bool) -> None:
    """A retry's scheduling-time owner survives later turns and rollback.

    Pending backoffs make only that owner partial, including in supersededTurns;
    a surviving start also uses the owner captured by its schedule.
    """
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    log.add(
        EventType.RETRY_SCHEDULED,
        _NS,
        turn_id=None,
        operation_id=_RETRY_1,
        payload={"reason_code": "stream_stall", "delay_ms": 3000, "retry_mode": "wire"},
    )
    _turn_finished(log, _TURN_A, 2 * _NS)
    rolled_back = branch_change in ("rollback", "rollback-then-turn")
    later_turn = branch_change in ("later-turn", "rollback-then-turn")
    if rolled_back:
        log.resolved_rollback(2, 3 * _NS)
    if later_turn:
        _turn_started(log, _TURN_B, 4 * _NS, number=2)
        _turn_finished(log, _TURN_B, 5 * _NS)
    if started:
        log.add(EventType.RETRY_STARTED, 6 * _NS, turn_id=None, operation_id=_RETRY_1)

    payload = _scan(log, tmp_path)
    turn = payload["supersededTurns"][0] if rolled_back else _turns(payload)[_TURN_A]

    assert turn["turnId"] == _TURN_A
    assert [retry["retryId"] for retry in turn["retries"]] == [_RETRY_1]
    assert turn["retries"][0]["started"] is started
    assert turn["retries"][0]["passId"] is None
    assert turn["retryCount"] == int(started)
    assert turn["coverage"] == ("exact" if started else "partial")
    assert payload["coverage"] == ("exact" if started else "partial")
    if later_turn:
        later = _turns(payload)[_TURN_B]
        assert later["retries"] == []
        assert later["retryCount"] == 0
        assert later["coverage"] == "exact"


def test_passless_backoff_without_an_owner_is_not_adopted_by_a_later_turn(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.add(
        EventType.RETRY_SCHEDULED,
        0,
        turn_id=None,
        operation_id=_RETRY_1,
        payload={"reason_code": "stream_stall", "delay_ms": 3000, "retry_mode": "wire"},
    )
    _turn_started(log, _TURN_A, _NS)
    _turn_finished(log, _TURN_A, 2 * _NS)

    payload = _scan(log, tmp_path)
    turn = _turns(payload)[_TURN_A]

    assert turn["retries"] == []
    assert turn["coverage"] == "exact"
    assert payload["coverage"] == "partial"


def test_orphan_retry_pass_is_unassigned_not_invisible(tmp_path: Path) -> None:
    """A pass the log recorded only through its retry still shows up as a pass."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    log.resolved_rollback(2, 2 * _NS)
    log.add(
        EventType.RETRY_STARTED,
        3 * _NS,
        turn_id=_TURN_B,
        operation_id=_RETRY_1,
        payload={"retry_mode": "wire"},
    )

    payload = _scan(log, tmp_path)

    assert [entry["passId"] for entry in payload["unassignedPasses"]] == [_TURN_B]
    assert payload["unassignedPasses"][0]["retryCount"] == 1
    assert payload["unassignedPasses"][0]["retries"][0]["retryId"] == _RETRY_1
    assert payload["coverage"] == "partial"


def test_session_level_call_without_a_terminal_makes_the_session_partial(tmp_path: Path) -> None:
    """Work the log left open outside every turn still means the scan is not final."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    _turn_finished(log, _TURN_A, _NS)
    log.add("model.exchange.started", 2 * _NS, turn_id=None, operation_id=_CALL_D)

    payload = _scan(log, tmp_path)

    assert payload["sessionLevel"]["modelCallCount"] == 1
    assert payload["sessionLevel"]["modelCalls"][0]["outcome"] is None
    assert payload["coverage"] == "partial"
    assert _turns(payload)[_TURN_A]["coverage"] == "exact"


def test_a_second_terminal_for_one_call_is_not_a_second_call(tmp_path: Path) -> None:
    """One exchange ends once: a repeated terminal must not double the call or its usage."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _exchange(log, _CALL_A, 0, pass_id=_TURN_A, input_tokens=10)
    log.add(
        "model.exchange.finished",
        2 * _NS,
        turn_id=_TURN_A,
        operation_id=_CALL_A,
        payload={"outcome": "success", "duration_ms": 1, "usage": {"normalized": {"input_total": 10}}},
    )
    _turn_finished(log, _TURN_A, 3 * _NS)

    payload = _scan(log, tmp_path)
    turn = _turns(payload)[_TURN_A]

    assert turn["modelCallCount"] == 1
    assert turn["usage"] == {"inputTokens": 10}


def test_a_superseded_turn_is_not_promoted_back_into_the_live_list(tmp_path: Path) -> None:
    """The registry merge replaces neither branch list with the other."""
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_A, 0)
    _turn_finished(log, _TURN_A, _NS)
    rollback_from = log.next_sequence
    _turn_started(log, _TURN_B, 2 * _NS)
    _exchange(log, _CALL_B, 2 * _NS, pass_id=_TURN_B, input_tokens=20)
    _turn_finished(log, _TURN_B, 3 * _NS)
    log.resolved_rollback(rollback_from, 4 * _NS)
    records = [
        TurnRecord(turn_number=1, turn_id=_TURN_A),
        TurnRecord(turn_number=2, turn_id=_TURN_B),
    ]

    payload = _scan(log, tmp_path, records=records)

    assert [turn["turnId"] for turn in payload["turns"]] == [_TURN_A]
    assert [turn["turnId"] for turn in payload["supersededTurns"]] == [_TURN_B]
    assert payload["turns"][0]["userMessageId"] is None


def test_registry_only_turn_shares_an_ordinal_without_reordering(tmp_path: Path) -> None:
    """The registry places an inherited turn before its logged successor.

    A fork inherits the first turn and reuses its number for the turn it runs
    itself; sorting by that number can only guess which came first, while the
    registry says it outright.
    """
    log = EventLog()
    log.coverage()
    _turn_started(log, _TURN_B, 0, number=1)
    _exchange(log, _CALL_B, 0, pass_id=_TURN_B, input_tokens=20)
    _turn_finished(log, _TURN_B, _NS)
    records = [
        TurnRecord(turn_number=1, turn_id=_TURN_A, origin_session_id="origin-session"),
        TurnRecord(turn_number=1, turn_id=_TURN_B),
    ]

    payload = _scan(log, tmp_path, records=records)

    assert [turn["turnId"] for turn in payload["turns"]] == [_TURN_A, _TURN_B]
    assert payload["turns"][0]["traceAvailable"] is False
    assert payload["turns"][1]["traceAvailable"] is True
