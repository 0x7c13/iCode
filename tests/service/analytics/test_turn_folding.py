# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Retry attempts fold into one logical turn without losing physical attempt identity."""

from __future__ import annotations

from array import array
from dataclasses import replace
from pathlib import Path

from chrys.foundation.trajectory.event_types import EventType
from chrys.service.analytics import (
    FLOW_TERMINAL_INDEX,
    ActionClass,
    Metric,
    Precision,
    TimelineOperation,
    TokenUsage,
    TurnAttemptRef,
    TurnFlow,
    UsageBucket,
    analyze_trajectory,
)
from chrys.service.analytics import aggregation as aggregation_module
from tests.service.analytics._events import NS, EventLog


def test_retry_attempts_fold_into_one_logical_turn(tmp_path: Path) -> None:
    first_turn_id = "4" * 32
    retry_turn_id = "5" * 32
    second_retry_turn_id = "6" * 32
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 0, turn_id=first_turn_id, payload={"turn_number": 1, "is_retry": False})
    log.add(
        EventType.TURN_FINISHED,
        2 * NS,
        turn_id=first_turn_id,
        payload={"end_reason": "interrupted", "duration_ms": 2_000},
    )
    log.settled(3 * NS, turn_id=first_turn_id, drained_scopes=[])
    log.add(
        EventType.TURN_STARTED,
        100 * NS,
        turn_id=retry_turn_id,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        EventType.TURN_FINISHED,
        102 * NS,
        turn_id=retry_turn_id,
        payload={"end_reason": "interrupted", "duration_ms": 2_000},
    )
    log.settled(103 * NS, turn_id=retry_turn_id, drained_scopes=[])
    log.add(
        EventType.TURN_STARTED,
        200 * NS,
        turn_id=second_retry_turn_id,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        EventType.TURN_FINISHED,
        204 * NS,
        turn_id=second_retry_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 4_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 1
    turn = analysis.turns[0]
    assert turn.turn_id == first_turn_id
    assert turn.turn_number == 1
    assert tuple(attempt.turn_id for attempt in turn.attempts) == (
        first_turn_id,
        retry_turn_id,
        second_retry_turn_id,
    )
    assert analysis.turn(retry_turn_id) is turn
    assert analysis.turn(second_retry_turn_id) is turn
    assert turn.elapsed_ns == Metric(10 * NS, Precision.EXACT)
    assert turn.axis_end_ns - turn.axis_start_ns == 10 * NS
    assert [attempt.is_retry for attempt in turn.attempts] == [False, True, True]
    assert analysis.change_verification is not None
    assert "turns' numbers cannot join" not in (analysis.change_verification.files_touched.reason or "")


def test_retry_timeline_diagnostic_keeps_physical_attempt_turn_id(tmp_path: Path) -> None:
    first_turn_id = "4" * 32
    retry_turn_id = "5" * 32
    hook_operation_id = "a" * 32
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 0, turn_id=first_turn_id, payload={"turn_number": 1, "is_retry": False})
    log.add(
        EventType.TURN_FINISHED,
        2 * NS,
        turn_id=first_turn_id,
        payload={"end_reason": "interrupted", "duration_ms": 2_000},
    )
    log.settled(3 * NS, turn_id=first_turn_id, drained_scopes=[])
    log.add(
        EventType.TURN_STARTED,
        100 * NS,
        turn_id=retry_turn_id,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        EventType.HOOK_OPERATION_STARTED,
        101 * NS,
        turn_id=retry_turn_id,
        operation_id=hook_operation_id,
        payload={
            "hook_key": "retry-cleanup",
            "hook_event": "after_turn",
            "execution_mode": "async",
            "drain_scope": "turn",
        },
    )
    log.add(
        EventType.TURN_FINISHED,
        102 * NS,
        turn_id=retry_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 2_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 1
    assert analysis.turns[0].turn_id == first_turn_id
    diagnostic = next(
        item for item in analysis.diagnostics.timeline_operations if item.operation_id == hook_operation_id
    )
    assert diagnostic.turn_id == retry_turn_id
    assert diagnostic.turn_number == 1


def test_same_number_fresh_turns_are_never_folded(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    for index, turn_id in enumerate(("4" * 32, "5" * 32)):
        log.add(
            EventType.TURN_STARTED,
            index * 2 * NS,
            turn_id=turn_id,
            payload={"turn_number": 1, "is_retry": False},
        )
        log.add(
            EventType.TURN_FINISHED,
            (index * 2 + 1) * NS,
            turn_id=turn_id,
            payload={"end_reason": "cancelled", "duration_ms": 1_000},
        )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 2
    assert [turn.turn_number for turn in analysis.turns] == [1, 1]


def test_orphan_retry_is_normalized_as_one_logical_turn(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 0, payload={"turn_number": 1, "is_retry": True})
    log.add(
        EventType.TURN_FINISHED,
        NS,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert len(turn.attempts) == 1
    assert turn.attempts[0].turn_id == turn.turn_id
    assert turn.attempts[0].is_retry is True


def test_normalized_single_retry_keeps_physical_identity_for_refolding(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 10 * NS, payload={"turn_number": 1, "is_retry": True})
    log.add(
        EventType.TURN_FINISHED,
        11 * NS,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)
    cached_retry = analyze_trajectory(path).turns[0]
    first_turn_id = "5" * 32
    predecessor = replace(
        cached_retry,
        turn_id=first_turn_id,
        start_sequence=0,
        axis_start_ns=0,
        axis_end_ns=NS,
        attempts=(
            replace(
                cached_retry.attempts[0],
                turn_id=first_turn_id,
                is_retry=False,
                physical_axis_start_ns=0,
                physical_axis_end_ns=NS,
                logical_axis_start_ns=0,
            ),
        ),
    )

    refolded = aggregation_module._fold_retry_turns([predecessor, cached_retry])

    assert len(refolded) == 1
    assert [attempt.is_retry for attempt in refolded[0].attempts] == [False, True]


def test_noncontiguous_retry_number_is_not_folded_or_joinable(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    for index, (turn_id, turn_number, is_retry) in enumerate(
        (("4" * 32, 1, False), ("5" * 32, 2, False), ("6" * 32, 1, True))
    ):
        log.add(
            EventType.TURN_STARTED,
            index * 2 * NS,
            turn_id=turn_id,
            payload={"turn_number": turn_number, "is_retry": is_retry},
        )
        log.add(
            EventType.TURN_FINISHED,
            (index * 2 + 1) * NS,
            turn_id=turn_id,
            payload={"end_reason": "cancelled", "duration_ms": 1_000},
        )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 3
    assert [turn.turn_number for turn in analysis.turns] == [1, 2, 1]
    assert analysis.change_verification is not None
    assert "turns' numbers cannot join" in (analysis.change_verification.files_touched.reason or "")


def test_duplicate_consistent_starts_keep_identity_and_fold_with_retry(tmp_path: Path) -> None:
    first_turn_id = "4" * 32
    retry_turn_id = "5" * 32
    log = EventLog()
    log.coverage()
    for monotonic_ns in (0, NS):
        log.add(
            EventType.TURN_STARTED,
            monotonic_ns,
            turn_id=first_turn_id,
            payload={"turn_number": 1, "is_retry": False},
        )
    log.add(
        EventType.TURN_FINISHED,
        2 * NS,
        turn_id=first_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 2_000},
    )
    log.add(
        EventType.TURN_STARTED,
        10 * NS,
        turn_id=retry_turn_id,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        EventType.TURN_FINISHED,
        11 * NS,
        turn_id=retry_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 1
    turn = analysis.turns[0]
    assert turn.turn_number == 1
    assert [attempt.turn_id for attempt in turn.attempts] == [first_turn_id, retry_turn_id]
    assert "turn lifecycle is not uniquely opened" in turn.diagnostics


def test_duplicate_retry_starts_restore_retry_identity_and_fold_with_predecessor(tmp_path: Path) -> None:
    first_turn_id = "4" * 32
    retry_turn_id = "5" * 32
    log = EventLog()
    log.coverage()
    log.add(
        EventType.TURN_STARTED,
        0,
        turn_id=first_turn_id,
        payload={"turn_number": 1, "is_retry": False},
    )
    log.add(
        EventType.TURN_FINISHED,
        NS,
        turn_id=first_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    for monotonic_ns in (10 * NS, 11 * NS):
        log.add(
            EventType.TURN_STARTED,
            monotonic_ns,
            turn_id=retry_turn_id,
            payload={"turn_number": 1, "is_retry": True},
        )
    log.add(
        EventType.TURN_FINISHED,
        12 * NS,
        turn_id=retry_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 1
    turn = analysis.turns[0]
    assert [attempt.turn_id for attempt in turn.attempts] == [first_turn_id, retry_turn_id]
    assert [attempt.is_retry for attempt in turn.attempts] == [False, True]
    assert "turn lifecycle is not uniquely opened" in turn.diagnostics


def test_fold_sums_attempt_tokens_actions_and_critical_contributions(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.turn(0, 5 * NS)
    path = tmp_path / "base.jsonl"
    log.write(path)
    base = analyze_trajectory(path).turns[0]
    optional_missing = Metric(None, Precision.MISSING, "not reported")

    def physical_attempt(
        turn_id: str,
        *,
        axis_start_ns: int,
        is_retry: bool,
        input_tokens: int,
        action_count: int,
        contribution_ns: int,
        elapsed_precision: Precision,
    ):
        axis_end_ns = axis_start_ns + 5 * NS
        return replace(
            base,
            turn_id=turn_id,
            start_sequence=1 if not is_retry else 10,
            elapsed_ns=Metric(5 * NS, elapsed_precision),
            axis_start_ns=axis_start_ns,
            axis_end_ns=axis_end_ns,
            attempts=(
                TurnAttemptRef(
                    turn_id=turn_id,
                    runtime_id=base.runtime_id,
                    is_retry=is_retry,
                    physical_axis_start_ns=axis_start_ns,
                    physical_axis_end_ns=axis_end_ns,
                    logical_axis_start_ns=axis_start_ns,
                    operation_start_index=0,
                    operation_end_index=len(base.operations),
                    slice_start_index=0,
                    slice_end_index=len(base.slices),
                ),
            ),
            action_counts={ActionClass.SEARCH: Metric(action_count, Precision.EXACT)},
            critical_tool_contributions_ns={"shared-operation": contribution_ns},
            server_critical_contributions_ns={"shared-server": contribution_ns + 1},
            token_usage=TokenUsage(
                buckets={
                    UsageBucket.INPUT: Metric(input_tokens, Precision.EXACT),
                    UsageBucket.OUTPUT: Metric(1, Precision.EXACT),
                    UsageBucket.REASONING: optional_missing,
                    UsageBucket.CACHE_READ: optional_missing,
                    UsageBucket.CACHE_CREATION: optional_missing,
                }
            ),
        )

    folded = aggregation_module._fold_retry_turns(
        [
            physical_attempt(
                "4" * 32,
                axis_start_ns=0,
                is_retry=False,
                input_tokens=10,
                action_count=2,
                contribution_ns=3,
                elapsed_precision=Precision.EXACT,
            ),
            physical_attempt(
                "5" * 32,
                axis_start_ns=100 * NS,
                is_retry=True,
                input_tokens=20,
                action_count=4,
                contribution_ns=7,
                elapsed_precision=Precision.ESTIMATED,
            ),
        ]
    )[0]

    assert folded.elapsed_ns == Metric(
        10 * NS,
        Precision.ESTIMATED,
        "one or more attempts of this turn are not exact",
    )
    assert folded.action_counts[ActionClass.SEARCH] == Metric(6, Precision.EXACT)
    assert folded.critical_tool_contributions_ns == {"shared-operation": 10}
    assert folded.server_critical_contributions_ns == {"shared-server": 12}
    assert folded.token_usage is not None
    assert folded.token_usage.buckets[UsageBucket.INPUT] == Metric(30, Precision.EXACT)
    assert folded.token_usage.buckets[UsageBucket.OUTPUT] == Metric(2, Precision.EXACT)


def test_fold_diagnoses_a_clamped_attempt_axis(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.turn(0, 5 * NS)
    path = tmp_path / "base.jsonl"
    log.write(path)
    base = analyze_trajectory(path).turns[0]
    malformed_ref = replace(
        base.attempts[0],
        physical_axis_start_ns=10 * NS,
        physical_axis_end_ns=5 * NS,
        logical_axis_start_ns=10 * NS,
    )
    first = replace(
        base,
        axis_start_ns=10 * NS,
        axis_end_ns=5 * NS,
        attempts=(malformed_ref,),
    )
    retry_ref = replace(
        base.attempts[0],
        turn_id="5" * 32,
        is_retry=True,
        physical_axis_start_ns=20 * NS,
        physical_axis_end_ns=25 * NS,
        logical_axis_start_ns=20 * NS,
    )
    retry = replace(
        base,
        turn_id=retry_ref.turn_id,
        start_sequence=10,
        axis_start_ns=20 * NS,
        axis_end_ns=25 * NS,
        attempts=(retry_ref,),
    )

    folded = aggregation_module._fold_retry_turns([first, retry])[0]

    assert "folded attempt axis end precedes its start" in folded.diagnostics
    assert [attempt.logical_axis_start_ns for attempt in folded.attempts] == [10 * NS, 10 * NS]


def test_fold_drops_nonfinal_attempt_response_terminal_edges(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.turn(0, 5 * NS)
    path = tmp_path / "base.jsonl"
    log.write(path)
    base = analyze_trajectory(path).turns[0]

    def physical_attempt(turn_id: str, start_ns: int, *, is_retry: bool):
        end_ns = start_ns + 5 * NS
        operation = TimelineOperation(
            operation_id=turn_id,
            family="model.exchange",
            depth=0,
            start_ns=start_ns,
            end_ns=end_ns,
            precision=Precision.EXACT,
        )
        flow = TurnFlow(
            turn_id=turn_id,
            root_index=0,
            has_terminal=True,
            parent_pairs=b"",
            causal_pairs=array("I", (0, FLOW_TERMINAL_INDEX)).tobytes(),
            acyclic=True,
        )
        return replace(
            base,
            turn_id=turn_id,
            start_sequence=1 if not is_retry else 10,
            axis_start_ns=start_ns,
            axis_end_ns=end_ns,
            operations=(operation,),
            flow=flow,
            attempts=(
                replace(
                    base.attempts[0],
                    turn_id=turn_id,
                    is_retry=is_retry,
                    physical_axis_start_ns=start_ns,
                    physical_axis_end_ns=end_ns,
                    logical_axis_start_ns=start_ns,
                    operation_end_index=1,
                ),
            ),
        )

    folded = aggregation_module._fold_retry_turns(
        [
            physical_attempt("4" * 32, 0, is_retry=False),
            physical_attempt("5" * 32, 100 * NS, is_retry=True),
        ]
    )[0]

    assert folded.flow is not None
    assert folded.flow.causal_edges() == ((1, FLOW_TERMINAL_INDEX),)
