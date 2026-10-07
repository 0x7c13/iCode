# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Verification-loop and non-additive submission-latency metrics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from chrys.foundation.trajectory.envelope import Actor, measurement
from chrys.service.analytics import (
    ActionClass,
    ActionOperation,
    ChangeVerification,
    Metric,
    Precision,
    SubmissionLatencyBucket,
    TimelineDiagnosticCode,
    TurnAnalysis,
    TurnAttemptRef,
    WallBucket,
    analyze_trajectory,
)
from chrys.service.analytics._actions import _validation_metrics
from chrys.service.analytics.aggregation import _overview
from tests.service.analytics._events import EventLog

_NS = 1_000_000_000


def test_overview_never_coerces_unavailable_turn_metrics_to_zero() -> None:
    unresolved = Metric(None, Precision.UNRESOLVED, "broken turn")
    turn = replace(
        _turn("turn-1", 1, 0, 100),
        elapsed_ns=unresolved,
        exclusive_work_ns=unresolved,
        overlap_gain_ns=unresolved,
        wall_time_ns=dict.fromkeys(WallBucket, unresolved),
    )

    overview = _overview([turn])

    assert overview.elapsed_ns == unresolved
    assert overview.exclusive_work_ns == unresolved
    assert overview.overlap_gain_ns == unresolved
    assert overview.parallelism.value is None
    assert all(metric.value is None for metric in overview.wall_time_ns.values())
    assert all(metric.value is None for metric in overview.utilization.values())


def test_every_validation_loop_metric_uses_projected_turn_time_and_recorded_outcomes() -> None:
    turns = (_turn("turn-1", 1, 1_000, 1_100), _turn("turn-2", 2, 10, 110))
    actions = (
        _action(1, "turn-1", 1, 1_010, ActionClass.SEARCH),
        _action(2, "turn-1", 1, 1_020, ActionClass.READ),
        _action(3, "turn-1", 1, 1_050, ActionClass.EDIT),
        _action(4, "turn-1", 1, 1_070, ActionClass.VERIFY, outcome="errored", end_ns=1_075),
        _action(5, "turn-1", 1, 1_080, ActionClass.OTHER, outcome="errored", fingerprint="repeated", end_ns=1_085),
        _action(
            6,
            "turn-1",
            1,
            1_090,
            ActionClass.OTHER,
            outcome="errored",
            fingerprint="repeated",
            tool_name="tool-5",
            end_ns=1_095,
        ),
        _action(7, "turn-2", 2, 30, ActionClass.VERIFY, tool_name="tool-4"),
        _action(8, "turn-2", 2, 40, ActionClass.OTHER, tool_name="tool-5"),
        _action(9, "turn-2", 2, 50, ActionClass.EDIT),
    )
    change = ChangeVerification(
        detail_available=True,
        detection_truncated=False,
        files_touched=Metric(1, Precision.EXACT),
        created=Metric(0, Precision.EXACT),
        modified=Metric(1, Precision.EXACT),
        deleted=Metric(0, Precision.EXACT),
        net_zero=Metric(1, Precision.EXACT),
    )

    metrics = _validation_metrics(actions, list(turns), change, Metric(100, Precision.EXACT))

    assert metrics.funnel.search == Metric(1, Precision.EXACT)
    assert metrics.funnel.read == Metric(1, Precision.EXACT)
    assert metrics.funnel.edit == Metric(2, Precision.EXACT)
    assert metrics.funnel.verify == Metric(2, Precision.EXACT)
    assert metrics.time_to_first_edit_ns == Metric(50, Precision.EXACT)
    assert metrics.first_edit_to_first_verify_ns == Metric(20, Precision.EXACT)
    assert metrics.edit_verify_cycle_count == Metric(1, Precision.EXACT)
    assert metrics.edit_verify_cycle_median_ns == Metric(70, Precision.EXACT)
    assert metrics.unverified_change_count == Metric(1, Precision.EXACT)
    assert metrics.net_zero_churn_count == Metric(1, Precision.EXACT)
    assert metrics.repeated_failure_signature_count == Metric(1, Precision.EXACT)
    assert metrics.failure_recovery_median_ns == Metric(45, Precision.EXACT)
    assert metrics.tool_failure_count == Metric(3, Precision.EXACT)
    assert metrics.tool_count == Metric(9, Precision.EXACT)
    assert metrics.retry_amplification_tokens == Metric(100, Precision.EXACT)


def test_verify_started_before_the_edit_terminal_never_vouches_for_it() -> None:
    turns = [_turn("turn-1", 1, 0, 100)]
    edit = _action(1, "turn-1", 1, 10, ActionClass.EDIT, end_sequence=4)
    overlapped_verify = _action(2, "turn-1", 1, 20, ActionClass.VERIFY)
    ordered_verify = _action(5, "turn-1", 1, 50, ActionClass.VERIFY)
    retry = Metric(0, Precision.EXACT)

    overlapped = _validation_metrics((edit, overlapped_verify), turns, _empty_change(), retry)
    ordered = _validation_metrics((edit, ordered_verify), turns, _empty_change(), retry)
    unlanded = _validation_metrics((replace(edit, end_sequence=None), ordered_verify), turns, _empty_change(), retry)

    assert overlapped.unverified_change_count == Metric(1, Precision.EXACT)
    assert ordered.unverified_change_count == Metric(0, Precision.EXACT)
    assert unlanded.unverified_change_count == Metric(1, Precision.EXACT)
    # The cycle metrics must agree with the unverified count: an overlapping
    # verify closes no cycle either.
    assert overlapped.edit_verify_cycle_count.value == 0
    assert ordered.edit_verify_cycle_count.value == 1
    assert unlanded.edit_verify_cycle_count.value == 0
    # And the first-verify metric: a verify that began before the edit
    # terminal landed is not the first verify of that edit.
    missing = Metric(None, Precision.MISSING, "no verify action followed the first edit")
    assert overlapped.first_edit_to_first_verify_ns == missing
    assert ordered.first_edit_to_first_verify_ns == Metric(40, Precision.EXACT)
    assert unlanded.first_edit_to_first_verify_ns == missing


def test_estimated_other_shell_actions_degrade_the_unverified_absence() -> None:
    """An estimated other-classified shell action may really have been
    verification work, so no metric that rests on the absence of a verify
    can claim more than the word-list heuristic proves."""
    turns = [_turn("turn-1", 1, 0, 100)]
    edit = _action(1, "turn-1", 1, 10, ActionClass.EDIT)
    lint = _action(
        5,
        "turn-1",
        1,
        50,
        ActionClass.OTHER,
        tool_kind="shell",
        classification_precision=Precision.ESTIMATED,
    )

    metrics = _validation_metrics((edit, lint), turns, _empty_change(), Metric(0, Precision.EXACT))

    reason = "one or more shell actions use the verification word-list heuristic"
    assert metrics.unverified_change_count == Metric(1, Precision.ESTIMATED, reason)
    assert metrics.first_edit_to_first_verify_ns == Metric(None, Precision.UNRESOLVED, reason)
    assert metrics.edit_verify_cycle_count == Metric(0, Precision.ESTIMATED, reason)
    assert metrics.edit_verify_cycle_median_ns == Metric(None, Precision.UNRESOLVED, reason)


def test_verify_overlapping_any_pending_edit_leaves_the_batch_open() -> None:
    """A cycle is a completed edit-batch → verify iteration: a verify that
    began while any pending edit was still landing closes nothing, and the
    whole batch stays pending for a later ordered verify."""
    turns = [_turn("turn-1", 1, 0, 100)]
    first = _action(1, "turn-1", 1, 10, ActionClass.EDIT, end_sequence=2)
    second = _action(3, "turn-1", 1, 30, ActionClass.EDIT, end_sequence=6)
    partial_verify = _action(4, "turn-1", 1, 40, ActionClass.VERIFY)
    covering_verify = _action(7, "turn-1", 1, 70, ActionClass.VERIFY)
    retry = Metric(0, Precision.EXACT)

    partial = _validation_metrics((first, second, partial_verify), turns, _empty_change(), retry)
    covered = _validation_metrics((first, second, partial_verify, covering_verify), turns, _empty_change(), retry)

    assert partial.edit_verify_cycle_count.value == 0
    assert partial.unverified_change_count.value == 1
    assert covered.edit_verify_cycle_count.value == 1
    assert covered.unverified_change_count.value == 0


def test_intervening_unknown_outcome_call_unresolves_the_recovery_latency() -> None:
    """A same-tool call with an unknown outcome between the failure and the
    recognized success may itself have been the recovery, so the measured
    latency is unresolved; a recorded failed retry proves nothing recovered
    and leaves the latency exact."""
    turns = [_turn("turn-1", 1, 0, 100)]
    fail = _action(1, "turn-1", 1, 10, ActionClass.OTHER, outcome="errored", tool_name="build")
    unknown = _action(2, "turn-1", 1, 30, ActionClass.OTHER, outcome="unknown", tool_name="build")
    failed_retry = _action(2, "turn-1", 1, 30, ActionClass.OTHER, outcome="errored", tool_name="build")
    success = _action(3, "turn-1", 1, 50, ActionClass.OTHER, tool_name="build")
    retry = Metric(0, Precision.EXACT)

    with_unknown = _validation_metrics((fail, unknown, success), turns, _empty_change(), retry)
    with_failed_retry = _validation_metrics((fail, failed_retry, success), turns, _empty_change(), retry)

    reason = "a same-tool call with an unknown outcome preceded the recognized recovery"
    assert with_unknown.failure_recovery_median_ns == Metric(35, Precision.UNRESOLVED, reason)
    assert with_failed_retry.failure_recovery_median_ns == Metric(25, Precision.EXACT)


def test_concurrent_same_tool_success_is_not_a_recovery() -> None:
    """A recovery responds to an observed failure: a same-tool success that
    started while the failing call was still running proves nothing, and a
    failure whose terminal never landed cannot order any later success."""
    turns = [_turn("turn-1", 1, 0, 100)]
    fail = _action(
        1, "turn-1", 1, 10, ActionClass.OTHER, outcome="errored", tool_name="build", end_ns=50, end_sequence=5
    )
    concurrent = _action(2, "turn-1", 1, 20, ActionClass.OTHER, tool_name="build", end_ns=30, end_sequence=3)
    retry_success = _action(6, "turn-1", 1, 70, ActionClass.OTHER, tool_name="build")
    retry = Metric(0, Precision.EXACT)

    ordered = _validation_metrics((fail, concurrent, retry_success), turns, _empty_change(), retry)
    unlanded = _validation_metrics(
        (replace(fail, end_sequence=None), concurrent, retry_success), turns, _empty_change(), retry
    )

    assert ordered.failure_recovery_median_ns == Metric(20, Precision.EXACT)
    assert unlanded.failure_recovery_median_ns == Metric(
        None, Precision.UNRESOLVED, "the failed call's terminal never landed"
    )


def test_missing_exit_code_does_not_rewrite_the_recorded_tool_outcome() -> None:
    turn = _turn("turn-1", 1, 0, 100)
    failed = _action(1, "turn-1", 1, 10, ActionClass.OTHER, outcome="errored")

    metrics = _validation_metrics(
        (failed,),
        [turn],
        _empty_change(),
        Metric(0, Precision.EXACT),
    )

    assert metrics.tool_failure_count == Metric(1, Precision.EXACT)


def test_retry_amplification_is_unresolved_when_retry_started_has_no_schedule(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "retry.started",
        1,
        operation_id="a" * 32,
        payload={"retry_mode": "run"},
    )
    log.add("turn.finished", 2, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)

    validation = analyze_trajectory(path).validation

    assert validation is not None
    assert validation.retry_amplification_tokens == Metric(
        None,
        Precision.UNRESOLVED,
        "retry lifecycle attribution is incomplete",
    )


def test_retry_cut_owned_exclusively_by_inactive_turn_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    removed_turn_id = "5" * 32
    retry_id = "a" * 32
    superseded_from = log.next_sequence
    log.add("turn.started", 0, turn_id=removed_turn_id, payload={"turn_number": 1})
    log.add(
        "retry.scheduled",
        _NS,
        turn_id=removed_turn_id,
        operation_id=retry_id,
        payload={"retry_mode": "run"},
    )
    log.add(
        "turn.finished",
        2 * _NS,
        turn_id=removed_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 2000},
    )
    log.resolved_rollback(superseded_from, 3 * _NS)
    log.add(
        "retry.started",
        4 * _NS,
        turn_id=removed_turn_id,
        operation_id=retry_id,
        branch_id="6" * 32,
        payload={"retry_mode": "run"},
    )
    log.write(path)

    validation = analyze_trajectory(path).validation

    assert validation is not None
    assert validation.retry_amplification_tokens == Metric(0, Precision.EXACT)


def test_retry_cut_owned_by_active_turn_remains_unresolved(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    retained_turn_id = "4" * 32
    removed_turn_id = "5" * 32
    retry_id = "a" * 32
    log.add("turn.started", 0, turn_id=retained_turn_id, payload={"turn_number": 1})
    log.add(
        "retry.scheduled",
        _NS,
        turn_id=retained_turn_id,
        operation_id=retry_id,
        payload={"retry_mode": "run"},
    )
    log.add(
        "turn.finished",
        2 * _NS,
        turn_id=retained_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 2000},
    )
    superseded_from = log.next_sequence
    log.add("turn.started", 3 * _NS, turn_id=removed_turn_id, payload={"turn_number": 2})
    log.add(
        "retry.started",
        4 * _NS,
        turn_id=retained_turn_id,
        operation_id=retry_id,
        payload={"retry_mode": "run"},
    )
    log.add(
        "turn.finished",
        5 * _NS,
        turn_id=removed_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 2000},
    )
    log.resolved_rollback(superseded_from, 6 * _NS, target_turn_id=retained_turn_id)
    log.write(path)

    validation = analyze_trajectory(path).validation

    assert validation is not None
    assert validation.retry_amplification_tokens == Metric(
        None,
        Precision.UNRESOLVED,
        "retry lifecycle attribution is incomplete",
    )


def test_side_call_lifecycle_cuts_do_not_degrade_main_retry_attribution(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    retained_turn_id = "4" * 32
    removed_turn_id = "5" * 32
    main_retry_id = "a" * 32
    side_actor = Actor(kind="side_call", role="approval_judge", actor_id="0" * 32)
    log.add("turn.started", 0, turn_id=retained_turn_id, payload={"turn_number": 1})
    log.add(
        "retry.scheduled",
        _NS,
        turn_id=retained_turn_id,
        operation_id=main_retry_id,
        payload={"retry_mode": "run"},
    )
    log.add(
        "retry.started",
        2 * _NS,
        turn_id=retained_turn_id,
        operation_id=main_retry_id,
        payload={"retry_mode": "run"},
    )
    log.add(
        "turn.finished",
        3 * _NS,
        turn_id=retained_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 3000},
    )
    side_nodes = (
        ("retry.scheduled", "b" * 32, {"retry_mode": "run"}),
        ("model.run.started", "c" * 32, {}),
        ("model.cycle.started", "d" * 32, {}),
        ("model.exchange.started", "e" * 32, {}),
    )
    for offset, (event_type, operation_id, payload) in enumerate(side_nodes, start=4):
        log.add(
            event_type,
            offset * _NS,
            turn_id=None,
            operation_id=operation_id,
            actor=side_actor,
            payload=payload,
        )
    superseded_from = log.next_sequence
    log.add("turn.started", 8 * _NS, turn_id=removed_turn_id, payload={"turn_number": 2})
    side_terminals = (
        ("retry.started", "b" * 32, {"retry_mode": "run"}),
        ("model.run.finished", "c" * 32, {"outcome": "success"}),
        ("model.cycle.finished", "d" * 32, {"outcome": "success"}),
        ("model.exchange.finished", "e" * 32, {"outcome": "success"}),
    )
    for offset, (event_type, operation_id, payload) in enumerate(side_terminals, start=9):
        log.add(
            event_type,
            offset * _NS,
            turn_id=None,
            operation_id=operation_id,
            actor=side_actor,
            payload=payload,
        )
    log.add(
        "turn.finished",
        13 * _NS,
        turn_id=removed_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 5000},
    )
    log.resolved_rollback(superseded_from, 14 * _NS, target_turn_id=retained_turn_id)
    log.write(path)

    validation = analyze_trajectory(path).validation

    assert validation is not None
    assert validation.retry_amplification_tokens == Metric(0, Precision.EXACT)


def test_unknown_outcome_is_neither_success_nor_failure_and_degrades_outcome_metrics() -> None:
    turn = _turn("turn-1", 1, 0, 100)
    unknown = _action(1, "turn-1", 1, 10, ActionClass.VERIFY, outcome="unknown")

    metrics = _validation_metrics(
        (unknown,),
        [turn],
        _empty_change(),
        Metric(0, Precision.EXACT),
    )

    reason = "one or more tool outcomes are unknown"
    assert metrics.tool_failure_count == Metric(0, Precision.UNRESOLVED, reason)
    assert metrics.repeated_failure_signature_count == Metric(0, Precision.UNRESOLVED, reason)
    assert metrics.unverified_change_count.precision is Precision.UNRESOLVED
    assert metrics.failure_recovery_median_ns == Metric(None, Precision.UNRESOLVED, reason)


def test_missing_and_invalid_recorded_outcomes_are_not_marked_exact(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    for index, outcome in enumerate((None, "not-a-tool-outcome"), start=1):
        operation_id = f"{index:032x}"
        log.add(
            "tool.operation.started",
            index * 10,
            operation_id=operation_id,
            payload={"tool_name": "read", "tool_kind": "filesystem.read"},
        )
        payload: dict[str, object] = {"duration_ms": 1}
        if outcome is not None:
            payload["outcome"] = outcome
        log.add(
            "tool.operation.finished",
            index * 10 + 1,
            operation_id=operation_id,
            payload=payload,
            measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
        )
    log.add("turn.finished", 100, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)

    missing, invalid = analyze_trajectory(path).actions

    assert missing.outcome_precision is Precision.MISSING
    assert missing.outcome_reason == "tool terminal outcome is missing"
    assert invalid.outcome == "not-a-tool-outcome"
    assert invalid.outcome_precision is Precision.UNRESOLVED
    assert invalid.outcome_reason == "tool terminal outcome is invalid"


def test_cross_scope_tool_terminal_cannot_supply_an_outcome_or_duration(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    operation_id = "a" * 32
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "tool.operation.started",
        10,
        operation_id=operation_id,
        payload={"tool_name": "read", "tool_kind": "filesystem.read"},
    )
    log.add(
        "tool.operation.finished",
        20,
        operation_id=operation_id,
        payload={"outcome": "errored", "duration_ms": 0},
        branch_id="5" * 32,
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add("turn.finished", 30, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)

    analysis = analyze_trajectory(path)
    (action,) = analysis.actions

    assert action.outcome is None
    assert action.outcome_precision is Precision.UNRESOLVED
    assert action.outcome_reason == "tool lifecycle endpoints cross scope"
    assert action.end_ns is None
    assert analysis.validation is not None
    assert analysis.validation.tool_failure_count.value == 0
    assert analysis.validation.tool_failure_count.precision is Precision.UNRESOLVED


def test_duplicate_tool_terminals_are_pairing_uncertainty_not_missing_evidence(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    operation_id = "a" * 32
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "tool.operation.started",
        10,
        operation_id=operation_id,
        payload={"tool_name": "read", "tool_kind": "filesystem.read"},
    )
    for monotonic_ns in (20, 21):
        log.add(
            "tool.operation.finished",
            monotonic_ns,
            operation_id=operation_id,
            payload={"outcome": "success", "duration_ms": 0},
            measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
        )
    log.add("turn.finished", 30, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)

    (action,) = analyze_trajectory(path).actions

    assert action.outcome is None
    assert action.outcome_precision is Precision.UNRESOLVED
    assert action.outcome_reason == "tool operation has more than one terminal event"
    assert action.end_ns is None


def test_corrupt_turn_region_degrades_action_projection_and_all_validation_metrics(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "tool.operation",
        "a" * 32,
        10,
        20,
        start_payload={"tool_name": "read", "tool_kind": "filesystem.read"},
    )
    log.add("turn.finished", 30, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)
    lines = path.read_bytes().splitlines(keepends=True)
    lines.insert(3, b"{corrupt action projection}\n")
    path.write_bytes(b"".join(lines))

    analysis = analyze_trajectory(path)
    (action,) = analysis.actions
    validation = analysis.validation

    assert action.outcome is None
    assert action.outcome_precision is Precision.UNRESOLVED
    assert validation is not None
    assert validation.funnel.read == Metric(1, Precision.UNRESOLVED, "corrupt trajectory line intersects the turn")
    assert validation.tool_count == Metric(1, Precision.UNRESOLVED, "corrupt trajectory line intersects the turn")
    assert all(
        metric.precision is Precision.UNRESOLVED
        for metric in (
            validation.time_to_first_edit_ns,
            validation.first_edit_to_first_verify_ns,
            validation.edit_verify_cycle_count,
            validation.edit_verify_cycle_median_ns,
            validation.unverified_change_count,
            validation.net_zero_churn_count,
            validation.repeated_failure_signature_count,
            validation.failure_recovery_median_ns,
            validation.tool_failure_count,
            validation.retry_amplification_tokens,
        )
    )


def test_missing_shell_command_makes_verify_dependent_metrics_unresolved() -> None:
    turn = _turn("turn-1", 1, 0, 100)
    edit = _action(1, "turn-1", 1, 10, ActionClass.EDIT, tool_kind="filesystem.write")
    degraded_shell = _action(
        2,
        "turn-1",
        1,
        20,
        ActionClass.OTHER,
        tool_kind="shell",
        classification_precision=Precision.UNRESOLVED,
        classification_reason="shell command carrier is unavailable",
    )

    metrics = _validation_metrics(
        (edit, degraded_shell),
        [turn],
        _empty_change(),
        Metric(0, Precision.EXACT),
    )

    assert metrics.funnel.verify == Metric(
        0, Precision.UNRESOLVED, "one or more shell command carriers are unavailable"
    )
    assert metrics.first_edit_to_first_verify_ns.precision is Precision.UNRESOLVED
    assert metrics.edit_verify_cycle_count.precision is Precision.UNRESOLVED
    assert metrics.edit_verify_cycle_median_ns.precision is Precision.UNRESOLVED
    assert metrics.unverified_change_count == Metric(
        1,
        Precision.UNRESOLVED,
        "one or more shell command carriers are unavailable",
    )


def test_submission_latency_uses_three_buckets_frozen_endpoints_and_excludes_unresolved_from_percentiles(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    _pre_turn(log, "a" * 32, 0, _NS, "fresh_turn")
    log.add(
        "turn.started",
        4 * _NS,
        payload={"turn_number": 1, "preparation_scope_operation_id": "a" * 32},
    )
    _pre_turn(log, "b" * 32, 5 * _NS, 8 * _NS, "injected", target_turn_id="4" * 32)
    _pre_turn(log, "c" * 32, 9 * _NS, 10 * _NS, "rejected")
    _pre_turn(log, "d" * 32, 11 * _NS, 12 * _NS, "fresh_turn")
    log.add("turn.finished", 20 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)

    submission = analyze_trajectory(path).submission_latency
    assert submission is not None
    by_bucket = {stats.bucket: stats for stats in submission.buckets}

    became = by_bucket[SubmissionLatencyBucket.BECAME_TURN]
    assert became.sample_count == 2
    assert became.unresolved_count == 1
    assert became.p50_ns == Metric(4 * _NS, Precision.EXACT)
    assert became.p90_ns == Metric(4 * _NS, Precision.EXACT)
    assert became.max_ns == Metric(4 * _NS, Precision.EXACT)
    injected = by_bucket[SubmissionLatencyBucket.INJECTED]
    assert injected.sample_count == 1
    assert injected.samples[0].duration_ns == Metric(3 * _NS, Precision.EXACT)
    assert injected.samples[0].end_ns == 8 * _NS
    did_not = by_bucket[SubmissionLatencyBucket.DID_NOT_BECOME_TURN]
    assert did_not.sample_count == 1
    assert did_not.samples[0].duration_ns == Metric(_NS, Precision.EXACT)
    assert all(
        [sample.start_sequence for sample in stats.samples] == sorted(sample.start_sequence for sample in stats.samples)
        for stats in submission.buckets
    )


def test_session_damage_caps_submission_percentiles_without_degrading_intact_samples(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    _pre_turn(log, "a" * 32, 0, _NS, "fresh_turn")
    log.add(
        "turn.started",
        4 * _NS,
        payload={"turn_number": 1, "preparation_scope_operation_id": "a" * 32},
    )
    log.add("turn.finished", 5 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)
    path.write_bytes(path.read_bytes() + b"{corrupt after the turn}\n")

    submission = analyze_trajectory(path).submission_latency
    assert submission is not None
    became_turn = next(stats for stats in submission.buckets if stats.bucket is SubmissionLatencyBucket.BECAME_TURN)
    reason = "session trajectory integrity is unresolved: corrupt lines"

    assert became_turn.sample_count == 1
    assert became_turn.unresolved_count == 0
    assert became_turn.samples[0].duration_ns == Metric(4 * _NS, Precision.EXACT)
    assert became_turn.p50_ns == Metric(4 * _NS, Precision.UNRESOLVED, reason)
    assert became_turn.p90_ns == Metric(4 * _NS, Precision.UNRESOLVED, reason)
    assert became_turn.max_ns == Metric(4 * _NS, Precision.UNRESOLVED, reason)


def test_unrecognized_preparation_outcome_is_not_read_as_exact_evidence(tmp_path: Path) -> None:
    """An outcome outside this build's vocabulary is damaged or from a
    different writer; the sample stays countable in the fallback bucket, but
    its latency cannot pass for read evidence."""
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    _pre_turn(log, "a" * 32, 0, _NS, "future_outcome")
    log.write(path)

    submission = analyze_trajectory(path).submission_latency
    assert submission is not None
    by_bucket = {stats.bucket: stats for stats in submission.buckets}
    did_not = by_bucket[SubmissionLatencyBucket.DID_NOT_BECOME_TURN]

    assert did_not.sample_count == 1
    assert did_not.samples[0].duration_ns == Metric(
        None, Precision.UNRESOLVED, "preparation terminal outcome is unrecognized"
    )


def test_every_resolved_submission_sample_has_end_not_before_start(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    _pre_turn(log, "a" * 32, _NS, 2 * _NS, "injected", target_turn_id="4" * 32)
    log.add("turn.finished", 3 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)

    submission = analyze_trajectory(path).submission_latency
    assert submission is not None
    samples = [sample for stats in submission.buckets for sample in stats.samples]

    assert samples
    assert all(sample.end_ns is not None and sample.start_ns is not None for sample in samples)
    assert all(
        sample.end_ns >= sample.start_ns
        for sample in samples
        if sample.end_ns is not None and sample.start_ns is not None
    )


def test_each_pre_turn_scope_has_exactly_one_finished_event(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    _pre_turn(log, "a" * 32, 0, _NS, "rejected")
    _pre_turn(log, "b" * 32, 2 * _NS, 4 * _NS, "cancelled")
    log.write(path)

    submission = analyze_trajectory(path).submission_latency
    assert submission is not None
    samples = [sample for stats in submission.buckets for sample in stats.samples]
    cancelled = next(sample for sample in samples if sample.outcome == "cancelled")

    assert samples
    assert all(sample.finished_count == 1 for sample in samples)
    assert cancelled.bucket is SubmissionLatencyBucket.DID_NOT_BECOME_TURN


def test_duplicate_preparation_terminal_is_retained_as_an_unresolved_sample(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    operation_id = "a" * 32
    log.add(
        "preparation.started",
        0,
        turn_id=None,
        operation_id=operation_id,
        payload={"scope": "pre_turn", "phase": "admission"},
    )
    for end_ns in (_NS, 2 * _NS):
        log.add(
            "preparation.finished",
            end_ns,
            turn_id=None,
            operation_id=operation_id,
            payload={"scope": "pre_turn", "outcome": "rejected", "duration_ms": 1_000},
            measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
        )
    log.write(path)

    submission = analyze_trajectory(path).submission_latency
    assert submission is not None
    sample = next(sample for stats in submission.buckets for sample in stats.samples)

    assert sample.finished_count == 2
    assert sample.duration_ns.precision is Precision.UNRESOLVED
    assert "exactly one finished" in (sample.duration_ns.reason or "")


def test_superseded_submission_cut_drops_without_hiding_retained_turn_cleanup(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    retained_turn_id = "4" * 32
    removed_turn_id = "5" * 32
    preparation_id = "a" * 32
    cleanup_hook_id = "b" * 32
    log.add("turn.started", 0, turn_id=retained_turn_id, payload={"turn_number": 1})
    log.add(
        "turn.finished",
        _NS,
        turn_id=retained_turn_id,
        payload={"end_reason": "completed", "duration_ms": 1000},
    )
    log.add(
        "preparation.started",
        2 * _NS,
        turn_id=None,
        operation_id=preparation_id,
        payload={"scope": "pre_turn", "phase": "input_admission"},
    )
    log.add(
        "hook.operation.started",
        3 * _NS,
        turn_id=retained_turn_id,
        operation_id=cleanup_hook_id,
        payload={"hook_event": "after_turn", "execution_mode": "async", "scope": "turn"},
    )
    log.add(
        "hook.operation.finished",
        4 * _NS,
        turn_id=retained_turn_id,
        operation_id=cleanup_hook_id,
        payload={"outcome": "success", "duration_ms": 1000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.settled(5 * _NS, turn_id=retained_turn_id, waited_hook_ids=[cleanup_hook_id])
    superseded_from = log.next_sequence
    log.add(
        "turn.started",
        6 * _NS,
        turn_id=removed_turn_id,
        payload={"turn_number": 2, "preparation_scope_operation_id": preparation_id},
    )
    log.add(
        "preparation.finished",
        7 * _NS,
        turn_id=None,
        operation_id=preparation_id,
        payload={"scope": "pre_turn", "outcome": "fresh_turn", "duration_ms": 5000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(
        "turn.finished",
        8 * _NS,
        turn_id=removed_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 2000},
    )
    log.resolved_rollback(superseded_from, 9 * _NS, target_turn_id=retained_turn_id)
    log.write(path)

    analysis = analyze_trajectory(path)

    assert analysis.diagnostics.rollback_projection_unresolved is False
    assert [turn.turn_id for turn in analysis.turns] == [retained_turn_id]
    retained = analysis.turns[0]
    assert not any("rollback projection" in reason for reason in retained.diagnostics)
    cleanup = next(operation for operation in retained.operations if operation.operation_id == cleanup_hook_id)
    assert cleanup.precision is Precision.EXACT
    assert analysis.submission_latency is not None
    assert sum(bucket.sample_count for bucket in analysis.submission_latency.buckets) == 0


def test_submission_cut_with_active_and_inactive_turn_owners_stays_unresolved(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    preparation_id = "a" * 32
    retained_turn_id = "4" * 32
    removed_turn_id = "5" * 32
    log.add(
        "preparation.started",
        0,
        turn_id=None,
        operation_id=preparation_id,
        payload={"scope": "pre_turn", "phase": "input_admission"},
    )
    log.add(
        "turn.started",
        _NS,
        turn_id=retained_turn_id,
        payload={"turn_number": 1, "preparation_scope_operation_id": preparation_id},
    )
    log.add(
        "turn.finished",
        2 * _NS,
        turn_id=retained_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1000},
    )
    superseded_from = log.next_sequence
    log.add(
        "turn.started",
        3 * _NS,
        turn_id=removed_turn_id,
        payload={"turn_number": 2, "preparation_scope_operation_id": preparation_id},
    )
    log.add(
        "preparation.finished",
        4 * _NS,
        turn_id=None,
        operation_id=preparation_id,
        payload={"scope": "pre_turn", "outcome": "fresh_turn", "duration_ms": 4000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(
        "turn.finished",
        5 * _NS,
        turn_id=removed_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 2000},
    )
    log.resolved_rollback(superseded_from, 6 * _NS, target_turn_id=retained_turn_id)
    log.write(path)

    submission = analyze_trajectory(path).submission_latency

    assert submission is not None
    samples = [sample for bucket in submission.buckets for sample in bucket.samples]
    assert len(samples) == 1
    assert samples[0].scope_operation_id == preparation_id
    assert samples[0].duration_ns == Metric(
        None,
        Precision.UNRESOLVED,
        "preparation scope does not have exactly one finished event",
    )


def test_ownerless_rejected_cut_drops_without_hiding_promoted_scope_missing_its_claim(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    rejected_id = "a" * 32
    damaged_fresh_id = "b" * 32
    removed_turn_id = "5" * 32
    for operation_id in (rejected_id, damaged_fresh_id):
        log.add(
            "preparation.started",
            _NS,
            turn_id=None,
            operation_id=operation_id,
            payload={"scope": "pre_turn", "phase": "input_admission"},
        )
    superseded_from = log.next_sequence
    log.add("turn.started", 2 * _NS, turn_id=removed_turn_id, payload={"turn_number": 1})
    for operation_id, outcome in ((rejected_id, "rejected"), (damaged_fresh_id, "fresh_turn")):
        log.add(
            "preparation.finished",
            3 * _NS,
            turn_id=None,
            operation_id=operation_id,
            payload={"scope": "pre_turn", "outcome": outcome, "duration_ms": 2000},
            measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
        )
    log.add(
        "turn.finished",
        4 * _NS,
        turn_id=removed_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 2000},
    )
    log.resolved_rollback(superseded_from, 5 * _NS)
    log.write(path)

    submission = analyze_trajectory(path).submission_latency

    assert submission is not None
    samples = [sample for bucket in submission.buckets for sample in bucket.samples]
    assert [sample.scope_operation_id for sample in samples] == [damaged_fresh_id]
    assert samples[0].duration_ns == Metric(
        None,
        Precision.UNRESOLVED,
        "preparation scope does not have exactly one finished event",
    )


def test_injected_cut_uses_explicit_target_turn_ownership(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    active_target_id = "4" * 32
    inactive_target_id = "5" * 32
    active_preparation_id = "a" * 32
    inactive_preparation_id = "b" * 32
    log.add("turn.started", 0, turn_id=active_target_id, payload={"turn_number": 1})
    log.add(
        "preparation.started",
        _NS,
        turn_id=None,
        operation_id=active_preparation_id,
        payload={"scope": "pre_turn", "phase": "input_admission"},
    )
    superseded_from = log.next_sequence
    log.add("turn.started", 2 * _NS, turn_id=inactive_target_id, payload={"turn_number": 2})
    log.add(
        "preparation.started",
        3 * _NS,
        turn_id=None,
        operation_id=inactive_preparation_id,
        payload={"scope": "pre_turn", "phase": "input_admission"},
    )
    log.add(
        "preparation.finished",
        4 * _NS,
        turn_id=None,
        operation_id=active_preparation_id,
        payload={
            "scope": "pre_turn",
            "outcome": "injected",
            "target_turn_id": active_target_id,
            "duration_ms": 3000,
        },
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(
        "turn.finished",
        5 * _NS,
        turn_id=inactive_target_id,
        payload={"end_reason": "cancelled", "duration_ms": 3000},
    )
    log.resolved_rollback(superseded_from, 6 * _NS, target_turn_id=active_target_id)
    log.add(
        "preparation.finished",
        7 * _NS,
        turn_id=None,
        operation_id=inactive_preparation_id,
        branch_id="6" * 32,
        payload={
            "scope": "pre_turn",
            "outcome": "injected",
            "target_turn_id": inactive_target_id,
            "duration_ms": 4000,
        },
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.write(path)

    submission = analyze_trajectory(path).submission_latency

    assert submission is not None
    samples = [sample for bucket in submission.buckets for sample in bucket.samples]
    assert [sample.scope_operation_id for sample in samples] == [active_preparation_id]
    assert samples[0].duration_ns == Metric(
        None,
        Precision.UNRESOLVED,
        "preparation scope does not have exactly one finished event",
    )


def test_missing_or_duplicate_raw_preparation_endpoints_are_not_dropped_as_cuts(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    missing_id = "a" * 32
    duplicate_id = "b" * 32
    log.add(
        "preparation.started",
        0,
        turn_id=None,
        operation_id=missing_id,
        payload={"scope": "pre_turn", "phase": "input_admission"},
    )
    log.add(
        "preparation.started",
        _NS,
        turn_id=None,
        operation_id=duplicate_id,
        payload={"scope": "pre_turn", "phase": "input_admission"},
    )
    superseded_from = log.next_sequence
    for end_ns in (2 * _NS, 3 * _NS):
        log.add(
            "preparation.finished",
            end_ns,
            turn_id=None,
            operation_id=duplicate_id,
            payload={"scope": "pre_turn", "outcome": "rejected", "duration_ms": 1000},
            measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
        )
    log.resolved_rollback(superseded_from, 4 * _NS)
    log.write(path)

    submission = analyze_trajectory(path).submission_latency

    assert submission is not None
    samples = {sample.scope_operation_id: sample for bucket in submission.buckets for sample in bucket.samples}
    assert samples[missing_id].duration_ns == Metric(
        None,
        Precision.UNRESOLVED,
        "preparation scope does not have exactly one finished event",
    )
    assert samples[duplicate_id].duration_ns == Metric(
        None,
        Precision.UNRESOLVED,
        "preparation scope does not have exactly one finished event",
    )


def test_finish_surviving_submission_cut_owned_by_inactive_turn_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    preparation_id = "a" * 32
    removed_turn_id = "5" * 32
    superseded_from = log.next_sequence
    log.add(
        "preparation.started",
        0,
        turn_id=None,
        operation_id=preparation_id,
        payload={"scope": "pre_turn", "phase": "input_admission"},
    )
    log.add(
        "turn.started",
        _NS,
        turn_id=removed_turn_id,
        payload={"turn_number": 1, "preparation_scope_operation_id": preparation_id},
    )
    log.add(
        "turn.finished",
        2 * _NS,
        turn_id=removed_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1000},
    )
    log.resolved_rollback(superseded_from, 3 * _NS)
    log.add(
        "preparation.finished",
        4 * _NS,
        turn_id=None,
        operation_id=preparation_id,
        branch_id="6" * 32,
        payload={"scope": "pre_turn", "outcome": "fresh_turn", "duration_ms": 4000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.write(path)

    submission = analyze_trajectory(path).submission_latency

    assert submission is not None
    assert sum(bucket.sample_count for bucket in submission.buckets) == 0


def test_hook_cut_in_retained_turn_is_diagnosed_in_structure_projection(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    retained_turn_id = "4" * 32
    removed_turn_id = "5" * 32
    hook_id = "a" * 32
    log.add("turn.started", 0, turn_id=retained_turn_id, payload={"turn_number": 1})
    log.add(
        "turn.finished",
        _NS,
        turn_id=retained_turn_id,
        payload={"end_reason": "completed", "duration_ms": 1000},
    )
    log.add(
        "hook.operation.started",
        2 * _NS,
        turn_id=retained_turn_id,
        operation_id=hook_id,
        payload={"hook_event": "after_turn", "execution_mode": "fire_and_forget", "scope": "turn"},
    )
    log.settled(3 * _NS, turn_id=retained_turn_id)
    superseded_from = log.next_sequence
    log.add("turn.started", 4 * _NS, turn_id=removed_turn_id, payload={"turn_number": 2})
    log.add(
        "hook.operation.finished",
        5 * _NS,
        turn_id=retained_turn_id,
        operation_id=hook_id,
        payload={"outcome": "success", "duration_ms": 3000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(
        "turn.finished",
        6 * _NS,
        turn_id=removed_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 2000},
    )
    log.resolved_rollback(superseded_from, 7 * _NS, target_turn_id=retained_turn_id)
    log.write(path)

    analysis = analyze_trajectory(path)
    retained = analysis.turns[0]

    reason = "hook.operation lifecycle crosses rollback projection; only its start endpoint remains active"
    assert reason in retained.diagnostics
    hook = next(operation for operation in retained.operations if operation.operation_id == hook_id)
    assert hook.precision is Precision.UNRESOLVED
    assert hook.reason == reason


def test_detached_hook_projects_spawn_only_duration_diagnostic(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    hook_id = "a" * 32
    hook_config_id = "register-session-to-git-after-turn"
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("turn.finished", _NS, payload={"end_reason": "completed", "duration_ms": 1000})
    log.span(
        "hook.operation",
        hook_id,
        2 * _NS,
        3 * _NS,
        start_payload={
            "hook_event": "after_turn",
            "hook_key": hook_config_id,
            "execution_mode": "fire_and_forget",
            "scope": "turn",
        },
        finish_payload={"outcome": "detached"},
    )
    log.settled(4 * _NS, waited_hook_ids=[hook_id])
    log.write(path)

    analysis = analyze_trajectory(path)
    hook = next(operation for operation in analysis.turns[0].operations if operation.operation_id == hook_id)
    diagnostic = next(item for item in analysis.diagnostics.timeline_operations if item.operation_id == hook_id)

    assert hook.start_ns is None
    assert hook.end_ns is None
    assert hook.precision is Precision.MISSING
    assert hook.diagnostic_code is TimelineDiagnosticCode.DETACHED_HOOK
    assert hook.identity == "after_turn"
    assert hook.hook_id == hook_config_id
    assert diagnostic.code is TimelineDiagnosticCode.DETACHED_HOOK
    assert diagnostic.reason == "detached hook records spawn latency, not work duration"
    assert diagnostic.identity == "after_turn"
    assert diagnostic.hook_id == hook_config_id


def test_hook_start_cut_out_of_retained_turn_keeps_surviving_terminal_diagnosable(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    retained_turn_id = "4" * 32
    removed_turn_id = "5" * 32
    hook_id = "a" * 32
    hook_config_id = "register-session-to-git-after-turn"
    log.add("turn.started", 0, turn_id=retained_turn_id, payload={"turn_number": 1})
    log.add(
        "turn.finished",
        _NS,
        turn_id=retained_turn_id,
        payload={"end_reason": "completed", "duration_ms": 1000},
    )
    superseded_from = log.next_sequence
    log.add(
        "hook.operation.started",
        2 * _NS,
        turn_id=retained_turn_id,
        operation_id=hook_id,
        payload={
            "hook_event": "after_turn",
            "hook_key": hook_config_id,
            "execution_mode": "fire_and_forget",
            "scope": "turn",
        },
    )
    log.add("turn.started", 3 * _NS, turn_id=removed_turn_id, payload={"turn_number": 2})
    log.add(
        "turn.finished",
        4 * _NS,
        turn_id=removed_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1000},
    )
    log.resolved_rollback(superseded_from, 5 * _NS, target_turn_id=retained_turn_id)
    log.add(
        "hook.operation.finished",
        6 * _NS,
        turn_id=retained_turn_id,
        operation_id=hook_id,
        branch_id="6" * 32,
        payload={"outcome": "success", "duration_ms": 4000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.settled(7 * _NS, turn_id=retained_turn_id, waited_hook_ids=[hook_id])
    log.write(path)

    analysis = analyze_trajectory(path)
    retained = analysis.turns[0]

    reason = "hook.operation lifecycle crosses rollback projection; only its terminal endpoint remains active"
    assert reason in retained.diagnostics
    hook = next(operation for operation in retained.operations if operation.operation_id == hook_id)
    diagnostic = next(item for item in analysis.diagnostics.timeline_operations if item.operation_id == hook_id)
    assert hook.precision is Precision.UNRESOLVED
    assert hook.reason == reason
    assert hook.identity == "after_turn"
    assert hook.hook_id == hook_config_id
    assert diagnostic.identity == "after_turn"
    assert diagnostic.hook_id == hook_config_id


def test_submission_terminal_before_start_is_unresolved_even_when_timestamps_are_nonnegative(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    operation_id = "a" * 32
    log.add(
        "preparation.finished",
        _NS,
        turn_id=None,
        operation_id=operation_id,
        payload={"scope": "pre_turn", "outcome": "rejected", "duration_ms": 1_000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(
        "preparation.started",
        0,
        turn_id=None,
        operation_id=operation_id,
        payload={"scope": "pre_turn", "phase": "admission"},
    )
    log.write(path)

    submission = analyze_trajectory(path).submission_latency
    assert submission is not None
    sample = next(sample for stats in submission.buckets for sample in stats.samples)

    assert sample.duration_ns == Metric(None, Precision.UNRESOLVED, "preparation lifecycle endpoints are not ordered")


def test_missing_preparation_outcome_is_unresolved_not_a_zero_or_exact_sample(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    operation_id = "a" * 32
    log.add(
        "preparation.started",
        0,
        turn_id=None,
        operation_id=operation_id,
        payload={"scope": "pre_turn", "phase": "admission"},
    )
    log.add(
        "preparation.finished",
        _NS,
        turn_id=None,
        operation_id=operation_id,
        payload={"scope": "pre_turn", "duration_ms": 1_000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.write(path)

    submission = analyze_trajectory(path).submission_latency
    assert submission is not None
    sample = next(sample for stats in submission.buckets for sample in stats.samples)

    assert sample.duration_ns == Metric(None, Precision.UNRESOLVED, "preparation terminal outcome is missing")


def _pre_turn(
    log: EventLog,
    operation_id: str,
    start_ns: int,
    end_ns: int,
    outcome: str,
    *,
    target_turn_id: str | None = None,
) -> None:
    finish_payload: dict[str, object] = {"scope": "pre_turn", "outcome": outcome}
    if target_turn_id is not None:
        finish_payload["target_turn_id"] = target_turn_id
    log.span(
        "preparation",
        operation_id,
        start_ns,
        end_ns,
        turn_id=None,
        start_payload={"scope": "pre_turn", "phase": "admission"},
        finish_payload=finish_payload,
    )


def _action(
    index: int,
    turn_id: str,
    turn_number: int,
    start_ns: int,
    classification: ActionClass,
    *,
    outcome: str = "success",
    end_ns: int | None = None,
    end_sequence: int | None = None,
    fingerprint: str | None = None,
    tool_name: str | None = None,
    tool_kind: str = "search",
    classification_precision: Precision = Precision.EXACT,
    classification_reason: str | None = None,
) -> ActionOperation:
    operation_id = f"{index + 100:032x}"
    return ActionOperation(
        evidence_key=f"classification:{index}",
        occurrence_id=f"tool:{index}:{operation_id}",
        operation_id=operation_id,
        turn_id=turn_id,
        turn_number=turn_number,
        tool_name=tool_name or f"tool-{index}",
        tool_kind=tool_kind,
        call_item_id=f"{index:032x}",
        argument_fingerprint=fingerprint or f"fingerprint-{index}",
        classification=classification,
        classification_precision=classification_precision,
        classification_reason=classification_reason,
        outcome=outcome,
        outcome_precision=Precision.EXACT,
        start_sequence=index,
        start_ns=start_ns,
        end_ns=end_ns if end_ns is not None else start_ns + 5,
        end_sequence=end_sequence if end_sequence is not None else index,
    )


def _turn(turn_id: str, turn_number: int, start_ns: int, end_ns: int) -> TurnAnalysis:
    elapsed = end_ns - start_ns
    exact_elapsed = Metric(elapsed, Precision.EXACT)
    zero = Metric(0, Precision.EXACT)
    return TurnAnalysis(
        turn_id=turn_id,
        turn_number=turn_number,
        runtime_id=f"runtime-{turn_number}",
        start_sequence=turn_number,
        end_sequence=turn_number + 1,
        elapsed_ns=exact_elapsed,
        compute_cp_ns=exact_elapsed,
        response_cp_ns=exact_elapsed,
        exclusive_work_ns=zero,
        parallelism=Metric(0.0, Precision.EXACT),
        overlap_gain_ns=zero,
        wall_time_ns=dict.fromkeys(WallBucket, zero),
        utilization=dict.fromkeys(WallBucket, zero),
        usage_tokens=zero,
        attempts=(
            TurnAttemptRef(
                turn_id=turn_id,
                runtime_id=f"runtime-{turn_number}",
                is_retry=False,
                physical_axis_start_ns=start_ns,
                physical_axis_end_ns=end_ns,
                logical_axis_start_ns=start_ns,
                operation_start_index=0,
                operation_end_index=0,
                slice_start_index=0,
                slice_end_index=0,
            ),
        ),
        axis_start_ns=start_ns,
        axis_end_ns=end_ns,
    )


def _empty_change() -> ChangeVerification:
    zero = Metric(0, Precision.EXACT)
    return ChangeVerification(True, False, zero, zero, zero, zero, zero)
