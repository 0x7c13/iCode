# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Positive, negative, and identity tests for the frozen P1 finding rules."""

from __future__ import annotations

from dataclasses import replace

import pytest

from chrys.service.analytics.findings import ContextCarryingLoad, evaluate_findings
from chrys.service.analytics.model import (
    ActionClass,
    ActionFunnel,
    ActionOperation,
    ChangeVerification,
    ChangeVerificationRow,
    ChangeVerificationState,
    FindingRow,
    Metric,
    Precision,
    TimeSlice,
    TurnAnalysis,
    TurnAttemptRef,
    ValidationMetrics,
    WallBucket,
    least_precision,
)


def test_unverified_change_rule_has_positive_and_negative_paths() -> None:
    edit = _action(1, classification=ActionClass.EDIT)
    verify = _action(2, classification=ActionClass.VERIFY)

    assert "unverified-change" in _rule_ids(_evaluate(actions=(edit,)))
    assert "unverified-change" not in _rule_ids(_evaluate(actions=(edit, verify)))


def test_unverified_change_rule_requires_the_verify_to_start_after_the_edit_terminal() -> None:
    edit = _action(1, classification=ActionClass.EDIT, end_sequence=4)
    overlapped_verify = _action(2, classification=ActionClass.VERIFY)
    ordered_verify = _action(5, classification=ActionClass.VERIFY)

    assert "unverified-change" in _rule_ids(_evaluate(actions=(edit, overlapped_verify)))
    assert "unverified-change" not in _rule_ids(_evaluate(actions=(edit, ordered_verify)))


def test_repeated_tool_fingerprint_rule_has_positive_and_negative_paths() -> None:
    first = _action(1, fingerprint="same", tool_name="same-tool")
    repeated = _action(2, fingerprint="same", tool_name="same-tool")
    distinct = _action(2, fingerprint="different", tool_name="same-tool")
    other_tool = _action(3, fingerprint="same", tool_name="other-tool")
    unknown_tool = replace(_action(4, fingerprint="same"), tool_name=None)

    assert "repeated-tool-fingerprint" in _rule_ids(_evaluate(actions=(first, repeated)))
    assert "repeated-tool-fingerprint" not in _rule_ids(_evaluate(actions=(first, distinct)))
    assert "repeated-tool-fingerprint" not in _rule_ids(_evaluate(actions=(first, other_tool)))
    assert "repeated-tool-fingerprint" not in _rule_ids(_evaluate(actions=(first, unknown_tool)))


def test_repeated_fingerprint_evidence_keys_distinguish_tools_without_call_item_ids() -> None:
    actions = tuple(
        replace(_action(index, fingerprint="same", tool_name=tool_name), call_item_id=None)
        for index, tool_name in enumerate(("first", "first", "second", "second"), start=1)
    )

    repeated = [finding for finding in _evaluate(actions=actions) if finding.rule_id == "repeated-tool-fingerprint"]

    assert len(repeated) == 2
    assert len({finding.evidence_key for finding in repeated}) == 2


def test_failed_attempt_critical_path_rule_has_positive_and_negative_paths() -> None:
    failed = _action(1, outcome="errored")
    dominating = _turn(critical={failed.operation_id: 30})
    minor = _turn(critical={failed.operation_id: 24})

    assert "failed-attempt-critical-path" in _rule_ids(_evaluate(actions=(failed,), turn=dominating))
    assert "failed-attempt-critical-path" not in _rule_ids(_evaluate(actions=(failed,), turn=minor))


def test_unknown_tool_outcome_does_not_emit_failed_attempt_finding() -> None:
    unknown = _action(1, outcome="unknown")
    dominating = _turn(critical={unknown.operation_id: 100})

    assert "failed-attempt-critical-path" not in _rule_ids(_evaluate(actions=(unknown,), turn=dominating))


def test_retry_token_amplification_rule_has_positive_and_negative_paths() -> None:
    action = _action(1)
    positive = _evaluate(
        actions=(action,),
        retry=Metric(500, Precision.EXACT),
        retry_evidence=("item:stable",),
        retry_target=(action.turn_id, action.operation_id, "occurrence"),
    )
    negative = _evaluate(actions=(action,), retry=Metric(0, Precision.EXACT), retry_evidence=("item:stable",))

    assert "retry-token-amplification" in _rule_ids(positive)
    assert "retry-token-amplification" not in _rule_ids(negative)


def test_net_zero_churn_rule_has_positive_and_negative_paths() -> None:
    edit = _action(1, classification=ActionClass.EDIT)
    evidence = (f"call:{edit.call_item_id}",)
    positive_change = _change(
        ChangeVerificationRow("a.py", ChangeVerificationState.NET_ZERO, 1, Precision.EXACT, evidence)
    )
    negative_change = _change(
        ChangeVerificationRow("a.py", ChangeVerificationState.UNVERIFIED, 1, Precision.EXACT, evidence)
    )
    # Withheld content backups leave the net change unprovable: the row only
    # believes the file returned to its original state, so the churn warning
    # must not present it as proven wasted work.
    unprovable_change = _change(
        ChangeVerificationRow("a.py", ChangeVerificationState.NET_ZERO, 1, Precision.UNRESOLVED, evidence)
    )

    assert "net-zero-churn" in _rule_ids(_evaluate(actions=(edit,), change=positive_change))
    assert "net-zero-churn" not in _rule_ids(_evaluate(actions=(edit,), change=negative_change))
    assert "net-zero-churn" not in _rule_ids(_evaluate(actions=(edit,), change=unprovable_change))


def test_approval_blocking_share_rule_has_positive_and_negative_paths() -> None:
    action = _action(1)
    high = _turn(slices=(_approval_slice(30),))
    low = _turn(slices=(_approval_slice(24),))

    assert "approval-blocking-share" in _rule_ids(_evaluate(actions=(action,), turn=high))
    assert "approval-blocking-share" not in _rule_ids(_evaluate(actions=(action,), turn=low))


def test_approval_blocking_share_unions_overlapping_waits() -> None:
    action = _action(1)
    # Two concurrent waits: the summed durations (20 + 14) would cross the
    # 25% threshold, but together they cover only 24ns of the 100ns turn.
    overlapping = _turn(slices=(_approval_slice(20), _approval_slice(14, start=10, index=1)))
    blocking = _turn(slices=(_approval_slice(20), _approval_slice(20, start=10, index=1)))

    assert "approval-blocking-share" not in _rule_ids(_evaluate(actions=(action,), turn=overlapping))
    assert "approval-blocking-share" in _rule_ids(_evaluate(actions=(action,), turn=blocking))


def test_context_carrying_load_rule_has_positive_and_negative_paths() -> None:
    carrying = (ContextCarryingLoad(900, "stable-item", "context:1", "turn", 1),)

    assert "context-carrying-load" in _rule_ids(_evaluate(context=carrying))
    assert "context-carrying-load" not in _rule_ids(_evaluate())


def test_finding_evidence_key_is_stable_across_projection_local_identity_changes() -> None:
    original = _action(1, classification=ActionClass.EDIT, operation_id="a" * 32, sequence=10)
    replayed = _action(1, classification=ActionClass.EDIT, operation_id="b" * 32, sequence=900)

    first = next(finding for finding in _evaluate(actions=(original,)) if finding.rule_id == "unverified-change")
    second = next(finding for finding in _evaluate(actions=(replayed,)) if finding.rule_id == "unverified-change")

    assert first.evidence_key == second.evidence_key
    assert first.occurrence_id != second.occurrence_id


def test_net_zero_finding_identity_depends_on_stable_evidence_not_file_path() -> None:
    edit = _action(1, classification=ActionClass.EDIT)
    evidence = (f"call:{edit.call_item_id}",)
    first_change = _change(
        ChangeVerificationRow("old.py", ChangeVerificationState.NET_ZERO, 1, Precision.EXACT, evidence)
    )
    second_change = _change(
        ChangeVerificationRow("renamed.py", ChangeVerificationState.NET_ZERO, 1, Precision.EXACT, evidence)
    )

    first = next(
        finding for finding in _evaluate(actions=(edit,), change=first_change) if finding.rule_id == "net-zero-churn"
    )
    second = next(
        finding for finding in _evaluate(actions=(edit,), change=second_change) if finding.rule_id == "net-zero-churn"
    )

    assert first.evidence_key == second.evidence_key


def test_deterministic_describes_every_rule() -> None:
    edit = _action(1, classification=ActionClass.EDIT)
    active = _evaluate(actions=(edit,))

    assert all(finding.deterministic for finding in active)


def _evaluate(
    *,
    actions: tuple[ActionOperation, ...] = (),
    turn: TurnAnalysis | None = None,
    retry: Metric | None = None,
    retry_evidence: tuple[str, ...] = (),
    retry_target: tuple[str | None, str | None, str | None] = (None, None, None),
    change: ChangeVerification | None = None,
    context: tuple[ContextCarryingLoad, ...] = (),
) -> tuple[FindingRow, ...]:
    selected_turn = turn or _turn()
    return evaluate_findings(
        actions=actions,
        turns=(selected_turn,),
        validation=_validation(actions, retry or Metric(0, Precision.EXACT)),
        change_verification=change or _change(),
        retry_amplification_evidence=retry_evidence,
        retry_target=retry_target,
        context_carrying_load=context,
    )


def _action(
    index: int,
    *,
    classification: ActionClass = ActionClass.SEARCH,
    outcome: str = "success",
    fingerprint: str | None = None,
    operation_id: str | None = None,
    sequence: int | None = None,
    end_sequence: int | None = None,
    tool_name: str | None = None,
) -> ActionOperation:
    call_item_id = f"{index:032x}"
    argument_fingerprint = fingerprint or f"fingerprint-{index}"
    operation = operation_id or f"{index + 100:032x}"
    start_sequence = sequence or index
    return ActionOperation(
        evidence_key=f"classification:{call_item_id}",
        occurrence_id=f"tool:{start_sequence}:{operation}",
        operation_id=operation,
        turn_id="turn",
        turn_number=1,
        tool_name=tool_name or f"tool-{index}",
        tool_kind="search",
        call_item_id=call_item_id,
        argument_fingerprint=argument_fingerprint,
        classification=classification,
        classification_precision=Precision.EXACT,
        classification_reason=None,
        outcome=outcome,
        outcome_precision=Precision.EXACT,
        start_sequence=start_sequence,
        start_ns=index * 10,
        end_ns=index * 10 + 5,
        end_sequence=end_sequence if end_sequence is not None else start_sequence,
    )


def _turn(
    *,
    critical: dict[str, int] | None = None,
    slices: tuple[TimeSlice, ...] = (),
) -> TurnAnalysis:
    exact_100 = Metric(100, Precision.EXACT)
    zero = Metric(0, Precision.EXACT)
    return TurnAnalysis(
        turn_id="turn",
        turn_number=1,
        runtime_id="runtime",
        start_sequence=1,
        end_sequence=99,
        elapsed_ns=exact_100,
        compute_cp_ns=exact_100,
        response_cp_ns=exact_100,
        exclusive_work_ns=exact_100,
        parallelism=Metric(1.0, Precision.EXACT),
        overlap_gain_ns=zero,
        wall_time_ns=dict.fromkeys(WallBucket, zero),
        utilization=dict.fromkeys(WallBucket, zero),
        usage_tokens=zero,
        attempts=(
            TurnAttemptRef(
                turn_id="turn",
                runtime_id="runtime",
                is_retry=False,
                physical_axis_start_ns=0,
                physical_axis_end_ns=100,
                logical_axis_start_ns=0,
                operation_start_index=0,
                operation_end_index=0,
                slice_start_index=0,
                slice_end_index=len(slices),
            ),
        ),
        axis_start_ns=0,
        axis_end_ns=100,
        slices=slices,
        critical_tool_contributions_ns=critical or {},
    )


def _approval_slice(duration: int, *, start: int = 0, index: int = 0) -> TimeSlice:
    return TimeSlice(
        family="approval",
        slice_index=index,
        turn_id="turn",
        runtime_id="runtime",
        operation_id="approval",
        owner="approval",
        start_ns=start,
        end_ns=start + duration,
        wall_bucket=WallBucket.WAIT,
        counts_as_work=False,
        compute_weight=False,
        response_weight=True,
    )


def _validation(actions: tuple[ActionOperation, ...], retry: Metric) -> ValidationMetrics:
    exact_zero = Metric(0, Precision.EXACT)
    return ValidationMetrics(
        funnel=ActionFunnel(exact_zero, exact_zero, exact_zero, exact_zero),
        time_to_first_edit_ns=exact_zero,
        first_edit_to_first_verify_ns=exact_zero,
        edit_verify_cycle_count=exact_zero,
        edit_verify_cycle_median_ns=exact_zero,
        unverified_change_count=Metric(
            sum(action.classification is ActionClass.EDIT for action in actions),
            Precision.EXACT,
        ),
        net_zero_churn_count=exact_zero,
        repeated_failure_signature_count=exact_zero,
        failure_recovery_median_ns=exact_zero,
        tool_failure_count=exact_zero,
        tool_count=Metric(len(actions), Precision.EXACT),
        retry_amplification_tokens=retry,
    )


def _change(row: ChangeVerificationRow | None = None) -> ChangeVerification:
    exact_zero = Metric(0, Precision.EXACT)
    return ChangeVerification(
        detail_available=True,
        detection_truncated=False,
        files_touched=Metric(1 if row is not None else 0, Precision.EXACT),
        created=exact_zero,
        modified=exact_zero,
        deleted=exact_zero,
        net_zero=Metric(row is not None and row.state is ChangeVerificationState.NET_ZERO, Precision.EXACT),
        rows=(row,) if row is not None else (),
    )


def _rule_ids(findings: tuple[FindingRow, ...]) -> set[str]:
    return {finding.rule_id for finding in findings}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ((), Precision.EXACT),
        ((Precision.EXACT, Precision.ESTIMATED), Precision.ESTIMATED),
        ((Precision.MISSING, Precision.ESTIMATED), Precision.MISSING),
        ((Precision.UNRESOLVED, Precision.MISSING, Precision.EXACT), Precision.UNRESOLVED),
    ],
)
def test_least_precision_takes_the_least_precise_value(values: tuple[Precision, ...], expected: Precision) -> None:
    assert least_precision(values) is expected
    assert least_precision(value for value in values) is expected
