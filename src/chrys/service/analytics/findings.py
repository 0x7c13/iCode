# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rollback-stable rules for compact trajectory finding rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from threading import Event
from typing import Final

from chrys.service.analytics.classification import evidence_key, tool_failed, tool_succeeded
from chrys.service.analytics.math import interval_length
from chrys.service.analytics.model import (
    ActionClass,
    ActionOperation,
    ChangeVerification,
    ChangeVerificationState,
    ContextCarryingLoad,
    FindingRow,
    FindingSeverity,
    Precision,
    TimeSlice,
    TurnAnalysis,
    ValidationMetrics,
    least_precision,
    verify_covers_edit,
)
from chrys.service.analytics.reader import raise_if_cancelled as _check_cancelled

_FAILED_CRITICAL_PATH_SHARE: Final = 0.25
_APPROVAL_BLOCKING_SHARE: Final = 0.25
_CONTEXT_CARRYING_LOAD_TOP_N: Final = 3


def evaluate_findings(
    *,
    actions: tuple[ActionOperation, ...],
    turns: tuple[TurnAnalysis, ...] | list[TurnAnalysis],
    validation: ValidationMetrics,
    change_verification: ChangeVerification,
    retry_amplification_evidence: tuple[str, ...],
    retry_target: tuple[str | None, str | None, str | None],
    context_carrying_load: tuple[ContextCarryingLoad, ...],
    cancel_event: Event | None = None,
) -> tuple[FindingRow, ...]:
    """Evaluate the frozen first rule batch using rollback-stable evidence only."""
    _check_cancelled(cancel_event)
    findings: list[FindingRow] = []
    turns_by_id = {turn.turn_id: turn for turn in turns}

    successful_verifies = [
        action for action in actions if action.classification is ActionClass.VERIFY and tool_succeeded(action.outcome)
    ]
    last_verify = successful_verifies[-1] if successful_verifies else None
    unverified_edits = [
        action
        for action in actions
        if action.classification is ActionClass.EDIT
        and (last_verify is None or not verify_covers_edit(action, last_verify))
    ]
    unverified_evidence = _action_evidence_set(unverified_edits)
    if unverified_edits and unverified_evidence:
        target = unverified_edits[0]
        findings.append(
            _finding(
                "unverified-change",
                unverified_evidence,
                occurrence_id=f"unverified:{target.occurrence_id}",
                severity=FindingSeverity.ERROR,
                detail_args=(("count", len(unverified_edits)),),
                precision=validation.unverified_change_count.precision,
                target=target,
            )
        )

    fingerprints: dict[tuple[str, str], list[ActionOperation]] = defaultdict(list)
    for action in actions:
        _check_cancelled(cancel_event)
        if action.argument_fingerprint is not None and action.tool_name is not None:
            fingerprints[(action.tool_name, action.argument_fingerprint)].append(action)
    for (tool_name, fingerprint), repeated in sorted(fingerprints.items()):
        _check_cancelled(cancel_event)
        if len(repeated) < 2:
            continue
        target = repeated[-1]
        findings.append(
            _finding(
                "repeated-tool-fingerprint",
                (f"tool:{tool_name}", f"arguments:{fingerprint}", *_action_evidence_set(repeated)),
                occurrence_id=f"repeated:{target.occurrence_id}",
                severity=FindingSeverity.WARNING,
                detail_args=(("count", len(repeated)),),
                precision=least_precision(
                    (
                        validation.tool_count.precision,
                        *(action.classification_precision for action in repeated),
                    )
                ),
                target=target,
            )
        )

    for action in actions:
        _check_cancelled(cancel_event)
        turn = turns_by_id.get(action.turn_id)
        if turn is None:
            continue
        response_cp = turn.response_cp_ns.value
        contribution = turn.critical_tool_contributions_ns.get(action.operation_id, 0)
        stable_evidence = _action_evidence(action)
        if (
            not tool_failed(action.outcome)
            or not isinstance(response_cp, int)
            or response_cp <= 0
            or contribution / response_cp < _FAILED_CRITICAL_PATH_SHARE
            or not stable_evidence
        ):
            continue
        findings.append(
            _finding(
                "failed-attempt-critical-path",
                stable_evidence,
                occurrence_id=f"failed-cp:{action.occurrence_id}",
                severity=FindingSeverity.WARNING,
                detail_args=(("percentage", int(f"{contribution / response_cp * 100:.0f}")),),
                precision=turn.response_cp_ns.precision,
                target=action,
            )
        )

    if validation.retry_amplification_tokens.value and retry_amplification_evidence:
        retry_turn_id, retry_operation_id, retry_occurrence = retry_target
        retry_turn = turns_by_id.get(retry_turn_id or "")
        findings.append(
            FindingRow(
                evidence_key=evidence_key("retry-token-amplification", 1, retry_amplification_evidence),
                occurrence_id=f"retry-amplification:{retry_occurrence or 'current'}",
                rule_id="retry-token-amplification",
                severity=FindingSeverity.WARNING,
                deterministic=True,
                detail_args=(("tokens", int(validation.retry_amplification_tokens.value)),),
                precision=validation.retry_amplification_tokens.precision,
                turn_id=retry_turn_id,
                turn_number=retry_turn.turn_number if retry_turn is not None else None,
                operation_id=retry_operation_id,
            )
        )

    # An unresolved net-zero row only *believes* the file returned to its
    # original state (withheld content backups, truncated detection); the
    # churn warning must not present such a row as proven wasted work.
    net_zero_rows = [
        row
        for row in change_verification.rows
        if row.state is ChangeVerificationState.NET_ZERO and row.precision is not Precision.UNRESOLVED
    ]
    net_zero_evidence = tuple(sorted({item for row in net_zero_rows for item in row.evidence}))
    if net_zero_rows and net_zero_evidence:
        last_turn_number = max(row.last_change_turn for row in net_zero_rows)
        target_action = next(
            (
                action
                for action in reversed(actions)
                if action.turn_number == last_turn_number and action.classification is ActionClass.EDIT
            ),
            None,
        )
        findings.append(
            FindingRow(
                evidence_key=evidence_key("net-zero-churn", 1, net_zero_evidence),
                occurrence_id=f"net-zero:{last_turn_number}",
                rule_id="net-zero-churn",
                severity=FindingSeverity.WARNING,
                deterministic=True,
                detail_args=(("count", len(net_zero_rows)),),
                precision=least_precision((change_verification.net_zero.precision, validation.tool_count.precision)),
                turn_id=target_action.turn_id if target_action is not None else None,
                turn_number=last_turn_number,
                operation_id=target_action.operation_id if target_action is not None else None,
            )
        )

    for turn in turns:
        _check_cancelled(cancel_event)
        elapsed = turn.elapsed_ns.value
        approval_ns = _approval_wait_ns(turn.slices)
        turn_actions = [action for action in actions if action.turn_id == turn.turn_id]
        stable_evidence = _action_evidence_set(turn_actions)
        if (
            not isinstance(elapsed, int)
            or elapsed <= 0
            or approval_ns / elapsed < _APPROVAL_BLOCKING_SHARE
            or not turn_actions
            or not stable_evidence
        ):
            continue
        findings.append(
            _finding(
                "approval-blocking-share",
                stable_evidence,
                occurrence_id=f"approval-share:{turn.start_sequence}",
                severity=FindingSeverity.WARNING,
                detail_args=(("percentage", int(f"{approval_ns / elapsed * 100:.0f}")),),
                precision=turn.elapsed_ns.precision,
                target=turn_actions[0],
            )
        )

    _check_cancelled(cancel_event)
    findings.extend(
        FindingRow(
            evidence_key=evidence_key("context-carrying-load", 1, (f"item:{item.item_id}",)),
            occurrence_id=item.occurrence_id,
            rule_id="context-carrying-load",
            severity=FindingSeverity.INFO,
            deterministic=True,
            detail_args=(("load", item.load),),
            precision=least_precision((Precision.ESTIMATED, validation.tool_count.precision)),
            turn_id=item.turn_id,
            turn_number=item.turn_number,
            operation_id=None,
        )
        for item in sorted(context_carrying_load, key=lambda value: (value.load, value.item_id), reverse=True)[
            :_CONTEXT_CARRYING_LOAD_TOP_N
        ]
    )

    _check_cancelled(cancel_event)
    return tuple(findings)


def _approval_wait_ns(slices: Iterable[TimeSlice]) -> int:
    """Wall-clock length of the union of a turn's approval waits.

    Concurrent approvals overlap; summing their durations would count the
    shared wall time once per request and push the share past the elapsed
    turn time.
    """
    return interval_length((item.start_ns, item.end_ns) for item in slices if item.owner == "approval")


def _action_evidence(action: ActionOperation) -> tuple[str, ...]:
    return tuple(
        item
        for item in (
            f"call:{action.call_item_id}" if action.call_item_id is not None else None,
            f"arguments:{action.argument_fingerprint}" if action.argument_fingerprint is not None else None,
        )
        if item is not None
    )


def _action_evidence_set(actions: list[ActionOperation]) -> tuple[str, ...]:
    return tuple(sorted({item for action in actions for item in _action_evidence(action)}))


def _finding(
    rule_id: str,
    stable_evidence: tuple[str, ...],
    *,
    occurrence_id: str,
    severity: FindingSeverity,
    detail_args: tuple[tuple[str, int], ...],
    precision: Precision,
    target: ActionOperation,
) -> FindingRow:
    return FindingRow(
        evidence_key=evidence_key(rule_id, 1, stable_evidence),
        occurrence_id=occurrence_id,
        rule_id=rule_id,
        severity=severity,
        deterministic=True,
        detail_args=detail_args,
        precision=precision,
        turn_id=target.turn_id,
        turn_number=target.turn_number,
        operation_id=target.operation_id,
    )


__all__ = ["ContextCarryingLoad", "evaluate_findings"]
