# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rollback-stable rules for compact trajectory finding rows."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import accumulate
from math import inf
from threading import Event
from typing import Final, NamedTuple

from chrys.foundation.tool_kinds import (
    KIND_FILESYSTEM_READ,
    KIND_SEARCH,
    KIND_SHELL,
    KIND_SUB_AGENT,
    KIND_TODO,
    KIND_WEB_FETCH,
    KIND_WEB_SEARCH,
)
from chrys.foundation.trajectory.event_types import ToolOutcome
from chrys.service.analytics.classification import evidence_key, tool_failed, tool_succeeded, unverified_edits
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
)
from chrys.service.analytics.reader import raise_if_cancelled as _check_cancelled

_FAILED_CRITICAL_PATH_SHARE: Final = 0.25
_APPROVAL_BLOCKING_SHARE: Final = 0.25
_CONTEXT_CARRYING_LOAD_TOP_N: Final = 3
# Below this many re-sent tokens an item costs less than one large request;
# the Insights table still ranks it, but it is not worth a finding.
_CONTEXT_CARRYING_LOAD_MIN_TOKENS: Final = 100_000
# Calls whose result depends only on state another call would have to change.
_REPEATABLE_READ_KINDS: Final = frozenset({KIND_FILESYSTEM_READ, KIND_SEARCH, KIND_WEB_SEARCH, KIND_WEB_FETCH})
# Work that cannot change what such a read returns; anything else running
# between two identical reads (an edit, a shell command, a sleep, a sub-agent,
# a question to the user) may have, so the second read is not redundant.
_STATE_NEUTRAL_KINDS: Final = _REPEATABLE_READ_KINDS | {KIND_TODO}
_NS_PER_MS: Final = 1_000_000


class SequenceSpan(NamedTuple):
    """The event sequence numbers from *first* to *last*, both included; *last* is None while the span is open."""

    first: int
    last: int | None


@dataclass(frozen=True, slots=True)
class RetriedExchange:
    """The retried model exchange a retry-amplification finding points at."""

    turn_id: str | None
    operation_id: str
    finish_sequence: int
    """The sequence of the exchange's finish event."""


class _ReadKey(NamedTuple):
    """What makes two read-only calls identical: who made them, in which turn, and the call itself."""

    actor_id: str | None
    turn_id: str | None
    tool_name: str
    argument_fingerprint: str


def evaluate_findings(
    *,
    actions: tuple[ActionOperation, ...],
    turns: tuple[TurnAnalysis, ...] | list[TurnAnalysis],
    validation: ValidationMetrics,
    change_verification: ChangeVerification,
    retry_amplification_evidence: tuple[str, ...],
    retried_exchange: RetriedExchange | None,
    context_carrying_load: tuple[ContextCarryingLoad, ...],
    outside_changes: tuple[SequenceSpan, ...] = (),
    cancel_event: Event | None = None,
) -> tuple[FindingRow, ...]:
    """Evaluate the frozen first rule batch using rollback-stable evidence only.

    *turns* are logical: a retried turn's attempts share one elapsed and
    response time, while each action names the attempt it ran in.
    *outside_changes* are where state changed outside any tool call:
    background tasks and context compaction.
    """
    _check_cancelled(cancel_event)
    findings: list[FindingRow] = []
    turn_of = {attempt.turn_id: turn for turn in turns for attempt in turn.attempts}

    verify_attempts = [action for action in actions if action.classification is ActionClass.VERIFY]
    successful_verifies = [action for action in verify_attempts if tool_succeeded(action.outcome)]
    last_verify = successful_verifies[-1] if successful_verifies else None
    unverified = unverified_edits(actions, last_verify)
    unverified_evidence = _action_evidence_set(unverified)
    # Edits after the last verify that only put files back the way they were
    # leave nothing to test: the per-file detail then shows every changed
    # file verified or provably cancelled out. A row that cannot prove its
    # state keeps the warning, and so does an edit the saved detail does not
    # cover yet: a turn's detail lands with its save, after the turn ends.
    row_states = {row.state for row in change_verification.rows}
    saved_through = change_verification.saved_through_sequence
    only_cancelled_out = (
        change_verification.detail_available
        and not change_verification.detection_truncated
        and ChangeVerificationState.NET_ZERO in row_states
        and row_states <= {ChangeVerificationState.VERIFIED, ChangeVerificationState.NET_ZERO}
        and all(row.precision is not Precision.UNRESOLVED for row in change_verification.rows)
        and saved_through is not None
        and all(action.start_sequence < saved_through for action in unverified)
    )
    if unverified and unverified_evidence and not only_cancelled_out:
        target = unverified[0]
        # A shell command whose text is unavailable and that did not fail
        # could have been a passing test of the edit if it started after the
        # edit ended, or after it started while the edit has no end yet (its
        # turn is still running, or a crash cut it off).
        edit_done = target.start_sequence if target.end_sequence is None else target.end_sequence
        unverified_args: list[tuple[str, int]] = [
            ("count", len(unverified)),
            ("verifies", len(successful_verifies)),
            ("verify_attempts", len(verify_attempts)),
            (
                "unclassified_shell",
                sum(
                    action.tool_kind == KIND_SHELL
                    and action.classification_precision is Precision.UNRESOLVED
                    and not tool_failed(action.outcome)
                    and action.start_sequence > edit_done
                    for action in actions
                ),
            ),
        ]
        if last_verify is not None and last_verify.turn_number is not None:
            unverified_args.append(("last_verify_turn", last_verify.turn_number))
        findings.append(
            _finding(
                "unverified-change",
                unverified_evidence,
                occurrence_id=f"unverified:{target.occurrence_id}",
                severity=FindingSeverity.ERROR,
                detail_args=tuple(unverified_args),
                precision=validation.unverified_change_count.precision,
                target=target,
            )
        )

    for key, calls in _repeated_reads(actions, outside_changes=outside_changes, cancel_event=cancel_event).items():
        _check_cancelled(cancel_event)
        repeats = calls[1:]
        target = repeats[-1]
        repeat_args: list[tuple[str, int]] = [("count", len(repeats))]
        durations = [action.duration_ns for action in repeats]
        if all(duration is not None for duration in durations):
            repeat_args.append(("duration_ms", sum(duration or 0 for duration in durations) // _NS_PER_MS))
        result_tokens = [action.payload_token_estimate for action in repeats]
        if all(tokens is not None for tokens in result_tokens):
            repeat_args.append(("result_tokens", sum(tokens or 0 for tokens in result_tokens)))
        findings.append(
            _finding(
                "repeated-read",
                (f"tool:{key.tool_name}", *_action_evidence_set(calls)),
                occurrence_id=f"repeated:{target.occurrence_id}",
                severity=FindingSeverity.WARNING,
                detail_args=tuple(repeat_args),
                precision=least_precision(
                    (
                        validation.tool_count.precision,
                        *(action.classification_precision for action in calls),
                    )
                ),
                target=target,
            )
        )

    for action in actions:
        _check_cancelled(cancel_event)
        turn = turn_of.get(action.turn_id or "")
        if turn is None:
            continue
        response_cp = turn.response_cp_ns.value
        contribution = turn.critical_tool_contributions_ns.get(action.operation_id, 0)
        stable_evidence = _action_evidence(action)
        if (
            not tool_failed(action.outcome)
            # A declined call never ran: its time is the approval wait, which
            # the approval rule reports.
            or action.outcome == ToolOutcome.REJECTED
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
                detail_args=(
                    ("percentage", int(f"{contribution / response_cp * 100:.0f}")),
                    ("duration_ms", contribution // _NS_PER_MS),
                ),
                precision=turn.response_cp_ns.precision,
                target=action,
            )
        )

    if validation.retry_amplification_tokens.value and retry_amplification_evidence and retried_exchange is not None:
        retry_turn = turn_of.get(retried_exchange.turn_id or "")
        findings.append(
            FindingRow(
                evidence_key=evidence_key("retry-token-amplification", 1, retry_amplification_evidence),
                occurrence_id=f"retry-amplification:{retried_exchange.finish_sequence}",
                rule_id="retry-token-amplification",
                severity=FindingSeverity.WARNING,
                deterministic=True,
                detail_args=(("tokens", int(validation.retry_amplification_tokens.value)),),
                precision=validation.retry_amplification_tokens.precision,
                turn_id=retried_exchange.turn_id,
                turn_number=retry_turn.turn_number if retry_turn is not None else None,
                operation_id=retried_exchange.operation_id,
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

    actions_by_turn: dict[str, list[ActionOperation]] = {}
    for action in actions:
        if (owner := turn_of.get(action.turn_id or "")) is not None:
            actions_by_turn.setdefault(owner.turn_id, []).append(action)
    for turn in turns:
        _check_cancelled(cancel_event)
        elapsed = turn.elapsed_ns.value
        approval_ns = _approval_wait_ns(turn.slices)
        turn_actions = actions_by_turn.get(turn.turn_id, [])
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
                detail_args=(
                    ("percentage", int(f"{approval_ns / elapsed * 100:.0f}")),
                    ("duration_ms", approval_ns // _NS_PER_MS),
                ),
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
            detail_args=(
                ("load", item.load),
                ("tokens", item.token_count),
                ("requests", item.carry_count),
                *((("origin_turn", item.origin_turn_number),) if item.origin_turn_number is not None else ()),
            ),
            precision=least_precision((Precision.ESTIMATED, validation.tool_count.precision)),
            turn_id=item.turn_id,
            turn_number=item.turn_number,
            operation_id=None,
        )
        for item in sorted(context_carrying_load, key=lambda value: (value.load, value.item_id), reverse=True)[
            :_CONTEXT_CARRYING_LOAD_TOP_N
        ]
        if item.load >= _CONTEXT_CARRYING_LOAD_MIN_TOKENS
    )

    _check_cancelled(cancel_event)
    return tuple(findings)


def _repeated_reads(
    actions: tuple[ActionOperation, ...],
    *,
    outside_changes: tuple[SequenceSpan, ...],
    cancel_event: Event | None,
) -> dict[_ReadKey, list[ActionOperation]]:
    """Read-only calls that repeated an identical call nothing could have changed.

    A repeat must come from the same agent in the same turn, succeed, and
    start after the last earlier identical call that succeeded had finished; a
    failed or declined call fetched nothing, so it is neither a repeat nor the
    call one is measured against. Nothing that may change state can overlap
    the stretch from that call's start to the repeat: no state-changing call
    from any agent in that turn and none of *outside_changes* (a running
    background task, or compaction dropping the turn's own results). Turns
    never overlap, so a call from another turn cannot fall in between, even
    one a crash left without an end. Re-reading a file after an edit, polling
    after a sleep, re-reading a background task's growing log, reading again
    what compaction dropped or retrying a failed read is expected work. The
    call that started a sub-agent does not count against that sub-agent's own
    reads; whatever the sub-agent changes is an action of its own.
    Each group lists the first call and then every redundant repeat of it.
    """
    by_turn: dict[str | None, list[ActionOperation]] = {}
    for action in sorted(actions, key=lambda action: action.start_sequence):
        by_turn.setdefault(action.turn_id, []).append(action)
    groups: dict[_ReadKey, list[ActionOperation]] = {}
    for ordered in by_turn.values():
        # Possible changes by start; the running maximum of their ends answers
        # "did any change that began before this repeat end after the earlier
        # call began" with one bisection. An open change never ended.
        changes = sorted(
            [
                *(
                    SequenceSpan(action.start_sequence, action.end_sequence)
                    for action in ordered
                    if action.tool_kind not in _STATE_NEUTRAL_KINDS and action.tool_kind != KIND_SUB_AGENT
                ),
                *outside_changes,
            ],
            key=lambda change: change.first,
        )
        change_starts = [change.first for change in changes]
        latest_change_end = list(accumulate((inf if change.last is None else change.last for change in changes), max))
        # Sub-agent calls are checked one by one: a repeat inside one is exempt
        # from the calls enclosing it.
        sub_agent_calls = [action for action in ordered if action.tool_kind == KIND_SUB_AGENT]
        previous_by_key: dict[_ReadKey, ActionOperation] = {}
        for action in ordered:
            _check_cancelled(cancel_event)
            if (
                action.tool_kind not in _REPEATABLE_READ_KINDS
                or action.tool_name is None
                or action.argument_fingerprint is None
                or not tool_succeeded(action.outcome)
            ):
                continue
            key = _ReadKey(action.actor_id, action.turn_id, action.tool_name, action.argument_fingerprint)
            previous = previous_by_key.get(key)
            previous_by_key[key] = action
            if previous is None or previous.end_sequence is None or previous.end_sequence >= action.start_sequence:
                continue
            began_before = bisect_left(change_starts, action.start_sequence)
            if began_before and latest_change_end[began_before - 1] > previous.start_sequence:
                continue
            if any(
                call.start_sequence < action.start_sequence
                and (call.end_sequence is None or call.end_sequence > previous.start_sequence)
                and call.operation_id not in action.enclosing_operation_ids
                for call in sub_agent_calls
            ):
                continue
            groups.setdefault(key, [previous]).append(action)
    return groups


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


__all__ = ["ContextCarryingLoad", "RetriedExchange", "SequenceSpan", "evaluate_findings"]
