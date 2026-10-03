# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tool action projection, validation metrics and per-file change verification.

Actions are projected from tool lifecycles; action counts, edit/verify
cycles, validation metrics and file verification states are derived from them
here. Action classes and tool-outcome predicates come from ``classification``.
``findings``' unverified-change rule repeats this module's unverified-edit
selection (``model.verify_covers_edit`` against the last successful verify)
and takes its precision from the count here, so a change to which edits count
as verified must change both. File verification states order differently: a
verify in the file's last-change turn must start after every potential mutator
in that turn.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import replace
from statistics import median
from threading import Event
from typing import Final, cast

from chrys.foundation.trajectory.event_types import SourceRefKind, ToolOutcome
from chrys.foundation.trajectory.ids import is_valid_analytics_id
from chrys.service.analytics._facts import (
    _active,
    _Endpoint,
    _EventScope,
    _Intermediate,
    _Node,
    _payload_int,
    _payload_str,
    _payload_value,
    _ToolPayloadExtras,
)
from chrys.service.analytics._metric_ops import _cap_session_metric, _with_unresolved_precision
from chrys.service.analytics._session_projection import _SessionProjection
from chrys.service.analytics._timeline import _tool_context_for_start
from chrys.service.analytics.classification import (
    KNOWN_TOOL_OUTCOMES,
    classification_evidence_key,
    classify_action,
    parse_verify_commands,
    tool_failed,
    tool_succeeded,
)
from chrys.service.analytics.model import (
    ActionClass,
    ActionFunnel,
    ActionOperation,
    ChangeVerification,
    ChangeVerificationRow,
    ChangeVerificationState,
    Metric,
    Precision,
    TurnAnalysis,
    ValidationMetrics,
    least_precision,
    verify_covers_edit,
)
from chrys.service.analytics.reader import raise_if_cancelled as _check_cancelled
from chrys.service.mutations.types import FileHashDiff, parse_skip_reason

_WORD_LIST_HEURISTIC_REASON: Final = "one or more shell actions use the verification word-list heuristic"


def _resolve_actions(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    turns: list[TurnAnalysis],
    *,
    session_projection: Callable[[], _SessionProjection],
    verify_commands: str,
    cancel_event: Event | None,
) -> tuple[ActionOperation, ...]:
    turns_by_id = {turn.turn_id: turn for turn in turns}
    commands = parse_verify_commands(verify_commands)
    actions: list[ActionOperation] = []
    for node in intermediate.nodes.get("tool.operation", {}).values():
        _check_cancelled(cancel_event)
        starts = [event for event in node.starts if _active(event.sequence, inactive_ranges) and not event.side_call]
        finishes = [
            event for event in node.finishes if _active(event.sequence, inactive_ranges) and not event.side_call
        ]
        if len(starts) != 1 or starts[0].side_call or starts[0].turn_id not in turns_by_id:
            continue
        start = starts[0]
        call_item_id = _payload_str(start.payload, "call_item_id")
        argument_fingerprint = _payload_str(start.payload, "argument_fingerprint")
        tool_kind = _payload_str(start.payload, "tool_kind")
        tool_name = _payload_str(start.payload, "tool_name")
        stable_key = classification_evidence_key(
            call_item_id=call_item_id,
            argument_fingerprint=argument_fingerprint,
        )
        command = session_projection().commands.get(call_item_id or "") if tool_kind == "shell" else None
        classification, classification_precision, classification_reason = classify_action(
            tool_kind,
            command=command,
            verify_commands=commands,
        )
        outcome, outcome_precision, outcome_reason, finish = _action_outcome(start, finishes)
        context = _tool_context_for_start(node, start)
        payload = _tool_payload_for_node(node, inactive_ranges, start.scope)
        turn = turns_by_id[cast("str", start.turn_id)]
        if turn.action_projection_precision is not Precision.EXACT:
            outcome = None
            outcome_precision = Precision.UNRESOLVED
            outcome_reason = turn.action_projection_reason or "tool action projection is incomplete"
            finish = None
        actions.append(
            ActionOperation(
                evidence_key=stable_key,
                occurrence_id=f"tool:{start.sequence}:{node.operation_id}",
                operation_id=node.operation_id,
                turn_id=turn.turn_id,
                turn_number=turn.turn_number,
                tool_name=tool_name,
                tool_kind=tool_kind,
                call_item_id=call_item_id,
                argument_fingerprint=argument_fingerprint,
                classification=classification,
                classification_precision=classification_precision,
                classification_reason=classification_reason,
                outcome=outcome,
                outcome_precision=outcome_precision,
                start_sequence=start.sequence,
                start_ns=start.monotonic_ns,
                end_ns=finish.monotonic_ns if finish is not None and finish.monotonic_measurement else None,
                end_sequence=finish.sequence if finish is not None else None,
                outcome_reason=outcome_reason,
                server_name=context.server_name if context is not None else None,
                remote_name=context.remote_name if context is not None else None,
                skill_name=context.skill_name if context is not None else None,
                skill_revision=context.skill_revision if context is not None else None,
                script_name=context.script_name if context is not None else None,
                resource_name=context.resource_name if context is not None else None,
                exit_code=_payload_int(finish.payload, "exit_code") if finish is not None else None,
                payload_observed=payload is not None,
                payload_bytes=payload.model_visible_bytes if payload is not None else None,
                payload_token_estimate=payload.local_token_estimate if payload is not None else None,
                payload_original_bytes=payload.original_bytes if payload is not None else None,
                payload_truncated=payload.truncated if payload is not None else None,
                payload_spilled=payload.spilled if payload is not None else None,
            )
        )
    return tuple(sorted(actions, key=lambda action: (action.start_sequence, action.operation_id)))


def _action_outcome(
    start: _Endpoint,
    finishes: list[_Endpoint],
) -> tuple[str | None, Precision, str | None, _Endpoint | None]:
    if not finishes:
        return None, Precision.MISSING, "tool operation has no terminal event", None
    if len(finishes) != 1:
        return None, Precision.UNRESOLVED, "tool operation has more than one terminal event", None
    finish = finishes[0]
    if finish.scope != start.scope:
        return None, Precision.UNRESOLVED, "tool lifecycle endpoints cross scope", None
    if finish.sequence <= start.sequence or finish.monotonic_ns < start.monotonic_ns:
        return None, Precision.UNRESOLVED, "tool lifecycle endpoints are not ordered", None
    outcome = _payload_str(finish.payload, "outcome")
    if outcome is None:
        return None, Precision.MISSING, "tool terminal outcome is missing", finish
    if outcome not in KNOWN_TOOL_OUTCOMES:
        return outcome, Precision.UNRESOLVED, "tool terminal outcome is invalid", finish
    return outcome, Precision.EXACT, None, finish


def _scope_matches(observed_scope: _EventScope, node_scope: _EventScope) -> bool:
    return (
        observed_scope.runtime_id == node_scope.runtime_id
        and observed_scope.branch_id == node_scope.branch_id
        and observed_scope.coverage_id == node_scope.coverage_id
        and observed_scope.actor_id == node_scope.actor_id
        and observed_scope.turn_id == node_scope.turn_id
    )


def _tool_payload_for_node(
    node: _Node,
    inactive_ranges: tuple[tuple[int, int], ...],
    scope: _EventScope,
) -> _ToolPayloadExtras | None:
    extras = node.extras
    if extras is None:
        return None
    active = [
        item
        for item in extras.payloads
        if _active(item.sequence, inactive_ranges) and _scope_matches(item.scope, scope)
    ]
    return active[0] if len(active) == 1 else None


def _action_count_metrics(
    actions: tuple[ActionOperation, ...],
    *,
    projection_precision: Precision = Precision.EXACT,
    projection_reason: str | None = None,
) -> dict[ActionClass, Metric]:
    degraded_shell = any(
        action.tool_kind == "shell" and action.classification_precision is Precision.UNRESOLVED for action in actions
    )
    estimated_shell = any(
        action.tool_kind == "shell" and action.classification_precision is Precision.ESTIMATED for action in actions
    )
    metrics: dict[ActionClass, Metric] = {}
    for action_class in ActionClass:
        matching = [action for action in actions if action.classification is action_class]
        precision = least_precision((projection_precision, *(action.classification_precision for action in matching)))
        reason = None
        if projection_precision is not Precision.EXACT:
            reason = projection_reason or "tool action projection is incomplete"
        elif action_class in {ActionClass.VERIFY, ActionClass.OTHER} and degraded_shell:
            precision = Precision.UNRESOLVED
            reason = "one or more shell command carriers are unavailable"
        elif (
            action_class in {ActionClass.VERIFY, ActionClass.OTHER} and estimated_shell and precision is Precision.EXACT
        ):
            # The word-list heuristic decides between the verify and other
            # buckets, so either count is only as precise as the heuristic
            # even when every shell action landed in the opposite bucket.
            precision = Precision.ESTIMATED
            reason = _WORD_LIST_HEURISTIC_REASON
        elif precision is Precision.ESTIMATED:
            reason = _WORD_LIST_HEURISTIC_REASON
        elif precision is Precision.UNRESOLVED:
            reason = "one or more action classifications are unresolved"
        metrics[action_class] = Metric(len(matching), precision, reason)
    return metrics


def _validation_metrics(
    actions: tuple[ActionOperation, ...],
    turns: list[TurnAnalysis],
    change_verification: ChangeVerification,
    retry_amplification: Metric,
    *,
    cancel_event: Event | None = None,
) -> ValidationMetrics:
    _check_cancelled(cancel_event)
    projection_precision = least_precision(turn.action_projection_precision for turn in turns)
    projection_reason = next(
        (turn.action_projection_reason for turn in turns if turn.action_projection_reason is not None),
        None,
    )
    counts = _action_count_metrics(
        actions,
        projection_precision=projection_precision,
        projection_reason=projection_reason,
    )
    degraded_shell = any(
        action.tool_kind == "shell" and action.classification_precision is Precision.UNRESOLVED for action in actions
    )
    estimated_shell = any(
        action.tool_kind == "shell" and action.classification_precision is Precision.ESTIMATED for action in actions
    )
    verification_reason = "one or more shell command carriers are unavailable"
    edits = [action for action in actions if action.classification is ActionClass.EDIT]
    verifies = [action for action in actions if action.classification is ActionClass.VERIFY]
    incomplete_outcomes = any(action.outcome_precision is not Precision.EXACT for action in actions)
    unknown_outcomes = any(action.outcome == ToolOutcome.UNKNOWN for action in actions)
    outcome_precision = Precision.UNRESOLVED if incomplete_outcomes or unknown_outcomes else Precision.EXACT
    outcome_reason = (
        next((action.outcome_reason for action in actions if action.outcome_reason is not None), None)
        if incomplete_outcomes
        else "one or more tool outcomes are unknown"
        if unknown_outcomes
        else None
    )
    verify_outcomes_unresolved = any(
        action.outcome_precision is not Precision.EXACT or action.outcome == ToolOutcome.UNKNOWN for action in verifies
    )
    first_edit = edits[0] if edits else None
    time_to_edit = _time_to_action(turns, first_edit)
    first_verify = next(
        (action for action in verifies if first_edit is not None and verify_covers_edit(first_edit, action)),
        None,
    )
    if first_edit is not None and first_verify is not None:
        edit_to_verify = _duration_between_actions(
            turns,
            first_edit,
            first_verify,
            precisions=(first_edit.classification_precision, first_verify.classification_precision),
        )
        if degraded_shell:
            edit_to_verify = _with_unresolved_precision(edit_to_verify, verification_reason)
    elif degraded_shell and first_edit is not None:
        edit_to_verify = Metric(None, Precision.UNRESOLVED, verification_reason)
    elif estimated_shell and first_edit is not None:
        # An estimated other-classified shell action may have been the
        # first verify, so "no verify followed" cannot be stated as fact.
        edit_to_verify = Metric(None, Precision.UNRESOLVED, _WORD_LIST_HEURISTIC_REASON)
    else:
        edit_to_verify = Metric(None, Precision.MISSING, "no verify action followed the first edit")
    cycle_metrics: list[Metric] = []
    pending_edits: list[ActionOperation] = []
    for action in actions:
        _check_cancelled(cancel_event)
        if action.classification is ActionClass.EDIT:
            pending_edits.append(action)
        # A verify closes the open iteration only when it began after every
        # pending edit's terminal landed; one overlapping any of them
        # leaves the whole batch pending for a later ordered verify.
        elif (
            action.classification is ActionClass.VERIFY
            and tool_succeeded(action.outcome)
            and pending_edits
            and all(verify_covers_edit(edit, action) for edit in pending_edits)
        ):
            cycle_metrics.append(
                _duration_between_actions(
                    turns,
                    pending_edits[0],
                    action,
                    precisions=(
                        *(edit.classification_precision for edit in pending_edits),
                        action.classification_precision,
                        action.outcome_precision,
                    ),
                )
            )
            pending_edits = []
    cycle_values = [metric.value for metric in cycle_metrics if isinstance(metric.value, int)]
    cycle_precisions = [metric.precision for metric in cycle_metrics]
    if degraded_shell or verify_outcomes_unresolved:
        cycle_precisions.append(Precision.UNRESOLVED)
    elif estimated_shell:
        # An unrecognized shell verify could have opened or closed more
        # cycles than the word list identified.
        cycle_precisions.append(Precision.ESTIMATED)
    cycle_precision = least_precision(cycle_precisions)
    cycle_reason = (
        verification_reason
        if degraded_shell
        else outcome_reason
        if verify_outcomes_unresolved
        else _WORD_LIST_HEURISTIC_REASON
        if estimated_shell
        else next((metric.reason for metric in cycle_metrics if metric.reason is not None), None)
    )
    successful_verifies = [action for action in verifies if tool_succeeded(action.outcome)]
    last_successful_verify = successful_verifies[-1] if successful_verifies else None
    unverified = [
        action
        for action in edits
        if last_successful_verify is None or not verify_covers_edit(action, last_successful_verify)
    ]
    unverified_precision = least_precision(action.classification_precision for action in (*edits, *verifies))
    unverified_reason = None
    if degraded_shell:
        unverified_precision = Precision.UNRESOLVED
        unverified_reason = verification_reason
    elif verify_outcomes_unresolved:
        unverified_precision = Precision.UNRESOLVED
        unverified_reason = outcome_reason
    elif estimated_shell and unverified_precision is Precision.EXACT:
        # An estimated other-classified shell action may really have been
        # verification work, so the absence this count rests on is only as
        # good as the word-list heuristic.
        unverified_precision = Precision.ESTIMATED
        unverified_reason = _WORD_LIST_HEURISTIC_REASON
    elif unverified_precision is Precision.ESTIMATED:
        unverified_reason = _WORD_LIST_HEURISTIC_REASON
    failures = [action for action in actions if tool_failed(action.outcome)]
    failure_groups: dict[tuple[str | None, str | None], list[ActionOperation]] = defaultdict(list)
    for action in failures:
        _check_cancelled(cancel_event)
        if action.tool_name is not None and action.argument_fingerprint is not None:
            failure_groups[(action.tool_name, action.argument_fingerprint)].append(action)
    repeated_signatures = sum(len(group) >= 2 for group in failure_groups.values())
    recovery_metrics: list[Metric] = []
    for failure in failures:
        _check_cancelled(cancel_event)
        # A recovery responds to an observed failure, so candidates must
        # begin after the failure's terminal — a concurrent same-tool
        # success is not a retry, and without the terminal nothing orders
        # any later success against the failure. A failure with no tool
        # name cannot be paired at all: matching nameless calls would
        # marry unrelated tools.
        if failure.tool_name is None:
            recovery_metrics.append(Metric(None, Precision.UNRESOLVED, "the failed call lacks a tool identity"))
            continue
        if failure.end_sequence is None:
            recovery_metrics.append(Metric(None, Precision.UNRESOLVED, "the failed call's terminal never landed"))
            continue
        recovered: ActionOperation | None = None
        # A same-tool action with an unknown or imprecise outcome between
        # the failure and the recognized success may itself have been the
        # recovery, so the measured latency cannot claim exactness past it.
        passed_unknown_outcome = False
        for action in actions:
            if action.start_sequence <= failure.end_sequence or action.tool_name != failure.tool_name:
                continue
            if tool_succeeded(action.outcome):
                recovered = action
                break
            if action.outcome == ToolOutcome.UNKNOWN or action.outcome_precision is not Precision.EXACT:
                passed_unknown_outcome = True
        if recovered is not None:
            metric = _duration_between_actions(
                turns,
                failure,
                recovered,
                from_end=True,
                precisions=(failure.outcome_precision, recovered.outcome_precision),
            )
            if passed_unknown_outcome:
                metric = _with_unresolved_precision(
                    metric, "a same-tool call with an unknown outcome preceded the recognized recovery"
                )
            recovery_metrics.append(metric)
    if any(action.tool_name is None or action.argument_fingerprint is None for action in failures):
        signature_precision = Precision.UNRESOLVED
        signature_reason = "one or more failed tools lack a name or argument fingerprint"
    else:
        signature_precision = outcome_precision
        signature_reason = outcome_reason
    recovery_values = [metric.value for metric in recovery_metrics if isinstance(metric.value, int)]
    recovery_precision = least_precision(metric.precision for metric in recovery_metrics)
    recovery_reason = next((metric.reason for metric in recovery_metrics if metric.reason is not None), outcome_reason)
    validation = ValidationMetrics(
        funnel=ActionFunnel(
            search=counts[ActionClass.SEARCH],
            read=counts[ActionClass.READ],
            edit=counts[ActionClass.EDIT],
            verify=counts[ActionClass.VERIFY],
        ),
        time_to_first_edit_ns=time_to_edit,
        first_edit_to_first_verify_ns=edit_to_verify,
        edit_verify_cycle_count=Metric(len(cycle_metrics), cycle_precision, cycle_reason),
        edit_verify_cycle_median_ns=(
            Metric(int(median(cycle_values)), cycle_precision, cycle_reason)
            if cycle_values and len(cycle_values) == len(cycle_metrics)
            else Metric(None, Precision.UNRESOLVED, cycle_reason)
            if cycle_metrics
            else Metric(None, Precision.UNRESOLVED, verification_reason)
            if degraded_shell
            else Metric(None, Precision.UNRESOLVED, outcome_reason)
            if verify_outcomes_unresolved and first_edit is not None
            else Metric(None, Precision.UNRESOLVED, _WORD_LIST_HEURISTIC_REASON)
            if estimated_shell and first_edit is not None
            else Metric(None, Precision.MISSING, "no completed edit-to-verify cycle was recorded")
        ),
        unverified_change_count=Metric(len(unverified), unverified_precision, unverified_reason),
        net_zero_churn_count=change_verification.net_zero,
        repeated_failure_signature_count=Metric(repeated_signatures, signature_precision, signature_reason),
        failure_recovery_median_ns=(
            Metric(int(median(recovery_values)), recovery_precision, recovery_reason)
            if recovery_values and len(recovery_values) == len(recovery_metrics)
            else Metric(None, Precision.UNRESOLVED, recovery_reason)
            if recovery_metrics
            else Metric(None, Precision.UNRESOLVED, outcome_reason)
            if outcome_precision is Precision.UNRESOLVED
            else Metric(None, Precision.MISSING, "no failed tool call was followed by a same-tool success")
        ),
        tool_failure_count=Metric(len(failures), outcome_precision, outcome_reason),
        tool_count=Metric(
            len(actions),
            projection_precision,
            None if projection_precision is Precision.EXACT else projection_reason,
        ),
        retry_amplification_tokens=retry_amplification,
    )
    if projection_precision is Precision.EXACT:
        return validation
    reason = projection_reason or "tool action projection is incomplete"
    return _degrade_validation_metrics(validation, reason)


def _time_to_action(turns: list[TurnAnalysis], action: ActionOperation | None) -> Metric:
    if action is None:
        return Metric(None, Precision.MISSING, "no edit action was recorded")
    elapsed = 0
    precisions = [action.classification_precision]
    for turn in turns:
        if turn.turn_id == action.turn_id:
            offset = action.start_ns - turn.axis_start_ns
            if offset < 0:
                return Metric(None, Precision.UNRESOLVED, "action precedes its owning turn")
            return Metric(elapsed + offset, least_precision(precisions), action.classification_reason)
        if not isinstance(turn.elapsed_ns.value, int):
            return Metric(None, Precision.UNRESOLVED, "an earlier turn duration is unresolved")
        elapsed += turn.elapsed_ns.value
        precisions.append(turn.elapsed_ns.precision)
    return Metric(None, Precision.UNRESOLVED, "action turn is absent from the active projection")


def _duration_between_actions(
    turns: list[TurnAnalysis],
    first: ActionOperation,
    second: ActionOperation,
    *,
    from_end: bool = False,
    precisions: tuple[Precision, ...],
) -> Metric:
    start_ns = first.end_ns if from_end else first.start_ns
    if start_ns is None:
        return Metric(None, Precision.MISSING, "the starting tool outcome has no terminal timestamp")
    if first.turn_id == second.turn_id:
        duration = second.start_ns - start_ns
        if duration < 0:
            return Metric(None, Precision.UNRESOLVED, "action order yields a negative duration")
        return Metric(duration, least_precision(precisions))
    turns_by_id = {turn.turn_id: index for index, turn in enumerate(turns)}
    first_index = turns_by_id.get(first.turn_id)
    second_index = turns_by_id.get(second.turn_id)
    if first_index is None or second_index is None or second_index <= first_index:
        return Metric(None, Precision.UNRESOLVED, "action turns are not ordered in the active projection")
    first_turn = turns[first_index]
    second_turn = turns[second_index]
    if not isinstance(first_turn.elapsed_ns.value, int):
        return Metric(None, Precision.UNRESOLVED, "the starting turn duration is unresolved")
    start_offset = start_ns - first_turn.axis_start_ns
    end_offset = second.start_ns - second_turn.axis_start_ns
    duration = first_turn.elapsed_ns.value - start_offset + end_offset
    metric_precisions = [*precisions, first_turn.elapsed_ns.precision]
    for turn in turns[first_index + 1 : second_index]:
        if not isinstance(turn.elapsed_ns.value, int):
            return Metric(None, Precision.UNRESOLVED, "an intervening turn duration is unresolved")
        duration += turn.elapsed_ns.value
        metric_precisions.append(turn.elapsed_ns.precision)
    if duration < 0:
        return Metric(None, Precision.UNRESOLVED, "action order yields a negative duration")
    return Metric(duration, least_precision(metric_precisions))


def _degrade_validation_metrics(
    validation: ValidationMetrics,
    reason: str,
    *,
    session_integrity_cap: bool = False,
) -> ValidationMetrics:
    def degrade(metric: Metric) -> Metric:
        return (
            _cap_session_metric(metric, reason) if session_integrity_cap else _with_unresolved_precision(metric, reason)
        )

    return ValidationMetrics(
        funnel=ActionFunnel(
            search=degrade(validation.funnel.search),
            read=degrade(validation.funnel.read),
            edit=degrade(validation.funnel.edit),
            verify=degrade(validation.funnel.verify),
        ),
        time_to_first_edit_ns=degrade(validation.time_to_first_edit_ns),
        first_edit_to_first_verify_ns=degrade(validation.first_edit_to_first_verify_ns),
        edit_verify_cycle_count=degrade(validation.edit_verify_cycle_count),
        edit_verify_cycle_median_ns=degrade(validation.edit_verify_cycle_median_ns),
        unverified_change_count=degrade(validation.unverified_change_count),
        net_zero_churn_count=degrade(validation.net_zero_churn_count),
        repeated_failure_signature_count=degrade(validation.repeated_failure_signature_count),
        failure_recovery_median_ns=degrade(validation.failure_recovery_median_ns),
        tool_failure_count=degrade(validation.tool_failure_count),
        tool_count=degrade(validation.tool_count),
        retry_amplification_tokens=degrade(validation.retry_amplification_tokens),
    )


def _resolve_change_verification(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    turns: list[TurnAnalysis],
    actions: tuple[ActionOperation, ...],
    projection: _SessionProjection,
    *,
    rollback_projection_unresolved: bool,
    cancel_event: Event | None,
) -> ChangeVerification:
    _check_cancelled(cancel_event)
    active_turn_numbers = {turn.turn_number for turn in turns if turn.turn_number is not None}
    active_turn_ids = {turn.turn_id for turn in turns}
    # The detailed mutation state joins on turn numbers; a missing,
    # damaged, or duplicated number silently drops or misattributes that
    # turn's rows, so the join cannot claim the detailed counts.
    seen_turn_numbers: set[int] = set()
    previous_turn_number: int | None = None
    turn_numbers_joinable = True
    for turn in turns:
        number = turn.turn_number
        repeated_without_contiguous_retry = number in seen_turn_numbers and (
            not turn.attempts[0].is_retry or number != previous_turn_number
        )
        if number is None or number <= 0 or repeated_without_contiguous_retry:
            turn_numbers_joinable = False
            break
        seen_turn_numbers.add(number)
        previous_turn_number = number
    summaries = [
        endpoint
        for endpoint in intermediate.mutation_summaries
        if _active(endpoint.sequence, inactive_ranges) and endpoint.turn_id in active_turn_ids
    ]
    # A summary that names no session checkpoint recorded mutations whose
    # save never completed: the readable document predates them, so its
    # detail cannot be read as the current truth. A later successful save
    # re-serializes the whole mutation log, so only the newest summary
    # decides freshness. Only a ref of the session-checkpoint kind carrying
    # an id the save path could have minted vouches for the document; any
    # other shape proves nothing about it.
    latest_summary = max(summaries, key=lambda endpoint: endpoint.sequence, default=None)
    source_ref = None if latest_summary is None else _payload_value(latest_summary.payload, "source_ref")
    detail_current = latest_summary is None or (
        isinstance(source_ref, dict)
        and source_ref.get("kind") == SourceRefKind.SESSION_CHECKPOINT
        and is_valid_analytics_id(source_ref.get("id"))
    )
    if (
        not projection.available
        or not projection.mutation_detail_available
        or not detail_current
        or not turn_numbers_joinable
    ):
        # Per-turn summaries carry counts but no file identities, so a file
        # touched in several turns counts once per turn, and a create followed
        # by a later modify never folds into one created file — the sums are
        # honest per-turn totals, never the session-wide folded counts the
        # detailed projection reports.
        precision = Precision.UNRESOLVED if rollback_projection_unresolved else Precision.ESTIMATED
        reason = (
            "recorded summary counts only; rollback projection and session.json detail are unavailable"
            if precision is Precision.UNRESOLVED
            else "summed per-turn summary counts; the active turns' numbers cannot join the session.json file detail"
            if not turn_numbers_joinable
            else "summed per-turn summary counts; usable session.json file detail is unavailable to fold repeat touches"
        )

        def summary(key: str) -> Metric:
            values = [_payload_int(endpoint.payload, key) for endpoint in summaries]
            if any(value is None or value < 0 for value in values):
                return Metric(None, Precision.MISSING, f"recorded mutation summary is missing or invalid for {key}")
            return Metric(sum(cast("int", value) for value in values), precision, reason)

        return ChangeVerification(
            detail_available=False,
            detection_truncated=False,
            files_touched=summary("files_touched"),
            created=summary("create"),
            modified=summary("modify"),
            deleted=summary("delete"),
            net_zero=summary("net_zero_count"),
        )
    selected = sorted(
        (
            mutation
            for mutation in projection.mutations
            if mutation.turn_number in active_turn_numbers and mutation.provenance != "foreign"
        ),
        key=lambda mutation: mutation.turn_number,
    )
    folded: dict[str, tuple[FileHashDiff, int]] = {}

    def fold(
        path: str,
        turn_number: int,
        *,
        before_hash: str | None,
        before_skip: str | None,
        after_hash: str | None,
        after_skip: str | None,
        inferred: bool,
        contested: bool,
    ) -> None:
        previous = folded.get(path)
        before = previous[0].before if previous is not None else before_hash
        folded_before_skip = previous[0].before_skip if previous is not None else parse_skip_reason(before_skip)
        folded[path] = (
            FileHashDiff(
                before=before,
                after=after_hash,
                before_skip=folded_before_skip,
                after_skip=parse_skip_reason(after_skip),
                contested=(previous is not None and previous[0].contested) or contested,
                inferred=(previous is not None and previous[0].inferred) or inferred,
            ),
            turn_number,
        )

    for mutation in selected:
        _check_cancelled(cancel_event)
        # Only a proven row attests that this session's own write produced
        # the change; anything else folded from a window diff may be a
        # concurrent third party's, so the folded diff keeps the badge.
        inferred = mutation.provenance != "proven"
        fold(
            mutation.path,
            mutation.turn_number,
            before_hash=mutation.before_hash,
            before_skip=mutation.before_skip,
            after_hash=mutation.after_hash,
            after_skip=mutation.after_skip,
            inferred=inferred,
            contested=mutation.contested,
        )
        if mutation.old_path is not None:
            # The move's own before_hash describes the destination; the
            # source folds as a delete from its snapshotted pre-state.
            fold(
                mutation.old_path,
                mutation.turn_number,
                before_hash=mutation.old_before_hash,
                before_skip=mutation.old_before_skip,
                after_hash=None,
                after_skip=None,
                inferred=inferred,
                contested=mutation.contested,
            )
    detection_truncated = bool(projection.detection_truncated_turns & active_turn_numbers)
    metric_precision = Precision.UNRESOLVED if detection_truncated else Precision.EXACT
    metric_reason = "recorded/observed file counts; mutation detection was truncated" if detection_truncated else None
    created = modified = deleted = net_zero = 0
    rows: list[ChangeVerificationRow] = []
    successful_verifies = [
        action for action in actions if action.classification is ActionClass.VERIFY and tool_succeeded(action.outcome)
    ]
    # Batched tool calls run concurrently, so an action having *started*
    # before a verify proves nothing about what the verify observed; the
    # verify vouches for a same-turn change only when it began after every
    # action that could have produced the change finished landing. Mutation
    # rows carry no operation attribution, so the ordering must clear every
    # potential mutator — edits, the other-classified actions (shell
    # commands and the like), and the verify-classified peers of the
    # candidate verify (a fixer like `ruff --fix` matches the word list
    # yet mutates) — while the row's evidence and the "orderable at all"
    # question stay with the edit-classified actions alone.
    edit_terminals_by_turn: dict[int, list[int | None]] = {turn_number: [] for turn_number in active_turn_numbers}
    mutator_terminals_by_turn: dict[int, list[int | None]] = {turn_number: [] for turn_number in active_turn_numbers}
    verify_terminals_by_turn: dict[int, list[tuple[str, int | None]]] = {
        turn_number: [] for turn_number in active_turn_numbers
    }
    for action in actions:
        if action.turn_number not in active_turn_numbers:
            continue
        if action.classification is ActionClass.EDIT:
            edit_terminals_by_turn[action.turn_number].append(action.end_sequence)
        if action.classification in (ActionClass.EDIT, ActionClass.OTHER):
            mutator_terminals_by_turn[action.turn_number].append(action.end_sequence)
        if action.classification is ActionClass.VERIFY:
            verify_terminals_by_turn[action.turn_number].append((action.operation_id, action.end_sequence))
    # A shell action whose command carrier is unavailable could have been
    # the verify an unverified row claims never happened, and a recognized
    # verify whose terminal outcome never resolved could have been the
    # success; an estimated shell action may have been verification work
    # the word list missed.
    verify_absence_precision = (
        Precision.UNRESOLVED
        if any(
            (action.tool_kind == "shell" and action.classification_precision is Precision.UNRESOLVED)
            or (
                action.classification is ActionClass.VERIFY
                and (action.outcome_precision is not Precision.EXACT or action.outcome == ToolOutcome.UNKNOWN)
            )
            for action in actions
        )
        else Precision.ESTIMATED
        if any(
            action.tool_kind == "shell" and action.classification_precision is Precision.ESTIMATED for action in actions
        )
        else Precision.EXACT
    )
    edit_evidence_by_turn = {
        turn_number: tuple(
            sorted(
                action.evidence_key
                for action in actions
                if action.turn_number == turn_number
                and action.classification is ActionClass.EDIT
                and action.evidence_key is not None
            )
        )
        for turn_number in active_turn_numbers
    }
    unprovable_net_zero = False
    for path, (diff, last_turn) in folded.items():
        _check_cancelled(cancel_event)
        if not diff.before_exists and diff.after_exists:
            created += 1
        elif diff.before_exists and not diff.after_exists:
            deleted += 1
        else:
            modified += 1
        # A window-inferred or peer-contested fold may describe another
        # writer's change, so no row built from one can claim exactness.
        provenance_precision = Precision.ESTIMATED if diff.inferred or diff.contested else Precision.EXACT
        if diff.is_net_zero:
            net_zero += 1
            state = ChangeVerificationState.NET_ZERO
            if diff.content_unavailable:
                # Both content backups were withheld: existence never
                # flipped, but a return to the original bytes cannot be
                # proven either way.
                unprovable_net_zero = True
                row_precision = Precision.UNRESOLVED
            else:
                row_precision = least_precision((metric_precision, provenance_precision))
        else:
            latest_verify = successful_verifies[-1] if successful_verifies else None
            edit_terminals = edit_terminals_by_turn.get(last_turn, [])
            mutator_terminals = list(mutator_terminals_by_turn.get(last_turn, []))
            if latest_verify is not None:
                mutator_terminals.extend(
                    end_sequence
                    for operation_id, end_sequence in verify_terminals_by_turn.get(last_turn, [])
                    if operation_id != latest_verify.operation_id
                )
            orderable = bool(edit_terminals) and None not in mutator_terminals
            if latest_verify is None or latest_verify.turn_number is None:
                state = ChangeVerificationState.UNVERIFIED
                row_precision = least_precision((metric_precision, provenance_precision, verify_absence_precision))
            elif latest_verify.turn_number == last_turn and not orderable:
                # The turn's changes came from actions that never classify
                # as edits (shell commands classify as verify or other) or
                # from a potential mutator whose terminal never landed, so
                # the verify cannot be ordered against the change it would
                # need to follow.
                state = ChangeVerificationState.UNVERIFIED
                row_precision = Precision.UNRESOLVED
            elif latest_verify.turn_number > last_turn or (
                latest_verify.turn_number == last_turn
                and latest_verify.start_sequence > max(cast("list[int]", mutator_terminals))
            ):
                state = ChangeVerificationState.VERIFIED
                row_precision = least_precision(
                    (metric_precision, provenance_precision, latest_verify.classification_precision)
                )
            else:
                # The state leans on having identified the candidate as a
                # verify AND on no unclassifiable shell action having been
                # a later verify that would upgrade the row, so both
                # precisions travel with it.
                state = ChangeVerificationState.AFTER_VERIFY
                row_precision = least_precision(
                    (
                        metric_precision,
                        provenance_precision,
                        latest_verify.classification_precision,
                        verify_absence_precision,
                    )
                )
        rows.append(
            ChangeVerificationRow(
                path,
                state,
                last_turn,
                row_precision,
                edit_evidence_by_turn.get(last_turn, ()),
            )
        )
    diagnostics = [metric_reason] if metric_reason is not None else []
    count_precision = metric_precision
    if any(diff.inferred or diff.contested for diff, _ in folded.values()):
        # The counts then include folds that may describe another writer's
        # change, so they cannot pass for exact per-session totals.
        count_precision = least_precision((count_precision, Precision.ESTIMATED))
        diagnostics.append("counts include window-inferred or peer-contested mutations")
    count_reason = "; ".join(diagnostics) if diagnostics else None
    net_zero_precision = count_precision
    net_zero_reason = count_reason
    if unprovable_net_zero:
        net_zero_precision = least_precision((net_zero_precision, Precision.UNRESOLVED))
        net_zero_reason = "; ".join(
            (*diagnostics, "count includes files whose withheld content backups leave the net change unprovable")
        )
    return ChangeVerification(
        detail_available=True,
        detection_truncated=detection_truncated,
        files_touched=Metric(len(folded), count_precision, count_reason),
        created=Metric(created, count_precision, count_reason),
        modified=Metric(modified, count_precision, count_reason),
        deleted=Metric(deleted, count_precision, count_reason),
        net_zero=Metric(net_zero, net_zero_precision, net_zero_reason),
        rows=tuple(sorted(rows, key=lambda row: row.path)),
    )


def _degrade_change_verification(change: ChangeVerification, reason: str) -> ChangeVerification:
    """Cap every session-wide mutation result when the source log is incomplete."""

    return ChangeVerification(
        detail_available=change.detail_available,
        detection_truncated=change.detection_truncated,
        files_touched=_cap_session_metric(change.files_touched, reason),
        created=_cap_session_metric(change.created, reason),
        modified=_cap_session_metric(change.modified, reason),
        deleted=_cap_session_metric(change.deleted, reason),
        net_zero=_cap_session_metric(change.net_zero, reason),
        rows=tuple(
            row if row.precision is Precision.MISSING else replace(row, precision=Precision.UNRESOLVED)
            for row in change.rows
        ),
    )
