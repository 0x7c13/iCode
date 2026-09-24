# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Physical turn, timeline, revision, and dependency resolution."""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise
from threading import Event
from typing import Final, cast

from chrys.foundation.trajectory.envelope import LinkRelation
from chrys.foundation.trajectory.event_types import RetryMode, ToolOutcome, TurnEndReason, WaitCategory
from chrys.foundation.trajectory.ids import is_valid_analytics_id
from chrys.foundation.trajectory.revisions import MembershipRef, membership_hash
from chrys.service.analytics._critical_path import _longest_interval_path, _PathResolution
from chrys.service.analytics._facts import (
    _HOOK_EXECUTION_MODES,
    _START_FAMILIES,
    _active,
    _Endpoint,
    _ExchangeUsage,
    _Intermediate,
    _lifecycle_cut,
    _LifecycleCut,
    _Node,
    _payload_bool,
    _payload_int,
    _payload_str,
    _payload_value,
    _RevisionEntry,
    _Segment,
    _ToolContextExtras,
    _Turn,
)
from chrys.service.analytics._timeline import (
    _endpoint_in_coverage_runtime,
    _infrastructure_branch_reaches,
    _materialize_timeline,
    _resolved_timeline_operation,
    _ResolvedNode,
    _timeline_hook_id,
    _timeline_identity,
    _timeline_parent_operation,
    _TimelineProjection,
)
from chrys.service.analytics.math import clip_interval, interval_length, interval_union, subtract_intervals
from chrys.service.analytics.model import (
    FLOW_TERMINAL_INDEX,
    ContainmentViolation,
    ContextSample,
    HookOwnership,
    Metric,
    Precision,
    SpanDurationMismatch,
    TimelineDiagnosticCode,
    TimelineOperation,
    TimeSlice,
    TokenUsage,
    TokenUsageSample,
    TurnAnalysis,
    TurnAttemptRef,
    TurnFlow,
    UsageBucket,
    WallBucket,
)
from chrys.service.analytics.reader import raise_if_cancelled as _check_cancelled
from chrys.service.hooks.events import HookEvent

_WEIGHTED_FAMILIES: Final = frozenset(_START_FAMILIES.values()) | {"approval", "compaction.phase"}
_WALL_PRIORITY: Final = {
    WallBucket.IDLE: 0,
    WallBucket.WAIT: 1,
    WallBucket.TOOLS: 2,
    WallBucket.MODEL: 3,
}
_KNOWN_TOOL_OUTCOMES: Final = frozenset(
    {
        ToolOutcome.SUCCESS,
        ToolOutcome.FAILED,
        ToolOutcome.ERRORED,
        ToolOutcome.INTERRUPTED,
        ToolOutcome.TIMED_OUT,
        ToolOutcome.REJECTED,
        ToolOutcome.INVALID_ARGUMENTS,
        ToolOutcome.UNKNOWN_TOOL,
        ToolOutcome.FILTERED,
        ToolOutcome.UNKNOWN,
    }
)
_FAILED_TOOL_OUTCOMES: Final = _KNOWN_TOOL_OUTCOMES - {
    ToolOutcome.SUCCESS,
    ToolOutcome.INTERRUPTED,
    ToolOutcome.UNKNOWN,
}
# Calls closed before dispatch have no tool preamble and no caused-by edge.
_NEVER_DISPATCHED_TOOL_OUTCOMES: Final = frozenset(
    {
        ToolOutcome.INVALID_ARGUMENTS,
        ToolOutcome.UNKNOWN_TOOL,
        ToolOutcome.FILTERED,
    }
)
_OPTIONAL_USAGE_BUCKETS: Final = (UsageBucket.REASONING, UsageBucket.CACHE_READ, UsageBucket.CACHE_CREATION)
_MODEL_OPERATION_FAMILIES: Final = frozenset({"model.run", "model.cycle", "model.exchange"})
_RETRY_EXCHANGE_MODES: Final = frozenset({RetryMode.WIRE, RetryMode.STALL_FALLBACK, RetryMode.VALIDATION})


@dataclass(frozen=True, slots=True)
class _DependencyProof:
    edges: dict[str, set[str]]
    turn_root_id: str | None
    response_terminal_id: str | None
    # Displacing child -> parent pairs; an edge absent here is a causal pointer.
    parents: dict[str, str]
    # Edges that fork a concurrent hook off its target; the hook only depends
    # on the target's triggering event, never on the target completing.
    fork_edges: frozenset[tuple[str, str]] = frozenset()


@dataclass(frozen=True, slots=True)
class _RevisionResolution:
    memberships: dict[str, tuple[str, ...]]
    endpoints: dict[str, _Endpoint]
    errors: dict[str, tuple[str, ...]]
    side_call_empty_shell_revisions: tuple[str, ...]
    # Replayed and hash-verified, but the producer reported items it could not
    # name: membership is a lower bound, good for positive claims only.
    unidentified_membership_revision_count: int = 0


@dataclass(frozen=True, slots=True)
class _Phase4Consumption:
    item_ids: frozenset[str]
    next_exchange_operation_id: str


@dataclass(frozen=True, slots=True)
class _CompactionConsumptionResolution:
    by_run: dict[str, _Phase4Consumption]
    exact: bool


@dataclass(slots=True)
class _ResolutionCache:
    """Resolve-derived state retained for exactly one live fact index."""

    counter_axis_diagnostics_by_turn: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _tool_context_for_start(node: _Node, start: _Endpoint) -> _ToolContextExtras | None:
    extras = node.extras
    if extras is None:
        return None
    return next((item for item in extras.contexts if item.sequence == start.sequence), None)


def _tool_failed(outcome: str | None) -> bool:
    return outcome in _FAILED_TOOL_OUTCOMES


def _known_duration_offset(node: _ResolvedNode) -> bool:
    if node.family in {"tool.operation", "compaction.phase", "approval"}:
        return True
    return node.family == "model.run" and (
        _payload_int(node.start.payload, "attempt_index") not in {None, 0}
        or _payload_str(node.start.payload, "previous_run_operation_id") is not None
    )


def _turn_token_usage(
    intermediate: _Intermediate,
    turn_id: str,
    inactive_ranges: tuple[tuple[int, int], ...],
    verdict: Metric,
) -> TokenUsage:
    exchanges = [
        item for item in intermediate.usage_by_turn.get(turn_id, ()) if _active(item.sequence, inactive_ranges)
    ]
    return _interned_token_usage(tuple(_turn_bucket_usage(exchanges, bucket, verdict) for bucket in UsageBucket))


def _turn_uses_session_carrier_fallback(
    intermediate: _Intermediate,
    turn_id: str,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    cancel_event: Event | None,
) -> bool:
    for node in intermediate.nodes_by_turn.get(turn_id, ()):
        _check_cancelled(cancel_event)
        if node.family != "tool.operation":
            continue
        for finish in node.finishes:
            if not _active(finish.sequence, inactive_ranges):
                continue
            if (
                _payload_str(finish.payload, "result_item_id") is not None
                and _payload_str(finish.payload, "result_carrier_item_id") is None
            ):
                return True
    return False


def _resolve_turn(
    intermediate: _Intermediate,
    turn: _Turn,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    resolution_cache: _ResolutionCache,
    revisions: _RevisionResolution,
    mapped_carrier: Callable[[str], str | None],
    rollback_projection_unresolved: bool,
    refresh_counter_axis: bool,
    cancel_event: Event | None,
) -> TurnAnalysis:
    _check_cancelled(cancel_event)
    diagnostics: list[str] = []
    starts = [event for event in turn.starts if _active(event.sequence, inactive_ranges)]
    finishes = [event for event in turn.finishes if _active(event.sequence, inactive_ranges)]
    if len(starts) != 1:
        return _empty_turn(
            intermediate,
            turn.turn_id,
            starts,
            "turn lifecycle is not uniquely opened",
            inactive_ranges=inactive_ranges,
            resolution_cache=resolution_cache,
            revisions=revisions,
            refresh=refresh_counter_axis,
            cancel_event=cancel_event,
        )
    start = starts[0]
    finish = finishes[0] if len(finishes) == 1 else None
    turn_lifecycle_exact = True
    if finish is None:
        diagnostics.append("turn lifecycle has no unique terminal")
        turn_lifecycle_exact = False
    elif finish.scope != start.scope or finish.sequence <= start.sequence or finish.monotonic_ns < start.monotonic_ns:
        diagnostics.append("turn interval has invalid monotonic endpoints")
        turn_lifecycle_exact = False
    tail_end, tail_exact, end_sequence, waited_hook_ids = _turn_tail(
        intermediate, turn, start, finish, inactive_ranges, diagnostics
    )
    visible_end = tail_end or (finish.monotonic_ns if finish is not None else start.monotonic_ns)
    turn_bounds = (start.monotonic_ns, max(start.monotonic_ns, visible_end))
    counter_axis_diagnostics = _counter_axis_diagnostics(
        intermediate,
        turn.turn_id,
        inactive_ranges,
        resolution_cache=resolution_cache,
        axis_start_ns=turn_bounds[0],
        axis_end_ns=turn_bounds[1],
        revisions=revisions,
        refresh=refresh_counter_axis,
        cancel_event=cancel_event,
    )
    integrity_diagnostics: list[str] = []
    coverage_exact = _turn_coverage_exact(
        intermediate,
        start,
        end_sequence,
        inactive_ranges,
        integrity_diagnostics,
        rollback_projection_unresolved=rollback_projection_unresolved,
    )
    diagnostics.extend(integrity_diagnostics)
    resolved_nodes, lifecycle_exact = _resolved_turn_nodes(intermediate, start, inactive_ranges, diagnostics)
    closed_exchange_terminal_sequences = frozenset(
        node.finish.sequence for node in resolved_nodes if node.family == "model.exchange"
    )
    resolved_nodes = [node for node in resolved_nodes if not node.start.side_call]
    compaction_consumption = _compaction_consumption_resolution(
        intermediate,
        resolved_nodes,
        inactive_ranges,
        cancel_event=cancel_event,
        diagnostics=diagnostics,
    )
    _check_cancelled(cancel_event)
    resolved_nodes.extend(_continuation_poll_nodes(resolved_nodes))
    suspension_nodes, suspension_deductions, suspension_exact = _suspension_nodes(
        turn, resolved_nodes, start, finish, inactive_ranges, diagnostics
    )
    resolved_nodes.extend(suspension_nodes)
    slices, dependency, timeline_exact, dag_exact, response_dependency_exact = _project_turn(
        resolved_nodes,
        turn_bounds,
        start=start,
        finish=finish,
        tail_end=tail_end,
        waited_hook_ids=waited_hook_ids,
        suspension_deductions=suspension_deductions,
        revisions=revisions,
        compaction_consumption=compaction_consumption,
        mapped_carrier=mapped_carrier,
        cancel_event=cancel_event,
        diagnostics=diagnostics,
    )
    wall_durations, idle_intervals = _wall_partition(turn_bounds, slices, cancel_event=cancel_event)
    idle_slices = tuple(
        TimeSlice(
            family="idle",
            slice_index=index,
            turn_id=turn.turn_id,
            runtime_id=start.runtime_id,
            operation_id=None,
            owner="idle(unattributed)",
            start_ns=idle_start,
            end_ns=idle_end,
            wall_bucket=WallBucket.IDLE,
            counts_as_work=False,
            compute_weight=False,
            response_weight=True,
        )
        for index, (idle_start, idle_end) in enumerate(idle_intervals)
    )
    all_slices = tuple(sorted((*slices, *idle_slices), key=lambda item: (item.start_ns, item.end_ns, item.slice_id)))
    work_slices = [item for item in all_slices if item.counts_as_work]
    exclusive_work = sum(item.duration_ns for item in work_slices)
    work_union = interval_length((item.start_ns, item.end_ns) for item in work_slices)
    overlap_gain = exclusive_work - work_union
    elapsed = turn_bounds[1] - turn_bounds[0]
    precision = (
        Precision.EXACT
        if coverage_exact
        and turn_lifecycle_exact
        and lifecycle_exact
        and tail_exact
        and timeline_exact
        and suspension_exact
        else Precision.UNRESOLVED
    )
    causes_exact = _required_causes_present(resolved_nodes, diagnostics)
    compute_cp_exact = (
        coverage_exact and turn_lifecycle_exact and lifecycle_exact and timeline_exact and dag_exact and causes_exact
    )
    response_cp_exact = compute_cp_exact and tail_exact and suspension_exact and response_dependency_exact
    compute_cp, response_cp, acyclic, response_reachable, compute_bounded, response_bounded = _critical_paths(
        resolved_nodes,
        all_slices,
        dependency,
        cancel_event=cancel_event,
    )
    if not compute_bounded:
        diagnostics.append("compute critical-path candidate cap exceeded")
        compute_cp_exact = False
    if not response_bounded:
        diagnostics.append("response critical-path candidate cap exceeded")
        response_cp_exact = False
    if not acyclic:
        diagnostics.append("operation dependency graph contains a cycle")
        compute_cp_exact = False
        response_cp_exact = False
    if not response_reachable:
        diagnostics.append("terminal response is not reachable from the turn root through typed edges")
        response_cp_exact = False
    critical_tool_contributions = (
        _failed_tool_cp_contributions(
            resolved_nodes,
            all_slices,
            dependency,
            response_cp=response_cp,
            cancel_event=cancel_event,
        )
        if response_cp_exact and response_cp is not None
        else {}
    )
    server_critical_contributions = (
        _server_tool_cp_contributions(
            intermediate,
            resolved_nodes,
            all_slices,
            dependency,
            response_cp=response_cp,
            cancel_event=cancel_event,
        )
        if response_cp_exact and response_cp is not None
        else {}
    )
    metric_reason = None if precision is Precision.EXACT else "; ".join(dict.fromkeys(diagnostics))
    unresolved_reason = "; ".join(dict.fromkeys(diagnostics))
    compute_cp_reason = None if compute_cp_exact else unresolved_reason
    response_cp_reason = None if response_cp_exact else unresolved_reason
    wall_metrics = {
        bucket: Metric(value=value, precision=precision, reason=metric_reason)
        for bucket, value in wall_durations.items()
    }
    utilization = {
        bucket: Metric(
            value=(
                sum(item.duration_ns for item in work_slices if item.wall_bucket is bucket) / elapsed
                if elapsed
                else 0.0
            ),
            precision=precision,
            reason=metric_reason,
        )
        for bucket in (WallBucket.MODEL, WallBucket.TOOLS)
    }
    usage = _turn_usage(
        intermediate,
        turn.turn_id,
        inactive_ranges,
        integrity_exact=coverage_exact,
        integrity_reason="; ".join(dict.fromkeys(integrity_diagnostics)),
        turn_lifecycle_exact=turn_lifecycle_exact,
        closed_exchange_terminal_sequences=closed_exchange_terminal_sequences,
        cancel_event=cancel_event,
    )
    token_usage = _turn_token_usage(intermediate, turn.turn_id, inactive_ranges, usage)
    turn_number = _payload_int(start.payload, "turn_number")
    action_projection_diagnostics: list[str] = []
    action_projection_exact = (
        coverage_exact
        and turn_lifecycle_exact
        and _tool_action_projection_exact(
            intermediate,
            start,
            finish.sequence if finish is not None else None,
            inactive_ranges,
            action_projection_diagnostics,
        )
    )
    action_projection_reason = "; ".join(dict.fromkeys((*integrity_diagnostics, *action_projection_diagnostics)))
    operations = _timeline_operations(
        intermediate,
        turn_id=turn.turn_id,
        inactive_ranges=inactive_ranges,
        resolved_nodes=resolved_nodes,
        cancel_event=cancel_event,
    )
    flow = _turn_flow(turn.turn_id, dependency, operations, acyclic=acyclic)
    is_retry = _payload_bool(start.payload, "is_retry")
    return TurnAnalysis(
        turn_id=turn.turn_id,
        turn_number=turn_number,
        runtime_id=start.runtime_id,
        start_sequence=start.sequence,
        end_sequence=end_sequence,
        elapsed_ns=Metric(elapsed, precision, metric_reason),
        compute_cp_ns=Metric(
            compute_cp if compute_cp_exact else None,
            Precision.EXACT if compute_cp_exact else Precision.UNRESOLVED,
            compute_cp_reason,
        ),
        response_cp_ns=Metric(
            response_cp if response_cp_exact else None,
            Precision.EXACT if response_cp_exact else Precision.UNRESOLVED,
            response_cp_reason,
        ),
        exclusive_work_ns=Metric(exclusive_work, precision, metric_reason),
        parallelism=Metric(exclusive_work / elapsed if elapsed else 0.0, precision, metric_reason),
        overlap_gain_ns=Metric(overlap_gain, precision, metric_reason),
        wall_time_ns=wall_metrics,
        utilization=utilization,
        usage_tokens=usage,
        attempts=(
            TurnAttemptRef(
                turn_id=turn.turn_id,
                runtime_id=start.runtime_id,
                is_retry=is_retry,
                physical_axis_start_ns=turn_bounds[0],
                physical_axis_end_ns=turn_bounds[1],
                logical_axis_start_ns=turn_bounds[0],
                operation_start_index=0,
                operation_end_index=len(operations),
                slice_start_index=0,
                slice_end_index=len(all_slices),
            ),
        ),
        axis_start_ns=turn_bounds[0],
        axis_end_ns=turn_bounds[1],
        operations=operations,
        slices=all_slices,
        diagnostics=tuple(dict.fromkeys((*diagnostics, *counter_axis_diagnostics))),
        critical_tool_contributions_ns=critical_tool_contributions,
        server_critical_contributions_ns=server_critical_contributions,
        action_projection_precision=Precision.EXACT if action_projection_exact else Precision.UNRESOLVED,
        action_projection_reason=None if action_projection_exact else action_projection_reason,
        token_usage=token_usage,
        flow=flow,
    )


def _turn_tail(
    intermediate: _Intermediate,
    turn: _Turn,
    start: _Endpoint,
    finish: _Endpoint | None,
    inactive_ranges: tuple[tuple[int, int], ...],
    diagnostics: list[str],
) -> tuple[int | None, bool, int | None, frozenset[str] | None]:
    if finish is None:
        return None, False, None, None
    if finish.scope != start.scope or finish.sequence <= start.sequence or finish.monotonic_ns < start.monotonic_ns:
        return finish.monotonic_ns, False, finish.sequence, None
    end_reason = _payload_str(finish.payload, "end_reason")
    if end_reason in {TurnEndReason.PROCESS_EXIT, TurnEndReason.CANCELLED}:
        return finish.monotonic_ns, True, finish.sequence, frozenset()
    markers = [marker for marker in turn.response_markers if _active(marker.sequence, inactive_ranges)]
    if len(markers) != 1:
        diagnostics.append("turn response fence is missing or duplicated")
        return finish.monotonic_ns, False, finish.sequence, None
    marker = markers[0]
    segment_end_sequence = max(
        (
            segment.sequence
            for segment in intermediate.segments.get(marker.event_id or "", ())
            if _active(segment.sequence, inactive_ranges)
        ),
        default=marker.sequence,
    )
    if marker.scope != start.scope or marker.sequence <= finish.sequence or marker.monotonic_ns < finish.monotonic_ns:
        diagnostics.append("turn response fence is outside the turn runtime or precedes turn.finished")
        return finish.monotonic_ns, False, segment_end_sequence, None
    waited_ids = _reassemble_waited_hook_ids(intermediate, marker, inactive_ranges)
    expected = _payload_int(marker.payload, "waited_hook_operation_count")
    if waited_ids is None or expected is None or len(waited_ids) != expected or len(set(waited_ids)) != expected:
        diagnostics.append("turn response fence hook membership is incomplete")
        return marker.monotonic_ns, False, segment_end_sequence, None
    drained_scopes = _payload_value(marker.payload, "drained_scopes")
    outcome = _payload_str(marker.payload, "outcome")
    if (
        not isinstance(drained_scopes, tuple)
        or not all(isinstance(scope, str) for scope in drained_scopes)
        or len(drained_scopes) != len(set(drained_scopes))
        or any(scope != "turn" for scope in drained_scopes)
        or outcome not in {"settled", "cancelled", "partial"}
        or (outcome == "settled" and drained_scopes not in {(), ("turn",)})
        or (waited_ids and "turn" not in drained_scopes)
    ):
        diagnostics.append("turn response fence drained_scopes are inconsistent with its outcome or membership")
        return marker.monotonic_ns, False, segment_end_sequence, None
    hook_nodes = {
        node.operation_id: node
        for node in intermediate.nodes_by_turn.get(turn.turn_id, ())
        if node.family == "hook.operation"
    }
    for operation_id in waited_ids:
        node = hook_nodes.get(operation_id)
        starts = [] if node is None else [event for event in node.starts if _active(event.sequence, inactive_ranges)]
        terminals = (
            [] if node is None else [event for event in node.finishes if _active(event.sequence, inactive_ranges)]
        )
        if (
            len(starts) != 1
            or len(terminals) != 1
            or _payload_str(starts[0].payload, "execution_mode") != "async"
            or _hook_drain_scope(starts[0]) != "turn"
            or starts[0].scope != start.scope
            or terminals[0].scope != start.scope
            or starts[0].sequence <= start.sequence
            or terminals[0].sequence <= starts[0].sequence
            or terminals[0].sequence >= marker.sequence
            or terminals[0].monotonic_ns < starts[0].monotonic_ns
            or not _endpoint_in_coverage_runtime(intermediate, starts[0], inactive_ranges)
            or not _endpoint_in_coverage_runtime(intermediate, terminals[0], inactive_ranges)
        ):
            diagnostics.append("a waited hook is not a uniquely closed async turn-scope hook before the response fence")
            return marker.monotonic_ns, False, segment_end_sequence, None
    return marker.monotonic_ns, True, segment_end_sequence, frozenset(waited_ids)


def _reassemble_waited_hook_ids(
    intermediate: _Intermediate,
    marker: _Endpoint,
    inactive_ranges: tuple[tuple[int, int], ...],
) -> list[str] | None:
    declarations = [
        declaration
        for declaration in marker.segmented_fields
        if declaration.field_pointer == "/payload/waited_hook_operation_ids"
    ]
    if len(declarations) != 1:
        return None
    declaration = declarations[0]
    segments = [
        segment
        for segment in intermediate.segments.get(marker.event_id, [])
        if _active(segment.sequence, inactive_ranges) and segment.field_pointer == declaration.field_pointer
    ]
    if len(segments) != declaration.segment_count:
        return None
    indexed_segments: list[tuple[int, _Segment]] = []
    for segment in segments:
        segment_index = segment.segment_index
        if segment_index is None:
            return None
        indexed_segments.append((segment_index, segment))
    indices = [segment_index for segment_index, _ in indexed_segments]
    if sorted(indices) != list(range(declaration.segment_count)):
        return None
    values: list[str] = []
    for _, segment in sorted(indexed_segments):
        if (
            segment.segment_count != declaration.segment_count
            or segment.segment_group_id != declaration.segment_group_id
            or segment.encoding != "array_slice"
            or segment.entry_oversized
            or segment.runtime_id != marker.runtime_id
            or segment.branch_id != marker.branch_id
            or segment.coverage_id != marker.coverage_id
            or segment.turn_id != marker.turn_id
            or segment.sequence <= marker.sequence
        ):
            return None
        entries = segment.entries
        if entries is None or not all(isinstance(item, str) for item in entries):
            return None
        values.extend(item for item in entries if isinstance(item, str))
    return values


def _turn_coverage_exact(
    intermediate: _Intermediate,
    start: _Endpoint,
    end_sequence: int | None,
    inactive_ranges: tuple[tuple[int, int], ...],
    diagnostics: list[str],
    *,
    rollback_projection_unresolved: bool,
) -> bool:
    if end_sequence is None:
        return False
    exact = True
    coverage_starts = [
        event
        for event in intermediate.coverage_starts.get(start.coverage_id, [])
        if event.sequence <= start.sequence
        and event.runtime_id == start.runtime_id
        and _infrastructure_branch_reaches(intermediate, event, start, inactive_ranges)
    ]
    if len(coverage_starts) != 1:
        diagnostics.append("turn is not covered by trajectory.coverage.started")
        exact = False
    coverage_ends = [
        event
        for event in intermediate.coverage_ends.get(start.coverage_id, ())
        if event.runtime_id == start.runtime_id
        and _infrastructure_branch_reaches(intermediate, start, event, inactive_ranges)
    ]
    if len(coverage_ends) > 1:
        diagnostics.append("trajectory coverage has duplicate terminal markers")
        exact = False
    elif coverage_ends:
        coverage_end = coverage_ends[0]
        last_sequence = _payload_int(coverage_end.payload, "last_sequence")
        if last_sequence != coverage_end.sequence - 1:
            diagnostics.append("trajectory coverage terminal has an invalid last_sequence")
            exact = False
        elif start.sequence > last_sequence or end_sequence > last_sequence:
            diagnostics.append("turn falls outside its trajectory coverage window")
            exact = False
    runtime_starts = [
        event
        for event in intermediate.runtime_starts.get(start.runtime_id, ())
        if _infrastructure_branch_reaches(intermediate, event, start, inactive_ranges)
    ]
    if len(runtime_starts) > 1:
        diagnostics.append("trajectory runtime has duplicate start endpoints")
        exact = False
    elif runtime_starts and runtime_starts[0].sequence >= start.sequence:
        diagnostics.append("turn begins before its trajectory runtime endpoint")
        exact = False
    runtime_recoveries = [
        event
        for event in intermediate.runtime_recoveries.get(start.runtime_id, ())
        if _infrastructure_branch_reaches(intermediate, event, start, inactive_ranges)
    ]
    if len(runtime_recoveries) > 1:
        diagnostics.append("trajectory runtime has duplicate recovery endpoints")
        exact = False
    elif runtime_recoveries and runtime_recoveries[0].sequence >= start.sequence:
        diagnostics.append("turn begins before its trajectory runtime recovery endpoint")
        exact = False
    runtime_finishes = [
        event
        for event in intermediate.runtime_finishes.get(start.runtime_id, ())
        if _infrastructure_branch_reaches(intermediate, start, event, inactive_ranges)
    ]
    if len(runtime_finishes) > 1:
        diagnostics.append("trajectory runtime has duplicate terminal markers")
        exact = False
    elif runtime_finishes:
        runtime_finish = runtime_finishes[0]
        if start.sequence >= runtime_finish.sequence or end_sequence >= runtime_finish.sequence:
            diagnostics.append("turn occurs after trajectory.runtime.finished")
            exact = False
        matching_coverage_end = coverage_ends[0] if len(coverage_ends) == 1 else None
        if matching_coverage_end is None or matching_coverage_end.sequence >= runtime_finish.sequence:
            diagnostics.append("trajectory runtime closure lacks an ordered coverage terminal")
            exact = False
    for first, last in intermediate.explicit_gaps:
        if first <= end_sequence and last >= start.sequence:
            diagnostics.append("trajectory gap intersects the turn")
            exact = False
            break
    if any(start.sequence <= sequence <= end_sequence for sequence in intermediate.unsupported_sequences):
        diagnostics.append("unsupported trajectory event intersects the turn")
        exact = False
    if any(start.sequence <= sequence + 1 <= end_sequence for sequence in intermediate.corrupt_after_sequences):
        diagnostics.append("corrupt trajectory line intersects the turn")
        exact = False
    if any(violation.first_sequence <= end_sequence for violation in intermediate.prefix_violations):
        diagnostics.append("trajectory accounted-prefix invariant failed")
        exact = False
    if rollback_projection_unresolved:
        diagnostics.append("rollback live-history range is unresolved")
        exact = False
    return exact


def _resolved_turn_nodes(
    intermediate: _Intermediate,
    turn_start: _Endpoint,
    inactive_ranges: tuple[tuple[int, int], ...],
    diagnostics: list[str],
) -> tuple[list[_ResolvedNode], bool]:
    resolved: list[_ResolvedNode] = []
    exact = True
    turn_id = turn_start.turn_id or ""
    for node in intermediate.nodes_by_turn.get(turn_id, ()):
        starts = [
            event for event in node.starts if event.turn_id == turn_id and _active(event.sequence, inactive_ranges)
        ]
        finishes = [
            event for event in node.finishes if event.turn_id == turn_id and _active(event.sequence, inactive_ranges)
        ]
        if not starts and not finishes:
            continue
        endpoints = (*starts, *finishes)
        if endpoints and all(event.side_call for event in endpoints):
            # Side-call lifecycles stay out of the turn's arithmetic; the
            # closed ones are kept only so their exchange terminals can vouch
            # for the usage they contributed, and are dropped right after.
            side_call_node = _resolved_side_call_node(intermediate, node, turn_start, starts, finishes, inactive_ranges)
            if side_call_node is not None:
                resolved.append(side_call_node)
            continue
        if any(event.side_call for event in endpoints):
            diagnostics.append(f"{node.family} lifecycle mixes main and non-main actors")
            exact = False
            continue
        cut = _lifecycle_cut(node, inactive_ranges)
        if cut is not _LifecycleCut.NONE:
            diagnostics.append(_lifecycle_cut_reason(node.family, cut))
            exact = False
            continue
        if node.family == "retry" and len(starts) == 1 and not finishes:
            # A lone scheduled marker is a cancelled backoff, not an open span.
            continue
        if len(starts) != 1 or len(finishes) != 1:
            diagnostics.append(f"{node.family} lifecycle is not uniquely closed")
            exact = False
            continue
        start, finish = starts[0], finishes[0]
        if (
            start.runtime_id != turn_start.runtime_id
            or finish.runtime_id != turn_start.runtime_id
            or start.branch_id != turn_start.branch_id
            or finish.branch_id != turn_start.branch_id
            or start.coverage_id != turn_start.coverage_id
            or finish.coverage_id != turn_start.coverage_id
            or start.sequence <= turn_start.sequence
            or finish.sequence <= turn_start.sequence
        ):
            diagnostics.append(f"{node.family} lifecycle does not belong to the owning turn runtime and branch")
            exact = False
            continue
        if (
            start.scope != finish.scope
            or (node.family != "compaction.phase" and finish.sequence <= start.sequence)
            or finish.monotonic_ns < start.monotonic_ns
        ):
            diagnostics.append(f"{node.family} interval has invalid monotonic endpoints")
            exact = False
            continue
        if not _endpoint_in_coverage_runtime(intermediate, start, inactive_ranges) or not _endpoint_in_coverage_runtime(
            intermediate, finish, inactive_ranges
        ):
            diagnostics.append(f"{node.family} lifecycle falls outside coverage or after runtime closure")
            exact = False
            continue
        if node.family in _WEIGHTED_FAMILIES and not finish.monotonic_measurement:
            diagnostics.append(f"{node.family} duration lacks monotonic provenance")
            exact = False
        resolved.append(
            _ResolvedNode(
                node_id=f"{node.family}:{node.operation_id}",
                family=node.family,
                operation_id=node.operation_id,
                start=start,
                finish=finish,
            )
        )
    return resolved, exact


def _resolved_side_call_node(
    intermediate: _Intermediate,
    node: _Node,
    turn_start: _Endpoint,
    starts: list[_Endpoint],
    finishes: list[_Endpoint],
    inactive_ranges: tuple[tuple[int, int], ...],
) -> _ResolvedNode | None:
    """Validate a side-call lifecycle in its own actor domain without entering main-turn metrics."""
    if len(starts) != 1 or len(finishes) != 1:
        return None
    start, finish = starts[0], finishes[0]
    if (
        start.scope != finish.scope
        or start.runtime_id != turn_start.runtime_id
        or start.branch_id != turn_start.branch_id
        or start.coverage_id != turn_start.coverage_id
        or start.sequence <= turn_start.sequence
        or finish.sequence <= start.sequence
        or finish.monotonic_ns < start.monotonic_ns
        or not _endpoint_in_coverage_runtime(intermediate, start, inactive_ranges)
        or not _endpoint_in_coverage_runtime(intermediate, finish, inactive_ranges)
    ):
        return None
    return _ResolvedNode(
        node_id=f"{node.family}:{node.operation_id}",
        family=node.family,
        operation_id=node.operation_id,
        start=start,
        finish=finish,
    )


def _tool_action_projection_exact(
    intermediate: _Intermediate,
    turn_start: _Endpoint,
    terminal_sequence: int | None,
    inactive_ranges: tuple[tuple[int, int], ...],
    diagnostics: list[str],
) -> bool:
    exact = True
    turn_id = turn_start.turn_id or ""
    for node in intermediate.nodes_by_turn.get(turn_id, ()):
        if node.family != "tool.operation":
            continue
        starts = [event for event in node.starts if _active(event.sequence, inactive_ranges)]
        finishes = [event for event in node.finishes if _active(event.sequence, inactive_ranges)]
        if not starts and not finishes:
            continue
        endpoints = (*starts, *finishes)
        if endpoints and all(event.side_call for event in endpoints):
            continue
        if any(event.side_call for event in endpoints):
            diagnostics.append("tool action projection mixes main and non-main actors")
            exact = False
            continue
        cut = _lifecycle_cut(node, inactive_ranges)
        if cut is not _LifecycleCut.NONE:
            diagnostics.append(_lifecycle_cut_reason("tool action", cut))
            exact = False
            continue
        if len(starts) != 1:
            diagnostics.append("tool action projection lacks a unique start event")
            exact = False
            continue
        start = starts[0]
        if (
            start.runtime_id != turn_start.runtime_id
            or start.branch_id != turn_start.branch_id
            or start.coverage_id != turn_start.coverage_id
            or start.turn_id != turn_start.turn_id
            or start.sequence <= turn_start.sequence
        ):
            diagnostics.append("tool action start does not belong to the owning turn scope")
            exact = False
            continue
        # The scope check above bounds the start from below; tool activity
        # past the turn terminal is just as unprovable as activity before the
        # turn opened. The bound is ``turn.finished`` itself, not the tail's
        # response fence — the fence settles after the turn closes and no
        # tool may run in between. Terminals count too: an outcome read from
        # a finish beyond the turn is no better founded than a stray start.
        # An open turn has no terminal yet, and is already inexact through
        # its lifecycle.
        if terminal_sequence is not None and any(event.sequence > terminal_sequence for event in (start, *finishes)):
            diagnostics.append("tool action endpoint lies beyond the turn terminal")
            exact = False
    return exact


def _timeline_operations(
    intermediate: _Intermediate,
    *,
    turn_id: str,
    inactive_ranges: tuple[tuple[int, int], ...],
    resolved_nodes: list[_ResolvedNode],
    cancel_event: Event | None,
) -> tuple[TimelineOperation, ...]:
    """Project lifecycle nodes for display without deriving metric values from them."""
    rows = [_resolved_timeline_operation(node) for node in resolved_nodes]
    resolved_ids = {node.node_id for node in resolved_nodes}
    for node in intermediate.nodes_by_turn.get(turn_id, ()):
        _check_cancelled(cancel_event)
        node_id = f"{node.family}:{node.operation_id}"
        if node_id in resolved_ids:
            continue
        starts = [
            event for event in node.starts if event.turn_id == turn_id and _active(event.sequence, inactive_ranges)
        ]
        finishes = [
            event for event in node.finishes if event.turn_id == turn_id and _active(event.sequence, inactive_ranges)
        ]
        endpoints = sorted((*starts, *finishes), key=lambda endpoint: endpoint.sequence)
        if not endpoints:
            continue
        if all(endpoint.side_call for endpoint in endpoints):
            continue
        cut = _lifecycle_cut(node, inactive_ranges)
        identity_endpoint = starts[0] if starts else endpoints[0]
        identity_source = (
            node.starts[0] if cut is _LifecycleCut.FINISH_SURVIVES and len(node.starts) == 1 else identity_endpoint
        )
        if cut is not _LifecycleCut.NONE:
            reason = _lifecycle_cut_reason(node.family, cut)
            diagnostic_code = _lifecycle_cut_diagnostic_code(cut)
        else:
            reason, diagnostic_code = _unresolved_operation_reason(node.family, starts, finishes)
        rows.append(
            _TimelineProjection(
                operation_id=node.operation_id,
                parent_operation_id=_timeline_parent_operation(node.family, identity_endpoint),
                start_sequence=endpoints[0].sequence,
                family=node.family,
                start_ns=None,
                end_ns=None,
                precision=Precision.UNRESOLVED,
                reason=reason,
                diagnostic_code=diagnostic_code,
                identity=_timeline_identity(node.family, identity_source),
                hook_id=_timeline_hook_id(node.family, identity_source),
            )
        )
    return _materialize_timeline(rows)


def _hook_drain_scope(start: _Endpoint) -> str | None:
    """Read the renamed hook lifetime scope while accepting retained logs."""
    return _payload_str(start.payload, "drain_scope") or _payload_str(start.payload, "scope")


def _unresolved_operation_reason(
    family: str,
    starts: list[_Endpoint],
    finishes: list[_Endpoint],
) -> tuple[str, TimelineDiagnosticCode]:
    if not starts:
        return f"{family} lifecycle has no start endpoint", TimelineDiagnosticCode.MISSING_START
    if not finishes:
        return f"{family} lifecycle has no terminal endpoint", TimelineDiagnosticCode.MISSING_TERMINAL
    if len(starts) != 1 or len(finishes) != 1:
        return f"{family} lifecycle is not uniquely closed", TimelineDiagnosticCode.NONUNIQUE_LIFECYCLE
    if finishes[0].sequence <= starts[0].sequence or finishes[0].monotonic_ns < starts[0].monotonic_ns:
        return f"{family} interval has invalid monotonic endpoints", TimelineDiagnosticCode.INVALID_ENDPOINTS
    return f"{family} lifecycle falls outside the owning turn coverage", TimelineDiagnosticCode.OUTSIDE_TURN_COVERAGE


def _lifecycle_cut_reason(family: str, cut: _LifecycleCut) -> str:
    """Describe which endpoint of a raw lifecycle remains active."""
    surviving = "start" if cut is _LifecycleCut.START_SURVIVES else "terminal"
    return f"{family} lifecycle crosses rollback projection; only its {surviving} endpoint remains active"


def _lifecycle_cut_diagnostic_code(cut: _LifecycleCut) -> TimelineDiagnosticCode:
    return (
        TimelineDiagnosticCode.ROLLBACK_START_SURVIVES
        if cut is _LifecycleCut.START_SURVIVES
        else TimelineDiagnosticCode.ROLLBACK_TERMINAL_SURVIVES
    )


def _continuation_poll_nodes(nodes: list[_ResolvedNode]) -> list[_ResolvedNode]:
    """Derive the uninstrumented fixed continuation-poll pauses."""
    exchanges = [node for node in nodes if node.family == "model.exchange"]
    retries = [node for node in nodes if node.family == "retry"]
    by_cycle: dict[str, list[_ResolvedNode]] = defaultdict(list)
    for exchange in exchanges:
        parent = exchange.start.parent_operation_id
        if parent is not None:
            by_cycle[parent].append(exchange)
    derived: list[_ResolvedNode] = []
    for cycle_operation_id, cycle_exchanges in by_cycle.items():
        ordered = sorted(cycle_exchanges, key=lambda node: (node.start.monotonic_ns, node.start.sequence))
        for previous, current in pairwise(ordered):
            if (
                _payload_str(previous.finish.payload, "outcome") != "success"
                or _payload_str(current.start.payload, "continuation_mode") != "poll"
                or previous.finish.monotonic_ns >= current.start.monotonic_ns
            ):
                continue
            if any(
                retry.start.monotonic_ns < current.start.monotonic_ns
                and retry.finish.monotonic_ns > previous.finish.monotonic_ns
                for retry in retries
            ):
                continue
            operation_id = f"continuation_poll:{previous.operation_id}:{current.operation_id}"
            start = _synthetic_endpoint(
                previous.finish,
                monotonic_ns=previous.finish.monotonic_ns,
                parent_operation_id=cycle_operation_id,
                payload={
                    "category": "continuation_poll",
                    "previous_exchange_operation_id": previous.operation_id,
                    "next_exchange_operation_id": current.operation_id,
                },
            )
            finish = _synthetic_endpoint(
                current.start,
                monotonic_ns=current.start.monotonic_ns,
                parent_operation_id=cycle_operation_id,
                payload={
                    "category": "continuation_poll",
                    "outcome": "completed",
                    "previous_exchange_operation_id": previous.operation_id,
                    "next_exchange_operation_id": current.operation_id,
                },
            )
            derived.append(
                _ResolvedNode(
                    node_id=f"continuation.poll:{operation_id}",
                    family="continuation.poll",
                    operation_id=operation_id,
                    start=start,
                    finish=finish,
                )
            )
    return derived


def _suspension_nodes(
    turn: _Turn,
    nodes: list[_ResolvedNode],
    turn_start: _Endpoint,
    finish: _Endpoint | None,
    inactive_ranges: tuple[tuple[int, int], ...],
    diagnostics: list[str],
) -> tuple[list[_ResolvedNode], dict[str, list[tuple[int, int]]], bool]:
    """Pair turn suspension markers and identify the sole deductible boundary."""
    suspended = sorted(
        (event for event in turn.suspended if _active(event.sequence, inactive_ranges)),
        key=lambda event: event.sequence,
    )
    resumed = sorted(
        (event for event in turn.resumed if _active(event.sequence, inactive_ranges)),
        key=lambda event: event.sequence,
    )
    resume_index = 0
    exact = True
    derived: list[_ResolvedNode] = []
    deductions: dict[str, list[tuple[int, int]]] = defaultdict(list)
    boundaries = [node for node in nodes if node.family == "sub_agent"]
    for index, start in enumerate(suspended):
        while resume_index < len(resumed) and resumed[resume_index].sequence <= start.sequence:
            diagnostics.append("turn.resumed has no preceding suspension")
            exact = False
            resume_index += 1
        if resume_index < len(resumed):
            terminal = resumed[resume_index]
            resume_index += 1
        elif finish is not None:
            terminal = finish
            diagnostics.append("turn suspension reaches the turn terminal without resume")
            exact = False
        else:
            diagnostics.append("turn suspension has no observable terminal")
            exact = False
            continue
        if (
            start.runtime_id != turn_start.runtime_id
            or terminal.runtime_id != turn_start.runtime_id
            or start.branch_id != turn_start.branch_id
            or terminal.branch_id != turn_start.branch_id
            or terminal.sequence <= start.sequence
            or terminal.monotonic_ns < start.monotonic_ns
        ):
            diagnostics.append("turn suspension has invalid monotonic endpoints")
            exact = False
            continue
        operation_id = f"turn_suspension:{turn.turn_id}:{index}"
        derived.append(
            _ResolvedNode(
                node_id=f"turn.suspension:{operation_id}",
                family="turn.suspension",
                operation_id=operation_id,
                start=_synthetic_endpoint(
                    start,
                    monotonic_ns=start.monotonic_ns,
                    parent_operation_id=None,
                    payload={"category": "sub_agent_suspension"},
                ),
                finish=_synthetic_endpoint(
                    terminal,
                    monotonic_ns=terminal.monotonic_ns,
                    parent_operation_id=None,
                    payload={"category": "sub_agent_suspension", "outcome": "completed"},
                ),
            )
        )
        candidates = [
            boundary
            for boundary in boundaries
            if boundary.start.monotonic_ns <= start.monotonic_ns
            and boundary.finish.monotonic_ns >= terminal.monotonic_ns
        ]
        if len(candidates) == 1:
            deductions[candidates[0].node_id].append((start.monotonic_ns, terminal.monotonic_ns))
        elif len(candidates) > 1:
            diagnostics.append("turn suspension overlaps multiple open sub-agent boundaries")
            exact = False
    if resume_index < len(resumed):
        diagnostics.append("turn.resumed has no preceding suspension")
        exact = False
    return derived, deductions, exact


def _synthetic_endpoint(
    source: _Endpoint,
    *,
    monotonic_ns: int,
    parent_operation_id: str | None,
    payload: dict[str, object],
) -> _Endpoint:
    return _Endpoint(
        event_id=source.event_id,
        sequence=source.sequence,
        scope=source.scope,
        monotonic_ns=monotonic_ns,
        parent_operation_id=parent_operation_id,
        side_call=source.side_call,
        payload=tuple(item for pair in payload.items() for item in pair),
        links=(),
        segmented_fields=(),
        monotonic_measurement=True,
    )


def _project_turn(
    nodes: list[_ResolvedNode],
    turn_bounds: tuple[int, int],
    *,
    start: _Endpoint,
    finish: _Endpoint | None,
    tail_end: int | None,
    waited_hook_ids: frozenset[str] | None,
    suspension_deductions: dict[str, list[tuple[int, int]]],
    revisions: _RevisionResolution,
    compaction_consumption: _CompactionConsumptionResolution,
    mapped_carrier: Callable[[str], str | None],
    cancel_event: Event | None,
    diagnostics: list[str],
) -> tuple[tuple[TimeSlice, ...], _DependencyProof, bool, bool, bool]:
    by_operation: dict[str, list[_ResolvedNode]] = defaultdict(list)
    for node in nodes:
        by_operation[node.operation_id].append(node)
    preparation_for_tool = {
        target: node
        for node in nodes
        if node.family == "preparation"
        and _payload_str(node.start.payload, "scope") == "tool_preamble"
        and (target := _payload_str(node.start.payload, "target_operation_id")) is not None
    }
    parents: dict[str, str] = {}
    graph_edges: dict[str, set[str]] = defaultdict(set)
    unsafe_work_nodes: set[str] = set()
    timeline_exact = True
    dag_exact = True
    for node in nodes:
        _check_cancelled(cancel_event)
        parent = _displacing_parent(node, by_operation, preparation_for_tool, diagnostics)
        if parent is False:
            timeline_exact = False
            dag_exact = False
            if node.family == "hook.operation":
                unsafe_work_nodes.add(node.node_id)
        elif isinstance(parent, str):
            parents[node.node_id] = parent
            graph_edges[parent].add(node.node_id)
        for relation, target_operation_id in node.start.links:
            if relation != LinkRelation.CAUSED_BY:
                continue
            # A retry lifecycle deliberately shares the next operation's id.
            # caused_by names the work operation, never that retry marker.
            targets = [
                target
                for target in by_operation.get(target_operation_id, [])
                if target.family != "retry" and target.node_id != node.node_id
            ]
            if len(targets) == 1:
                graph_edges[targets[0].node_id].add(node.node_id)
            else:
                diagnostics.append("caused_by target cannot be resolved uniquely")
                dag_exact = False
    response_dependency_exact = _validate_async_hook_fences(nodes, waited_hook_ids, diagnostics)
    children: dict[str, set[str]] = defaultdict(set)
    for child, parent in parents.items():
        children[parent].add(child)
    node_by_id = {node.node_id: node for node in nodes}
    if _has_cycle(children):
        diagnostics.append("displacing-edge graph contains a cycle")
        return (), _DependencyProof(graph_edges, None, None, parents), False, False, response_dependency_exact
    slices: list[TimeSlice] = []
    for node in nodes:
        _check_cancelled(cancel_event)
        if _skip_node(node):
            continue
        descendants = _descendants(node.node_id, children)
        removed = [node_by_id[descendant].interval for descendant in descendants]
        removed.extend(suspension_deductions.get(node.node_id, ()))
        residuals = subtract_intervals(node.interval, removed)
        attributes = _slice_attributes(node)
        if attributes is None:
            continue
        bucket, counts_as_work, compute_weight, response_weight, owner = attributes
        if node.family == "hook.operation":
            mode = _payload_str(node.start.payload, "execution_mode")
            hook_event = _payload_str(node.start.payload, "hook_event")
            if hook_event == "user_interrupt" or mode == "fire_and_forget":
                response_weight = False
            elif mode == "async":
                response_weight = waited_hook_ids is not None and node.operation_id in waited_hook_ids
        if node.node_id in unsafe_work_nodes:
            counts_as_work = False
            compute_weight = False
            response_weight = False
        for index, residual in enumerate(residuals):
            clipped = clip_interval(residual, turn_bounds)
            if clipped is None:
                continue
            slices.append(
                TimeSlice(
                    family=node.family,
                    slice_index=index,
                    turn_id=start.turn_id or "",
                    runtime_id=node.start.runtime_id,
                    operation_id=node.operation_id,
                    owner=owner,
                    start_ns=clipped[0],
                    end_ns=clipped[1],
                    wall_bucket=bucket,
                    counts_as_work=counts_as_work,
                    compute_weight=compute_weight,
                    response_weight=response_weight,
                    outcome=_payload_str(node.finish.payload, "outcome"),
                    tool_name=_payload_str(node.start.payload, "tool_name"),
                    tool_kind=_payload_str(node.start.payload, "tool_kind"),
                )
            )
    typed_exact, response_source_id = _connect_typed_dependencies(
        nodes,
        graph_edges,
        main_actor_id=start.actor_id,
        revisions=revisions,
        compaction_consumption=compaction_consumption,
        mapped_carrier=mapped_carrier,
        cancel_event=cancel_event,
        diagnostics=diagnostics,
    )
    dag_exact = dag_exact and typed_exact
    roots = [
        node.node_id
        for node in nodes
        if node.family == "preparation" and _payload_str(node.start.payload, "scope") == "turn_preamble"
    ]
    turn_root_id = roots[0] if len(roots) == 1 else None
    response_terminal_id = (
        f"turn.response:{start.turn_id}"
        if finish is not None and tail_end is not None and response_source_id is not None
        else None
    )
    if response_terminal_id is not None and response_source_id is not None:
        graph_edges[response_source_id].add(response_terminal_id)
    fork_edges: set[tuple[str, str]] = set()
    hook_dag_exact, hook_response_exact = _connect_hook_dependencies(
        nodes,
        graph_edges,
        waited_hook_ids=waited_hook_ids,
        turn_root_id=turn_root_id,
        response_source_id=response_source_id,
        response_terminal_id=response_terminal_id,
        fork_edges=fork_edges,
        diagnostics=diagnostics,
    )
    dag_exact = dag_exact and hook_dag_exact
    response_dependency_exact = response_dependency_exact and hook_response_exact
    if response_source_id is not None and response_terminal_id is not None:
        response_descendants = _descendants(response_source_id, children)
        for node_id in response_descendants:
            if not children.get(node_id, set()) & response_descendants:
                graph_edges[node_id].add(response_terminal_id)
    return (
        tuple(slices),
        _DependencyProof(graph_edges, turn_root_id, response_terminal_id, parents, frozenset(fork_edges)),
        timeline_exact,
        dag_exact,
        response_dependency_exact,
    )


def _turn_flow(
    turn_id: str,
    dependency: _DependencyProof,
    operations: tuple[TimelineOperation, ...],
    *,
    acyclic: bool,
) -> TurnFlow:
    """Index-encode the typed per-turn dependency graph against the operations tuple."""
    index_by_node: dict[str, int] = {}
    for index, operation in enumerate(operations):
        index_by_node.setdefault(f"{operation.family}:{operation.operation_id}", index)
    terminal_id = dependency.response_terminal_id
    parent_pairs = array("I")
    causal_pairs = array("I")
    for source, targets in sorted(dependency.edges.items()):
        source_index = index_by_node.get(source)
        if source_index is None:
            continue
        for target in sorted(targets):
            if target == terminal_id:
                target_index = FLOW_TERMINAL_INDEX
            else:
                found = index_by_node.get(target)
                if found is None:
                    continue
                target_index = found
            pairs = parent_pairs if dependency.parents.get(target) == source else causal_pairs
            pairs.append(source_index)
            pairs.append(target_index)
    root_id = dependency.turn_root_id
    return TurnFlow(
        turn_id=turn_id,
        root_index=index_by_node.get(root_id) if root_id is not None else None,
        has_terminal=terminal_id is not None,
        parent_pairs=parent_pairs.tobytes(),
        causal_pairs=causal_pairs.tobytes(),
        acyclic=acyclic,
    )


def _usage_samples_by_turn(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
) -> dict[str, tuple[TokenUsageSample, ...]]:
    """Project timestamped per-exchange usage counters on demand for export."""
    finish_ns = {
        endpoint.sequence: endpoint.monotonic_ns
        for node in intermediate.nodes.get("model.exchange", {}).values()
        for endpoint in node.finishes
    }
    samples: dict[str, tuple[TokenUsageSample, ...]] = {}
    for turn_id, items in intermediate.usage_by_turn.items():
        rows = tuple(
            TokenUsageSample(
                sequence=item.sequence,
                end_ns=finish_ns.get(item.sequence),
                input_tokens=item.input_total,
                output_tokens=item.output_total,
                reasoning_tokens=item.extras.reasoning if item.extras is not None else None,
                cache_read_tokens=item.extras.cache_read if item.extras is not None else None,
                cache_creation_tokens=item.extras.cache_creation if item.extras is not None else None,
            )
            for item in items
            if _active(item.sequence, inactive_ranges)
            and (item.input_total is not None or item.output_total is not None or item.extras is not None)
        )
        if rows:
            samples[turn_id] = rows
    return samples


def _context_samples_by_turn(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
) -> dict[str, tuple[ContextSample, ...]]:
    """Project timestamped context-size counters on demand for export.

    The declared item_count is read straight off each uniquely defined revision
    endpoint, so no membership replay is needed; declaration inconsistencies
    already surface through the diagnostics pipeline.
    """
    by_turn: dict[str, list[ContextSample]] = defaultdict(list)
    for candidates in intermediate.context_revisions.values():
        active = [endpoint for endpoint in candidates if _active(endpoint.sequence, inactive_ranges)]
        if len(active) != 1:
            continue
        endpoint = active[0]
        item_count = _payload_int(endpoint.payload, "item_count")
        if endpoint.turn_id is None or item_count is None or item_count < 0 or endpoint.side_call:
            continue
        by_turn[endpoint.turn_id].append(
            ContextSample(
                sequence=endpoint.sequence,
                ns=endpoint.monotonic_ns,
                item_count=item_count,
            )
        )
    return {
        turn_id: tuple(sorted(rows, key=lambda sample: sample.sequence)) for turn_id, rows in sorted(by_turn.items())
    }


def _displacing_parent(
    node: _ResolvedNode,
    by_operation: dict[str, list[_ResolvedNode]],
    preparation_for_tool: dict[str, _ResolvedNode],
    diagnostics: list[str],
) -> str | bool | None:
    if node.family == "preparation":
        scope = _payload_str(node.start.payload, "scope")
        if scope == "turn_preamble":
            return None
        if scope == "pre_turn":
            return None
    if node.family == "hook.operation":
        hook_event = _payload_str(node.start.payload, "hook_event")
        mode = _payload_str(node.start.payload, "execution_mode")
        ownership = classify_hook_ownership(hook_event, mode)
        if ownership is HookOwnership.CONCURRENT:
            return None
        target = _payload_str(node.start.payload, "target_operation_id")
        if ownership is HookOwnership.TOOL_PREAMBLE and target is not None:
            preamble = preparation_for_tool.get(target)
            if preamble is not None:
                return preamble.node_id
        elif ownership in {
            HookOwnership.TOOL_TAIL,
            HookOwnership.APPROVAL,
            HookOwnership.TURN_PREAMBLE,
            HookOwnership.SUB_AGENT,
            HookOwnership.COMPACTION,
        }:
            families = _hook_target_families(hook_event)
            resolved = [candidate for candidate in by_operation.get(target or "", []) if candidate.family in families]
            if len(resolved) == 1:
                return resolved[0].node_id
        elif ownership in {HookOwnership.TURN_TAIL, HookOwnership.SESSION_ROOT, HookOwnership.WORKFLOW_ROOT}:
            return None
        diagnostics.append(f"blocking hook {hook_event or 'unknown'} has no placeable displacing edge")
        return False
    if node.family == "tool.operation":
        parent = _payload_str(node.start.payload, "parent_model_operation_id")
        if parent is None:
            diagnostics.append("tool operation lacks parent_model_operation_id")
            return False
        candidates = [
            candidate for candidate in by_operation.get(parent, []) if candidate.family in _MODEL_OPERATION_FAMILIES
        ]
        if len(candidates) == 1:
            return candidates[0].node_id
        diagnostics.append("tool parent model operation cannot be resolved uniquely")
        return False
    if node.family == "turn.suspension":
        return None
    target = None
    families: frozenset[str] | None = None
    if node.family == "approval":
        generic_target = _payload_str(node.start.payload, "target_operation_id")
        target = generic_target or _payload_str(node.start.payload, "target_tool_operation_id")
        if generic_target is None:
            families = frozenset({"tool.operation"})
    elif node.family in {"preparation", "compaction"}:
        target = node.start.parent_operation_id
        families = _MODEL_OPERATION_FAMILIES
    elif node.family == "compaction.phase":
        target = node.start.parent_operation_id
        families = frozenset({"compaction"})
    elif node.family == "model.cycle":
        target = node.start.parent_operation_id
        families = frozenset({"model.run", "sub_agent"})
    elif node.family == "model.exchange":
        target = node.start.parent_operation_id
        families = frozenset({"model.run", "model.cycle"})
    elif node.family == "sub_agent":
        target = node.start.parent_operation_id
        families = frozenset({"tool.operation", *_MODEL_OPERATION_FAMILIES})
    elif node.family == "wait":
        target = node.start.parent_operation_id
        category = _payload_str(node.start.payload, "category")
        families = _wait_parent_families(category)
    elif node.family == "retry":
        target = node.start.parent_operation_id
        families = _retry_parent_families(_payload_str(node.start.payload, "retry_mode"))
    else:
        target = node.start.parent_operation_id
    if target is None:
        return None
    candidates = [
        candidate
        for candidate in by_operation.get(target, [])
        if candidate.node_id != node.node_id
        and candidate.family != "retry"
        and (families is None or candidate.family in families)
    ]
    if len(candidates) == 1:
        return candidates[0].node_id
    if candidates:
        diagnostics.append(f"{node.family} parent operation is ambiguous")
        return False
    diagnostics.append(f"{node.family} parent operation is missing")
    return False


def _wait_parent_families(category: str | None) -> frozenset[str] | None:
    if category == WaitCategory.USER_INPUT:
        return frozenset({"tool.operation", "sub_agent"})
    if category == WaitCategory.MCP_CONNECT:
        return frozenset({"tool.operation"})
    if category in {WaitCategory.INPUT_ADMISSION, WaitCategory.TOOL_ADMISSION}:
        return frozenset({"preparation"})
    if category == WaitCategory.RATE_LIMIT:
        return _MODEL_OPERATION_FAMILIES
    if category == WaitCategory.SUB_AGENT_CONCURRENCY:
        return frozenset({"sub_agent"})
    return None


def _retry_parent_families(retry_mode: str | None) -> frozenset[str] | None:
    if retry_mode in _RETRY_EXCHANGE_MODES:
        return frozenset({"model.run", "model.cycle"})
    if retry_mode == RetryMode.RUN:
        return frozenset({"sub_agent"})
    if retry_mode == RetryMode.COMPACTION:
        return frozenset({"compaction"})
    if retry_mode == RetryMode.CONNECTION:
        return frozenset({"sub_agent"})
    return None


def _hook_target_families(hook_event: str | None) -> frozenset[str]:
    if hook_event in {HookEvent.APPROVAL_REQUESTED, HookEvent.APPROVAL_RESOLVED}:
        return frozenset({"approval"})
    if hook_event in {HookEvent.BEFORE_TOOL_CALL, HookEvent.AFTER_TOOL_CALL, HookEvent.TOOL_ERROR}:
        return frozenset({"tool.operation"})
    if hook_event == HookEvent.BEFORE_TURN:
        return frozenset({"preparation"})
    if hook_event in {HookEvent.SUB_AGENT_START, HookEvent.SUB_AGENT_END}:
        return frozenset({"sub_agent"})
    if hook_event == HookEvent.PRE_COMPACT:
        return frozenset({"compaction"})
    return frozenset()


def classify_hook_ownership(hook_event: str | None, execution_mode: str | None) -> HookOwnership:
    """Apply the ordered, exhaustive hook ownership matrix without timing guesses."""
    if execution_mode not in _HOOK_EXECUTION_MODES:
        return HookOwnership.UNSAFE
    if hook_event == HookEvent.USER_INTERRUPT or execution_mode != "blocking":
        return HookOwnership.CONCURRENT
    if hook_event == HookEvent.BEFORE_TOOL_CALL:
        return HookOwnership.TOOL_PREAMBLE
    if hook_event in {HookEvent.AFTER_TOOL_CALL, HookEvent.TOOL_ERROR}:
        return HookOwnership.TOOL_TAIL
    if hook_event in {HookEvent.APPROVAL_REQUESTED, HookEvent.APPROVAL_RESOLVED}:
        return HookOwnership.APPROVAL
    if hook_event == HookEvent.BEFORE_TURN:
        return HookOwnership.TURN_PREAMBLE
    if hook_event == HookEvent.AFTER_TURN:
        return HookOwnership.TURN_TAIL
    if hook_event == HookEvent.USER_PROMPT_SUBMIT:
        return HookOwnership.PRE_TURN
    if hook_event in {HookEvent.SUB_AGENT_START, HookEvent.SUB_AGENT_END}:
        return HookOwnership.SUB_AGENT
    if hook_event == HookEvent.PRE_COMPACT:
        return HookOwnership.COMPACTION
    if hook_event in {HookEvent.WORKFLOW_RUN_START, HookEvent.WORKFLOW_RUN_END}:
        return HookOwnership.WORKFLOW_ROOT
    if hook_event in {HookEvent.SESSION_START, HookEvent.SESSION_RESTORED, HookEvent.SESSION_END}:
        return HookOwnership.SESSION_ROOT
    return HookOwnership.UNSAFE


def _slice_attributes(
    node: _ResolvedNode,
) -> tuple[WallBucket, bool, bool, bool, str] | None:
    family = node.family
    if family in {"model.run", "model.cycle", "model.exchange"}:
        return WallBucket.MODEL, True, True, True, "model"
    if family in {"compaction", "compaction.phase"}:
        return WallBucket.MODEL, True, True, True, "compaction"
    if family == "tool.operation":
        if _payload_str(node.start.payload, "tool_kind") == "sleep":
            return WallBucket.WAIT, False, False, True, "tool sleep"
        return WallBucket.TOOLS, True, True, True, "tool"
    if family == "sub_agent":
        return WallBucket.TOOLS, True, True, True, "sub-agent"
    if family == "continuation.poll":
        return WallBucket.WAIT, False, False, True, "continuation poll"
    if family == "turn.suspension":
        return WallBucket.WAIT, False, False, True, "sub-agent suspension"
    if family in {"wait", "approval", "retry"}:
        if family == "wait" and _payload_str(node.start.payload, "category") == "input_admission":
            return None
        owner = (
            "approval" if family == "approval" or _payload_str(node.start.payload, "category") == "approval" else family
        )
        return WallBucket.WAIT, False, False, True, owner
    if family == "preparation":
        scope = _payload_str(node.start.payload, "scope")
        if scope == "pre_turn":
            return None
        return WallBucket.TOOLS, True, True, True, "preparation"
    if family == "hook.operation":
        outcome = _payload_str(node.finish.payload, "outcome")
        if outcome == "detached":
            return None
        mode = _payload_str(node.start.payload, "execution_mode")
        hook_event = _payload_str(node.start.payload, "hook_event")
        if mode == "fire_and_forget" or hook_event == "user_interrupt":
            return WallBucket.TOOLS, True, True, False, "hook"
        return WallBucket.TOOLS, True, True, True, "hook"
    return None


def _skip_node(node: _ResolvedNode) -> bool:
    return node.family == "tool.operation" and _payload_bool(node.finish.payload, "abandoned")


def _never_dispatched(node: _ResolvedNode) -> bool:
    return (
        node.family == "tool.operation"
        and _payload_str(node.finish.payload, "outcome") in _NEVER_DISPATCHED_TOOL_OUTCOMES
    )


def _required_causes_present(nodes: list[_ResolvedNode], diagnostics: list[str]) -> bool:
    preambles = [
        node
        for node in nodes
        if node.family == "preparation" and _payload_str(node.start.payload, "scope") == "turn_preamble"
    ]
    runs = sorted((node for node in nodes if node.family == "model.run"), key=lambda node: node.start.sequence)
    exact = True
    if len(preambles) != 1 or not runs:
        diagnostics.append("turn_preamble or first model.run is missing")
        exact = False
    elif (LinkRelation.CAUSED_BY, preambles[0].operation_id) not in runs[0].start.links:
        diagnostics.append("first model.run lacks caused_by link to turn_preamble")
        exact = False
    tool_preambles = {
        target: node
        for node in nodes
        if node.family == "preparation"
        and _payload_str(node.start.payload, "scope") == "tool_preamble"
        and (target := _payload_str(node.start.payload, "target_operation_id")) is not None
    }
    tools = [node for node in nodes if node.family == "tool.operation" and not _never_dispatched(node)]
    if len(tool_preambles) != len(tools):
        diagnostics.append("tool_preamble to tool pairing is not one-to-one")
        exact = False
    for tool in tools:
        preamble = tool_preambles.get(tool.operation_id)
        if preamble is None or (LinkRelation.CAUSED_BY, preamble.operation_id) not in tool.start.links:
            diagnostics.append("tool operation lacks caused_by link to its tool_preamble")
            exact = False
    return exact


def _validate_async_hook_fences(
    nodes: list[_ResolvedNode],
    waited_hook_ids: frozenset[str] | None,
    diagnostics: list[str],
) -> bool:
    async_turn_hooks = {
        node.operation_id
        for node in nodes
        if node.family == "hook.operation"
        and _payload_str(node.start.payload, "execution_mode") == "async"
        and _hook_drain_scope(node.start) == "turn"
    }
    if not async_turn_hooks:
        return True
    if waited_hook_ids is None:
        diagnostics.append("async hook response dependencies lack a complete turn fence")
        return False
    known_hooks = {node.operation_id: node for node in nodes if node.family == "hook.operation"}
    invalid_members = [
        operation_id
        for operation_id in waited_hook_ids
        if operation_id not in known_hooks
        or _payload_str(known_hooks[operation_id].start.payload, "execution_mode") != "async"
        or _hook_drain_scope(known_hooks[operation_id].start) != "turn"
    ]
    if invalid_members:
        diagnostics.append("turn fence contains a hook that is not an async turn-scope dependency")
        return False
    return True


def _connect_typed_dependencies(
    nodes: list[_ResolvedNode],
    edges: dict[str, set[str]],
    *,
    main_actor_id: str | None,
    revisions: _RevisionResolution,
    compaction_consumption: _CompactionConsumptionResolution,
    mapped_carrier: Callable[[str], str | None],
    cancel_event: Event | None,
    diagnostics: list[str],
) -> tuple[bool, str | None]:
    """Connect only producer-declared causal pointers; never infer adjacency."""
    by_operation: dict[str, list[_ResolvedNode]] = defaultdict(list)
    for node in nodes:
        by_operation[node.operation_id].append(node)
    exact = compaction_consumption.exact

    compaction_bridges: dict[str, str] = {}
    for run_id, consumption in compaction_consumption.by_run.items():
        _check_cancelled(cancel_event)
        compactions = [candidate for candidate in by_operation.get(run_id, ()) if candidate.family == "compaction"]
        if len(compactions) != 1:
            diagnostics.append("Phase-4 consumed items cannot resolve their compaction run uniquely")
            exact = False
            continue
        compaction = compactions[0]
        candidates = [
            candidate
            for candidate in by_operation.get(consumption.next_exchange_operation_id, ())
            if candidate.family == "model.exchange"
            and candidate.start.scope == compaction.start.scope
            and candidate.start.sequence > compaction.finish.sequence
            and candidate.start.monotonic_ns >= compaction.finish.monotonic_ns
        ]
        if len(candidates) != 1:
            diagnostics.append("Phase-4 compaction successor exchange cannot be resolved uniquely")
            exact = False
            continue
        edges[compaction.node_id].add(candidates[0].node_id)
        compaction_bridges[compaction.operation_id] = compaction.node_id

    for run in (node for node in nodes if node.family == "model.run"):
        _check_cancelled(cancel_event)
        previous = _payload_str(run.start.payload, "previous_run_operation_id")
        if previous is not None:
            exact = (
                _connect_unique(
                    by_operation,
                    previous,
                    run.node_id,
                    edges,
                    diagnostics,
                    "previous model run",
                    families=frozenset({"model.run"}),
                )
                and exact
            )

    for retry in (node for node in nodes if node.family == "retry"):
        retry_mode = _payload_str(retry.start.payload, "retry_mode")
        neighbor_families = _retry_neighbor_families(retry_mode)
        previous = _payload_str(retry.start.payload, "previous_operation_id")
        following = _payload_str(retry.start.payload, "next_operation_id") or _payload_str(
            retry.finish.payload, "next_operation_id"
        )
        if previous is not None:
            exact = (
                _connect_unique(
                    by_operation,
                    previous,
                    retry.node_id,
                    edges,
                    diagnostics,
                    "retry predecessor",
                    families=neighbor_families,
                )
                and exact
            )
        if following is not None:
            exact = (
                _connect_unique(
                    by_operation,
                    following,
                    None,
                    edges,
                    diagnostics,
                    "retry successor",
                    source=retry.node_id,
                    families=neighbor_families,
                )
                and exact
            )

    for poll in (node for node in nodes if node.family == "continuation.poll"):
        _check_cancelled(cancel_event)
        previous = _payload_str(poll.start.payload, "previous_exchange_operation_id")
        following = _payload_str(poll.start.payload, "next_exchange_operation_id")
        if previous is None or following is None:
            diagnostics.append("continuation poll lacks typed neighboring exchanges")
            exact = False
            continue
        exact = (
            _connect_unique(
                by_operation,
                previous,
                poll.node_id,
                edges,
                diagnostics,
                "continuation poll predecessor",
                families=frozenset({"model.exchange"}),
            )
            and exact
        )
        exact = (
            _connect_unique(
                by_operation,
                following,
                None,
                edges,
                diagnostics,
                "continuation poll successor",
                source=poll.node_id,
                families=frozenset({"model.exchange"}),
            )
            and exact
        )

    response_source_id: str | None = None
    response_source_operation_id: str | None = None
    # Sub-agent cycles are inlined in the turn; only the turn's own actor
    # answers the user, so the response source is its last final exchange.
    cycles_with_final = [
        (node, final_exchange)
        for node in nodes
        if node.family == "model.cycle"
        and node.start.actor_id == main_actor_id
        and (final_exchange := _payload_str(node.finish.payload, "final_exchange_operation_id")) is not None
    ]
    if not cycles_with_final:
        diagnostics.append("response linkage lacks final_exchange_operation_id")
    else:
        _, final_exchange_id = max(cycles_with_final, key=lambda item: item[0].finish.sequence)
        candidates = [node for node in by_operation.get(final_exchange_id, ()) if node.family == "model.exchange"]
        if len(candidates) != 1:
            diagnostics.append("final exchange operation cannot be resolved uniquely")
            exact = False
        else:
            response_source_id = candidates[0].node_id
            response_source_operation_id = candidates[0].operation_id

    exchanges_by_revision: dict[str, list[_ResolvedNode]] = defaultdict(list)
    for exchange in (node for node in nodes if node.family == "model.exchange"):
        _check_cancelled(cancel_event)
        revision_id = _payload_str(exchange.start.payload, "context_revision_id")
        if revision_id is not None:
            exchanges_by_revision[revision_id].append(exchange)

    valid_exchange_memberships: dict[str, tuple[str, ...]] = {}
    invalid_exchange_membership_diagnostics: list[str] = []
    for revision_id, exchanges in exchanges_by_revision.items():
        _check_cancelled(cancel_event)
        revision_errors = revisions.errors.get(revision_id, ())
        if revision_errors:
            invalid_exchange_membership_diagnostics.extend(revision_errors)
            continue
        membership = revisions.memberships.get(revision_id)
        revision = revisions.endpoints.get(revision_id)
        if membership is None or revision is None:
            invalid_exchange_membership_diagnostics.append(f"context revision {revision_id} cannot be resolved")
            continue
        if len(exchanges) != 1:
            invalid_exchange_membership_diagnostics.append(
                f"context revision {revision_id} is claimed by multiple exchanges"
            )
            continue
        exchange = exchanges[0]
        if (
            revision.parent_operation_id != exchange.operation_id
            or revision.runtime_id != exchange.start.runtime_id
            or revision.branch_id != exchange.start.branch_id
            or revision.coverage_id != exchange.start.coverage_id
            or revision.actor_id != exchange.start.actor_id
            or revision.sequence >= exchange.start.sequence
        ):
            invalid_exchange_membership_diagnostics.append(
                f"context revision {revision_id} does not belong to its claiming exchange"
            )
            continue
        valid_exchange_memberships[revision_id] = membership
    diagnostics.extend(invalid_exchange_membership_diagnostics)

    for tool in (node for node in nodes if node.family == "tool.operation" and not _skip_node(node)):
        _check_cancelled(cancel_event)
        parent = _payload_str(tool.start.payload, "parent_model_operation_id")
        if parent is None:
            diagnostics.append("tool call producer lacks parent_model_operation_id")
            exact = False
        else:
            exact = (
                _connect_unique(
                    by_operation,
                    parent,
                    tool.node_id,
                    edges,
                    diagnostics,
                    "tool call producer",
                    families=frozenset({"model.run", "model.cycle", "model.exchange"}),
                )
                and exact
            )
        call_item_id = _payload_str(tool.start.payload, "call_item_id")
        result_item_id = _payload_str(tool.finish.payload, "result_item_id")
        if result_item_id is None and _never_dispatched(tool):
            # A call the kernel closed without dispatching may own no result
            # item at all (a filtered call never gets one), so there is no
            # fan-in edge to demand. One that does carry a result (invalid
            # arguments, unknown tool) is consumed like any other and falls
            # through to the ordinary requirement.
            continue
        if call_item_id is None or result_item_id is None:
            diagnostics.append("tool result fan-in lacks call_item_id or result_item_id")
            exact = False
            continue
        result_carrier_item_id = _payload_str(tool.finish.payload, "result_carrier_item_id")
        fan_in_item_ids = {result_item_id}
        if result_carrier_item_id is not None:
            fan_in_item_ids.add(result_carrier_item_id)
        consumers = {
            exchange.node_id
            for revision_id, memberships in valid_exchange_memberships.items()
            if fan_in_item_ids.intersection(memberships)
            for exchange in exchanges_by_revision.get(revision_id, ())
        }
        if not consumers:
            needs_consumer = response_source_operation_id is None or parent != response_source_operation_id
            carrier_mapping_missing = False
            if needs_consumer and result_carrier_item_id is None:
                result_carrier_item_id = mapped_carrier(result_item_id)
                if result_carrier_item_id is None:
                    carrier_mapping_missing = True
                else:
                    fan_in_item_ids.add(result_carrier_item_id)
                    consumers = {
                        exchange.node_id
                        for revision_id, memberships in valid_exchange_memberships.items()
                        if result_carrier_item_id in memberships
                        for exchange in exchanges_by_revision.get(revision_id, ())
                    }
            if needs_consumer and not consumers:
                compaction_runs = [
                    run_id
                    for run_id, consumption in compaction_consumption.by_run.items()
                    if fan_in_item_ids.intersection(consumption.item_ids)
                ]
                if len(compaction_runs) == 1:
                    compaction_node_id = compaction_bridges.get(compaction_runs[0])
                    if compaction_node_id is not None:
                        consumers = {compaction_node_id}
                elif len(compaction_runs) > 1:
                    diagnostics.append("tool result fan-in matches multiple Phase-4 compaction runs")
                    exact = False
            if needs_consumer and not consumers:
                if carrier_mapping_missing:
                    diagnostics.append("carrier mapping unavailable")
                diagnostics.append("tool result fan-in cannot resolve a consuming context revision")
                exact = False
        if not consumers:
            continue
        edges[tool.node_id].update(consumers)
    return exact, response_source_id


def _retry_neighbor_families(retry_mode: str | None) -> frozenset[str] | None:
    if retry_mode in _RETRY_EXCHANGE_MODES:
        return frozenset({"model.exchange"})
    if retry_mode == RetryMode.RUN:
        return frozenset({"model.run"})
    return None


def _connect_unique(
    by_operation: dict[str, list[_ResolvedNode]],
    operation_id: str,
    target_node_id: str | None,
    edges: dict[str, set[str]],
    diagnostics: list[str],
    label: str,
    *,
    source: str | None = None,
    families: frozenset[str] | None = None,
) -> bool:
    candidates = [
        candidate
        for candidate in by_operation.get(operation_id, ())
        if candidate.node_id != source
        and candidate.node_id != target_node_id
        and (families is None or candidate.family in families)
    ]
    if len(candidates) != 1:
        diagnostics.append(f"{label} cannot be resolved uniquely")
        return False
    if source is None and target_node_id is not None:
        edges[candidates[0].node_id].add(target_node_id)
    elif source is not None:
        edges[source].add(candidates[0].node_id)
    return True


def _compaction_consumption_resolution(
    intermediate: _Intermediate,
    nodes: list[_ResolvedNode],
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    cancel_event: Event | None,
    diagnostics: list[str],
) -> _CompactionConsumptionResolution:
    by_run: dict[str, _Phase4Consumption] = {}
    seen_runs: set[str] = set()
    exact = True
    for phase in (node for node in nodes if node.family == "compaction.phase"):
        _check_cancelled(cancel_event)
        if _payload_str(phase.finish.payload, "phase") != "phase4":
            continue
        endpoint = phase.finish
        run_id = endpoint.parent_operation_id
        if run_id is None:
            diagnostics.append("Phase-4 consumed item declaration lacks a compaction run")
            exact = False
            continue
        if run_id in seen_runs:
            diagnostics.append("compaction run declares more than one Phase-4 consumption event")
            by_run.pop(run_id, None)
            exact = False
            continue
        seen_runs.add(run_id)
        entries = _segmented_string_entries(
            intermediate,
            endpoint,
            inactive_ranges,
            field_pointer="/payload/consumed_item_ids",
            cancel_event=cancel_event,
        )
        if entries is None:
            diagnostics.append("Phase-4 compaction has invalid segmented consumed item ids")
            exact = False
            continue
        next_exchange = _payload_str(endpoint.payload, "next_exchange_operation_id")
        if next_exchange is None:
            diagnostics.append("Phase-4 compaction lacks next_exchange_operation_id")
            exact = False
            continue
        by_run[run_id] = _Phase4Consumption(
            item_ids=frozenset(entries),
            next_exchange_operation_id=next_exchange,
        )
    return _CompactionConsumptionResolution(
        by_run=by_run,
        exact=exact,
    )


def _segmented_string_entries(
    intermediate: _Intermediate,
    endpoint: _Endpoint,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    field_pointer: str,
    cancel_event: Event | None,
) -> tuple[str, ...] | None:
    declarations = [item for item in endpoint.segmented_fields if item.field_pointer == field_pointer]
    if len(declarations) != 1 or endpoint.event_id is None:
        return None
    declaration = declarations[0]
    segments = [
        segment
        for segment in intermediate.segments.get(endpoint.event_id, ())
        if _active(segment.sequence, inactive_ranges) and segment.field_pointer == field_pointer
    ]
    if len(segments) != declaration.segment_count:
        return None
    indexed_segments: list[tuple[int, _Segment]] = []
    for segment in segments:
        _check_cancelled(cancel_event)
        if segment.segment_index is None:
            return None
        indexed_segments.append((segment.segment_index, segment))
    if sorted(index for index, _ in indexed_segments) != list(range(declaration.segment_count)):
        return None
    entries: list[str] = []
    for _, segment in sorted(indexed_segments):
        _check_cancelled(cancel_event)
        if (
            segment.segment_count != declaration.segment_count
            or segment.segment_group_id != declaration.segment_group_id
            or segment.encoding != "array_slice"
            or segment.entry_oversized
            or segment.runtime_id != endpoint.runtime_id
            or segment.branch_id != endpoint.branch_id
            or segment.coverage_id != endpoint.coverage_id
            or segment.turn_id != endpoint.turn_id
            or segment.sequence <= endpoint.sequence
            or not _endpoint_in_coverage_runtime(intermediate, segment, inactive_ranges)
        ):
            return None
        values = segment.entries
        if values is None or not all(isinstance(value, str) and is_valid_analytics_id(value) for value in values):
            return None
        entries.extend(cast("str", value) for value in values)
    return tuple(entries)


def _revision_memberships(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    fingerprint_key: bytes | None,
    cancel_event: Event | None,
) -> _RevisionResolution:
    ordered_memberships: dict[str, tuple[MembershipRef, ...]] = {}
    endpoints: dict[str, _Endpoint] = {}
    errors: dict[str, tuple[str, ...]] = {}
    side_call_empty_shell_revisions: list[str] = []
    unidentified_membership_revisions = 0
    definitions = {
        revision_id: [endpoint for endpoint in candidates if _active(endpoint.sequence, inactive_ranges)]
        for revision_id, candidates in intermediate.context_revisions.items()
    }
    for revision_id, candidates in definitions.items():
        _check_cancelled(cancel_event)
        if len(candidates) > 1:
            errors[revision_id] = (f"context revision {revision_id} is defined more than once",)
        elif candidates:
            endpoints[revision_id] = candidates[0]
    for revision_id, endpoint in sorted(endpoints.items(), key=lambda item: item[1].sequence):
        _check_cancelled(cancel_event)
        if revision_id in errors:
            continue
        revision_errors: list[str] = []
        membership: tuple[MembershipRef, ...] | None = None
        entries = None
        if not _endpoint_in_coverage_runtime(intermediate, endpoint, inactive_ranges):
            revision_errors.append(f"context revision {revision_id} falls outside coverage or after runtime closure")
        else:
            entries = _segmented_revision_entries(intermediate, endpoint, inactive_ranges, cancel_event=cancel_event)
        if entries is None:
            revision_errors.append(f"context revision {revision_id} has invalid segmented membership")
        elif _payload_bool(endpoint.payload, "is_checkpoint"):
            membership = _replay_checkpoint(entries)
            if membership is None:
                revision_errors.append(f"context revision {revision_id} has invalid checkpoint membership")
        else:
            parent_id = _payload_str(endpoint.payload, "parent_revision_id")
            parent = endpoints.get(parent_id or "")
            parent_membership = ordered_memberships.get(parent_id or "")
            if (
                parent_id is None
                or parent is None
                or parent_membership is None
                or parent.sequence >= endpoint.sequence
                or parent.actor_id is None
                or parent.actor_id != endpoint.actor_id
                or parent.runtime_id != endpoint.runtime_id
                or parent.branch_id != endpoint.branch_id
            ):
                revision_errors.append(f"context revision {revision_id} has no valid same-actor/runtime parent")
                membership = None
            else:
                membership = _replay_delta(parent_membership, entries)
                if membership is None:
                    revision_errors.append(f"context revision {revision_id} has an invalid delta replay")
        if entries is not None and membership is not None:
            item_count = _payload_int(endpoint.payload, "item_count")
            unidentified = _payload_int(endpoint.payload, "unidentified_item_count")
            if item_count is None or item_count != len(membership):
                revision_errors.append(f"context revision {revision_id} item_count does not match replayed membership")
            if unidentified is None or unidentified < 0:
                revision_errors.append(f"context revision {revision_id} has an invalid unidentified item count")
            elif unidentified and endpoint.side_call and item_count == 0:
                # The known empty-shell shape of a side call: nothing named,
                # nothing to replay; its volume travels through exchange usage.
                side_call_empty_shell_revisions.append(revision_id)
            elif unidentified:
                # The named items replay and verify; the request also carried
                # items nothing can name, so the membership only supports
                # positive claims and the count is surfaced as information.
                unidentified_membership_revisions += 1
            expected_hash = _payload_str(endpoint.payload, "membership_hash")
            if expected_hash is not None:
                actual_hash = membership_hash(fingerprint_key, membership)
                if actual_hash is None:
                    revision_errors.append(f"context revision {revision_id} membership_hash cannot be verified")
                elif actual_hash != expected_hash:
                    revision_errors.append(f"context revision {revision_id} membership_hash does not match replay")
        if revision_errors:
            errors[revision_id] = tuple(revision_errors)
            continue
        if membership is None:
            raise RuntimeError("A validated context revision has no resolved membership.")
        ordered_memberships[revision_id] = membership
    return _RevisionResolution(
        memberships={
            revision_id: tuple(item_id for item_id, _ in membership)
            for revision_id, membership in ordered_memberships.items()
        },
        endpoints=endpoints,
        errors=errors,
        side_call_empty_shell_revisions=tuple(side_call_empty_shell_revisions),
        unidentified_membership_revision_count=unidentified_membership_revisions,
    )


def _segmented_revision_entries(
    intermediate: _Intermediate,
    endpoint: _Endpoint,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    cancel_event: Event | None,
) -> tuple[_RevisionEntry, ...] | None:
    field_pointer = "/payload/refs"
    declarations = [item for item in endpoint.segmented_fields if item.field_pointer == field_pointer]
    if len(declarations) != 1 or endpoint.event_id is None:
        return None
    declaration = declarations[0]
    segments = [
        segment
        for segment in intermediate.segments.get(endpoint.event_id, ())
        if _active(segment.sequence, inactive_ranges) and segment.field_pointer == field_pointer
    ]
    if len(segments) != declaration.segment_count:
        return None
    indexed_segments: list[tuple[int, _Segment]] = []
    for segment in segments:
        _check_cancelled(cancel_event)
        segment_index = segment.segment_index
        if segment_index is None:
            return None
        indexed_segments.append((segment_index, segment))
    if sorted(segment_index for segment_index, _ in indexed_segments) != list(range(declaration.segment_count)):
        return None
    entries: list[_RevisionEntry] = []
    for _, segment in sorted(indexed_segments):
        _check_cancelled(cancel_event)
        if (
            segment.segment_count != declaration.segment_count
            or segment.segment_group_id != declaration.segment_group_id
            or segment.encoding != "array_slice"
            or segment.entry_oversized
            or segment.runtime_id != endpoint.runtime_id
            or segment.branch_id != endpoint.branch_id
            or segment.coverage_id != endpoint.coverage_id
            or segment.turn_id != endpoint.turn_id
            or segment.sequence <= endpoint.sequence
            or not _endpoint_in_coverage_runtime(intermediate, segment, inactive_ranges)
        ):
            return None
        values = segment.entries
        if values is None or not all(isinstance(value, _RevisionEntry) for value in values):
            return None
        entries.extend(value for value in values if isinstance(value, _RevisionEntry))
    return tuple(entries)


def _replay_checkpoint(entries: tuple[_RevisionEntry, ...]) -> tuple[MembershipRef, ...] | None:
    if any(entry.action != "add" for entry in entries):
        return None
    ordered = sorted(entries, key=lambda entry: entry.position)
    if [entry.position for entry in ordered] != list(range(len(ordered))):
        return None
    membership = tuple((entry.item_id, entry.occurrence) for entry in ordered)
    return membership if _valid_membership_occurrences(membership) else None


def _replay_delta(
    parent: tuple[MembershipRef, ...],
    entries: tuple[_RevisionEntry, ...],
) -> tuple[MembershipRef, ...] | None:
    removals = [entry for entry in entries if entry.action == "remove"]
    additions = [entry for entry in entries if entry.action == "add"]
    if len(removals) + len(additions) != len(entries):
        return None
    removal_positions = [entry.position for entry in removals]
    if len(removal_positions) != len(set(removal_positions)):
        return None
    for entry in removals:
        if entry.position >= len(parent) or parent[entry.position] != (entry.item_id, entry.occurrence):
            return None
    replayed = list(parent)
    for entry in sorted(removals, key=lambda item: item.position, reverse=True):
        del replayed[entry.position]
    addition_positions = [entry.position for entry in additions]
    if len(addition_positions) != len(set(addition_positions)):
        return None
    for entry in sorted(additions, key=lambda item: item.position):
        ref = (entry.item_id, entry.occurrence)
        if entry.position > len(replayed) or ref in replayed:
            return None
        replayed.insert(entry.position, ref)
    membership = tuple(replayed)
    return membership if _valid_membership_occurrences(membership) else None


def _valid_membership_occurrences(membership: tuple[MembershipRef, ...]) -> bool:
    seen: dict[str, int] = {}
    for item_id, occurrence in membership:
        if occurrence != seen.get(item_id, 0):
            return False
        seen[item_id] = occurrence + 1
    return True


def _connect_hook_dependencies(
    nodes: list[_ResolvedNode],
    edges: dict[str, set[str]],
    *,
    waited_hook_ids: frozenset[str] | None,
    turn_root_id: str | None,
    response_source_id: str | None,
    response_terminal_id: str | None,
    fork_edges: set[tuple[str, str]],
    diagnostics: list[str],
) -> tuple[bool, bool]:
    by_operation: dict[str, list[_ResolvedNode]] = defaultdict(list)
    for node in nodes:
        by_operation[node.operation_id].append(node)
    runs = sorted((node for node in nodes if node.family == "model.run"), key=lambda node: node.start.sequence)
    dag_exact = True
    response_exact = True
    after_tool_hooks: list[tuple[_ResolvedNode, _ResolvedNode]] = []
    blocking_dispatches: dict[tuple[str, str | None, str], list[_ResolvedNode]] = defaultdict(list)
    for hook in (node for node in nodes if node.family == "hook.operation"):
        mode = _payload_str(hook.start.payload, "execution_mode")
        event = _payload_str(hook.start.payload, "hook_event")
        if mode not in _HOOK_EXECUTION_MODES:
            continue
        target = _payload_str(hook.start.payload, "target_operation_id")
        if mode == "blocking" and event != HookEvent.USER_INTERRUPT:
            scope = _hook_drain_scope(hook.start)
            if event is None or scope is None:
                diagnostics.append("blocking hook dispatch lacks field-based ordering evidence")
                dag_exact = False
            else:
                blocking_dispatches[(event, target, scope)].append(hook)
        target_families = _hook_target_families(event)
        targets = [
            candidate
            for candidate in by_operation.get(target or "", ())
            if candidate.family != "retry" and (not target_families or candidate.family in target_families)
        ]
        if mode != "blocking" or event == HookEvent.USER_INTERRUPT:
            if target is not None and len(targets) == 1:
                edges[targets[0].node_id].add(hook.node_id)
                fork_edges.add((targets[0].node_id, hook.node_id))
        elif event == HookEvent.BEFORE_TOOL_CALL and len(targets) == 1:
            edges[hook.node_id].add(targets[0].node_id)
        elif event == HookEvent.BEFORE_TURN and runs:
            edges[hook.node_id].add(runs[0].node_id)
        elif event in {HookEvent.AFTER_TOOL_CALL, HookEvent.TOOL_ERROR} and len(targets) == 1:
            after_tool_hooks.append((hook, targets[0]))
        elif event == HookEvent.AFTER_TURN and response_source_id is not None:
            edges[response_source_id].add(hook.node_id)
            if response_terminal_id is not None:
                edges[hook.node_id].add(response_terminal_id)
    after_tool_hook_node_ids = {hook.node_id for hook, _ in after_tool_hooks}
    for hook, target in after_tool_hooks:
        for successor in tuple(edges.get(target.node_id, ())):
            if successor not in after_tool_hook_node_ids:
                edges[hook.node_id].add(successor)
    for dispatch_hooks in blocking_dispatches.values():
        ordered = sorted(dispatch_hooks, key=lambda hook: hook.start.sequence)
        for previous, current in pairwise(ordered):
            edges[previous.node_id].add(current.node_id)
            if (
                previous.start.runtime_id != current.start.runtime_id
                or previous.start.branch_id != current.start.branch_id
                or previous.finish.sequence >= current.start.sequence
                or previous.finish.monotonic_ns > current.start.monotonic_ns
            ):
                diagnostics.append("blocking hook dispatch has inconsistent serial intervals")
                dag_exact = False
    if waited_hook_ids is not None and response_terminal_id is not None:
        hooks = {node.operation_id: node for node in nodes if node.family == "hook.operation"}
        for operation_id in waited_hook_ids:
            hook = hooks.get(operation_id)
            if hook is None:
                diagnostics.append("turn response fence names an unknown hook operation")
                response_exact = False
            else:
                edges[hook.node_id].add(response_terminal_id)
                if turn_root_id is None or not _reachable(edges, turn_root_id, hook.node_id):
                    diagnostics.append("waited async hook lacks a typed fork path from the turn root")
                    response_exact = False
    return dag_exact, response_exact


def _reachable(edges: dict[str, set[str]], source: str, target: str) -> bool:
    pending = [source]
    seen: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id == target:
            return True
        if node_id in seen:
            continue
        seen.add(node_id)
        pending.extend(edges.get(node_id, ()))
    return False


def _critical_paths(
    nodes: list[_ResolvedNode],
    slices: tuple[TimeSlice, ...],
    dependency: _DependencyProof,
    *,
    cancel_event: Event | None,
) -> tuple[int | None, int | None, bool, bool, bool, bool]:
    compute_intervals, response_intervals = _critical_path_interval_maps(nodes, slices, cancel_event=cancel_event)
    terminal_id = dependency.response_terminal_id
    if terminal_id is not None:
        compute_intervals[terminal_id] = []
        response_intervals[terminal_id] = [
            (item.start_ns, item.end_ns) for item in slices if item.operation_id is None and item.response_weight
        ]
    compute = _longest_interval_path(
        compute_intervals,
        dependency.edges,
        parents=dependency.parents,
        fork_edges=dependency.fork_edges,
        cancel_event=cancel_event,
    )
    if dependency.turn_root_id is None or terminal_id is None:
        response = _PathResolution(None, compute.acyclic, True)
        response_reachable = False
    else:
        response = _longest_interval_path(
            response_intervals,
            dependency.edges,
            parents=dependency.parents,
            fork_edges=dependency.fork_edges,
            root_id=dependency.turn_root_id,
            terminal_id=terminal_id,
            cancel_event=cancel_event,
        )
        response_reachable = _reachable(dependency.edges, dependency.turn_root_id, terminal_id)
    return (
        compute.value,
        response.value,
        compute.acyclic and response.acyclic,
        response_reachable,
        compute.bounded,
        response.bounded,
    )


def _critical_path_interval_maps(
    nodes: list[_ResolvedNode],
    slices: tuple[TimeSlice, ...],
    *,
    cancel_event: Event | None,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]]]:
    compute_intervals = {node.node_id: [] for node in nodes}
    response_intervals = {node.node_id: [] for node in nodes}
    for item in slices:
        _check_cancelled(cancel_event)
        if item.operation_id is None:
            continue
        candidates = [node for node in nodes if node.operation_id == item.operation_id]
        if len(candidates) != 1:
            candidates = [
                node
                for node in candidates
                if (attributes := _slice_attributes(node)) is not None and attributes[4] == item.owner
            ]
        if len(candidates) != 1:
            continue
        node_id = candidates[0].node_id
        if item.compute_weight:
            compute_intervals[node_id].append((item.start_ns, item.end_ns))
        if item.response_weight:
            response_intervals[node_id].append((item.start_ns, item.end_ns))
    return compute_intervals, response_intervals


def _failed_tool_cp_contributions(
    nodes: list[_ResolvedNode],
    slices: tuple[TimeSlice, ...],
    dependency: _DependencyProof,
    *,
    response_cp: int,
    cancel_event: Event | None,
) -> dict[str, int]:
    groups = {
        node.operation_id: frozenset((node.node_id,))
        for node in nodes
        if node.family == "tool.operation" and _tool_failed(_payload_str(node.finish.payload, "outcome"))
    }
    return {
        operation_id: contribution
        for operation_id, contribution in _grouped_cp_contributions(
            nodes,
            slices,
            dependency,
            groups=groups,
            response_cp=response_cp,
            cancel_event=cancel_event,
        ).items()
        if contribution
    }


def _server_tool_cp_contributions(
    intermediate: _Intermediate,
    nodes: list[_ResolvedNode],
    slices: tuple[TimeSlice, ...],
    dependency: _DependencyProof,
    *,
    response_cp: int,
    cancel_event: Event | None,
) -> dict[str, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    tool_nodes = intermediate.nodes.get("tool.operation", {})
    for node in nodes:
        _check_cancelled(cancel_event)
        if node.family != "tool.operation":
            continue
        raw_node = tool_nodes.get(node.operation_id)
        context = _tool_context_for_start(raw_node, node.start) if raw_node is not None else None
        if context is not None and context.server_name is not None:
            grouped[context.server_name].add(node.node_id)
    return _grouped_cp_contributions(
        nodes,
        slices,
        dependency,
        groups={name: frozenset(node_ids) for name, node_ids in grouped.items()},
        response_cp=response_cp,
        cancel_event=cancel_event,
    )


def _grouped_cp_contributions(
    nodes: list[_ResolvedNode],
    slices: tuple[TimeSlice, ...],
    dependency: _DependencyProof,
    *,
    groups: dict[str, frozenset[str]],
    response_cp: int,
    cancel_event: Event | None,
) -> dict[str, int]:
    """Resolve one leave-one-group-out response path per supplied group."""
    terminal_id = dependency.response_terminal_id
    root_id = dependency.turn_root_id
    if not groups or root_id is None or terminal_id is None or response_cp <= 0:
        return {}
    _, response_intervals = _critical_path_interval_maps(nodes, slices, cancel_event=cancel_event)
    response_intervals[terminal_id] = [
        (item.start_ns, item.end_ns) for item in slices if item.operation_id is None and item.response_weight
    ]
    children: dict[str, set[str]] = defaultdict(set)
    for child, parent in dependency.parents.items():
        children[parent].add(child)
    contributions: dict[str, int] = {}
    for group, node_ids in groups.items():
        _check_cancelled(cancel_event)
        without = dict(response_intervals)
        # A member's contribution spans everything it displaced: the approval
        # and sub-agent inside a tool left with it.
        for node_id in node_ids:
            without[node_id] = []
            for descendant in _descendants(node_id, children):
                without[descendant] = []
        resolved = _longest_interval_path(
            without,
            dependency.edges,
            parents=dependency.parents,
            fork_edges=dependency.fork_edges,
            root_id=root_id,
            terminal_id=terminal_id,
            cancel_event=cancel_event,
        )
        if resolved.value is not None and resolved.acyclic and resolved.bounded:
            contributions[group] = max(0, response_cp - resolved.value)
    return contributions


def _wall_partition(
    bounds: tuple[int, int],
    slices: tuple[TimeSlice, ...],
    *,
    cancel_event: Event | None,
) -> tuple[dict[WallBucket, int], tuple[tuple[int, int], ...]]:
    points = {bounds[0], bounds[1]}
    changes: dict[int, Counter[WallBucket]] = defaultdict(Counter)
    for item in slices:
        _check_cancelled(cancel_event)
        if item.wall_bucket is None:
            continue
        start = max(bounds[0], item.start_ns)
        end = min(bounds[1], item.end_ns)
        if end <= start:
            continue
        points.add(start)
        points.add(end)
        changes[start][item.wall_bucket] += 1
        changes[end][item.wall_bucket] -= 1
    ordered = sorted(points)
    durations = dict.fromkeys(WallBucket, 0)
    idle: list[tuple[int, int]] = []
    active_counts: Counter[WallBucket] = Counter()
    for left, right in pairwise(ordered):
        _check_cancelled(cancel_event)
        for bucket, delta in changes.get(left, {}).items():
            count = active_counts[bucket] + delta
            if count > 0:
                active_counts[bucket] = count
            else:
                active_counts.pop(bucket, None)
        if right <= left:
            continue
        bucket = max(active_counts, key=lambda value: _WALL_PRIORITY[value]) if active_counts else WallBucket.IDLE
        durations[bucket] += right - left
        if bucket is WallBucket.IDLE:
            idle.append((left, right))
    return durations, interval_union(idle)


def _turn_usage(
    intermediate: _Intermediate,
    turn_id: str,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    integrity_exact: bool,
    integrity_reason: str,
    turn_lifecycle_exact: bool,
    closed_exchange_terminal_sequences: frozenset[int],
    cancel_event: Event | None,
) -> Metric:
    unclosed_start = False
    exchange_projection_exact = True
    for node in intermediate.nodes_by_turn.get(turn_id, ()):
        _check_cancelled(cancel_event)
        if node.family != "model.exchange":
            continue
        starts = [
            event for event in node.starts if event.turn_id == turn_id and _active(event.sequence, inactive_ranges)
        ]
        finishes = [
            event for event in node.finishes if event.turn_id == turn_id and _active(event.sequence, inactive_ranges)
        ]
        endpoints = (*starts, *finishes)
        if not endpoints:
            continue
        if _lifecycle_cut(node, inactive_ranges) is not _LifecycleCut.NONE:
            exchange_projection_exact = False
            continue
        if len(starts) == 1 and not finishes:
            unclosed_start = True
            continue
        if len(starts) != 1 or len(finishes) != 1 or finishes[0].sequence not in closed_exchange_terminal_sequences:
            exchange_projection_exact = False
    if unclosed_start:
        return Metric(None, Precision.MISSING, "one or more model exchanges have no terminal usage")
    exchanges = [
        item for item in intermediate.usage_by_turn.get(turn_id, ()) if _active(item.sequence, inactive_ranges)
    ]
    if any(
        item.normalization_unavailable or item.input_total is None or item.output_total is None for item in exchanges
    ):
        return Metric(None, Precision.MISSING, "one or more normalized usage totals are missing")
    value = sum(cast("int", item.input_total) + cast("int", item.output_total) for item in exchanges)
    if not integrity_exact:
        return Metric(value, Precision.UNRESOLVED, integrity_reason or "turn sequence integrity is unresolved")
    usage_terminals = [item.sequence for item in exchanges]
    if (
        not turn_lifecycle_exact
        or not exchange_projection_exact
        or any(sequence not in closed_exchange_terminal_sequences for sequence in usage_terminals)
    ):
        return Metric(value, Precision.UNRESOLVED, "usage requires a uniquely closed exchange and exact turn lifecycle")
    if not all(
        item.bucket_has_provider_provenance(UsageBucket.INPUT)
        and item.bucket_has_provider_provenance(UsageBucket.OUTPUT)
        for item in exchanges
    ):
        return Metric(value, Precision.UNRESOLVED, "normalized usage provenance is incomplete")
    return Metric(value, Precision.EXACT)


def _timeline_diagnostics(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    cancel_event: Event | None,
) -> tuple[tuple[SpanDurationMismatch, ...], tuple[ContainmentViolation, ...]]:
    resolved: dict[str, list[_ResolvedNode]] = defaultdict(list)
    span_mismatches: list[SpanDurationMismatch] = []
    for family_nodes in intermediate.nodes.values():
        for node in family_nodes.values():
            _check_cancelled(cancel_event)
            starts = [
                event for event in node.starts if _active(event.sequence, inactive_ranges) and not event.side_call
            ]
            finishes = [
                event for event in node.finishes if _active(event.sequence, inactive_ranges) and not event.side_call
            ]
            if len(starts) != 1 or len(finishes) != 1:
                continue
            start, finish = starts[0], finishes[0]
            if (
                start.scope != finish.scope
                or (node.family != "compaction.phase" and finish.sequence <= start.sequence)
                or finish.monotonic_ns < start.monotonic_ns
            ):
                continue
            resolved_node = _ResolvedNode(
                node_id=f"{node.family}:{node.operation_id}",
                family=node.family,
                operation_id=node.operation_id,
                start=start,
                finish=finish,
            )
            resolved[node.operation_id].append(resolved_node)
            duration_ms = _payload_int(finish.payload, "wait_ms" if node.family == "approval" else "duration_ms")
            if duration_ms is None or _known_duration_offset(resolved_node):
                continue
            interval_ns = finish.monotonic_ns - start.monotonic_ns
            if abs(interval_ns - duration_ms * 1_000_000) > max(2_000_000, interval_ns // 100):
                span_mismatches.append(
                    SpanDurationMismatch(
                        family=node.family,
                        operation_id=node.operation_id,
                        start_sequence=start.sequence,
                        finish_sequence=finish.sequence,
                        interval_ns=interval_ns,
                        recorded_duration_ms=duration_ms,
                    )
                )
    containment_violations: list[ContainmentViolation] = []
    for operations in resolved.values():
        for node in operations:
            _check_cancelled(cancel_event)
            parent_id = node.start.parent_operation_id
            parents = resolved.get(parent_id or "", ())
            if len(parents) != 1:
                continue
            parent = parents[0]
            if _known_noncontained_shape(node, parent):
                continue
            if (
                node.start.runtime_id == parent.start.runtime_id
                and node.start.branch_id == parent.start.branch_id
                and node.start.turn_id == parent.start.turn_id
                and (
                    node.start.monotonic_ns < parent.start.monotonic_ns
                    or node.finish.monotonic_ns > parent.finish.monotonic_ns
                )
            ):
                containment_violations.append(
                    ContainmentViolation(
                        family=node.family,
                        operation_id=node.operation_id,
                        parent_family=parent.family,
                        parent_operation_id=parent.operation_id,
                        start_sequence=node.start.sequence,
                        finish_sequence=node.finish.sequence,
                    )
                )
    return tuple(span_mismatches), tuple(containment_violations)


def _known_noncontained_shape(node: _ResolvedNode, parent: _ResolvedNode) -> bool:
    if node.start.side_call or node.family in {"turn.suspension", "continuation.poll"}:
        return True
    if node.family == "approval" and parent.family == "model.exchange":
        return True
    if node.family == "hook.operation":
        hook_event = _payload_str(node.start.payload, "hook_event")
        execution_mode = _payload_str(node.start.payload, "execution_mode")
        return hook_event in {
            HookEvent.BEFORE_TOOL_CALL,
            HookEvent.AFTER_TOOL_CALL,
            HookEvent.TOOL_ERROR,
            HookEvent.USER_INTERRUPT,
        } or execution_mode in {"async", "fire_and_forget"}
    if parent.family == "model.exchange" and parent.finish.monotonic_ns <= node.start.monotonic_ns:
        if node.family == "tool.operation":
            return True
        if node.family == "preparation" and _payload_str(node.start.payload, "scope") == "tool_preamble":
            return True
    return node.family == "model.run" and any(relation == LinkRelation.CAUSED_BY for relation, _ in node.start.links)


def _turn_bucket_usage(exchanges: list[_ExchangeUsage], bucket: UsageBucket, verdict: Metric) -> Metric:
    """Split one turn's usage verdict into a per-bucket display metric.

    The turn-level verdict caps every bucket: a missing or unresolved turn can
    never yield an exact bucket. Optional buckets additionally degrade on
    partial reporting because providers normalize them inconsistently.
    """
    if verdict.value is None:
        return Metric(None, verdict.precision, verdict.reason)
    reported = [value for item in exchanges if (value := item.bucket_value(bucket)) is not None]
    if bucket in _OPTIONAL_USAGE_BUCKETS and exchanges and not reported:
        return Metric(None, Precision.MISSING, "no exchange reported this bucket")
    value = sum(reported)
    if verdict.precision is not Precision.EXACT:
        return Metric(value, Precision.UNRESOLVED, verdict.reason)
    if bucket is UsageBucket.REASONING and any(
        item.extras is not None
        and item.extras.reasoning is not None
        and item.output_total is not None
        and item.extras.reasoning > item.output_total
        for item in exchanges
    ):
        return Metric(value, Precision.UNRESOLVED, "reasoning tokens exceed normalized output tokens")
    if len(reported) < len(exchanges):
        return Metric(value, Precision.ESTIMATED, "not every exchange reported this bucket")
    if not all(item.bucket_has_provider_provenance(bucket) for item in exchanges):
        return Metric(value, Precision.UNRESOLVED, "normalized usage provenance is incomplete")
    return Metric(value, Precision.EXACT)


def _counter_axis_diagnostics(
    intermediate: _Intermediate,
    turn_id: str,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    resolution_cache: _ResolutionCache,
    axis_start_ns: int,
    axis_end_ns: int,
    revisions: _RevisionResolution,
    refresh: bool,
    cancel_event: Event | None,
) -> tuple[str, ...]:
    """Validate one dirty physical turn's counter timestamps without a session scan."""
    if not refresh:
        return resolution_cache.counter_axis_diagnostics_by_turn.get(turn_id, ())
    diagnostics: list[str] = []
    usage_sequences: set[int] = set()
    for item in intermediate.usage_by_turn.get(turn_id, ()):
        _check_cancelled(cancel_event)
        if _active(item.sequence, inactive_ranges) and (
            item.input_total is not None or item.output_total is not None or item.extras is not None
        ):
            usage_sequences.add(item.sequence)
    usage_outside = False
    if usage_sequences:
        for node in intermediate.nodes_by_turn.get(turn_id, ()):
            _check_cancelled(cancel_event)
            if node.family != "model.exchange":
                continue
            for endpoint in node.finishes:
                _check_cancelled(cancel_event)
                if (
                    endpoint.turn_id == turn_id
                    and endpoint.sequence in usage_sequences
                    and _active(endpoint.sequence, inactive_ranges)
                    and not axis_start_ns <= endpoint.monotonic_ns <= axis_end_ns
                ):
                    usage_outside = True
                    break
            if usage_outside:
                break
    if usage_outside:
        diagnostics.append("usage sample lies outside its owning attempt axis")

    for revision_id in intermediate.context_revision_ids_by_turn.get(turn_id, ()):
        _check_cancelled(cancel_event)
        endpoint = revisions.endpoints.get(revision_id)
        if endpoint is None:
            continue
        item_count = _payload_int(endpoint.payload, "item_count")
        if (
            item_count is not None
            and item_count >= 0
            and not endpoint.side_call
            and not axis_start_ns <= endpoint.monotonic_ns <= axis_end_ns
        ):
            diagnostics.append("context sample lies outside its owning attempt axis")
            break
    result = tuple(diagnostics)
    resolution_cache.counter_axis_diagnostics_by_turn[turn_id] = result
    return result


@lru_cache(maxsize=256)
def _interned_token_usage(metrics: tuple[Metric, ...]) -> TokenUsage:
    """Share value-identical per-turn usage; every retained turn carries one.

    Long sessions repeat the same bucket shapes turn after turn, and a private
    five-metric dict per turn is what the residency ceiling notices first.
    """
    return TokenUsage(buckets=dict(zip(UsageBucket, metrics, strict=True)))


def _empty_turn(
    intermediate: _Intermediate,
    turn_id: str,
    starts: list[_Endpoint],
    reason: str,
    *,
    inactive_ranges: tuple[tuple[int, int], ...],
    resolution_cache: _ResolutionCache,
    revisions: _RevisionResolution,
    refresh: bool,
    cancel_event: Event | None,
) -> TurnAnalysis:
    """Build an unresolved physical attempt without discarding stable opener identity.

    Duplicate starts may still prove one ``turn_number``/``is_retry`` pair and
    therefore participate in logical retry folding. With no surviving start
    (for example, an opener hidden by an unaccounted gap), no such association
    is provable and the attempt deliberately remains an unnumbered turn.
    """
    start = starts[0] if starts else None
    turn_numbers = {_payload_int(candidate.payload, "turn_number") for candidate in starts}
    retry_flags = {_payload_bool(candidate.payload, "is_retry") for candidate in starts}
    runtime_ids = {candidate.runtime_id for candidate in starts}
    turn_number = next(iter(turn_numbers)) if len(turn_numbers) == 1 else None
    is_retry = next(iter(retry_flags)) if len(retry_flags) == 1 else False
    diagnostics = [reason]
    if starts and (len(turn_numbers) != 1 or len(retry_flags) != 1):
        diagnostics.append("turn starts disagree on logical retry identity")
    if len(runtime_ids) > 1:
        diagnostics.append("turn starts disagree on runtime ownership")
    unresolved = Metric(None, Precision.UNRESOLVED, reason)
    runtime_id = start.runtime_id if start is not None else ""
    sequence = start.sequence if start is not None else 0
    axis_ns = start.monotonic_ns if start is not None else 0
    diagnostics.extend(
        _counter_axis_diagnostics(
            intermediate,
            turn_id,
            inactive_ranges,
            resolution_cache=resolution_cache,
            axis_start_ns=axis_ns,
            axis_end_ns=axis_ns,
            revisions=revisions,
            refresh=refresh,
            cancel_event=cancel_event,
        )
    )
    wall = dict.fromkeys(WallBucket, unresolved)
    utilization = dict.fromkeys((WallBucket.MODEL, WallBucket.TOOLS), unresolved)
    return TurnAnalysis(
        turn_id=turn_id,
        turn_number=turn_number,
        runtime_id=runtime_id,
        start_sequence=sequence,
        end_sequence=None,
        elapsed_ns=unresolved,
        compute_cp_ns=unresolved,
        response_cp_ns=unresolved,
        exclusive_work_ns=unresolved,
        parallelism=unresolved,
        overlap_gain_ns=unresolved,
        wall_time_ns=wall,
        utilization=utilization,
        usage_tokens=Metric(None, Precision.MISSING, reason),
        attempts=(
            TurnAttemptRef(
                turn_id=turn_id,
                runtime_id=runtime_id,
                is_retry=is_retry,
                physical_axis_start_ns=axis_ns,
                physical_axis_end_ns=axis_ns,
                logical_axis_start_ns=axis_ns,
                operation_start_index=0,
                operation_end_index=0,
                slice_start_index=0,
                slice_end_index=0,
            ),
        ),
        axis_start_ns=axis_ns,
        axis_end_ns=axis_ns,
        diagnostics=tuple(diagnostics),
        action_projection_precision=Precision.UNRESOLVED,
        action_projection_reason=reason,
    )


def _descendants(node_id: str, children: dict[str, set[str]]) -> set[str]:
    result: set[str] = set()
    pending = list(children.get(node_id, set()))
    while pending:
        child = pending.pop()
        if child in result:
            continue
        result.add(child)
        pending.extend(children.get(child, set()))
    return result


def _has_cycle(children: dict[str, set[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        if any(visit(child) for child in children.get(node_id, set())):
            return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(visit(node_id) for node_id in children)
