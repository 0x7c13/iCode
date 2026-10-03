# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rollback-aware lifecycle projection shared by the analytics resolvers.

Runtime/coverage reachability and timeline rows serve both Chat turns and
Workflow runs. Lifecycle ownership under rollback serves the session totals in
``aggregation`` and the insight panels; the per-start tool context and hook
drain-scope readers serve the turn, turn-graph and action resolvers; the
session-wide span/containment diagnostics feed ``aggregation``'s trajectory
diagnostics.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from threading import Event

from chrys.foundation.trajectory.envelope import LinkRelation
from chrys.service.analytics._facts import (
    WORKFLOW_NODE_FAMILY,
    WORKFLOW_RUN_FAMILY,
    _active,
    _Endpoint,
    _Intermediate,
    _Node,
    _payload_int,
    _payload_str,
    _projection_membership,
    _Segment,
    _ToolContextExtras,
)
from chrys.service.analytics.model import (
    ContainmentViolation,
    Precision,
    SpanDurationMismatch,
    TimelineDiagnosticCode,
    TimelineOperation,
    TimelineOperationDetail,
)
from chrys.service.analytics.reader import raise_if_cancelled as _check_cancelled
from chrys.service.hooks.events import HookEvent


@dataclass(frozen=True, slots=True)
class _ResolvedNode:
    node_id: str
    family: str
    operation_id: str
    start: _Endpoint
    finish: _Endpoint

    @property
    def interval(self) -> tuple[int, int]:
        return (self.start.monotonic_ns, self.finish.monotonic_ns)


@dataclass(frozen=True, slots=True)
class _TimelineProjection:
    operation_id: str
    parent_operation_id: str | None
    start_sequence: int
    family: str
    start_ns: int | None
    end_ns: int | None
    precision: Precision
    reason: str | None
    diagnostic_code: TimelineDiagnosticCode | None
    identity: str | None
    hook_id: str | None


def _infrastructure_branch_reaches(
    intermediate: _Intermediate,
    anchor: _Endpoint | _Segment,
    endpoint: _Endpoint | _Segment,
    inactive_ranges: tuple[tuple[int, int], ...],
) -> bool:
    """Whether a physical runtime/coverage anchor can cover *endpoint*.

    Rollback supersedes logical history, not the recorder prelude that made
    the current runtime and coverage observable.  A cold resumed recorder can
    write that prelude on the recovered branch immediately before rollback
    opens its successor, so infrastructure crosses branches only through a
    fully paired, active rollback transition in sequence order.
    """
    if anchor.branch_id == endpoint.branch_id:
        return True
    if anchor.sequence > endpoint.sequence:
        return False
    supersessions = [
        (sequence, old_branch, new_branch)
        for sequence, old_branch, new_branch in intermediate.branch_supersessions
        if _active(sequence, inactive_ranges) and old_branch is not None and new_branch is not None
    ]
    reachable = {anchor.branch_id}
    for sequence, old_branch, new_branch in sorted(intermediate.rollback_branch_pairs):
        if sequence <= anchor.sequence or sequence > endpoint.sequence or not _active(sequence, inactive_ranges):
            continue
        if old_branch in reachable and any(
            sequence < supersession_sequence <= endpoint.sequence
            and superseded_branch == old_branch
            and successor_branch == new_branch
            for supersession_sequence, superseded_branch, successor_branch in supersessions
        ):
            reachable.add(new_branch)
    return endpoint.branch_id in reachable


def _endpoint_in_coverage_runtime(
    intermediate: _Intermediate,
    endpoint: _Endpoint | _Segment,
    inactive_ranges: tuple[tuple[int, int], ...],
) -> bool:
    coverage_starts = [
        event
        for event in intermediate.coverage_starts.get(endpoint.coverage_id, ())
        if event.runtime_id == endpoint.runtime_id
        and _infrastructure_branch_reaches(intermediate, event, endpoint, inactive_ranges)
        and event.sequence <= endpoint.sequence
    ]
    if len(coverage_starts) != 1:
        return False
    coverage_ends = [
        event
        for event in intermediate.coverage_ends.get(endpoint.coverage_id, ())
        if event.runtime_id == endpoint.runtime_id
        and _infrastructure_branch_reaches(intermediate, endpoint, event, inactive_ranges)
    ]
    if len(coverage_ends) > 1:
        return False
    if coverage_ends:
        last_sequence = _payload_int(coverage_ends[0].payload, "last_sequence")
        if last_sequence is None or endpoint.sequence > last_sequence:
            return False
    runtime_starts = [
        event
        for event in intermediate.runtime_starts.get(endpoint.runtime_id, ())
        if _infrastructure_branch_reaches(intermediate, event, endpoint, inactive_ranges)
    ]
    if len(runtime_starts) > 1 or (runtime_starts and endpoint.sequence <= runtime_starts[0].sequence):
        return False
    runtime_recoveries = [
        event
        for event in intermediate.runtime_recoveries.get(endpoint.runtime_id, ())
        if _infrastructure_branch_reaches(intermediate, event, endpoint, inactive_ranges)
    ]
    if len(runtime_recoveries) > 1 or (runtime_recoveries and endpoint.sequence <= runtime_recoveries[0].sequence):
        return False
    runtime_finishes = [
        event
        for event in intermediate.runtime_finishes.get(endpoint.runtime_id, ())
        if _infrastructure_branch_reaches(intermediate, endpoint, event, inactive_ranges)
    ]
    return len(runtime_finishes) <= 1 and (not runtime_finishes or endpoint.sequence < runtime_finishes[0].sequence)


def _materialize_timeline(rows: list[_TimelineProjection]) -> tuple[TimelineOperation, ...]:
    """Shared lifecycle row ordering and parent depth for Chat turns and Workflow runs."""
    ordered = sorted(rows, key=lambda row: (row.start_sequence, row.family, row.operation_id))
    depths = _timeline_depths(ordered)
    return tuple(
        TimelineOperation(
            operation_id=row.operation_id,
            family=row.family,
            depth=depths[row.operation_id],
            start_ns=row.start_ns,
            end_ns=row.end_ns,
            precision=row.precision,
            identity=row.identity,
            detail=(
                TimelineOperationDetail(
                    reason=row.reason,
                    diagnostic_code=row.diagnostic_code,
                    hook_id=row.hook_id,
                )
                if row.reason is not None or row.diagnostic_code is not None or row.hook_id is not None
                else None
            ),
        )
        for row in ordered
    )


def _resolved_timeline_operation(node: _ResolvedNode) -> _TimelineProjection:
    detached = node.family == "hook.operation" and _payload_str(node.finish.payload, "outcome") == "detached"
    return _TimelineProjection(
        operation_id=node.operation_id,
        parent_operation_id=_timeline_parent_operation(node.family, node.start),
        start_sequence=node.start.sequence,
        family=node.family,
        start_ns=None if detached else node.start.monotonic_ns,
        end_ns=None if detached else node.finish.monotonic_ns,
        precision=Precision.MISSING if detached else Precision.EXACT,
        reason="detached hook records spawn latency, not work duration" if detached else None,
        diagnostic_code=TimelineDiagnosticCode.DETACHED_HOOK if detached else None,
        identity=_timeline_identity(node.family, node.start),
        hook_id=_timeline_hook_id(node.family, node.start),
    )


def _timeline_depths(rows: list[_TimelineProjection]) -> dict[str, int]:
    by_id = {row.operation_id: row for row in rows}
    depths: dict[str, int] = {}

    def depth(row: _TimelineProjection, trail: frozenset[str]) -> int:
        cached = depths.get(row.operation_id)
        if cached is not None:
            return cached
        parent_id = row.parent_operation_id
        if parent_id is None or parent_id not in by_id or parent_id in trail:
            result = 0
        else:
            result = min(4, depth(by_id[parent_id], trail | {row.operation_id}) + 1)
        depths[row.operation_id] = result
        return result

    for row in rows:
        depth(row, frozenset())
    return depths


def _timeline_identity(family: str, endpoint: _Endpoint) -> str | None:
    if family == WORKFLOW_RUN_FAMILY:
        return _payload_str(endpoint.payload, "workflow")
    if family == WORKFLOW_NODE_FAMILY:
        activation = _payload_str(endpoint.payload, "activation")
        attempt = _payload_int(endpoint.payload, "attempt")
        return f"{activation} · #{attempt}"
    if family == "tool.operation":
        name = _payload_str(endpoint.payload, "tool_name")
        fingerprint = _payload_str(endpoint.payload, "argument_fingerprint")
        if name is not None and fingerprint is not None:
            return f"{name} (#{fingerprint[:8]})"
        return name
    if family == "hook.operation":
        hook_event = _payload_str(endpoint.payload, "hook_event")
        hook_id = _timeline_hook_id(family, endpoint)
        return hook_event or hook_id
    if family in {"wait", "continuation.poll", "turn.suspension"}:
        return _payload_str(endpoint.payload, "category")
    if family == "sub_agent":
        return _payload_str(endpoint.payload, "agent_profile")
    return None


def _timeline_hook_id(family: str, endpoint: _Endpoint) -> str | None:
    if family != "hook.operation":
        return None
    return _payload_str(endpoint.payload, "hook_key") or _payload_str(endpoint.payload, "hook_id")


def _timeline_parent_operation(family: str, start: _Endpoint) -> str | None:
    if family == "tool.operation":
        return _payload_str(start.payload, "parent_model_operation_id") or start.parent_operation_id
    if family in {"hook.operation", "wait", "approval"}:
        return start.parent_operation_id or _payload_str(start.payload, "target_operation_id")
    return start.parent_operation_id


def _tool_context_for_start(node: _Node, start: _Endpoint) -> _ToolContextExtras | None:
    extras = node.extras
    if extras is None:
        return None
    return next((item for item in extras.contexts if item.sequence == start.sequence), None)


def _hook_drain_scope(start: _Endpoint) -> str | None:
    """Read the renamed hook lifetime scope while accepting retained logs."""
    return _payload_str(start.payload, "drain_scope") or _payload_str(start.payload, "scope")


def _merge_projection_memberships(*memberships: tuple[bool, bool]) -> tuple[bool, bool]:
    """Combine ``(active, inactive)`` ownership evidence."""
    return any(active for active, _ in memberships), any(inactive for _, inactive in memberships)


def _exclusively_inactive(membership: tuple[bool, bool]) -> bool:
    """Return whether ownership is proven inactive without an active claimant."""
    active, inactive = membership
    return inactive and not active


def _turn_start_membership(
    intermediate: _Intermediate,
    turn_ids: set[str],
    inactive_ranges: tuple[tuple[int, int], ...],
) -> tuple[bool, bool]:
    starts = [start for turn_id in turn_ids if (turn := intermediate.turns.get(turn_id)) for start in turn.starts]
    return _projection_membership(starts, inactive_ranges)


def _operation_start_membership(
    intermediate: _Intermediate,
    operation_id: str | None,
    inactive_ranges: tuple[tuple[int, int], ...],
) -> tuple[bool, bool]:
    if operation_id is None:
        return False, False
    starts = [
        start
        for family_nodes in intermediate.nodes.values()
        if (candidate := family_nodes.get(operation_id)) is not None
        for start in candidate.starts
    ]
    return _projection_membership(starts, inactive_ranges)


def _node_owner_membership(
    intermediate: _Intermediate,
    node: _Node,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    target_operation_id: str | None = None,
) -> tuple[bool, bool]:
    """Resolve lifecycle ownership from its turn and optional target operation."""
    turn_ids = {endpoint.turn_id for endpoint in (*node.starts, *node.finishes) if endpoint.turn_id is not None}
    return _merge_projection_memberships(
        _turn_start_membership(intermediate, turn_ids, inactive_ranges),
        _operation_start_membership(intermediate, target_operation_id, inactive_ranges),
    )


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


def _known_duration_offset(node: _ResolvedNode) -> bool:
    if node.family in {"tool.operation", "compaction.phase", "approval"}:
        return True
    return node.family == "model.run" and (
        _payload_int(node.start.payload, "attempt_index") not in {None, 0}
        or _payload_str(node.start.payload, "previous_run_operation_id") is not None
    )
