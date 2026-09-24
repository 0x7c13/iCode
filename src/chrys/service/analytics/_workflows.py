# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow run roots and their operation trees, independent of Chat turns.

Only measured lifecycle durations are exposed here. Chat's dependency proof
cannot establish a Workflow critical path; unsupported aggregates stay missing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from threading import Event

from chrys.service.analytics._facts import WORKFLOW_RUN_FAMILY, _active, _Endpoint, _Intermediate, _Node, _payload_str
from chrys.service.analytics._timeline import (
    _endpoint_in_coverage_runtime,
    _materialize_timeline,
    _resolved_timeline_operation,
    _ResolvedNode,
    _timeline_identity,
    _TimelineProjection,
)
from chrys.service.analytics.model import (
    Metric,
    Precision,
    TimelineDiagnosticCode,
    TrajectoryOverview,
    WorkflowRunAnalysis,
)
from chrys.service.analytics.reader import raise_if_cancelled


def workflow_runs(
    facts: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    integrity_reason: str | None,
    cancel_event: Event | None,
) -> tuple[WorkflowRunAnalysis, ...]:
    """Index parents once; each run owns its descendants even without a turn_id."""
    run_nodes = facts.nodes.get(WORKFLOW_RUN_FAMILY, {})
    if not run_nodes:
        return ()
    children: dict[str, list[_Node]] = defaultdict(list)
    for family in facts.nodes.values():
        for node in family.values():
            raise_if_cancelled(cancel_event)
            parents = {e.parent_operation_id for e in (*node.starts, *node.finishes)}
            for parent in parents:
                if parent is not None:
                    children[parent].append(node)

    def endpoints(node: _Node) -> tuple[list[_Endpoint], list[_Endpoint]]:
        return (
            [e for e in node.starts if _active(e.sequence, inactive_ranges)],
            [e for e in node.finishes if _active(e.sequence, inactive_ranges)],
        )

    def project(node: _Node, parent: _ResolvedNode | None) -> tuple[_TimelineProjection, _ResolvedNode | None]:
        starts, finishes = endpoints(node)
        anchor = (starts or finishes or list(node.starts) or list(node.finishes))[0]
        reason, code = integrity_reason, TimelineDiagnosticCode.INVALID_ENDPOINTS
        if not starts:
            reason, code = "Missing lifecycle start", TimelineDiagnosticCode.MISSING_START
        elif not finishes:
            reason, code = "Missing lifecycle terminal", TimelineDiagnosticCode.MISSING_TERMINAL
        elif len(starts) != 1 or len(finishes) != 1:
            reason, code = "Ambiguous lifecycle endpoints", TimelineDiagnosticCode.NONUNIQUE_LIFECYCLE
        else:
            start, finish = starts[0], finishes[0]
            valid = (
                start.scope == finish.scope
                and start.parent_operation_id == finish.parent_operation_id
                and start.sequence < finish.sequence
                and start.monotonic_ns <= finish.monotonic_ns
                and finish.monotonic_measurement
                and _endpoint_in_coverage_runtime(facts, start, inactive_ranges)
                and _endpoint_in_coverage_runtime(facts, finish, inactive_ranges)
            )
            if parent is not None:
                valid = valid and (
                    start.runtime_id == parent.start.runtime_id
                    and start.branch_id == parent.start.branch_id
                    and start.coverage_id == parent.start.coverage_id
                    and parent.start.sequence <= start.sequence < finish.sequence <= parent.finish.sequence
                    and parent.start.monotonic_ns
                    <= start.monotonic_ns
                    <= finish.monotonic_ns
                    <= parent.finish.monotonic_ns
                )
            if valid and reason is None:
                resolved = _ResolvedNode(
                    f"{node.family}:{node.operation_id}", node.family, node.operation_id, start, finish
                )
                return _resolved_timeline_operation(resolved), resolved
            reason = reason or "Lifecycle has invalid timing, coverage or parent containment"
        return _TimelineProjection(
            operation_id=node.operation_id,
            parent_operation_id=anchor.parent_operation_id,
            start_sequence=anchor.sequence,
            family=node.family,
            start_ns=None,
            end_ns=None,
            precision=Precision.UNRESOLVED,
            reason=reason,
            diagnostic_code=code,
            identity=_timeline_identity(node.family, anchor),
            hook_id=None,
        ), None

    result = []
    roots = sorted(run_nodes.values(), key=lambda n: min(e.sequence for e in (*n.starts, *n.finishes)))
    for root in roots:
        starts, finishes = endpoints(root)
        if not starts and not finishes:
            continue
        rows = []
        stack: list[tuple[_Node, _ResolvedNode | None]] = [(root, None)]
        visited: set[str] = set()
        while stack:
            raise_if_cancelled(cancel_event)
            node, parent = stack.pop()
            if node.operation_id in visited:
                continue
            visited.add(node.operation_id)
            row, resolved = project(node, parent)
            rows.append(row)
            stack.extend(
                (child, resolved)
                for child in children.get(node.operation_id, ())
                if child.family != WORKFLOW_RUN_FAMILY
            )
        operations = _materialize_timeline(rows)
        root_row = next(row for row in operations if row.operation_id == root.operation_id)
        anchor = (starts or finishes)[0]
        result.append(
            WorkflowRunAnalysis(
                run_id=root.operation_id,
                workflow_id=_payload_str(anchor.payload, "workflow") or "",
                runtime_id=anchor.runtime_id,
                outcome=_payload_str(finishes[0].payload, "outcome") if len(finishes) == 1 else None,
                elapsed_ns=Metric(root_row.duration_ns, root_row.precision, root_row.reason),
                operations=operations,
            )
        )
    return tuple(result)


def include_workflow_overview(
    overview: TrajectoryOverview, runs: tuple[WorkflowRunAnalysis, ...]
) -> TrajectoryOverview:
    metrics = [overview.elapsed_ns, *(run.elapsed_ns for run in runs)]
    elapsed = (
        Metric(sum(m.value for m in metrics if m.value is not None), Precision.EXACT)
        if all(m.precision is Precision.EXACT and m.value is not None for m in metrics)
        else Metric(None, Precision.UNRESOLVED, "One or more execution lifecycles are unresolved")
    )
    missing = Metric(None, Precision.MISSING, "Workflow scheduling and usage aggregates are not supported yet")
    return replace(
        overview,
        elapsed_ns=elapsed,
        compute_cp_ns=missing,
        response_cp_ns=missing,
        exclusive_work_ns=missing,
        parallelism=missing,
        overlap_gain_ns=missing,
        usage_tokens=missing,
        wall_time_ns=dict.fromkeys(overview.wall_time_ns, missing),
        utilization=dict.fromkeys(overview.utilization, missing),
    )
