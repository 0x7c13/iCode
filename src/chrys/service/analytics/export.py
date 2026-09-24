# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pure serializers turning a resolved trajectory analysis into export payloads.

Every function here is a transient projection: nothing is retained, so a
200 MiB session exports without touching the analyzer's residency budget.
Exports are redacted by default — file paths are replaced with stable digests
so repeated exports of the same session stay diffable without leaking the
workspace layout.
"""

from __future__ import annotations

import csv
import io
from dataclasses import replace
from hashlib import sha256

from chrys.service.analytics.model import (
    FLOW_TERMINAL_INDEX,
    ChangeVerification,
    ContextSample,
    FindingRow,
    Metric,
    SessionCounterSamples,
    TimelineOperation,
    TokenUsage,
    TokenUsageSample,
    TrajectoryAnalysis,
    TrajectoryDiagnostics,
    TrajectoryOverview,
    TurnAnalysis,
    TurnAttemptRef,
    UsageBucket,
    ValidationMetrics,
    WallBucket,
)

EXPORT_SCHEMA = "chrys.trajectory.export/1"


def _redacted(value: str) -> str:
    digest = sha256(value.encode("utf-8", "surrogatepass")).hexdigest()
    return f"redacted:{digest[:12]}"


def _redacted_detail_key(key: str, *, include_sensitive: bool) -> str:
    if include_sensitive or ("/" not in key and "\\" not in key):
        return key
    return _redacted(key)


def _metric(metric: Metric) -> dict[str, object]:
    return {"value": metric.value, "precision": metric.precision, "reason": metric.reason}


def _token_usage(usage: TokenUsage | None) -> dict[str, object] | None:
    if usage is None:
        return None
    return {bucket: _metric(metric) for bucket, metric in usage.buckets.items()}


def _overview(overview: TrajectoryOverview | None) -> dict[str, object] | None:
    if overview is None:
        return None
    return {
        "elapsed_ns": _metric(overview.elapsed_ns),
        "compute_cp_ns": _metric(overview.compute_cp_ns),
        "response_cp_ns": _metric(overview.response_cp_ns),
        "exclusive_work_ns": _metric(overview.exclusive_work_ns),
        "parallelism": _metric(overview.parallelism),
        "overlap_gain_ns": _metric(overview.overlap_gain_ns),
        "wall_time_ns": {bucket: _metric(metric) for bucket, metric in overview.wall_time_ns.items()},
        "utilization": {bucket: _metric(metric) for bucket, metric in overview.utilization.items()},
        "usage_tokens": _metric(overview.usage_tokens),
    }


def _diagnostics(diagnostics: TrajectoryDiagnostics) -> dict[str, object]:
    return {
        "line_count": diagnostics.line_count,
        "byte_count": diagnostics.byte_count,
        "torn_tail_bytes": diagnostics.torn_tail_bytes,
        "corrupt_line_count": diagnostics.corrupt_line_count,
        "corrupt_lines": [
            {
                "line_number": item.line_number,
                "byte_offset": item.byte_offset,
                "byte_length": item.byte_length,
                "reason": item.reason,
                "after_sequence": item.after_sequence,
            }
            for item in diagnostics.corrupt_lines
        ],
        "unsupported_event_count": diagnostics.unsupported_event_count,
        "unsupported_lines": [
            {
                "line_number": item.line_number,
                "byte_offset": item.byte_offset,
                "sequence": item.sequence,
                "schema_version": item.schema_version,
            }
            for item in diagnostics.unsupported_lines
        ],
        "accounted_prefix_violations": list(diagnostics.accounted_prefix_violations),
        "accounted_prefix_violation_details": [
            {
                "message": item.message,
                "first_sequence": item.first_sequence,
                "last_sequence": item.last_sequence,
            }
            for item in diagnostics.accounted_prefix_violation_details
        ],
        "explicit_gap_count": diagnostics.explicit_gap_count,
        "explicit_gaps": [
            {
                "message": item.message,
                "first_sequence": item.first_sequence,
                "last_sequence": item.last_sequence,
            }
            for item in diagnostics.explicit_gaps
        ],
        "rollback_projection_unresolved": diagnostics.rollback_projection_unresolved,
        "span_duration_mismatch_count": diagnostics.span_duration_mismatch_count,
        "span_duration_mismatches": [
            {
                "family": item.family,
                "operation_id": item.operation_id,
                "start_sequence": item.start_sequence,
                "finish_sequence": item.finish_sequence,
                "interval_ns": item.interval_ns,
                "recorded_duration_ms": item.recorded_duration_ms,
            }
            for item in diagnostics.span_duration_mismatches
        ],
        "containment_violation_count": diagnostics.containment_violation_count,
        "containment_violations": [
            {
                "family": item.family,
                "operation_id": item.operation_id,
                "parent_family": item.parent_family,
                "parent_operation_id": item.parent_operation_id,
                "start_sequence": item.start_sequence,
                "finish_sequence": item.finish_sequence,
            }
            for item in diagnostics.containment_violations
        ],
        "malformed_hook_execution_mode_count": diagnostics.malformed_hook_execution_mode_count,
        "timeline_operations": [
            {
                "turn_id": item.turn_id,
                "turn_number": item.turn_number,
                "operation_id": item.operation_id,
                "family": item.family,
                "precision": item.precision,
                "code": item.code,
                "reason": item.reason,
                "identity": item.identity,
                "hook_id": item.hook_id,
            }
            for item in diagnostics.timeline_operations
        ],
        "side_call_empty_shell_revisions": list(diagnostics.side_call_empty_shell_revisions),
        "unidentified_membership_revision_count": diagnostics.unidentified_membership_revision_count,
    }


def _validation(validation: ValidationMetrics | None) -> dict[str, object] | None:
    if validation is None:
        return None
    return {
        "funnel": {
            "search": _metric(validation.funnel.search),
            "read": _metric(validation.funnel.read),
            "edit": _metric(validation.funnel.edit),
            "verify": _metric(validation.funnel.verify),
        },
        "time_to_first_edit_ns": _metric(validation.time_to_first_edit_ns),
        "first_edit_to_first_verify_ns": _metric(validation.first_edit_to_first_verify_ns),
        "edit_verify_cycle_count": _metric(validation.edit_verify_cycle_count),
        "edit_verify_cycle_median_ns": _metric(validation.edit_verify_cycle_median_ns),
        "unverified_change_count": _metric(validation.unverified_change_count),
        "net_zero_churn_count": _metric(validation.net_zero_churn_count),
        "repeated_failure_signature_count": _metric(validation.repeated_failure_signature_count),
        "failure_recovery_median_ns": _metric(validation.failure_recovery_median_ns),
        "tool_failure_count": _metric(validation.tool_failure_count),
        "tool_count": _metric(validation.tool_count),
        "retry_amplification_tokens": _metric(validation.retry_amplification_tokens),
    }


def _change_verification(change: ChangeVerification | None, *, include_sensitive: bool) -> dict[str, object] | None:
    if change is None:
        return None
    return {
        "detail_available": change.detail_available,
        "detection_truncated": change.detection_truncated,
        "files_touched": _metric(change.files_touched),
        "created": _metric(change.created),
        "modified": _metric(change.modified),
        "deleted": _metric(change.deleted),
        "net_zero": _metric(change.net_zero),
        "rows": [
            {
                "path": row.path if include_sensitive else _redacted(row.path),
                "state": row.state,
                "last_change_turn": row.last_change_turn,
                "precision": row.precision,
                "evidence": list(row.evidence),
            }
            for row in change.rows
        ],
    }


def _operation(operation: TimelineOperation) -> dict[str, object]:
    return {
        "operation_id": operation.operation_id,
        "family": operation.family,
        "depth": operation.depth,
        "start_ns": operation.start_ns,
        "end_ns": operation.end_ns,
        "precision": operation.precision,
        "reason": operation.reason,
        "diagnostic_code": operation.diagnostic_code,
        "identity": operation.identity,
        "hook_id": operation.hook_id,
    }


def _usage_sample(sample: TokenUsageSample) -> dict[str, object]:
    return {
        "sequence": sample.sequence,
        "end_ns": sample.end_ns,
        "input_tokens": sample.input_tokens,
        "output_tokens": sample.output_tokens,
        "reasoning_tokens": sample.reasoning_tokens,
        "cache_read_tokens": sample.cache_read_tokens,
        "cache_creation_tokens": sample.cache_creation_tokens,
    }


def _context_sample(sample: ContextSample) -> dict[str, object]:
    return {"sequence": sample.sequence, "ns": sample.ns, "item_count": sample.item_count}


def _attempt_payload(attempt: TurnAttemptRef) -> dict[str, object]:
    return {
        "turn_id": attempt.turn_id,
        "runtime_id": attempt.runtime_id,
        "is_retry": attempt.is_retry,
        "physical_axis_start_ns": attempt.physical_axis_start_ns,
        "physical_axis_end_ns": attempt.physical_axis_end_ns,
        "logical_axis_start_ns": attempt.logical_axis_start_ns,
        "operation_start_index": attempt.operation_start_index,
        "operation_end_index": attempt.operation_end_index,
        "slice_start_index": attempt.slice_start_index,
        "slice_end_index": attempt.slice_end_index,
    }


def _logical_usage_sample(sample: TokenUsageSample, attempt: TurnAttemptRef) -> dict[str, object]:
    """Project a physical usage timestamp onto its turn's stitched logical axis."""
    payload = _usage_sample(sample)
    if sample.end_ns is not None:
        payload["end_ns"] = (
            attempt.to_logical_ns(sample.end_ns) if attempt.contains_physical_ns(sample.end_ns) else None
        )
    return payload


def _logical_context_sample(sample: ContextSample, attempt: TurnAttemptRef) -> dict[str, object]:
    """Project a physical context timestamp onto its turn's stitched logical axis."""
    payload = _context_sample(sample)
    payload["ns"] = attempt.to_logical_ns(sample.ns) if attempt.contains_physical_ns(sample.ns) else None
    return payload


def _turn_payload(turn: TurnAnalysis, samples: SessionCounterSamples) -> dict[str, object]:
    flow = turn.flow
    usage_samples = sorted(
        ((sample, attempt) for attempt in turn.attempts for sample in samples.usage_by_turn.get(attempt.turn_id, ())),
        key=lambda item: item[0].sequence,
    )
    context_samples = sorted(
        ((sample, attempt) for attempt in turn.attempts for sample in samples.context_by_turn.get(attempt.turn_id, ())),
        key=lambda item: item[0].sequence,
    )
    return {
        "turn_id": turn.turn_id,
        "attempts": [_attempt_payload(attempt) for attempt in turn.attempts],
        "turn_number": turn.turn_number,
        "start_sequence": turn.start_sequence,
        "end_sequence": turn.end_sequence,
        "axis_start_ns": turn.axis_start_ns,
        "axis_end_ns": turn.axis_end_ns,
        "metrics": {
            "elapsed_ns": _metric(turn.elapsed_ns),
            "compute_cp_ns": _metric(turn.compute_cp_ns),
            "response_cp_ns": _metric(turn.response_cp_ns),
            "exclusive_work_ns": _metric(turn.exclusive_work_ns),
            "parallelism": _metric(turn.parallelism),
            "overlap_gain_ns": _metric(turn.overlap_gain_ns),
            "wall_time_ns": {bucket: _metric(metric) for bucket, metric in turn.wall_time_ns.items()},
            "utilization": {bucket: _metric(metric) for bucket, metric in turn.utilization.items()},
            "usage_tokens": _metric(turn.usage_tokens),
        },
        "token_usage": _token_usage(turn.token_usage),
        "diagnostics": list(turn.diagnostics),
        "operations": [_operation(operation) for operation in turn.operations],
        "flow": None
        if flow is None
        else {
            "root_index": flow.root_index,
            "has_terminal": flow.has_terminal,
            "acyclic": flow.acyclic,
            "terminal_index": FLOW_TERMINAL_INDEX,
            "parent_edges": [list(edge) for edge in flow.parent_edges()],
            "causal_edges": [list(edge) for edge in flow.causal_edges()],
        },
        "usage_samples": [_logical_usage_sample(sample, attempt) for sample, attempt in usage_samples],
        "context_samples": [_logical_context_sample(sample, attempt) for sample, attempt in context_samples],
    }


def _finding(row: FindingRow, *, include_sensitive: bool) -> dict[str, object]:
    return {
        "evidence_key": row.evidence_key,
        "occurrence_id": row.occurrence_id,
        "rule_id": row.rule_id,
        "severity": row.severity,
        "deterministic": row.deterministic,
        "precision": row.precision,
        "turn_number": row.turn_number,
        "detail_args": [
            [_redacted_detail_key(key, include_sensitive=include_sensitive), value] for key, value in row.detail_args
        ],
    }


def analysis_json(
    analysis: TrajectoryAnalysis,
    samples: SessionCounterSamples,
    *,
    include_sensitive: bool = False,
) -> dict[str, object]:
    """Project aggregate results onto the stitched logical turn axes.

    Operation and counter timestamps share each logical turn's gap-free axis.
    A counter outside its owning physical attempt keeps its values but receives
    a null timestamp and a turn diagnostic instead of an inferred placement.
    Perfetto deliberately uses each runtime's physical axis instead.
    """
    path_text = str(analysis.path)
    return {
        "schema": EXPORT_SCHEMA,
        "session": {
            "availability": analysis.availability,
            "generation": analysis.generation,
            "path": path_text if include_sensitive else _redacted(path_text),
        },
        "diagnostics": _diagnostics(analysis.diagnostics),
        "overview": _overview(analysis.overview),
        "token_usage": _token_usage(analysis.token_usage),
        "validation": _validation(analysis.validation),
        "change_verification": _change_verification(analysis.change_verification, include_sensitive=include_sensitive),
        "turns": [_turn_payload(turn, samples) for turn in analysis.turns],
        "workflow_runs": [
            {
                "run_id": run.run_id,
                "workflow_id": run.workflow_id,
                "runtime_id": run.runtime_id,
                "outcome": run.outcome,
                "elapsed_ns": _metric(run.elapsed_ns),
                "operations": [_operation(operation) for operation in run.operations],
            }
            for run in analysis.workflow_runs
        ],
        "findings": [_finding(row, include_sensitive=include_sensitive) for row in analysis.findings],
    }


def turns_csv(analysis: TrajectoryAnalysis) -> str:
    """Render one spreadsheet-friendly row per resolved turn, values in raw ns."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(
        [
            "turn_number",
            "turn_id",
            "start_sequence",
            "elapsed_ns",
            "compute_cp_ns",
            "response_cp_ns",
            "exclusive_work_ns",
            "parallelism",
            "overlap_gain_ns",
            *(f"wall_{bucket}_ns" for bucket in WallBucket),
            *(f"tokens_{bucket}" for bucket in UsageBucket),
            "precision",
            "compute_cp_precision",
            "response_cp_precision",
            "exclusive_work_precision",
            "parallelism_precision",
            "overlap_gain_precision",
            *(f"wall_{bucket}_precision" for bucket in WallBucket),
            *(f"tokens_{bucket}_precision" for bucket in UsageBucket),
        ]
    )
    for turn in analysis.turns:
        token_buckets = turn.token_usage.buckets if turn.token_usage is not None else {}
        writer.writerow(
            [
                turn.turn_number,
                turn.turn_id,
                turn.start_sequence,
                turn.elapsed_ns.value,
                turn.compute_cp_ns.value,
                turn.response_cp_ns.value,
                turn.exclusive_work_ns.value,
                turn.parallelism.value,
                turn.overlap_gain_ns.value,
                *(
                    metric.value if (metric := turn.wall_time_ns.get(bucket)) is not None else None
                    for bucket in WallBucket
                ),
                *(
                    metric.value if (metric := token_buckets.get(bucket)) is not None else None
                    for bucket in UsageBucket
                ),
                turn.elapsed_ns.precision,
                turn.compute_cp_ns.precision,
                turn.response_cp_ns.precision,
                turn.exclusive_work_ns.precision,
                turn.parallelism.precision,
                turn.overlap_gain_ns.precision,
                *(
                    metric.precision if (metric := turn.wall_time_ns.get(bucket)) is not None else None
                    for bucket in WallBucket
                ),
                *(
                    metric.precision if (metric := token_buckets.get(bucket)) is not None else None
                    for bucket in UsageBucket
                ),
            ]
        )
    return out.getvalue()


def findings_csv(analysis: TrajectoryAnalysis, *, include_sensitive: bool = False) -> str:
    """Render one row per finding for CI-style regression tracking."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(
        [
            "rule_id",
            "severity",
            "turn_number",
            "evidence_key",
            "occurrence_id",
            "deterministic",
            "precision",
            "details",
        ]
    )
    for row in analysis.findings:
        details = ";".join(
            f"{_redacted_detail_key(key, include_sensitive=include_sensitive)}={value}"
            for key, value in row.detail_args
        )
        writer.writerow(
            [
                row.rule_id,
                row.severity,
                row.turn_number,
                row.evidence_key,
                row.occurrence_id,
                row.deterministic,
                row.precision,
                details,
            ]
        )
    return out.getvalue()


def _packed_lanes(operations: tuple[TimelineOperation, ...]) -> tuple[dict[int, int], int]:
    """Assign overlap-free lanes so every slice imports losslessly."""
    placeable = sorted(
        (
            (index, operation.start_ns, operation.end_ns)
            for index, operation in enumerate(operations)
            if operation.start_ns is not None and operation.end_ns is not None
        ),
        key=lambda item: (item[1], item[2], item[0]),
    )
    lane_ends: list[int] = []
    lane_by_index: dict[int, int] = {}
    for index, start_ns, end_ns in placeable:
        for lane, end in enumerate(lane_ends):
            if end <= start_ns:
                lane_ends[lane] = end_ns
                lane_by_index[index] = lane
                break
        else:
            lane_by_index[index] = len(lane_ends)
            lane_ends.append(end_ns)
    return lane_by_index, max(1, len(lane_ends))


def perfetto_trace(
    analysis: TrajectoryAnalysis,
    samples: SessionCounterSamples,
    *,
    include_sensitive: bool = False,
) -> dict[str, object]:
    """Project operations, counters, and causal flow into a Chrome JSON trace.

    Operations become complete slices on overlap-free per-turn lanes, token and
    context samples plus swept work concurrency become counter tracks, and only
    producer-declared causal edges become flow arrows — sequence adjacency is
    never drawn as causality. Each runtime is its own trace process with its
    own time origin, because monotonic timestamps are only comparable within
    one process run — a resumed session must not interleave unrelated clocks
    on a shared axis.
    """
    events: list[dict[str, object]] = []
    turns = analysis.turns
    runtime_order: list[str] = []
    for turn in turns:
        for attempt in turn.attempts:
            if attempt.runtime_id not in runtime_order:
                runtime_order.append(attempt.runtime_id)
    for run in analysis.workflow_runs:
        if run.runtime_id not in runtime_order:
            runtime_order.append(run.runtime_id)
    pid_by_runtime = {runtime: index + 1 for index, runtime in enumerate(runtime_order)}
    attempt_by_turn_id = {attempt.turn_id: attempt for turn in turns for attempt in turn.attempts}
    origin_values: dict[str, list[int]] = {runtime: [] for runtime in runtime_order}
    for turn in turns:
        for attempt in turn.attempts:
            origin_values[attempt.runtime_id].append(attempt.physical_axis_start_ns)
    for turn_id, usage_rows in samples.usage_by_turn.items():
        attempt = attempt_by_turn_id.get(turn_id)
        if attempt is None:
            continue
        origin_values[attempt.runtime_id].extend(
            sample.end_ns
            for sample in usage_rows
            if sample.end_ns is not None and attempt.contains_physical_ns(sample.end_ns)
        )
    for turn_id, context_rows in samples.context_by_turn.items():
        attempt = attempt_by_turn_id.get(turn_id)
        if attempt is None:
            continue
        origin_values[attempt.runtime_id].extend(
            sample.ns for sample in context_rows if attempt.contains_physical_ns(sample.ns)
        )
    for run in analysis.workflow_runs:
        origin_values[run.runtime_id].extend(
            operation.start_ns for operation in run.operations if operation.start_ns is not None
        )
    origin_by_runtime = {runtime: min(values, default=0) for runtime, values in origin_values.items()}

    def us(ns: int, runtime: str) -> float:
        return (ns - origin_by_runtime[runtime]) / 1000.0

    path_text = str(analysis.path)
    session_label = path_text if include_sensitive else _redacted(path_text)
    for position, runtime in enumerate(runtime_order):
        name = f"chrys trajectory {session_label}"
        if len(runtime_order) > 1:
            name = f"{name} · run {position + 1} ({runtime[:8]})"
        events.append({"ph": "M", "pid": pid_by_runtime[runtime], "name": "process_name", "args": {"name": name}})
    flow_id = 0
    next_tid_by_pid = dict.fromkeys(pid_by_runtime.values(), 1)
    for turn in turns:
        turn_label = f"turn {turn.turn_number}" if turn.turn_number is not None else f"turn {turn.turn_id[:8]}"
        anchors: dict[int, tuple[int, int, float, float]] = {}
        base_tid_by_runtime: dict[str, int] = {}
        # One logical turn is one track per runtime, not one per attempt.
        # Attempts sharing a runtime share its clock, so their slices lane-pack
        # together and the retry reads as a gap; a separate track per attempt
        # would put two identically named "turn N" lanes in one process.
        attempts_by_runtime: dict[str, list[TurnAttemptRef]] = {}
        for attempt in turn.attempts:
            attempts_by_runtime.setdefault(attempt.runtime_id, []).append(attempt)
        for runtime, attempts in attempts_by_runtime.items():
            pid = pid_by_runtime[runtime]
            indexed_operations = [
                (operation_index, operation)
                for attempt in attempts
                for operation_index, operation in enumerate(
                    turn.operations[attempt.operation_start_index : attempt.operation_end_index],
                    start=attempt.operation_start_index,
                )
            ]
            physical_operations = tuple(
                replace(
                    operation,
                    start_ns=(attempt.to_physical_ns(operation.start_ns) if operation.start_ns is not None else None),
                    end_ns=attempt.to_physical_ns(operation.end_ns) if operation.end_ns is not None else None,
                )
                for attempt in attempts
                for operation in turn.operations[attempt.operation_start_index : attempt.operation_end_index]
            )
            lane_by_index, lane_count = _packed_lanes(physical_operations)
            base_tid = next_tid_by_pid[pid]
            base_tid_by_runtime[runtime] = base_tid
            next_tid_by_pid[pid] = base_tid + lane_count
            for lane in range(lane_count):
                tid = base_tid + lane
                name = turn_label if lane == 0 else f"{turn_label} +{lane}"
                events.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name", "args": {"name": name}})
                events.append(
                    {
                        "ph": "M",
                        "pid": pid,
                        "tid": tid,
                        "name": "thread_sort_index",
                        "args": {"sort_index": tid},
                    }
                )
            for local_index, ((operation_index, operation), physical) in enumerate(
                zip(indexed_operations, physical_operations, strict=True)
            ):
                lane = lane_by_index.get(local_index)
                if lane is None or physical.start_ns is None or physical.end_ns is None:
                    continue
                tid = base_tid + lane
                ts = us(physical.start_ns, runtime)
                anchors[operation_index] = (pid, tid, ts, us(physical.end_ns, runtime))
                events.append(
                    {
                        "ph": "X",
                        "pid": pid,
                        "tid": tid,
                        "ts": ts,
                        "dur": (physical.end_ns - physical.start_ns) / 1000.0,
                        "name": operation.identity or operation.family,
                        "cat": operation.family,
                        "args": {"operation_id": operation.operation_id, "precision": operation.precision},
                    }
                )
        flow = turn.flow
        if flow is None:
            continue
        terminal_anchor: tuple[int, int, float, float] | None = None
        if flow.has_terminal:
            terminal_attempt = turn.attempts[-1]
            runtime = terminal_attempt.runtime_id
            pid = pid_by_runtime[runtime]
            tid = base_tid_by_runtime[runtime]
            terminal_ts = us(terminal_attempt.physical_axis_end_ns, runtime)
            terminal_anchor = (pid, tid, terminal_ts, terminal_ts)
            events.append(
                {
                    "ph": "X",
                    "pid": pid,
                    "tid": tid,
                    "ts": terminal_ts,
                    "dur": 0.0,
                    "name": "response",
                    "cat": "turn.response",
                    "args": {},
                }
            )
        for source, target in flow.causal_edges():
            source_anchor = anchors.get(source)
            target_anchor = terminal_anchor if target == FLOW_TERMINAL_INDEX else anchors.get(target)
            if source_anchor is None or target_anchor is None:
                continue
            if source_anchor[0] == target_anchor[0] and source_anchor[3] > target_anchor[2]:
                continue
            flow_id += 1
            events.append(
                {
                    "ph": "s",
                    "id": flow_id,
                    "pid": source_anchor[0],
                    "tid": source_anchor[1],
                    "ts": source_anchor[3],
                    "name": "causal",
                    "cat": "flow",
                }
            )
            events.append(
                {
                    "ph": "f",
                    "bp": "e",
                    "id": flow_id,
                    "pid": target_anchor[0],
                    "tid": target_anchor[1],
                    "ts": target_anchor[2],
                    "name": "causal",
                    "cat": "flow",
                }
            )
    for turn_id, usage_rows in samples.usage_by_turn.items():
        attempt = attempt_by_turn_id.get(turn_id)
        if attempt is None:
            continue
        runtime = attempt.runtime_id
        for sample in usage_rows:
            if sample.end_ns is None or not attempt.contains_physical_ns(sample.end_ns):
                continue
            args = {
                key: value
                for key, value in (
                    ("input", sample.input_tokens),
                    ("output", sample.output_tokens),
                    ("reasoning", sample.reasoning_tokens),
                    ("cache_read", sample.cache_read_tokens),
                    ("cache_creation", sample.cache_creation_tokens),
                )
                if value is not None
            }
            events.append(
                {
                    "ph": "C",
                    "pid": pid_by_runtime[runtime],
                    "ts": us(sample.end_ns, runtime),
                    "name": "tokens",
                    "args": args,
                }
            )
    for turn_id, context_rows in samples.context_by_turn.items():
        attempt = attempt_by_turn_id.get(turn_id)
        if attempt is None:
            continue
        runtime = attempt.runtime_id
        events.extend(
            {
                "ph": "C",
                "pid": pid_by_runtime[runtime],
                "ts": us(sample.ns, runtime),
                "name": "context items",
                "args": {"items": sample.item_count},
            }
            for sample in context_rows
            if attempt.contains_physical_ns(sample.ns)
        )
    # Work concurrency is swept per runtime: overlap between slices of
    # different runtimes is a clock artifact, never real parallelism.
    boundaries_by_runtime: dict[str, list[tuple[int, int]]] = {runtime: [] for runtime in runtime_order}
    for turn in turns:
        for attempt in turn.attempts:
            boundaries = boundaries_by_runtime[attempt.runtime_id]
            for item in turn.slices[attempt.slice_start_index : attempt.slice_end_index]:
                if item.counts_as_work:
                    boundaries.append((attempt.to_physical_ns(item.start_ns), 1))
                    boundaries.append((attempt.to_physical_ns(item.end_ns), -1))
    for runtime, boundaries in boundaries_by_runtime.items():
        boundaries.sort()
        running = 0
        for position, (ns, delta) in enumerate(boundaries):
            running += delta
            if position + 1 < len(boundaries) and boundaries[position + 1][0] == ns:
                continue
            events.append(
                {
                    "ph": "C",
                    "pid": pid_by_runtime[runtime],
                    "ts": us(ns, runtime),
                    "name": "active work",
                    "args": {"count": running},
                }
            )
    for index, run in enumerate(analysis.workflow_runs, 1):
        pid = pid_by_runtime[run.runtime_id]
        base_tid = next_tid_by_pid[pid]
        lanes, count = _packed_lanes(run.operations)
        next_tid_by_pid[pid] += count
        events.extend(
            {
                "ph": "M",
                "pid": pid,
                "tid": base_tid + lane,
                "name": "thread_name",
                "args": {"name": f"{run.workflow_id} · Run {index}"},
            }
            for lane in range(count)
        )
        for position, operation in enumerate(run.operations):
            lane = lanes.get(position)
            if lane is None or operation.start_ns is None or operation.end_ns is None:
                continue
            events.append(
                {
                    "ph": "X",
                    "pid": pid,
                    "tid": base_tid + lane,
                    "ts": us(operation.start_ns, run.runtime_id),
                    "dur": (operation.end_ns - operation.start_ns) / 1000.0,
                    "name": operation.identity or operation.family,
                    "cat": operation.family,
                    "args": {
                        "operation_id": operation.operation_id,
                        "precision": operation.precision,
                        "run_id": run.run_id,
                    },
                }
            )
    return {"traceEvents": events, "displayTimeUnit": "ms"}
