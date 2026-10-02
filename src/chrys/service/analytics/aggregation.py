# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Three-stage trajectory aggregation for the P0 dashboard.

The loader performs one physical JSONL scan into compact, reversible facts.
Resolution happens only after the scan reaches EOF (or the complete live-tail
prefix), so rollback projection and forward references are settled before any
metric is published.
"""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import replace
from io import BufferedReader
from os import fstat
from pathlib import Path
from statistics import median
from threading import Event
from typing import Final, cast

import chrys.service.analytics._turns as _turns
from chrys.foundation.config.settings import DEFAULT_TRAJECTORY_VERIFY_COMMANDS
from chrys.foundation.platform.files import secure_open_owner_only_binary
from chrys.foundation.tool_kinds import TOOL_KINDS
from chrys.foundation.trajectory.event_types import SourceRefKind, ToolOutcome
from chrys.foundation.trajectory.fingerprint import FINGERPRINT_KEY_BYTES
from chrys.foundation.trajectory.ids import is_valid_analytics_id
from chrys.foundation.trajectory.keys import TRAJECTORY_KEY_FILE_NAME
from chrys.service.analytics._facts import (
    _active,
    _closed_sequence_union,
    _Endpoint,
    _EventScope,
    _Intermediate,
    _lifecycle_cut,
    _LifecycleCut,
    _Node,
    _payload_int,
    _payload_str,
    _payload_value,
    _projection_membership,
    _ToolPayloadExtras,
)
from chrys.service.analytics._session_projection import (
    _lazy_session_projection,
    _SessionProjection,
    _SessionProjectionCache,
)
from chrys.service.analytics._workflows import include_workflow_overview, workflow_runs
from chrys.service.analytics.classification import (
    classification_evidence_key,
    classify_action,
    parse_verify_commands,
)
from chrys.service.analytics.findings import evaluate_findings
from chrys.service.analytics.model import (
    FLOW_TERMINAL_INDEX,
    ActionClass,
    ActionFunnel,
    ActionOperation,
    AnalysisAvailability,
    ChangeVerification,
    ChangeVerificationRow,
    ChangeVerificationState,
    ContextCarryingLoad,
    InsightsAnalysis,
    McpInsights,
    McpRemoteRow,
    McpServerRow,
    Metric,
    NamedCountRow,
    Precision,
    SequenceRangeDiagnostic,
    SessionCounterSamples,
    SessionSpan,
    SkillActivityRow,
    SkillInsightRow,
    SkillInsights,
    SubmissionLatencyBucket,
    SubmissionLatencyOverview,
    SubmissionLatencySample,
    SubmissionLatencyStats,
    TimelineOperation,
    TimelineOperationDiagnostic,
    TimeSlice,
    TokenUsage,
    ToolInsightRow,
    ToolInsights,
    ToolUsagePanel,
    TrajectoryAnalysis,
    TrajectoryDiagnostics,
    TrajectoryOverview,
    TurnAnalysis,
    TurnAttemptRef,
    TurnFlow,
    UsageBucket,
    ValidationMetrics,
    WallBucket,
    least_precision,
    verify_covers_edit,
)
from chrys.service.analytics.reader import (
    ScanCursor,
    TrajectoryScanCancelled,
    _PrefixHasher,
    scan_open_trajectory_batch,
    verify_prefix,
)
from chrys.service.analytics.reader import (
    raise_if_cancelled as _check_cancelled,
)
from chrys.service.mutations.types import FileHashDiff, parse_skip_reason
from chrys.service.trajectory.preparation import PreparationOutcome

MAX_RESIDENT_MEMORY_BYTES: Final = 200 * 1024 * 1024
"""P0 acceptance ceiling for a 200 MiB input fixture."""

_WORD_LIST_HEURISTIC_REASON: Final = "one or more shell actions use the verification word-list heuristic"

_PREFIX_REVERIFY_FLOOR_BYTES: Final = 1 << 20
"""Smallest growth that triggers a full prefix replay between doublings."""

_PREFIX_PROBE_BYTES: Final = 1 << 12
"""Consumed-suffix window reread before trusting growth as an append."""


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


def _all_side_call_lifecycle(node: _Node) -> bool:
    """Whether every raw endpoint belongs to a non-main actor."""
    endpoints = (*node.starts, *node.finishes)
    return bool(endpoints) and all(endpoint.side_call for endpoint in endpoints)


def _read_prefix_probe(handle: BufferedReader, byte_offset: int) -> bytes:
    length = min(byte_offset, _PREFIX_PROBE_BYTES)
    handle.seek(byte_offset - length)
    return handle.read(length)


class TrajectoryAnalyzer:
    """Path-identity cache with append-only live-tail scans and dirty-turn resolves."""

    def __init__(
        self,
        *,
        fingerprint_key: bytes | None = None,
        verify_commands: str = DEFAULT_TRAJECTORY_VERIFY_COMMANDS,
    ) -> None:
        self._path: Path | None = None
        self._identity: tuple[int, int] | None = None
        self._size = 0
        self._mtime_ns = 0
        self._prefix_digest = b""
        self._prefix_hasher: _PrefixHasher | None = None
        self._prefix_probe = b""
        self._verified_offset = 0
        self._cursor = ScanCursor()
        self._intermediate: _Intermediate | None = None
        self._resolution_cache = _turns._ResolutionCache()
        self._analysis: TrajectoryAnalysis | None = None
        self._generation = 0
        self._fingerprint_key = fingerprint_key
        self._verify_commands = verify_commands
        self._session_projection_cache = _SessionProjectionCache()

    def load(self, path: Path, *, cancel_event: Event | None = None) -> TrajectoryAnalysis:
        """Perform the one initial physical scan and resolve after EOF."""
        self._path = path
        self._generation += 1
        try:
            handle = path.open("rb")
        except FileNotFoundError:
            self._reset_cache()
            self._path = path
            self._analysis = TrajectoryAnalysis(
                availability=AnalysisAvailability.UNAVAILABLE,
                path=path,
                generation=self._generation,
            )
            return self._analysis
        except OSError as exc:
            return self._read_error(path, exc)
        with handle:
            return self._load_opened(path, handle, cancel_event=cancel_event)

    def _load_opened(
        self,
        path: Path,
        handle: BufferedReader,
        *,
        cancel_event: Event | None,
    ) -> TrajectoryAnalysis:
        intermediate = _Intermediate(path)
        cursor = ScanCursor()
        try:
            batch = scan_open_trajectory_batch(
                handle,
                intermediate.consume,
                cursor=cursor,
                cancel_event=cancel_event,
            )
        except TrajectoryScanCancelled:
            self._reset_cache()
            raise
        except OSError as exc:
            return self._read_error(path, exc)
        if cancel_event is not None and cancel_event.is_set():
            self._reset_cache()
            raise TrajectoryScanCancelled
        intermediate.absorb_batch(batch)
        stat = fstat(handle.fileno())
        self._cursor = cursor
        self._intermediate = intermediate
        self._identity = (stat.st_dev, stat.st_ino)
        self._size = cursor.byte_offset + batch.torn_tail_bytes
        self._mtime_ns = stat.st_mtime_ns
        self._prefix_digest = batch.prefix_digest
        self._prefix_hasher = batch.prefix_hasher
        self._prefix_probe = _read_prefix_probe(handle, cursor.byte_offset)
        self._verified_offset = cursor.byte_offset
        self._resolution_cache = _turns._ResolutionCache()
        try:
            self._analysis = _resolve(
                intermediate,
                generation=self._generation,
                resolution_cache=self._resolution_cache,
                cancel_event=cancel_event,
                fingerprint_key=self._fingerprint_key or _read_installed_fingerprint_key(path),
                verify_commands=self._verify_commands,
                session_cache=self._session_projection_cache,
            )
        except TrajectoryScanCancelled:
            self._reset_cache()
            raise
        intermediate.mark_resolved()
        return self._analysis

    def refresh(self, *, cancel_event: Event | None = None) -> TrajectoryAnalysis:
        """Consume one append batch; replacement or truncation performs a fresh load."""
        path = self._path
        if path is None:
            raise RuntimeError("TrajectoryAnalyzer.refresh() requires load() first")
        try:
            handle = path.open("rb")
        except OSError:
            return self.load(path, cancel_event=cancel_event)
        with handle:
            stat = fstat(handle.fileno())
            identity = (stat.st_dev, stat.st_ino)
            intermediate = self._intermediate
            self._generation += 1
            if intermediate is None or identity != self._identity or stat.st_size < self._cursor.byte_offset:
                return self._load_opened(path, handle, cancel_event=cancel_event)
            if stat.st_size == self._size:
                # An unchanged log still goes stale when the companion
                # session document moved under it: carriers, commands, and
                # mutation detail all fold from that document.
                if (
                    stat.st_mtime_ns == self._mtime_ns
                    and self._analysis is not None
                    and not self._session_projection_cache.changed(path)
                ):
                    self._generation -= 1
                    return self._analysis
                return self._load_opened(path, handle, cancel_event=cancel_event)
            retained = self._prefix_hasher
            appended_since_verify = self._cursor.byte_offset - self._verified_offset
            # Replaying the whole consumed prefix on every append batch is
            # quadratic against a log polled twice a second, so the retained
            # digest state continues across batches and the replay re-verifies
            # only once the log outgrows the last verified length again:
            # amortized linear, still fail-closed at every doubling and load.
            if retained is not None and appended_since_verify < max(
                self._verified_offset, _PREFIX_REVERIFY_FLOOR_BYTES
            ):
                # Growth alone is not append provenance: a same-inode
                # replacement also grows the file. Rereading the consumed
                # suffix catches a replaced prefix at the boundary before any
                # stale state is combined with the new bytes; a replacement
                # that forges the probe window still fails the next replay.
                if _read_prefix_probe(handle, self._cursor.byte_offset) != self._prefix_probe:
                    return self._load_opened(path, handle, cancel_event=cancel_event)
                prefix_hasher = retained
            else:
                try:
                    prefix_hasher = verify_prefix(
                        handle,
                        length=self._cursor.byte_offset,
                        expected_digest=self._prefix_digest,
                        cancel_event=cancel_event,
                    )
                except TrajectoryScanCancelled:
                    self._reset_cache()
                    raise
                if prefix_hasher is None:
                    return self._load_opened(path, handle, cancel_event=cancel_event)
                self._verified_offset = self._cursor.byte_offset
            try:
                batch = scan_open_trajectory_batch(
                    handle,
                    intermediate.consume,
                    cursor=self._cursor,
                    cancel_event=cancel_event,
                    prefix_hasher=prefix_hasher,
                )
            except TrajectoryScanCancelled:
                self._reset_cache()
                raise
            except OSError as exc:
                return self._read_error(path, exc)
            if cancel_event is not None and cancel_event.is_set():
                self._reset_cache()
                raise TrajectoryScanCancelled
            intermediate.absorb_batch(batch)
            final_stat = fstat(handle.fileno())
            self._size = self._cursor.byte_offset + batch.torn_tail_bytes
            self._mtime_ns = final_stat.st_mtime_ns
            self._prefix_digest = batch.prefix_digest
            self._prefix_hasher = batch.prefix_hasher
            self._prefix_probe = _read_prefix_probe(handle, self._cursor.byte_offset)
            try:
                self._analysis = _resolve(
                    intermediate,
                    generation=self._generation,
                    resolution_cache=self._resolution_cache,
                    previous=self._analysis,
                    cancel_event=cancel_event,
                    fingerprint_key=self._fingerprint_key or _read_installed_fingerprint_key(path),
                    verify_commands=self._verify_commands,
                    session_cache=self._session_projection_cache,
                )
            except TrajectoryScanCancelled:
                self._reset_cache()
                raise
            intermediate.mark_resolved()
            return self._analysis

    def release(self) -> None:
        """Release all generation-bound state owned by the analyzer."""
        self._path = None
        self._analysis = None
        self._session_projection_cache = _SessionProjectionCache()
        self._reset_cache()

    def counter_samples(self) -> SessionCounterSamples:
        """Project per-turn counter samples from the retained scan facts.

        Computed transiently for export; retaining timestamped samples on every
        TurnAnalysis would scale the resident footprint with session length.
        """
        intermediate = self._intermediate
        if intermediate is None:
            return SessionCounterSamples(usage_by_turn={}, context_by_turn={})
        inactive_ranges = _closed_sequence_union(intermediate.rollback_ranges)
        return SessionCounterSamples(
            usage_by_turn=_turns._usage_samples_by_turn(intermediate, inactive_ranges),
            context_by_turn=_turns._context_samples_by_turn(intermediate, inactive_ranges),
        )

    def _read_error(self, path: Path, exc: OSError) -> TrajectoryAnalysis:
        self._reset_cache()
        self._path = path
        self._analysis = TrajectoryAnalysis(
            availability=AnalysisAvailability.READ_ERROR,
            path=path,
            generation=self._generation,
            read_error=str(exc),
        )
        return self._analysis

    def _reset_cache(self) -> None:
        self._identity = None
        self._size = 0
        self._mtime_ns = 0
        self._prefix_digest = b""
        # A scan abandoned mid-batch leaves the retained digest state partially
        # updated, so every reset path must also drop it.
        self._prefix_hasher = None
        self._prefix_probe = b""
        self._verified_offset = 0
        self._cursor = ScanCursor()
        self._intermediate = None
        self._resolution_cache = _turns._ResolutionCache()


def analyze_trajectory(
    path: Path,
    *,
    fingerprint_key: bytes | None = None,
    verify_commands: str = DEFAULT_TRAJECTORY_VERIFY_COMMANDS,
) -> TrajectoryAnalysis:
    """Analyze *path* without retaining a live-tail cache."""
    return TrajectoryAnalyzer(fingerprint_key=fingerprint_key, verify_commands=verify_commands).load(path)


def _resolve(
    intermediate: _Intermediate,
    *,
    generation: int,
    resolution_cache: _turns._ResolutionCache,
    previous: TrajectoryAnalysis | None = None,
    cancel_event: Event | None = None,
    fingerprint_key: bytes | None = None,
    session_cache: _SessionProjectionCache | None = None,
    verify_commands: str = DEFAULT_TRAJECTORY_VERIFY_COMMANDS,
) -> TrajectoryAnalysis:
    _check_cancelled(cancel_event)
    active_ranges = _closed_sequence_union(intermediate.rollback_ranges)
    rollback_projection_unresolved = _rollback_projection_unresolved(intermediate, active_ranges)
    session_integrity_reason = _session_integrity_reason(
        intermediate,
        rollback_projection_unresolved=rollback_projection_unresolved,
    )
    active_turn_ids: set[str] = set()
    for turn_id, turn in intermediate.turns.items():
        _check_cancelled(cancel_event)
        if any(_active(event.sequence, active_ranges) for event in turn.starts):
            active_turn_ids.add(turn_id)
    revisions = _turns._revision_memberships(
        intermediate,
        active_ranges,
        fingerprint_key=fingerprint_key,
        cancel_event=cancel_event,
    )
    session_projection = (
        session_cache.lazy(intermediate.path, cancel_event=cancel_event)
        if session_cache is not None
        else _lazy_session_projection(intermediate.path, cancel_event=cancel_event)
    )

    def mapped_carrier(item_id: str) -> str | None:
        return session_projection().carriers.get(item_id)

    # A folded logical turn is not a valid cache entry for any one of its
    # physical attempts. Re-resolve those attempts and fold them again; regular
    # one-attempt turns retain the live-tail fast path.
    previous_turns = (
        {turn.turn_id: turn for turn in previous.turns if len(turn.attempts) == 1} if previous is not None else {}
    )
    session_carrier_turn_ids = {
        turn_id
        for turn_id in active_turn_ids
        if _turns._turn_uses_session_carrier_fallback(
            intermediate,
            turn_id,
            active_ranges,
            cancel_event=cancel_event,
        )
    }
    resolved_turns: list[TurnAnalysis] = []
    for turn_id in active_turn_ids:
        _check_cancelled(cancel_event)
        if (
            not intermediate.dirty.full
            and turn_id not in intermediate.dirty.turn_ids
            and turn_id not in session_carrier_turn_ids
            and turn_id in previous_turns
        ):
            resolved_turns.append(previous_turns[turn_id])
        else:
            resolved_turns.append(
                _turns._resolve_turn(
                    intermediate,
                    intermediate.turns[turn_id],
                    active_ranges,
                    resolution_cache=resolution_cache,
                    revisions=revisions,
                    mapped_carrier=mapped_carrier,
                    rollback_projection_unresolved=rollback_projection_unresolved,
                    refresh_counter_axis=intermediate.dirty.full or turn_id in intermediate.dirty.turn_ids,
                    cancel_event=cancel_event,
                )
            )
    resolved_turns.sort(key=lambda turn: turn.start_sequence)
    actions = _resolve_actions(
        intermediate,
        active_ranges,
        resolved_turns,
        session_projection=session_projection,
        verify_commands=verify_commands,
        cancel_event=cancel_event,
    )
    actions_by_turn: dict[str, list[ActionOperation]] = defaultdict(list)
    for action in actions:
        actions_by_turn[action.turn_id].append(action)
    resolved_turns = [
        replace(
            turn,
            action_counts=_action_count_metrics(
                tuple(actions_by_turn.get(turn.turn_id, ())),
                projection_precision=turn.action_projection_precision,
                projection_reason=turn.action_projection_reason,
            ),
        )
        for turn in resolved_turns
    ]
    change_verification = _resolve_change_verification(
        intermediate,
        active_ranges,
        resolved_turns,
        actions,
        session_projection(),
        rollback_projection_unresolved=rollback_projection_unresolved,
        cancel_event=cancel_event,
    )
    if session_integrity_reason is not None:
        change_verification = _degrade_change_verification(change_verification, session_integrity_reason)
    retry_amplification, retry_evidence, retry_target = _retry_amplification(
        intermediate,
        active_ranges,
        revisions,
        cancel_event=cancel_event,
    )
    validation = _validation_metrics(
        actions,
        resolved_turns,
        change_verification,
        retry_amplification,
        cancel_event=cancel_event,
    )
    if session_integrity_reason is not None:
        validation = _degrade_validation_metrics(
            validation,
            session_integrity_reason,
            session_integrity_cap=True,
        )
    context_carrying_load = _context_carrying_load(
        intermediate,
        active_ranges,
        resolved_turns,
        revisions,
        session_projection(),
        cancel_event=cancel_event,
    )
    findings = evaluate_findings(
        actions=actions,
        turns=resolved_turns,
        validation=validation,
        change_verification=change_verification,
        retry_amplification_evidence=retry_evidence,
        retry_target=retry_target,
        context_carrying_load=context_carrying_load,
        cancel_event=cancel_event,
    )
    span_mismatches, containment_violations = _turns._timeline_diagnostics(
        intermediate,
        active_ranges,
        cancel_event=cancel_event,
    )
    diagnostics = TrajectoryDiagnostics(
        line_count=intermediate.line_count,
        byte_count=intermediate.byte_count,
        torn_tail_bytes=intermediate.torn_tail_bytes,
        corrupt_line_count=len(intermediate.corrupt_after_sequences),
        unsupported_event_count=intermediate.unsupported_event_count,
        accounted_prefix_violations=tuple(violation.message for violation in intermediate.prefix_violations),
        accounted_prefix_violation_details=tuple(
            SequenceRangeDiagnostic(violation.message, violation.first_sequence, violation.last_sequence)
            for violation in intermediate.prefix_violations
        ),
        explicit_gap_count=len(intermediate.explicit_gaps),
        explicit_gaps=tuple(
            SequenceRangeDiagnostic("trajectory gap", first, last) for first, last in intermediate.explicit_gaps
        ),
        rollback_projection_unresolved=rollback_projection_unresolved,
        span_duration_mismatch_count=len(span_mismatches),
        span_duration_mismatches=span_mismatches,
        containment_violation_count=len(containment_violations),
        containment_violations=containment_violations,
        malformed_hook_execution_mode_count=intermediate.malformed_hook_execution_mode_count,
        side_call_empty_shell_revisions=revisions.side_call_empty_shell_revisions,
        unidentified_membership_revision_count=revisions.unidentified_membership_revision_count,
        corrupt_lines=tuple(intermediate.corrupt_lines),
        unsupported_lines=tuple(intermediate.unsupported_lines),
    )
    overview = _overview(resolved_turns, cancel_event=cancel_event)
    token_usage = _token_usage(intermediate, active_ranges, resolved_turns, cancel_event=cancel_event)
    skill_usage = _tool_usage_panel(
        actions,
        resolved_turns,
        tool_kind="skill",
        display_name=lambda action: session_projection().skill_names.get(action.call_item_id or ""),
    )
    mcp_usage = _tool_usage_panel(
        actions,
        resolved_turns,
        tool_kind="mcp",
        display_name=lambda action: action.tool_name,
    )
    logical_turns = _fold_retry_turns(resolved_turns, cancel_event=cancel_event)
    diagnostics = replace(
        diagnostics,
        timeline_operations=_timeline_operation_diagnostics(logical_turns),
    )
    insights = _insights_analysis(
        intermediate,
        active_ranges,
        resolved_turns,
        logical_turns,
        actions,
        context_carrying_load=context_carrying_load,
        cancel_event=cancel_event,
    )
    submission_latency = _submission_latency(intermediate, active_ranges, cancel_event=cancel_event)
    if session_integrity_reason is not None:
        overview = _degrade_overview(overview, session_integrity_reason)
        token_usage = _degrade_token_usage(token_usage, session_integrity_reason)
        skill_usage = _degrade_tool_usage_panel(skill_usage, session_integrity_reason)
        mcp_usage = _degrade_tool_usage_panel(mcp_usage, session_integrity_reason)
        insights = _degrade_insights(insights, session_integrity_reason)
        submission_latency = _degrade_submission_latency(submission_latency, session_integrity_reason)
    runs = workflow_runs(
        intermediate, active_ranges, integrity_reason=session_integrity_reason, cancel_event=cancel_event
    )
    if runs:
        overview = include_workflow_overview(overview, runs)
    return TrajectoryAnalysis(
        availability=AnalysisAvailability.AVAILABLE,
        path=intermediate.path,
        generation=generation,
        overview=overview,
        workflow_runs=runs,
        turns=tuple(logical_turns),
        diagnostics=diagnostics,
        actions=actions,
        validation=None if runs else validation,
        findings=findings,
        submission_latency=None if runs else submission_latency,
        change_verification=None if runs else change_verification,
        token_usage=None if runs else token_usage,
        skill_usage=None if runs else skill_usage,
        mcp_usage=None if runs else mcp_usage,
        insights=None if runs else insights,
        session_span=SessionSpan(
            first_turn_started_at=intermediate.first_turn_started_at,
            last_turn_finished_at=intermediate.last_turn_finished_at,
            runtime_count=len(intermediate.runtime_starts),
        ),
    )


def _timeline_operation_diagnostics(
    turns: list[TurnAnalysis],
) -> tuple[TimelineOperationDiagnostic, ...]:
    """Project sparse row diagnostics without losing physical retry ownership."""
    diagnostics: list[TimelineOperationDiagnostic] = []
    for turn in turns:
        for attempt in turn.attempts:
            for operation in turn.operations[attempt.operation_start_index : attempt.operation_end_index]:
                code = operation.diagnostic_code
                reason = operation.reason
                if code is None or reason is None:
                    continue
                diagnostics.append(
                    TimelineOperationDiagnostic(
                        turn_id=attempt.turn_id,
                        turn_number=turn.turn_number,
                        operation_id=operation.operation_id,
                        family=operation.family,
                        precision=operation.precision,
                        code=code,
                        reason=reason,
                        identity=operation.identity,
                        hook_id=operation.hook_id,
                    )
                )
    return tuple(diagnostics)


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
        context = _turns._tool_context_for_start(node, start)
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
            and _tool_succeeded(action.outcome)
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
    successful_verifies = [action for action in verifies if _tool_succeeded(action.outcome)]
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
    failures = [action for action in actions if _turns._tool_failed(action.outcome)]
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
            if _tool_succeeded(action.outcome):
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


def _degrade_overview(overview: TrajectoryOverview, reason: str) -> TrajectoryOverview:
    """Cap selection-wide metrics without changing still-provable turn metrics."""

    return TrajectoryOverview(
        elapsed_ns=_cap_session_metric(overview.elapsed_ns, reason),
        compute_cp_ns=_cap_session_metric(overview.compute_cp_ns, reason),
        response_cp_ns=_cap_session_metric(overview.response_cp_ns, reason),
        exclusive_work_ns=_cap_session_metric(overview.exclusive_work_ns, reason),
        parallelism=_cap_session_metric(overview.parallelism, reason),
        overlap_gain_ns=_cap_session_metric(overview.overlap_gain_ns, reason),
        wall_time_ns={bucket: _cap_session_metric(metric, reason) for bucket, metric in overview.wall_time_ns.items()},
        utilization={bucket: _cap_session_metric(metric, reason) for bucket, metric in overview.utilization.items()},
        usage_tokens=_cap_session_metric(overview.usage_tokens, reason),
    )


def _degrade_token_usage(usage: TokenUsage, reason: str) -> TokenUsage:
    """Cap normalized session token totals when events may be missing."""

    return TokenUsage(buckets={bucket: _cap_session_metric(metric, reason) for bucket, metric in usage.buckets.items()})


def _degrade_submission_latency(latency: SubmissionLatencyOverview, reason: str) -> SubmissionLatencyOverview:
    """Cap session percentile summaries without degrading intact samples."""

    return SubmissionLatencyOverview(
        buckets=tuple(
            replace(
                stats,
                p50_ns=_cap_session_metric(stats.p50_ns, reason),
                p90_ns=_cap_session_metric(stats.p90_ns, reason),
                max_ns=_cap_session_metric(stats.max_ns, reason),
            )
            for stats in latency.buckets
        )
    )


def _degrade_tool_usage_panel(panel: ToolUsagePanel, reason: str) -> ToolUsagePanel:
    precision, combined_reason = _cap_session_precision(panel.precision, panel.reason, reason)
    return replace(panel, precision=precision, reason=combined_reason)


def _degrade_insights(insights: InsightsAnalysis, reason: str) -> InsightsAnalysis:
    tools_precision, tools_reason = _cap_session_precision(
        insights.tools.precision,
        insights.tools.reason,
        reason,
    )
    tools = replace(
        insights.tools,
        rows=tuple(
            replace(
                row,
                duration_share=_cap_session_metric(row.duration_share, reason),
                p50_ns=_cap_session_metric(row.p50_ns, reason),
                p95_ns=_cap_session_metric(row.p95_ns, reason),
            )
            for row in insights.tools.rows
        ),
        precision=tools_precision,
        reason=tools_reason,
    )
    mcp_precision, mcp_reason = _cap_session_precision(insights.mcp.precision, insights.mcp.reason, reason)
    mcp = replace(
        insights.mcp,
        rows=tuple(_degrade_mcp_server_row(row, reason) for row in insights.mcp.rows),
        precision=mcp_precision,
        reason=mcp_reason,
    )
    skills_precision, skills_reason = _cap_session_precision(
        insights.skills.precision,
        insights.skills.reason,
        reason,
    )
    skills = replace(
        insights.skills,
        rows=tuple(
            replace(
                row,
                first_action_median_ns=_cap_session_metric(row.first_action_median_ns, reason),
                injected_tokens=_cap_session_metric(row.injected_tokens, reason),
            )
            for row in insights.skills.rows
        ),
        precision=skills_precision,
        reason=skills_reason,
    )
    context_carrying_precision, context_carrying_reason = _cap_session_precision(
        insights.context_carrying_precision,
        insights.context_carrying_reason,
        reason,
    )
    return replace(
        insights,
        tools=tools,
        mcp=mcp,
        skills=skills,
        context_carrying_precision=context_carrying_precision,
        context_carrying_reason=context_carrying_reason,
    )


def _degrade_mcp_server_row(row: McpServerRow, reason: str) -> McpServerRow:
    return replace(
        row,
        duration_share=_cap_session_metric(row.duration_share, reason),
        p50_ns=_cap_session_metric(row.p50_ns, reason),
        p95_ns=_cap_session_metric(row.p95_ns, reason),
        approval_blocking_share=_cap_session_metric(row.approval_blocking_share, reason),
        result_bytes=_cap_session_metric(row.result_bytes, reason),
        result_tokens=_cap_session_metric(row.result_tokens, reason),
        truncated_count=_cap_session_metric(row.truncated_count, reason),
        spill_count=_cap_session_metric(row.spill_count, reason),
        critical_path_exclusive_ns=_cap_session_metric(row.critical_path_exclusive_ns, reason),
        connection_wait_count=_cap_session_metric(row.connection_wait_count, reason),
        connection_wait_ns=_cap_session_metric(row.connection_wait_ns, reason),
        remotes=tuple(
            replace(
                remote,
                p50_ns=_cap_session_metric(remote.p50_ns, reason),
                p95_ns=_cap_session_metric(remote.p95_ns, reason),
            )
            for remote in row.remotes
        ),
    )


def _cap_session_metric(metric: Metric, reason: str) -> Metric:
    if metric.precision is Precision.MISSING:
        return metric
    if metric.precision is Precision.UNRESOLVED and metric.reason is not None:
        return metric
    return Metric(metric.value, Precision.UNRESOLVED, _merge_reasons(metric.reason, reason))


def _cap_session_precision(
    precision: Precision,
    existing_reason: str | None,
    integrity_reason: str,
) -> tuple[Precision, str | None]:
    if precision is Precision.MISSING:
        return precision, existing_reason
    if precision is Precision.UNRESOLVED and existing_reason is not None:
        return precision, existing_reason
    return Precision.UNRESOLVED, _merge_reasons(existing_reason, integrity_reason)


def _merge_reasons(*reasons: str | None) -> str:
    return "; ".join(dict.fromkeys(reason for reason in reasons if reason))


def _with_unresolved_precision(metric: Metric, reason: str) -> Metric:
    return Metric(metric.value, Precision.UNRESOLVED, reason)


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


def _retry_amplification(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    revisions: _turns._RevisionResolution,
    *,
    cancel_event: Event | None,
) -> tuple[Metric, tuple[str, ...], tuple[str | None, str | None, str | None]]:
    attribution_exact = True
    retry_ids: set[str] = set()
    for node in intermediate.nodes.get("retry", {}).values():
        _check_cancelled(cancel_event)
        if _all_side_call_lifecycle(node):
            continue
        cut = _lifecycle_cut(node, inactive_ranges)
        if cut is not _LifecycleCut.NONE:
            ownership = _node_owner_membership(
                intermediate,
                node,
                inactive_ranges,
                target_operation_id=node.starts[0].parent_operation_id,
            )
            if not _exclusively_inactive(ownership):
                attribution_exact = False
            continue
        starts = [event for event in node.starts if _active(event.sequence, inactive_ranges)]
        finishes = [event for event in node.finishes if _active(event.sequence, inactive_ranges)]
        endpoints = (*starts, *finishes)
        if not endpoints or all(event.side_call for event in endpoints):
            continue
        if any(event.side_call for event in endpoints):
            attribution_exact = False
            continue
        if len(starts) == 1 and not finishes:
            # A scheduled retry whose backoff never starts adds no retry usage.
            continue
        if (
            len(starts) != 1
            or len(finishes) != 1
            or starts[0].scope != finishes[0].scope
            or finishes[0].sequence <= starts[0].sequence
        ):
            attribution_exact = False
            continue
        retry_ids.add(node.operation_id)
    if not retry_ids:
        metric = (
            Metric(0, Precision.EXACT)
            if attribution_exact
            else Metric(None, Precision.UNRESOLVED, "retry lifecycle attribution is incomplete")
        )
        return metric, (), (None, None, None)
    parent_by_operation: dict[str, str | None] = {}
    for family in ("model.run", "model.cycle", "model.exchange"):
        for node in intermediate.nodes.get(family, {}).values():
            _check_cancelled(cancel_event)
            if _all_side_call_lifecycle(node):
                continue
            cut = _lifecycle_cut(node, inactive_ranges)
            if cut is not _LifecycleCut.NONE:
                ownership = _node_owner_membership(
                    intermediate,
                    node,
                    inactive_ranges,
                    target_operation_id=node.starts[0].parent_operation_id,
                )
                if not _exclusively_inactive(ownership):
                    attribution_exact = False
                continue
            starts = [event for event in node.starts if _active(event.sequence, inactive_ranges)]
            finishes = [event for event in node.finishes if _active(event.sequence, inactive_ranges)]
            endpoints = (*starts, *finishes)
            if not endpoints or all(event.side_call for event in endpoints):
                continue
            if any(event.side_call for event in endpoints) or len(starts) != 1:
                attribution_exact = False
            else:
                parent_by_operation[node.operation_id] = starts[0].parent_operation_id

    def retried(operation_id: str | None) -> bool:
        current = operation_id
        visited: set[str] = set()
        while current is not None and current not in visited:
            _check_cancelled(cancel_event)
            if current in retry_ids:
                return True
            visited.add(current)
            current = parent_by_operation.get(current)
        return False

    retried_exchanges: dict[int, tuple[str, _Endpoint]] = {}
    for node in intermediate.nodes.get("model.exchange", {}).values():
        _check_cancelled(cancel_event)
        if _all_side_call_lifecycle(node):
            continue
        if not retried(node.operation_id):
            continue
        cut = _lifecycle_cut(node, inactive_ranges)
        if cut is not _LifecycleCut.NONE:
            ownership = _node_owner_membership(
                intermediate,
                node,
                inactive_ranges,
                target_operation_id=node.starts[0].parent_operation_id,
            )
            if not _exclusively_inactive(ownership):
                attribution_exact = False
            continue
        starts = [event for event in node.starts if _active(event.sequence, inactive_ranges)]
        finishes = [event for event in node.finishes if _active(event.sequence, inactive_ranges)]
        endpoints = (*starts, *finishes)
        if endpoints and all(event.side_call for event in endpoints):
            continue
        if (
            any(event.side_call for event in endpoints)
            or len(starts) != 1
            or len(finishes) != 1
            or starts[0].scope != finishes[0].scope
            or finishes[0].sequence <= starts[0].sequence
        ):
            attribution_exact = False
        else:
            retried_exchanges[finishes[0].sequence] = (node.operation_id, starts[0])
    amplified = [
        usage
        for usages in intermediate.usage_by_turn.values()
        for usage in usages
        if _active(usage.sequence, inactive_ranges) and usage.sequence in retried_exchanges
    ]
    if not amplified:
        metric = (
            Metric(0, Precision.EXACT)
            if attribution_exact
            else Metric(None, Precision.UNRESOLVED, "retry exchange attribution is incomplete")
        )
        return metric, (), (None, None, None)
    missing = any(
        usage.normalization_unavailable or usage.input_total is None or usage.output_total is None
        for usage in amplified
    )
    value = sum((usage.input_total or 0) + (usage.output_total or 0) for usage in amplified)
    provenance_exact = all(
        usage.bucket_has_provider_provenance(UsageBucket.INPUT)
        and usage.bucket_has_provider_provenance(UsageBucket.OUTPUT)
        for usage in amplified
    )
    precision = (
        Precision.MISSING
        if missing
        else Precision.EXACT
        if provenance_exact and attribution_exact
        else Precision.UNRESOLVED
    )
    reason = (
        "normalized usage is missing for one or more retried exchanges"
        if missing
        else None
        if provenance_exact and attribution_exact
        else "retry exchange attribution is incomplete"
        if not attribution_exact
        else "normalized retry usage provenance is incomplete"
    )
    stable_items: set[str] = set()
    target: tuple[str | None, str | None, str | None] = (None, None, None)
    for usage in amplified:
        _check_cancelled(cancel_event)
        operation_id, start = retried_exchanges[usage.sequence]
        revision_id = _payload_str(start.payload, "context_revision_id")
        stable_items.update(revisions.memberships.get(revision_id or "", ()))
        if target == (None, None, None):
            target = (start.turn_id, operation_id, str(usage.sequence))
    return Metric(value if not missing else None, precision, reason), tuple(sorted(stable_items)), target


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
        action for action in actions if action.classification is ActionClass.VERIFY and _tool_succeeded(action.outcome)
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


# The vocabulary this build's writer can emit; an outcome outside it is
# damaged or from a different writer and cannot pass for read evidence.
_RECOGNIZED_PREPARATION_OUTCOMES: Final = frozenset(
    value for name, value in vars(PreparationOutcome).items() if not name.startswith("_") and isinstance(value, str)
)
_PROMOTED_PREPARATION_OUTCOMES: Final = frozenset({PreparationOutcome.FRESH_TURN, PreparationOutcome.RETRY_TURN})


def _drop_submission_lifecycle_cut(
    intermediate: _Intermediate,
    node: _Node,
    inactive_ranges: tuple[tuple[int, int], ...],
    turn_scope_membership: dict[str, tuple[bool, bool]],
) -> bool:
    """Whether a complete cut lifecycle has no active submission owner."""
    if _lifecycle_cut(node, inactive_ranges) is _LifecycleCut.NONE:
        return False
    raw_finish = node.finishes[0]
    outcome = _payload_str(raw_finish.payload, "outcome")
    if outcome not in _RECOGNIZED_PREPARATION_OUTCOMES:
        return False
    scope_membership = turn_scope_membership.get(node.operation_id, (False, False))
    target_turn_id = _payload_str(raw_finish.payload, "target_turn_id")
    target_membership = _turn_start_membership(
        intermediate,
        {target_turn_id} if target_turn_id is not None else set(),
        inactive_ranges,
    )
    ownership = _merge_projection_memberships(scope_membership, target_membership)
    if ownership[0]:
        return False
    if outcome in _PROMOTED_PREPARATION_OUTCOMES:
        # A promoted turn must claim the preparation scope. Absence is damaged
        # evidence rather than proof that the lifecycle was superseded.
        return _exclusively_inactive(scope_membership)
    if outcome == PreparationOutcome.INJECTED:
        # Successful injection terminals explicitly identify their target turn.
        return target_turn_id is not None and _exclusively_inactive(target_membership)
    # Rejected/cancelled/failed admission normally creates no turn claim. For
    # these derived samples, a complete cut with no active owner is superseded.
    return True


def _submission_latency(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    cancel_event: Event | None,
) -> SubmissionLatencyOverview:
    turn_by_scope: dict[str, list[_Endpoint]] = defaultdict(list)
    turn_scope_membership: dict[str, tuple[bool, bool]] = {}
    for turn in intermediate.turns.values():
        _check_cancelled(cancel_event)
        for start in turn.starts:
            scope_id = _payload_str(start.payload, "preparation_scope_operation_id")
            if scope_id is None:
                continue
            active = _active(start.sequence, inactive_ranges)
            turn_scope_membership[scope_id] = _merge_projection_memberships(
                turn_scope_membership.get(scope_id, (False, False)),
                (active, not active),
            )
            if active:
                turn_by_scope[scope_id].append(start)
    samples: list[SubmissionLatencySample] = []
    for node in intermediate.nodes.get("preparation", {}).values():
        _check_cancelled(cancel_event)
        starts = [event for event in node.starts if _active(event.sequence, inactive_ranges)]
        finishes = [event for event in node.finishes if _active(event.sequence, inactive_ranges)]
        endpoints = sorted((*starts, *finishes), key=lambda endpoint: endpoint.sequence)
        if not endpoints or _payload_str(endpoints[0].payload, "scope") != "pre_turn":
            continue
        if all(endpoint.side_call for endpoint in endpoints):
            continue
        if _drop_submission_lifecycle_cut(intermediate, node, inactive_ranges, turn_scope_membership):
            # The raw lifecycle is complete, but rollback cut its projection
            # away from every active owner. Its surviving endpoint is not a
            # live submission sample; ownership ambiguity remains diagnosable.
            continue
        start = starts[0] if len(starts) == 1 else None
        finish = finishes[0] if len(finishes) == 1 else None
        outcome = _payload_str(finish.payload, "outcome") if finish is not None else None
        bucket = _submission_bucket(outcome)
        turn_candidates = turn_by_scope.get(node.operation_id, ())
        if outcome in {"fresh_turn", "retry_turn"}:
            end = turn_candidates[0] if len(turn_candidates) == 1 else None
            turn_id = (
                end.turn_id if end is not None else _payload_str(finish.payload, "target_turn_id") if finish else None
            )
        else:
            end = finish
            turn_id = _payload_str(finish.payload, "target_turn_id") if finish is not None else None
        exact = (
            start is not None
            and finish is not None
            and end is not None
            and outcome in _RECOGNIZED_PREPARATION_OUTCOMES
            and not start.side_call
            and finish.monotonic_measurement
            and finish.scope == start.scope
            and finish.sequence > start.sequence
            and finish.monotonic_ns >= start.monotonic_ns
            and end.runtime_id == start.runtime_id
            and end.branch_id == start.branch_id
            and end.coverage_id == start.coverage_id
            and end.actor_id == start.actor_id
            and not end.side_call
            and end.sequence > start.sequence
            and end.monotonic_ns >= start.monotonic_ns
        )
        reason = None
        if len(starts) != 1:
            reason = "preparation scope does not have exactly one start"
        elif len(finishes) != 1:
            reason = "preparation scope does not have exactly one finished event"
        elif outcome is None:
            reason = "preparation terminal outcome is missing"
        elif outcome not in _RECOGNIZED_PREPARATION_OUTCOMES:
            reason = "preparation terminal outcome is unrecognized"
        elif end is None:
            reason = "promoted turn start cannot be resolved uniquely"
        elif start is not None and start.side_call:
            reason = "preparation scope belongs to a side-call actor"
        elif finish is not None and not finish.monotonic_measurement:
            reason = "preparation duration lacks monotonic provenance"
        elif start is not None and finish is not None and finish.scope != start.scope:
            reason = "preparation endpoints cross scope"
        elif (
            start is not None
            and finish is not None
            and (finish.sequence <= start.sequence or finish.monotonic_ns < start.monotonic_ns)
        ):
            reason = "preparation lifecycle endpoints are not ordered"
        elif (
            start is not None
            and end is not None
            and (
                end.runtime_id != start.runtime_id
                or end.branch_id != start.branch_id
                or end.coverage_id != start.coverage_id
                or end.actor_id != start.actor_id
                or end.side_call
            )
        ):
            reason = "submission endpoints cross scope"
        elif end is not None and start is not None and end.sequence <= start.sequence:
            reason = "submission endpoints are not ordered"
        elif end is not None and start is not None and end.monotonic_ns < start.monotonic_ns:
            reason = "submission latency has a negative endpoint interval"
        start_ns = start.monotonic_ns if start is not None else None
        raw_end_ns = end.monotonic_ns if end is not None else None
        end_ns = raw_end_ns if start_ns is None or raw_end_ns is None or raw_end_ns >= start_ns else None
        samples.append(
            SubmissionLatencySample(
                scope_operation_id=node.operation_id,
                outcome=outcome,
                bucket=bucket,
                start_sequence=start.sequence if start is not None else endpoints[0].sequence,
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=Metric(
                    end_ns - start_ns if exact and end_ns is not None and start_ns is not None else None,
                    Precision.EXACT if exact else Precision.UNRESOLVED,
                    reason,
                ),
                turn_id=turn_id,
                finished_count=len(finishes),
            )
        )
    ordered = tuple(sorted(samples, key=lambda sample: sample.start_sequence))
    stats: list[SubmissionLatencyStats] = []
    for bucket in SubmissionLatencyBucket:
        _check_cancelled(cancel_event)
        bucket_samples = tuple(sample for sample in ordered if sample.bucket is bucket)
        resolved = [
            cast("int", sample.duration_ns.value)
            for sample in bucket_samples
            if sample.duration_ns.precision is Precision.EXACT
        ]
        stats.append(
            SubmissionLatencyStats(
                bucket=bucket,
                sample_count=len(bucket_samples),
                unresolved_count=len(bucket_samples) - len(resolved),
                p50_ns=_percentile_metric(resolved, 0.50),
                p90_ns=_percentile_metric(resolved, 0.90),
                max_ns=Metric(max(resolved), Precision.EXACT)
                if resolved
                else Metric(None, Precision.MISSING, "no resolved samples"),
                samples=bucket_samples,
            )
        )
    return SubmissionLatencyOverview(tuple(stats))


def _context_carrying_load(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    turns: list[TurnAnalysis],
    revisions: _turns._RevisionResolution,
    projection: _SessionProjection,
    *,
    cancel_event: Event | None,
) -> tuple[ContextCarryingLoad, ...]:
    if not projection.item_tokens:
        return ()
    turns_by_id = {turn.turn_id: turn for turn in turns}
    claims: dict[str, tuple[str, _Endpoint]] = {}
    ambiguous_claims: set[str] = set()
    for node in intermediate.nodes.get("model.exchange", {}).values():
        _check_cancelled(cancel_event)
        for start in node.starts:
            if not _active(start.sequence, inactive_ranges) or start.side_call:
                continue
            revision_id = _payload_str(start.payload, "context_revision_id")
            if revision_id is None or revision_id in ambiguous_claims:
                continue
            if revision_id in claims:
                claims.pop(revision_id)
                ambiguous_claims.add(revision_id)
            else:
                claims[revision_id] = (node.operation_id, start)

    carried: dict[str, tuple[int, _Endpoint, _Endpoint]] = {}
    for revision_id, (exchange_operation_id, start) in sorted(claims.items(), key=lambda item: item[1][1].sequence):
        _check_cancelled(cancel_event)
        revision = revisions.endpoints.get(revision_id)
        membership = revisions.memberships.get(revision_id)
        if (
            revision is None
            or membership is None
            or revision_id in revisions.errors
            or revision.side_call
            or revision.parent_operation_id != exchange_operation_id
            or revision.runtime_id != start.runtime_id
            or revision.branch_id != start.branch_id
            or revision.coverage_id != start.coverage_id
            or revision.actor_id != start.actor_id
            or revision.sequence >= start.sequence
        ):
            continue
        counted: set[str] = set()
        for item_id in membership:
            _check_cancelled(cancel_event)
            if item_id in counted or item_id not in projection.item_tokens:
                continue
            counted.add(item_id)
            previous = carried.get(item_id)
            carried[item_id] = (
                (previous[0] + 1) if previous is not None else 1,
                previous[1] if previous is not None else revision,
                revision,
            )

    rows: list[ContextCarryingLoad] = []
    for item_id, (carry_count, first, last) in carried.items():
        _check_cancelled(cancel_event)
        token_count = projection.item_tokens[item_id]
        if token_count <= 0:
            continue
        turn = turns_by_id.get(last.turn_id or "")
        origin_turn = turns_by_id.get(first.turn_id or "")
        rows.append(
            ContextCarryingLoad(
                load=token_count * carry_count,
                item_id=item_id,
                occurrence_id=f"context-load:{last.sequence}:{last.event_id or 'revision'}",
                turn_id=last.turn_id,
                turn_number=turn.turn_number if turn is not None else None,
                token_count=token_count,
                carry_count=carry_count,
                origin_turn_number=origin_turn.turn_number if origin_turn is not None else None,
                role=projection.item_roles.get(item_id),
                tool_names=projection.item_tool_names.get(item_id, ()),
            )
        )
    return tuple(rows)


def _submission_bucket(outcome: str | None) -> SubmissionLatencyBucket:
    if outcome in {"fresh_turn", "retry_turn"}:
        return SubmissionLatencyBucket.BECAME_TURN
    if outcome == "injected":
        return SubmissionLatencyBucket.INJECTED
    return SubmissionLatencyBucket.DID_NOT_BECOME_TURN


def _percentile_metric(values: list[int], quantile: float) -> Metric:
    if not values:
        return Metric(None, Precision.MISSING, "no resolved samples")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * quantile) + 0.999999999) - 1))
    return Metric(ordered[index], Precision.EXACT)


def _tool_succeeded(outcome: str | None) -> bool:
    return outcome == ToolOutcome.SUCCESS


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
    if outcome not in _turns._KNOWN_TOOL_OUTCOMES:
        return outcome, Precision.UNRESOLVED, "tool terminal outcome is invalid", finish
    return outcome, Precision.EXACT, None, finish


def _overview(
    turns: list[TurnAnalysis],
    *,
    cancel_event: Event | None = None,
    metric_subject: str = "selected turns",
) -> TrajectoryOverview:
    for _turn in turns:
        _check_cancelled(cancel_event)
    elapsed = _sum_metrics([turn.elapsed_ns for turn in turns], metric_subject=metric_subject)
    compute_cp = _sum_metrics([turn.compute_cp_ns for turn in turns], metric_subject=metric_subject)
    response_cp = _sum_metrics([turn.response_cp_ns for turn in turns], metric_subject=metric_subject)
    exclusive = _sum_metrics([turn.exclusive_work_ns for turn in turns], metric_subject=metric_subject)
    overlap = _sum_metrics([turn.overlap_gain_ns for turn in turns], metric_subject=metric_subject)
    usage = _sum_metrics(
        [turn.usage_tokens for turn in turns],
        missing_is_missing=True,
        metric_subject=metric_subject,
    )
    ratio_precision = (
        Precision.EXACT
        if elapsed.precision is Precision.EXACT and exclusive.precision is Precision.EXACT
        else Precision.UNRESOLVED
    )
    elapsed_value = int(elapsed.value) if elapsed.value is not None else None
    exclusive_value = int(exclusive.value) if exclusive.value is not None else None
    wall = {
        bucket: _sum_metrics(
            [turn.wall_time_ns[bucket] for turn in turns],
            metric_subject=metric_subject,
        )
        for bucket in WallBucket
    }
    utilization: dict[WallBucket, Metric] = {}
    for bucket in (WallBucket.MODEL, WallBucket.TOOLS):
        work_ns = 0
        for turn in turns:
            _check_cancelled(cancel_event)
            work_ns += sum(
                item.duration_ns for item in turn.slices if item.counts_as_work and item.wall_bucket is bucket
            )
        utilization[bucket] = Metric(
            work_ns / elapsed_value
            if elapsed_value is not None and elapsed_value > 0
            else 0.0
            if elapsed_value == 0
            else None,
            ratio_precision,
            None if ratio_precision is Precision.EXACT else f"one or more {metric_subject} are unresolved",
        )
    return TrajectoryOverview(
        elapsed_ns=elapsed,
        compute_cp_ns=compute_cp,
        response_cp_ns=response_cp,
        exclusive_work_ns=exclusive,
        parallelism=Metric(
            exclusive_value / elapsed_value
            if elapsed_value is not None and elapsed_value > 0 and exclusive_value is not None
            else 0.0
            if elapsed_value == 0 and exclusive_value is not None
            else None,
            ratio_precision,
            None if ratio_precision is Precision.EXACT else f"one or more {metric_subject} are unresolved",
        ),
        overlap_gain_ns=overlap,
        wall_time_ns=wall,
        utilization=utilization,
        usage_tokens=usage,
    )


def _sum_metrics(
    metrics: list[Metric],
    *,
    missing_is_missing: bool = False,
    metric_subject: str = "selected turns",
) -> Metric:
    if any(metric.value is None for metric in metrics):
        missing = any(metric.precision is Precision.MISSING for metric in metrics)
        precision = Precision.MISSING if missing_is_missing and missing else Precision.UNRESOLVED
        reason = next((metric.reason for metric in metrics if metric.value is None and metric.reason is not None), None)
        return Metric(
            None,
            precision,
            reason or f"one or more {metric_subject} are unresolved",
        )
    values = [metric.value for metric in metrics if metric.value is not None]
    precision = least_precision(metric.precision for metric in metrics)
    return Metric(
        sum(values),
        precision,
        None if precision is Precision.EXACT else f"one or more {metric_subject} are not exact",
    )


def _sum_optional_bucket_metrics(metrics: list[Metric], *, metric_subject: str = "selected turns") -> Metric:
    """Sum an optional bucket across turns without letting absence poison it."""
    reported = [metric for metric in metrics if metric.value is not None]
    if not reported:
        reason = next((metric.reason for metric in metrics if metric.reason is not None), None)
        return Metric(None, Precision.MISSING, reason or "no exchange reported this bucket")
    value = sum(cast("int | float", metric.value) for metric in reported)
    if any(metric.precision is Precision.UNRESOLVED for metric in metrics):
        return Metric(value, Precision.UNRESOLVED, f"one or more {metric_subject} are unresolved")
    if len(reported) < len(metrics) or any(metric.precision is Precision.ESTIMATED for metric in reported):
        return Metric(value, Precision.ESTIMATED, "not every exchange reported this bucket")
    return Metric(value, Precision.EXACT)


def _fold_retry_turns(turns: list[TurnAnalysis], *, cancel_event: Event | None = None) -> list[TurnAnalysis]:
    """Fold consecutive retry/resume attempts into their one user-level turn.

    Producers keep a fresh physical ``turn_id`` per pass so every lifecycle can
    still be validated independently. ``turn_number`` is the user-level
    ordinal, and ``is_retry`` explicitly says that a later pass continues the
    preceding ordinal rather than opening another turn.
    """
    groups: list[list[TurnAnalysis]] = []
    for turn in turns:
        _check_cancelled(cancel_event)
        if (
            turn.attempts[0].is_retry
            and groups
            and turn.turn_number is not None
            and turn.turn_number == groups[-1][0].turn_number
        ):
            groups[-1].append(turn)
        else:
            groups.append([turn])
    return [_fold_turn_group(group, cancel_event=cancel_event) for group in groups]


def _fold_turn_group(attempts: list[TurnAnalysis], *, cancel_event: Event | None) -> TurnAnalysis:
    canonical = attempts[0]
    if len(attempts) == 1:
        attempt = attempts[0]
        source = attempt.attempts[0]
        diagnostics = list(attempt.diagnostics)
        if source.physical_axis_end_ns < source.physical_axis_start_ns:
            diagnostics.append("folded attempt axis end precedes its start")
        normalized = replace(
            source,
            logical_axis_start_ns=attempt.axis_start_ns,
            operation_start_index=0,
            operation_end_index=len(attempt.operations),
            slice_start_index=0,
            slice_end_index=len(attempt.slices),
        )
        return replace(
            attempt,
            axis_end_ns=attempt.axis_start_ns + normalized.duration_ns,
            diagnostics=tuple(dict.fromkeys(diagnostics)),
            attempts=(normalized,),
        )

    overview = _overview(
        attempts,
        cancel_event=cancel_event,
        metric_subject="attempts of this turn",
    )
    axis_start = canonical.axis_start_ns
    cursor = axis_start
    operations: list[TimelineOperation] = []
    slices: list[TimeSlice] = []
    attempt_refs: list[TurnAttemptRef] = []
    fold_diagnostics: list[str] = []
    parent_pairs = array("I")
    causal_pairs = array("I")
    root_index: int | None = None
    flows_exact = all(attempt.flow is not None for attempt in attempts)
    if len({attempt.runtime_id for attempt in attempts}) > 1:
        fold_diagnostics.append("logical turn spans multiple trajectory runtimes")
    for attempt_index, attempt in enumerate(attempts):
        _check_cancelled(cancel_event)
        source = attempt.attempts[0]
        if len(attempt.attempts) != 1:
            fold_diagnostics.append("fold input already contains multiple physical attempts")
        if source.physical_axis_end_ns < source.physical_axis_start_ns:
            fold_diagnostics.append("folded attempt axis end precedes its start")
        shift = cursor - attempt.axis_start_ns
        operation_offset = len(operations)
        operations.extend(
            replace(
                operation,
                start_ns=operation.start_ns + shift if operation.start_ns is not None else None,
                end_ns=operation.end_ns + shift if operation.end_ns is not None else None,
            )
            for operation in attempt.operations
        )
        operation_end = len(operations)
        slice_offset = len(slices)
        for item in attempt.slices:
            slices.append(
                replace(
                    item,
                    slice_index=len(slices),
                    start_ns=item.start_ns + shift,
                    end_ns=item.end_ns + shift,
                )
            )
        attempt_refs.append(
            TurnAttemptRef(
                turn_id=source.turn_id,
                runtime_id=source.runtime_id,
                is_retry=source.is_retry,
                physical_axis_start_ns=source.physical_axis_start_ns,
                physical_axis_end_ns=source.physical_axis_end_ns,
                logical_axis_start_ns=cursor,
                operation_start_index=operation_offset,
                operation_end_index=operation_end,
                slice_start_index=slice_offset,
                slice_end_index=len(slices),
            )
        )
        flow = attempt.flow
        if flow is not None:
            if root_index is None and flow.root_index is not None:
                root_index = flow.root_index + operation_offset
            # A cancelled/interrupted physical attempt may still have a typed
            # response terminal. Only the final attempt owns the response of
            # the stitched logical turn, so earlier terminal edges stop here.
            for edge_source, target in flow.parent_edges():
                if target == FLOW_TERMINAL_INDEX and attempt_index < len(attempts) - 1:
                    continue
                # FLOW_TERMINAL_INDEX is producer-defined as a target-only
                # response sentinel, so every source is a real operation index.
                parent_pairs.extend(
                    (
                        edge_source + operation_offset,
                        FLOW_TERMINAL_INDEX if target == FLOW_TERMINAL_INDEX else target + operation_offset,
                    )
                )
            for edge_source, target in flow.causal_edges():
                if target == FLOW_TERMINAL_INDEX and attempt_index < len(attempts) - 1:
                    continue
                causal_pairs.extend(
                    (
                        edge_source + operation_offset,
                        FLOW_TERMINAL_INDEX if target == FLOW_TERMINAL_INDEX else target + operation_offset,
                    )
                )
        cursor += attempt_refs[-1].duration_ns

    action_counts = {
        action_class: _sum_metrics(
            [attempt.action_counts.get(action_class, Metric(0, Precision.EXACT)) for attempt in attempts],
            metric_subject="attempts of this turn",
        )
        for action_class in ActionClass
    }
    action_precision = least_precision(attempt.action_projection_precision for attempt in attempts)
    action_reason = next(
        (
            attempt.action_projection_reason
            for attempt in attempts
            if attempt.action_projection_precision is not Precision.EXACT and attempt.action_projection_reason
        ),
        None,
    )
    token_usages = [attempt.token_usage for attempt in attempts]
    token_usage = (
        _merge_turn_token_usage([usage for usage in token_usages if usage is not None])
        if all(usage is not None for usage in token_usages)
        else None
    )
    flow = (
        TurnFlow(
            turn_id=canonical.turn_id,
            root_index=root_index,
            has_terminal=bool(attempts[-1].flow and attempts[-1].flow.has_terminal),
            parent_pairs=parent_pairs.tobytes(),
            causal_pairs=causal_pairs.tobytes(),
            acyclic=all(attempt.flow is not None and attempt.flow.acyclic for attempt in attempts),
        )
        if flows_exact
        else None
    )
    return replace(
        canonical,
        end_sequence=attempts[-1].end_sequence,
        elapsed_ns=overview.elapsed_ns,
        compute_cp_ns=overview.compute_cp_ns,
        response_cp_ns=overview.response_cp_ns,
        exclusive_work_ns=overview.exclusive_work_ns,
        parallelism=overview.parallelism,
        overlap_gain_ns=overview.overlap_gain_ns,
        wall_time_ns=overview.wall_time_ns,
        utilization=overview.utilization,
        usage_tokens=overview.usage_tokens,
        axis_end_ns=cursor,
        operations=tuple(operations),
        slices=tuple(slices),
        diagnostics=tuple(
            dict.fromkeys(
                (
                    *(diagnostic for attempt in attempts for diagnostic in attempt.diagnostics),
                    *fold_diagnostics,
                )
            )
        ),
        action_counts=action_counts,
        critical_tool_contributions_ns=dict(
            sum((Counter(attempt.critical_tool_contributions_ns) for attempt in attempts), Counter())
        ),
        server_critical_contributions_ns=dict(
            sum((Counter(attempt.server_critical_contributions_ns) for attempt in attempts), Counter())
        ),
        action_projection_precision=action_precision,
        action_projection_reason=action_reason,
        token_usage=token_usage,
        flow=flow,
        attempts=tuple(attempt_refs),
    )


def _merge_turn_token_usage(usages: list[TokenUsage]) -> TokenUsage:
    per_bucket = {bucket: [usage.buckets[bucket] for usage in usages] for bucket in UsageBucket}
    buckets = {
        UsageBucket.INPUT: _sum_metrics(
            per_bucket[UsageBucket.INPUT],
            missing_is_missing=True,
            metric_subject="attempts of this turn",
        ),
        UsageBucket.OUTPUT: _sum_metrics(
            per_bucket[UsageBucket.OUTPUT],
            missing_is_missing=True,
            metric_subject="attempts of this turn",
        ),
    }
    for bucket in _turns._OPTIONAL_USAGE_BUCKETS:
        buckets[bucket] = _sum_optional_bucket_metrics(
            per_bucket[bucket],
            metric_subject="attempts of this turn",
        )
    return TokenUsage(buckets=buckets)


def _token_usage(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    turns: list[TurnAnalysis],
    *,
    cancel_event: Event | None,
) -> TokenUsage:
    per_turn: dict[UsageBucket, list[Metric]] = {bucket: [] for bucket in UsageBucket}
    for turn in turns:
        _check_cancelled(cancel_event)
        usage = turn.token_usage
        if usage is None:
            usage = _turns._turn_token_usage(intermediate, turn.turn_id, inactive_ranges, turn.usage_tokens)
        for bucket in UsageBucket:
            per_turn[bucket].append(usage.buckets[bucket])
    buckets = {
        UsageBucket.INPUT: _sum_metrics(per_turn[UsageBucket.INPUT], missing_is_missing=True),
        UsageBucket.OUTPUT: _sum_metrics(per_turn[UsageBucket.OUTPUT], missing_is_missing=True),
    }
    for bucket in _turns._OPTIONAL_USAGE_BUCKETS:
        buckets[bucket] = _sum_optional_bucket_metrics(per_turn[bucket])
    return TokenUsage(buckets=buckets)


def _insights_analysis(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    turns: list[TurnAnalysis],
    logical_turns: list[TurnAnalysis],
    actions: tuple[ActionOperation, ...],
    *,
    context_carrying_load: tuple[ContextCarryingLoad, ...],
    cancel_event: Event | None,
) -> InsightsAnalysis:
    return InsightsAnalysis(
        tools=_tool_insights(actions, turns),
        mcp=_mcp_insights(intermediate, inactive_ranges, actions, turns, cancel_event=cancel_event),
        skills=_skill_insights(actions, turns, logical_turns),
        context_carrying_load=context_carrying_load,
    )


def _tool_insights(actions: tuple[ActionOperation, ...], turns: list[TurnAnalysis]) -> ToolInsights:
    grouped: dict[tuple[str, str | None], list[ActionOperation]] = defaultdict(list)
    unclassified = 0
    for action in actions:
        tool_kind = action.tool_kind if action.tool_kind in TOOL_KINDS else "unclassified"
        if tool_kind == "unclassified":
            unclassified += 1
        grouped[(tool_kind, action.tool_name)].append(action)
    total_duration = _complete_duration_total(actions)
    rows = tuple(
        sorted(
            (
                ToolInsightRow(
                    tool_kind=tool_kind,
                    tool_name=tool_name,
                    calls=len(group),
                    duration_share=_duration_share(group, total_duration),
                    p50_ns=_duration_percentile(group, 0.50),
                    p95_ns=_duration_percentile(group, 0.95),
                    outcomes=_outcome_rows(group),
                )
                for (tool_kind, tool_name), group in grouped.items()
            ),
            key=lambda row: (
                -(float(row.duration_share.value) if row.duration_share.value is not None else -1.0),
                -row.calls,
                row.tool_kind,
                row.tool_name or "",
            ),
        )
    )
    precision, reason = _action_panel_precision(turns)
    return ToolInsights(
        total=len(actions),
        rows=rows,
        unclassified=unclassified,
        precision=precision,
        reason=reason,
    )


def _mcp_insights(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    actions: tuple[ActionOperation, ...],
    turns: list[TurnAnalysis],
    *,
    cancel_event: Event | None,
) -> McpInsights:
    selected = [action for action in actions if action.tool_kind == "mcp"]
    grouped: dict[str, list[ActionOperation]] = defaultdict(list)
    unattributed = 0
    for action in selected:
        if action.server_name is None:
            unattributed += 1
        else:
            grouped[action.server_name].append(action)
    total_duration = _complete_duration_total(actions)
    approval_by_tool = _approval_durations_by_tool(intermediate, inactive_ranges, cancel_event=cancel_event)
    waits_by_server, unattributed_waits = _mcp_connection_waits(
        intermediate,
        inactive_ranges,
        cancel_event=cancel_event,
    )
    turns_by_id = {turn.turn_id: turn for turn in turns}
    rows = tuple(
        sorted(
            (
                _mcp_server_row(
                    server_name,
                    group,
                    total_duration=total_duration,
                    approval_by_tool=approval_by_tool,
                    connection_waits=waits_by_server.get(server_name, ()),
                    turns_by_id=turns_by_id,
                )
                for server_name, group in grouped.items()
            ),
            key=lambda row: (
                -(float(row.duration_share.value) if row.duration_share.value is not None else -1.0),
                -row.calls,
                row.server_name,
            ),
        )
    )
    precision, reason = _action_panel_precision(turns)
    return McpInsights(
        total=len(selected),
        rows=rows,
        unattributed=unattributed,
        unattributed_connection_waits=unattributed_waits,
        precision=precision,
        reason=reason,
    )


def _mcp_server_row(
    server_name: str,
    actions: list[ActionOperation],
    *,
    total_duration: int | None,
    approval_by_tool: dict[str, tuple[int | None, ...]],
    connection_waits: tuple[int | None, ...],
    turns_by_id: dict[str, TurnAnalysis],
) -> McpServerRow:
    remote_groups: dict[str | None, list[ActionOperation]] = defaultdict(list)
    for action in actions:
        remote_groups[action.remote_name].append(action)
    remotes = tuple(
        McpRemoteRow(
            remote_name=remote_name,
            calls=len(group),
            p50_ns=_duration_percentile(group, 0.50),
            p95_ns=_duration_percentile(group, 0.95),
            outcomes=_outcome_rows(group),
        )
        for remote_name, group in sorted(remote_groups.items(), key=lambda item: (-len(item[1]), item[0] or ""))
    )
    approval_samples = tuple(
        duration for action in actions for duration in approval_by_tool.get(action.operation_id, ())
    )
    action_duration = _complete_duration_total(actions)
    approval_total = sum(duration for duration in approval_samples if duration is not None)
    if any(duration is None for duration in approval_samples) or action_duration is None:
        approval_share = Metric(None, Precision.UNRESOLVED, "one or more approval or tool intervals are unresolved")
    else:
        # A tool operation stays open for its whole approval wait, so the
        # operation durations already contain the approval time; adding the
        # approval total to the denominator would count that wait twice.
        approval_share = Metric(approval_total / action_duration if action_duration else 0.0, Precision.EXACT)
    wait_count, wait_duration = _connection_wait_metrics(connection_waits)
    return McpServerRow(
        server_name=server_name,
        calls=len(actions),
        duration_share=_duration_share(actions, total_duration),
        p50_ns=_duration_percentile(actions, 0.50),
        p95_ns=_duration_percentile(actions, 0.95),
        outcomes=_outcome_rows(actions),
        approval_blocking_share=approval_share,
        result_bytes=_payload_sum_metric(actions, "bytes"),
        result_tokens=_payload_sum_metric(actions, "tokens", inherently_estimated=True),
        truncated_count=_payload_sum_metric(actions, "truncated"),
        spill_count=_payload_sum_metric(actions, "spill"),
        critical_path_exclusive_ns=_server_cp_metric(server_name, actions, turns_by_id),
        connection_wait_count=wait_count,
        connection_wait_ns=wait_duration,
        remotes=remotes,
    )


def _skill_insights(
    actions: tuple[ActionOperation, ...],
    turns: list[TurnAnalysis],
    logical_turns: list[TurnAnalysis],
) -> SkillInsights:
    selected = [action for action in actions if action.tool_kind == "skill"]
    grouped: dict[str, list[ActionOperation]] = defaultdict(list)
    not_found: Counter[str] = Counter()
    unattributed = 0
    for action in selected:
        if action.skill_name is None:
            unattributed += 1
        elif action.skill_revision is None:
            if action.tool_name == "load_skill":
                not_found[action.skill_name] += 1
            else:
                unattributed += 1
        else:
            grouped[action.skill_name].append(action)
    turns_by_id = {turn.turn_id: turn for turn in turns}
    canonical_turn_ids = {
        attempt.turn_id: logical_turn.turn_id for logical_turn in logical_turns for attempt in logical_turn.attempts
    }
    rows = tuple(
        sorted(
            (_skill_row(skill_name, group, turns_by_id, canonical_turn_ids) for skill_name, group in grouped.items()),
            key=lambda row: (-row.load_count, -row.script_count - row.resource_count, row.skill_name),
        )
    )
    precision, reason = _action_panel_precision(turns)
    return SkillInsights(
        total=len(selected),
        rows=rows,
        not_found=tuple(
            NamedCountRow(name=name, count=count)
            for name, count in sorted(not_found.items(), key=lambda item: (-item[1], item[0]))
        ),
        unattributed=unattributed,
        precision=precision,
        reason=reason,
    )


def _skill_row(
    skill_name: str,
    actions: list[ActionOperation],
    turns_by_id: dict[str, TurnAnalysis],
    canonical_turn_ids: dict[str, str],
) -> SkillInsightRow:
    loads = [action for action in actions if action.tool_name == "load_skill"]
    scripts = [action for action in actions if action.tool_name == "run_skill_script"]
    resources = [action for action in actions if action.tool_name == "read_skill_resource"]
    script_groups: dict[str | None, list[ActionOperation]] = defaultdict(list)
    resource_groups: dict[str | None, list[ActionOperation]] = defaultdict(list)
    for action in scripts:
        script_groups[action.script_name].append(action)
    for action in resources:
        resource_groups[action.resource_name].append(action)
    return SkillInsightRow(
        skill_name=skill_name,
        load_count=len(loads),
        turn_count=len({canonical_turn_ids.get(action.turn_id, action.turn_id) for action in actions}),
        first_action_median_ns=_skill_first_action_latency(loads, (*scripts, *resources), turns_by_id),
        script_count=len(scripts),
        script_outcomes=_outcome_rows(scripts),
        script_exit_codes=_exit_code_rows(scripts),
        resource_count=len(resources),
        injected_tokens=_payload_sum_metric(loads, "tokens", inherently_estimated=True),
        revisions=tuple(sorted({action.skill_revision for action in actions if action.skill_revision is not None})),
        scripts=tuple(
            SkillActivityRow(
                name=name,
                count=len(group),
                outcomes=_outcome_rows(group),
                exit_codes=_exit_code_rows(group),
            )
            for name, group in sorted(script_groups.items(), key=lambda item: (-len(item[1]), item[0] or ""))
        ),
        resources=tuple(
            SkillActivityRow(name=name, count=len(group), outcomes=_outcome_rows(group))
            for name, group in sorted(resource_groups.items(), key=lambda item: (-len(item[1]), item[0] or ""))
        ),
    )


def _action_panel_precision(turns: list[TurnAnalysis]) -> tuple[Precision, str | None]:
    degraded = next((turn for turn in turns if turn.action_projection_precision is not Precision.EXACT), None)
    if degraded is None:
        return Precision.EXACT, None
    return Precision.UNRESOLVED, degraded.action_projection_reason or "tool action projection is incomplete"


def _complete_duration_total(actions: Iterable[ActionOperation]) -> int | None:
    durations = [action.duration_ns for action in actions]
    return (
        sum(cast("int", duration) for duration in durations)
        if all(duration is not None for duration in durations)
        else None
    )


def _duration_share(actions: Iterable[ActionOperation], total_duration: int | None) -> Metric:
    duration = _complete_duration_total(actions)
    if duration is None or total_duration is None:
        return Metric(None, Precision.UNRESOLVED, "one or more tool durations are unresolved")
    return Metric(duration / total_duration if total_duration else 0.0, Precision.EXACT)


def _duration_percentile(actions: Iterable[ActionOperation], quantile: float) -> Metric:
    group = list(actions)
    durations = [action.duration_ns for action in group]
    if not group:
        return Metric(None, Precision.MISSING, "no matching tool calls")
    if any(duration is None for duration in durations):
        return Metric(None, Precision.UNRESOLVED, "one or more tool durations are unresolved")
    return _percentile_metric([cast("int", duration) for duration in durations], quantile)


def _outcome_rows(actions: Iterable[ActionOperation]) -> tuple[NamedCountRow, ...]:
    counts = Counter(action.outcome or "unresolved" for action in actions)
    return tuple(
        NamedCountRow(name=name, count=count)
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _exit_code_rows(actions: Iterable[ActionOperation]) -> tuple[NamedCountRow, ...]:
    counts = Counter(str(action.exit_code) for action in actions if action.exit_code is not None)
    return tuple(
        NamedCountRow(name=name, count=count)
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _payload_sum_metric(
    actions: Iterable[ActionOperation],
    field: str,
    *,
    inherently_estimated: bool = False,
) -> Metric:
    group = list(actions)
    observed = [action for action in group if action.payload_observed]
    if not observed:
        return Metric(None, Precision.MISSING, "no tool payload observation is available")
    values: list[int] = []
    for action in observed:
        value: int | None
        if field == "bytes":
            value = action.payload_bytes
        elif field == "tokens":
            value = action.payload_token_estimate
        elif field == "truncated":
            value = int(action.payload_truncated) if action.payload_truncated is not None else None
        else:
            value = int(action.payload_spilled) if action.payload_spilled is not None else None
        if value is not None:
            values.append(value)
    if not values:
        return Metric(None, Precision.MISSING, "the observed payload did not report this field")
    partial = len(observed) < len(group) or len(values) < len(observed)
    precision = Precision.ESTIMATED if partial or inherently_estimated else Precision.EXACT
    reason = (
        "local tokenizer estimate"
        if inherently_estimated and not partial
        else "one or more tool payload observations are missing"
        if partial
        else None
    )
    return Metric(sum(values), precision, reason)


def _approval_durations_by_tool(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    cancel_event: Event | None,
) -> dict[str, tuple[int | None, ...]]:
    durations: dict[str, tuple[int | None, ...]] = {}
    for node in intermediate.nodes.get("approval", {}).values():
        _check_cancelled(cancel_event)
        cut = _lifecycle_cut(node, inactive_ranges)
        if cut is not _LifecycleCut.NONE:
            raw_start = node.starts[0]
            if raw_start.side_call:
                continue
            target = _payload_str(raw_start.payload, "target_tool_operation_id")
            if target is None:
                continue
            ownership = _node_owner_membership(
                intermediate,
                node,
                inactive_ranges,
                target_operation_id=target,
            )
            if _exclusively_inactive(ownership):
                continue
            durations[target] = (*durations.get(target, ()), None)
            continue
        starts = [event for event in node.starts if _active(event.sequence, inactive_ranges) and not event.side_call]
        finishes = [
            event for event in node.finishes if _active(event.sequence, inactive_ranges) and not event.side_call
        ]
        if not starts:
            continue
        target = _payload_str(starts[0].payload, "target_tool_operation_id") if len(starts) == 1 else None
        if target is None:
            continue
        duration = None
        if (
            len(finishes) == 1
            and finishes[0].monotonic_measurement
            and finishes[0].scope == starts[0].scope
            and finishes[0].sequence > starts[0].sequence
            and finishes[0].monotonic_ns >= starts[0].monotonic_ns
        ):
            duration = finishes[0].monotonic_ns - starts[0].monotonic_ns
        durations[target] = (*durations.get(target, ()), duration)
    return durations


def _mcp_connection_waits(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
    *,
    cancel_event: Event | None,
) -> tuple[dict[str, tuple[int | None, ...]], int]:
    grouped: dict[str, tuple[int | None, ...]] = {}
    unattributed = 0
    for node in intermediate.nodes.get("wait", {}).values():
        _check_cancelled(cancel_event)
        cut = _lifecycle_cut(node, inactive_ranges)
        if cut is not _LifecycleCut.NONE:
            raw_start = node.starts[0]
            if raw_start.side_call or _payload_str(raw_start.payload, "category") != "mcp_connect":
                continue
            target = _payload_str(raw_start.payload, "target_operation_id")
            ownership = _node_owner_membership(
                intermediate,
                node,
                inactive_ranges,
                target_operation_id=target,
            )
            if _exclusively_inactive(ownership):
                continue
            server_name = _payload_str(raw_start.payload, "server_name")
            if server_name is None:
                unattributed += 1
            else:
                grouped[server_name] = (*grouped.get(server_name, ()), None)
            continue
        starts = [event for event in node.starts if _active(event.sequence, inactive_ranges) and not event.side_call]
        finishes = [
            event for event in node.finishes if _active(event.sequence, inactive_ranges) and not event.side_call
        ]
        if not starts or not any(_payload_str(start.payload, "category") == "mcp_connect" for start in starts):
            continue
        server_name = _payload_str(starts[0].payload, "server_name") if len(starts) == 1 else None
        if server_name is None:
            unattributed += 1
            continue
        duration = None
        if (
            len(finishes) == 1
            and finishes[0].monotonic_measurement
            and finishes[0].scope == starts[0].scope
            and finishes[0].sequence > starts[0].sequence
            and finishes[0].monotonic_ns >= starts[0].monotonic_ns
        ):
            duration = finishes[0].monotonic_ns - starts[0].monotonic_ns
        grouped[server_name] = (*grouped.get(server_name, ()), duration)
    return grouped, unattributed


def _connection_wait_metrics(samples: tuple[int | None, ...]) -> tuple[Metric, Metric]:
    if not samples:
        return Metric(0, Precision.EXACT), Metric(0, Precision.EXACT)
    if any(sample is None for sample in samples):
        reason = "one or more MCP connection waits are unresolved"
        return Metric(len(samples), Precision.EXACT), Metric(None, Precision.UNRESOLVED, reason)
    return Metric(len(samples), Precision.EXACT), Metric(
        sum(cast("int", sample) for sample in samples), Precision.EXACT
    )


def _server_cp_metric(
    server_name: str,
    actions: list[ActionOperation],
    turns_by_id: dict[str, TurnAnalysis],
) -> Metric:
    relevant = [turns_by_id[turn_id] for turn_id in {action.turn_id for action in actions} if turn_id in turns_by_id]
    exact = [turn for turn in relevant if turn.response_cp_ns.precision is Precision.EXACT]
    unresolved_count = len(relevant) - len(exact)
    if not exact:
        return Metric(None, Precision.UNRESOLVED, "no related turn has an exact response critical path")
    value = sum(turn.server_critical_contributions_ns.get(server_name, 0) for turn in exact)
    if unresolved_count:
        return Metric(
            value,
            Precision.ESTIMATED,
            f"{unresolved_count} related turn(s) have an unresolved response critical path",
        )
    return Metric(value, Precision.EXACT)


def _skill_first_action_latency(
    loads: list[ActionOperation],
    related: tuple[ActionOperation, ...],
    turns_by_id: dict[str, TurnAnalysis],
) -> Metric:
    samples: list[int] = []
    unresolved = 0
    ordered_related = sorted(related, key=lambda action: action.start_sequence)
    for load in loads:
        following = next((action for action in ordered_related if action.start_sequence > load.start_sequence), None)
        if following is None:
            continue
        load_turn = turns_by_id.get(load.turn_id)
        action_turn = turns_by_id.get(following.turn_id)
        # An action that began before the load's terminal landed cannot
        # prove it used the loaded skill, however its timestamps read.
        if (
            load.end_ns is None
            or load.end_sequence is None
            or following.start_sequence <= load.end_sequence
            or load_turn is None
            or action_turn is None
            or load_turn.runtime_id != action_turn.runtime_id
            or following.start_ns < load.end_ns
        ):
            unresolved += 1
            continue
        samples.append(following.start_ns - load.end_ns)
    if not samples:
        if unresolved:
            return Metric(None, Precision.UNRESOLVED, "load-to-action latency endpoints are unresolved")
        return Metric(None, Precision.MISSING, "no load was followed by a related skill action")
    value = int(median(samples))
    if unresolved:
        return Metric(value, Precision.ESTIMATED, "one or more load-to-action latency samples are unresolved")
    return Metric(value, Precision.EXACT)


def _tool_usage_panel(
    actions: tuple[ActionOperation, ...],
    turns: list[TurnAnalysis],
    *,
    tool_kind: str,
    display_name: Callable[[ActionOperation], str | None],
) -> ToolUsagePanel:
    selected = [action for action in actions if action.tool_kind == tool_kind]
    counts: Counter[str] = Counter()
    unattributed = 0
    for action in selected:
        name = display_name(action)
        if name is None:
            unattributed += 1
        else:
            counts[name] += 1
    rows = tuple(
        NamedCountRow(name=name, count=count)
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    degraded = next((turn for turn in turns if turn.action_projection_precision is not Precision.EXACT), None)
    if degraded is not None:
        return ToolUsagePanel(
            total=len(selected),
            rows=rows,
            unattributed=unattributed,
            precision=Precision.UNRESOLVED,
            reason=degraded.action_projection_reason or "tool action projection is incomplete",
        )
    return ToolUsagePanel(total=len(selected), rows=rows, unattributed=unattributed)


def _session_integrity_reason(
    intermediate: _Intermediate,
    *,
    rollback_projection_unresolved: bool,
) -> str | None:
    """Explain physical damage or an unresolvable live-history projection."""

    problems: list[str] = []
    if intermediate.line_count == 0 and intermediate.byte_count == 0 and intermediate.torn_tail_bytes == 0:
        problems.append("empty log")
    if intermediate.corrupt_after_sequences:
        problems.append("corrupt lines")
    if intermediate.unsupported_event_count:
        problems.append("unsupported events")
    if intermediate.prefix_violations:
        problems.append("accounted-prefix violations")
    if intermediate.explicit_gaps:
        problems.append("explicit gaps")
    if intermediate.torn_tail_bytes:
        problems.append("torn tail")
    if rollback_projection_unresolved:
        problems.append("unresolved rollback projection")
    if not problems:
        return None
    return f"session trajectory integrity is unresolved: {', '.join(problems)}"


def _rollback_projection_unresolved(
    intermediate: _Intermediate,
    inactive_ranges: tuple[tuple[int, int], ...],
) -> bool:
    if any(_active(sequence, inactive_ranges) for sequence in intermediate.rollback_errors):
        return True
    rollback_pairs = {
        (old_branch, new_branch)
        for sequence, old_branch, new_branch in intermediate.rollback_branch_pairs
        if _active(sequence, inactive_ranges)
    }
    return any(
        _active(sequence, inactive_ranges)
        and (old_branch is None or new_branch is None or (old_branch, new_branch) not in rollback_pairs)
        for sequence, old_branch, new_branch in intermediate.branch_supersessions
    )


def _read_installed_fingerprint_key(path: Path) -> bytes | None:
    if (
        len(path.parents) < 4
        or path.name != "events.jsonl"
        or path.parent.name != "trajectory"
        or path.parents[2].name != "sessions"
    ):
        return None
    from chrys.foundation.platform import get_platform

    # The recorder keeps the key in the platform config directory. Beside the
    # session root covers the default layout (the root IS the config
    # directory) and trees copied along with their key; the config directory
    # is the fallback for a custom session root, whose sessions live
    # elsewhere while the key stays put.
    candidates = [path.parents[3] / TRAJECTORY_KEY_FILE_NAME]
    config_key_path = get_platform().config_dir / TRAJECTORY_KEY_FILE_NAME
    if config_key_path not in candidates:
        candidates.append(config_key_path)
    for key_path in candidates:
        try:
            with secure_open_owner_only_binary(key_path) as handle:
                key = handle.read(FINGERPRINT_KEY_BYTES + 1)
        except OSError:
            continue
        if len(key) == FINGERPRINT_KEY_BYTES:
            return key
    return None
