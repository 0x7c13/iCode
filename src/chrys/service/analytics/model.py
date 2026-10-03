# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Serializable value objects produced by trajectory analysis."""

from __future__ import annotations

from array import array
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Precision(StrEnum):
    """The four display states frozen by the trajectory contract."""

    EXACT = "exact"
    ESTIMATED = "estimated"
    MISSING = "missing"
    UNRESOLVED = "unresolved"


_PRECISION_ORDER = {
    Precision.EXACT: 0,
    Precision.ESTIMATED: 1,
    Precision.MISSING: 2,
    Precision.UNRESOLVED: 3,
}


def least_precision(values: Iterable[Precision]) -> Precision:
    """The least precise of *values*, in the order exact, estimated, missing, unresolved; exact when empty."""
    precisions = list(values)
    return max(precisions, key=_PRECISION_ORDER.__getitem__) if precisions else Precision.EXACT


class TimelineDiagnosticCode(StrEnum):
    """Stable reason codes for lifecycle rows that cannot expose a duration."""

    DETACHED_HOOK = "detached_hook"
    MISSING_START = "missing_start"
    MISSING_TERMINAL = "missing_terminal"
    NONUNIQUE_LIFECYCLE = "nonunique_lifecycle"
    INVALID_ENDPOINTS = "invalid_endpoints"
    OUTSIDE_TURN_COVERAGE = "outside_turn_coverage"
    ROLLBACK_START_SURVIVES = "rollback_start_survives"
    ROLLBACK_TERMINAL_SURVIVES = "rollback_terminal_survives"


class AnalysisAvailability(StrEnum):
    """Whether the trajectory source can be analyzed at all."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    READ_ERROR = "read_error"


class WallBucket(StrEnum):
    """The mutually exclusive wall-clock partition shown in Overview."""

    MODEL = "model"
    TOOLS = "tools"
    WAIT = "wait"
    IDLE = "idle"


class ActionClass(StrEnum):
    """Validation-loop action classes derived from recorded tool evidence."""

    SEARCH = "search"
    READ = "read"
    EDIT = "edit"
    VERIFY = "verify"
    OTHER = "other"


class FindingSeverity(StrEnum):
    """Renderer-facing finding severity."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class SubmissionLatencyBucket(StrEnum):
    """The three non-additive preparation-outcome groups."""

    BECAME_TURN = "became_turn"
    INJECTED = "injected"
    DID_NOT_BECOME_TURN = "did_not_become_turn"


class UsageBucket(StrEnum):
    """The normalized usage buckets shown in the Overview token panel."""

    INPUT = "input"
    OUTPUT = "output"
    REASONING = "reasoning"
    CACHE_READ = "cache_read"
    CACHE_CREATION = "cache_creation"


class ChangeVerificationState(StrEnum):
    """Latest known verification state for one changed file."""

    VERIFIED = "verified"
    AFTER_VERIFY = "after_verify"
    UNVERIFIED = "unverified"
    NET_ZERO = "net_zero"


class HookOwnership(StrEnum):
    """Field-derived ownership classes from the hook event/mode matrix."""

    CONCURRENT = "concurrent"
    TOOL_PREAMBLE = "tool_preamble"
    TOOL_TAIL = "tool_tail"
    APPROVAL = "approval"
    TURN_PREAMBLE = "turn_preamble"
    TURN_TAIL = "turn_tail"
    PRE_TURN = "pre_turn"
    SUB_AGENT = "sub_agent"
    COMPACTION = "compaction"
    SESSION_ROOT = "session_root"
    WORKFLOW_ROOT = "workflow_root"
    UNSAFE = "unsafe"


@dataclass(frozen=True, slots=True)
class SequenceRangeDiagnostic:
    """One sequence-region diagnostic retained for rendering and export."""

    message: str
    first_sequence: int
    last_sequence: int


@dataclass(frozen=True, slots=True)
class CorruptLineDiagnostic:
    """One corrupt physical line and its last trustworthy sequence anchor."""

    line_number: int
    byte_offset: int
    byte_length: int
    reason: str
    after_sequence: int


@dataclass(frozen=True, slots=True)
class UnsupportedLineDiagnostic:
    """One unsupported physical line whose envelope sequence is readable."""

    line_number: int
    byte_offset: int
    sequence: int
    schema_version: int | None


@dataclass(frozen=True, slots=True)
class SpanDurationMismatch:
    """A recorded duration that differs from its lifecycle interval."""

    family: str
    operation_id: str
    start_sequence: int
    finish_sequence: int
    interval_ns: int
    recorded_duration_ms: int


@dataclass(frozen=True, slots=True)
class ContainmentViolation:
    """A child lifecycle extending beyond its declared parent's interval."""

    family: str
    operation_id: str
    parent_family: str
    parent_operation_id: str
    start_sequence: int
    finish_sequence: int


@dataclass(frozen=True, slots=True)
class Metric:
    """One value and the evidence state under which it may be displayed."""

    value: int | float | None
    precision: Precision
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TimeSlice:
    """One projected timeline slice, already clipped to its owning turn."""

    family: str
    slice_index: int
    turn_id: str
    runtime_id: str
    operation_id: str | None
    owner: str
    start_ns: int
    end_ns: int
    wall_bucket: WallBucket | None
    counts_as_work: bool
    compute_weight: bool
    response_weight: bool
    outcome: str | None = None
    tool_name: str | None = None
    tool_kind: str | None = None

    @property
    def slice_id(self) -> str:
        """Derived, not stored: a private string per retained slice is what
        the residency ceiling notices first."""
        if self.operation_id is None:
            return f"{self.turn_id}:{self.family}:{self.slice_index}"
        return f"{self.family}:{self.operation_id}:{self.slice_index}"

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


FLOW_TERMINAL_INDEX = 0xFFFF_FFFF
"""Sentinel edge endpoint for the synthetic response terminal, which has no operations row."""


def _decode_edge_pairs(packed: bytes) -> tuple[tuple[int, int], ...]:
    values = array("I", packed)
    return tuple((values[index], values[index + 1]) for index in range(0, len(values), 2))


@dataclass(frozen=True, slots=True)
class TurnFlow:
    """The typed per-turn dependency graph, index-encoded for retention.

    Edge endpoints are indices into the owning ``TurnAnalysis.operations``
    tuple, except ``FLOW_TERMINAL_INDEX`` which marks the synthetic response
    terminal. Mere sequence adjacency is never recorded as an edge; the packed
    uint32 encoding keeps thousands of retained turns inside the residency
    ceiling.
    """

    turn_id: str
    root_index: int | None
    has_terminal: bool
    # Native-endian uint32 (source, target) pairs; parent = displacing
    # containment, causal = producer-declared pointer.
    parent_pairs: bytes
    causal_pairs: bytes
    acyclic: bool

    def parent_edges(self) -> tuple[tuple[int, int], ...]:
        """Decode the displacing containment edges as (source, target) index pairs."""
        return _decode_edge_pairs(self.parent_pairs)

    def causal_edges(self) -> tuple[tuple[int, int], ...]:
        """Decode the producer-declared causal edges as (source, target) index pairs."""
        return _decode_edge_pairs(self.causal_pairs)


@dataclass(frozen=True, slots=True)
class TokenUsageSample:
    """One closed exchange's normalized usage, timestamped for counter export."""

    sequence: int
    end_ns: int | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cache_read_tokens: int | None
    cache_creation_tokens: int | None


@dataclass(frozen=True, slots=True)
class ContextSample:
    """One context revision's declared item count, timestamped for counter export."""

    sequence: int
    ns: int
    item_count: int


@dataclass(frozen=True, slots=True)
class SessionCounterSamples:
    """Per-turn counter samples computed on demand for export, never retained."""

    usage_by_turn: dict[str, tuple[TokenUsageSample, ...]]
    context_by_turn: dict[str, tuple[ContextSample, ...]]


@dataclass(frozen=True, slots=True)
class TimelineOperationDetail:
    """Sparse metadata retained only for hook or unresolved timeline rows."""

    reason: str | None = None
    diagnostic_code: TimelineDiagnosticCode | None = None
    hook_id: str | None = None

    def __post_init__(self) -> None:
        if (self.reason is None) != (self.diagnostic_code is None):
            msg = "timeline operation reason and diagnostic code must be set together"
            raise ValueError(msg)
        if self.reason is None and self.hook_id is None:
            msg = "timeline operation detail must not be empty"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TimelineOperation:
    """One lifecycle operation rendered by the turn-scoped Timeline."""

    operation_id: str
    family: str
    depth: int
    start_ns: int | None
    end_ns: int | None
    precision: Precision
    identity: str | None = None
    detail: TimelineOperationDetail | None = None

    @property
    def reason(self) -> str | None:
        return self.detail.reason if self.detail is not None else None

    @property
    def diagnostic_code(self) -> TimelineDiagnosticCode | None:
        return self.detail.diagnostic_code if self.detail is not None else None

    @property
    def hook_id(self) -> str | None:
        return self.detail.hook_id if self.detail is not None else None

    @property
    def duration_ns(self) -> int | None:
        if self.start_ns is None or self.end_ns is None:
            return None
        return self.end_ns - self.start_ns


@dataclass(frozen=True, slots=True)
class TimelineOperationDiagnostic:
    """One structured duration diagnostic projected out of a timeline row.

    ``turn_id`` preserves the physical attempt owner while ``turn_number`` is
    the user-visible logical turn ordinal shared by folded retries.
    """

    turn_id: str
    turn_number: int | None
    operation_id: str
    family: str
    precision: Precision
    code: TimelineDiagnosticCode
    reason: str
    identity: str | None = None
    hook_id: str | None = None


@dataclass(frozen=True, slots=True)
class TurnAttemptRef:
    """Physical ownership and physical↔logical axis map for one turn attempt."""

    turn_id: str
    runtime_id: str
    is_retry: bool
    physical_axis_start_ns: int
    physical_axis_end_ns: int
    logical_axis_start_ns: int
    operation_start_index: int
    operation_end_index: int
    slice_start_index: int
    slice_end_index: int

    @property
    def duration_ns(self) -> int:
        """Non-negative logical width retained for this physical attempt."""
        return max(0, self.physical_axis_end_ns - self.physical_axis_start_ns)

    @property
    def logical_axis_end_ns(self) -> int:
        """Logical-axis end boundary of this attempt's retained window."""
        return self.logical_axis_start_ns + self.duration_ns

    def contains_physical_ns(self, value: int) -> bool:
        """Whether a physical timestamp is proven inside this attempt's window."""
        return self.physical_axis_start_ns <= value <= self.physical_axis_end_ns

    def to_logical_ns(self, physical_ns: int) -> int:
        """Map one physical monotonic timestamp onto the folded logical axis."""
        return physical_ns - self.physical_axis_start_ns + self.logical_axis_start_ns

    def to_physical_ns(self, logical_ns: int) -> int:
        """Map one folded logical timestamp back to this attempt's runtime axis."""
        return logical_ns - self.logical_axis_start_ns + self.physical_axis_start_ns


@dataclass(frozen=True, slots=True)
class TurnAnalysis:
    """All P0 metrics and slices for one turn."""

    turn_id: str
    turn_number: int | None
    runtime_id: str
    start_sequence: int
    end_sequence: int | None
    elapsed_ns: Metric
    compute_cp_ns: Metric
    response_cp_ns: Metric
    exclusive_work_ns: Metric
    parallelism: Metric
    overlap_gain_ns: Metric
    wall_time_ns: dict[WallBucket, Metric]
    utilization: dict[WallBucket, Metric]
    usage_tokens: Metric
    attempts: tuple[TurnAttemptRef, ...]
    axis_start_ns: int = 0
    axis_end_ns: int = 0
    operations: tuple[TimelineOperation, ...] = ()
    slices: tuple[TimeSlice, ...] = ()
    diagnostics: tuple[str, ...] = ()
    action_counts: dict[ActionClass, Metric] = field(default_factory=dict)
    critical_tool_contributions_ns: dict[str, int] = field(default_factory=dict)
    server_critical_contributions_ns: dict[str, int] = field(default_factory=dict)
    action_projection_precision: Precision = Precision.EXACT
    action_projection_reason: str | None = None
    token_usage: TokenUsage | None = None
    flow: TurnFlow | None = None

    def owns_turn_id(self, turn_id: str) -> bool:
        """Whether *turn_id* is this logical turn or one of its physical attempts."""
        return self.turn_id == turn_id or any(attempt.turn_id == turn_id for attempt in self.attempts)

    def attempt(self, turn_id: str) -> TurnAttemptRef | None:
        """Return the physical attempt carrying *turn_id*, if this turn owns it."""
        return next((attempt for attempt in self.attempts if attempt.turn_id == turn_id), None)


@dataclass(frozen=True, slots=True)
class ActionOperation:
    """One compact tool action retained after live-history projection."""

    evidence_key: str | None
    occurrence_id: str
    operation_id: str
    turn_id: str
    turn_number: int | None
    tool_name: str | None
    tool_kind: str | None
    call_item_id: str | None
    argument_fingerprint: str | None
    classification: ActionClass
    classification_precision: Precision
    classification_reason: str | None
    outcome: str | None
    outcome_precision: Precision
    start_sequence: int
    start_ns: int
    end_ns: int | None
    end_sequence: int | None = None
    outcome_reason: str | None = None
    server_name: str | None = None
    remote_name: str | None = None
    skill_name: str | None = None
    skill_revision: str | None = None
    script_name: str | None = None
    resource_name: str | None = None
    exit_code: int | None = None
    payload_observed: bool = False
    payload_bytes: int | None = None
    payload_token_estimate: int | None = None
    payload_original_bytes: int | None = None
    payload_truncated: bool | None = None
    payload_spilled: bool | None = None

    @property
    def duration_ns(self) -> int | None:
        if self.end_ns is None:
            return None
        return self.end_ns - self.start_ns


def verify_covers_edit(edit: ActionOperation, verify: ActionOperation) -> bool:
    """True when *verify* began only after *edit*'s terminal landed.

    Batched tool calls run concurrently, so start order alone proves
    nothing about what the verify observed; an edit whose terminal never
    landed can never be vouched for.
    """
    return edit.end_sequence is not None and verify.start_sequence > edit.end_sequence


@dataclass(frozen=True, slots=True)
class ActionFunnel:
    """Search/read/edit/verify counts with classification precision."""

    search: Metric
    read: Metric
    edit: Metric
    verify: Metric

    def metric(self, action: ActionClass) -> Metric:
        """Return the displayed metric for one funnel action."""
        return {
            ActionClass.SEARCH: self.search,
            ActionClass.READ: self.read,
            ActionClass.EDIT: self.edit,
            ActionClass.VERIFY: self.verify,
        }[action]


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """The §5.2 validation-loop metrics for the active projection."""

    funnel: ActionFunnel
    time_to_first_edit_ns: Metric
    first_edit_to_first_verify_ns: Metric
    edit_verify_cycle_count: Metric
    edit_verify_cycle_median_ns: Metric
    unverified_change_count: Metric
    net_zero_churn_count: Metric
    repeated_failure_signature_count: Metric
    failure_recovery_median_ns: Metric
    tool_failure_count: Metric
    tool_count: Metric
    retry_amplification_tokens: Metric


@dataclass(frozen=True, slots=True)
class FindingRow:
    """Compact renderer row with separate stable and projection-local identities."""

    evidence_key: str
    occurrence_id: str
    rule_id: str
    severity: FindingSeverity
    deterministic: bool
    detail_args: tuple[tuple[str, int], ...]
    precision: Precision
    turn_id: str | None = None
    turn_number: int | None = None
    operation_id: str | None = None


@dataclass(frozen=True, slots=True)
class SubmissionLatencySample:
    """One pre-turn preparation scope; samples are never additive."""

    scope_operation_id: str
    outcome: str | None
    bucket: SubmissionLatencyBucket
    start_sequence: int
    start_ns: int | None
    end_ns: int | None
    duration_ns: Metric
    turn_id: str | None
    finished_count: int


@dataclass(frozen=True, slots=True)
class SubmissionLatencyStats:
    """One outcome bucket's count and non-additive percentile summary."""

    bucket: SubmissionLatencyBucket
    sample_count: int
    unresolved_count: int
    p50_ns: Metric
    p90_ns: Metric
    max_ns: Metric
    samples: tuple[SubmissionLatencySample, ...]


@dataclass(frozen=True, slots=True)
class SubmissionLatencyOverview:
    """All three frozen submission-latency buckets."""

    buckets: tuple[SubmissionLatencyStats, ...]


@dataclass(frozen=True, slots=True)
class ChangeVerificationRow:
    """One file row sourced from session mutation state."""

    path: str
    state: ChangeVerificationState
    last_change_turn: int
    precision: Precision
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChangeVerification:
    """Per-file detail or the count-only mutation-summary fallback."""

    detail_available: bool
    detection_truncated: bool
    files_touched: Metric
    created: Metric
    modified: Metric
    deleted: Metric
    net_zero: Metric
    rows: tuple[ChangeVerificationRow, ...] = ()


@dataclass(frozen=True, slots=True)
class TrajectoryOverview:
    """Selection aggregate; P0 selects all resolved live-history turns."""

    elapsed_ns: Metric
    compute_cp_ns: Metric
    response_cp_ns: Metric
    exclusive_work_ns: Metric
    parallelism: Metric
    overlap_gain_ns: Metric
    wall_time_ns: dict[WallBucket, Metric]
    utilization: dict[WallBucket, Metric]
    usage_tokens: Metric


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Selection-wide normalized usage split into display buckets."""

    buckets: dict[UsageBucket, Metric]


@dataclass(frozen=True, slots=True)
class NamedCountRow:
    """One name-grouped occurrence count inside a usage panel."""

    name: str
    count: int


@dataclass(frozen=True, slots=True)
class ToolUsagePanel:
    """Occurrence counts for one tool family, grouped by display name."""

    total: int
    rows: tuple[NamedCountRow, ...]
    unattributed: int = 0
    precision: Precision = Precision.EXACT
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ToolInsightRow:
    """One ``(tool_kind, tool_name)`` row in the Tools insight."""

    tool_kind: str
    tool_name: str | None
    calls: int
    duration_share: Metric
    p50_ns: Metric
    p95_ns: Metric
    outcomes: tuple[NamedCountRow, ...]


@dataclass(frozen=True, slots=True)
class ToolInsights:
    """Session-wide tool timing and outcome groups."""

    total: int
    rows: tuple[ToolInsightRow, ...]
    unclassified: int = 0
    precision: Precision = Precision.EXACT
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class McpRemoteRow:
    """One statically attributed remote tool below an MCP server."""

    remote_name: str | None
    calls: int
    p50_ns: Metric
    p95_ns: Metric
    outcomes: tuple[NamedCountRow, ...]


@dataclass(frozen=True, slots=True)
class McpServerRow:
    """One structured MCP-server group and its presentation metrics."""

    server_name: str
    calls: int
    duration_share: Metric
    p50_ns: Metric
    p95_ns: Metric
    outcomes: tuple[NamedCountRow, ...]
    approval_blocking_share: Metric
    result_bytes: Metric
    result_tokens: Metric
    truncated_count: Metric
    spill_count: Metric
    critical_path_exclusive_ns: Metric
    connection_wait_count: Metric
    connection_wait_ns: Metric
    remotes: tuple[McpRemoteRow, ...]


@dataclass(frozen=True, slots=True)
class McpInsights:
    """MCP calls grouped strictly by structured server provenance."""

    total: int
    rows: tuple[McpServerRow, ...]
    unattributed: int = 0
    unattributed_connection_waits: int = 0
    precision: Precision = Precision.EXACT
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SkillActivityRow:
    """One script or resource child row below a skill."""

    name: str | None
    count: int
    outcomes: tuple[NamedCountRow, ...] = ()
    exit_codes: tuple[NamedCountRow, ...] = ()


@dataclass(frozen=True, slots=True)
class SkillInsightRow:
    """One canonical skill usage group."""

    skill_name: str
    load_count: int
    turn_count: int
    first_action_median_ns: Metric
    script_count: int
    script_outcomes: tuple[NamedCountRow, ...]
    script_exit_codes: tuple[NamedCountRow, ...]
    resource_count: int
    injected_tokens: Metric
    revisions: tuple[str, ...]
    scripts: tuple[SkillActivityRow, ...]
    resources: tuple[SkillActivityRow, ...]


@dataclass(frozen=True, slots=True)
class SkillInsights:
    """Skill usage grouped strictly by canonical structured provenance."""

    total: int
    rows: tuple[SkillInsightRow, ...]
    not_found: tuple[NamedCountRow, ...] = ()
    unattributed: int = 0
    precision: Precision = Precision.EXACT
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ContextCarryingLoad:
    """One stable context item ranked by estimated token-revision load.

    ``load`` is ``token_count`` multiplied by ``carry_count``, the number of
    non-side-call model requests whose context included the item; ``turn_number``
    names the last carrying turn and ``origin_turn_number`` the first.  ``role``
    and ``tool_names`` describe the session message the item maps to, when the
    session projection could identify it.
    """

    load: int
    item_id: str
    occurrence_id: str
    turn_id: str | None
    turn_number: int | None
    token_count: int = 0
    carry_count: int = 0
    origin_turn_number: int | None = None
    role: str | None = None
    tool_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InsightsAnalysis:
    """All aggregated data consumed by the four non-Findings insight pages."""

    tools: ToolInsights
    mcp: McpInsights
    skills: SkillInsights
    context_carrying_load: tuple[ContextCarryingLoad, ...] = ()
    context_carrying_precision: Precision = Precision.EXACT
    context_carrying_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SessionSpan:
    """Wall-clock anchors the events log records for the whole session.

    ``occurred_at`` is display-only by contract, so these stay the producer's
    RFC 3339 UTC strings; presentation converts them to the viewer's clock.
    """

    first_turn_started_at: str | None = None
    last_turn_finished_at: str | None = None
    runtime_count: int = 0


@dataclass(frozen=True, slots=True)
class TrajectoryDiagnostics:
    """Reader and resolver diagnostics retained without event materialization."""

    line_count: int = 0
    byte_count: int = 0
    torn_tail_bytes: int = 0
    corrupt_line_count: int = 0
    unsupported_event_count: int = 0
    accounted_prefix_violations: tuple[str, ...] = ()
    accounted_prefix_violation_details: tuple[SequenceRangeDiagnostic, ...] = ()
    explicit_gap_count: int = 0
    explicit_gaps: tuple[SequenceRangeDiagnostic, ...] = ()
    rollback_projection_unresolved: bool = False
    span_duration_mismatch_count: int = 0
    span_duration_mismatches: tuple[SpanDurationMismatch, ...] = ()
    containment_violation_count: int = 0
    containment_violations: tuple[ContainmentViolation, ...] = ()
    malformed_hook_execution_mode_count: int = 0
    side_call_empty_shell_revisions: tuple[str, ...] = ()
    unidentified_membership_revision_count: int = 0
    corrupt_lines: tuple[CorruptLineDiagnostic, ...] = ()
    unsupported_lines: tuple[UnsupportedLineDiagnostic, ...] = ()
    timeline_operations: tuple[TimelineOperationDiagnostic, ...] = ()

    @property
    def integrity_unresolved(self) -> bool:
        """Whether source damage can make the session projection incomplete.

        This predicate presumes the diagnostics belong to an ``AVAILABLE``
        analysis. The default diagnostics carried by ``UNAVAILABLE`` and
        ``READ_ERROR`` analyses also have the empty-source signature.
        """

        return bool(
            (self.line_count == 0 and self.byte_count == 0 and self.torn_tail_bytes == 0)
            or self.corrupt_line_count
            or self.unsupported_event_count
            or self.accounted_prefix_violations
            or self.explicit_gap_count
            or self.torn_tail_bytes
            or self.rollback_projection_unresolved
        )


@dataclass(frozen=True, slots=True)
class WorkflowRunAnalysis:
    """One Workflow execution, with activation attempts and nested agent operations."""

    run_id: str
    workflow_id: str
    runtime_id: str
    outcome: str | None
    elapsed_ns: Metric
    operations: tuple[TimelineOperation, ...]


@dataclass(frozen=True, slots=True)
class TrajectoryAnalysis:
    """Top-level P0 view model; independent of Textual and JSON serializable by fields."""

    availability: AnalysisAvailability
    path: Path
    generation: int
    overview: TrajectoryOverview | None = None
    turns: tuple[TurnAnalysis, ...] = ()
    workflow_runs: tuple[WorkflowRunAnalysis, ...] = ()
    diagnostics: TrajectoryDiagnostics = field(default_factory=TrajectoryDiagnostics)
    read_error: str | None = None
    actions: tuple[ActionOperation, ...] = ()
    validation: ValidationMetrics | None = None
    findings: tuple[FindingRow, ...] = ()
    submission_latency: SubmissionLatencyOverview | None = None
    change_verification: ChangeVerification | None = None
    token_usage: TokenUsage | None = None
    skill_usage: ToolUsagePanel | None = None
    mcp_usage: ToolUsagePanel | None = None
    insights: InsightsAnalysis | None = None
    session_span: SessionSpan | None = None

    def turn(self, turn_id: str) -> TurnAnalysis | None:
        """Return one resolved turn without maintaining a duplicate index."""
        return next((turn for turn in self.turns if turn.owns_turn_id(turn_id)), None)
