# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The trajectory dashboard's Insights page: tool, MCP and skill activity,
per-turn token load, findings and the diagnostics wall."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from rich.style import Style
from rich.text import Text

from chrys.app.tui.widgets.trajectory.chartkit import percentage_meter, section_interior_width
from chrys.app.tui.widgets.trajectory.presentation import (
    BUCKET_IDLE,
    BUCKET_MODEL,
    BUCKET_TOOLS,
    BUCKET_WAIT,
    METRIC_CP_COMPUTE,
    METRIC_CP_RESPONSE,
    METRIC_ELAPSED,
    METRIC_OVERLAP,
    METRIC_PARALLELISM,
    METRIC_USAGE,
    METRIC_WORK,
    PRECISION_SYMBOLS,
    TOKEN_CACHE_HIT,
    TOKEN_CACHE_READ,
    TOKEN_INPUT,
    TOKEN_OUTPUT,
    TOKEN_REASONING,
    TOOL_USAGE_MORE,
    TURN_LABEL,
    DashboardLook,
    RenderContext,
    ResponsiveTier,
    align_edges,
    badged_section_title,
    bordered_section_row,
    cache_hit_metric,
    callsite,
    derived_metric_precision,
    format_duration,
    format_tokens,
    identity_with_hook_id,
    precision_badge,
    precision_label,
    precision_style,
    section_box,
    section_row_widths,
    section_style,
)
from chrys.foundation.i18n import MessageDef, MessageRef, msg
from chrys.service.analytics import (
    ContextCarryingLoad,
    FindingRow,
    FindingSeverity,
    McpServerRow,
    Metric,
    NamedCountRow,
    Precision,
    SkillInsightRow,
    TimelineDiagnosticCode,
    TimelineOperationDiagnostic,
    TokenUsage,
    ToolInsightRow,
    TrajectoryAnalysis,
    TrajectoryDiagnostics,
    TrajectoryOverview,
    TurnAnalysis,
    UsageBucket,
    WallBucket,
)

if TYPE_CHECKING:
    from collections.abc import Callable


_DIAGNOSTICS = msg("tui.trajectory.diagnostics.title", fallback="Diagnostics")
_DIAGNOSTICS_INTRO = msg(
    "tui.trajectory.diagnostics.intro",
    fallback="Data-integrity notes for this session's events log.",
)
_DIAGNOSTICS_HEALTHY = msg(
    "tui.trajectory.diagnostics.healthy",
    fallback="The events log is intact; no problems were found.",
)
_CORRUPT_LINES = msg(
    "tui.trajectory.diagnostics.corrupt",
    fallback="Corrupt line {count} ({sequences}); metrics in the affected range degrade to unresolved.",
    plural_fallback="Corrupt lines {count} ({sequences}); metrics in the affected ranges degrade to unresolved.",
)
_UNSUPPORTED_LINES = msg(
    "tui.trajectory.diagnostics.unsupported",
    fallback="Unsupported line {count} ({sequences}); metrics in the affected range degrade to unresolved.",
    plural_fallback="Unsupported lines {count} ({sequences}); metrics in the affected ranges degrade to unresolved.",
)
_AFTER_SEQUENCE = msg("tui.trajectory.diagnostics.after_sequence", fallback="after seq {sequence}")
_SEQUENCE = msg("tui.trajectory.diagnostics.sequence", fallback="seq {sequence}")
_ACCOUNTED_PREFIX = msg(
    "tui.trajectory.diagnostics.accounted_prefix",
    fallback="Accounted-prefix seq {first}-{last}: {reason}",
)
_UNRESOLVED_METRIC = msg(
    "tui.trajectory.diagnostics.unresolved_metric",
    fallback="{metric} unresolved: {reason}",
)
_OPERATION_DIAGNOSTIC = msg(
    "tui.trajectory.diagnostics.operation",
    fallback="Turn {turn} · {operation}: {reason}",
)
_OPERATION_REASON_DETACHED_HOOK = msg(
    "tui.trajectory.diagnostics.operation.detached_hook",
    fallback="detached hook records spawn latency, not work duration",
)
_OPERATION_REASON_MISSING_START = msg(
    "tui.trajectory.diagnostics.operation.missing_start",
    fallback="lifecycle has no start endpoint",
)
_OPERATION_REASON_MISSING_TERMINAL = msg(
    "tui.trajectory.diagnostics.operation.missing_terminal",
    fallback="lifecycle has no terminal endpoint",
)
_OPERATION_REASON_NONUNIQUE = msg(
    "tui.trajectory.diagnostics.operation.nonunique",
    fallback="lifecycle is not uniquely closed",
)
_OPERATION_REASON_INVALID_ENDPOINTS = msg(
    "tui.trajectory.diagnostics.operation.invalid_endpoints",
    fallback="interval has invalid monotonic endpoints",
)
_OPERATION_REASON_OUTSIDE_COVERAGE = msg(
    "tui.trajectory.diagnostics.operation.outside_coverage",
    fallback="lifecycle falls outside the owning turn coverage",
)
_OPERATION_REASON_ROLLBACK_START = msg(
    "tui.trajectory.diagnostics.operation.rollback_start",
    fallback="lifecycle crosses rollback projection; only its start endpoint remains active",
)
_OPERATION_REASON_ROLLBACK_TERMINAL = msg(
    "tui.trajectory.diagnostics.operation.rollback_terminal",
    fallback="lifecycle crosses rollback projection; only its terminal endpoint remains active",
)
_OPERATION_REASON_MESSAGES = {
    TimelineDiagnosticCode.DETACHED_HOOK: _OPERATION_REASON_DETACHED_HOOK,
    TimelineDiagnosticCode.MISSING_START: _OPERATION_REASON_MISSING_START,
    TimelineDiagnosticCode.MISSING_TERMINAL: _OPERATION_REASON_MISSING_TERMINAL,
    TimelineDiagnosticCode.NONUNIQUE_LIFECYCLE: _OPERATION_REASON_NONUNIQUE,
    TimelineDiagnosticCode.INVALID_ENDPOINTS: _OPERATION_REASON_INVALID_ENDPOINTS,
    TimelineDiagnosticCode.OUTSIDE_TURN_COVERAGE: _OPERATION_REASON_OUTSIDE_COVERAGE,
    TimelineDiagnosticCode.ROLLBACK_START_SURVIVES: _OPERATION_REASON_ROLLBACK_START,
    TimelineDiagnosticCode.ROLLBACK_TERMINAL_SURVIVES: _OPERATION_REASON_ROLLBACK_TERMINAL,
}
_DURATION_MISMATCH_SUMMARY = msg(
    "tui.trajectory.diagnostics.duration_mismatch_summary",
    fallback=(
        "{value} span's recorded duration drifts from its lifecycle interval ({families}; up to {delta})"
        " — write-through acknowledgement and scheduling jitter; every metric uses the lifecycle interval."
    ),
    plural_fallback=(
        "{value} spans' recorded durations drift from their lifecycle intervals ({families}; up to {delta})"
        " — write-through acknowledgement and scheduling jitter; every metric uses the lifecycle interval."
    ),
)
_CONTAINMENT = msg(
    "tui.trajectory.diagnostics.containment",
    fallback="Containment {family} @{callsite} outside {parent_family} @{parent_callsite}",
)
_TORN_TAIL = msg("tui.trajectory.diagnostics.torn_tail", fallback="Torn tail: {bytes} bytes")
_EXPLICIT_GAP = msg(
    "tui.trajectory.diagnostics.gap",
    fallback="Trajectory gap seq {first}-{last}: {reason}",
)
_ROLLBACK_UNRESOLVED = msg(
    "tui.trajectory.diagnostics.rollback",
    fallback="Rollback live-history projection is unresolved.",
)
_MALFORMED_HOOK_MODES = msg(
    "tui.trajectory.diagnostics.hook_modes",
    fallback="Malformed hook execution mode: {count}",
    plural_fallback="Malformed hook execution modes: {count}",
)
_SIDE_CALL_EMPTY_SHELLS = msg(
    "tui.trajectory.diagnostics.side_call_empty_shells",
    fallback=(
        "{value} side call (title generation, approval judging, etc.) recorded an empty"
        " context snapshot — a known benign shape, excluded from all metrics."
    ),
    plural_fallback=(
        "{value} side calls (title generation, approval judging, etc.) recorded empty"
        " context snapshots — a known benign shape, excluded from all metrics."
    ),
)
_UNIDENTIFIED_MEMBERSHIP = msg(
    "tui.trajectory.diagnostics.unidentified_membership",
    fallback=(
        "{value} context snapshot carried an item without analytics identity (e.g. a sub-agent's"
        " seed prompt); token re-send cost is unknown for such items, timing is unaffected."
    ),
    plural_fallback=(
        "{value} context snapshots carried items without analytics identity (e.g. sub-agent"
        " seed prompts); token re-send cost is unknown for such items, timing is unaffected."
    ),
)
_WALL_METRIC = msg("tui.trajectory.diagnostics.wall_metric", fallback="{bucket} wall time")
_UTILIZATION_METRIC = msg(
    "tui.trajectory.diagnostics.utilization_metric",
    fallback="{bucket} busy share",
)
_FINDINGS = msg("tui.trajectory.findings.title", fallback="Findings")
_NO_FINDINGS = msg("tui.trajectory.findings.none", fallback="No active findings.")
_FINDING_TITLE_UNVERIFIED = msg(
    "tui.trajectory.finding.unverified_change.title",
    fallback="Unverified change",
)
_FINDING_DETAIL_UNVERIFIED = msg(
    "tui.trajectory.finding.unverified_change.detail",
    fallback="{count} edit action occurred after the last successful verification.",
    plural_fallback="{count} edit actions occurred after the last successful verification.",
)
_FINDING_TITLE_REPEATED = msg(
    "tui.trajectory.finding.repeated_tool_fingerprint.title",
    fallback="Repeated tool fingerprint",
)
_FINDING_DETAIL_REPEATED = msg(
    "tui.trajectory.finding.repeated_tool_fingerprint.detail",
    fallback="The same argument fingerprint occurred {count} time.",
    plural_fallback="The same argument fingerprint occurred {count} times.",
)
_FINDING_TITLE_FAILED_CP = msg(
    "tui.trajectory.finding.failed_attempt_critical_path.title",
    fallback="Failed attempt dominates the critical path",
)
_FINDING_DETAIL_FAILED_CP = msg(
    "tui.trajectory.finding.failed_attempt_critical_path.detail",
    fallback="One failed tool attempt accounts for {percentage}% of response critical-path time.",
)
_FINDING_TITLE_RETRY_AMPLIFICATION = msg(
    "tui.trajectory.finding.retry_token_amplification.title",
    fallback="Retry token amplification",
)
_FINDING_DETAIL_RETRY_AMPLIFICATION = msg(
    "tui.trajectory.finding.retry_token_amplification.detail",
    fallback="Retries added {tokens} normalized token.",
    plural_fallback="Retries added {tokens} normalized tokens.",
)
_FINDING_TITLE_NET_ZERO = msg(
    "tui.trajectory.finding.net_zero_churn.title",
    fallback="Changes cancelled out",
)
_FINDING_DETAIL_NET_ZERO = msg(
    "tui.trajectory.finding.net_zero_churn.detail",
    fallback="{count} file returned to its original state.",
    plural_fallback="{count} files returned to their original state.",
)
_FINDING_TITLE_APPROVAL_SHARE = msg(
    "tui.trajectory.finding.approval_blocking_share.title",
    fallback="Approval blocking share is high",
)
_FINDING_DETAIL_APPROVAL_SHARE = msg(
    "tui.trajectory.finding.approval_blocking_share.detail",
    fallback="Approval waits occupied {percentage}% of this turn.",
)
_FINDING_TITLE_CONTEXT_LOAD = msg(
    "tui.trajectory.finding.context_carrying_load.title",
    fallback="Heavily re-sent context item",
)
_FINDING_DETAIL_CONTEXT_LOAD = msg(
    "tui.trajectory.finding.context_carrying_load.detail",
    fallback="One context item cost an estimated {load} token across the model requests that re-sent it.",
    plural_fallback="One context item cost an estimated {load} tokens across the model requests that re-sent it.",
)
_FINDING_TITLES = {
    "unverified-change": _FINDING_TITLE_UNVERIFIED,
    "repeated-tool-fingerprint": _FINDING_TITLE_REPEATED,
    "failed-attempt-critical-path": _FINDING_TITLE_FAILED_CP,
    "retry-token-amplification": _FINDING_TITLE_RETRY_AMPLIFICATION,
    "net-zero-churn": _FINDING_TITLE_NET_ZERO,
    "approval-blocking-share": _FINDING_TITLE_APPROVAL_SHARE,
    "context-carrying-load": _FINDING_TITLE_CONTEXT_LOAD,
}
_FINDING_DETAILS = {
    "unverified-change": _FINDING_DETAIL_UNVERIFIED,
    "repeated-tool-fingerprint": _FINDING_DETAIL_REPEATED,
    "failed-attempt-critical-path": _FINDING_DETAIL_FAILED_CP,
    "retry-token-amplification": _FINDING_DETAIL_RETRY_AMPLIFICATION,
    "net-zero-churn": _FINDING_DETAIL_NET_ZERO,
    "approval-blocking-share": _FINDING_DETAIL_APPROVAL_SHARE,
    "context-carrying-load": _FINDING_DETAIL_CONTEXT_LOAD,
}
_TOKEN_CACHE_CREATION = msg("tui.trajectory.token_usage.cache_creation", fallback="cache creation")
_INSIGHTS_TOOLS = msg("tui.trajectory.insights.tools.title", fallback="Tool activity")
_INSIGHTS_INTEGRATIONS_MCP = msg("tui.trajectory.insights.integrations.mcp", fallback="MCP servers")
_INSIGHTS_INTEGRATIONS_SKILLS = msg(
    "tui.trajectory.insights.integrations.skills",
    fallback="Skills",
)
_INSIGHTS_NO_TOOLS = msg(
    "tui.trajectory.insights.tools.none",
    fallback="This session has no tool calls.",
)
_INSIGHTS_NO_MCP = msg(
    "tui.trajectory.insights.integrations.mcp_none",
    fallback="This session has no MCP calls.",
)
_INSIGHTS_NO_SKILLS = msg(
    "tui.trajectory.insights.integrations.skills_none",
    fallback="This session has no skill usage.",
)
_INSIGHTS_NO_TURN_TOKENS = msg(
    "tui.trajectory.insights.tokens.no_turn_data",
    fallback="No per-turn data.",
)
_INSIGHTS_PER_TURN_TOKENS = msg(
    "tui.trajectory.insights.tokens.per_turn",
    fallback="Tokens per turn",
)
_INSIGHTS_CARRYING_LOAD = msg(
    "tui.trajectory.insights.tokens.carrying_load",
    fallback="Context re-send cost · top {value}",
)
_INSIGHTS_CARRYING_EXPLAINER = msg(
    "tui.trajectory.insights.tokens.carrying_explainer",
    fallback="tokens × model requests that re-sent the item",  # noqa: RUF001
)
_INSIGHTS_CARRYING_ROW_HEAD = msg(
    "tui.trajectory.insights.tokens.carrying_row_head",
    fallback="{item} · since turn {turn}",
)
_INSIGHTS_CARRYING_ROW_DETAIL = msg(
    "tui.trajectory.insights.tokens.carrying_row_detail",
    fallback="{tokens} tok × {carries} re-sends",  # noqa: RUF001
)
_INSIGHTS_CARRYING_USER = msg("tui.trajectory.insights.tokens.carrying_user", fallback="user message")
_INSIGHTS_CARRYING_ASSISTANT = msg("tui.trajectory.insights.tokens.carrying_assistant", fallback="assistant message")
_INSIGHTS_CARRYING_TOOL_RESULT = msg("tui.trajectory.insights.tokens.carrying_tool_result", fallback="tool result")
_INSIGHTS_CARRYING_ITEM = msg("tui.trajectory.insights.tokens.carrying_item", fallback="context item")
_INSIGHTS_INPUT_SHARE = msg(
    "tui.trajectory.insights.tokens.input_share",
    fallback="input share",
)
_INSIGHTS_UNATTRIBUTED = msg(
    "tui.trajectory.insights.status.unattributed",
    fallback="Unresolved attribution: {value}",
)
_INSIGHTS_UNCLASSIFIED = msg(
    "tui.trajectory.insights.status.unclassified",
    fallback="Unclassified: {value}",
)
_INSIGHTS_SKILL_CHANGED = msg(
    "tui.trajectory.insights.status.skill_changed",
    fallback="Skill changed during the session",
)
_INSIGHTS_SKILL_NOT_FOUND = msg(
    "tui.trajectory.insights.status.skill_not_found",
    fallback="Skill not found (configuration issue)",
)
_INSIGHTS_CALLS = msg("tui.trajectory.insights.column.calls", fallback="calls")
_INSIGHTS_DURATION_SHARE = msg("tui.trajectory.insights.column.duration_share", fallback="time share")
_INSIGHTS_P50 = msg("tui.trajectory.insights.column.p50", fallback="p50")
_INSIGHTS_P95 = msg("tui.trajectory.insights.column.p95", fallback="p95")
_INSIGHTS_RESULTS = msg("tui.trajectory.insights.column.results", fallback="results")
_INSIGHTS_APPROVAL = msg("tui.trajectory.insights.column.approval", fallback="approval blocking")
_INSIGHTS_RETURN_VOLUME = msg("tui.trajectory.insights.column.return_volume", fallback="return volume")
_INSIGHTS_TRUNCATED_SPILL = msg(
    "tui.trajectory.insights.column.truncated_spill",
    fallback="truncated · spilled",
)
_INSIGHTS_CP_EXCLUSIVE = msg(
    "tui.trajectory.insights.column.cp_exclusive",
    fallback="critical-path exclusive contribution",
)
_INSIGHTS_CONNECTION_WAIT = msg(
    "tui.trajectory.insights.column.connection_wait",
    fallback="connection wait",
)
_INSIGHTS_OTEL_CROSS_CHECK = msg(
    "tui.trajectory.insights.column.otel_cross_check",
    fallback="OTel cross-check",
)
_INSIGHTS_LOADS = msg("tui.trajectory.insights.column.loads", fallback="loads")
_INSIGHTS_TURNS = msg("tui.trajectory.insights.column.turns", fallback="turns")
_INSIGHTS_FIRST_ACTION = msg(
    "tui.trajectory.insights.column.first_action",
    fallback="median time to first action",
)
_INSIGHTS_SCRIPT_RUNS = msg("tui.trajectory.insights.column.script_runs", fallback="script runs")
_INSIGHTS_RESOURCE_READS = msg("tui.trajectory.insights.column.resource_reads", fallback="resource reads")
_INSIGHTS_INJECTED_TOKENS = msg("tui.trajectory.insights.column.injected_tokens", fallback="injected tokens")
_INSIGHTS_REVISIONS = msg("tui.trajectory.insights.column.revisions", fallback="revision")
_INSIGHTS_EXIT_CODES = msg("tui.trajectory.insights.column.exit_codes", fallback="exit codes")

# Side-by-side boxes pad the shorter one to the taller one's height; past this
# gap the dead space outweighs the width saving and the pair stacks full-width.
_SECTION_PAIR_MAX_HEIGHT_GAP = 10


def has_diagnostic_content(diagnostics: TrajectoryDiagnostics) -> bool:
    """Whether the Insights diagnostic wall has session-specific content to show."""

    return bool(
        diagnostics.torn_tail_bytes
        or diagnostics.corrupt_line_count
        or diagnostics.corrupt_lines
        or diagnostics.unsupported_event_count
        or diagnostics.unsupported_lines
        or diagnostics.accounted_prefix_violations
        or diagnostics.accounted_prefix_violation_details
        or diagnostics.explicit_gap_count
        or diagnostics.explicit_gaps
        or diagnostics.rollback_projection_unresolved
        or diagnostics.span_duration_mismatch_count
        or diagnostics.span_duration_mismatches
        or diagnostics.containment_violation_count
        or diagnostics.containment_violations
        or diagnostics.malformed_hook_execution_mode_count
        or diagnostics.timeline_operations
        or diagnostics.side_call_empty_shell_revisions
        or diagnostics.unidentified_membership_revision_count
    )


def insights_lines(look: DashboardLook, context: RenderContext, analysis: TrajectoryAnalysis) -> list[Text]:
    width = context.width
    tier = context.tier
    interior = section_interior_width(width)
    insights = analysis.insights
    carrying = (
        sorted(
            insights.context_carrying_load,
            key=lambda row: (row.load, row.item_id),
            reverse=True,
        )[:5]
        if insights is not None
        else []
    )
    skills_precision = insights.skills.precision if insights is not None else Precision.MISSING
    mcp_precision = insights.mcp.precision if insights is not None else Precision.MISSING
    tools_precision = insights.tools.precision if insights is not None else Precision.MISSING
    carrying_precision = insights.context_carrying_precision if insights is not None else Precision.MISSING
    # Row order: the two integration summaries first, then the tool and
    # context-cost details, the per-turn token table, and the findings
    # and diagnostics wall last so the log-integrity notes close the page.
    lines: list[Text] = []
    lines.extend(
        _adaptive_section_pair(
            look,
            badged_section_title(look, _INSIGHTS_INTEGRATIONS_SKILLS, skills_precision),
            lambda box_width: _skills_insight_lines(look, analysis, width=box_width),
            badged_section_title(look, _INSIGHTS_INTEGRATIONS_MCP, mcp_precision),
            lambda box_width: _mcp_insight_lines(look, analysis, width=box_width),
            width=width,
            paired=tier is ResponsiveTier.WIDE,
            balanced=False,
        )
    )
    lines.extend(
        _adaptive_section_pair(
            look,
            badged_section_title(look, _INSIGHTS_TOOLS, tools_precision),
            lambda box_width: _tool_activity_lines(look, context, analysis, width=box_width),
            badged_section_title(look, _INSIGHTS_CARRYING_LOAD.bind(value=5), carrying_precision),
            lambda box_width: _carrying_load_lines(look, context, carrying, width=box_width),
            width=width,
            paired=tier is ResponsiveTier.WIDE,
            balanced=False,
        )
    )
    lines.extend(
        section_box(
            look,
            _INSIGHTS_PER_TURN_TOKENS,
            _per_turn_token_content(look, context, analysis, width=interior),
            width=width,
        )
    )
    lines.extend(
        _adaptive_section_pair(
            look,
            _FINDINGS,
            lambda _: _finding_lines(look, analysis),
            _DIAGNOSTICS,
            lambda _: _diagnostic_lines(look, analysis),
            width=width,
            paired=tier in {ResponsiveTier.WIDE, ResponsiveTier.MID},
        )
    )
    return lines


def _adaptive_section_pair(
    look: DashboardLook,
    left_title: MessageDef | MessageRef | Text,
    build_left: Callable[[int], list[Text]],
    right_title: MessageDef | MessageRef | Text,
    build_right: Callable[[int], list[Text]],
    *,
    width: int,
    paired: bool,
    balanced: bool = True,
) -> list[Text]:
    """Lay two sections side by side, or stack them full-width.

    *balanced* pairs keep the side-by-side layout only while the two
    boxes stay within the height-gap threshold; an unbalanced pair stays
    side by side whenever *paired* holds, accepting the padding.
    """
    if paired:
        halves = section_row_widths(width, 2)
        left_lines = build_left(section_interior_width(halves[0]))
        right_lines = build_right(section_interior_width(halves[1]))
        left_box = section_box(look, left_title, left_lines, width=halves[0])
        right_box = section_box(look, right_title, right_lines, width=halves[1])
        if not balanced or abs(len(left_box) - len(right_box)) <= _SECTION_PAIR_MAX_HEIGHT_GAP:
            return bordered_section_row(
                look,
                (
                    (left_title, left_lines, halves[0]),
                    (right_title, right_lines, halves[1]),
                ),
            )
    interior = section_interior_width(width)
    return [
        *section_box(look, left_title, build_left(interior), width=width),
        *section_box(look, right_title, build_right(interior), width=width),
    ]


def _tool_activity_lines(
    look: DashboardLook, context: RenderContext, analysis: TrajectoryAnalysis, *, width: int
) -> list[Text]:
    insights = analysis.insights
    if insights is None or not insights.tools.total:
        return [Text(look.message(_INSIGHTS_NO_TOOLS.bind()), style="dim")]
    content = []
    visible = insights.tools.rows[:12]
    for row in visible:
        content.extend(_tool_insight_row(look, context, row, width=width))
    if len(insights.tools.rows) > len(visible):
        content.append(
            Text(
                look.message(TOOL_USAGE_MORE.bind(value=len(insights.tools.rows) - len(visible))),
                style="dim",
            )
        )
    if insights.tools.unclassified:
        content.append(
            Text(
                look.message(_INSIGHTS_UNCLASSIFIED.bind(value=insights.tools.unclassified)),
                style=look.semantic_style("warning", "yellow"),
            )
        )
    return content


def _tool_insight_row(look: DashboardLook, context: RenderContext, row: ToolInsightRow, *, width: int) -> list[Text]:
    name = f"{row.tool_kind} · {row.tool_name or '—'}"
    calls = look.message(_INSIGHTS_CALLS.bind())
    share = look.message(_INSIGHTS_DURATION_SHARE.bind())
    p50 = look.message(_INSIGHTS_P50.bind())
    p95 = look.message(_INSIGHTS_P95.bind())
    outcomes = look.message(_INSIGHTS_RESULTS.bind())
    if context.tier is ResponsiveTier.FLOOR:
        share_text = Text(_format_percentage_metric(row.duration_share))
    else:
        # Same bracket meter as the Overview operation shares, so every
        # percentage on the dashboard reads the same way.
        share_text = Text.assemble(
            percentage_meter(
                None if row.duration_share.value is None else float(row.duration_share.value) * 100,
                width=max(6, min(16, width // 4)),
                style=look.semantic_style("primary", "blue"),
            ),
            Text(" "),
            precision_badge(look, row.duration_share.precision),
        )
    return [
        align_edges(
            Text(name, style=section_style(look)),
            Text(f"{calls} {row.calls:,}"),
            width,
        ),
        align_edges(Text(share), share_text, width),
        align_edges(
            Text(f"{p50} {_format_metric_duration(row.p50_ns)} · {p95} {_format_metric_duration(row.p95_ns)}"),
            Text(f"{outcomes} {_format_named_counts(row.outcomes)}"),
            width,
        ),
    ]


def _mcp_insight_lines(look: DashboardLook, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
    insights = analysis.insights
    if insights is None:
        return [Text(look.message(_INSIGHTS_NO_MCP.bind()), style="dim")]
    mcp_lines: list[Text] = []
    if not insights.mcp.total:
        mcp_lines.append(Text(look.message(_INSIGHTS_NO_MCP.bind()), style="dim"))
    else:
        for row in insights.mcp.rows[:8]:
            mcp_lines.extend(_mcp_server_lines(look, row, width=width))
        if len(insights.mcp.rows) > 8:
            mcp_lines.append(Text(look.message(TOOL_USAGE_MORE.bind(value=len(insights.mcp.rows) - 8)), style="dim"))
    mcp_unattributed = insights.mcp.unattributed + insights.mcp.unattributed_connection_waits
    if mcp_unattributed:
        mcp_lines.append(
            Text(
                look.message(_INSIGHTS_UNATTRIBUTED.bind(value=mcp_unattributed)),
                style=look.semantic_style("warning", "yellow"),
            )
        )
    return mcp_lines


def _skills_insight_lines(look: DashboardLook, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
    insights = analysis.insights
    if insights is None:
        return [Text(look.message(_INSIGHTS_NO_SKILLS.bind()), style="dim")]
    skill_lines: list[Text] = []
    if not insights.skills.rows:
        skill_lines.append(Text(look.message(_INSIGHTS_NO_SKILLS.bind()), style="dim"))
    else:
        for row in insights.skills.rows[:10]:
            skill_lines.extend(_skill_insight_lines(look, row, width=width))
        if len(insights.skills.rows) > 10:
            skill_lines.append(
                Text(look.message(TOOL_USAGE_MORE.bind(value=len(insights.skills.rows) - 10)), style="dim")
            )
    if insights.skills.unattributed:
        skill_lines.append(
            Text(
                look.message(_INSIGHTS_UNATTRIBUTED.bind(value=insights.skills.unattributed)),
                style=look.semantic_style("warning", "yellow"),
            )
        )
    skill_lines.extend(
        (
            align_edges(
                Text(
                    look.message(_INSIGHTS_SKILL_NOT_FOUND.bind()),
                    style=look.semantic_style("error", "red"),
                ),
                Text(f"{row.name} x{row.count}"),
                width,
            )
        )
        for row in insights.skills.not_found
    )
    return skill_lines


def _mcp_server_lines(look: DashboardLook, row: McpServerRow, *, width: int) -> list[Text]:
    calls = look.message(_INSIGHTS_CALLS.bind())
    share = look.message(_INSIGHTS_DURATION_SHARE.bind())
    p50 = look.message(_INSIGHTS_P50.bind())
    p95 = look.message(_INSIGHTS_P95.bind())
    results = look.message(_INSIGHTS_RESULTS.bind())
    approval = look.message(_INSIGHTS_APPROVAL.bind())
    volume = look.message(_INSIGHTS_RETURN_VOLUME.bind())
    truncated = look.message(_INSIGHTS_TRUNCATED_SPILL.bind())
    cp = look.message(_INSIGHTS_CP_EXCLUSIVE.bind())
    connection = look.message(_INSIGHTS_CONNECTION_WAIT.bind())
    otel = look.message(_INSIGHTS_OTEL_CROSS_CHECK.bind())
    lines = [
        align_edges(
            Text(row.server_name, style=section_style(look)),
            Text(f"{calls} {row.calls:,} · {share} {_format_percentage_metric(row.duration_share)}"),
            width,
        ),
        align_edges(
            Text(f"{p50} {_format_metric_duration(row.p50_ns)} · {p95} {_format_metric_duration(row.p95_ns)}"),
            Text(f"{results} {_format_named_counts(row.outcomes)}"),
            width,
        ),
        align_edges(
            Text(f"{approval} {_format_percentage_metric(row.approval_blocking_share)}"),
            Text(f"{volume} {_format_metric_bytes(row.result_bytes)} + {_format_metric_tokens(row.result_tokens)}"),
            width,
        ),
        align_edges(
            Text(f"{truncated} {_format_metric_count(row.truncated_count)} · {_format_metric_count(row.spill_count)}"),
            Text(f"{cp} {_format_metric_duration(row.critical_path_exclusive_ns)}"),
            width,
        ),
        align_edges(
            Text(
                f"{connection} {_format_metric_count(row.connection_wait_count)} · "
                f"{_format_metric_duration(row.connection_wait_ns)}"
            ),
            Text(f"{otel} —"),
            width,
        ),
    ]
    lines.extend(
        align_edges(
            Text(f"  ↳ {remote.remote_name or '—'}", style="dim"),
            Text(
                f"{calls} {remote.calls:,} · {p50} {_format_metric_duration(remote.p50_ns)} · "
                f"{p95} {_format_metric_duration(remote.p95_ns)} · {_format_named_counts(remote.outcomes)}"
            ),
            width,
        )
        for remote in row.remotes[:5]
    )
    if len(row.remotes) > 5:
        lines.append(Text(f"  {look.message(TOOL_USAGE_MORE.bind(value=len(row.remotes) - 5))}", style="dim"))
    return lines


def _skill_insight_lines(look: DashboardLook, row: SkillInsightRow, *, width: int) -> list[Text]:
    loads = look.message(_INSIGHTS_LOADS.bind())
    turns = look.message(_INSIGHTS_TURNS.bind())
    first_action = look.message(_INSIGHTS_FIRST_ACTION.bind())
    script_runs = look.message(_INSIGHTS_SCRIPT_RUNS.bind())
    resources = look.message(_INSIGHTS_RESOURCE_READS.bind())
    injected = look.message(_INSIGHTS_INJECTED_TOKENS.bind())
    revisions = look.message(_INSIGHTS_REVISIONS.bind())
    exit_codes = look.message(_INSIGHTS_EXIT_CODES.bind())
    lines = [
        align_edges(
            Text(row.skill_name, style=section_style(look)),
            Text(f"{loads} {row.load_count:,} · {turns} {row.turn_count:,}"),
            width,
        ),
        align_edges(
            Text(f"{first_action} {_format_metric_duration(row.first_action_median_ns)}"),
            Text(f"{script_runs} {row.script_count:,} · {resources} {row.resource_count:,}"),
            width,
        ),
        align_edges(
            Text(f"{injected} {_format_metric_tokens(row.injected_tokens)}"),
            Text(f"{revisions} {', '.join(row.revisions) or '—'}"),
            width,
        ),
    ]
    if len(row.revisions) > 1:
        lines.append(
            Text(
                look.message(_INSIGHTS_SKILL_CHANGED.bind()),
                style=look.semantic_style("warning", "yellow"),
            )
        )
    lines.extend(
        align_edges(
            Text(f"  ↳ {child.name or '—'}", style="dim"),
            Text(
                f"{script_runs} {child.count:,} · {_format_named_counts(child.outcomes)}"
                + (f" · {exit_codes} {_format_named_counts(child.exit_codes)}" if child.exit_codes else "")
            ),
            width,
        )
        for child in row.scripts[:5]
    )
    if len(row.scripts) > 5:
        lines.append(Text(f"  {look.message(TOOL_USAGE_MORE.bind(value=len(row.scripts) - 5))}", style="dim"))
    lines.extend(
        align_edges(
            Text(f"  ↳ {child.name or '—'}", style="dim"),
            Text(f"{resources} {child.count:,}"),
            width,
        )
        for child in row.resources[:5]
    )
    if len(row.resources) > 5:
        lines.append(Text(f"  {look.message(TOOL_USAGE_MORE.bind(value=len(row.resources) - 5))}", style="dim"))
    return lines


def _per_turn_token_content(
    look: DashboardLook, context: RenderContext, analysis: TrajectoryAnalysis, *, width: int
) -> list[Text]:
    per_turn: list[Text] = []
    for turn in analysis.turns:
        usage = turn.token_usage
        if usage is None:
            continue
        per_turn.extend(_per_turn_token_lines(look, context, turn, usage, session=analysis.token_usage, width=width))
    if not per_turn:
        per_turn.append(Text(look.message(_INSIGHTS_NO_TURN_TOKENS.bind()), style="dim"))
    return per_turn


def _per_turn_token_lines(
    look: DashboardLook,
    context: RenderContext,
    turn: TurnAnalysis,
    usage: TokenUsage,
    *,
    session: TokenUsage | None,
    width: int,
) -> list[Text]:
    label = look.message(TURN_LABEL.bind(turn=turn.turn_number or "—"))
    buckets = " · ".join(
        f"{look.message(definition.bind())} {_format_metric_tokens(usage.buckets[bucket])}"
        for bucket, definition in (
            (UsageBucket.INPUT, TOKEN_INPUT),
            (UsageBucket.OUTPUT, TOKEN_OUTPUT),
            (UsageBucket.REASONING, TOKEN_REASONING),
            (UsageBucket.CACHE_READ, TOKEN_CACHE_READ),
            (UsageBucket.CACHE_CREATION, _TOKEN_CACHE_CREATION),
        )
    )
    lines = [
        align_edges(
            Text(label, style=section_style(look)),
            Text(buckets),
            width,
        )
    ]
    if context.tier is not ResponsiveTier.FLOOR:
        share = _input_share_metric(usage, session)
        cache_hit = cache_hit_metric(usage)
        meter_width = max(6, min(30, (width - 46) // 2))
        left = Text.assemble(
            Text(f"  {look.message(_INSIGHTS_INPUT_SHARE.bind())} "),
            percentage_meter(
                None if share.value is None else float(share.value),
                width=meter_width,
                style=look.semantic_style("primary", "blue"),
            ),
            Text(" "),
            precision_badge(look, share.precision),
        )
        right = Text.assemble(
            Text(f"{look.message(TOKEN_CACHE_HIT.bind())} "),
            percentage_meter(
                None if cache_hit.value is None else float(cache_hit.value),
                width=meter_width,
                style=_cache_hit_style(look, cache_hit.value),
            ),
            Text(" "),
            precision_badge(look, cache_hit.precision),
        )
        lines.append(align_edges(left, right, width))
    return lines


def _carrying_load_lines(
    look: DashboardLook, context: RenderContext, rows: list[ContextCarryingLoad], *, width: int
) -> list[Text]:
    lines = [Text(look.message(_INSIGHTS_CARRYING_EXPLAINER.bind()), style="dim")]
    if not rows:
        lines.append(Text("—", style="dim"))
        return lines
    maximum = max(row.load for row in rows)
    bar_width = min(16, max(4, width // 4))
    for row in rows:
        # Two lines per item: the identity and its total cost on the
        # first, the cost formula and a relative bar on the second, so
        # neither half gets truncated into the other at pair widths.
        head = look.message(
            _INSIGHTS_CARRYING_ROW_HEAD.bind(
                item=_carrying_item_label(look, row),
                turn=row.origin_turn_number or "—",
            )
        )
        detail = look.message(
            _INSIGHTS_CARRYING_ROW_DETAIL.bind(
                tokens=format_tokens(row.token_count),
                carries=row.carry_count,
            )
        )
        lines.append(
            align_edges(
                Text(head, style=section_style(look)),
                Text(format_tokens(row.load), style="bold"),
                width,
            )
        )
        bar = Text()
        if context.tier is not ResponsiveTier.FLOOR and maximum:
            filled = max(1, round(row.load / maximum * bar_width))
            bar = Text.assemble(
                Text("▬" * filled, style=look.semantic_style("primary", "blue")),
                Text(" " * (bar_width - filled)),
            )
        lines.append(align_edges(Text(f"  {detail}", style="dim"), bar, width))
    return lines


def _carrying_item_label(look: DashboardLook, row: ContextCarryingLoad) -> str:
    if row.role == "user":
        kind = _INSIGHTS_CARRYING_USER
    elif row.role == "assistant":
        kind = _INSIGHTS_CARRYING_ASSISTANT
    elif row.role == "tool":
        kind = _INSIGHTS_CARRYING_TOOL_RESULT
    else:
        kind = _INSIGHTS_CARRYING_ITEM
    label = look.message(kind.bind())
    if not row.tool_names:
        return label
    names = ", ".join(
        f"{name} ×{count}" if count > 1 else name  # noqa: RUF001
        for name, count in Counter(row.tool_names).items()
    )
    return f"{label} ({names})"


def _finding_lines(look: DashboardLook, analysis: TrajectoryAnalysis) -> list[Text]:
    lines: list[Text] = []
    if not analysis.findings:
        lines.append(Text(look.message(_NO_FINDINGS.bind()), style="dim"))
    else:
        for finding in analysis.findings:
            severity_style = _finding_style(look, finding.severity)
            lines.append(
                Text.assemble(
                    Text(f"{_finding_glyph(finding.severity)} ", style=severity_style),
                    Text(
                        look.message(_FINDING_TITLES[finding.rule_id].bind()),
                        style=severity_style,
                    ),
                    Text("  "),
                    precision_badge(look, finding.precision),
                )
            )
            lines.append(
                Text.assemble(
                    Text("    "),
                    Text(_finding_detail(look, finding), style="dim"),
                )
            )
    return lines


def _finding_detail(look: DashboardLook, finding: FindingRow) -> str:
    definition = _FINDING_DETAILS[finding.rule_id]
    args = dict(finding.detail_args)
    if finding.rule_id in {"unverified-change", "repeated-tool-fingerprint", "net-zero-churn"}:
        return look.message(definition.bind(count=args["count"]))
    if finding.rule_id == "retry-token-amplification":
        tokens = args["tokens"]
        return look.message(definition.bind(count=tokens, tokens=format_tokens(tokens)))
    if finding.rule_id == "context-carrying-load":
        load = args["load"]
        return look.message(definition.bind(count=load, load=format_tokens(load)))
    return look.message(definition.bind(**args))


def _finding_style(look: DashboardLook, severity: FindingSeverity) -> Style:
    if severity is FindingSeverity.ERROR:
        return look.semantic_style("error", "red", bold=True)
    if severity is FindingSeverity.WARNING:
        return look.semantic_style("warning", "yellow", bold=True)
    return Style(dim=True)


def _diagnostic_lines(look: DashboardLook, analysis: TrajectoryAnalysis) -> list[Text]:
    diagnostics = analysis.diagnostics
    lines = [Text(look.message(_DIAGNOSTICS_INTRO.bind()), style="dim")]
    problems: list[Text] = []
    if diagnostics.corrupt_line_count:
        corrupt_sequences = ", ".join(
            look.message(_AFTER_SEQUENCE.bind(sequence=item.after_sequence)) for item in diagnostics.corrupt_lines
        )
        problems.append(
            Text(
                look.message(_CORRUPT_LINES.bind(count=diagnostics.corrupt_line_count, sequences=corrupt_sequences)),
                style=look.semantic_style("error", "red"),
            )
        )
    if diagnostics.unsupported_event_count:
        unsupported_sequences = ", ".join(
            look.message(_SEQUENCE.bind(sequence=item.sequence)) for item in diagnostics.unsupported_lines
        )
        problems.append(
            Text(
                look.message(
                    _UNSUPPORTED_LINES.bind(
                        count=diagnostics.unsupported_event_count,
                        sequences=unsupported_sequences,
                    )
                ),
                style=look.semantic_style("error", "red"),
            )
        )
    problems.extend(
        Text(
            look.message(
                _ACCOUNTED_PREFIX.bind(
                    first=item.first_sequence,
                    last=item.last_sequence,
                    reason=item.message,
                )
            ),
            style=look.semantic_style("error", "red"),
        )
        for item in diagnostics.accounted_prefix_violation_details
    )
    for label, metric in _overview_metric_items(look, analysis.overview):
        if metric.precision is Precision.UNRESOLVED:
            problems.append(_unresolved_metric_line(look, label, metric))
    for turn in analysis.turns:
        turn_label = look.message(TURN_LABEL.bind(turn=turn.turn_number or "—"))
        for label, metric in _turn_metric_items(look, turn):
            if metric.precision is Precision.UNRESOLVED:
                problems.append(_unresolved_metric_line(look, f"{turn_label} {label}", metric))
    problems.extend(_operation_diagnostic_line(look, item) for item in diagnostics.timeline_operations)
    problems.extend(
        Text(
            look.message(
                _CONTAINMENT.bind(
                    family=item.family,
                    callsite=callsite(item.operation_id),
                    parent_family=item.parent_family,
                    parent_callsite=callsite(item.parent_operation_id),
                )
            ),
            style=look.semantic_style("error", "red"),
        )
        for item in diagnostics.containment_violations
    )
    if diagnostics.torn_tail_bytes:
        problems.append(
            Text(
                look.message(_TORN_TAIL.bind(bytes=diagnostics.torn_tail_bytes)),
                style=look.semantic_style("error", "red"),
            )
        )
    problems.extend(
        Text(
            look.message(_EXPLICIT_GAP.bind(first=item.first_sequence, last=item.last_sequence, reason=item.message)),
            style=look.semantic_style("error", "red"),
        )
        for item in diagnostics.explicit_gaps
    )
    if diagnostics.rollback_projection_unresolved:
        problems.append(
            Text(
                look.message(_ROLLBACK_UNRESOLVED.bind()),
                style=look.semantic_style("warning", "yellow"),
            )
        )
    if diagnostics.malformed_hook_execution_mode_count:
        problems.append(
            Text(
                look.message(_MALFORMED_HOOK_MODES.bind(count=diagnostics.malformed_hook_execution_mode_count)),
                style=look.semantic_style("error", "red"),
            )
        )
    if problems:
        lines.extend(problems)
    else:
        lines.append(
            Text(
                look.message(_DIAGNOSTICS_HEALTHY.bind()),
                style=look.semantic_style("success", "green"),
            )
        )
    # Recorded-duration drift is a pure producer signal: no metric reads
    # duration_ms, so it is information, not a problem — one summary line
    # rather than a wall of per-span notes.
    if diagnostics.span_duration_mismatches:
        mismatches = diagnostics.span_duration_mismatches
        families = ", ".join(
            f"{family} ×{count}" if count > 1 else family  # noqa: RUF001
            for family, count in Counter(item.family for item in mismatches).items()
        )
        delta_ns = max(abs(item.interval_ns - item.recorded_duration_ms * 1_000_000) for item in mismatches)
        lines.append(
            Text(
                look.message(
                    _DURATION_MISMATCH_SUMMARY.bind(
                        count=len(mismatches),
                        value=len(mismatches),
                        families=families,
                        delta=format_duration(delta_ns),
                    )
                ),
                style="dim",
            )
        )
    if diagnostics.side_call_empty_shell_revisions:
        shell_count = len(diagnostics.side_call_empty_shell_revisions)
        lines.append(
            Text(
                look.message(_SIDE_CALL_EMPTY_SHELLS.bind(count=shell_count, value=shell_count)),
                style="dim",
            )
        )
    if diagnostics.unidentified_membership_revision_count:
        unidentified_count = diagnostics.unidentified_membership_revision_count
        lines.append(
            Text(
                look.message(_UNIDENTIFIED_MEMBERSHIP.bind(count=unidentified_count, value=unidentified_count)),
                style="dim",
            )
        )
    return lines


def _unresolved_metric_line(look: DashboardLook, label: str, metric: Metric) -> Text:
    return Text(
        look.message(_UNRESOLVED_METRIC.bind(metric=label, reason=metric.reason or _precision(look, metric))),
        style=look.semantic_style("warning", "yellow"),
    )


def _operation_diagnostic_line(look: DashboardLook, diagnostic: TimelineOperationDiagnostic) -> Text:
    identity = diagnostic.identity or diagnostic.family
    identity = identity_with_hook_id(identity, diagnostic.hook_id)
    operation = f"{identity} @{callsite(diagnostic.operation_id)}"
    reason = look.message(_OPERATION_REASON_MESSAGES[diagnostic.code].bind())
    return Text(
        look.message(
            _OPERATION_DIAGNOSTIC.bind(
                turn=diagnostic.turn_number or "—",
                operation=operation,
                reason=reason,
            )
        ),
        style=precision_style(look, diagnostic.precision),
    )


def _overview_metric_items(look: DashboardLook, overview: TrajectoryOverview | None) -> list[tuple[str, Metric]]:
    if overview is None:
        return []
    return _metric_items(
        look,
        overview.elapsed_ns,
        overview.response_cp_ns,
        overview.compute_cp_ns,
        overview.exclusive_work_ns,
        overview.parallelism,
        overview.overlap_gain_ns,
        overview.usage_tokens,
        overview.wall_time_ns,
        overview.utilization,
    )


def _turn_metric_items(look: DashboardLook, turn: TurnAnalysis) -> list[tuple[str, Metric]]:
    return _metric_items(
        look,
        turn.elapsed_ns,
        turn.response_cp_ns,
        turn.compute_cp_ns,
        turn.exclusive_work_ns,
        turn.parallelism,
        turn.overlap_gain_ns,
        turn.usage_tokens,
        turn.wall_time_ns,
        turn.utilization,
    )


def _metric_items(
    look: DashboardLook,
    elapsed: Metric,
    response_cp: Metric,
    compute_cp: Metric,
    work: Metric,
    parallelism: Metric,
    overlap: Metric,
    usage: Metric,
    wall: dict[WallBucket, Metric],
    utilization: dict[WallBucket, Metric],
) -> list[tuple[str, Metric]]:
    items = [
        (look.message(METRIC_ELAPSED.bind()), elapsed),
        (look.message(METRIC_CP_RESPONSE.bind()), response_cp),
        (look.message(METRIC_CP_COMPUTE.bind()), compute_cp),
        (look.message(METRIC_WORK.bind()), work),
        (look.message(METRIC_PARALLELISM.bind()), parallelism),
        (look.message(METRIC_OVERLAP.bind()), overlap),
        (look.message(METRIC_USAGE.bind()), usage),
    ]
    bucket_labels = {
        WallBucket.MODEL: look.message(BUCKET_MODEL.bind()),
        WallBucket.TOOLS: look.message(BUCKET_TOOLS.bind()),
        WallBucket.WAIT: look.message(BUCKET_WAIT.bind()),
        WallBucket.IDLE: look.message(BUCKET_IDLE.bind()),
    }
    items.extend(
        (
            look.message(_WALL_METRIC.bind(bucket=bucket_labels[bucket])),
            wall[bucket],
        )
        for bucket in WallBucket
    )
    items.extend(
        (
            look.message(_UTILIZATION_METRIC.bind(bucket=bucket_labels[bucket])),
            utilization[bucket],
        )
        for bucket in (WallBucket.MODEL, WallBucket.TOOLS)
    )
    return items


def _cache_hit_style(look: DashboardLook, value: int | float | None) -> Style:
    if value is None or value > 60:
        return look.semantic_style("success", "green")
    if value >= 30:
        return look.semantic_style("warning", "yellow")
    return look.semantic_style("error", "red")


def _precision(look: DashboardLook, metric: Metric) -> str:
    return precision_label(look, metric.precision)


def _finding_glyph(severity: FindingSeverity) -> str:
    if severity is FindingSeverity.ERROR:
        return "!"
    if severity is FindingSeverity.WARNING:
        return "△"
    return "•"


def _input_share_metric(usage: TokenUsage, session: TokenUsage | None) -> Metric:
    """Derive this turn's share of the session's input tokens."""
    if session is None:
        return Metric(None, Precision.MISSING)
    turn_input = usage.buckets[UsageBucket.INPUT]
    total_input = session.buckets[UsageBucket.INPUT]
    if turn_input.value is None:
        return Metric(None, turn_input.precision, turn_input.reason)
    if total_input.value is None:
        return Metric(None, total_input.precision, total_input.reason)
    if float(total_input.value) == 0:
        return Metric(None, Precision.MISSING, total_input.reason)
    if (
        float(total_input.value) < 0
        or float(turn_input.value) < 0
        or float(turn_input.value) > float(total_input.value)
    ):
        return Metric(None, Precision.UNRESOLVED)
    percent = 100 * float(turn_input.value) / float(total_input.value)
    precision, reason = derived_metric_precision(turn_input, total_input)
    return Metric(percent, precision, reason)


def _format_metric_duration(metric: Metric) -> str:
    value = "—" if metric.value is None else format_duration(metric.value)
    return f"{value} {PRECISION_SYMBOLS[metric.precision]}"


def _format_percentage_metric(metric: Metric) -> str:
    value = "—" if metric.value is None else f"{float(metric.value) * 100:.1f}%"
    return f"{value} {PRECISION_SYMBOLS[metric.precision]}"


def _format_metric_tokens(metric: Metric) -> str:
    return f"{format_tokens(metric.value)} {PRECISION_SYMBOLS[metric.precision]}"


def _format_metric_count(metric: Metric) -> str:
    value = "—" if metric.value is None else f"{int(metric.value):,}"
    return f"{value} {PRECISION_SYMBOLS[metric.precision]}"


def _format_metric_bytes(metric: Metric) -> str:
    if metric.value is None:
        value = "—"
    else:
        size = float(metric.value)
        value = f"{size / 1024:.1f} KiB" if size >= 1024 else f"{int(size)} B"
    return f"{value} {PRECISION_SYMBOLS[metric.precision]}"


def _format_named_counts(rows: tuple[NamedCountRow, ...]) -> str:
    return ", ".join(f"{row.name}:{row.count}" for row in rows) or "—"
