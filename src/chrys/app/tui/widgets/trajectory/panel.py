# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""In-place trajectory Overview, turn Timeline, placeholders, and session data."""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from threading import Event
from time import monotonic
from typing import TYPE_CHECKING, Any, ClassVar

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.cache import LRUCache
from textual.color import Color, ColorParseError
from textual.containers import Container, VerticalGroup
from textual.css.query import NoMatches
from textual.geometry import Size
from textual.message import Message
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widgets import Tab, Tabs

from chrys.app.tui.binding_display import localized_binding
from chrys.app.tui.clipboard import copy_text_to_clipboards
from chrys.app.tui.copy_messages import COPIED_TITLE
from chrys.app.tui.i18n import render_str
from chrys.app.tui.support.file_manager import can_open_in_file_manager, open_in_file_manager
from chrys.app.tui.support.gc_freeze import (
    DetachedLruCache,
    GcFreezeBlockReason,
    detach_lru_cache,
    renew_lru_cache,
)
from chrys.app.tui.util.formatting import format_byte_size, format_token_count
from chrys.app.tui.util.rich_style import rich_style_from_textual_color
from chrys.app.tui.widgets.chat.session_json import SessionJsonPanel
from chrys.app.tui.widgets.hatch import hatch_text_style, hatched_text_line
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
from chrys.app.tui.widgets.trajectory.chartkit import (
    bordered_section,
    coverage_bar,
    fit_cells,
    percentage_meter,
    section_interior_width,
    time_ruler,
    timeline_bar,
    unresolved_bar,
    waterfall_lanes,
)
from chrys.app.tui.widgets.trajectory.session_info import SessionStorage, collect_session_storage
from chrys.foundation.config.settings import DEFAULT_TRAJECTORY_VERIFY_COMMANDS
from chrys.foundation.i18n import MessageDef, MessageRef, msg
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.platform.files import surrogate_safe_text
from chrys.foundation.util.session_ids import session_short_id
from chrys.service.analytics import (
    FLOW_TERMINAL_INDEX,
    ActionClass,
    AnalysisAvailability,
    ChangeVerificationState,
    ContextCarryingLoad,
    FindingRow,
    FindingSeverity,
    McpServerRow,
    Metric,
    NamedCountRow,
    Precision,
    SkillInsightRow,
    SubmissionLatencyBucket,
    SubmissionLatencySample,
    TimelineDiagnosticCode,
    TimelineOperation,
    TimelineOperationDiagnostic,
    TokenUsage,
    ToolInsightRow,
    ToolUsagePanel,
    TrajectoryAnalysis,
    TrajectoryAnalyzer,
    TrajectoryDiagnostics,
    TrajectoryOverview,
    TrajectoryScanCancelled,
    TurnAnalysis,
    UsageBucket,
    WallBucket,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.app import ComposeResult
    from textual.timer import Timer
    from textual.worker import Worker

    from chrys.app.tui.i18n import LocaleController

# The session folder's sizes live outside the events log, so the live refresh
# recollects them on this coarser clock instead of walking the tree every tick.
_STORAGE_REFRESH_INTERVAL_S = 5.0

_DASHBOARD_TITLE = msg("tui.trajectory.title", fallback="Trajectory")
_OVERVIEW_TAB = msg("tui.trajectory.tab.overview", fallback="Overview")
_TIMELINE_TAB = msg("tui.trajectory.tab.timeline", fallback="Timeline")
_INSIGHTS_TAB = msg("tui.trajectory.tab.insights", fallback="Insights")
_SESSION_DATA_TAB = msg("tui.trajectory.tab.session_data", fallback="Session data")
_GRAPH_LEGEND = msg(
    "tui.trajectory.graph.legend",
    fallback="│ parent · ⇠ causal · ┄ adjacent only (no proven dependency)",
)
_GRAPH_TITLE = msg("tui.trajectory.graph.title", fallback="Dependency graph · turn {turn}")
_GRAPH_RESPONSE = msg("tui.trajectory.graph.response", fallback="response")
_GRAPH_NONE = msg(
    "tui.trajectory.graph.none",
    fallback="No dependency graph is available for this turn.",
)
_GRAPH_CYCLE_WARNING = msg(
    "tui.trajectory.graph.cycle",
    fallback="The dependency graph contains a cycle; rows fall back to first-occurrence order.",
)
_GRAPH_HINT = msg("tui.trajectory.timeline.graph_hint", fallback="Space: dependency graph")
_TIMELINE_HINT = msg("tui.trajectory.graph.timeline_hint", fallback="Space: timeline")
_TOGGLE_GRAPH_BINDING = msg("tui.binding.toggle_graph", fallback="Dependency graph")
_UNAVAILABLE = msg("tui.trajectory.unavailable", fallback="No trajectory data is available for this session.")
_READ_ERROR = msg("tui.trajectory.read_error", fallback="Trajectory read error: {error}")
_NO_TURNS = msg("tui.trajectory.no_turns", fallback="No completed turns are available.")
_NO_ACTIVE_SESSION = msg("tui.main.session_json.no_active_session", fallback="No active session.")
_ELAPSED_SCOPE = msg(
    "tui.trajectory.elapsed_scope",
    fallback="Total time excludes preparation between submission and the start of the turn.",
)
_PRECISION_EXACT = msg("tui.trajectory.precision.exact", fallback="exact")
_PRECISION_ESTIMATED = msg("tui.trajectory.precision.estimated", fallback="estimated")
_PRECISION_MISSING = msg("tui.trajectory.precision.missing", fallback="missing")
_PRECISION_UNRESOLVED = msg("tui.trajectory.precision.unresolved", fallback="unresolved")
_PRECISION = {
    Precision.EXACT: _PRECISION_EXACT,
    Precision.ESTIMATED: _PRECISION_ESTIMATED,
    Precision.MISSING: _PRECISION_MISSING,
    Precision.UNRESOLVED: _PRECISION_UNRESOLVED,
}
_PRECISION_SYMBOLS = {
    Precision.EXACT: "✓",
    Precision.ESTIMATED: "~",
    Precision.MISSING: "−",  # noqa: RUF001
    Precision.UNRESOLVED: "✗",
}
_ELAPSED = msg("tui.trajectory.metric.elapsed", fallback="total time")
_CP_RESPONSE = msg("tui.trajectory.metric.cp_response", fallback="bottleneck (response)")
_CP_COMPUTE = msg("tui.trajectory.metric.cp_compute", fallback="bottleneck (compute)")
_WORK = msg("tui.trajectory.metric.work", fallback="actual work time")
_PARALLELISM = msg("tui.trajectory.metric.parallelism", fallback="parallelism")
_OVERLAP = msg("tui.trajectory.metric.overlap", fallback="parallel time saved")
_USAGE = msg("tui.trajectory.metric.usage", fallback="total token usage")
_UTILIZATION = msg(
    "tui.trajectory.utilization",
    fallback="Busy share (model and tools independent; >100% = parallel work)",
)
_MODEL = msg("tui.trajectory.bucket.model", fallback="model")
_TOOLS = msg("tui.trajectory.bucket.tools", fallback="tools")
_WAIT = msg("tui.trajectory.bucket.wait", fallback="wait")
_IDLE = msg("tui.trajectory.bucket.idle", fallback="idle")
_COVERAGE = msg("tui.trajectory.coverage", fallback="data confidence")
_COVERAGE_SHARES = msg(
    "tui.trajectory.coverage_shares",
    fallback="{exact}✓/{estimated}~/{missing}−/{unresolved}✗",  # noqa: RUF001
)
_KPI_TIME = msg("tui.trajectory.kpi.time", fallback="Time & usage")
_KPI_SPLIT = msg("tui.trajectory.kpi.split", fallback="Where time went")
_KPI_PARALLEL = msg("tui.trajectory.kpi.parallel", fallback="Parallelism & busy")
_SESSION_INFO = msg("tui.trajectory.session_info.title", fallback="Session info")
_SESSION_INFO_PATH = msg("tui.trajectory.session_info.path", fallback="folder")
_SESSION_INFO_COPY_PATH = msg("tui.trajectory.session_info.copy_path", fallback="Copy path")
_SESSION_INFO_PATH_COPIED = msg("tui.trajectory.session_info.path_copied", fallback="Path copied")
_SESSION_INFO_OPEN_FOLDER = msg("tui.trajectory.session_info.open_folder", fallback="Open folder")
_SESSION_INFO_ON_DISK = msg("tui.trajectory.session_info.on_disk", fallback="on disk")
_SESSION_INFO_MUTATIONS = msg("tui.trajectory.session_info.mutations", fallback="diff backups")
_SESSION_INFO_SNAPSHOTS = msg("tui.trajectory.session_info.snapshots", fallback="rollback snapshots")
_SESSION_INFO_SUB_AGENTS = msg("tui.trajectory.session_info.sub_agents", fallback="sub-agent sessions")
_SESSION_INFO_FIRST_MESSAGE = msg("tui.trajectory.session_info.first_message", fallback="first message")
_SESSION_INFO_LAST_REPLY = msg("tui.trajectory.session_info.last_reply", fallback="last reply")
_SESSION_INFO_SPAN = msg("tui.trajectory.session_info.span", fallback="first → last")
_SESSION_INFO_TURNS = msg("tui.trajectory.session_info.turns", fallback="turns")
_SESSION_INFO_EVENTS = msg("tui.trajectory.session_info.events", fallback="events logged")
_SESSION_INFO_RUNTIMES = msg("tui.trajectory.session_info.runtimes", fallback="times opened")
_SESSION_INFO_FILES = msg("tui.trajectory.session_info.files", fallback="{count} file", plural_fallback="{count} files")
_SESSION_INFO_OPEN_FAILED = msg(
    "tui.trajectory.session_info.open_failed",
    fallback="Could not open the session folder: {error}",
)
_SESSION_INFO_OPEN_UNAVAILABLE = msg(
    "tui.trajectory.session_info.open_unavailable",
    fallback="Opening the session folder is not available in the current environment.",
)
_WATERFALL = msg(
    "tui.trajectory.waterfall",
    fallback="Per-turn time breakdown · turns {first}-{last}",
)
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
_TURN = msg("tui.trajectory.turn", fallback="Turn {turn}")
_TIME_RULER = msg("tui.trajectory.time_ruler", fallback="time")
_CATEGORY_MODEL = msg("tui.trajectory.category.model", fallback="Model")
_CATEGORY_TOOL = msg("tui.trajectory.category.tool", fallback="Tool")
_CATEGORY_WAIT = msg("tui.trajectory.category.wait", fallback="Wait")
_CATEGORY_HOOK = msg("tui.trajectory.category.hook", fallback="Hook")
_CATEGORY_AGENT = msg("tui.trajectory.category.agent", fallback="Agent")
_CATEGORY_PREPARATION = msg("tui.trajectory.category.preparation", fallback="Prepare")
_CATEGORY_COMPACTION = msg("tui.trajectory.category.compaction", fallback="Compact")
_CATEGORY_APPROVAL = msg("tui.trajectory.category.approval", fallback="Approval")
_CATEGORY_RETRY = msg("tui.trajectory.category.retry", fallback="Retry")
_CATEGORY_OPERATION = msg("tui.trajectory.category.operation", fallback="Operation")
_WALL_METRIC = msg("tui.trajectory.diagnostics.wall_metric", fallback="{bucket} wall time")
_UTILIZATION_METRIC = msg(
    "tui.trajectory.diagnostics.utilization_metric",
    fallback="{bucket} busy share",
)
_CHARTS_TOO_NARROW = msg(
    "tui.trajectory.charts_too_narrow",
    fallback="Terminal too narrow; widen it to display charts.",
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
_ACTION_FUNNEL = msg("tui.trajectory.action_funnel.title", fallback="Action breakdown")
_ACTION_SEARCH = msg("tui.trajectory.action.search", fallback="search")
_ACTION_READ = msg("tui.trajectory.action.read", fallback="read")
_ACTION_EDIT = msg("tui.trajectory.action.edit", fallback="edit")
_ACTION_VERIFY = msg("tui.trajectory.action.verify", fallback="verify")
_FAILURE_RECOVERY = msg("tui.trajectory.failure_recovery.title", fallback="Failure recovery")
_TOOL_FAILURES = msg("tui.trajectory.failure_recovery.failures", fallback="tool failures")
_MEDIAN_RECOVERY = msg("tui.trajectory.failure_recovery.median", fallback="median recovery after failure")
_RETRY_AMPLIFICATION = msg(
    "tui.trajectory.failure_recovery.amplification",
    fallback="retry overhead",
)
_REPEATED_SIGNATURES = msg(
    "tui.trajectory.failure_recovery.repeated",
    fallback="repeated identical failures",
)
_TOKEN_COUNT = msg("tui.trajectory.token_count", fallback="{tokens} tokens")
_CHANGE_VERIFICATION = msg("tui.trajectory.change_verification.title", fallback="Change verification")
_CHANGE_FILES = msg("tui.trajectory.change_verification.files", fallback="files")
_CHANGE_COUNTS = msg(
    "tui.trajectory.change_verification.counts",
    fallback="{files} · +{created} ~{modified} -{deleted} · cancelled out {net_zero}",
)
_CHANGE_DETAIL_UNAVAILABLE = msg(
    "tui.trajectory.change_verification.unavailable",
    fallback="File detail unavailable; showing recorded mutation summaries.",
)
_CHANGE_DETECTION_TRUNCATED = msg(
    "tui.trajectory.change_verification.truncated",
    fallback="Recorded/observed counts; mutation detection was truncated.",
)
_CHANGE_STATE_VERIFIED = msg("tui.trajectory.change_verification.state.verified", fallback="verified")
_CHANGE_STATE_AFTER = msg("tui.trajectory.change_verification.state.after", fallback="after verify")
_CHANGE_STATE_UNVERIFIED = msg("tui.trajectory.change_verification.state.unverified", fallback="unverified")
_CHANGE_STATE_NET_ZERO = msg("tui.trajectory.change_verification.state.net_zero", fallback="cancelled out")
_TOKEN_USAGE = msg("tui.trajectory.token_usage.title", fallback="Token usage")
_TOKEN_INPUT = msg("tui.trajectory.token_usage.input", fallback="input")
_TOKEN_OUTPUT = msg("tui.trajectory.token_usage.output", fallback="output")
_TOKEN_REASONING = msg("tui.trajectory.token_usage.reasoning", fallback="reasoning")
_TOKEN_CACHE_READ = msg("tui.trajectory.token_usage.cache_read", fallback="cache read")
_TOKEN_CACHE_CREATION = msg("tui.trajectory.token_usage.cache_creation", fallback="cache creation")
_TOKEN_CACHE_HIT = msg("tui.trajectory.token_usage.cache_hit", fallback="cache hit")
_SKILL_USAGE = msg("tui.trajectory.skill_usage.title", fallback="Skill usage")
_SKILL_USAGE_NONE = msg("tui.trajectory.skill_usage.none", fallback="No skills were used.")
_MCP_USAGE = msg("tui.trajectory.mcp_usage.title", fallback="MCP usage")
_MCP_USAGE_NONE = msg("tui.trajectory.mcp_usage.none", fallback="No MCP tools were called.")
_TOOL_USAGE_UNATTRIBUTED = msg("tui.trajectory.tool_usage.unattributed", fallback="(name unavailable)")
_TOOL_USAGE_MORE = msg("tui.trajectory.tool_usage.more", fallback="+{value} more")
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
_SUBMISSION_LATENCY = msg(
    "tui.trajectory.submission_latency.title",
    fallback="Submission wait (submit → work starts)",
)
_SUBMISSION_INTRO = msg(
    "tui.trajectory.submission_latency.intro",
    fallback="How long each message waited between being sent and the agent starting on it.",
)
_SUBMISSION_NONE = msg("tui.trajectory.submission_latency.none", fallback="No submissions were recorded.")
_SUBMISSION_BECAME_TURN = msg(
    "tui.trajectory.submission_latency.became_turn",
    fallback="started a new turn",
)
_SUBMISSION_INJECTED = msg(
    "tui.trajectory.submission_latency.injected",
    fallback="injected into an ongoing turn",
)
_SUBMISSION_DID_NOT_BECOME = msg(
    "tui.trajectory.submission_latency.did_not_become",
    fallback="never became a turn",
)
_SUBMISSION_STATS = msg(
    "tui.trajectory.submission_latency.stats",
    fallback="{value} sample · median {p50} · p90 {p90} · slowest {maximum}",
    plural_fallback="{value} samples · median {p50} · p90 {p90} · slowest {maximum}",
)
_SUBMISSION_UNRESOLVED = msg(
    "tui.trajectory.submission_latency.unresolved",
    fallback="{count} unresolved sample",
    plural_fallback="{count} unresolved samples",
)
_SUBMISSION_SAMPLE = msg(
    "tui.trajectory.submission_latency.sample",
    fallback="{outcome} · {duration}",
)
_PREPARATION_OUTCOME_HANDOFF = msg("tui.trajectory.preparation_outcome.handoff", fallback="handoff")
_PREPARATION_OUTCOME_COMPLETED = msg("tui.trajectory.preparation_outcome.completed", fallback="completed")
_PREPARATION_OUTCOME_FAILED = msg("tui.trajectory.preparation_outcome.failed", fallback="failed")
_PREPARATION_OUTCOME_INTERRUPTED = msg("tui.trajectory.preparation_outcome.interrupted", fallback="interrupted")
_PREPARATION_OUTCOME_FRESH_TURN = msg("tui.trajectory.preparation_outcome.fresh_turn", fallback="fresh turn")
_PREPARATION_OUTCOME_RETRY_TURN = msg("tui.trajectory.preparation_outcome.retry_turn", fallback="retry turn")
_PREPARATION_OUTCOME_INJECTED = msg("tui.trajectory.preparation_outcome.injected", fallback="injected")
_PREPARATION_OUTCOME_ABANDONED_NO_TARGET = msg(
    "tui.trajectory.preparation_outcome.abandoned_no_target",
    fallback="abandoned: no target",
)
_PREPARATION_OUTCOME_CANCELLED = msg("tui.trajectory.preparation_outcome.cancelled", fallback="cancelled")
_PREPARATION_OUTCOME_TARGET_STALE = msg(
    "tui.trajectory.preparation_outcome.target_stale",
    fallback="target stale",
)
_PREPARATION_OUTCOME_REJECTED = msg("tui.trajectory.preparation_outcome.rejected", fallback="rejected")
_PREPARATION_OUTCOME_IMAGE_REJECTED = msg(
    "tui.trajectory.preparation_outcome.image_rejected",
    fallback="image rejected",
)
_PREPARATION_OUTCOME_NOT_READY = msg("tui.trajectory.preparation_outcome.not_ready", fallback="not ready")
_PREPARATION_OUTCOME_PREPARATION_FAILED = msg(
    "tui.trajectory.preparation_outcome.preparation_failed",
    fallback="preparation failed",
)
_PREPARATION_OUTCOME_CONFLICT = msg("tui.trajectory.preparation_outcome.conflict", fallback="conflict")
_PREPARATION_OUTCOME_OWNER_CHANGED = msg(
    "tui.trajectory.preparation_outcome.owner_changed",
    fallback="owner changed",
)
_PREPARATION_OUTCOME_SUPERSEDED = msg("tui.trajectory.preparation_outcome.superseded", fallback="superseded")
_PREPARATION_OUTCOME_DROPPED = msg("tui.trajectory.preparation_outcome.dropped", fallback="dropped")
_PREPARATION_OUTCOMES = {
    "handoff": _PREPARATION_OUTCOME_HANDOFF,
    "completed": _PREPARATION_OUTCOME_COMPLETED,
    "failed": _PREPARATION_OUTCOME_FAILED,
    "interrupted": _PREPARATION_OUTCOME_INTERRUPTED,
    "fresh_turn": _PREPARATION_OUTCOME_FRESH_TURN,
    "retry_turn": _PREPARATION_OUTCOME_RETRY_TURN,
    "injected": _PREPARATION_OUTCOME_INJECTED,
    "abandoned_no_target": _PREPARATION_OUTCOME_ABANDONED_NO_TARGET,
    "cancelled": _PREPARATION_OUTCOME_CANCELLED,
    "target_stale": _PREPARATION_OUTCOME_TARGET_STALE,
    "rejected": _PREPARATION_OUTCOME_REJECTED,
    "image_rejected": _PREPARATION_OUTCOME_IMAGE_REJECTED,
    "not_ready": _PREPARATION_OUTCOME_NOT_READY,
    "preparation_failed": _PREPARATION_OUTCOME_PREPARATION_FAILED,
    "conflict": _PREPARATION_OUTCOME_CONFLICT,
    "owner_changed": _PREPARATION_OUTCOME_OWNER_CHANGED,
    "superseded": _PREPARATION_OUTCOME_SUPERSEDED,
    "dropped": _PREPARATION_OUTCOME_DROPPED,
}

_WIDE_MIN_COLUMNS = 120
_MID_MIN_COLUMNS = 80
_NARROW_MIN_COLUMNS = 60
# Side-by-side boxes pad the shorter one to the taller one's height; past this
# gap the dead space outweighs the width saving and the pair stacks full-width.
_SECTION_PAIR_MAX_HEIGHT_GAP = 10
_TABS_HEIGHT = 2
_SESSION_INFO_FOUR_COLUMN_MIN_WIDTH = 144


class DashboardTab(StrEnum):
    """The frozen four-tab dashboard information architecture."""

    OVERVIEW = "overview"
    TIMELINE = "timeline"
    INSIGHTS = "insights"
    SESSION_DATA = "session-data"


_OPERATION_CATEGORY_WIDTH = 8
_OPERATION_CATEGORY_SEPARATOR = " "
_TIMELINE_LABEL_WIDTH = 27
_TIMELINE_WIDE_LABEL_WIDTH = 39
_HOOK_ID_DISPLAY_MAX_LENGTH = 16
_HOOK_ID_TRUNCATION_SUFFIX = "..."
_HOOK_EVENT_DISPLAY_NAMES = {
    "before_tool_call": "before_tool",
    "after_tool_call": "after_tool",
}


def _identity_with_hook_id(identity: str, hook_id: str | None) -> str:
    displayed_identity = _HOOK_EVENT_DISPLAY_NAMES.get(identity, identity)
    if hook_id is None:
        return displayed_identity
    displayed_hook_id = hook_id
    if len(displayed_hook_id) > _HOOK_ID_DISPLAY_MAX_LENGTH:
        prefix_length = _HOOK_ID_DISPLAY_MAX_LENGTH - len(_HOOK_ID_TRUNCATION_SUFFIX)
        displayed_hook_id = f"{displayed_hook_id[:prefix_length]}{_HOOK_ID_TRUNCATION_SUFFIX}"
    if hook_id == identity:
        return displayed_hook_id
    return f"{displayed_identity} ({displayed_hook_id})"


class ResponsiveTier(StrEnum):
    """Panel-local width tiers frozen by the trajectory UI contract."""

    WIDE = "wide"
    MID = "mid"
    NARROW = "narrow"
    FLOOR = "floor"


def _has_diagnostic_content(diagnostics: TrajectoryDiagnostics) -> bool:
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


class TrajectoryTextView(ScrollView):
    """Virtualized, cell-width-aware line surface for dashboard presentations."""

    can_focus = True

    DEFAULT_CSS = """
    TrajectoryTextView {
        height: 1fr;
        overflow: auto auto;
        scrollbar-size: 1 1;
    }
    TrajectoryTextView.-overview {
        overflow-x: hidden;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._lines: list[Text] = []
        self._width = 0
        self._strips: LRUCache[tuple[int, int, int], Strip] | DetachedLruCache = LRUCache(maxsize=500)

    def set_lines(self, lines: list[Text], *, reset_scroll: bool = True) -> None:
        old_x = self.scroll_offset.x
        old_y = self.scroll_offset.y
        self._lines = lines
        self._width = max((cell_len(line.plain) for line in lines), default=0)
        if not isinstance(self._strips, DetachedLruCache):
            self._strips.clear()
        self.virtual_size = Size(self._width, len(lines))
        if reset_scroll:
            self.scroll_home(animate=False)
        else:
            self.scroll_to(x=old_x, y=old_y, animate=False, force=True, immediate=True)
        self.refresh(layout=True)

    def clear_lines(self) -> None:
        self.set_lines([])

    def release(self) -> None:
        """Release content and the render LRU at the dashboard ownership boundary."""
        self.clear_lines()
        self._strips = LRUCache(maxsize=500)

    def detach_cache(self) -> None:
        self._strips = detach_lru_cache(self._strips)

    def renew_cache(self) -> None:
        self._strips = renew_lru_cache(self._strips)

    def notify_style_update(self) -> None:
        """Discard strips that contain resolved theme styles."""
        super().notify_style_update()
        if not isinstance(self._strips, DetachedLruCache):
            self._strips.clear()

    def on_resize(self, _event: events.Resize) -> None:
        """Sibling chrome (the turn tab strip) resizes this view only after
        lines commit; the dashboard rebuilds them for the settled region."""
        parent = self.parent
        if isinstance(parent, TrajectoryDashboard):
            parent.render_for_settled_view()

    def action_open_session_folder(self) -> None:
        """``@click`` target of the session info section's folder control."""
        parent = self.parent
        if isinstance(parent, TrajectoryDashboard):
            parent.open_session_folder()

    def action_copy_session_path(self) -> None:
        """``@click`` target of the session info section's copy control."""
        parent = self.parent
        if isinstance(parent, TrajectoryDashboard):
            parent.copy_session_path()

    def render_line(self, y: int) -> Strip:
        width = self.scrollable_content_region.width
        scroll_x = round(self.scroll_offset.x)
        absolute_line = y + round(self.scroll_offset.y)
        widget_style = self.visual_style.rich_style
        if absolute_line < 0 or absolute_line >= len(self._lines):
            return Strip.blank(width, widget_style)
        key = (absolute_line, scroll_x, width)
        cache = self._strips
        if isinstance(cache, DetachedLruCache):
            cache = renew_lru_cache(cache)
            self._strips = cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        line = self._lines[absolute_line]
        strip = Strip(list(line.render(self.app.console)), cell_len(line.plain))
        rendered = strip.crop(scroll_x, scroll_x + width).adjust_cell_length(width)
        line_style = self.app.console.get_style(line.style)
        rendered = rendered.apply_style(widget_style + line_style)
        cache[key] = rendered
        return rendered


class TrajectoryDashboard(Container):
    """Own dashboard foreground and active-tab state as one atomic pair."""

    can_focus = True

    BINDINGS: ClassVar[list] = [
        localized_binding("space", "toggle_timeline_dependencies", _TOGGLE_GRAPH_BINDING, show=False),
    ]

    COMPONENT_CLASSES: ClassVar[set[str]] = {"trajectory-dashboard--muted-label", "hatch--pattern"}

    DEFAULT_CSS = """
    TrajectoryDashboard > .hatch--pattern {
        color: $hatch-color;
    }
    TrajectoryDashboard > .trajectory-dashboard--muted-label {
        color: $text-muted;
    }
    TrajectoryDashboard {
        height: 1fr;
        display: none;
        layer: overlay;
        layers: default loading;
        background: $background;
        padding: 0 1;
        border: round $tui-border-accent $border-opacity;
        border-title-align: left;
        border-title-color: $tui-border-title-accent;
        border-subtitle-align: right;
        border-subtitle-color: $tui-border-title-accent;
    }
    TrajectoryDashboard > Tabs {
        height: 2;
    }
    TrajectoryDashboard > #timeline-turn-tabs {
        display: none;
    }
    /* The loading state floats on its own layer below the tab strip, so the
       same indicator covers either the hidden text view or the Session Data
       viewer while that viewer's background load is still running. */
    TrajectoryDashboard > #trajectory-loading-state {
        display: none;
        layer: loading;
        margin-top: 2;
        width: 100%;
        height: 1fr;
        min-height: 1;
        align: center middle;
        background: $background;
    }
    TrajectoryDashboard > #trajectory-loading-state > ChrysLoadingIndicator {
        width: 12;
        height: 1;
        color: $primary;
    }
    TrajectoryDashboard > SessionJsonPanel {
        height: 1fr;
        border: none;
    }
    """

    class StateChanged(Message):
        """The foreground derivations need synchronization by MainScreen."""

    def __init__(
        self,
        *,
        locale_controller: LocaleController | None = None,
        verify_commands: str = DEFAULT_TRAJECTORY_VERIFY_COMMANDS,
    ) -> None:
        super().__init__()
        self._locale_controller = locale_controller
        self._verify_commands = verify_commands
        self.foreground = False
        self.active_tab = DashboardTab.OVERVIEW
        self._timeline_dependencies = False
        self._selected_turn_id: str | None = None
        self._selected_operation_id: str | None = None
        self._selected_timeline_line: int | None = None
        self._session_id = ""
        self._path: Path | None = None
        self._session_json_loaded = False
        self._turn_tab_key: tuple[tuple[str, str], ...] = ()
        self._width_delta = 0
        self._height_delta = 0
        self._analyzer: TrajectoryAnalyzer | None = None
        self._analysis: TrajectoryAnalysis | None = None
        self._session_storage: SessionStorage | None = None
        self._can_open_session_folder = can_open_in_file_manager()
        self._storage_collected_at = -_STORAGE_REFRESH_INTERVAL_S
        self._load_generation = 0
        self._load_pending = False
        self._worker: Worker[Any] | None = None
        self._scan_cancel_event: Event | None = None
        self._live_timer: Timer | None = None
        self._shell_restore_foreground = False
        self._available_size = Size(0, 0)
        self._presentation_key: tuple[int, int, int] | None = None
        self._presentation_revision = 0
        self._render_identity: tuple[DashboardTab, str | None] | None = None
        self._presentation_cache: LRUCache[tuple[object, ...], tuple[Text, ...]] | DetachedLruCache = LRUCache(
            maxsize=64
        )
        self._update_border_labels()

    @property
    def chat_foreground(self) -> bool:
        return not self.foreground

    @property
    def session_json_visible(self) -> bool:
        return self.foreground and self.active_tab is DashboardTab.SESSION_DATA

    @property
    def responsive_tier(self) -> ResponsiveTier:
        width = self._available_width()
        if width >= _WIDE_MIN_COLUMNS:
            return ResponsiveTier.WIDE
        if width >= _MID_MIN_COLUMNS:
            return ResponsiveTier.MID
        if width >= _NARROW_MIN_COLUMNS:
            return ResponsiveTier.NARROW
        return ResponsiveTier.FLOOR

    def compose(self) -> ComposeResult:
        yield Tabs(
            Tab(Text(self._render_message(_OVERVIEW_TAB.bind())), id=DashboardTab.OVERVIEW),
            Tab(Text(self._render_message(_TIMELINE_TAB.bind())), id=DashboardTab.TIMELINE),
            Tab(Text(self._render_message(_INSIGHTS_TAB.bind())), id=DashboardTab.INSIGHTS),
            Tab(Text(self._render_message(_SESSION_DATA_TAB.bind())), id=DashboardTab.SESSION_DATA),
            active=self.active_tab,
            id="trajectory-tabs",
        )
        yield Tabs(id="timeline-turn-tabs")
        with VerticalGroup(id="trajectory-loading-state"):
            yield ChrysLoadingIndicator(id="trajectory-loading")
        yield TrajectoryTextView()
        yield SessionJsonPanel(locale_controller=self._locale_controller)

    def on_mount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.register_surface(self)
        self._live_timer = self.set_interval(0.5, self._refresh_live)
        self._live_timer.pause()
        self._sync_content_visibility()

    def on_unmount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.unregister_surface(self)
        self._cancel_worker()
        analyzer = self._analyzer
        if analyzer is not None:
            analyzer.release()
        self._analyzer = None
        self._analysis = None

    def on_resize(self, event: events.Resize) -> None:
        """Invalidate presentation only; trajectory analysis never depends on size."""
        content_size = self.size
        if content_size == self._available_size:
            return
        self._available_size = content_size
        if self.foreground and not self.session_json_visible:
            self._render_active_view()

    def render_for_settled_view(self) -> None:
        """Re-render after the text view's own size settles asynchronously."""
        if self.is_mounted and self.foreground and not self.session_json_visible:
            self._render_active_view()

    def notify_style_update(self) -> None:
        """Rebuild presentation lines that contain resolved theme colors."""
        super().notify_style_update()
        self._presentation_revision += 1
        self._update_border_labels()
        self._clear_presentation_cache()
        self._render_active_view()

    def refresh_localization(self) -> None:
        self._presentation_revision += 1
        self._update_border_labels()
        for tab, label in (
            (DashboardTab.OVERVIEW, _OVERVIEW_TAB),
            (DashboardTab.TIMELINE, _TIMELINE_TAB),
            (DashboardTab.INSIGHTS, _INSIGHTS_TAB),
            (DashboardTab.SESSION_DATA, _SESSION_DATA_TAB),
        ):
            self.query_one(f"#{tab}", Tab).label = Text(self._render_message(label.bind()))
        self._clear_presentation_cache()
        self._render_active_view()

    def show_session(self, session_id: str, path: Path | None) -> None:
        changed = session_id != self._session_id or path != self._path
        self.foreground = True
        self.display = True
        self._session_id = session_id
        self._path = path
        self._update_border_labels()
        if changed:
            self._session_json_loaded = False
        if changed or self._analysis is None:
            self._release_analysis()
            self._start_load()
        self._sync_live_timer()
        self._sync_content_visibility()
        self.query_one("#trajectory-tabs", Tabs).focus()

    def set_verify_commands(self, value: str) -> None:
        """Apply the live classification word list and rebuild the current projection."""
        if value == self._verify_commands:
            return
        self._verify_commands = value
        if self.foreground and self._path is not None:
            self._release_analysis()
            self._start_load()

    def hide_dashboard(self) -> None:
        self.foreground = False
        self.display = False
        self._sync_live_timer()
        self._sync_content_visibility()
        self._release_analysis()

    def select_turn(self, turn_id: str, operation_id: str | None = None) -> None:
        turn = self._analysis.turn(turn_id) if self._analysis is not None else None
        self._selected_turn_id = turn.turn_id if turn is not None else turn_id
        self._selected_operation_id = operation_id
        self._timeline_dependencies = False
        self.active_tab = DashboardTab.TIMELINE
        self.query_one("#trajectory-tabs", Tabs).active = DashboardTab.TIMELINE
        self._update_border_labels()
        self._sync_live_timer()
        self._sync_content_visibility()
        self._render_active_view()
        self.post_message(self.StateChanged())

    def suspend_for_shell_mode(self) -> bool:
        self._shell_restore_foreground = self.foreground
        restore_json = self.session_json_visible
        self.foreground = False
        self.display = False
        self._sync_live_timer()
        self._release_analysis()
        return restore_json

    def finish_shell_mode(self) -> bool:
        restore = self._shell_restore_foreground
        self._shell_restore_foreground = False
        if restore:
            self.show_session(self._session_id, self._path)
        return restore

    def gc_freeze_block_reason(self) -> GcFreezeBlockReason | None:
        if self.foreground and not self.session_json_visible:
            return GcFreezeBlockReason.TRAJECTORY_DASHBOARD_VISIBLE
        return None

    def prepare_for_gc_freeze(self) -> None:
        if self.foreground and not self.session_json_visible:
            return
        self.query_one(TrajectoryTextView).detach_cache()
        self._presentation_cache = detach_lru_cache(self._presentation_cache)

    def after_gc_freeze(self) -> None:
        self.query_one(TrajectoryTextView).renew_cache()
        self._presentation_cache = renew_lru_cache(self._presentation_cache)

    def abort_gc_freeze(self) -> None:
        self.after_gc_freeze()

    @on(Tabs.TabActivated, "#trajectory-tabs")
    def _on_tab_activated(self, event: Tabs.TabActivated) -> None:
        if event.tab.id is None:
            return
        self.active_tab = DashboardTab(event.tab.id)
        self._update_border_labels()
        self._sync_live_timer()
        self._sync_content_visibility()
        self._render_active_view()
        if self.foreground:
            if self.session_json_visible:
                self.query_one(SessionJsonPanel).focus()
            else:
                self.query_one(TrajectoryTextView).focus()
        self.post_message(self.StateChanged())

    @on(Tabs.TabActivated, "#timeline-turn-tabs")
    def _on_turn_tab_activated(self, event: Tabs.TabActivated) -> None:
        event.stop()
        if event.tabs is not self._query_turn_tabs() or event.tab.id is None:
            return
        index = int(event.tab.id.removeprefix("turn-"))
        if index >= len(self._turn_tab_key):
            return
        turn_id = self._turn_tab_key[index][0]
        if turn_id == self._selected_turn_id:
            return
        self._selected_turn_id = turn_id
        self._selected_operation_id = None
        self._render_active_view()

    def action_toggle_timeline_dependencies(self) -> None:
        """Flip the timeline tab between the time axis and the dependency graph."""
        if not self.foreground or self.active_tab is not DashboardTab.TIMELINE or self.session_json_visible:
            return
        self._timeline_dependencies = not self._timeline_dependencies
        self._render_active_view()

    @on(SessionJsonPanel.LoadStateChanged)
    def _on_session_json_load_state_changed(self, event: SessionJsonPanel.LoadStateChanged) -> None:
        event.stop()
        self._sync_content_visibility()

    def _sync_content_visibility(self) -> None:
        if not self.is_mounted:
            return
        session_json = self.query_one(SessionJsonPanel)
        text_view = self.query_one(TrajectoryTextView)
        show_json = self.session_json_visible
        if show_json:
            session_json.display = True
            if not self._session_json_loaded:
                if self._session_id:
                    session_json.load_session(self._session_id)
                else:
                    session_json.set_status(self._render_message(_NO_ACTIVE_SESSION.bind()))
                self._session_json_loaded = True
        else:
            session_json.hide_session_json()
            self._session_json_loaded = False
        # One indicator for both slow paths: the analysis scan behind the
        # hidden text view, and the Session Data viewer's background load
        # (which must stay displayed so its worker can commit).
        show_loading = self.foreground and (session_json.is_loading if show_json else self._load_pending)
        text_view.set_class(self.active_tab is DashboardTab.OVERVIEW, "-overview")
        text_view.set_class(self.active_tab is DashboardTab.TIMELINE, "-timeline")
        text_view.display = self.foreground and not show_json and not show_loading
        self.query_one("#trajectory-loading-state").display = show_loading
        turn_tabs = self._query_turn_tabs()
        if turn_tabs is not None:
            turn_tabs.display = self._turn_tabs_visible()

    def _turn_tabs_visible(self) -> bool:
        return (
            self.foreground
            and not self.session_json_visible
            and bool(self._turn_tab_key)
            and self.active_tab is DashboardTab.TIMELINE
        )

    def _sync_turn_tabs(self, analysis: TrajectoryAnalysis | None) -> None:
        turns = (
            analysis.turns if analysis is not None and analysis.availability is AnalysisAvailability.AVAILABLE else ()
        )
        key = tuple((turn.turn_id, self._render_message(_TURN.bind(turn=turn.turn_number or "—"))) for turn in turns)
        if key != self._turn_tab_key:
            self._turn_tab_key = key
            # Rebuild by replacement: remove-then-mount runs atomically inside
            # one callback, and building at execution time keeps the active tab
            # aligned with whatever the selection is once the callback runs.
            self.call_next(self._replace_turn_tabs)
            return
        tabs = self._query_turn_tabs()
        if tabs is None:
            return
        active = self._active_turn_tab_id()
        if active is not None and tabs.active != active and tabs.query(f"#{active}"):
            tabs.active = active
        tabs.display = self._turn_tabs_visible()

    def _query_turn_tabs(self) -> Tabs | None:
        # A pending replacement leaves a gap between remove and mount; callers
        # skip the sync then because the mount applies the fresh state itself.
        try:
            return self.query_one("#timeline-turn-tabs", Tabs)
        except NoMatches:
            return None

    def _active_turn_tab_id(self) -> str | None:
        return next(
            (
                f"turn-{index}"
                for index, (turn_id, _) in enumerate(self._turn_tab_key)
                if turn_id == self._selected_turn_id
            ),
            None,
        )

    async def _replace_turn_tabs(self) -> None:
        old = self.query_one("#timeline-turn-tabs", Tabs)
        anchor = self.query_one("#trajectory-tabs", Tabs)
        replacement = Tabs(
            *(Tab(Text(label), id=f"turn-{index}") for index, (_, label) in enumerate(self._turn_tab_key)),
            active=self._active_turn_tab_id(),
            id="timeline-turn-tabs",
        )
        await old.remove()
        replacement.display = self._turn_tabs_visible()
        await self.mount(replacement, after=anchor)

    def _update_border_labels(self) -> None:
        self.border_title = Text(self._render_message(_DASHBOARD_TITLE.bind()))
        if not self._session_id:
            self.border_subtitle = Text("")
            return
        if self.active_tab is DashboardTab.SESSION_DATA and self._path is not None:
            self.border_subtitle = Text(surrogate_safe_text(str(self._path.parents[1] / "session.json")))
            return
        subtitle = self._precision_legend()
        subtitle.append(" · ")
        subtitle.append(session_short_id(self._session_id))
        self.border_subtitle = subtitle

    def _sync_live_timer(self) -> None:
        timer = self._live_timer
        if timer is None:
            return
        if self.foreground and self.active_tab in {DashboardTab.OVERVIEW, DashboardTab.TIMELINE}:
            timer.resume()
        else:
            timer.pause()

    def _start_load(self) -> None:
        self._cancel_worker()
        self._load_generation += 1
        generation = self._load_generation
        analyzer = TrajectoryAnalyzer(verify_commands=self._verify_commands)
        path = self._path
        cancel_event = Event()
        self._analyzer = analyzer
        self._scan_cancel_event = cancel_event
        self.query_one(TrajectoryTextView).clear_lines()
        # The text view stays hidden behind the loading indicator until the
        # first analysis lands; a long scan must not read as "no data".
        self._load_pending = True
        self._worker = self.run_worker(
            partial(self._load, generation, analyzer, path, cancel_event),
            group="trajectory-analysis",
            exclusive=True,
        )
        self._sync_turn_tabs(None)
        self._sync_content_visibility()

    async def _load(
        self,
        generation: int,
        analyzer: TrajectoryAnalyzer,
        path: Path | None,
        cancel_event: Event,
    ) -> None:
        storage: SessionStorage | None = None
        if path is None:
            analysis = None
        else:
            try:
                analysis, storage = await asyncio.to_thread(
                    partial(_load_with_storage, analyzer, path, cancel_event=cancel_event)
                )
            except TrajectoryScanCancelled:
                return
        if generation != self._load_generation or analyzer is not self._analyzer or not self.foreground:
            return
        self._analysis = analysis
        self._session_storage = storage
        if storage is not None:
            self._storage_collected_at = monotonic()
        self._load_pending = False
        self._presentation_revision += 1
        self._sync_content_visibility()
        self._render_active_view()
        # The display flip above lands in the next layout pass, so the text
        # view's region is still the placeholder's here; render again once the
        # view settles or the first frame keeps the stale measurements.
        self.call_after_refresh(self.render_for_settled_view)

    async def _refresh_live(self) -> None:
        analyzer = self._analyzer
        worker = self._worker
        if (
            not self.foreground
            or analyzer is None
            or self._analysis is None
            or (worker is not None and worker.is_running)
        ):
            return
        self._load_generation += 1
        generation = self._load_generation
        cancel_event = Event()
        self._scan_cancel_event = cancel_event
        self._worker = self.run_worker(
            partial(self._refresh, generation, analyzer, cancel_event),
            group="trajectory-analysis",
            exclusive=True,
        )

    async def _refresh(self, generation: int, analyzer: TrajectoryAnalyzer, cancel_event: Event) -> None:
        # Storage lives outside the events log, so a quiet log must not pin it
        # forever: recollect on a coarse clock even when the analysis is unchanged.
        collect_storage = monotonic() - self._storage_collected_at >= _STORAGE_REFRESH_INTERVAL_S
        try:
            analysis, storage = await asyncio.to_thread(
                partial(
                    _refresh_with_storage,
                    analyzer,
                    collect_storage=collect_storage,
                    cancel_event=cancel_event,
                )
            )
        except TrajectoryScanCancelled:
            return
        if generation != self._load_generation or analyzer is not self._analyzer or not self.foreground:
            return
        changed = analysis is not self._analysis
        if storage is not None:
            self._storage_collected_at = monotonic()
            if storage != self._session_storage:
                self._session_storage = storage
                changed = True
        if not changed:
            return
        self._analysis = analysis
        self._presentation_revision += 1
        self._render_active_view()

    def _cancel_worker(self) -> None:
        self._load_generation += 1
        self._load_pending = False
        cancel_event = self._scan_cancel_event
        if cancel_event is not None:
            cancel_event.set()
        self._scan_cancel_event = None
        worker = self._worker
        if worker is not None:
            worker.cancel()
        self._worker = None

    def _release_analysis(self) -> None:
        self._cancel_worker()
        analyzer = self._analyzer
        if analyzer is not None:
            analyzer.release()
        self._analyzer = None
        self._analysis = None
        self._session_storage = None
        self._storage_collected_at = -_STORAGE_REFRESH_INTERVAL_S
        self._presentation_key = None
        self._render_identity = None
        self._clear_presentation_cache()
        if self.is_mounted:
            self.query_one(TrajectoryTextView).release()

    def _clear_presentation_cache(self) -> None:
        if not isinstance(self._presentation_cache, DetachedLruCache):
            self._presentation_cache.clear()

    def _render_active_view(self) -> None:
        if not self.is_mounted or self.session_json_visible or self._load_pending:
            return
        analysis = self._analysis
        generation = analysis.generation if analysis is not None else -1
        width = self._available_width()
        height = self._available_height()
        self._presentation_key = (generation, width, height)
        timeline = self.active_tab is DashboardTab.TIMELINE
        selected = self._selected_turn_id if timeline else None
        selected_operation = self._selected_operation_id if timeline and not self._timeline_dependencies else None
        # The empty state and the hatch fill below short content are built for
        # the text view's own region, which settles a beat after this widget
        # resizes; the region (and the scrollbar state feeding the settle math)
        # must key the cache so a build for a transitional layout is missed --
        # not hit -- by the re-render that follows the settled view.
        text_view = self.query_one(TrajectoryTextView)
        region = text_view.scrollable_content_region
        cache_key = (
            self.active_tab,
            self._timeline_dependencies,
            selected,
            selected_operation,
            generation,
            width,
            height,
            region.width,
            region.height,
            text_view.show_vertical_scrollbar,
            text_view.show_horizontal_scrollbar,
            self._verify_commands,
            self._presentation_revision,
        )
        cache = None if isinstance(self._presentation_cache, DetachedLruCache) else self._presentation_cache
        if analysis is None or analysis.availability is not AnalysisAvailability.AVAILABLE or not analysis.turns:
            cache = None
        if not region.width or not region.height:
            cache = None
        cached = cache.get(cache_key) if cache is not None else None
        if cached is not None:
            self._commit_lines(list(cached))
        else:
            lines = self._settled_lines(self._active_lines(analysis), analysis)
            if cache is not None:
                cache[cache_key] = tuple(lines)
            self._commit_lines(lines)
        if self.active_tab is DashboardTab.TIMELINE:
            self._sync_turn_tabs(analysis)

    def _active_lines(self, analysis: TrajectoryAnalysis | None) -> list[Text]:
        if analysis is None or analysis.availability is AnalysisAvailability.UNAVAILABLE:
            return self._empty_state_lines(_UNAVAILABLE)
        if analysis.availability is AnalysisAvailability.READ_ERROR:
            return [Text(self._render_message(_READ_ERROR.bind(error=analysis.read_error or "")))]
        if (
            not analysis.turns
            and self.active_tab
            in {
                DashboardTab.OVERVIEW,
                DashboardTab.TIMELINE,
                DashboardTab.INSIGHTS,
            }
            and (self.active_tab is not DashboardTab.INSIGHTS or not _has_diagnostic_content(analysis.diagnostics))
        ):
            return self._empty_state_lines(_NO_TURNS)
        if self.active_tab is DashboardTab.INSIGHTS:
            return self._insights_lines(analysis)
        if self.active_tab is DashboardTab.OVERVIEW:
            return self._overview_lines(analysis)
        if self.active_tab is DashboardTab.TIMELINE:
            if self._timeline_dependencies:
                return self._dependency_graph_lines(analysis)
            return self._timeline_lines(analysis)
        return []

    def _settled_lines(self, lines: list[Text], analysis: TrajectoryAnalysis | None) -> list[Text]:
        """Rebuild for the region the asynchronous scrollbars settle on.

        The scrollbars this commit provokes (or retires) resize the region one
        cell after the fact; building for the settled region keeps every right
        edge real instead of cropping it off.
        """
        text_view = self.query_one(TrajectoryTextView)
        region = text_view.scrollable_content_region
        if not region.width or not region.height:
            return lines
        region_width = self._effective_region_width()
        available = self._available_width()
        base_width = region_width + (1 if text_view.show_vertical_scrollbar else 0)
        if available:
            base_width = min(base_width, available)
        base_height = region.height + (1 if text_view.show_horizontal_scrollbar else 0)
        overflows_y = len(lines) > base_height
        settled_width = base_width - (1 if overflows_y else 0)
        overflows_x = any(line.cell_len > settled_width for line in lines)
        settled_height = base_height - (1 if overflows_x else 0)
        self._width_delta = settled_width - region_width
        self._height_delta = settled_height - region.height
        try:
            if self._width_delta or self._height_delta:
                lines = self._active_lines(analysis)
        finally:
            self._width_delta = 0
            self._height_delta = 0
        return self._hatch_filled(lines, width=settled_width, height=settled_height)

    def _hatch_filled(self, lines: list[Text], *, width: int, height: int) -> list[Text]:
        """Fill the viewport rows below short content with the hatch pattern."""
        if len(lines) >= height:
            return lines
        hatch_style = self._hatch_style()
        label_style = self._muted_label_style()
        return [
            *lines,
            *(
                hatched_text_line(width, hatch_style=hatch_style, label_style=label_style)
                for _ in range(height - len(lines))
            ),
        ]

    def _effective_region_width(self) -> int:
        """The text view's region width, bounded by the dashboard's own size.

        The view's region lags a display flip by one layout pass, so right
        after a load it can still report the width of a previous layout; the
        dashboard's content size is current and caps it.
        """
        width = self.query_one(TrajectoryTextView).scrollable_content_region.width
        available = self._available_width()
        return min(width, available) if width and available else width

    def _content_width(self) -> int:
        width = self._effective_region_width()
        if not width:
            return max(1, self._available_width())
        return max(1, width + self._width_delta)

    def _empty_state_lines(self, message: MessageDef) -> list[Text]:
        text_view = self.query_one(TrajectoryTextView)
        width = self._content_width()
        region_height = text_view.scrollable_content_region.height
        height = max(1, region_height + self._height_delta) if region_height else self._available_height()
        hatch_style = self._hatch_style()
        label_style = self._muted_label_style()
        label = self._render_message(message.bind())
        return [
            hatched_text_line(
                width, label if y == height // 2 else None, hatch_style=hatch_style, label_style=label_style
            )
            for y in range(height)
        ]

    def _commit_lines(self, lines: list[Text]) -> None:
        # The identity deliberately excludes the analysis generation: a live
        # session bumps it on every appended event, and a refresh of the view
        # the user is already reading must keep their scroll position. Session
        # switches reset the identity through ``_release_analysis``.
        identity = (
            self.active_tab,
            self._selected_turn_id if self.active_tab is DashboardTab.TIMELINE else None,
        )
        reset_scroll = identity != self._render_identity
        self._render_identity = identity
        text_view = self.query_one(TrajectoryTextView)
        text_view.set_lines(lines, reset_scroll=reset_scroll)
        if self.active_tab is DashboardTab.TIMELINE and self._selected_timeline_line is not None:
            text_view.scroll_to(y=max(0, self._selected_timeline_line - 2), animate=False, force=True, immediate=True)

    def _overview_lines(self, analysis: TrajectoryAnalysis) -> list[Text]:
        overview = analysis.overview
        if overview is None:
            return [Text(self._render_message(_UNAVAILABLE.bind()))]
        tier = self.responsive_tier
        width = self._content_width()
        interior_width = section_interior_width(width)
        if tier in {ResponsiveTier.WIDE, ResponsiveTier.MID}:
            lower_widths = _section_row_widths(width, 3)
            lower_interiors = tuple(section_interior_width(box_width) for box_width in lower_widths)
            kpi = self._bordered_section_row(
                (
                    (_KPI_TIME, self._kpi_time_lines(overview, width=lower_interiors[0]), lower_widths[0]),
                    (_KPI_SPLIT, self._kpi_split_lines(overview, width=lower_interiors[1]), lower_widths[1]),
                    (_KPI_PARALLEL, self._kpi_parallel_lines(overview, width=lower_interiors[2]), lower_widths[2]),
                )
            )
        else:
            lower_widths = (width, width, width)
            lower_interiors = (interior_width, interior_width, interior_width)
            kpi = [
                *self._section_box(_KPI_TIME, self._kpi_time_lines(overview, width=interior_width), width=width),
                *self._section_box(_KPI_SPLIT, self._kpi_split_lines(overview, width=interior_width), width=width),
                *self._section_box(
                    _KPI_PARALLEL, self._kpi_parallel_lines(overview, width=interior_width), width=width
                ),
            ]
        if tier is ResponsiveTier.FLOOR:
            notice = Text(self._render_message(_CHARTS_TOO_NARROW.bind()), style="dim")
            return [*kpi, *notice.wrap(self.app.console, width)]
        session_info = self._section_box(
            self._badged_section_title(
                _SESSION_INFO,
                Precision.UNRESOLVED if analysis.diagnostics.integrity_unresolved else Precision.EXACT,
            ),
            self._session_info_lines(
                analysis,
                width=interior_width,
                columns=4 if interior_width >= _SESSION_INFO_FOUR_COLUMN_MIN_WIDTH else 2,
            ),
            width=width,
        )
        token_usage = self._token_usage_lines(analysis, width=lower_interiors[0])
        skill_usage = self._tool_usage_lines(analysis.skill_usage, empty=_SKILL_USAGE_NONE, width=lower_interiors[1])
        mcp_usage = self._tool_usage_lines(analysis.mcp_usage, empty=_MCP_USAGE_NONE, width=lower_interiors[2])
        funnel = self._action_funnel_lines(analysis, width=lower_interiors[0])
        recovery = self._failure_recovery_lines(analysis, width=lower_interiors[1])
        changes = self._change_verification_lines(analysis, width=lower_interiors[2])
        if tier in {ResponsiveTier.WIDE, ResponsiveTier.MID}:
            usage_trio = self._bordered_section_row(
                (
                    (_TOKEN_USAGE, token_usage, lower_widths[0]),
                    (_SKILL_USAGE, skill_usage, lower_widths[1]),
                    (_MCP_USAGE, mcp_usage, lower_widths[2]),
                )
            )
            lower = self._bordered_section_row(
                (
                    (_ACTION_FUNNEL, funnel, lower_widths[0]),
                    (_FAILURE_RECOVERY, recovery, lower_widths[1]),
                    (_CHANGE_VERIFICATION, changes, lower_widths[2]),
                )
            )
        else:
            usage_trio = [
                *self._section_box(_TOKEN_USAGE, token_usage, width=width),
                *self._section_box(_SKILL_USAGE, skill_usage, width=width),
                *self._section_box(_MCP_USAGE, mcp_usage, width=width),
            ]
            lower = [
                *self._section_box(_ACTION_FUNNEL, funnel, width=width),
                *self._section_box(_FAILURE_RECOVERY, recovery, width=width),
                *self._section_box(_CHANGE_VERIFICATION, changes, width=width),
            ]
        first = analysis.turns[0].turn_number if analysis.turns else "—"
        last = analysis.turns[-1].turn_number if analysis.turns else "—"
        waterfall_title = _WATERFALL.bind(first=first or "—", last=last or "—")
        return [
            *session_info,
            *kpi,
            *self._section_box(
                waterfall_title,
                self._waterfall_lines(analysis, width=interior_width),
                width=width,
            ),
            *usage_trio,
            *lower,
            *self._section_box(
                _SUBMISSION_LATENCY,
                [
                    *self._submission_latency_lines(analysis, width=interior_width),
                    Text(self._render_message(_ELAPSED_SCOPE.bind()), style="dim"),
                ],
                width=width,
            ),
        ]

    def _insights_lines(self, analysis: TrajectoryAnalysis) -> list[Text]:
        width = self._content_width()
        tier = self.responsive_tier
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
            self._adaptive_section_pair(
                self._badged_section_title(_INSIGHTS_INTEGRATIONS_SKILLS, skills_precision),
                lambda box_width: self._skills_insight_lines(analysis, width=box_width),
                self._badged_section_title(_INSIGHTS_INTEGRATIONS_MCP, mcp_precision),
                lambda box_width: self._mcp_insight_lines(analysis, width=box_width),
                width=width,
                paired=tier is ResponsiveTier.WIDE,
                balanced=False,
            )
        )
        lines.extend(
            self._adaptive_section_pair(
                self._badged_section_title(_INSIGHTS_TOOLS, tools_precision),
                lambda box_width: self._tool_activity_lines(analysis, width=box_width),
                self._badged_section_title(_INSIGHTS_CARRYING_LOAD.bind(value=5), carrying_precision),
                lambda box_width: self._carrying_load_lines(carrying, width=box_width),
                width=width,
                paired=tier is ResponsiveTier.WIDE,
                balanced=False,
            )
        )
        lines.extend(
            self._section_box(
                _INSIGHTS_PER_TURN_TOKENS, self._per_turn_token_content(analysis, width=interior), width=width
            )
        )
        lines.extend(
            self._adaptive_section_pair(
                _FINDINGS,
                lambda _: self._finding_lines(analysis),
                _DIAGNOSTICS,
                lambda _: self._diagnostic_lines(analysis),
                width=width,
                paired=tier in {ResponsiveTier.WIDE, ResponsiveTier.MID},
            )
        )
        return lines

    def _adaptive_section_pair(
        self,
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
            halves = _section_row_widths(width, 2)
            left_lines = build_left(section_interior_width(halves[0]))
            right_lines = build_right(section_interior_width(halves[1]))
            left_box = self._section_box(left_title, left_lines, width=halves[0])
            right_box = self._section_box(right_title, right_lines, width=halves[1])
            if not balanced or abs(len(left_box) - len(right_box)) <= _SECTION_PAIR_MAX_HEIGHT_GAP:
                return self._bordered_section_row(
                    (
                        (left_title, left_lines, halves[0]),
                        (right_title, right_lines, halves[1]),
                    )
                )
        interior = section_interior_width(width)
        return [
            *self._section_box(left_title, build_left(interior), width=width),
            *self._section_box(right_title, build_right(interior), width=width),
        ]

    def _tool_activity_lines(self, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
        insights = analysis.insights
        if insights is None or not insights.tools.total:
            return [Text(self._render_message(_INSIGHTS_NO_TOOLS.bind()), style="dim")]
        content = []
        visible = insights.tools.rows[:12]
        for row in visible:
            content.extend(self._tool_insight_row(row, width=width))
        if len(insights.tools.rows) > len(visible):
            content.append(
                Text(
                    self._render_message(_TOOL_USAGE_MORE.bind(value=len(insights.tools.rows) - len(visible))),
                    style="dim",
                )
            )
        if insights.tools.unclassified:
            content.append(
                Text(
                    self._render_message(_INSIGHTS_UNCLASSIFIED.bind(value=insights.tools.unclassified)),
                    style=self._semantic_style("warning", "yellow"),
                )
            )
        return content

    def _tool_insight_row(self, row: ToolInsightRow, *, width: int) -> list[Text]:
        name = f"{row.tool_kind} · {row.tool_name or '—'}"
        calls = self._render_message(_INSIGHTS_CALLS.bind())
        share = self._render_message(_INSIGHTS_DURATION_SHARE.bind())
        p50 = self._render_message(_INSIGHTS_P50.bind())
        p95 = self._render_message(_INSIGHTS_P95.bind())
        outcomes = self._render_message(_INSIGHTS_RESULTS.bind())
        if self.responsive_tier is ResponsiveTier.FLOOR:
            share_text = Text(_format_percentage_metric(row.duration_share))
        else:
            # Same bracket meter as the Overview operation shares, so every
            # percentage on the dashboard reads the same way.
            share_text = Text.assemble(
                percentage_meter(
                    None if row.duration_share.value is None else float(row.duration_share.value) * 100,
                    width=max(6, min(16, width // 4)),
                    style=self._semantic_style("primary", "blue"),
                ),
                Text(" "),
                self._precision_badge(row.duration_share.precision),
            )
        return [
            _align_edges(
                Text(name, style=self._section_style()),
                Text(f"{calls} {row.calls:,}"),
                width,
            ),
            _align_edges(Text(share), share_text, width),
            _align_edges(
                Text(f"{p50} {_format_metric_duration(row.p50_ns)} · {p95} {_format_metric_duration(row.p95_ns)}"),
                Text(f"{outcomes} {_format_named_counts(row.outcomes)}"),
                width,
            ),
        ]

    def _mcp_insight_lines(self, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
        insights = analysis.insights
        if insights is None:
            return [Text(self._render_message(_INSIGHTS_NO_MCP.bind()), style="dim")]
        mcp_lines: list[Text] = []
        if not insights.mcp.total:
            mcp_lines.append(Text(self._render_message(_INSIGHTS_NO_MCP.bind()), style="dim"))
        else:
            for row in insights.mcp.rows[:8]:
                mcp_lines.extend(self._mcp_server_lines(row, width=width))
            if len(insights.mcp.rows) > 8:
                mcp_lines.append(
                    Text(self._render_message(_TOOL_USAGE_MORE.bind(value=len(insights.mcp.rows) - 8)), style="dim")
                )
        mcp_unattributed = insights.mcp.unattributed + insights.mcp.unattributed_connection_waits
        if mcp_unattributed:
            mcp_lines.append(
                Text(
                    self._render_message(_INSIGHTS_UNATTRIBUTED.bind(value=mcp_unattributed)),
                    style=self._semantic_style("warning", "yellow"),
                )
            )
        return mcp_lines

    def _skills_insight_lines(self, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
        insights = analysis.insights
        if insights is None:
            return [Text(self._render_message(_INSIGHTS_NO_SKILLS.bind()), style="dim")]
        skill_lines: list[Text] = []
        if not insights.skills.rows:
            skill_lines.append(Text(self._render_message(_INSIGHTS_NO_SKILLS.bind()), style="dim"))
        else:
            for row in insights.skills.rows[:10]:
                skill_lines.extend(self._skill_insight_lines(row, width=width))
            if len(insights.skills.rows) > 10:
                skill_lines.append(
                    Text(self._render_message(_TOOL_USAGE_MORE.bind(value=len(insights.skills.rows) - 10)), style="dim")
                )
        if insights.skills.unattributed:
            skill_lines.append(
                Text(
                    self._render_message(_INSIGHTS_UNATTRIBUTED.bind(value=insights.skills.unattributed)),
                    style=self._semantic_style("warning", "yellow"),
                )
            )
        skill_lines.extend(
            (
                _align_edges(
                    Text(
                        self._render_message(_INSIGHTS_SKILL_NOT_FOUND.bind()),
                        style=self._semantic_style("error", "red"),
                    ),
                    Text(f"{row.name} x{row.count}"),
                    width,
                )
            )
            for row in insights.skills.not_found
        )
        return skill_lines

    def _mcp_server_lines(self, row: McpServerRow, *, width: int) -> list[Text]:
        calls = self._render_message(_INSIGHTS_CALLS.bind())
        share = self._render_message(_INSIGHTS_DURATION_SHARE.bind())
        p50 = self._render_message(_INSIGHTS_P50.bind())
        p95 = self._render_message(_INSIGHTS_P95.bind())
        results = self._render_message(_INSIGHTS_RESULTS.bind())
        approval = self._render_message(_INSIGHTS_APPROVAL.bind())
        volume = self._render_message(_INSIGHTS_RETURN_VOLUME.bind())
        truncated = self._render_message(_INSIGHTS_TRUNCATED_SPILL.bind())
        cp = self._render_message(_INSIGHTS_CP_EXCLUSIVE.bind())
        connection = self._render_message(_INSIGHTS_CONNECTION_WAIT.bind())
        otel = self._render_message(_INSIGHTS_OTEL_CROSS_CHECK.bind())
        lines = [
            _align_edges(
                Text(row.server_name, style=self._section_style()),
                Text(f"{calls} {row.calls:,} · {share} {_format_percentage_metric(row.duration_share)}"),
                width,
            ),
            _align_edges(
                Text(f"{p50} {_format_metric_duration(row.p50_ns)} · {p95} {_format_metric_duration(row.p95_ns)}"),
                Text(f"{results} {_format_named_counts(row.outcomes)}"),
                width,
            ),
            _align_edges(
                Text(f"{approval} {_format_percentage_metric(row.approval_blocking_share)}"),
                Text(f"{volume} {_format_metric_bytes(row.result_bytes)} + {_format_metric_tokens(row.result_tokens)}"),
                width,
            ),
            _align_edges(
                Text(
                    f"{truncated} {_format_metric_count(row.truncated_count)} · {_format_metric_count(row.spill_count)}"
                ),
                Text(f"{cp} {_format_metric_duration(row.critical_path_exclusive_ns)}"),
                width,
            ),
            _align_edges(
                Text(
                    f"{connection} {_format_metric_count(row.connection_wait_count)} · "
                    f"{_format_metric_duration(row.connection_wait_ns)}"
                ),
                Text(f"{otel} —"),
                width,
            ),
        ]
        lines.extend(
            _align_edges(
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
            lines.append(
                Text(f"  {self._render_message(_TOOL_USAGE_MORE.bind(value=len(row.remotes) - 5))}", style="dim")
            )
        return lines

    def _skill_insight_lines(self, row: SkillInsightRow, *, width: int) -> list[Text]:
        loads = self._render_message(_INSIGHTS_LOADS.bind())
        turns = self._render_message(_INSIGHTS_TURNS.bind())
        first_action = self._render_message(_INSIGHTS_FIRST_ACTION.bind())
        script_runs = self._render_message(_INSIGHTS_SCRIPT_RUNS.bind())
        resources = self._render_message(_INSIGHTS_RESOURCE_READS.bind())
        injected = self._render_message(_INSIGHTS_INJECTED_TOKENS.bind())
        revisions = self._render_message(_INSIGHTS_REVISIONS.bind())
        exit_codes = self._render_message(_INSIGHTS_EXIT_CODES.bind())
        lines = [
            _align_edges(
                Text(row.skill_name, style=self._section_style()),
                Text(f"{loads} {row.load_count:,} · {turns} {row.turn_count:,}"),
                width,
            ),
            _align_edges(
                Text(f"{first_action} {_format_metric_duration(row.first_action_median_ns)}"),
                Text(f"{script_runs} {row.script_count:,} · {resources} {row.resource_count:,}"),
                width,
            ),
            _align_edges(
                Text(f"{injected} {_format_metric_tokens(row.injected_tokens)}"),
                Text(f"{revisions} {', '.join(row.revisions) or '—'}"),
                width,
            ),
        ]
        if len(row.revisions) > 1:
            lines.append(
                Text(
                    self._render_message(_INSIGHTS_SKILL_CHANGED.bind()),
                    style=self._semantic_style("warning", "yellow"),
                )
            )
        lines.extend(
            _align_edges(
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
            lines.append(
                Text(f"  {self._render_message(_TOOL_USAGE_MORE.bind(value=len(row.scripts) - 5))}", style="dim")
            )
        lines.extend(
            _align_edges(
                Text(f"  ↳ {child.name or '—'}", style="dim"),
                Text(f"{resources} {child.count:,}"),
                width,
            )
            for child in row.resources[:5]
        )
        if len(row.resources) > 5:
            lines.append(
                Text(f"  {self._render_message(_TOOL_USAGE_MORE.bind(value=len(row.resources) - 5))}", style="dim")
            )
        return lines

    def _per_turn_token_content(self, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
        per_turn: list[Text] = []
        for turn in analysis.turns:
            usage = turn.token_usage
            if usage is None:
                continue
            per_turn.extend(self._per_turn_token_lines(turn, usage, session=analysis.token_usage, width=width))
        if not per_turn:
            per_turn.append(Text(self._render_message(_INSIGHTS_NO_TURN_TOKENS.bind()), style="dim"))
        return per_turn

    def _per_turn_token_lines(
        self,
        turn: TurnAnalysis,
        usage: TokenUsage,
        *,
        session: TokenUsage | None,
        width: int,
    ) -> list[Text]:
        label = self._render_message(_TURN.bind(turn=turn.turn_number or "—"))
        buckets = " · ".join(
            f"{self._render_message(definition.bind())} {_format_metric_tokens(usage.buckets[bucket])}"
            for bucket, definition in (
                (UsageBucket.INPUT, _TOKEN_INPUT),
                (UsageBucket.OUTPUT, _TOKEN_OUTPUT),
                (UsageBucket.REASONING, _TOKEN_REASONING),
                (UsageBucket.CACHE_READ, _TOKEN_CACHE_READ),
                (UsageBucket.CACHE_CREATION, _TOKEN_CACHE_CREATION),
            )
        )
        lines = [
            _align_edges(
                Text(label, style=self._section_style()),
                Text(buckets),
                width,
            )
        ]
        if self.responsive_tier is not ResponsiveTier.FLOOR:
            share = _input_share_metric(usage, session)
            cache_hit = _cache_hit_metric(usage)
            meter_width = max(6, min(30, (width - 46) // 2))
            left = Text.assemble(
                Text(f"  {self._render_message(_INSIGHTS_INPUT_SHARE.bind())} "),
                percentage_meter(
                    None if share.value is None else float(share.value),
                    width=meter_width,
                    style=self._semantic_style("primary", "blue"),
                ),
                Text(" "),
                self._precision_badge(share.precision),
            )
            right = Text.assemble(
                Text(f"{self._render_message(_TOKEN_CACHE_HIT.bind())} "),
                percentage_meter(
                    None if cache_hit.value is None else float(cache_hit.value),
                    width=meter_width,
                    style=self._cache_hit_style(cache_hit.value),
                ),
                Text(" "),
                self._precision_badge(cache_hit.precision),
            )
            lines.append(_align_edges(left, right, width))
        return lines

    def _carrying_load_lines(self, rows: list[ContextCarryingLoad], *, width: int) -> list[Text]:
        lines = [Text(self._render_message(_INSIGHTS_CARRYING_EXPLAINER.bind()), style="dim")]
        if not rows:
            lines.append(Text("—", style="dim"))
            return lines
        maximum = max(row.load for row in rows)
        bar_width = min(16, max(4, width // 4))
        for row in rows:
            # Two lines per item: the identity and its total cost on the
            # first, the cost formula and a relative bar on the second, so
            # neither half gets truncated into the other at pair widths.
            head = self._render_message(
                _INSIGHTS_CARRYING_ROW_HEAD.bind(
                    item=self._carrying_item_label(row),
                    turn=row.origin_turn_number or "—",
                )
            )
            detail = self._render_message(
                _INSIGHTS_CARRYING_ROW_DETAIL.bind(
                    tokens=_format_tokens(row.token_count),
                    carries=row.carry_count,
                )
            )
            lines.append(
                _align_edges(
                    Text(head, style=self._section_style()),
                    Text(_format_tokens(row.load), style="bold"),
                    width,
                )
            )
            bar = Text()
            if self.responsive_tier is not ResponsiveTier.FLOOR and maximum:
                filled = max(1, round(row.load / maximum * bar_width))
                bar = Text.assemble(
                    Text("▬" * filled, style=self._semantic_style("primary", "blue")),
                    Text(" " * (bar_width - filled)),
                )
            lines.append(_align_edges(Text(f"  {detail}", style="dim"), bar, width))
        return lines

    def _carrying_item_label(self, row: ContextCarryingLoad) -> str:
        if row.role == "user":
            kind = _INSIGHTS_CARRYING_USER
        elif row.role == "assistant":
            kind = _INSIGHTS_CARRYING_ASSISTANT
        elif row.role == "tool":
            kind = _INSIGHTS_CARRYING_TOOL_RESULT
        else:
            kind = _INSIGHTS_CARRYING_ITEM
        label = self._render_message(kind.bind())
        if not row.tool_names:
            return label
        names = ", ".join(
            f"{name} ×{count}" if count > 1 else name  # noqa: RUF001
            for name, count in Counter(row.tool_names).items()
        )
        return f"{label} ({names})"

    def _session_info_lines(self, analysis: TrajectoryAnalysis, *, width: int, columns: int) -> list[Text]:
        """Folder, on-disk footprint, and wall-clock anchors of the session."""
        folder = self._displayed_folder()
        storage = self._session_storage
        span = analysis.session_span
        label_style = Style()

        def label(message: MessageDef) -> Text:
            return Text(self._render_message(message.bind()), style=label_style)

        def size(value: int | None) -> Text:
            return Text("—" if value is None else format_byte_size(value))

        controls = Text()
        if folder is not None:
            controls.append(
                f"⎘ {self._render_message(_SESSION_INFO_COPY_PATH.bind())}",
                style=self._semantic_style("primary", "blue", bold=True)
                + Style(underline=True)
                + Style.from_meta({"@click": "copy_session_path"}),
            )
        if self._can_open_session_folder:
            if controls:
                controls.append("  ")
            controls.append(
                f"⧉ {self._render_message(_SESSION_INFO_OPEN_FOLDER.bind())}",
                style=self._semantic_style("primary", "blue", bold=True)
                + Style(underline=True)
                + Style.from_meta({"@click": "open_session_folder"}),
            )
        folder_label = label(_SESSION_INFO_PATH)
        path_room = width - cell_len(folder_label.plain) - 2
        if controls:
            path_room -= cell_len(controls.plain) + 2
        folder_text = Text(
            _fit_path_tail(surrogate_safe_text(str(folder)), max(1, path_room)) if folder is not None else "—"
        )
        lines = [_align_edges(Text.assemble(folder_label, Text("  "), folder_text), controls, width)]
        on_disk = Text("—")
        if storage is not None:
            on_disk = Text.assemble(
                Text(format_byte_size(storage.total_bytes)),
                Text(" · ", style="dim"),
                Text(self._render_message(_SESSION_INFO_FILES.bind(count=storage.file_count)), style="dim"),
            )
        first_message = _local_clock(span.first_turn_started_at) if span is not None else None
        last_reply = _local_clock(span.last_turn_finished_at) if span is not None else None
        clock_span = (
            _clock_span_ns(span.first_turn_started_at, span.last_turn_finished_at) if span is not None else None
        )
        groups: list[list[tuple[Text, Text]]] = [
            [
                (label(_SESSION_INFO_ON_DISK), on_disk),
                (Text("session.json", style=label_style), size(storage.session_json_bytes if storage else None)),
                (Text("events.jsonl", style=label_style), size(storage.events_bytes if storage else None)),
            ],
            [
                (label(_SESSION_INFO_MUTATIONS), size(storage.mutations_bytes if storage else None)),
                (label(_SESSION_INFO_SNAPSHOTS), size(storage.snapshots_bytes if storage else None)),
                (label(_SESSION_INFO_SUB_AGENTS), size(storage.sub_agents_bytes if storage else None)),
            ],
            [
                (label(_SESSION_INFO_FIRST_MESSAGE), Text(first_message or "—")),
                (label(_SESSION_INFO_LAST_REPLY), Text(last_reply or "—")),
                (label(_SESSION_INFO_SPAN), Text("—" if clock_span is None else _format_duration(clock_span))),
            ],
            [
                (label(_SESSION_INFO_TURNS), Text(str(len(analysis.turns)))),
                (label(_SESSION_INFO_EVENTS), Text(f"{analysis.diagnostics.line_count:,}")),
                (label(_SESSION_INFO_RUNTIMES), Text("—" if span is None else str(span.runtime_count))),
            ],
        ]
        lines.extend(_grouped_grid_lines(groups, width=width, columns=columns))
        return lines

    def _displayed_folder(self) -> Path | None:
        """The folder the session info section names: the session directory when
        the events log sits in the store layout, otherwise the log's own parent."""
        path = self._path
        if path is None:
            return None
        session_dir = _session_directory(path)
        return session_dir if session_dir is not None else path.parent

    def open_session_folder(self) -> None:
        """Reveal the displayed session folder in the desktop file manager."""
        folder = self._displayed_folder()
        if folder is None:
            return
        if not can_open_in_file_manager():
            self.notify(self._render_message(_SESSION_INFO_OPEN_UNAVAILABLE.bind()), severity="warning", markup=False)
            return
        try:
            open_in_file_manager(folder)
        except OSError as error:
            # The OS error text is data, not markup: "[Errno 2] ..." would
            # otherwise be parsed as a content tag.
            self.notify(
                self._render_message(_SESSION_INFO_OPEN_FAILED.bind(error=surrogate_safe_text(str(error)))),
                severity="error",
                markup=False,
            )

    def copy_session_path(self) -> None:
        """Copy the displayed session folder path to the available clipboards."""
        folder = self._displayed_folder()
        if folder is None:
            return
        copy_text_to_clipboards(self.app, str(folder))
        self.notify(
            self._render_message(_SESSION_INFO_PATH_COPIED.bind()),
            title=self._render_message(COPIED_TITLE.bind()),
            timeout=1,
            markup=False,
        )

    def _kpi_time_lines(self, overview: TrajectoryOverview, *, width: int) -> list[Text]:
        cells = [
            self._metric_cell(_ELAPSED, overview.elapsed_ns),
            self._metric_cell(_WORK, overview.exclusive_work_ns),
            self._metric_cell(_CP_RESPONSE, overview.response_cp_ns),
            self._metric_cell(_CP_COMPUTE, overview.compute_cp_ns),
            self._metric_cell(_USAGE, overview.usage_tokens, tokens=True),
            self._coverage_cell(overview),
        ]
        return [_align_edges(left, right, width) for left, right in cells]

    def _kpi_split_lines(self, overview: TrajectoryOverview, *, width: int) -> list[Text]:
        wall_percentages = _partition_percentages(overview)
        cells = [
            self._percentage_cell(label, overview.wall_time_ns[bucket], wall_percentages[bucket], bucket=bucket)
            for bucket, label in (
                (WallBucket.MODEL, _MODEL),
                (WallBucket.TOOLS, _TOOLS),
                (WallBucket.WAIT, _WAIT),
                (WallBucket.IDLE, _IDLE),
            )
        ]
        return [_align_edges(left, right, width) for left, right in cells]

    def _kpi_parallel_lines(self, overview: TrajectoryOverview, *, width: int) -> list[Text]:
        lines = [
            _align_edges(*self._metric_cell(_PARALLELISM, overview.parallelism), width),
            _align_edges(*self._metric_cell(_OVERLAP, overview.overlap_gain_ns), width),
            Text(self._render_message(_UTILIZATION.bind()), style="dim"),
        ]
        for bucket, label in ((WallBucket.MODEL, _MODEL), (WallBucket.TOOLS, _TOOLS)):
            lines.append(_align_edges(*self._utilization_cell(label, overview.utilization[bucket], bucket), width))
        return lines

    def _precision_legend(self) -> Text:
        # Only the glyphs carry semantic colour; the labels inherit the border
        # subtitle's own colour so they read like the session id beside them.
        legend = Text()
        for index, precision in enumerate(_PRECISION_SYMBOLS):
            if index:
                legend.append("   ")
            legend.append(_PRECISION_SYMBOLS[precision], self._precision_style(precision))
            legend.append(" ")
            legend.append(self._precision_value(precision))
        return legend

    def _coverage_cell(self, overview: TrajectoryOverview) -> tuple[Text, Text]:
        metrics = _overview_metrics(overview)
        total = max(1, len(metrics))
        exact = sum(metric.precision is Precision.EXACT for metric in metrics) * 100 / total
        estimated = sum(metric.precision is Precision.ESTIMATED for metric in metrics) * 100 / total
        missing = sum(metric.precision is Precision.MISSING for metric in metrics) * 100 / total
        unresolved = 100.0 - exact - estimated - missing
        return (
            Text(self._render_message(_COVERAGE.bind())),
            Text.assemble(
                coverage_bar(
                    exact,
                    estimated,
                    missing,
                    unresolved,
                    width=5,
                    exact_style=self._precision_style(Precision.EXACT),
                    estimated_style=self._precision_style(Precision.ESTIMATED),
                    missing_style=self._precision_style(Precision.MISSING),
                    unresolved_style=self._precision_style(Precision.UNRESOLVED),
                ),
                Text(" "),
                Text(
                    self._render_message(
                        _COVERAGE_SHARES.bind(
                            exact=f"{exact:.0f}",
                            estimated=f"{estimated:.0f}",
                            missing=f"{missing:.0f}",
                            unresolved=f"{unresolved:.0f}",
                        )
                    )
                ),
            ),
        )

    def _waterfall_lines(self, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
        if not analysis.turns:
            return [Text(self._render_message(_NO_TURNS.bind()))]
        chart_width = max(12, width - 10)
        # Lane order doubles as the tie-break when two buckets cover a cell
        # equally; tools and waits outrank the model so short calls stay visible.
        lane_specs = (
            (WallBucket.TOOLS, _TOOLS, "▬"),
            (WallBucket.WAIT, _WAIT, "▒"),
            (WallBucket.MODEL, _MODEL, "█"),
            (WallBucket.IDLE, _IDLE, "·"),
        )
        turns: list[tuple[int, dict[WallBucket, list[tuple[int, int]]]]] = []
        for turn in analysis.turns:
            span = max(0, turn.axis_end_ns - turn.axis_start_ns)
            intervals: dict[WallBucket, list[tuple[int, int]]] = {bucket: [] for bucket, _, _ in lane_specs}
            for item in turn.slices:
                if (
                    item.wall_bucket in intervals
                    and item.end_ns > turn.axis_start_ns
                    and item.start_ns < turn.axis_end_ns
                ):
                    intervals[item.wall_bucket].append(
                        (
                            max(0, item.start_ns - turn.axis_start_ns),
                            min(span, item.end_ns - turn.axis_start_ns),
                        )
                    )
            turns.append((span, intervals))
        lanes = waterfall_lanes(
            turns,
            width=chart_width,
            lanes=[(bucket, glyph, self._bucket_style(bucket)) for bucket, _, glyph in lane_specs],
        )
        lines = [
            Text.assemble(
                Text(fit_cells(self._render_message(label.bind()), 8)),
                Text(" "),
                lanes[bucket],
            )
            for bucket, label, _ in (
                (WallBucket.MODEL, _MODEL, "█"),
                (WallBucket.TOOLS, _TOOLS, "▬"),
                (WallBucket.WAIT, _WAIT, "▒"),
                (WallBucket.IDLE, _IDLE, "·"),
            )
        ]
        # The axis is cumulative turn time (turn spans concatenated, gaps
        # between turns excluded) — the same scale the lanes are drawn on.
        ruler = time_ruler(sum(span for span, _ in turns), width=chart_width)
        ruler.stylize("dim")
        lines.append(
            Text.assemble(
                Text(fit_cells(self._render_message(_TIME_RULER.bind()), 8)),
                Text(" "),
                ruler,
            )
        )
        return lines

    def _finding_lines(self, analysis: TrajectoryAnalysis) -> list[Text]:
        lines: list[Text] = []
        if not analysis.findings:
            lines.append(Text(self._render_message(_NO_FINDINGS.bind()), style="dim"))
        else:
            for finding in analysis.findings:
                severity_style = self._finding_style(finding.severity)
                lines.append(
                    Text.assemble(
                        Text(f"{_finding_glyph(finding.severity)} ", style=severity_style),
                        Text(
                            self._render_message(_FINDING_TITLES[finding.rule_id].bind()),
                            style=severity_style,
                        ),
                        Text("  "),
                        self._precision_badge(finding.precision),
                    )
                )
                lines.append(
                    Text.assemble(
                        Text("    "),
                        Text(self._finding_detail(finding), style="dim"),
                    )
                )
        return lines

    def _finding_detail(self, finding: FindingRow) -> str:
        definition = _FINDING_DETAILS[finding.rule_id]
        args = dict(finding.detail_args)
        if finding.rule_id in {"unverified-change", "repeated-tool-fingerprint", "net-zero-churn"}:
            return self._render_message(definition.bind(count=args["count"]))
        if finding.rule_id == "retry-token-amplification":
            tokens = args["tokens"]
            return self._render_message(definition.bind(count=tokens, tokens=_format_tokens(tokens)))
        if finding.rule_id == "context-carrying-load":
            load = args["load"]
            return self._render_message(definition.bind(count=load, load=_format_tokens(load)))
        return self._render_message(definition.bind(**args))

    def _action_funnel_lines(self, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
        validation = analysis.validation
        if validation is None:
            return []
        metrics = (
            (_ACTION_SEARCH, validation.funnel.search, ActionClass.SEARCH),
            (_ACTION_READ, validation.funnel.read, ActionClass.READ),
            (_ACTION_EDIT, validation.funnel.edit, ActionClass.EDIT),
            (_ACTION_VERIFY, validation.funnel.verify, ActionClass.VERIFY),
        )
        count_width = max(cell_len(_format_count(metric)) for _, metric, _ in metrics)
        badge_width = max(cell_len(self._precision_badge(metric.precision).plain) for _, metric, _ in metrics)
        label_reserve = min(8, max(cell_len(self._render_message(label.bind())) for label, _, _ in metrics))
        bar_width = max(1, min(12, width - label_reserve - count_width - badge_width - 3))
        maximum = max((int(metric.value) for _, metric, _ in metrics if metric.value is not None), default=0)
        lines: list[Text] = []
        for label, metric, action in metrics:
            count = int(metric.value) if metric.value is not None else None
            bar_length = 0 if maximum == 0 or count is None else max(1, round(count / maximum * bar_width))
            lines.append(
                _align_edges(
                    Text(self._render_message(label.bind())),
                    Text.assemble(
                        _fit_text_right(Text(_format_count(metric)), count_width),
                        Text(" "),
                        Text("▬" * bar_length, style=self._action_style(action)),
                        Text(" " * (bar_width - bar_length)),
                        Text(" "),
                        _fit_text_right(self._precision_badge(metric.precision), badge_width),
                    ),
                    width,
                )
            )
        return lines

    def _failure_recovery_lines(self, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
        validation = analysis.validation
        if validation is None:
            return []
        rows = (
            (
                self._render_message(_TOOL_FAILURES.bind()),
                Text(f"{_format_count(validation.tool_failure_count)}/{_format_count(validation.tool_count)}"),
                validation.tool_failure_count,
            ),
            (
                self._render_message(_MEDIAN_RECOVERY.bind()),
                Text(self._metric_value(validation.failure_recovery_median_ns)),
                validation.failure_recovery_median_ns,
            ),
            (
                self._render_message(_RETRY_AMPLIFICATION.bind()),
                Text(
                    self._render_message(
                        _TOKEN_COUNT.bind(tokens=_format_count(validation.retry_amplification_tokens, signed=True))
                    )
                ),
                validation.retry_amplification_tokens,
            ),
            (
                self._render_message(_REPEATED_SIGNATURES.bind()),
                Text(_format_count(validation.repeated_failure_signature_count)),
                validation.repeated_failure_signature_count,
            ),
        )
        value_width = max(cell_len(value.plain) for _, value, _ in rows)
        badge_width = max(cell_len(self._precision_badge(metric.precision).plain) for _, _, metric in rows)
        return [
            _align_edges(
                Text(label),
                Text.assemble(
                    _fit_text_right(value, value_width),
                    Text(" "),
                    _fit_text_right(self._precision_badge(metric.precision), badge_width),
                ),
                width,
            )
            for label, value, metric in rows
        ]

    def _token_usage_lines(self, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
        usage = analysis.token_usage
        if usage is None:
            return []
        cache_hit = _cache_hit_metric(usage)
        rows = [
            (
                self._render_message(label.bind()),
                Text(_format_tokens(usage.buckets[bucket].value)),
                usage.buckets[bucket],
            )
            for bucket, label in (
                (UsageBucket.INPUT, _TOKEN_INPUT),
                (UsageBucket.OUTPUT, _TOKEN_OUTPUT),
                (UsageBucket.REASONING, _TOKEN_REASONING),
                (UsageBucket.CACHE_READ, _TOKEN_CACHE_READ),
            )
        ]
        rows.append(
            (
                self._render_message(_TOKEN_CACHE_HIT.bind()),
                Text("—" if cache_hit.value is None else f"{cache_hit.value}%"),
                cache_hit,
            )
        )
        value_width = max(cell_len(value.plain) for _, value, _ in rows)
        badge_width = max(cell_len(self._precision_badge(metric.precision).plain) for _, _, metric in rows)
        return [
            _align_edges(
                Text(label),
                Text.assemble(
                    _fit_text_right(value, value_width),
                    Text(" "),
                    _fit_text_right(self._precision_badge(metric.precision), badge_width),
                ),
                width,
            )
            for label, value, metric in rows
        ]

    def _tool_usage_lines(self, panel: ToolUsagePanel | None, *, empty: MessageDef, width: int) -> list[Text]:
        if panel is None:
            return []
        if not panel.total:
            return [Text(self._render_message(empty.bind()), style="dim")]
        rows = [(row.name, row.count) for row in panel.rows]
        if panel.unattributed:
            rows.append((self._render_message(_TOOL_USAGE_UNATTRIBUTED.bind()), panel.unattributed))
            rows.sort(key=lambda row: -row[1])
        visible = rows[:5]
        count_width = max(cell_len(f"{count:,}") for _, count in visible)
        badge = self._precision_badge(panel.precision)
        lines = [
            _align_edges(
                Text(name),
                Text.assemble(
                    _fit_text_right(Text(f"{count:,}"), count_width),
                    Text(" "),
                    badge.copy(),
                ),
                width,
            )
            for name, count in visible
        ]
        if len(rows) > len(visible):
            lines.append(
                Text(
                    self._render_message(_TOOL_USAGE_MORE.bind(value=len(rows) - len(visible))),
                    style="dim",
                )
            )
        return lines

    def _change_verification_lines(self, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
        change = analysis.change_verification
        if change is None:
            return []
        lines: list[Text] = []
        counts = self._render_message(
            _CHANGE_COUNTS.bind(
                files=_format_count(change.files_touched),
                created=_format_count(change.created),
                modified=_format_count(change.modified),
                deleted=_format_count(change.deleted),
                net_zero=_format_count(change.net_zero),
            )
        )
        # The counts fold provenance and skip evidence into their precision;
        # one badge for the worst of them keeps that visible in the line.
        precision_order = (Precision.EXACT, Precision.ESTIMATED, Precision.MISSING, Precision.UNRESOLVED)
        counts_precision = max(
            (
                metric.precision
                for metric in (
                    change.files_touched,
                    change.created,
                    change.modified,
                    change.deleted,
                    change.net_zero,
                )
            ),
            key=precision_order.index,
        )
        lines.append(
            _align_edges_badged(
                Text(self._render_message(_CHANGE_FILES.bind())),
                Text(counts),
                self._precision_badge(counts_precision),
                width,
            )
        )
        if not change.detail_available:
            lines.append(Text(self._render_message(_CHANGE_DETAIL_UNAVAILABLE.bind()), style="dim"))
        else:
            states = {
                ChangeVerificationState.VERIFIED: _CHANGE_STATE_VERIFIED,
                ChangeVerificationState.AFTER_VERIFY: _CHANGE_STATE_AFTER,
                ChangeVerificationState.UNVERIFIED: _CHANGE_STATE_UNVERIFIED,
                ChangeVerificationState.NET_ZERO: _CHANGE_STATE_NET_ZERO,
            }
            for row in change.rows:
                style = (
                    self._semantic_style("success", "green")
                    if row.state is ChangeVerificationState.VERIFIED
                    else self._semantic_style("warning", "yellow")
                )
                lines.append(
                    _align_path_edges_badged(
                        surrogate_safe_text(row.path),
                        style,
                        Text(self._render_message(states[row.state].bind()), style=style),
                        self._precision_badge(row.precision),
                        width,
                    )
                )
        if change.detection_truncated:
            lines.append(Text(self._render_message(_CHANGE_DETECTION_TRUNCATED.bind()), style="dim"))
        return lines

    def _submission_latency_lines(self, analysis: TrajectoryAnalysis, *, width: int) -> list[Text]:
        submission = analysis.submission_latency
        if submission is None:
            return []
        bucket_labels = {
            SubmissionLatencyBucket.BECAME_TURN: _SUBMISSION_BECAME_TURN,
            SubmissionLatencyBucket.INJECTED: _SUBMISSION_INJECTED,
            SubmissionLatencyBucket.DID_NOT_BECOME_TURN: _SUBMISSION_DID_NOT_BECOME,
        }
        lines: list[Text] = [Text(self._render_message(_SUBMISSION_INTRO.bind()), style="dim")]
        rendered_any = False
        for stats in submission.buckets:
            if not stats.sample_count and not stats.unresolved_count and not stats.samples:
                continue
            rendered_any = True
            label = self._render_message(bucket_labels[stats.bucket].bind())
            aggregate_precision, _ = _derived_metric_precision(stats.p50_ns, stats.p90_ns, stats.max_ns)
            lines.append(
                _align_edges_badged(
                    Text(label),
                    Text(
                        self._render_message(
                            _SUBMISSION_STATS.bind(
                                count=stats.sample_count,
                                value=stats.sample_count,
                                p50=self._metric_value(stats.p50_ns),
                                p90=self._metric_value(stats.p90_ns),
                                maximum=self._metric_value(stats.max_ns),
                            )
                        )
                    ),
                    self._precision_badge(aggregate_precision),
                    width,
                )
            )
            if stats.unresolved_count:
                lines.append(
                    _align_edges(
                        Text(),
                        Text(
                            self._render_message(_SUBMISSION_UNRESOLVED.bind(count=stats.unresolved_count)),
                            style=self._semantic_style("warning", "yellow"),
                        ),
                        width,
                    )
                )
            lines.extend(
                _align_edges(
                    Text(self._submission_sample_label(analysis, sample)),
                    Text.assemble(
                        Text(
                            self._render_message(
                                _SUBMISSION_SAMPLE.bind(
                                    outcome=self._preparation_outcome(sample.outcome),
                                    duration=self._metric_value(sample.duration_ns),
                                )
                            )
                        ),
                        Text(" "),
                        self._precision_badge(sample.duration_ns.precision),
                    ),
                    width,
                )
                for sample in stats.samples
            )
        if not rendered_any:
            lines.append(Text(self._render_message(_SUBMISSION_NONE.bind()), style="dim"))
        return lines

    def _submission_sample_label(self, analysis: TrajectoryAnalysis, sample: SubmissionLatencySample) -> str:
        turn = analysis.turn(sample.turn_id) if sample.turn_id is not None else None
        if turn is not None and turn.turn_number is not None:
            return self._render_message(_TURN.bind(turn=turn.turn_number))
        return _callsite(sample.scope_operation_id)

    def _preparation_outcome(self, outcome: str | None) -> str:
        if outcome is None:
            return "—"
        definition = _PREPARATION_OUTCOMES.get(outcome)
        return self._render_message(definition.bind()) if definition is not None else surrogate_safe_text(outcome)

    def _bordered_section_row(
        self,
        specs: tuple[tuple[MessageDef | MessageRef | Text, list[Text], int], ...],
    ) -> list[Text]:
        boxes = [self._section_box(title, lines, width=box_width) for title, lines, box_width in specs]
        content_height = max(len(box) - 2 for box in boxes)
        boxes = [
            self._section_box(title, lines, width=box_width, content_height=content_height)
            for title, lines, box_width in specs
        ]
        rows: list[Text] = []
        for index in range(len(boxes[0])):
            parts: list[Text] = []
            for box in boxes:
                if parts:
                    parts.append(Text(" "))
                parts.append(box[index])
            rows.append(Text.assemble(*parts))
        return rows

    def _finding_style(self, severity: FindingSeverity) -> Style:
        if severity is FindingSeverity.ERROR:
            return self._semantic_style("error", "red", bold=True)
        if severity is FindingSeverity.WARNING:
            return self._semantic_style("warning", "yellow", bold=True)
        return Style(dim=True)

    def _action_style(self, action: ActionClass) -> Style:
        if action is ActionClass.EDIT:
            return self._semantic_style("warning", "yellow", bold=True)
        if action is ActionClass.VERIFY:
            return self._semantic_style("success", "green", bold=True)
        if action is ActionClass.SEARCH:
            return self._semantic_style("primary", "blue", bold=True)
        return self._semantic_style("secondary", "cyan", bold=True)

    def _diagnostic_lines(self, analysis: TrajectoryAnalysis) -> list[Text]:
        diagnostics = analysis.diagnostics
        lines = [Text(self._render_message(_DIAGNOSTICS_INTRO.bind()), style="dim")]
        problems: list[Text] = []
        if diagnostics.corrupt_line_count:
            corrupt_sequences = ", ".join(
                self._render_message(_AFTER_SEQUENCE.bind(sequence=item.after_sequence))
                for item in diagnostics.corrupt_lines
            )
            problems.append(
                Text(
                    self._render_message(
                        _CORRUPT_LINES.bind(count=diagnostics.corrupt_line_count, sequences=corrupt_sequences)
                    ),
                    style=self._semantic_style("error", "red"),
                )
            )
        if diagnostics.unsupported_event_count:
            unsupported_sequences = ", ".join(
                self._render_message(_SEQUENCE.bind(sequence=item.sequence)) for item in diagnostics.unsupported_lines
            )
            problems.append(
                Text(
                    self._render_message(
                        _UNSUPPORTED_LINES.bind(
                            count=diagnostics.unsupported_event_count,
                            sequences=unsupported_sequences,
                        )
                    ),
                    style=self._semantic_style("error", "red"),
                )
            )
        problems.extend(
            Text(
                self._render_message(
                    _ACCOUNTED_PREFIX.bind(
                        first=item.first_sequence,
                        last=item.last_sequence,
                        reason=item.message,
                    )
                ),
                style=self._semantic_style("error", "red"),
            )
            for item in diagnostics.accounted_prefix_violation_details
        )
        for label, metric in self._overview_metric_items(analysis.overview):
            if metric.precision is Precision.UNRESOLVED:
                problems.append(self._unresolved_metric_line(label, metric))
        for turn in analysis.turns:
            turn_label = self._render_message(_TURN.bind(turn=turn.turn_number or "—"))
            for label, metric in self._turn_metric_items(turn):
                if metric.precision is Precision.UNRESOLVED:
                    problems.append(self._unresolved_metric_line(f"{turn_label} {label}", metric))
        problems.extend(self._operation_diagnostic_line(item) for item in diagnostics.timeline_operations)
        problems.extend(
            Text(
                self._render_message(
                    _CONTAINMENT.bind(
                        family=item.family,
                        callsite=_callsite(item.operation_id),
                        parent_family=item.parent_family,
                        parent_callsite=_callsite(item.parent_operation_id),
                    )
                ),
                style=self._semantic_style("error", "red"),
            )
            for item in diagnostics.containment_violations
        )
        if diagnostics.torn_tail_bytes:
            problems.append(
                Text(
                    self._render_message(_TORN_TAIL.bind(bytes=diagnostics.torn_tail_bytes)),
                    style=self._semantic_style("error", "red"),
                )
            )
        problems.extend(
            Text(
                self._render_message(
                    _EXPLICIT_GAP.bind(first=item.first_sequence, last=item.last_sequence, reason=item.message)
                ),
                style=self._semantic_style("error", "red"),
            )
            for item in diagnostics.explicit_gaps
        )
        if diagnostics.rollback_projection_unresolved:
            problems.append(
                Text(
                    self._render_message(_ROLLBACK_UNRESOLVED.bind()),
                    style=self._semantic_style("warning", "yellow"),
                )
            )
        if diagnostics.malformed_hook_execution_mode_count:
            problems.append(
                Text(
                    self._render_message(
                        _MALFORMED_HOOK_MODES.bind(count=diagnostics.malformed_hook_execution_mode_count)
                    ),
                    style=self._semantic_style("error", "red"),
                )
            )
        if problems:
            lines.extend(problems)
        else:
            lines.append(
                Text(
                    self._render_message(_DIAGNOSTICS_HEALTHY.bind()),
                    style=self._semantic_style("success", "green"),
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
                    self._render_message(
                        _DURATION_MISMATCH_SUMMARY.bind(
                            count=len(mismatches),
                            value=len(mismatches),
                            families=families,
                            delta=_format_duration(delta_ns),
                        )
                    ),
                    style="dim",
                )
            )
        if diagnostics.side_call_empty_shell_revisions:
            shell_count = len(diagnostics.side_call_empty_shell_revisions)
            lines.append(
                Text(
                    self._render_message(_SIDE_CALL_EMPTY_SHELLS.bind(count=shell_count, value=shell_count)),
                    style="dim",
                )
            )
        if diagnostics.unidentified_membership_revision_count:
            unidentified_count = diagnostics.unidentified_membership_revision_count
            lines.append(
                Text(
                    self._render_message(
                        _UNIDENTIFIED_MEMBERSHIP.bind(count=unidentified_count, value=unidentified_count)
                    ),
                    style="dim",
                )
            )
        return lines

    def _unresolved_metric_line(self, label: str, metric: Metric) -> Text:
        return Text(
            self._render_message(
                _UNRESOLVED_METRIC.bind(metric=label, reason=metric.reason or self._precision(metric))
            ),
            style=self._semantic_style("warning", "yellow"),
        )

    def _operation_diagnostic_line(self, diagnostic: TimelineOperationDiagnostic) -> Text:
        identity = diagnostic.identity or diagnostic.family
        identity = _identity_with_hook_id(identity, diagnostic.hook_id)
        operation = f"{identity} @{_callsite(diagnostic.operation_id)}"
        reason = self._render_message(_OPERATION_REASON_MESSAGES[diagnostic.code].bind())
        return Text(
            self._render_message(
                _OPERATION_DIAGNOSTIC.bind(
                    turn=diagnostic.turn_number or "—",
                    operation=operation,
                    reason=reason,
                )
            ),
            style=self._precision_style(diagnostic.precision),
        )

    def _timeline_lines(self, analysis: TrajectoryAnalysis) -> list[Text]:
        if not analysis.turns:
            return [Text(self._render_message(_NO_TURNS.bind()))]
        self._selected_timeline_line = None
        turn = analysis.turn(self._selected_turn_id) if self._selected_turn_id is not None else None
        turn = turn or analysis.turns[-1]
        self._selected_turn_id = turn.turn_id
        title = self._render_message(_TURN.bind(turn=turn.turn_number or "—"))
        lines = [
            Text.assemble(
                Text(title, style=self._section_style()),
                Text("  "),
                self._metric_text(turn.elapsed_ns),
                Text(f"  {self._render_message(_GRAPH_HINT.bind())}", style="dim"),
            )
        ]
        width = self._content_width()
        category_width = _OPERATION_CATEGORY_WIDTH
        suffix_reserve = 20
        # Below the narrowest fitting layout the timeline draws on a fixed
        # 92-cell canvas and scrolls horizontally instead of squeezing its
        # columns into illegibility; the Timeline view owns a horizontal
        # scrollbar, unlike Overview.
        minimum_fit_width = category_width + _TIMELINE_LABEL_WIDTH + suffix_reserve + 4 + 12
        canvas_width = width if width >= minimum_fit_width else max(width, 92)
        label_width = _TIMELINE_WIDE_LABEL_WIDTH if canvas_width >= _WIDE_MIN_COLUMNS else _TIMELINE_LABEL_WIDTH
        # The duration column hugs its widest value so the bars run right up
        # to the figures instead of leaving a gutter of reserved cells.
        resolved_suffixes = [
            _format_duration(operation.duration_ns or 0)
            for operation in turn.operations
            if operation.start_ns is not None and operation.end_ns is not None
        ]
        unresolved_suffixes = [
            f"{_PRECISION_SYMBOLS[operation.precision]} {self._precision_value(operation.precision)}"
            for operation in turn.operations
            if operation.start_ns is None or operation.end_ns is None
        ]
        suffix_width = min(
            suffix_reserve,
            max([8, *(cell_len(suffix) for suffix in (*resolved_suffixes, *unresolved_suffixes))]),
        )
        bar_width = max(12, canvas_width - category_width - label_width - suffix_width - 4)
        prefix_width = category_width + label_width + 3
        span = max(0, turn.axis_end_ns - turn.axis_start_ns)
        ruler = time_ruler(span, width=bar_width)
        ruler.stylize("dim")
        lines.append(
            Text.assemble(
                Text(fit_cells(self._render_message(_TIME_RULER.bind()), prefix_width), style="dim"),
                ruler,
            )
        )
        for row_index, operation in enumerate(turn.operations):
            category = self._operation_category(operation)
            identity = self._operation_identity(operation)
            operation_style = self._operation_style(operation.family)
            bar_style = self._operation_bar_style(operation.family)
            indented = Text.assemble(
                Text("│ " * operation.depth, style="dim"),
                Text(identity, style="dim"),
            )
            if operation.start_ns is None or operation.end_ns is None:
                bar = unresolved_bar(bar_width, style=self._precision_style(operation.precision))
                suffix = Text.assemble(
                    self._precision_badge(operation.precision),
                    Text(
                        f" {self._precision_value(operation.precision)}",
                        style=self._precision_style(operation.precision),
                    ),
                )
            else:
                bar = timeline_bar(
                    operation.start_ns,
                    operation.end_ns,
                    origin=turn.axis_start_ns,
                    span=span,
                    width=bar_width,
                    glyph=_operation_glyph(operation.family),
                    style=bar_style,
                )
                suffix = Text(_format_duration(operation.duration_ns or 0), style="dim")
            line = Text.assemble(
                _fit_text(Text(category, style=operation_style), category_width),
                Text(_OPERATION_CATEGORY_SEPARATOR),
                _fit_text(indented, label_width),
                Text(" "),
                bar,
                Text("  "),
                _fit_text_right(suffix, suffix_width),
            )
            if row_index % 2:
                line.stylize(self._zebra_style(), 0, len(line))
            if operation.operation_id == self._selected_operation_id:
                line.stylize(Style(reverse=True), 0, len(line))
                self._selected_timeline_line = len(lines)
            lines.append(line)
        lines.extend([Text(), Text(self._render_message(_ELAPSED_SCOPE.bind()), style="dim")])
        return lines

    def _dependency_graph_lines(self, analysis: TrajectoryAnalysis) -> list[Text]:
        turn = analysis.turn(self._selected_turn_id) if self._selected_turn_id is not None else None
        turn = turn or analysis.turns[-1]
        self._selected_turn_id = turn.turn_id
        lines = [
            Text.assemble(
                Text(
                    self._render_message(_GRAPH_TITLE.bind(turn=turn.turn_number or "—")),
                    style=self._section_style(),
                ),
                Text(f"  {self._render_message(_TIMELINE_HINT.bind())}", style="dim"),
            ),
            Text(self._render_message(_GRAPH_LEGEND.bind()), style="dim"),
            Text(),
        ]
        flow = turn.flow
        if flow is None:
            lines.append(Text(self._render_message(_GRAPH_NONE.bind()), style="dim"))
            return lines
        if not flow.acyclic:
            lines.append(
                Text(
                    self._render_message(_GRAPH_CYCLE_WARNING.bind()),
                    style=self._precision_style(Precision.UNRESOLVED),
                )
            )
            lines.append(Text())
        operations = turn.operations
        children: dict[int, list[int]] = defaultdict(list)
        has_parent: set[int] = set()
        for source, target in flow.parent_edges():
            if FLOW_TERMINAL_INDEX in (source, target):
                continue
            children[source].append(target)
            has_parent.add(target)
        causal_in: dict[int, list[int]] = defaultdict(list)
        terminal_fan_in = 0
        for source, target in flow.causal_edges():
            if target == FLOW_TERMINAL_INDEX:
                terminal_fan_in += 1
            elif source != FLOW_TERMINAL_INDEX:
                causal_in[target].append(source)

        def order_key(index: int) -> tuple[bool, int, int]:
            operation = operations[index]
            return (operation.start_ns is None, operation.start_ns or 0, index)

        roots = sorted((index for index in range(len(operations)) if index not in has_parent), key=order_key)
        for child_list in children.values():
            child_list.sort(key=order_key)
        seen: set[int] = set()
        stack = [(index, 0) for index in reversed(roots)]
        while stack:
            index, depth = stack.pop()
            if index in seen:
                continue
            seen.add(index)
            unproven = depth == 0 and index != flow.root_index and not causal_in.get(index)
            lines.append(
                self._dependency_node_line(
                    operations,
                    index,
                    depth=depth,
                    causal_sources=causal_in.get(index, []),
                    unproven=unproven,
                )
            )
            stack.extend((child, depth + 1) for child in reversed(children.get(index, [])))
            # A parent cycle leaves its members without a reachable root; once
            # the reachable forest drains, surface them flat rather than drop
            # them (the cycle warning above explains the shape).
            if not stack:
                leftovers = sorted((index for index in range(len(operations)) if index not in seen), key=order_key)
                stack = [(index, 0) for index in reversed(leftovers)]
        if flow.has_terminal:
            lines.append(
                Text.assemble(
                    Text(" " * _OPERATION_CATEGORY_WIDTH),
                    Text(_OPERATION_CATEGORY_SEPARATOR),
                    Text("◆ ", style=self._section_style()),
                    Text(self._render_message(_GRAPH_RESPONSE.bind()), style=self._section_style()),
                    Text(f"  ⇠ {terminal_fan_in}", style="dim"),
                )
            )
        return lines

    def _dependency_node_line(
        self,
        operations: tuple[TimelineOperation, ...],
        index: int,
        *,
        depth: int,
        causal_sources: list[int],
        unproven: bool,
    ) -> Text:
        operation = operations[index]
        style = self._operation_style(operation.family)
        parts = [
            _fit_text(Text(self._family_category(operation.family), style=style), _OPERATION_CATEGORY_WIDTH),
            Text(_OPERATION_CATEGORY_SEPARATOR),
            Text("│ " * depth, style="dim"),
        ]
        if unproven:
            parts.append(Text("┄ ", style="dim"))
        parts.append(Text(self._operation_identity(operation), style=style))
        duration = operation.duration_ns
        if duration is not None:
            parts.append(Text(f"  {_format_duration(duration)}", style="dim"))
        parts.extend((Text(" "), self._precision_badge(operation.precision)))
        if causal_sources:
            labels = ", ".join(self._operation_identity(operations[source]) for source in causal_sources[:2])
            extra = len(causal_sources) - 2
            suffix = f" +{extra}" if extra > 0 else ""
            parts.append(Text(f"  ⇠ {labels}{suffix}", style="dim"))
        return Text.assemble(*parts)

    def _operation_category(self, operation: TimelineOperation) -> str:
        return self._family_category(operation.family)

    def _family_category(self, family: str) -> str:
        if family.startswith("workflow."):
            from chrys.app.tui.widgets.workflow.text import TITLE

            label = TITLE
        elif family.startswith("model."):
            label = _CATEGORY_MODEL
        elif family == "tool.operation":
            label = _CATEGORY_TOOL
        elif family in {"wait", "continuation.poll", "turn.suspension"}:
            label = _CATEGORY_WAIT
        elif family == "hook.operation":
            label = _CATEGORY_HOOK
        elif family == "sub_agent":
            label = _CATEGORY_AGENT
        elif family == "preparation":
            label = _CATEGORY_PREPARATION
        elif family.startswith("compaction"):
            label = _CATEGORY_COMPACTION
        elif family == "approval":
            label = _CATEGORY_APPROVAL
        elif family == "retry":
            label = _CATEGORY_RETRY
        else:
            label = _CATEGORY_OPERATION
        return self._render_message(label.bind())

    def _overview_metric_items(self, overview: TrajectoryOverview | None) -> list[tuple[str, Metric]]:
        if overview is None:
            return []
        return self._metric_items(
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

    def _turn_metric_items(self, turn: TurnAnalysis) -> list[tuple[str, Metric]]:
        return self._metric_items(
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
        self,
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
            (self._render_message(_ELAPSED.bind()), elapsed),
            (self._render_message(_CP_RESPONSE.bind()), response_cp),
            (self._render_message(_CP_COMPUTE.bind()), compute_cp),
            (self._render_message(_WORK.bind()), work),
            (self._render_message(_PARALLELISM.bind()), parallelism),
            (self._render_message(_OVERLAP.bind()), overlap),
            (self._render_message(_USAGE.bind()), usage),
        ]
        bucket_labels = {
            WallBucket.MODEL: self._render_message(_MODEL.bind()),
            WallBucket.TOOLS: self._render_message(_TOOLS.bind()),
            WallBucket.WAIT: self._render_message(_WAIT.bind()),
            WallBucket.IDLE: self._render_message(_IDLE.bind()),
        }
        items.extend(
            (
                self._render_message(_WALL_METRIC.bind(bucket=bucket_labels[bucket])),
                wall[bucket],
            )
            for bucket in WallBucket
        )
        items.extend(
            (
                self._render_message(_UTILIZATION_METRIC.bind(bucket=bucket_labels[bucket])),
                utilization[bucket],
            )
            for bucket in (WallBucket.MODEL, WallBucket.TOOLS)
        )
        return items

    def _operation_identity(self, operation: TimelineOperation) -> str:
        if operation.identity is not None:
            return _identity_with_hook_id(operation.identity, operation.hook_id)
        if operation.family == "tool.operation":
            return self._render_message(_CATEGORY_TOOL.bind())
        if operation.family == "hook.operation":
            return self._render_message(_CATEGORY_HOOK.bind())
        if operation.family in {"wait", "continuation.poll", "turn.suspension"}:
            return self._render_message(_CATEGORY_WAIT.bind())
        if operation.family == "sub_agent":
            return self._render_message(_CATEGORY_AGENT.bind())
        return operation.family.removeprefix("model.").removeprefix("compaction.")

    def _metric_cell(self, label: MessageDef, metric: Metric, *, tokens: bool = False) -> tuple[Text, Text]:
        value = _format_tokens(metric.value) if tokens else self._metric_value(metric)
        return (
            Text(self._render_message(label.bind())),
            Text.assemble(Text(value), Text(" "), self._precision_badge(metric.precision)),
        )

    def _percentage_cell(
        self,
        label: MessageDef,
        metric: Metric,
        value: float | None,
        *,
        bucket: WallBucket,
    ) -> tuple[Text, Text]:
        return (
            Text(self._render_message(label.bind()), style=self._bucket_style(bucket)),
            Text.assemble(
                percentage_meter(value, width=8, style=self._bucket_style(bucket)),
                Text(" "),
                self._precision_badge(metric.precision),
            ),
        )

    def _utilization_cell(self, label: MessageDef, metric: Metric, bucket: WallBucket) -> tuple[Text, Text]:
        value = None if metric.value is None else float(metric.value) * 100
        return (
            Text(self._render_message(label.bind()), style=self._bucket_style(bucket)),
            Text.assemble(
                percentage_meter(value, width=8, style=self._bucket_style(bucket)),
                Text(" "),
                self._precision_badge(metric.precision),
            ),
        )

    def _metric_text(self, metric: Metric) -> Text:
        return Text.assemble(Text(self._metric_value(metric)), Text(" "), self._precision_badge(metric.precision))

    def _metric_value(self, metric: Metric) -> str:
        if metric.value is None:
            return "—"
        if isinstance(metric.value, float):
            return f"{metric.value:.2f}×"  # noqa: RUF001
        return _format_duration(metric.value)

    def _precision_badge(self, precision: Precision) -> Text:
        return Text(_PRECISION_SYMBOLS[precision], style=self._precision_style(precision))

    def _precision_style(self, precision: Precision) -> Style:
        if precision is Precision.EXACT:
            return self._semantic_style("success", "green", bold=True)
        if precision is Precision.ESTIMATED:
            return self._semantic_style("warning", "yellow", bold=True)
        if precision is Precision.UNRESOLVED:
            return self._semantic_style("error", "red", bold=True)
        return Style(dim=True)

    def _cache_hit_style(self, value: int | float | None) -> Style:
        if value is None or value > 60:
            return self._semantic_style("success", "green")
        if value >= 30:
            return self._semantic_style("warning", "yellow")
        return self._semantic_style("error", "red")

    def _bucket_style(self, bucket: WallBucket) -> Style:
        if bucket is WallBucket.MODEL:
            return self._semantic_style("primary", "blue")
        if bucket is WallBucket.TOOLS:
            return self._semantic_style("warning", "yellow")
        if bucket is WallBucket.WAIT:
            return self._semantic_style("accent", "magenta")
        return Style(dim=True)

    def _operation_style(self, family: str) -> Style:
        if family.startswith(("model.", "compaction", "workflow.")):
            return self._semantic_style("primary", "blue", bold=True)
        if family == "tool.operation" or family == "preparation":
            return self._semantic_style("warning", "yellow", bold=True)
        if family == "hook.operation":
            return rich_style_from_textual_color("ansi_cyan", bold=True)
        if family in {"wait", "approval", "retry", "continuation.poll", "turn.suspension"}:
            return self._semantic_style("accent", "magenta", bold=True)
        if family == "sub_agent":
            return self._semantic_style("secondary", "cyan", bold=True)
        return self._semantic_style("foreground", "white")

    def _operation_bar_style(self, family: str) -> Style:
        if family == "model.run":
            return self._semantic_style("accent", "magenta", bold=True)
        return self._operation_style(family)

    def _section_style(self) -> Style:
        return self._semantic_style("primary", "blue", bold=True)

    def _border_style(self) -> Style:
        return Style.combine([self._semantic_style("secondary", "bright_black"), Style(dim=True)])

    def _zebra_style(self) -> Style:
        return self._blended_style(0.08, fill_background=True) or Style()

    def _hatch_style(self) -> Style:
        return hatch_text_style(self)

    def _blended_style(self, factor: float, *, fill_background: bool) -> Style | None:
        """Blend the theme background toward the foreground; None when not blendable (ANSI)."""
        variables = self.app.theme_variables
        try:
            background = Color.parse(variables.get("background", ""))
            foreground = Color.parse(variables.get("foreground", ""))
        except ColorParseError:
            return None
        if background.ansi is not None or foreground.ansi is not None:
            return None
        blended = background.blend(foreground, factor)
        if fill_background:
            return Style(bgcolor=blended.rich_color)
        return Style(color=blended.rich_color)

    def _semantic_style(
        self,
        name: str,
        fallback: str,
        *,
        bold: bool | None = None,
    ) -> Style:
        return rich_style_from_textual_color(self.app.theme_variables.get(name, fallback), bold=bold)

    def _muted_label_style(self) -> Style:
        # $text-muted is a composite value on every theme ("auto 60%",
        # "ansi_white 40%"), which a plain color parse silently drops; only
        # the stylesheet can resolve the alpha blend. Strip the stamped
        # background so the label stays transparent over the hatch.
        full = self.get_component_rich_style("trajectory-dashboard--muted-label")
        return full.without_color + Style.from_color(full.color)

    def _precision(self, metric: Metric) -> str:
        return self._precision_value(metric.precision)

    def _precision_value(self, precision: Precision) -> str:
        return self._render_message(_PRECISION[precision].bind())

    def _section_box(
        self,
        title: MessageDef | MessageRef | Text,
        lines: list[Text],
        *,
        width: int,
        content_height: int | None = None,
    ) -> list[Text]:
        if isinstance(title, Text):
            rendered_title = title
        else:
            reference = title.bind() if isinstance(title, MessageDef) else title
            rendered_title = self._render_message(reference)
        return bordered_section(
            rendered_title,
            lines,
            width=width,
            console=self.app.console,
            border_style=self._border_style(),
            title_style=self._section_style(),
            content_height=content_height,
        )

    def _badged_section_title(self, title: MessageDef | MessageRef, precision: Precision) -> Text:
        reference = title.bind() if isinstance(title, MessageDef) else title
        return Text.assemble(
            Text(self._render_message(reference)),
            Text(" "),
            self._precision_badge(precision),
        )

    def _available_width(self) -> int:
        return self._available_size.width or self.size.width

    def _available_height(self) -> int:
        height = self._available_size.height or self.size.height
        return max(1, height - _TABS_HEIGHT)

    def _render_message(self, reference: MessageRef) -> str:
        if self._locale_controller is None:
            return format_message(reference)
        return render_str(self._locale_controller.localizer, reference)


def _session_directory(path: Path) -> Path | None:
    """The session folder owning *path*, only when it sits in the store layout.

    Anything else (a loose events file under a scratch directory) has no
    session folder to size or open; walking its parent would measure an
    unrelated tree.
    """
    if (
        len(path.parents) < 3
        or path.name != "events.jsonl"
        or path.parent.name != "trajectory"
        or path.parents[2].name != "sessions"
    ):
        return None
    return path.parents[1]


def _load_with_storage(
    analyzer: TrajectoryAnalyzer,
    path: Path,
    *,
    cancel_event: Event,
) -> tuple[TrajectoryAnalysis, SessionStorage | None]:
    analysis = analyzer.load(path, cancel_event=cancel_event)
    session_dir = _session_directory(path)
    if session_dir is None:
        return analysis, None
    return analysis, collect_session_storage(session_dir, cancel_event=cancel_event)


def _refresh_with_storage(
    analyzer: TrajectoryAnalyzer,
    *,
    collect_storage: bool,
    cancel_event: Event,
) -> tuple[TrajectoryAnalysis, SessionStorage | None]:
    analysis = analyzer.refresh(cancel_event=cancel_event)
    if not collect_storage:
        # Storage recollection stays on the caller's coarse clock even while
        # the log is appending, or a busy session walks the directory tree on
        # every poll tick; the panel keeps the previous figures meanwhile.
        return analysis, None
    session_dir = _session_directory(analysis.path)
    if session_dir is None:
        return analysis, None
    return analysis, collect_session_storage(session_dir, cancel_event=cancel_event)


def _parse_clock(value: str | None) -> datetime | None:
    """Parse a producer RFC 3339 stamp; naive or malformed stamps are unusable."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _local_clock(value: str | None) -> str | None:
    """Render a producer RFC 3339 UTC stamp on the viewer's wall clock."""
    parsed = _parse_clock(value)
    return None if parsed is None else parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _clock_span_ns(first: str | None, last: str | None) -> int | None:
    start = _parse_clock(first)
    end = _parse_clock(last)
    if start is None or end is None or end < start:
        return None
    return int((end - start).total_seconds() * 1_000_000_000)


def _fit_path_tail(value: str, width: int) -> str:
    """Keep the end of a path visible when it must be cropped."""
    usable = max(1, width)
    if cell_len(value) <= usable:
        return value
    tail = value
    while tail and cell_len(tail) > usable - 1:
        tail = tail[1:]
    return f"…{tail}"


def _fit_path_middle(value: str, width: int) -> str:
    """Crop a file path in the middle, preserving its full basename first."""
    usable = max(0, width)
    if usable == 0:
        return ""
    if cell_len(value) <= usable:
        return value
    if usable == 1:
        return "…"

    separator_index = max(value.rfind("/"), value.rfind("\\"))
    if separator_index >= 0:
        suffix = value[separator_index:]
        if cell_len(suffix) + 1 <= usable:
            head_room = usable - cell_len(suffix) - 1
            head = Text(value[:separator_index])
            head.truncate(head_room, overflow="crop")
            return f"{head.plain}…{suffix}"

    tail = value[separator_index + 1 :]
    while tail and cell_len(tail) > usable - 1:
        tail = tail[1:]
    return f"…{tail}"


def _fit_text(value: Text, width: int) -> Text:
    fitted = value.copy()
    fitted.truncate(max(0, width), overflow="ellipsis", pad=True)
    return fitted


def _fit_text_right(value: Text, width: int) -> Text:
    fitted = value.copy()
    fitted.truncate(max(0, width), overflow="ellipsis")
    return Text.assemble(Text(" " * max(0, width - cell_len(fitted.plain))), fitted)


def _align_edges(left: Text, right: Text, width: int) -> Text:
    usable = max(0, width)
    fitted_right = right.copy()
    fitted_right.truncate(usable, overflow="ellipsis")
    right_width = cell_len(fitted_right.plain)
    separator_width = int(bool(left.plain and fitted_right.plain and right_width < usable))
    fitted_left = left.copy()
    fitted_left.truncate(max(0, usable - right_width - separator_width), overflow="ellipsis")
    gap = usable - cell_len(fitted_left.plain) - right_width
    return Text.assemble(fitted_left, Text(" " * gap), fitted_right)


def _align_edges_badged(left: Text, value: Text, badge: Text, width: int) -> Text:
    """Two-edge alignment whose right side ends in a badge that survives.

    ``_align_edges`` truncates the right side from its tail, which would
    drop an appended precision badge exactly in the narrow columns where
    the value is most compressed; the value is fitted first so the badge
    always stays visible.
    """
    badge_width = cell_len(badge.plain)
    fitted_value = value.copy()
    fitted_value.truncate(max(0, width - badge_width - 1), overflow="ellipsis")
    return _align_edges(left, Text.assemble(fitted_value, Text(" "), badge), width)


def _align_path_edges_badged(path: str, path_style: Style, value: Text, badge: Text, width: int) -> Text:
    """Align a path while reserving its available cells for middle cropping."""
    usable = max(0, width)
    badge_width = cell_len(badge.plain)
    fitted_value = value.copy()
    fitted_value.truncate(max(0, usable - badge_width - 1), overflow="ellipsis")
    right = Text.assemble(fitted_value, Text(" "), badge)
    right_width = cell_len(right.plain)
    separator_width = int(bool(path and right.plain and right_width < usable))
    path_room = max(0, usable - right_width - separator_width)
    fitted_path = Text(_fit_path_middle(path, path_room), style=path_style)
    return _align_edges(fitted_path, right, usable)


def _grouped_grid_lines(
    groups: list[list[tuple[Text, Text]]],
    *,
    width: int,
    columns: int,
    gap: int = 3,
) -> list[Text]:
    """Lay each semantic group down one column before starting the next band."""
    columns = max(1, columns)
    usable = max(columns, width - gap * (columns - 1))
    base = usable // columns
    widths = [base] * (columns - 1) + [usable - base * (columns - 1)]
    lines: list[Text] = []
    for group_start in range(0, len(groups), columns):
        band = groups[group_start : group_start + columns]
        if lines:
            lines.append(Text())
        for row_index in range(max((len(group) for group in band), default=0)):
            parts: list[Text] = []
            for column_index, column_width in enumerate(widths):
                if column_index:
                    parts.append(Text(" " * gap))
                if column_index < len(band) and row_index < len(band[column_index]):
                    label, value = band[column_index][row_index]
                else:
                    label, value = Text(), Text()
                parts.append(_align_edges(label, value, column_width))
            lines.append(Text.assemble(*parts))
    return lines


def _section_row_widths(width: int, count: int) -> tuple[int, ...]:
    gaps = count - 1
    base = max(6, (width - gaps) // count)
    return (*(base,) * (count - 1), max(6, width - base * (count - 1) - gaps))


def _partition_percentages(overview: TrajectoryOverview) -> dict[WallBucket, float | None]:
    elapsed = overview.elapsed_ns.value
    values = [overview.wall_time_ns[bucket].value for bucket in WallBucket]
    if elapsed is None or float(elapsed) <= 0 or any(value is None for value in values):
        return dict.fromkeys(WallBucket)
    numeric_values = [value for value in values if value is not None]
    raw_tenths = [float(value) / float(elapsed) * 1000 for value in numeric_values]
    tenths = [int(value) for value in raw_tenths]
    for index in sorted(
        range(len(tenths)),
        key=lambda item: raw_tenths[item] - tenths[item],
        reverse=True,
    )[: 1000 - sum(tenths)]:
        tenths[index] += 1
    return {bucket: tenths[index] / 10 for index, bucket in enumerate(WallBucket)}


def _overview_metrics(overview: TrajectoryOverview | None) -> list[Metric]:
    if overview is None:
        return []
    return [
        overview.elapsed_ns,
        overview.response_cp_ns,
        overview.compute_cp_ns,
        overview.exclusive_work_ns,
        overview.parallelism,
        overview.overlap_gain_ns,
        overview.usage_tokens,
        *(overview.wall_time_ns[bucket] for bucket in WallBucket),
        *(overview.utilization[bucket] for bucket in (WallBucket.MODEL, WallBucket.TOOLS)),
    ]


def _operation_glyph(family: str) -> str:
    if family.startswith("model."):
        return "▮"
    if family in {"wait", "approval", "retry", "continuation.poll", "turn.suspension"}:
        return "▭"
    if family == "hook.operation":
        return "◆"
    return "▨"


def _finding_glyph(severity: FindingSeverity) -> str:
    if severity is FindingSeverity.ERROR:
        return "!"
    if severity is FindingSeverity.WARNING:
        return "△"
    return "•"


def _callsite(operation_id: str) -> str:
    return operation_id[:8]


def _format_duration(value_ns: int | float) -> str:
    seconds = float(value_ns) / 1_000_000_000
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


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
    precision, reason = _derived_metric_precision(turn_input, total_input)
    return Metric(percent, precision, reason)


def _cache_hit_metric(usage: TokenUsage) -> Metric:
    """Derive the cache-hit share of input tokens from the display buckets."""
    cache_read = usage.buckets[UsageBucket.CACHE_READ]
    total_input = usage.buckets[UsageBucket.INPUT]
    if cache_read.value is None:
        return Metric(None, cache_read.precision, cache_read.reason)
    if total_input.value is None:
        return Metric(None, total_input.precision, total_input.reason)
    if float(total_input.value) == 0:
        return Metric(None, Precision.MISSING, total_input.reason)
    if (
        float(total_input.value) < 0
        or float(cache_read.value) < 0
        or float(cache_read.value) > float(total_input.value)
    ):
        return Metric(None, Precision.UNRESOLVED)
    percent = 100 * float(cache_read.value) / float(total_input.value)
    value = min(100, max(1, round(percent))) if cache_read.value else 0
    precision, reason = _derived_metric_precision(cache_read, total_input)
    return Metric(value, precision, reason)


def _derived_metric_precision(*metrics: Metric) -> tuple[Precision, str | None]:
    for precision in (Precision.UNRESOLVED, Precision.MISSING, Precision.ESTIMATED):
        for metric in metrics:
            if metric.precision is precision:
                return precision, metric.reason
    return Precision.EXACT, None


def _format_tokens(value: int | float | None) -> str:
    """Compact token count sharing the chat status bar's k/m/b units."""
    if value is None:
        return "—"
    return format_token_count(int(value))


def _format_metric_duration(metric: Metric) -> str:
    value = "—" if metric.value is None else _format_duration(metric.value)
    return f"{value} {_PRECISION_SYMBOLS[metric.precision]}"


def _format_percentage_metric(metric: Metric) -> str:
    value = "—" if metric.value is None else f"{float(metric.value) * 100:.1f}%"
    return f"{value} {_PRECISION_SYMBOLS[metric.precision]}"


def _format_metric_tokens(metric: Metric) -> str:
    return f"{_format_tokens(metric.value)} {_PRECISION_SYMBOLS[metric.precision]}"


def _format_metric_count(metric: Metric) -> str:
    value = "—" if metric.value is None else f"{int(metric.value):,}"
    return f"{value} {_PRECISION_SYMBOLS[metric.precision]}"


def _format_metric_bytes(metric: Metric) -> str:
    if metric.value is None:
        value = "—"
    else:
        size = float(metric.value)
        value = f"{size / 1024:.1f} KiB" if size >= 1024 else f"{int(size)} B"
    return f"{value} {_PRECISION_SYMBOLS[metric.precision]}"


def _format_named_counts(rows: tuple[NamedCountRow, ...]) -> str:
    return ", ".join(f"{row.name}:{row.count}" for row in rows) or "—"


def _format_count(metric: Metric, *, signed: bool = False) -> str:
    if metric.value is None:
        return "—"
    value = int(metric.value)
    return f"{value:+,}" if signed else f"{value:,}"
