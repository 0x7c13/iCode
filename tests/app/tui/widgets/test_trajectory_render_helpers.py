# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pure chartkit, presentation and page-builder units, rendered without mounting an App."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

from chrys.app.tui.widgets.trajectory import insights as trajectory_insights
from chrys.app.tui.widgets.trajectory import overview as trajectory_overview
from chrys.app.tui.widgets.trajectory.chartkit import (
    bordered_section,
    coverage_bar,
    fit_cells,
    fit_text_cells,
    section_interior_width,
    time_ruler,
    unresolved_bar,
    waterfall_lanes,
)
from chrys.app.tui.widgets.trajectory.insights import _input_share_metric, has_diagnostic_content
from chrys.app.tui.widgets.trajectory.overview import _fit_path_middle
from chrys.app.tui.widgets.trajectory.presentation import (
    RenderContext,
    ResponsiveTier,
    align_edges,
    cache_hit_metric,
    identity_with_hook_id,
)
from chrys.service.analytics import (
    AnalysisAvailability,
    ChangeVerification,
    ChangeVerificationRow,
    ChangeVerificationState,
    Metric,
    Precision,
    SubmissionLatencyBucket,
    TimelineDiagnosticCode,
    TokenUsage,
    TrajectoryAnalysis,
    TrajectoryAnalyzer,
    TrajectoryDiagnostics,
    UsageBucket,
)
from tests.app.tui.widgets._trajectory_fixtures import _NS, _write_p1_operations, _write_p2_operations, plain_look


def test_coverage_bar_keeps_missing_distinct_from_unresolved() -> None:
    bar = coverage_bar(25, 25, 25, 25, width=8)

    assert bar.plain == "██▒▒░░··"


@pytest.mark.parametrize(
    ("identity", "hook_id", "expected"),
    [
        ("before_tool_call", "hook_id_1", "before_tool (hook_id_1)"),
        ("after_tool_call", "hook_abcdefghijklmnop", "after_tool (hook_abcdefgh...)"),
        ("after_turn", None, "after_turn"),
    ],
)
def test_hook_identity_shortens_tool_event_names_and_caps_only_the_hook_id(
    identity: str,
    hook_id: str | None,
    expected: str,
) -> None:
    assert identity_with_hook_id(identity, hook_id) == expected


def test_operation_reason_messages_cover_every_timeline_diagnostic_code() -> None:
    assert set(trajectory_insights._OPERATION_REASON_MESSAGES) == set(TimelineDiagnosticCode)


def test_grouped_grid_lines_keeps_each_semantic_group_in_one_column() -> None:
    groups = [[(Text(f"g{group}r{row}"), Text(str(row))) for row in range(3)] for group in range(4)]

    lines = trajectory_overview._grouped_grid_lines(groups, width=120, columns=4)

    assert len(lines) == 3
    for row, line in enumerate(lines):
        assert all(f"g{group}r{row}" in line.plain for group in range(4))


@pytest.mark.parametrize(
    "diagnostics",
    [
        TrajectoryDiagnostics(span_duration_mismatch_count=1),
        TrajectoryDiagnostics(containment_violation_count=1),
        TrajectoryDiagnostics(malformed_hook_execution_mode_count=1),
        TrajectoryDiagnostics(side_call_empty_shell_revisions=("a" * 32,)),
        TrajectoryDiagnostics(unidentified_membership_revision_count=1),
    ],
)
def test_diagnostic_content_gate_does_not_depend_on_overview_precision(
    diagnostics: TrajectoryDiagnostics,
) -> None:
    assert has_diagnostic_content(diagnostics) is True
    assert has_diagnostic_content(TrajectoryDiagnostics()) is False


@pytest.mark.parametrize("width", [1, 2, 4, 11])
def test_unknown_timeline_primitives_never_exceed_requested_width(width: int) -> None:
    assert cell_len(unresolved_bar(width).plain) == width
    assert cell_len(time_ruler(_NS, width=width).plain) == width


def test_time_ruler_spreads_tick_labels_across_the_axis() -> None:
    ruler = time_ruler(12 * _NS, width=60).plain

    assert cell_len(ruler) == 60
    assert ruler.startswith("0s")
    assert ruler.endswith("12s")
    assert "4.8s" in ruler and "7.2s" in ruler


def test_time_ruler_tick_units_follow_the_axis_magnitude() -> None:
    sub_second = time_ruler(800_000_000, width=60).plain
    minutes = time_ruler(638 * _NS, width=90).plain
    hours = time_ruler(3 * 3600 * _NS + 1800 * _NS, width=90).plain

    assert sub_second.startswith("0ms") and sub_second.endswith("800ms")
    assert minutes.startswith("0s") and minutes.endswith("10m38s")
    assert "1m46s" in minutes and "3m33s" in minutes
    assert hours.startswith("0s") and hours.endswith("3h30m")
    assert "35m00s" in hours and "1h10m" in hours


def test_derived_token_ratios_preserve_missing_and_inconsistent_evidence() -> None:
    estimated_usage = TokenUsage(
        {
            UsageBucket.INPUT: Metric(40, Precision.ESTIMATED),
            UsageBucket.CACHE_READ: Metric(20, Precision.MISSING),
        }
    )
    exact_session = TokenUsage({UsageBucket.INPUT: Metric(100, Precision.EXACT)})
    zero_session = TokenUsage({UsageBucket.INPUT: Metric(0, Precision.EXACT)})
    inconsistent_usage = TokenUsage(
        {
            UsageBucket.INPUT: Metric(40, Precision.EXACT),
            UsageBucket.CACHE_READ: Metric(50, Precision.EXACT),
        }
    )

    assert _input_share_metric(estimated_usage, exact_session).precision is Precision.ESTIMATED
    assert _input_share_metric(estimated_usage, zero_session).precision is Precision.MISSING
    assert cache_hit_metric(estimated_usage).precision is Precision.MISSING
    assert cache_hit_metric(inconsistent_usage).precision is Precision.UNRESOLVED


def test_bordered_section_is_cell_width_safe_for_cjk_titles() -> None:
    lines = bordered_section("发现", [Text("一行内容")], width=18, console=Console())

    assert lines[0].plain.startswith("┌─ 发现 ")
    assert all(cell_len(line.plain) == 18 for line in lines)


def test_bordered_section_pads_one_cell_inside_each_vertical_border() -> None:
    lines = bordered_section("T", [Text("x" * 20)], width=12, console=Console())

    assert section_interior_width(12) == 8
    assert lines[1].plain == "│ xxxxxxxx │"
    assert lines[2].plain == "│ xxxxxxxx │"
    assert lines[-1].plain == "└──────────┘"
    assert all(cell_len(line.plain) == 12 for line in lines)


def test_waterfall_lanes_paint_each_cell_in_exactly_one_lane() -> None:
    # One turn, 10 cells: the model covers everything except a tool window that
    # owns most of cells 4-5; model slivers around it must not repaint them.
    turn = (
        1000,
        {
            "model": [(0, 420), (421, 430), (580, 1000)],
            "tool": [(430, 580)],
        },
    )
    lanes = waterfall_lanes([turn], width=10, lanes=[("tool", "▬", ""), ("model", "█", "")])

    assert lanes["model"].plain == "████  ████"
    assert lanes["tool"].plain == "    ▬▬    "
    assert all(
        (model_cell == " ") or (tool_cell == " ")
        for model_cell, tool_cell in zip(lanes["model"].plain, lanes["tool"].plain, strict=True)
    )


def test_waterfall_lanes_keep_turn_separators_and_empty_canvas() -> None:
    lanes = waterfall_lanes(
        [(100, {"model": [(0, 100)]}), (100, {"model": [(0, 100)]})],
        width=9,
        lanes=[("model", "█", "")],
    )

    assert lanes["model"].plain == "████┊████"
    assert waterfall_lanes([], width=4, lanes=[("model", "█", "")])["model"].plain == "    "


def test_two_edge_alignment_is_cell_width_safe_for_cjk_labels() -> None:
    line = align_edges(Text("标签"), Text("值 [精确]"), 20)

    assert cell_len(line.plain) == 20
    assert line.plain.startswith("标签")
    assert line.plain.endswith("值 [精确]")


@pytest.mark.parametrize(
    "path",
    [
        "/Users/0x7c13/Repos/deeply/nested/report.final.json",
        "C:\\Users\\0x7c13\\Repos\\deeply\\nested\\report.final.json",
    ],
)
def test_fit_path_middle_preserves_the_full_filename_and_extension(path: str) -> None:
    fitted = _fit_path_middle(path, 30)
    extension_only_fallback = _fit_path_middle(path, 10)

    assert cell_len(fitted) <= 30
    assert fitted.endswith(("/report.final.json", "\\report.final.json"))
    assert fitted.startswith(("/", "C:"))
    assert "…" in fitted
    assert cell_len(extension_only_fallback) <= 10
    assert extension_only_fallback.startswith("…")
    assert extension_only_fallback.endswith(".json")


@pytest.mark.parametrize(
    ("cache_hit", "semantic_name", "fallback"),
    [
        (29, "error", "red"),
        (30, "warning", "yellow"),
        (60, "warning", "yellow"),
        (61, "success", "green"),
    ],
)
def test_cache_hit_meter_style_uses_threshold_theme_colors(
    cache_hit: int,
    semantic_name: str,
    fallback: str,
) -> None:
    look = plain_look({"error": "#aa0000", "warning": "#bbbb00", "success": "#00cc00"})

    assert trajectory_insights._cache_hit_style(look, cache_hit) == look.semantic_style(semantic_name, fallback)


def _change_verification_analysis(
    rows: tuple[ChangeVerificationRow, ...],
    *,
    modified: Metric | None = None,
) -> TrajectoryAnalysis:
    """An available analysis whose change section holds *rows*, with exact counts unless overridden."""
    exact_zero = Metric(0, Precision.EXACT)
    exact_rows = Metric(len(rows), Precision.EXACT)
    return TrajectoryAnalysis(
        availability=AnalysisAvailability.AVAILABLE,
        path=Path("events.jsonl"),
        generation=1,
        change_verification=ChangeVerification(
            detail_available=True,
            detection_truncated=False,
            files_touched=exact_rows,
            created=exact_zero,
            modified=exact_rows if modified is None else modified,
            deleted=exact_zero,
            net_zero=exact_zero,
            rows=rows,
        ),
    )


def test_change_verification_path_display_copy_is_surrogate_safe() -> None:
    raw_path = "changed-\udcff.py"
    analysis = _change_verification_analysis(
        (
            ChangeVerificationRow(
                path=raw_path,
                state=ChangeVerificationState.VERIFIED,
                last_change_turn=1,
                precision=Precision.EXACT,
            ),
        )
    )

    lines = trajectory_overview._change_verification_lines(plain_look(), analysis, width=80)

    assert any("changed-\\udcff.py" in line.plain for line in lines)


def test_change_verification_rows_and_counts_carry_precision_badges() -> None:
    """Each row shows its own precision and the counts line shows the worst
    of the five count precisions, so a degraded change section cannot pass
    for exact measurements."""
    analysis = _change_verification_analysis(
        (
            ChangeVerificationRow(
                path="proven.py",
                state=ChangeVerificationState.VERIFIED,
                last_change_turn=1,
                precision=Precision.EXACT,
            ),
            ChangeVerificationRow(
                path="unprovable.py",
                state=ChangeVerificationState.NET_ZERO,
                last_change_turn=1,
                precision=Precision.UNRESOLVED,
            ),
        ),
        modified=Metric(2, Precision.ESTIMATED, "counts include window-inferred or peer-contested mutations"),
    )

    lines = trajectory_overview._change_verification_lines(plain_look(), analysis, width=80)
    narrow = trajectory_overview._change_verification_lines(plain_look(), analysis, width=22)

    assert lines[0].plain.endswith("~")
    assert next(line.plain for line in lines if "proven.py" in line.plain).endswith("✓")
    assert next(line.plain for line in lines if "unprovable.py" in line.plain).endswith("✗")
    # A compressed column truncates the counts, never the badge.
    assert "…" in narrow[0].plain
    assert narrow[0].plain.endswith("~")


def test_change_verification_middle_crops_paths_before_the_filename() -> None:
    analysis = _change_verification_analysis(
        (
            ChangeVerificationRow(
                path="/Users/0x7c13/Repos/deeply/nested/report.final.json",
                state=ChangeVerificationState.UNVERIFIED,
                last_change_turn=1,
                precision=Precision.EXACT,
            ),
        )
    )

    lines = trajectory_overview._change_verification_lines(plain_look(), analysis, width=40)

    row = next(line.plain for line in lines if "report.final.json" in line.plain)
    assert row.startswith("/Users/")
    assert "…/report.final.json" in row
    assert row.endswith("✓")
    assert cell_len(row) == 40


@pytest.mark.parametrize(
    ("precision", "badge"),
    [(Precision.EXACT, "✓"), (Precision.UNRESOLVED, "✗")],
)
def test_insights_section_titles_render_their_panel_precision(
    tmp_path: Path,
    precision: Precision,
    badge: str,
) -> None:
    path = tmp_path / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p2_operations(path)
    analysis = TrajectoryAnalyzer().load(path)
    assert analysis.insights is not None
    reason = "sentinel unresolved panel" if precision is Precision.UNRESOLVED else None
    insights = replace(
        analysis.insights,
        tools=replace(analysis.insights.tools, precision=precision, reason=reason),
        mcp=replace(analysis.insights.mcp, precision=precision, reason=reason),
        skills=replace(analysis.insights.skills, precision=precision, reason=reason),
        context_carrying_precision=precision,
        context_carrying_reason=reason,
    )

    context = RenderContext(width=220, tier=ResponsiveTier.WIDE)
    page = "\n".join(
        line.plain
        for line in trajectory_insights.insights_lines(plain_look(), context, replace(analysis, insights=insights))
    )

    for title in ("Skills", "MCP servers", "Tool activity", "Context re-send cost · top 5"):
        assert f"{title} {badge}" in page


@pytest.mark.parametrize(
    ("precision", "badge"),
    [(Precision.EXACT, "✓"), (Precision.UNRESOLVED, "✗")],
)
def test_submission_aggregate_renders_derived_precision_for_the_same_duration(
    tmp_path: Path,
    precision: Precision,
    badge: str,
) -> None:
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p1_operations(path)
    analysis = TrajectoryAnalyzer().load(path)
    assert analysis.submission_latency is not None
    stats = next(
        bucket for bucket in analysis.submission_latency.buckets if bucket.bucket is SubmissionLatencyBucket.BECAME_TURN
    )
    reason = "sentinel unresolved aggregate" if precision is Precision.UNRESOLVED else None
    four_seconds = Metric(4 * _NS, precision, reason)
    submission = replace(
        analysis.submission_latency,
        buckets=(replace(stats, p50_ns=four_seconds, p90_ns=four_seconds, max_ns=four_seconds),),
    )

    lines = trajectory_overview._submission_latency_lines(
        plain_look(), replace(analysis, submission_latency=submission), width=80
    )

    aggregate = next(line.plain for line in lines if "started a new turn" in line.plain)
    assert aggregate.count("4.00 s") == 3
    assert aggregate.endswith(badge)


@pytest.mark.parametrize("value", ["", "abc", "中文", "e\u0301", "a中b"])
@pytest.mark.parametrize("width", [-1, 0, 1, 2, 5, 12])
def test_fit_cells_pads_and_crops_to_terminal_width(value: str, width: int) -> None:
    assert cell_len(fit_cells(value, width)) == max(0, width)


@pytest.mark.parametrize("value", ["", "abc", "中文", "e\u0301", "a中b"])
@pytest.mark.parametrize("width", [-1, 0, 1, 2, 5, 12])
def test_fit_text_cells_pads_and_crops_styled_text_to_terminal_width(value: str, width: int) -> None:
    assert cell_len(fit_text_cells(Text(value, style="bold"), width).plain) == max(0, width)


def test_fit_text_cells_marks_a_crop_with_an_ellipsis_and_keeps_the_style() -> None:
    fitted = fit_text_cells(Text("abcdef", style="bold"), 4)

    assert fitted.plain == "abc…"
    assert fitted.get_style_at_offset(Console(), 0).bold is True
