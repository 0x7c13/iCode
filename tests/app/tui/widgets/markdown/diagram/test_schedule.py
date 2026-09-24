# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Schedule-family parsing, geometry, fidelity, and fail-closed regressions."""

from __future__ import annotations

from datetime import date

import pytest

from chrys.app.tui.widgets.markdown.diagram import compile_mermaid
from chrys.app.tui.widgets.markdown.diagram.model import DiagnosticCode, DiagramKind
from chrys.app.tui.widgets.markdown.diagram.parser import parse_mermaid
from chrys.app.tui.widgets.markdown.diagram.renderers import schedule
from chrys.app.tui.widgets.markdown.diagram.renderers.common import ChartCanvasLimit
from chrys.app.tui.widgets.markdown.diagram.specs.schedule import GanttChart, GitChart, PacketChart

_GANTT = """gantt
title 软件开发计划
dateFormat YYYY-MM-DD
section 设计
需求分析 :a1, 2026-01-01, 7d
架构设计 :a2, after a1, 5d
section 开发
后端开发 :b1, after a2, 10d
前端开发 :b2, after a2, 10d
section 测试
测试与上线 :c1, after b1, 6d
"""

_GIT = """gitGraph
commit id: "init"
branch feature
commit id: "add feature"
commit id: "add tests"
checkout main
merge feature
commit id: "release v1.0"
"""

_PACKET = """packet-beta
title IPv4 报文头
0-3: "版本"
4-7: "首部长度"
8-15: "服务类型"
16-31: "总长度"
32-63: "标识"
64-127: "源地址"
128-191: "目的地址"
"""


@pytest.mark.parametrize(
    ("source", "kind", "labels"),
    [
        (_GANTT, DiagramKind.GANTT, ("软件开发计划", "需求分析", "测试与上线")),
        (_GIT, DiagramKind.GIT, ("main", "feature", "add feature", "release v1.0")),
        (_PACKET, DiagramKind.PACKET, ("IPv4 报文头", "首部长度", "源地址", "目的地址")),
    ],
)
def test_schedule_user_examples_render(source: str, kind: DiagramKind, labels: tuple[str, ...]) -> None:
    result = compile_mermaid(source)
    assert not result.diagnostics
    assert result.kind is kind
    assert result.width > 0 and result.height > 0
    for label in labels:
        assert label in "\n".join(result.rows)


def test_gantt_dependencies_preserve_parallel_tasks_and_exact_dates() -> None:
    ir = parse_mermaid(_GANTT)
    assert isinstance(ir.chart, GanttChart)
    first, second, backend, frontend, final = ir.chart.tasks
    assert first.start == date(2026, 1, 1)
    assert first.end == second.start == date(2026, 1, 8)
    assert backend.start == frontend.start == second.end == date(2026, 1, 13)
    assert backend.end == frontend.end == final.start == date(2026, 1, 23)
    assert final.end == date(2026, 1, 29)


@pytest.mark.parametrize(
    "label", ["Customer's rollout", "'90s migration", "Customers' rollout", "'Release'", 'The "release"', '"Release"']
)
def test_gantt_task_quotes_are_literal_text(label: str) -> None:
    source = f"gantt\n{label}: 2026-01-01, 7d"
    ir = parse_mermaid(source)
    assert not ir.diagnostics
    assert isinstance(ir.chart, GanttChart)
    task = ir.chart.tasks[0]
    assert task.label == label
    assert task.start == date(2026, 1, 1)
    assert task.end == date(2026, 1, 8)
    diagram = compile_mermaid(source)
    assert not diagram.diagnostics
    assert label in "\n".join(diagram.rows)


def test_gantt_forward_multiple_dependencies_and_implicit_previous_task() -> None:
    ir = parse_mermaid("gantt\nCombined :c, after a b, 1w\nA :a, 2026-01-01, 2d\nB :b, 2026-01-01, 4d\nNext :1d")
    assert not ir.diagnostics
    assert isinstance(ir.chart, GanttChart)
    combined, _, longest, next_task = ir.chart.tasks
    assert combined.start == longest.end == date(2026, 1, 5)
    assert combined.end == date(2026, 1, 12)
    assert next_task.start == longest.end


def test_gantt_until_statuses_and_zero_duration_milestone_are_retained() -> None:
    source = (
        "gantt\nPrepare :done, crit, a, 2026-01-01, until b\n"
        "Release :milestone, b, 2026-01-06, 0d\nFollow up :active, after b, 2d"
    )
    ir = parse_mermaid(source)
    assert not ir.diagnostics
    assert isinstance(ir.chart, GanttChart)
    assert ir.chart.tasks[0].end == date(2026, 1, 6)
    assert ir.chart.tasks[0].statuses == ("done", "crit")
    text = "\n".join(compile_mermaid(source).rows)
    assert "◆" in text and "[done, crit]" in text and "[active]" in text


@pytest.mark.parametrize(
    "body",
    [
        "A :after missing, 2d",
        "A :a, after b, 1d\nB :b, after a, 1d",
        "A :2026-02-30, 2d",
        "A :9999-12-31, 1w",
        "A :2026-01-01, 0d",
        "A :2026-01-01, 6h",
        "A :2026-01-01, 1.5d",
        "A :1d",
        "excludes weekends\nA :2026-01-01, 7d",
        "dateFormat DD-MM-YYYY\nA :01-01-2026, 7d",
        "A :a, 2026-01-01, 1d\nB :a, 2026-01-02, 1d",
    ],
)
def test_gantt_unsupported_or_invalid_schedule_fails_closed(body: str) -> None:
    assert parse_mermaid("gantt\n" + body).has_fatal_error


def test_git_branch_switch_and_merge_retain_actual_parent_indices() -> None:
    ir = parse_mermaid(_GIT)
    assert not ir.diagnostics
    assert isinstance(ir.chart, GitChart)
    assert ir.chart.branches == ("main", "feature")
    assert [commit.branch for commit in ir.chart.commits] == ["main", "feature", "feature", "main", "main"]
    assert [commit.parents for commit in ir.chart.commits] == [(), (0,), (1,), (0, 2), (3,)]


@pytest.mark.parametrize("orientation", ["LR", "TB", "BT"])
def test_git_orientations_keep_commit_ids_and_tags(orientation: str) -> None:
    source = (
        f'gitGraph {orientation}\ncommit id: "开始" tag: "v1.0"\n'
        'branch "特性"\ncommit id: "撤回" type: REVERSE\nswitch main\n'
        'merge "特性" id: "完成" type: HIGHLIGHT tag: "v2.0"'
    )
    result = compile_mermaid(source)
    assert not result.diagnostics
    text = "\n".join(result.rows)
    assert all(label in text for label in ("开始", "撤回", "完成", "v1.0", "v2.0", "REVERSE", "HIGHLIGHT"))
    assert "← 1, 2" in text


@pytest.mark.parametrize(
    "body",
    [
        "commit\nmerge main",
        "commit\ncheckout missing",
        "commit\nbranch main",
        'commit id: "same"\ncommit id: "same"',
        "commit type: INVALID",
        'commit\ncherry-pick id: "1"',
        "commit\nbranch feature\ncheckout main\nmerge feature",
    ],
)
def test_git_invalid_operations_fail_closed(body: str) -> None:
    assert parse_mermaid("gitGraph\n" + body).has_fatal_error


def test_packet_inclusive_offsets_and_relative_lengths_cross_rows() -> None:
    source = 'packet\n0: "Flag"\n+7: "Kind"\n8-39: "Crossing"\n+24: "Tail"'
    ir = parse_mermaid(source)
    assert not ir.diagnostics
    assert isinstance(ir.chart, PacketChart)
    assert [(field.start, field.end) for field in ir.chart.fields] == [(0, 0), (1, 7), (8, 39), (40, 63)]
    result = compile_mermaid(source)
    text = "\n".join(result.rows)
    assert "0-31" in text and "32-63" in text
    assert "[8-39] Crossing" in text


def test_packet_long_narrow_field_labels_survive_in_legend() -> None:
    label = "这是一个非常长的单比特字段名字"
    result = compile_mermaid(f'packet-beta\n0: "{label}"\n5-8: "Later"')
    assert not result.diagnostics
    assert label in "\n".join(result.rows)
    assert "[0-0]" in "\n".join(result.rows)
    assert "[5-8]" in "\n".join(result.rows)


@pytest.mark.parametrize("body", ['3-1: "Reverse"', '+0: "Empty"', '0-4: "A"\n4-6: "Overlap"'])
def test_packet_bad_ranges_fail_closed(body: str) -> None:
    assert parse_mermaid("packet\n" + body).has_fatal_error


def test_packet_extreme_offset_is_rejected_before_drawing() -> None:
    ir = parse_mermaid('packet\n999999: "Huge"')
    assert ir.has_fatal_error
    assert any(diagnostic.code is DiagnosticCode.CANVAS_LIMIT for diagnostic in ir.diagnostics)


@pytest.mark.parametrize("header", ["gantt", "gitGraph", "packet-beta"])
def test_empty_schedule_diagrams_fail_closed(header: str) -> None:
    assert parse_mermaid(header).has_fatal_error


@pytest.mark.parametrize(
    ("header", "statement"),
    [
        ("gantt", "Task {index}: task{index}, 2026-01-01, 1d"),
        ("gitGraph", 'commit id: "commit{index}"'),
        ("packet", '{index}: "Field {index}"'),
    ],
)
def test_schedule_item_limits_are_enforced(header: str, statement: str) -> None:
    source = header + "\n" + "\n".join(statement.format(index=index) for index in range(201))
    result = parse_mermaid(source)
    assert result.has_fatal_error
    assert any(diagnostic.code is DiagnosticCode.NODE_LIMIT for diagnostic in result.diagnostics)


@pytest.mark.parametrize("source", [_GANTT, _GIT, _PACKET])
def test_schedule_preflights_before_creating_canvas(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_canvas() -> None:
        pytest.fail("drawing started before the canvas budget check")

    monkeypatch.setattr(schedule, "TerminalCanvas", unexpected_canvas)
    with pytest.raises(ChartCanvasLimit):
        schedule.compile_schedule(source, parse_mermaid(source), lambda width, height: True)


def test_git_long_labels_are_complete_and_do_not_expand_branch_lanes() -> None:
    branch = "feature-" + "very-long-name-" * 15
    commit_id = "A commit ID with a long description " * 10
    source = f'gitGraph\ncommit\nbranch "{branch}"\ncommit id: "{commit_id}"'
    result = compile_mermaid(source)
    assert not result.diagnostics
    # Wrapping may break words at cell boundaries, but no content is cropped.
    compact_rows = "".join("".join(result.rows).split())
    assert branch in compact_rows
    assert "".join(commit_id.split()) in compact_rows
    assert result.width <= 110


@pytest.mark.parametrize("source", [_GANTT, _GIT, _PACKET])
def test_schedule_comments_before_header_do_not_change_rendering(source: str) -> None:
    commented = "%% Introduction\n\n%% Another comment\n" + source.replace("\n", "\n%% Comment\n")
    result = compile_mermaid(commented)
    assert not result.diagnostics
    assert result.rows == compile_mermaid(source).rows


@pytest.mark.parametrize("header", ["gitGraph:", "gitGraph LR:", "gitGraph TB:", "gitGraph BT:"])
def test_git_colon_header_is_recognized_through_facade(header: str) -> None:
    result = compile_mermaid(f'%% Comment\n{header}\ncommit id: "Initial"')
    assert not result.diagnostics
    assert "Initial" in "\n".join(result.rows)


@pytest.mark.parametrize(
    ("header", "body"),
    [
        ("gantt garbage", "A: 2026-01-01, 2d"),
        ("gantt:", "A: 2026-01-01, 2d"),
        ("packet garbage", '0-7: "A"'),
        ("packet-beta garbage", '0-7: "A"'),
        ("gitGraph LR garbage", "commit"),
    ],
)
def test_schedule_rejects_malformed_header_suffixes(header: str, body: str) -> None:
    ir = parse_mermaid(f"%% Leading comment\n{header}\n{body}")
    assert ir.has_fatal_error
    assert any(diagnostic.line == 2 for diagnostic in ir.diagnostics)
