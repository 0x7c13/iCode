# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Semantic and terminal-geometry coverage for planning diagram subsets."""

from __future__ import annotations

import pytest
from rich.cells import cell_len

from chrys.app.tui.widgets.markdown.diagram import compile_mermaid
from chrys.app.tui.widgets.markdown.diagram.model import DiagnosticCode, DiagramKind, Direction, NodeShape
from chrys.app.tui.widgets.markdown.diagram.parser import MAX_EDGES, MAX_NODES, parse_mermaid
from chrys.app.tui.widgets.markdown.diagram.renderers import planning as planning_renderer
from chrys.app.tui.widgets.markdown.diagram.renderers.common import ChartCanvasLimit
from chrys.app.tui.widgets.markdown.diagram.specs.planning import JourneyChart, KanbanChart, TimelineChart

JOURNEY = """journey
title 电商购物体验
section 浏览
查找商品: 5: 用户
加入购物车: 4: 用户
section 结算
填写地址: 3: 用户
确认支付: 4: 用户, 支付服务
section 收货
收到包裹: 5: 用户
"""

TIMELINE = """timeline
title Web 技术演进
1990s : HTML
: HTTP
2000s : AJAX : CSS 流行
2010s : SPA
: 移动优先
2020s : PWA
: Serverless
"""

KANBAN = """kanban
  todo[待办]
    docs[编写文档]@{ assigned: "me" }
    prototype[设计原型]@{ assigned: "you" }
  doing[进行中]
    login[实现登录]@{ assigned: "you", ticket: ABC-42, priority: 'High' }
  done[已完成]
    setup[环境搭建]@{ assigned: "me" }
"""

MINDMAP = """mindmap
  root((软件系统))
    展示层
      网页端
      移动端 APP
    应用层
      API 网关
      微服务
    数据层
      数据库
      缓存
"""


def test_journey_preserves_sections_scores_actors_and_order() -> None:
    parsed = parse_mermaid(JOURNEY)
    assert not parsed.diagnostics
    assert parsed.kind is DiagramKind.JOURNEY
    assert isinstance(parsed.chart, JourneyChart)
    assert parsed.chart.title == "电商购物体验"
    assert [task.section for task in parsed.chart.tasks] == ["浏览", "浏览", "结算", "结算", "收货"]
    assert [task.score for task in parsed.chart.tasks] == [5, 4, 3, 4, 5]
    assert parsed.chart.tasks[3].actors == ("用户", "支付服务")
    compiled = compile_mermaid(JOURNEY)
    text = "\n".join(compiled.rows)
    assert "●●●○○ 3/5" in text
    assert "@ 支付服务" in text
    assert text.index("查找商品") < text.index("加入购物车")


def test_journey_accepts_apostrophes_and_quoted_actor_commas() -> None:
    parsed = parse_mermaid('journey\nUser\'s cart: 4: "Customer, admin", Staff')
    assert not parsed.diagnostics
    assert isinstance(parsed.chart, JourneyChart)
    assert parsed.chart.tasks[0].label == "User's cart"
    assert parsed.chart.tasks[0].actors == ("Customer, admin", "Staff")


@pytest.mark.parametrize(
    ("actor", "expected"),
    [
        ("O'Brien", "O'Brien"),
        ("'90s team", "'90s team"),
        ("Customers' team", "Customers' team"),
        ("'Lead'", "'Lead'"),
        ('"O\'Brien, lead"', "O'Brien, lead"),
        (r'"\"Lead\""', '"Lead"'),
    ],
)
def test_journey_actors_keep_literal_quotes_without_breaking_actor_lists(actor: str, expected: str) -> None:
    source = f"journey\n'Deploy': 5: {actor}, Alice"
    parsed = parse_mermaid(source)
    assert not parsed.diagnostics
    assert isinstance(parsed.chart, JourneyChart)
    task = parsed.chart.tasks[0]
    assert task.label == "'Deploy'"
    assert task.actors == (expected, "Alice")
    diagram = compile_mermaid(source)
    assert not diagram.diagnostics
    assert expected in "\n".join(diagram.rows)


def test_timeline_preserves_continuations_and_source_order() -> None:
    parsed = parse_mermaid(TIMELINE)
    assert not parsed.diagnostics
    assert isinstance(parsed.chart, TimelineChart)
    assert [period.label for period in parsed.chart.periods] == ["1990s", "2000s", "2010s", "2020s"]
    assert parsed.chart.periods[0].events == ("HTML", "HTTP")
    assert parsed.chart.periods[1].events == ("AJAX", "CSS 流行")
    compiled = compile_mermaid(TIMELINE)
    assert "Serverless" in "\n".join(compiled.rows)
    assert sum(row.count("▶") for row in compiled.rows) == 3


def test_timeline_vertical_sections_and_apostrophes() -> None:
    source = "timeline TD\ntitle History\nsection Early\nNow : User's work\nsection Later\nThen : Done"
    parsed = parse_mermaid(source)
    assert parsed.direction is Direction.TOP_DOWN
    assert isinstance(parsed.chart, TimelineChart)
    assert parsed.chart.vertical
    assert [period.section for period in parsed.chart.periods] == ["Early", "Later"]
    compiled = compile_mermaid(source)
    assert not compiled.diagnostics
    assert "▼" in "\n".join(compiled.rows)
    assert next(index for index, row in enumerate(compiled.rows) if "Now" in row) < next(
        index for index, row in enumerate(compiled.rows) if "Then" in row
    )


def test_kanban_retains_columns_tasks_and_metadata() -> None:
    parsed = parse_mermaid(KANBAN)
    assert not parsed.diagnostics
    assert isinstance(parsed.chart, KanbanChart)
    assert [column.label for column in parsed.chart.columns] == ["待办", "进行中", "已完成"]
    task = parsed.chart.columns[1].tasks[0]
    assert task.task_id == "login"
    assert task.metadata == (("assigned", "you"), ("ticket", "ABC-42"), ("priority", "High"))
    compiled = compile_mermaid(KANBAN)
    text = "\n".join(compiled.rows)
    assert "assigned: you" in text
    assert "ticket: ABC-42" in text
    assert "priority: High" in text
    assert "▶" not in text


def test_inline_tabs_do_not_change_kanban_columns_or_mindmap_hierarchy() -> None:
    parsed = parse_mermaid("kanban\n  todo[Todo]\n  done[Do\tit]")
    assert not parsed.diagnostics
    assert isinstance(parsed.chart, KanbanChart)
    assert [column.label for column in parsed.chart.columns] == ["Todo", "Do it"]
    assert all(not column.tasks for column in parsed.chart.columns)

    # Two equally indented roots remain invalid regardless of inline tabs.
    for label in ("sib ling", "sib\tling"):
        assert parse_mermaid(f"mindmap\n  root\n  {label}").has_fatal_error
    siblings = parse_mermaid("mindmap\nroot\n  first\n  sib\tling")
    assert not siblings.diagnostics
    assert len(siblings.edges) == 2
    assert siblings.edges[0].source == siblings.edges[1].source


def test_kanban_plain_anonymous_and_empty_columns() -> None:
    source = "kanban\nTodo\n  [Create Documentation]\n[In progress]\n  docs[Can't reproduce]\nempty[Done]"
    parsed = parse_mermaid(source)
    assert not parsed.diagnostics
    assert isinstance(parsed.chart, KanbanChart)
    assert parsed.chart.columns[0].tasks[0].label == "Create Documentation"
    assert parsed.chart.columns[1].tasks[0].label == "Can't reproduce"
    assert not parsed.chart.columns[2].tasks
    assert "Done" in "\n".join(compile_mermaid(source).rows)


def test_kanban_quoted_brackets_and_metadata_punctuation() -> None:
    source = 'kanban\ntodo[Tasks]\n  task["Check [header]"]@{ assigned: "User, Admin", ticket: "app:42" }'
    parsed = parse_mermaid(source)
    assert not parsed.diagnostics
    assert isinstance(parsed.chart, KanbanChart)
    task = parsed.chart.columns[0].tasks[0]
    assert task.label == "Check [header]"
    assert task.metadata == (("assigned", "User, Admin"), ("ticket", "app:42"))


def test_mindmap_user_example_preserves_hierarchy() -> None:
    parsed = parse_mermaid(MINDMAP)
    assert not parsed.diagnostics
    assert parsed.kind is DiagramKind.MINDMAP
    assert parsed.direction is Direction.LEFT_RIGHT
    assert len(parsed.nodes) == 10
    assert len(parsed.edges) == 9
    labels = {node.node_id: node.label for node in parsed.nodes}
    relationships = [(labels[edge.source], labels[edge.target]) for edge in parsed.edges]
    assert relationships == [
        ("软件系统", "展示层"),
        ("展示层", "网页端"),
        ("展示层", "移动端 APP"),
        ("软件系统", "应用层"),
        ("应用层", "API 网关"),
        ("应用层", "微服务"),
        ("软件系统", "数据层"),
        ("数据层", "数据库"),
        ("数据层", "缓存"),
    ]
    assert all(not edge.directed for edge in parsed.edges)
    assert not compile_mermaid(MINDMAP).diagnostics


def test_mindmap_repeated_labels_remain_distinct_and_uneven_indents_use_nearest_parent() -> None:
    parsed = parse_mermaid("mindmap\nRoot\n    Same\n        Leaf\n      Same\n    Same")
    assert not parsed.diagnostics
    assert len(parsed.nodes) == 5
    assert len({node.node_id for node in parsed.nodes}) == 5
    assert [edge.source for edge in parsed.edges] == ["@mindmap:2", "@mindmap:3", "@mindmap:3", "@mindmap:2"]


@pytest.mark.parametrize(
    ("label", "shape"),
    [
        ("id[Square]", NodeShape.RECTANGLE),
        ("id(Rounded)", NodeShape.ROUNDED),
        ("id((Circle))", NodeShape.CIRCLE),
        ("id{{Hexagon}}", NodeShape.HEXAGON),
        ("id))Bang((", NodeShape.RECTANGLE),
        ("id)Cloud(", NodeShape.ROUNDED),
    ],
)
def test_mindmap_common_shapes(label: str, shape: NodeShape) -> None:
    parsed = parse_mermaid(f"mindmap\n{label}")
    assert not parsed.diagnostics
    assert parsed.nodes[0].shape is shape


def test_mindmap_icons_are_retained_as_text_and_classes_warn() -> None:
    parsed = parse_mermaid("mindmap\nRoot\n  A\n  ::icon(fa fa-book)\n  :::urgent large")
    assert not parsed.has_fatal_error
    assert parsed.nodes[1].notes == ("fa fa-book",)
    assert [diagnostic.code for diagnostic in parsed.diagnostics] == [DiagnosticCode.UNSUPPORTED_DIRECTIVE]


@pytest.mark.parametrize("source", [JOURNEY, TIMELINE, KANBAN, MINDMAP])
def test_planning_accepts_comments_and_common_codeblock_indentation(source: str) -> None:
    decorated = "%% comment\n" + "\n".join("    " + line for line in source.splitlines())
    parsed = parse_mermaid(decorated)
    assert not parsed.diagnostics
    assert not compile_mermaid(decorated).diagnostics


@pytest.mark.parametrize("source", [JOURNEY, TIMELINE, KANBAN, MINDMAP])
def test_planning_configuration_falls_back_without_interpreting_unsupported_settings(source: str) -> None:
    decorated = "---\nconfig:\n  layout: custom\n---\n" + source
    assert parse_mermaid(decorated).has_fatal_error


@pytest.mark.parametrize(
    "source",
    [
        "journey",
        "journey unsupported\nTask: 4: User",
        "journey\nTask: 6: User",
        "journey\nTask: 4",
        "journey\nTask: 4: User,",
        "timeline",
        "timeline BT\nNow: Event",
        "timeline\n: Orphan event",
        "timeline\nNow: Event\nsection Other\n: Orphan event",
        "timeline\nNow: Event:",
        "kanban",
        "kanban unsupported\nTodo\n  Task",
        "kanban\nTodo\n  Task\n    Nested task",
        "kanban\ntodo[Todo]\n  task[Task]@{ unexpected: value }",
        "kanban\ntodo[Todo]\n  task[Task]@{ priority: Highest }",
        "kanban\ntodo[Todo]\n  task[Task]@{ assigned: me, assigned: you }",
        "kanban\ntodo[Todo]\n  task[Task]\n  task[Again]",
        "kanban\ntodo[Todo]\n  task[Unclosed",
        "mindmap",
        "mindmap unsupported\nRoot",
        "mindmap\nRoot\nOther root",
        "mindmap\nroot((Unclosed)",
        "mindmap\n::icon(fa fa-book)",
    ],
)
def test_planning_invalid_or_unsupported_syntax_fails_closed(source: str) -> None:
    assert parse_mermaid(source).has_fatal_error
    assert compile_mermaid(source).diagnostics


@pytest.mark.parametrize("source", [JOURNEY, TIMELINE, KANBAN])
def test_planning_preflights_before_allocating_canvas(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_canvas() -> None:
        pytest.fail("canvas allocation must not precede the preflight")

    monkeypatch.setattr(planning_renderer, "TerminalCanvas", unexpected_canvas)
    with pytest.raises(ChartCanvasLimit):
        planning_renderer.compile_planning(source, parse_mermaid(source), lambda _width, _height: True)


@pytest.mark.parametrize(
    "source",
    [
        "journey\n" + "\n".join(f"Task {index}: 4: User" for index in range(MAX_NODES + 1)),
        "timeline\n" + "\n".join(f"Period {index}: Event" for index in range(MAX_NODES + 1)),
        "timeline\nNow: First\n" + "\n".join(": Event" for _ in range(MAX_EDGES)),
        "kanban\nTodo\n" + "\n".join(f"  task{index}[Task]" for index in range(MAX_NODES)),
        "mindmap\nRoot\n" + "\n".join(f"  Node {index}" for index in range(MAX_NODES)),
    ],
)
def test_planning_limits_are_fatal_and_bounded(source: str) -> None:
    parsed = parse_mermaid(source)
    assert parsed.has_fatal_error
    assert any(
        diagnostic.code in {DiagnosticCode.NODE_LIMIT, DiagnosticCode.EDGE_LIMIT} for diagnostic in parsed.diagnostics
    )


@pytest.mark.parametrize("kind", ["journey", "timeline", "kanban"])
def test_planning_wraps_full_wide_labels_and_sanitizes_before_measurement(kind: str) -> None:
    label = "高优先级" * 18 + " END\x1b"
    sources = {
        "journey": f"journey\n{label}: 4: 用户",
        "timeline": f"timeline\nNow: {label}",
        "kanban": f"kanban\nTodo\n  task[{label}]",
    }
    compiled = compile_mermaid(sources[kind])
    assert not compiled.diagnostics
    text = "\n".join(compiled.rows)
    assert text.count("高") == 18
    assert "END�" in text
    assert all(cell_len(row) <= compiled.width for row in compiled.rows)
