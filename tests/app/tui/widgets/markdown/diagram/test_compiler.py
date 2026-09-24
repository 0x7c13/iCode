# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for deterministic terminal Mermaid compilation."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

import pytest
from rich.cells import cell_len

from chrys.app.tui.widgets.markdown.diagram import compile_mermaid
from chrys.app.tui.widgets.markdown.diagram.canvas import TerminalCanvas, crop_cell_text
from chrys.app.tui.widgets.markdown.diagram.charts import ChartCanvasLimit, compile_chart
from chrys.app.tui.widgets.markdown.diagram.layout import MAX_CANVAS_AXIS
from chrys.app.tui.widgets.markdown.diagram.model import (
    DiagnosticCode,
    DiagramEdge,
    DiagramNode,
    Direction,
    EdgeStyle,
    NodeShape,
    PlacedNode,
    Point,
)
from chrys.app.tui.widgets.markdown.diagram.parser import parse_mermaid
from chrys.app.tui.widgets.markdown.diagram.router import route_edges
from chrys.foundation.i18n import Localizer


def test_simple_flowchart_has_stable_plain_text_golden() -> None:
    source = "flowchart TB\nA[Start] --> B(Work)"

    first = compile_mermaid(source)
    second = compile_mermaid(source)

    assert first == second
    assert first.rows == (
        "┌─────────┐",
        "│  Start  │",
        "└─────────┘",
        "     │",
        "     │",
        "     │",
        "     │",
        "     ▼",
        "╭────────╮",
        "│  Work  │",
        "╰────────╯",
    )


def test_malformed_and_unsupported_input_returns_diagnostic_canvas() -> None:
    source = "flowchart TB\nA -->"
    diagram = compile_mermaid(source)

    assert diagram.source == source
    assert diagram.diagnostics
    assert diagram.width > 0
    assert diagram.height == len(diagram.rows)
    assert "could not be rendered" in "\n".join(diagram.rows)


def test_crop_plain_row_is_exact_for_cjk_emoji_and_partial_graphemes() -> None:
    text = "A用户👩🏽\u200d💻B"

    for start in range(cell_len(text) + 3):
        cropped = crop_cell_text(text, start, 5)
        assert cell_len(cropped) == 5

    diagram = compile_mermaid("flowchart LR\nA[用户 👩🏽\u200d💻] --> B[完成]")
    for y in range(-1, diagram.height + 1):
        assert cell_len(diagram.crop_plain_row(y, 1, 17)) == 17


def test_common_flowchart_shapes_have_stable_terminal_approximations() -> None:
    diagram = compile_mermaid(
        """flowchart LR
        A((Circle)) --> B[(Database)] --> C[[Subroutine]] --> D([Stadium]) --> E{{Hexagon}}
        """
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    assert "╭" in rendered
    assert "││  Subroutine  ││" in rendered
    assert "⬡ Hexagon" in rendered


def test_class_annotations_and_state_choice_have_terminal_approximations() -> None:
    class_diagram = compile_mermaid('classDiagram\nclass Service["Public API"] <<interface>>')
    state_diagram = compile_mermaid("stateDiagram-v2\nstate decision <<choice>>\nA --> decision")

    assert "«interface» Public API" in "\n".join(class_diagram.rows)
    assert "◇ decision" in "\n".join(state_diagram.rows)
    assert not class_diagram.diagnostics
    assert not state_diagram.diagnostics


def test_notes_fork_join_typed_participants_and_slanted_shapes_render() -> None:
    state_diagram = compile_mermaid(
        """stateDiagram-v2
        state fork_state <<fork>>
        A --> fork_state
        fork_state --> B
        note right of B : worker
        """
    )
    sequence_diagram = compile_mermaid(
        'sequenceDiagram\nparticipant DB@{ "type": "database" } as Storage\nDB->>Client: rows'
    )
    flowchart = compile_mermaid(
        r"""flowchart LR
        A[/Input/] --> B(((Done)))
        """
    )

    rendered_state = "\n".join(state_diagram.rows)
    rendered_sequence = "\n".join(sequence_diagram.rows)
    rendered_flowchart = "\n".join(flowchart.rows)
    assert "━" * 9 in rendered_state
    assert "📝 worker" in rendered_state
    assert "«database» Storage" in rendered_sequence
    assert "╭" in rendered_sequence
    assert "▱ Input" in rendered_flowchart
    assert "Done" in rendered_flowchart
    assert not state_diagram.diagnostics
    assert not sequence_diagram.diagnostics
    assert not flowchart.diagnostics


def test_sequence_notes_class_generics_and_shape_metadata_render() -> None:
    sequence_diagram = compile_mermaid(
        """sequenceDiagram
        participant A
        participant B
        Note over A,B: shared context
        A->>B: request
        """
    )
    class_diagram = compile_mermaid("classDiagram\nclass Repository~Entity~\nRepository --> Service")
    flowchart = compile_mermaid(
        'flowchart LR\nA@{ shape: diam, label: "Choose" } --> B@{ shape: database, label: "Store" }'
    )

    rendered_sequence = "\n".join(sequence_diagram.rows)
    rendered_class = "\n".join(class_diagram.rows)
    rendered_flowchart = "\n".join(flowchart.rows)
    assert rendered_sequence.count("📝 shared context") == 2
    assert "Repository<Entity>" in rendered_class
    assert "◇ Choose" in rendered_flowchart
    assert "Store" in rendered_flowchart
    assert not class_diagram.diagnostics
    assert not flowchart.diagnostics
    assert {diagnostic.severity.value for diagnostic in sequence_diagram.diagnostics} == {"warning"}


def test_routes_contain_only_cardinal_segments() -> None:
    first = DiagramNode("A", "A")
    second = DiagramNode("B", "B")
    placed = {
        "A": PlacedNode(first, 10, 2, 7, 3, ("A",)),
        "B": PlacedNode(second, 10, 10, 7, 3, ("B",)),
    }
    routed = route_edges((DiagramEdge("A", "B"), DiagramEdge("B", "A")), placed, {"A": 0, "B": 1}, Direction.TOP_DOWN)

    for edge in routed:
        assert all(
            first_point.x == second_point.x or first_point.y == second_point.y
            for first_point, second_point in pairwise(edge.points)
        )
        assert edge.arrow_at == edge.points[-1]


def test_left_right_edge_label_stays_between_node_borders() -> None:
    diagram = compile_mermaid("flowchart LR\nA[Alpha] -->|dispatch| B[Beta]")
    label_row = next(row for row in diagram.rows if "dispatch" in row)
    alpha_left = label_row.index("┌")
    alpha_right = label_row.index("┐", alpha_left) + 1
    beta_left = label_row.index("┌", alpha_right)
    label_start = label_row.index("dispatch")

    assert not diagram.diagnostics
    assert alpha_right < label_start
    assert label_start + len("dispatch") < beta_left


def test_non_adjacent_backward_and_self_routes_finish_toward_target() -> None:
    nodes = {node_id: DiagramNode(node_id, node_id) for node_id in ("A", "B", "C")}
    ranks = {"A": 0, "B": 1, "C": 2}
    edges = (DiagramEdge("A", "C"), DiagramEdge("C", "A"), DiagramEdge("B", "B"))
    placements = {
        Direction.TOP_DOWN: {
            node_id: PlacedNode(nodes[node_id], 10, 2 + rank * 8, 7, 3, (node_id,)) for node_id, rank in ranks.items()
        },
        Direction.LEFT_RIGHT: {
            node_id: PlacedNode(nodes[node_id], 2 + rank * 13, 8, 7, 3, (node_id,)) for node_id, rank in ranks.items()
        },
    }

    for direction, placed in placements.items():
        routed = route_edges(edges, placed, ranks, direction)
        for edge in routed:
            previous, final = edge.points[-2:]
            if direction is Direction.TOP_DOWN:
                assert previous.x == final.x
                assert previous.y < final.y
                assert edge.arrow == "▼"
            else:
                assert previous.y == final.y
                assert previous.x < final.x
                assert edge.arrow == "▶"


def test_parallel_adjacent_edges_keep_every_label_visible() -> None:
    for direction in ("TB", "LR"):
        diagram = compile_mermaid(
            f"""flowchart {direction}
            A[Alpha] -->|first| B[Beta]
            A -->|second| B
            """
        )
        rendered = "\n".join(diagram.rows)

        assert not diagram.diagnostics
        assert rendered.count("first") == 1
        assert rendered.count("second") == 1


def test_crossing_edges_keep_every_label_visible() -> None:
    diagram = compile_mermaid(
        """flowchart TB
        S --> A
        S --> B
        A -->|ac| C
        A -->|ad| D
        B -->|bc| C
        B -->|bd| D
        """
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    for label in ("ac", "ad", "bc", "bd"):
        assert rendered.count(label) == 1
    for row in diagram.rows:
        if any(label in row for label in ("ac", "ad", "bc", "bd")):
            assert "  B  " not in row


def test_shared_class_endpoint_labels_keep_every_label_visible() -> None:
    sources = (
        'classDiagram\ndirection TB\nN2 "S0" --> "T0" N0 : L0\nN2 "S1" --> "T1" N4 : L1',
        'classDiagram\ndirection TB\nN0 "S0" --> "T0" N2 : L0\nN4 "S1" --> "T1" N2 : L1',
        'classDiagram\ndirection LR\nN2 "S0" --> "T0" N0 : L0\nN2 "S1" --> "T1" N4 : L1',
        'classDiagram\ndirection LR\nN0 "S0" --> "T0" N2 : L0\nN4 "S1" --> "T1" N2 : L1',
    )
    for source in sources:
        diagram = compile_mermaid(source)
        rendered = "\n".join(diagram.rows)

        assert not diagram.diagnostics
        for label in ("S0", "T0", "L0", "S1", "T1", "L1"):
            assert rendered.count(label) == 1


def test_many_parallel_back_edges_stay_in_bounds_and_keep_labels() -> None:
    labels = tuple(f"back-{index}" for index in range(6))
    back_edges = "\n".join(f"B -->|{label}| A" for label in labels)
    for direction in ("TB", "LR"):
        diagram = compile_mermaid(f"flowchart {direction}\nA --> B\n{back_edges}")
        rendered = "\n".join(diagram.rows)

        assert not diagram.diagnostics
        for label in labels:
            assert rendered.count(label) == 1


def test_parallel_self_and_non_adjacent_edges_keep_every_label_visible() -> None:
    sources = (
        "flowchart TB\nA[Alpha] -->|first| A\nA -->|second| A",
        "flowchart LR\nA[Alpha] -->|first| A\nA -->|second| A",
        "flowchart TB\nA --> B --> C\nA -->|first| C\nA -->|second| C",
        "flowchart TB\nA --> B --> C\nC -->|first| A\nC -->|second| A",
    )
    for source in sources:
        diagram = compile_mermaid(source)
        rendered = "\n".join(diagram.rows)

        assert not diagram.diagnostics
        assert rendered.count("first") == 1
        assert rendered.count("second") == 1


def test_directional_markers_follow_vertical_routes() -> None:
    bidirectional = "\n".join(compile_mermaid("flowchart TB\nA <--> B").rows)
    association = "\n".join(compile_mermaid("classDiagram\nA --> B").rows)
    inheritance = "\n".join(compile_mermaid("classDiagram\nA --|> B").rows)

    assert "▲" in bidirectional
    assert "▼" in bidirectional
    assert not {"◀", "▶"} & set(bidirectional)
    assert "▼" in association
    assert "▽" in inheritance


def test_class_realization_renders_dotted_hollow_triangle_toward_interface() -> None:
    diagram = compile_mermaid(
        """classDiagram
        class OperationBinding {
          <<interface>>
        }
        class SubAgentToolShell
        OperationBinding <|.. SubAgentToolShell
        """
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    assert "«interface» OperationBinding" in rendered
    assert "SubAgentToolShell" in rendered
    assert "▽" in rendered
    assert "┊" in rendered


def test_er_diagram_renders_entity_tables_cardinalities_and_relationship_labels() -> None:
    diagram = compile_mermaid(
        """erDiagram
        direction LR
        Student ||--o{ Enrollment : enrolls
        Course ||--o{ Enrollment : offered_in
        Student {
            string student_id PK
            string name
        }
        Course {
            string course_id PK
            string title
        }
        Enrollment {
            string student_id PK, FK
            string course_id PK, FK
            date enroll_date "首次选课日期"
        }
        """
    )
    rendered = "\n".join(diagram.rows)

    assert diagram.kind.value == "er"
    assert not diagram.diagnostics
    for text in (
        "Student",
        "Course",
        "Enrollment",
        "string student_id [PK]",
        "string student_id [PK,FK]",
        "date enroll_date — 首次选课日期",
        "enrolls",
        "offered_in",
    ):
        assert rendered.count(text) == 1
    assert rendered.count("0..*") == 2
    assert rendered.count("├") >= 3


def test_er_non_identifying_relation_renders_dotted_without_direction_arrow() -> None:
    diagram = compile_mermaid('erDiagram\n"驾驶员 档案" }o..|{ CAR : "可以驾驶"')
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    assert "驾驶员 档案" in rendered
    assert "可以驾驶" in rendered
    assert "0..*" in rendered
    assert "1..*" in rendered
    assert "┊" in rendered or "┄" in rendered
    assert not {"▲", "▼", "◀", "▶"} & set(rendered)


def test_er_reverse_directions_reverse_entity_placement() -> None:
    diagrams = {
        direction: compile_mermaid(f"erDiagram\ndirection {direction}\nAlpha ||--o{{ Beta : contains")
        for direction in ("TB", "BT", "LR", "RL")
    }

    def position(direction: str, label: str) -> tuple[int, int]:
        return next((row.index(label), y) for y, row in enumerate(diagrams[direction].rows) if label in row)

    assert position("TB", "Alpha")[1] < position("TB", "Beta")[1]
    assert position("BT", "Alpha")[1] > position("BT", "Beta")[1]
    assert position("LR", "Alpha")[0] < position("LR", "Beta")[0]
    assert position("RL", "Alpha")[0] > position("RL", "Beta")[0]
    assert all(not diagram.diagnostics for diagram in diagrams.values())


@pytest.mark.parametrize(
    "source_template",
    (
        "flowchart {direction}\nA --> B",
        "classDiagram\ndirection {direction}\nA --> B",
        "stateDiagram-v2\ndirection {direction}\nA --> B",
    ),
)
@pytest.mark.parametrize(("direction", "arrow"), (("BT", "▲"), ("RL", "◀")))
def test_reverse_graph_directions_preserve_placement_and_arrow_semantics(
    source_template: str,
    direction: str,
    arrow: str,
) -> None:
    diagram = compile_mermaid(source_template.format(direction=direction))

    def position(label: str) -> tuple[int, int]:
        return next((row.index(label), y) for y, row in enumerate(diagram.rows) if label in row)

    alpha = position("A")
    beta = position("B")
    axis = 1 if direction == "BT" else 0
    assert alpha[axis] > beta[axis]
    assert arrow in "\n".join(diagram.rows)
    assert not diagram.diagnostics


@pytest.mark.parametrize(("forward", "reverse", "axis"), (("TB", "BT", 1), ("LR", "RL", 0)))
@pytest.mark.parametrize("attached", (False, True))
def test_reverse_directions_reverse_cycle_members_and_connected_nodes(
    forward: str, reverse: str, axis: int, *, attached: bool
) -> None:
    body = "A -->|ab| B\nB -->|bc| C\nC -->|ca| A"
    if attached:
        body = f"Start --> A\n{body}\nC --> End"
    labels = ("Start", "A", "B", "C", "End") if attached else ("A", "B", "C")
    for direction in (forward, reverse):
        diagram = compile_mermaid(f"flowchart {direction}\n{body}")
        positions = [
            next((row.index(f"  {label}  "), y)[axis] for y, row in enumerate(diagram.rows) if f"  {label}  " in row)
            for label in labels
        ]

        assert positions == sorted(positions, reverse=direction == reverse)
        assert len(set(positions)) == len(labels)
        assert not diagram.diagnostics
        rendered = "\n".join(diagram.rows)
        for edge_label in ("ab", "bc", "ca"):
            assert rendered.count(edge_label) == 1


def test_pie_chart_renders_proportional_bars_and_show_data() -> None:
    diagram = compile_mermaid(
        """pie showData title 浏览器市场份额
        "Chrome" : 65
        "Safari" : 15
        "Firefox" : 10
        "Edge" : 5
        "Other" : 5
        """
    )
    rendered = "\n".join(diagram.rows)
    chrome_row = next(row for row in diagram.rows if "Chrome" in row)
    other_row = next(row for row in diagram.rows if "Other" in row)

    assert not diagram.diagnostics
    assert "浏览器市场份额" in rendered
    assert "65 65%" in chrome_row
    assert chrome_row.count("█") > other_row.count("█")


@pytest.mark.parametrize(
    "labels",
    (
        ("Customer acquisition channel - Organic", "Customer acquisition channel - Paid"),
        ("客户获客渠道来源分类统计分析报告 自然搜索", "客户获客渠道来源分类统计分析报告 付费推广"),
        ("a" * 40 + "Organic", "a" * 40 + "Paid"),
    ),
)
def test_pie_preserves_complete_labels_and_their_value_association(labels: tuple[str, str]) -> None:
    diagram = compile_mermaid(f'pie\n"{labels[0]}": 60\n"{labels[1]}": 40')
    groups = "\n".join(diagram.rows).split("\n\n")

    assert not diagram.diagnostics
    assert len(groups) == 2
    for group, label, percentage in zip(groups, labels, ("60%", "40%"), strict=True):
        rendered_label = "".join(crop_cell_text(row, 0, 28).strip() for row in group.splitlines())
        assert "".join(rendered_label.split()) == "".join(label.split())
        assert percentage in group
        assert sum("█" in row for row in group.splitlines()) == 1


def test_xychart_renders_bars_lines_axes_and_categories() -> None:
    diagram = compile_mermaid(
        """xychart
        title "Sales Revenue"
        x-axis "Quarter" [Q1, Q2, Q3, Q4]
        y-axis "Revenue" 0 --> 100
        bar [45, 72, 60, 90]
        line [40, 65, 75, 85]
        """
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    for text in ("Sales Revenue", "Quarter", "Revenue", "Q1", "Q4", "bar 1", "line 2"):
        assert text in rendered
    assert "█" in rendered
    assert "●" in rendered
    assert "└" in rendered


def test_xychart_renders_named_series_in_the_legend() -> None:
    diagram = compile_mermaid(
        """xychart
        x-axis [Q1, Q2]
        y-axis Revenue
        bar "Bookings" [25, 45]
        line average [20, 50]
        """
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    assert "Bookings" in rendered
    assert "average" in rendered


@pytest.mark.parametrize("header", ("xychart", "xychart horizontal"))
@pytest.mark.parametrize("values", ("0, 100", "100, 0", "50, 50"))
def test_xychart_line_follows_linear_interpolation(header: str, values: str) -> None:
    diagram = compile_mermaid(f"{header}\nx-axis [A, B]\ny-axis 0 --> 100\nline [{values}]")

    assert not diagram.diagnostics
    markers = [(x, y) for y, row in enumerate(diagram.rows) for x, glyph in enumerate(row) if glyph == "●"]
    assert len(markers) == 2
    first, second = markers
    major = 0 if abs(second[0] - first[0]) >= abs(second[1] - first[1]) else 1
    minor = 1 - major
    path_glyphs = "─│┌┐└┘├┤┬┴┼"
    for coordinate in range(min(first[major], second[major]) + 1, max(first[major], second[major])):
        expected = first[minor] + (second[minor] - first[minor]) * (coordinate - first[major]) / (
            second[major] - first[major]
        )
        strokes = []
        for other in range(min(first[minor], second[minor]), max(first[minor], second[minor]) + 1):
            x, y = (coordinate, other) if major == 0 else (other, coordinate)
            if crop_cell_text(diagram.rows[y], x, 1) in path_glyphs:
                strokes.append(other)
        assert strokes, (coordinate, expected)
        assert all(abs(other - expected) <= 1 for other in strokes), (coordinate, expected, strokes)


@pytest.mark.parametrize("kind", ("bar", "line"))
def test_xychart_renders_series_names_containing_brackets(kind: str) -> None:
    diagram = compile_mermaid(f'xychart\n{kind} "Latency [ms]" [10, 20]')

    assert not diagram.diagnostics
    assert "Latency [ms]" in "\n".join(diagram.rows)


def test_xychart_keeps_lines_visible_over_bars_and_preserves_long_categories() -> None:
    long_label = "A category label too long for one chart slot"
    diagram = compile_mermaid(
        f'''xychart
        x-axis ["{long_label}", Short]
        y-axis 0 --> 100
        line [50, 60]
        bar [50, 60]
        '''
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    assert "●" in rendered
    assert f"1. {long_label}" in rendered


def test_xychart_line_paths_stay_above_grid_lines_and_bars() -> None:
    grid_crossing = compile_mermaid("xychart\nx-axis [A, B]\ny-axis 0 --> 100\nbar [50, 50]\nline [50, 50]")
    grid_row = next(row for row in grid_crossing.rows if row.count("●") == 2)
    first_marker = grid_row.index("●")
    second_marker = grid_row.index("●", first_marker + 1)

    header = "xychart\nx-axis [A, B]\ny-axis 0 --> 100\n"
    bar_crossing = compile_mermaid(header + "bar [0, 100]\nline [0, 100]")
    line_only = compile_mermaid(header + "line [0, 100]")
    bars_only = compile_mermaid(header + "bar [0, 100]")
    crossings = [
        (x, y, glyph)
        for y, row in enumerate(line_only.rows)
        for x, glyph in enumerate(row)
        if glyph in "─│┌┐└┘├┤┬┴┼" and crop_cell_text(bars_only.rows[y], x, 1) == "█"
    ]

    assert "─" in grid_row[first_marker + 1 : second_marker]
    assert crossings
    for x, y, glyph in crossings:
        assert crop_cell_text(bar_crossing.rows[y], x, 1) == glyph


@pytest.mark.parametrize("header", ["xychart", "xychart horizontal"])
def test_xychart_multiple_line_paths_preserve_every_series_marker(header: str) -> None:
    diagram = compile_mermaid(
        f'{header}\nx-axis [A, B, C]\ny-axis 0 --> 100\nline "first" [20, 50, 20]\nline "second" [100, 0, 100]'
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    assert rendered.count("●") == 4
    assert rendered.count("◆") == 4


@pytest.mark.parametrize(("header", "coordinate"), (("xychart", 0), ("xychart horizontal", 1)))
def test_numeric_xychart_spans_the_axis_for_each_series(
    header: str,
    coordinate: int,
) -> None:
    diagram = compile_mermaid(f'{header}\nx-axis 0 --> 6\ny-axis 0 --> 5\nline "four" [1, 2, 3, 4]\nline "two" [4, 1]')
    plot_limit = 14 if header == "xychart" else 12
    circles = [
        (x, y)[coordinate]
        for y, row in enumerate(diagram.rows)
        for x, glyph in enumerate(row)
        if glyph == "●" and (y <= plot_limit)
    ]
    diamonds = [
        (x, y)[coordinate]
        for y, row in enumerate(diagram.rows)
        for x, glyph in enumerate(row)
        if glyph == "◆" and (y <= plot_limit)
    ]

    assert len(circles) == 4
    assert len(diamonds) == 2
    assert (min(diamonds), max(diamonds)) == (min(circles), max(circles))
    assert not diagram.diagnostics


def test_xychart_preflights_oversized_plot_before_drawing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_draw_path = TerminalCanvas.draw_path

    def guarded_draw_path(canvas: TerminalCanvas, points: Iterable[Point]) -> None:
        materialized = tuple(points)
        assert all(point.x <= MAX_CANVAS_AXIS and point.y <= MAX_CANVAS_AXIS for point in materialized)
        original_draw_path(canvas, materialized)

    monkeypatch.setattr(TerminalCanvas, "draw_path", guarded_draw_path)
    bars = "\n".join(f'bar "series {index}" [1]' for index in range(249))
    long_line = ", ".join("1" for _ in range(251))
    diagram = compile_mermaid(f"xychart\n{bars}\nline [{long_line}]")

    assert diagram.diagnostics[-1].code is DiagnosticCode.CANVAS_LIMIT
    assert diagram.width < 100
    assert diagram.height < 20


def test_horizontal_xychart_renders_categories_and_numeric_ticks() -> None:
    diagram = compile_mermaid(
        """xychart horizontal
        x-axis [Jan, Feb, Mar]
        y-axis 0 --> 30
        bar [10, 20, 30]
        """
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    assert all(label in rendered for label in ("Jan", "Feb", "Mar", "0", "30"))
    assert "█" in rendered


@pytest.mark.parametrize(("category_title", "value_title"), (("Month", "Revenue"), ("月份", "收入")))
def test_horizontal_xychart_keeps_axis_titles_with_their_data(category_title: str, value_title: str) -> None:
    diagram = compile_mermaid(
        f'xychart horizontal\nx-axis "{category_title}" [Jan, Feb]\ny-axis "{value_title}" 0 --> 10\nbar [2, 5]'
    )

    assert not diagram.diagnostics
    title_y = next(y for y, row in enumerate(diagram.rows) if category_title in row)
    category_y = next(y for y, row in enumerate(diagram.rows) if "Jan" in row)
    left = diagram.rows[category_y].index("│")
    assert title_y < category_y
    assert cell_len(diagram.rows[title_y]) < left
    ticks_y = next(y for y, row in enumerate(diagram.rows) if row.strip().split() == ["0", "5", "10"])
    assert diagram.rows[ticks_y + 1].strip() == value_title
    assert cell_len(diagram.rows[ticks_y + 1].split(value_title)[0]) >= left


def test_horizontal_xychart_line_path_stays_above_bars() -> None:
    diagram = compile_mermaid("xychart horizontal\nx-axis [A, B]\ny-axis 0 --> 100\nbar [100, 100]\nline [0, 100]")
    target_row = next(row for row in diagram.rows if "B" in row and "●" in row)
    marker = target_row.index("●")

    assert not diagram.diagnostics
    assert "─" in target_row[:marker]


def test_xychart_respects_explicit_nonzero_y_axis_bounds() -> None:
    positive = compile_mermaid("xychart\nx-axis [A, B]\ny-axis 10 --> 20\nbar [10, 20]")
    negative = compile_mermaid("xychart\nx-axis [A, B]\ny-axis -20 --> -10\nbar [-20, -10]")

    assert not positive.diagnostics
    assert not negative.diagnostics
    assert any(row.lstrip().startswith("10 ") for row in positive.rows)
    assert not any(row.lstrip().startswith("0 ") for row in positive.rows)
    assert any(row.lstrip().startswith("-10 ") for row in negative.rows)
    assert not any(row.lstrip().startswith("0 ") for row in negative.rows)


def test_single_point_xychart_uses_the_explicit_numeric_x_axis() -> None:
    diagram = compile_mermaid("xychart\nx-axis 2026 --> 2030\ny-axis 0 --> 100\nline [50]")
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    assert "2026" in rendered
    assert diagram.rows[-1].strip() == "2026"


def test_horizontal_xychart_does_not_draw_a_bar_for_zero() -> None:
    diagram = compile_mermaid("xychart horizontal\nx-axis [Zero, One]\ny-axis 0 --> 1\nbar [0, 1]")
    zero_row = next(row for row in diagram.rows if "Zero" in row)
    one_row = next(row for row in diagram.rows if "One" in row)

    assert not diagram.diagnostics
    assert "█" not in zero_row
    assert "█" in one_row


@pytest.mark.parametrize(
    ("header", "body"),
    [
        ("xychart", "bar [1, 2]"),
        ("xychart horizontal", "bar [1, 2]"),
        ("pie", '"A": 1'),
        ("quadrantChart", "P: [0.8, 0.9]"),
    ],
)
@pytest.mark.parametrize(
    "title",
    [
        "Monthly revenue for internal accounts excluding partner transactions and regional adjustments",
        "内部账户每月收入，不含合作伙伴交易及地区调整。" * 3,  # noqa: RUF001 — exercise full-width punctuation
    ],
)
def test_chart_titles_wrap_completely_above_unchanged_plot(header: str, body: str, title: str) -> None:
    plain = compile_mermaid(f"{header}\n{body}")
    diagram = compile_mermaid(f"{header}\ntitle {title}\n{body}")
    assert not diagram.diagnostics
    title_height = next(index for index, row in enumerate(diagram.rows) if not row.strip())
    assert title_height >= 2
    assert "".join("".join(diagram.rows[:title_height]).split()) == "".join(title.split())
    assert diagram.rows[title_height + 1 :] == plain.rows
    assert diagram.height == plain.height + title_height + 1


@pytest.mark.parametrize(
    ("header", "body"),
    [
        ("xychart", "bar [1, 2]"),
        ("xychart horizontal", "bar [1, 2]"),
        ("pie", '"A": 1'),
        ("quadrantChart", "P: [0.8, 0.9]"),
    ],
)
def test_wrapped_title_height_is_checked_before_drawing(
    header: str, body: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = f"{header}\ntitle {'完整标题 ' * 100}\n{body}"
    rendered = compile_mermaid(source)
    title_height = next(index for index, row in enumerate(rendered.rows) if not row.strip())
    base_source = f"{header}\n{body}"
    estimates: list[tuple[int, int]] = []

    def capture_budget(width: int, height: int) -> bool:
        estimates.append((width, height))
        return False

    compile_chart(base_source, parse_mermaid(base_source), capture_budget)
    baseline_height = estimates[0][1]

    def deny_extra_height(width: int, height: int) -> bool:
        assert height == baseline_height + title_height + 1
        return True

    def must_not_draw(*args: object, **kwargs: object) -> None:
        pytest.fail("chart was drawn before checking its wrapped title height")

    monkeypatch.setattr(TerminalCanvas, "draw_text", must_not_draw)
    with pytest.raises(ChartCanvasLimit):
        compile_chart(source, parse_mermaid(source), deny_extra_height)


@pytest.mark.parametrize(
    "title",
    [
        "Elapsed time since first successful deployment in milliseconds",
        "自首次成功部署以来经过的时间（毫秒）",  # noqa: RUF001 — exercise full-width punctuation
    ],
)
def test_xy_vertical_wraps_complete_axis_title_before_series_legend(title: str) -> None:
    plain = compile_mermaid("xychart\nx-axis [A, B]\nbar Revenue [1, 2]")
    diagram = compile_mermaid(f'xychart\nx-axis "{title}" [A, B]\nbar Revenue [1, 2]')
    assert not diagram.diagnostics
    first_title_row = next(index for index, row in enumerate(diagram.rows) if row.split() == ["A", "B"]) + 1
    legend_row = next(index for index, row in enumerate(diagram.rows) if "█ Revenue" in row)
    title_rows = diagram.rows[first_title_row : legend_row - 1]
    assert len(title_rows) >= 2
    assert "".join("".join(title_rows).split()) == "".join(title.split())
    assert diagram.rows[:first_title_row] == plain.rows[:first_title_row]
    assert diagram.rows[legend_row:] == plain.rows[first_title_row + 1 :]
    assert diagram.width == plain.width


@pytest.mark.parametrize(
    "source",
    [
        f'xychart\nx-axis "{"Elapsed time " * 30}milliseconds" [A, B]\nbar [1, 2]',
        (f'quadrantChart\nx-axis "{"Elapsed time " * 10}milliseconds" --> High\ny-axis Low --> "{"Revenue " * 10}USD"'),
    ],
)
def test_complete_axis_labels_are_budgeted_before_drawing(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    diagram = compile_mermaid(source)
    assert not diagram.diagnostics

    def reject_budget(width: int, height: int) -> bool:
        assert width >= diagram.width
        assert height >= diagram.height
        return True

    def must_not_draw(*args: object, **kwargs: object) -> None:
        pytest.fail("chart was drawn before checking complete axis labels")

    monkeypatch.setattr(TerminalCanvas, "draw_text", must_not_draw)
    with pytest.raises(ChartCanvasLimit):
        compile_chart(source, parse_mermaid(source), reject_budget)


def test_quadrant_chart_renders_quadrants_points_axes_and_legend() -> None:
    diagram = compile_mermaid(
        """quadrantChart
        title Reach and engagement
        x-axis Low Reach --> High Reach
        y-axis Low Engagement --> High Engagement
        quadrant-1 Expand
        quadrant-2 Promote
        quadrant-3 Re-evaluate
        quadrant-4 Improve
        Campaign A: [0.3, 0.6]
        Campaign B: [0.78, 0.34]
        """
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    for text in (
        "Reach and engagement",
        "Low Reach",
        "High Reach",
        "Low Engagement",
        "High Engagement",
        "Expand",
        "Promote",
        "Re-evaluate",
        "Improve",
        "Campaign A [0.3, 0.6]",
        "Campaign B [0.78, 0.34]",
    ):
        assert text in rendered
    assert "┼" in rendered


def test_quadrant_chart_legend_uses_the_collision_marker() -> None:
    diagram = compile_mermaid("quadrantChart\nFirst: [0.5, 0.5]\nSecond: [0.5, 0.5]")
    rendered = "\n".join(diagram.rows)

    assert "● First [0.5, 0.5]" in rendered
    assert "◆ Second [0.5, 0.5]" in rendered
    assert not diagram.diagnostics


@pytest.mark.parametrize(
    ("quadrant", "label", "x", "y", "cell_x", "cell_y"),
    (
        (1, "高优先级", "0.52", "1", 37, 2),
        (1, "高优先级", "0.54", "1", 38, 2),
        (1, "Focus", "0.54", "1", 38, 2),
        (2, "高优先级", "0.02", "1", 4, 2),
        (3, "高优先级", "0.02", "0.4", 4, 12),
        (4, "高优先级", "0.54", "0.4", 38, 12),
    ),
)
def test_quadrant_chart_preserves_points_and_overlapping_captions(
    quadrant: int, label: str, x: str, y: str, cell_x: int, cell_y: int
) -> None:
    diagram = compile_mermaid(f"quadrantChart\nquadrant-{quadrant} {label}\nP: [{x}, {y}]")

    assert not diagram.diagnostics
    assert crop_cell_text(diagram.rows[cell_y], cell_x, 1) == "●"
    assert f"quadrant-{quadrant}: {label}" in "\n".join(diagram.rows[21:])
    assert f"● P [{x}, {y}]" in "\n".join(diagram.rows[21:])
    assert cell_len(diagram.rows[cell_y]) == cell_len(diagram.rows[1])


@pytest.mark.parametrize("body", ("title Priorities", "x-axis Low --> High", "quadrant-1 Focus"))
def test_quadrant_chart_renders_label_only_content(body: str) -> None:
    diagram = compile_mermaid(f"quadrantChart\n{body}")

    assert not diagram.diagnostics
    rendered = "\n".join(diagram.rows)
    assert body.split()[-1] in rendered
    assert "┼" in rendered


@pytest.mark.parametrize("quadrant", [1, 2, 3, 4])
@pytest.mark.parametrize("point", ["", "\nP: [0.54, 1]"])
@pytest.mark.parametrize(
    "label",
    ["The most important actions for next quarter", "下个季度最重要的行动以及应当优先安排的任务"],
)
def test_quadrant_long_captions_are_kept_in_full_legend(quadrant: int, point: str, label: str) -> None:
    diagram = compile_mermaid(f"quadrantChart\nquadrant-{quadrant} {label}{point}")
    assert not diagram.diagnostics
    assert cell_len(label) > 32
    assert f"quadrant-{quadrant}: {label}" in "\n".join(diagram.rows[21:])
    assert "\n".join(diagram.rows).count(label) == 1
    assert label[:10] not in "\n".join(diagram.rows[:21])
    if point:
        assert "● P [0.54, 1]" in "\n".join(diagram.rows[21:])
        assert crop_cell_text(diagram.rows[2], 38, 1) == "●"


@pytest.mark.parametrize("axis", ["x-axis", "y-axis"])
@pytest.mark.parametrize("endpoint", [0, 1])
@pytest.mark.parametrize(
    "label",
    [
        "Elapsed time since first successful deployment in milliseconds",
        "自首次成功部署以来经过的时间单位为毫秒",
    ],
)
def test_quadrant_long_axis_endpoints_survive_in_full_axis_legend(axis: str, endpoint: int, label: str) -> None:
    labels = ["Low", "High"]
    labels[endpoint] = label
    declaration = f'{axis} "{labels[0]}" --> "{labels[1]}"'
    diagram = compile_mermaid(f"quadrantChart\n{declaration}\nquadrant-1 Focus\nP: [0.8, 0.9]")
    assert not diagram.diagnostics
    assert cell_len(label) > 34
    legend = "\n".join(diagram.rows[23:])
    assert f'{axis}: "{labels[0]}" --> "{labels[1]}"' in legend
    assert "● P [0.8, 0.9]" in legend
    assert "Focus" in "\n".join(diagram.rows[:23])


def test_treemap_renders_nested_proportional_rectangles_and_full_legend() -> None:
    diagram = compile_mermaid(
        """treemap-beta
        "Products"
            "Desktop": 40
            "Mobile": 35
        "Services"
            "Cloud": 20
            "Support": 5
        """
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    for text in (
        "Products 75",
        "Desktop 40",
        "Mobile 35",
        "Services 25",
        "Cloud 20",
        "Support 5",
        "Products / Desktop: 40",
        "Services / Support: 5",
    ):
        assert text in rendered
    assert rendered.count("┌") >= 6


def test_treemap_legend_keeps_descendants_of_too_small_rectangles() -> None:
    diagram = compile_mermaid(
        """treemap-beta
        "Large"
            "Visible": 999
        "Tiny"
            "Hidden descendant": 1
        """
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    assert "• Tiny: 1" in rendered
    assert "• Tiny / Hidden descendant: 1" in rendered


def test_treemap_preflights_deep_legend_before_drawing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_draw_text = TerminalCanvas.draw_text

    def guarded_draw_text(canvas: TerminalCanvas, x: int, y: int, text: str) -> None:
        assert cell_len(text) <= MAX_CANVAS_AXIS
        original_draw_text(canvas, x, y, text)

    monkeypatch.setattr(TerminalCanvas, "draw_text", guarded_draw_text)
    rows = [f'{" " * depth}"{"x" * 80}{depth}"' for depth in range(60)]
    rows[-1] += ": 1"
    diagram = compile_mermaid("treemap-beta\n" + "\n".join(rows))

    assert diagram.diagnostics[-1].code is DiagnosticCode.CANVAS_LIMIT
    assert diagram.width < 100
    assert diagram.height < 20


def test_treemap_without_drawable_rectangles_starts_at_the_legend() -> None:
    leaves = "\n".join(f'"Item {index}": 1' for index in range(160))
    diagram = compile_mermaid(f"treemap-beta\n{leaves}")

    assert diagram.rows[0].startswith("• Item 0: 1")
    assert all(row for row in diagram.rows)
    assert "┌" not in "\n".join(diagram.rows)
    assert not diagram.diagnostics


def test_backward_labels_from_different_sources_get_distinct_target_lanes() -> None:
    diagram = compile_mermaid("flowchart TB\nA --> B --> C\nB -->|from B| A\nC -->|from C| A")
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    assert rendered.count("from B") == 1
    assert rendered.count("from C") == 1


def test_sequence_diagram_renders_lifelines_messages_and_no_phantom_participants() -> None:
    diagram = compile_mermaid(
        """sequenceDiagram
        actor U as 用户
        participant S as Service
        U->>S: 请求
        S-->>U: response
        """
    )
    rendered = "\n".join(diagram.rows)

    assert not diagram.diagnostics
    assert "用户" in rendered
    assert "Service" in rendered
    assert "请求" in rendered
    assert "response" in rendered
    assert "S-" not in rendered
    assert "┊" in rendered


def test_state_cycle_keeps_every_branch_label_visible() -> None:
    source = """stateDiagram-v2
        [*] --> Idle
        Idle --> Running : 请求到达 Submit
        Running --> WaitingForTool : 请求工具 Call Tool
        WaitingForTool --> Running : 工具返回 Tool Result
        Running --> Completed : 任务完成 Done
        Running --> Failed : 发生错误 Error
        Failed --> Idle : 重置 Reset
        Completed --> [*]
        """
    ir = parse_mermaid(source)
    diagram = compile_mermaid(source)
    rendered = "\n".join(diagram.rows)
    pseudo_start = next(node.node_id for node in ir.nodes if node.shape is NodeShape.PSEUDO_START)
    pseudo_end = next(node.node_id for node in ir.nodes if node.shape is NodeShape.PSEUDO_END)

    assert not diagram.diagnostics
    assert [(edge.source, edge.target) for edge in ir.edges] == [
        (pseudo_start, "Idle"),
        ("Idle", "Running"),
        ("Running", "WaitingForTool"),
        ("WaitingForTool", "Running"),
        ("Running", "Completed"),
        ("Running", "Failed"),
        ("Failed", "Idle"),
        ("Completed", pseudo_end),
    ]
    for label in (
        "请求到达 Submit",
        "请求工具 Call Tool",
        "工具返回 Tool Result",
        "任务完成 Done",
        "发生错误 Error",
        "重置 Reset",
    ):
        assert rendered.count(label) == 1


def test_sequence_preserves_all_declared_participants_and_message_directions() -> None:
    source = """sequenceDiagram
        actor User
        participant TUI
        participant Engine
        participant Model
        participant Tool
        User->>TUI: ✍️ 提交问题 Submit Question
        TUI->>Engine: 构建会话 Build Session
        Engine->>Model: 发送提示 Send Prompt
        Model->>Tool: 调用工具 Call Tool
        Tool-->>Model: 返回结果 Return Result
        Model-->>Engine: 输出 Token Token
        Engine-->>TUI: 流式更新 Stream Update
        TUI-->>User: 流式回复 Stream Reply
        """
    ir = parse_mermaid(source)
    diagram = compile_mermaid(source)
    rendered = "\n".join(diagram.rows)
    participant_header = "\n".join(diagram.rows[:3])

    assert not diagram.diagnostics
    assert [(edge.source, edge.target) for edge in ir.edges] == [
        ("User", "TUI"),
        ("TUI", "Engine"),
        ("Engine", "Model"),
        ("Model", "Tool"),
        ("Tool", "Model"),
        ("Model", "Engine"),
        ("Engine", "TUI"),
        ("TUI", "User"),
    ]
    assert [edge.style for edge in ir.edges] == [
        EdgeStyle.SOLID,
        EdgeStyle.SOLID,
        EdgeStyle.SOLID,
        EdgeStyle.SOLID,
        EdgeStyle.DOTTED,
        EdgeStyle.DOTTED,
        EdgeStyle.DOTTED,
        EdgeStyle.DOTTED,
    ]
    for participant in ("User", "TUI", "Engine", "Model", "Tool"):
        assert participant_header.count(participant) == 1
    for label in (
        "✍️ 提交问题 Submit Question",
        "构建会话 Build Session",
        "发送提示 Send Prompt",
        "调用工具 Call Tool",
        "返回结果 Return Result",
        "输出 Token Token",
        "流式更新 Stream Update",
        "流式回复 Stream Reply",
    ):
        assert rendered.count(label) == 1


def test_sequence_message_cannot_expand_canvas_past_axis_limit() -> None:
    diagram = compile_mermaid(f"sequenceDiagram\nA->>B: {'x' * 5000}")

    assert diagram.diagnostics
    assert diagram.width < 100
    assert diagram.diagnostics[-1].code is DiagnosticCode.CANVAS_LIMIT


def test_sequence_total_canvas_area_is_bounded_before_materialization() -> None:
    participants = "\n".join(f"participant P{index}" for index in range(175))
    messages = "\n".join(f"P{index % 174}->>P{(index % 174) + 1}: message {index}" for index in range(500))
    diagram = compile_mermaid(f"sequenceDiagram\n{participants}\n{messages}")

    assert diagram.diagnostics[-1].code is DiagnosticCode.CANVAS_LIMIT
    assert diagram.width < 100
    assert diagram.height < 20


def test_sequence_self_message_arrow_follows_final_segment_back_to_lifeline() -> None:
    diagram = compile_mermaid("sequenceDiagram\nparticipant A\nA->>A: retry")
    rendered = "\n".join(diagram.rows)

    assert "◀" in rendered
    assert "▶" not in rendered
    assert not diagram.diagnostics


def test_diagnostic_canvas_uses_the_requested_locale() -> None:
    localizer = Localizer("zh-Hans")
    diagram = compile_mermaid(
        "flowchart LR\nA -->",
        render_message=localizer.render,
    )
    rendered = "\n".join(diagram.rows)

    assert "Mermaid 图表无法渲染" in rendered
    assert "第 2 行" in rendered
    assert "缺少目标" not in rendered
    assert "Unsupported or malformed" not in rendered


def test_canvas_sanitizes_terminal_control_characters() -> None:
    canvas = TerminalCanvas()
    canvas.draw_text(0, 0, "safe\x1b[31m")

    row = canvas.rows(canvas.natural_width, canvas.natural_height)[0]
    assert "\x1b" not in row
    assert "�" in row


def test_node_geometry_measures_sanitized_control_characters() -> None:
    controls = "\x1b" * 12
    diagram = compile_mermaid(f"flowchart LR\nA[before{controls}after]")
    rendered = "\n".join(diagram.rows)
    content_row = next(row for row in diagram.rows if "before" in row)
    top_row = diagram.rows[diagram.rows.index(content_row) - 1]
    left = top_row.index("┌")
    right = top_row.index("┐")

    assert "�" * 12 in rendered
    assert content_row[left] == "│"
    assert content_row[right] == "│"
