# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fixed-grid and explicitly simplified diagrams preserve source semantics."""

from __future__ import annotations

from decimal import Decimal
from itertools import pairwise

import pytest

from chrys.app.tui.widgets.markdown.diagram import compile_mermaid
from chrys.app.tui.widgets.markdown.diagram.model import (
    DiagnosticCode,
    DiagramEdge,
    DiagramKind,
    DiagramNode,
    Direction,
    NodeShape,
    PlacedNode,
    XYChart,
)
from chrys.app.tui.widgets.markdown.diagram.parser import parse_mermaid
from chrys.app.tui.widgets.markdown.diagram.renderers.structure import compile_block
from chrys.app.tui.widgets.markdown.diagram.renderers.xy import _xy_labels
from chrys.app.tui.widgets.markdown.diagram.router import route_edges
from chrys.app.tui.widgets.markdown.diagram.specs.structure import BlockChart
from chrys.foundation.i18n import Localizer

BLOCK = """block-beta
columns 3
a["数据库"] b["缓存"] c["消息队列"]
d["API 服务"] e["Web 前端"] f["移动端"]
a --> d
b --> d
c --> f
e --> d
"""
ARCHITECTURE = """architecture-beta
group api(cloud)[API 层]
service gateway(server)[网关] in api
service order(server)[订单服务] in api
service pay(server)[支付服务] in api
service db(database)[数据库] in api
service cache(database)[缓存] in api
gateway:R --> L:order
gateway:B --> T:pay
order:R --> L:db
order:B --> T:cache
"""
SANKEY = """sankey-beta
能源,电力,40
能源,交通,25
能源,工业,35
电力,居民,15
电力,商业,25
交通,物流,10
工业,制造业,20
"""


@pytest.mark.parametrize(
    ("source", "kind"),
    [(BLOCK, DiagramKind.BLOCK), (ARCHITECTURE, DiagramKind.ARCHITECTURE), (SANKEY, DiagramKind.SANKEY)],
)
def test_structure_examples_render_through_public_entrypoint(source: str, kind: DiagramKind) -> None:
    diagram = compile_mermaid("%% Leading comment\n" + source)
    assert diagram.kind is kind
    assert not diagram.diagnostics
    assert diagram == compile_mermaid("%% Leading comment\n" + source)
    assert diagram.width <= 4096
    assert diagram.width * diagram.height <= 1_000_000


@pytest.mark.parametrize("source", ["flowchart LR\nA --> B", "xychart\nbar [1, 2]"])
def test_block_renderer_rejects_missing_or_wrong_chart_data(source: str) -> None:
    with pytest.raises(TypeError, match="not a block chart"):
        compile_block(source, parse_mermaid(source), lambda width, height: False)


def test_block_keeps_columns_and_source_slot_order() -> None:
    parsed = parse_mermaid(BLOCK)
    assert isinstance(parsed.chart, BlockChart)
    assert parsed.chart.columns == 3
    assert [cell.node_id for cell in parsed.chart.cells] == list("abcdef")
    rows = compile_mermaid(BLOCK).rows
    first = next(row for row in rows if "数据库" in row)
    second = next(row for row in rows if "API 服务" in row)
    assert first.index("数据库") < first.index("缓存") < first.index("消息队列")
    assert second.index("API 服务") < second.index("Web 前端") < second.index("移动端")
    assert rows.index(first) < rows.index(second)


@pytest.mark.parametrize("operator", ["-->", "<--"])
@pytest.mark.parametrize("declarations_first", [False, True])
def test_block_connection_updates_endpoint_labels_and_shapes_without_new_slots(
    operator: str, declarations_first: bool
) -> None:
    declarations = "a b"
    connection = f"a(Alpha) {operator} b{{Beta}}"
    body = f"{declarations}\n{connection}" if declarations_first else f"{connection}\n{declarations}"
    source = f"block-beta\ncolumns 2\n{body}"
    parsed = parse_mermaid(source)

    assert not parsed.diagnostics
    assert isinstance(parsed.chart, BlockChart)
    assert [cell.node_id for cell in parsed.chart.cells] == ["a", "b"]
    assert [(node.node_id, node.label, node.shape) for node in parsed.nodes] == [
        ("a", "Alpha", NodeShape.ROUNDED),
        ("b", "Beta", NodeShape.DECISION),
    ]
    diagram = compile_mermaid(source)
    assert not diagram.diagnostics
    assert "Alpha" in "\n".join(diagram.rows)
    assert "Beta" in "\n".join(diagram.rows)


def test_block_spans_and_intentional_empty_slots() -> None:
    source = "block-beta\ncolumns 3\na[Wide]:2 space\nb[Bottom] c[Next] space"
    parsed = parse_mermaid(source)
    assert isinstance(parsed.chart, BlockChart)
    assert [cell.span for cell in parsed.chart.cells] == [2, 1, 1, 1, 1]
    diagram = compile_mermaid(source)
    assert not diagram.diagnostics
    top = next(row for row in diagram.rows if "Wide" in row)
    bottom = next(row for row in diagram.rows if "Bottom" in row)
    assert top.count("│") == 2
    assert bottom.count("│") == 4


def test_block_parallel_labels_and_self_edge_survive() -> None:
    diagram = compile_mermaid(
        "block-beta\ncolumns 1\na[Alpha] b[Beta]\na -->|first| b\na -->|second| b\nb -->|retry| b"
    )
    assert not diagram.diagnostics
    rendered = "\n".join(diagram.rows)
    for text in ("Alpha", "Beta", "first", "second", "retry"):
        assert text in rendered


@pytest.mark.parametrize("direction", [Direction.TOP_DOWN, Direction.LEFT_RIGHT])
@pytest.mark.parametrize("parallel", [False, True])
def test_rank_channels_avoid_taller_or_wider_sibling(direction: Direction, parallel: bool) -> None:
    boxes = {"a": (10, 2, 7, 3), "b": (30, 2, 7, 15), "c": (10, 30, 7, 3), "d": (30, 30, 7, 3)}
    if direction is Direction.LEFT_RIGHT:
        boxes = {key: (y, x, height, width) for key, (x, y, width, height) in boxes.items()}
    placed = {key: PlacedNode(DiagramNode(key, key), *box, (key,)) for key, box in boxes.items()}
    edges = (DiagramEdge("a", "d"),) * (2 if parallel else 1)
    for routed in route_edges(edges, placed, {"a": 0, "b": 0, "c": 1, "d": 1}, direction):
        for first, second in pairwise(routed.points):
            for x in range(min(first.x, second.x), max(first.x, second.x) + 1):
                for y in range(min(first.y, second.y), max(first.y, second.y) + 1):
                    obstacle = placed["b"]
                    assert not (
                        obstacle.x <= x < obstacle.x + obstacle.width and obstacle.y <= y < obstacle.y + obstacle.height
                    )


def test_block_repeated_self_routes_have_top_margin() -> None:
    diagram = compile_mermaid("block-beta\na[Alpha]\na -->|one| a\na -->|two| a")
    assert not diagram.diagnostics
    assert "one" in "\n".join(diagram.rows)
    assert "two" in "\n".join(diagram.rows)


def test_architecture_preserves_group_ancestry_ports_and_icon_names() -> None:
    source = "architecture-beta\ngroup outer(cloud)[外部]\ngroup inner[内部] in outer\nservice a(server)[A] in inner\nservice b(database)[B]\na:B <--> T:b"
    parsed = parse_mermaid(source)
    assert not parsed.diagnostics
    assert parsed.simplified
    assert "outer[外部] (cloud) / inner[内部]" in parsed.nodes[0].sections[0][0]
    assert parsed.nodes[0].annotation == "server"
    assert parsed.edges[0].source_label == "B"
    assert parsed.edges[0].target_label == "T"
    assert parsed.edges[0].source_marker
    assert parsed.edges[0].directed
    assert "Schematic layout" in "\n".join(compile_mermaid(source).rows)


@pytest.mark.parametrize(
    ("source", "heading"),
    [
        (ARCHITECTURE, "示意布局：分组与端口以标签保留，不约束空间位置。"),  # noqa: RUF001 — exact translation
        (SANKEY, "带数值的流向图：保留各条流与数值，线宽不按数值比例变化。"),  # noqa: RUF001 — exact translation
    ],
)
def test_schematic_heading_uses_the_requested_locale(source: str, heading: str) -> None:
    diagram = compile_mermaid(source, render_message=Localizer("zh-Hans").render)
    assert not diagram.diagnostics
    assert diagram.rows[0] == heading
    assert "Schematic layout" not in "\n".join(diagram.rows)
    assert "Weighted flow diagram" not in "\n".join(diagram.rows)


def test_empty_architecture_group_remains_visible() -> None:
    diagram = compile_mermaid("architecture-beta\ngroup api(cloud)[API]")
    assert not diagram.diagnostics
    assert "api[API]" in "\n".join(diagram.rows)


def test_occupied_architecture_groups_do_not_create_extra_empty_group_nodes() -> None:
    parsed = parse_mermaid(
        "architecture-beta\ngroup outer[Outer]\ngroup inner[Inner] in outer\nservice api[API] in inner"
    )
    assert not parsed.diagnostics
    assert [node.node_id for node in parsed.nodes] == ["api"]
    assert parsed.nodes[0].sections == (("in outer[Outer] / inner[Inner]",),)


def test_sankey_csv_and_parallel_values_are_not_lost() -> None:
    source = 'sankey-beta\n"Energy, primary",Power,40.125\n"Energy, primary",Power,2.5'
    parsed = parse_mermaid(source)
    assert parsed.nodes[0].label == "Energy, primary"
    assert [edge.label for edge in parsed.edges] == ["40.125", "2.5"]
    diagram = compile_mermaid(source)
    assert not diagram.diagnostics
    rendered = "\n".join(diagram.rows)
    assert "40.125" in rendered and "2.5" in rendered
    assert "not proportional" in rendered


@pytest.mark.parametrize(("field", "name"), [("'A'", "'A'"), ('"""A"""', '"A"'), ('"""A, B"""', '"A, B"')])
def test_sankey_decoded_literal_quotes_do_not_merge_node_identities(field: str, name: str) -> None:
    source = f"sankey-beta\n{field},A,1\nA,{field},2"
    parsed = parse_mermaid(source)
    assert not parsed.diagnostics
    assert [node.label for node in parsed.nodes] == [name, "A"]
    quoted, plain = parsed.nodes
    assert quoted.node_id != plain.node_id
    assert [(edge.source, edge.target) for edge in parsed.edges] == [
        (quoted.node_id, plain.node_id),
        (plain.node_id, quoted.node_id),
    ]
    diagram = compile_mermaid(source)
    assert not diagram.diagnostics
    assert name in "\n".join(diagram.rows)


@pytest.mark.parametrize(
    "source",
    [
        "block-beta invalid\na",
        "block-beta\ncolumns 0\na",
        "block-beta\ncolumns 2\na:3",
        "block-beta\na --> missing",
        "block-beta\na a",
        "block-beta\nblock:outer\na\nend",
        "architecture-beta\nservice a\na --> a",
        "architecture-beta\nservice a in missing",
        "architecture-beta\ngroup a in b\ngroup b in a",
        "architecture-beta\nservice a\na:R --> L:missing",
        "sankey-beta\nSource Target 10",
        "sankey-beta\nSource,Target,NaN",
        "sankey-beta\nSource,Target,-1",
    ],
)
def test_unimplemented_or_invalid_structure_fails_closed(source: str) -> None:
    assert parse_mermaid(source).has_fatal_error
    assert compile_mermaid(source).diagnostics


def test_excessive_block_grid_is_rejected_before_canvas(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_draw(*args: object, **kwargs: object) -> None:
        pytest.fail("oversized block allocated a canvas")

    monkeypatch.setattr(
        "chrys.app.tui.widgets.markdown.diagram.renderers.structure.TerminalCanvas.draw_text", must_not_draw
    )
    # Call renderer directly: the diagnostic fallback is itself drawn on a canvas.
    from chrys.app.tui.widgets.markdown.diagram.renderers.common import ChartCanvasLimit
    from chrys.app.tui.widgets.markdown.diagram.renderers.structure import compile_block

    source = "block-beta\ncolumns 200\n" + " ".join(f"n{i}[{'Long label' * 4}]" for i in range(200))
    with pytest.raises(ChartCanvasLimit):
        compile_block(source, parse_mermaid(source), lambda width, height: width > 4096 or width * height > 1_000_000)


@pytest.mark.parametrize("orientation", ["", " horizontal"])
def test_computed_xy_ticks_stay_inline_and_readable(orientation: str) -> None:
    source = f"xychart{orientation}\nx-axis 0 --> 100\ny-axis 0 --> 40\nbar [12,24,31,18]"
    chart = parse_mermaid(source).chart
    assert isinstance(chart, XYChart)
    assert _xy_labels(chart, 4) == ("0", "33.3", "66.7", "100")
    diagram = compile_mermaid(source)
    rendered = "\n".join(diagram.rows)
    assert not diagram.diagnostics
    assert "33.3" in rendered and "66.7" in rendered
    assert "1. 0" not in rendered
    assert "3333333" not in rendered


@pytest.mark.parametrize("count", [4, 7, 8, 10, 12])
def test_xy_tick_precision_adapts_to_small_offset_ranges(count: int) -> None:
    chart = parse_mermaid("xychart\nx-axis 123.000001 --> 123.000002\nline [1,2]").chart
    assert isinstance(chart, XYChart)
    labels = _xy_labels(chart, count)
    assert len(set(labels)) == count
    assert Decimal(labels[0]) == chart.x_min
    assert Decimal(labels[-1]) == chart.x_max
    assert all(len(label) < 20 for label in labels)


def test_pie_source_literals_keep_their_precision() -> None:
    diagram = compile_mermaid('pie showData\n"Exact" : 1.234567')
    assert "1.234567" in "\n".join(diagram.rows)


@pytest.mark.parametrize("source", [BLOCK, ARCHITECTURE, SANKEY])
def test_semantic_configuration_is_not_silently_ignored(source: str) -> None:
    result = parse_mermaid('%%{init: {"packet": {"bitsPerRow": 16}}}%%\n' + source)
    assert result.has_fatal_error
    assert result.diagnostics[0].code is DiagnosticCode.UNSUPPORTED_DIRECTIVE


def test_block_reserves_tracks_for_returns_to_different_entries() -> None:
    diagram = compile_mermaid(
        "block-beta\ncolumns 3\na b c\nd e f\na --> d\nb --> e\nc --> f\nd --> a\ne --> b\nf --> c"
    )
    assert not diagram.diagnostics
    assert "\n".join(diagram.rows).count("▼") == 6
