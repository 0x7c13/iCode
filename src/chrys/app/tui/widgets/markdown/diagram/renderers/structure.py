# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fixed grid blocks, sharing graph node drawing and orthogonal routing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from chrys.app.tui.widgets.markdown.diagram.canvas import TerminalCanvas
from chrys.app.tui.widgets.markdown.diagram.geometry import node_geometry
from chrys.app.tui.widgets.markdown.diagram.model import CompiledDiagram, DiagramIR, Direction, PlacedNode
from chrys.app.tui.widgets.markdown.diagram.router import detour_channels, draw_edges, draw_nodes, route_edges
from chrys.app.tui.widgets.markdown.diagram.specs.structure import BlockChart

from .common import ChartCanvasLimit, _finish


def compile_block(source: str, ir: DiagramIR, exceeds_budget: Callable[[int, int], bool]) -> CompiledDiagram:
    """Preserve source slot order and spans; route in the gaps between rows."""
    chart = ir.chart
    if not isinstance(chart, BlockChart):
        raise TypeError("not a block chart")
    nodes = {node.node_id: node for node in ir.nodes}
    geometry = {node.node_id: node_geometry(node) for node in ir.nodes}
    slots: list[tuple[str, int, int, int]] = []
    row = column = 0
    for cell in chart.cells:
        if column + cell.span > chart.columns:
            row += 1
            column = 0
        if cell.node_id is not None:
            slots.append((cell.node_id, row, column, cell.span))
        column += cell.span
    parallel_counts = Counter((edge.source, edge.target) for edge in ir.edges)
    parallel_lanes = max(parallel_counts.values(), default=1)
    self_lanes = max((count for (start, end), count in parallel_counts.items() if start == end), default=1)
    ranks = {node_id: rank for node_id, rank, _, _ in slots}
    _, departures, arrivals = detour_channels(ir.edges, ranks)
    backward_lanes = sum(ranks[edge.target] <= ranks[edge.source] for edge in ir.edges if edge.source != edge.target)
    gap_x = 6 + (parallel_lanes - 1) * 2
    gap_y = 6 + 2 * max((sum(edge.source == node.node_id for edge in ir.edges) for node in ir.nodes), default=0)
    gap_y = max(gap_y, 6 + 2 * (max(departures.values(), default=1) + max(arrivals.values(), default=1) - 2))
    slot_width = max((geometry[node_id][0] for node_id, _, _, _ in slots), default=7)
    row_heights = [0] * (row + 1)
    for node_id, rank, _, _ in slots:
        row_heights[rank] = max(row_heights[rank], geometry[node_id][1])
    # Backward/self/parallel routes can use outer lanes to the left of the grid.
    margin = 6 + 2 * max(parallel_lanes - 1, backward_lanes - 1)
    row_tops = [2 + 2 * max(self_lanes - 1, arrivals.get(0, 0) - 1)]
    for height in row_heights[:-1]:
        row_tops.append(row_tops[-1] + max(3, height) + gap_y)
    width = margin + chart.columns * (slot_width + gap_x) + 2 * len(ir.edges) + 4
    height = row_tops[-1] + max(3, row_heights[-1]) + gap_y + 2 * len(ir.edges)
    if exceeds_budget(width, height):
        raise ChartCanvasLimit
    placed: dict[str, PlacedNode] = {}
    for node_id, rank, column, span in slots:
        _, node_height, lines, breaks = geometry[node_id]
        placed[node_id] = PlacedNode(
            nodes[node_id],
            margin + column * (slot_width + gap_x),
            row_tops[rank],
            span * (slot_width + gap_x) - gap_x,
            node_height,
            lines,
            breaks,
        )
    routed = route_edges(ir.edges, placed, ranks, Direction.TOP_DOWN)
    # Check complete routes/labels before touching the sparse canvas.
    for edge in routed:
        if any(point.x < 0 or point.y < 0 for point in edge.points):
            raise ChartCanvasLimit
    canvas = TerminalCanvas()
    draw_edges(canvas, routed)
    draw_nodes(canvas, placed.values())
    return _finish(source, ir, canvas, exceeds_budget)
