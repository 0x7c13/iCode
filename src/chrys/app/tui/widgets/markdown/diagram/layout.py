# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Deterministic terminal-cell layout for parsed Mermaid diagrams."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import replace

from rich.cells import cell_len

from chrys.foundation.i18n import MessageRef

from .canvas import TerminalCanvas, crop_cell_text, sanitize_terminal_text, wrap_cell_text
from .charts import ChartCanvasLimit, compile_chart
from .diagnostics import render_diagnostic, render_diagnostic_heading, render_more_diagnostics
from .geometry import node_geometry as _node_geometry
from .model import (
    CompiledDiagram,
    Diagnostic,
    DiagnosticCode,
    DiagnosticSeverity,
    DiagramEdge,
    DiagramIR,
    DiagramKind,
    Direction,
    PlacedNode,
    Point,
    RoutedEdge,
)
from .presentation import graph_heading
from .router import arrow_for_points, detour_channels, draw_edges, draw_nodes, route_edges, trunk_tracks

MAX_CANVAS_AXIS = 4096
MAX_CANVAS_CELLS = 1_000_000
_NODE_GAP = 6
_RANK_GAP = 5


def _exceeds_canvas_budget(width: int, height: int) -> bool:
    return width > MAX_CANVAS_AXIS or height > MAX_CANVAS_AXIS or width * height > MAX_CANVAS_CELLS


def _strong_components(ir: DiagramIR) -> tuple[list[list[str]], dict[str, int]]:
    adjacency: dict[str, list[str]] = {node.node_id: [] for node in ir.nodes}
    for edge in ir.edges:
        adjacency[edge.source].append(edge.target)
    index = 0
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node_id: str) -> None:
        nonlocal index
        indexes[node_id] = index
        lowlinks[node_id] = index
        index += 1
        stack.append(node_id)
        on_stack.add(node_id)
        for target in adjacency[node_id]:
            if target not in indexes:
                visit(target)
                lowlinks[node_id] = min(lowlinks[node_id], lowlinks[target])
            elif target in on_stack:
                lowlinks[node_id] = min(lowlinks[node_id], indexes[target])
        if lowlinks[node_id] != indexes[node_id]:
            return
        component: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node_id:
                break
        components.append(component)

    for node in ir.nodes:
        if node.node_id not in indexes:
            visit(node.node_id)
    component_for = {node_id: component for component, members in enumerate(components) for node_id in members}
    return components, component_for


def _assign_ranks(ir: DiagramIR, *, reverse_components: bool = False) -> dict[str, int]:
    ir = replace(ir, edges=tuple(edge for edge in ir.edges if edge.constrains_rank))
    components, component_for = _strong_components(ir)
    source_order = {node.node_id: index for index, node in enumerate(ir.nodes)}
    for component in components:
        component.sort(key=source_order.__getitem__, reverse=reverse_components)
    successors: dict[int, set[int]] = defaultdict(set)
    indegrees = [0] * len(components)
    for edge in ir.edges:
        source = component_for[edge.source]
        target = component_for[edge.target]
        if source == target or target in successors[source]:
            continue
        successors[source].add(target)
        indegrees[target] += 1
    ranks = [0] * len(components)
    ready = deque(index for index, indegree in enumerate(indegrees) if indegree == 0)
    while ready:
        source = ready.popleft()
        for target in sorted(successors[source]):
            ranks[target] = max(ranks[target], ranks[source] + len(components[source]))
            indegrees[target] -= 1
            if indegrees[target] == 0:
                ready.append(target)
    member_offsets = {node_id: offset for component in components for offset, node_id in enumerate(component)}
    return {node.node_id: ranks[component_for[node.node_id]] + member_offsets[node.node_id] for node in ir.nodes}


def _ordered_ranks(ir: DiagramIR, ranks: dict[str, int]) -> list[list[str]]:
    source_order = {node.node_id: index for index, node in enumerate(ir.nodes)}
    groups: dict[int, list[str]] = defaultdict(list)
    for node in ir.nodes:
        groups[ranks[node.node_id]].append(node.node_id)
    neighbors_before: dict[str, list[str]] = defaultdict(list)
    neighbors_after: dict[str, list[str]] = defaultdict(list)
    for edge in ir.edges:
        neighbors_after[edge.source].append(edge.target)
        neighbors_before[edge.target].append(edge.source)
    ordered = [groups[rank] for rank in range(max(groups, default=0) + 1)]
    for _ in range(2):
        positions = {node_id: index for group in ordered for index, node_id in enumerate(group)}
        for rank in range(1, len(ordered)):
            ordered[rank].sort(
                key=lambda node_id: (
                    sum(positions[value] for value in neighbors_before[node_id]) / len(neighbors_before[node_id])
                    if neighbors_before[node_id]
                    else source_order[node_id],
                    source_order[node_id],
                )
            )
        positions = {node_id: index for group in ordered for index, node_id in enumerate(group)}
        for rank in range(len(ordered) - 2, -1, -1):
            ordered[rank].sort(
                key=lambda node_id: (
                    sum(positions[value] for value in neighbors_after[node_id]) / len(neighbors_after[node_id])
                    if neighbors_after[node_id]
                    else source_order[node_id],
                    source_order[node_id],
                )
            )
    return ordered


def _with_trunk_room(
    place: Callable[[], dict[str, PlacedNode]],
    rank_gaps: list[int],
    edges: tuple[DiagramEdge, ...],
    ranks: dict[str, int],
    direction: Direction,
) -> dict[str, PlacedNode]:
    """Place the nodes, then widen each gap that needs more than one trunk and place them again.

    A trunk's track depends only on positions along the rank, which the gaps do not move, so the
    router finds the same tracks in the final placement.
    """
    placed = place()
    _, trunks = trunk_tracks(edges, placed, ranks, direction)
    if max(trunks.values(), default=1) > 1:
        for rank, count in trunks.items():
            rank_gaps[rank] += 2 * (count - 1)
        placed = place()
    return placed


def _place_top_down(ir: DiagramIR, ordered: list[list[str]]) -> dict[str, PlacedNode]:
    nodes = {node.node_id: node for node in ir.nodes}
    geometry = {node_id: _node_geometry(node) for node_id, node in nodes.items()}
    rank_for = {node_id: rank for rank, group in enumerate(ordered) for node_id in group}
    parallel_counts = Counter((edge.source, edge.target) for edge in ir.edges)
    backward_label_counts = Counter(
        edge.target for edge in ir.edges if edge.label and rank_for[edge.target] < rank_for[edge.source]
    )
    self_lanes = max((count for (source, target), count in parallel_counts.items() if source == target), default=1)
    backward_lanes = sum(
        count for (source, target), count in parallel_counts.items() if rank_for[target] < rank_for[source]
    )
    rank_gaps = [_RANK_GAP] * max(0, len(ordered) - 1)
    for (source, target), count in parallel_counts.items():
        source_rank = rank_for[source]
        target_rank = rank_for[target]
        if target_rank < source_rank:
            boundary_rank = target_rank
        elif target_rank == source_rank + 1:
            boundary_rank = source_rank
        else:
            continue
        rank_gaps[boundary_rank] = max(rank_gaps[boundary_rank], _RANK_GAP + max(0, count - 1) * 2)
    for target, count in backward_label_counts.items():
        rank_gaps[rank_for[target]] = max(
            rank_gaps[rank_for[target]],
            _RANK_GAP + max(0, count - 1) * 2,
        )
    _, departures, arrivals = detour_channels(ir.edges, rank_for)
    for rank in range(len(rank_gaps)):
        rank_gaps[rank] = max(
            rank_gaps[rank],
            _RANK_GAP + 2 * (max(0, departures.get(rank, 0) - 1) + max(0, arrivals.get(rank + 1, 0) - 1)),
        )
    label_widths: defaultdict[str, int] = defaultdict(int)
    for edge in ir.edges:
        if edge.label and rank_for[edge.target] == rank_for[edge.source] + 1:
            width = cell_len(sanitize_terminal_text(edge.label))
            for node_id in (edge.source, edge.target):
                label_widths[node_id] = max(label_widths[node_id], width)
    node_gaps = {
        node_id: max(_NODE_GAP, label_widths[node_id] - node_geometry[0] // 2 + 2)
        for node_id, node_geometry in geometry.items()
    }
    rank_widths = [
        sum(geometry[node_id][0] for node_id in group) + sum(node_gaps[node_id] for node_id in group[:-1])
        for group in ordered
    ]
    content_width = max(rank_widths, default=1)

    def place() -> dict[str, PlacedNode]:
        placed: dict[str, PlacedNode] = {}
        y = 2 + max(0, self_lanes - 1, arrivals.get(0, 0) - 1) * 2
        for rank, group in enumerate(ordered):
            rank_height = max((geometry[node_id][1] for node_id in group), default=1)
            x = 10 + max(0, backward_lanes - 1) * 2 + (content_width - rank_widths[rank]) // 2
            for node_id in group:
                width, height, lines, breaks = geometry[node_id]
                placed[node_id] = PlacedNode(nodes[node_id], x, y, width, height, lines, breaks)
                x += width + node_gaps[node_id]
            y += rank_height
            if rank < len(rank_gaps):
                y += rank_gaps[rank]
        return placed

    return _with_trunk_room(place, rank_gaps, ir.edges, rank_for, Direction.TOP_DOWN)


def _place_left_right(ir: DiagramIR, ordered: list[list[str]]) -> dict[str, PlacedNode]:
    nodes = {node.node_id: node for node in ir.nodes}
    geometry = {node_id: _node_geometry(node) for node_id, node in nodes.items()}
    rank_for = {node_id: rank for rank, group in enumerate(ordered) for node_id in group}
    parallel_counts = Counter((edge.source, edge.target) for edge in ir.edges)
    backward_lanes = sum(
        count for (source, target), count in parallel_counts.items() if rank_for[target] < rank_for[source]
    )
    rank_gaps = [_RANK_GAP + 2] * max(0, len(ordered) - 1)
    successors: defaultdict[str, set[str]] = defaultdict(set)
    predecessors: defaultdict[str, set[str]] = defaultdict(set)
    for edge in ir.edges:
        successors[edge.source].add(edge.target)
        predecessors[edge.target].add(edge.source)
    for edge in ir.edges:
        source_rank = rank_for[edge.source]
        if edge.label and rank_for[edge.target] == source_rank + 1:
            # Branch labels sit on their own side of the shared vertical stem.
            branching = len(successors[edge.source]) > 1 or len(predecessors[edge.target]) > 1
            rank_gaps[source_rank] = max(
                rank_gaps[source_rank],
                cell_len(sanitize_terminal_text(edge.label)) * (2 if branching else 1) + 4,
            )
    _, departures, arrivals = detour_channels(ir.edges, rank_for)
    for rank in range(len(rank_gaps)):
        rank_gaps[rank] += 2 * (max(0, departures.get(rank, 0) - 1) + max(0, arrivals.get(rank + 1, 0) - 1))
    rank_heights = [
        sum(geometry[node_id][1] for node_id in group) + _NODE_GAP * max(0, len(group) - 1) for group in ordered
    ]
    content_height = max(rank_heights, default=1)

    def place() -> dict[str, PlacedNode]:
        placed: dict[str, PlacedNode] = {}
        x = 2 + max(0, arrivals.get(0, 0) - 1) * 2
        for rank, group in enumerate(ordered):
            rank_width = max((geometry[node_id][0] for node_id in group), default=1)
            y = 8 + max(0, backward_lanes - 1) * 2 + (content_height - rank_heights[rank]) // 2
            for node_id in group:
                width, height, lines, breaks = geometry[node_id]
                placed[node_id] = PlacedNode(nodes[node_id], x, y, width, height, lines, breaks)
                y += height + _NODE_GAP
            x += rank_width
            if rank < len(rank_gaps):
                x += rank_gaps[rank]
        return placed

    return _with_trunk_room(place, rank_gaps, ir.edges, rank_for, Direction.LEFT_RIGHT)


def _diagnostic_diagram(
    source: str,
    kind: DiagramKind,
    diagnostics: tuple[Diagnostic, ...],
    render_message: Callable[[MessageRef], str] | None,
) -> CompiledDiagram:
    shown = diagnostics[:8]
    message_lines: list[str] = [render_diagnostic_heading(render_message)]
    for diagnostic in shown:
        message_lines.extend(wrap_cell_text(render_diagnostic(diagnostic, render_message), 84))
    if len(diagnostics) > len(shown):
        message_lines.append(render_more_diagnostics(len(diagnostics) - len(shown), render_message))
    width = min(90, max(24, max(cell_len(line) for line in message_lines) + 4))
    height = len(message_lines) + 2
    canvas = TerminalCanvas()
    canvas.draw_box(0, 0, width, height)
    for y, line in enumerate(message_lines, 1):
        canvas.draw_text(2, y, line)
    return CompiledDiagram(source, kind, width, height, canvas.rows(width, height), diagnostics)


def _compile_graph(
    source: str,
    ir: DiagramIR,
    render_message: Callable[[MessageRef], str] | None,
    geometry: dict[str, PlacedNode],
) -> CompiledDiagram:
    reverse = ir.direction in {Direction.BOTTOM_UP, Direction.RIGHT_LEFT}
    axis_direction = (
        Direction.TOP_DOWN if ir.direction in {Direction.TOP_DOWN, Direction.BOTTOM_UP} else Direction.LEFT_RIGHT
    )
    layout_ir = ir
    if reverse:
        layout_ir = replace(
            ir,
            direction=axis_direction,
            edges=tuple(
                replace(
                    edge,
                    source=edge.target,
                    target=edge.source,
                    directed=False,
                    source_marker=edge.target_marker or ("▶" if edge.directed else ""),
                    target_marker=edge.source_marker,
                    source_label=edge.target_label,
                    target_label=edge.source_label,
                )
                for edge in ir.edges
            ),
        )
    # Edge reversal preserves component membership; reverse its internal order too.
    ranks = _assign_ranks(layout_ir, reverse_components=reverse)
    ordered = _ordered_ranks(layout_ir, ranks)
    placed = (
        _place_top_down(layout_ir, ordered)
        if axis_direction is Direction.TOP_DOWN
        else _place_left_right(layout_ir, ordered)
    )
    placed_width = max((node.x + node.width for node in placed.values()), default=0)
    placed_height = max((node.y + node.height for node in placed.values()), default=0)
    if _exceeds_canvas_budget(placed_width, placed_height):
        diagnostic = Diagnostic(0, DiagnosticCode.CANVAS_LIMIT, (("limit", MAX_CANVAS_AXIS),))
        return _diagnostic_diagram(source, ir.kind, (*ir.diagnostics, diagnostic), render_message)
    routed = route_edges(layout_ir.edges, placed, ranks, axis_direction)
    canvas = TerminalCanvas()
    draw_edges(canvas, routed)
    draw_nodes(canvas, placed.values())
    width = canvas.natural_width
    height = canvas.natural_height
    if _exceeds_canvas_budget(width, height):
        diagnostic = Diagnostic(0, DiagnosticCode.CANVAS_LIMIT, (("limit", MAX_CANVAS_AXIS),))
        return _diagnostic_diagram(source, ir.kind, (*ir.diagnostics, diagnostic), render_message)
    rows = canvas.rows(width, height)
    occupied = [y for y, row in enumerate(rows) if row.strip()]
    top, bottom = occupied[0], occupied[-1] + 1
    left = min(len(rows[y]) - len(rows[y].lstrip(" ")) for y in occupied)
    right = max(cell_len(rows[y]) for y in occupied)
    geometry.update({key: replace(box, x=box.x - left, y=box.y - top) for key, box in placed.items()})
    return CompiledDiagram(
        source,
        ir.kind,
        right - left,
        bottom - top,
        tuple(crop_cell_text(row, left, right - left).rstrip() for row in rows[top:bottom]),
        ir.diagnostics,
        tuple(route.translated(-left, -top) for route in routed),
    )


def _sequence_geometry(ir: DiagramIR) -> tuple[dict[str, PlacedNode], dict[str, int], int, int]:
    longest_message = max((cell_len(sanitize_terminal_text(edge.label)) for edge in ir.edges), default=0)
    gap = min(48, max(12, longest_message + 4))
    placed: dict[str, PlacedNode] = {}
    centers: dict[str, int] = {}
    x = 2
    for node in ir.nodes:
        width, height, lines, breaks = _node_geometry(node)
        placed[node.node_id] = PlacedNode(node, x, 1, width, height, lines, breaks)
        centers[node.node_id] = x + width // 2
        x += width + gap
    height = 8 + len(ir.edges) * 4
    return placed, centers, max(1, x - gap + 2), height


def _compile_sequence(
    source: str,
    ir: DiagramIR,
    render_message: Callable[[MessageRef], str] | None,
) -> CompiledDiagram:
    placed, centers, width, height = _sequence_geometry(ir)
    if _exceeds_canvas_budget(width, height):
        diagnostic = Diagnostic(0, DiagnosticCode.CANVAS_LIMIT, (("limit", MAX_CANVAS_AXIS),))
        return _diagnostic_diagram(source, ir.kind, (*ir.diagnostics, diagnostic), render_message)
    canvas = TerminalCanvas()
    draw_nodes(canvas, placed.values())
    lifeline_top = max((node.y + node.height for node in placed.values()), default=3)
    lifeline_bottom = height - 2
    for center in centers.values():
        canvas.draw_path((Point(center, lifeline_top), Point(center, lifeline_bottom)))
        for y in range(lifeline_top + 1, lifeline_bottom, 2):
            canvas.put(center, y, "┊")
    routed: list[RoutedEdge] = []
    for index, edge in enumerate(ir.edges):
        y = lifeline_top + 2 + index * 4
        source_x = centers[edge.source]
        target_x = centers[edge.target]
        if source_x == target_x:
            points = (Point(source_x, y), Point(source_x + 6, y), Point(source_x + 6, y + 2), Point(source_x, y + 2))
        else:
            step = 1 if target_x > source_x else -1
            points = (Point(source_x + step, y), Point(target_x, y))
        label_x = min(source_x, target_x) + 1 if source_x != target_x else source_x + 1
        marker = edge.target_marker or (arrow_for_points(points) if edge.directed else "")
        routed.append(
            RoutedEdge(
                edge,
                points,
                Point(label_x, y - 1) if edge.label else None,
                points[-1] if marker else None,
                marker,
                source_marker_at=points[0] if edge.source_marker else None,
            )
        )
    draw_edges(canvas, routed)
    width = max(width, canvas.natural_width)
    height = max(height, canvas.natural_height)
    if _exceeds_canvas_budget(width, height):
        diagnostic = Diagnostic(0, DiagnosticCode.CANVAS_LIMIT, (("limit", MAX_CANVAS_AXIS),))
        return _diagnostic_diagram(source, ir.kind, (*ir.diagnostics, diagnostic), render_message)
    return CompiledDiagram(source, ir.kind, width, height, canvas.rows(width, height), ir.diagnostics)


def compile_ir(
    source: str,
    ir: DiagramIR,
    *,
    render_message: Callable[[MessageRef], str] | None = None,
) -> CompiledDiagram:
    """Lay out a parsed diagram or return its bounded diagnostic canvas."""
    errors = tuple(diagnostic for diagnostic in ir.diagnostics if diagnostic.severity is DiagnosticSeverity.ERROR)
    if ir.has_fatal_error or errors:
        return _diagnostic_diagram(source, ir.kind, ir.diagnostics, render_message)
    if ir.chart is not None:
        try:
            return compile_chart(source, ir, _exceeds_canvas_budget)
        except ChartCanvasLimit:
            diagnostic = Diagnostic(0, DiagnosticCode.CANVAS_LIMIT, (("limit", MAX_CANVAS_AXIS),))
            return _diagnostic_diagram(source, ir.kind, (*ir.diagnostics, diagnostic), render_message)
    if not ir.nodes:
        diagnostic = Diagnostic(0, DiagnosticCode.NO_NODES)
        return _diagnostic_diagram(source, ir.kind, (*ir.diagnostics, diagnostic), render_message)
    if ir.kind is DiagramKind.SEQUENCE:
        return _compile_sequence(source, ir, render_message)
    return compile_ir_with_geometry(source, ir, render_message=render_message)[0]


def compile_ir_with_geometry(
    source: str, ir: DiagramIR, *, render_message: Callable[[MessageRef], str] | None = None
) -> tuple[CompiledDiagram, dict[str, PlacedNode]]:
    """Compile graph IR and retain node boxes in the resulting canvas coordinates."""
    geometry: dict[str, PlacedNode] = {}
    if (
        ir.has_fatal_error
        or not ir.nodes
        or ir.chart is not None
        or ir.kind is DiagramKind.SEQUENCE
        or any(item.severity is DiagnosticSeverity.ERROR for item in ir.diagnostics)
    ):
        return compile_ir(source, ir, render_message=render_message), geometry
    compiled = _compile_graph(source, ir, render_message, geometry)
    if not any(item.severity is DiagnosticSeverity.ERROR for item in compiled.diagnostics) and (
        heading := graph_heading(ir, render_message)
    ):
        width = max(compiled.width, *(cell_len(line) for line in heading))
        height = compiled.height + len(heading)
        if _exceeds_canvas_budget(width, height):
            diagnostic = Diagnostic(0, DiagnosticCode.CANVAS_LIMIT, (("limit", MAX_CANVAS_AXIS),))
            return _diagnostic_diagram(source, ir.kind, (*ir.diagnostics, diagnostic), render_message), {}
        compiled = replace(
            compiled,
            width=width,
            height=height,
            rows=(*heading, *compiled.rows),
            routed_edges=tuple(edge.translated(0, len(heading)) for edge in compiled.routed_edges),
        )
        geometry = {key: replace(box, y=box.y + len(heading)) for key, box in geometry.items()}
    return compiled, geometry
