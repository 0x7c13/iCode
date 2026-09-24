# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cell-indexed connection ink, independent of graph layout and animation frames."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

from rich.cells import cell_len
from rich.style import Style

from chrys.app.tui.widgets.markdown.diagram.canvas import CellStyleSpan
from chrys.app.tui.widgets.markdown.diagram.model import CompiledDiagram, PlacedNode
from chrys.app.tui.widgets.workflow import text


@dataclass(frozen=True, slots=True)
class EdgeCell:
    """One visible route cell and its distance from the source node."""

    x: int
    edge: int
    distance: int


def connection_state(source: str, target: str) -> str:
    """Show endpoint readiness, not a claim that a conditional edge was taken.

    Only connections feeding an executing node animate. A running source alone
    must never light up its as-yet undecided outgoing branches.
    """
    if "skipped" in (source, target):
        return "skipped"
    if source not in {"completed", "running"}:
        return "pending"
    return target if target in text.STATES else "pending"


def edge_cells(diagram: CompiledDiagram, nodes: Mapping[str, PlacedNode]) -> dict[int, tuple[EdgeCell, ...]]:
    """Index once per layout, excluding cards and all labels (including wide text)."""
    occupied: dict[int, list[tuple[int, int]]] = {}
    for box in nodes.values():
        for y in range(box.y, box.y + box.height):
            occupied.setdefault(y, []).append((box.x, box.x + box.width))
    for route in diagram.routed_edges:
        for point, label in (
            (route.label_at, route.edge.label),
            (route.source_label_at, route.edge.source_label),
            (route.target_label_at, route.edge.target_label),
        ):
            if point is not None:
                occupied.setdefault(point.y, []).append((point.x, point.x + cell_len(label)))
    rows: dict[int, list[EdgeCell]] = {}
    for index, route in enumerate(diagram.routed_edges):
        distance = 0
        points = route.points
        for start, end in pairwise(points):
            dx, dy = (end.x > start.x) - (end.x < start.x), (end.y > start.y) - (end.y < start.y)
            length = abs(end.x - start.x) + abs(end.y - start.y)
            for step in range(length):
                x, y = start.x + dx * step, start.y + dy * step
                if not any(left <= x < right for left, right in occupied.get(y, ())):
                    rows.setdefault(y, []).append(EdgeCell(x, index, distance + step))
            distance += length
        if points:
            x, y = points[-1].x, points[-1].y
            if not any(left <= x < right for left, right in occupied.get(y, ())):
                rows.setdefault(y, []).append(EdgeCell(x, index, distance))
    return {y: tuple(cells) for y, cells in rows.items()}


def pulse_strength(distance: int, frame: int) -> int:
    """A compact head and fading tail travel toward the target without blinking."""
    behind = (frame - distance) % 16
    return 3 if behind < 2 else 2 if behind < 4 else 1 if behind < 7 else 0


def ink_spans(cells: Mapping[int, Style]) -> tuple[CellStyleSpan, ...]:
    """Merge adjacent ink so long connections stay cheap to crop and repaint."""
    spans: list[CellStyleSpan] = []
    for x, style in sorted(cells.items()):
        if spans and spans[-1].end == x and spans[-1].style == style:
            previous = spans[-1]
            spans[-1] = CellStyleSpan(previous.start, x + 1, style)
        else:
            spans.append(CellStyleSpan(x, x + 1, style))
    return tuple(spans)
