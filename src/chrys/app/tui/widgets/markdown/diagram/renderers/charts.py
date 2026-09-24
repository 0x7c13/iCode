# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pie, quadrant, and treemap terminal layouts."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from math import ceil, sqrt

from rich.cells import cell_len

from chrys.app.tui.widgets.markdown.diagram.canvas import TerminalCanvas, wrap_cell_text
from chrys.app.tui.widgets.markdown.diagram.model import (
    CompiledDiagram,
    DiagramIR,
    PieChart,
    Point,
    QuadrantChart,
    TreemapChart,
    TreemapItem,
)

from .common import ChartCanvasLimit, _cell_width, _decimal_text, _draw_title, _finish, _fit, _repeat, _title_lines


def _compile_pie(
    source: str,
    ir: DiagramIR,
    chart: PieChart,
    exceeds_budget: Callable[[int, int], bool],
) -> CompiledDiagram:
    label_width = min(28, max((_cell_width(item.label) for item in chart.slices), default=1))
    wrapped_labels = tuple(wrap_cell_text(item.label, label_width) for item in chart.slices)
    bar_width = 40
    total = sum((item.value for item in chart.slices), Decimal(0))
    metrics: list[str] = []
    for item in chart.slices:
        percent = item.value / total * 100
        percent_text = f"{_decimal_text(percent.quantize(Decimal('0.1')))}%"
        metrics.append(f"{_decimal_text(item.value)} {percent_text}" if chart.show_data else percent_text)
    value_width = max(7, max((cell_len(metric) for metric in metrics), default=1) + 1)
    chart_width = label_width + 2 + value_width + bar_width
    title_lines = _title_lines(chart.title, chart_width)
    chart_height = len(title_lines) + bool(title_lines) + sum(len(lines) + 1 for lines in wrapped_labels) - 1
    if exceeds_budget(chart_width, chart_height):
        raise ChartCanvasLimit
    canvas = TerminalCanvas()
    row = _draw_title(canvas, title_lines, chart_width)
    for item, value_text, label_lines in zip(chart.slices, metrics, wrapped_labels, strict=True):
        ratio = item.value / total
        filled = max(1, min(bar_width, int(ratio * bar_width + Decimal("0.5"))))
        for offset, label_line in enumerate(label_lines):
            canvas.draw_text(0, row + offset, label_line)
        canvas.draw_text(label_width + 2, row, value_text.rjust(value_width - 1))
        canvas.draw_text(label_width + value_width + 2, row, _repeat("█", filled) + _repeat("░", bar_width - filled))
        row += len(label_lines) + 1
    return _finish(source, ir, canvas, exceeds_budget)


def _compile_quadrant(
    source: str,
    ir: DiagramIR,
    chart: QuadrantChart,
    exceeds_budget: Callable[[int, int], bool],
) -> CompiledDiagram:
    width = 68
    height = 20
    left = 2
    title_lines = _title_lines(chart.title, left + width)
    row = len(title_lines) + bool(title_lines)
    top = row + 1
    bottom = top + height - 1
    right = left + width - 1
    middle_x = left + width // 2
    middle_y = top + height // 2
    point_legends = tuple(
        f"{point.label} [{_decimal_text(point.x)}, {_decimal_text(point.y)}]" for point in chart.points
    )
    axis_legends = tuple(
        f'{name}: "{labels[0]}" --> "{labels[1]}"'
        for name, labels, limit in (("x-axis", chart.x_axis, width // 2), ("y-axis", chart.y_axis, 24))
        if any(_cell_width(label) > limit for label in labels)
    )
    estimated_width = max(
        left + width + (25 if any(chart.y_axis) else 0),
        max((left + 2 + _cell_width(text) for text in point_legends), default=0),
        max((left + _cell_width(text) for text in axis_legends), default=0),
        max(
            (left + _cell_width(f"quadrant-{index}: {label}") for index, label in enumerate(chart.quadrants, 1)),
            default=0,
        ),
    )
    estimated_height = (
        bottom + 3 + len(axis_legends) + sum(bool(label) for label in chart.quadrants) + len(chart.points)
    )
    if exceeds_budget(estimated_width, estimated_height):
        raise ChartCanvasLimit
    canvas = TerminalCanvas()
    _draw_title(canvas, title_lines, left + width)
    canvas.draw_box(left, top, width, height)
    canvas.draw_path((Point(middle_x, top), Point(middle_x, bottom)))
    canvas.draw_path((Point(left, middle_y), Point(right, middle_y)))
    quadrant_positions = (
        (middle_x + 1, top + 1),
        (left + 1, top + 1),
        (left + 1, middle_y + 1),
        (middle_x + 1, middle_y + 1),
    )
    point_positions = tuple(
        (
            left + 1 + int(point.x * (width - 3) + Decimal("0.5")),
            bottom - 1 - int(point.y * (height - 3) + Decimal("0.5")),
        )
        for point in chart.points
    )
    caption_legend: list[str] = []
    for index, (label, (x, y)) in enumerate(zip(chart.quadrants, quadrant_positions, strict=True), 1):
        if label:
            fitted = _fit(label, width // 2 - 2)
            # Keep complete captions in the legend if they cannot fit safely.
            if _cell_width(label) > width // 2 - 2 or any(
                py == y and x <= px < x + cell_len(fitted) for px, py in point_positions
            ):
                caption_legend.append(f"quadrant-{index}: {label}")
            else:
                canvas.draw_text(x, y, fitted)
    occupied: set[tuple[int, int]] = set()
    markers: list[str] = []
    for x, y in point_positions:
        marker = "◆" if (x, y) in occupied else "●"
        canvas.put(x, y, marker)
        occupied.add((x, y))
        markers.append(marker)
    if chart.x_axis[0]:
        canvas.draw_text(left, bottom + 1, _fit(chart.x_axis[0], width // 2))
    if chart.x_axis[1]:
        fitted = _fit(chart.x_axis[1], width // 2)
        canvas.draw_text(right - cell_len(fitted) + 1, bottom + 1, fitted)
    if chart.y_axis[1]:
        canvas.draw_text(right + 2, top, _fit(chart.y_axis[1], 24))
    if chart.y_axis[0]:
        canvas.draw_text(right + 2, bottom, _fit(chart.y_axis[0], 24))
    legend_row = bottom + 3
    for caption in (*axis_legends, *caption_legend):
        canvas.draw_text(left, legend_row, caption)
        legend_row += 1
    for legend, marker in zip(point_legends, markers, strict=True):
        canvas.draw_text(left, legend_row, f"{marker} {legend}")
        legend_row += 1
    return _finish(source, ir, canvas, exceeds_budget)


@dataclass(frozen=True, slots=True)
class _Rect:
    x: int
    y: int
    width: int
    height: int


def _partition(items: Sequence[TreemapItem], rect: _Rect) -> list[_Rect]:
    if not items:
        return []
    horizontal = rect.width >= rect.height * 2 or len(items) > 4
    extent = rect.width if horizontal else rect.height
    total = sum((item.value for item in items), Decimal(0))
    cursor = rect.x if horizontal else rect.y
    end = cursor + extent
    rectangles: list[_Rect] = []
    remaining = total
    for index, item in enumerate(items):
        available = end - cursor
        if index == len(items) - 1:
            size = available
        elif available <= 0:
            size = 0
        else:
            reserved = len(items) - index - 1
            proportional = int(Decimal(available) * item.value / remaining + Decimal("0.5"))
            size = max(1, min(max(1, available - reserved), proportional))
        if horizontal:
            rectangles.append(_Rect(cursor, rect.y, size, rect.height))
        else:
            rectangles.append(_Rect(rect.x, cursor, rect.width, size))
        cursor += size
        remaining -= item.value
    return rectangles


def _draw_treemap_items(
    canvas: TerminalCanvas,
    items: Sequence[TreemapItem],
    rect: _Rect,
    prefix: str = "",
) -> None:
    for item, item_rect in zip(items, _partition(items, rect), strict=True):
        path = f"{prefix} / {item.label}" if prefix else item.label
        if item_rect.width >= 2 and item_rect.height >= 2:
            canvas.draw_box(item_rect.x, item_rect.y, item_rect.width, item_rect.height)
        if item_rect.width >= 4 and item_rect.height >= 3:
            label = f"{item.label} {_decimal_text(item.value)}"
            canvas.draw_text(item_rect.x + 1, item_rect.y + 1, _fit(label, item_rect.width - 2))
        if item.children and item_rect.width >= 4 and item_rect.height >= 5:
            inner = _Rect(item_rect.x + 1, item_rect.y + 2, item_rect.width - 2, item_rect.height - 3)
            _draw_treemap_items(canvas, item.children, inner, path)


def _iter_treemap_paths(
    items: Sequence[TreemapItem],
    prefix: str = "",
) -> Iterator[tuple[str, Decimal]]:
    for item in items:
        path = f"{prefix} / {item.label}" if prefix else item.label
        yield path, item.value
        yield from _iter_treemap_paths(item.children, path)


def _treemap_legend_metrics(
    items: Sequence[TreemapItem],
    prefix_width: int = 0,
) -> tuple[int, int]:
    count = 0
    maximum_width = 0
    for item in items:
        path_width = prefix_width + (3 if prefix_width else 0) + _cell_width(item.label)
        line_width = 2 + path_width + 2 + _cell_width(_decimal_text(item.value))
        child_count, child_width = _treemap_legend_metrics(item.children, path_width)
        count += 1 + child_count
        maximum_width = max(maximum_width, line_width, child_width)
    return count, maximum_width


def _treemap_stats(items: Sequence[TreemapItem], depth: int = 1) -> tuple[int, int, int]:
    count = 0
    leaves = 0
    maximum_depth = depth
    for item in items:
        count += 1
        if not item.children:
            leaves += 1
            continue
        child_count, child_leaves, child_depth = _treemap_stats(item.children, depth + 1)
        count += child_count
        leaves += child_leaves
        maximum_depth = max(maximum_depth, child_depth)
    return count, leaves, maximum_depth


def _compile_treemap(
    source: str,
    ir: DiagramIR,
    chart: TreemapChart,
    exceeds_budget: Callable[[int, int], bool],
) -> CompiledDiagram:
    _, leaves, depth = _treemap_stats(chart.items)
    width = min(160, max(64, ceil(sqrt(max(1, leaves))) * 20))
    height = min(64, max(20, ceil(leaves / max(1, width // 16)) * 5 + depth * 2))
    root_rect = _Rect(0, 0, width, height)
    has_drawable_rectangle = any(
        rectangle.width >= 2 and rectangle.height >= 2 for rectangle in _partition(chart.items, root_rect)
    )
    legend_top = height + 1 if has_drawable_rectangle else 0
    legend_rows, legend_width = _treemap_legend_metrics(chart.items)
    if exceeds_budget(max(width if has_drawable_rectangle else 0, legend_width), legend_top + legend_rows):
        raise ChartCanvasLimit
    canvas = TerminalCanvas()
    if has_drawable_rectangle:
        _draw_treemap_items(canvas, chart.items, root_rect)
    row = legend_top
    for path, value in _iter_treemap_paths(chart.items):
        canvas.draw_text(0, row, f"• {path}: {_decimal_text(value)}")
        row += 1
    return _finish(source, ir, canvas, exceeds_budget)
