# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Numeric/categorical XY axes and terminal plot series."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from decimal import Decimal
from itertools import pairwise

from rich.cells import cell_len

from chrys.app.tui.widgets.markdown.diagram.canvas import TerminalCanvas
from chrys.app.tui.widgets.markdown.diagram.model import (
    ChartSeriesKind,
    CompiledDiagram,
    DiagramIR,
    Point,
    XYChart,
    XYSeries,
)

from .common import (
    ChartCanvasLimit,
    _cell_width,
    _center,
    _decimal_text,
    _draw_title,
    _finish,
    _fit,
    _repeat,
    _title_lines,
)

_BAR_GLYPHS = ("█", "▓", "▒", "░")


_LINE_GLYPHS = ("●", "◆", "■", "▲")


def _xy_range(chart: XYChart) -> tuple[Decimal, Decimal]:
    values = [value for series in chart.series for value in series.values]
    low = chart.y_min if chart.y_min is not None else min(values)
    high = chart.y_max if chart.y_max is not None else max(values)
    if any(series.kind is ChartSeriesKind.BAR for series in chart.series):
        if chart.y_min is None:
            low = min(low, Decimal(0))
        if chart.y_max is None:
            high = max(high, Decimal(0))
    if low == high:
        low -= 1
        high += 1
    return low, high


def _xy_labels(chart: XYChart, count: int) -> tuple[str, ...]:
    if chart.x_labels:
        return chart.x_labels
    if chart.x_min is not None and chart.x_max is not None:
        if count == 1:
            return (_decimal_text(chart.x_min),)
        span = chart.x_max - chart.x_min
        # These are computed ticks, not source literals. Keep three significant
        # digits of the step so repeating Decimal divisions remain readable,
        # without collapsing closely spaced values on an offset/small axis.
        step = span / (count - 1)
        quantum = Decimal(1).scaleb(step.adjusted() - 2)
        return tuple(
            _decimal_text(
                chart.x_min
                if index == 0
                else chart.x_max
                if index == count - 1
                else (chart.x_min + step * index).quantize(quantum)
            )
            for index in range(count)
        )
    return tuple(str(index + 1) for index in range(count))


def _compact_axis_labels(labels: tuple[str, ...], limit: int) -> tuple[tuple[str, ...], bool]:
    if all(_cell_width(label) <= limit for label in labels):
        return labels, False
    return tuple(str(index) for index in range(1, len(labels) + 1)), True


def _draw_category_legend(canvas: TerminalCanvas, labels: tuple[str, ...], row: int) -> int:
    for index, label in enumerate(labels, 1):
        canvas.draw_text(0, row, f"{index}. {label}")
        row += 1
    return row


def _series_positions(centers: Sequence[int], count: int, *, numeric_axis: bool) -> list[int]:
    if not numeric_axis or count >= len(centers):
        return list(centers[:count])
    if count == 1 or len(centers) == 1:
        return [centers[0]]
    span = centers[-1] - centers[0]
    return [centers[0] + int(Decimal(span * index) / (count - 1) + Decimal("0.5")) for index in range(count)]


def _plot_y(value: Decimal, low: Decimal, high: Decimal, top: int, height: int) -> int:
    scaled = min(Decimal(1), max(Decimal(0), (value - low) / (high - low)))
    return top + height - int(scaled * height + Decimal("0.5"))


def _plot_x(value: Decimal, low: Decimal, high: Decimal, left: int, width: int) -> int:
    scaled = min(Decimal(1), max(Decimal(0), (value - low) / (high - low)))
    return left + int(scaled * width + Decimal("0.5"))


def _interpolated_line(first: Point, second: Point) -> Iterator[Point]:
    """Rasterize a straight segment using connected horizontal/vertical cells."""
    dx = abs(second.x - first.x)
    dy = abs(second.y - first.y)
    step_x = 1 if second.x > first.x else -1
    step_y = 1 if second.y > first.y else -1
    x, y = first.x, first.y
    error = 0
    yield first
    while x != second.x or y != second.y:
        # Choose the next cardinal cell nearest the ideal line. The error is
        # progress_x * dy - progress_y * dx, so no floating-point work is needed.
        if x != second.x and (y == second.y or abs(error + dy) <= abs(error - dx)):
            x += step_x
            error += dy
        else:
            y += step_y
            error -= dx
        yield Point(x, y)


def _xy_legend_entries(series: Sequence[XYSeries]) -> tuple[str, ...]:
    if len(series) <= 1 and not any(item.name for item in series):
        return ()
    entries: list[str] = []
    bar_index = 0
    line_index = 0
    for index, item in enumerate(series, 1):
        if item.kind is ChartSeriesKind.BAR:
            glyph = _BAR_GLYPHS[bar_index % len(_BAR_GLYPHS)]
            bar_index += 1
        else:
            glyph = _LINE_GLYPHS[line_index % len(_LINE_GLYPHS)]
            line_index += 1
        entries.append(f"{glyph} {item.name or f'{item.kind.value} {index}'}")
    return tuple(entries)


def _draw_xy_legend(canvas: TerminalCanvas, series: Sequence[XYSeries], row: int) -> int:
    entries = _xy_legend_entries(series)
    if not entries:
        return row
    x = 0
    for label in entries:
        canvas.draw_text(x, row, label)
        x += _cell_width(label) + 3
    return row + 1


def _xy_preflight_width(
    chart: XYChart,
    labels: tuple[str, ...],
    *,
    left: int,
    plot_width: int,
    needs_category_legend: bool,
) -> int:
    entries = _xy_legend_entries(chart.series)
    legend_width = sum(_cell_width(entry) for entry in entries) + max(0, len(entries) - 1) * 3
    category_width = (
        max((_cell_width(f"{index}. {label}") for index, label in enumerate(labels, 1)), default=0)
        if needs_category_legend
        else 0
    )
    return max(left + plot_width + 1, left + _cell_width(chart.y_axis_title), legend_width, category_width)


def _compile_xy_vertical(
    source: str,
    ir: DiagramIR,
    chart: XYChart,
    exceeds_budget: Callable[[int, int], bool],
) -> CompiledDiagram:
    count = max(len(series.values) for series in chart.series)
    labels = _xy_labels(chart, count)
    display_labels, needs_category_legend = _compact_axis_labels(labels, 12)
    low, high = _xy_range(chart)
    midpoint = (low + high) / 2
    tick_texts = tuple(_decimal_text(value) for value in (high, midpoint, low))
    left = max(cell_len(text) for text in tick_texts) + 2
    bar_count = sum(series.kind is ChartSeriesKind.BAR for series in chart.series)
    label_cells = max((_cell_width(label) for label in display_labels), default=1)
    slot = max(5, min(14, label_cells + 2), bar_count * 2 + 2)
    plot_width = max(24, count * slot)
    plot_height = 14
    title_lines = _title_lines(chart.title, left + plot_width + 1)
    x_title_lines = _title_lines(chart.x_axis_title, plot_width)
    top = len(title_lines) + bool(title_lines) + bool(chart.y_axis_title)
    estimated_width = _xy_preflight_width(
        chart,
        labels,
        left=left,
        plot_width=plot_width,
        needs_category_legend=needs_category_legend,
    )
    estimated_height = (
        top
        + plot_height
        + 7
        + len(x_title_lines)
        + bool(_xy_legend_entries(chart.series))
        + (len(labels) if needs_category_legend else 0)
    )
    if exceeds_budget(estimated_width, estimated_height):
        raise ChartCanvasLimit
    canvas = TerminalCanvas()
    row = _draw_title(canvas, title_lines, left + plot_width + 1)
    if chart.y_axis_title:
        canvas.draw_text(left, row, chart.y_axis_title)
        row += 1
    top = row
    bottom = top + plot_height
    canvas.draw_path((Point(left, top), Point(left, bottom), Point(left + plot_width, bottom)))
    for tick_index, (tick, text) in enumerate(zip((high, midpoint, low), tick_texts, strict=True)):
        y = _plot_y(tick, low, high, top, plot_height)
        canvas.draw_text(max(0, left - cell_len(text) - 1), y, text)
        if tick_index < 2:
            canvas.draw_text(left + 1, y, _repeat("┄", plot_width))
    baseline = _plot_y(min(high, max(low, Decimal(0))), low, high, top, plot_height)
    centers = [left + slot * index + slot // 2 for index in range(count)]
    numeric_x_axis = not chart.x_labels
    bar_index = 0
    line_index = 0
    line_plots: list[tuple[str, list[Point]]] = []
    plot_series = tuple(series for series in chart.series if series.kind is ChartSeriesKind.BAR) + tuple(
        series for series in chart.series if series.kind is ChartSeriesKind.LINE
    )
    for series in plot_series:
        series_centers = _series_positions(centers, len(series.values), numeric_axis=numeric_x_axis)
        if series.kind is ChartSeriesKind.BAR:
            glyph = _BAR_GLYPHS[bar_index % len(_BAR_GLYPHS)]
            offset = bar_index - (bar_count - 1) // 2
            for x, value in zip(series_centers, series.values, strict=True):
                value_y = _plot_y(value, low, high, top, plot_height)
                for y in range(min(value_y, baseline), max(value_y, baseline) + 1):
                    if y != baseline:
                        canvas.put(x + offset, y, glyph)
            bar_index += 1
            continue
        glyph = _LINE_GLYPHS[line_index % len(_LINE_GLYPHS)]
        points = [
            Point(x, _plot_y(value, low, high, top, plot_height))
            for x, value in zip(series_centers, series.values, strict=True)
        ]
        line_plots.append((glyph, points))
        line_index += 1
    for _, points in line_plots:
        for first, second in pairwise(points):
            canvas.draw_path_on_top(_interpolated_line(first, second))
    for glyph, points in line_plots:
        for point in points:
            canvas.put(point.x, point.y, glyph)
    label_row = bottom + 1
    for x, label in zip(centers, display_labels, strict=False):
        fitted = _fit(label, slot - 1)
        canvas.draw_text(max(left + 1, x - cell_len(fitted) // 2), label_row, fitted)
    next_row = label_row + 1
    for line in x_title_lines:
        _center(canvas, next_row, line, left, plot_width)
        next_row += 1
    next_row = _draw_xy_legend(canvas, chart.series, next_row + 1)
    if needs_category_legend:
        _draw_category_legend(canvas, labels, next_row + 1)
    return _finish(source, ir, canvas, exceeds_budget)


def _compile_xy_horizontal(
    source: str,
    ir: DiagramIR,
    chart: XYChart,
    exceeds_budget: Callable[[int, int], bool],
) -> CompiledDiagram:
    count = max(len(series.values) for series in chart.series)
    labels = _xy_labels(chart, count)
    display_labels, needs_category_legend = _compact_axis_labels(labels, 24)
    low, high = _xy_range(chart)
    label_width = max(
        _cell_width(chart.x_axis_title),
        min(24, max((_cell_width(label) for label in display_labels), default=1)),
    )
    left = label_width + 2
    plot_width = 52
    bar_count = sum(series.kind is ChartSeriesKind.BAR for series in chart.series)
    row_gap = max(3, bar_count + 2)
    title_lines = _title_lines(chart.title, left + plot_width + 1)
    top = len(title_lines) + bool(title_lines) + bool(chart.x_axis_title)
    estimated_width = _xy_preflight_width(
        chart,
        labels,
        left=left,
        plot_width=plot_width,
        needs_category_legend=needs_category_legend,
    )
    estimated_height = (
        top
        + count * row_gap
        + 7
        + bool(_xy_legend_entries(chart.series))
        + (len(labels) if needs_category_legend else 0)
    )
    if exceeds_budget(estimated_width, estimated_height):
        raise ChartCanvasLimit
    canvas = TerminalCanvas()
    row = _draw_title(canvas, title_lines, left + plot_width + 1)
    if chart.x_axis_title:
        canvas.draw_text(label_width - _cell_width(chart.x_axis_title), row, chart.x_axis_title)
        row += 1
    top = row
    bottom = top + count * row_gap
    canvas.draw_path((Point(left, top), Point(left, bottom), Point(left + plot_width, bottom)))
    baseline = _plot_x(min(high, max(low, Decimal(0))), low, high, left, plot_width)
    centers = [top + index * row_gap + row_gap // 2 for index in range(count)]
    numeric_x_axis = not chart.x_labels
    for y, label in zip(centers, display_labels, strict=False):
        fitted = _fit(label, label_width)
        canvas.draw_text(label_width - cell_len(fitted), y, fitted)
    bar_index = 0
    line_index = 0
    line_plots: list[tuple[str, list[Point]]] = []
    plot_series = tuple(series for series in chart.series if series.kind is ChartSeriesKind.BAR) + tuple(
        series for series in chart.series if series.kind is ChartSeriesKind.LINE
    )
    for series in plot_series:
        series_centers = _series_positions(centers, len(series.values), numeric_axis=numeric_x_axis)
        if series.kind is ChartSeriesKind.BAR:
            glyph = _BAR_GLYPHS[bar_index % len(_BAR_GLYPHS)]
            offset = bar_index - (bar_count - 1) // 2
            for y, value in zip(series_centers, series.values, strict=True):
                value_x = _plot_x(value, low, high, left, plot_width)
                start, end = sorted((baseline, value_x))
                if start == end:
                    continue
                canvas.draw_text(start + (start == baseline), y + offset, _repeat(glyph, max(1, end - start)))
            bar_index += 1
            continue
        glyph = _LINE_GLYPHS[line_index % len(_LINE_GLYPHS)]
        points = [
            Point(_plot_x(value, low, high, left, plot_width), y)
            for y, value in zip(series_centers, series.values, strict=True)
        ]
        line_plots.append((glyph, points))
        line_index += 1
    for _, points in line_plots:
        for first, second in pairwise(points):
            canvas.draw_path_on_top(_interpolated_line(first, second))
    for glyph, points in line_plots:
        for point in points:
            canvas.put(point.x, point.y, glyph)
    for value in (low, (low + high) / 2, high):
        text = _decimal_text(value)
        x = _plot_x(value, low, high, left, plot_width)
        canvas.draw_text(max(left, x - cell_len(text) // 2), bottom + 1, text)
    next_row = bottom + 2
    if chart.y_axis_title:
        canvas.draw_text(
            left + max(0, (plot_width - _cell_width(chart.y_axis_title)) // 2), next_row, chart.y_axis_title
        )
        next_row += 1
    next_row = _draw_xy_legend(canvas, chart.series, next_row + 1)
    if needs_category_legend:
        _draw_category_legend(canvas, labels, next_row + 1)
    return _finish(source, ir, canvas, exceeds_budget)


def _compile_xy(
    source: str,
    ir: DiagramIR,
    chart: XYChart,
    exceeds_budget: Callable[[int, int], bool],
) -> CompiledDiagram:
    if chart.horizontal:
        return _compile_xy_horizontal(source, ir, chart, exceeds_budget)
    return _compile_xy_vertical(source, ir, chart, exceeds_budget)
