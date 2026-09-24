# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Parsers for statistical Mermaid chart data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from chrys.app.tui.widgets.markdown.diagram.model import (
    ChartSeriesKind,
    DiagnosticCode,
    DiagramIR,
    DiagramKind,
    Direction,
    PieChart,
    PieSlice,
    QuadrantChart,
    QuadrantPoint,
    TreemapChart,
    TreemapItem,
    XYChart,
    XYSeries,
)

from .common import (
    _NUMBER_PATTERN,
    MAX_EDGES,
    MAX_NODES,
    _Builder,
    _clean_label,
    _find_unquoted_delimiter,
    _parse_list_items,
    _source_lines,
)


def _parse_number_list(raw: str) -> tuple[Decimal, ...] | None:
    values = _parse_list_items(raw)
    if values is None:
        return None
    if any(re.fullmatch(_NUMBER_PATTERN, value) is None for value in values):
        return None
    return tuple(Decimal(value) for value in values)


def _parse_pie(
    lines: list[tuple[int, str]],
    header_index: int,
    show_data: bool,
    inline_title: str = "",
) -> DiagramIR:
    builder = _Builder(DiagramKind.PIE, Direction.LEFT_RIGHT)
    title = _clean_label(inline_title)
    slices: list[PieSlice] = []
    for line, raw in lines[header_index + 1 :]:
        if raw.startswith("%%"):
            continue
        if title_match := re.fullmatch(r"title\s+(.+)", raw, re.IGNORECASE):
            title = _clean_label(title_match.group(1))
            continue
        item = re.fullmatch(rf'"([^"]+)"\s*:\s*({_NUMBER_PATTERN})', raw)
        if item is None:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        value = Decimal(item.group(2))
        if value <= 0:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        if len(slices) >= MAX_NODES:
            builder.error(line, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
            continue
        slices.append(PieSlice(_clean_label(item.group(1)), value))
    if not slices:
        builder.error(lines[header_index][0], DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    return builder.finish(PieChart(title, show_data, tuple(slices)))


def _parse_axis(
    raw: str,
    *,
    categorical: bool,
) -> tuple[str, tuple[str, ...], Decimal | None, Decimal | None] | None:
    delimiter = _find_unquoted_delimiter(raw, ("[", "-->"))
    if delimiter is None:
        return None
    if delimiter < 0:
        title = _clean_label(raw)
        return (title, (), None, None) if title else None
    if raw[delimiter] == "[":
        if not categorical or not raw.endswith("]"):
            return None
        labels = _parse_list_items(raw[delimiter + 1 : -1])
        if labels is None:
            return None
        return _clean_label(raw[:delimiter]), labels, None, None
    numeric = re.fullmatch(
        rf"(?:(.+?)\s+)?({_NUMBER_PATTERN})\s*-->\s*({_NUMBER_PATTERN})",
        raw,
    )
    if numeric is None:
        return None
    title, minimum, maximum = numeric.groups()
    low = Decimal(minimum)
    high = Decimal(maximum)
    if low >= high:
        return None
    return _clean_label(title or ""), (), low, high


def _parse_xychart(
    lines: list[tuple[int, str]],
    header_index: int,
    *,
    horizontal: bool,
) -> DiagramIR:
    builder = _Builder(DiagramKind.XYCHART, Direction.LEFT_RIGHT)
    title = ""
    x_axis: tuple[str, tuple[str, ...], Decimal | None, Decimal | None] = ("", (), None, None)
    y_axis: tuple[str, tuple[str, ...], Decimal | None, Decimal | None] = ("", (), None, None)
    series_with_lines: list[tuple[int, XYSeries]] = []
    point_count = 0
    for line, raw in lines[header_index + 1 :]:
        if raw.startswith("%%"):
            continue
        if title_match := re.fullmatch(r"title\s+(.+)", raw, re.IGNORECASE):
            title = _clean_label(title_match.group(1))
            continue
        if axis_match := re.fullmatch(r"x-axis\s+(.+)", raw, re.IGNORECASE):
            parsed = _parse_axis(axis_match.group(1), categorical=True)
            if parsed is None:
                builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            else:
                x_axis = parsed
            continue
        if axis_match := re.fullmatch(r"y-axis\s+(.+)", raw, re.IGNORECASE):
            parsed = _parse_axis(axis_match.group(1), categorical=False)
            if parsed is None:
                builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            else:
                y_axis = parsed
            continue
        plot = re.match(r"(bar|line)(?=\s|\[|$)", raw, re.IGNORECASE)
        if plot is None:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        body = raw[plot.end() :].strip()
        opening = _find_unquoted_delimiter(body, ("[",))
        if opening is None or opening < 0 or not body.endswith("]"):
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        values = _parse_number_list(body[opening + 1 : -1])
        if values is None:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        point_count += len(values)
        if point_count > MAX_EDGES:
            builder.error(line, DiagnosticCode.EDGE_LIMIT, limit=MAX_EDGES)
            continue
        raw_name = body[:opening].strip()
        name = ""
        if raw_name:
            if (len(raw_name) >= 2 and raw_name[0] == raw_name[-1] and raw_name[0] in {'"', "'"}) or not any(
                char.isspace() for char in raw_name
            ):
                name = _clean_label(raw_name)
            else:
                builder.warning(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
        kind = ChartSeriesKind.BAR if plot.group(1).casefold() == "bar" else ChartSeriesKind.LINE
        series_with_lines.append((line, XYSeries(kind, values, name)))
    if not series_with_lines:
        builder.error(lines[header_index][0], DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    for line, series in series_with_lines:
        if x_axis[1] and len(series.values) != len(x_axis[1]):
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
        # Point-wise clamping changes lines and falsely represents out-of-range
        # values as boundary samples. Until clipping is supported, show source.
        if (
            y_axis[2] is not None
            and y_axis[3] is not None
            and any(value < y_axis[2] or value > y_axis[3] for value in series.values)
        ):
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    chart = XYChart(
        title,
        horizontal,
        x_axis[0],
        x_axis[1],
        x_axis[2],
        x_axis[3],
        y_axis[0],
        y_axis[2],
        y_axis[3],
        tuple(series for _, series in series_with_lines),
    )
    return builder.finish(chart)


def _axis_labels(raw: str) -> tuple[str, str] | None:
    separator = _find_unquoted_delimiter(raw, ("-->",), quote_chars='"')
    if separator is None:
        return None
    if separator < 0:
        return _clean_label(raw, quote_chars='"'), ""
    left, right = raw[:separator], raw[separator + 3 :]
    if _find_unquoted_delimiter(right, ("-->",), quote_chars='"') != -1:
        return None
    return _clean_label(left, quote_chars='"'), _clean_label(right, quote_chars='"')


def _parse_quadrant(lines: list[tuple[int, str]], header_index: int) -> DiagramIR:
    builder = _Builder(DiagramKind.QUADRANT, Direction.LEFT_RIGHT)
    title = ""
    x_axis = ("", "")
    y_axis = ("", "")
    quadrants = ["", "", "", ""]
    points: list[QuadrantPoint] = []
    for line, raw in lines[header_index + 1 :]:
        if raw.startswith("%%"):
            continue
        if title_match := re.fullmatch(r"title\s+(.+)", raw, re.IGNORECASE):
            title = _clean_label(title_match.group(1))
            continue
        if axis_match := re.fullmatch(r"x-axis\s+(.+)", raw, re.IGNORECASE):
            parsed_axis = _axis_labels(axis_match.group(1))
            if parsed_axis is None:
                builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            else:
                x_axis = parsed_axis
            continue
        if axis_match := re.fullmatch(r"y-axis\s+(.+)", raw, re.IGNORECASE):
            parsed_axis = _axis_labels(axis_match.group(1))
            if parsed_axis is None:
                builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            else:
                y_axis = parsed_axis
            continue
        if quadrant := re.fullmatch(r"quadrant-([1-4])\s+(.+)", raw, re.IGNORECASE):
            quadrants[int(quadrant.group(1)) - 1] = _clean_label(quadrant.group(2), quote_chars='"')
            continue
        if raw.casefold().startswith("classdef "):
            builder.warning(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        # Quadrant strings use double quotes. Apostrophes are punctuation,
        # including at the start of a name such as '90s baseline.
        opening = _find_unquoted_delimiter(raw, ("[",), quote_chars='"')
        if opening is None or opening < 0 or not raw[:opening].rstrip().endswith(":"):
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        label = raw[:opening].rstrip()[:-1].strip()
        point = re.fullmatch(
            rf"\[\s*({_NUMBER_PATTERN})\s*,\s*({_NUMBER_PATTERN})\s*\](.*)",
            raw[opening:],
        )
        if point is None:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        x_text, y_text, styles = point.groups()
        class_at = _find_unquoted_delimiter(label, (":::",), quote_chars='"')
        if class_at is None or (class_at >= 0 and re.fullmatch(r':::[^\s:"]+', label[class_at:]) is None):
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        clean_label = _clean_label(label[:class_at] if class_at >= 0 else label, quote_chars='"')
        if not clean_label:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        if class_at >= 0 or styles.strip():
            builder.warning(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
        x = Decimal(x_text)
        y = Decimal(y_text)
        if not (Decimal(0) <= x <= Decimal(1) and Decimal(0) <= y <= Decimal(1)):
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        if len(points) >= MAX_NODES:
            builder.error(line, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
            continue
        points.append(QuadrantPoint(clean_label, x, y))
    quadrant_labels = (quadrants[0], quadrants[1], quadrants[2], quadrants[3])
    if not (points or title or any(x_axis) or any(y_axis) or any(quadrants)):
        builder.error(lines[header_index][0], DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    return builder.finish(QuadrantChart(title, x_axis, y_axis, quadrant_labels, tuple(points)))


@dataclass(slots=True)
class _TreemapDraft:
    label: str
    value: Decimal | None
    line: int
    children: list[_TreemapDraft]


def _parse_treemap(source: str) -> DiagramIR:
    lines = _source_lines(source, preserve_indent=True)
    header_index = next((index for index, (_, line) in enumerate(lines) if not line.lstrip().startswith("%%")), 0)
    builder = _Builder(DiagramKind.TREEMAP, Direction.LEFT_RIGHT)
    roots: list[_TreemapDraft] = []
    stack: list[tuple[int, _TreemapDraft]] = []
    item_count = 0
    for line, raw in lines[header_index + 1 :]:
        content = raw.strip()
        if content.startswith("%%"):
            continue
        if content.casefold().startswith(("classdef ", "class ", "style ")):
            builder.warning(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        item = re.fullmatch(
            rf'"([^"]+)"(?:(:::[^\s:]+))?(?:\s*:\s*({_NUMBER_PATTERN})(?:(:::[^\s:]+))?)?',
            content,
        )
        if item is None:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        label, leading_style, value_text, trailing_style = item.groups()
        value = Decimal(value_text) if value_text is not None else None
        if value is not None and value <= 0:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        if leading_style or trailing_style:
            builder.warning(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
        item_count += 1
        if item_count > MAX_NODES:
            builder.error(line, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
            continue
        indent = len(raw.expandtabs(4)) - len(raw.expandtabs(4).lstrip())
        while stack and indent <= stack[-1][0]:
            stack.pop()
        draft = _TreemapDraft(_clean_label(label), value, line, [])
        if stack:
            parent = stack[-1][1]
            if parent.value is not None:
                builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
                continue
            parent.children.append(draft)
        else:
            roots.append(draft)
        stack.append((indent, draft))

    def freeze(draft: _TreemapDraft) -> TreemapItem | None:
        children = tuple(item for child in draft.children if (item := freeze(child)) is not None)
        if draft.value is None and not children:
            builder.error(draft.line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            return None
        return TreemapItem(draft.label, draft.value or sum((child.value for child in children), Decimal(0)), children)

    items = tuple(item for root in roots if (item := freeze(root)) is not None)
    if not items:
        builder.error(lines[header_index][0], DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    return builder.finish(TreemapChart(items))
