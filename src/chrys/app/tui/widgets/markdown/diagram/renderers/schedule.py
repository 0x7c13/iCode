# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fixed-budget date bars, commit lanes, and exact-width packet fields."""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from math import ceil

from chrys.app.tui.widgets.markdown.diagram.canvas import TerminalCanvas, wrap_cell_text
from chrys.app.tui.widgets.markdown.diagram.model import CompiledDiagram, DiagramIR, Point
from chrys.app.tui.widgets.markdown.diagram.specs.schedule import GanttChart, GitChart, PacketChart

from .common import ChartCanvasLimit, _cell_width, _center, _finish


def _compile_gantt(
    source: str, ir: DiagramIR, chart: GanttChart, exceeds_budget: Callable[[int, int], bool]
) -> CompiledDiagram:
    left = 34
    plot_width = 64
    width = left + plot_width + 25
    title_lines = wrap_cell_text(chart.title, width) if chart.title else ()
    first = min(task.start for task in chart.tasks)
    last = max(task.end for task in chart.tasks)
    span = max(1, (last - first).days)
    labels = [
        wrap_cell_text(task.label + (f" [{', '.join(task.statuses)}]" if task.statuses else ""), left - 2)
        for task in chart.tasks
    ]
    row = len(title_lines) + (1 if title_lines else 0) + 2
    task_rows: list[int] = []
    section_rows: list[tuple[int, tuple[str, ...]]] = []
    previous_section: str | None = None
    for task, lines in zip(chart.tasks, labels, strict=True):
        if task.section != previous_section and task.section:
            section_lines = wrap_cell_text(task.section, width)
            section_rows.append((row, section_lines))
            row += len(section_lines) + 1
        previous_section = task.section
        task_rows.append(row)
        row += max(1, len(lines)) + 1
    if exceeds_budget(width, row):
        raise ChartCanvasLimit
    canvas = TerminalCanvas()
    for y, line in enumerate(title_lines):
        canvas.draw_text(0, y, line)
    ruler = len(title_lines) + (1 if title_lines else 0)
    canvas.draw_text(left, ruler, first.isoformat())
    canvas.draw_text(left + plot_width - 9, ruler, last.isoformat())
    middle = first + timedelta(days=span // 2)
    _center(canvas, ruler, middle.isoformat(), left + 20, 24)
    canvas.draw_path((Point(left, ruler + 1), Point(left + plot_width, ruler + 1)))
    for y, lines in section_rows:
        for offset, line in enumerate(lines):
            canvas.draw_text(0, y + offset, line)
    for task, label_lines, y in zip(chart.tasks, labels, task_rows, strict=True):
        for offset, line in enumerate(label_lines):
            canvas.draw_text(0, y + offset, line)
        start_offset = (task.start - first).days
        end_offset = (task.end - first).days
        if "milestone" in task.statuses:
            x = left + round((start_offset + end_offset) * plot_width / (2 * span))
            canvas.put(x, y, "◆")
        else:
            start_x = left + start_offset * plot_width // span
            end_x = left + ceil(end_offset * plot_width / span)
            canvas.draw_text(start_x, y, "█" * max(1, end_x - start_x))
        canvas.draw_text(left + plot_width + 2, y, f"{task.start.isoformat()} → {task.end.isoformat()}")
    return _finish(source, ir, canvas, exceeds_budget)


def _git_legend(chart: GitChart) -> tuple[str, ...]:
    lines: list[str] = []
    for index, commit in enumerate(chart.commits, 1):
        metadata = f"{index}. {commit.commit_id} [{commit.branch}]"
        if commit.parents:
            metadata += " ← " + ", ".join(str(parent + 1) for parent in commit.parents)
        if commit.kind != "NORMAL":
            metadata += f" [{commit.kind}]"
        if commit.tags:
            metadata += " [tag: " + ", ".join(commit.tags) + "]"
        lines.extend(wrap_cell_text(metadata, 110))
    return tuple(lines)


def _compile_git(
    source: str, ir: DiagramIR, chart: GitChart, exceeds_budget: Callable[[int, int], bool]
) -> CompiledDiagram:
    branch_lines = [wrap_cell_text(branch, 22) for branch in chart.branches]
    legend = _git_legend(chart)
    branch_indexes = {branch: index for index, branch in enumerate(chart.branches)}
    horizontal = chart.orientation == "LR"
    if horizontal:
        left = max((_cell_width(line) for lines in branch_lines for line in lines), default=1) + 3
        lanes: list[int] = []
        y = 1
        for lines in branch_lines:
            lanes.append(y)
            y += max(3, len(lines) + 1)
        graph_width = left + max(0, len(chart.commits) - 1) * 6 + 4
        graph_height = y
        points = [
            Point(left + index * 6, lanes[branch_indexes[commit.branch]]) for index, commit in enumerate(chart.commits)
        ]
    else:
        top = max((len(lines) for lines in branch_lines), default=1) + 2
        graph_width = len(chart.branches) * 26
        graph_height = top + len(chart.commits) * 3
        points = [
            Point(
                branch_indexes[commit.branch] * 26 + 12,
                top + (len(chart.commits) - index - 1 if chart.orientation == "BT" else index) * 3,
            )
            for index, commit in enumerate(chart.commits)
        ]
    width = max(graph_width, max((_cell_width(line) for line in legend), default=1))
    height = graph_height + 1 + len(legend)
    if exceeds_budget(width, height):
        raise ChartCanvasLimit
    canvas = TerminalCanvas()
    for branch_index, lines in enumerate(branch_lines):
        for offset, line in enumerate(lines):
            if horizontal:
                canvas.draw_text(0, lanes[branch_index] + offset, line)
            else:
                _center(canvas, offset, line, branch_index * 26, 24)
    for commit, point in zip(chart.commits, points, strict=True):
        for parent_index in commit.parents:
            parent = points[parent_index]
            if horizontal:
                bend = point.x - 2
                path = (parent, Point(bend, parent.y), Point(bend, point.y), point)
            else:
                bend = point.y + (1 if chart.orientation == "BT" else -1)
                path = (parent, Point(parent.x, bend), Point(point.x, bend), point)
            canvas.draw_path(path)
    for index, (commit, point) in enumerate(zip(chart.commits, points, strict=True), 1):
        marker = "◎" if len(commit.parents) > 1 else "●"
        if commit.kind == "REVERSE":
            marker = "x"
        elif commit.kind == "HIGHLIGHT":
            marker = "◆"
        canvas.put(point.x, point.y, marker)
        canvas.draw_text(point.x + (0 if horizontal else 2), point.y + (1 if horizontal else 0), str(index))
    for offset, line in enumerate(legend):
        canvas.draw_text(0, graph_height + 1 + offset, line)
    return _finish(source, ir, canvas, exceeds_budget)


def _compile_packet(
    source: str, ir: DiagramIR, chart: PacketChart, exceeds_budget: Callable[[int, int], bool]
) -> CompiledDiagram:
    left = 13
    bit_width = 3
    width = left + 32 * bit_width + 1
    title_lines = wrap_cell_text(chart.title, width) if chart.title else ()
    top = len(title_lines) + (1 if title_lines else 0)
    row_count = chart.fields[-1].end // 32 + 1
    legend: list[str] = []
    for index, field in enumerate(chart.fields, 1):
        legend.extend(wrap_cell_text(f"{index}. [{field.start}-{field.end}] {field.label}", width))
    height = top + 1 + row_count * 4 + 1 + len(legend)
    if exceeds_budget(width, height):
        raise ChartCanvasLimit
    canvas = TerminalCanvas()
    for y, line in enumerate(title_lines):
        canvas.draw_text(0, y, line)
    for bit in range(32):
        canvas.draw_text(left + bit * bit_width, top, str(bit).rjust(2))
    for row in range(row_count):
        y = top + 1 + row * 4
        canvas.draw_text(0, y + 1, f"{row * 32}-{row * 32 + 31}")
        canvas.draw_box(left, y, 32 * bit_width + 1, 3)
    for index, field in enumerate(chart.fields, 1):
        for row in range(field.start // 32, field.end // 32 + 1):
            start = max(field.start, row * 32) % 32
            end = min(field.end, row * 32 + 31) % 32 + 1
            x = left + start * bit_width
            field_width = (end - start) * bit_width + 1
            y = top + 1 + row * 4
            canvas.draw_box(x, y, field_width, 3)
            label = field.label if _cell_width(field.label) <= field_width - 2 else f"#{index}"
            if _cell_width(label) > field_width - 2:
                label = "·"
            _center(canvas, y + 1, label, x + 1, field_width - 2)
    for offset, line in enumerate(legend):
        canvas.draw_text(0, top + 1 + row_count * 4 + 1 + offset, line)
    return _finish(source, ir, canvas, exceeds_budget)


def compile_schedule(source: str, ir: DiagramIR, exceeds_budget: Callable[[int, int], bool]) -> CompiledDiagram:
    """Dispatch only schedule-family typed data, independent of source syntax."""
    chart = ir.chart
    if isinstance(chart, GanttChart):
        return _compile_gantt(source, ir, chart, exceeds_budget)
    if isinstance(chart, GitChart):
        return _compile_git(source, ir, chart, exceeds_budget)
    if isinstance(chart, PacketChart):
        return _compile_packet(source, ir, chart, exceeds_budget)
    raise TypeError("not a schedule-family diagram")
