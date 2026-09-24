# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Wrapped terminal cards for journey, timeline, and read-only kanban boards."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from chrys.app.tui.widgets.markdown.diagram.canvas import TerminalCanvas, wrap_cell_text
from chrys.app.tui.widgets.markdown.diagram.model import CompiledDiagram, DiagramIR, Point
from chrys.app.tui.widgets.markdown.diagram.specs.planning import JourneyChart, KanbanChart, TimelineChart

from .common import ChartCanvasLimit, _center, _finish

_CARD_WIDTH = 28
_CARD_GAP = 3
_TEXT_WIDTH = _CARD_WIDTH - 4


@dataclass(frozen=True, slots=True)
class _Card:
    heading: tuple[str, ...]
    body: tuple[str, ...]

    @property
    def height(self) -> int:
        return len(self.heading) + len(self.body) + (3 if self.body else 2)


def _wrapped(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(line for value in values for line in wrap_cell_text(value, _TEXT_WIDTH))


def _draw_card(canvas: TerminalCanvas, x: int, y: int, card: _Card) -> None:
    canvas.draw_box(x, y, _CARD_WIDTH, card.height)
    row = y + 1
    for line in card.heading:
        canvas.draw_text(x + 2, row, line)
        row += 1
    if card.body:
        canvas.draw_horizontal(x, x + _CARD_WIDTH - 1, row)
        row += 1
        for line in card.body:
            canvas.draw_text(x + 2, row, line)
            row += 1


def _compile_sequence_cards(
    source: str,
    ir: DiagramIR,
    title: str,
    cards: tuple[_Card, ...],
    *,
    vertical: bool,
    exceeds_budget: Callable[[int, int], bool],
) -> CompiledDiagram:
    width = _CARD_WIDTH if vertical else len(cards) * (_CARD_WIDTH + _CARD_GAP) - _CARD_GAP
    title_lines = wrap_cell_text(title, min(width, 80)) if title else ()
    top = len(title_lines) + (1 if title_lines else 0)
    height = top + (
        sum(card.height for card in cards) + _CARD_GAP * (len(cards) - 1)
        if vertical
        else max(card.height for card in cards)
    )
    if exceeds_budget(width, height):
        raise ChartCanvasLimit
    canvas = TerminalCanvas()
    for row, line in enumerate(title_lines):
        _center(canvas, row, line, 0, width)
    x, y = 0, top
    for index, card in enumerate(cards):
        _draw_card(canvas, x, y, card)
        if index < len(cards) - 1:
            if vertical:
                start = Point(_CARD_WIDTH // 2, y + card.height)
                end = Point(start.x, start.y + _CARD_GAP - 1)
                canvas.draw_path((start, end))
                canvas.put(end.x, end.y, "▼")
            else:
                start = Point(x + _CARD_WIDTH, top + 1)
                end = Point(start.x + _CARD_GAP - 1, start.y)
                canvas.draw_path((start, end))
                canvas.put(end.x, end.y, "▶")
        if vertical:
            y += card.height + _CARD_GAP
        else:
            x += _CARD_WIDTH + _CARD_GAP
    return _finish(source, ir, canvas, exceeds_budget)


def _compile_journey(
    source: str, ir: DiagramIR, chart: JourneyChart, exceeds_budget: Callable[[int, int], bool]
) -> CompiledDiagram:
    cards = tuple(
        _Card(
            _wrapped((task.section, task.label) if task.section else (task.label,)),
            _wrapped(
                (f"{'●' * task.score}{'○' * (5 - task.score)} {task.score}/5", *(f"@ {actor}" for actor in task.actors))
            ),
        )
        for task in chart.tasks
    )
    return _compile_sequence_cards(source, ir, chart.title, cards, vertical=False, exceeds_budget=exceeds_budget)


def _compile_timeline(
    source: str, ir: DiagramIR, chart: TimelineChart, exceeds_budget: Callable[[int, int], bool]
) -> CompiledDiagram:
    cards = tuple(
        _Card(
            _wrapped((period.section, period.label) if period.section else (period.label,)),
            _wrapped(f"• {event}" for event in period.events),
        )
        for period in chart.periods
    )
    return _compile_sequence_cards(
        source, ir, chart.title, cards, vertical=chart.vertical, exceeds_budget=exceeds_budget
    )


def _compile_kanban(
    source: str, ir: DiagramIR, chart: KanbanChart, exceeds_budget: Callable[[int, int], bool]
) -> CompiledDiagram:
    columns = tuple(
        (
            _Card(_wrapped((column.label,)), ()),
            *(
                _Card(_wrapped((task.label,)), _wrapped(f"{key}: {value}" for key, value in task.metadata))
                for task in column.tasks
            ),
        )
        for column in chart.columns
    )
    width = len(columns) * (_CARD_WIDTH + _CARD_GAP) - _CARD_GAP
    height = max(sum(card.height for card in cards) + len(cards) - 1 for cards in columns)
    if exceeds_budget(width, height):
        raise ChartCanvasLimit
    canvas = TerminalCanvas()
    for index, cards in enumerate(columns):
        x, y = index * (_CARD_WIDTH + _CARD_GAP), 0
        for card in cards:
            _draw_card(canvas, x, y, card)
            y += card.height + 1
    return _finish(source, ir, canvas, exceeds_budget)


def compile_planning(source: str, ir: DiagramIR, exceeds_budget: Callable[[int, int], bool]) -> CompiledDiagram:
    """Render planning data while bounding canvas allocation before drawing."""
    chart = ir.chart
    if isinstance(chart, JourneyChart):
        return _compile_journey(source, ir, chart, exceeds_budget)
    if isinstance(chart, TimelineChart):
        return _compile_timeline(source, ir, chart, exceeds_budget)
    if isinstance(chart, KanbanChart):
        return _compile_kanban(source, ir, chart, exceeds_budget)
    raise TypeError("not a planning chart")
