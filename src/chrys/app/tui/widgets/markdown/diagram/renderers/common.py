# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared terminal chart measurement and budget checks."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal

from rich.cells import cell_len

from chrys.app.tui.widgets.markdown.diagram.canvas import (
    TerminalCanvas,
    crop_cell_text,
    sanitize_terminal_text,
    wrap_cell_text,
)
from chrys.app.tui.widgets.markdown.diagram.model import (
    CompiledDiagram,
    DiagramIR,
)


class ChartCanvasLimit(Exception):
    """Raised before a chart canvas would exceed renderer limits."""


def _cell_width(text: str) -> int:
    return cell_len(sanitize_terminal_text(text))


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    return crop_cell_text(sanitize_terminal_text(text), 0, width).rstrip()


def _center(canvas: TerminalCanvas, y: int, text: str, left: int, width: int) -> None:
    fitted = _fit(text, width)
    canvas.draw_text(left + max(0, (width - cell_len(fitted)) // 2), y, fitted)


def _title_lines(title: str, width: int) -> tuple[str, ...]:
    return wrap_cell_text(title, width) if title.strip() else ()


def _draw_title(canvas: TerminalCanvas, lines: tuple[str, ...], width: int) -> int:
    """Draw a premeasured title and return the first content row after its gap."""
    for row, line in enumerate(lines):
        _center(canvas, row, line, 0, width)
    return len(lines) + 1 if lines else 0


def _repeat(glyph: str, count: int) -> str:
    return glyph * max(0, count)


def _finish(
    source: str,
    ir: DiagramIR,
    canvas: TerminalCanvas,
    exceeds_budget: Callable[[int, int], bool],
) -> CompiledDiagram:
    width = canvas.natural_width
    height = canvas.natural_height
    if exceeds_budget(width, height):
        raise ChartCanvasLimit
    return CompiledDiagram(source, ir.kind, width, height, canvas.rows(width, height), ir.diagnostics)
