# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Drawing emulator rows: cells and pens in, Textual strips out.

Palette colors stay palette colors here. Textual's ANSI filter resolves them against the app's
terminal theme on the way out, so the embedded terminal follows the theme without being redrawn.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Final

from rich.color import Color
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

from chrys.app.tui.terminal.emulator import Attribute, CursorShape, Pen, Rgb, Row

_BLOCK_CURSOR: Final = Style(reverse=True)
_UNDERLINE_CURSOR: Final = Style(underline=True)
# A bar cannot be drawn between two cells. Over a blank it is a thin block at the cell's left
# edge, which is where a bar cursor nearly always sits; over a glyph it falls back to underline.
_BAR_GLYPH: Final = "▏"


def _rich_color(color: int | Rgb | None) -> Color | None:
    if color is None:
        return None
    return Color.from_rgb(*color) if isinstance(color, Rgb) else Color.from_ansi(color)


@lru_cache(maxsize=2048)
def pen_style(pen: Pen) -> Style:
    """The Rich style a pen draws with. Unset attributes stay unset, so the widget's style shows through."""
    held = pen.attributes

    def when(attribute: Attribute) -> bool | None:
        return True if held & attribute else None

    return Style(
        color=_rich_color(pen.foreground),
        bgcolor=_rich_color(pen.background),
        bold=when(Attribute.BOLD),
        dim=when(Attribute.DIM),
        italic=when(Attribute.ITALIC),
        underline=when(Attribute.UNDERLINE),
        underline2=when(Attribute.DOUBLE_UNDERLINE),
        blink=when(Attribute.BLINK),
        reverse=when(Attribute.REVERSE),
        conceal=when(Attribute.CONCEAL),
        strike=when(Attribute.STRIKE),
        overline=when(Attribute.OVERLINE),
        link=pen.link,
    )


def render_row(row: Row) -> Strip:
    """A row as a strip exactly as wide as its content: one segment per run of cells sharing a pen."""
    cells, pens = row.cells, row.pens
    segments: list[Segment] = []
    start, total = 0, len(cells)
    while start < total:
        pen = pens[start]
        end = start + 1
        while end < total and pens[end] == pen:
            end += 1
        segments.append(Segment("".join(cells[start:end]), pen_style(pen)))
        start = end
    return Strip(segments, total)


def _divide(strip: Strip, row: Row, columns: Sequence[int]) -> list[Strip]:
    """Cut the strip of ``row`` before each of ``columns``, which are in order and begin cells.

    `Strip.divide` cuts by measured width, and inside a cell that several characters share (a
    joiner sequence, a base and its marks) a width is no place: it takes the sequence apart and
    blanks what it cannot show. The row knows where its cells are, so the cuts are made in its
    characters. Past the row a strip is blanks, one to a cell.
    """
    cells = row.cells
    parts: list[Strip] = []
    segments = iter(strip)
    held: Segment | None = None
    characters_before = columns_before = 0
    for column in columns:
        inside = min(column, len(cells))
        characters = sum(map(len, cells[:inside])) + column - inside
        wanted = characters - characters_before
        part: list[Segment] = []
        while wanted > 0 and (segment := held or next(segments, None)) is not None:
            held = None
            text, style, _control = segment
            if len(text) > wanted:
                segment, held = Segment(text[:wanted], style), Segment(text[wanted:], style)
            part.append(segment)
            wanted -= len(segment.text)
        parts.append(Strip(part, column - columns_before))
        characters_before, columns_before = characters, column
    rest = list(segments) if held is None else [held, *segments]
    parts.append(Strip(rest, strip.cell_length - columns_before))
    return parts


def restyle(strip: Strip, row: Row, start: int, end: int, style: Style) -> Strip:
    """Lay ``style`` over the cells in ``[start, end)`` of the strip of ``row``, padded out to reach them."""
    if start >= end:
        return strip
    strip = strip.adjust_cell_length(max(strip.cell_length, end))
    before, inside, after = _divide(strip, row, [start, end])
    return Strip.join([before, Strip(Segment.apply_style(inside, post_style=style), end - start), after])


def draw_cursor(strip: Strip, row: Row, column: int, shape: CursorShape) -> Strip:
    """Show the cursor at ``column``. On a double-width character it covers both cells."""
    cells = row.cells
    if 0 < column < len(cells) and not cells[column]:
        column -= 1
    width = 2 if column + 1 < len(cells) and not cells[column + 1] else 1
    if shape is CursorShape.BLOCK:
        return restyle(strip, row, column, column + width, _BLOCK_CURSOR)
    if shape is CursorShape.BAR and (column >= len(cells) or cells[column] == " "):
        strip = strip.adjust_cell_length(max(strip.cell_length, column + 1))
        before, under, after = _divide(strip, row, [column, column + 1])
        style = next((segment.style for segment in under), None)
        return Strip.join([before, Strip([Segment(_BAR_GLYPH, style)], 1), after])
    return restyle(strip, row, column, column + width, _UNDERLINE_CURSOR)


def character_span_to_cells(cells: Sequence[str], start: int, end: int, width: int) -> tuple[int, int]:
    """Convert a span of a row's text, in characters, to the cells it covers on a ``width``-wide row.

    Text selection counts characters, and a cell is not always one: a combining sequence is
    several characters in one cell, a double-width character is one character in two. A negative
    ``end`` means the rest of the row. The span never parts a cell.
    """
    first: int | None = None
    position = 0
    for column, cell in enumerate(cells):
        if first is None and position + len(cell) > start:
            first = column
        if first is not None and 0 <= end <= position and cell:
            return first, column
        position += len(cell)
    # Past the content every cell is one blank character.
    beyond = len(cells) - position
    if first is None:
        first = start + beyond
    return min(first, width), (width if end < 0 else min(end + beyond, width))
