# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Sparse Unicode-cell-aware canvas for terminal diagrams."""

from __future__ import annotations

import unicodedata
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style

from .model import Point

_LEFT = 1
_RIGHT = 2
_UP = 4
_DOWN = 8

_LINE_GLYPHS = {
    _LEFT: "─",
    _RIGHT: "─",
    _UP: "│",
    _DOWN: "│",
    _LEFT | _RIGHT: "─",
    _UP | _DOWN: "│",
    _RIGHT | _DOWN: "┌",
    _LEFT | _DOWN: "┐",
    _RIGHT | _UP: "└",
    _LEFT | _UP: "┘",
    _RIGHT | _UP | _DOWN: "├",
    _LEFT | _UP | _DOWN: "┤",
    _LEFT | _RIGHT | _DOWN: "┬",
    _LEFT | _RIGHT | _UP: "┴",
    _LEFT | _RIGHT | _UP | _DOWN: "┼",
}


@dataclass(frozen=True, slots=True)
class CellStyleSpan:
    """A half-open terminal-cell range; the leading cell owns a wide grapheme's style."""

    start: int
    end: int
    style: Style


@dataclass(frozen=True, slots=True)
class _CellGlyph:
    text: str
    start: int
    end: int


class CellRow:
    """Index a static row once and composite visible overlays in terminal cells."""

    def __init__(self, row: str) -> None:
        glyphs = []
        position = 0
        for glyph in iter_graphemes(row):
            end = position + cell_len(glyph)
            if end > position and glyph != " ":
                glyphs.append(_CellGlyph(glyph, position, end))
            position = end
        self._glyphs = tuple(glyphs)
        self._starts = tuple(glyph.start for glyph in glyphs)

    def composite(self, start: int, width: int, overlays: Iterable[tuple[int, str]] = ()) -> str:
        """Later overlays win; clipped or partly covered wide glyphs become spaces."""
        if width <= 0:
            return ""
        end = start + width
        cells: list[_CellGlyph | None] = [None] * width

        def put(glyph: _CellGlyph) -> None:
            left, right = max(start, glyph.start), min(end, glyph.end)
            if left < right:
                cells[left - start : right - start] = [glyph] * (right - left)

        first = max(0, bisect_right(self._starts, start) - 1)
        for index in range(first, len(self._glyphs)):
            glyph = self._glyphs[index]
            if glyph.start >= end:
                break
            put(glyph)
        for x, overlay in overlays:
            if x >= end or x + cell_len(overlay) <= start:
                continue
            for value in iter_graphemes(overlay):
                next_x = x + cell_len(value)
                put(_CellGlyph(value, x, next_x))
                x = next_x
                if x >= end:
                    break
        result: list[str] = []
        offset = 0
        while offset < width:
            glyph = cells[offset]
            if (
                glyph is not None
                and glyph.start == start + offset
                and glyph.end <= end
                and all(cell is glyph for cell in cells[offset : glyph.end - start])
            ):
                result.append(glyph.text)
                offset = glyph.end - start
            else:
                result.append(" ")
                offset += 1
        return "".join(result)


def styled_cell_segments(
    row: str, spans: tuple[CellStyleSpan, ...], start: int, width: int, base: Style
) -> list[Segment]:
    """Crop and style without splitting wide graphemes at span or viewport boundaries."""
    if width <= 0:
        return []

    def style_at(position: int) -> Style:
        style = base
        for span in spans:
            if span.start <= position < span.end:
                style += span.style
        return style

    segments: list[Segment] = []
    position = 0
    end = start + width
    for glyph in iter_graphemes(row):
        next_position = position + cell_len(glyph)
        if next_position > start and position < end:
            overlap = min(end, next_position) - max(start, position)
            text = glyph if start <= position and next_position <= end else " " * overlap
            segments.append(Segment(text, style_at(position)))
        position = next_position
        if position >= end:
            break
    if position < end:
        # rows() trims blank cells. Restore them in runs split at style boundaries.
        padding_start = max(start, position)
        boundaries = {padding_start, end}
        boundaries.update(
            boundary for span in spans for boundary in (span.start, span.end) if padding_start < boundary < end
        )
        for left, right in pairwise(sorted(boundaries)):
            segments.append(Segment(" " * (right - left), style_at(left)))
    return list(Segment.simplify(segments))


def sanitize_terminal_text(text: str) -> str:
    """Replace terminal control characters with visible width-one glyphs."""
    return "".join("�" if ord(char) < 32 or 0x7F <= ord(char) < 0xA0 else char for char in text)


def iter_graphemes(text: str) -> Iterator[str]:
    """Yield practical terminal grapheme clusters without an extra dependency."""
    cluster = ""
    regional_count = 0
    for char in text:
        codepoint = ord(char)
        regional = 0x1F1E6 <= codepoint <= 0x1F1FF
        joins_previous = bool(cluster) and (
            cluster.endswith("\u200d")
            or char == "\u200d"
            or unicodedata.combining(char) != 0
            or unicodedata.category(char) in {"Mn", "Me"}
            or 0xFE00 <= codepoint <= 0xFE0F
            or 0x1F3FB <= codepoint <= 0x1F3FF
            or codepoint == 0x20E3
            or (regional and regional_count == 1)
        )
        if not joins_previous and cluster:
            yield cluster
            cluster = ""
            regional_count = 0
        cluster += char
        if regional:
            regional_count += 1
        elif char != "\u200d":
            regional_count = 0
    if cluster:
        yield cluster


def crop_cell_text(text: str, start: int, width: int) -> str:
    """Crop *text* by terminal cells, replacing partial graphemes with spaces."""
    if width <= 0:
        return ""
    end = start + width
    position = 0
    output: list[str] = []
    output_width = 0
    for grapheme in iter_graphemes(text):
        grapheme_width = max(0, cell_len(grapheme))
        grapheme_end = position + grapheme_width
        if grapheme_end <= start:
            position = grapheme_end
            continue
        if position >= end:
            break
        overlap_start = max(start, position)
        overlap_end = min(end, grapheme_end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap:
            if position >= start and grapheme_end <= end:
                output.append(grapheme)
            else:
                output.append(" " * overlap)
            output_width += overlap
        position = grapheme_end
    if output_width < width:
        output.append(" " * (width - output_width))
    return "".join(output)


def wrap_cell_text(text: str, width: int) -> tuple[str, ...]:
    """Wrap plain text to a fixed terminal-cell width."""
    width = max(1, width)
    normalized = " ".join(sanitize_terminal_text(text).split())
    if not normalized:
        return ("",)
    lines: list[str] = []
    current: list[str] = []
    current_width = 0
    last_space = -1
    for grapheme in iter_graphemes(normalized):
        grapheme_width = max(0, cell_len(grapheme))
        if current and current_width + grapheme_width > width:
            if last_space >= 0:
                head = "".join(current[:last_space]).rstrip()
                tail = current[last_space + 1 :]
                lines.append(head)
                current = tail
                current_width = cell_len("".join(current))
            else:
                lines.append("".join(current))
                current = []
                current_width = 0
            last_space = max((index for index, value in enumerate(current) if value == " "), default=-1)
        if grapheme == " " and not current:
            continue
        current.append(grapheme)
        current_width += grapheme_width
        if grapheme == " ":
            last_space = len(current) - 1
    if current:
        lines.append("".join(current).rstrip())
    return tuple(lines or [""])


class TerminalCanvas:
    """Sparse plain-text terminal canvas with merged box-drawing junctions."""

    def __init__(self) -> None:
        self._glyphs: dict[tuple[int, int], str] = {}
        self._continuations: set[tuple[int, int]] = set()
        self._lines: dict[tuple[int, int], int] = {}
        self._max_x = -1
        self._max_y = -1
        self.styles: dict[int, tuple[CellStyleSpan, ...]] = {}

    def style_span(self, y: int, start: int, end: int, style: Style) -> None:
        """Overlay a style range without altering text or canvas dimensions."""
        self.styles[y] = (*self.styles.get(y, ()), CellStyleSpan(start, end, style))

    def render_line(self, y: int, width: int, *, x: int = 0, style: Style | None = None) -> list[Segment]:
        """Render a styled cell crop; rows() remains the plain-text contract."""
        row = self.rows(self.natural_width, y + 1)[y]
        return styled_cell_segments(row, self.styles.get(y, ()), x, width, style or Style())

    def _touch(self, x: int, y: int) -> None:
        if x < 0 or y < 0:
            raise ValueError("terminal canvas coordinates must be non-negative")
        self._max_x = max(self._max_x, x)
        self._max_y = max(self._max_y, y)

    def put(self, x: int, y: int, glyph: str) -> None:
        """Overlay one grapheme at *(x, y)*."""
        if not glyph:
            return
        width = max(1, cell_len(glyph))
        self._touch(x + width - 1, y)
        self._glyphs[(x, y)] = glyph
        for offset in range(1, width):
            self._continuations.add((x + offset, y))

    def draw_text(self, x: int, y: int, text: str) -> None:
        """Draw dynamic plain text without interpreting markup."""
        cursor = x
        for grapheme in iter_graphemes(sanitize_terminal_text(text)):
            self.put(cursor, y, grapheme)
            cursor += max(0, cell_len(grapheme))

    def _connect(self, first: Point, second: Point) -> None:
        self._touch(first.x, first.y)
        self._touch(second.x, second.y)
        if first.y == second.y:
            step = 1 if second.x > first.x else -1
            for x in range(first.x, second.x, step):
                left = Point(x, first.y)
                right = Point(x + step, first.y)
                self._lines[(left.x, left.y)] = self._lines.get((left.x, left.y), 0) | (_RIGHT if step > 0 else _LEFT)
                self._lines[(right.x, right.y)] = self._lines.get((right.x, right.y), 0) | (
                    _LEFT if step > 0 else _RIGHT
                )
            return
        if first.x == second.x:
            step = 1 if second.y > first.y else -1
            for y in range(first.y, second.y, step):
                top = Point(first.x, y)
                bottom = Point(first.x, y + step)
                self._lines[(top.x, top.y)] = self._lines.get((top.x, top.y), 0) | (_DOWN if step > 0 else _UP)
                self._lines[(bottom.x, bottom.y)] = self._lines.get((bottom.x, bottom.y), 0) | (
                    _UP if step > 0 else _DOWN
                )
            return
        raise ValueError("diagram paths must be horizontal or vertical")

    def draw_path(self, points: Iterable[Point]) -> None:
        """Draw a polyline whose segments must be cardinal."""
        iterator = iter(points)
        try:
            previous = next(iterator)
        except StopIteration:
            return
        self._touch(previous.x, previous.y)
        for point in iterator:
            self._connect(previous, point)
            previous = point

    def draw_path_on_top(self, points: Iterable[Point]) -> None:
        """Draw a cardinal path above existing text and fill glyphs."""
        path = tuple(points)
        self.draw_path(path)
        cells: set[tuple[int, int]] = set()
        for first, second in pairwise(path):
            if first.y == second.y:
                start, end = sorted((first.x, second.x))
                cells.update((x, first.y) for x in range(start, end + 1))
            elif first.x == second.x:
                start, end = sorted((first.y, second.y))
                cells.update((first.x, y) for y in range(start, end + 1))
            else:
                raise ValueError("diagram paths must be horizontal or vertical")
        for x, y in cells:
            if glyph := _LINE_GLYPHS.get(self._lines.get((x, y), 0)):
                self.put(x, y, glyph)

    def draw_box(self, x: int, y: int, width: int, height: int, *, rounded: bool = False) -> None:
        """Draw a rectangular node box."""
        if width < 2 or height < 2:
            raise ValueError("diagram boxes must be at least two cells wide and high")
        self.draw_path(
            (
                Point(x, y),
                Point(x + width - 1, y),
                Point(x + width - 1, y + height - 1),
                Point(x, y + height - 1),
                Point(x, y),
            )
        )
        if rounded:
            self.put(x, y, "╭")
            self.put(x + width - 1, y, "╮")
            self.put(x, y + height - 1, "╰")
            self.put(x + width - 1, y + height - 1, "╯")

    def draw_horizontal(self, x1: int, x2: int, y: int) -> None:
        """Draw a horizontal segment."""
        self.draw_path((Point(x1, y), Point(x2, y)))

    def rows(self, width: int, height: int) -> tuple[str, ...]:
        """Materialize sparse contents into plain rows without trailing spaces."""
        if width < 0 or height < 0:
            raise ValueError("diagram dimensions must be non-negative")
        rendered: list[str] = []
        for y in range(height):
            cells: list[str] = []
            x = 0
            while x < width:
                glyph = self._glyphs.get((x, y))
                if glyph is not None:
                    cells.append(glyph)
                    x += max(1, cell_len(glyph))
                    continue
                if (x, y) in self._continuations:
                    x += 1
                    continue
                mask = self._lines.get((x, y), 0)
                cells.append(_LINE_GLYPHS.get(mask, " "))
                x += 1
            rendered.append("".join(cells).rstrip())
        return tuple(rendered)

    @property
    def natural_width(self) -> int:
        """Width of all drawn cells."""
        return self._max_x + 1

    @property
    def natural_height(self) -> int:
        """Height of all drawn cells."""
        return self._max_y + 1
