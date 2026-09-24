# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The cell grid: rows of styled cells, the screen window over them, and the history above it.

A buffer is one continuous list of physical rows. The last ``lines`` of them are the screen, the
rows before ``top`` are history; scrolling a line into history only moves ``top``, so nothing is
copied and a row keeps its identity (and its cached rendering) for as long as it lives. A row that
was filled by autowrap is flagged ``wrapped``, which is all reflow needs to rebuild the original
lines at another width.

Cells hold text, not code points: a combining sequence lives in the cell of its base character, and
a double-width character owns two cells, the second holding the empty string. Joining a row's cells
therefore yields its text, and the text's cell width is the number of cells.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from itertools import count, islice

from chrys.app.tui.terminal.emulator.pen import DEFAULT_PEN, Pen

_stamps = count(1)


class Row:
    """One physical line. Cells past the end are blank, so a row is only as long as its content."""

    __slots__ = ("cells", "pens", "stamp", "wrapped")

    def __init__(self, cells: list[str] | None = None, pens: list[Pen] | None = None, *, wrapped: bool = False) -> None:
        self.cells: list[str] = [] if cells is None else cells
        self.pens: list[Pen] = [] if pens is None else pens
        self.wrapped = wrapped
        """The line continues on the next row because autowrap, not the program, ended this one."""
        self.stamp = next(_stamps)
        """Changes whenever the row does and is never shared between rows: a rendering cache key."""

    @property
    def text(self) -> str:
        return "".join(self.cells)

    @property
    def is_blank(self) -> bool:
        """Whether the row shows nothing: no glyphs, no painted background, no decoration."""
        return all(cell == " " for cell in self.cells) and all(
            pen.background is None and not pen.attributes for pen in self.pens
        )

    def put(self, column: int, cells: Sequence[str], pen: Pen) -> None:
        """Overwrite cells starting at ``column``."""
        end = column + len(cells)
        self._extend_to(column)
        self._split_wide(column)
        self._split_wide(end)
        self.cells[column:end] = cells
        self.pens[column:end] = [pen] * len(cells)
        self.stamp = next(_stamps)

    def insert(self, column: int, cells: Sequence[str], pen: Pen, columns: int) -> None:
        """Insert cells at ``column``; what is pushed past the right edge is lost."""
        self._extend_to(column)
        self._split_wide(column)
        self.cells[column:column] = cells
        self.pens[column:column] = [pen] * len(cells)
        self.crop(columns)
        self.stamp = next(_stamps)

    def delete(self, column: int, count: int, eraser: Pen, columns: int) -> None:
        """Delete cells at ``column``; the rest of the row closes up and blanks enter on the right."""
        end = min(column + count, len(self.cells))
        if column < end:
            self._split_wide(column)
            self._split_wide(end)
            del self.cells[column:end]
            del self.pens[column:end]
        if eraser != DEFAULT_PEN:
            self.erase(max(column, columns - count), columns, eraser, columns)
        self.wrapped = False
        self.stamp = next(_stamps)

    def erase(self, start: int, end: int, eraser: Pen, columns: int) -> None:
        """Blank the cells in ``[start, end)``. An ``end`` at the right edge takes the whole tail."""
        length = len(self.cells)
        if end >= columns:
            end = max(end, length)
            self.wrapped = False
        if start >= end:
            return
        self.stamp = next(_stamps)
        if eraser == DEFAULT_PEN:
            if start >= length:
                return
            if end >= length:
                self._split_wide(start)
                del self.cells[start:]
                del self.pens[start:]
                return
        self._extend_to(start)
        self._split_wide(start)
        self._split_wide(end)
        self.cells[start:end] = " " * (end - start)
        self.pens[start:end] = [eraser] * (end - start)

    def crop(self, columns: int) -> None:
        """Drop everything past the right edge."""
        if len(self.cells) > columns:
            self._split_wide(columns)
            del self.cells[columns:]
            del self.pens[columns:]
            self.stamp = next(_stamps)

    def _extend_to(self, column: int) -> None:
        if (missing := column - len(self.cells)) > 0:
            self.cells.extend(" " * missing)
            self.pens.extend([DEFAULT_PEN] * missing)

    def _split_wide(self, column: int) -> None:
        """Blank a double-width character straddling ``column`` so that an edit may begin or end there."""
        cells = self.cells
        if 0 < column < len(cells) and not cells[column]:
            cells[column - 1] = cells[column] = " "


class Cursor:
    """The active position, in screen coordinates."""

    __slots__ = ("pending_wrap", "x", "y")

    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.pending_wrap = False
        """The last column was just written, and the cursor is past it in all but position.

        DEC's deferred wrap: the next character wraps first. Whether it does is the emulator's to
        say when that character comes, for with autowrap off it is written over the last column; the
        flag is raised either way, since a character completing the last one has to know which cell
        that was.
        """


class ScreenBuffer:
    """A screen of ``lines`` rows, its cursor and margins, and up to ``history_limit`` rows above it."""

    def __init__(self, columns: int, lines: int, *, history_limit: int) -> None:
        self.columns = columns
        self.lines = lines
        self.history_limit = history_limit
        self.rows: list[Row] = [Row() for _ in range(lines)]
        self.top = 0
        """Index in ``rows`` of the first screen row. Everything before it is history."""
        self.cursor = Cursor()
        self.margin_top = 0
        self.margin_bottom = lines - 1
        self._layout_is_stale = False
        self._damaged: set[int] = set()
        self._damaged_everything = True

    # -- reading ---------------------------------------------------------------------------

    def row(self, y: int) -> Row:
        """The screen row ``y``, for reading."""
        return self.rows[self.top + y]

    @property
    def cursor_index(self) -> int:
        """The cursor row as an index into ``rows``."""
        return self.top + self.cursor.y

    @property
    def used_height(self) -> int:
        """How many rows, history included, reach down to the cursor or to something visible."""
        return self._last_used_index() + 1

    @property
    def screen_text(self) -> list[str]:
        """The text of every screen row, without trailing blanks."""
        return [row.text.rstrip() for row in islice(self.rows, self.top, None)]

    # -- damage ----------------------------------------------------------------------------

    def damage(self, index: int) -> None:
        """Record that the row at ``index`` must be redrawn."""
        self._damaged.add(index)

    def damage_everything(self) -> None:
        self._damaged_everything = True

    def take_damage(self) -> set[int] | None:
        """Rows (as indices into ``rows``) changed since the last call; ``None`` means all of them."""
        damaged = None if self._damaged_everything else self._damaged
        self._damaged = set()
        self._damaged_everything = False
        return damaged

    def edit(self, y: int) -> Row:
        """The screen row ``y``, about to be changed."""
        index = self.top + y
        self._damaged.add(index)
        return self.rows[index]

    # -- vertical movement -----------------------------------------------------------------

    def set_margins(self, top: int, bottom: int) -> None:
        """Set the scrolling region to the screen rows ``top`` through ``bottom``, inclusive."""
        self.margin_top = top
        self.margin_bottom = bottom

    def index(self, eraser: Pen) -> None:
        """Move down a row, scrolling the region when the cursor sits on its last row."""
        cursor = self.cursor
        cursor.pending_wrap = False
        if cursor.y == self.margin_bottom:
            self.scroll_up(1, eraser)
        elif cursor.y < self.lines - 1:
            cursor.y += 1

    def reverse_index(self, eraser: Pen) -> None:
        """Move up a row, scrolling the region down when the cursor sits on its first row."""
        cursor = self.cursor
        cursor.pending_wrap = False
        if cursor.y == self.margin_top:
            self.scroll_down(1, eraser)
        elif cursor.y > 0:
            cursor.y -= 1

    def scroll_up(self, count: int, eraser: Pen) -> None:
        """Scroll the region up. Rows leaving the top of the screen become history."""
        if self.margin_top or not self.history_limit:
            self._rotate_up(self.margin_top, self.margin_bottom, count, eraser)
            return
        rows = self.rows
        count = min(count, self.margin_bottom + 1)
        for _ in range(count):
            self.top += 1
            rows.insert(self.top + self.margin_bottom, self._blank_row(eraser))
        # Rows inside the region kept their place in ``rows``; only the new blanks, and whatever
        # sits below the region, moved.
        first_new = self.top + self.margin_bottom - count + 1
        self._unwrap(first_new - 1)
        self._damaged.update(range(first_new, len(rows)))

    def scroll_down(self, count: int, eraser: Pen) -> None:
        """Scroll the region down; rows leaving its bottom are lost."""
        self._rotate_down(self.margin_top, self.margin_bottom, count, eraser)

    def insert_lines(self, count: int, eraser: Pen) -> None:
        """Open blank rows at the cursor, pushing the rest of the region down."""
        if self.margin_top <= self.cursor.y <= self.margin_bottom:
            self._rotate_down(self.cursor.y, self.margin_bottom, count, eraser)

    def delete_lines(self, count: int, eraser: Pen) -> None:
        """Remove rows at the cursor, pulling the rest of the region up."""
        if self.margin_top <= self.cursor.y <= self.margin_bottom:
            self._rotate_up(self.cursor.y, self.margin_bottom, count, eraser)

    def _rotate_up(self, first: int, last: int, count: int, eraser: Pen) -> None:
        count = min(count, last - first + 1)
        start, stop = self.top + first, self.top + last + 1
        rows = self.rows
        del rows[start : start + count]
        rows[stop - count : stop - count] = [self._blank_row(eraser) for _ in range(count)]
        self._unwrap(start - 1)
        self._unwrap(stop - count - 1)
        self._damaged.update(range(start, stop))

    def _rotate_down(self, first: int, last: int, count: int, eraser: Pen) -> None:
        count = min(count, last - first + 1)
        start, stop = self.top + first, self.top + last + 1
        rows = self.rows
        del rows[stop - count : stop]
        rows[start:start] = [self._blank_row(eraser) for _ in range(count)]
        self._unwrap(start - 1)
        self._unwrap(stop - 1)
        self._damaged.update(range(start, stop))

    def _unwrap(self, index: int) -> None:
        """The row at ``index`` no longer runs on into the row below it."""
        if 0 <= index < len(self.rows):
            self.rows[index].wrapped = False

    def _blank_row(self, eraser: Pen) -> Row:
        if eraser == DEFAULT_PEN:
            return Row()
        return Row([" "] * self.columns, [eraser] * self.columns)

    # -- erasing ---------------------------------------------------------------------------

    def erase_rows(self, first: int, last: int, eraser: Pen) -> None:
        """Blank the screen rows ``first`` through ``last``, inclusive."""
        if first > last:
            return
        for index in range(self.top + first, self.top + last + 1):
            self.rows[index] = self._blank_row(eraser)
            self._damaged.add(index)
        self._unwrap(self.top + first - 1)

    def fill(self, character: str) -> None:
        """Fill the screen with one character (the DECALN alignment pattern)."""
        for index in range(self.top, self.top + self.lines):
            self.rows[index] = Row([character] * self.columns, [DEFAULT_PEN] * self.columns)
        self._damaged_everything = True

    def clear_history(self) -> int:
        """Forget every row above the screen; returns how many rows were dropped from the front."""
        dropped = self.top
        if dropped:
            del self.rows[:dropped]
            self.top = 0
            self._damaged_everything = True
        return dropped

    def trim_history(self) -> int:
        """Enforce ``history_limit``; returns how many rows were dropped from the front."""
        if (excess := self.top - self.history_limit) <= 0:
            return 0
        del self.rows[:excess]
        self.top -= excess
        self._damaged_everything = True
        return excess

    # -- geometry --------------------------------------------------------------------------

    def resize(self, columns: int, lines: int, *, reflow: bool, pull_history: bool) -> None:
        """Change the screen size.

        Args:
            columns: New width.
            lines: New height.
            reflow: Rewrap lines to the new width. Without it rows keep their layout, as when
                whoever drives the terminal repaints the screen itself after a resize.
            pull_history: A taller screen reveals history at its top, the way a desktop terminal
                grows, instead of gaining blank rows at its bottom.
        """
        if columns != self.columns or (reflow and self._layout_is_stale):
            self.columns = columns
            self._layout_is_stale = not reflow
            if reflow:
                self._reflow()
        if lines != self.lines:
            self._set_lines(lines, pull_history=pull_history)
        self.margin_top, self.margin_bottom = 0, self.lines - 1
        cursor = self.cursor
        if cursor.pending_wrap and cursor.x < self.columns - 1:
            # The row grew: there is room again for what was waiting to wrap.
            cursor.x += 1
            cursor.pending_wrap = False
        cursor.x = min(cursor.x, self.columns - 1)
        self._damaged_everything = True

    def anchor_cursor_row(self, y: int) -> None:
        """Move the screen window so the row the cursor is on becomes screen row ``y``.

        For a host that keeps its own screen and readdresses ours after a resize: its first absolute
        position names the row it believes the cursor is on, which tells us where its screen begins.
        """
        cursor_index = self.cursor_index
        top = max(0, cursor_index - min(max(y, 0), self.lines - 1))
        if top == self.top:
            return
        self.top = top
        self.cursor.y = cursor_index - top
        self._fit_rows()
        self._damaged_everything = True

    def _set_lines(self, lines: int, *, pull_history: bool) -> None:
        cursor = self.cursor
        if lines < self.lines:
            # Blank rows below the output go first. Only when those run out do rows leave through
            # the top, and never the row the cursor is on.
            spare = len(self.rows) - 1 - self._last_used_index()
            moved = min(max(self.lines - lines - spare, 0), cursor.y)
        else:
            moved = -min(lines - self.lines, self.top) if pull_history else 0
        self.top += moved
        cursor.y -= moved
        self.lines = lines
        self._fit_rows()
        if not self.history_limit:
            # Rows that left through the top have no history to go to. Where there is one, it is
            # trimmed after the next feed, which is where whoever numbers the rows hears of it.
            self.trim_history()

    def _fit_rows(self) -> None:
        """Restore the invariant that the screen is exactly the last ``lines`` rows."""
        end = self.top + self.lines
        del self.rows[end:]
        self.rows.extend(Row() for _ in range(end - len(self.rows)))

    def _last_used_index(self) -> int:
        """Index of the last row that holds the cursor or anything visible."""
        index = len(self.rows) - 1
        floor = self.cursor_index
        while index > floor and self.rows[index].is_blank:
            index -= 1
        return index

    def _reflow(self) -> None:
        """Rebuild every line at the current width, carrying the cursor and the screen origin along."""
        columns, rows, cursor = self.columns, self.rows, self.cursor
        cursor_index = self.cursor_index
        last = self._last_used_index()
        reflowed: list[Row] = []
        top = cursor_row = 0
        index = 0
        while index <= last:
            first = index
            while index < last and rows[index].wrapped:
                index += 1
            index += 1
            line = rows[first:index]
            if len(line) == 1 and len(line[0].cells) <= columns:
                pieces, starts = line, [0]
            else:
                pieces, starts = _wrap(line, columns)
            if first <= self.top < index:
                offset = sum(len(row.cells) for row in rows[first : self.top])
                top = len(reflowed) + bisect_right(starts, offset) - 1
            if first <= cursor_index < index:
                offset = sum(len(row.cells) for row in rows[first:cursor_index]) + cursor.x + cursor.pending_wrap
                piece = bisect_right(starts, offset) - 1
                cursor_row = len(reflowed) + piece
                x = offset - starts[piece]
                cursor.pending_wrap = x == columns == len(pieces[piece].cells)
                cursor.x = min(x, columns - 1)
            reflowed.extend(pieces)
        # The origin follows its line, but never so far up that content falls off the bottom, and
        # never so far down that the cursor is left above the screen.
        self.top = min(max(top, len(reflowed) - self.lines, 0), cursor_row)
        cursor.y = cursor_row - self.top
        self.rows = reflowed
        self._fit_rows()


def _wrap(line: Sequence[Row], columns: int) -> tuple[list[Row], list[int]]:
    """Lay one logical line out at ``columns`` wide; returns the rows and each row's cell offset."""
    cells = [cell for row in line for cell in row.cells]
    pens = [pen for row in line for pen in row.pens]
    while cells and cells[-1] == " " and pens[-1] == DEFAULT_PEN:
        cells.pop()
        pens.pop()
    pieces: list[Row] = []
    starts: list[int] = []
    start, total = 0, len(cells)
    while True:
        end = min(start + columns, total)
        if end < total and not cells[end]:
            # Never part a double-width character from its second cell.
            end = end - 1 if end - 1 > start else end + 1
        pieces.append(Row(cells[start:end], pens[start:end], wrapped=end < total))
        starts.append(start)
        start = end
        if start >= total:
            return pieces, starts
