# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Boxes that show one value of a tool call a dialog asks about, whole, under its name.

A value short enough is laid out at once (``ValueText``); a longer one scrolls
in its own box and draws only the rows in view (``LongValueView``), so a dialog
shows every value in full without laying out megabytes. Both show the value
sanitized and copy it as given. The approval dialog and a hook's question use them.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rich.cells import split_graphemes
from rich.text import Text
from textual import events
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widgets import Static

from chrys.app.tui.util.source_text import mark_hidden_format, sanitize_source_text
from chrys.foundation.i18n.formatting import sanitize_legacy_scalar

if TYPE_CHECKING:
    from textual.geometry import Offset
    from textual.selection import Selection
    from textual.widget import Widget

# A request can carry a whole file it is about to write, and laying out a long
# value blocks the app for about a second per MiB, on every resize too. A dialog
# lays out its values whole up to this many characters in all; a value that does
# not fit goes into a box that scrolls through all of it and draws only the rows
# in view.
VALUE_DISPLAY_MAX_CHARS = 64 * 1024

# A stretch of printable ASCII this long is cut into rows by count; the rest of
# a line is measured grapheme by grapheme, as rich draws it. A stretch leaves
# its first and last character to the measured side, so a joiner or variation
# selector next to it still pairs the way rich pairs it.
_ASCII_STRETCH = re.compile(r"[\x20-\x7e]{64,}")

# A value's box replaces its control characters, bidi controls and zero-width
# characters, one for one, so a remote agent's text cannot hide, reorder or
# restyle what is confirmed; it shows each line break as one LF and each tab as
# spaces up to the next stop this many columns apart. A copy hands back the
# value as given.
_TAB_SIZE = 8
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def _fold_line(line: str, width: int) -> list[str]:
    """Cut one line into pieces at most *width* cells wide, each as full as it can be."""
    if line.isascii():
        return [line[index : index + width] for index in range(0, len(line), width)] or [""]
    if len(line) * 2 <= width:
        return [line]
    pieces: list[str] = []
    row_start = row_cells = 0

    def measured(start: int, end: int) -> None:
        nonlocal row_start, row_cells
        spans, _ = split_graphemes(line[start:end])
        for span_start, _span_end, cells in spans:
            if row_cells and row_cells + cells > width:
                pieces.append(line[row_start : start + span_start])
                row_start, row_cells = start + span_start, 0
            row_cells += cells

    def counted(start: int, end: int) -> None:
        nonlocal row_start, row_cells
        position = start
        while end - position > (room := max(width - row_cells, 0)):
            position += room
            pieces.append(line[row_start:position])
            row_start, row_cells = position, 0
        row_cells += end - position

    position = 0
    for stretch in _ASCII_STRETCH.finditer(line):
        measured(position, stretch.start() + 1)
        counted(stretch.start() + 1, stretch.end() - 1)
        position = stretch.end() - 1
    measured(position, len(line))
    pieces.append(line[row_start:])
    return pieces


def shown_text(value: str) -> str:
    """Return *value* as a value's box shows it."""
    return mark_hidden_format(sanitize_source_text(value, tab_size=_TAB_SIZE))


def shown_line(value: str) -> str:
    """Return remote text shown on one line, such as a title, as the boxes would."""
    return mark_hidden_format(sanitize_legacy_scalar(value))


def _value_index(value: str, index: int, *, end: bool) -> int:
    """Return where in *value* the box's text at *index* falls.

    An index inside the spaces a tab is shown as falls before the tab, or after
    it with *end*, so a selection that touches a tab copies all of it.
    """
    shown = line_start = 0
    line_end = len(value)
    for line_break in _LINE_BREAK.finditer(value):
        width = len(value[line_start : line_break.start()].expandtabs(_TAB_SIZE))
        if index <= shown + width:
            line_end = line_break.start()
            break
        shown += width + 1
        line_start = line_break.end()
    line = value[line_start:line_end]
    column = index - shown
    position = shown = 0
    while (tab := line.find("\t", position)) != -1 and column > shown + tab - position:
        shown += tab - position
        width = _TAB_SIZE - shown % _TAB_SIZE
        if column < shown + width:
            return line_start + tab + (1 if end else 0)
        shown += width
        position = tab + 1
    return line_start + min(position + column - shown, len(line))


def _value_slice(value: str, start: int, end: int) -> str:
    """Return the part of *value* the box's text shows from *start* to *end*."""
    return value[_value_index(value, start, end=False) : _value_index(value, end, end=True)]


def fold_rows(text: str, width: int) -> tuple[list[str], list[int]]:
    """Cut *text* into rows at most *width* cells wide, with where each row starts in it."""
    rows: list[str] = []
    starts: list[int] = []
    line_start = 0
    for line in text.split("\n"):
        start = line_start
        for piece in _fold_line(line, width):
            rows.append(piece)
            starts.append(start)
            start += len(piece)
        line_start += len(line) + 1
    return rows, starts


class LongValueView(ScrollView):
    """A value too long to lay out at once, shown whole in a scrolling box.

    Its text is cut into rows as wide as the box, again when the box's width
    changes, and only the rows in view are drawn. A selection copies the value
    as given, its tabs and line breaks included, without the breaks the rows add.
    """

    # Its scrollbar sits against the right border; the stable gutter keeps the
    # scrollbar from changing the width its rows are cut to.
    DEFAULT_CSS = """
    LongValueView {
        width: 100%;
        height: auto;
        max-height: 20;
        border: round $tui-border-warning 50%;
        border-title-align: left;
        border-title-color: $warning;
        color: $text;
        padding: 0 0 0 1;
        margin-bottom: 1;
        overflow-x: hidden;
        overflow-y: auto;
        scrollbar-gutter: stable;
        scrollbar-size-vertical: 1;
    }
    LongValueView:focus {
        border: round $tui-border-warning;
    }
    """

    def __init__(self, value: str, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self._value = value
        self._text = shown_text(value)
        self._rows: list[str] = []
        self._row_starts: list[int] = []
        self._fold_width = 0

    @property
    def value(self) -> str:
        """The value as given, not as shown."""
        return self._value

    def on_resize(self, _event: events.Resize) -> None:
        # One blank column keeps the text off the scrollbar, as the left
        # padding keeps it off the border.
        width = self.scrollable_content_region.width - 1
        if width < 1 or width == self._fold_width:
            return
        self._fold_width = width
        self._rows, self._row_starts = fold_rows(self._text, width)
        self.virtual_size = Size(width, len(self._rows))
        self.refresh()

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        row = scroll_y + y
        width = self.size.width
        rich_style = self.rich_style
        if row >= len(self._rows):
            return Strip.blank(width, rich_style)
        # Text.render drops the base style of a text without spans, so the box's
        # colors go on as a span.
        text = Text(self._rows[row], no_wrap=True)
        text.stylize(rich_style)
        selection = self.text_selection
        if selection is not None and (span := selection.get_span(row)) is not None:
            start, end = span
            selection_style = self.screen.get_component_rich_style("screen--selection")
            text.stylize(selection_style, start, len(text) if end == -1 else end)
        strip = Strip(text.render(self.app.console), text.cell_len)
        return strip.crop_extend(scroll_x, scroll_x + width, rich_style).apply_offsets(scroll_x, row)

    def selection_updated(self, selection: Selection | None) -> None:
        self.refresh()

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        start = 0 if selection.start is None else self._text_index(selection.start)
        end = len(self._text) if selection.end is None else self._text_index(selection.end)
        return _value_slice(self._value, start, end), "\n"

    def _text_index(self, offset: Offset) -> int:
        if offset.y >= len(self._rows):
            return len(self._text)
        row = max(offset.y, 0)
        return self._row_starts[row] + min(max(offset.x, 0), len(self._rows[row]))


class ValueText(Static):
    """A value short enough to lay out at once, shown and copied as the long box does."""

    # Bordered only as an argument's box (``value_box``): a dialog may show a value
    # on its own as a plain line.
    DEFAULT_CSS = """
    ValueText.-boxed {
        width: 100%;
        height: auto;
        border: round $tui-border-warning 50%;
        border-title-align: left;
        border-title-color: $warning;
        color: $text;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, value: str, *, id: str | None = None, classes: str | None = None) -> None:
        self._value = value
        self._text = shown_text(value)
        super().__init__(Text(self._text), id=id, classes=classes)

    @property
    def value(self) -> str:
        """The value as given, not as shown."""
        return self._value

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        # A selection offset names a line of the text and a character in it.
        line_starts = [0, *(line_break.end() for line_break in re.finditer("\n", self._text))]
        line_ends = [start - 1 for start in line_starts[1:]] + [len(self._text)]

        def text_index(offset: Offset) -> int:
            if offset.y >= len(line_starts):
                return len(self._text)
            line = max(offset.y, 0)
            return min(line_starts[line] + max(offset.x, 0), line_ends[line])

        start = 0 if selection.start is None else text_index(selection.start)
        end = len(self._text) if selection.end is None else text_index(selection.end)
        return _value_slice(self._value, start, end), "\n"


def value_box(value: str, label: str, *, whole: bool, classes: str | None = None) -> Widget:
    """Return the bordered box that shows one argument *value* under *label*, laid out *whole* or scrolling."""
    box: Widget = ValueText(value, classes=classes) if whole else LongValueView(value, classes=classes)
    box.add_class("-boxed")
    box.border_title = Text(shown_line(label))
    return box
