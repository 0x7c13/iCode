# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Emulator rows drawn as Textual strips: pens to styles, the cursor, and selection spans."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from rich.color import Color
from rich.style import Style
from textual.strip import Strip

from chrys.app.tui.terminal.emulator import Attribute, CursorShape, Pen, Rgb, Row, TerminalEmulator
from chrys.app.tui.terminal.rendering import character_span_to_cells, draw_cursor, pen_style, render_row, restyle

_RED = Pen(foreground=1)

# Several characters that share a cell, or a pair of them: what a cut by width takes apart.
_CLUSTERS = [
    pytest.param("👩\u200d👧", id="joiner-sequence"),
    pytest.param("e\u0301\u0302", id="combining-marks"),
    pytest.param("☺\ufe0f", id="variation-selector"),
    pytest.param("1\ufe0f\u20e3", id="keycap"),
    pytest.param("👍🏽", id="skin-tone"),
]


def _row(output: str, *, columns: int = 10) -> Row:
    """The first screen row after a terminal of ``columns`` has been sent ``output``."""
    emulator = TerminalEmulator(columns, 3)
    emulator.feed(output)
    return emulator.buffer.row(0)


def _segments(strip: Strip) -> list[tuple[str, Style | None]]:
    return [(segment.text, segment.style) for segment in strip]


# -- pen_style ---------------------------------------------------------------------------------------


def test_default_pen_sets_nothing_so_the_widget_style_shows_through() -> None:
    assert pen_style(Pen()) == Style()


def test_palette_colors_stay_palette_colors() -> None:
    style = pen_style(Pen(foreground=1, background=250))

    # Left as indices, they follow the app's terminal theme without the row being drawn again.
    assert style.color == Color.from_ansi(1)
    assert style.bgcolor == Color.from_ansi(250)


def test_direct_colors_become_truecolor() -> None:
    style = pen_style(Pen(foreground=Rgb(1, 2, 3), background=Rgb(250, 251, 252)))

    assert style.color == Color.from_rgb(1, 2, 3)
    assert style.bgcolor == Color.from_rgb(250, 251, 252)


@pytest.mark.parametrize(
    ("attribute", "style_field"),
    [
        (Attribute.BOLD, "bold"),
        (Attribute.DIM, "dim"),
        (Attribute.ITALIC, "italic"),
        (Attribute.UNDERLINE, "underline"),
        (Attribute.DOUBLE_UNDERLINE, "underline2"),
        (Attribute.BLINK, "blink"),
        (Attribute.REVERSE, "reverse"),
        (Attribute.CONCEAL, "conceal"),
        (Attribute.STRIKE, "strike"),
        (Attribute.OVERLINE, "overline"),
    ],
)
def test_each_attribute_sets_its_style_and_no_other(attribute: Attribute, style_field: str) -> None:
    assert pen_style(Pen(attributes=attribute)) == Style.parse(style_field)


def test_every_attribute_has_a_style() -> None:
    drawn = pen_style(Pen(attributes=Attribute(sum(Attribute))))

    assert drawn == Style(
        bold=True,
        dim=True,
        italic=True,
        underline=True,
        underline2=True,
        blink=True,
        reverse=True,
        conceal=True,
        strike=True,
        overline=True,
    )


def test_unset_attributes_stay_unset_rather_than_off() -> None:
    style = pen_style(Pen(attributes=Attribute.BOLD))

    # ``False`` would switch off an italic the widget's own style asks for; ``None`` leaves it be.
    assert style.italic is None
    assert (Style(italic=True) + style).italic is True


def test_link_rides_on_the_style() -> None:
    assert pen_style(Pen(link="https://example.com")).link == "https://example.com"
    assert pen_style(Pen()).link is None


def test_equal_pens_share_one_style() -> None:
    assert pen_style(Pen(foreground=4, attributes=Attribute.BOLD)) is pen_style(
        Pen(foreground=4, attributes=Attribute.BOLD)
    )


# -- render_row --------------------------------------------------------------------------------------


def test_row_renders_one_segment_per_run_of_cells_sharing_a_pen() -> None:
    strip = render_row(_row("a\x1b[31mbc\x1b[0md"))

    assert _segments(strip) == [("a", pen_style(Pen())), ("bc", pen_style(_RED)), ("d", pen_style(Pen()))]


def test_row_is_exactly_as_wide_as_its_cells() -> None:
    row = _row("ab")

    strip = render_row(row)

    assert strip.text == "ab"
    assert strip.cell_length == len(row.cells) == 2


def test_double_width_character_counts_both_of_its_cells() -> None:
    row = _row("a中b")

    strip = render_row(row)

    assert row.cells == ["a", "中", "", "b"]
    assert strip.text == "a中b"
    assert strip.cell_length == 4
    assert sum(segment.cell_length for segment in strip) == 4


def test_empty_row_renders_an_empty_strip() -> None:
    strip = render_row(Row())

    assert _segments(strip) == []
    assert strip.cell_length == 0


# -- restyle -----------------------------------------------------------------------------------------


def test_restyle_lays_the_style_over_the_span_only() -> None:
    row = _row("abcdef")

    restyled = restyle(render_row(row), row, 2, 4, Style(reverse=True))

    assert restyled.text == "abcdef"
    assert restyled.cell_length == 6
    assert [text for text, style in _segments(restyled) if style is not None and style.reverse] == ["cd"]


def test_restyle_keeps_the_style_already_there() -> None:
    row = _row("\x1b[31mabc")

    restyled = restyle(render_row(row), row, 1, 2, Style(reverse=True))

    assert ("b", pen_style(_RED) + Style(reverse=True)) in _segments(restyled)


def test_restyle_pads_the_strip_out_to_a_span_beyond_it() -> None:
    row = _row("ab")

    restyled = restyle(render_row(row), row, 4, 6, Style(reverse=True))

    assert restyled.text == "ab    "
    assert restyled.cell_length == 6
    assert _segments(restyled)[-1] == ("  ", Style(reverse=True))


@pytest.mark.parametrize("cluster", _CLUSTERS)
def test_restyle_beside_characters_sharing_a_cell_leaves_them_whole(cluster: str) -> None:
    row = _row(f"ab{cluster}C")
    cells = row.cells
    after = len(cells) - 1
    assert cells[2] == cluster

    for start, end in [(0, 2), (1, 2), (2, after), (after, after + 1), (1, after + 1), (2, after + 3)]:
        restyled = restyle(render_row(row), row, start, end, Style(reverse=True))

        assert restyled.text.rstrip() == row.text, (start, end)
        assert restyled.cell_length == max(len(cells), end), (start, end)
        assert _cursor_cells(restyled, _reversed).rstrip() == "".join(cells[start:end]), (start, end)


def test_restyle_of_an_empty_span_returns_the_strip_untouched() -> None:
    row = _row("ab")
    strip = render_row(row)

    assert restyle(strip, row, 1, 1, Style(reverse=True)) is strip
    assert restyle(strip, row, 2, 1, Style(reverse=True)) is strip


# -- draw_cursor -------------------------------------------------------------------------------------


def _reversed(style: Style) -> bool | None:
    return style.reverse


def _underlined(style: Style) -> bool | None:
    return style.underline


def _cursor_cells(strip: Strip, marked: Callable[[Style], bool | None]) -> str:
    """The text of the cells the cursor's style is on."""
    return "".join(segment.text for segment in strip if segment.style is not None and marked(segment.style))


def test_block_cursor_reverses_its_cell() -> None:
    row = _row("abc")

    strip = draw_cursor(render_row(row), row, 1, CursorShape.BLOCK)

    assert strip.text == "abc"
    assert _cursor_cells(strip, _reversed) == "b"


def test_underline_cursor_underlines_its_cell() -> None:
    row = _row("abc")

    strip = draw_cursor(render_row(row), row, 1, CursorShape.UNDERLINE)

    assert _cursor_cells(strip, _underlined) == "b"
    assert _cursor_cells(strip, _reversed) == ""


@pytest.mark.parametrize("column", [1, 2], ids=["first-half", "second-half"])
def test_cursor_on_either_half_of_a_double_width_character_covers_both(column: int) -> None:
    row = _row("a中b")

    strip = draw_cursor(render_row(row), row, column, CursorShape.BLOCK)

    assert _cursor_cells(strip, _reversed) == "中"
    assert strip.cell_length == 4


@pytest.mark.parametrize("shape", [CursorShape.BLOCK, CursorShape.UNDERLINE, CursorShape.BAR])
@pytest.mark.parametrize("cluster", _CLUSTERS)
def test_cursor_beside_characters_sharing_a_cell_leaves_them_whole(cluster: str, shape: CursorShape) -> None:
    row = _row(f"{cluster}C")
    marked = _reversed if shape is CursorShape.BLOCK else _underlined

    beside = draw_cursor(render_row(row), row, len(row.cells) - 1, shape)
    upon = draw_cursor(render_row(row), row, 0, shape)

    assert beside.text == upon.text == row.text
    assert _cursor_cells(beside, marked) == "C"
    assert _cursor_cells(upon, marked) == cluster


@pytest.mark.parametrize("cluster", _CLUSTERS)
def test_bar_cursor_over_a_blank_beside_characters_sharing_a_cell_leaves_them_whole(cluster: str) -> None:
    row = _row(f"{cluster} x")

    strip = draw_cursor(render_row(row), row, len(row.cells) - 2, CursorShape.BAR)

    assert strip.text == f"{cluster}▏x"
    assert strip.cell_length == len(row.cells)


def test_cursor_keeps_the_color_of_the_cell_under_it() -> None:
    row = _row("\x1b[31mabc")

    strip = draw_cursor(render_row(row), row, 0, CursorShape.BLOCK)

    assert _segments(strip)[0] == ("a", pen_style(_RED) + Style(reverse=True))


def test_cursor_past_the_end_of_the_row_pads_out_to_it() -> None:
    row = _row("ab")

    strip = draw_cursor(render_row(row), row, 5, CursorShape.BLOCK)

    assert strip.text == "ab    "
    assert strip.cell_length == 6
    assert _segments(strip)[-1] == (" ", Style(reverse=True))


def test_bar_cursor_over_a_blank_draws_a_bar_in_the_cell_style() -> None:
    row = _row("\x1b[41mab  ")

    strip = draw_cursor(render_row(row), row, 2, CursorShape.BAR)

    assert strip.text == "ab▏ "
    assert strip.cell_length == 4
    assert ("▏", pen_style(Pen(background=1))) in _segments(strip)


def test_bar_cursor_past_the_end_of_the_row_draws_a_bar() -> None:
    row = _row("ab")

    strip = draw_cursor(render_row(row), row, 2, CursorShape.BAR)

    assert strip.text == "ab▏"
    assert strip.cell_length == 3


def test_bar_cursor_over_a_glyph_falls_back_to_underline() -> None:
    row = _row("abc")

    strip = draw_cursor(render_row(row), row, 1, CursorShape.BAR)

    assert strip.text == "abc"
    assert _cursor_cells(strip, _underlined) == "b"


# -- character_span_to_cells -------------------------------------------------------------------------


def test_ascii_span_is_its_own_cells() -> None:
    assert character_span_to_cells(_row("abcdef").cells, 1, 4, 10) == (1, 4)


def test_double_width_character_is_one_character_in_two_cells() -> None:
    cells = _row("a中b").cells

    assert character_span_to_cells(cells, 1, 2, 10) == (1, 3)
    # The character after it starts a cell later than its index says.
    assert character_span_to_cells(cells, 2, 3, 10) == (3, 4)


def test_combining_sequence_is_several_characters_in_one_cell() -> None:
    cells = _row("éx中y").cells
    assert cells == ["é", "x", "中", "", "y"]

    assert character_span_to_cells(cells, 0, 2, 10) == (0, 1)
    assert character_span_to_cells(cells, 2, 4, 10) == (1, 4)
    assert character_span_to_cells(cells, 4, 5, 10) == (4, 5)


def test_span_never_parts_a_cell() -> None:
    cells = _row("éx").cells

    # Characters 1..2 are the accent alone; the cell it shares with its base is what gets marked.
    assert character_span_to_cells(cells, 1, 2, 10) == (0, 1)


def test_negative_end_means_the_rest_of_the_row() -> None:
    assert character_span_to_cells(_row("abc").cells, 1, -1, 10) == (1, 10)
    assert character_span_to_cells([], 0, -1, 10) == (0, 10)


def test_span_past_the_content_counts_blanks_as_one_cell_each() -> None:
    cells = _row("a中").cells

    # Two characters in three cells: character 4 is the blank in cell 5.
    assert character_span_to_cells(cells, 4, 6, 10) == (5, 7)
    assert character_span_to_cells([], 2, 5, 10) == (2, 5)


def test_span_is_clamped_to_the_width() -> None:
    cells = _row("abc").cells

    assert character_span_to_cells(cells, 1, 40, 10) == (1, 10)
    assert character_span_to_cells(cells, 20, 40, 10) == (10, 10)
