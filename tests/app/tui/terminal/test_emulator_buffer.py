# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The cell grid: editing a row, scrolling rows into history, and changing the screen's size."""

from __future__ import annotations

import pytest

from chrys.app.tui.terminal.emulator import DEFAULT_PEN, Attribute, Pen, Row, ScreenBuffer

_RED = Pen(foreground=1)
_ON_BLUE = Pen(background=4)


def row_of(text: str, pen: Pen = DEFAULT_PEN, *, wrapped: bool = False) -> Row:
    """A row holding ``text``, double-width characters taking the two cells they do on screen."""
    cells: list[str] = []
    for character in text:
        cells.append(character)
        if ord(character) >= 0x2E80:
            cells.append("")
    return Row(cells, [pen] * len(cells), wrapped=wrapped)


def buffer_of(*lines: str, columns: int = 10, height: int | None = None, history_limit: int = 100) -> ScreenBuffer:
    """A buffer these lines were written to from the top, the cursor left after the last of them."""
    buffer = ScreenBuffer(columns, height or len(lines), history_limit=history_limit)
    for number, line in enumerate(lines):
        if number:
            buffer.index(DEFAULT_PEN)
        buffer.edit(buffer.cursor.y).put(0, row_of(line).cells, DEFAULT_PEN)
    buffer.cursor.x = min(len(row_of(lines[-1]).cells), columns - 1) if lines else 0
    buffer.take_damage()
    return buffer


def texts(buffer: ScreenBuffer) -> list[str]:
    return [row.text.rstrip() for row in buffer.rows]


# -- rows --------------------------------------------------------------------------------------------


def test_new_row_is_empty_and_blank() -> None:
    row = Row()

    assert (row.cells, row.pens, row.text, row.wrapped) == ([], [], "", False)
    assert row.is_blank


def test_put_pads_up_to_the_column() -> None:
    row = Row()

    row.put(2, "ab", _RED)

    assert row.cells == [" ", " ", "a", "b"]
    assert row.pens == [DEFAULT_PEN, DEFAULT_PEN, _RED, _RED]


def test_put_overwrites_in_place() -> None:
    row = row_of("abcdef")

    row.put(1, "XY", _RED)

    assert row.text == "aXYdef"
    assert row.pens == [DEFAULT_PEN, _RED, _RED, DEFAULT_PEN, DEFAULT_PEN, DEFAULT_PEN]


def test_text_of_a_row_with_wide_characters_has_one_cell_per_column() -> None:
    row = row_of("a你b")

    assert row.cells == ["a", "你", "", "b"]
    assert row.text == "a你b"


@pytest.mark.parametrize(
    ("column", "written", "expected"),
    [
        (0, ["x"], ["x", " ", "b"]),
        (1, ["x"], [" ", "x", "b"]),
        (0, ["好", ""], ["好", "", "b"]),
        (1, ["好", ""], [" ", "好", ""]),
    ],
)
def test_put_blanks_the_other_half_of_a_wide_character_it_cuts(
    column: int, written: list[str], expected: list[str]
) -> None:
    row = row_of("你b")

    row.put(column, written, DEFAULT_PEN)

    assert row.cells == expected


def test_insert_pushes_cells_off_the_right_edge() -> None:
    row = row_of("abcd")

    row.insert(1, "XY", _RED, 5)

    assert row.text == "aXYbc"
    assert row.pens == [DEFAULT_PEN, _RED, _RED, DEFAULT_PEN, DEFAULT_PEN]


def test_insert_never_leaves_half_a_wide_character() -> None:
    inside = row_of("你b")
    inside.insert(1, "X", DEFAULT_PEN, 5)
    assert inside.cells == [" ", "X", " ", "b"]

    at_the_edge = row_of("a你")
    at_the_edge.insert(0, "X", DEFAULT_PEN, 3)
    assert at_the_edge.cells == ["X", "a", " "]


def test_delete_closes_up_and_ends_the_wrap() -> None:
    row = row_of("abcde", wrapped=True)

    row.delete(1, 2, DEFAULT_PEN, 5)

    assert row.text == "ade"
    assert not row.wrapped


def test_delete_lets_painted_blanks_in_on_the_right() -> None:
    row = row_of("abcde")

    row.delete(1, 2, _ON_BLUE, 5)

    assert row.text == "ade  "
    assert row.pens[3:] == [_ON_BLUE, _ON_BLUE]


def test_delete_past_the_content_is_harmless() -> None:
    row = row_of("abc")

    row.delete(1, 99, DEFAULT_PEN, 10)
    row.delete(7, 2, DEFAULT_PEN, 10)

    assert row.text == "a"


def test_delete_blanks_the_wide_characters_it_cuts() -> None:
    row = row_of("你好吗")

    row.delete(1, 2, DEFAULT_PEN, 10)

    assert row.cells == [" ", " ", "吗", ""]


def test_erase_with_the_default_pen_shortens_the_row() -> None:
    row = row_of("abcdef", wrapped=True)

    row.erase(3, 10, DEFAULT_PEN, 10)

    assert row.cells == ["a", "b", "c"]
    assert not row.wrapped


def test_erase_inside_the_row_blanks_cells_and_keeps_the_wrap() -> None:
    row = row_of("abcdef", wrapped=True)

    row.erase(1, 3, DEFAULT_PEN, 10)

    assert row.text == "a  def"
    assert row.wrapped


def test_erase_with_a_background_paints_to_the_right_edge() -> None:
    row = row_of("abc")

    row.erase(1, 6, _ON_BLUE, 6)

    assert row.text == "a     "
    assert row.pens == [DEFAULT_PEN] + [_ON_BLUE] * 5


def test_erase_past_the_content_with_a_background_paints_there() -> None:
    row = row_of("a")

    row.erase(3, 5, _ON_BLUE, 10)

    assert row.text == "a    "
    assert row.pens == [DEFAULT_PEN, DEFAULT_PEN, DEFAULT_PEN, _ON_BLUE, _ON_BLUE]


def test_erase_blanks_the_wide_characters_it_cuts() -> None:
    row = row_of("你好吗")

    row.erase(1, 3, DEFAULT_PEN, 10)

    assert row.cells == [" ", " ", " ", " ", "吗", ""]


def test_erase_of_nothing_changes_nothing() -> None:
    row = row_of("abc")
    stamp = row.stamp

    row.erase(2, 2, DEFAULT_PEN, 10)
    row.erase(3, 1, _ON_BLUE, 10)

    assert (row.text, row.stamp) == ("abc", stamp)


def test_crop_drops_what_is_past_the_edge() -> None:
    row = row_of("ab你")
    fits = row_of("ab")
    stamp = fits.stamp

    row.crop(3)
    fits.crop(3)

    assert row.cells == ["a", "b", " "]
    assert (fits.text, fits.stamp) == ("ab", stamp)


@pytest.mark.parametrize(
    ("row", "blank"),
    [
        (row_of("   "), True),
        (row_of("   ", _RED), True),
        (row_of("  x"), False),
        (row_of("   ", _ON_BLUE), False),
        (row_of("   ", Pen(attributes=Attribute.UNDERLINE)), False),
        (row_of("   ", Pen(attributes=Attribute.REVERSE)), False),
    ],
)
def test_blank_means_nothing_to_see(row: Row, blank: bool) -> None:
    assert row.is_blank is blank


def test_stamp_is_unique_and_changes_with_the_row() -> None:
    row, other = Row(), Row()
    assert row.stamp != other.stamp

    seen = {row.stamp}
    for edit in (
        lambda: row.put(0, "abcdef", DEFAULT_PEN),
        lambda: row.insert(1, "X", DEFAULT_PEN, 10),
        lambda: row.delete(0, 1, DEFAULT_PEN, 10),
        lambda: row.erase(1, 2, DEFAULT_PEN, 10),
        lambda: row.crop(3),
    ):
        edit()
        assert row.stamp not in seen
        seen.add(row.stamp)


# -- the screen and its history ----------------------------------------------------------------------


def test_new_buffer_is_one_blank_screen() -> None:
    buffer = ScreenBuffer(10, 4, history_limit=100)

    assert len(buffer.rows) == 4
    assert (buffer.top, buffer.cursor.x, buffer.cursor.y, buffer.cursor.pending_wrap) == (0, 0, 0, False)
    assert (buffer.margin_top, buffer.margin_bottom) == (0, 3)
    assert buffer.used_height == 1
    assert buffer.screen_text == ["", "", "", ""]


def test_damage_is_everything_at_first_then_what_was_touched() -> None:
    buffer = ScreenBuffer(10, 4, history_limit=100)
    assert buffer.take_damage() is None
    assert buffer.take_damage() == set()

    buffer.edit(2)
    buffer.damage(0)
    assert buffer.take_damage() == {0, 2}
    assert buffer.take_damage() == set()

    buffer.edit(1)
    buffer.damage_everything()
    assert buffer.take_damage() is None
    assert buffer.take_damage() == set()


def test_edit_and_row_address_the_screen_not_the_history() -> None:
    buffer = buffer_of("a", "b", "c", "d", "e", height=3)

    assert buffer.top == 2
    assert buffer.row(0).text == "c"
    assert buffer.edit(1) is buffer.rows[3]
    assert buffer.take_damage() == {3}
    assert buffer.cursor_index == 4


def test_index_moves_down_and_scrolls_at_the_bottom() -> None:
    buffer = buffer_of("a", "b", "c")
    oldest = buffer.rows[0]

    buffer.cursor.pending_wrap = True
    buffer.index(DEFAULT_PEN)

    assert texts(buffer) == ["a", "b", "c", ""]
    assert (buffer.top, buffer.cursor.y, buffer.cursor.pending_wrap) == (1, 2, False)
    # Scrolled into history the row is still the same row, so what was drawn for it still holds.
    assert buffer.rows[0] is oldest
    assert buffer.take_damage() == {3}


def test_index_below_the_scrolling_region_stops_at_the_last_row() -> None:
    buffer = buffer_of("a", "b", "c", "d")
    buffer.set_margins(0, 1)
    buffer.cursor.y = 3

    buffer.index(DEFAULT_PEN)

    assert texts(buffer) == ["a", "b", "c", "d"]
    assert buffer.cursor.y == 3


def test_reverse_index_scrolls_down_at_the_top_of_the_region() -> None:
    buffer = buffer_of("a", "b", "c", "d", "e", height=3)
    buffer.cursor.y = 0

    buffer.reverse_index(DEFAULT_PEN)

    assert texts(buffer) == ["a", "b", "", "c", "d"]
    assert (buffer.top, buffer.cursor.y) == (2, 0)


def test_reverse_index_above_the_region_stops_at_the_first_row() -> None:
    buffer = buffer_of("a", "b", "c")
    buffer.set_margins(1, 2)
    buffer.cursor.y = 0

    buffer.reverse_index(DEFAULT_PEN)

    assert texts(buffer) == ["a", "b", "c"]
    assert buffer.cursor.y == 0


def test_scroll_up_keeps_what_leaves_the_top_as_history() -> None:
    buffer = buffer_of("a", "b", "c")

    buffer.scroll_up(2, DEFAULT_PEN)

    assert texts(buffer) == ["a", "b", "c", "", ""]
    assert buffer.top == 2
    assert buffer.screen_text == ["c", "", ""]


def test_scroll_up_by_more_than_the_screen_scrolls_the_screen_away() -> None:
    buffer = buffer_of("a", "b", "c")

    buffer.scroll_up(99, DEFAULT_PEN)

    assert texts(buffer) == ["a", "b", "c", "", "", ""]
    assert buffer.top == 3


def test_scroll_up_of_a_region_at_the_top_leaves_the_rows_below_it() -> None:
    buffer = buffer_of("a", "b", "c", "d")
    buffer.set_margins(0, 1)

    buffer.scroll_up(1, DEFAULT_PEN)

    assert texts(buffer) == ["a", "b", "", "c", "d"]
    assert buffer.screen_text == ["b", "", "c", "d"]


def test_scroll_up_of_a_region_below_the_top_keeps_no_history() -> None:
    buffer = buffer_of("a", "b", "c", "d")
    buffer.set_margins(1, 2)

    buffer.scroll_up(1, DEFAULT_PEN)

    assert texts(buffer) == ["a", "c", "", "d"]
    assert buffer.top == 0
    assert buffer.take_damage() == {1, 2}


def test_scroll_up_without_history_loses_the_rows() -> None:
    buffer = buffer_of("a", "b", "c", history_limit=0)

    buffer.scroll_up(1, DEFAULT_PEN)

    assert texts(buffer) == ["b", "c", ""]
    assert buffer.top == 0


def test_scroll_down_loses_what_leaves_the_region() -> None:
    buffer = buffer_of("a", "b", "c", "d")
    buffer.set_margins(1, 2)

    buffer.scroll_down(1, DEFAULT_PEN)

    assert texts(buffer) == ["a", "", "b", "d"]

    buffer.scroll_down(99, DEFAULT_PEN)

    assert texts(buffer) == ["a", "", "", "d"]


def test_scrolled_in_rows_take_the_erasers_background() -> None:
    buffer = buffer_of("a", "b", columns=4)

    buffer.scroll_up(1, _ON_BLUE)

    assert buffer.rows[-1].text == "    "
    assert buffer.rows[-1].pens == [_ON_BLUE] * 4


def test_insert_and_delete_lines_work_from_the_cursor_to_the_bottom_margin() -> None:
    buffer = buffer_of("a", "b", "c", "d", "e")
    buffer.set_margins(0, 3)
    buffer.cursor.y = 1

    buffer.insert_lines(2, DEFAULT_PEN)
    assert texts(buffer) == ["a", "", "", "b", "e"]

    buffer.delete_lines(1, DEFAULT_PEN)
    assert texts(buffer) == ["a", "", "b", "", "e"]

    buffer.delete_lines(99, DEFAULT_PEN)
    assert texts(buffer) == ["a", "", "", "", "e"]


def test_insert_and_delete_lines_outside_the_region_do_nothing() -> None:
    buffer = buffer_of("a", "b", "c", "d")
    buffer.set_margins(0, 1)
    buffer.cursor.y = 3

    buffer.insert_lines(1, DEFAULT_PEN)
    buffer.delete_lines(1, DEFAULT_PEN)

    assert texts(buffer) == ["a", "b", "c", "d"]


def test_a_row_stops_running_on_once_its_continuation_is_moved_away() -> None:
    scrolled = buffer_of("a", "b", "c", "d")
    scrolled.rows[1].wrapped = True
    scrolled.set_margins(0, 1)
    scrolled.scroll_up(1, DEFAULT_PEN)
    assert not scrolled.rows[1].wrapped

    opened = buffer_of("a", "b", "c", "d")
    opened.rows[0].wrapped = True
    opened.cursor.y = 1
    opened.insert_lines(1, DEFAULT_PEN)
    assert not opened.rows[0].wrapped

    erased = buffer_of("a", "b", "c", "d")
    erased.rows[0].wrapped = True
    erased.erase_rows(1, 2, DEFAULT_PEN)
    assert not erased.rows[0].wrapped


def test_erase_rows_replaces_them() -> None:
    buffer = buffer_of("a", "b", "c", "d", columns=3)
    before = buffer.rows[1]

    buffer.erase_rows(1, 2, _ON_BLUE)
    buffer.erase_rows(3, 2, _ON_BLUE)

    assert texts(buffer) == ["a", "", "", "d"]
    assert buffer.rows[1] is not before
    assert buffer.rows[1].pens == [_ON_BLUE] * 3
    assert buffer.take_damage() == {1, 2}


def test_fill_covers_the_screen_and_only_the_screen() -> None:
    buffer = buffer_of("a", "b", "c", "d", columns=3, height=2)

    buffer.fill("E")

    assert texts(buffer) == ["a", "b", "EEE", "EEE"]
    assert buffer.take_damage() is None


def test_used_height_reaches_the_last_thing_to_see_or_the_cursor() -> None:
    buffer = buffer_of("a", "b", height=6)
    assert buffer.used_height == 2

    buffer.cursor.y = 3
    assert buffer.used_height == 4

    buffer.edit(4).erase(0, 10, _ON_BLUE, 10)
    assert buffer.used_height == 5


def test_screen_text_has_no_trailing_blanks() -> None:
    buffer = buffer_of("a", "b", "c", height=2)
    buffer.edit(1).put(3, "  ", DEFAULT_PEN)

    assert buffer.screen_text == ["b", "c"]


def test_trim_history_drops_the_oldest_rows() -> None:
    buffer = buffer_of("a", "b", "c", "d", "e", "f", height=2, history_limit=100)
    buffer.history_limit = 1

    assert buffer.trim_history() == 3
    assert texts(buffer) == ["d", "e", "f"]
    assert buffer.top == 1
    assert buffer.take_damage() is None
    assert buffer.trim_history() == 0
    assert buffer.take_damage() == set()


def test_clear_history_keeps_the_screen() -> None:
    buffer = buffer_of("a", "b", "c", "d", height=2)

    assert buffer.clear_history() == 2

    assert texts(buffer) == ["c", "d"]
    assert (buffer.top, buffer.cursor_index) == (0, 1)
    assert buffer.take_damage() is None
    # With no history there is nothing to forget, and nothing to redraw.
    assert buffer.clear_history() == 0
    assert buffer.take_damage() == set()


# -- a new width -------------------------------------------------------------------------------------


def wrapped_buffer(columns: int, height: int, *lines: str) -> ScreenBuffer:
    """A buffer these logical lines were typed into, autowrap laying each out at ``columns`` wide."""
    buffer = ScreenBuffer(columns, height, history_limit=100)
    for number, line in enumerate(lines):
        if number:
            buffer.index(DEFAULT_PEN)
        cells = row_of(line).cells
        for start in range(0, max(len(cells), 1), columns):
            if start:
                buffer.index(DEFAULT_PEN)
                buffer.rows[buffer.cursor_index - 1].wrapped = True
            buffer.edit(buffer.cursor.y).put(0, cells[start : start + columns], DEFAULT_PEN)
            buffer.cursor.x = min(len(cells) - start, columns - 1)
    buffer.take_damage()
    return buffer


def test_wider_screen_rejoins_wrapped_rows() -> None:
    buffer = wrapped_buffer(5, 4, "abcdefghijkl", "next")
    assert texts(buffer) == ["abcde", "fghij", "kl", "next"]

    buffer.resize(12, 4, reflow=True, pull_history=True)

    assert texts(buffer) == ["abcdefghijkl", "next", "", ""]
    assert [row.wrapped for row in buffer.rows] == [False, False, False, False]
    assert buffer.take_damage() is None


def test_narrower_screen_wraps_long_rows() -> None:
    buffer = wrapped_buffer(12, 2, "abcdefghijkl", "next")

    buffer.resize(5, 2, reflow=True, pull_history=True)

    assert texts(buffer) == ["abcde", "fghij", "kl", "next"]
    assert [row.wrapped for row in buffer.rows] == [True, True, False, False]
    assert buffer.top == 2


def test_hard_line_breaks_survive_any_width() -> None:
    buffer = wrapped_buffer(4, 6, "abcdef", "", "gh")

    buffer.resize(3, 6, reflow=True, pull_history=True)
    buffer.resize(20, 6, reflow=True, pull_history=True)

    assert texts(buffer)[:3] == ["abcdef", "", "gh"]


def test_reflow_keeps_pens_with_their_cells() -> None:
    buffer = ScreenBuffer(4, 2, history_limit=100)
    buffer.edit(0).put(0, "abcd", _RED)
    buffer.rows[0].wrapped = True
    buffer.edit(1).put(0, "ef", _ON_BLUE)
    buffer.cursor.y = 1

    buffer.resize(3, 2, reflow=True, pull_history=True)

    assert texts(buffer) == ["abc", "def"]
    assert buffer.rows[0].pens == [_RED] * 3
    assert buffer.rows[1].pens == [_RED, _ON_BLUE, _ON_BLUE]


def test_reflow_never_parts_a_wide_character() -> None:
    buffer = wrapped_buffer(10, 3, "ab你好")

    buffer.resize(3, 3, reflow=True, pull_history=True)

    assert [row.cells for row in buffer.rows] == [["a", "b"], ["你", ""], ["好", ""]]
    assert [row.wrapped for row in buffer.rows] == [True, True, False]


def test_reflow_drops_the_blanks_a_line_ends_with() -> None:
    buffer = wrapped_buffer(5, 3, "abcdefg")
    buffer.edit(1).put(2, "   ", DEFAULT_PEN)

    buffer.resize(4, 3, reflow=True, pull_history=True)

    assert [row.cells for row in buffer.rows[:2]] == [["a", "b", "c", "d"], ["e", "f", "g"]]


def test_reflow_leaves_a_row_that_still_fits_alone() -> None:
    buffer = wrapped_buffer(10, 3, "short", "abcdefghij")
    short = buffer.rows[0]
    stamp = short.stamp

    buffer.resize(6, 3, reflow=True, pull_history=True)

    assert buffer.rows[0] is short
    assert short.stamp == stamp


def test_reflow_carries_the_cursor_with_its_text() -> None:
    buffer = wrapped_buffer(5, 4, "abcdefgh")
    assert (buffer.cursor.x, buffer.cursor.y) == (3, 1)

    buffer.resize(10, 4, reflow=True, pull_history=True)
    assert (buffer.cursor.x, buffer.cursor.y) == (8, 0)

    buffer.resize(3, 4, reflow=True, pull_history=True)
    assert (buffer.cursor.x, buffer.cursor.y) == (2, 2)
    assert texts(buffer)[:3] == ["abc", "def", "gh"]


def test_reflow_carries_a_cursor_waiting_to_wrap() -> None:
    buffer = wrapped_buffer(6, 3, "abcdef")
    buffer.cursor.x, buffer.cursor.pending_wrap = 5, True

    buffer.resize(3, 3, reflow=True, pull_history=True)
    assert (buffer.cursor.x, buffer.cursor.y, buffer.cursor.pending_wrap) == (2, 1, True)

    buffer.resize(10, 3, reflow=True, pull_history=True)
    assert (buffer.cursor.x, buffer.cursor.y, buffer.cursor.pending_wrap) == (6, 0, False)


def test_reflow_keeps_the_screen_on_the_line_it_began_with() -> None:
    buffer = wrapped_buffer(10, 3, "aaaaaaaa", "b", "c", "d", "e")
    assert buffer.screen_text == ["c", "d", "e"]

    buffer.resize(4, 3, reflow=True, pull_history=True)

    assert texts(buffer) == ["aaaa", "aaaa", "b", "c", "d", "e"]
    assert buffer.screen_text == ["c", "d", "e"]
    assert (buffer.top, buffer.cursor.y) == (3, 2)


def test_reflow_moves_the_screen_down_rather_than_lose_what_is_below_it() -> None:
    buffer = wrapped_buffer(10, 3, "cccccccc", "d", "e")

    buffer.resize(4, 3, reflow=True, pull_history=True)

    assert texts(buffer) == ["cccc", "cccc", "d", "e"]
    assert buffer.screen_text == ["cccc", "d", "e"]
    assert buffer.cursor.y == 2


def test_reflow_never_leaves_the_cursor_above_the_screen() -> None:
    buffer = wrapped_buffer(10, 3, "x", "bbbbbbbb", "cccccccc")
    buffer.cursor.x, buffer.cursor.y = 1, 0

    buffer.resize(4, 3, reflow=True, pull_history=True)

    assert (buffer.top, buffer.cursor.x, buffer.cursor.y) == (0, 1, 0)
    assert buffer.screen_text == ["x", "bbbb", "bbbb"]


def test_reflow_forgets_blank_rows_below_the_cursor_and_restores_the_screen() -> None:
    buffer = wrapped_buffer(10, 5, "abcdefgh")

    buffer.resize(4, 5, reflow=True, pull_history=True)

    assert len(buffer.rows) == 5
    assert texts(buffer) == ["abcd", "efgh", "", "", ""]


def test_resize_without_reflow_keeps_the_layout_until_a_reflow_is_asked_for() -> None:
    buffer = wrapped_buffer(5, 3, "abcdefgh")

    buffer.resize(10, 3, reflow=False, pull_history=False)
    assert texts(buffer) == ["abcde", "fgh", ""]
    assert buffer.columns == 10

    # Same width again: the layout is still owed.
    buffer.resize(10, 3, reflow=True, pull_history=False)
    assert texts(buffer) == ["abcdefgh", "", ""]

    buffer.resize(10, 3, reflow=True, pull_history=False)
    assert texts(buffer) == ["abcdefgh", "", ""]


def test_resize_keeps_the_cursor_inside_the_new_width() -> None:
    buffer = buffer_of("abcdefgh")

    buffer.resize(4, 1, reflow=False, pull_history=False)

    assert buffer.cursor.x == 3


def test_resize_resets_the_margins() -> None:
    buffer = buffer_of("a", "b", "c", "d")
    buffer.set_margins(1, 2)

    buffer.resize(10, 6, reflow=True, pull_history=False)

    assert (buffer.margin_top, buffer.margin_bottom) == (0, 5)


# -- a new height ------------------------------------------------------------------------------------


def test_taller_screen_reveals_history() -> None:
    buffer = buffer_of("a", "b", "c", "d", "e", height=3)

    buffer.resize(10, 4, reflow=True, pull_history=True)
    assert (buffer.screen_text, buffer.cursor.y) == (["b", "c", "d", "e"], 3)

    buffer.resize(10, 8, reflow=True, pull_history=True)
    assert (buffer.screen_text, buffer.cursor.y) == (["a", "b", "c", "d", "e", "", "", ""], 4)
    assert buffer.top == 0


def test_taller_screen_of_a_repainting_host_gains_blank_rows_below() -> None:
    buffer = buffer_of("a", "b", "c", "d", "e", height=3)

    buffer.resize(10, 5, reflow=False, pull_history=False)

    assert (buffer.screen_text, buffer.top, buffer.cursor.y) == (["c", "d", "e", "", ""], 2, 2)


def test_shorter_screen_gives_up_blank_rows_below_the_output_first() -> None:
    buffer = buffer_of("a", "b", height=5)

    buffer.resize(10, 3, reflow=True, pull_history=True)

    assert (texts(buffer), buffer.top, buffer.cursor.y) == (["a", "b", ""], 0, 1)


def test_shorter_screen_then_sends_rows_into_history() -> None:
    buffer = buffer_of("a", "b", "c", "d", height=5)

    buffer.resize(10, 2, reflow=True, pull_history=True)

    assert texts(buffer) == ["a", "b", "c", "d"]
    assert (buffer.screen_text, buffer.top, buffer.cursor.y) == (["c", "d"], 2, 1)


def test_shorter_screen_never_scrolls_the_cursor_row_away() -> None:
    buffer = buffer_of("a", "b", "c", "d", "e")
    buffer.cursor.y = 1

    buffer.resize(10, 2, reflow=True, pull_history=True)

    assert (buffer.screen_text, buffer.top, buffer.cursor.y) == (["b", "c"], 1, 0)
    assert texts(buffer) == ["a", "b", "c"]


def test_shorter_screen_leaves_the_history_limit_to_the_next_trim() -> None:
    """Whoever numbers the rows must hear of every trim, and a resize has no way of telling."""
    buffer = buffer_of("a", "b", "c", "d", history_limit=1)

    buffer.resize(10, 1, reflow=True, pull_history=True)

    assert texts(buffer) == ["a", "b", "c", "d"]
    assert buffer.top == 3
    assert buffer.trim_history() == 2
    assert (texts(buffer), buffer.top) == (["c", "d"], 1)


def test_shorter_screen_without_history_loses_the_rows_that_leave_it() -> None:
    buffer = buffer_of("a", "b", "c", "d", history_limit=0)

    buffer.resize(10, 2, reflow=False, pull_history=False)

    assert (texts(buffer), buffer.top, buffer.cursor.y) == (["c", "d"], 0, 1)


# -- a host that says where the screen is ------------------------------------------------------------


def test_anchor_moves_the_screen_so_the_cursor_row_is_where_the_host_says() -> None:
    buffer = buffer_of("a", "b", "c", "d", "e", "f", height=4)
    assert (buffer.top, buffer.cursor.y) == (2, 3)

    buffer.anchor_cursor_row(1)

    assert (buffer.top, buffer.cursor.y, buffer.cursor_index) == (4, 1, 5)
    assert buffer.screen_text == ["e", "f", "", ""]
    assert len(buffer.rows) == 8
    assert buffer.take_damage() is None


def test_anchor_that_agrees_changes_nothing() -> None:
    buffer = buffer_of("a", "b", "c", "d", "e", "f", height=4)

    buffer.anchor_cursor_row(3)

    assert (buffer.top, buffer.cursor.y) == (2, 3)
    assert buffer.take_damage() == set()


def test_anchor_is_held_to_the_screen_and_to_the_rows_there_are() -> None:
    buffer = buffer_of("a", "b", "c", "d", "e", "f", height=4)
    buffer.cursor.y = 0

    buffer.anchor_cursor_row(99)
    assert (buffer.top, buffer.cursor.y) == (0, 2)
    assert texts(buffer) == ["a", "b", "c", "d"]

    buffer.anchor_cursor_row(-5)
    assert (buffer.top, buffer.cursor.y) == (2, 0)
    assert texts(buffer) == ["a", "b", "c", "d", "", ""]
