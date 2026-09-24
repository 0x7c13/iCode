# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The emulator's screen: what printing, the C0 controls and the editing sequences do to it."""

from __future__ import annotations

import pytest
from rich.cells import cell_len

from chrys.app.tui.terminal.emulator import DEFAULT_PEN, Attribute, Pen, Rgb, TerminalEmulator
from chrys.app.tui.terminal.emulator.core import DEFAULT_HISTORY_LIMIT


def fed(*streams: str, columns: int = 20, lines: int = 5, history_limit: int = 100) -> TerminalEmulator:
    emulator = TerminalEmulator(columns, lines, history_limit=history_limit)
    for stream in streams:
        emulator.feed(stream)
    return emulator


def screen(emulator: TerminalEmulator) -> list[str]:
    return emulator.buffer.screen_text


def everything(emulator: TerminalEmulator) -> list[str]:
    """History and screen, top to bottom."""
    return [row.text.rstrip() for row in emulator.buffer.rows]


def cursor(emulator: TerminalEmulator) -> tuple[int, int]:
    return emulator.buffer.cursor.x, emulator.buffer.cursor.y


def modes(emulator: TerminalEmulator) -> tuple[object, ...]:
    """Everything a front end reads off the emulator besides the screen."""
    return (
        emulator.pen,
        emulator.cursor_shape,
        emulator.cursor_visible,
        emulator.mouse_tracking,
        emulator.mouse_encoding,
        emulator.application_cursor_keys,
        emulator.bracketed_paste,
        emulator.focus_reporting,
        emulator.alternate_scroll,
        emulator.key_protocol,
    )


def lines_as_typed(emulator: TerminalEmulator) -> list[str]:
    """The rows with autowrap undone: what the program wrote, line by line."""
    lines: list[str] = []
    continues = False
    for row in emulator.buffer.rows:
        if continues:
            lines[-1] += row.text
        else:
            lines.append(row.text)
        continues = row.wrapped
    return [line.rstrip() for line in lines]


# -- printing ----------------------------------------------------------------------------------------


def test_text_lands_at_the_cursor_which_moves_past_it() -> None:
    emulator = fed("hello")

    assert screen(emulator) == ["hello", "", "", "", ""]
    assert cursor(emulator) == (5, 0)


def test_text_is_drawn_with_the_pen_in_force() -> None:
    emulator = fed("a\x1b[1;31mb\x1b[48;2;1;2;3mc\x1b[0md")

    assert emulator.buffer.row(0).pens == [
        DEFAULT_PEN,
        Pen(1, None, Attribute.BOLD),
        Pen(1, Rgb(1, 2, 3), Attribute.BOLD),
        DEFAULT_PEN,
    ]
    assert emulator.pen == DEFAULT_PEN


def test_line_too_long_for_the_row_wraps_onto_the_next() -> None:
    emulator = fed("abcdefg", columns=5)

    assert screen(emulator)[:2] == ["abcde", "fg"]
    assert [row.wrapped for row in emulator.buffer.rows[:2]] == [True, False]
    assert cursor(emulator) == (2, 1)


def test_wrap_waits_for_the_next_character() -> None:
    emulator = fed("abcde", columns=5)

    assert cursor(emulator) == (4, 0)
    assert emulator.buffer.cursor.pending_wrap
    assert not emulator.buffer.row(0).wrapped
    assert screen(emulator)[1] == ""


@pytest.mark.parametrize(
    ("then", "first_row", "position"),
    [
        ("\rX", "Xbcde", (1, 0)),
        ("\bX", "abcXe", (4, 0)),
        ("\x1b[DX", "abcXe", (4, 0)),
        ("\x1b[1;5HX", "abcdX", (4, 0)),
        ("\x1b[K", "abcd", (4, 0)),
        ("\x1b[X", "abcd", (4, 0)),
        ("\x1b[P", "abcd", (4, 0)),
        ("\x1b[@X", "abcdX", (4, 0)),
    ],
)
def test_pending_wrap_is_given_up_when_the_cursor_is_put_somewhere(
    then: str, first_row: str, position: tuple[int, int]
) -> None:
    emulator = fed("abcde", then, columns=5)

    assert screen(emulator)[:2] == [first_row, ""]
    assert cursor(emulator) == position


def test_wrap_at_the_bottom_scrolls_and_still_marks_the_row_as_continued() -> None:
    emulator = fed("\x1b[3;1Habcdefgh", columns=5, lines=3)

    assert everything(emulator) == ["", "", "abcde", "fgh"]
    assert [row.wrapped for row in emulator.buffer.rows] == [False, False, True, False]
    assert lines_as_typed(emulator) == ["", "", "abcdefgh"]


def test_wrap_below_the_scrolling_region_stays_on_the_last_row() -> None:
    emulator = fed("\x1b[1;2r\x1b[3;1Habcdefg", columns=5, lines=3)

    assert screen(emulator) == ["", "", "fgcde"]
    assert not emulator.buffer.row(2).wrapped


def test_without_autowrap_the_last_column_is_overwritten() -> None:
    emulator = fed("\x1b[?7labcdef", "g", columns=5)

    assert screen(emulator)[:2] == ["abcdg", ""]
    assert cursor(emulator) == (4, 0)
    assert not emulator.buffer.row(0).wrapped


def test_switching_autowrap_off_gives_up_a_pending_wrap() -> None:
    emulator = fed("abcde\x1b[?7lX", columns=5)

    assert screen(emulator)[:2] == ["abcdX", ""]


def test_switching_autowrap_off_leaves_the_last_column_the_one_just_written() -> None:
    emulator = fed("abcde\x1b[?7l", "\u0301", columns=5)

    assert emulator.buffer.row(0).cells == ["a", "b", "c", "d", "e\u0301"]


def test_wrap_saved_with_the_cursor_is_not_made_once_autowrap_is_off() -> None:
    emulator = fed("abcde\x1b7\x1b[?7l\x1b8X", columns=5)

    assert screen(emulator)[:2] == ["abcdX", ""]


def test_autowrap_is_asked_about_when_the_next_character_comes() -> None:
    """As xterm has it: writing the last column decides nothing yet."""
    emulator = fed("\x1b[?7labcde\x1b[?7hX", columns=5)

    assert screen(emulator)[:2] == ["abcde", "X"]
    assert emulator.buffer.row(0).wrapped


def test_hard_newline_after_a_wrapped_bottom_line_starts_a_new_row() -> None:
    emulator = TerminalEmulator(100, 24)
    for number in range(23):
        emulator.feed(f"old {number}\r\n")
    emulator.feed("PS D:\\Repos\\chrys> ")

    emulator.feed(
        "asdfasdf\r\n"
        "\x1b[0m\x1b[0m\x1b[31;1masdfasdf: "
        "\x1b[31;1mThe term 'asdfasdf' is not recognized as a name of a cmdlet, "
        "function, script file, or executable program.\x1b[0m\r\n"
        "\x1b[31;1m\x1b[31;1mCheck the spelling of the name, or if a path was included, "
        "verify that the path is correct and try again.\x1b[0m\r\n"
        "PS D:\\Repos\\chrys> "
    )

    typed = lines_as_typed(emulator)
    assert typed[-4] == "PS D:\\Repos\\chrys> asdfasdf"
    assert typed[-3].startswith("asdfasdf: The term")
    assert typed[-3].endswith("executable program.")
    assert typed[-2].startswith("Check the spelling")
    assert typed[-1] == "PS D:\\Repos\\chrys>"
    assert emulator.buffer.row(emulator.buffer.cursor.y).text == "PS D:\\Repos\\chrys> "


# -- characters that are not one cell ----------------------------------------------------------------


def test_wide_character_takes_two_cells() -> None:
    emulator = fed("a你b")

    assert emulator.buffer.row(0).cells == ["a", "你", "", "b"]
    assert cursor(emulator) == (4, 0)


def test_wide_character_that_does_not_fit_wraps_whole() -> None:
    emulator = fed("abcd你", columns=5)

    assert [emulator.buffer.row(y).cells for y in (0, 1)] == [["a", "b", "c", "d"], ["你", ""]]
    assert emulator.buffer.row(0).wrapped
    assert cursor(emulator) == (2, 1)


def test_wide_character_that_does_not_fit_is_written_over_the_last_two_columns_without_autowrap() -> None:
    """As a narrow one is over the last: what was printed last is on the screen, whatever it was."""
    assert fed("\x1b[?7labcd你", columns=5).buffer.row(0).cells == ["a", "b", "c", "你", ""]
    assert fed("\x1b[?7labcd你x", columns=5).buffer.row(0).cells == ["a", "b", "c", " ", "x"]


def test_wide_character_has_no_room_on_a_screen_one_column_wide() -> None:
    assert fed("你x", columns=1, lines=2).buffer.row(0).cells == ["x"]
    assert fed("\x1b[?7l你x", columns=1, lines=2).buffer.row(0).cells == ["x"]


def test_cursor_position_past_the_end_of_a_row_pads_with_blanks() -> None:
    emulator = fed("\x1b[1;1Habcd\x1b[1;10HX")

    assert emulator.buffer.row(0).text == "abcd     X"
    assert len(emulator.buffer.row(0).cells) == 10


def test_cursor_position_past_the_end_of_a_wide_row_pads_by_cells() -> None:
    emulator = fed("\x1b[1;1H你是谁?\x1b[1;10HX")

    assert emulator.buffer.row(0).text == "你是谁?  X"
    assert len(emulator.buffer.row(0).cells) == 10


def test_wide_characters_over_narrow_ones_keep_the_row_adding_up() -> None:
    emulator = fed("\x1b[1;1HABCDEFGHIJ\x1b[1;1H你好")

    assert emulator.buffer.row(0).text == "你好EFGHIJ"
    assert len(emulator.buffer.row(0).cells) == 10


def test_narrow_characters_over_wide_ones_blank_the_half_left_over() -> None:
    emulator = fed("\x1b[1;1H你好世界AB\x1b[1;1Hxyz")

    assert emulator.buffer.row(0).text == "xyz 世界AB"
    assert len(emulator.buffer.row(0).cells) == 10


def test_writing_on_the_right_half_of_a_wide_character_blanks_the_left() -> None:
    emulator = fed("\x1b[1;1H你BCD\x1b[1;2HX")

    assert emulator.buffer.row(0).text == " XBCD"
    assert len(emulator.buffer.row(0).cells) == 5


def test_combining_character_joins_the_cell_before_it() -> None:
    emulator = fed("e\u0301x")

    assert emulator.buffer.row(0).cells == ["e\u0301", "x"]
    assert cursor(emulator) == (2, 0)


def test_combining_character_arriving_later_still_joins() -> None:
    assert fed("e", "\u0301x").buffer.row(0).cells == ["e\u0301", "x"]
    assert fed("你", "\u0301x").buffer.row(0).cells == ["你\u0301", "", "x"]


def test_combining_character_joins_the_last_column_while_the_wrap_is_pending() -> None:
    emulator = fed("abc", "\u0301", columns=3)

    assert emulator.buffer.row(0).cells == ["a", "b", "c\u0301"]
    assert emulator.buffer.cursor.pending_wrap


def test_combining_character_joins_the_last_column_without_autowrap_too() -> None:
    emulator = fed("\x1b[?7labc", "\u0301", columns=3)

    assert emulator.buffer.row(0).cells == ["a", "b", "c\u0301"]

    emulator.feed("x")

    assert emulator.buffer.row(0).cells == ["a", "b", "x"]


def test_combining_character_joins_the_column_before_a_cursor_put_in_the_last_one() -> None:
    """The cursor is in the same place as when that column has just been written, and is not past it."""
    emulator = fed("\x1b[?7labc\x1b[1;3H", "\u0301", columns=3)

    assert emulator.buffer.row(0).cells == ["a", "b\u0301", "c"]


def test_combining_character_with_nothing_before_it_is_dropped() -> None:
    emulator = fed("\u0301x")

    assert emulator.buffer.row(0).cells == ["x"]


def test_zero_width_joiner_sequence_is_one_character() -> None:
    family = "👨\u200d👩\u200d👧"

    assert fed(f"{family}x").buffer.row(0).cells == [family, "", "x"]
    # However the sequence was cut up on its way here.
    assert fed("👨\u200d", "👩\u200d👧x").buffer.row(0).cells == [family, "", "x"]
    assert fed("👨", "\u200d", "👩", "\u200d", "👧", "x").buffer.row(0).cells == [family, "", "x"]


def test_zero_width_joiner_takes_whatever_comes_next() -> None:
    """Which is how the row is measured when it is drawn, so that is how it is laid out."""
    assert fed("👩\u200dax").buffer.row(0).cells == ["👩\u200da", "", "x"]
    assert fed("👩\u200d", "ax").buffer.row(0).cells == ["👩\u200da", "", "x"]


def test_zero_width_joiner_taken_by_another_joins_nothing() -> None:
    text = "❤\u200d\u200d你"
    cells = fed(text).buffer.row(0).cells

    assert cells == ["❤\u200d\u200d", "你", ""]
    assert len(cells) == cell_len(text)
    assert fed("❤\u200d\u200d", "你").buffer.row(0).cells == cells


def test_variation_selector_widens_its_character() -> None:
    emulator = fed("a☺\ufe0fb")

    assert emulator.buffer.row(0).cells == ["a", "☺\ufe0f", "", "b"]
    assert cursor(emulator) == (4, 0)


def test_variation_selector_arriving_later_still_widens_its_character() -> None:
    emulator = fed("a☺", "\ufe0f", "b")

    assert emulator.buffer.row(0).cells == ["a", "☺\ufe0f", "", "b"]
    assert cursor(emulator) == (4, 0)


def test_character_widened_later_takes_the_cell_after_it() -> None:
    emulator = fed("☺bc\x1b[1;2H", "\ufe0f")

    assert emulator.buffer.row(0).cells == ["☺\ufe0f", "", "c"]
    assert cursor(emulator) == (2, 0)


def test_character_widened_later_keeps_its_pen() -> None:
    emulator = fed("\x1b[31m☺\x1b[m", "\ufe0f")

    assert emulator.buffer.row(0).pens == [Pen(foreground=1)] * 2


def test_character_widened_later_in_the_last_column_moves_to_the_next_row() -> None:
    emulator = fed("ab☺", "\ufe0f", "c", columns=3)

    assert [row.cells for row in emulator.buffer.rows[:2]] == [["a", "b"], ["☺\ufe0f", "", "c"]]
    assert emulator.buffer.row(0).wrapped
    assert cursor(emulator) == (2, 1)
    assert emulator.buffer.cursor.pending_wrap


def test_character_widened_later_in_the_last_column_without_autowrap_is_written_over_the_last_two() -> None:
    emulator = fed("\x1b[?7lab☺", "\ufe0f", columns=3)

    assert emulator.buffer.row(0).cells == ["a", "☺\ufe0f", ""]
    assert emulator.buffer.row(0).cells == fed("\x1b[?7lab☺\ufe0f", columns=3).buffer.row(0).cells
    assert cursor(emulator) == (2, 0)


def test_character_widened_later_up_to_the_last_column_leaves_the_wrap_pending() -> None:
    emulator = fed("a☺", "\ufe0f", columns=3)

    assert emulator.buffer.row(0).cells == ["a", "☺\ufe0f", ""]
    assert cursor(emulator) == (2, 0)
    assert emulator.buffer.cursor.pending_wrap


def test_character_widened_later_in_insert_mode_pushes_the_rest_of_the_row_right() -> None:
    emulator = fed("xyz\r\x1b[4h☺", "\ufe0f")

    assert emulator.buffer.row(0).cells == ["☺\ufe0f", "", "x", "y", "z"]
    assert cursor(emulator) == (2, 0)


@pytest.mark.parametrize("columns", [6, 7, 20])
@pytest.mark.parametrize("mode", ["", "\x1b[?7l", "\x1b[4h"], ids=["autowrap", "no-autowrap", "insert"])
@pytest.mark.parametrize(
    "text",
    [
        "a☺\ufe0fb",
        "abcd☺\ufe0fxyz",
        "abcde☺\ufe0fxyz",
        "abcde1\ufe0f\u20e3",
        "abcd👨\u200d👩\u200d👧z",
        "e\u0301\u0302x",
        "abcdef\u0301\u0302g",
        "abcde你\u0301\u200d👧x",
        "👩\u200dab",
        "❤\u200d\u200d你x",
        # And what REP repeats is the character however late it was completed.
        "☺\ufe0f\x1b[2b",
        "abcde\u0301\x1b[2bx",
        "abc👩\u200d👧\x1b[b",
    ],
)
def test_screen_does_not_depend_on_where_the_output_was_cut(text: str, mode: str, columns: int) -> None:
    """A pseudoterminal hands output over in whatever pieces its reads end on."""

    def outcome(*streams: str) -> tuple[object, ...]:
        emulator = fed(mode, *streams, columns=columns)
        position = emulator.buffer.cursor
        rows = [(row.cells, row.pens, row.wrapped) for row in emulator.buffer.rows]
        return rows, position.x, position.y, position.pending_wrap

    whole = outcome(text)

    for cut in range(1, len(text)):
        assert outcome(text[:cut], text[cut:]) == whole, f"cut before {text[cut:]!r}"


def test_skin_tone_modifier_joins_its_emoji() -> None:
    assert fed("👍🏽x").buffer.row(0).cells == ["👍🏽", "", "x"]


def test_repeat_prints_the_last_character_again() -> None:
    assert screen(fed("a\x1b[3b"))[0] == "aaaa"
    assert screen(fed("ab\x1b[b"))[0] == "abb"
    assert fed("你\x1b[2b").buffer.row(0).cells == ["你", "", "你", "", "你", ""]
    assert fed("e\u0301\x1b[2b").buffer.row(0).cells == ["e\u0301"] * 3


def test_repeat_prints_a_character_completed_later_as_completed() -> None:
    widened = fed("☺", "\ufe0f\x1b[2b")

    assert widened.buffer.row(0).cells == ["☺\ufe0f", ""] * 3
    assert cursor(widened) == (6, 0)
    assert fed("e", "\u0301\x1b[2b").buffer.row(0).cells == ["e\u0301"] * 3
    assert fed("👩\u200d", "👧\x1b[b").buffer.row(0).cells == ["👩\u200d👧", ""] * 2


def test_repeat_with_nothing_printed_does_nothing() -> None:
    assert screen(fed("\x1b[3b"))[0] == ""


def test_repeat_is_held_to_one_screenful() -> None:
    emulator = fed("a\x1b[65535b", columns=4, lines=2, history_limit=100)

    assert emulator.buffer.used_height <= 4


def test_insert_mode_pushes_the_rest_of_the_row_right() -> None:
    emulator = fed("ab\r\x1b[4hX")
    assert screen(emulator)[0] == "Xab"

    emulator.feed("\r\x1b[4lY")
    assert screen(emulator)[0] == "Yab"


def test_insert_mode_loses_what_is_pushed_off_the_row() -> None:
    emulator = fed("abcde\r\x1b[4hXY", columns=5)

    assert screen(emulator)[:2] == ["XYabc", ""]


# -- C0 controls -------------------------------------------------------------------------------------


def test_line_feed_keeps_the_column() -> None:
    emulator = fed("ab\ncd")

    assert screen(emulator)[:2] == ["ab", "  cd"]


def test_line_feed_in_new_line_mode_returns_the_carriage() -> None:
    emulator = fed("\x1b[20hab\ncd")
    assert screen(emulator)[:2] == ["ab", "cd"]

    emulator.feed("\x1b[20l\nef")
    assert screen(emulator)[2] == "  ef"


@pytest.mark.parametrize("control", ["\x0b", "\x0c"])
def test_vertical_tab_and_form_feed_are_line_feeds(control: str) -> None:
    assert everything(fed(f"first{control}second")) == everything(fed("first\nsecond"))
    assert screen(fed(f"first{control}\rsecond"))[:2] == ["first", "second"]


def test_carriage_return_and_backspace() -> None:
    assert screen(fed("abc\rX"))[0] == "Xbc"
    assert screen(fed("abc\b\bX"))[0] == "aXc"
    assert cursor(fed("a\b\b\b")) == (0, 0)


def test_horizontal_tab_moves_without_writing() -> None:
    emulator = fed("a\tb")

    assert emulator.buffer.row(0).text == "a       b"
    assert cursor(emulator) == (9, 0)


def test_bell_is_swallowed() -> None:
    emulator = fed("before", "\x07", "\x07\x07after")

    assert emulator.buffer.row(0).text == "beforeafter"
    assert cursor(emulator) == (len("beforeafter"), 0)


@pytest.mark.parametrize("control", [*map(chr, range(0x20)), "\x7f", "\x80", "\x9b", "\x9f"])
def test_control_characters_are_never_stored(control: str) -> None:
    emulator = fed(f"before{control}after")

    for row in emulator.buffer.rows:
        assert control not in row.text
        assert all(cell == "" or cell.isprintable() for cell in row.cells)


def test_shift_out_and_shift_in_pick_the_character_set() -> None:
    assert screen(fed("\x1b)0q\x0eq\x0fq"))[0] == "q─q"


def test_line_feed_at_the_bottom_adds_a_row_of_history() -> None:
    emulator = fed("line 1\r\nline 2\r\nline 3", "\r\nnext prompt", lines=3)

    assert everything(emulator) == ["line 1", "line 2", "line 3", "next prompt"]
    assert emulator.buffer.top == 1


def test_blank_lines_at_the_bottom_are_kept() -> None:
    emulator = fed("[line 1]\r\n[line 2]\r\n[line 3]\r\n\r\n[next]", lines=3)

    assert everything(emulator) == ["[line 1]", "[line 2]", "[line 3]", "", "[next]"]


def test_line_feed_at_the_bottom_of_a_region_scrolls_only_the_region() -> None:
    emulator = fed("a\r\nb\r\nc\r\nd\r\ne", "\x1b[2;4r\x1b[4;4H\nX")

    assert screen(emulator) == ["a", "c", "d", "   X", "e"]
    assert cursor(emulator) == (4, 3)
    assert emulator.buffer.top == 0


def test_line_feed_in_a_region_at_the_top_keeps_history_and_the_rows_below() -> None:
    emulator = fed("a\r\nb\r\nstatus", "\x1b[1;2r\x1b[2;4H\nX", lines=3)

    assert everything(emulator) == ["a", "b", "   X", "status"]
    assert screen(emulator) == ["b", "   X", "status"]
    assert cursor(emulator) == (4, 1)


def test_index_next_line_and_reverse_index() -> None:
    assert screen(fed("ab\x1bDcd"))[:2] == ["ab", "  cd"]
    assert screen(fed("ab\x1bEcd"))[:2] == ["ab", "cd"]
    assert screen(fed("ab\r\ncd\x1bMX"))[:2] == ["abX", "cd"]


def test_reverse_index_at_the_top_scrolls_the_screen_down() -> None:
    emulator = fed("one\r\ntwo\r\nthree", "\x1b[1;1H\x1bM", lines=3)

    assert screen(emulator) == ["", "one", "two"]


def test_reverse_index_scrolls_the_screen_and_not_the_history() -> None:
    emulator = fed("old0\r\nold1\r\nscreen0\r\nscreen1\r\nscreen2", "\x1b[1;1H\x1bM", lines=3)

    assert everything(emulator) == ["old0", "old1", "", "screen0", "screen1"]
    assert emulator.buffer.top == 2


# -- moving the cursor -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("\x1b[A", (5, 1)),
        ("\x1b[2A", (5, 0)),
        ("\x1b[99A", (5, 0)),
        ("\x1b[B", (5, 3)),
        ("\x1b[99B", (5, 4)),
        ("\x1b[C", (6, 2)),
        ("\x1b[99C", (19, 2)),
        ("\x1b[D", (4, 2)),
        ("\x1b[99D", (0, 2)),
        ("\x1b[0D", (4, 2)),
        ("\x1b[E", (0, 3)),
        ("\x1b[2F", (0, 0)),
        ("\x1b[G", (0, 2)),
        ("\x1b[9G", (8, 2)),
        ("\x1b[99G", (19, 2)),
        ("\x1b[9`", (8, 2)),
        ("\x1b[3a", (8, 2)),
        ("\x1b[d", (5, 0)),
        ("\x1b[4d", (5, 3)),
        ("\x1b[99d", (5, 4)),
        ("\x1b[2e", (5, 4)),
        ("\x1b[H", (0, 0)),
        ("\x1b[4H", (0, 3)),
        ("\x1b[;7H", (6, 0)),
        ("\x1b[4;7H", (6, 3)),
        ("\x1b[4;7f", (6, 3)),
        ("\x1b[99;99H", (19, 4)),
        ("\x1b[0;0H", (0, 0)),
    ],
)
def test_cursor_movement(sequence: str, expected: tuple[int, int]) -> None:
    assert cursor(fed("\x1b[3;6H", sequence)) == expected


def test_scrolling_region_is_set_and_homes_the_cursor() -> None:
    emulator = fed("\x1b[3;3H\x1b[2;4r")

    assert (emulator.buffer.margin_top, emulator.buffer.margin_bottom) == (1, 3)
    assert cursor(emulator) == (0, 0)


@pytest.mark.parametrize("sequence", ["\x1b[3;3r", "\x1b[4;2r", "\x1b[9;12r"])
def test_scrolling_region_that_is_no_region_is_ignored(sequence: str) -> None:
    emulator = fed("\x1b[2;4r\x1b[3;3H", sequence)

    assert (emulator.buffer.margin_top, emulator.buffer.margin_bottom) == (1, 3)
    assert cursor(emulator) == (2, 2)


@pytest.mark.parametrize("sequence", ["\x1b[r", "\x1b[1;99r", "\x1b[0;0r"])
def test_scrolling_region_defaults_to_the_whole_screen(sequence: str) -> None:
    emulator = fed("\x1b[2;4r", sequence)

    assert (emulator.buffer.margin_top, emulator.buffer.margin_bottom) == (0, 4)


def test_relative_movement_stops_at_a_margin_it_starts_inside_of() -> None:
    assert cursor(fed("\x1b[2;4r\x1b[3;1H\x1b[99A")) == (0, 1)
    assert cursor(fed("\x1b[2;4r\x1b[3;1H\x1b[99B")) == (0, 3)
    # From outside the region there is no margin in the way.
    assert cursor(fed("\x1b[3;4r\x1b[1;1H\x1b[99B")) == (0, 3)
    assert cursor(fed("\x1b[2;3r\x1b[5;1H\x1b[99A")) == (0, 1)
    assert cursor(fed("\x1b[2;3r\x1b[5;1H\x1b[99B")) == (0, 4)
    assert cursor(fed("\x1b[3;4r\x1b[1;1H\x1b[2;1H\x1b[99A")) == (0, 0)


def test_origin_mode_counts_rows_from_the_top_margin_and_stays_inside() -> None:
    emulator = fed("\x1b[2;4r\x1b[?6h")
    assert cursor(emulator) == (0, 1)

    emulator.feed("\x1b[2;3H")
    assert cursor(emulator) == (2, 2)

    emulator.feed("\x1b[99;1H")
    assert cursor(emulator) == (0, 3)

    emulator.feed("\x1b[2d")
    assert cursor(emulator) == (0, 2)

    emulator.feed("\x1b[5G")
    assert cursor(emulator) == (4, 2)

    emulator.feed("\x1b[?6l")
    assert cursor(emulator) == (0, 0)


# -- tab stops ---------------------------------------------------------------------------------------


def test_tab_stops_start_every_eight_columns() -> None:
    emulator = fed(columns=30)
    stops = []
    for _ in range(5):
        emulator.feed("\t")
        stops.append(emulator.buffer.cursor.x)

    assert stops == [8, 16, 24, 29, 29]


def test_tab_stop_set_and_cleared() -> None:
    emulator = fed("\x1b[1;4H\x1bH\r\t")
    assert cursor(emulator) == (3, 0)

    emulator.feed("\x1b[g\r\t")
    assert cursor(emulator) == (8, 0)

    emulator.feed("\x1b[3g\r\t")
    assert cursor(emulator) == (19, 0)


def test_tabbing_forward_and_backward_by_count() -> None:
    emulator = fed("\x1b[2I", columns=40)
    assert cursor(emulator) == (16, 0)

    emulator.feed("\x1b[1;30H\x1b[Z")
    assert cursor(emulator) == (24, 0)

    emulator.feed("\x1b[2Z")
    assert cursor(emulator) == (8, 0)

    emulator.feed("\x1b[99Z")
    assert cursor(emulator) == (0, 0)


# -- erasing and editing -----------------------------------------------------------------------------

_FIVE_ROWS = "aaaaa\r\nbbbbb\r\nccccc\r\nddddd\r\neeeee"


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("\x1b[J", ["aaaaa", "bbbbb", "cc", "", ""]),
        ("\x1b[0J", ["aaaaa", "bbbbb", "cc", "", ""]),
        ("\x1b[1J", ["", "", "   cc", "ddddd", "eeeee"]),
        ("\x1b[2J", ["", "", "", "", ""]),
        ("\x1b[?J", ["aaaaa", "bbbbb", "cc", "", ""]),
        ("\x1b[K", ["aaaaa", "bbbbb", "cc", "ddddd", "eeeee"]),
        ("\x1b[1K", ["aaaaa", "bbbbb", "   cc", "ddddd", "eeeee"]),
        ("\x1b[2K", ["aaaaa", "bbbbb", "", "ddddd", "eeeee"]),
        ("\x1b[?2K", ["aaaaa", "bbbbb", "", "ddddd", "eeeee"]),
        ("\x1b[X", ["aaaaa", "bbbbb", "cc cc", "ddddd", "eeeee"]),
        ("\x1b[2X", ["aaaaa", "bbbbb", "cc  c", "ddddd", "eeeee"]),
        ("\x1b[99X", ["aaaaa", "bbbbb", "cc", "ddddd", "eeeee"]),
        ("\x1b[P", ["aaaaa", "bbbbb", "cccc", "ddddd", "eeeee"]),
        ("\x1b[99P", ["aaaaa", "bbbbb", "cc", "ddddd", "eeeee"]),
        ("\x1b[@", ["aaaaa", "bbbbb", "cc ccc", "ddddd", "eeeee"]),
        ("\x1b[2@", ["aaaaa", "bbbbb", "cc  ccc", "ddddd", "eeeee"]),
        ("\x1b[L", ["aaaaa", "bbbbb", "", "ccccc", "ddddd"]),
        ("\x1b[2L", ["aaaaa", "bbbbb", "", "", "ccccc"]),
        ("\x1b[M", ["aaaaa", "bbbbb", "ddddd", "eeeee", ""]),
        ("\x1b[99M", ["aaaaa", "bbbbb", "", "", ""]),
        ("\x1b[T", ["", "aaaaa", "bbbbb", "ccccc", "ddddd"]),
        ("\x1b[2T", ["", "", "aaaaa", "bbbbb", "ccccc"]),
    ],
)
def test_erasing_and_editing(sequence: str, expected: list[str]) -> None:
    emulator = fed(_FIVE_ROWS, "\x1b[3;3H", sequence)

    assert screen(emulator) == expected
    assert emulator.buffer.top == 0


def test_erasing_leaves_the_cursor_where_it_is() -> None:
    for sequence in ("\x1b[J", "\x1b[1J", "\x1b[2J", "\x1b[3J", "\x1b[K", "\x1b[X", "\x1b[P", "\x1b[@"):
        assert cursor(fed(_FIVE_ROWS, "\x1b[3;3H", sequence)) == (2, 2)


def test_inserting_and_deleting_lines_returns_the_carriage() -> None:
    assert cursor(fed(_FIVE_ROWS, "\x1b[3;3H\x1b[L")) == (0, 2)
    assert cursor(fed(_FIVE_ROWS, "\x1b[3;3H\x1b[M")) == (0, 2)


def test_scroll_up_sends_rows_into_history() -> None:
    emulator = fed(_FIVE_ROWS, "\x1b[2S")

    assert everything(emulator) == ["aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee", "", ""]
    assert screen(emulator) == ["ccccc", "ddddd", "eeeee", "", ""]


def test_erase_display_keeps_history_and_erase_saved_lines_clears_it() -> None:
    emulator = fed("one\r\ntwo\r\nthree\r\nfour", lines=2)
    assert everything(emulator) == ["one", "two", "three", "four"]

    emulator.feed("\x1b[2J")
    assert everything(emulator) == ["one", "two", "", ""]
    assert cursor(emulator) == (4, 1)

    emulator.feed("new\x1b[3J")
    assert everything(emulator) == ["", "    new"]
    assert emulator.buffer.top == 0


def test_erased_cells_take_the_background_in_force() -> None:
    blue = Pen(background=4)
    emulator = fed("abcdef\x1b[1;31;44m\x1b[1;3H\x1b[K", columns=8)

    row = emulator.buffer.row(0)
    assert row.text == "ab      "
    assert row.pens == [DEFAULT_PEN, DEFAULT_PEN] + [blue] * 6


@pytest.mark.parametrize(
    ("sequence", "text", "painted"),
    [
        ("\x1b[2X", "ab  ef", [2, 3]),
        ("\x1b[2@", "ab  cdef", [2, 3]),
        ("\x1b[2P", "abef    ", [6, 7]),
        ("\x1b[1K", "   def", [0, 1, 2]),
    ],
)
def test_editing_a_row_paints_the_blanks_it_makes(sequence: str, text: str, painted: list[int]) -> None:
    emulator = fed("abcdef\x1b[44m\x1b[1;3H", sequence, columns=8)

    row = emulator.buffer.row(0)
    assert row.text == text
    assert [column for column, pen in enumerate(row.pens) if pen == Pen(background=4)] == painted


def test_rows_scrolled_or_erased_in_take_the_background_in_force() -> None:
    blue = Pen(background=4)

    for sequence in ("\x1b[2J", "\x1b[5;1H\n", "\x1b[S", "\x1b[T", "\x1b[L", "\x1b[M"):
        emulator = fed(_FIVE_ROWS, "\x1b[44m", sequence, columns=6)
        painted = [row for row in emulator.buffer.rows if row.pens == [blue] * 6]
        assert painted, sequence
        assert all(row.text == " " * 6 for row in painted)


def test_writing_on_the_second_row_of_a_wrapped_line() -> None:
    emulator = fed("abcdefghi\x1b[2;2HX", columns=5, lines=3)

    assert screen(emulator)[:2] == ["abcde", "fXhi"]
    assert lines_as_typed(emulator)[0] == "abcdefXhi"


def test_erasing_to_the_end_of_the_second_row_of_a_wrapped_line() -> None:
    emulator = fed("abcdefghi\x1b[2;2H\x1b[K", columns=5, lines=3)

    assert lines_as_typed(emulator)[0] == "abcdef"


def test_erasing_a_character_on_the_second_row_of_a_wrapped_line() -> None:
    emulator = fed("abcdefghi\x1b[2;2H\x1b[X", columns=5, lines=3)

    assert lines_as_typed(emulator)[0] == "abcdef hi"


def test_erasing_the_end_of_a_wrapped_row_ends_the_wrap() -> None:
    emulator = fed("abcdefghi\x1b[1;3H\x1b[K", columns=5, lines=3)

    assert screen(emulator)[:2] == ["ab", "fghi"]
    assert lines_as_typed(emulator)[:2] == ["ab", "fghi"]


def test_alignment_pattern_fills_the_screen() -> None:
    emulator = fed("\x1b#8", columns=4, lines=2)

    assert screen(emulator) == ["EEEE", "EEEE"]


# -- saving and restoring the cursor -----------------------------------------------------------------


def test_cursor_is_saved_and_restored_with_its_pen() -> None:
    emulator = fed("ab\x1b[31m\x1b7\r\x1b[0mX\x1b8Y")

    assert screen(emulator)[0] == "XbY"
    assert emulator.buffer.row(0).pens[2] == Pen(foreground=1)


def test_sco_save_and_restore() -> None:
    assert screen(fed("ab\x1b[s\rX\x1b[uY"))[0] == "XbY"


def test_restore_without_a_save_homes_the_cursor_with_a_plain_pen() -> None:
    emulator = fed("\x1b[3;3H\x1b[1;31m\x1b8")

    assert cursor(emulator) == (0, 0)
    assert emulator.pen == DEFAULT_PEN


def test_restored_cursor_keeps_a_pending_wrap() -> None:
    emulator = fed("abcde\x1b7\x1b[1;1H\x1b8X", columns=5)

    assert screen(emulator)[:2] == ["abcde", "X"]


def test_restored_cursor_keeps_origin_mode() -> None:
    emulator = fed("\x1b[2;4r\x1b[?6h\x1b[2;2H\x1b7\x1b[?6l\x1b8")
    assert cursor(emulator) == (1, 2)

    # Rows count from the top margin again.
    emulator.feed("\x1b[1;1H")
    assert cursor(emulator) == (0, 1)


def test_restore_does_not_close_a_hyperlink_opened_since() -> None:
    emulator = fed("\x1b7\x1b]8;;https://example.com\x1b\\\x1b8x")

    assert emulator.buffer.row(0).pens[0].link == "https://example.com"


def test_save_and_restore_around_a_margin_reset_keeps_the_prompt_line_intact() -> None:
    emulator = fed(
        "user@host workspace % mock-tui\r\n",
        "\x1b7\x1b[r\x1b8 ▐▛███▜▌\x1b[3CMock\x1b[1CTUI\x1b[1Cv1.2.3",
        columns=80,
    )

    assert screen(emulator)[:2] == ["user@host workspace % mock-tui", " ▐▛███▜▌   Mock TUI v1.2.3"]


# -- the alternate screen ----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [47, 1047, 1049])
def test_alternate_screen_is_blank_and_leaves_the_primary_untouched(mode: int) -> None:
    emulator = fed("one\r\ntwo\r\nthree\r\nfour", lines=3)
    assert not emulator.alternate_screen

    emulator.feed(f"\x1b[?{mode}h")
    assert emulator.alternate_screen
    assert screen(emulator) == ["", "", ""]

    emulator.feed("\x1b[H\x1b[2Jfull screen\x1b[3J")
    emulator.feed(f"\x1b[?{mode}l")
    assert not emulator.alternate_screen
    assert everything(emulator) == ["one", "two", "three", "four"]
    assert cursor(emulator) == (4, 2)


def test_alternate_screen_starts_with_the_cursor_where_it_was() -> None:
    emulator = fed("\x1b[3;7H\x1b[?1049h")

    assert cursor(emulator) == (6, 2)


def test_alternate_screen_is_new_each_time() -> None:
    emulator = fed("\x1b[?1049hleft behind\x1b[?1049l\x1b[?1049h")

    assert screen(emulator) == ["", "", "", "", ""]


def test_entering_the_alternate_screen_twice_keeps_it() -> None:
    emulator = fed("\x1b[?1049hkept\x1b[?1049h\x1b[?47h")

    assert screen(emulator)[0] == "kept"


def test_alternate_screen_has_no_history() -> None:
    emulator = fed("\x1b[?1049h", "".join(f"row {number}\r\n" for number in range(20)), lines=3)

    assert len(emulator.buffer.rows) == 3
    assert emulator.buffer.top == 0
    assert screen(emulator) == ["row 18", "row 19", ""]


def test_mode_1049_restores_the_pen_and_the_plain_modes_do_not() -> None:
    assert fed("\x1b[?1049h\x1b[31m\x1b[?1049l").pen == DEFAULT_PEN
    assert fed("\x1b[?1047h\x1b[31m\x1b[?1047l").pen == Pen(foreground=1)


def test_each_screen_has_its_own_saved_cursor() -> None:
    emulator = fed("\x1b[2;2H\x1b7\x1b[?47h\x1b[4;4H\x1b7\x1b[1;1H\x1b8")
    assert cursor(emulator) == (3, 3)

    emulator.feed("\x1b[?47l\x1b[1;1H\x1b8")
    assert cursor(emulator) == (1, 1)


# -- resets ------------------------------------------------------------------------------------------


def test_soft_reset_restores_modes_and_keeps_the_screen() -> None:
    emulator = fed(
        _FIVE_ROWS,
        "\x1b[2;4r\x1b[?6h\x1b[?1h\x1b=\x1b[4h\x1b[?25l\x1b[?7l\x1b[1;31;44m\x1b[5 q\x1b7",
        "\x1b]8;;https://example.com\x1b\\",
        "\x1b[!p",
        columns=5,
    )

    assert screen(emulator) == ["aaaaa", "bbbbb", "ccccc", "ddddd", "eeeee"]
    assert (emulator.buffer.margin_top, emulator.buffer.margin_bottom) == (0, 4)
    assert emulator.cursor_visible
    assert not emulator.application_cursor_keys
    assert emulator.pen == Pen(link="https://example.com")
    assert emulator.cursor_shape.value == "block"

    emulator.feed("\x1b[1;1HX\x1b[1;5HYZ\x1b8")
    assert screen(emulator)[:2] == ["XaaaY", "Zbbbb"]
    assert cursor(emulator) == (0, 0)


def test_hard_reset_is_a_new_terminal() -> None:
    emulator = fed(
        "one\r\ntwo\r\nthree\r\nfour",
        "\x1b[?1049h\x1b[?1000h\x1b[?1006h\x1b[?2004h\x1b[?1004h\x1b[?1h\x1b[?25l\x1b[>1u\x1b[31m\x1b[3 q\x1b(0",
        "\x1bc",
        lines=3,
    )
    fresh = TerminalEmulator(20, 3)

    assert not emulator.alternate_screen
    assert everything(emulator) == ["", "", ""]
    assert modes(emulator) == modes(fresh)
    emulator.feed("q")
    assert screen(emulator)[0] == "q"


def test_hard_reset_forgets_a_sequence_cut_short() -> None:
    # The reset arrives inside an unfinished string; what follows is text again.
    emulator = fed("\x1b]0;never finished", "\x1bc", "text")

    assert screen(emulator)[0] == "text"


# -- what a feed reports -----------------------------------------------------------------------------


def test_first_update_damages_everything_and_later_ones_what_changed() -> None:
    emulator = TerminalEmulator(20, 5)

    assert emulator.feed("").damaged is None
    assert emulator.feed("").damaged == set()
    assert emulator.feed("hello").damaged == {0}
    assert emulator.feed("\x1b[3;1Hthere").damaged == {0, 2}


def test_single_row_update_on_the_alternate_screen_damages_that_row() -> None:
    emulator = fed("\x1b[?1049h\x1b[1;1Hfirst")

    assert emulator.feed("\x1b[1;1Hsecond").damaged == {0}


def test_moving_the_cursor_damages_the_row_it_left_and_the_row_it_reached() -> None:
    emulator = fed("\x1b[2;1H")

    assert emulator.feed("\x1b[4;1H").damaged == {1, 3}
    assert emulator.feed("\x1b[4;1H").damaged == set()
    assert emulator.feed("\x1b[C").damaged == {3}


def test_showing_or_hiding_the_cursor_damages_its_row() -> None:
    emulator = fed("\x1b[2;1H")

    assert emulator.feed("\x1b[?25l").damaged == {1}
    assert emulator.feed("\x1b[?25l").damaged == set()
    assert emulator.feed("\x1b[?25h").damaged == {1}


def test_reshaping_the_cursor_damages_its_row() -> None:
    emulator = fed("\x1b[2;1H")

    assert emulator.feed("\x1b[6 q").damaged == {1}
    assert emulator.feed("\x1b[6 q").damaged == set()
    # Blinking is not drawn: a bar stays the bar it was.
    assert emulator.feed("\x1b[5 q").damaged == set()
    assert emulator.feed("\x1b[!p").damaged == {1}


def test_damage_counts_rows_from_the_top_of_the_history() -> None:
    emulator = fed("one\r\ntwo\r\nthree\r\nfour", lines=2)

    assert emulator.feed("\x1b[1;1HX").damaged == {2, 3}


def test_switching_screens_damages_everything() -> None:
    emulator = fed("text")

    assert emulator.feed("\x1b[?1049h").damaged is None
    assert emulator.feed("x").damaged == {0}
    assert emulator.feed("\x1b[?1049l").damaged is None


def test_hard_reset_damages_everything() -> None:
    emulator = fed("text")

    assert emulator.feed("\x1bc").damaged is None


def test_history_over_the_limit_is_trimmed_and_reported() -> None:
    emulator = TerminalEmulator(20, 2, history_limit=3)

    update = emulator.feed("".join(f"row {number}\r\n" for number in range(10)))

    assert update.trimmed == 6
    assert update.damaged is None
    assert everything(emulator) == ["row 6", "row 7", "row 8", "row 9", ""]
    assert emulator.buffer.top == 3

    update = emulator.feed("more\r\n")
    assert update.trimmed == 1
    assert emulator.feed("x").trimmed == 0


def test_history_pushed_over_the_limit_by_a_resize_is_reported_by_the_next_feed() -> None:
    emulator = TerminalEmulator(20, 4, history_limit=2)
    emulator.feed("".join(f"row {number}\r\n" for number in range(5)) + "row 5")
    assert emulator.buffer.top == 2

    emulator.resize(20, 2)

    assert emulator.buffer.top == 4
    assert emulator.feed("").trimmed == 2
    assert everything(emulator) == ["row 2", "row 3", "row 4", "row 5"]
    assert emulator.buffer.top == 2


def test_history_the_program_clears_is_reported_as_trimmed() -> None:
    emulator = fed("".join(f"row {number}\r\n" for number in range(8)), lines=3)
    assert emulator.buffer.top == 6

    update = emulator.feed("\x1b[3J")

    assert update.trimmed == 6
    assert update.damaged is None
    assert everything(emulator) == ["row 6", "row 7", ""]
    assert emulator.feed("\x1b[3J").trimmed == 0


def test_history_cleared_and_history_over_the_limit_are_reported_together() -> None:
    emulator = TerminalEmulator(20, 2, history_limit=3)
    emulator.feed("".join(f"old {number}\r\n" for number in range(3)))
    assert emulator.buffer.top == 2

    update = emulator.feed("\x1b[3J" + "".join(f"row {number}\r\n" for number in range(5)))

    assert update.trimmed == 2 + 2
    assert everything(emulator) == ["row 1", "row 2", "row 3", "row 4", ""]


def test_history_lost_to_a_hard_reset_is_reported_as_trimmed() -> None:
    emulator = fed("".join(f"row {number}\r\n" for number in range(8)), lines=3)

    assert emulator.feed("\x1bc").trimmed == 6
    assert emulator.feed("x").trimmed == 0


def test_history_cleared_is_reported_once_the_primary_screen_is_back() -> None:
    emulator = fed("".join(f"row {number}\r\n" for number in range(8)), lines=3)

    assert emulator.feed("\x1b[3J\x1b[?1049h").trimmed == 0
    assert emulator.feed("\x1b[3J").trimmed == 0
    assert emulator.feed("\x1b[?1049l").trimmed == 6


def test_nothing_is_trimmed_on_the_alternate_screen() -> None:
    emulator = TerminalEmulator(20, 2, history_limit=3)
    emulator.feed("\x1b[?1049h")

    assert emulator.feed("".join(f"row {number}\r\n" for number in range(10))).trimmed == 0


def test_history_is_capped_by_default() -> None:
    emulator = TerminalEmulator(20, 2)

    # From the bottom row, where every line feed scrolls.
    update = emulator.feed("\x1b[2;1H" + "x\r\n" * (DEFAULT_HISTORY_LIMIT + 50))

    assert update.trimmed == 50
    assert emulator.buffer.top == DEFAULT_HISTORY_LIMIT
    assert len(emulator.buffer.rows) == DEFAULT_HISTORY_LIMIT + 2


def test_replies_and_events_belong_to_the_feed_that_caused_them() -> None:
    emulator = TerminalEmulator(20, 5)

    update = emulator.feed("\x1b[5n\x1b]2026;ls\x07\x1b[6n")
    assert update.replies == "\x1b[0n\x1b[1;1R"
    assert [type(event).__name__ for event in update.events] == ["CommandSubmitted"]

    update = emulator.feed("plain")
    assert (update.replies, update.events) == ("", ())
