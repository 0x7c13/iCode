# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Resizing the emulator: the way a desktop terminal does it, and under a host that repaints."""

from __future__ import annotations

import random

import pytest
from rich.cells import cell_len

from chrys.app.tui.terminal.emulator import TerminalEmulator

_PROMPT = "PS D:\\Repos\\chrys> "
_ERROR = (
    "asdfasdf: The term 'asdfasdf' is not recognized as a name of a cmdlet, "
    "function, script file, or executable program.\r\n"
    "Check the spelling of the name, or if a path was included, verify that the path is correct and try again.\r\n"
)
_LISTING = "".join(f"item{number}\r\n" for number in range(8)) + "PS> "


def fed(*streams: str, columns: int = 20, lines: int = 4) -> TerminalEmulator:
    emulator = TerminalEmulator(columns, lines)
    for stream in streams:
        emulator.feed(stream)
    return emulator


def everything(emulator: TerminalEmulator) -> list[str]:
    return [row.text.rstrip() for row in emulator.buffer.rows]


def cursor_row(emulator: TerminalEmulator) -> str:
    return emulator.buffer.rows[emulator.buffer.cursor_index].text


# -- the way a desktop terminal resizes --------------------------------------------------------------


def test_both_screens_take_the_new_size() -> None:
    emulator = fed("text")

    emulator.resize(33, 7)
    assert (emulator.columns, emulator.lines) == (33, 7)
    assert (emulator.buffer.columns, emulator.buffer.lines, len(emulator.buffer.rows)) == (33, 7, 7)

    emulator.feed("\x1b[?1049h")
    assert (emulator.buffer.columns, emulator.buffer.lines, len(emulator.buffer.rows)) == (33, 7, 7)

    emulator.resize(12, 3)
    emulator.feed("\x1b[99;99H")
    assert (emulator.buffer.cursor.x, emulator.buffer.cursor.y) == (11, 2)
    emulator.feed("\x1b[?1049l\x1b[99;99H")
    assert (emulator.buffer.cursor.x, emulator.buffer.cursor.y) == (11, 2)


def test_resize_redraws_everything() -> None:
    emulator = fed("text")

    emulator.resize(30, 6)

    assert emulator.feed("").damaged is None


def test_resize_to_the_same_size_changes_nothing_on_the_screen() -> None:
    emulator = fed(_LISTING)
    before = (everything(emulator), emulator.buffer.top, emulator.buffer.cursor.x, emulator.buffer.cursor.y)

    emulator.resize(20, 4)

    assert (everything(emulator), emulator.buffer.top, emulator.buffer.cursor.x, emulator.buffer.cursor.y) == before


def test_wider_screen_has_tab_stops_all_the_way_across() -> None:
    emulator = fed("\x1b[1;4H\x1bH\r")

    emulator.resize(40, 4)

    stops = []
    for _ in range(7):
        emulator.feed("\t")
        stops.append(emulator.buffer.cursor.x)
    assert stops == [3, 8, 16, 24, 32, 39, 39]


def test_narrower_screen_tabs_no_further_than_its_last_column() -> None:
    emulator = fed(columns=40)

    emulator.resize(12, 4)

    emulator.feed("\t")
    assert emulator.buffer.cursor.x == 8
    emulator.feed("\t")
    assert emulator.buffer.cursor.x == 11


def test_lines_are_rewrapped_to_the_new_width() -> None:
    emulator = fed("abcdefg", columns=5, lines=3)
    assert everything(emulator) == ["abcde", "fg", ""]

    emulator.resize(10, 3)
    assert everything(emulator) == ["abcdefg", "", ""]
    assert (emulator.buffer.cursor.x, emulator.buffer.cursor.y) == (7, 0)

    emulator.resize(3, 3)
    assert everything(emulator) == ["abc", "def", "g"]
    assert (emulator.buffer.cursor.x, emulator.buffer.cursor.y) == (1, 2)

    emulator.feed("h")
    assert everything(emulator) == ["abc", "def", "gh"]


def test_line_wrapped_at_the_bottom_of_the_screen_is_rejoined() -> None:
    emulator = fed("a\r\nb\r\nabcdefghijklm", columns=5, lines=3)
    assert everything(emulator) == ["a", "b", "abcde", "fghij", "klm"]

    emulator.resize(10, 3)

    assert everything(emulator) == ["a", "b", "abcdefghij", "klm", ""]
    assert emulator.buffer.top == 2


def test_hard_line_breaks_are_kept() -> None:
    emulator = fed("abcde\r\nfg", columns=5, lines=3)

    emulator.resize(10, 3)

    assert everything(emulator) == ["abcde", "fg", ""]


def test_history_is_rewrapped_too() -> None:
    emulator = fed("abcdefgh\r\n1\r\n2\r\n3\r\n4", columns=4, lines=2)
    assert everything(emulator) == ["abcd", "efgh", "1", "2", "3", "4"]

    emulator.resize(8, 2)

    assert everything(emulator) == ["abcdefgh", "1", "2", "3", "4"]
    assert emulator.buffer.top == 3


def test_wrap_pending_at_the_old_width_is_not_pending_at_a_wider_one() -> None:
    for reflow in (True, False):
        emulator = fed("abcde", columns=5, lines=3)
        assert emulator.buffer.cursor.pending_wrap

        emulator.resize(10, 3, reflow=reflow)
        emulator.feed("X")

        assert everything(emulator) == ["abcdeX", "", ""], reflow


def test_cursor_is_held_inside_a_narrower_screen() -> None:
    emulator = fed("\x1b[2;18H")

    emulator.resize(10, 4, reflow=False)

    assert (emulator.buffer.cursor.x, emulator.buffer.cursor.y) == (9, 1)


def test_scrolling_region_is_forgotten() -> None:
    emulator = fed("\x1b[2;3r")

    emulator.resize(20, 6)

    assert (emulator.buffer.margin_top, emulator.buffer.margin_bottom) == (0, 5)
    assert emulator.feed("\x1bP$qr\x1b\\").replies == "\x1bP1$r1;6r\x1b\\"


def test_taller_screen_shows_history_again() -> None:
    emulator = fed(_LISTING)
    assert (emulator.buffer.top, emulator.buffer.cursor.y) == (5, 3)

    emulator.resize(20, 6)

    assert (emulator.buffer.top, emulator.buffer.cursor.y) == (3, 5)
    assert emulator.buffer.screen_text == ["item3", "item4", "item5", "item6", "item7", "PS>"]


def test_screen_taller_than_the_history_gains_blank_rows_below() -> None:
    emulator = fed(_LISTING)

    emulator.resize(20, 12)

    assert (emulator.buffer.top, emulator.buffer.cursor.y, len(emulator.buffer.rows)) == (0, 8, 12)
    assert cursor_row(emulator) == "PS> "


def test_shorter_screen_gives_up_blank_rows_before_output() -> None:
    emulator = fed("a\r\nb\r\nPS> ", lines=6)

    emulator.resize(20, 3)
    assert (everything(emulator), emulator.buffer.top, emulator.buffer.cursor.y) == (["a", "b", "PS>"], 0, 2)

    emulator.resize(20, 2)
    assert (everything(emulator), emulator.buffer.top, emulator.buffer.cursor.y) == (["a", "b", "PS>"], 1, 1)


def test_cursor_stays_on_the_prompt_through_any_resize() -> None:
    emulator = TerminalEmulator(100, 24)
    for number in range(23):
        emulator.feed(f"old {number}\r\n")
    emulator.feed(f"{_PROMPT}asdfasdf\r\n{_ERROR}{_PROMPT}")

    for columns, lines in [(80, 12), (100, 24), (30, 5), (200, 60), (7, 2), (100, 24)]:
        emulator.resize(columns, lines)

        # However many rows the prompt takes at this width, the cursor is after its last character.
        x = emulator.buffer.cursor.x
        assert cursor_row(emulator).ljust(x)[:x].endswith("> "), (columns, lines)
        assert all(row.is_blank for row in emulator.buffer.rows[emulator.buffer.cursor_index + 1 :])

    emulator.feed("typed")
    assert cursor_row(emulator) == f"{_PROMPT}typed"


def test_resize_without_reflow_leaves_rows_as_they_are_until_asked() -> None:
    emulator = fed("abcdefghij", columns=10, lines=3)

    emulator.resize(5, 3, reflow=False)
    assert everything(emulator) == ["abcdefghij", "", ""]

    # Same width again, but this time the layout is ours to fix.
    emulator.resize(5, 3, reflow=True)
    assert everything(emulator) == ["abcde", "fghij", ""]


# -- the alternate screen ----------------------------------------------------------------------------


def test_alternate_screen_keeps_what_is_on_it() -> None:
    emulator = fed("\x1b[?1049hone\r\ntwo\r\nthree\r\nfour\r\nfive", lines=5)

    emulator.resize(10, 3)

    assert emulator.alternate_screen
    assert emulator.buffer.screen_text == ["three", "four", "five"]
    assert (len(emulator.buffer.rows), emulator.buffer.top) == (3, 0)
    assert (emulator.buffer.cursor.x, emulator.buffer.cursor.y) == (4, 2)


def test_alternate_screen_grows_blank_rows_and_never_a_history() -> None:
    emulator = fed("\x1b[?1049hone\r\ntwo\r\nthree\r\nfour\r\nfive", lines=5)

    emulator.resize(10, 3)
    emulator.resize(30, 6)

    assert emulator.buffer.screen_text == ["three", "four", "five", "", "", ""]
    assert (len(emulator.buffer.rows), emulator.buffer.top) == (6, 0)


def test_alternate_screen_is_not_rewrapped() -> None:
    emulator = fed("\x1b[?1049h0123456789", columns=10, lines=3)

    emulator.resize(5, 3)
    assert emulator.buffer.screen_text == ["0123456789", "", ""]

    # The program repaints at the new size; nothing of the old layout is in its way.
    emulator.feed("\x1b[H\x1b[2J01234\r\n56789")
    assert emulator.buffer.screen_text == ["01234", "56789", ""]


def test_primary_screen_is_resized_behind_the_alternate_screen() -> None:
    emulator = fed("abcdefghij\r\n$ ", "\x1b[?1049hfull screen", columns=10, lines=3)

    emulator.resize(5, 3)
    emulator.feed("\x1b[?1049l")

    assert everything(emulator) == ["abcde", "fghij", "$"]
    assert (emulator.buffer.cursor.x, emulator.buffer.cursor.y) == (2, 2)


def test_cursor_saved_on_the_way_in_comes_back_to_the_prompt_wherever_that_went() -> None:
    emulator = fed(_LISTING, "\x1b[?1049h\x1b[H\x1b[2Jeditor")

    emulator.resize(20, 6)
    emulator.feed("\x1b[?1049l")

    assert cursor_row(emulator) == "PS> "
    assert (emulator.buffer.cursor.x, emulator.buffer.cursor.y) == (4, 5)


def test_cursor_saved_somewhere_else_comes_back_to_where_it_was_saved() -> None:
    emulator = fed(_LISTING, "\x1b[2;3H\x1b7\x1b[4;5H")

    emulator.resize(20, 6)
    emulator.feed("\x1b8")

    assert (emulator.buffer.cursor.x, emulator.buffer.cursor.y) == (2, 1)


# -- under a host that repaints ----------------------------------------------------------------------


def test_repainting_host_gets_blank_rows_below_instead_of_history_above() -> None:
    emulator = fed("row1\r\nrow2\r\nrow3\r\nrow4\r\nPS> ", columns=80)
    assert (emulator.buffer.top, emulator.buffer.cursor.y) == (1, 3)

    emulator.resize(80, 8, reflow=False, host_repaints=True)
    emulator.feed("\x1b[4;5Htyped")

    assert everything(emulator)[4] == "PS> typed"
    assert (emulator.buffer.top, len(emulator.buffer.rows)) == (1, 9)


def test_first_position_a_repainting_host_sends_says_where_the_cursor_row_is() -> None:
    emulator = fed(_LISTING)
    emulator.resize(20, 6, reflow=False, host_repaints=True)
    assert (emulator.buffer.top, emulator.buffer.cursor.y) == (5, 3)

    # The host has the prompt on its second row, not our fourth.
    emulator.feed("\x1b[2;5Htyped")

    assert everything(emulator)[8] == "PS> typed"
    assert (emulator.buffer.top, emulator.buffer.cursor.y, len(emulator.buffer.rows)) == (7, 1, 13)


def test_only_the_first_position_is_taken_that_way() -> None:
    emulator = fed(_LISTING)
    emulator.resize(20, 6, reflow=False, host_repaints=True)

    emulator.feed("\x1b[2;5Htyped\x1b[1;1HX")

    assert everything(emulator)[7:9] == ["Xtem7", "PS> typed"]
    assert emulator.buffer.top == 7


def test_line_position_absolute_says_it_too() -> None:
    emulator = fed(_LISTING)
    emulator.resize(20, 6, reflow=False, host_repaints=True)

    emulator.feed("\x1b[2d")

    assert (emulator.buffer.top, emulator.buffer.cursor.y) == (7, 1)
    assert cursor_row(emulator) == "PS> "


def test_position_that_agrees_with_ours_moves_nothing() -> None:
    emulator = fed(_LISTING)
    emulator.resize(20, 6, reflow=False, host_repaints=True)

    emulator.feed("\x1b[4;5Htyped")

    assert everything(emulator)[8] == "PS> typed"
    assert (emulator.buffer.top, emulator.buffer.cursor.y) == (5, 3)


def test_moving_and_erasing_before_the_first_position_change_nothing_about_it() -> None:
    emulator = fed(_LISTING)
    emulator.resize(20, 6, reflow=False, host_repaints=True)

    emulator.feed("\r\x1b[K\x1b[?25l\x1b[2;1HY")

    assert everything(emulator)[8] == "Y"
    assert emulator.buffer.top == 7


def test_text_before_any_position_means_the_host_is_not_repainting() -> None:
    emulator = fed(_LISTING)
    emulator.resize(20, 6, reflow=False, host_repaints=True)

    emulator.feed("x\x1b[2;1HY")

    assert everything(emulator)[6:9] == ["Ytem6", "item7", "PS> x"]
    assert emulator.buffer.top == 5


def test_alternate_screen_is_addressed_as_it_is() -> None:
    emulator = fed(_LISTING, "\x1b[?1049hone\r\ntwo\r\nthree\r\nfour")
    emulator.resize(20, 6, reflow=False, host_repaints=True)

    emulator.feed("\x1b[2;1HY")
    assert emulator.buffer.screen_text == ["    one", "Ywo", "three", "four", "", ""]

    # Nor is the primary screen moved by the first position sent once the program has left.
    emulator.feed("\x1b[?1049l\x1b[2;1HZ")
    assert everything(emulator)[6:9] == ["Ztem6", "item7", "PS>"]
    assert emulator.buffer.top == 5


def test_resize_of_our_own_does_not_wait_for_a_position() -> None:
    emulator = TerminalEmulator(100, 24)
    emulator.feed(f"line0\r\nline1\r\nline2\r\nline3\r\nline4\r\nline5\r\n{_PROMPT}")

    emulator.resize(80, 24)
    emulator.feed("\x1b[5;1Htoprow")

    assert everything(emulator)[4] == "toprow"
    assert everything(emulator)[6] == _PROMPT.rstrip()
    assert emulator.buffer.top == 0


def test_output_that_fits_the_screen_is_laid_out_from_its_top_as_the_host_does() -> None:
    emulator = TerminalEmulator(100, 24)
    emulator.feed(f"line0\r\nline1\r\nline2\r\nline3\r\nline4\r\nline5\r\n{_PROMPT}")

    emulator.resize(80, 24, host_repaints=True)
    assert (emulator.buffer.top, emulator.buffer.cursor.y) == (0, 6)

    emulator.feed("\x1b[7;20Htyped")
    assert everything(emulator)[6] == f"{_PROMPT}typed"
    assert emulator.buffer.top == 0


def test_rewrapped_history_is_found_again_by_the_first_position() -> None:
    emulator = TerminalEmulator(100, 24)
    emulator.feed(
        f"{_PROMPT}ls\r\n\r\n    Directory: D:\\Repos\\chrys\r\n\r\n"
        "Mode                 LastWriteTime         Length Name\r\n"
        "----                 -------------         ------ ----\r\n"
    )
    for number in range(17):
        emulator.feed(f"d----           5/10/2026 12:00 PM                item{number}\r\n")
    emulator.feed(f"\r\n{_PROMPT}")
    prompt_index = emulator.buffer.cursor_index

    emulator.resize(80, 24, reflow=True, host_repaints=True)
    emulator.feed("\x1b[5;20Hsadfasdf")

    assert emulator.buffer.cursor_index == prompt_index
    assert everything(emulator)[prompt_index] == f"{_PROMPT}sadfasdf"
    assert emulator.buffer.top == prompt_index - 4
    assert "sadfasdf" not in everything(emulator)[4]


def test_narrower_and_shorter_under_a_repainting_host_keeps_the_prompt_under_the_cursor() -> None:
    emulator = TerminalEmulator(100, 24)
    for number in range(23):
        emulator.feed(f"old {number}\r\n")
    emulator.feed(f"{_PROMPT}asdfasdf\r\n{_ERROR}{_PROMPT}")

    emulator.resize(80, 12, reflow=True, host_repaints=True)
    emulator.feed("\x1b[12;20Hsdffasfd")

    assert cursor_row(emulator) == f"{_PROMPT}sdffasfd"
    assert emulator.buffer.cursor.y == 11


def test_wider_and_taller_under_a_repainting_host_keeps_the_prompt_under_the_cursor() -> None:
    emulator = TerminalEmulator(80, 12)
    for number in range(23):
        emulator.feed(f"old {number}\r\n")
    emulator.feed(f"{_PROMPT}asdfasdf\r\n{_ERROR}{_PROMPT}\r\n{_PROMPT}\r\n{_PROMPT}")
    assert emulator.buffer.cursor.y == 11

    emulator.resize(100, 24, reflow=True, host_repaints=True)
    emulator.feed("\x1b[12;20Hsdfffffffffff")

    assert cursor_row(emulator) == f"{_PROMPT}sdfffffffffff"
    assert emulator.buffer.cursor.y == 11


# -- whatever happens --------------------------------------------------------------------------------

_PIECES = (
    "plain text ",
    "a line long enough to wrap around more than once on a narrow screen ",
    "你好世界 wide ",
    "e\u0301 👨\u200d👩\u200d👧 ☺\ufe0f ",
    "\r\n",
    "\n",
    "\r",
    "\b",
    "\t",
    "\x1b[H",
    "\x1b[2J",
    "\x1b[3J",
    "\x1b[K",
    "\x1b[1K",
    "\x1b[2;3r",
    "\x1b[r",
    "\x1b[?6h",
    "\x1b[?6l",
    "\x1b[?7l",
    "\x1b[?7h",
    "\x1b[4h",
    "\x1b[4l",
    "\x1b[?1049h",
    "\x1b[?1049l",
    "\x1b[?47h",
    "\x1b[?47l",
    "\x1b7",
    "\x1b8",
    "\x1bM",
    "\x1bD",
    "\x1bE",
    "\x1b#8",
    "\x1b[5b",
    "\x1bc",
    "\x1b[!p",
)
_PARAMETERIZED = "ABCDEFGHLMPSTXZ@`adeI"
_FUZZ_HISTORY_LIMIT = 50


def assert_consistent(emulator: TerminalEmulator) -> None:
    buffer = emulator.buffer
    assert (buffer.columns, buffer.lines) == (emulator.columns, emulator.lines)
    assert len(buffer.rows) == buffer.top + buffer.lines
    assert 0 <= buffer.cursor.y < buffer.lines
    assert 0 <= buffer.cursor.x < buffer.columns
    assert not buffer.cursor.pending_wrap or buffer.cursor.x == buffer.columns - 1
    assert 0 <= buffer.margin_top < buffer.margin_bottom < buffer.lines or buffer.lines == 1
    assert not emulator.alternate_screen or buffer.top == 0
    for row in buffer.rows:
        assert len(row.cells) == len(row.pens) == cell_len(row.text)
        assert all(cell or column for column, cell in enumerate(row.cells))


@pytest.mark.parametrize("seed", range(40))
def test_the_screen_adds_up_whatever_is_fed_and_however_it_is_resized(seed: int) -> None:
    rng = random.Random(seed)
    emulator = TerminalEmulator(rng.randint(2, 30), rng.randint(1, 8), history_limit=_FUZZ_HISTORY_LIMIT)

    for _ in range(300):
        roll = rng.random()
        if roll < 0.12:
            emulator.resize(
                rng.randint(2, 30),
                rng.randint(1, 8),
                reflow=rng.random() < 0.7,
                host_repaints=rng.random() < 0.4,
            )
        else:
            if roll < 0.5:
                stream = f"\x1b[{rng.randint(0, 12)};{rng.randint(0, 40)}{rng.choice(_PARAMETERIZED)}"
            else:
                stream = rng.choice(_PIECES)
            update = emulator.feed(stream)
            if update.damaged is not None:
                assert all(0 <= index < len(emulator.buffer.rows) for index in update.damaged)
            # History is cut back to its limit by the feed, which is what reports the cut.
            assert emulator.alternate_screen or emulator.buffer.top <= _FUZZ_HISTORY_LIMIT
        assert_consistent(emulator)
