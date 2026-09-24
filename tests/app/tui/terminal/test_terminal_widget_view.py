# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The terminal widget as a view: size, scrolling over history, drawing, the cursor, selection."""

from __future__ import annotations

import base64

import pytest
from rich.color import Color
from rich.style import Style
from textual.geometry import Offset, Region, Size
from textual.selection import Selection
from textual.strip import Strip

from chrys.app.tui.support.gc_freeze import DetachedLruCache
from chrys.app.tui.terminal import widget as widget_module
from chrys.app.tui.terminal.emulator import Row, TerminalEmulator
from chrys.app.tui.terminal.widget import Terminal
from tests.app.tui.terminal._widget_harness import (
    GUTTER_WIDTH,
    TerminalApp,
    connect_stdin,
    numbered_lines,
    select_text,
    shown_lines,
)
from tests.support.waiting import wait_for

_FOUR_LINES_AND_A_PROMPT = "one\r\ntwo\r\nthree\r\nfour\r\nPS> "


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _marked_text(strip: Strip, *, reverse: bool = False, underline: bool = False) -> str:
    """The text of the segments carrying the given cursor style."""
    return "".join(
        segment.text
        for segment in strip
        if segment.style is not None
        and ((reverse and segment.style.reverse) or (underline and segment.style.underline))
    )


def _line_number(strip: Strip) -> int:
    """The line a rendered strip tells Textual's text selection it is."""
    return next(
        segment.style.meta["offset"][1]
        for segment in strip
        if segment.style is not None and "offset" in segment.style.meta
    )


class RefreshSpy:
    """Stands in for `Widget.refresh`, recording the regions each repaint asked for.

    Layout requests are Textual's own, made whenever the scrollable area changes size; they say
    nothing about what the terminal chose to draw, and are left out.
    """

    def __init__(self, terminal: Terminal) -> None:
        self._terminal = terminal
        self.calls: list[tuple[Region, ...]] = []

    def __call__(
        self, *regions: Region, repaint: bool = True, layout: bool = False, recompose: bool = False
    ) -> Terminal:
        if not layout:
            self.calls.append(regions)
        return self._terminal


def _spy_on_refresh(monkeypatch: pytest.MonkeyPatch, terminal: Terminal) -> RefreshSpy:
    spy = RefreshSpy(terminal)
    monkeypatch.setattr(terminal, "refresh", spy)
    return spy


# -- size --------------------------------------------------------------------------------------------


async def test_mount_sizes_the_emulator_to_the_content_region_and_announces_it() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await wait_for(
            lambda: app.messages_of(Terminal.SizeChanged), pilot=pilot, description="size announced after mount"
        )

        assert terminal.scrollable_content_region.size == Size(40 - GUTTER_WIDTH, 10)
        assert (terminal.width, terminal.height) == (38, 10)
        assert (terminal.emulator.columns, terminal.emulator.lines) == (38, 10)
        announced = app.messages_of(Terminal.SizeChanged)
        assert [(message.width, message.height, message.control) for message in announced] == [(38, 10, terminal)]


async def test_app_resize_resizes_the_emulator_and_announces_each_new_size() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal

        await pilot.resize_terminal(50, 8)
        await wait_for(lambda: (terminal.width, terminal.height) == (48, 7), pilot=pilot, description="resized")
        await wait_for(
            lambda: len(app.messages_of(Terminal.SizeChanged)) == 2, pilot=pilot, description="resize announced"
        )

        assert (terminal.emulator.columns, terminal.emulator.lines) == (48, 7)
        assert [(message.width, message.height) for message in app.messages_of(Terminal.SizeChanged)][-1] == (48, 7)


@pytest.mark.parametrize("host_repaints", [False, True], ids=["local-reflow", "repainting-host"])
async def test_empty_size_of_a_hidden_widget_is_ignored(host_repaints: bool) -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        terminal.host_repaints = host_repaints
        await wait_for(lambda: app.messages_of(Terminal.SizeChanged), pilot=pilot, description="mounted size")
        app.terminal_messages.clear()

        terminal.update_size(0, 0, immediate=True)
        terminal.update_size(0, 10)
        terminal.update_size(38, -1, force_reflow=True)

        assert (terminal.width, terminal.height) == (38, 10)
        assert (terminal.emulator.columns, terminal.emulator.lines) == (38, 10)
        # A real size afterwards is the only one announced: the empty ones said nothing before it.
        terminal.update_size(30, 8, immediate=True)
        await wait_for(lambda: app.messages_of(Terminal.SizeChanged), pilot=pilot, description="real size announced")
        assert [(message.width, message.height) for message in app.messages_of(Terminal.SizeChanged)] == [(30, 8)]


async def test_hiding_the_widget_keeps_the_size_its_program_knows() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write(numbered_lines(3))

        terminal.display = False
        await pilot.pause()

        assert (terminal.width, terminal.height) == (38, 10)
        terminal.display = True
        await wait_for(lambda: terminal.scrollable_content_region.height == 10, pilot=pilot, description="shown again")
        assert (terminal.width, terminal.height) == (38, 10)
        assert shown_lines(terminal)[:3] == ["line 0", "line 1", "line 2"]


def test_unmounted_terminal_renders_without_a_screen() -> None:
    terminal = Terminal(size=(20, 4))

    # No region to fill yet, so nothing to draw; what matters is that asking is safe.
    assert (terminal.width, terminal.height) == (20, 4)
    assert terminal.render_line(0).cell_length == 0
    assert terminal.render_line(99).cell_length == 0


async def test_lines_outside_the_rows_render_blank_at_full_width() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)):
        terminal = app.terminal
        await terminal.write("only")

        below = terminal.render_line(terminal.height + 5)

        assert below.cell_length == 38
        assert below.text.strip() == ""
        assert terminal.render_line(0).cell_length == 38


# -- following the output ----------------------------------------------------------------------------


async def test_view_follows_output_into_history() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal

        await terminal.write(numbered_lines(30) + "$ ")

        # Drawn at once, ahead of the layout that lets the scroll position catch up.
        assert shown_lines(terminal)[-2:] == ["line 29", "$"]
        await pilot.pause()
        top = terminal.emulator.buffer.top
        assert top == 21
        assert terminal.virtual_size == Size(38, 31)
        assert terminal.scroll_y == terminal.max_scroll_y == top
        assert shown_lines(terminal) == [*(f"line {number}" for number in range(21, 30)), "$"]


async def test_view_scrolled_into_history_stays_put_while_output_arrives() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write(numbered_lines(30) + "$ ")
        await pilot.pause()

        terminal.scroll_to(y=5, animate=False)
        await pilot.pause()
        await terminal.write("more\r\n$ ")
        await pilot.pause()

        assert terminal.scroll_y == 5
        assert terminal.max_scroll_y == 22
        assert shown_lines(terminal) == [f"line {number}" for number in range(5, 15)]


async def test_view_scrolled_into_history_stays_on_its_text_while_history_is_trimmed() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        terminal.emulator = TerminalEmulator(terminal.width, terminal.height, history_limit=20)
        await terminal.write(numbered_lines(30))
        await pilot.pause()
        terminal.scroll_to(y=8, animate=False)
        await pilot.pause()
        shown = shown_lines(terminal)
        assert shown[0] == "line 9"

        await terminal.write("".join(f"more {number}\r\n" for number in range(5)))
        await pilot.pause()

        # Five rows left through the top; the view moved up five to stay on the same text.
        assert terminal.emulator.buffer.top == 20
        assert terminal.scroll_y == 3
        assert shown_lines(terminal) == shown


async def test_view_whose_text_was_trimmed_away_rests_on_the_oldest_rows() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        terminal.emulator = TerminalEmulator(terminal.width, terminal.height, history_limit=20)
        await terminal.write(numbered_lines(30))
        await pilot.pause()
        terminal.scroll_to(y=2, animate=False)
        await pilot.pause()

        await terminal.write("".join(f"more {number}\r\n" for number in range(5)))
        await pilot.pause()

        assert terminal.scroll_y == 0
        assert shown_lines(terminal)[0] == terminal.emulator.buffer.rows[0].text == "line 6"


async def test_scrolling_back_to_the_bottom_follows_again() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write(numbered_lines(30) + "$ ")
        await pilot.pause()
        terminal.scroll_to(y=5, animate=False)
        await pilot.pause()

        terminal.scroll_to(y=terminal.max_scroll_y, animate=False)
        await pilot.pause()
        await terminal.write("more\r\n$ ")
        await pilot.pause()

        assert terminal.scroll_y == terminal.max_scroll_y == terminal.emulator.buffer.top
        assert shown_lines(terminal)[-2:] == ["$ more", "$"]


async def test_following_keeps_the_scroll_target_with_the_view() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal

        await terminal.write(numbered_lines(30) + "$ ")
        await pilot.pause()

        # Textual scrolls the wheel on from the target. Left at 0 it would jump to the first row.
        assert terminal.scroll_target_y == terminal.scroll_y == 21


async def test_typing_brings_the_view_back_to_the_prompt() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write(numbered_lines(30) + "$ ")
        await pilot.pause()
        terminal.scroll_to(y=0, animate=False)
        await pilot.pause()
        assert shown_lines(terminal)[0] == "line 0"

        await pilot.press("enter")
        await wait_for(lambda: stdin.writes == ["\r"], pilot=pilot, description="Enter reaches the program")

        assert shown_lines(terminal)[-2:] == ["line 29", "$"]
        assert terminal.scroll_y == terminal.max_scroll_y


async def test_pasting_brings_the_view_back_to_the_prompt() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write(numbered_lines(30) + "$ ")
        await pilot.pause()
        terminal.scroll_to(y=0, animate=False)
        await pilot.pause()

        await terminal.paste("ls")
        await pilot.pause()

        assert stdin.writes == ["ls"]
        assert shown_lines(terminal)[-1] == "$"
        assert terminal.scroll_y == terminal.max_scroll_y


async def test_protocol_replies_do_not_move_a_view_scrolled_into_history() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write(numbered_lines(30) + "$ ")
        await pilot.pause()
        terminal.scroll_to(y=5, animate=False)
        await pilot.pause()

        await terminal.write("\x1b[6n")
        await pilot.pause()

        # The program asked, the terminal answered; the user did nothing and keeps their place.
        assert stdin.writes == ["\x1b[10;3R"]
        assert terminal.scroll_y == 5
        assert shown_lines(terminal)[0] == "line 5"


# -- resizing with history ---------------------------------------------------------------------------


async def test_height_shrink_keeps_the_view_at_the_bottom_and_the_cursor_in_sight() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 5)) as pilot:
        terminal = app.terminal
        await terminal.write(_FOUR_LINES_AND_A_PROMPT)
        await pilot.pause()
        assert terminal.max_scroll_y == 1

        await pilot.resize_terminal(40, 3)
        await wait_for(lambda: terminal.height == 2, pilot=pilot, description="terminal two rows high")
        await terminal.write("x")
        await pilot.pause()

        assert terminal.scroll_y == terminal.max_scroll_y == 3
        assert shown_lines(terminal) == ["four", "PS> x"]


async def test_height_grow_shows_history_again_and_needs_no_scrolling_once_it_fits() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 4)) as pilot:
        terminal = app.terminal
        await terminal.write("one\r\ntwo\r\nthree\r\nPS> ")
        await pilot.pause()
        assert terminal.max_scroll_y == 1

        await pilot.resize_terminal(40, 11)
        await wait_for(lambda: terminal.height == 10, pilot=pilot, description="terminal ten rows high")
        await pilot.pause()

        # The way a desktop terminal grows: rows come back from history, none are added below.
        assert terminal.emulator.buffer.top == 0
        assert terminal.max_scroll_y == 0
        assert terminal.scroll_y == 0
        assert shown_lines(terminal)[:4] == ["one", "two", "three", "PS>"]


async def test_height_grow_under_a_repainting_host_leaves_the_origin_to_the_host() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 4)) as pilot:
        terminal = app.terminal
        terminal.host_repaints = True
        await terminal.write("one\r\ntwo\r\nthree\r\nPS> ")
        await pilot.pause()
        assert terminal.emulator.buffer.top == 1

        await pilot.resize_terminal(40, 11)
        await wait_for(lambda: terminal.height == 10, pilot=pilot, description="terminal ten rows high")
        await pilot.pause()

        # The host has yet to say where its screen begins, but everything fits and is shown.
        assert terminal.emulator.buffer.top == 1
        assert terminal.max_scroll_y == 0
        assert terminal.scroll_y == 0
        assert shown_lines(terminal)[:4] == ["one", "two", "three", "PS>"]


async def test_height_shrink_after_a_fit_takes_the_origin_from_the_hosts_first_position() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 4)) as pilot:
        terminal = app.terminal
        terminal.host_repaints = True
        await terminal.write("one\r\ntwo\r\nthree\r\nPS> ")
        await pilot.resize_terminal(40, 11)
        await wait_for(lambda: terminal.height == 10, pilot=pilot, description="terminal ten rows high")

        await pilot.resize_terminal(40, 4)
        await wait_for(lambda: terminal.height == 3, pilot=pilot, description="terminal three rows high")
        await terminal.write("\x1b[3;5Htyped")
        await pilot.pause()

        assert terminal.emulator.buffer.top == 1
        assert terminal.max_scroll_y == 1
        assert terminal.scroll_y == 1
        assert terminal.emulator.buffer.rows[-1].text == "PS> typed"
        assert shown_lines(terminal) == ["two", "three", "PS> typed"]


async def test_height_shrink_under_a_repainting_host_moves_the_origin_to_the_hosts_cursor_row() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 25)) as pilot:
        terminal = app.terminal
        await terminal.write("row1\r\nrow2\r\nrow3\r\nrow4\r\nPS> ")
        prompt_index = terminal.emulator.buffer.cursor_index
        assert prompt_index == 4

        terminal.host_repaints = True
        await pilot.resize_terminal(40, 4)
        await wait_for(lambda: terminal.height == 3, pilot=pilot, description="terminal three rows high")
        await terminal.write("\x1b[3;5Htyped")
        await pilot.pause()

        buffer = terminal.emulator.buffer
        assert buffer.rows[prompt_index].text == "PS> typed"
        assert buffer.top == prompt_index - 2
        assert shown_lines(terminal) == ["row3", "row4", "PS> typed"]


async def test_catching_up_a_repainting_host_keeps_its_origin_within_reach_of_the_scrollbar() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 13)) as pilot:
        terminal = app.terminal
        terminal.host_repaints = True
        await terminal.write(numbered_lines(30) + "PS> ")
        await pilot.pause()
        assert terminal.emulator.buffer.top == 19

        # The size changed while the terminal was hidden; its owner catches it up on showing it.
        await pilot.resize_terminal(50, 25)
        await wait_for(lambda: (terminal.width, terminal.height) == (48, 24), pilot=pilot, description="resized")
        terminal.update_size(48, 24, immediate=True, force_reflow=True)
        await pilot.pause()

        # More output than fits: the origin stays where the host has it, below blank rows that
        # count towards the scrollable height so that the bottom of the scrollbar is the screen.
        top = terminal.emulator.buffer.top
        assert top == 19
        assert terminal.emulator.buffer.used_height == 31
        assert terminal.virtual_size.height == top + terminal.height
        assert terminal.scroll_y == terminal.max_scroll_y == top
        assert shown_lines(terminal)[0] == "line 19"
        assert shown_lines(terminal)[11] == "PS>"


async def test_catching_up_a_repainting_host_starts_from_the_first_row_when_everything_fits() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 13)) as pilot:
        terminal = app.terminal
        terminal.host_repaints = True
        await terminal.write(numbered_lines(23) + "PS> ")
        await pilot.pause()
        assert terminal.emulator.buffer.top == 12

        await pilot.resize_terminal(50, 25)
        await wait_for(lambda: (terminal.width, terminal.height) == (48, 24), pilot=pilot, description="resized")
        terminal.update_size(48, 24, immediate=True, force_reflow=True)
        await pilot.pause()

        assert terminal.emulator.buffer.top == 0
        assert terminal.max_scroll_y == 0
        assert shown_lines(terminal)[0] == "line 0"
        assert shown_lines(terminal)[23] == "PS>"


async def test_forced_reflow_rewraps_what_a_repainting_host_resize_left_alone() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        terminal.host_repaints = True
        await terminal.write("x" * 50)
        assert [row.text for row in terminal.emulator.buffer.rows[:2]] == ["x" * 38, "x" * 12]

        await pilot.resize_terminal(62, 11)
        await wait_for(lambda: terminal.width == 60, pilot=pilot, description="terminal sixty columns wide")
        # The host is expected to send the screen again, so the rows wait for it as they are.
        assert [row.text for row in terminal.emulator.buffer.rows[:2]] == ["x" * 38, "x" * 12]

        terminal.update_size(60, 10, immediate=True, force_reflow=True)

        assert terminal.emulator.buffer.rows[0].text == "x" * 50
        # Nothing new to tell the program: the size is the one already announced.
        await pilot.pause()
        assert [(message.width, message.height) for message in app.messages_of(Terminal.SizeChanged)] == [
            (38, 10),
            (60, 10),
        ]


async def test_local_resize_rewraps_at_once() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("x" * 50)

        await pilot.resize_terminal(62, 11)
        await wait_for(lambda: terminal.width == 60, pilot=pilot, description="terminal sixty columns wide")

        assert terminal.emulator.buffer.rows[0].text == "x" * 50
        assert shown_lines(terminal)[0] == "x" * 50


# -- the scrollable area -----------------------------------------------------------------------------


async def test_scrollable_area_is_only_as_tall_as_the_output_while_it_fits() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal

        await terminal.write("a\r\nb\r\n")
        await pilot.pause()

        assert terminal.emulator.buffer.used_height == 3
        assert terminal.virtual_size == Size(38, 3)
        assert terminal.max_scroll_y == 0


async def test_alternate_screen_has_nothing_to_scroll_and_does_not_change_the_width() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write(numbered_lines(30) + "$ ")
        await pilot.pause()
        assert terminal.allow_vertical_scroll

        await terminal.write("\x1b[?1049h\x1b[Hfull screen")
        await wait_for(
            lambda: app.messages_of(Terminal.AlternateScreenChanged), pilot=pilot, description="alternate announced"
        )

        assert terminal.alternate_screen
        assert terminal.virtual_size == Size(38, 10)
        assert terminal.max_scroll_y == 0
        assert not terminal.allow_vertical_scroll
        assert not terminal.allow_horizontal_scroll
        assert shown_lines(terminal) == ["full screen", *[""] * 9]
        # A gutter that closed here would resize the program on every entry and exit.
        assert terminal.scrollable_content_region.width == terminal.width == 38
        assert len(app.messages_of(Terminal.SizeChanged)) == 1


async def test_alternate_screen_is_announced_both_ways_and_history_is_back_after_it() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write(numbered_lines(30) + "$ ")
        await pilot.pause()

        await terminal.write("\x1b[?1049h\x1b[Hfull screen")
        await terminal.write("\x1b[?1049l")
        await wait_for(
            lambda: len(app.messages_of(Terminal.AlternateScreenChanged)) == 2,
            pilot=pilot,
            description="alternate screen entry and exit announced",
        )

        changes = app.messages_of(Terminal.AlternateScreenChanged)
        assert [(change.enabled, change.control) for change in changes] == [(True, terminal), (False, terminal)]
        assert not terminal.alternate_screen
        assert terminal.scroll_y == terminal.max_scroll_y == 21
        assert shown_lines(terminal)[-2:] == ["line 29", "$"]


async def test_reset_leaves_the_alternate_screen_and_says_so() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write(numbered_lines(30) + "\x1b[?1049h\x1b[Hfull screen")

        terminal.reset()
        await wait_for(
            lambda: len(app.messages_of(Terminal.AlternateScreenChanged)) == 2,
            pilot=pilot,
            description="reset announced the primary screen",
        )

        assert [change.enabled for change in app.messages_of(Terminal.AlternateScreenChanged)] == [True, False]
        assert not terminal.alternate_screen
        assert (terminal.width, terminal.height) == (38, 10)
        assert shown_lines(terminal) == [""] * 10
        assert terminal.virtual_size == Size(38, 1)
        assert terminal.max_scroll_y == 0


async def test_reset_on_the_primary_screen_announces_no_screen_change() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write(numbered_lines(30))

        terminal.reset()
        await terminal.write("\x1b[?1049h")
        await wait_for(
            lambda: app.messages_of(Terminal.AlternateScreenChanged), pilot=pilot, description="alternate announced"
        )

        # The entry is the first change heard of; the reset before it announced none.
        assert [change.enabled for change in app.messages_of(Terminal.AlternateScreenChanged)] == [True]


async def test_shell_reports_become_messages() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal

        await terminal.write(f"\x1b]2025;b64:{_b64('/tmp/a;b')}\x1b\\\x1b]2026;b64:{_b64('echo 1; echo 2')}\x07")
        await wait_for(
            lambda: app.messages_of(Terminal.CommandSubmitted), pilot=pilot, description="shell reports delivered"
        )

        directories = app.messages_of(Terminal.DirectoryChanged)
        commands = app.messages_of(Terminal.CommandSubmitted)
        assert [(message.path, message.control) for message in directories] == [("/tmp/a;b", terminal)]
        assert [(message.command, message.control) for message in commands] == [("echo 1; echo 2", terminal)]
        assert shown_lines(terminal) == [""] * 10


async def test_write_says_whether_anything_on_display_changed() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)):
        terminal = app.terminal

        assert await terminal.write("x") is True
        assert await terminal.write("\x1b[?2004h") is False


# -- drawing -----------------------------------------------------------------------------------------


async def test_write_redraws_only_the_rows_it_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("one\r\ntwo\r\nthree")
        await pilot.pause()
        refreshes = _spy_on_refresh(monkeypatch, terminal)

        await terminal.write("\x1b[2;1Hchanged")

        # Row 1 changed; the cursor came from row 2, which is redrawn to take it off.
        assert len(refreshes.calls) == 1
        assert set(refreshes.calls[0]) == {Region(0, 1, 38, 1), Region(0, 2, 38, 1)}


async def test_write_that_moves_the_view_redraws_all_of_it(monkeypatch: pytest.MonkeyPatch) -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write(numbered_lines(9) + "$ ")
        await pilot.pause()
        refreshes = _spy_on_refresh(monkeypatch, terminal)

        await terminal.write("\r\n$ ")

        # Textual repaints for the scroll on its own account; the terminal's full redraw comes last.
        assert refreshes.calls
        assert refreshes.calls[-1] == ()


async def test_write_that_changes_nothing_redraws_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("$ ")
        await pilot.pause()
        refreshes = _spy_on_refresh(monkeypatch, terminal)

        await terminal.write("\x1b[?2004h")
        assert refreshes.calls == []

        # The spy does see a redraw when there is one.
        await terminal.write("x")
        assert refreshes.calls == [(Region(0, 0, 38, 1),)]


async def test_resize_under_a_repainting_host_draws_once_the_repaint_has_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(widget_module, "_REPAINT_SETTLE_DELAY", 0.01)
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        terminal.host_repaints = True
        await terminal.write("PS> ")
        await pilot.pause()
        refreshes = _spy_on_refresh(monkeypatch, terminal)

        terminal.update_size(30, 8)
        await wait_for(lambda: refreshes.calls == [()], pilot=pilot, description="one full redraw after the settle")


async def test_resize_under_a_repainting_host_draws_nothing_while_the_repaint_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(widget_module, "_REPAINT_SETTLE_DELAY", 60.0)
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        terminal.host_repaints = True
        await terminal.write("PS> ")
        await pilot.pause()
        refreshes = _spy_on_refresh(monkeypatch, terminal)

        terminal.update_size(30, 8)
        await terminal.write("\x1b[2J\x1b[HPS> half of the repaint")
        assert refreshes.calls == []

        # An owner that cannot wait (it is showing the terminal right now) draws at once, and
        # writes draw again from then on.
        terminal.update_size(30, 8, immediate=True, force_reflow=True)
        assert refreshes.calls == [()]
        await terminal.write("x")
        assert len(refreshes.calls) == 2


async def test_local_resize_draws_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("$ ")
        await pilot.pause()
        refreshes = _spy_on_refresh(monkeypatch, terminal)

        terminal.update_size(30, 8)

        assert refreshes.calls == [()]


async def test_rendered_rows_are_cached_until_the_row_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered: list[str] = []
    render_row = widget_module.render_row

    def counting_render_row(row: Row) -> Strip:
        rendered.append(row.text)
        return render_row(row)

    monkeypatch.setattr(widget_module, "render_row", counting_render_row)
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("first\r\nsecond")
        await pilot.pause()
        terminal._strip_cache.clear()
        rendered.clear()

        for _ in range(2):
            terminal.render_line(0)
            terminal.render_line(1)
        assert rendered == ["first", "second"]

        # Only the row that changed is rendered again.
        rendered.clear()
        await terminal.write("\x1b[1;1Hthird")
        assert terminal.render_line(0).text.rstrip() == "third"
        terminal.render_line(1)
        assert rendered == ["third"]


async def test_cached_rows_take_a_new_widget_style_without_being_rendered_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rendered: list[str] = []
    render_row = widget_module.render_row

    def counting_render_row(row: Row) -> Strip:
        rendered.append(row.text)
        return render_row(row)

    monkeypatch.setattr(widget_module, "render_row", counting_render_row)
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("themed")
        await pilot.pause()
        before = terminal.render_line(0)
        rendered.clear()

        terminal.styles.background = "#123456"
        await pilot.pause()
        after = terminal.render_line(0)

        # The cache holds rows as the program styled them; the widget's style goes on afterwards.
        assert rendered == []
        assert after.text == before.text
        assert {segment.style.bgcolor for segment in after if segment.style is not None} == {
            Color.from_rgb(0x12, 0x34, 0x56)
        }


async def test_render_cache_is_let_go_for_a_gc_freeze_and_comes_back_on_demand() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)):
        terminal = app.terminal
        await terminal.write("kept")
        assert terminal.render_line(0).text.rstrip() == "kept"
        capacity = terminal._strips.maxsize

        terminal.detach_render_cache()
        assert isinstance(terminal._strips, DetachedLruCache)

        # Drawing does not wait for the owner to renew the cache.
        assert terminal.render_line(0).text.rstrip() == "kept"
        assert not isinstance(terminal._strips, DetachedLruCache)
        assert terminal._strips.maxsize == capacity

        terminal.detach_render_cache()
        terminal.renew_render_cache()
        assert not isinstance(terminal._strips, DetachedLruCache)
        assert terminal._strips.maxsize == capacity


async def test_hyperlinks_are_the_programs_not_guessed_from_the_text() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)):
        terminal = app.terminal

        await terminal.write("https://plain.example \x1b]8;;https://linked.example\x1b\\here\x1b]8;;\x1b\\")

        links = {segment.text: segment.style.link for segment in terminal.render_line(0) if segment.style is not None}
        assert links["here"] == "https://linked.example"
        assert links["https://plain.example "] is None
        assert not terminal.auto_links


# -- the cursor --------------------------------------------------------------------------------------


async def test_cursor_is_drawn_only_while_the_terminal_has_focus() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("abc\x1b[2D")
        await wait_for(lambda: terminal.has_focus, pilot=pilot, description="terminal focused")
        assert _marked_text(terminal.render_line(0), reverse=True) == "b"

        app.query_one("#elsewhere").focus()
        await wait_for(lambda: not terminal.has_focus, pilot=pilot, description="focus elsewhere")
        assert _marked_text(terminal.render_line(0), reverse=True) == ""

        terminal.focus()
        await wait_for(lambda: terminal.has_focus, pilot=pilot, description="terminal focused again")
        assert _marked_text(terminal.render_line(0), reverse=True) == "b"


async def test_program_hides_shows_and_shapes_the_cursor() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("abc\x1b[2D")
        await wait_for(lambda: terminal.has_focus, pilot=pilot, description="terminal focused")

        await terminal.write("\x1b[?25l")
        assert _marked_text(terminal.render_line(0), reverse=True, underline=True) == ""

        await terminal.write("\x1b[?25h\x1b[4 q")
        assert _marked_text(terminal.render_line(0), underline=True) == "b"
        assert _marked_text(terminal.render_line(0), reverse=True) == ""


async def test_new_cursor_shape_alone_is_a_change_to_redraw(monkeypatch: pytest.MonkeyPatch) -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("one\r\nabc\x1b[2D")
        await wait_for(lambda: terminal.has_focus, pilot=pilot, description="terminal focused")
        assert _marked_text(terminal.render_line(1), reverse=True) == "b"
        refreshes = _spy_on_refresh(monkeypatch, terminal)

        assert await terminal.write("\x1b[4 q")

        assert refreshes.calls == [(Region(0, 1, 38, 1),)]
        assert _marked_text(terminal.render_line(1), underline=True) == "b"
        assert not await terminal.write("\x1b[4 q")


async def test_cursor_beside_a_joiner_sequence_leaves_it_whole() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("👩\u200d👧C\x1b[D")
        await wait_for(lambda: terminal.has_focus, pilot=pilot, description="terminal focused")

        strip = terminal.render_line(0)

        assert strip.text.rstrip() == "👩\u200d👧C"
        assert strip.cell_length == 38
        assert _marked_text(strip, reverse=True) == "C"


async def test_cursor_past_the_text_keeps_the_line_at_full_width() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("ab\x1b[1;20H")
        await wait_for(lambda: terminal.has_focus, pilot=pilot, description="terminal focused")

        strip = terminal.render_line(0)

        assert strip.cell_length == 38
        assert _marked_text(strip, reverse=True) == " "
        assert strip.text[:19] == "ab" + " " * 17


class _OffsetTerminalApp(TerminalApp):
    CSS = TerminalApp.CSS + "Terminal { margin: 2 0 0 3; }"


async def test_cursor_screen_offset_is_where_the_cursor_is_drawn() -> None:
    app = _OffsetTerminalApp()
    async with app.run_test(size=(40, 13)) as pilot:
        terminal = app.terminal
        await terminal.write("one\r\nab")
        await pilot.pause()

        offset = terminal.cursor_screen_offset

        # An `Offset` and nothing looser: the app's input-method anchor accepts no other type.
        assert type(offset) is Offset
        assert offset == Offset(3 + 2, 2 + 1)


async def test_cursor_screen_offset_stays_inside_the_terminal_after_the_last_column() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)):
        terminal = app.terminal

        await terminal.write("x" * 38)

        assert terminal.emulator.buffer.cursor.pending_wrap
        assert terminal.cursor_screen_offset == Offset(37, 0)


async def test_cursor_screen_offset_is_none_while_the_cursor_is_scrolled_out_of_view() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write(numbered_lines(30) + "$ ")
        await pilot.pause()
        assert terminal.cursor_screen_offset == Offset(2, 9)

        terminal.scroll_to(y=0, animate=False)
        await pilot.pause()

        assert terminal.cursor_screen_offset is None


# -- selection ---------------------------------------------------------------------------------------


async def test_selected_text_joins_wrapped_rows_and_drops_blanks_nobody_typed() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)):
        terminal = app.terminal
        long_line = "x" * 50
        await terminal.write(f"{long_line}\r\nshort\r\n\r\nlast")
        # A blank the program wrote far to the right of "short", as clearing a line does.
        await terminal.write("\x1b[3;30H \x1b[5;5H")
        assert terminal.emulator.buffer.rows[2].text == "short" + " " * 25

        everything = terminal.get_selection(Selection(None, None))

        # The terminal wrapped the first line; the program broke the others. The rows below the
        # cursor, which nothing was written to, are not part of the text.
        assert everything == (f"{long_line}\nshort\n\nlast", "\n")


async def test_selection_inside_a_wrapped_line_has_no_line_break() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)):
        terminal = app.terminal
        await terminal.write("0123456789" * 5)

        selected = terminal.get_selection(Selection(Offset(36, 0), Offset(4, 1)))

        # Two characters end the first row and four begin the second.
        assert selected == ("678901", "\n")


async def test_selection_of_nothing_is_no_selection() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)):
        terminal = app.terminal
        await terminal.write("text")

        assert terminal.get_selection(Selection(Offset(2, 0), Offset(2, 0))) is None


async def test_selection_keeps_its_text_while_history_is_trimmed_from_under_it() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        terminal.emulator = TerminalEmulator(terminal.width, terminal.height, history_limit=20)
        await terminal.write(numbered_lines(40))
        await pilot.pause()
        strip = terminal.render_line(3)
        assert strip.text.rstrip() == "line 34"
        line = _line_number(strip)
        selection = Selection(Offset(0, line), Offset(7, line))
        assert terminal.get_selection(selection) == ("line 34", "\n")

        await terminal.write("".join(f"more {number}\r\n" for number in range(6)))

        # The row moved up six places; its line number, which is what a selection holds, did not.
        assert terminal.emulator.buffer.rows[17].text == "line 34"
        assert terminal.get_selection(selection) == ("line 34", "\n")
        await pilot.pause()
        terminal.scroll_to(y=10, animate=False)
        await pilot.pause()
        assert shown_lines(terminal)[7] == "line 34"
        assert _line_number(terminal.render_line(7)) == line

        # Once the row itself has left through the top, there is nothing left to select.
        await terminal.write("".join(f"later {number}\r\n" for number in range(30)))
        assert terminal.get_selection(selection) is None


async def test_selection_keeps_its_text_when_the_program_clears_the_history() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write(numbered_lines(40))
        await pilot.pause()
        strip = terminal.render_line(3)
        assert strip.text.rstrip() == "line 34"
        line = _line_number(strip)
        selection = Selection(Offset(0, line), Offset(7, line))

        await terminal.write("\x1b[3J")
        await pilot.pause()

        # The history above the screen is gone and the row is 31 places up: still the same line.
        assert terminal.emulator.buffer.rows[3].text == "line 34"
        assert terminal.get_selection(selection) == ("line 34", "\n")
        assert shown_lines(terminal)[3] == "line 34"
        assert _line_number(terminal.render_line(3)) == line


async def test_selection_is_drawn_over_the_cells_of_the_selected_characters() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("\x1b[31ma中bc\x1b[0m")
        app.query_one("#elsewhere").focus()
        await wait_for(lambda: not terminal.has_focus, pilot=pilot, description="no cursor in the way")
        plain = {segment.style.bgcolor for segment in terminal.render_line(0) if segment.style is not None}
        assert len(plain) == 1

        select_text(terminal, Offset(1, 0), Offset(3, 0))
        await pilot.pause()

        selection_style = app.screen.get_component_rich_style("screen--selection")
        highlighted = [
            segment
            for segment in terminal.render_line(0)
            if segment.style is not None and segment.style.bgcolor == selection_style.bgcolor
        ]
        assert selection_style.bgcolor not in plain
        assert "".join(segment.text for segment in highlighted) == "中b"
        assert sum(segment.cell_length for segment in highlighted) == 3
        # Still the program's red: the theme here gives selected text no color of its own.
        assert {segment.style.color for segment in highlighted if segment.style is not None} == {
            Style.parse("color(1)").color
        }


@pytest.mark.parametrize(
    ("start", "end", "selected"),
    [(0, 5, "e\u0301👩\u200d👧"), (5, 6, "C"), (2, 6, "👩\u200d👧C")],
    ids=["up-to-its-end", "from-its-end", "from-its-start"],
)
async def test_selection_ending_at_a_joiner_sequence_leaves_it_whole(start: int, end: int, selected: str) -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("e\u0301👩\u200d👧C")
        app.query_one("#elsewhere").focus()
        await wait_for(lambda: not terminal.has_focus, pilot=pilot, description="no cursor in the way")

        select_text(terminal, Offset(start, 0), Offset(end, 0))
        await pilot.pause()

        strip = terminal.render_line(0)
        selection_style = app.screen.get_component_rich_style("screen--selection")
        highlighted = "".join(
            segment.text
            for segment in strip
            if segment.style is not None and segment.style.bgcolor == selection_style.bgcolor
        )
        assert strip.text.rstrip() == "e\u0301👩\u200d👧C"
        assert highlighted == selected


async def test_selection_to_the_end_of_a_row_is_drawn_to_the_edge() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("ab\r\ncd")
        select_text(terminal, Offset(1, 0), Offset(1, 1))
        await pilot.pause()

        selection_style = app.screen.get_component_rich_style("screen--selection")
        first = terminal.render_line(0)
        highlighted = sum(
            segment.cell_length
            for segment in first
            if segment.style is not None and segment.style.bgcolor == selection_style.bgcolor
        )

        assert first.cell_length == 38
        assert highlighted == 37


async def test_width_change_drops_a_selection_and_a_height_change_keeps_it() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        await terminal.write("some text")
        select_text(terminal, Offset(0, 0), Offset(4, 0))

        await pilot.resize_terminal(40, 8)
        await wait_for(lambda: terminal.height == 7, pilot=pilot, description="terminal seven rows high")
        assert terminal in app.screen.selections

        # Rewrapping moves text to other rows and columns, away from under what was selected.
        await pilot.resize_terminal(30, 8)
        await wait_for(lambda: terminal.width == 28, pilot=pilot, description="terminal narrower")
        assert terminal not in app.screen.selections


async def test_reset_drops_a_selection() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)):
        terminal = app.terminal
        await terminal.write("some text")
        select_text(terminal, Offset(0, 0), Offset(4, 0))

        terminal.reset()

        assert terminal not in app.screen.selections


async def test_text_cannot_be_selected_while_the_program_has_the_pointer_or_the_whole_screen() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)):
        terminal = app.terminal
        assert terminal.allow_select

        await terminal.write("\x1b[?1000h")
        assert not terminal.allow_select
        await terminal.write("\x1b[?1000l")
        assert terminal.allow_select

        await terminal.write("\x1b[?1049h")
        assert not terminal.allow_select
        await terminal.write("\x1b[?1049l")
        assert terminal.allow_select
