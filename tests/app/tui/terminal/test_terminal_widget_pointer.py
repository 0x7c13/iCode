# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The terminal widget under the pointer: reports to a program that tracks it, scrolling otherwise."""

from __future__ import annotations

import pytest
from textual import events

from chrys.app.tui.terminal import widget as widget_module
from chrys.app.tui.terminal.widget import Terminal
from tests.app.tui.terminal._widget_harness import (
    TerminalApp,
    connect_stdin,
    numbered_lines,
    post_mouse_event,
    shown_lines,
)
from tests.support.waiting import wait_for

_LEFT, _MIDDLE, _RIGHT = 1, 2, 3

# Long enough that a motion report waiting for its turn waits for the whole of a test.
_NEVER = 60.0
# Short enough that it is sent while the test waits for it.
_SOON = 0.001


# -- buttons -----------------------------------------------------------------------------------------


async def test_pointer_is_not_reported_to_a_program_that_never_asked() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)

        await post_mouse_event(pilot, terminal, events.MouseDown, 3, 4, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseMove, 5, 4, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseUp, 5, 4, button=_LEFT)

        assert stdin.writes == []
        assert app.mouse_captured is None


async def test_press_and_release_are_reported_in_one_based_cells() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1000h\x1b[?1006h")

        await post_mouse_event(pilot, terminal, events.MouseDown, 3, 4, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseUp, 3, 4, button=_LEFT)

        assert stdin.writes == ["\x1b[<0;4;5M", "\x1b[<0;4;5m"]


@pytest.mark.parametrize(("button", "code"), [(_LEFT, 0), (_MIDDLE, 1), (_RIGHT, 2)], ids=["left", "middle", "right"])
async def test_release_names_its_button(button: int, code: int) -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1000h\x1b[?1006h")

        await post_mouse_event(pilot, terminal, events.MouseDown, 0, 0, button=button)
        await post_mouse_event(pilot, terminal, events.MouseUp, 0, 0, button=button)

        assert stdin.writes == [f"\x1b[<{code};1;1M", f"\x1b[<{code};1;1m"]


async def test_held_modifiers_ride_on_the_report() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1000h\x1b[?1006h")

        await post_mouse_event(pilot, terminal, events.MouseDown, 0, 0, button=_LEFT, shift=True, ctrl=True)
        await post_mouse_event(pilot, terminal, events.MouseUp, 0, 0, button=_LEFT, meta=True)

        assert stdin.writes == ["\x1b[<20;1;1M", "\x1b[<8;1;1m"]


async def test_report_is_spelled_in_the_format_the_program_selected() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1000h")

        await post_mouse_event(pilot, terminal, events.MouseDown, 3, 4, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseUp, 3, 4, button=_LEFT)

        # One character each for button, column and row, offset by 32; a release names no button.
        assert stdin.writes == ["\x1b[M $%", "\x1b[M#$%"]


async def test_program_that_asked_for_presses_only_hears_of_no_release_and_the_pointer_is_let_go() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?9h\x1b[?1006h")

        await post_mouse_event(pilot, terminal, events.MouseDown, 3, 4, button=_LEFT)
        assert app.mouse_captured is terminal
        await post_mouse_event(pilot, terminal, events.MouseUp, 3, 4, button=_LEFT)

        assert stdin.writes == ["\x1b[<0;4;5M"]
        assert app.mouse_captured is None


async def test_cells_are_counted_from_the_screen_not_from_the_history_above_it() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write(numbered_lines(30) + "$ \x1b[?1000h\x1b[?1006h")
        await pilot.pause()
        assert terminal.emulator.buffer.top == 21

        await post_mouse_event(pilot, terminal, events.MouseDown, 3, 4, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseUp, 3, 4, button=_LEFT)

        assert stdin.writes == ["\x1b[<0;4;5M", "\x1b[<0;4;5m"]


async def test_view_scrolled_into_history_reports_only_what_is_still_on_the_screen() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write(numbered_lines(30) + "$ \x1b[?1000h\x1b[?1006h")
        await pilot.pause()
        terminal.scroll_to(y=18, animate=False)
        await pilot.pause()
        assert shown_lines(terminal)[:4] == ["line 18", "line 19", "line 20", "line 21"]

        # The first three lines shown are history, which the program knows nothing of.
        await post_mouse_event(pilot, terminal, events.MouseDown, 3, 1, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseUp, 3, 1, button=_LEFT)
        assert stdin.writes == []
        assert app.mouse_captured is None

        # The fourth is the first row of its screen.
        await post_mouse_event(pilot, terminal, events.MouseDown, 3, 3, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseUp, 3, 3, button=_LEFT)
        assert stdin.writes == ["\x1b[<0;4;1M", "\x1b[<0;4;1m"]


@pytest.mark.parametrize("tracking", [False, True], ids=["plain", "tracked"])
async def test_click_focuses_the_terminal(tracking: bool) -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        if tracking:
            await terminal.write("\x1b[?1000h\x1b[?1006h")
        app.query_one("#elsewhere").focus()
        await wait_for(lambda: not terminal.has_focus, pilot=pilot, description="focus elsewhere")

        await pilot.click(terminal, offset=(2, 2))
        await wait_for(lambda: terminal.has_focus, pilot=pilot, description="terminal focused by the click")


# -- motion ------------------------------------------------------------------------------------------


async def test_drag_is_reported_at_a_measured_pace_the_latest_cell_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(widget_module, "_MOTION_REPORT_INTERVAL", _NEVER)
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1002h\x1b[?1006h")

        await post_mouse_event(pilot, terminal, events.MouseDown, 3, 4, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseMove, 4, 4, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseMove, 5, 4, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseMove, 6, 5, button=_LEFT)
        assert stdin.writes == ["\x1b[<0;4;5M"]

        # The release must not overtake the motion before it, so that goes out first.
        await post_mouse_event(pilot, terminal, events.MouseUp, 6, 5, button=_LEFT)
        assert stdin.writes == ["\x1b[<0;4;5M", "\x1b[<32;7;6M", "\x1b[<0;7;6m"]


async def test_motion_within_one_cell_is_reported_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(widget_module, "_MOTION_REPORT_INTERVAL", _SOON)
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1002h\x1b[?1006h")
        await post_mouse_event(pilot, terminal, events.MouseDown, 3, 4, button=_LEFT)

        await post_mouse_event(pilot, terminal, events.MouseMove, 5, 4, button=_LEFT)
        await wait_for(lambda: len(stdin.writes) == 2, pilot=pilot, description="motion reported")
        await post_mouse_event(pilot, terminal, events.MouseMove, 5, 4, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseMove, 6, 4, button=_LEFT)
        await wait_for(lambda: len(stdin.writes) == 3, pilot=pilot, description="motion into the next cell reported")
        await post_mouse_event(pilot, terminal, events.MouseUp, 6, 4, button=_LEFT)

        # In order, with nothing between the two cells: the second event in cell 6 said nothing new.
        assert stdin.writes == ["\x1b[<0;4;5M", "\x1b[<32;6;5M", "\x1b[<32;7;5M", "\x1b[<0;7;5m"]


async def test_drag_keeps_the_pointer_and_reports_from_the_edge_it_left_by(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(widget_module, "_MOTION_REPORT_INTERVAL", _NEVER)
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1002h\x1b[?1006h")

        await post_mouse_event(pilot, terminal, events.MouseDown, 3, 4, button=_LEFT)
        assert app.mouse_captured is terminal

        # Over the button below the terminal, past the last column: still the terminal's drag.
        await post_mouse_event(pilot, terminal, events.MouseMove, 39, 10, button=_LEFT)
        await post_mouse_event(pilot, terminal, events.MouseUp, 39, 10, button=_LEFT)

        assert stdin.writes == ["\x1b[<0;4;5M", "\x1b[<32;38;10M", "\x1b[<0;38;10m"]
        assert app.mouse_captured is None


async def test_motion_without_a_button_is_reported_only_to_a_program_tracking_all_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(widget_module, "_MOTION_REPORT_INTERVAL", _SOON)
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)

        await terminal.write("\x1b[?1002h\x1b[?1006h")
        await post_mouse_event(pilot, terminal, events.MouseMove, 5, 4)
        assert stdin.writes == []
        assert terminal._pending_motion is None

        await terminal.write("\x1b[?1003h")
        await post_mouse_event(pilot, terminal, events.MouseMove, 6, 4)
        await wait_for(lambda: stdin.writes == ["\x1b[<35;7;5M"], pilot=pilot, description="motion reported")


async def test_press_lets_waiting_motion_out_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(widget_module, "_MOTION_REPORT_INTERVAL", _NEVER)
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1003h\x1b[?1006h")

        await post_mouse_event(pilot, terminal, events.MouseMove, 5, 4)
        assert stdin.writes == []
        await post_mouse_event(pilot, terminal, events.MouseDown, 5, 4, button=_LEFT)

        assert stdin.writes == ["\x1b[<35;6;5M", "\x1b[<0;6;5M"]


def _motion_over(terminal: Terminal, x: int, y: int) -> events.MouseMove:
    origin = terminal.region.offset
    return events.MouseMove(terminal, x, y, 0, 0, 0, False, False, False, screen_x=origin.x + x, screen_y=origin.y + y)


@pytest.mark.parametrize(
    "goodbye",
    [
        pytest.param("\x1b[?1003l", id="tracking switched off"),
        pytest.param("\x1b[?1003l\x1b[?1000h", id="tracking that leaves motion out"),
        pytest.param("\x1b[?1006l", id="reports spelled another way"),
        pytest.param("\x1bc", id="terminal reset by the program"),
    ],
)
async def test_motion_still_waiting_when_the_program_stops_listening_is_never_sent(
    goodbye: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(widget_module, "_MOTION_REPORT_INTERVAL", _SOON)
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1003h\x1b[?1006h")

        # Handed over directly, and followed by the program's goodbye without a turn of the loop
        # between them: however soon the timer is due, it finds the program already gone. What reads
        # the input by then is a shell prompt.
        terminal.on_mouse_move(_motion_over(terminal, 5, 4))
        assert terminal._pending_motion is not None
        await terminal.write(goodbye)

        await wait_for(lambda: terminal._motion_timer is None, pilot=pilot, description="motion timer fired")
        assert terminal._pending_motion is None
        assert stdin.writes == []

        # Asked again, the pointer is reported where it is: nobody has been told yet. Anything the
        # timer let through would be ahead of this.
        await terminal.write("\x1b[?1003h\x1b[?1006h")
        await post_mouse_event(pilot, terminal, events.MouseMove, 5, 4)
        await wait_for(lambda: bool(stdin.writes), pilot=pilot, description="motion reported")
        assert stdin.writes == ["\x1b[<35;6;5M"]


async def test_reset_forgets_motion_waiting_to_be_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(widget_module, "_MOTION_REPORT_INTERVAL", _NEVER)
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1003h\x1b[?1006h")
        await post_mouse_event(pilot, terminal, events.MouseMove, 5, 4)
        timer = terminal._motion_timer
        assert timer is not None

        terminal.reset()

        assert terminal._pending_motion is None
        assert terminal._motion_timer is None
        # The next program asks for the same reports: it hears of its own pointer, not the last one's.
        await terminal.write("\x1b[?1003h\x1b[?1006h")
        await post_mouse_event(pilot, terminal, events.MouseDown, 8, 2, button=_LEFT)
        assert stdin.writes == ["\x1b[<0;9;3M"]


# -- the wheel ---------------------------------------------------------------------------------------


async def test_wheel_is_reported_to_a_program_tracking_the_pointer_and_scrolls_nothing() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write(numbered_lines(30) + "$ \x1b[?1000h\x1b[?1006h")
        await pilot.pause()

        await post_mouse_event(pilot, terminal, events.MouseScrollUp, 3, 4)
        await post_mouse_event(pilot, terminal, events.MouseScrollDown, 3, 4)
        await post_mouse_event(pilot, terminal, events.MouseScrollLeft, 3, 4)
        await post_mouse_event(pilot, terminal, events.MouseScrollRight, 3, 4)
        await pilot.pause()

        assert stdin.writes == ["\x1b[<64;4;5M", "\x1b[<65;4;5M", "\x1b[<66;4;5M", "\x1b[<67;4;5M"]
        assert terminal.scroll_y == terminal.max_scroll_y == 21


async def test_wheel_moves_the_cursor_of_a_full_screen_program() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1049h")

        await post_mouse_event(pilot, terminal, events.MouseScrollUp, 3, 4)
        await post_mouse_event(pilot, terminal, events.MouseScrollDown, 3, 4)
        assert stdin.writes == ["\x1b[A", "\x1b[B"]

        # The cursor keys such a program asked for, in application mode, are the ones it gets.
        await terminal.write("\x1b[?1h")
        await post_mouse_event(pilot, terminal, events.MouseScrollUp, 3, 4)
        await post_mouse_event(pilot, terminal, events.MouseScrollDown, 3, 4)
        assert stdin.writes[2:] == ["\x1bOA", "\x1bOB"]

        # Sideways there is no cursor key to stand for the wheel.
        await post_mouse_event(pilot, terminal, events.MouseScrollLeft, 3, 4)
        assert len(stdin.writes) == 4


async def test_wheel_speaks_the_key_protocol_the_full_screen_program_asked_for() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        # Application cursor keys and the kitty protocol both: the second knows nothing of the first,
        # and a wheel that answered in SS3 would be typing at a program that reads keys one way only.
        await terminal.write("\x1b[?1049h\x1b[?1h\x1b[>1u")

        await post_mouse_event(pilot, terminal, events.MouseScrollUp, 3, 4)
        await post_mouse_event(pilot, terminal, events.MouseScrollDown, 3, 4)
        assert stdin.writes == ["\x1b[A", "\x1b[B"]

        await terminal.write("\x1b[<u")
        await post_mouse_event(pilot, terminal, events.MouseScrollUp, 3, 4)
        assert stdin.writes[2:] == ["\x1bOA"]


async def test_full_screen_program_can_decline_the_wheel() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1049h\x1b[?1007l")

        await post_mouse_event(pilot, terminal, events.MouseScrollUp, 3, 4)
        assert stdin.writes == []

        await terminal.write("\x1b[?1007h")
        await post_mouse_event(pilot, terminal, events.MouseScrollUp, 3, 4)
        assert stdin.writes == ["\x1b[A"]


async def test_full_screen_program_tracking_the_pointer_gets_the_wheel_as_a_report() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write("\x1b[?1049h\x1b[?1000h\x1b[?1006h")

        await post_mouse_event(pilot, terminal, events.MouseScrollUp, 3, 4)

        assert stdin.writes == ["\x1b[<64;4;5M"]


async def test_wheel_scrolls_the_history_step_by_step_and_back_to_following() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        stdin = connect_stdin(terminal)
        await terminal.write(numbered_lines(30) + "$ ")
        await pilot.pause()
        step = app.scroll_sensitivity_y
        assert 0 < step < 21

        # One step up from the bottom, not a leap to the first row.
        await post_mouse_event(pilot, terminal, events.MouseScrollUp, 3, 4)
        await wait_for(lambda: terminal.scroll_y == 21 - step, pilot=pilot, description="scrolled one step up")
        assert shown_lines(terminal)[0] == f"line {21 - int(step)}"

        await post_mouse_event(pilot, terminal, events.MouseScrollDown, 3, 4)
        await wait_for(lambda: terminal.scroll_y == 21, pilot=pilot, description="scrolled back down")
        await terminal.write("more\r\n$ ")
        await pilot.pause()

        assert terminal.scroll_y == terminal.max_scroll_y == 22
        assert shown_lines(terminal)[-2:] == ["$ more", "$"]
        assert stdin.writes == []


async def test_wheel_scrolls_on_from_where_trimmed_history_left_the_view() -> None:
    app = TerminalApp()
    async with app.run_test(size=(40, 11)) as pilot:
        terminal = app.terminal
        terminal.emulator = widget_module.TerminalEmulator(terminal.width, terminal.height, history_limit=20)
        await terminal.write(numbered_lines(30))
        await pilot.pause()
        terminal.scroll_to(y=12, animate=False)
        await pilot.pause()
        await terminal.write("".join(f"more {number}\r\n" for number in range(5)))
        await pilot.pause()
        assert terminal.scroll_y == 7
        step = app.scroll_sensitivity_y

        await post_mouse_event(pilot, terminal, events.MouseScrollUp, 3, 4)
        await wait_for(lambda: terminal.scroll_y == 7 - step, pilot=pilot, description="scrolled one step up")
