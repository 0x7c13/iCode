# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The emulator's modes, what it answers when asked, and what it hears from the shell."""

from __future__ import annotations

import base64
import socket
from collections.abc import Callable

import pytest

from chrys.app.tui.terminal.emulator import (
    CommandSubmitted,
    CursorShape,
    DirectoryChanged,
    KeyProtocol,
    MouseEncoding,
    MouseTracking,
    Pen,
    TerminalEmulator,
)


def fed(*streams: str, columns: int = 20, lines: int = 5) -> TerminalEmulator:
    emulator = TerminalEmulator(columns, lines)
    for stream in streams:
        emulator.feed(stream)
    return emulator


def replies(stream: str, *, after: str = "", columns: int = 20, lines: int = 5) -> str:
    """What the terminal answers to a stream, once it has been fed what comes before."""
    return fed(after, columns=columns, lines=lines).feed(stream).replies


def b64(text: str) -> str:
    return "b64:" + base64.b64encode(text.encode()).decode()


# -- reports -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("request_", "reply"),
    [
        ("\x1b[c", "\x1b[?62;22c"),
        ("\x1b[0c", "\x1b[?62;22c"),
        ("\x1b[1c", ""),
        ("\x1b[>c", "\x1b[>1;0;0c"),
        ("\x1b[>0c", "\x1b[>1;0;0c"),
        ("\x1b[>1c", ""),
        ("\x1b[5n", "\x1b[0n"),
        ("\x1b[n", ""),
        ("\x1b[6n", "\x1b[1;1R"),
        ("\x1b[?6n", "\x1b[?1;1R"),
        ("\x1b[?5n", ""),
        ("\x1b[18t", "\x1b[8;5;20t"),
        ("\x1b[14t", ""),
        ("\x1b[?u", "\x1b[?0u"),
        ("\x1b[=c", ""),
    ],
)
def test_reports(request_: str, reply: str) -> None:
    assert replies(request_) == reply


def test_cursor_position_report_counts_from_the_top_of_the_screen() -> None:
    assert replies("\x1b[6n", after="one\r\ntwo\r\nthree\r\nfour\r\nPS> ", lines=3) == "\x1b[3;5R"


def test_cursor_position_report_on_the_alternate_screen() -> None:
    assert replies("\x1b[6n", after="\x1b[?1049h\x1b[2;4H", lines=3) == "\x1b[2;4R"


def test_cursor_position_report_counts_from_the_margin_in_origin_mode() -> None:
    assert replies("\x1b[6n\x1b[?6n", after="\x1b[2;4r\x1b[?6h\x1b[2;3H") == "\x1b[2;3R\x1b[?2;3R"


def test_cursor_position_report_with_a_wrap_pending_names_the_last_column() -> None:
    assert replies("\x1b[6n", after="abcde", columns=5) == "\x1b[1;5R"


def test_size_report_follows_a_resize() -> None:
    emulator = fed()
    emulator.resize(33, 7)

    assert emulator.feed("\x1b[18t").replies == "\x1b[8;7;33t"


def test_replies_come_in_the_order_they_were_asked_for() -> None:
    assert replies("\x1b[5n\x1b[c\x1b[2;2H\x1b[6n") == "\x1b[0n\x1b[?62;22c\x1b[2;2R"


@pytest.mark.parametrize(
    ("after", "mode", "status"),
    [
        ("", 4, 2),
        ("\x1b[4h", 4, 1),
        ("\x1b[4h\x1b[4l", 4, 2),
        ("", 20, 2),
        ("\x1b[20h", 20, 1),
        ("\x1b[4;20h", 20, 1),
        ("", 2, 0),
        ("", 12, 0),
    ],
)
def test_ansi_mode_report(after: str, mode: int, status: int) -> None:
    assert replies(f"\x1b[{mode}$p", after=after) == f"\x1b[{mode};{status}$y"


@pytest.mark.parametrize(
    ("after", "mode", "status"),
    [
        ("", 1, 2),
        ("\x1b[?1h", 1, 1),
        ("", 6, 2),
        ("\x1b[?6h", 6, 1),
        ("", 7, 1),
        ("\x1b[?7l", 7, 2),
        ("", 12, 2),
        ("\x1b[?12h", 12, 1),
        ("", 25, 1),
        ("\x1b[?25l", 25, 2),
        ("", 66, 2),
        ("\x1b=", 66, 1),
        ("\x1b=\x1b>", 66, 2),
        ("\x1b[?66h", 66, 1),
        ("", 1004, 2),
        ("\x1b[?1004h", 1004, 1),
        ("", 1007, 1),
        ("\x1b[?1007l", 1007, 2),
        ("", 2004, 2),
        ("\x1b[?2004h", 2004, 1),
        ("", 1049, 2),
        ("\x1b[?1049h", 1049, 1),
        ("\x1b[?1049h", 47, 1),
        ("\x1b[?47h", 1047, 1),
        ("\x1b[?1049h\x1b[?1049l", 1049, 2),
        ("", 1000, 2),
        ("\x1b[?1000h", 1000, 1),
        ("\x1b[?1002h", 1000, 2),
        ("\x1b[?1002h", 1002, 1),
        ("\x1b[?1003h", 1003, 1),
        ("\x1b[?9h", 9, 1),
        ("", 1006, 2),
        ("\x1b[?1006h", 1006, 1),
        ("\x1b[?1006h", 1005, 2),
        ("\x1b[?1015h", 1015, 1),
        ("", 9999, 0),
        ("", 2026, 0),
        ("", 0, 0),
    ],
)
def test_private_mode_report(after: str, mode: int, status: int) -> None:
    assert replies(f"\x1b[?{mode}$p", after=after) == f"\x1b[?{mode};{status}$y"


def test_scrolling_region_setting_report() -> None:
    assert replies("\x1bP$qr\x1b\\") == "\x1bP1$r1;5r\x1b\\"
    assert replies("\x1bP$qr\x1b\\", after="\x1b[2;4r") == "\x1bP1$r2;4r\x1b\\"


@pytest.mark.parametrize("style", range(7))
def test_cursor_style_setting_report_names_the_shape_in_force(style: int) -> None:
    emulator = fed(f"\x1b[{style} q")
    shape = emulator.cursor_shape

    reply = emulator.feed("\x1bP$q q\x1b\\").replies
    assert reply.startswith("\x1bP1$r")
    assert reply.endswith(" q\x1b\\")

    # Sent back as a DECSCUSR, the answer selects the shape it describes.
    other = fed(f"\x1b[{reply.removeprefix('\x1bP1$r').removesuffix('\x1b\\')}")
    assert other.cursor_shape is shape


@pytest.mark.parametrize(("style", "reported"), [(0, 2), (1, 2), (2, 2), (3, 4), (4, 4), (5, 6), (6, 6)])
def test_cursor_style_setting_report_names_the_steady_variant(style: int, reported: int) -> None:
    """The cursor is drawn without blinking, so that is what every shape reports."""
    assert fed(f"\x1b[{style} q").feed("\x1bP$q q\x1b\\").replies == f"\x1bP1$r{reported} q\x1b\\"


@pytest.mark.parametrize("setting", ["m", '"p', "", "xyz"])
def test_setting_this_terminal_does_not_report_is_refused(setting: str) -> None:
    assert replies(f"\x1bP$q{setting}\x1b\\") == "\x1bP0$r\x1b\\"


def test_other_device_control_strings_are_ignored() -> None:
    emulator = fed()

    update = emulator.feed("\x1bP+q544e\x1b\\\x1bP1000p\x1b\\text")

    assert update.replies == ""
    assert emulator.buffer.screen_text[0] == "text"


# -- switches ----------------------------------------------------------------------------------------


def test_modes_a_new_terminal_starts_in() -> None:
    emulator = TerminalEmulator()

    assert (emulator.columns, emulator.lines) == (80, 24)
    assert emulator.cursor_visible
    assert emulator.alternate_scroll
    assert not emulator.alternate_screen
    assert not emulator.application_cursor_keys
    assert not emulator.bracketed_paste
    assert not emulator.focus_reporting
    assert emulator.cursor_shape is CursorShape.BLOCK
    assert emulator.mouse_tracking is MouseTracking.OFF
    assert emulator.mouse_encoding is MouseEncoding.DEFAULT
    assert emulator.key_protocol is KeyProtocol.LEGACY


@pytest.mark.parametrize(
    ("mode", "read", "at_reset"),
    [
        pytest.param(1, lambda emulator: emulator.application_cursor_keys, False, id="application_cursor_keys"),
        pytest.param(25, lambda emulator: emulator.cursor_visible, True, id="cursor_visible"),
        pytest.param(1004, lambda emulator: emulator.focus_reporting, False, id="focus_reporting"),
        pytest.param(1007, lambda emulator: emulator.alternate_scroll, True, id="alternate_scroll"),
        pytest.param(2004, lambda emulator: emulator.bracketed_paste, False, id="bracketed_paste"),
    ],
)
def test_switch(mode: int, read: Callable[[TerminalEmulator], bool], at_reset: bool) -> None:
    emulator = fed()
    states = []
    for sequence in ("", f"\x1b[?{mode}h", f"\x1b[?{mode}h", f"\x1b[?{mode}l", f"\x1b[?{mode}l"):
        emulator.feed(sequence)
        states.append(read(emulator))

    assert states == [at_reset, True, True, False, False]


def test_several_modes_in_one_sequence() -> None:
    emulator = fed("\x1b[?1;1004;2004;1000;1006h")

    assert emulator.application_cursor_keys
    assert emulator.focus_reporting
    assert emulator.bracketed_paste
    assert emulator.mouse_tracking is MouseTracking.PRESS_RELEASE
    assert emulator.mouse_encoding is MouseEncoding.SGR

    emulator.feed("\x1b[?1;2004l")
    assert not emulator.application_cursor_keys
    assert not emulator.bracketed_paste
    assert emulator.focus_reporting


def test_unknown_modes_are_ignored() -> None:
    emulator = fed("\x1b[?9999h\x1b[?0h\x1b[99h\x1b[?2026h\x1b[?1h")

    assert emulator.application_cursor_keys
    assert emulator.buffer.screen_text == [""] * 5


def test_ansi_and_private_modes_with_the_same_number_are_different_modes() -> None:
    # ANSI 4 is insert mode; DEC private 4 is smooth scroll, which means nothing here.
    assert fed("\x1b[?4hab\rX").buffer.screen_text[0] == "Xb"
    assert fed("\x1b[4hab\rX").buffer.screen_text[0] == "Xab"
    # And the other way around: private 25 shows the cursor, ANSI 25 is nothing.
    assert fed("\x1b[25l").cursor_visible


@pytest.mark.parametrize("tracking", [tracking for tracking in MouseTracking if tracking is not MouseTracking.OFF])
def test_mouse_tracking_is_switched_on_and_off_by_its_mode(tracking: MouseTracking) -> None:
    emulator = fed(f"\x1b[?{tracking.value}h")
    assert emulator.mouse_tracking is tracking

    emulator.feed(f"\x1b[?{tracking.value}l")
    assert emulator.mouse_tracking is MouseTracking.OFF


def test_newest_mouse_tracking_mode_wins_and_only_its_reset_switches_it_off() -> None:
    emulator = fed("\x1b[?1000h\x1b[?1003h")
    assert emulator.mouse_tracking is MouseTracking.MOTION

    # Programs reset every mode they know of on the way out, in any order.
    emulator.feed("\x1b[?1000l\x1b[?1002l\x1b[?9l")
    assert emulator.mouse_tracking is MouseTracking.MOTION

    emulator.feed("\x1b[?1003l")
    assert emulator.mouse_tracking is MouseTracking.OFF


@pytest.mark.parametrize("encoding", [encoding for encoding in MouseEncoding if encoding is not MouseEncoding.DEFAULT])
def test_mouse_encoding_is_switched_on_and_off_by_its_mode(encoding: MouseEncoding) -> None:
    emulator = fed(f"\x1b[?{encoding.value}h")
    assert emulator.mouse_encoding is encoding

    emulator.feed("\x1b[?1005l\x1b[?1006l\x1b[?1015l")
    assert emulator.mouse_encoding is MouseEncoding.DEFAULT


def test_resetting_another_mouse_encoding_keeps_the_one_in_force() -> None:
    emulator = fed("\x1b[?1006h\x1b[?1015l\x1b[?1005l")

    assert emulator.mouse_encoding is MouseEncoding.SGR


def test_mouse_modes_outlive_a_screen_switch() -> None:
    emulator = fed("\x1b[?1049h\x1b[?1002h\x1b[?1006h\x1b[?1049l")

    assert emulator.mouse_tracking is MouseTracking.DRAG
    assert emulator.mouse_encoding is MouseEncoding.SGR


@pytest.mark.parametrize(
    ("style", "shape"),
    [
        (0, CursorShape.BLOCK),
        (1, CursorShape.BLOCK),
        (2, CursorShape.BLOCK),
        (3, CursorShape.UNDERLINE),
        (4, CursorShape.UNDERLINE),
        (5, CursorShape.BAR),
        (6, CursorShape.BAR),
    ],
)
def test_cursor_shape(style: int, shape: CursorShape) -> None:
    assert fed("\x1b[5 q", f"\x1b[{style} q").cursor_shape is (shape)
    assert fed("\x1b[3 q", f"\x1b[{style} q").cursor_shape is (shape)


def test_cursor_shape_defaults_to_a_block() -> None:
    assert fed("\x1b[5 q\x1b[ q").cursor_shape is CursorShape.BLOCK


@pytest.mark.parametrize("style", [7, 8, 100, 65535])
def test_cursor_shape_that_does_not_exist_is_ignored(style: int) -> None:
    assert fed(f"\x1b[5 q\x1b[{style} q").cursor_shape is CursorShape.BAR


# -- key protocols -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stream", "protocol"),
    [
        ("\x1b[>4;2m", KeyProtocol.MODIFY_OTHER_KEYS_2),
        ("\x1b[>4;1m", KeyProtocol.MODIFY_OTHER_KEYS_1),
        ("\x1b[>4;2m\x1b[>4;1m", KeyProtocol.MODIFY_OTHER_KEYS_1),
        # Level 3 is not on offer; the most there is answers for it.
        ("\x1b[>4;3m", KeyProtocol.MODIFY_OTHER_KEYS_2),
        ("\x1b[>4;2m\x1b[>4;0m", KeyProtocol.LEGACY),
        ("\x1b[>4;2m\x1b[>4m", KeyProtocol.LEGACY),
        ("\x1b[>4;2m\x1b[>4n", KeyProtocol.LEGACY),
        ("\x1b[>1;2m", KeyProtocol.LEGACY),
        ("\x1b[>4;2m\x1b[>1m", KeyProtocol.MODIFY_OTHER_KEYS_2),
        ("\x1b[>4;2m\x1b[>2n", KeyProtocol.MODIFY_OTHER_KEYS_2),
        ("\x1b[>1u", KeyProtocol.KITTY),
        ("\x1b[>1u\x1b[<u", KeyProtocol.LEGACY),
        ("\x1b[>u", KeyProtocol.LEGACY),
        ("\x1b[=1u", KeyProtocol.KITTY),
        ("\x1b[=1;1u", KeyProtocol.KITTY),
        ("\x1b[=1u\x1b[=0u", KeyProtocol.LEGACY),
        ("\x1b[=1;2u", KeyProtocol.KITTY),
        ("\x1b[=1u\x1b[=0;2u", KeyProtocol.KITTY),
        ("\x1b[=1u\x1b[=1;3u", KeyProtocol.LEGACY),
        ("\x1b[=1u\x1b[=2;3u", KeyProtocol.KITTY),
        # Kitty's protocol, being the one a program has to ask for last, wins while it is on.
        ("\x1b[>4;2m\x1b[>1u", KeyProtocol.KITTY),
        ("\x1b[>4;2m\x1b[>1u\x1b[<u", KeyProtocol.MODIFY_OTHER_KEYS_2),
        ("\x1b[>4;1m\x1b[>1u\x1b[<u", KeyProtocol.MODIFY_OTHER_KEYS_1),
    ],
)
def test_key_protocol(stream: str, protocol: KeyProtocol) -> None:
    assert fed(stream).key_protocol is protocol


@pytest.mark.parametrize(
    ("stream", "flags"),
    [
        ("", 0),
        ("\x1b[>1u", 1),
        ("\x1b[>31u", 1),
        ("\x1b[>30u", 0),
        ("\x1b[=31u", 1),
        ("\x1b[=2u", 0),
        ("\x1b[>1u\x1b[>0u", 0),
        ("\x1b[>1u\x1b[>0u\x1b[<u", 1),
        ("\x1b[>1u\x1b[>0u\x1b[<1u", 1),
        ("\x1b[>1u\x1b[>1u\x1b[<2u", 0),
        ("\x1b[>1u\x1b[<99u", 0),
        ("\x1b[<u", 0),
        ("\x1b[=1u\x1b[<u", 0),
    ],
)
def test_kitty_flags_report_what_took(stream: str, flags: int) -> None:
    assert replies("\x1b[?u", after=stream) == f"\x1b[?{flags}u"


def test_popping_every_kitty_entry_still_leaves_flags_to_set() -> None:
    assert replies("\x1b[?u", after="\x1b[<9u\x1b[=1u") == "\x1b[?1u"


def test_kitty_stack_forgets_its_oldest_entries() -> None:
    emulator = fed("\x1b[>1u" * 40)

    emulator.feed("\x1b[<15u")
    assert emulator.key_protocol is KeyProtocol.KITTY

    emulator.feed("\x1b[<u")
    assert emulator.key_protocol is KeyProtocol.LEGACY


def test_each_screen_has_its_own_kitty_flags() -> None:
    emulator = fed("\x1b[?1049h\x1b[>1u")
    assert emulator.key_protocol is KeyProtocol.KITTY

    # The program died without popping: the shell it leaves behind still reads plain keys.
    emulator.feed("\x1b[?1049l")
    assert emulator.key_protocol is KeyProtocol.LEGACY

    emulator.feed("\x1b[?1049h")
    assert emulator.key_protocol is KeyProtocol.LEGACY


def test_primary_screen_kitty_flags_come_back_after_the_alternate_screen() -> None:
    emulator = fed("\x1b[>1u\x1b[?1049h")
    assert emulator.key_protocol is KeyProtocol.LEGACY

    emulator.feed("\x1b[?1049l")
    assert emulator.key_protocol is KeyProtocol.KITTY


# -- hyperlinks --------------------------------------------------------------------------------------


def links(emulator: TerminalEmulator, y: int = 0) -> list[str | None]:
    return [pen.link for pen in emulator.buffer.row(y).pens]


def test_hyperlink_covers_the_text_written_while_it_is_open() -> None:
    emulator = fed("a\x1b]8;;https://example.com\x1b\\bc\x1b]8;;\x1b\\d")

    assert links(emulator) == [None, "https://example.com", "https://example.com", None]
    assert emulator.pen == Pen()


def test_hyperlink_parameters_are_not_part_of_the_address() -> None:
    emulator = fed("\x1b]8;id=42:foo=bar;https://example.com/?a=1;b=2\x07x")

    assert links(emulator) == ["https://example.com/?a=1;b=2"]


def test_hyperlink_is_replaced_by_the_next_one() -> None:
    emulator = fed("\x1b]8;;https://a.example\x07a\x1b]8;;https://b.example\x07b")

    assert links(emulator) == ["https://a.example", "https://b.example"]


def test_hyperlink_is_not_a_rendition() -> None:
    emulator = fed("\x1b]8;;https://example.com\x07\x1b[1ma\x1b[0mb\x1b[mc")

    assert links(emulator) == ["https://example.com"] * 3


def test_hyperlink_runs_across_rows() -> None:
    emulator = fed("\x1b]8;;https://example.com\x07abcdefg\r\nh", columns=5)

    assert [links(emulator, y) for y in range(3)] == [["https://example.com"] * cells for cells in (5, 2, 1)]


def test_erased_cells_are_not_linked() -> None:
    emulator = fed("abc\x1b]8;;https://example.com\x07\x1b[44m\x1b[1;2H\x1b[K", columns=4)

    assert emulator.buffer.row(0).pens == [Pen(), Pen(background=4), Pen(background=4), Pen(background=4)]


def test_hyperlink_with_nothing_in_it_is_no_link() -> None:
    assert links(fed("\x1b]8\x07\x1b]8;\x07\x1b]8;id=1\x07\x1b]8;id=1;\x07x")) == [None]


# -- what the shell reports --------------------------------------------------------------------------


def events(*streams: str) -> list[DirectoryChanged | CommandSubmitted]:
    emulator = TerminalEmulator(20, 5)
    return [event for stream in streams for event in emulator.feed(stream).events]


@pytest.mark.parametrize(
    ("url", "path"),
    [
        ("file:///home/me/src", "/home/me/src"),
        ("file://localhost/home/me", "/home/me"),
        ("file://LOCALHOST/home/me", "/home/me"),
        ("file://this-machine/home/me", "/home/me"),
        ("file://this-machine.example.com/home/me", "/home/me"),
        ("file://This-Machine.local/home/me", "/home/me"),
        ("file:///tmp/a%20b/%E7%9B%AE%E5%BD%95", "/tmp/a b/目录"),
        ("file:///tmp/semi;colon", "/tmp/semi;colon"),
        ("file:///tmp/100%25", "/tmp/100%"),
        ("file:///C:/Users/me", "C:/Users/me"),
        ("file:///c:/", "c:/"),
        ("file:///C:", "C:"),
        ("file:///Cabinet:/x", "/Cabinet:/x"),
        ("file:///", "/"),
    ],
)
def test_directory_reported_as_a_file_url(url: str, path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "This-Machine.example.com")

    assert events(f"\x1b]7;{url}\x1b\\") == [DirectoryChanged(path)]


@pytest.mark.parametrize(
    "url",
    [
        "file://build-server/home/me",
        "file://this-machine-2/home/me",
        "https://example.com/home/me",
        "sftp://localhost/home/me",
        "/home/me",
        "",
    ],
)
def test_directory_somewhere_else_is_not_ours(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "this-machine")

    assert events(f"\x1b]7;{url}\x1b\\") == []


@pytest.mark.parametrize("url", ["file://[", "file://[::1/tmp", "file://tmp]/x", "file://[not-an-address]/tmp"])
def test_directory_url_that_cannot_be_read_is_ignored_and_costs_nothing_after_it(url: str) -> None:
    emulator = TerminalEmulator(20, 5)

    update = emulator.feed(f"\x1b]7;{url}\x07$ \x1b]7;file:///tmp\x07")

    assert update.events == (DirectoryChanged("/tmp"),)
    assert emulator.buffer.rows[0].text.rstrip() == "$"


@pytest.mark.parametrize(
    ("payload", "path"),
    [
        ("/home/me", "/home/me"),
        ("/tmp/a b/目录", "/tmp/a b/目录"),
        ("/tmp/a;b", "/tmp/a;b"),
        ("C:\\Users\\me", "C:\\Users\\me"),
        (b64("/tmp/a;b c"), "/tmp/a;b c"),
        (b64("/tmp/目录/\x1b[31m"), "/tmp/目录/\x1b[31m"),
        (b64("D:\\Repos\\chrys"), "D:\\Repos\\chrys"),
    ],
)
def test_directory_reported_by_shell_integration(payload: str, path: str) -> None:
    assert events(f"\x1b]2025;{payload}\x1b\\") == [DirectoryChanged(path)]


@pytest.mark.parametrize("payload", ["", "b64:", "b64:!!!!", "b64:L3Rtc", "b64:L3 Rt"])
def test_directory_that_is_empty_or_garbled_is_not_reported(payload: str) -> None:
    assert events(f"\x1b]2025;{payload}\x1b\\", "\x1b]2025\x07") == []


def test_directory_that_is_not_utf8_is_still_reported() -> None:
    payload = "b64:" + base64.b64encode(b"/tmp/\xff").decode()

    assert events(f"\x1b]2025;{payload}\x07") == [DirectoryChanged("/tmp/\ufffd")]


@pytest.mark.parametrize(
    ("payload", "command"),
    [
        ("ls -la", "ls -la"),
        ("echo a;b", "echo a;b"),
        (b64("echo a;b && printf '\\033'"), "echo a;b && printf '\\033'"),
        (b64("echo 你好\nsecond line"), "echo 你好\nsecond line"),
    ],
)
def test_command_reported_by_shell_integration(payload: str, command: str) -> None:
    assert events(f"\x1b]2026;{payload}\x1b\\") == [CommandSubmitted(command)]


def test_garbled_command_is_not_reported() -> None:
    assert events("\x1b]2026;b64:!!!!\x07") == []


@pytest.mark.parametrize("payload", ["", "b64:"])
def test_empty_command_is_not_reported(payload: str) -> None:
    """Enter at an empty prompt ran nothing."""
    assert events(f"\x1b]2026;{payload}\x07") == []


def test_reports_arrive_in_order_with_the_feed_that_completed_them() -> None:
    emulator = TerminalEmulator(20, 5)

    assert emulator.feed("\x1b]2026;make\x07\x1b]2025;/src\x07\x1b]2026;ls").events == (
        CommandSubmitted("make"),
        DirectoryChanged("/src"),
    )
    assert emulator.feed(" -la").events == ()
    assert emulator.feed("\x1b\\").events == (CommandSubmitted("ls -la"),)
    assert emulator.feed("").events == ()


def test_report_terminated_by_a_bell_in_a_later_read() -> None:
    emulator = TerminalEmulator(20, 5)

    assert emulator.feed(f"\x1b]2025;{b64('/tmp/bell-terminated')}").events == ()
    assert emulator.feed("\x07").events == (DirectoryChanged("/tmp/bell-terminated"),)
    emulator.feed("ready")

    assert emulator.buffer.screen_text[0] == "ready"


def test_reports_write_nothing_on_the_screen() -> None:
    emulator = fed(
        "\x1b]2025;/tmp\x07\x1b]2026;ls\x1b\\\x1b]7;file:///tmp\x07\x1b]8;;https://example.com\x07\x1b]8;;\x07"
    )

    assert emulator.buffer.screen_text == [""] * 5
    assert (emulator.buffer.cursor.x, emulator.buffer.cursor.y) == (0, 0)


@pytest.mark.parametrize("command", ["0", "1", "2"])
def test_window_title_is_swallowed(command: str) -> None:
    emulator = TerminalEmulator(20, 5)

    update = emulator.feed(f"\x1b]{command};title")
    assert emulator.feed("\x07visible").events == update.events == ()

    assert emulator.buffer.screen_text[0] == "visible"


@pytest.mark.parametrize(
    "payload", ["4;1;rgb:00/00/00", "10;?", "11;?", "52;c;aGVsbG8=", "133;A", "633;A", "1337;File=:", "9999", "", ";"]
)
def test_operating_system_command_this_terminal_does_not_have_is_ignored(payload: str) -> None:
    emulator = TerminalEmulator(20, 5)

    update = emulator.feed(f"\x1b]{payload}\x1b\\text")

    assert (update.replies, update.events) == ("", ())
    assert emulator.buffer.screen_text[0] == "text"
