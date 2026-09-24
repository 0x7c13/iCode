# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The control-sequence parser: what it recognizes, how it recovers, and that read boundaries do not matter."""

from __future__ import annotations

import pytest

from chrys.app.tui.terminal.emulator.parser import SequenceParser

type Call = tuple[str, ...]


class Recorder:
    """A `SequenceHandler` that writes down what it is told."""

    def __init__(self) -> None:
        self.calls: list[Call] = []

    def print(self, text: str) -> None:
        self.calls.append(("print", text))

    def execute(self, control: str) -> None:
        self.calls.append(("execute", control))

    def escape(self, intermediates: str, final: str) -> None:
        self.calls.append(("escape", intermediates, final))

    def control_sequence(self, marker: str, parameters: str, intermediates: str, final: str) -> None:
        self.calls.append(("csi", marker, parameters, intermediates, final))

    def operating_system_command(self, payload: str) -> None:
        self.calls.append(("osc", payload))

    def device_control(self, payload: str) -> None:
        self.calls.append(("dcs", payload))


def parse(*chunks: str) -> list[Call]:
    """What the parser reports for a stream read in these pieces, runs of text joined back together."""
    recorder = Recorder()
    parser = SequenceParser(recorder)
    for chunk in chunks:
        parser.feed(chunk)
    merged: list[Call] = []
    for call in recorder.calls:
        if call[0] == "print" and merged and merged[-1][0] == "print":
            merged[-1] = ("print", merged[-1][1] + call[1])
        else:
            merged.append(call)
    return merged


# -- what it recognizes ------------------------------------------------------------------------------


def test_text_is_reported_in_runs() -> None:
    recorder = Recorder()
    SequenceParser(recorder).feed("hello, wörld 你好")

    assert recorder.calls == [("print", "hello, wörld 你好")]


@pytest.mark.parametrize("control", [chr(code) for code in range(0x20) if chr(code) != "\x1b"])
def test_c0_controls_are_executed(control: str) -> None:
    assert parse(f"a{control}b") == [("print", "a"), ("execute", control), ("print", "b")]


def test_delete_and_c1_code_points_neither_print_nor_execute() -> None:
    assert parse("a\x7fb\x80c\x9bd\x9fe") == [("print", "abcde")]


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("\x1b7", ("escape", "", "7")),
        ("\x1bM", ("escape", "", "M")),
        ("\x1b=", ("escape", "", "=")),
        ("\x1b#8", ("escape", "#", "8")),
        ("\x1b(0", ("escape", "(", "0")),
        ("\x1b(%5", ("escape", "(%", "5")),
    ],
)
def test_escape_sequences(sequence: str, expected: Call) -> None:
    assert parse(sequence) == [expected]


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("\x1b[H", ("csi", "", "", "", "H")),
        ("\x1b[12;40H", ("csi", "", "12;40", "", "H")),
        ("\x1b[;5H", ("csi", "", ";5", "", "H")),
        ("\x1b[?25h", ("csi", "?", "25", "", "h")),
        ("\x1b[>4;2m", ("csi", ">", "4;2", "", "m")),
        ("\x1b[<1u", ("csi", "<", "1", "", "u")),
        ("\x1b[=1;2u", ("csi", "=", "1;2", "", "u")),
        ("\x1b[2 q", ("csi", "", "2", " ", "q")),
        ("\x1b[!p", ("csi", "", "", "!", "p")),
        ("\x1b[?2004$p", ("csi", "?", "2004", "$", "p")),
        ("\x1b[38:2::10:20:30m", ("csi", "", "38:2::10:20:30", "", "m")),
        ("\x1b[1~", ("csi", "", "1", "", "~")),
    ],
)
def test_control_sequences(sequence: str, expected: Call) -> None:
    assert parse(sequence) == [expected]
    # The same one read a character at a time, which takes the other road through the parser.
    assert parse(*sequence) == [expected]


def test_operating_system_command_ends_at_bel() -> None:
    assert parse("\x1b]0;a title\x07after") == [("osc", "0;a title"), ("print", "after")]


def test_operating_system_command_ends_at_string_terminator() -> None:
    # ST is itself an escape sequence, and is reported as the one it is.
    assert parse("\x1b]8;;https://example.com\x1b\\after") == [
        ("osc", "8;;https://example.com"),
        ("escape", "", "\\"),
        ("print", "after"),
    ]


def test_operating_system_command_ends_at_c1_string_terminator() -> None:
    assert parse("\x1b]2;title\x9cafter") == [("osc", "2;title"), ("print", "after")]


def test_any_escape_sequence_ends_a_string() -> None:
    assert parse("\x1b]0;title\x1b[2Jafter") == [("osc", "0;title"), ("csi", "", "2", "", "J"), ("print", "after")]


def test_operating_system_command_keeps_semicolons_and_unicode() -> None:
    assert parse("\x1b]2025;/tmp/a;b 目录\x07") == [("osc", "2025;/tmp/a;b 目录")]


def test_device_control_string() -> None:
    assert parse("\x1bP$q q\x1b\\") == [("dcs", "$q q"), ("escape", "", "\\")]


def test_bel_does_not_end_a_device_control_string() -> None:
    assert parse("\x1bP1$r\x07more\x1b\\") == [("dcs", "1$rmore"), ("escape", "", "\\")]


@pytest.mark.parametrize("introducer", ["X", "^", "_"])
def test_sos_pm_and_apc_strings_are_consumed(introducer: str) -> None:
    assert parse(f"before\x1b{introducer}Gf=100;secret\x07still inside\x1b\\after") == [
        ("print", "before"),
        ("escape", "", "\\"),
        ("print", "after"),
    ]


def test_controls_inside_a_string_are_dropped() -> None:
    assert parse("\x1b]0;ti\r\n\ttle\x07") == [("osc", "0;title")]


# -- recovery ----------------------------------------------------------------------------------------


def test_control_character_inside_a_control_sequence_acts_without_ending_it() -> None:
    assert parse("\x1b[1\n;2H") == [("execute", "\n"), ("csi", "", "1;2", "", "H")]


def test_control_character_inside_an_escape_sequence_acts_without_ending_it() -> None:
    assert parse("\x1b(\r0") == [("execute", "\r"), ("escape", "(", "0")]


def test_escape_abandons_the_sequence_in_progress() -> None:
    assert parse("\x1b[1;2\x1b[3;4H") == [("csi", "", "3;4", "", "H")]
    assert parse("\x1b(\x1b7") == [("escape", "", "7")]


@pytest.mark.parametrize("cancel", ["\x18", "\x1a"])
def test_can_and_sub_cancel_a_sequence(cancel: str) -> None:
    assert parse(f"\x1b[1;2{cancel}H") == [("print", "H")]
    assert parse(f"\x1b{cancel}7") == [("print", "7")]
    assert parse(f"\x1b]0;ti{cancel}tle") == [("print", "tle")]
    assert parse(f"\x1bP$q{cancel}r") == [("print", "r")]


def test_delete_inside_a_sequence_is_ignored() -> None:
    assert parse("\x1b[1\x7f;2H") == [("csi", "", "1;2", "", "H")]
    assert parse("\x1b\x7f7") == [("escape", "", "7")]


def test_text_inside_a_sequence_ends_it_and_still_prints() -> None:
    assert parse("\x1b[1;2你好") == [("print", "你好")]
    assert parse("\x1b(é") == [("print", "é")]


def test_c1_code_point_inside_a_sequence_ends_it_and_is_dropped() -> None:
    assert parse("\x1b[1;2\x9bH") == [("print", "H")]


def test_parameter_after_an_intermediate_swallows_the_sequence() -> None:
    assert parse("\x1b[1 2qafter") == [("print", "after")]


def test_second_marker_swallows_the_sequence() -> None:
    assert parse("\x1b[?1?hafter") == [("print", "after")]
    assert parse("\x1b[1?hafter") == [("print", "after")]


def test_swallowed_sequence_still_executes_controls() -> None:
    assert parse("\x1b[1 2\rq") == [("execute", "\r")]


@pytest.mark.parametrize("chunked", [False, True])
def test_parameter_length_is_capped(chunked: bool) -> None:
    fits = "1;" * 128
    too_long = fits + "1"
    assert len(fits) == 256

    def read(sequence: str) -> list[Call]:
        return parse(*sequence) if chunked else parse(sequence)

    assert read(f"\x1b[{fits}mafter") == [("csi", "", fits, "", "m"), ("print", "after")]
    assert read(f"\x1b[{too_long}mafter") == [("print", "after")]


def test_string_length_is_capped() -> None:
    fits = "x" * (1 << 18)

    assert parse(f"\x1b]{fits}\x07after") == [("osc", fits), ("print", "after")]
    assert parse(f"\x1b]{fits}y\x07after") == [("print", "after")]
    # Read in pieces the string stops growing once over the cap, and is dropped all the same.
    assert parse("\x1b]", fits, "y" * 10, "z" * 10, "\x07after") == [("print", "after")]


def test_reset_forgets_the_sequence_in_progress() -> None:
    recorder = Recorder()
    parser = SequenceParser(recorder)

    parser.feed("\x1b[1;2")
    parser.reset()
    parser.feed("H")
    parser.feed("\x1b]0;tit")
    parser.reset()
    parser.feed("le")

    assert recorder.calls == [("print", "H"), ("print", "le")]


class Choking(Recorder):
    """A recorder that fails on the first thing it is told, and only on that."""

    def __init__(self) -> None:
        super().__init__()
        self.choked = False

    def _choke(self) -> None:
        if not self.choked:
            self.choked = True
            raise ValueError("cannot make sense of this")

    def execute(self, control: str) -> None:
        self._choke()
        super().execute(control)

    def control_sequence(self, marker: str, parameters: str, intermediates: str, final: str) -> None:
        self._choke()
        super().control_sequence(marker, parameters, intermediates, final)

    def operating_system_command(self, payload: str) -> None:
        self._choke()
        super().operating_system_command(payload)

    def device_control(self, payload: str) -> None:
        self._choke()
        super().device_control(payload)


@pytest.mark.parametrize(
    "chunks",
    [
        ("\x1b]7;bad\x07lost",),
        ("\x1b]7;bad\x1b\\lost",),
        ("\x1b]7;b", "ad\x07lost"),
        ("\x1bP$qbad\x1b\\lost",),
        ("\x1b[1;2Hlost",),
        # Cut by a read, a control sequence is finished a character at a time.
        ("\x1b[1;", "2Hlost"),
        # A control character acts from inside the sequence it interrupts, which then goes on.
        ("\x1b[1\n;2Hlost",),
        ("\x1b(\nBlost",),
    ],
)
def test_a_handler_that_fails_costs_nothing_that_is_fed_afterwards(chunks: tuple[str, ...]) -> None:
    recorder = Choking()
    parser = SequenceParser(recorder)

    with pytest.raises(ValueError, match="cannot make sense"):
        for chunk in chunks:
            parser.feed(chunk)
    # Read from the ground up: no text taken into the string that failed, no parameter or
    # intermediate left over for the next sequence.
    parser.feed("h\x1b]0;title\x07\x1b[3m")

    assert recorder.calls == [("print", "h"), ("osc", "0;title"), ("csi", "", "3", "", "m")]


# -- read boundaries ---------------------------------------------------------------------------------

_STREAM = (
    "plain 你好 text\r\n"
    "\x1b[1;31mred\x1b[0m\x1b[m"
    "\x1b[?1049h\x1b[2J\x1b[12;40H\x1b[2 q\x1b[>4;2m\x1b[?2004$p\x1b[38:2::1:2:3m"
    "\x1b7\x1b8\x1b#8\x1b(0lqk\x1b(B\x1b(%5"
    "\x1b]0;title\x07\x1b]8;;https://example.com/a;b\x1b\\link\x1b]8;;\x1b\\"
    "\x1b]2025;b64:L3RtcC9hO2I=\x9c"
    "\x1bP$q q\x1b\\\x1bXignored\x1b\\\x1b^ignored\x07too\x1b\\\x1b_ignored\x1b\\"
    "\x1b[1\n;2H\x1b[1;2\x1b[3;4H\x1b[1;2\x18H\x1b[1\x7f;2H\x1b[1 2q\x1b[?1?h\x1b[1;2é"
    "\x1b]0;cancel\x1aled\x1b]0;interrupted\x1b[Jtail\x07\x7f\x9b"
)


def test_stream_covers_every_kind_of_report() -> None:
    assert {call[0] for call in parse(_STREAM)} == {"print", "execute", "escape", "csi", "osc", "dcs"}


@pytest.mark.parametrize("split", range(1, len(_STREAM)))
def test_a_read_boundary_anywhere_changes_nothing(split: int) -> None:
    assert parse(_STREAM[:split], _STREAM[split:]) == parse(_STREAM)


def test_reading_a_character_at_a_time_changes_nothing() -> None:
    assert parse(*_STREAM) == parse(_STREAM)


@pytest.mark.parametrize("size", [2, 3, 5, 7])
def test_reading_in_small_pieces_changes_nothing(size: int) -> None:
    pieces = [_STREAM[start : start + size] for start in range(0, len(_STREAM), size)]

    assert parse(*pieces) == parse(_STREAM)
