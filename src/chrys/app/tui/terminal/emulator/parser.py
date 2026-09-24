# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Control-sequence parser: splits a terminal's output stream into printable runs and controls.

The grammar is ECMA-48's, driven as a resumable state machine so a sequence may be cut anywhere by
a read boundary, and the recovery rules are the ones DEC terminals established: a control
character inside a sequence acts immediately without ending it, ESC abandons the sequence in
progress and starts another, CAN and SUB cancel it. Whole runs of text and whole control sequences
are matched in one step when they arrive intact, which is nearly always; the character-at-a-time
states only finish what a boundary or an embedded control interrupted.

The parser holds no terminal state. It reports what it recognized to a `SequenceHandler`. A handler
that fails costs the rest of that `feed` and the sequence in progress, never the parse of what is
fed next.
"""

from __future__ import annotations

import re
from typing import Final, Protocol

ESC: Final = "\x1b"
BEL: Final = "\x07"
_CAN: Final = "\x18"
_SUB: Final = "\x1a"
_DEL: Final = "\x7f"
_ST: Final = "\x9c"

# C0, DEL and C1 never print. Decoded C1 code points are dropped rather than obeyed: in a UTF-8
# stream they are far likelier to be damage than an 8-bit control.
_PRINTABLE_RUN = re.compile(r"[^\x00-\x1f\x7f-\x9f]+")
_CONTROL_SEQUENCE = re.compile(r"([<=>?]?)([0-9:;]*)([ -/]*)([@-~])")
_STRING_RUN = re.compile(r"[^\x00-\x1f\x9c]+")

_MAX_PARAMETER_LENGTH: Final = 256
_MAX_STRING_LENGTH: Final = 1 << 18

_GROUND: Final = 0
_ESCAPE: Final = 1
_CSI_PARAMETERS: Final = 2
_CSI_INTERMEDIATES: Final = 3
_CSI_IGNORE: Final = 4
_OSC_STRING: Final = 5
_DCS_STRING: Final = 6
_IGNORED_STRING: Final = 7

# OSC and DCS carry data we act on; SOS, PM and APC are consumed so their text never prints.
_STRING_INTRODUCERS: Final = {
    "]": _OSC_STRING,
    "P": _DCS_STRING,
    "X": _IGNORED_STRING,
    "^": _IGNORED_STRING,
    "_": _IGNORED_STRING,
}


class SequenceHandler(Protocol):
    """What a parser reports to."""

    def print(self, text: str) -> None:
        """A run of printable characters."""

    def execute(self, control: str) -> None:
        """A C0 control character."""

    def escape(self, intermediates: str, final: str) -> None:
        """An escape sequence: ``ESC intermediates final``."""

    def control_sequence(self, marker: str, parameters: str, intermediates: str, final: str) -> None:
        """A control sequence: ``CSI marker parameters intermediates final``.

        ``marker`` is the private-use prefix (one of ``<=>?``) or empty; ``parameters`` is the raw
        parameter text, digits with ``;`` between parameters and ``:`` between sub-parameters.
        """

    def operating_system_command(self, payload: str) -> None:
        """The text of an OSC string."""

    def device_control(self, payload: str) -> None:
        """The text of a DCS string, introducer parameters included."""


class SequenceParser:
    """Feeds a `SequenceHandler` from a stream of decoded terminal output."""

    __slots__ = ("_collected", "_collected_length", "_handler", "_intermediates", "_marker", "_state")

    def __init__(self, handler: SequenceHandler) -> None:
        self._handler = handler
        self._state = _GROUND
        self._marker = ""
        self._collected: list[str] = []
        self._collected_length = 0
        self._intermediates = ""

    def reset(self) -> None:
        """Forget any sequence in progress."""
        self._state = _GROUND
        self._begin_collecting()

    def feed(self, data: str) -> None:
        """Parse the next piece of the stream."""
        try:
            self._feed(data)
        except BaseException:
            # Reports are made from inside a sequence as well: a string is still open while its
            # text is handed over, a control character acts in the middle of whatever it interrupts.
            # Left there, the parser would read what comes next as more of the same.
            self.reset()
            raise

    def _feed(self, data: str) -> None:
        handler = self._handler
        position = 0
        end = len(data)
        while position < end:
            state = self._state
            if state == _GROUND:
                if (run := _PRINTABLE_RUN.match(data, position)) is not None:
                    handler.print(run.group())
                    position = run.end()
                    continue
                char = data[position]
                position += 1
                if char == ESC:
                    self._state = _ESCAPE
                    self._begin_collecting()
                elif char < " ":
                    handler.execute(char)
                continue

            if state >= _OSC_STRING:
                position = self._feed_string(data, position)
                continue

            char = data[position]
            position += 1
            if char == ESC:
                self._state = _ESCAPE
                self._begin_collecting()
            elif char in (_CAN, _SUB):
                self._state = _GROUND
            elif char < " ":
                handler.execute(char)
            elif char >= _DEL:
                if char != _DEL:
                    # Not part of any sequence: the sequence is damaged, the character still prints.
                    self._state = _GROUND
                    position -= 1
            elif state == _ESCAPE:
                position = self._feed_escape(char, data, position)
            else:
                self._feed_control_sequence(char)

    def _begin_collecting(self) -> None:
        self._marker = ""
        self._collected = []
        self._collected_length = 0
        self._intermediates = ""

    def _feed_escape(self, char: str, data: str, position: int) -> int:
        if char < "0":
            self._intermediates += char
            return position
        self._state = _GROUND
        if self._intermediates:
            self._handler.escape(self._intermediates, char)
        elif char == "[":
            if (sequence := _CONTROL_SEQUENCE.match(data, position)) is not None:
                if len(sequence[2]) <= _MAX_PARAMETER_LENGTH:
                    self._handler.control_sequence(*sequence.groups())
                return sequence.end()
            self._state = _CSI_PARAMETERS
        elif (string_state := _STRING_INTRODUCERS.get(char)) is not None:
            self._state = string_state
        else:
            self._handler.escape("", char)
        return position

    def _feed_control_sequence(self, char: str) -> None:
        if char >= "@":
            if self._state != _CSI_IGNORE:
                self._handler.control_sequence(self._marker, "".join(self._collected), self._intermediates, char)
            self._state = _GROUND
        elif self._state == _CSI_IGNORE:
            pass
        elif char < "0":
            self._intermediates += char
            self._state = _CSI_INTERMEDIATES
        elif self._state == _CSI_INTERMEDIATES:
            # A parameter byte after an intermediate is malformed; swallow through the final byte.
            self._state = _CSI_IGNORE
        elif char < "<":
            self._collected.append(char)
            if len(self._collected) > _MAX_PARAMETER_LENGTH:
                self._state = _CSI_IGNORE
        elif self._collected or self._marker:
            self._state = _CSI_IGNORE
        else:
            self._marker = char

    def _feed_string(self, data: str, position: int) -> int:
        if (run := _STRING_RUN.match(data, position)) is not None:
            if self._state != _IGNORED_STRING and self._collected_length <= _MAX_STRING_LENGTH:
                self._collected.append(run.group())
                self._collected_length += run.end() - position
            return run.end()
        char = data[position]
        if char == ESC:
            # Normally the first half of ST (ESC \); whatever follows, the string is over.
            self._end_string()
            self._state = _ESCAPE
            self._begin_collecting()
        elif char == _ST or (char == BEL and self._state == _OSC_STRING):
            self._end_string()
            self._state = _GROUND
        elif char in (_CAN, _SUB):
            self._state = _GROUND
        return position + 1

    def _end_string(self) -> None:
        if self._collected_length > _MAX_STRING_LENGTH:
            return
        if self._state == _OSC_STRING:
            self._handler.operating_system_command("".join(self._collected))
        elif self._state == _DCS_STRING:
            self._handler.device_control("".join(self._collected))
