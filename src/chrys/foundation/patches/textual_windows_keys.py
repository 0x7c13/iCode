# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Translate win32-input-mode before Textual parses the terminal input stream.

With ENABLE_VIRTUAL_TERMINAL_INPUT, conhost's legacy encoder loses Shift on
Enter before ReadConsoleInputW sees it. Mode 9001 preserves the KEY_EVENT_RECORD
fields in ``CSI Vk;Sc;Uc;Kd;Cs;Rc _``. Keep Textual's VT console mode and mouse,
paste and resize handling; only translate the lossless keyboard transport.

Synthetic records (Vk 0 / VK_PACKET) carry literal terminal bytes, including
bracketed-paste delimiters and terminal replies. Unwrap them before parsing, so
they can form whole escape sequences rather than become individual key events.
UTF-16 surrogate pairs may span records or input reads.
"""

from __future__ import annotations

import codecs
import sys
from functools import lru_cache
from time import monotonic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from textual._xterm_parser import XTermParser
    from textual.message import Message

_NAVIGATION = {
    0x21: (5, "~"),  # Page Up
    0x22: (6, "~"),
    0x23: (1, "F"),  # End
    0x24: (1, "H"),
    0x25: (1, "D"),  # Left
    0x26: (1, "A"),
    0x27: (1, "C"),
    0x28: (1, "B"),
    0x2D: (2, "~"),  # Insert
    0x2E: (3, "~"),
}
_FUNCTION_KEYS = (11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24)
_CONTROL_KEYS = {0x08: 127, 0x09: 9, 0x0D: 13, 0x1B: 27}
_MODIFIER_KEYS = {0x10, 0x11, 0x12, 0x14, 0x5B, 0x5C, 0x90, 0x91, *range(0xA0, 0xA6)}
_MAX_RECORD_LENGTH = 128


@lru_cache(maxsize=1)
def _load_virtual_key_mapper() -> Callable[[int, int], int] | None:
    """Load the native layout lookup only when Windows needs a missing character."""
    if sys.platform != "win32":
        return None
    import ctypes

    try:
        mapper = ctypes.WinDLL("user32").MapVirtualKeyW
        mapper.argtypes = [ctypes.c_uint, ctypes.c_uint]
        mapper.restype = ctypes.c_uint
    except OSError, AttributeError:
        return None
    return mapper


def _virtual_key_character(virtual_key: int) -> int:
    """Approximate punctuation using the input thread's keyboard layout.

    MapVirtualKeyW does not necessarily follow layout changes in the terminal
    host. Consult it per key, without caching character results.
    """
    mapper = _load_virtual_key_mapper()
    # MAPVK_VK_TO_CHAR returns the unshifted character in the low word; the
    # high bit flags dead keys. AltGr composition is excluded by the caller.
    return mapper(virtual_key, 2) & 0xFFFF if mapper is not None else 0


def _key_sequence(virtual_key: int, codepoint: int, down: int, state: int) -> str:
    """Convert one native key to the protocols the existing parser understands."""
    if not down:
        # Alt+numpad composition commits its character on Alt release.
        return chr(codepoint) if virtual_key == 0x12 and codepoint else ""
    if virtual_key in (0, 0xE7):
        return chr(codepoint) if codepoint else ""
    if virtual_key in _MODIFIER_KEYS:
        return ""
    shift, alt, ctrl = bool(state & 0x10), bool(state & 0x03), bool(state & 0x0C)
    if alt and not ctrl and 0x60 <= virtual_key <= 0x69:
        return ""
    modifier = 1 + shift + 2 * alt + 4 * ctrl
    if virtual_key in _CONTROL_KEYS:
        control = _CONTROL_KEYS[virtual_key]
        if modifier == 1 and virtual_key != 0x1B:
            # Console paste can carry Return/Tab as native key records even
            # between bracketed-paste markers. Keep them as literal text.
            return chr(control)
        return f"\x1b[27;{modifier};{control}~"
    if virtual_key in _NAVIGATION:
        number, final = _NAVIGATION[virtual_key]
        return f"\x1b[{number};{modifier}{final}"
    if 0x70 <= virtual_key <= 0x87:
        number, final = (
            (_FUNCTION_KEYS[virtual_key - 0x70], "~") if virtual_key < 0x7C else (57376 + virtual_key - 0x7C, "u")
        )
        return f"\x1b[{number};{modifier}{final}"
    character = chr(codepoint)
    if shift and codepoint == 32 and not (alt or ctrl):
        return f"\x1b[27;{modifier};32~"
    # Windows layouts use Ctrl+Alt for AltGr text. Preserve the supplied text,
    # including dead-key/IME commits, instead of turning it into a shortcut.
    if codepoint >= 32 and codepoint != 127 and (not (alt or ctrl) or (alt and ctrl)):
        return character
    # Right Alt + Left Ctrl is Windows' AltGr signature. A zero character is
    # uncommitted composition, including dead keys on digits and letters.
    if not codepoint and state & 0x09 == 0x09:
        return ""
    if alt or ctrl:
        if 0x30 <= virtual_key <= 0x39 or 0x41 <= virtual_key <= 0x5A or virtual_key == 0x20:
            codepoint = virtual_key
        elif ctrl and 0 < codepoint < 32:
            codepoint += 64
        elif not codepoint:
            # Ctrl+/ and other punctuation shortcuts can have no character.
            codepoint = _virtual_key_character(virtual_key)
        if codepoint:
            return f"\x1b[27;{modifier};{codepoint}~"
    return character if codepoint else ""


def _decode_record(parameters: str) -> str:
    parts = parameters.split(";")
    if len(parts) > 6:
        return ""
    values = [0, 0, 0, 0, 0, 1]
    for index, part in enumerate(parts):
        if part:
            values[index] = int(part)
    virtual_key, scan, codepoint, down, state, repeat = values
    if (
        any(value > 0xFFFF for value in (virtual_key, scan, codepoint, repeat))
        or state > 0xFFFFFFFF
        or down not in (0, 1)
    ):
        return ""
    return _key_sequence(virtual_key, codepoint, down, state) * repeat


class _InputDecoder:
    """Buffer only a possible win32 record; pass other terminal input through."""

    def __init__(self) -> None:
        self.pending = ""
        self.updated_at = 0.0
        self.unicode = codecs.getincrementaldecoder("utf-16-le")("replace")

    def feed(self, data: str) -> str:
        output: list[str] = []
        for char in data:
            if char == "\x1b":
                output.append(self.pending)
                self.pending = char
            elif (self.pending == "\x1b" and char == "[") or (
                self.pending.startswith("\x1b[") and char in "0123456789;" and len(self.pending) < _MAX_RECORD_LENGTH
            ):
                self.pending += char
            elif self.pending.startswith("\x1b[") and char == "_":
                output.append(_decode_record(self.pending[2:]))
                self.pending = ""
            else:
                output.extend((self.pending, char))
                self.pending = ""
        self.updated_at = monotonic()
        return self._unicode("".join(output))

    def flush(self, *, final: bool = False) -> str:
        pending, self.pending = self.pending, ""
        return self._unicode(pending, final=final)

    def _unicode(self, text: str, *, final: bool = False) -> str:
        return self.unicode.decode(text.encode("utf-16-le", "surrogatepass"), final=final)


@lru_cache(maxsize=1)
def get_parser_class() -> type[XTermParser]:
    """Create the lazy Textual adapter used by Windows EventMonitor."""
    from textual import constants
    from textual._xterm_parser import XTermParser

    class WindowsInputParser(XTermParser):
        def __init__(self, debug: bool = False) -> None:
            self._windows_input = _InputDecoder()
            super().__init__(debug=debug)

        def feed(self, data: str) -> Iterable[Message]:
            if data:
                if decoded := self._windows_input.feed(data):
                    yield from super().feed(decoded)
            else:
                if decoded := self._windows_input.flush(final=True):
                    yield from super().feed(decoded)
                yield from super().feed("")

        def tick(self) -> Iterable[Message]:
            if self._windows_input.pending:
                if monotonic() - self._windows_input.updated_at < constants.ESCAPE_DELAY:
                    # An inner VT sequence may be waiting for the character
                    # carried by this incomplete outer keyboard record.
                    return
                if decoded := self._windows_input.flush():
                    yield from super().feed(decoded)
                if self._timeout_time is not None:
                    # The escape delay was already spent in the transport
                    # buffer. Do not make a legacy Escape wait a second time.
                    self._timeout_time = 0.0
            yield from super().tick()

    return WindowsInputParser
