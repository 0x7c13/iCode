# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Support xterm's modified text keys alongside Textual's Kitty protocol.

Textual 8.2.7 mistakes ``CSI 27;modifier;codepoint ~`` for a Kitty sequence,
using 27 as the key and the actual key as associated text. Decode that wire
format before Textual's normal parser, retaining printable shifted text and
keeping control/alt/meta shortcuts out of the text insertion path.

POSIX drivers request modifyOtherKeys level 2 before their Kitty push. Terminals
that understand Kitty can use that protocol; older xterm-compatible terminals
can report modified Enter through modifyOtherKeys. Reset the xterm mode before
the Kitty pop (also on suspend), with a close fallback for partial startup.
The request is independent of TERM_PROGRAM, which may be missing over SSH.

Windows drivers request win32-input-mode instead. Its adapter unwraps native
keyboard records before the existing VT parser, preserving Enter's modifiers.

This is a runtime-only patch: startup applies it to already-imported Textual
classes, without rewriting any upstream source or replacing a driver class.
"""

from __future__ import annotations

import logging
import re
from functools import wraps
from typing import TYPE_CHECKING

from chrys.foundation.platform import get_platform

if TYPE_CHECKING:
    from collections.abc import Iterable

    from textual._xterm_parser import XTermParser
    from textual.driver import Driver
    from textual.events import Key

_TEXTUAL_VERSION = "8.2.7"
_PATCH_MARKER = "_chrys_extended_keys"
_ACTIVE_ATTRIBUTE = "_chrys_extended_keys_active"
_XTERM_KEY = re.compile(r"\x1b\[27;([0-9]+);([0-9]+)~")
_KITTY_PUSH = re.compile(r"\x1b\[>[0-9]+u")
_ENABLE = "\x1b[>4;2m"
_RESET = "\x1b[>4;0m"
_KITTY_POP = "\x1b[<u"
logger = logging.getLogger(__name__)


def _decode_key(modifier: int, codepoint: int) -> Key:
    """Decode xterm's four modifier bits, preserving text only for typing."""
    from textual.events import Key
    from textual.keys import Keys, _character_to_key

    if not 1 <= modifier <= 16 or not 0 <= codepoint <= 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
        return Key(Keys.Ignore, None)
    character = chr(codepoint)
    bits = modifier - 1
    if bits == 1 and codepoint in (8, 27, 127):
        # Retain the legacy correction/cancel keys while Shift is held.
        bits = 0
    if bits == 4 and character == "[":
        return Key("escape", None)
    text = character if character.isprintable() and not bits & 0b1110 else None
    # xterm reports the shifted character; shortcut bindings use lower-case
    # letters plus explicit modifiers (for example ctrl+shift+v).
    key_character = character
    if bits & 0b1110 and character.isascii() and character.isalpha():
        key_character = character.lower()
    key = {8: "backspace", 9: "tab", 13: "enter", 27: "escape", 127: "backspace"}.get(codepoint)
    if key is None:
        key = _character_to_key(key_character)
    names = [name for bit, name in ((2, "alt"), (4, "ctrl"), (8, "meta")) if bits & bit]
    # A shifted symbol already encodes Shift in its character (Ctrl+_).
    # Letters, digits, whitespace and control keys keep explicit modifiers.
    explicit_shift = character.isalnum() or character.isspace() or not character.isprintable()
    if bits & 1 and (text is None or character.isspace()) and explicit_shift:
        names.append("shift")
    return Key("+".join([*sorted(names), key]), text)


def _patch_parser() -> None:
    from textual._xterm_parser import XTermParser

    original = XTermParser._sequence_to_key_events
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def sequence_to_key_events(self: XTermParser, sequence: str, alt: bool = False) -> Iterable[Key]:
        if match := _XTERM_KEY.fullmatch(sequence):
            yield _decode_key(*map(int, match.groups()))
            return
        yield from original(self, sequence, alt=alt)

    setattr(sequence_to_key_events, _PATCH_MARKER, True)
    XTermParser._sequence_to_key_events = sequence_to_key_events


def _patch_driver(driver_class: type[Driver], *, windows: bool = False) -> None:
    """Pair mode changes with Textual's own start/stop/suspend writes."""
    from textual import constants

    original_write = driver_class.write
    original_close = driver_class.close
    if getattr(original_write, _PATCH_MARKER, False):
        return
    enable, reset = ("\x1b[?9001h", "\x1b[?9001l") if windows else (_ENABLE, _RESET)

    @wraps(original_write)
    def write(self: Driver, data: str) -> None:
        active = getattr(self, _ACTIVE_ATTRIBUTE, False)
        resetting = active and data == _KITTY_POP
        if not active and not constants.DISABLE_KITTY_KEY and _KITTY_PUSH.fullmatch(data):
            # Record ownership before writing, so close can reset after a
            # partial write as well as after a later startup failure.
            setattr(self, _ACTIVE_ATTRIBUTE, True)
            data = enable + data
        elif resetting:
            data = reset + data
        original_write(self, data)
        if resetting:
            setattr(self, _ACTIVE_ATTRIBUTE, False)

    @wraps(original_close)
    def close(self: Driver) -> None:
        try:
            if getattr(self, _ACTIVE_ATTRIBUTE, False):
                original_write(self, reset)
                self.flush()
                setattr(self, _ACTIVE_ATTRIBUTE, False)
        finally:
            original_close(self)

    setattr(write, _PATCH_MARKER, True)
    driver_class.write = write
    driver_class.close = close


def apply_runtime_patch() -> None:
    """Install extended keyboard decoding and platform-specific negotiation."""
    try:
        import textual
    except ImportError:
        return
    if textual.__version__ != _TEXTUAL_VERSION:
        logger.warning("Skipping extended keyboard patch for unsupported Textual %s", textual.__version__)
        return
    _patch_parser()
    if get_platform().is_windows:
        from textual.drivers import win32
        from textual.drivers.windows_driver import WindowsDriver

        from chrys.foundation.patches.textual_windows_keys import get_parser_class

        win32.XTermParser = get_parser_class()
        _patch_driver(WindowsDriver, windows=True)
        return
    from textual.drivers.linux_driver import LinuxDriver
    from textual.drivers.linux_inline_driver import LinuxInlineDriver

    _patch_driver(LinuxDriver)
    _patch_driver(LinuxInlineDriver)
