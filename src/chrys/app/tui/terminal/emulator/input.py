# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What a terminal sends to the program: keys, pointer reports, pastes and focus changes.

Keys follow xterm until the program asks for more. A function key is a fixed sequence with the
held modifiers in its second parameter, one more than their bits. A text key sends its text, Ctrl
folds it into a C0 control and Alt prefixes ESC, which is where the classic encoding runs out:
Shift+Enter is Enter, Ctrl+Shift+A is Ctrl+A. A program that wants those told apart asks for one of
the extended encodings (`KeyProtocol`), and only then are they sent, because a shell that never
asked would print the unfamiliar sequence as text.

The extended encodings are two families, and each is followed by its own definition. xterm's
modifyOtherKeys respells text keys only, and how many of them is a matter of the level asked for.
The kitty keyboard protocol has its own modifier bits, and respells the function keys as well:
none of them is an SS3 sequence any more, whatever the cursor key mode, and F3 stops looking like
a cursor position report.

Key names are Textual's: modifiers first, ``+`` between (``ctrl+shift+left``).
"""

from __future__ import annotations

import unicodedata
from enum import Enum
from typing import Final

from chrys.app.tui.terminal.emulator.core import KeyProtocol, MouseEncoding, MouseTracking

FOCUS_IN: Final = "\x1b[I"
FOCUS_OUT: Final = "\x1b[O"

_PASTE_START: Final = "\x1b[200~"
_PASTE_END: Final = "\x1b[201~"

_SHIFT, _ALT, _CTRL = 1, 2, 4
# xterm has a bit for Meta and none for Super or Hyper, which it cannot say at all. kitty has all
# three, and its Meta is not where xterm's is.
_XTERM_MODIFIER_BITS: Final = {"shift": _SHIFT, "alt": _ALT, "ctrl": _CTRL, "meta": 8, "super": 0, "hyper": 0}
_KITTY_MODIFIER_BITS: Final = {"shift": _SHIFT, "alt": _ALT, "ctrl": _CTRL, "super": 8, "hyper": 16, "meta": 32}

# Keys spelled ``CSI 1 ; modifiers <final>``. Bare, xterm has the cursor keys honor application mode
# (DECCKM) and F1-F4 always SS3; the kitty protocol keeps both where they are.
_CURSOR_KEYS: Final = {"up": "A", "down": "B", "right": "C", "left": "D", "home": "H", "end": "F"}
_FUNCTION_KEYS_SS3: Final = {"f1": "P", "f2": "Q", "f3": "R", "f4": "S"}
# Keys spelled ``CSI <number> ; modifiers ~``. The gaps in the numbering are DEC's.
_TILDE_KEYS: Final = {
    "insert": 2,
    "delete": 3,
    "pageup": 5,
    "pagedown": 6,
    "f5": 15,
    "f6": 17,
    "f7": 18,
    "f8": 19,
    "f9": 20,
    "f10": 21,
    "f11": 23,
    "f12": 24,
    "f13": 25,
    "f14": 26,
    "f15": 28,
    "f16": 29,
    "f17": 31,
    "f18": 32,
    "f19": 33,
    "f20": 34,
}
# Where kitty parts with xterm: F3 by number, since ``CSI 1 ; modifiers R`` is also how a terminal
# reports the cursor position, and the keys past F12 by numbers of kitty's own, up to F35.
_FIRST_KITTY_NUMBERED_FUNCTION_KEY: Final = 13
_KITTY_FUNCTION_KEYS: Final = {
    "f3": (13, "~"),
    **{f"f{number}": (57376 + number - _FIRST_KITTY_NUMBERED_FUNCTION_KEY, "u") for number in range(13, 36)},
}

# Keys that are a character but not a printable one, and the names Textual gives punctuation where
# the Unicode name would not lead back to the character.
_EDITING_KEYS: Final = {"enter": "\r", "tab": "\t", "escape": "\x1b", "backspace": "\x7f", "space": " "}
_PUNCTUATION_NAMES: Final = {
    "slash": "/",
    "backslash": "\\",
    "at": "@",
    "minus": "-",
    "plus": "+",
    "underscore": "_",
    "less_than_sign": "<",
    "greater_than_sign": ">",
}
# What a key types with Shift held, where that is not the capital of what it types without. Textual
# names such a key by the latter and nothing records the layout that turns one into the other, so
# it is the US one: what xterm's own table of these encodings is written for.
_US_SHIFTED: Final = dict(zip("`1234567890-=[]\\;',./", '~!@#$%^&*()_+{}|:"<>?', strict=True))
# What Ctrl makes of a character besides the letters: the C0 control sharing its low five bits,
# plus the digit row and the few aliases every terminal has inherited from the VT100 keyboard.
_CONTROL_CHARACTERS: Final = {
    "@": "\x00",
    " ": "\x00",
    "`": "\x00",
    "2": "\x00",
    "[": "\x1b",
    "{": "\x1b",
    "3": "\x1b",
    "\\": "\x1c",
    "|": "\x1c",
    "4": "\x1c",
    "]": "\x1d",
    "}": "\x1d",
    "5": "\x1d",
    "^": "\x1e",
    "~": "\x1e",
    "6": "\x1e",
    "_": "\x1f",
    "-": "\x1f",
    "/": "\x1f",
    "7": "\x1f",
    "8": "\x7f",
    "?": "\x7f",
    "\x7f": "\x08",
}
# The two of those xterm does not count as well known: it sends "-" as itself, and makes DEL of "?"
# only once it has decided how the key is spelled.
_LESSER_CONTROL_ALIASES: Final = frozenset("-?")


def encode_key(
    key: str,
    character: str | None,
    *,
    application_cursor_keys: bool = False,
    protocol: KeyProtocol = KeyProtocol.LEGACY,
) -> str | None:
    """The bytes for a key press, or ``None`` if the key sends nothing.

    Args:
        key: Textual's key name.
        character: The text the key produced, if any. It may be several code points: an input
            method commits a whole word as one key.
        application_cursor_keys: Whether the program switched the cursor keys to application mode.
        protocol: The encoding the program asked for.
    """
    kitty = protocol is KeyProtocol.KITTY
    bits = _KITTY_MODIFIER_BITS if kitty else _XTERM_MODIFIER_BITS
    *modifiers, base = key.split("+")
    if any(modifier not in bits for modifier in modifiers):
        # Not a modified key after all, just text that happens to contain a plus sign.
        return character
    held = sum(bits[modifier] for modifier in modifiers)

    if (final := _CURSOR_KEYS.get(base)) is not None:
        application = application_cursor_keys and not held and not kitty
        return f"\x1bO{final}" if application else _control_sequence(1, held, final)
    if kitty and (spelling := _KITTY_FUNCTION_KEYS.get(base)) is not None:
        return _control_sequence(spelling[0], held, spelling[1])
    if (final := _FUNCTION_KEYS_SS3.get(base)) is not None:
        return _control_sequence(1, held, final) if held or kitty else f"\x1bO{final}"
    if (number := _TILDE_KEYS.get(base)) is not None:
        return _control_sequence(number, held, "~")

    typed = _key_character(base)
    if typed is None:
        return character
    if kitty:
        if typed != typed.lower() and len(typed.lower()) == 1:
            # The key is named by what it types unshifted, and Shift is a modifier like any other.
            typed, held = typed.lower(), held | _SHIFT
        if typed == "\x1b" or held & ~_SHIFT or (held and base in _EDITING_KEYS):
            return _control_sequence(ord(typed), held, "u")
    if held & _SHIFT:
        # xterm goes the other way: by what was typed, Shift's part in it included. That is what Ctrl
        # folds, what is asked whether it has a classic spelling, and the code that is reported.
        typed = _shifted(typed)
    if _modify_other_keys_applies(protocol, base, typed, held):
        return f"\x1b[27;{held + 1};{ord(typed)}~"
    return _classic_encoding(base, typed, held, character)


def _control_sequence(number: int, held: int, final: str) -> str:
    """``CSI number ; modifiers final``, less the parameters that would only state a default."""
    if held:
        return f"\x1b[{number};{held + 1}{final}"
    return f"\x1b[{final}" if number == 1 else f"\x1b[{number}{final}"


def _key_character(base: str) -> str | None:
    """The character a key name stands for, or ``None`` for a key that is not a character."""
    if len(base) == 1:
        return base
    if (known := _EDITING_KEYS.get(base) or _PUNCTUATION_NAMES.get(base)) is not None:
        return known
    try:
        return unicodedata.lookup(base.replace("_", " "))
    except KeyError:
        return None


def _shifted(typed: str) -> str:
    """What the key that types this types with Shift held, as far as that can be told."""
    capital = typed.upper()
    if capital != typed:
        return capital if len(capital) == 1 else typed
    return _US_SHIFTED.get(typed, typed)


def _classic_encoding(base: str, typed: str, held: int, character: str | None) -> str | None:
    if base == "tab" and held & _SHIFT:
        typed = "\x1b[Z"
    elif held & _CTRL:
        is_letter = typed.isascii() and typed.isalpha()
        typed = chr(ord(typed) & 0x1F) if is_letter else _CONTROL_CHARACTERS.get(typed, typed)
    elif base not in _EDITING_KEYS:
        # Whatever the key name says, the text the key produced is what was typed.
        typed = character if character is not None else typed
    return f"\x1b{typed}" if held & _ALT else typed


def _modify_other_keys_applies(protocol: KeyProtocol, base: str, typed: str, held: int) -> bool:
    """Whether xterm spells a text key ``CSI 27 ; modifiers ; code ~`` at the level the program asked for.

    xterm as it is when Alt sends ESC, the only Alt the classic encoding here knows, going as it
    does by ``typed`` with Shift's part already in it. One departure: at level 2 xterm also
    respells Shift with a letter, and not Shift with a digit. Shift with something printable is
    sent as what it types, everywhere.
    """
    if not held or protocol not in (KeyProtocol.MODIFY_OTHER_KEYS_1, KeyProtocol.MODIFY_OTHER_KEYS_2):
        return False
    if base == "backspace" and held == _CTRL:
        # Not a chord to xterm but the other backspace: Ctrl swaps which of BS and DEL the key sends.
        return False
    if held & ~(_SHIFT | _ALT | _CTRL):
        # Meta. There is no classic spelling to keep.
        return True
    if protocol is KeyProtocol.MODIFY_OTHER_KEYS_2:
        return held != _SHIFT or base in _EDITING_KEYS
    # Level 1 leaves alone whatever has a well-known spelling, even one that drops a modifier.
    if base in ("tab", "enter"):
        # Back-tab is the one such spelling these two have; Alt has none, as ESC is not put before them.
        return (base, held) != ("tab", _SHIFT)
    if base == "escape":
        return bool(held & _ALT)
    if base in _EDITING_KEYS:
        return False
    is_letter = typed.isascii() and typed.isalpha()
    has_control = is_letter or (typed in _CONTROL_CHARACTERS and typed not in _LESSER_CONTROL_ALIASES)
    return bool(held & _CTRL) and not has_control


def encode_paste(text: str, *, bracketed: bool) -> str:
    """The bytes for pasted text.

    Bracketed, the program is told where the paste starts and ends, so the paste itself must not be
    able to say so first. Unbracketed, the text arrives as if typed, where Return ends a line.
    """
    if bracketed:
        return f"{_PASTE_START}{text.replace(_PASTE_END, '')}{_PASTE_END}"
    return text.replace("\r\n", "\r").replace("\n", "\r")


class PointerAction(Enum):
    PRESS = "press"
    RELEASE = "release"
    MOVE = "move"
    WHEEL_UP = "wheel_up"
    WHEEL_DOWN = "wheel_down"
    WHEEL_LEFT = "wheel_left"
    WHEEL_RIGHT = "wheel_right"


_NO_BUTTON: Final = 3
_MOTION: Final = 32
_WHEEL: Final = {
    PointerAction.WHEEL_UP: 64,
    PointerAction.WHEEL_DOWN: 65,
    PointerAction.WHEEL_LEFT: 66,
    PointerAction.WHEEL_RIGHT: 67,
}
_MAX_DEFAULT_COORDINATE: Final = 223
_MAX_UTF8_COORDINATE: Final = 2015
_FIRST_BYTE_THAT_IS_NOT_TEXT: Final = 0x80
_SURROGATE_ESCAPE: Final = 0xDC00


def encode_pointer(
    tracking: MouseTracking,
    encoding: MouseEncoding,
    action: PointerAction,
    *,
    x: int,
    y: int,
    button: int | None = None,
    shift: bool = False,
    alt: bool = False,
    ctrl: bool = False,
) -> str | None:
    """The report for a pointer event, or ``None`` when the program did not ask to hear of it.

    xterm's original format spends one byte on each coordinate, and from column 96 on that byte is
    not text. It is returned as the lone surrogate the ``surrogateescape`` error handler turns back
    into the byte, which is how whoever encodes the report for the program has to encode it. Only
    the UTF-8 format (mode 1005) sends such a value as a character.

    Args:
        tracking: The tracking mode the program selected.
        encoding: The report format the program selected.
        action: What the pointer did.
        x: Zero-based column.
        y: Zero-based row.
        button: 0 left, 1 middle, 2 right; ``None`` when no button is involved or held.
        shift: Shift is held.
        alt: Alt (Meta) is held.
        ctrl: Control is held.
    """
    if tracking is MouseTracking.OFF:
        return None
    if tracking is MouseTracking.PRESS and action is not PointerAction.PRESS:
        return None
    if action is PointerAction.MOVE and (
        tracking is MouseTracking.PRESS_RELEASE or (tracking is MouseTracking.DRAG and button is None)
    ):
        return None

    released = action is PointerAction.RELEASE
    if (code := _WHEEL.get(action)) is None:
        # Only the SGR format can say which button was released; the others report "no button".
        code = _NO_BUTTON if button is None or (released and encoding is not MouseEncoding.SGR) else button
        if action is PointerAction.MOVE:
            code += _MOTION
    if tracking is not MouseTracking.PRESS:
        code += 4 * shift + 8 * alt + 16 * ctrl

    column, row = x + 1, y + 1
    if encoding is MouseEncoding.SGR:
        return f"\x1b[<{code};{column};{row}{'m' if released else 'M'}"
    if encoding is MouseEncoding.URXVT:
        return f"\x1b[{code + 32};{column};{row}M"
    limit = _MAX_UTF8_COORDINATE if encoding is MouseEncoding.UTF8 else _MAX_DEFAULT_COORDINATE
    if column > limit or row > limit:
        # These formats spend one character, or one byte, per coordinate and cannot count any higher.
        return None
    values = (code + 32, column + 32, row + 32)
    if encoding is not MouseEncoding.UTF8:
        values = (value + _SURROGATE_ESCAPE * (value >= _FIRST_BYTE_THAT_IS_NOT_TEXT) for value in values)
    return "\x1b[M" + "".join(map(chr, values))
