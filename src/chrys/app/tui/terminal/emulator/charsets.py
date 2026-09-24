# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Character sets: the four designation slots and the shifts that pick which one prints.

Input is already Unicode, so only the 7-bit (GL) half of the ISO 2022 machinery means anything:
a designated set remaps ASCII code points, which is how line drawing reaches programs that still
speak ``ESC ( 0``. Right-half (GR) invocations are accepted and ignored.
"""

from __future__ import annotations

_SPECIAL_GRAPHICS = str.maketrans(
    "_`abcdefghijklmnopqrstuvwxyz{|}~",
    " ◆▒␉␌␍␊°±␤␋┘┐┌└┼⎺⎻─⎼⎽├┤┴┬│≤≥π≠£·",
)


def _national(replaced: str, replacements: str) -> dict[int, int]:
    return str.maketrans(replaced, replacements)


# National replacement sets differ from ASCII only in the twelve code points ISO 646 leaves open.
_BRITISH = _national("#", "£")
_DUTCH = _national("#@[\\]{|}~", "£¾ĳ½|¨ƒ¼´")  # noqa: RUF001 - the acute accent is the mapping
_FINNISH = _national("[\\]^`{|}~", "ÄÖÅÜéäöåü")
_FRENCH = _national("#@[\\]{|}~", "£à°ç§éùè¨")
_FRENCH_CANADIAN = _national("@[\\]^`{|}~", "àâçêîôéùèû")
_GERMAN = _national("@[\\]{|}~", "§ÄÖÜäöüß")
_ITALIAN = _national("#@[\\]`{|}~", "£§°çéùàòèì")
_NORWEGIAN_DANISH = _national("@[\\]^`{|}~", "ÄÆØÅÜäæøåü")
_SPANISH = _national("#@[\\]{|}", "£§¡Ñ¿°ñç")
_SWEDISH = _national("@[\\]^`{|}~", "ÉÄÖÅÜéäöåü")
_SWISS = _national("#@[\\]^_`{|}~", "ùàéçêîèôäöüû")

# DEC Supplemental Graphic is Latin-1's upper half moved down to 7 bits, except where DEC's
# multinational set predates Latin-1 and disagrees with it, or leaves the position unassigned.
_SUPPLEMENTAL_DIFFERENCES = {0x28: "¤", 0x57: "Œ", 0x5D: "Ÿ", 0x77: "œ", 0x7D: "ÿ"}
_SUPPLEMENTAL_UNASSIGNED = (0x24, 0x26, 0x2C, 0x2D, 0x2E, 0x2F, 0x34, 0x38, 0x3E, 0x50, 0x5E, 0x70, 0x7E)
_SUPPLEMENTAL = {
    code: ord(_SUPPLEMENTAL_DIFFERENCES.get(code, chr(code + 0x80)))
    for code in range(0x21, 0x7F)
    if code not in _SUPPLEMENTAL_UNASSIGNED
}

_ASCII: dict[int, int] = {}
_DESIGNATORS: dict[str, dict[int, int]] = {
    "B": _ASCII,
    "0": _SPECIAL_GRAPHICS,
    "A": _BRITISH,
    "4": _DUTCH,
    "C": _FINNISH,
    "5": _FINNISH,
    "R": _FRENCH,
    "f": _FRENCH,
    "Q": _FRENCH_CANADIAN,
    "9": _FRENCH_CANADIAN,
    "K": _GERMAN,
    "Y": _ITALIAN,
    "E": _NORWEGIAN_DANISH,
    "6": _NORWEGIAN_DANISH,
    "`": _NORWEGIAN_DANISH,
    "Z": _SPANISH,
    "H": _SWEDISH,
    "7": _SWEDISH,
    "=": _SWISS,
    "<": _SUPPLEMENTAL,
    "%5": _SUPPLEMENTAL,
}

type CharsetSnapshot = tuple[tuple[dict[int, int], ...], int]


class Charsets:
    """G0-G3 designations plus the locking and single shifts that select among them."""

    __slots__ = ("_locked", "_single", "_slots")

    def __init__(self) -> None:
        self._slots: list[dict[int, int]] = [_ASCII, _ASCII, _ASCII, _ASCII]
        self._locked = 0
        self._single: int | None = None

    def designate(self, slot: int, designator: str) -> None:
        """Load the set named by an SCS designator into G0-G3; unknown designators mean ASCII."""
        self._slots[slot] = _DESIGNATORS.get(designator, _ASCII)

    def lock(self, slot: int) -> None:
        """Locking shift: print through ``slot`` until the next shift (SI, SO, LS2, LS3)."""
        self._locked = slot

    def shift_once(self, slot: int) -> None:
        """Single shift: only the next printed character goes through ``slot`` (SS2, SS3)."""
        self._single = slot

    def translate(self, text: str) -> str:
        """Map a printable run through the active sets."""
        if self._single is not None:
            shifted = text[0].translate(self._slots[self._single])
            self._single = None
            text = shifted + text[1:].translate(self._slots[self._locked])
            return text
        table = self._slots[self._locked]
        return text.translate(table) if table else text

    def snapshot(self) -> CharsetSnapshot:
        """The state DECSC saves."""
        return tuple(self._slots), self._locked

    def restore(self, snapshot: CharsetSnapshot) -> None:
        """Reinstate a state captured by `snapshot`."""
        slots, self._locked = snapshot
        self._slots = list(slots)
        self._single = None
