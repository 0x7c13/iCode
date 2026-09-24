# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Graphic rendition: the pen that styles every cell, and the SGR sequence that changes it."""

from __future__ import annotations

from enum import IntFlag
from functools import lru_cache
from typing import NamedTuple


class Rgb(NamedTuple):
    """A direct 24-bit color."""

    red: int
    green: int
    blue: int


type Color = int | Rgb
"""A palette index (0-255) or a direct color. ``None`` stands for the terminal default."""


class Attribute(IntFlag):
    """Character attributes that SGR can switch on and off."""

    NONE = 0
    BOLD = 1
    DIM = 2
    ITALIC = 4
    UNDERLINE = 8
    DOUBLE_UNDERLINE = 16
    BLINK = 32
    REVERSE = 64
    CONCEAL = 128
    STRIKE = 256
    OVERLINE = 512


class Pen(NamedTuple):
    """The rendition applied to a cell. Immutable, so cells share pens freely."""

    foreground: Color | None = None
    background: Color | None = None
    attributes: Attribute = Attribute.NONE
    link: str | None = None

    @property
    def eraser(self) -> Pen:
        """The pen erased cells take: only the background survives (background color erase)."""
        if self.background is None:
            return DEFAULT_PEN
        return Pen(background=self.background)


DEFAULT_PEN = Pen()

_SET_ATTRIBUTE = {
    1: Attribute.BOLD,
    2: Attribute.DIM,
    3: Attribute.ITALIC,
    4: Attribute.UNDERLINE,
    5: Attribute.BLINK,
    6: Attribute.BLINK,
    7: Attribute.REVERSE,
    8: Attribute.CONCEAL,
    9: Attribute.STRIKE,
    21: Attribute.DOUBLE_UNDERLINE,
    53: Attribute.OVERLINE,
}
_CLEAR_ATTRIBUTE = {
    22: Attribute.BOLD | Attribute.DIM,
    23: Attribute.ITALIC,
    24: Attribute.UNDERLINE | Attribute.DOUBLE_UNDERLINE,
    25: Attribute.BLINK,
    27: Attribute.REVERSE,
    28: Attribute.CONCEAL,
    29: Attribute.STRIKE,
    55: Attribute.OVERLINE,
}
_UNDERLINES = Attribute.UNDERLINE | Attribute.DOUBLE_UNDERLINE
_EXTENDED_FOREGROUND, _EXTENDED_BACKGROUND, _EXTENDED_UNDERLINE = 38, 48, 58
_MAX_PARAMETER = 0xFFFF


def _number(field: str) -> int:
    """An SGR field as a number; empty means zero and junk can never match a code."""
    if not field:
        return 0
    return min(int(field), _MAX_PARAMETER) if field.isdecimal() else -1


def _direct_color(fields: list[int]) -> Color | None:
    """Decode the fields after a 38/48/58 introducer, or ``None`` when they name no color."""
    match fields:
        case [5, index, *_] if 0 <= index <= 255:
            return index
        case [2, red, green, blue] | [2, _, red, green, blue, *_]:
            if all(0 <= channel <= 255 for channel in (red, green, blue)):
                return Rgb(red, green, blue)
    return None


@lru_cache(maxsize=4096)
def apply_sgr(pen: Pen, parameters: str) -> Pen:
    """Return ``pen`` after a Select Graphic Rendition sequence with the given raw parameters.

    Extended colors are accepted in both spellings: the legacy ``38;2;R;G;B`` that spends following
    parameters, and the ITU T.416 ``38:2::R:G:B`` that keeps everything in one parameter's
    sub-fields (with or without the color-space slot). Unknown codes are skipped.
    """
    foreground, background, attributes, link = pen
    groups = parameters.split(";")
    position = 0
    while position < len(groups):
        code_field, *sub_fields = groups[position].split(":")
        code = _number(code_field)
        position += 1
        if code == 0:
            foreground = background = None
            attributes = Attribute.NONE
        elif code == 4 and sub_fields:
            attributes &= ~_UNDERLINES
            if (shape := _number(sub_fields[0])) == 2:
                attributes |= Attribute.DOUBLE_UNDERLINE
            elif shape:
                attributes |= Attribute.UNDERLINE
        elif (flag := _SET_ATTRIBUTE.get(code)) is not None:
            attributes |= flag
        elif (flags := _CLEAR_ATTRIBUTE.get(code)) is not None:
            attributes &= ~flags
        elif 30 <= code <= 37:
            foreground = code - 30
        elif 40 <= code <= 47:
            background = code - 40
        elif 90 <= code <= 97:
            foreground = code - 90 + 8
        elif 100 <= code <= 107:
            background = code - 100 + 8
        elif code == 39:
            foreground = None
        elif code == 49:
            background = None
        elif code in (_EXTENDED_FOREGROUND, _EXTENDED_BACKGROUND, _EXTENDED_UNDERLINE):
            if sub_fields:
                color = _direct_color([_number(field) for field in sub_fields])
            else:
                following = [_number(field.partition(":")[0]) for field in groups[position : position + 4]]
                color = _direct_color(following[:2] if following[:1] == [5] else following)
                # A legacy introducer spends its operands even when they name no usable color.
                position += 2 if following[:1] == [5] else 4 if following[:1] == [2] else 0
            if color is None:
                continue
            if code == _EXTENDED_FOREGROUND:
                foreground = color
            elif code == _EXTENDED_BACKGROUND:
                background = color
    return Pen(foreground, background, attributes, link)
