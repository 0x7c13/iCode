# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rich styles built from Textual color values."""

from __future__ import annotations

from rich.style import Style
from textual.color import Color, ColorParseError


def rich_style_from_textual_color(
    color: str | None,
    *,
    background: str | None = None,
    bold: bool | None = None,
    dim: bool | None = None,
    reverse: bool | None = None,
) -> Style:
    """Convert a plain Textual color token to a concrete Rich style.

    Textual's ANSI color names use an ``ansi_`` prefix that Rich doesn't
    accept in style strings. Parsing through Textual first preserves the
    ANSI color number while giving Rich a native color object. Composite
    CSS values such as ``auto 60%`` and text-style values must instead be
    resolved through a component class and ``get_component_rich_style()``.
    Invalid input falls back to an attribute-only style rather than crashing.
    """
    try:
        rich_color = Color.parse(color).rich_color if color is not None else None
    except ColorParseError:
        rich_color = None
    try:
        rich_background = Color.parse(background).rich_color if background is not None else None
    except ColorParseError:
        rich_background = None
    return Style(color=rich_color, bgcolor=rich_background, bold=bold, dim=dim, reverse=reverse)
