# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pixel rendering engine for buddies using Unicode half-blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from PIL import Image
from rich.color import Color
from rich.style import Style
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_HALF_BLOCK = "\u2580"
DEFAULT_BG_RGB: tuple[int, int, int] = (15, 18, 24)


def image_to_half_block_lines(
    image: Image.Image,
    *,
    bg_rgb: tuple[int, int, int] | None = DEFAULT_BG_RGB,
) -> list[Text]:
    """Convert a PIL RGBA image into TrueColor Rich Text lines using half-block characters.

    Each terminal character cell represents 2 vertical pixels (1:1 square pixel aspect ratio).
    Top pixel -> foreground color, Bottom pixel -> background color.

    Args:
        image: The PIL Image (RGBA or RGB) to render.
        bg_rgb: Background RGB color to composite transparent pixels against.
            None preserves the terminal background with binary transparency
            (alpha below 128 is transparent), since its RGB value is unknown.

    Returns:
        List of Rich Text instances representing terminal character rows.
    """
    image = image.convert("RGBA")
    # Ensure height is even for half-block pairing
    if image.height % 2 != 0:
        padded = Image.new("RGBA", (image.width, image.height + 1), (0, 0, 0, 0))
        padded.paste(image, (0, 0))
        image = padded

    if bg_rgb is not None:
        background = Image.new("RGBA", image.size, (*bg_rgb, 255))
        background.alpha_composite(image)
        image = background

    pixels = image.load()
    if pixels is None:
        return []

    lines: list[Text] = []
    width = image.width
    height = image.height

    for y in range(0, height, 2):
        line = Text()
        for x in range(width):
            upper = cast(tuple[int, int, int, int], pixels[x, y])
            lower = cast(tuple[int, int, int, int], pixels[x, y + 1])
            upper_color = Color.from_rgb(*upper[:3]) if upper[3] >= 128 else None
            lower_color = Color.from_rgb(*lower[:3]) if lower[3] >= 128 else None
            if upper_color is not None:
                line.append(_HALF_BLOCK, style=Style(color=upper_color, bgcolor=lower_color))
            elif lower_color is not None:
                # Invert the glyph, not the colors: its upper half must inherit
                # the terminal background rather than painting a guessed RGB.
                line.append("▄", style=Style(color=lower_color))
            else:
                line.append(" ")
        lines.append(line)

    return lines


def matrix_to_image(
    rows: Sequence[str | Sequence[int]],
    palette: Mapping[int, tuple[int, int, int, int]],
) -> Image.Image:
    """Construct a PIL RGBA Image from a 2D palette index matrix and color map.

    Args:
        rows: Sequence of rows containing palette indices (as numeric values or digits '0'-'9').
        palette: Mapping of index -> RGBA tuple.

    Returns:
        Constructed PIL RGBA Image.
    """
    height = len(rows)
    width = max((len(row) for row in rows), default=0) if height > 0 else 0

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    pixels = img.load()
    if pixels is None:
        return img

    for y, row in enumerate(rows):
        for x, char_or_val in enumerate(row):
            if isinstance(char_or_val, str):
                try:
                    idx = int(char_or_val)
                except ValueError:
                    idx = 0
            else:
                idx = int(char_or_val)

            pixels[x, y] = palette.get(idx, (0, 0, 0, 0))

    return img
