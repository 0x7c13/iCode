# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The blinking dot before a running row's title, the band of light that sweeps across the title, and the dot
that breathes where a waiting turn's next row will appear.

A tool group draws the blinking dot and the band while its calls run, in place of its fold arrow.
"""

from __future__ import annotations

import math

from rich.color import Color, ColorType, blend_rgb
from rich.style import Style
from rich.text import Text

FRAME_SECONDS = 1 / 16
"""How often a running row repaints, as Textual's loading indicator."""
RUNNING_DOT = "•"
"""The dot lit, and the still dot when animations are off."""
_RUNNING_DOT_DIM = "◦"
_BLINK_SECONDS = 0.6
"""How long the dot stays lit, then dim."""
_SWEEP_SECONDS = 2.0
"""How long the band takes to cross the title and the margins either side of it."""
_SWEEP_MARGIN = 10
"""Characters the band travels past each end of the title, so it enters and leaves out of sight."""
_BAND_HALF_WIDTH = 5.0
"""Characters from the band's centre to where it fades out."""
_BREATH_SECONDS = 1.2
"""How long the breathing dot takes to grow from its smallest to its largest and back."""
_BREATH_DOTS = (("·", 0.0), ("•", 0.3), ("●", 0.8))
"""The breathing dot's sizes, smallest first, each with the brightness from which it shows."""


def running_dot(now: float) -> tuple[str, bool]:
    """The dot at *now* and whether it is lit: a lit ``•`` and a dim ``◦`` take turns."""
    lit = int(now / _BLINK_SECONDS) % 2 == 0
    return (RUNNING_DOT if lit else _RUNNING_DOT_DIM), lit


def shimmer(text: Text, end: int, base: Color | None, highlight: Color | None, now: float) -> None:
    """Light *text*'s first *end* characters where the band is at *now*, from *base* towards *highlight*.

    Two truecolor colours blend smoothly; a theme of terminal colours lights only the band's middle, in *highlight*.
    """
    if base is None or highlight is None or end <= 0:
        return
    centre = (now % _SWEEP_SECONDS) / _SWEEP_SECONDS * (end + 2 * _SWEEP_MARGIN) - _SWEEP_MARGIN
    for index in range(
        max(math.ceil(centre - _BAND_HALF_WIDTH), 0), min(math.floor(centre + _BAND_HALF_WIDTH) + 1, end)
    ):
        level = 0.5 * (1 + math.cos(math.pi * min(abs(index - centre) / _BAND_HALF_WIDTH, 1.0)))
        if (color := _lit(base, highlight, level)) is not None:
            text.stylize(Style(color=color), index, index + 1)


def breathing_dot(base: Color | None, highlight: Color | None, now: float) -> Text:
    """The breathing dot at *now*: it grows ``·`` → ``•`` → ``●`` and back, from *base* towards *highlight* as it grows.

    Without both colours, as when animations are off, it is a still ``•`` in the row's own colour.
    """
    if base is None or highlight is None:
        return Text(RUNNING_DOT)
    level = 0.5 * (1 - math.cos(2 * math.pi * (now % _BREATH_SECONDS) / _BREATH_SECONDS))
    dot = next(dot for dot, threshold in reversed(_BREATH_DOTS) if level >= threshold)
    color = _lit(base, highlight, level)
    return Text(dot, style=Style(color=color) if color is not None else "")


def _lit(base: Color, highlight: Color, level: float) -> Color | None:
    """The colour *level* of the way from *base* to *highlight*; None where it leaves *base* as it is.

    Two truecolor colours blend; a theme of terminal colours takes *highlight* only from halfway.
    """
    if base.type is ColorType.TRUECOLOR and highlight.type is ColorType.TRUECOLOR:
        return Color.from_triplet(blend_rgb(base.get_truecolor(), highlight.get_truecolor(), level))
    return highlight if level >= 0.5 else None
