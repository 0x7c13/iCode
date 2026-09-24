# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Buddy portraits with separate rarity corners and a compact nameplate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

from chrys.app.features.buddy.model import Rarity
from chrys.app.features.buddy.pixel_renderer import DEFAULT_BG_RGB
from chrys.app.features.buddy.pixel_sprites import PIXEL_HEIGHT, PIXEL_WIDTH, render_pixel_sprite

if TYPE_CHECKING:
    from chrys.app.features.buddy.model import Appearance

# The colour a tier is shown in, wherever it is shown.
RARITY_COLORS: dict[Rarity, str] = {
    Rarity.N: "#808080",
    Rarity.R: "#00ff00",
    Rarity.SR: "#bf00ff",
    Rarity.SSR: "#ffa500",
}

PORTRAIT_WIDTH = PIXEL_WIDTH + 4
PORTRAIT_HEIGHT = PIXEL_HEIGHT // 2 + 3
SHINY_FPS = 10


def _nameplate(look: Appearance, name: str, width: int, effect_tick: int, bg_rgb: tuple[int, int, int] | None) -> Text:
    badge = Text(f"[{look.rarity.value}]", style=RARITY_COLORS[look.rarity])
    if look.shiny:
        badge.append(" ✧", style="gold1")
        # A short sweep followed by a rest, confined to the badge. Keep the
        # highlight readable on light themes as well as dark terminal panels.
        phase = effect_tick % (len(badge) + 6)
        highlight = "bold reverse" if bg_rgb is None else "bold #6b4400" if sum(bg_rgb) > 384 else "bold #fff1b8"
        if phase < len(badge):
            badge.stylize(highlight, phase, phase + 1)
    badge.truncate(width, overflow="ellipsis")
    label = Text(" ".join(name.split()))
    label.truncate(max(0, width - badge.cell_len - 1), overflow="ellipsis")
    if label:
        label.append(" ")
    label.append(badge)
    label.align("center", width)
    return label


def render_portrait(
    look: Appearance,
    name: str,
    frame: int = 0,
    blink: bool = False,
    *,
    width: int = PORTRAIT_WIDTH,
    effect_tick: int = 0,
    bg_rgb: tuple[int, int, int] | None = DEFAULT_BG_RGB,
) -> list[Text]:
    """Render a fixed-height pixel portrait at the available panel width."""
    width = max(1, min(width, PORTRAIT_WIDTH))
    body = render_pixel_sprite(look.species, frame, blink, bg_rgb=bg_rgb, width=width)
    for row in body:
        row.truncate(width)
        row.align("center", width)

    if width >= PIXEL_WIDTH + 2:
        color = RARITY_COLORS[look.rarity]
        top = Text("┌" + " " * (width - 2) + "┐", style=color)
        bottom = Text("└" + " " * (width - 2) + "┘", style=color)
    else:
        # Corners need their own columns; omit them before crowding the body.
        top = Text(" " * width)
        bottom = Text(" " * width)
    return [top, *body, bottom, _nameplate(look, name, width, effect_tick, bg_rgb)]
