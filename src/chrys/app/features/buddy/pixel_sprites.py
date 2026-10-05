# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pixel artwork, blinking, and asset loading for buddies.

Each species' palette and poses are data, kept one file per species in
``sprites/<species>.toml`` beside this module; this module loads them, applies
user PNG overrides and builds the frames the portrait renders.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import cache, lru_cache
from importlib import resources
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from PIL import Image

from chrys.app.features.buddy.animation import IDLE_FRAME_COUNT
from chrys.app.features.buddy.model import Species
from chrys.app.features.buddy.pixel_renderer import DEFAULT_BG_RGB, image_to_half_block_lines, matrix_to_image
from chrys.foundation.platform import get_platform

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from rich.text import Text

# Resolution: 20 pixels wide x 16 pixels high (maps to 20 columns x 8 rows of terminal character cells)
PIXEL_WIDTH = 20
PIXEL_HEIGHT = 16

type Rgba = tuple[int, int, int, int]

# -----------------------------------------------------------------------------
# Palette roles
# -----------------------------------------------------------------------------

_SHADOW_PALETTE_KEY = 3  # palette index used for the species' own shadow
_EYE_PALETTE_KEY = 4  # palette index used for the primary eye colour

# -----------------------------------------------------------------------------
# Built-in artwork
# -----------------------------------------------------------------------------

_SPRITES_PACKAGE = "chrys.app.features.buddy"
_SPRITES_DIR = "sprites"


@dataclass(frozen=True, slots=True)
class SpeciesSprite:
    """One species' built-in artwork."""

    palette: Mapping[int, Rgba]
    """Palette index to RGBA color; ``3`` is the body shadow and ``4`` the pupils."""
    frames: tuple[tuple[str, ...], ...]
    """Idle poses, each ``PIXEL_HEIGHT`` rows of ``PIXEL_WIDTH`` palette-index digits."""


@cache
def species_sprite(species: Species) -> SpeciesSprite:
    """Load the packaged artwork for *species*, once per process."""
    name = f"{species.value}.toml"
    resource = resources.files(_SPRITES_PACKAGE) / _SPRITES_DIR / name
    try:
        data = tomllib.loads(resource.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{name}: {exc}") from exc
    return _parse_sprite(name, data)


def _parse_sprite(name: str, data: dict[str, Any]) -> SpeciesSprite:
    """Check the shape the frame builder relies on, so a bad file fails by name."""
    palette_table = data.get("palette")
    if not isinstance(palette_table, dict):
        raise ValueError(f"{name}: no [palette] table")
    palette = {_parse_index(name, key): _parse_rgba(name, key, value) for key, value in palette_table.items()}
    missing_roles = {_SHADOW_PALETTE_KEY, _EYE_PALETTE_KEY} - palette.keys()
    if missing_roles:
        raise ValueError(f"{name}: the palette has no index {min(missing_roles)}")
    frame_tables = data.get("frames")
    if not isinstance(frame_tables, list) or not frame_tables:
        raise ValueError(f"{name}: no frames")
    indices = {str(key) for key in palette}
    frames: list[tuple[str, ...]] = []
    for frame_idx, frame in enumerate(frame_tables):
        rows = frame.get("rows") if isinstance(frame, dict) else None
        if (
            not isinstance(rows, list)
            or len(rows) != PIXEL_HEIGHT
            or any(not isinstance(row, str) or len(row) != PIXEL_WIDTH for row in rows)
        ):
            raise ValueError(f"{name}: frame {frame_idx} is not {PIXEL_WIDTH}x{PIXEL_HEIGHT}")
        unknown = set("".join(rows)) - indices
        if unknown:
            raise ValueError(f"{name}: frame {frame_idx} uses indices missing from the palette: {sorted(unknown)}")
        frames.append(tuple(rows))
    return SpeciesSprite(palette=MappingProxyType(palette), frames=tuple(frames))


def _parse_index(name: str, key: str) -> int:
    # Each pixel of a frame row is one palette-index digit.
    if len(key) != 1 or key not in "0123456789":
        raise ValueError(f"{name}: palette key {key!r} is not a digit 0-9")
    return int(key)


def _parse_rgba(name: str, key: str, value: object) -> Rgba:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(
            isinstance(channel, int) and not isinstance(channel, bool) and 0 <= channel <= 255 for channel in value
        )
    ):
        raise ValueError(f"{name}: palette index {key} is not four 0-255 RGBA channels")
    red, green, blue, alpha = value
    return (red, green, blue, alpha)


def _get_assets_dir() -> Path:
    """Get path to custom buddy pixel assets directory."""
    return get_platform().config_dir / "extras" / "buddy" / "assets"


@lru_cache(maxsize=128)
def _load_external_asset(target_path: Path, _revision: tuple[int, int, int]) -> Image.Image:
    """Keep only resized artwork in a bounded cache, invalidated by file changes."""
    # Custom artwork is PNG by contract; no other decoder reads the file.
    with Image.open(target_path, formats=("PNG",)) as opened:
        img = opened.convert("RGBA")
        if img.size != (PIXEL_WIDTH, PIXEL_HEIGHT):
            img = img.resize((PIXEL_WIDTH, PIXEL_HEIGHT), Image.Resampling.NEAREST)
        # Source metadata may dwarf the thumbnail; the renderer needs pixels only.
        img.info.clear()
        return img


def load_external_pixel_frame(species: Species, frame_idx: int) -> Image.Image | None:
    """Load custom artwork from ~/.chrys/extras/buddy/assets/<species>_<frame_idx>.png.

    Checking the revision keeps edits and removals live without decoding the
    same PNG on every badge repaint. Callers receive their own mutable image.
    """
    target_path = _get_assets_dir() / f"{species.value}_{frame_idx}.png"
    try:
        stat = target_path.stat()
        image = _load_external_asset(target_path, (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size))
    except Exception:
        # Failed reads must remain retryable: a Windows sharing violation can
        # clear without changing the file revision. The LRU caches successes.
        return None
    return image.copy()


# -----------------------------------------------------------------------------
# Frame Builder — private helpers
# -----------------------------------------------------------------------------


def _build_petting_frame(frames: Sequence[Sequence[str]], frame_idx: int) -> list[str]:
    """Animate intact species poses at the caller's faster petting cadence."""
    pet_phase = frame_idx % IDLE_FRAME_COUNT
    raw = list(frames[pet_phase % len(frames)])
    if pet_phase == 0 and all(index == "0" for index in raw[0]):
        # Hop only when the top row is empty; tall ears, horns, and antennae
        # already touching the canvas edge must not be clipped off.
        return [*raw[1:], "0" * PIXEL_WIDTH]
    return raw


def _apply_blink(
    pixels: Any,
    raw_frame: list[str],
    palette: Mapping[int, Rgba],
) -> None:
    """Close the upper eye pixels while retaining a native-colored lower lid."""
    for y, row in enumerate(raw_frame[:-1]):
        for x, index in enumerate(row):
            if int(index) == _EYE_PALETTE_KEY and int(raw_frame[y + 1][x]) == _EYE_PALETTE_KEY:
                # Single-pixel eyes stay intact; taller eyes become a thin
                # line instead of disappearing into the face's shadow color.
                pixels[x, y] = palette[_SHADOW_PALETTE_KEY]


# -----------------------------------------------------------------------------
# Frame Builder
# -----------------------------------------------------------------------------


def build_pixel_frame(
    species: Species,
    frame_idx: int = 0,
    blink: bool = False,
    *,
    width: int = PIXEL_WIDTH,
) -> Image.Image:
    """Construct 20x16 RGBA artwork, fitting it to narrower panels if requested."""
    # 1. Prefer user-supplied external PNG asset
    external_img = load_external_pixel_frame(species, frame_idx)
    if external_img is not None:
        return _fit_pixel_frame(external_img, width)

    # 2. Keep species colours intact; portrait chrome owns rarity and shiny.
    sprite = species_sprite(species)
    palette = sprite.palette

    # 3. Resolve the raw pixel-matrix frame (idle or dynamic petting)
    frames = sprite.frames
    raw_frame = (
        _build_petting_frame(frames, frame_idx)
        if frame_idx >= IDLE_FRAME_COUNT
        else list(frames[frame_idx % len(frames)])
    )

    # 4. Preserve the authored pupil colors; only blinking changes their shape.
    img = matrix_to_image(raw_frame, palette)
    pixels = img.load()
    if pixels is None:
        raise RuntimeError("The buddy sprite has no pixel buffer.")

    if blink:
        _apply_blink(pixels, raw_frame, palette)

    if width < PIXEL_WIDTH:
        # Nearest-neighbor sampling alone can skip a one-column pupil. Keep
        # its visible pixels at their scaled positions, including closed lids.
        pupils = [
            (x, y)
            for y, row in enumerate(raw_frame)
            for x, key in enumerate(row)
            if key == "4" and pixels[x, y] == palette[_EYE_PALETTE_KEY]
        ]
        return _fit_pixel_frame(img, width, pupils=pupils)
    return img


def _fit_pixel_frame(image: Image.Image, width: int, *, pupils: list[tuple[int, int]] | None = None) -> Image.Image:
    """Fit the whole canvas, preserving pupil contrast and a fixed row count."""
    width = max(1, min(width, PIXEL_WIDTH))
    if width == PIXEL_WIDTH:
        return image
    height = max(1, round(PIXEL_HEIGHT * width / PIXEL_WIDTH))
    fitted = image.resize((width, height), Image.Resampling.NEAREST)
    for x, y in pupils or ():
        target = (int((x + 0.5) * width / PIXEL_WIDTH), int((y + 0.5) * height / PIXEL_HEIGHT))
        color = image.getpixel((x, y))
        if color is None:
            raise RuntimeError("The buddy pupil has no pixel value.")
        fitted.putpixel(target, color)
    canvas = Image.new("RGBA", (width, PIXEL_HEIGHT))
    canvas.paste(fitted, (0, (PIXEL_HEIGHT - height) // 2))
    return canvas


# -----------------------------------------------------------------------------
# Main Render Function
# -----------------------------------------------------------------------------


def render_pixel_sprite(
    species: Species,
    frame: int = 0,
    blink: bool = False,
    *,
    bg_rgb: tuple[int, int, int] | None = DEFAULT_BG_RGB,
    width: int = PIXEL_WIDTH,
) -> list[Text]:
    """Fit pixel artwork to the available columns, retaining eight terminal rows."""
    img = build_pixel_frame(
        species,
        frame_idx=frame,
        blink=blink,
        width=width,
    )
    return image_to_half_block_lines(img, bg_rgb=bg_rgb)
