# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for high-precision half-block pixel rendering and pixel sprite engine."""

import pytest
from PIL import Image
from rich.console import Console
from rich.text import Text

from chrys.app.features.buddy.model import Species
from chrys.app.features.buddy.pixel_renderer import image_to_half_block_lines, matrix_to_image
from chrys.app.features.buddy.pixel_sprites import (
    PIXEL_HEIGHT,
    PIXEL_WIDTH,
    SPECIES_PALETTES,
    build_pixel_frame,
    render_pixel_sprite,
)


def test_matrix_to_image_construction():
    """matrix_to_image converts a 2D palette matrix to a PIL RGBA Image."""
    matrix = [
        "0123",
        "3210",
    ]
    palette = {
        0: (0, 0, 0, 0),
        1: (255, 0, 0, 255),
        2: (0, 255, 0, 255),
        3: (0, 0, 255, 255),
    }
    img = matrix_to_image(matrix, palette)
    assert img.size == (4, 2)
    assert img.getpixel((0, 0)) == (0, 0, 0, 0)
    assert img.getpixel((1, 0)) == (255, 0, 0, 255)
    assert img.getpixel((2, 0)) == (0, 255, 0, 255)
    assert img.getpixel((3, 0)) == (0, 0, 255, 255)


def test_image_to_half_block_lines_pairing():
    """image_to_half_block_lines converts 10-pixel height image into 5 Rich Text lines."""
    img = Image.new("RGBA", (16, 10), (255, 255, 255, 255))
    lines = image_to_half_block_lines(img, bg_rgb=(15, 18, 24))
    assert len(lines) == 5
    for line in lines:
        assert isinstance(line, Text)
        assert line.cell_len == 16


def test_build_pixel_frame_returns_valid_image():
    """build_pixel_frame constructs a 20x16 PIL RGBA Image for all species."""
    for species in Species:
        img = build_pixel_frame(species, frame_idx=0)
        assert isinstance(img, Image.Image)
        assert img.size == (PIXEL_WIDTH, PIXEL_HEIGHT)


def test_render_pixel_sprite_all_species():
    """render_pixel_sprite renders every species into 8 terminal character lines."""
    for species in Species:
        lines = render_pixel_sprite(species, frame=0)
        assert len(lines) == PIXEL_HEIGHT // 2
        for line in lines:
            assert isinstance(line, Text)
            assert line.cell_len == PIXEL_WIDTH


def test_render_pixel_sprite_blink_mode():
    """Blink mode executes without error and returns 8 lines."""
    lines = render_pixel_sprite(Species.CAT, frame=0, blink=True)
    assert len(lines) == PIXEL_HEIGHT // 2


@pytest.mark.parametrize("size", [(16, 10), (20, 16)])
def test_external_png_loading_override(tmp_path, monkeypatch, size):
    """External PNG asset overrides default matrix frame when present in assets dir."""
    assets_dir = tmp_path / "extras" / "buddy" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Both legacy-sized and native-sized custom assets remain usable.
    custom_img = Image.new("RGBA", size, (255, 0, 0, 255))
    custom_img.save(assets_dir / "duck_0.png")

    monkeypatch.setattr(
        "chrys.app.features.buddy.pixel_sprites._get_assets_dir",
        lambda: assets_dir,
    )

    frame_img = build_pixel_frame(Species.DUCK, frame_idx=0)
    assert frame_img.size == (20, 16)
    assert frame_img.getpixel((0, 0)) == (255, 0, 0, 255)


def test_all_species_have_pixel_palettes():
    """Every Species enum has an entry in SPECIES_PALETTES."""
    for species in Species:
        assert species in SPECIES_PALETTES, f"Missing SPECIES_PALETTES entry for {species}"


def test_all_species_have_unique_pixel_frames():
    """Every Species enum has an explicit entry in DEFAULT_PIXEL_FRAMES with 3 frames of 20x16."""
    from chrys.app.features.buddy.pixel_sprites import DEFAULT_PIXEL_FRAMES

    for species in Species:
        assert species in DEFAULT_PIXEL_FRAMES, f"Missing DEFAULT_PIXEL_FRAMES for {species}"
        frames = DEFAULT_PIXEL_FRAMES[species]
        assert len(frames) >= 3, f"{species} needs at least 3 pixel frames, got {len(frames)}"
        assert len({tuple(frame) for frame in frames}) == len(frames), f"{species} repeats an idle frame"
        for frame_idx, frame in enumerate(frames):
            assert len(frame) == PIXEL_HEIGHT, f"{species} frame {frame_idx} must have {PIXEL_HEIGHT} rows"
            for row_idx, row in enumerate(frame):
                assert len(row) == PIXEL_WIDTH, f"{species} frame {frame_idx} row {row_idx} must be {PIXEL_WIDTH} wide"


def test_half_block_colors_preserve_pairing_and_composite_alpha() -> None:
    image = Image.new("RGBA", (1, 2))
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.putpixel((0, 1), (0, 0, 255, 128))
    line = image_to_half_block_lines(image, bg_rgb=(255, 255, 255))[0]
    style = line.get_style_at_offset(Console(force_terminal=False, _environ={}), 0)
    assert line.plain == "▀"
    assert style.color is not None and style.color.triplet == (255, 0, 0)
    assert style.bgcolor is not None and style.bgcolor.triplet == (127, 127, 255)


def test_terminal_background_survives_all_half_block_transparency_combinations() -> None:
    image = Image.new("RGBA", (4, 2))
    red, blue = (255, 0, 0, 255), (0, 0, 255, 255)
    for x, upper, lower in ((0, red, blue), (1, red, (0, 0, 0, 0)), (2, (0, 0, 0, 0), blue)):
        image.putpixel((x, 0), upper)
        image.putpixel((x, 1), lower)
    line = image_to_half_block_lines(image, bg_rgb=None)[0]
    assert line.plain == "▀▀▄ "
    console = Console(force_terminal=False, _environ={})
    styles = [line.get_style_at_offset(console, x) for x in range(4)]
    assert styles[0].color.triplet == styles[1].color.triplet == red[:3]
    assert styles[0].bgcolor.triplet == styles[2].color.triplet == blue[:3]
    assert all(style.bgcolor is None for style in styles[1:])
    assert styles[3].color is None


@pytest.mark.parametrize("alpha", [0, 127, 128, 255])
def test_terminal_transparency_handles_partial_alpha_and_odd_height(alpha: int) -> None:
    image = Image.new("RGBA", (1, 1), (12, 34, 56, alpha))
    line = image_to_half_block_lines(image, bg_rgb=None)[0]
    assert line.plain == ("▀" if alpha >= 128 else " ")
    style = line.get_style_at_offset(Console(force_terminal=False, _environ={}), 0)
    assert style.bgcolor is None
    assert (style.color is not None) == (alpha >= 128)


@pytest.mark.parametrize("height", [1, 2])
def test_palette_transparency_survives_odd_height_padding(height: int) -> None:
    image = Image.new("P", (1, height), 0)
    image.putpalette([255, 0, 0] + [0, 0, 0] * 255)
    image.info["transparency"] = 0
    line = image_to_half_block_lines(image, bg_rgb=(10, 20, 30))[0]
    style = line.get_style_at_offset(Console(force_terminal=False, _environ={}), 0)
    assert style.color is not None and style.color.triplet == (10, 20, 30)
    assert style.bgcolor is not None and style.bgcolor.triplet == (10, 20, 30)
