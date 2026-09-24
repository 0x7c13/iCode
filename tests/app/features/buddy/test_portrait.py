# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Portrait chrome must identify rarity without modifying the artwork."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rich.console import Console
from rich.text import Text

from chrys.app.features.buddy.model import Appearance, Rarity, Species
from chrys.app.features.buddy.pixel_sprites import PIXEL_HEIGHT, PIXEL_WIDTH, render_pixel_sprite
from chrys.app.features.buddy.portrait import PORTRAIT_HEIGHT, PORTRAIT_WIDTH, RARITY_COLORS, render_portrait


def _look(species: Species = Species.RABBIT) -> Appearance:
    return Appearance(species, Rarity.N)


def _styles(text: Text) -> list:
    console = Console(force_terminal=False, _environ={})
    return [text.get_style_at_offset(console, offset) for offset in range(len(text))]


@pytest.mark.parametrize("species", list(Species))
def test_rarity_and_shiny_leave_the_entire_body_canvas_unchanged(species: Species) -> None:
    look = _look(species)
    for frame in range(6):
        body = render_pixel_sprite(species, frame)
        assert len(body) == PIXEL_HEIGHT // 2
        for rarity in Rarity:
            for shiny in (False, True):
                portrait = render_portrait(replace(look, rarity=rarity, shiny=shiny), "Momo", frame)
                assert len(portrait) == PORTRAIT_HEIGHT
                for expected, row in zip(body, portrait[1:-2], strict=True):
                    actual = row[2 : 2 + PIXEL_WIDTH]
                    assert actual.plain == expected.plain
                    assert _styles(actual) == _styles(expected)


@pytest.mark.parametrize("rarity", list(Rarity))
def test_colored_corners_and_nameplate_identify_rarity(rarity: Rarity) -> None:
    portrait = render_portrait(replace(_look(), rarity=rarity), "Momo")
    assert portrait[0].plain == "┌" + " " * (PORTRAIT_WIDTH - 2) + "┐"
    assert portrait[-2].plain == "└" + " " * (PORTRAIT_WIDTH - 2) + "┘"
    assert portrait[-1].plain.strip() == f"Momo [{rarity.value}]"
    assert _styles(portrait[0])[0].color is not None
    assert _styles(portrait[0])[0].color.name == RARITY_COLORS[rarity]


@pytest.mark.parametrize("bg_rgb", [(15, 18, 24), (245, 245, 245)])
def test_shiny_sweep_is_confined_to_badge(bg_rgb: tuple[int, int, int]) -> None:
    look = replace(_look(), rarity=Rarity.SSR, shiny=True)
    frames = [render_portrait(look, "Momo", effect_tick=tick, bg_rgb=bg_rgb) for tick in range(12)]
    baseline = frames[0]
    for frame in frames[1:]:
        assert [line.plain for line in frame] == [line.plain for line in baseline]
        for before, after in zip(baseline[:-1], frame[:-1], strict=True):
            assert _styles(before) == _styles(after)
        name_end = frame[-1].plain.index("[")
        assert _styles(frame[-1])[:name_end] == _styles(baseline[-1])[:name_end]
    assert _styles(frames[0][-1]) != _styles(frames[1][-1])
    assert baseline[-1].plain.strip() == "Momo [SSR] ✧"


@pytest.mark.parametrize("width", [12, 16, 19, 20, 21, 22, 24, 40])
@pytest.mark.parametrize("name", ["Momo", "小兔子的名字非常非常长", "[red]", "two\nlines\tname"])
def test_nameplate_fits_and_preserves_rarity_with_long_or_literal_names(width: int, name: str) -> None:
    portrait = render_portrait(replace(_look(), rarity=Rarity.SSR, shiny=True), name, width=width)
    assert len(portrait) == PORTRAIT_HEIGHT
    assert all(row.cell_len == min(width, PORTRAIT_WIDTH) for row in portrait)
    assert portrait[-1].plain.rstrip().endswith("[SSR] ✧")
    assert "\n" not in portrait[-1].plain and "\t" not in portrait[-1].plain
    if name == "[red]" and width >= 16:
        assert "[red] [SSR]" in portrait[-1].plain
    assert all("▀" in row.plain for row in portrait[1:-2])


def test_external_artwork_gets_the_same_frame_and_nameplate(tmp_path, monkeypatch) -> None:
    from PIL import Image

    Image.new("RGBA", (16, 10), (210, 30, 80, 255)).save(tmp_path / "rabbit_0.png")
    monkeypatch.setattr("chrys.app.features.buddy.pixel_sprites._get_assets_dir", lambda: tmp_path)
    portrait = render_portrait(replace(_look(), rarity=Rarity.SSR, shiny=True), "Custom")
    assert portrait[-1].plain.strip() == "Custom [SSR] ✧"
    for row in portrait[1:-2]:
        assert all(
            style.color is not None and style.color.triplet == (210, 30, 80)
            for style in _styles(row[2 : 2 + PIXEL_WIDTH])
        )


@pytest.mark.parametrize("width", [1, 10, 16, 19])
def test_narrow_custom_art_keeps_canvas_edges_and_vertical_alignment(tmp_path, monkeypatch, width: int) -> None:
    from PIL import Image

    # An opaque canvas with different left/right halves detects cropping,
    # skipped overrides, and asymmetric padding during narrow rendering.
    red, green, background = (210, 30, 80), (20, 160, 70), (245, 245, 245)
    image = Image.new("RGBA", (20, 16), (*red, 255))
    image.paste((*green, 255), (10, 0, 20, 16))
    image.save(tmp_path / "rabbit_0.png")
    monkeypatch.setattr("chrys.app.features.buddy.pixel_sprites._get_assets_dir", lambda: tmp_path)
    portrait = render_portrait(_look(), "Custom", width=width, bg_rgb=background)
    assert len(portrait) == PORTRAIT_HEIGHT
    assert all(row.cell_len == width for row in portrait)
    pixels = []
    for row in portrait[1:-2]:
        styles = _styles(row)
        pixels.extend(([style.color.triplet for style in styles], [style.bgcolor.triplet for style in styles]))
    occupied = [y for y, row in enumerate(pixels) if any(color != background for color in row)]
    assert len(occupied) == max(1, round(16 * width / 20))
    assert abs(occupied[0] - (15 - occupied[-1])) <= 1
    for y in occupied:
        assert pixels[y][-1] == green
        if width > 1:
            assert pixels[y][0] == red


def test_ansi_shiny_sweep_uses_terminal_contrast_without_a_fixed_rgb_highlight() -> None:
    look = replace(_look(), rarity=Rarity.SSR, shiny=True)
    frames = [render_portrait(look, "Momo", effect_tick=tick, bg_rgb=None)[-1] for tick in (0, 1)]
    for tick, frame in enumerate(frames):
        badge_start = frame.plain.index("[")
        styles = _styles(frame)
        assert styles[badge_start + tick].bold and styles[badge_start + tick].reverse
        assert not styles[badge_start + 1 - tick].reverse
        assert styles[badge_start + tick].color == styles[badge_start + 1 - tick].color
