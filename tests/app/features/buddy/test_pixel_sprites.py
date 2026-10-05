# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Regressions for complete pixel artwork, visible blinks, and asset loading."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from chrys.app.features.buddy.model import Species
from chrys.app.features.buddy.pixel_sprites import (
    PIXEL_HEIGHT,
    PIXEL_WIDTH,
    build_pixel_frame,
    load_external_pixel_frame,
    species_sprite,
)
from tests.support.images import image_bytes


@pytest.mark.parametrize("species", list(Species))
def test_idle_artwork_has_no_unmapped_or_transparent_features(species: Species) -> None:
    for frame_idx, rows in enumerate(species_sprite(species).frames):
        image = build_pixel_frame(species, frame_idx)
        for y, row in enumerate(rows):
            for x, index in enumerate(row):
                assert int(index) in species_sprite(species).palette, (species, frame_idx, x, y, index)
                assert bool(image.getpixel((x, y))[3]) == (index != "0")


@pytest.mark.parametrize("species", list(Species))
def test_petting_preserves_complete_species_artwork(species: Species) -> None:
    pet_frames: list[bytes] = []
    for frame_idx in range(3):
        idle = build_pixel_frame(species, frame_idx)
        pet = build_pixel_frame(species, frame_idx + 3)
        assert pet.size == (PIXEL_WIDTH, PIXEL_HEIGHT)
        idle_bounds = idle.getbbox()
        pet_bounds = pet.getbbox()
        assert idle_bounds is not None and pet_bounds is not None
        # The whole pose may move, but ears, antennae, feet, and every other
        # pixel must survive; floating particles must not replace body rows.
        idle_body = idle.crop(idle_bounds)
        pet_body = pet.crop(pet_bounds)
        assert pet_body.size == idle_body.size, (species, frame_idx)
        assert pet_body.tobytes() == idle_body.tobytes(), (species, frame_idx)
        pet_frames.append(pet.tobytes())
    assert len(set(pet_frames)) == 3


@pytest.mark.parametrize("species", list(Species))
@pytest.mark.parametrize("frame", range(6))
def test_every_builtin_pose_has_a_visible_blink(species: Species, frame: int) -> None:
    open_eyes = build_pixel_frame(species, frame)
    closed_eyes = build_pixel_frame(species, frame, blink=True)
    assert closed_eyes.tobytes() != open_eyes.tobytes(), (species, frame)


@pytest.mark.parametrize("species", list(Species))
@pytest.mark.parametrize("blink", [False, True])
def test_blinks_preserve_visible_pupils(species: Species, blink: bool) -> None:
    for frame_idx in range(6):
        rows = species_sprite(species).frames[frame_idx % 3]
        idle = build_pixel_frame(species, frame_idx % 3)
        base = build_pixel_frame(species, frame_idx)
        actual = build_pixel_frame(species, frame_idx, blink=blink)
        idle_bounds, bounds = idle.getbbox(), base.getbbox()
        assert idle_bounds is not None and bounds is not None
        offset_y = bounds[1] - idle_bounds[1]
        eye_pixels = {(x, y + offset_y) for y, row in enumerate(rows) for x, value in enumerate(row) if value == "4"}
        assert eye_pixels
        # Every vertical pupil stroke must keep its bottom pixel in the native
        # eye color, even for single-pixel eyes and during a closed-eye pose.
        for x, y in eye_pixels:
            if (x, y + 1) not in eye_pixels:
                assert actual.getpixel((x, y)) == species_sprite(species).palette[4], (species, frame_idx, blink)
        for y in range(base.height):
            for x in range(base.width):
                if (x, y) not in eye_pixels:
                    assert actual.getpixel((x, y)) == base.getpixel((x, y))


def test_blink_preserves_other_features_with_the_same_color() -> None:
    base = build_pixel_frame(Species.GHOST)
    styled = build_pixel_frame(Species.GHOST, blink=True)
    # Eyes and mouth intentionally share their RGB color, but only key 4 is an eye.
    assert base.getpixel((7, 6)) == base.getpixel((9, 8))
    assert styled.getpixel((7, 6)) != base.getpixel((7, 6))
    assert styled.getpixel((9, 8)) == base.getpixel((9, 8))


def test_robot_blinks_its_led_eyes_without_changing_its_antenna() -> None:
    base = build_pixel_frame(Species.ROBOT)
    blink = build_pixel_frame(Species.ROBOT, blink=True)
    assert base.getpixel((7, 6)) == (0, 215, 255, 255)
    assert blink.getpixel((7, 6)) != base.getpixel((7, 6))
    assert blink.getpixel((9, 1)) == base.getpixel((9, 1)) == (255, 55, 85, 255)


def test_external_artwork_is_decoded_once_and_returns_independent_frames(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rabbit_0.png"
    metadata = PngInfo()
    metadata.add_text("comment", "source metadata " * 1000)
    Image.new("RGBA", (1024, 1024), (20, 40, 60, 255)).save(path, pnginfo=metadata)
    monkeypatch.setattr("chrys.app.features.buddy.pixel_sprites._get_assets_dir", lambda: tmp_path)
    with patch("chrys.app.features.buddy.pixel_sprites.Image.open", autospec=True, side_effect=Image.open) as opened:
        first = build_pixel_frame(Species.RABBIT)
        assert first.size == (PIXEL_WIDTH, PIXEL_HEIGHT)
        assert first.info == {}
        first.putpixel((0, 0), (255, 0, 0, 255))
        for _ in range(10):
            frame = build_pixel_frame(Species.RABBIT)
            assert frame.getpixel((0, 0)) == (20, 40, 60, 255)
        opened.assert_called_once_with(path, formats=("PNG",))


def test_external_artwork_cache_tracks_same_size_edits(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rabbit_0.png"
    Image.new("RGBA", (16, 10), (20, 40, 60, 255)).save(path, compress_level=0)
    before = path.stat()
    monkeypatch.setattr("chrys.app.features.buddy.pixel_sprites._get_assets_dir", lambda: tmp_path)
    assert build_pixel_frame(Species.RABBIT).getpixel((0, 0)) == (20, 40, 60, 255)
    Image.new("RGBA", (16, 10), (80, 100, 120, 255)).save(path, compress_level=0)
    assert path.stat().st_size == before.st_size
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    assert build_pixel_frame(Species.RABBIT).getpixel((0, 0)) == (80, 100, 120, 255)


def test_external_artwork_recovers_after_missing_invalid_and_removed_files(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rabbit_0.png"
    monkeypatch.setattr("chrys.app.features.buddy.pixel_sprites._get_assets_dir", lambda: tmp_path)
    assert load_external_pixel_frame(Species.RABBIT, 0) is None
    path.write_bytes(b"not a png")
    assert load_external_pixel_frame(Species.RABBIT, 0) is None
    Image.new("RGBA", (16, 10), (20, 40, 60, 255)).save(path)
    assert build_pixel_frame(Species.RABBIT).getpixel((0, 0)) == (20, 40, 60, 255)
    path.unlink()
    assert load_external_pixel_frame(Species.RABBIT, 0) is None


def test_external_artwork_is_read_only_as_png(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rabbit_0.png"
    monkeypatch.setattr("chrys.app.features.buddy.pixel_sprites._get_assets_dir", lambda: tmp_path)
    path.write_bytes(image_bytes("BMP", size=(16, 10)))
    assert load_external_pixel_frame(Species.RABBIT, 0) is None
    Image.new("RGBA", (16, 10), (20, 40, 60, 255)).save(path)
    assert build_pixel_frame(Species.RABBIT).getpixel((0, 0)) == (20, 40, 60, 255)


def test_external_artwork_retries_transient_read_errors_without_a_file_edit(tmp_path, monkeypatch) -> None:
    path = tmp_path / "rabbit_0.png"
    Image.new("RGBA", (16, 10), (20, 40, 60, 255)).save(path)
    monkeypatch.setattr("chrys.app.features.buddy.pixel_sprites._get_assets_dir", lambda: tmp_path)
    original_open = Image.open
    attempts = 0

    def transient_open(fp, mode="r", formats=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("asset temporarily locked by its editor")
        return original_open(fp, mode, formats=formats)

    with patch(
        "chrys.app.features.buddy.pixel_sprites.Image.open", autospec=True, side_effect=transient_open
    ) as opened:
        assert load_external_pixel_frame(Species.RABBIT, 0) is None
        assert build_pixel_frame(Species.RABBIT).getpixel((0, 0)) == (20, 40, 60, 255)
        assert build_pixel_frame(Species.RABBIT).getpixel((0, 0)) == (20, 40, 60, 255)
        assert opened.call_count == 2


@pytest.mark.parametrize("species", list(Species))
@pytest.mark.parametrize("width", [12, 16, 19])
@pytest.mark.parametrize("blink", [False, True])
def test_narrow_pixel_frames_keep_visible_pupils(species: Species, width: int, blink: bool) -> None:
    for frame in range(6):
        image = build_pixel_frame(species, frame, blink=blink, width=width)
        assert image.size == (width, PIXEL_HEIGHT)
        colors = {color for _, color in image.getcolors()}
        assert species_sprite(species).palette[4] in colors
        assert colors <= set(species_sprite(species).palette.values())
