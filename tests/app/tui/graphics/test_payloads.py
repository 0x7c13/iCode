# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What a terminal is sent for one image."""

from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image, UnidentifiedImageError

from chrys.app.tui.graphics import payloads
from chrys.app.tui.widgets.chat.image_preview import TRANSPARENT_BACKGROUND
from tests.support.images import image_bytes
from tests.support.sixel_decoder import decode_sixel


def _near(color: tuple[int, int, int] | None, expected: tuple[int, int, int]) -> bool:
    """Sixel colors are percentages, so a channel can come back up to 2 off."""
    return color is not None and all(abs(got - want) <= 2 for got, want in zip(color, expected, strict=True))


def _data_uri(image: Image.Image, image_format: str = "PNG") -> str:
    out = BytesIO()
    image.save(out, format=image_format)
    return f"data:image/{image_format.lower()};base64,{base64.b64encode(out.getvalue()).decode()}"


def test_kitty_png_fits_the_box_pixels_and_keeps_transparency() -> None:
    image = Image.new("RGBA", (640, 360), (255, 0, 0, 255))
    image.paste((0, 0, 0, 0), (0, 0, 320, 360))

    png = payloads.kitty_png(_data_uri(image), columns=20, rows=5, cell_pixels=(10, 20))

    with Image.open(BytesIO(png), formats=("PNG",)) as sent:
        assert sent.size == (178, 100)
        assert sent.mode == "RGBA"
        assert sent.getchannel("A").getpixel((0, 0)) == 0
        assert sent.getchannel("A").getpixel((177, 99)) == 255


def test_kitty_png_never_upscales_the_original() -> None:
    png = payloads.kitty_png(_data_uri(Image.new("RGB", (40, 20))), columns=50, rows=10, cell_pixels=(16, 34))

    with Image.open(BytesIO(png), formats=("PNG",)) as sent:
        assert (sent.size, sent.mode) == ((40, 20), "RGB")


def test_sixel_rows_fill_the_half_block_box_and_pad_its_odd_last_row() -> None:
    image = Image.new("RGBA", (30, 30), (0, 200, 0, 255))
    image.paste((0, 0, 0, 0), (0, 0, 10, 10))

    rows = payloads.sixel_rows(_data_uri(image), columns=3, rows=2, pixel_height=3)

    first, second = (decode_sixel(row) for row in rows)
    assert first.raster == second.raster == (1, 1, 30, 20)
    assert _near(first.rgb(0, 0), TRANSPARENT_BACKGROUND)
    assert _near(first.rgb(29, 0), (0, 200, 0))
    assert _near(second.rgb(29, 9), (0, 200, 0))
    assert _near(second.rgb(29, 10), TRANSPARENT_BACKGROUND)
    assert _near(second.rgb(0, 19), TRANSPARENT_BACKGROUND)


def test_images_over_the_pixel_limit_are_not_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(payloads, "MAX_PREVIEW_INPUT_PIXELS", 99)

    with pytest.raises(ValueError, match="pixel limit"):
        payloads.sixel_rows(_data_uri(Image.new("RGB", (10, 10))), columns=1, rows=1, pixel_height=1)


def test_only_preview_formats_are_decoded() -> None:
    uri = "data:image/tiff;base64," + base64.b64encode(image_bytes("TIFF")).decode()

    with pytest.raises(UnidentifiedImageError):
        payloads.kitty_png(uri, columns=1, rows=1, cell_pixels=(10, 20))
