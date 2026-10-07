# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The image formats model APIs read, and the ones Pillow may decode."""

from __future__ import annotations

import pytest

from chrys.foundation.text.images import (
    IMAGE_DECODE_FORMATS,
    ImageProcessingError,
    compress_image_data,
    inspect_image_dimensions,
    wire_image_media_type,
)
from tests.support.images import image_bytes


@pytest.mark.parametrize(
    ("image_format", "declared", "expected"),
    [
        ("PNG", "image/png", "image/png"),
        ("JPEG", "image/png", "image/jpeg"),
        ("GIF", None, "image/gif"),
        ("WEBP", "image/jpeg", "image/webp"),
        ("BMP", "image/png", None),
    ],
)
def test_bytes_name_the_type_whatever_is_declared(
    image_format: str, declared: str | None, expected: str | None
) -> None:
    assert wire_image_media_type(image_bytes(image_format), declared) == expected


def test_bytes_without_a_signature_are_no_image_whatever_is_declared() -> None:
    assert wire_image_media_type(b"image-bytes", "image/png") is None


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("image/png", "image/png"),
        ("IMAGE/JPEG; charset=binary", "image/jpeg"),
        ("image/jpg", "image/jpeg"),
        ("image/bmp", None),
        ("image/svg+xml", None),
        (None, None),
    ],
)
def test_without_bytes_the_declared_type_decides(declared: str | None, expected: str | None) -> None:
    assert wire_image_media_type(None, declared) == expected


@pytest.mark.parametrize("image_format", IMAGE_DECODE_FORMATS)
def test_whitelisted_formats_decode_and_convert(image_format: str) -> None:
    data = image_bytes(image_format, size=(3, 2))

    assert inspect_image_dimensions(data) == (3, 2)
    assert compress_image_data(data).startswith(b"\xff\xd8\xff")


@pytest.mark.parametrize("image_format", ["TIFF", "PPM", "PCX", "SGI", "DDS", "JPEG2000"])
def test_formats_outside_the_whitelist_are_never_decoded(image_format: str) -> None:
    """Pillow reads every one of these; behind an image name they are still refused."""
    data = image_bytes(image_format)

    with pytest.raises(ImageProcessingError, match="could not be read as a supported image"):
        inspect_image_dimensions(data)
    with pytest.raises(ImageProcessingError, match="could not be read as a supported image"):
        compress_image_data(data)


def test_a_caller_names_any_wider_format_set_it_decodes() -> None:
    data = image_bytes("TIFF", size=(3, 2))
    formats = (*IMAGE_DECODE_FORMATS, "TIFF")

    assert inspect_image_dimensions(data, formats=formats) == (3, 2)
    assert compress_image_data(data, formats=formats).startswith(b"\xff\xd8\xff")
