# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The image formats model APIs read."""

from __future__ import annotations

import pytest

from chrys.foundation.text.images import wire_image_media_type
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
