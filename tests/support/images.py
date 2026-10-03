# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real image bytes, for tests whose code reads an image's signature rather than its name."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


def image_bytes(image_format: str = "PNG", *, size: tuple[int, int] = (2, 2)) -> bytes:
    """Encode a solid *size* image in Pillow's *image_format* (``PNG``, ``JPEG``, ``GIF``, ``WEBP``, ``BMP``)."""
    out = BytesIO()
    Image.new("RGB", size, (200, 40, 40)).save(out, format=image_format)
    return out.getvalue()
