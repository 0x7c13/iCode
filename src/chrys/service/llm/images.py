# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Images as the model APIs accept them, for every wire protocol's history encoder.

An image in a format the APIs don't read (BMP, SVG, HEIC, a mislabelled
file) goes out as :data:`UNSUPPORTED_IMAGE_TEXT` instead, so one bad image
doesn't fail every later request of the session. Stored history keeps the
image as it is.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from chrys.foundation.text.images import wire_image_media_type

if TYPE_CHECKING:
    from chrys.kernel import Content

UNSUPPORTED_IMAGE_TEXT: Final = "[Image omitted: unsupported or invalid image data. Use PNG, JPEG, GIF, or WebP.]"

# Base64 characters that decode to the 12 bytes every image signature fits in.
_SIGNATURE_CHARS: Final = 16


@dataclass(frozen=True, slots=True)
class WireImage:
    """An image as one request sends it."""

    media_type: str
    uri: str
    """A data URI naming :attr:`media_type`, or the image's URL."""
    data: str | None
    """The base64 payload of a data URI; None for a URL."""


def wire_image(content: Content) -> WireImage | None:
    """*content*'s image as model APIs accept it; None when they would reject it.

    The bytes of a data URI decide its type, whatever the content declares;
    only their leading bytes are read, since the payload was encoded or
    checked where it entered. An http(s) URL keeps its declared type when that
    is one the APIs read; the APIs can't fetch any other URL (``file://``).
    """
    uri = content.uri
    if not isinstance(uri, str):
        return None
    if not uri.startswith("data:"):
        if not uri.lower().startswith(("http://", "https://")):
            return None
        media_type = wire_image_media_type(None, content.media_type)
        return WireImage(media_type, uri, None) if media_type is not None else None
    header, separator, payload = uri.partition(",")
    if not separator or not header.endswith(";base64"):
        return None
    try:
        leading = base64.b64decode(payload[:_SIGNATURE_CHARS], validate=True)
    except binascii.Error, ValueError:
        return None
    if (media_type := wire_image_media_type(leading, content.media_type)) is None:
        return None
    return WireImage(media_type, f"data:{media_type};base64,{payload}", payload)
