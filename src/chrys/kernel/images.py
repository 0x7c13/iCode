# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Kernel-local image content predicates."""

from __future__ import annotations

from typing import Any

from chrys.foundation.text.images import is_image_media_type as _is_image_media_type

from ._content import Content


def is_image_media_type(media_type: Any) -> bool:
    """Return True when *media_type* is an image MIME type."""
    return _is_image_media_type(media_type)


def is_image_data_uri(uri: Any) -> bool:
    """Return True when *uri* declares an image MIME type in a data URI prefix."""
    if not isinstance(uri, str) or not uri.lower().startswith("data:"):
        return False
    prefix = uri.split(",", 1)[0]
    media_type = prefix[5:].split(";", 1)[0]
    return is_image_media_type(media_type)


def is_image_content(content: Content) -> bool:
    """Return True for image data/URI content without raising on missing media type."""
    if content.type not in ("data", "uri"):
        return False
    if is_image_media_type(content.media_type):
        return True
    if is_image_media_type(content.additional_properties.get("media_type")):
        return True
    return is_image_data_uri(content.uri)
