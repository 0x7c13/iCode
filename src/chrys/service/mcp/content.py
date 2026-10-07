# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Media an MCP server sends as base64.

Data that does not decode becomes a placeholder text item in place of the
media, so the rest of the result still reaches the model.
"""

from __future__ import annotations

import base64
import binascii
from typing import Final

INVALID_IMAGE_TEXT: Final = "[Image omitted: invalid base64 data.]"
INVALID_AUDIO_TEXT: Final = "[Audio omitted: invalid base64 data.]"
INVALID_RESOURCE_TEXT: Final = "[Resource omitted: invalid base64 data.]"


def decode_media_base64(value: str) -> bytes | None:
    """The bytes of *value*, bare base64 or a ``data:…;base64,`` URI; None when it is neither.

    Whitespace in the payload is ignored: servers may send line-wrapped base64.
    """
    if value.startswith("data:"):
        header, separator, value = value.partition(",")
        if not separator or not header.endswith(";base64"):
            return None
    try:
        return base64.b64decode("".join(value.split()), validate=True)
    except binascii.Error, ValueError:
        return None
