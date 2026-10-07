# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Text layout shared by the profile pickers."""

from __future__ import annotations

import textwrap

_DESC_INDENT = "    "
"""Indent for description lines (matches width of '  - ')."""


def wrap_description(desc: str, width: int) -> str:
    """Wrap description text so continuation lines align after '  - '."""
    if not desc:
        return ""
    indent = _DESC_INDENT
    first_prefix = "  - "
    # Available width for text on the first and subsequent lines
    text_width = max(width - len(indent), 20)
    lines = textwrap.wrap(desc, width=text_width)
    if not lines:
        return ""
    result = first_prefix + lines[0]
    for line in lines[1:]:
        result += "\n" + indent + line
    return result
