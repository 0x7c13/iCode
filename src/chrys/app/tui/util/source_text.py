# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Display-only normalization and control-character filtering for literal source text."""

from __future__ import annotations

from chrys.foundation.i18n.formatting import sanitize_terminal_block


def sanitize_source_text(source: str, *, tab_size: int = 4) -> str:
    """Preserve line breaks and indentation before replacing C0/C1 and DEL controls."""
    return sanitize_terminal_block(source).expandtabs(tab_size)
