# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Literal workflow source display, independent of the bytes trusted and executed."""

from __future__ import annotations

from rich.syntax import Syntax

from chrys.app.tui.util.source_text import sanitize_source_text


def workflow_source_syntax(source: bytes) -> Syntax:
    """Normalize source whitespace before replacing terminal control characters."""
    code = sanitize_source_text(source.decode("utf-8", errors="replace"))
    return Syntax(code, "python", line_numbers=True, word_wrap=False)
