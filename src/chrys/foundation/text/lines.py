# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Line splitting for the tools that show line numbers or take them back."""

from __future__ import annotations


def split_lines(text: str) -> list[str]:
    """Split LF-only *text* into its lines, numbered as editors and ``grep -n`` number them.

    ``str.splitlines`` also breaks at vertical tabs, form feeds, ``\\x1c`` to ``\\x1e``,
    ``\\x85`` and U+2028/U+2029 (a PowerPoint soft line break is a vertical tab),
    which would shift every later line number away from the file's. A trailing
    newline ends the last line and opens no new one.
    """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def normalize_line_endings(text: str) -> str:
    """Turn CRLF and lone CR line endings into LF, as Python's text-mode reads do."""
    return text.replace("\r\n", "\n").replace("\r", "\n")
