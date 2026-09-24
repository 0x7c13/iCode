# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded text preview helpers for file mutation renderers."""

from __future__ import annotations

WRITE_FILE_PREVIEW_MAX_LINE_CHARS = 240

_UNSAFE_DISPLAY_CONTROL_TRANSLATION = {i: "?" for i in range(32) if chr(i) not in {"\t", "\n"}}
_UNSAFE_DISPLAY_CONTROL_TRANSLATION[0x7F] = "?"


def sanitize_preview_line(line: str) -> str:
    """Return a terminal-safe line for bounded previews only."""
    return line.translate(_UNSAFE_DISPLAY_CONTROL_TRANSLATION)


def bounded_preview_text(text: str, *, max_lines: int, max_line_chars: int) -> str:
    """Build a bounded preview without scanning or splitting the full snapshot."""
    if not text or max_lines <= 0 or max_line_chars <= 0:
        return ""

    lines: list[str] = []
    pos = 0
    text_len = len(text)
    while pos < text_len and len(lines) < max_lines:
        scan_end = min(text_len, pos + max_line_chars + 1)
        newline = text.find("\n", pos, scan_end)
        if newline >= 0:
            line = text[pos:newline]
            line = line.removesuffix("\r")
            lines.append(sanitize_preview_line(line))
            pos = newline + 1
            continue

        chunk = text[pos:scan_end]
        if scan_end >= text_len:
            truncated = len(chunk) > max_line_chars
            line = chunk[:max_line_chars]
            line = line.removesuffix("\r")
            if truncated:
                line += "..."
            lines.append(sanitize_preview_line(line))
            break

        line = text[pos : pos + max_line_chars]
        line = line.removesuffix("\r")
        lines.append(sanitize_preview_line(line) + "...")
        break

    return "\n".join(lines)


def write_file_preview_text(
    text: str, *, max_lines: int, max_line_chars: int = WRITE_FILE_PREVIEW_MAX_LINE_CHARS
) -> str:
    """Build the bounded write-file display preview."""
    return bounded_preview_text(text, max_lines=max_lines, max_line_chars=max_line_chars)
