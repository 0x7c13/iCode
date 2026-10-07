# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Conversions between editor (row, column) locations and text offsets."""

from __future__ import annotations

from chrys.app.tui.widgets.editor.types import EditorLocation


def _lines(text: str) -> list[str]:
    return text.split("\n")


def location_to_offset(text: str, location: EditorLocation) -> int:
    """The offset of *location* in *text*, clamped to the text."""
    lines = _lines(text)
    row = min(max(location[0], 0), len(lines) - 1)
    column = min(max(location[1], 0), len(lines[row]))
    return sum(len(line) + 1 for line in lines[:row]) + column


def offset_to_location(text: str, offset: int) -> EditorLocation:
    """The location of *offset* in *text*, clamped to the text."""
    lines = _lines(text)
    remaining = min(max(offset, 0), len(text))
    for row, line in enumerate(lines[:-1]):
        if remaining <= len(line):
            return row, remaining
        remaining -= len(line) + 1
    return len(lines) - 1, min(remaining, len(lines[-1]))
