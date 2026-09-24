# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pure table-of-contents data types shared by chat and sidebar widgets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TocItem:
    """TOC item data structure."""

    turn_id: str
    summary: str
    children: list[TocItem] = field(default_factory=list)
    compressed: bool = False
    turn_index: int | None = None


_SUMMARY_MAX_LENGTH = 120


def summarize_prompt(text: str, max_length: int = _SUMMARY_MAX_LENGTH) -> str:
    if not text:
        return "(empty)"
    first_line = text.split("\n", 1)[0]
    if len(first_line) > max_length:
        return first_line[: max_length - 3] + "..."
    return first_line
