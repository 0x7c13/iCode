# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reminder rendering for the session todo list."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from chrys.foundation.models.todos import TodoItem

_STATUS_MARKS: Final[dict[str, str]] = {
    "completed": "x",
    "in_progress": ">",
    "pending": " ",
}


def format_todo_reminder(items: tuple[TodoItem, ...]) -> str | None:
    """Render *items* as the per-turn reminder text, or ``None`` when empty.

    Empty means cleared-on-purpose or never used — no "list is empty" nudge
    (completed items stay in the list, so finished work still renders).
    """
    if not items:
        return None
    lines = ["Current todo list (todo_write to update):"]
    lines.extend(f"- [{_STATUS_MARKS.get(item.status, ' ')}] {item.content}" for item in items)
    return "\n".join(lines)
