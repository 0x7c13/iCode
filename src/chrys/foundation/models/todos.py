# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Todo-list data model shared by service, orchestration, TUI and ACP layers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final, Literal, TypeIs, get_args

logger = logging.getLogger(__name__)

TodoStatus = Literal["pending", "in_progress", "completed"]

TODO_STATUSES: Final[frozenset[str]] = frozenset(get_args(TodoStatus))

MAX_TODO_ITEMS: Final[int] = 128
MAX_TODO_TEXT_LENGTH: Final[int] = 500
"""Per-field cap for ``content`` and ``active_form`` (both persisted and rendered)."""


def _is_todo_status(value: object) -> TypeIs[TodoStatus]:
    """Return whether *value* is a supported todo status."""
    return isinstance(value, str) and value in TODO_STATUSES


@dataclass(frozen=True, slots=True)
class TodoItem:
    """One entry of the session todo list."""

    content: str
    """Imperative description, e.g. ``"Run the test suite"``."""

    status: TodoStatus = "pending"

    active_form: str = ""
    """Optional present-continuous label shown while in progress, e.g. ``"Running tests"``."""


def parse_todo_items(raw: object) -> tuple[TodoItem, ...]:
    """Tolerantly decode persisted ``chrys_todos`` payloads into :class:`TodoItem` tuples.

    The ONE raw→``TodoItem`` decoder: both ``TodoTracker.restore()`` and the
    TUI session reseed go through it, so malformed or legacy state degrades to
    skipped items instead of crashing or desyncing the sidebar. Non-list input
    yields ``()``; per-item bad shape is skipped with a log; never raises.
    """
    if not isinstance(raw, list):
        if raw is not None:
            logger.warning("Ignoring non-list todo payload of type %s", type(raw).__name__)
        return ()
    items: list[TodoItem] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict todo entry of type %s", type(entry).__name__)
            continue
        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            logger.warning("Skipping todo entry without usable content: %r", entry)
            continue
        status = entry.get("status", "pending")
        if not _is_todo_status(status):
            logger.warning("Skipping todo entry with unknown status %r", status)
            continue
        active_form = entry.get("active_form", "")
        if not isinstance(active_form, str):
            active_form = ""
        items.append(TodoItem(content=content, status=status, active_form=active_form))
    return tuple(items)


def serialize_todo_items(items: tuple[TodoItem, ...]) -> list[dict[str, str]]:
    """Encode *items* as the plain-dict ``chrys_todos`` payload."""
    return [{"content": item.content, "status": item.status, "active_form": item.active_form} for item in items]
