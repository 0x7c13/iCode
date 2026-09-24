# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared Rich rendering for todo checklists (Tasks sidebar + chat card)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.style import Style
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from chrys.foundation.models.todos import TodoItem

_MARKER_WIDTH = 2
"""Cell width of the status-marker column (marker glyph + trailing space)."""


def _todo_marker_and_body(item: TodoItem, *, in_progress_style: Style) -> tuple[Text, Text]:
    """Split one todo item into its status marker and content text."""
    if item.status == "completed":
        return Text("✓ ", style="green"), Text(item.content, style="dim strike")
    if item.status == "in_progress":
        marker = Text("▸ ", style=in_progress_style)
        return marker, Text(item.active_form or item.content, style=in_progress_style)
    return Text("☐ "), Text(item.content)


def todo_checklist_renderable(items: tuple[TodoItem, ...], *, in_progress_style: Style) -> Table:
    """Render *items* as a two-column marker + content grid.

    The grid re-wraps at whatever width it is rendered into, and wrapped
    continuation lines stay aligned under the first content character
    rather than sliding back under the status marker.

    ``in_progress_style`` is a resolved Rich style (theme variables can't be
    referenced from Rich markup, so callers resolve ``$warning`` themselves).
    """
    grid = Table.grid()
    grid.add_column(width=_MARKER_WIDTH)
    grid.add_column(overflow="fold")
    for item in items:
        marker, body = _todo_marker_and_body(item, in_progress_style=in_progress_style)
        grid.add_row(marker, body)
    return grid


def todo_done_count(items: tuple[TodoItem, ...]) -> int:
    """Number of completed items in *items*."""
    return sum(1 for item in items if item.status == "completed")
