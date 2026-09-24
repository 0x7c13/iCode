# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared node measurement, independent of layout strategy."""

from __future__ import annotations

from rich.cells import cell_len

from .canvas import wrap_cell_text
from .model import DiagramNode, NodeShape

MAX_LABEL_CELLS = 40


def node_geometry(node: DiagramNode) -> tuple[int, int, tuple[str, ...], tuple[int, ...]]:
    """Measure the same normalized, wrapped lines that the canvas will draw."""
    if node.shape in {NodeShape.PSEUDO_START, NodeShape.PSEUDO_END}:
        return 1, 1, (node.label,), ()
    if node.shape is NodeShape.FORK_JOIN:
        return 9, 1, (), ()
    annotation = f"«{node.annotation}» " if node.annotation else ""
    if node.shape is NodeShape.DECISION:
        label = f"◇ {annotation}{node.label}"
    elif node.shape is NodeShape.HEXAGON:
        label = f"⬡ {annotation}{node.label}"
    elif node.shape is NodeShape.SLANTED:
        label = f"▱ {annotation}{node.label}"
    else:
        label = f"{annotation}{node.label}"
    lines = list(wrap_cell_text(label, MAX_LABEL_CELLS))
    breaks: list[int] = []
    for section in node.sections:
        if not section:
            continue
        breaks.append(len(lines))
        lines.append("")
        for member in section:
            lines.extend(wrap_cell_text(member, MAX_LABEL_CELLS))
    if node.notes:
        breaks.append(len(lines))
        lines.append("")
        for note in node.notes:
            lines.extend(wrap_cell_text(f"📝 {note}", MAX_LABEL_CELLS))
    horizontal_chrome = 8 if node.shape is NodeShape.SUBROUTINE else 6
    width = max(7, node.min_width, max((cell_len(line) for line in lines), default=1) + horizontal_chrome)
    return width, len(lines) + 2, tuple(lines), tuple(breaks)
