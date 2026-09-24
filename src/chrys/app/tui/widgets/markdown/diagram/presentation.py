# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Localized headings for graph-level titles and explicitly simplified layouts."""

from __future__ import annotations

from collections.abc import Callable

from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

from .canvas import wrap_cell_text
from .model import DiagramIR, DiagramKind

_ARCHITECTURE_LAYOUT = msg(
    "tui.diagram.presentation.architecture_layout",
    fallback="Schematic layout: groups and ports are shown as labels, not spatial constraints.",
)
_SANKEY_LAYOUT = msg(
    "tui.diagram.presentation.sankey_layout",
    fallback="Weighted flow diagram: values are preserved; line widths are not proportional.",
)


def graph_heading(ir: DiagramIR, renderer: Callable[[MessageRef], str] | None) -> tuple[str, ...]:
    """Return safe wrapped display lines, leaving source data untouched."""
    lines = list(wrap_cell_text(ir.title, 80)) if ir.title.strip() else []
    if ir.simplified:
        definition = _ARCHITECTURE_LAYOUT if ir.kind is DiagramKind.ARCHITECTURE else _SANKEY_LAYOUT
        text = (renderer or format_message)(definition.bind())
        lines.extend(wrap_cell_text(text, 80))
    return (*lines, "") if lines else ()
