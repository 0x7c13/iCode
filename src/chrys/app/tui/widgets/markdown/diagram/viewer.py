# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Two-axis terminal viewport for a compiled diagram."""

from __future__ import annotations

from collections.abc import Mapping

from textual.geometry import Offset, Size
from textual.scroll_view import ScrollView
from textual.strip import Strip

from .canvas import CellStyleSpan, styled_cell_segments
from .model import CompiledDiagram


class DiagramViewport(ScrollView, can_focus=True):
    """Virtualized, cell-aware surface for one compiled terminal diagram."""

    DEFAULT_CSS = """
    DiagramViewport {
        width: 100%;
        height: 1fr;
        overflow: auto auto;
        scrollbar-size: 1 1;
    }
    """

    def __init__(
        self,
        diagram: CompiledDiagram,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        center_diagram: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.center_diagram = center_diagram
        self.diagram = diagram
        self.cell_styles: dict[int, tuple[CellStyleSpan, ...]] = {}
        self.virtual_size = Size(diagram.width, diagram.height)

    def update_styles(self, styles: Mapping[int, tuple[CellStyleSpan, ...]]) -> None:
        """Repaint only: retain the canvas, geometry and both scroll offsets."""
        self.cell_styles = dict(styles)
        self.refresh()

    def set_diagram(self, diagram: CompiledDiagram) -> None:
        """Replace the displayed cell canvas and reset its scroll position."""
        self.diagram = diagram
        self.cell_styles.clear()
        self.virtual_size = Size(diagram.width, diagram.height)
        self.scroll_to(x=0, y=0, animate=False, force=True, immediate=True)
        self.refresh(layout=True)

    @property
    def diagram_origin(self) -> Offset:
        """Center a fitting canvas; larger diagrams retain their natural scroll origin."""
        if not self.center_diagram:
            return Offset(0, 0)
        region = self.scrollable_content_region
        return Offset(
            max(0, (region.width - self.diagram.width) // 2), max(0, (region.height - self.diagram.height) // 2)
        )

    def render_diagram_row(
        self, row: str, content_y: int, *, overlays: tuple[CellStyleSpan, ...] = (), row_start: int = 0
    ) -> Strip:
        width = self.scrollable_content_region.width
        style = self.visual_style.rich_style
        scroll_x = round(self.scroll_offset.x)
        left = self.diagram_origin.x
        spans = tuple(
            CellStyleSpan(span.start - row_start, span.end - row_start, span.style)
            for span in (*self.cell_styles.get(content_y, ()), *overlays)
            if span.end > scroll_x and span.start < scroll_x + width - left
        )
        segments = styled_cell_segments(row, spans, scroll_x - row_start, width - left, style)
        content = Strip(segments, width - left)
        return Strip.join((Strip.blank(left, style), content)).apply_offsets(scroll_x - left, content_y)

    def render_line(self, y: int) -> Strip:
        """Render one visible row from the diagram's natural cell canvas."""
        content_y = y + round(self.scroll_offset.y) - self.diagram_origin.y
        if not 0 <= content_y < self.diagram.height:
            return Strip.blank(self.scrollable_content_region.width, self.visual_style.rich_style)
        return self.render_diagram_row(self.diagram.rows[content_y], content_y)


__all__ = ["DiagramViewport"]
