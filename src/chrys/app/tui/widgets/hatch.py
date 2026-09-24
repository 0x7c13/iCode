# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hatched placeholder widgets shared by TUI empty states."""

from __future__ import annotations

from typing import ClassVar

from rich.cells import cell_len, set_cell_size
from rich.style import Style
from rich.text import Text
from textual.content import Content
from textual.geometry import Region
from textual.strip import Strip
from textual.style import Style as VisualStyle
from textual.widget import Widget

HATCH_GLYPH = "╲"


def _fit_label(label: str, width: int) -> str:
    if width <= 2:
        return " " * width
    padded = f" {label} "
    if cell_len(padded) <= width:
        return padded
    if width <= 3:
        return "." * width
    return f"{set_cell_size(label, width - 3)}..."


def hatched_content_line(
    width: int, label: str | None = None, *, pattern_style: VisualStyle, label_style: str = "$text-muted"
) -> Content:
    """One full-width hatch row, optionally carrying a centered label."""
    if width <= 0:
        return Content("")
    if not label:
        return Content.styled(HATCH_GLYPH * width, pattern_style)
    fitted = _fit_label(label, width)
    label_width = cell_len(fitted)
    start = max(0, (width - label_width) // 2)
    end = start + label_width
    return (
        Content.styled(HATCH_GLYPH * start, pattern_style)
        + Content.styled(fitted, label_style)
        + Content.styled(HATCH_GLYPH * max(0, width - end), pattern_style)
    )


def hatch_style(widget: Widget) -> VisualStyle:
    """Resolve composite variables through CSS, also for Content renderers."""
    return VisualStyle.from_rich_style(hatch_text_style(widget))


def hatch_text_style(widget: Widget) -> Style:
    """Use CSS's resolved alpha/ANSI color in Rich-rendered empty states too.

    Callers declare the ``hatch--pattern`` component. Discard its stamped
    background so the glyph does not paint over the caller's own surface.
    """
    return Style.from_color(widget.get_component_rich_style("hatch--pattern").color)


def hatched_text_line(width: int, label: str | None = None, *, hatch_style: Style, label_style: Style) -> Text:
    """One full-width Rich hatch row, optionally carrying a centered, padded label.

    The Rich twin of :func:`hatched_content_line` for widgets that render
    ``Text`` lines through the app console; labels are fitted to the row so
    narrow layouts preserve as much of the empty-state message as possible.

    Every style is carried as a span: ``Text.render()`` yields unstyled
    segments for a span-less ``Text`` and ignores the ``Text``'s own style,
    so a line-renderer that does not re-apply ``line.style`` would otherwise
    paint the hatch in the widget's foreground colour.
    """
    if width <= 0:
        return Text()
    if label:
        fitted = _fit_label(label, width)
        label_width = cell_len(fitted)
        start = (width - label_width) // 2
        return Text.assemble(
            (HATCH_GLYPH * start, hatch_style),
            (fitted, label_style),
            (HATCH_GLYPH * (width - start - label_width), hatch_style),
        )
    return Text.assemble((HATCH_GLYPH * width, hatch_style))


class HatchedEmptyState(Widget):
    """Full-width empty-state label rendered over a hatch pattern."""

    COMPONENT_CLASSES: ClassVar[set[str]] = {"hatch--pattern"}

    DEFAULT_CSS = """
    HatchedEmptyState > .hatch--pattern {
        color: $hatch-color;
    }
    HatchedEmptyState {
        width: 1fr;
        height: 5;
    }
    """

    def __init__(
        self,
        label: str,
        *,
        label_style: str = "$text-muted",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.label = label
        self.label_style = label_style

    def update_label(self, label: str) -> None:
        """Replace the centered label and redraw the hatch row."""
        self.label = label
        self.refresh()

    def render_lines(self, crop: Region) -> list[Strip]:
        if not self.is_attached:
            return [Strip.blank(crop.width) for _ in crop.line_range]
        return super().render_lines(crop)

    def render_line(self, y: int) -> Strip:
        width = self.size.width
        if width <= 0:
            return Strip.blank(0, self.visual_style.rich_style)

        label = self.label if y == self.size.height // 2 else None
        line = hatched_content_line(width, label, pattern_style=hatch_style(self), label_style=self.label_style)
        visual_style = self.visual_style
        strip = Strip(line.render_segments(visual_style), cell_length=line.cell_length)
        return strip.adjust_cell_length(width, visual_style.rich_style)
