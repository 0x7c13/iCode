# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A column of cells beside the code: line numbers, or the ``+`` and ``-`` of changed lines."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, ClassVar, Self

from textual.cache import LRUCache
from textual.geometry import Region, Size
from textual.strip import Strip
from textual.widget import Widget

from chrys.app.tui.support.gc_freeze import detach_lru_cache, renew_lru_cache
from chrys.app.tui.widgets.diff_view.palette import DiffLook

if TYPE_CHECKING:
    from textual.content import Content

    from chrys.app.tui.widgets.diff_view.code import CodeColumn
    from chrys.app.tui.widgets.diff_view.rows import DiffRow

type GutterCell = Callable[[DiffRow, DiffLook], Content]
"""Draws what a gutter shows for one row. See `chrys.app.tui.widgets.diff_view.cells`."""


class GutterColumn(Widget):
    """One cell per row, in step with a `CodeColumn`.

    The gutter does not scroll. It reads the position of its code column and draws the cells of
    the rows that column shows, and the code column refreshes it when the position changes.
    """

    COMPONENT_CLASSES: ClassVar[set[str]] = {"hatch--pattern"}

    DEFAULT_CSS = """
    GutterColumn > .hatch--pattern {
        color: $hatch-color;
    }
    GutterColumn {
        width: auto;
        height: 1fr;
    }
    """

    def __init__(
        self,
        rows: Sequence[DiffRow],
        cell: GutterCell,
        cell_width: int,
        code_column: CodeColumn,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.rows = rows
        self.code_column = code_column
        self._cell = cell
        self._cell_width = cell_width
        self._strip_cache: LRUCache[int, Strip] = LRUCache(maxsize=500)
        self._look_key: tuple[str, str | None] | None = None
        self._look: DiffLook | None = None
        self._painted: tuple[tuple[int, int, Region], list[Strip]] | None = None

    # -- caches ------------------------------------------------------------------------------------

    def prepare_for_gc_freeze(self) -> None:
        """Detach the cyclic strip LRU before the permanent generation changes."""
        # gc-freeze swaps in an acyclic capacity token; renew restores a real cache before next use.
        self._strip_cache = detach_lru_cache(self._strip_cache)  # ty: ignore[invalid-assignment]
        self._painted = None

    def after_gc_freeze(self) -> None:
        """Recreate the strip LRU after the permanent generation changes."""
        self._strip_cache = renew_lru_cache(self._strip_cache)

    def abort_gc_freeze(self) -> None:
        """Restore the strip LRU after an incomplete hook pass."""
        self._strip_cache = renew_lru_cache(self._strip_cache)

    def _invalidate_render_cache(self) -> None:
        self._strip_cache.clear()
        self._look_key = None
        self._look = None
        self._painted = None

    def notify_style_update(self) -> None:
        """Forget rendered cells when the theme or a CSS variable changes."""
        super().notify_style_update()
        self._invalidate_render_cache()
        self.refresh()

    def refresh(self, *regions: Region, repaint: bool = True, layout: bool = False, recompose: bool = False) -> Self:
        """Whoever asks for a refresh knows of a change that the scroll position does not show."""
        self._painted = None
        return super().refresh(*regions, repaint=repaint, layout=layout, recompose=recompose)

    # -- rendering ---------------------------------------------------------------------------------

    def get_content_width(self, container: Size, viewport: Size) -> int:
        return self._cell_width

    def _window(self) -> tuple[int, int]:
        """The first row on display, and how many: never more than the code column has lines for.

        The code column may be the shorter of the two by the scrollbar under it. Cells drawn
        beside that scrollbar would belong to rows the column is not showing.
        """
        look_key = (self.app.theme, self.app.console.color_system)
        if look_key != self._look_key:
            self._invalidate_render_cache()
            self._look_key = look_key
        code_column = self.code_column
        return round(code_column.scroll_offset.y), code_column.scrollable_content_region.height

    def render_lines(self, crop: Region) -> list[Strip]:
        if not self.is_attached:
            return [Strip.blank(crop.width) for _ in crop.line_range]
        first_row, height = self._window()
        key = (first_row, height, crop)
        if self._painted is not None and self._painted[0] == key:
            return self._painted[1]

        width = self._cell_width
        line_filters = self.get_line_filters()
        _base_background, background = self.background_colors
        strips: list[Strip] = []
        for y in crop.line_range:
            strip = self._render_cell(first_row + y) if y < height else self._blank()
            for line_filter in line_filters:
                strip = strip.apply_filter(line_filter, background)
            if crop.column_span != (0, width):
                strip = strip.crop(crop.x, crop.x + crop.width)
            strips.append(strip)
        self._painted = (key, strips)
        return strips

    def render_line(self, y: int) -> Strip:
        if not self.is_attached:
            return Strip.blank(self._cell_width)
        first_row, height = self._window()
        return self._render_cell(first_row + y) if y < height else self._blank()

    def _blank(self) -> Strip:
        return Strip.blank(self._cell_width, self.visual_style.rich_style)

    def _render_cell(self, index: int) -> Strip:
        if not 0 <= index < len(self.rows):
            return self._blank()
        strip = self._strip_cache.get(index)
        if strip is None:
            if self._look is None:
                self._look = DiffLook.of(self)
            visual_style = self.visual_style
            cell = self._cell(self.rows[index], self._look)
            strip = Strip(cell.render_segments(visual_style), cell_length=cell.cell_length)
            strip = self._strip_cache[index] = strip.adjust_cell_length(self._cell_width, visual_style.rich_style)
        return strip
