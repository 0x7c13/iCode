# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The blank row above a fold row that follows a folded one, its guide running on under their fold arrows.

An open row's text or calls end with a blank row under its guide; a folded row is its one row. The fold row below a
folded one, or below the summary of a turn's activity, keeps a blank row above it instead, so a turn's rows stand one
blank row apart, open or folded, and the guide under their fold arrows strings them together. Nothing follows the
last of them: the margin before what comes next parts it from that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from rich.style import Style
from rich.text import Text
from textual.geometry import Offset
from textual.layout import WidgetPlacement

if TYPE_CHECKING:
    from collections.abc import Iterable

    from textual.widget import Widget

FOLD_GAP_ROWS: Final = 1
"""The rows of the gap above a fold row."""

_GAP_OFFSET = Offset(0, FOLD_GAP_ROWS)

if TYPE_CHECKING:
    _FoldGapBase = Widget
else:
    # A plain mixin at runtime, as GatedAnimation: a Widget base listed first would hide the CSS of the widget class
    # it is mixed into.
    _FoldGapBase = object


class FoldGap(_FoldGapBase):
    """Mixin for a transcript entry a gap parts from a folded row above it, or that is a folded row itself.

    The transcript shows the gaps as it lays its entries out (:func:`show_fold_gaps`); an entry with its gap
    shown places what it shows below it and draws the gap's guide under its fold arrow (:meth:`gap_text`).
    """

    gap_above = False
    """Whether its gap shows above it: the entry above it is a folded row."""

    def takes_fold_gap(self) -> bool:
        """Whether a gap parts it from a folded row above it."""
        return False

    def ends_folded(self) -> bool:
        """Whether it is a folded row: one row, with no blank row under it for a fold row below to share."""
        return False

    def show_gap_above(self, gap: bool) -> None:
        if gap is self.gap_above:
            return
        self.gap_above = gap
        # Set as the transcript arranges it, before measuring it: what it shows moved, which neither the
        # arrangement nor the box model Textual caches for it knows. Dropped as Textual drops them for an ancestor
        # of a widget laid out anew; a layout refresh would lay the whole transcript out again. Its own rows are
        # painted anew, though its size may not change: text that streams keeps the rows it took.
        self._clear_arrangement_cache()
        self._layout_updates += 1
        self.refresh()

    def process_layout(self, placements: list[WidgetPlacement]) -> list[WidgetPlacement]:
        """What it shows starts below its gap."""
        return WidgetPlacement.translate(placements, _GAP_OFFSET) if self.gap_above else placements

    def gap_text(self) -> Text:
        """The gap's rows: the guide under its fold arrow, in the colour of its own guide."""
        _edge, color = self.styles.border_left
        style = Style(color=(self.background_colors[1] + color).rich_color)
        return Text("\n".join(["│"] * FOLD_GAP_ROWS), style=style)


def _parted_by_gap(above: Widget | None, below: Widget) -> bool:
    """Whether a gap parts *below* from *above*, the entry shown before it."""
    return isinstance(below, FoldGap) and below.takes_fold_gap() and isinstance(above, FoldGap) and above.ends_folded()


def show_fold_gaps(entries: Iterable[Widget]) -> None:
    """Show the gap of each of the *entries* shown, in order, that follows a folded row; hide the others'.

    A transcript calls it as it lays its entries out (``pre_layout``): an entry folding, opening, showing, hiding,
    mounting or going lays them out again, and only that does, so the gaps follow without a pass of their own.
    """
    above: Widget | None = None
    for entry in entries:
        if isinstance(entry, FoldGap):
            entry.show_gap_above(_parted_by_gap(above, entry))
        above = entry
