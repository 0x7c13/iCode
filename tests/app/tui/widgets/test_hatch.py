# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the shared hatch style helpers."""

from __future__ import annotations

from rich.cells import cell_len
from rich.style import Style
from textual.geometry import Region

from chrys.app.tui.widgets.hatch import HATCH_GLYPH, HatchedEmptyState, hatched_text_line
from tests.support.tui_helpers import WidgetApp


def test_hatched_text_line_truncates_a_label_instead_of_dropping_it() -> None:
    line = hatched_text_line(8, "No trajectory data", hatch_style=Style(dim=True), label_style=Style(bold=True))

    assert cell_len(line.plain) == 8
    assert line.plain.endswith("...")
    assert line.plain != "╲" * 8


def test_hatched_empty_state_render_lines_unattached_returns_blank() -> None:
    rendered = HatchedEmptyState("Empty").render_lines(Region(0, 0, 20, 2))

    assert [strip.cell_length for strip in rendered] == [20, 20]


async def test_hatched_empty_state_centers_wide_glyph_labels_by_cell_width() -> None:
    """A CJK label is 2 cells per glyph; centring by code points shoves it right."""

    async with WidgetApp(lambda: HatchedEmptyState("没有已保存的会话。", id="hatch")).run_test(size=(40, 6)) as pilot:
        await pilot.pause()
        widget = pilot.app.query_one("#hatch", HatchedEmptyState)
        text = widget.render_line(widget.size.height // 2).text
        label = " 没有已保存的会话。 "
        start = text.index(label)
        left = text[:start].count(HATCH_GLYPH)
        right = text[start + len(label) :].count(HATCH_GLYPH)
        assert (left, right) == ((40 - 20) // 2, 40 - 20 - (40 - 20) // 2)
