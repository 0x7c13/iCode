# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Golden render tests for :class:`DiffView`.

A diff is drawn three ways: a unified one by a single widget, a unified one as gutter columns
around a code column, and a split one as two such groups. These tests pin the text of every row
in each, and that a row comes out the same whether a repaint or a lone ``render_line`` asks for it.
"""

from __future__ import annotations

import pytest
from rich.color import ColorSystem
from textual.app import App, ComposeResult
from textual.geometry import Region
from textual.strip import Strip

from chrys.app.tui.widgets import HATCH_GLYPH
from chrys.app.tui.widgets.diff_view import CodeColumn, DiffView, GutterColumn
from chrys.app.tui.widgets.diff_view.unified import UnifiedDiffLines

_BEFORE = "alpha\nbeta\n"
_AFTER = "alpha\nBETA\ngamma\n"


def _row_text(strip: Strip) -> str:
    return "".join(segment.text for segment in strip).rstrip()


def _segments(strips: list[Strip]) -> list[tuple[int, list[tuple[str, object]]]]:
    return [(strip.cell_length, [(segment.text, segment.style) for segment in strip]) for strip in strips]


def _column_rows(column: CodeColumn | GutterColumn, row_count: int) -> list[str]:
    return [_row_text(strip) for strip in column.render_lines(Region(0, 0, column.size.width, row_count))]


class _DiffApp(App):
    def __init__(self, before: str, after: str, *, split: bool, scrollbars: bool) -> None:
        super().__init__()
        self._before = before
        self._after = after
        self._split = split
        self._scrollbars = scrollbars

    def compose(self) -> ComposeResult:
        diff = DiffView("file.py", "file.py", self._before, self._after)
        diff.split = self._split
        diff.auto_height = True
        diff.show_scrollbars = self._scrollbars
        yield diff


@pytest.mark.asyncio
async def test_flat_unified_diff_rows() -> None:
    async with _DiffApp(_BEFORE, _AFTER, split=False, scrollbars=False).run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        flat = pilot.app.query_one(UnifiedDiffLines)

        strips = flat.render_lines(Region(0, 0, flat.size.width, flat.size.height))

    assert [_row_text(strip) for strip in strips] == [
        "▎1  1    alpha",
        "▎2     - beta",
        "▎   2  + BETA",
        "▎   3  + gamma",
    ]
    assert {strip.cell_length for strip in strips} == {80}


@pytest.mark.asyncio
async def test_unified_columns_draw_the_rows_of_the_flat_widget() -> None:
    async with _DiffApp(_BEFORE, _AFTER, split=False, scrollbars=True).run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        code = pilot.app.query_one(CodeColumn)
        numbers_before, numbers_after, annotations = pilot.app.query(GutterColumn)
        row_count = len(code.rows)

        columns = [
            [strip.text for strip in column.render_lines(Region(0, 0, column.size.width, row_count))]
            for column in (numbers_before, numbers_after, annotations, code)
        ]

    assert ["".join(cells).rstrip() for cells in zip(*columns, strict=True)] == [
        "▎1  1    alpha",
        "▎2     - beta",
        "▎   2  + BETA",
        "▎   3  + gamma",
    ]


@pytest.mark.asyncio
async def test_split_diff_rows_pair_removed_with_added_and_fill_the_rest() -> None:
    async with _DiffApp(_BEFORE, _AFTER, split=True, scrollbars=True).run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        left_code, right_code = pilot.app.query(CodeColumn)
        left_numbers, left_annotations, right_numbers, right_annotations = pilot.app.query(GutterColumn)

        assert left_code.scroll_sync is right_code
        assert right_code.scroll_sync is left_code
        left_width = left_code.scrollable_content_region.width
        left = [_column_rows(column, 3) for column in (left_numbers, left_annotations, left_code)]
        right = [_column_rows(column, 3) for column in (right_numbers, right_annotations, right_code)]

    assert left == [
        ["▎1", "▎2", HATCH_GLYPH * 3],
        ["", " -", HATCH_GLYPH * 3],
        ["alpha", "beta", HATCH_GLYPH * left_width],
    ]
    assert right == [
        ["▎1", "▎2", "▎3"],
        ["", " +", " +"],
        ["alpha", "BETA", "gamma"],
    ]


@pytest.mark.asyncio
async def test_break_between_hunks_is_one_row_hatched_from_edge_to_edge() -> None:
    before_lines = [f"line {index}" for index in range(30)]
    after_lines = before_lines.copy()
    after_lines[0] = "changed 0"
    after_lines[29] = "changed 29"
    before = "\n".join(before_lines) + "\n"
    after = "\n".join(after_lines) + "\n"

    async with _DiffApp(before, after, split=False, scrollbars=False).run_test(size=(41, 20)) as pilot:
        await pilot.pause()
        flat = pilot.app.query_one(UnifiedDiffLines)

        rows = [strip.text for strip in flat.render_lines(Region(0, 0, flat.size.width, flat.size.height))]

    # "-" and "+" of the first line, three lines of context, the break, three more, "-" and "+" of the last.
    assert len(rows) == 11
    assert rows[5] == HATCH_GLYPH * 41
    assert [row for row in rows if HATCH_GLYPH in row] == [rows[5]]


@pytest.mark.asyncio
@pytest.mark.parametrize("split", [False, True])
async def test_break_is_hatched_in_every_column_of_a_scrolling_diff(split: bool) -> None:
    before_lines = [f"line {index}" for index in range(30)]
    after_lines = before_lines.copy()
    after_lines[0] = "changed 0"
    after_lines[29] = "changed 29"
    before = "\n".join(before_lines) + "\n"
    after = "\n".join(after_lines) + "\n"

    async with _DiffApp(before, after, split=split, scrollbars=True).run_test(size=(100, 20)) as pilot:
        await pilot.pause()
        code_columns = list(pilot.app.query(CodeColumn))
        break_index = next(index for index, row in enumerate(code_columns[0].rows) if row.is_placeholder)
        columns = [*pilot.app.query(GutterColumn), *code_columns]

        cells = [_column_rows(column, break_index + 1)[break_index] for column in columns]
        widths = [
            column.scrollable_content_region.width if isinstance(column, CodeColumn) else column.size.width
            for column in columns
        ]

    assert len(columns) == (6 if split else 4)
    assert cells == [HATCH_GLYPH * width for width in widths]


@pytest.mark.asyncio
async def test_unified_diff_render_is_deterministic_across_repeats() -> None:
    async with _DiffApp(_BEFORE, _AFTER, split=False, scrollbars=False).run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        flat = pilot.app.query_one(UnifiedDiffLines)
        crop = Region(0, 0, flat.size.width, flat.size.height)

        first = flat.render_lines(crop)
        flat._invalidate_render_cache()
        second = flat.render_lines(crop)

    assert second is not first
    assert _segments(first) == _segments(second)


@pytest.mark.asyncio
@pytest.mark.parametrize("split", [False, True])
async def test_render_line_outside_a_repaint_draws_the_row_of_the_repaint(split: bool) -> None:
    """A lone ``render_line`` measures for itself what a repaint measures once for all its rows.

    The 256-color backgrounds are part of that: forced here, because they are what a row drawn
    from stale measurements would get wrong.
    """
    async with _DiffApp("same\n", "added\n", split=split, scrollbars=split).run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        pilot.app.console._color_system = ColorSystem.EIGHT_BIT
        widget = pilot.app.query(CodeColumn).last() if split else pilot.app.query_one(UnifiedDiffLines)
        row_count = len(widget.rows)

        widget.render_lines(Region(0, 0, widget.size.width, row_count))
        during_repaint = [widget.render_line(y) for y in range(row_count)]
        widget._invalidate_render_cache()
        standalone = [widget.render_line(y) for y in range(row_count)]

    # Split, the removed line and the added one share a row.
    assert row_count == (1 if split else 2)
    assert _segments(during_repaint) == _segments(standalone)
    assert any(
        segment.style is not None and segment.style.bgcolor is not None and segment.style.bgcolor.name == "#005f00"
        for segment in standalone[-1]
    )
