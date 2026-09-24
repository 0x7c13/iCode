# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for in-house markdown selection styling."""

from __future__ import annotations

import pytest
from rich.color import ColorSystem
from rich.style import Style as RichStyle
from textual.app import App, ComposeResult
from textual.color import Color
from textual.geometry import Offset, Region
from textual.selection import SELECT_ALL, Selection
from textual.style import Style as TextualStyle
from textual.widget import Widget

from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown, _normalize_selection_rich_style


class _MarkdownSelectionApp(App):
    CSS = """
    VirtualizedMarkdown {
        width: 40;
        height: 3;
        padding: 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield VirtualizedMarkdown("abcdefghijklmnopqrstuvwxyz")


class _CopySelectionApp(App):
    CSS = """
    VirtualizedMarkdown {
        width: 18;
        height: 12;
        padding: 0;
    }
    """

    def __init__(self, markdown: str) -> None:
        super().__init__()
        self._markdown = markdown

    def compose(self) -> ComposeResult:
        yield VirtualizedMarkdown(self._markdown)


class _NarrowCopySelectionApp(App):
    CSS = """
    VirtualizedMarkdown {
        width: 4;
        height: 8;
        padding: 0;
    }
    """

    def __init__(self, markdown: str) -> None:
        super().__init__()
        self._markdown = markdown

    def compose(self) -> ComposeResult:
        yield VirtualizedMarkdown(self._markdown)


def _selection_style(background: str, foreground: str) -> RichStyle:
    return TextualStyle(Color.parse(background), Color.parse(foreground)).rich_style


def test_markdown_selection_preserves_text_color_for_transparent_foreground() -> None:
    """Textual's default transparent selection foreground resolves to the selection background."""
    selection_style = _selection_style("#0178D47F", "transparent")
    base_style = RichStyle(color="#E0E0E0")

    assert selection_style.color == selection_style.bgcolor

    highlighted = base_style + _normalize_selection_rich_style(selection_style)

    assert highlighted.bgcolor == selection_style.bgcolor
    assert highlighted.color == base_style.color


def test_markdown_selection_keeps_explicit_selection_foreground() -> None:
    selection_style = _selection_style("#0178D47F", "#F8F8F2")
    base_style = RichStyle(color="#E0E0E0")

    highlighted = base_style + _normalize_selection_rich_style(selection_style)

    assert highlighted.bgcolor == selection_style.bgcolor
    assert highlighted.color == selection_style.color


@pytest.mark.asyncio
async def test_markdown_same_row_selection_hit_testing_keeps_rendered_offsets() -> None:
    """Selection highlighting must not corrupt Textual's offset metadata."""
    app = _MarkdownSelectionApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)
        app.screen.selections = {markdown: Selection(Offset(5, 0), Offset(11, 0))}
        await pilot.pause()

        widget, offset = app.screen.get_widget_and_offset_at(markdown.region.x + 12, markdown.region.y)

        assert widget is markdown
        assert offset == Offset(12, 0)


@pytest.mark.asyncio
async def test_markdown_copy_selection_collapses_soft_wrapped_lines() -> None:
    text = "This paragraph wraps across several visual rows when copied."
    app = _CopySelectionApp(text)
    async with app.run_test() as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        selected = markdown.get_selection(SELECT_ALL)

        assert markdown._total_lines > 1
        assert selected == (text, "\n")


@pytest.mark.asyncio
async def test_markdown_copy_selection_drops_soft_wrap_list_continuation_indent() -> None:
    text = "- 通用 Agent 平台 —— 提供内置的 Agent 角色 (Code, QA, Explore, Plan, General 等)"
    app = _CopySelectionApp(text)
    async with app.run_test() as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        selected = markdown.get_selection(SELECT_ALL)

        assert markdown._total_lines > 1
        assert selected == ("  • 通用 Agent 平台 —— 提供内置的 Agent 角色 (Code, QA, Explore, Plan, General 等)", "\n")


@pytest.mark.asyncio
async def test_markdown_copy_selection_keeps_inline_code_tab_at_soft_wrap() -> None:
    app = _NarrowCopySelectionApp("`aa\tbb`")
    async with app.run_test() as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        selected = markdown.get_selection(SELECT_ALL)

        assert markdown._total_lines == 2
        assert selected == ("aa\tbb", "\n")


@pytest.mark.asyncio
async def test_markdown_copy_selection_preserves_real_line_breaks() -> None:
    app = _CopySelectionApp("First paragraph wraps across rows  \nSecond real line")
    async with app.run_test() as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        selected = markdown.get_selection(SELECT_ALL)

        assert markdown._total_lines > 2
        assert selected == ("First paragraph wraps across rows\nSecond real line", "\n")


@pytest.mark.asyncio
async def test_markdown_direct_render_cache_tracks_color_system() -> None:
    """Direct clipped rendering must not reuse filtered strips across terminal color modes."""
    app = _MarkdownSelectionApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)
        crop = Region(0, 0, markdown.size.width, 1)

        # Anchor the starting color system explicitly so the second render
        # always observes a transition (Windows CI defaults to a downsampled
        # system that may otherwise match EIGHT_BIT).
        app.console._color_system = ColorSystem.TRUECOLOR
        first = markdown.render_lines(crop)
        assert markdown._lines_cache_result is first

        app.console._color_system = ColorSystem.EIGHT_BIT
        second = markdown.render_lines(crop)

        assert markdown._lines_cache_result is second
        assert second is not first


@pytest.mark.asyncio
async def test_markdown_render_lines_after_remove_returns_blank() -> None:
    app = _MarkdownSelectionApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)

        await markdown.remove()
        await pilot.pause()

        rendered = markdown.render_lines(Region(0, 0, 20, 2))

    assert [strip.cell_length for strip in rendered] == [20, 20]


@pytest.mark.asyncio
async def test_markdown_offset_hover_does_not_apply_link_hover_style() -> None:
    """Mouse hover offset metadata is for hit testing, not link highlighting."""
    app = _MarkdownSelectionApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        markdown = app.query_one(VirtualizedMarkdown)
        crop = Region(0, 0, markdown.size.width, 1)

        baseline = markdown.render_lines(crop)[0]
        hover_style = baseline._segments[0].style
        assert hover_style is not None
        assert hover_style._link_id
        assert "@click" not in hover_style.meta

        markdown._lines_cache_result = None
        markdown.set_reactive(Widget.hover_style, hover_style)
        hovered = markdown.render_lines(crop)[0]

        assert [(text, style) for text, style, _control in hovered._segments] == [
            (text, style) for text, style, _control in baseline._segments
        ]
