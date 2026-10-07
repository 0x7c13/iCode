# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Visible container nesting, spacing, wrapping and clipboard regressions."""

from __future__ import annotations

import pytest
from markdown_it.token import Token
from textual.app import ComposeResult
from textual.content import Content
from textual.selection import SELECT_ALL

from chrys.app.tui.widgets.markdown.blocks import MarkdownBlock
from chrys.app.tui.widgets.markdown.copyable import CopyableMarkdown
from chrys.app.tui.widgets.markdown.parser import _parse_tokens
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from tests.app.tui.widgets.test_markdown_diagram import _DiagramMarkdownApp
from tests.app.tui.widgets.test_markdown_render_golden import _MdApp, _render_all_rows, _row_text
from tests.support.tui_helpers import click_when_settled, delay_resize_dispatch, resize_when_settled
from tests.support.waiting import wait_for


def test_external_token_handler_retains_its_flat_layout() -> None:
    block = MarkdownBlock("custom", Content("notice"), indent=3, prefix="* ", border_left="│ ")
    blocks = _parse_tokens([Token("custom", "", 0)], lambda _: block)
    assert blocks == [block]
    assert (block.indent, block.prefix, block.border_left) == (3, "* ", "│ ")


async def test_nested_quote_list_keeps_borders_and_return_to_outer_quote() -> None:
    source = "> outer\n>\n> > inner\n> >\n> > - item\n>\n> back\n"
    rows = await _render_all_rows(source)
    assert rows[:7] == ["▌ outer", "▌", "▌ ▌ inner", "▌ ▌", "▌ ▌   • item", "▌", "▌ back"]


async def test_alternating_list_and_quote_containers_keep_source_order() -> None:
    source = "- outer\n  > quote\n  > - inner\n  >\n  > back\n\n  tail\n"
    rows = await _render_all_rows(source)
    assert [row for row in rows if row.strip() and row.strip() != "▌"] == [
        "  • outer",
        "    ▌ quote",
        "    ▌ ▪ inner",
        "    ▌ back",
        "    tail",
    ]
    inner = rows.index("    ▌ ▪ inner")
    assert rows[inner + 1] == "    ▌"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("- one\n- two", ["  • one", "  • two"]),
        ("- one\n\n- two", ["  • one", "", "  • two"]),
        ("- one\n\n  two", ["  • one", "", "    two"]),
        ("> one\n>\n> two", ["▌ one", "▌", "▌ two"]),
        ("> one\n\noutside\n\n> two", ["▌ one", "", "outside", "", "▌ two"]),
        ("> one\n\n> two", ["▌ one", "", "▌ two"]),
        (">\n", ["▌"]),
        ("-\n", ["  •"]),
    ],
)
async def test_paragraph_gaps_follow_container_boundaries(source: str, expected: list[str]) -> None:
    rows = await _render_all_rows(source)
    while rows and not rows[-1]:
        rows.pop()
    assert rows == expected


@pytest.mark.parametrize(
    ("body", "label"),
    [
        ("## Heading", "Heading"),
        ("| a | b |\n|---|---|\n| 1 | 2 |", "│ a │ b │"),
        ("***", "───"),
        ("```text\ncode\n```", "code"),
        ("$$x^2$$", "x²"),
        ("```mermaid\nflowchart LR\nA[Start] --> B[End]\n```", "Start"),
    ],
)
async def test_every_block_kind_stays_in_quote_and_consumes_list_marker(body: str, label: str) -> None:
    lines = body.splitlines()
    source = "> - " + lines[0] + "\n" + "".join(">   " + line + "\n" for line in lines[1:])
    source += ">\n>   tail\n"
    app = _MdApp(source)
    async with app.run_test(size=(60, 50)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None
        rows = [_row_text(md.render_line(y)) for y in range(md._total_lines)]
        assert all(row.startswith("▌") for row in rows)
        assert any(label in row for row in rows)
        assert sum("•" in row for row in rows) == 1
        marker_line = next(i for i, row in enumerate(rows) if "•" in row)
        assert marker_line < next(i for i, row in enumerate(rows) if "tail" in row)
        assert "tail" not in rows[marker_line]
        assert all(md.render_line(y).cell_length == md._width_at_last_layout for y in range(md._total_lines))


async def test_quoted_heading_preserves_heading_style() -> None:
    app = _MdApp("> ## Heading\n>\n> paragraph")
    async with app.run_test() as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None
        strip = md.render_line(0)
        style = next(segment.style for segment in strip if "Heading" in segment.text)
        heading_style = md.get_visual_style("virtualized-markdown--h2").rich_style
        assert style is not None
        assert (style.color, style.bold, style.underline) == (
            heading_style.color,
            heading_style.bold,
            heading_style.underline,
        )
        assert style.bgcolor == md._get_bq_depth_style(1).rich_style.bgcolor


async def test_quote_paragraph_respects_custom_quote_text_style() -> None:
    class StyledApp(_MdApp):
        CSS = """
        VirtualizedMarkdown > .virtualized-markdown--block-quote { color: red; text-style: italic; }
        VirtualizedMarkdown > .virtualized-markdown--paragraph { color: blue; }
        """

    app = StyledApp("> quoted paragraph")
    async with app.run_test() as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None
        style = next(segment.style for segment in md.render_line(0) if "quoted paragraph" in segment.text)
        assert style is not None
        assert style.color == md.get_visual_style("virtualized-markdown--block-quote").rich_style.color
        assert style.italic


async def test_quoted_table_header_keeps_quote_background_and_header_emphasis() -> None:
    app = _MdApp("> | Header | Value |\n> |---|---|\n> | one | two |")
    async with app.run_test() as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None
        style = next(segment.style for segment in md.render_line(1) if "Header" in segment.text)
        assert style is not None
        assert style.bgcolor == md._get_bq_depth_style(1).rich_style.bgcolor
        assert style.bold


@pytest.mark.parametrize("width", [16, 22, 40])
async def test_long_list_numbers_wrap_and_copy_without_losing_characters(width: int) -> None:
    text = "abcdefghijklmnopqrstuvwxyz 中文结束"
    source = f"9999. {text}\n10000. {text}"
    app = _MdApp(source)
    async with app.run_test(size=(width, 30)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None
        assert md.get_selection(SELECT_ALL) == (source, "\n")


async def test_quoted_list_wrapping_keeps_borders_and_copies_one_paragraph() -> None:
    text = "中文 English 123 mixed text wraps inside the nested quote."
    app = _MdApp("> > - " + text)
    async with app.run_test(size=(24, 30)) as pilot:
        await pilot.pause()
        md = app.widget
        assert md is not None
        rows = [_row_text(md.render_line(y)) for y in range(md._total_lines)]
        assert len(rows) > 2
        assert rows[0].startswith("▌ ▌   • ")
        assert all(row.startswith("▌ ▌     ") for row in rows[1:])
        assert md.get_selection(SELECT_ALL) == ("▌ ▌   • " + text, "\n")


class _CopyApp(_MdApp):
    def compose(self) -> ComposeResult:
        self.widget = CopyableMarkdown(self._md, copy_button_text="Copy")
        yield self.widget


@pytest.mark.parametrize("prefix", ["", "> ", "> - "])
async def test_final_code_copy_button_stays_visible_inside_containers(
    prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    continuation = prefix.replace("-", " ")
    source = f"{prefix}```text\n{continuation}hello\n{continuation}```"
    app = _CopyApp(source)
    async with app.run_test(size=(30, 15)) as pilot:
        await pilot.pause()
        md = app.query_one(CopyableMarkdown)
        line = next(iter(md._copy_button_lines))
        assert line < md._total_lines
        row = _row_text(md.render_line(line))
        assert "Copy" in row
        if prefix:
            assert row.startswith("▌ ")
        await click_when_settled(pilot, md, offset=(row.index("Copy"), line))
        assert app.clipboard == "hello"
        delay_resize_dispatch(app, monkeypatch, 0.05)
        await resize_when_settled(pilot, 24, 15)
        await wait_for(
            lambda: md._width_at_last_layout == 24,
            pilot=pilot,
            description="markdown relaid out at the resized width",
        )
        assert next(iter(md._copy_button_lines)) == line
        assert _row_text(md.render_line(line)).index("Copy") == row.index("Copy") - 6


async def test_copy_button_hit_testing_accounts_for_padding_and_border() -> None:
    class PaddedApp(_CopyApp):
        CSS = "VirtualizedMarkdown { padding: 1 2; border: solid red; }"

    app = PaddedApp("> ```text\n> hello\n> ```")
    async with app.run_test(size=(30, 15)) as pilot:
        await pilot.pause()
        md = app.query_one(CopyableMarkdown)
        line = next(iter(md._copy_button_lines))
        row = _row_text(md.render_line(line))
        offset = md.content_region.offset - md.region.offset
        await click_when_settled(pilot, md, offset=(row.index("Copy") + offset.x, line + offset.y))
        assert app.clipboard == "hello"


async def test_padding_cannot_activate_a_scrolled_out_copy_button() -> None:
    class PaddedApp(_CopyApp):
        CSS = "VirtualizedMarkdown { padding: 1 2; }"

    app = PaddedApp("```text\nhello\n```\n\n" + "paragraph\n\n" * 20)
    async with app.run_test(size=(30, 10)) as pilot:
        await pilot.pause()
        md = app.query_one(CopyableMarkdown)
        line = next(iter(md._copy_button_lines))
        row = _row_text(md.render_line(line))
        md.scroll_to(y=line + 1, animate=False)
        await pilot.pause()
        assert md.scroll_offset.y == line + 1
        await click_when_settled(pilot, md, offset=(row.index("Copy") + 2, 0))
        assert app.clipboard == ""


async def test_padding_cannot_activate_a_scrolled_out_diagram_action() -> None:
    class PaddedApp(_DiagramMarkdownApp):
        CSS = "VirtualizedMarkdown { padding: 1 2; }"

    source = "```mermaid\nflowchart LR\nA --> B\n```\n\n" + "paragraph\n\n" * 20
    app = PaddedApp(source)
    async with app.run_test(size=(60, 24)) as pilot:
        await pilot.pause()
        md = app.query_one(VirtualizedMarkdown)
        line = next(iter(md._diagram_action_lines))
        row = _row_text(md.render_line(line))
        x = row.index("Open full diagram") + 2
        await click_when_settled(pilot, md, offset=(x, line + 1))
        assert app.opened is not None
        app.opened = None
        md.scroll_to(y=line + 1, animate=False)
        await pilot.pause()
        assert md.scroll_offset.y == line + 1
        await click_when_settled(pilot, md, offset=(x, 0))
        assert app.opened is None
        await pilot.hover(md, offset=(x, 0))
        assert md.styles.pointer == "default"
