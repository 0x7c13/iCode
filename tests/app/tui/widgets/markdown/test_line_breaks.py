# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The line-break parser keeps each newline inside a paragraph; the default parser joins them."""

from __future__ import annotations

from textual.app import App

from chrys.app.tui.widgets.markdown.parser import _create_markdown_parser, create_line_break_markdown_parser
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from tests.support.waiting import wait_for

_SOURCE = "first line\nsecond line\n\n- item one\n  continued\n"


def _inline_types(parser_tokens: list) -> list[str]:
    return [child.type for token in parser_tokens for child in token.children or [] if child.type.endswith("break")]


def test_soft_breaks_become_hard_breaks_only_in_the_line_break_parser() -> None:
    assert _inline_types(create_line_break_markdown_parser().parse(_SOURCE)) == ["hardbreak", "hardbreak"]
    assert _inline_types(_create_markdown_parser().parse(_SOURCE)) == ["softbreak", "softbreak"]


async def test_line_break_parser_renders_each_source_line_on_its_own_row() -> None:
    app = App()
    async with app.run_test(size=(80, 20)) as pilot:
        widget = VirtualizedMarkdown(_SOURCE, parser_factory=create_line_break_markdown_parser)
        await app.mount(widget)
        await wait_for(
            lambda: widget.scrollable_content_region.width > 0 and widget.virtual_size.height >= 5,
            pilot=pilot,
            description="line-break markdown laid out",
        )
        rows = [widget.render_line(y).text.rstrip() for y in range(widget.virtual_size.height)]
        assert [row.strip() for row in rows if row.strip()] == ["first line", "second line", "• item one", "continued"]
