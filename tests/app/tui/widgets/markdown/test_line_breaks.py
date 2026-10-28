# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The line-break parser keeps each newline inside a paragraph; the default parser joins them.

Every parser keeps HTML-looking text as written; ``<br>`` breaks the line except in typed text.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from markdown_it import MarkdownIt
from textual.app import App

from chrys.app.tui.widgets.markdown.parser import (
    _create_markdown_parser,
    _parse_tokens,
    create_line_break_markdown_parser,
    create_user_text_markdown_parser,
)
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from tests.support.waiting import wait_for

_SOURCE = "first line\nsecond line\n\n- item one\n  continued\n"


def _inline_types(parser_tokens: list) -> list[str]:
    return [child.type for token in parser_tokens for child in token.children or [] if child.type.endswith("break")]


def test_soft_breaks_become_hard_breaks_only_in_the_line_break_parser() -> None:
    assert _inline_types(create_line_break_markdown_parser().parse(_SOURCE)) == ["hardbreak", "hardbreak"]
    assert _inline_types(_create_markdown_parser().parse(_SOURCE)) == ["softbreak", "softbreak"]
    assert _inline_types(create_user_text_markdown_parser().parse(_SOURCE)) == ["hardbreak", "hardbreak"]


def _paragraph_text(parser: MarkdownIt, source: str) -> str:
    [block] = _parse_tokens(parser.parse(source))
    return block.content.plain


@pytest.mark.parametrize(
    "factory", [_create_markdown_parser, create_line_break_markdown_parser, create_user_text_markdown_parser]
)
def test_html_looking_text_stays_as_written(factory: Callable[[], MarkdownIt]) -> None:
    source = "Fix List<String> in <file>\n<div>\nblock\n</div>"

    tokens = factory().parse(source)

    types = [token.type for token in tokens] + [child.type for token in tokens for child in token.children or []]
    assert not [token_type for token_type in types if token_type.startswith("html")]
    inline = "".join(child.content for token in tokens for child in token.children or [] if child.type == "text")
    assert inline == "Fix List<String> in <file><div>block</div>"


@pytest.mark.parametrize("factory", [_create_markdown_parser, create_line_break_markdown_parser])
def test_br_breaks_the_line_once_outside_code(factory: Callable[[], MarkdownIt]) -> None:
    parser = factory()

    assert _paragraph_text(parser, "one<br>two <BR/> three") == "one\ntwo\nthree"
    assert _paragraph_text(parser, "one<br>\ntwo") == "one\ntwo"
    assert _paragraph_text(parser, "one<br><br>two") == "one\n\ntwo"
    assert _paragraph_text(parser, "one <br> ") == "one"
    assert _paragraph_text(parser, "use `<br>` here") == "use <br> here"


def test_br_stays_as_typed_in_user_text() -> None:
    assert _paragraph_text(create_user_text_markdown_parser(), "one<br>two") == "one<br>two"


@pytest.mark.parametrize(
    "source",
    [
        r"open D:\project\.code",
        r"C:\Users\me\_cache\file.txt",
        r"a \* b \\ c \` d",
        r"line ends with \\",
    ],
)
def test_backslashes_stay_as_typed_in_user_text(source: str) -> None:
    assert _paragraph_text(create_user_text_markdown_parser(), source) == source


def test_a_typed_backslash_still_keeps_the_next_character_plain() -> None:
    parser = create_user_text_markdown_parser()

    assert _paragraph_text(parser, r"\*not emphasis\*") == r"\*not emphasis\*"
    assert _paragraph_text(parser, r"\[dir\]\#x \`no code\`") == r"\[dir\]\#x \`no code\`"
    tokens = parser.parse(r"\*not emphasis\* and \`no code\`")
    assert not [
        child.type for token in tokens for child in token.children or [] if child.type in {"em_open", "code_inline"}
    ]


def test_a_backslash_before_a_newline_stays_as_typed_and_the_line_still_breaks() -> None:
    assert _paragraph_text(create_user_text_markdown_parser(), "one\\\ntwo") == "one\\\ntwo"


def _table_cell_text(parser: MarkdownIt, cell: str) -> str:
    *_, body_cell = [token for token in parser.parse(f"| head |\n|---|\n| {cell} |\n") if token.type == "inline"]
    return "".join(child.content for child in body_cell.children or [] if child.type == "text")


def test_a_path_in_a_typed_table_cell_stays_as_typed() -> None:
    assert _table_cell_text(create_user_text_markdown_parser(), r"C:\Users\me\.code") == r"C:\Users\me\.code"


def test_a_typed_table_cell_still_reads_a_backslashed_pipe_as_a_literal_pipe() -> None:
    # Standard table syntax, kept on purpose: \| is the only way to put a pipe in a cell.
    assert _table_cell_text(create_user_text_markdown_parser(), r"x\|y") == "x|y"


@pytest.mark.parametrize("factory", [_create_markdown_parser, create_line_break_markdown_parser])
def test_backslash_escapes_still_apply_outside_user_text(factory: Callable[[], MarkdownIt]) -> None:
    assert _paragraph_text(factory(), r"D:\project\.code \*not italic\*") == r"D:\project.code *not italic*"


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
