# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Data structures for parsed markdown blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from textual.content import Content

if TYPE_CHECKING:
    from chrys.app.tui.widgets.markdown.diagram.model import CompiledDiagram
    from chrys.app.tui.widgets.markdown.math import CompiledMath

type TableOfContentsType = list[tuple[int, str, str | None]]
"""Information about the table of contents of a markdown document.

The triples encode the level, the label, and the optional block id of each heading.
"""

BULLETS = ["\u2022 ", "\u25aa ", "\u2023 ", "\u2b51 ", "\u25e6 "]
"""Unicode bullets used for unordered lists."""

NUMERALS = " \u2160\u2161\u2162\u2163\u2164\u2165\u2166"


@dataclass(frozen=True)
class MarkdownGutter:
    """One ordered container in a block's left gutter, shared by wrapped rows."""

    container_id: int
    kind: Literal["list", "quote"]
    width: int
    prefix: str = ""


@dataclass
class MarkdownBlock:
    """Data object representing a parsed markdown block element."""

    block_type: str
    """The type of block (e.g. 'paragraph', 'heading', 'fence', 'hr', etc.)."""
    content: Content
    """The rendered Content for this block."""
    level: int = 0
    """The heading level (1-6) for heading blocks, or nesting level for lists."""
    block_id: str | None = None
    """An optional ID for the block (used for heading anchors)."""
    source_range: tuple[int, int] = (0, 0)
    """The (start_line, end_line) range in the source document."""
    style_name: str = ""
    """CSS component class name for styling."""
    top_margin: int = 0
    """Lines of margin above this block."""
    bottom_margin: int = 1
    """Lines of margin below this block."""
    indent: int = 0
    """Left indentation in cells."""
    prefix: str = ""
    """A prefix string (e.g. bullet character) to render before the first line."""
    padding_top: int = 0
    """Lines of padding above content (rendered with block style, unlike margin)."""
    padding_bottom: int = 0
    """Lines of padding below content (rendered with block style, unlike margin)."""
    padding_left: int = 0
    """Cells of padding to the left of content."""
    padding_right: int = 0
    """Cells of padding to the right of content (inside block background)."""
    border_left: str = ""
    """Character to render as a left border on every content line."""
    bq_depth: int = 0
    """Blockquote nesting depth (0 = not in blockquote)."""
    gutter: tuple[MarkdownGutter, ...] = ()
    """List and quote containers in source order; IDs distinguish adjacent quotes."""
    text_align: str = "left"
    """Text alignment: 'left', 'center', or 'right'."""
    code_language: str = ""
    """Language for code blocks."""
    is_header_row: bool = False
    """Whether this is a table header row."""
    table_headers: list[Content] | None = None
    """Header contents for table blocks."""
    table_rows: list[list[Content]] | None = None
    """Row contents for table blocks."""
    diagram: CompiledDiagram | None = None
    """Compiled terminal diagram payload for diagram blocks."""
    math: CompiledMath | None = None
    """Formula canvas; content retains its source for lossless narrow layouts."""


class MarkdownFence:
    """Compatibility shim - references MarkdownBlock internally.

    This class exists to preserve the public API for code that checks
    `isinstance(block, MarkdownFence)`.
    """


@dataclass
class _BlockLineInfo:
    """Cached rendering information for a markdown block."""

    block_index: int
    """Index into the blocks list."""
    start_line: int
    """First virtual line occupied by this block."""
    height: int
    """Total height in virtual lines (including margins)."""
    content_height: int
    """Height of the content itself (without margins)."""
    top_margin: int
    """Top margin lines."""
    bottom_margin: int
    """Bottom margin lines."""
