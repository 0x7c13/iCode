# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Virtualized Markdown widget package for Textual.

Provides a high-performance, virtualizing Markdown renderer with streaming support.
Named VirtualizedMarkdown to avoid conflicts with Textual's built-in Markdown widget.
"""

from chrys.app.tui.widgets.markdown.blocks import MarkdownBlock, MarkdownFence, TableOfContentsType
from chrys.app.tui.widgets.markdown.navigator import Navigator
from chrys.app.tui.widgets.markdown.stream import MarkdownStream
from chrys.app.tui.widgets.markdown.toc import MarkdownTableOfContents
from chrys.app.tui.widgets.markdown.viewer import VirtualizedMarkdownViewer
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown

__all__ = [
    "MarkdownBlock",
    "MarkdownFence",
    "MarkdownStream",
    "MarkdownTableOfContents",
    "Navigator",
    "TableOfContentsType",
    "VirtualizedMarkdown",
    "VirtualizedMarkdownViewer",
]
