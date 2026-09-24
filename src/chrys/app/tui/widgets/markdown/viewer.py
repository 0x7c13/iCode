# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""VirtualizedMarkdownViewer — combines VirtualizedMarkdown with a table of contents sidebar."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePath

from markdown_it import MarkdownIt
from textual.app import ComposeResult
from textual.events import Mount
from textual.message import Message
from textual.reactive import reactive, var
from textual.widget import Widget

from chrys.app.tui.widgets.markdown.navigator import Navigator
from chrys.app.tui.widgets.markdown.toc import MarkdownTableOfContents
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown


class VirtualizedMarkdownViewer(Widget, can_focus=False, can_focus_children=True):
    """A Markdown viewer widget with optional table of contents sidebar."""

    SCOPED_CSS = False

    DEFAULT_CSS = """
    VirtualizedMarkdownViewer {
        height: 1fr;
        layout: horizontal;
        background: $surface;
        & > VirtualizedMarkdown {
            width: 1fr;
            height: 1fr;
        }
        & > MarkdownTableOfContents {
            display: none;
            dock:left;
        }
    }

    VirtualizedMarkdownViewer.-show-table-of-contents > MarkdownTableOfContents {
        display: block;
    }
    """

    show_table_of_contents = reactive(True)
    """Show the table of contents?"""
    top_block = reactive("")

    navigator: var[Navigator] = var[Navigator](Navigator)

    class NavigatorUpdated(Message):
        """Navigator has been changed (clicked link etc)."""

    def __init__(
        self,
        markdown: str | None = None,
        *,
        show_table_of_contents: bool = True,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        parser_factory: Callable[[], MarkdownIt] | None = None,
        open_links: bool = True,
    ):
        """Create a VirtualizedMarkdownViewer object.

        Args:
            markdown: String containing Markdown, or None to leave blank.
            show_table_of_contents: Show a table of contents in a sidebar.
            name: The name of the widget.
            id: The ID of the widget in the DOM.
            classes: The CSS classes of the widget.
            parser_factory: A factory function to return a configured MarkdownIt instance.
            open_links: Open links automatically.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.show_table_of_contents = show_table_of_contents
        self._markdown = markdown
        self._parser_factory = parser_factory
        self._open_links = open_links

    @property
    def document(self) -> VirtualizedMarkdown:
        """The VirtualizedMarkdown document widget."""
        return self.query_one(VirtualizedMarkdown)

    @property
    def table_of_contents(self) -> MarkdownTableOfContents:
        """The table of contents widget."""
        return self.query_one(MarkdownTableOfContents)

    async def _on_mount(self, _: Mount) -> None:  # ty: ignore[invalid-method-override]  # Textual dispatch awaits async handlers.
        await self.document.update(self._markdown or "")

    async def go(self, location: str | PurePath) -> None:
        """Navigate to a new document path."""
        location_str = str(location)
        if location_str.startswith(("http://", "https://", "mailto:")):
            return
        path, anchor = self.document.sanitize_location(location_str)
        if path == Path(".") and anchor:
            self.document.goto_anchor(anchor)
        else:
            try:
                await self.document.load(self.navigator.go(location))
            except OSError:
                return
            self.post_message(self.NavigatorUpdated())

    async def back(self) -> None:
        """Go back one level in the history."""
        if self.navigator.back():
            await self.document.load(self.navigator.location)
            self.post_message(self.NavigatorUpdated())

    async def forward(self) -> None:
        """Go forward one level in the history."""
        if self.navigator.forward():
            await self.document.load(self.navigator.location)
            self.post_message(self.NavigatorUpdated())

    async def _on_virtualized_markdown_link_clicked(self, message: VirtualizedMarkdown.LinkClicked) -> None:
        message.stop()
        await self.go(message.href)

    def watch_show_table_of_contents(self, show_table_of_contents: bool) -> None:
        self.set_class(show_table_of_contents, "-show-table-of-contents")

    def compose(self) -> ComposeResult:
        markdown = VirtualizedMarkdown(parser_factory=self._parser_factory, open_links=self._open_links)
        yield markdown
        yield MarkdownTableOfContents(markdown)

    def _on_virtualized_markdown_table_of_contents_updated(
        self, message: VirtualizedMarkdown.TableOfContentsUpdated
    ) -> None:
        self.query_one(MarkdownTableOfContents).table_of_contents = message.table_of_contents
        message.stop()

    def _on_virtualized_markdown_table_of_contents_selected(
        self, message: VirtualizedMarkdown.TableOfContentsSelected
    ) -> None:
        self.document.scroll_to_block_id(message.block_id)
        message.stop()
