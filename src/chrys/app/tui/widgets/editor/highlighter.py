# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Clean-room, lazy Markdown highlighting for the message editor."""

from __future__ import annotations

from typing import Any

from rich.text import Text
from textual.content import Content
from textual.highlight import highlight
from textual.widgets.text_area import Edit, EditResult

from chrys.app.tui.widgets.syntax_theme import NoErrorHighlightTheme
from chrys.app.tui.widgets.text_area import EnhancedTextArea
from chrys.foundation.i18n import MessageRef, msg

MARKDOWN_HIGHLIGHT_CHARACTER_CUTOFF = 16_384
"""Suspend highlighting at or above this many Unicode code points."""

MARKDOWN_HIGHLIGHT_LINE_CUTOFF = 1_000
"""Suspend highlighting above this many logical lines."""

MARKDOWN_HIGHLIGHT_PLAIN_STATUS = msg(
    "tui.editor.status.markdown_highlighting_paused",
    fallback="PLAIN · Markdown highlighting paused for large draft",
)
"""Visible status used while the safe highlighting limits are exceeded."""


class MarkdownHighlightTheme(NoErrorHighlightTheme):
    """Chrys Markdown theme using the shared error-token suppression policy."""

    STYLES: dict[tuple[str, ...], str] = dict(  # noqa: RUF012 - Textual requires a mutable theme mapping.
        NoErrorHighlightTheme.STYLES
    )


class MarkdownHighlightedTextArea(EnhancedTextArea):
    """Render Markdown styles in-place without changing the editable document."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._markdown_content_lines: list[Content] | None = None
        self._markdown_rendered_lines: dict[int, Text] = {}
        self._markdown_plain = False
        super().__init__(*args, **kwargs)
        self._markdown_plain = self._exceeds_highlight_limits()

    @property
    def markdown_highlighting_degraded(self) -> bool:
        """Whether the document currently renders through the plain fallback."""
        return self._markdown_plain

    @property
    def markdown_highlight_status(self) -> MessageRef | str:
        """Return the large-document status, or an empty string when highlighted."""
        return MARKDOWN_HIGHLIGHT_PLAIN_STATUS.bind() if self._markdown_plain else ""

    def get_line(self, line_index: int) -> Text:
        """Return an independently mutable rendered copy of one document line."""
        if self._markdown_plain:
            return super().get_line(line_index)

        cached = self._markdown_rendered_lines.get(line_index)
        if cached is not None:
            return cached.copy()

        lines = self._highlighted_document_lines()
        if not 0 <= line_index < len(lines):
            return super().get_line(line_index)

        rendered = Text(end="", no_wrap=True)
        for segment, style, _control in lines[line_index].render_segments(self.visual_style):
            rendered.append(segment, style)
        self._markdown_rendered_lines[line_index] = rendered
        return rendered.copy()

    def notify_style_update(self) -> None:
        """Discard style-resolved caches before Textual applies a new style context."""
        self._invalidate_markdown_caches()
        super().notify_style_update()

    def edit(self, edit: Edit) -> EditResult:
        """Invalidate rendered Markdown synchronously with a document edit."""
        result = super().edit(edit)
        self._sync_markdown_document()
        return result

    def load_text(self, text: str) -> None:
        """Invalidate rendered Markdown synchronously with a document replacement."""
        super().load_text(text)
        self._sync_markdown_document()

    def undo(self) -> None:
        """Invalidate rendered Markdown synchronously with an undo batch."""
        super().undo()
        self._sync_markdown_document()

    def redo(self) -> None:
        """Invalidate rendered Markdown synchronously with a redo batch."""
        super().redo()
        self._sync_markdown_document()

    def _sync_markdown_document(self) -> None:
        """Refresh derived highlighting state before the next compositor paint."""
        was_plain = self._markdown_plain
        self._markdown_plain = self._exceeds_highlight_limits()
        self._invalidate_markdown_caches()
        if self._markdown_plain != was_plain:
            self._on_markdown_highlight_status_changed()

    def _highlighted_document_lines(self) -> list[Content]:
        lines = self._markdown_content_lines
        if lines is None:
            highlighted = self._highlight_markdown_document(self.text)
            lines = highlighted.split("\n", allow_blank=True)[: self.document.line_count]
            self._markdown_content_lines = lines
        return lines

    def _highlight_markdown_document(self, source: str) -> Content:
        """Highlight a synthetic copy closed by a fence for stable open-fence tokens."""
        synthetic_source = f"{source}\n```"
        return highlight(synthetic_source, language="markdown", theme=MarkdownHighlightTheme)

    def _exceeds_highlight_limits(self) -> bool:
        return (
            len(self.text) >= MARKDOWN_HIGHLIGHT_CHARACTER_CUTOFF
            or self.document.line_count > MARKDOWN_HIGHLIGHT_LINE_CUTOFF
        )

    def _invalidate_markdown_caches(self) -> None:
        self._markdown_content_lines = None
        self._markdown_rendered_lines.clear()

    def _on_markdown_highlight_status_changed(self) -> None:
        """Hook for subclasses that expose the degraded transition as a message."""

    def on_unmount(self) -> None:
        self._invalidate_markdown_caches()
