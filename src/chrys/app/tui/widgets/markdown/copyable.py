# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""CopyableMarkdown — VirtualizedMarkdown with copy buttons on code fences.

Each fence block gets a clickable "copy" button rendered in its bottom
margin.  Clicking the button copies the raw code to the clipboard.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from rich.segment import Segment, cell_len
from textual.events import Click
from textual.strip import Strip

from chrys.app.tui.clipboard import copy_text_to_clipboards
from chrys.app.tui.copy_messages import COPIED_TITLE
from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.widgets.markdown.blocks import MarkdownBlock, _BlockLineInfo
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from chrys.foundation.i18n import MessageRef, msg

if TYPE_CHECKING:
    pass

_COPY_BUTTON_RIGHT_MARGIN = 4
_CODE_BLOCK_COPIED = msg(
    "tui.markdown.copy.code_block",
    fallback="Code block copied to clipboard",
)
_CODE_BLOCK_COPIED_REF = _CODE_BLOCK_COPIED.bind()
_COPIED_TITLE_REF = COPIED_TITLE.bind()


class CopyableMarkdown(VirtualizedMarkdown):
    """Markdown widget with copyable code blocks.

    Each fence block has a copy button displayed below the code.
    Clicking the button copies the code to the clipboard.
    """

    COMPONENT_CLASSES: ClassVar[set[str]] = VirtualizedMarkdown.COMPONENT_CLASSES | {
        "markdown--copy-button",
    }

    DEFAULT_CSS = """
    CopyableMarkdown {
        background: transparent;

        & > .markdown--copy-button {
            background: transparent;
            text-style: underline;
            color: $text-muted;
        }
    }
    """

    def __init__(
        self,
        markdown: str | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        parser_factory: Callable | None = None,
        open_links: bool = True,
        copy_button_text: str = "📋 Copy",
        copy_success_text: MessageRef | str = _CODE_BLOCK_COPIED_REF,
        copy_success_title: MessageRef | str = _COPIED_TITLE_REF,
    ):
        super().__init__(
            markdown,
            name=name,
            id=id,
            classes=classes,
            parser_factory=parser_factory,
            open_links=open_links,
        )
        self._copy_button_text = copy_button_text
        self._copy_success_text = copy_success_text
        self._copy_success_title = copy_success_title
        self._copy_button_lines: dict[int, int] = {}
        """Mapping from virtual line number (copy button line) to block index."""

    def _layout_blocks(self, width: int | None = None) -> None:
        # Ensure fence blocks have at least 2 lines of bottom margin
        # so there is room for a blank separator + the copy button line.
        for block in self._blocks:
            if block.block_type == "fence" and block.bottom_margin < 2:
                block.bottom_margin = 2

        super()._layout_blocks(width)

        # Build copy-button-line → block-index mapping.
        self._copy_button_lines.clear()
        for idx, block in enumerate(self._blocks):
            if block.block_type == "fence":
                info = self._block_line_info[idx]
                # Place the copy button on the first bottom-margin line
                # (right below the fence content; second line is spacing before next block).
                copy_line = info.start_line + info.top_margin + info.content_height
                self._copy_button_lines[copy_line] = idx

    def _render_block_line(self, block: MarkdownBlock, info: _BlockLineInfo, line: int, width: int) -> Strip:
        if line in self._copy_button_lines:
            return self._render_copy_button(width)
        return super()._render_block_line(block, info, line, width)

    def _render_copy_button(self, width: int) -> Strip:
        copy_text = self._copy_button_text
        button_style = self._safe_component_style("markdown--copy-button").rich_style
        base_rs = self.visual_style.rich_style

        copy_cells = cell_len(copy_text)
        left_pad = max(0, width - copy_cells - _COPY_BUTTON_RIGHT_MARGIN)
        segments = [
            Segment(" " * left_pad, base_rs),
            Segment(copy_text, button_style),
            Segment(" " * _COPY_BUTTON_RIGHT_MARGIN, base_rs),
        ]
        return Strip(segments, width)

    def on_click(self, event: Click) -> None:
        virtual_y = event.y + self.scroll_offset.y

        if virtual_y not in self._copy_button_lines:
            super().on_click(event)
            return

        block_index = self._copy_button_lines[virtual_y]
        block = self._blocks[block_index]

        # Hit-test: is the click within the button text?
        copy_text = self._copy_button_text
        width = self.scrollable_content_region.width
        copy_cells = cell_len(copy_text)

        padding_left = self.styles.padding.left
        x_in_content = event.x - padding_left

        button_start = width - copy_cells - _COPY_BUTTON_RIGHT_MARGIN
        button_end = width - _COPY_BUTTON_RIGHT_MARGIN

        if button_start <= x_in_content < button_end:
            code = block.content.plain
            copy_text_to_clipboards(self.app, code)
            self.notify(
                self._render_toast(self._copy_success_text),
                title=self._render_toast(self._copy_success_title),
                markup=False,
            )
            event.prevent_default()
            event.stop()

    def _render_toast(self, message: MessageRef | str) -> str:
        if isinstance(message, str):
            return message
        return render_str(widget_localizer(self), message)
