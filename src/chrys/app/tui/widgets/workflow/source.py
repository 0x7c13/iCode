# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Literal workflow source display, independent of the bytes trusted and executed."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.segment import Segment
from rich.syntax import Syntax

from chrys.app.tui.util.source_text import sanitize_source_text

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult

type _RenderKey = tuple[int, bool | None, bool, bool, str | None]

# One layout pass renders the view at two widths (with and without the vertical
# scrollbar of the scroll view around it); older widths are rendered again.
_RENDERED_WIDTHS = 2


class WorkflowSourceSyntax(Syntax):
    """Line-numbered, non-wrapping view of workflow source that is lexed once per render width.

    Textual renders a Rich renderable again for every height query and every paint,
    and each ``Syntax`` render re-lexes the whole file. The code and every render
    setting are fixed at construction, and with them ``Syntax`` output depends only
    on the option and console fields in the render key, so the segments rendered
    for a recent key are replayed instead.
    """

    def __init__(self, source: bytes) -> None:
        # Normalize source whitespace before replacing terminal control characters.
        super().__init__(
            sanitize_source_text(source.decode("utf-8", errors="replace")),
            "python",
            line_numbers=True,
            word_wrap=False,
        )
        self.source = source
        self._rendered: dict[_RenderKey, list[Segment]] = {}

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        key: _RenderKey = (
            options.max_width,
            options.no_wrap,
            options.ascii_only,
            options.legacy_windows,
            console.color_system,
        )
        segments = self._rendered.pop(key, None)
        if segments is None:
            # Console.render renders nested renderables with the height reset.
            nested = options.reset_height()
            segments = [
                segment
                for output in super().__rich_console__(console, options)
                for segment in ((output,) if isinstance(output, Segment) else console.render(output, nested))
            ]
            if len(self._rendered) >= _RENDERED_WIDTHS:
                del self._rendered[next(iter(self._rendered))]
        self._rendered[key] = segments
        yield from segments
