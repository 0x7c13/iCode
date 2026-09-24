# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared, centered welcome surface with a configurable logo and context."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

from rich.cells import cell_len, set_cell_size
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual.widget import Widget

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult
    from textual.geometry import Size


class _WelcomeRenderable:
    """Fill the available area with a centered logo, title and working directory."""

    def __init__(
        self,
        logo_lines: tuple[str, ...],
        title: str,
        cwd: str,
        width: int,
        height: int,
        logo_style: Style,
    ) -> None:
        self.logo_lines = logo_lines
        self.title = title
        self.cwd = cwd
        self.width = width
        self.height = height
        self.logo_style = logo_style

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width, height = self.width, self.height
        if width <= 0 or height <= 0:
            return

        logo_width = max((cell_len(line) for line in self.logo_lines), default=0)
        logo_left = max(0, (width - logo_width) // 2)
        rows = [(logo_left, line, self.logo_style) for line in self.logo_lines]
        info = [(self.title, Style(bold=True)), (self.cwd, Style(dim=True))]
        if any(value for value, _style in info):
            rows.append((0, "", Style()))
        for value, style in info:
            if value:
                line = Text(value)
                line.truncate(width, overflow="ellipsis")
                rows.append((max(0, (width - line.cell_len) // 2), line.plain, style))

        top = max(0, (height - len(rows)) // 2)
        for y in range(height):
            row = y - top
            if 0 <= row < len(rows):
                left, value, style = rows[row]
                yield Segment(" " * left)
                yield Segment(set_cell_size(value, width - left), style)
            else:
                yield Segment(" " * width)
            yield Segment.line()


class WelcomeWidget(Widget):
    """Reusable empty state; callers own mode selection and localized display text."""

    DEFAULT_CSS = """
    WelcomeWidget {
        width: 100%;
        height: 100%;
        color: $foreground;
    }
    """

    def __init__(
        self,
        logo: str,
        *,
        title: str = "",
        cwd: str = "",
        compact_logo: str = "",
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self._logo_lines = tuple(
            dedent("\n".join(line.rstrip() for line in logo.splitlines())).strip("\n").splitlines()
        )
        self._logo_width = max((cell_len(line) for line in self._logo_lines), default=0)
        self._title = title
        self._cwd = cwd
        self._compact_logo = compact_logo

    def update_info(self, *, title: str | None = None, cwd: str | None = None, compact_logo: str | None = None) -> None:
        """Update context without scheduling a repaint for unchanged values."""
        values = (
            self._title if title is None else title,
            self._cwd if cwd is None else cwd,
            self._compact_logo if compact_logo is None else compact_logo,
        )
        if values != (self._title, self._cwd, self._compact_logo):
            self._title, self._cwd, self._compact_logo = values
            height = self.styles.height
            self.refresh(layout=height is not None and height.is_auto)

    def _visible_logo(self, width: int) -> tuple[str, ...]:
        # A compact wordmark keeps wide ASCII art readable in narrow panels.
        return (self._compact_logo,) if self._compact_logo and width < self._logo_width else self._logo_lines

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        info_height = bool(self._title) + bool(self._cwd)
        return len(self._visible_logo(width)) + (1 + info_height if info_height else 0)

    def render(self) -> _WelcomeRenderable:
        return _WelcomeRenderable(
            self._visible_logo(self.size.width),
            self._title,
            self._cwd,
            self.size.width,
            self.size.height,
            Style(color=self.rich_style.color, bold=True),
        )
