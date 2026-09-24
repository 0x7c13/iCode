# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Selection helpers for selectable content and excluded chrome."""

from __future__ import annotations

from typing import ClassVar

from rich.style import Style as RichStyle
from textual.selection import Selection
from textual.widgets import Static


class NonSelectableTextMixin:
    """Prevent a text-rendering widget from participating in selection."""

    ALLOW_SELECT: ClassVar[bool] = False

    @property
    def text_selection(self) -> None:
        """Ignore Textual endpoint selections that bypass ``allow_select``."""
        return None

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Exclude this widget from copied screen-selection text."""
        _ = selection
        return None

    def selection_updated(self, selection: Selection | None) -> None:
        """Avoid repainting chrome for an ignored selection update."""
        _ = selection


class NonSelectableStatic(NonSelectableTextMixin, Static):
    """Static chrome that cannot start, render, or export text selection."""


def normalize_selection_rich_style(style: RichStyle) -> RichStyle:
    """Preserve text color when Textual's transparent selection foreground resolves to background."""
    if style.color is None or style.bgcolor is None or style.color != style.bgcolor:
        return style
    return RichStyle(
        bgcolor=style.bgcolor,
        bold=style.bold,
        dim=style.dim,
        italic=style.italic,
        underline=style.underline,
        blink=style.blink,
        blink2=style.blink2,
        reverse=style.reverse,
        conceal=style.conceal,
        strike=style.strike,
        underline2=style.underline2,
        frame=style.frame,
        encircle=style.encircle,
        overline=style.overline,
        link=style.link,
        meta=style.meta,
    )
