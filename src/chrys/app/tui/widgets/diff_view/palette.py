# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The colors of a diff, and how the app's present look selects among them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from chrys.app.tui.widgets import hatch_style
from chrys.app.tui.widgets.diff_view.rows import RowKind

if TYPE_CHECKING:
    from textual.style import Style as VisualStyle
    from textual.widget import Widget

REMOVED_EMPHASIS = "on $error 40%"
"""Marks the characters of a removed line that the line replacing it does not have."""
ADDED_EMPHASIS = "on $success 40%"
"""Marks the characters of an added line that the line it replaces did not have."""


@dataclass(frozen=True, slots=True)
class DiffPalette:
    """Styles for a row's line number, the edge bar ahead of it, and the line of code, by row kind."""

    number: Mapping[RowKind, str]
    edge: Mapping[RowKind, str]
    line: Mapping[RowKind, str]
    line_in_256_colors: Mapping[RowKind, str] = field(default_factory=dict)
    """Backgrounds to use instead on a 256-color terminal, where the nearest match Rich would pick
    for a subtle tint comes out grey. These are exact xterm colors."""

    def line_style(self, kind: RowKind, color_system: str | None) -> str:
        if color_system == "256" and kind in self.line_in_256_colors:
            return self.line_in_256_colors[kind]
        return self.line.get(kind, "")


DARK = DiffPalette(
    number={
        RowKind.ADDED: "$text-success on #1f4d2a",
        RowKind.REMOVED: "$text-error on #5a2628",
        RowKind.CONTEXT: "$foreground 30%",
    },
    edge={
        RowKind.ADDED: "$success on #1f4d2a",
        RowKind.REMOVED: "$error on #5a2628",
        RowKind.CONTEXT: "$foreground 15%",
    },
    line={RowKind.ADDED: "on #243f30", RowKind.REMOVED: "on #512b35"},
    line_in_256_colors={RowKind.ADDED: "on #005F00", RowKind.REMOVED: "on #5F0000"},
)

LIGHT = DiffPalette(
    number={
        RowKind.ADDED: "#1f7a3a on #D4ECD9",
        RowKind.REMOVED: "#8f1f25 on #F0D4D8",
        RowKind.CONTEXT: "$foreground 30%",
    },
    edge={
        RowKind.ADDED: "#2E8B57 on #D4ECD9",
        RowKind.REMOVED: "#C75261 on #F0D4D8",
        RowKind.CONTEXT: "$foreground 15%",
    },
    line={RowKind.ADDED: "on #d8f0dc", RowKind.REMOVED: "on #F8E8EA"},
)


@dataclass(frozen=True, slots=True)
class DiffLook:
    """What drawing a row depends on beyond the row: resolved once per look of the app, not per row."""

    palette: DiffPalette
    color_system: str | None
    hatch: VisualStyle

    @classmethod
    def of(cls, widget: Widget) -> DiffLook:
        """The look for ``widget``, which is mounted and declares the ``hatch--pattern`` component."""
        theme = widget.app.current_theme
        palette = DARK if theme is None or theme.dark else LIGHT
        return cls(palette, widget.app.console.color_system, hatch_style(widget))

    def line_style(self, kind: RowKind) -> str:
        return self.palette.line_style(kind, self.color_system)
