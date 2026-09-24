# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Diff widgets: two texts compared, with syntax highlighting and the edits within lines marked.

`DiffView` is the viewer, split or unified, scrolling under its own scrollbars and drawing only
the rows on screen. `chrys.app.tui.widgets.diff_view.inline` has the single-widget unified diff
that the chat embeds.
"""

from chrys.app.tui.widgets.diff_view.code import CodeColumn
from chrys.app.tui.widgets.diff_view.gutter import GutterColumn
from chrys.app.tui.widgets.diff_view.protocols import SupportsPrepare
from chrys.app.tui.widgets.diff_view.widget import DiffView

__all__ = [
    "CodeColumn",
    "DiffView",
    "GutterColumn",
    "SupportsPrepare",
]
