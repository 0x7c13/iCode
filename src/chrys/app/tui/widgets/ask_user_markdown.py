# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Deferred Markdown widget for ask-user question panes."""

from __future__ import annotations

from textual.geometry import Size

from chrys.app.tui.widgets.markdown import VirtualizedMarkdown


class AskUserQuestionMarkdown(VirtualizedMarkdown, can_focus=False):
    """Question Markdown that follows its virtual content height."""

    def watch_virtual_size(self, _old: Size, new: Size) -> None:
        if new.height > 0:
            self.styles.height = new.height
