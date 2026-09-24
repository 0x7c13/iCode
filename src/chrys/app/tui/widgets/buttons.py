# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared TUI button variants."""

from __future__ import annotations

import contextlib
from typing import Any

from rich.text import Text
from textual import events
from textual.content import Content
from textual.dom import NoScreen
from textual.widgets import Button


class ConfigActionButton(Button):
    """Button used by configuration screens.

    Textual's default mouse click leaves focus/hover state visible after
    click. Clear both only for the mouse path, leaving keyboard activation
    and programmatic ``press()`` focus behavior intact.
    """

    async def _on_click(self, event: events.Click) -> None:
        event.stop()
        # We call press() ourselves; suppress Textual's later Button._on_click MRO handler.
        event.prevent_default()
        if not self.has_class("-active"):
            super().press()
        with contextlib.suppress(NoScreen):
            self.screen.set_focus(None)
        self.mouse_hover = False


class ConfigAddButton(ConfigActionButton):
    """Add-style button used by configuration screens."""

    def __init__(
        self,
        label: Content | Text | str = "+ Add",
        *,
        compact: bool = False,
        classes: str | None = None,
        **kwargs: Any,
    ) -> None:
        shared_classes = "config-add-button"
        if compact:
            shared_classes = f"{shared_classes} config-add-button-compact"
        if classes:
            shared_classes = f"{classes} {shared_classes}"
        super().__init__(label, classes=shared_classes, **kwargs)
