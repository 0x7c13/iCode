# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared click-consuming affordance widgets."""

from __future__ import annotations

from typing import ClassVar

from textual.events import Click
from textual.message import Message
from textual.widgets import Static


class ClickAffordance(Static):
    """Static pseudo-button that owns Chrys' click consumption invariant."""

    CLICK_MESSAGE: ClassVar[type[Message] | None] = None

    def on_click(self, event: Click) -> None:
        event.prevent_default()
        event.stop()
        message_type = self.CLICK_MESSAGE
        if message_type is not None:
            self.post_message(message_type())
