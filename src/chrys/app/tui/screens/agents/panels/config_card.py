# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared card chrome for agent configuration panels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Label

if TYPE_CHECKING:
    from textual.app import ComposeResult


class ConfigCard(Vertical):
    """Base card with shared visual chrome and delete-message behavior."""

    DEFAULT_CSS = """
    ConfigCard {
        height: auto;
        background: $foreground 5%;
        border: solid $tui-border-foreground 15%;
        padding: 1 2;
        margin: 0 2 1 0;
    }
    ConfigCard .config-card-title {
        width: 1fr;
        height: 1;
        color: $secondary;
        text-style: bold;
        content-align: left middle;
    }
    ConfigCard Button.config-card-delete-btn.-style-default.-error {
        min-width: 3;
        border: none;
        background: transparent;
        color: $error;
        content-align: center middle;
        text-align: center;
        offset-x: 1;
    }
    """

    _delete_button_prefix: str = ""

    class Removed(Message):
        """Posted when the user clicks delete."""

        def __init__(self, index: int) -> None:
            self.index = index
            super().__init__()

    def __init__(self, index: int, *, classes: str | None = None, read_only: bool = False) -> None:
        self._index = index
        self._read_only = read_only
        card_classes = "agent-config-card"
        if classes:
            card_classes = f"{classes} {card_classes}"
        super().__init__(classes=card_classes)

    @property
    def _delete_button_id(self) -> str:
        return f"{self._delete_button_prefix}-{self._index}"

    def compose_header(self, title: str, *, row_class: str, title_class: str) -> ComposeResult:
        """Yield a standard card header with the subclass's existing classes."""
        with Horizontal(classes=f"{row_class} config-card-header-row"):
            yield Label(title, classes=f"{title_class} config-card-title")
            delete_button = Button("✕", variant="error", id=self._delete_button_id, classes="config-card-delete-btn")
            if self._read_only:
                delete_button.disabled = True
                delete_button.display = False
            yield delete_button

    def _removed_message(self) -> Message:
        """Build the remove message for this card."""
        return self.Removed(self._index)

    @on(Button.Pressed)
    def _on_config_card_button_pressed(self, event: Button.Pressed) -> None:
        if self._read_only:
            return
        if event.button.id != self._delete_button_id:
            return
        self.call_later(self.post_message, self._removed_message())
