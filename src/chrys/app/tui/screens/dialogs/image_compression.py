# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ImageCompressionDialog — modal shown while image attachments are compressed."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.containers import VerticalGroup
from textual.screen import ModalScreen

from chrys.app.tui.behaviors.insert_clipboard import INSERT_CLIPBOARD_BINDINGS, InsertClipboardScreenMixin
from chrys.app.tui.i18n import render_str
from chrys.app.tui.widgets import ChrysLoadingIndicator
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController

type ImageCompressionTitle = MessageRef | str

_PREPARING_IMAGE = msg("tui.image_compression.default_title", fallback="Preparing Image")
_DEFAULT_TITLE = _PREPARING_IMAGE.bind()


class ImageCompressionDialog(InsertClipboardScreenMixin, ModalScreen[None]):
    """Show a compact loading-only modal while image compression is running."""

    BINDINGS: ClassVar[list] = [
        Binding("escape", "ignore_escape", show=False, priority=True),
        *INSERT_CLIPBOARD_BINDINGS,
    ]

    CSS_PATH = "image_compression.tcss"

    def __init__(
        self,
        title: ImageCompressionTitle = _DEFAULT_TITLE,
        *,
        locale_controller: LocaleController | None = None,
    ) -> None:
        self._locale_controller = locale_controller
        self._title_message = title
        self._title = self._render_message(title)
        self._mounted = False
        self._dismiss_pending = False
        self._dismiss_on_resume = False
        self._dismissed = False
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="image-compression-container") as container:
            container.border_title = Text(self._render_message(self._title_message))
            with VerticalGroup(id="image-compression-inner"):
                yield ChrysLoadingIndicator(id="image-compression-loading")

    def on_mount(self) -> None:
        self._mounted = True
        if self._dismiss_pending:
            self._safe_dismiss()

    def finish(self) -> None:
        """Dismiss after mounting, or remember the finish if mounting is still pending."""
        # Owners may still report after removal starts, which prunes the children before this
        # dialog unmounts. Buffer as before mount; nothing will render it.
        if not self._mounted or self._pruning or self._closing:
            self._dismiss_pending = True
            return
        self._safe_dismiss()

    def _safe_dismiss(self) -> None:
        if self._dismissed:
            return
        if not (self.is_attached and self.app.screen is self):
            if self.is_attached:
                self.query_one("#image-compression-container", VerticalGroup).display = False
            self._dismiss_on_resume = True
            return
        self._dismiss_on_resume = False
        self._dismissed = True
        self.dismiss(None)

    def on_screen_resume(self, _event: events.ScreenResume) -> None:
        if self._dismiss_on_resume:
            self._dismiss_on_resume = False
            self._safe_dismiss()

    def action_ignore_escape(self) -> None:
        """Keep compression modal open until backend preparation finishes."""
        return

    def _render_message(self, message: ImageCompressionTitle) -> str:
        if isinstance(message, str):
            return message
        controller = self._locale_controller
        return format_message(message) if controller is None else render_str(controller.localizer, message)
