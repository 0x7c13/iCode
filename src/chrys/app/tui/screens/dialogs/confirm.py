# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Generic confirmation dialog — reusable yes/no modal overlay."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on
from textual.binding import Binding
from textual.containers import VerticalGroup, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from chrys.app.tui.binding_display import CANCEL_BINDING, localized_binding
from chrys.app.tui.i18n import render_str
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.widgets import DialogButtonRow, DialogButtonSpec
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController


_CONFIRM_TITLE = msg("tui.confirm.title", fallback="Confirm")
_CONFIRM_MESSAGE = msg("tui.confirm.message", fallback="Are you sure?")
_CONFIRM_LABEL = msg("tui.confirm.button.confirm", fallback="Confirm")
_CANCEL_LABEL = msg("tui.confirm.button.cancel", fallback="Cancel")

_DEFAULT_CONFIRM_TITLE = _CONFIRM_TITLE.bind()
_DEFAULT_CONFIRM_MESSAGE = _CONFIRM_MESSAGE.bind()
_DEFAULT_CONFIRM_LABEL = _CONFIRM_LABEL.bind()
_DEFAULT_CANCEL_LABEL = _CANCEL_LABEL.bind()

type ConfirmLabel = MessageRef | str
type ConfirmMessage = ConfirmLabel | Text


class ConfirmDialog(BaseDialog[bool]):
    """Compact confirmation dialog.

    Dismisses with ``True`` (confirmed) or ``False`` (cancelled / Esc).
    """

    BINDINGS: ClassVar[list] = [
        localized_binding("escape", "cancel", CANCEL_BINDING, show=False, priority=True),
        Binding("left", "switch_focus", show=False),
        Binding("right", "switch_focus", show=False),
    ]

    CSS_PATH = "confirm.tcss"

    def __init__(
        self,
        title: ConfirmLabel = _DEFAULT_CONFIRM_TITLE,
        message: ConfirmMessage = _DEFAULT_CONFIRM_MESSAGE,
        confirm_label: ConfirmLabel = _DEFAULT_CONFIRM_LABEL,
        cancel_label: ConfirmLabel | None = _DEFAULT_CANCEL_LABEL,
        confirm_variant: str = "primary",
        *,
        locale_controller: LocaleController | None = None,
        bold_message_prefix: bool = False,
    ) -> None:
        self._locale_controller = locale_controller
        self._title_message = title
        self._message_value = message
        self._confirm_label_message = confirm_label
        self._cancel_label_message = cancel_label
        self._bold_message_prefix = bold_message_prefix
        self._title = self._render_message(title)
        self._message = self._render_dialog_message(message)
        self._confirm_label = self._render_message(confirm_label)
        self._cancel_label = self._render_message(cancel_label) if cancel_label is not None else None
        self._confirm_variant = confirm_variant
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="confirm-container") as container:
            container.border_title = Text(self._render_message(self._title_message))
            with VerticalGroup(id="confirm-inner"):
                yield from self.compose_message()
                confirm_label = self._render_message(self._confirm_label_message)
                specs = [DialogButtonSpec(Text(confirm_label), id="confirm-yes", variant=self._confirm_variant)]
                if self._cancel_label_message is not None:
                    cancel_label = self._render_message(self._cancel_label_message)
                    specs.append(DialogButtonSpec(Text(cancel_label), id="confirm-no", variant="warning"))
                yield DialogButtonRow(*specs, id="confirm-buttons")

    def compose_message(self) -> ComposeResult:
        body = self._render_dialog_message(self._message_value)
        yield Static(body if isinstance(body, Text) else Text(body), id="confirm-message")

    def _render_message(self, message: ConfirmLabel) -> str:
        if isinstance(message, str):
            return message
        controller = self._locale_controller
        return format_message(message) if controller is None else render_str(controller.localizer, message)

    def _render_dialog_message(self, message: ConfirmMessage) -> str | Text:
        if isinstance(message, Text):
            return message
        rendered = self._render_message(message)
        if not self._bold_message_prefix:
            return rendered
        text = Text(rendered)
        prefix_end = rendered.find("\n")
        text.stylize("bold", 0, len(rendered) if prefix_end < 0 else prefix_end)
        return text

    def on_mount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.register_surface(self)
        # The buttons are composed by the nested DialogButtonRow. Defer the
        # autofocus one refresh so it runs after that nested mount has
        # settled, and never let a missing button crash the app: a compose
        # interrupted by shutdown leaves the row empty.
        self.call_after_refresh(self._focus_confirm_button)

    def on_unmount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.unregister_surface(self)

    def refresh_localization(self) -> None:
        self._title = self._render_message(self._title_message)
        self._message = self._render_dialog_message(self._message_value)
        self._confirm_label = self._render_message(self._confirm_label_message)
        self.query_one("#confirm-container").border_title = Text(self._title)
        self.query_one("#confirm-message", Static).update(
            self._message if isinstance(self._message, Text) else Text(self._message)
        )
        self.query_one("#confirm-yes", Button).label = Text(self._confirm_label)
        if self._cancel_label_message is not None:
            self._cancel_label = self._render_message(self._cancel_label_message)
            self.query_one("#confirm-no", Button).label = Text(self._cancel_label)

    def _focus_confirm_button(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#confirm-yes", Button).focus()

    @on(Button.Pressed, "#confirm-yes")
    def _on_confirm(self, event: Button.Pressed) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#confirm-no")
    def _on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(False)

    def action_switch_focus(self) -> None:
        if self._cancel_label is None:
            return
        yes = self.query_one("#confirm-yes", Button)
        no = self.query_one("#confirm-no", Button)
        if yes.has_focus:
            no.focus()
        else:
            yes.focus()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def _default_dismiss_result(self) -> bool:  # type: ignore[override]
        return False


class NoticeDialog(ConfirmDialog):
    """Scrollable feedback and cancellable progress using the standard dialog controls."""

    CSS_PATH: ClassVar[list[str]] = ["confirm.tcss", "notice.tcss"]

    def compose_message(self) -> ComposeResult:
        with VerticalScroll(id="notice-scroll"):
            yield from super().compose_message()

    def finish(self) -> None:
        self.dismiss_when_topmost(False)

    def _focus_confirm_button(self) -> None:
        if self.is_attached and self.app.screen is self and not self.dismiss_requested:
            super()._focus_confirm_button()
