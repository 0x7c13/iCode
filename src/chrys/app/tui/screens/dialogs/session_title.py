# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session title editor — set or clear the session's custom title."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on
from textual.containers import VerticalGroup
from textual.widgets import Button, Static

from chrys.app.tui.binding_display import CANCEL_BINDING, localized_binding
from chrys.app.tui.i18n import render_str
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.widgets import DialogButtonRow, DialogButtonSpec, EnhancedInput
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController


_HINT = msg(
    "tui.session_title.hint",
    fallback="A custom title for this session. Leave empty for the auto-generated title.",
)
_TITLE = msg("tui.session_title.title", fallback="Session Title")
_PLACEHOLDER = msg("tui.session_title.placeholder", fallback="Session title")
_SAVE = msg("tui.session_title.button.save", fallback="Save")
_CANCEL = msg("tui.session_title.button.cancel", fallback="Cancel")


class SessionTitleDialog(BaseDialog[str | None]):
    """Edit the current session's custom title.

    Dismisses with the new custom title (an empty string clears it back to
    the automatic title) or ``None`` when cancelled.
    """

    BINDINGS: ClassVar[list] = [
        localized_binding("escape", "cancel", CANCEL_BINDING, show=False, priority=True),
    ]

    CSS_PATH = "session_title.tcss"

    def __init__(
        self,
        *,
        custom_title: str = "",
        auto_title: str = "",
        locale_controller: LocaleController | None = None,
    ) -> None:
        self._locale_controller = locale_controller
        self._custom_title = custom_title
        self._auto_title = auto_title
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="session-title-container") as container:
            container.border_title = Text(self._render_message(_TITLE.bind()))
            with VerticalGroup(id="session-title-inner"):
                yield Static(Text(self._render_message(_HINT.bind())), id="session-title-hint")
                yield EnhancedInput(
                    value=self._custom_title,
                    placeholder=self._auto_title or self._render_message(_PLACEHOLDER.bind()),
                    id="session-title-input",
                )
                yield DialogButtonRow(
                    DialogButtonSpec(
                        Text(self._render_message(_SAVE.bind())),
                        id="session-title-save",
                        variant="primary",
                    ),
                    DialogButtonSpec(
                        Text(self._render_message(_CANCEL.bind())),
                        id="session-title-cancel",
                        variant="warning",
                    ),
                    id="session-title-buttons",
                )

    def on_mount(self) -> None:
        self.query_one("#session-title-input", EnhancedInput).focus()

    @on(EnhancedInput.Submitted, "#session-title-input")
    def _on_input_submitted(self, event: EnhancedInput.Submitted) -> None:
        event.stop()
        self._submit()

    @on(Button.Pressed, "#session-title-save")
    def _on_save(self, _event: Button.Pressed) -> None:
        self._submit()

    @on(Button.Pressed, "#session-title-cancel")
    def _on_cancel(self, _event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _submit(self) -> None:
        self.dismiss(self.query_one("#session-title-input", EnhancedInput).value.strip())

    def _render_message(self, reference: MessageRef) -> str:
        controller = self._locale_controller
        return format_message(reference) if controller is None else render_str(controller.localizer, reference)
