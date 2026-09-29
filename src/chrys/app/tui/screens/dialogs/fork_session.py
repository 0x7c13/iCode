# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ForkSessionDialog — choose what to do after creating a session fork."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from rich.text import Text
from textual import on
from textual.binding import Binding
from textual.containers import HorizontalGroup, VerticalGroup
from textual.widgets import Button, Static

from chrys.app.tui.i18n import render_str
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.widgets import ChrysLoadingIndicator, DialogButtonRow, DialogButtonSpec
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController

ForkSessionResult = Literal["stay", "switch", "new_window"]
ForkSessionState = Literal["loading", "success", "error"]
type ForkSessionMessage = MessageRef | str

_FORK_CREATING = msg("tui.fork_session.creating", fallback="Creating session fork...")
_FORK_CREATED = msg("tui.fork_session.created", fallback="Created fork {fork_short_id}.")
_FORK_TITLE_LOADING = msg("tui.fork_session.title.loading", fallback="Forking Session")
_FORK_TITLE_ERROR = msg("tui.fork_session.title.error", fallback="Fork Failed")
_FORK_TITLE_SUCCESS = msg("tui.fork_session.title.success", fallback="Session Forked")
_FORK_SWITCH = msg("tui.fork_session.button.switch", fallback="Switch")
_FORK_OPEN_NEW_WINDOW = msg("tui.fork_session.button.open_new_window", fallback="Open New Window")
_FORK_STAY = msg("tui.fork_session.button.stay", fallback="Stay")
_FORK_OK = msg("tui.fork_session.button.ok", fallback="Ok")


class ForkSessionDialog(BaseDialog[ForkSessionResult | None]):
    """Dialog shown after the backend creates a forked session."""

    BINDINGS: ClassVar[list] = [
        Binding("escape", "dismiss_after_result", show=False, priority=True),
        Binding("left", "focus_previous", show=False),
        Binding("right", "focus_next", show=False),
    ]

    CSS_PATH = "fork_session.tcss"

    def __init__(
        self,
        fork_short_id: str = "",
        *,
        show_new_window: bool = True,
        locale_controller: LocaleController | None = None,
    ) -> None:
        self._locale_controller = locale_controller
        self._fork_short_id = fork_short_id
        self._show_new_window = show_new_window
        self._state: ForkSessionState = "success" if fork_short_id else "loading"
        self._message_value: ForkSessionMessage = (
            _FORK_CREATED.bind(fork_short_id=fork_short_id) if fork_short_id else _FORK_CREATING.bind()
        )
        self._message = self._render_message(self._message_value)
        self._mounted = False
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="fork-session-container") as container:
            container.border_title = Text(self._title)
            with VerticalGroup(id="fork-session-inner"):
                yield ChrysLoadingIndicator(id="fork-session-loading")
                yield Static(Text(self._render_message(self._message_value)), id="fork-session-message")
                success_specs = [
                    DialogButtonSpec(
                        Text(self._render_message(_FORK_SWITCH.bind())),
                        id="fork-session-switch",
                        variant="primary",
                    )
                ]
                if self._show_new_window:
                    success_specs.append(
                        DialogButtonSpec(
                            Text(self._render_message(_FORK_OPEN_NEW_WINDOW.bind())),
                            id="fork-session-new-window",
                            variant="success",
                        )
                    )
                success_specs.append(
                    DialogButtonSpec(
                        Text(self._render_message(_FORK_STAY.bind())),
                        id="fork-session-stay",
                        variant="warning",
                    )
                )
                yield DialogButtonRow(*success_specs, id="fork-session-buttons")
                yield DialogButtonRow(
                    DialogButtonSpec(
                        Text(self._render_message(_FORK_OK.bind())),
                        id="fork-session-ok",
                        variant="warning",
                    ),
                    id="fork-session-error-buttons",
                )

    def on_mount(self) -> None:
        self._mounted = True
        self._apply_state()

    @property
    def _title(self) -> str:
        return self._render_message(self._title_message)

    @property
    def _title_message(self) -> MessageRef:
        if self._state == "loading":
            return _FORK_TITLE_LOADING.bind()
        if self._state == "error":
            return _FORK_TITLE_ERROR.bind()
        return _FORK_TITLE_SUCCESS.bind()

    def set_success(self, fork_short_id: str) -> None:
        """Swap the loading state for the success action row."""
        self._fork_short_id = fork_short_id
        self._state = "success"
        self._message_value = _FORK_CREATED.bind(fork_short_id=fork_short_id)
        self._message = self._render_message(self._message_value)
        self._apply_state()

    def set_error(self, message: str) -> None:
        """Swap the loading state for an error result."""
        self._state = "error"
        self._message_value = message
        self._message = self._render_message(self._message_value)
        self._apply_state()

    def _apply_state(self) -> None:
        # Owners may still report after removal starts, which prunes the children before this
        # dialog unmounts. Buffer as before mount; nothing will render it.
        if not self._mounted or self._pruning or self._closing:
            return

        self.remove_class("-loading", "-success", "-error")
        self.add_class(f"-{self._state}")
        self.query_one("#fork-session-container", VerticalGroup).border_title = Text(
            self._render_message(self._title_message)
        )
        self.query_one("#fork-session-loading", ChrysLoadingIndicator).display = self._state == "loading"
        self.query_one("#fork-session-message", Static).update(Text(self._render_message(self._message_value)))
        self.query_one("#fork-session-buttons", HorizontalGroup).display = self._state == "success"
        self.query_one("#fork-session-error-buttons", HorizontalGroup).display = self._state == "error"
        if self._state == "success":
            self.query_one("#fork-session-switch", Button).focus()
        elif self._state == "error":
            self.query_one("#fork-session-ok", Button).focus()

    @on(Button.Pressed, "#fork-session-stay")
    def _on_stay(self, event: Button.Pressed) -> None:
        event.stop()
        if self._state != "success":
            return
        self.dismiss("stay")

    @on(Button.Pressed, "#fork-session-switch")
    def _on_switch(self, event: Button.Pressed) -> None:
        event.stop()
        if self._state != "success":
            return
        self.dismiss("switch")

    @on(Button.Pressed, "#fork-session-new-window")
    def _on_new_window(self, event: Button.Pressed) -> None:
        event.stop()
        if self._state != "success" or not self._show_new_window:
            return
        self.dismiss("new_window")

    @on(Button.Pressed, "#fork-session-ok")
    def _on_ok(self, event: Button.Pressed) -> None:
        event.stop()
        if self._state == "loading":
            return
        self.dismiss(None)

    def action_dismiss_after_result(self) -> None:
        if self._state == "loading":
            return
        self.dismiss(self._default_dismiss_result())

    def action_focus_previous(self) -> None:
        self._focus_relative(-1)

    def action_focus_next(self) -> None:
        self._focus_relative(1)

    def _focus_relative(self, offset: int) -> None:
        buttons = [button for button in self.query(Button) if self._button_is_visible(button)]
        if not buttons:
            return
        current = next((index for index, button in enumerate(buttons) if button.has_focus), 0)
        buttons[(current + offset) % len(buttons)].focus()

    @staticmethod
    def _button_is_visible(button: Button) -> bool:
        return button.display and button.parent is not None and button.parent.display

    def _allow_click_outside_dismiss(self) -> bool:
        return self._state != "loading"

    def _default_dismiss_result(self) -> ForkSessionResult | None:  # type: ignore[override]
        return "stay" if self._state == "success" else None

    def _render_message(self, message: ForkSessionMessage) -> str:
        if isinstance(message, str):
            return message
        controller = self._locale_controller
        return format_message(message) if controller is None else render_str(controller.localizer, message)
