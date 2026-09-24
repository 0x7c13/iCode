# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""VisionUnsupportedDialog — modal shown when image input is unavailable."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

from rich.text import Text
from textual import on
from textual.binding import Binding
from textual.containers import VerticalGroup
from textual.widgets import Button, Static

from chrys.app.tui.i18n import render_str
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.widgets import DialogButtonRow, DialogButtonSpec
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController

USE_IMAGE_PATHS_RESULT = "use_image_paths"
VisionUnsupportedDialogResult = Literal["use_image_paths"] | None

VISION_UNSUPPORTED_TITLE = msg("tui.vision_unsupported.title", fallback="Image Input Not Available")
_VISION_USE_PATHS = msg("tui.vision_unsupported.button.use_paths", fallback="Use Paths Instead")
_VISION_OK = msg("tui.vision_unsupported.button.ok", fallback="OK")
_DEFAULT_VISION_TITLE = VISION_UNSUPPORTED_TITLE.bind()
type VisionTitle = MessageRef | str


class VisionUnsupportedDialog(BaseDialog[VisionUnsupportedDialogResult]):
    """Inform the user that image input was not attached or cannot be sent."""

    BINDINGS: ClassVar[list] = [
        Binding("escape", "dismiss_dialog", show=False, priority=True),
        Binding("left", "switch_focus", show=False),
        Binding("right", "switch_focus", show=False),
    ]

    CSS_PATH = "vision_unsupported.tcss"

    def __init__(
        self,
        message: str | Text,
        title: VisionTitle = _DEFAULT_VISION_TITLE,
        *,
        show_path_action: bool = False,
        locale_controller: LocaleController | None = None,
    ) -> None:
        self._locale_controller = locale_controller
        self._message = message
        self._title_message = title
        self._title = self._render_message(title)
        self._show_path_action = show_path_action
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="vision-unsupported-container") as container:
            container.border_title = Text(self._render_message(self._title_message))
            with VerticalGroup(id="vision-unsupported-inner"):
                yield Static(
                    self._message if isinstance(self._message, Text) else Text(self._message),
                    id="vision-unsupported-message",
                )
            specs: list[DialogButtonSpec] = []
            if self._show_path_action:
                specs.append(
                    DialogButtonSpec(
                        Text(self._render_message(_VISION_USE_PATHS.bind())),
                        id="vision-unsupported-use-paths",
                        variant="warning",
                    )
                )
            specs.append(
                DialogButtonSpec(
                    Text(self._render_message(_VISION_OK.bind())),
                    id="vision-unsupported-ok",
                    variant="primary",
                )
            )
            yield DialogButtonRow(
                *specs,
                id="vision-unsupported-buttons",
            )

    def on_mount(self) -> None:
        button_id = "#vision-unsupported-use-paths" if self._show_path_action else "#vision-unsupported-ok"
        self.query_one(button_id, Button).focus()

    @on(Button.Pressed, "#vision-unsupported-ok")
    def _on_ok(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    @on(Button.Pressed, "#vision-unsupported-use-paths")
    def _on_use_paths(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(USE_IMAGE_PATHS_RESULT)

    def action_switch_focus(self) -> None:
        if not self._show_path_action:
            return
        use_paths = self.query_one("#vision-unsupported-use-paths", Button)
        ok = self.query_one("#vision-unsupported-ok", Button)
        if use_paths.has_focus:
            ok.focus()
        else:
            use_paths.focus()

    def action_dismiss_dialog(self) -> None:
        self.dismiss(None)

    def _render_message(self, message: VisionTitle) -> str:
        if isinstance(message, str):
            return message
        controller = self._locale_controller
        return format_message(message) if controller is None else render_str(controller.localizer, message)
