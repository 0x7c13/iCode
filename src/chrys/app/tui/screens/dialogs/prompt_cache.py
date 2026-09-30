# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""PromptCacheDialog — save-time reminder that a Claude profile has prompt caching off."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar, Literal

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
from chrys.service.profiles.models.options import ANTHROPIC_PROMPT_CACHE_CONTROL

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController

ADD_AND_SAVE_RESULT = "add_and_save"
SAVE_AS_IS_RESULT = "save_as_is"
PromptCacheDialogResult = Literal["add_and_save", "save_as_is"] | None

_TITLE = msg("tui.prompt_cache.title", fallback="Prompt Caching Is Off")
_MESSAGE = msg(
    "tui.prompt_cache.message",
    fallback=(
        "Claude models on the Anthropic protocol cache a prompt only when the request asks for it, "
        "and this profile doesn't. Without caching, every model call pays full price for the whole "
        "conversation again.\n\n"
        "Add and Save puts {option} into the extra_body chat option. Reading from the cache costs "
        "a tenth of the normal input price or less; writing to it costs 25% more. If your endpoint "
        "rejects the field, remove it from extra_body."
    ),
    multiline=True,
)
# Bound as ``{option}``: catalog text holds named slots only, never literal braces.
# Rendered from the value Add and Save writes, so the two cannot drift apart.
_CACHE_CONTROL_OPTION = f'"cache_control": {json.dumps(dict(ANTHROPIC_PROMPT_CACHE_CONTROL))}'
_ADD_AND_SAVE = msg("tui.prompt_cache.button.add_and_save", fallback="Add and Save")
_SAVE_AS_IS = msg("tui.prompt_cache.button.save_as_is", fallback="Save as Is")


class PromptCacheDialog(BaseDialog[PromptCacheDialogResult]):
    """Offer to turn on Anthropic prompt caching before saving a Claude profile.

    Dismisses with ``ADD_AND_SAVE_RESULT``, ``SAVE_AS_IS_RESULT``, or ``None``
    (Esc or a backdrop click): back to the form, nothing saved.
    """

    BINDINGS: ClassVar[list] = [
        localized_binding("escape", "cancel", CANCEL_BINDING, show=False, priority=True),
        Binding("left", "switch_focus", show=False),
        Binding("right", "switch_focus", show=False),
    ]

    CSS_PATH = "prompt_cache.tcss"

    def __init__(self, *, locale_controller: LocaleController | None = None) -> None:
        self._locale_controller = locale_controller
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="prompt-cache-container") as container:
            container.border_title = Text(self._render_message(_TITLE.bind()))
            # Scrolls when a short terminal leaves the text less room than it needs.
            with VerticalScroll(id="prompt-cache-inner"):
                yield Static(Text(self._message_text()), id="prompt-cache-message")
            yield DialogButtonRow(
                DialogButtonSpec(
                    Text(self._render_message(_ADD_AND_SAVE.bind())),
                    id="prompt-cache-add",
                    variant="primary",
                ),
                DialogButtonSpec(
                    Text(self._render_message(_SAVE_AS_IS.bind())),
                    id="prompt-cache-save",
                    variant="warning",
                ),
                id="prompt-cache-buttons",
            )

    def on_mount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.register_surface(self)
        # The buttons belong to the nested DialogButtonRow; focus once it has mounted.
        self.call_after_refresh(self._focus_add_button)

    def on_unmount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.unregister_surface(self)

    def refresh_localization(self) -> None:
        self.query_one("#prompt-cache-container").border_title = Text(self._render_message(_TITLE.bind()))
        self.query_one("#prompt-cache-message", Static).update(Text(self._message_text()))
        self.query_one("#prompt-cache-add", Button).label = Text(self._render_message(_ADD_AND_SAVE.bind()))
        self.query_one("#prompt-cache-save", Button).label = Text(self._render_message(_SAVE_AS_IS.bind()))

    def _focus_add_button(self) -> None:
        try:
            self.query_one("#prompt-cache-add", Button).focus()
        except NoMatches:
            return

    @on(Button.Pressed, "#prompt-cache-add")
    def _on_add(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(ADD_AND_SAVE_RESULT)

    @on(Button.Pressed, "#prompt-cache-save")
    def _on_save_as_is(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(SAVE_AS_IS_RESULT)

    def action_switch_focus(self) -> None:
        add = self.query_one("#prompt-cache-add", Button)
        save = self.query_one("#prompt-cache-save", Button)
        if add.has_focus:
            save.focus()
        else:
            add.focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _default_dismiss_result(self) -> PromptCacheDialogResult:
        return None

    def _message_text(self) -> str:
        return self._render_message(_MESSAGE.bind(option=_CACHE_CONTROL_OPTION))

    def _render_message(self, message: MessageRef) -> str:
        controller = self._locale_controller
        return format_message(message) if controller is None else render_str(controller.localizer, message)
