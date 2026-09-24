# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AskUserDialog — modal dialog for agent ask_user questions."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import events, on
from textual.binding import Binding
from textual.containers import VerticalGroup
from textual.screen import ModalScreen

from chrys.app.tui.behaviors.insert_clipboard import INSERT_CLIPBOARD_BINDINGS, InsertClipboardScreenMixin
from chrys.app.tui.behaviors.right_click_copy import RightClickScreenCopyMixin
from chrys.app.tui.widgets import (
    AskUserActiveQuestionChanged,
    AskUserInlineRequested,
    AskUserPrompt,
    AskUserSubmitted,
    PromptDraft,
)
from chrys.foundation.models.ask_user import AskUserAnswer, AskUserQuestion

if TYPE_CHECKING:
    from textual.app import ComposeResult


@dataclass(frozen=True, slots=True)
class AskUserInlineResult:
    """Modal result indicating the prompt should move into the chat transcript."""

    request_id: str
    draft: PromptDraft


AskUserDialogResult = tuple[str, tuple[AskUserAnswer, ...]] | AskUserInlineResult | None


class AskUserDialog(RightClickScreenCopyMixin, InsertClipboardScreenMixin, ModalScreen[AskUserDialogResult]):
    """Modal multi-question prompt with a timeout-only dismissal channel."""

    BINDINGS: ClassVar[list] = [
        Binding("escape", "noop", show=False, priority=True),
        *INSERT_CLIPBOARD_BINDINGS,
    ]

    CSS_PATH = "ask_user.tcss"

    def __init__(
        self,
        request_id: str,
        questions: tuple[AskUserQuestion, ...],
        caller_name: str = "",
        draft: PromptDraft | None = None,
        allow_inline: bool = True,
    ) -> None:
        self._request_id = request_id
        self._questions = questions
        self._caller_name = caller_name
        self._draft = draft
        self._allow_inline = allow_inline
        self._dismiss_requested = False
        self._dismiss_on_resume = False
        self._dismiss_result: AskUserDialogResult = None
        self._dismissed = False
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="askuser-container") as container:
            if self._caller_name:
                container.border_subtitle = Text(self._caller_name)
            yield AskUserPrompt(
                self._request_id,
                self._questions,
                inline=False,
                allow_inline=self._allow_inline,
                draft=self._draft,
            )

    def on_mount(self) -> None:
        self._refresh_title()
        if self._dismiss_requested:
            # The dialog counts as mounted only once its mount handlers have returned.
            self.call_later(self._dismiss_if_top)

    def _refresh_title(self) -> None:
        with contextlib.suppress(Exception):
            prompt = self.query_one(AskUserPrompt)
            self.query_one("#askuser-container").border_title = prompt.active_title()

    def _safe_dismiss(self, result: AskUserDialogResult) -> None:
        """Dismiss once and hide immediately so timeout/user clicks cannot race."""
        if self._dismiss_requested:
            return
        self._dismiss_requested = True
        self._dismiss_result = result
        with contextlib.suppress(Exception):
            self.query_one("#askuser-container").display = False
        self._dismiss_if_top()

    def _dismiss_if_top(self) -> None:
        if self._dismissed:
            return
        if not self.is_mounted:
            # Removing a dialog that is still composing mounts its widgets without their children.
            return
        is_top = False
        with contextlib.suppress(Exception):
            is_top = self.app.screen is self
        if not is_top:
            self._dismiss_on_resume = True
            return
        self._dismiss_on_resume = False
        self._dismissed = True
        self.dismiss(self._dismiss_result)

    def on_screen_resume(self, _event: events.ScreenResume) -> None:
        if self._dismiss_on_resume:
            self._dismiss_on_resume = False
            self._dismiss_if_top()

    def dismiss_due_to_timeout(self) -> None:
        """Close the dialog because the backend ask-user request expired."""
        self._safe_dismiss(None)

    @on(AskUserActiveQuestionChanged)
    def _on_active_question_changed(self, event: AskUserActiveQuestionChanged) -> None:
        if event.request_id == self._request_id:
            self._refresh_title()

    @on(AskUserSubmitted)
    def _on_ask_user_submitted(self, event: AskUserSubmitted) -> None:
        event.stop()
        self._safe_dismiss((event.request_id, event.answers))

    @on(AskUserInlineRequested)
    def _on_ask_user_inline_requested(self, event: AskUserInlineRequested) -> None:
        event.stop()
        if isinstance(event.draft, PromptDraft):
            self._safe_dismiss(AskUserInlineResult(event.request_id, event.draft))

    def action_noop(self) -> None:
        """Swallow Esc — user must explicitly respond."""
