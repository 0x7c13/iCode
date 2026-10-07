# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow notices and progress overlays with one owner for their deferred dismissal."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chrys.app.tui.binding_display import CLOSE_BINDING
from chrys.app.tui.screens.dialogs.agent_load import AgentLoadDialog
from chrys.app.tui.screens.dialogs.confirm import NoticeDialog
from chrys.app.tui.widgets.workflow import text
from chrys.foundation.i18n import MessageRef

if TYPE_CHECKING:
    from textual.screen import Screen

    from chrys.app.tui.i18n import LocaleController
    from chrys.app.tui.widgets.workflow.panel import WorkflowPanel


_DISMISS_LABEL = CLOSE_BINDING.bind()


@dataclass(frozen=True, slots=True)
class WorkflowNoticeAction:
    label: MessageRef
    run: Callable[[], None]


@dataclass(frozen=True, slots=True)
class _Notice:
    message: str | MessageRef
    dismiss_label: MessageRef
    action: WorkflowNoticeAction | None


class WorkflowFeedback:
    def __init__(
        self,
        *,
        panel: WorkflowPanel,
        refresh: Callable[[], None],
        enabled: Callable[[], bool],
        locale: LocaleController | None,
    ) -> None:
        self._panel = panel
        self._refresh = refresh
        self._enabled = enabled
        self._locale = locale
        self.notice: NoticeDialog | None = None
        self.loading: AgentLoadDialog | None = None
        self.pending_action: Callable[[], None] | None = None
        self._notice_text = ""
        self._pending_notices: deque[_Notice] = deque()

    def render(self, message: str | MessageRef) -> str:
        return message if isinstance(message, str) else text.render(message, self._locale)

    def notify(
        self,
        message: str | MessageRef,
        *,
        dismiss_label: MessageRef = _DISMISS_LABEL,
        action: WorkflowNoticeAction | None = None,
    ) -> None:
        if isinstance(message, str):
            # Error text can carry file names and paths: keep their control characters off the terminal.
            message = text.shown(message, block=True)
        rendered = self.render(message)
        if not rendered or rendered == self._notice_text:
            return
        if all(self.render(pending.message) != rendered for pending in self._pending_notices):
            self._pending_notices.append(_Notice(message, dismiss_label, action))
            self._refresh()

    def present_notice(self, *, over: Screen | None = None) -> None:
        panel = self._panel
        if self.notice is not None or not self._pending_notices or not self._enabled():
            return
        if panel.app.screen not in (panel.screen, over):
            return
        notice = self._pending_notices.popleft()
        dialog = NoticeDialog(
            title=text.TITLE.bind(),
            message=notice.message,
            confirm_label=notice.action.label if notice.action is not None else notice.dismiss_label,
            cancel_label=notice.dismiss_label if notice.action is not None else None,
            locale_controller=self._locale,
        )
        self.notice, self._notice_text = dialog, self.render(notice.message)

        def closed(accepted: bool | None) -> None:
            if self.notice is dialog:
                self.notice, self._notice_text = None, ""
                if accepted and notice.action is not None and self._enabled():
                    notice.action.run()
                self._refresh()

        panel.app.push_screen(dialog, closed)

    def clear(self) -> None:
        self._pending_notices.clear()
        self.pending_action = None
        dialog, self.notice = self.notice, None
        self._notice_text = ""
        if dialog is not None:
            dialog.finish()
        self.close_loading()

    def present_pending(self) -> bool:
        action, self.pending_action = self.pending_action, None
        if action is None:
            return False
        action()
        return True

    def close_loading(self) -> None:
        dialog, self.loading = self.loading, None
        if dialog is not None:
            dialog.request_dismiss()

    def show_loading(self, *, title: MessageRef, message: MessageRef, phase: str) -> None:
        self.close_loading()
        dialog = AgentLoadDialog(title=title, locale_controller=self._locale)
        dialog.update_progress(message, phase=phase)
        self.loading = dialog

        def closed(_result: None) -> None:
            if self.loading is dialog:
                self.loading = None
            self._refresh()

        self._panel.app.push_screen(dialog, closed)
