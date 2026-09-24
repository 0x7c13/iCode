# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A docked theme editor that shares the main screen's layout and live preview."""

from __future__ import annotations

from collections.abc import Callable

from textual import on
from textual.app import ComposeResult
from textual.theme import Theme
from textual.widget import Widget

from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog
from chrys.app.tui.support.gc_freeze import GcReclaimReason, GcReclaimRequested
from chrys.app.tui.themes.store import ThemeFileRevision, UserThemeStore

from . import messages as M
from .editor import ResettableThemeEditor


class ThemeEditorPanel(Widget):
    """Own the draft while the existing chat remains mounted beside it."""

    def __init__(
        self,
        theme: Theme,
        store: UserThemeStore,
        revision: ThemeFileRevision | None = None,
        *,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.editor = ResettableThemeEditor(theme, store, revision)
        self._confirming_close = False
        self._editor_closing = False
        self._on_close = on_close
        self._after_close: Callable[[], None] | None = None

    def compose(self) -> ComposeResult:
        yield self.editor

    def resume(self) -> None:
        self.editor.resume()

    @on(ResettableThemeEditor.CloseRequested)
    def _close_requested(self, event: ResettableThemeEditor.CloseRequested) -> None:
        event.stop()
        self.action_close()

    def action_close(self) -> None:
        self.request_close()

    def request_close(self, *, after_close: Callable[[], None] | None = None) -> None:
        """Run the continuation only after an accepted close has unmounted the editor."""
        if self._confirming_close or self._editor_closing or self.editor.delete_pending:
            return
        self._after_close = after_close
        if not self.editor.document.unsaved:
            self._remove_panel()
            return
        self._confirming_close = True

        def decided(discard: bool | None) -> None:
            self._confirming_close = False
            if discard:
                self.app.call_after_refresh(self._remove_panel)
            else:
                self._after_close = None

        self.app.push_screen(
            ConfirmDialog(
                title=M.DISCARD_TITLE.bind(),
                message=M.DISCARD_BODY.bind(),
                confirm_label=M.DISCARD.bind(),
                cancel_label=M.CANCEL.bind(),
                confirm_variant="error",
                locale_controller=self.editor.host.locale_controller,
            ),
            decided,
        )

    def on_unmount(self) -> None:
        if self.app.is_running:
            self.app.call_after_refresh(self._finish_close)

    def _finish_close(self) -> None:
        if self._on_close is not None:
            self._on_close()
        after_close, self._after_close = self._after_close, None
        if after_close is not None:
            after_close()

    def _remove_panel(self) -> None:
        if self._editor_closing:
            return
        self._editor_closing = True
        # The App owns this callback: the panel's message pump must be free to
        # finish unmounting its children before we request their reclamation.
        if not self.app.call_later(self._remove_and_reclaim):
            self._editor_closing = False
            self._after_close = None

    async def _remove_and_reclaim(self) -> None:
        app = self.app
        parent = self.parent
        await self.remove()
        if isinstance(parent, Widget):
            # NodeList's cached displayed children can retain a removed dock
            # even after reflow. Drop parent-owned references before reclaiming.
            parent._clear_arrangement_cache()
            _ = parent.displayed_children
        app.post_message(GcReclaimRequested(GcReclaimReason.STABLE_CONTENT_REMOVED, prompt=False))
