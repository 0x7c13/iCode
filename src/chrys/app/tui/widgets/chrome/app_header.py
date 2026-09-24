# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reusable app header widget with app title and optional subtitle."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.content import Content
from textual.events import Click
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from chrys.app.tui.i18n import render_str
from chrys.app.tui.widgets.selection import NonSelectableStatic
from chrys.app.tui.widgets.workflow import text as workflow_text
from chrys.foundation.branding import format_app_version_title
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message
from chrys.service.approval.policy import ApprovalMode

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController
    from chrys.foundation.i18n import Localizer

_APPROVAL_BADGE = msg(
    "tui.chrome.approval_mode.badge",
    fallback=" APPROVAL MODE: {mode} ",
)
_APPROVAL_MODE_MANUAL = msg("tui.chrome.approval_mode.manual", fallback="MANUAL")
_APPROVAL_MODE_AUTO = msg("tui.chrome.approval_mode.auto", fallback="AUTO")
_APPROVAL_MODE_BYPASS = msg("tui.chrome.approval_mode.bypass", fallback="BYPASS")

_APPROVAL_CLASSES = {
    ApprovalMode.MANUAL: "approval-manual",
    ApprovalMode.AUTO: "approval-auto",
    ApprovalMode.BYPASS: "approval-bypass",
}

APPROVAL_MODE_MESSAGES = {
    ApprovalMode.MANUAL: _APPROVAL_MODE_MANUAL,
    ApprovalMode.AUTO: _APPROVAL_MODE_AUTO,
    ApprovalMode.BYPASS: _APPROVAL_MODE_BYPASS,
}


class _HeaderBadge(NonSelectableStatic):
    """Clickable header chip excluded from drag selection and copy."""


class AppHeader(Widget):
    """Top-of-screen header bar showing the application title and optional subtitle.

    Uses a horizontal layout with a centered title (``1fr``) and a
    right-aligned approval mode badge when enabled.

    Usage::

        header = AppHeader()
        yield header

        # Later, update with profile/model info:
        header.set_subtitle("Code Agent", "claude-sonnet-4-6")
    """

    class ApprovalBadgeClicked(Message):
        """Posted when the approval mode badge is clicked."""

    class ModeClicked(Message):
        """Switch between chat and workflow presentation."""

    ALLOW_SELECT = False

    subtitle_parts: reactive[tuple[str, ...]] = reactive(())
    approval_mode: reactive[ApprovalMode] = reactive(ApprovalMode.MANUAL)

    def __init__(
        self,
        *,
        show_approval_badge: bool = True,
        show_mode_badge: bool = False,
        locale_controller: LocaleController | None = None,
    ) -> None:
        super().__init__(id="app-header")
        self._show_approval_badge = show_approval_badge
        self._show_mode_badge = show_mode_badge
        self._workflow_mode = False
        self._locale_controller = locale_controller

    def compose(self) -> ComposeResult:
        yield Static(id="header-title")
        if self._show_mode_badge:
            yield _HeaderBadge(self._mode_content(), id="mode-badge")
        if self._show_approval_badge:
            yield _HeaderBadge(id="approval-badge")

    def on_mount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.register_surface(self)
        self._refresh_title()
        self._refresh_mode()
        if self._show_approval_badge:
            self._refresh_approval_badge(self.approval_mode)

    def on_unmount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.unregister_surface(self)

    def refresh_localization(self) -> None:
        """Retranslate the approval badge without rebuilding the header."""
        self._refresh_mode()
        if self.is_mounted and self._show_approval_badge:
            self._refresh_approval_badge(self.approval_mode)

    def watch_subtitle_parts(self, _parts: tuple[str, ...]) -> None:
        if self.is_mounted:
            self._refresh_title()

    def watch_approval_mode(self, mode: ApprovalMode) -> None:
        if self.is_mounted:
            self._refresh_approval_badge(mode)

    def _refresh_title(self) -> None:
        """Rebuild the title text from the current subtitle parts."""
        from chrys import __version__

        title = Content(format_app_version_title(__version__))
        if self.subtitle_parts:
            sub_text = " \u2502 ".join(self.subtitle_parts)
            content = Content.assemble(title, (" \u2014 ", "dim"), Content(sub_text).stylize("dim"))
        else:
            content = title
        self.query_one("#header-title", Static).update(content)

    def _refresh_approval_badge(self, mode: ApprovalMode) -> None:
        """Refresh the approval mode badge from the current reactive value."""
        if not self._show_approval_badge:
            return
        badge = self.query_one("#approval-badge", Static)
        mode_text = self._render_message(APPROVAL_MODE_MESSAGES[mode].bind())
        badge_text = self._render_message(_APPROVAL_BADGE.bind(mode=mode_text))
        badge.update(Content.from_text(badge_text, markup=False))
        # Swap CSS class for color
        for cls in _APPROVAL_CLASSES.values():
            badge.remove_class(cls)
        badge.add_class(_APPROVAL_CLASSES[mode])

    def _localizer(self) -> Localizer | None:
        controller = self._locale_controller
        if controller is None:
            return None
        return controller.localizer

    def _render_message(self, reference: MessageRef) -> str:
        localizer = self._localizer()
        if localizer is None:
            return format_message(reference)
        return render_str(localizer, reference)

    def on_click(self, event: Click) -> None:
        """Open approval mode picker when the badge is clicked."""
        if self._show_mode_badge and self.query_one("#mode-badge").region.contains(event.screen_x, event.screen_y):
            event.prevent_default()
            event.stop()
            self.post_message(self.ModeClicked())
            return
        if not self._show_approval_badge:
            return
        badge = self.query_one("#approval-badge", Static)
        if badge.region.contains(event.screen_x, event.screen_y):
            event.prevent_default()
            event.stop()
            self.post_message(self.ApprovalBadgeClicked())

    def set_subtitle(self, *parts: str) -> None:
        """Set subtitle parts (e.g. profile name, model id) and refresh."""
        self.subtitle_parts = tuple(p for p in parts if p)

    def set_workflow_mode(self, workflow: bool) -> None:
        if workflow == self._workflow_mode:
            return
        self._workflow_mode = workflow
        self._refresh_mode()

    def _mode_content(self) -> Content:
        reference = workflow_text.MODE_WORKFLOW if self._workflow_mode else workflow_text.MODE_CHAT
        label = self._render_message(reference.bind())
        badge = self._render_message(workflow_text.MODE_BADGE.bind(mode=label))
        return Content.from_text(badge, markup=False)

    def _refresh_mode(self) -> None:
        if not self._show_mode_badge:
            return
        self.query_one("#mode-badge", Static).update(self._mode_content())
