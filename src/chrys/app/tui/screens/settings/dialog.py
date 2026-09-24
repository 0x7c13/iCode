# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The Settings dialog: a tabbed modal over the layout table."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual.containers import VerticalGroup, VerticalScroll
from textual.widgets import Button, Checkbox, Input, Select, TabbedContent, TabPane

from chrys.app.tui.binding_display import CLOSE_BINDING, localized_binding
from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.screens.settings.layout import (
    GENERAL_TAB_ID,
    NOTIFICATIONS_TAB_ID,
    TABS,
    RowKind,
    SettingRowSpec,
    SettingsTab,
    tab_by_id,
)
from chrys.app.tui.screens.settings.panes.notifications import NotificationsPane
from chrys.app.tui.screens.settings.panes.sessions import SessionRootRow
from chrys.app.tui.screens.settings.rows import SettingRow, row_class_for
from chrys.foundation.config.settings import Settings
from chrys.foundation.config.spec import specs_by_key
from chrys.foundation.i18n import msg

_CONTROL_TYPES = (Checkbox, Select, Input, Button)

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController
    from chrys.app.tui.screens.settings.ports import SettingsPanelPorts

_TITLE = msg("tui.settings.dialog.title", fallback="Settings")
THEME_PREVIEW_HINT = msg(
    "tui.settings.hint.theme_preview", fallback="Close the theme editor before changing the applied theme."
)
_STATUS_IDLE = msg("tui.settings.status.idle", fallback="Changes are saved as you make them")
_STATUS_RESTART = msg(
    "tui.settings.status.restart",
    fallback="{count} change applies after restart",
    plural_fallback="{count} changes apply after restart",
)
_STATUS_RELOAD_ON_CLOSE = msg("tui.settings.status.reload_on_close", fallback="Changes apply on close (reload)")
_STATUS_RELOAD_AFTER_TURN = msg(
    "tui.settings.status.reload_after_turn",
    fallback="Changes apply when the current turn ends",
)

STATUS_SEPARATOR = " · "


def pane_id(tab_id: str) -> str:
    return f"settings-tab-{tab_id}"


class SettingsDialog(BaseDialog[None]):
    """Tabbed settings modal; every row saves itself through the ports."""

    CSS_PATH = "settings.tcss"

    BINDINGS: ClassVar[list] = [
        localized_binding("escape", "close", CLOSE_BINDING, show=False, priority=True),
    ]

    def __init__(
        self,
        ports: SettingsPanelPorts,
        *,
        initial_tab: str = GENERAL_TAB_ID,
        locale_controller: LocaleController | None = None,
    ) -> None:
        self._ports = ports
        self._initial_tab = initial_tab if tab_by_id(initial_tab) is not None else GENERAL_TAB_ID
        self._locale_controller = locale_controller
        self._specs = specs_by_key(Settings)
        super().__init__()

    # ── composition ─────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        localizer = widget_localizer(self)
        with VerticalGroup(id="settings-container") as container:
            container.border_title = Text(render_str(localizer, _TITLE.bind()))
            with TabbedContent(id="settings-tabs", initial=pane_id(self._initial_tab)):
                for tab in TABS:
                    with (
                        TabPane(render_str(localizer, tab.title.bind()), id=pane_id(tab.id)),
                        VerticalScroll(classes="settings-pane-scroll"),
                    ):
                        yield from self._compose_tab(tab)

    def _compose_tab(self, tab: SettingsTab) -> ComposeResult:
        if tab.id == NOTIFICATIONS_TAB_ID:
            yield NotificationsPane(self._ports.notifications())
            return
        localizer = widget_localizer(self)
        for section in tab.sections:
            with VerticalGroup(classes="settings-section") as group:
                group.border_title = Text(render_str(localizer, section.title.bind()))
                for row in section.rows:
                    yield self._build_row(row)

    def _build_row(self, row: SettingRowSpec) -> SettingRow:
        from chrys.app.tui.app import ChrysApp

        spec = self._specs[row.key]
        if row.special is RowKind.SESSION_ROOT:
            return SessionRootRow(spec, row, self._ports)
        widget = row_class_for(spec, row)(spec, row, self._ports)
        if row.key == "ui.theme" and isinstance(self.app, ChrysApp) and self.app.theme_preview is not None:
            widget.read_only_reason = THEME_PREVIEW_HINT.bind()
        return widget

    def on_mount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.register_surface(self)
        self.refresh_status()
        # Defer focus until the mounted pane's layout is ready.
        self.call_after_refresh(self._focus_first_control, self._initial_tab)

    def on_unmount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.unregister_surface(self)

    def _focus_first_control(self, tab_id: str) -> None:
        panes = self.query(f"#{pane_id(tab_id)}")
        if not panes:
            return
        for widget in panes.first().query("*"):
            if not isinstance(widget, _CONTROL_TYPES):
                continue
            if widget.display and not widget.disabled:
                widget.focus()
                return

    # ── refresh ────────────────────────────────────────────────────
    def rows(self) -> list[SettingRow]:
        return list(self.query(SettingRow))

    def reproject(self) -> None:
        """Re-read every value/badge from the ports; controls are not rebuilt."""
        for row in self.rows():
            row.project()
        for pane in self.query(NotificationsPane):
            pane.project()
        self.refresh_status()

    def refresh_localization(self) -> None:
        """Replace text in place: tab titles, sections, rows, status."""
        localizer = widget_localizer(self)
        self.query_one("#settings-container", VerticalGroup).border_title = Text(render_str(localizer, _TITLE.bind()))
        tabs = self.query_one("#settings-tabs", TabbedContent)
        for tab in TABS:
            tabs.get_tab(pane_id(tab.id)).label = render_str(localizer, tab.title.bind())
            pane = self.query_one(f"#{pane_id(tab.id)}", TabPane)
            if tab.id == NOTIFICATIONS_TAB_ID:
                continue
            groups = list(pane.query(".settings-section"))
            for group, section in zip(groups, tab.sections, strict=True):
                group.border_title = Text(render_str(localizer, section.title.bind()))
        for row in self.rows():
            row.refresh_localization()
        for notifications in self.query(NotificationsPane):
            notifications.refresh_localization()
        self.refresh_status()

    def refresh_status(self) -> None:
        localizer = widget_localizer(self)
        parts: list[str] = []
        pending = len(self._ports.restart_pending_keys())
        if pending:
            parts.append(render_str(localizer, _STATUS_RESTART.bind(count=pending)))
        if self._ports.reload_dirty():
            definition = _STATUS_RELOAD_AFTER_TURN if self._ports.turn_in_progress() else _STATUS_RELOAD_ON_CLOSE
            parts.append(render_str(localizer, definition.bind()))
        text = STATUS_SEPARATOR.join(parts) if parts else render_str(localizer, _STATUS_IDLE.bind())
        container = self.query_one("#settings-container", VerticalGroup)
        container.set_class(bool(parts), "-status-active")
        container.border_subtitle = Text(text)

    # ── close ──────────────────────────────────────────────────────
    def _before_dismiss(self, _result: object | None = None) -> None:
        self.commit_pending()
        self._ports.on_dialog_closed()

    def commit_pending(self) -> None:
        """Commit edits the controls have not reported yet (text being typed).

        Called on close, and by the app before it exits with this dialog on
        top: Ctrl+Q is an app-level priority binding, and by the time the
        dialog unmounts its rows are already gone.
        """
        for row in self.rows():
            row.commit_pending()

    def action_close(self) -> None:
        self.dismiss(None)


__all__ = ["STATUS_SEPARATOR", "SettingsDialog", "pane_id"]
