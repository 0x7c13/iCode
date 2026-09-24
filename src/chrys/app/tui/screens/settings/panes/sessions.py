# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The session-root row: path input, folder browser, in-use line and Migrate."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual import on
from textual.containers import VerticalGroup
from textual.widgets import Button, Label, Static

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.screens.settings.rows import InputRow
from chrys.app.tui.widgets import EnhancedInput as Input
from chrys.foundation.i18n import DisplayPath, msg

if TYPE_CHECKING:
    from textual.app import ComposeResult

SESSION_ROOT_KEY = "storage.session_root_dir"

_BROWSE = msg("tui.settings.session_root.browse", fallback="Browse")
_IN_USE = msg("tui.settings.session_root.in_use", fallback="In use: {path}")
_MIGRATE = msg("tui.settings.session_root.migrate", fallback="Migrate sessions")
_INVALID_ROOT = msg(
    "tui.settings.session_root.invalid",
    fallback="Not a usable session root: a writable sessions folder could not be created there.",
)
_WRITTEN = msg(
    "tui.settings.session_root.written",
    fallback=(
        "Restart required · sessions keep saving under the current root until then · "
        "use Migrate sessions to copy existing sessions"
    ),
)


class SessionRootRow(InputRow):
    """``storage.session_root_dir``: validated on commit, browsable, migratable."""

    def compose_main(self) -> ComposeResult:
        yield Label("", classes="settings-row-label")
        yield Input(classes="settings-row-input session-root-input")
        yield Button("", classes="settings-row-link session-root-browse", flat=True)

    def compose_control(self) -> ComposeResult:
        yield from ()

    def compose(self) -> ComposeResult:
        yield from super().compose()
        with VerticalGroup(classes="session-root-footer"):
            yield Static("", classes="session-root-in-use")
            yield Button("", classes="settings-row-link session-root-migrate", flat=True)

    def write_control(self, value: Any, *, editable: bool) -> None:
        super().write_control(value, editable=editable)
        self.query_one(".session-root-browse", Button).disabled = not editable

    def refresh_localization(self) -> None:
        localizer = widget_localizer(self)
        self.query_one(".session-root-browse", Button).label = render_str(localizer, _BROWSE.bind())
        self.query_one(".session-root-migrate", Button).label = render_str(localizer, _MIGRATE.bind())
        in_use = self._ports.session_storage().in_force_sessions_dir()
        self.query_one(".session-root-in-use", Static).update(
            Text(render_str(localizer, _IN_USE.bind(path=DisplayPath(in_use))))
        )
        super().refresh_localization()

    def hint_text(self) -> str:
        if (
            self._error is None
            and self._provenance.hint is None
            and self.spec.key in self._ports.restart_pending_keys()
        ):
            return render_str(widget_localizer(self), _WRITTEN.bind())
        return super().hint_text()

    def commit_pending(self) -> None:
        field = self.query_one(Input)
        if field.disabled:
            return
        raw = field.value
        current = "" if self._last_value is None else str(self._last_value)
        if raw.strip() == current.strip():
            if raw != current:
                self._paint(field, current)
            return
        self._commit(raw)

    def _commit(self, raw: str) -> None:
        if self._ports.session_storage().probe_root(raw) is None:
            self.show_error(_INVALID_ROOT.bind())
            return
        value = "" if not raw.strip() else raw.strip()
        self._paint(self.query_one(Input), value)
        self._edited(value)

    def _projected_sessions_dir(self) -> Path | None:
        """``<projected root>/sessions`` when it differs from the in-force one.

        Reads the row's own value rather than asking the ports again: a root
        typed a moment ago is committed but still in the debounced write queue,
        and the migration dialog should already point at it.
        """
        raw = self._last_value
        storage = self._ports.session_storage()
        candidate = storage.probe_root("" if raw is None else str(raw))
        if candidate is None or candidate == storage.in_force_sessions_dir():
            return None
        return candidate

    @on(Button.Pressed, ".session-root-browse")
    def _on_browse(self, event: Button.Pressed) -> None:
        event.stop()
        from chrys.app.tui.screens.dialogs.file_picker import FilePicker, FilePickerMode

        current = self.query_one(Input).value.strip()
        initial = current or str(self._ports.session_storage().in_force_sessions_dir().parent)

        def _on_result(result: str | None) -> None:
            if result:
                self._commit(result)

        self.app.push_screen(FilePicker(mode=FilePickerMode.FOLDER, initial_path=initial), _on_result)

    @on(Button.Pressed, ".session-root-migrate")
    def _on_migrate(self, event: Button.Pressed) -> None:
        event.stop()
        from chrys.app.tui.screens.settings.migrate_dialog import MigrateSessionsDialog

        storage = self._ports.session_storage()
        source = storage.in_force_sessions_dir()
        destination = self._projected_sessions_dir()
        if destination is None:
            default_dir = storage.default_sessions_dir()
            if default_dir != source and default_dir.is_dir():
                # No root change pending and this launch already runs on a
                # custom root: what is left at the default location is what
                # "restart and run Migrate again" is about. Both fields stay
                # editable for any other pair of roots.
                source, destination = default_dir, source
        self.app.push_screen(MigrateSessionsDialog(storage, source=source, destination=destination))


__all__ = ["SESSION_ROOT_KEY", "SessionRootRow"]
