# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The session-root row and the migration dialog."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.color import Color
from textual.widgets import Button, Input, Static

from chrys.app.tui.screens.dialogs.file_picker import FilePicker
from chrys.app.tui.screens.settings import SettingsDialog
from chrys.app.tui.screens.settings.migrate_dialog import MigrateSessionsDialog
from chrys.app.tui.screens.settings.panes.sessions import SESSION_ROOT_KEY, SessionRootRow
from chrys.foundation.config.settings import Settings
from chrys.service.state.session_migration import (
    MigrationItem,
    MigrationPlan,
    MigrationReport,
    SessionMigrationError,
)
from tests.app.tui.screens.settings.support import Host, StubPorts, StubSessionStorage


async def _open_sessions_tab(app: Host, ports: StubPorts) -> SessionRootRow:
    dialog = SettingsDialog(ports, initial_tab="sessions")
    await app.push_screen(dialog)
    return dialog.query_one(SessionRootRow)


def _hint(row: SessionRootRow) -> str:
    return str(row.query_one(".settings-row-hint", Static).render())


@pytest.mark.asyncio
async def test_session_root_row_shows_the_root_in_force_and_validates_edits(tmp_path: Path) -> None:
    ports = StubPorts(root=tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        row = await _open_sessions_tab(app, ports)
        await pilot.pause()

        assert row.spec.key == SESSION_ROOT_KEY
        in_use = str(row.query_one(".session-root-in-use", Static).render())
        assert in_use.startswith("In use: ")
        assert in_use.endswith("sessions")
        assert _hint(row).startswith("Blank = ")

        field = row.query_one(Input)
        field.value = str(tmp_path / "bad")
        row.commit_pending()
        await pilot.pause()
        assert ports.persisted == []
        assert _hint(row).startswith("Not a usable session root")
        assert ports.storage.probed == [str(tmp_path / "bad")]

        field.value = f"  {tmp_path / 'good'}  "
        row.commit_pending()
        await pilot.pause()
        assert ports.persisted == [{SESSION_ROOT_KEY: str(tmp_path / "good")}]
        assert field.value == str(tmp_path / "good")
        assert not row.query_one(".settings-row-hint", Static).has_class("-error")

        # Once written, the hint says the change waits for a restart.
        ports.desired[SESSION_ROOT_KEY] = str(tmp_path / "good")
        ports.restart_pending = frozenset({SESSION_ROOT_KEY})
        row.project()
        assert _hint(row).startswith("Restart required")


@pytest.mark.asyncio
async def test_session_root_row_clearing_the_field_writes_the_default(tmp_path: Path) -> None:
    ports = StubPorts(Settings(session_root_dir=str(tmp_path / "custom")), root=tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        row = await _open_sessions_tab(app, ports)
        await pilot.pause()
        field = row.query_one(Input)
        assert field.value == str(tmp_path / "custom")

        field.value = "   "
        row.commit_pending()
        await pilot.pause()

        assert ports.persisted == [{SESSION_ROOT_KEY: ""}]
        assert field.value == ""


@pytest.mark.asyncio
async def test_migrating_back_to_the_default_root_prefills_the_default_sessions_dir(tmp_path: Path) -> None:
    ports = StubPorts(
        Settings(session_root_dir=str(tmp_path / "custom")),
        root=tmp_path / "custom",
        default_root=tmp_path / "default",
    )
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        row = await _open_sessions_tab(app, ports)
        await pilot.pause()
        row.query_one(Input).value = ""
        row.commit_pending()
        await pilot.pause()
        assert ports.persisted == [{SESSION_ROOT_KEY: ""}]

        row.query_one(".session-root-migrate", Button).press()
        await pilot.pause()
        migrate = app.screen
        assert isinstance(migrate, MigrateSessionsDialog)
        assert migrate.query_one("#migrate-from", Input).value == str(tmp_path / "custom" / "sessions")
        assert migrate.query_one("#migrate-to", Input).value == str(tmp_path / "default" / "sessions")


@pytest.mark.asyncio
async def test_migrating_after_a_restart_onto_a_custom_root_prefills_the_default_as_source(tmp_path: Path) -> None:
    """No root change pending, custom root in force: the sessions a previous
    launch left at the default location are the ones to bring over."""
    (tmp_path / "default" / "sessions").mkdir(parents=True)
    ports = StubPorts(
        Settings(session_root_dir=str(tmp_path / "custom")),
        root=tmp_path / "custom",
        default_root=tmp_path / "default",
    )
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        row = await _open_sessions_tab(app, ports)
        await pilot.pause()
        row.query_one(".session-root-migrate", Button).press()
        await pilot.pause()
        migrate = app.screen
        assert isinstance(migrate, MigrateSessionsDialog)
        assert migrate.query_one("#migrate-from", Input).value == str(tmp_path / "default" / "sessions")
        assert migrate.query_one("#migrate-to", Input).value == str(tmp_path / "custom" / "sessions")


@pytest.mark.asyncio
async def test_migrating_with_nothing_at_the_default_location_leaves_the_destination_blank(tmp_path: Path) -> None:
    ports = StubPorts(
        Settings(session_root_dir=str(tmp_path / "custom")),
        root=tmp_path / "custom",
        default_root=tmp_path / "default",
    )
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        row = await _open_sessions_tab(app, ports)
        await pilot.pause()
        row.query_one(".session-root-migrate", Button).press()
        await pilot.pause()
        migrate = app.screen
        assert isinstance(migrate, MigrateSessionsDialog)
        assert migrate.query_one("#migrate-from", Input).value == str(tmp_path / "custom" / "sessions")
        assert migrate.query_one("#migrate-to", Input).value == ""


@pytest.mark.asyncio
async def test_session_root_browse_and_migrate_open_their_dialogs(tmp_path: Path) -> None:
    ports = StubPorts(Settings(session_root_dir=str(tmp_path / "next")), root=tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        row = await _open_sessions_tab(app, ports)
        await pilot.pause()

        row.query_one(".session-root-browse", Button).press()
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, FilePicker)
        picker.dismiss(str(tmp_path / "picked"))
        await pilot.pause()
        assert ports.persisted == [{SESSION_ROOT_KEY: str(tmp_path / "picked")}]

        # The write is still queued (nothing projected yet), but the migration
        # destination already follows the root just committed, not the one in force.
        assert ports.desired == {}
        row.query_one(".session-root-migrate", Button).press()
        await pilot.pause()
        migrate = app.screen
        assert isinstance(migrate, MigrateSessionsDialog)
        assert migrate.query_one("#migrate-from", Input).value == str(tmp_path / "sessions")
        # The destination is the projected root's sessions folder.
        assert migrate.query_one("#migrate-to", Input).value == str(tmp_path / "picked" / "sessions")


def _plan(tmp_path: Path, *ids: str) -> MigrationPlan:
    return MigrationPlan(
        source_dir=tmp_path / "sessions",
        destination_dir=tmp_path / "next" / "sessions",
        items=tuple(
            MigrationItem(
                session_id=session_id,
                source=tmp_path / "sessions" / session_id,
                destination=tmp_path / "next" / "sessions" / session_id,
                legacy_file=False,
                already_present=False,
            )
            for session_id in ids
        ),
        rejected=(),
    )


@pytest.mark.asyncio
async def test_migrate_dialog_copies_and_reports_then_turns_cancel_into_close(tmp_path: Path) -> None:
    storage = StubSessionStorage(tmp_path)
    storage.plan_result = _plan(tmp_path, "a", "b", "c")
    storage.report_result = MigrationReport(
        copied=("a",),
        skipped_present=("b",),
        skipped_active=("c",),
        skipped_busy=(),
        failed=((tmp_path / "sessions" / "d", "permission denied"),),
    )
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = MigrateSessionsDialog(
            storage, source=tmp_path / "sessions", destination=tmp_path / "next" / "sessions"
        )
        await app.push_screen(dialog)
        await pilot.pause()
        assert isinstance(dialog.focused, Button) and dialog.focused.id == "migrate-run"

        dialog.query_one("#migrate-run", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert storage.plans == [(tmp_path / "sessions", tmp_path / "next" / "sessions")]
        assert storage.reports == [storage.plan_result]
        status = str(dialog.query_one("#migrate-status", Static).render())
        assert status.splitlines()[0] == "Copied 1 · already present 1 · active 1 · busy 0 · failed 1"
        assert "restart and run Migrate again" in status
        assert status.splitlines()[-1].endswith("d: permission denied")
        assert dialog.query_one("#migrate-run", Button).disabled is True
        assert str(dialog.query_one("#migrate-cancel", Button).label) == "Close"

        dialog.query_one("#migrate-cancel", Button).press()
        await pilot.pause()
        assert app.screen is not dialog


@pytest.mark.asyncio
async def test_migrate_dialog_needs_both_paths_and_surfaces_planner_errors(tmp_path: Path) -> None:
    class _FailingStorage(StubSessionStorage):
        def plan_migration(self, source: Path, destination: Path) -> MigrationPlan:
            raise SessionMigrationError("destination is inside the source")

    storage = _FailingStorage(tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = MigrateSessionsDialog(storage, source=tmp_path / "sessions", destination=None)
        await app.push_screen(dialog)
        await pilot.pause()

        dialog.query_one("#migrate-run", Button).press()
        await pilot.pause()
        assert str(dialog.query_one("#migrate-status", Static).render()) == "Choose both a source and a destination."
        assert dialog.query_one("#migrate-run", Button).disabled is False

        dialog.query_one("#migrate-to", Input).value = str(tmp_path / "sessions" / "inner")
        dialog.query_one("#migrate-run", Button).press()
        await pilot.pause()
        await pilot.pause()
        status = dialog.query_one("#migrate-status", Static)
        assert str(status.render()) == "Migration could not run: destination is inside the source"
        assert status.has_class("-error")
        # A failed attempt can be corrected and retried.
        assert dialog.query_one("#migrate-run", Button).disabled is False


@pytest.mark.asyncio
async def test_migrate_dialog_reports_an_empty_source(tmp_path: Path) -> None:
    storage = StubSessionStorage(tmp_path)
    storage.plan_result = _plan(tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = MigrateSessionsDialog(
            storage, source=tmp_path / "sessions", destination=tmp_path / "next" / "sessions"
        )
        await app.push_screen(dialog)
        await pilot.pause()

        dialog.query_one("#migrate-run", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert str(dialog.query_one("#migrate-status", Static).render()).startswith("No sessions to copy from ")
        assert storage.reports == []
        assert dialog.query_one("#migrate-run", Button).disabled is True


@pytest.mark.asyncio
async def test_migrate_dialog_locks_every_control_while_copying(tmp_path: Path) -> None:
    import asyncio

    gate = asyncio.Event()

    class _SlowStorage(StubSessionStorage):
        async def run_migration(self, plan: MigrationPlan) -> MigrationReport:
            await gate.wait()
            return await super().run_migration(plan)

    storage = _SlowStorage(tmp_path)
    storage.plan_result = _plan(tmp_path, "a")
    storage.report_result = MigrationReport(
        copied=("a",), skipped_present=(), skipped_active=(), skipped_busy=(), failed=()
    )
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = MigrateSessionsDialog(
            storage, source=tmp_path / "sessions", destination=tmp_path / "next" / "sessions"
        )
        await app.push_screen(dialog)
        await pilot.pause()

        dialog.query_one("#migrate-run", Button).press()
        await pilot.pause()
        await pilot.pause()

        # Mid-copy: no button, no path field can be used, and Esc does not close.
        for selector in ("#migrate-run", "#migrate-cancel", "#migrate-browse-from", "#migrate-browse-to"):
            assert dialog.query_one(selector, Button).disabled is True, selector
        for selector in ("#migrate-from", "#migrate-to"):
            assert dialog.query_one(selector, Input).disabled is True, selector
        assert dialog.query_one("#migrate-loading").display is True, "the dots spin next to the status"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is dialog

        gate.set()
        await pilot.pause()
        await pilot.pause()

        # Done: only Close remains live.
        assert dialog.query_one("#migrate-run", Button).disabled is True
        cancel = dialog.query_one("#migrate-cancel", Button)
        assert cancel.disabled is False and str(cancel.label) == "Close"
        assert dialog.query_one("#migrate-from", Input).disabled is False
        assert dialog.query_one("#migrate-loading").display is False
        cancel.press()
        await pilot.pause()
        assert app.screen is not dialog


@pytest.mark.asyncio
async def test_migrate_dialog_browse_buttons_use_the_panel_link_style(tmp_path: Path) -> None:
    """The theme's flat default-button foreground assumes a coloured background;
    the wizard's Browse buttons must get the same link ink and tint as the panel's."""
    storage = StubSessionStorage(tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = MigrateSessionsDialog(
            storage, source=tmp_path / "sessions", destination=tmp_path / "next" / "sessions"
        )
        await app.push_screen(dialog)
        await pilot.pause()

        primary = Color.parse(app.get_css_variables()["primary"])
        for selector in ("#migrate-browse-from", "#migrate-browse-to"):
            button = dialog.query_one(selector, Button)
            assert button.has_class("settings-row-link"), selector
            assert button.styles.color == primary, selector
            assert button.styles.background == primary.with_alpha(0.2), selector
