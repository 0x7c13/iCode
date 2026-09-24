# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for sessions screen utility functions."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.events import MouseMove
from textual.widgets import Button, DataTable, Static

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog
from chrys.app.tui.screens.sessions.presenter import format_size
from chrys.app.tui.screens.sessions.screen import (
    SessionsScreen,
    _next_cursor_row_after_delete,
    _next_session_id_after_delete,
    _SessionTable,
)
from chrys.app.tui.util.rich_style import rich_style_from_textual_color
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
from chrys.foundation.config.settings import Settings
from chrys.service.state.store import ChatSessionMeta, SessionMeta
from tests.support.waiting import wait_for


class _SessionsApp(App):
    locale_controller = LocaleController(Settings(locale="en"))

    def compose(self) -> ComposeResult:
        yield Static("placeholder")


class _SessionStore:
    def __init__(self, count: int) -> None:
        base = datetime(2026, 1, 1, tzinfo=UTC)
        self.sessions = [
            ChatSessionMeta(
                session_id=f"session-{index}",
                agent_profile="Code",
                agent_display_name="Code",
                created_at=base - timedelta(minutes=index),
                updated_at=base - timedelta(minutes=index),
                message_count=1,
                title=f"Session {index}",
            )
            for index in range(count)
        ]

    async def list_sessions(self) -> list[SessionMeta]:
        return list(self.sessions)

    async def stream_session_metas(self, *, batch_size: int = 32, kind=None):
        for start in range(0, len(self.sessions), batch_size):
            yield list(self.sessions[start : start + batch_size])

    async def delete_session(self, session_id: str, *, allow_active: bool = False) -> None:
        self.sessions = [session for session in self.sessions if session.session_id != session_id]


class _DelayedFinalBatchStore(_SessionStore):
    def __init__(self, count: int, initial_count: int) -> None:
        super().__init__(count)
        self.initial_count = initial_count
        self.initial_batch_yielded = asyncio.Event()
        self.release_final_batch = asyncio.Event()

    async def stream_session_metas(self, *, batch_size: int = 32, kind=None):
        yield list(self.sessions[: self.initial_count])
        self.initial_batch_yielded.set()
        await self.release_final_batch.wait()
        remaining = self.sessions[self.initial_count :]
        if remaining:
            yield list(remaining)


class _BlockedReloadStore(_SessionStore):
    def __init__(self, count: int) -> None:
        super().__init__(count)
        self.stream_calls = 0
        self.reload_started = asyncio.Event()
        self.release_reload = asyncio.Event()

    async def stream_session_metas(self, *, batch_size: int = 32, kind=None):
        self.stream_calls += 1
        if self.stream_calls > 1:
            self.reload_started.set()
            await self.release_reload.wait()
        yield list(self.sessions)


class _PartialBlockedReloadStore(_SessionStore):
    def __init__(self, count: int) -> None:
        super().__init__(count)
        self.stream_calls = 0
        self.reload_started = asyncio.Event()
        self.partial_batch_yielded = asyncio.Event()
        self.release_reload = asyncio.Event()

    async def stream_session_metas(self, *, batch_size: int = 32, kind=None):
        self.stream_calls += 1
        if self.stream_calls == 1:
            yield list(self.sessions)
            return

        self.reload_started.set()
        sessions = list(self.sessions)
        if sessions:
            yield sessions[:1]
        self.partial_batch_yielded.set()
        await self.release_reload.wait()
        if len(sessions) > 1:
            yield sessions[1:]


async def _wait_for_row_count(table: DataTable, pilot, count: int) -> None:
    await wait_for(
        lambda: table.row_count == count,
        pilot=pilot,
        description=f"sessions table reaches {count} rows",
    )


async def _wait_for_load_idle(screen: SessionsScreen, pilot) -> None:
    await wait_for(
        lambda: not screen._loading,
        pilot=pilot,
        description="Expected sessions load to become idle",
    )


async def _wait_for_event(event: asyncio.Event, pilot) -> None:
    await wait_for(
        event.is_set,
        pilot=pilot,
        description="Expected event to be set",
    )


async def _wait_for_results(results: list[str | None], pilot, count: int) -> None:
    await wait_for(
        lambda: len(results) == count,
        pilot=pilot,
        description=f"session dialog produces {count} dismiss results",
    )


async def _open_delete_dialog(screen: SessionsScreen, pilot) -> ConfirmDialog:
    screen.action_delete_session()
    await pilot.pause()
    dialog = pilot.app.screen
    assert isinstance(dialog, ConfirmDialog)
    return dialog


async def _confirm_delete(screen: SessionsScreen, pilot) -> None:
    dialog = await _open_delete_dialog(screen, pilot)
    dialog.query_one("#confirm-yes", Button).press()
    await pilot.pause()


@pytest.mark.asyncio
async def test_chat_session_browser_excludes_workflow_sessions() -> None:
    store = _SessionStore(3)
    store.sessions[1].kind = "workflow"
    app = _SessionsApp()
    async with app.run_test() as pilot:
        screen = SessionsScreen(store)
        await app.push_screen(screen)
        await _wait_for_load_idle(screen, pilot)
        assert screen._session_ids == ["session-0", "session-2"]
        assert screen.query_one("#sessions", DataTable).row_count == 2


@pytest.mark.asyncio
async def test_session_table_mouse_move_dispatch_invokes_data_table_base_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session table hover customizations should not double-run DataTable mouse handling."""

    class _MouseMoveProbe(_SessionTable):
        pass

    base_calls: list[MouseMove] = []

    def data_table_mouse_move_spy(_table: DataTable, event: MouseMove) -> None:
        base_calls.append(event)

    monkeypatch.setattr(DataTable, "_on_mouse_move", data_table_mouse_move_spy)
    table = _MouseMoveProbe()
    event = MouseMove(
        table,
        x=0,
        y=0,
        delta_x=0,
        delta_y=0,
        button=0,
        shift=False,
        meta=False,
        ctrl=False,
        style=Style(),
    )

    await table._on_message(event)

    assert base_calls == [event]


class TestFormatSize:
    def test_zero_bytes(self) -> None:
        assert format_size(0) == "0 B"

    def test_small_bytes(self) -> None:
        assert format_size(512) == "512 B"

    def test_one_byte_below_kb(self) -> None:
        assert format_size(1023) == "1023 B"

    def test_exactly_one_kb(self) -> None:
        assert format_size(1024) == "1.0 KB"

    def test_kilobytes(self) -> None:
        assert format_size(1536) == "1.5 KB"

    def test_one_byte_below_mb(self) -> None:
        assert format_size(1024 * 1024 - 1) == "1024.0 KB"

    def test_exactly_one_mb(self) -> None:
        assert format_size(1024 * 1024) == "1.0 MB"

    def test_megabytes(self) -> None:
        assert format_size(int(2.5 * 1024 * 1024)) == "2.5 MB"

    def test_exactly_one_gb(self) -> None:
        assert format_size(1024 * 1024 * 1024) == "1.0 GB"

    def test_gigabytes(self) -> None:
        assert format_size(int(3.7 * 1024 * 1024 * 1024)) == "3.7 GB"


class TestNextCursorRowAfterDelete:
    def test_keeps_same_visible_index_when_later_rows_remain(self) -> None:
        assert _next_cursor_row_after_delete(deleted_row=4, remaining_count=8) == 4

    def test_moves_to_previous_row_when_deleted_row_was_last(self) -> None:
        assert _next_cursor_row_after_delete(deleted_row=4, remaining_count=4) == 3

    def test_returns_none_when_no_rows_remain(self) -> None:
        assert _next_cursor_row_after_delete(deleted_row=0, remaining_count=0) is None


class TestNextSessionIdAfterDelete:
    def test_prefers_following_visible_session(self) -> None:
        assert _next_session_id_after_delete(["a", "b", "c"], 1) == "c"

    def test_falls_back_to_previous_for_last_row(self) -> None:
        assert _next_session_id_after_delete(["a", "b", "c"], 2) == "b"

    def test_returns_none_for_only_row(self) -> None:
        assert _next_session_id_after_delete(["a"], 0) is None

    def test_returns_none_for_invalid_row(self) -> None:
        assert _next_session_id_after_delete(["a"], 5) is None


@pytest.mark.asyncio
async def test_initial_load_shows_loading_until_final_rows_ready() -> None:
    store = _DelayedFinalBatchStore(count=4, initial_count=2)
    store.sessions[3].updated_at = datetime(2026, 1, 2, tzinfo=UTC)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_event(store.initial_batch_yielded, pilot)

        assert table.row_count == 0
        assert not bool(table.display)
        assert bool(screen.query_one("#sessions-loading-state").display)
        assert isinstance(screen.query_one("#sessions-loading"), ChrysLoadingIndicator)
        assert not bool(screen.query_one("#footer").display)
        assert screen.query_one("#container").border_subtitle == "Loading sessions"

        store.release_final_batch.set()
        await _wait_for_row_count(table, pilot, 4)

        assert not bool(screen.query_one("#sessions-loading-state").display)
        assert bool(table.display)
        assert bool(screen.query_one("#footer").display)
        assert screen.query_one("#container").border_subtitle == "4 sessions"
        assert table.get_row_at(0)[0].plain == "session3"
        assert screen._get_selected_session_id() == "session-3"


@pytest.mark.asyncio
async def test_delete_keeps_same_visible_row_when_following_rows_remain() -> None:
    store = _SessionStore(8)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_row_count(table, pilot, 8)

        table.move_cursor(row=4)
        await pilot.pause()
        await _confirm_delete(screen, pilot)
        await _wait_for_row_count(table, pilot, 7)

        assert table.cursor_coordinate.row == 4
        assert screen._get_selected_session_id() == "session-5"


@pytest.mark.asyncio
async def test_delete_selects_previous_row_when_deleted_row_was_last() -> None:
    store = _SessionStore(5)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_row_count(table, pilot, 5)

        table.move_cursor(row=4)
        await pilot.pause()
        await _confirm_delete(screen, pilot)
        await _wait_for_row_count(table, pilot, 4)

        assert table.cursor_coordinate.row == 3
        assert screen._get_selected_session_id() == "session-3"


@pytest.mark.asyncio
async def test_delete_leaves_no_selection_when_no_rows_remain() -> None:
    store = _SessionStore(1)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_row_count(table, pilot, 1)

        await _confirm_delete(screen, pilot)
        await _wait_for_row_count(table, pilot, 0)

        assert screen._get_selected_session_id() is None
        assert bool(screen.query_one("#empty-note").display)
        assert screen.query_one("#resume", Button).disabled
        assert screen.query_one("#delete", Button).disabled


@pytest.mark.asyncio
async def test_delete_other_session_does_not_yank_cursor_to_current() -> None:
    """Post-delete the cursor lands on the neighbor, not the loaded session."""
    store = _SessionStore(5)
    screen = SessionsScreen(store, current_session_id="session-0")

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_row_count(table, pilot, 5)
        assert screen._get_selected_session_id() == "session-0"

        table.move_cursor(row=3)
        await pilot.pause()
        await _confirm_delete(screen, pilot)
        await _wait_for_row_count(table, pilot, 4)

        assert table.cursor_coordinate.row == 3
        assert screen._get_selected_session_id() == "session-4"


@pytest.mark.asyncio
async def test_delete_after_sort_selects_visible_neighbor_by_session_id() -> None:
    store = _SessionStore(4)
    for index, turns in enumerate((40, 10, 20, 30)):
        store.sessions[index].turn_count = turns
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", _SessionTable)
        await _wait_for_row_count(table, pilot, 4)

        _click_header(screen, 4)  # Turns — descending first.
        await pilot.pause()
        assert [table.get_row_at(i)[4].plain for i in range(4)] == ["40", "30", "20", "10"]

        table.move_cursor(row=1)
        await pilot.pause()
        assert screen._get_selected_session_id() == "session-3"

        await _confirm_delete(screen, pilot)
        await _wait_for_row_count(table, pilot, 3)

        assert [table.get_row_at(i)[4].plain for i in range(3)] == ["40", "20", "10"]
        assert table.cursor_coordinate.row == 1
        assert screen._get_selected_session_id() == "session-2"


@pytest.mark.asyncio
async def test_delete_reload_final_render_preserves_user_cursor_move() -> None:
    store = _BlockedReloadStore(4)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_row_count(table, pilot, 4)

        table.move_cursor(row=1)
        await pilot.pause()
        await _confirm_delete(screen, pilot)
        await _wait_for_event(store.reload_started, pilot)
        await _wait_for_row_count(table, pilot, 3)
        assert screen._get_selected_session_id() == "session-2"

        table.move_cursor(row=2)
        await pilot.pause()
        assert screen._get_selected_session_id() == "session-3"

        store.release_reload.set()
        await _wait_for_load_idle(screen, pilot)

        assert screen._get_selected_session_id() == "session-3"


@pytest.mark.asyncio
async def test_rapid_delete_optimistic_render_uses_last_rendered_full_source() -> None:
    store = _PartialBlockedReloadStore(4)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_row_count(table, pilot, 4)

        table.move_cursor(row=0)
        await pilot.pause()
        await _confirm_delete(screen, pilot)
        await _wait_for_event(store.partial_batch_yielded, pilot)
        await _wait_for_row_count(table, pilot, 3)
        assert [table.get_row_at(i)[0].plain for i in range(3)] == ["session1", "session2", "session3"]

        table.move_cursor(row=1)
        await pilot.pause()
        await _confirm_delete(screen, pilot)
        await _wait_for_row_count(table, pilot, 2)

        assert [table.get_row_at(i)[0].plain for i in range(2)] == ["session1", "session3"]
        assert screen._get_selected_session_id() == "session-3"

        store.release_reload.set()
        await _wait_for_load_idle(screen, pilot)


@pytest.mark.asyncio
async def test_delete_confirmation_uses_neighbor_from_dialog_open_order() -> None:
    store = _SessionStore(4)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_row_count(table, pilot, 4)

        table.move_cursor(row=1)
        await pilot.pause()
        assert screen._get_selected_session_id() == "session-1"
        dialog = await _open_delete_dialog(screen, pilot)

        store.sessions[3].updated_at = datetime(2026, 1, 3, tzinfo=UTC)
        screen._render_table()
        await pilot.pause()
        assert [table.get_row_at(i)[0].plain for i in range(4)] == ["session3", "session0", "session1", "session2"]

        dialog.query_one("#confirm-yes", Button).press()
        await _wait_for_row_count(table, pilot, 3)

        assert screen._get_selected_session_id() == "session-2"


@pytest.mark.asyncio
async def test_delete_optimistic_render_uses_current_stable_source_after_dialog_refresh() -> None:
    store = _BlockedReloadStore(3)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_row_count(table, pilot, 3)

        table.move_cursor(row=1)
        await pilot.pause()
        assert screen._get_selected_session_id() == "session-1"
        dialog = await _open_delete_dialog(screen, pilot)

        discovered_session = ChatSessionMeta(
            session_id="session-new",
            agent_profile="Code",
            agent_display_name="Code",
            created_at=datetime(2026, 1, 3, tzinfo=UTC),
            updated_at=datetime(2026, 1, 3, tzinfo=UTC),
            message_count=1,
            title="Session New",
        )
        store.sessions.append(discovered_session)
        screen._render_table(source_sessions=list(store.sessions))
        await pilot.pause()
        assert screen._session_ids == ["session-new", "session-0", "session-1", "session-2"]

        dialog.query_one("#confirm-yes", Button).press()
        await _wait_for_event(store.reload_started, pilot)
        await _wait_for_row_count(table, pilot, 3)

        assert screen._session_ids == ["session-new", "session-0", "session-2"]
        assert screen._get_selected_session_id() == "session-2"

        store.release_reload.set()
        await _wait_for_load_idle(screen, pilot)


@pytest.mark.asyncio
async def test_delete_session_cancel_leaves_row_intact() -> None:
    store = _SessionStore(2)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_row_count(table, pilot, 2)

        dialog = await _open_delete_dialog(screen, pilot)
        message = dialog.query_one("#confirm-message", Static)
        confirm = dialog.query_one("#confirm-yes", Button)
        assert message.render().plain == 'Delete session\n"session0"?\n\nThis cannot be undone.'
        assert str(confirm.label) == "Delete"
        assert confirm.variant == "error"

        dialog.query_one("#confirm-no", Button).press()
        await pilot.pause()

        assert table.row_count == 2
        assert [session.session_id for session in store.sessions] == ["session-0", "session-1"]


@pytest.mark.asyncio
async def test_delete_session_escape_cancels_confirmation() -> None:
    store = _SessionStore(2)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_row_count(table, pilot, 2)

        await _open_delete_dialog(screen, pilot)
        await pilot.press("escape")
        await pilot.pause()

        assert table.row_count == 2
        assert [session.session_id for session in store.sessions] == ["session-0", "session-1"]


@pytest.mark.asyncio
async def test_delete_current_session_requires_confirmation_before_dismiss() -> None:
    store = _SessionStore(2)
    screen = SessionsScreen(store, current_session_id="session-0")
    results: list[str | None] = []

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen, callback=results.append)
        table = screen.query_one("#sessions", DataTable)
        await _wait_for_row_count(table, pilot, 2)

        dialog = await _open_delete_dialog(screen, pilot)

        assert results == []
        dialog.query_one("#confirm-yes", Button).press()

        await _wait_for_results(results, pilot, 1)

    assert results == [f"{SessionsScreen.DELETE_AND_NEW_PREFIX}session-0"]
    assert [session.session_id for session in store.sessions] == ["session-0", "session-1"]


@pytest.mark.asyncio
async def test_sessions_table_treats_session_metadata_as_plain_text() -> None:
    store = _SessionStore(1)
    store.sessions[0].agent_profile_history = ["Code [/home/jack]"]
    store.sessions[0].primary_cwd = "/tmp/[/project]"
    store.sessions[0].title = "Review crash from [/home/jack]"
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", _SessionTable)
        await _wait_for_row_count(table, pilot, 1)

        cells = table.get_row_at(0)
        assert all(isinstance(cell, Text) for cell in cells)
        assert cells[1].plain == "Review crash from [/home/jack]"
        assert cells[2].plain == "project]"
        assert isinstance(table._row_tooltips[0], Text)
        last_interaction = store.sessions[0].updated_at.astimezone().strftime("%Y/%m/%d %H:%M")
        assert table._row_tooltips[0].plain == (
            "Title: Review crash from [/home/jack]\n"
            "Agent: Code [/home/jack]\n"
            "Directory: /tmp/[/project]\n"
            "Turns: 0\n"
            "Total tokens: 0\n"
            f"Last interaction: {last_interaction}"
        )


def _click_header(screen: SessionsScreen, column_index: int) -> None:
    """Simulate a header click by posting the DataTable message."""
    table = screen.query_one("#sessions", _SessionTable)
    column = table.ordered_columns[column_index]
    table.post_message(DataTable.HeaderSelected(table, column.key, column_index, label=column.label))


@pytest.mark.asyncio
async def test_header_click_sorts_and_click_again_reverses() -> None:
    store = _SessionStore(3)
    store.sessions[0].turn_count = 5
    store.sessions[1].turn_count = 20
    store.sessions[2].turn_count = 1
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", _SessionTable)
        await _wait_for_row_count(table, pilot, 3)

        _click_header(screen, 4)  # Turns — numeric, descending first
        await pilot.pause()
        assert [table.get_row_at(i)[4].plain for i in range(3)] == ["20", "5", "1"]
        assert "↓" in str(table.ordered_columns[4].label)

        _click_header(screen, 4)
        await pilot.pause()
        assert [table.get_row_at(i)[4].plain for i in range(3)] == ["1", "5", "20"]
        assert "↑" in str(table.ordered_columns[4].label)


@pytest.mark.asyncio
async def test_header_click_keeps_selected_session() -> None:
    store = _SessionStore(4)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", _SessionTable)
        await _wait_for_row_count(table, pilot, 4)

        table.move_cursor(row=2)
        await pilot.pause()
        selected = screen._get_selected_session_id()

        _click_header(screen, 0)  # Session ID ascending
        await pilot.pause()
        assert screen._get_selected_session_id() == selected


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "ansi-dark", "ansi-light"])
async def test_search_filters_rows_and_highlights_matches(theme: str) -> None:
    from textual.widgets import Input

    store = _SessionStore(5)
    store.sessions[3].title = "unique needle title"
    screen = SessionsScreen(store)

    app = _SessionsApp()
    async with app.run_test(size=(120, 40)) as pilot:
        app.theme = theme
        await app.push_screen(screen)
        table = screen.query_one("#sessions", _SessionTable)
        await _wait_for_row_count(table, pilot, 5)

        screen.query_one("#search", Input).value = "needle"
        await _wait_for_row_count(table, pilot, 1)

        title_cell = table.get_row_at(0)[1]
        assert title_cell.plain == "unique needle title"
        assert screen.query_one("#container").border_subtitle == "1/5 sessions"
        # Match style is reverse-video in the theme's warning color.
        warning = app.theme_variables["warning"]
        expected_style = rich_style_from_textual_color(warning, reverse=True)
        match_spans = [span for span in title_cell.spans if span.style == expected_style]
        assert match_spans
        assert screen._get_selected_session_id() == "session-3"

        screen.query_one("#search", Input).value = ""
        await _wait_for_row_count(table, pilot, 5)


@pytest.mark.asyncio
async def test_search_escape_clears_then_returns_focus_to_table() -> None:
    from textual.widgets import Input

    store = _SessionStore(2)
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", _SessionTable)
        await _wait_for_row_count(table, pilot, 2)

        search = screen.query_one("#search", Input)
        search.focus()
        await wait_for(lambda: search.has_focus, pilot=pilot, description="control focus before interaction")
        await pilot.pause()
        search.value = "session"
        await pilot.pause()

        await pilot.press("escape")
        assert search.value == ""
        assert screen.app.focused is search
        assert isinstance(screen.app.screen, SessionsScreen)

        await pilot.press("escape")
        await pilot.pause()
        assert screen.app.focused is table
        assert isinstance(screen.app.screen, SessionsScreen)


@pytest.mark.asyncio
async def test_forked_sessions_nest_under_parent_with_tree_guides() -> None:
    store = _SessionStore(3)
    # session-2 is a fork of session-0; forks carry the parent's full id.
    store.sessions[2].parent_session_id = "session-0"
    screen = SessionsScreen(store)

    async with _SessionsApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        table = screen.query_one("#sessions", _SessionTable)
        await _wait_for_row_count(table, pilot, 3)

        # Default sort: newest first — session-0 root, its fork right below.
        first_column = [table.get_row_at(i)[0].plain for i in range(3)]
        assert first_column == ["session0", "└ session2", "session1"]
        assert "Forked from: session0" in table._row_tooltips[1].plain
