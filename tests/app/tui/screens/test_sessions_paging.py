# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Surface filters, paging and the empty states of the sessions browser."""

from __future__ import annotations

import pytest
from textual.widgets import Button, DataTable, Input, Static

from chrys.app.tui.screens.sessions.screen import SessionsScreen, _SessionTable
from chrys.app.tui.widgets import Checkbox, HatchedEmptyState, PageNavigator
from chrys.foundation.models.session_surface import SessionSurface
from chrys.service.state.session_listing import SESSION_PAGE_SIZE
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for

from ._sessions_support import (
    FakeSessionStore,
    GatedPageStore,
    SessionsHostApp,
    confirm_delete,
    wait_for_blocked_loads,
    wait_for_load_idle,
)

TUI, CLI, ACP = SessionSurface.TUI, SessionSurface.CLI, SessionSurface.ACP


def _checked(screen: SessionsScreen) -> dict[str, bool]:
    return {surface.value: screen.query_one(f"#surface-{surface.value}", Checkbox).value for surface in SessionSurface}


def _pager(screen: SessionsScreen) -> tuple[bool, str, bool]:
    """Previous disabled, the page label, Next disabled."""
    navigator = screen.query_one("#session-pages", PageNavigator)
    return (
        navigator.query_one("#previous-page", Button).disabled,
        str(navigator.query_one("#page-number", Static).content),
        navigator.query_one("#next-page", Button).disabled,
    )


def _subtitle(screen: SessionsScreen) -> str:
    return str(screen.query_one("#container").border_subtitle)


def _note(screen: SessionsScreen) -> str | None:
    """The empty note's text while it shows."""
    note = screen.query_one("#empty-note", HatchedEmptyState)
    return note.label if note.display else None


def _controls_shown(screen: SessionsScreen) -> bool:
    return bool(screen.query_one("#filters").display) and bool(screen.query_one("#footer").display)


async def _click_pager(pilot, screen: SessionsScreen, button: str) -> None:
    """Click Previous or Next: Textual ignores a click while the button still shows its last press."""
    target = screen.query_one(f"#session-pages #{button}", Button)
    await wait_for(lambda: not target.has_class("-active"), pilot=pilot, description=f"{button} takes a click")
    await click_when_settled(pilot, target)


async def _wait_for_page(screen: SessionsScreen, pilot, page: int) -> None:
    await wait_for(
        lambda: screen._page == page and not screen._loading,
        pilot=pilot,
        description=f"page {page} is loaded",
    )


@pytest.mark.asyncio
async def test_opens_on_the_first_page_of_tui_sessions_counting_unrecorded_ones_as_tui() -> None:
    store = FakeSessionStore(4, surfaces=[CLI, TUI, None, ACP])
    screen = SessionsScreen(store)

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)

        assert _checked(screen) == {"tui": True, "cli": False, "acp": False}
        assert store.pages_requested == [(frozenset({TUI}), 1)]
        assert screen._session_ids == ["session-1", "session-2"]
        assert _subtitle(screen) == "2 sessions"
        assert _pager(screen) == (True, "Page 1 of 1", True)
        tooltips = screen.query_one("#sessions", _SessionTable)._row_tooltips
        assert tooltips[0].plain.endswith("\nLast used in: TUI")
        assert tooltips[1].plain.endswith("\nLast used in: Unknown")


@pytest.mark.asyncio
async def test_opens_with_the_surfaces_it_is_given() -> None:
    store = FakeSessionStore(3, surfaces=[CLI, TUI, ACP])
    screen = SessionsScreen(store, surfaces={CLI, ACP})

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)

        assert _checked(screen) == {"tui": False, "cli": True, "acp": True}
        assert screen._session_ids == ["session-0", "session-2"]
        assert screen.query_one("#sessions", _SessionTable)._row_tooltips[1].plain.endswith("\nLast used in: ACP")


@pytest.mark.asyncio
async def test_workflow_sessions_offer_only_the_surfaces_that_start_runs() -> None:
    store = FakeSessionStore(0)
    reported: list[frozenset[SessionSurface]] = []
    screen = SessionsScreen(store, workflow_mode=True, surfaces={CLI, ACP}, on_surfaces_changed=reported.append)

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)

        assert [checkbox.id for checkbox in screen.query("#surface-filters Checkbox")] == [
            "surface-tui",
            "surface-cli",
        ]
        # ACP has no checkbox here, so it could never be unchecked.
        assert store.pages_requested == [(frozenset({CLI}), 1)]
        screen.query_one("#surface-tui", Checkbox).value = True
        await wait_for(lambda: reported == [frozenset({TUI, CLI})], pilot=pilot, description="TUI is added")


@pytest.mark.asyncio
async def test_toggling_a_surface_lists_the_first_page_of_the_new_selection() -> None:
    store = FakeSessionStore(SESSION_PAGE_SIZE + 20, surfaces=[CLI] * 5)
    reported: list[frozenset[SessionSurface]] = []
    screen = SessionsScreen(store, on_surfaces_changed=reported.append)

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)
        assert _subtitle(screen) == "115 sessions"
        await _click_pager(pilot, screen, "next-page")
        await _wait_for_page(screen, pilot, 2)
        assert screen._session_ids == [f"session-{index}" for index in range(105, 120)]

        await click_when_settled(pilot, "#surface-cli")
        await wait_for(
            lambda: store.pages_requested[-1] == (frozenset({TUI, CLI}), 1) and not screen._loading,
            pilot=pilot,
            description="the first page of TUI and CLI sessions is loaded",
        )

        assert reported == [frozenset({TUI, CLI})]
        assert _pager(screen) == (True, "Page 1 of 2", False)
        assert screen._session_ids[:6] == [f"session-{index}" for index in range(6)]
        assert _subtitle(screen) == "120 sessions"
        # Every load pages through the listing taken when the browser opened.
        assert store.listings_opened == ["chat"]


@pytest.mark.asyncio
async def test_filters_that_match_nothing_keep_the_controls_that_undo_them() -> None:
    store = FakeSessionStore(3, surfaces=[CLI] * 3)
    screen = SessionsScreen(store)

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)

        assert _note(screen) == "No sessions match the selected filters."
        assert not screen.query_one("#sessions").display
        assert _controls_shown(screen)
        assert not screen.query_one("#container").has_class("-empty")
        assert _subtitle(screen) == "0 sessions"
        assert _pager(screen) == (True, "Page 1 of 1", True)
        assert screen.query_one("#resume", Button).disabled
        assert not screen.query_one("#cancel", Button).disabled

        await click_when_settled(pilot, "#surface-cli")
        await wait_for(lambda: len(screen._session_ids) == 3, pilot=pilot, description="CLI sessions are listed")
        assert _note(screen) is None
        assert screen.query_one("#sessions").display


@pytest.mark.asyncio
async def test_without_saved_sessions_only_the_note_shows() -> None:
    screen = SessionsScreen(FakeSessionStore(0))

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)

        assert _note(screen) == "No saved sessions."
        assert not screen.query_one("#filters").display
        assert not screen.query_one("#footer").display
        assert screen.query_one("#container").has_class("-empty")
        assert _subtitle(screen) == ""


@pytest.mark.asyncio
async def test_search_covers_only_the_page_shown() -> None:
    store = FakeSessionStore(SESSION_PAGE_SIZE + 10)
    store.sessions[SESSION_PAGE_SIZE + 3].title = "needle on page two"
    screen = SessionsScreen(store)

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)

        screen.query_one("#search", Input).value = "needle"
        await wait_for(lambda: _note(screen) is not None, pilot=pilot, description="the search empties the page")
        assert _note(screen) == "No sessions on this page match your search."
        assert _subtitle(screen) == "0/100 sessions"
        assert _controls_shown(screen)

        await _click_pager(pilot, screen, "next-page")
        await wait_for(
            lambda: screen._session_ids == [f"session-{SESSION_PAGE_SIZE + 3}"],
            pilot=pilot,
            description="the search finds the session on page two",
        )
        assert _note(screen) is None
        assert _subtitle(screen) == "1/10 sessions"


@pytest.mark.parametrize(
    ("query", "lands_on"),
    [("needle", "#sessions"), ("nothing like it", "#search")],
    ids=["the-last-page-has-matches", "the-last-page-has-none"],
)
@pytest.mark.asyncio
async def test_turning_to_the_last_page_under_a_search_that_emptied_the_table_leaves_the_pager(
    query: str, lands_on: str
) -> None:
    store = FakeSessionStore(SESSION_PAGE_SIZE + 1)
    store.sessions[SESSION_PAGE_SIZE].title = "needle on page two"
    screen = SessionsScreen(store)

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)
        screen.query_one("#search", Input).value = query
        await wait_for(lambda: _note(screen) is not None, pilot=pilot, description="the search empties the page")

        next_page = screen.query_one("#session-pages #next-page", Button)
        next_page.focus()
        await wait_for(lambda: next_page.has_focus, pilot=pilot, description="Next has focus")
        await pilot.press("enter")
        await _wait_for_page(screen, pilot, 2)
        assert _pager(screen) == (False, "Page 2 of 2", True)
        # Not the Previous button, where another Enter would walk back to page one.
        assert pilot.app.focused is screen.query_one(lands_on)


@pytest.mark.asyncio
async def test_pager_walks_the_pages_and_hands_focus_to_the_table() -> None:
    store = FakeSessionStore(2 * SESSION_PAGE_SIZE + 50)
    screen = SessionsScreen(store)

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)
        table = screen.query_one("#sessions", DataTable)
        assert _pager(screen) == (True, "Page 1 of 3", False)
        assert table.row_count == SESSION_PAGE_SIZE
        assert _subtitle(screen) == "250 sessions"

        next_page = screen.query_one("#session-pages #next-page", Button)
        next_page.focus()
        await wait_for(lambda: next_page.has_focus, pilot=pilot, description="Next has focus")
        await pilot.press("enter")
        await _wait_for_page(screen, pilot, 2)
        assert _pager(screen) == (False, "Page 2 of 3", False)
        assert screen._session_ids[0] == f"session-{SESSION_PAGE_SIZE}"
        assert pilot.app.focused is table

        await _click_pager(pilot, screen, "next-page")
        await _wait_for_page(screen, pilot, 3)
        assert _pager(screen) == (False, "Page 3 of 3", True)
        assert table.row_count == 50
        # Not the Previous button, which Textual would pick when Next is disabled under focus.
        assert pilot.app.focused is table

        await _click_pager(pilot, screen, "previous-page")
        await _wait_for_page(screen, pilot, 2)
        assert _pager(screen) == (False, "Page 2 of 3", False)


@pytest.mark.asyncio
async def test_the_last_requested_load_wins() -> None:
    store = GatedPageStore(2 * SESSION_PAGE_SIZE, surfaces=[CLI] * 10)
    screen = SessionsScreen(store)

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)

        await _click_pager(pilot, screen, "next-page")
        await wait_for_blocked_loads(store, pilot, 1)
        await click_when_settled(pilot, "#surface-cli")
        await wait_for_blocked_loads(store, pilot, 2)
        await wait_for(lambda: store.cancelled_loads == 1, pilot=pilot, description="the page 2 load is cancelled")
        store.release_all()
        await wait_for_load_idle(screen, pilot)

        assert store.pages_requested[1:] == [(frozenset({TUI}), 2), (frozenset({TUI, CLI}), 1)]
        assert screen._page == 1
        assert screen._session_ids[0] == "session-0"


@pytest.mark.asyncio
async def test_deleting_from_a_full_page_pulls_the_next_session_up() -> None:
    store = FakeSessionStore(SESSION_PAGE_SIZE + 1)
    screen = SessionsScreen(store)

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)
        assert _pager(screen) == (True, "Page 1 of 2", False)

        screen.query_one("#sessions", DataTable).move_cursor(row=0)
        await confirm_delete(screen, pilot)
        await wait_for(
            lambda: f"session-{SESSION_PAGE_SIZE}" in screen._session_ids and not screen._loading,
            pilot=pilot,
            description="the next page's first session moves up",
        )

        assert len(screen._session_ids) == SESSION_PAGE_SIZE
        assert _pager(screen) == (True, "Page 1 of 1", True)
        assert _subtitle(screen) == "100 sessions"
        assert screen._get_selected_session_id() == "session-1"


@pytest.mark.asyncio
async def test_deleting_the_only_session_of_the_last_page_shows_the_page_before() -> None:
    store = FakeSessionStore(SESSION_PAGE_SIZE + 1)
    screen = SessionsScreen(store)

    async with SessionsHostApp().run_test(size=(120, 40)) as pilot:
        await pilot.app.push_screen(screen)
        await wait_for_load_idle(screen, pilot)
        await _click_pager(pilot, screen, "next-page")
        await _wait_for_page(screen, pilot, 2)
        assert screen._session_ids == [f"session-{SESSION_PAGE_SIZE}"]

        await confirm_delete(screen, pilot)
        await _wait_for_page(screen, pilot, 1)

        assert len(screen._session_ids) == SESSION_PAGE_SIZE
        assert _pager(screen) == (True, "Page 1 of 1", True)
        assert _note(screen) is None
