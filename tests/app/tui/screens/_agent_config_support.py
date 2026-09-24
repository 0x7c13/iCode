# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared host app, waits, and factories for the agent configuration screen tests."""

from __future__ import annotations

import asyncio
import contextlib
import stat
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from textual.app import App
from textual.css.query import NoMatches
from textual.pilot import Pilot
from textual.widgets import Button, Input, OptionList, TabbedContent

from chrys.app.tui.screens.agents.config import AgentsConfigScreen
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import AgentProfile
from tests.support.tui_helpers import PlaceholderHostApp
from tests.support.waiting import shared_wait_deadline

# One polling budget is shared by every wait helper in a test (see
# shared_wait_deadline). Heavy multi-tab tests spend it across several
# hydration waits, and Windows CI under xdist load has pushed past 20s
# total — the deadline is a ceiling, not a sleep, so fast tests are
# unaffected by the headroom.
_DEFAULT_WAIT_TIMEOUT = 45.0


async def _wait_for_selectors(screen, pilot, *selectors: str, timeout: float = _DEFAULT_WAIT_TIMEOUT) -> None:
    """Poll until every selector resolves on the screen.

    Re-mounting a config panel (e.g. after Clone) yields its children in
    batches; a single ``pilot.pause()`` is not always enough to drain
    them on slower CI hosts. Waiting on the specific widgets that the
    next interaction needs makes the test deterministic.
    """
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        try:
            for sel in selectors:
                screen.query_one(sel)
            return
        except NoMatches:
            if loop.time() > deadline:
                raise
            await pilot.pause(0.05)


async def _wait_for_input_enabled(screen, pilot, selector: str, timeout: float = _DEFAULT_WAIT_TIMEOUT) -> None:
    """Poll until an Input widget matching *selector* exists and is enabled."""
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        try:
            widget = screen.query_one(selector, Input)
            if not widget.disabled:
                return
        except NoMatches:
            pass
        if loop.time() > deadline:
            raise AssertionError(f"{selector} did not become enabled within {timeout}s")
        await pilot.pause(0.05)


def _agent_config_clone_save_debug(screen: AgentsConfigScreen, registry: AgentProfileRegistry, copied_name: str) -> str:
    """Return a compact snapshot for diagnosing clone-save timing failures."""
    selected = screen._drafts.get(screen._selected_draft_key)
    dirty_names = [draft.profile.name for draft in screen._drafts.values() if draft.dirty]
    registry_names = registry.list_names()
    return (
        f"selected={screen._selected_profile_name!r}; "
        f"selected_draft={selected.profile.name if selected is not None else None!r}; "
        f"hydrating={screen._hydrating!r}; "
        f"save_disabled={screen.query_one('#ac-save', Button).disabled!r}; "
        f"copied_registered={registry.get(copied_name) is not None!r}; "
        f"dirty={dirty_names!r}; "
        f"registry={registry_names!r}"
    )


async def _wait_for_hydrated(screen: AgentsConfigScreen, pilot, timeout: float = _DEFAULT_WAIT_TIMEOUT) -> None:
    """Wait until the agent config screen has finished hydrating its panels.

    ``_finish_hydrating`` clears ``screen._hydrating`` after its retry
    budget is exhausted even when the mounted panels never matched the
    draft (only a warning is logged). On slower Windows CI under xdist
    load that give-up path can fire for profiles with several sub-agent
    rows, so checking the flag alone is a false-positive. Also require
    a fresh panel rebuild to equal the draft before returning.
    """
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        last_state = f"hydrating={screen._hydrating!r}"
        if not screen._hydrating:
            draft = screen._drafts.get(screen._selected_draft_key)
            if draft is None:
                return
            try:
                if screen._build_profile_from_mounted_panels(draft) == draft.profile:
                    return
                last_state += "; mounted panels != draft"
            except Exception as exc:  # transient — a panel is still mounting
                last_state += f"; panels not readable: {exc!r}"
        if loop.time() > deadline:
            raise AssertionError(f"agent config screen did not finish hydrating ({last_state})")
        await pilot.pause(0.05)


async def _wait_for_confirm_button(app: App, pilot, timeout: float = _DEFAULT_WAIT_TIMEOUT) -> Button:
    """Poll until a ConfirmDialog is the active screen and its confirm button is mounted.

    ``push_screen`` composes the dialog asynchronously and its buttons live
    inside a nested ``DialogButtonRow``; a single ``pilot.pause()`` after the
    triggering press is not a reliable mount barrier on slower CI hosts.
    """
    from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog

    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        dialog = app.screen
        if isinstance(dialog, ConfirmDialog):
            try:
                button = dialog.query_one("#confirm-yes", Button)
            except NoMatches:
                pass
            else:
                if button.is_mounted:
                    return button
        if loop.time() > deadline:
            raise AssertionError(f"confirm dialog did not open (active screen: {app.screen!r})")
        await pilot.pause(0.05)


async def _wait_for_active_screen(app: App, pilot, screen, timeout: float = _DEFAULT_WAIT_TIMEOUT) -> None:
    """Poll until *screen* is the active screen again (e.g. a modal was dismissed)."""
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while app.screen is not screen:
        if loop.time() > deadline:
            raise AssertionError(f"{screen!r} did not become active (active screen: {app.screen!r})")
        await pilot.pause(0.05)


async def _activate_agent_config_tab(screen: AgentsConfigScreen, pilot, tab_id: str) -> None:
    screen.query_one("#ac-tabs", TabbedContent).active = tab_id
    await pilot.pause()
    await _wait_for_hydrated(screen, pilot)


async def _wait_for_panel_display_name(
    screen: AgentsConfigScreen,
    pilot,
    expected: str,
    timeout: float = _DEFAULT_WAIT_TIMEOUT,
) -> None:
    """Wait until the mounted panels are readable and reflect an edited display name.

    Pressing Save while ``_build_profile_from_mounted_panels`` can still raise
    ``_AgentConfigPanelsNotReady`` (a tab transiently not queryable under slow
    Windows CI) makes ``_on_save`` abort with a "Save Error" and silently skip the
    write. Confirm the edit is harvestable before saving so the press is deterministic.
    """
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        draft = screen._drafts.get(screen._selected_draft_key)
        if draft is not None and not screen._hydrating:
            try:
                if screen._build_profile_from_mounted_panels(draft).display_name == expected:
                    return
            except Exception:
                pass  # transient — a panel is still mounting
        if loop.time() > deadline:
            raise AssertionError(f"panels did not reflect display name {expected!r}")
        await pilot.pause(0.05)


async def _wait_for_selected_profile_name(
    screen: AgentsConfigScreen,
    pilot,
    expected: str,
    timeout: float = _DEFAULT_WAIT_TIMEOUT,
) -> None:
    """Wait until the selected draft has the expected profile name.

    Some callers use this after editing the mounted Basic tab; others use it
    after a clone button press. In both cases the thing under test is the
    staged draft name, not the full tab hydration lifecycle. Requiring
    ``_hydrating`` to be false made clone-name tests fail under CI load even
    after the clone had been selected correctly.
    """
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        draft = screen._drafts.get(screen._selected_draft_key)
        if draft is not None and draft.profile.name == expected:
            return
        if loop.time() > deadline:
            actual = draft.profile.name if draft is not None else None
            raise AssertionError(
                f"selected profile did not become {expected!r}; got {actual!r}; hydrating={screen._hydrating!r}"
            )
        await pilot.pause(0.05)


def _registry() -> AgentProfileRegistry:
    registry = AgentProfileRegistry()
    registry.load_builtins()
    return registry


def _draft_for_profile(screen: AgentsConfigScreen, profile_name: str):
    for draft in screen._drafts.values():
        if draft.profile.name == profile_name:
            return draft
    raise AssertionError(f"Draft for {profile_name!r} not found")


def _draft_for_original(screen: AgentsConfigScreen, original_name: str):
    for draft in screen._drafts.values():
        if draft.original_name == original_name:
            return draft
    raise AssertionError(f"Draft originally named {original_name!r} not found")


def _sidebar_text(screen: AgentsConfigScreen, draft_key: str) -> str:
    option = screen.query_one("#ac-list", OptionList).get_option(draft_key)
    return option.prompt.plain


def _file_mode(path: Path) -> int | None:
    """Return the permission bits of *path* on POSIX; Windows has no comparable mode."""
    if sys.platform == "win32":
        return None
    return stat.S_IMODE(path.stat().st_mode)


_DEFAULT_INSTRUCTIONS = "Follow the user's instructions."


def make_profile(name: str, **overrides: Any) -> AgentProfile:
    """Build a minimal valid custom profile; *overrides* are ``AgentProfile`` fields."""
    fields: dict[str, Any] = {"instructions": _DEFAULT_INSTRUCTIONS}
    fields.update(overrides)
    return AgentProfile(name=name, **fields)


def registry_with(*profiles: AgentProfile) -> AgentProfileRegistry:
    """Return a registry without built-ins holding exactly *profiles*."""
    registry = AgentProfileRegistry()
    for profile in profiles:
        registry.register(profile)
    return registry


@contextlib.asynccontextmanager
async def open_agent_config(
    registry: AgentProfileRegistry,
    *,
    size: tuple[int, int] = (120, 40),
    hydrated: bool = True,
    **screen_kwargs: Any,
) -> AsyncIterator[tuple[AgentsConfigScreen, Pilot]]:
    """Mount ``AgentsConfigScreen`` on a bare host app and yield it with the pilot.

    ``hydrated=False`` returns right after the first ``pilot.pause()`` for
    tests that probe the screen before panel hydration has finished.
    """
    app = PlaceholderHostApp()
    async with app.run_test(size=size) as pilot:
        screen = AgentsConfigScreen(registry, **screen_kwargs)
        await app.push_screen(screen)
        await pilot.pause()
        if hydrated:
            await _wait_for_hydrated(screen, pilot)
        yield screen, pilot
