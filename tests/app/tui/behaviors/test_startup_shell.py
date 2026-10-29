# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The initial loading screen owns and drains its deferred startup work."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

import pytest
from rich.cells import cell_len

from chrys.app.tui import app as tui_module
from chrys.app.tui.app import ChrysApp
from chrys.app.tui.util.logo import TUI_LOGO
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import AgentProfile
from chrys.service.state.store import JsonFileStateStore
from tests.support.tui_app_harness import EmptyAgentRegistry, ShutdownOnlyEngine
from tests.support.waiting import DEFAULT_WAIT_TIMEOUT, MAIN_SCREEN_BUILD_TIMEOUT, wait_for


class StartupApp(ChrysApp):
    CSS_PATH: ClassVar = [Path(tui_module.__file__).parent / path for path in ChrysApp.CSS_PATH]


def _startup_app(
    root: Path,
    cls: type[StartupApp] = StartupApp,
    *,
    agent_registry: AgentProfileRegistry | None = None,
) -> StartupApp:
    return cls(
        EventBus(),
        ShutdownOnlyEngine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(root),
        agent_registry=agent_registry if agent_registry is not None else EmptyAgentRegistry(),  # type: ignore[arg-type]
        show_startup_loading=True,
        gc_freeze_enabled=False,
    )


def _held_startup() -> tuple[type[StartupApp], asyncio.Event]:
    """An app whose loading shell stays up until the app shuts down, and the event set once the startup holds it."""
    preparing = asyncio.Event()

    class HeldStartup(StartupApp):
        async def _prepare_main_screen(self):
            preparing.set()
            # Held until the app's shutdown cancels the startup.
            await asyncio.Event().wait()
            raise AssertionError("nothing releases the startup")

    return HeldStartup, preparing


@pytest.mark.parametrize(("width", "height"), [(100, 30), (20, 12)], ids=["wide", "narrower-than-the-logo"])
async def test_loading_shell_centers_the_logo_over_its_dots(tmp_path: Path, width: int, height: int) -> None:
    held, preparing = _held_startup()
    app = _startup_app(tmp_path, held)
    async with app.run_test(size=(width, height)):
        task = app._startup_task
        assert task is not None
        await wait_for(lambda: preparing.is_set() or task.done(), description="the loading shell held on screen")
        if task.done():
            await task
        assert preparing.is_set()
        rows = [strip.text.rstrip() for strip in app.screen._compositor.render_strips()]
        logo_color = app.screen.query_one("#startup-logo").styles.color
        dots_color = app.screen.query_one("#startup-loading").styles.color
    # Leaving run_test while the startup is held cancels and drains it, as an exit during loading does, so no main
    # screen is built: the shell is all this test reads, and a loaded runner can stretch that CPU-bound build past a
    # default wait.
    assert task.cancelled()

    logo = TUI_LOGO.strip("\n").splitlines()
    left = max(0, (width - max(cell_len(line) for line in logo)) // 2)
    top = next(y for y, row in enumerate(rows) if row)
    # The art keeps its rows whole: a narrow terminal crops it instead of wrapping it.
    assert rows[top : top + len(logo)] == [(" " * left + line)[:width].rstrip() for line in logo]
    blank, dots, *below = rows[top + len(logo) :]
    assert blank == ""
    assert dots.strip()
    assert abs(2 * (len(dots) - len(dots.lstrip())) + cell_len(dots.lstrip()) - width) <= 1
    assert not any(below)
    assert abs(top - len(below)) <= 1
    assert dots_color == logo_color


async def test_loading_shell_finishes_without_an_available_profile(tmp_path: Path) -> None:
    app = _startup_app(tmp_path)
    async with app.run_test():
        task = app._startup_task
        assert task is not None
        await wait_for(
            task.done, description="loading shell without an agent profile", timeout=MAIN_SCREEN_BUILD_TIMEOUT
        )
        await task
        assert app.screen is app._main_screen
        indicator = app._startup_loading_indicator
        assert indicator is not None
        timer = indicator._auto_refresh_timer
        assert timer is not None
        assert not timer._active.is_set()


async def test_exit_during_loading_drains_the_startup_task(tmp_path: Path) -> None:
    held, preparing = _held_startup()
    app = _startup_app(tmp_path, held)
    async with app.run_test():
        task = app._startup_task
        assert task is not None
        await wait_for(lambda: preparing.is_set() or task.done(), description="deferred main screen construction")
        if task.done():
            await task
        assert preparing.is_set()
        app.exit()
    assert task.cancelled()
    assert app._main_screen is None


@pytest.mark.parametrize("exit_at_refresh", [1, 2], ids=["loading-shell", "main-screen"])
async def test_exit_requested_at_startup_refresh_stops_further_startup(tmp_path: Path, exit_at_refresh: int) -> None:
    exit_requested = asyncio.Event()
    exit_states: list[tuple[bool, bool, bool]] = []
    main_builds: list[bool] = []
    engine_starts: list[bool] = []
    refresh_count = 0
    exit_after_next_refresh = False

    class ExitAtRefresh(StartupApp):
        async def _wait_for_startup_refresh(self) -> bool:
            nonlocal refresh_count, exit_after_next_refresh
            refresh_count += 1
            exit_after_next_refresh = refresh_count == exit_at_refresh
            return await super()._wait_for_startup_refresh()

        def call_after_refresh(self, callback, *args, **kwargs) -> bool:
            nonlocal exit_after_next_refresh
            if not exit_after_next_refresh:
                return super().call_after_refresh(callback, *args, **kwargs)
            exit_after_next_refresh = False

            def paint_then_exit() -> None:
                callback(*args, **kwargs)
                self.exit()
                exit_states.append((self._exit, self.is_running, self._closing))
                exit_requested.set()

            return super().call_after_refresh(paint_then_exit)

        def _build_main_screen(self):
            main_builds.append(self._exit)
            return super()._build_main_screen()

        async def _start_engine(self, profile, screen) -> None:
            engine_starts.append(self._exit)

    registry = AgentProfileRegistry()
    registry.register(AgentProfile(name="Code"))
    app = _startup_app(tmp_path, ExitAtRefresh, agent_registry=registry)
    async with app.run_test():
        task = app._startup_task
        assert task is not None
        await wait_for(
            lambda: exit_requested.is_set() or task.done(),
            description="exit requested at startup paint",
            # The main-screen case asks only once the screen is built.
            timeout=MAIN_SCREEN_BUILD_TIMEOUT if exit_at_refresh == 2 else DEFAULT_WAIT_TIMEOUT,
        )
        if task.done():
            await task
        assert exit_requested.is_set()
    assert exit_states == [(True, True, False)]
    assert main_builds == ([] if exit_at_refresh == 1 else [False])
    assert engine_starts == []
    assert task.done()
    assert not task.cancelled()


async def test_deferred_screen_error_reaches_textual_crash_handling(tmp_path: Path) -> None:
    class BrokenStartup(StartupApp):
        def _build_main_screen(self):
            raise RuntimeError("main screen construction failed")

    app = _startup_app(tmp_path, BrokenStartup)
    with pytest.raises(RuntimeError, match="main screen construction failed"):
        async with app.run_test():
            task = app._startup_task
            assert task is not None
            await wait_for(task.done, description="failed deferred main screen construction")
            await task
