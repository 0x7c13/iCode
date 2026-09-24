# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the ChrysApp AppFocus/AppBlur handlers on the native and web hosts."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Static

from tests.support.tui_app_harness import make_chrys_app
from tests.support.waiting import wait_for, wait_until_quiet


async def test_native_app_focus_preserves_internal_focus_without_layout_or_binding_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Native focus transitions must stay independent of mounted transcript size."""
    from textual import events

    from chrys.app.tui.widgets.chrome.input_bar import InputBar

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    app = make_chrys_app(tmp_path)

    async with app.run_test() as pilot:
        await app.screen.mount(*(Static("") for _ in range(64)))
        app.screen.query_one(InputBar).focus_input()
        await wait_for(
            lambda: app.screen.query_one(InputBar).query_one("#chat-input").has_focus,
            pilot=pilot,
            description="composer focus before interaction",
        )
        await pilot.pause()
        focused = app.screen.focused
        assert focused is not None
        assert len(app.screen.walk_children()) >= 64

        full_restyles: list[bool] = []
        binding_refreshes: list[None] = []
        layout_refreshes: list[None] = []
        monkeypatch.setattr(
            app.screen,
            "update_node_styles",
            lambda animate=True: full_restyles.append(animate),
        )
        monkeypatch.setattr(app.screen, "refresh_bindings", lambda: binding_refreshes.append(None))
        monkeypatch.setattr(app.screen, "_refresh_layout", lambda *_args, **_kwargs: layout_refreshes.append(None))

        # Deferred work from the 64-widget mount can land arbitrarily late on
        # loaded CI workers; drain it so anything recorded below is caused by
        # the focus events under test.
        await wait_until_quiet(
            lambda: (len(full_restyles), len(binding_refreshes), len(layout_refreshes)),
            description="initial focus refresh counters",
            pilot=pilot,
        )
        full_restyles.clear()
        binding_refreshes.clear()
        layout_refreshes.clear()

        await app.on_event(events.AppBlur())
        await pilot.pause()
        assert app.app_focus is False
        assert app.screen.focused is focused
        assert focused.has_focus is True
        assert app._last_focused_on_app_blur is None

        await app.on_event(events.AppFocus())
        await pilot.pause()
        assert app.app_focus is True
        assert app.screen.focused is focused
        assert focused.has_focus is True
        assert app._last_focused_on_app_blur is None
        assert full_restyles == []
        assert binding_refreshes == []
        assert layout_refreshes == []


async def test_native_app_focus_recovers_deferred_diff_footer_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A DiffScreen completed while unfocused must refresh its stock Footer."""
    import asyncio

    from textual import events

    from chrys.app.tui.screens.diff.screen import DiffScreen
    from chrys.app.tui.util.diff_entries import DiffFileEntry, DiffLoadResult
    from chrys.service.mutations.types import MutationOp

    load_started = asyncio.Event()
    release_load = asyncio.Event()
    entry = DiffFileEntry(
        path=str(tmp_path / "async.py"),
        rel_path="async.py",
        operation=MutationOp.MODIFY,
        old_path=None,
        before_text="before\n",
        after_text="after\n",
        is_binary=False,
        encoding="utf-8",
        bytes_changed=True,
    )

    async def load_data() -> DiffLoadResult:
        load_started.set()
        await release_load.wait()
        return DiffLoadResult(all_entries=[entry], per_period_entries={1: [entry]}, total_periods=1)

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    app = make_chrys_app(tmp_path)

    async with app.run_test() as pilot:
        screen = DiffScreen({}, cwd=str(tmp_path), load_data=load_data)
        await app.push_screen(screen)
        # Generous deadline: a loaded CI worker can take >1s just to run
        # the deferred load worker; a short cap here only converts
        # scheduler contention into flakes (no cost on the happy path).
        await asyncio.wait_for(load_started.wait(), timeout=10)
        await app.on_event(events.AppBlur())
        release_load.set()

        await wait_for(
            lambda: screen._content_ready,
            timeout=10.0,
            pilot=pilot,
            description="diff screen content ready",
        )
        await wait_for(
            lambda: "go_back" in {key.action for key in screen.query("FooterKey")},
            pilot=pilot,
            description="blurred diff footer Back binding",
        )
        footer_actions = {key.action for key in screen.query("FooterKey")}
        assert "go_back" in footer_actions
        assert "toggle_view" not in footer_actions
        assert "toggle_change_list" not in footer_actions

        await app.on_event(events.AppFocus())
        await wait_for(
            lambda: (
                {"go_back", "toggle_view", "toggle_change_list"} <= {key.action for key in screen.query("FooterKey")}
            ),
            pilot=pilot,
            description="refocused diff footer full bindings",
        )


async def test_web_app_focus_retains_textual_full_style_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The web host may visibly depend on Textual's App:focus/App:blur CSS."""
    from textual import events

    monkeypatch.setenv("TEXTUAL_DRIVER", "textual.drivers.web_driver:WebDriver")
    app = make_chrys_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.pause()
        full_restyles: list[bool] = []
        monkeypatch.setattr(
            app.screen,
            "update_node_styles",
            lambda animate=True: full_restyles.append(animate),
        )

        await app.on_event(events.AppBlur())
        await app.on_event(events.AppFocus())

        assert full_restyles == [True, True]
