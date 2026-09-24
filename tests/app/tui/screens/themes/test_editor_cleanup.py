# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Recover malformed user files and reclaim the detached, frozen editor dock."""

from __future__ import annotations

import asyncio
import gc
import weakref
from pathlib import Path
from unittest.mock import patch

import pytest
from textual._time import get_time
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Select
from textual.widgets._toast import Toast

from chrys.app.tui.screens.main import MainScreen
from chrys.app.tui.screens.themes.editor import ResettableThemeEditor
from chrys.app.tui.screens.themes.picker import ThemesScreen
from chrys.app.tui.support.gc_freeze import GcAbsorbReason, GcReclaimReason
from chrys.app.tui.theme import CHRYS_LEGACY_THEME
from chrys.app.tui.themes.document import copy_theme
from chrys.app.tui.themes.store import UserThemeStore
from chrys.foundation.config.settings import Settings
from tests.support.tui_app_harness import make_chrys_app
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for

from .helpers import NotificationCapture, make_app, open_editor, wait_for_confirmation, wait_for_editor, wait_for_themes


@pytest.mark.parametrize(
    ("payload", "diagnostic", "locale"),
    [("primary: [oops\n", "[oops", "en"), ("primary: '[bold]oops[/bold]'\n", "[bold]oops[/bold]", "zh-Hans")],
)
async def test_malformed_applied_theme_shows_literal_error_and_can_be_reopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, locale: str, payload: str, diagnostic: str
) -> None:
    directory = tmp_path / "themes"
    monkeypatch.setattr("chrys.app.tui.themes.store.default_theme_directory", lambda: directory)
    monkeypatch.setattr("chrys.app.tui.theme_loader.default_theme_directory", lambda: directory)
    revision = UserThemeStore(directory).save(copy_theme(CHRYS_LEGACY_THEME, name="custom"), None)
    original = revision.path.read_bytes()
    app = make_app(tmp_path, "custom", locale)
    capture = NotificationCapture()
    async with app.run_test(size=(140, 50), notifications=True, message_hook=capture) as pilot:
        await pilot.pause()
        main = app.screen
        assert isinstance(main, MainScreen)
        revision.path.write_text(payload, encoding="utf-8")
        await pilot.press("f9")
        await wait_for_themes(pilot)
        await pilot.press("end", "enter")
        await wait_for(lambda: bool(capture.notifications), pilot=pilot, description="theme error delivered")
        (toast,) = capture.notifications
        # Exercise Textual's literal rendering even if the live toast has expired.
        assert diagnostic in Toast(toast).render().plain
        assert toast.message.startswith("Could not open the theme:" if locale == "en" else "无法打开主题")
        assert not toast.markup and toast.severity == "error"
        assert isinstance(app.screen, ThemesScreen)
        assert main.theme_editor is None and app.theme_preview is None
        assert app.theme == app._settings.theme == "custom"

        revision.path.write_bytes(original)
        await pilot.press("end", "enter")
        # Wait for the dock to be mounted, not merely assigned: ending the test while its
        # editor still composes hands that compose to the app's shutdown prune.
        editor = await wait_for_editor(pilot)
        assert editor.document.draft.name == "custom"
        assert editor.revision == revision


async def test_dock_removed_before_the_editor_composes_leaves_no_stale_refresh(tmp_path: Path) -> None:
    """A prune that lands mid-compose mounts nothing; the Mount-time refresh must not look for the controls."""
    app = make_app(tmp_path, "chrys-legacy")
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        main = app.screen
        assert isinstance(main, MainScreen)
        source = app.get_theme("chrys-legacy")
        assert source is not None
        compose = ResettableThemeEditor.compose

        def compose_under_removal(editor: ResettableThemeEditor) -> ComposeResult:
            assert main.theme_editor is not None
            main.theme_editor.remove()  # marks the dock and this editor pruning before its children can mount
            yield from compose(editor)

        with patch.object(ResettableThemeEditor, "compose", compose_under_removal):
            await main.open_theme_editor(source, UserThemeStore(tmp_path / "themes"))
        await wait_for(
            lambda: main.theme_editor is None and app.theme_preview is None,
            pilot=pilot,
            description="removed dock closed and ended its preview",
        )
        editor = await open_editor(app, pilot, tmp_path)
        assert editor.query_one("#theme-select", Select).value == "chrys-legacy"


def _alive(reference: weakref.ReferenceType[Widget]) -> bool:
    """Read a weakref without retaining its target in the test's async frame."""
    return reference() is not None


@pytest.mark.parametrize("dirty", [False, True])
async def test_close_reclaims_frozen_editor_only_after_removal(tmp_path: Path, dirty: bool) -> None:
    app = make_chrys_app(tmp_path, settings=Settings(theme="chrys-legacy"), gc_freeze_enabled=True)
    coordinator = app._gc_freeze
    offset = 0.0

    def clock() -> float:
        return get_time() + offset

    with patch.object(coordinator, "_clock", clock):
        async with app.run_test(size=(140, 50)) as pilot:
            await wait_for(lambda: coordinator.frozen, pilot=pilot)
            assert app._gc_freeze_watchdog is not None
            app._gc_freeze_watchdog.pause()
            main = app.screen
            assert isinstance(main, MainScreen)
            await main.manage_themes()
            panel = main.theme_editor
            assert panel is not None
            await wait_for(
                lambda: panel is not None and app.focused is panel.editor._color_buttons["primary"], pilot=pilot
            )
            if dirty:
                token = panel.editor.document.begin("color:primary")
                panel.editor.preview_edit(token, "#123456")
                assert panel.editor.commit_edit(token)
            references = tuple(
                weakref.ref(widget) for widget in (panel, panel.editor, panel.editor._color_buttons["primary"])
            )
            previous_metrics = coordinator.last_action_metrics
            coordinator.request_absorb(reason=GcAbsorbReason.TURN_TERMINAL, terminal_boundary=True)
            await wait_for(lambda: coordinator.last_action_metrics is not previous_metrics, pilot=pilot)
            assert coordinator.last_action_metrics is not None
            assert coordinator.last_action_metrics.action == "absorb"

            request_reclaim = coordinator.request_reclaim

            def reclaimed(*, reason: GcReclaimReason, prompt: bool, requested_at: float | None = None) -> None:
                if reason is GcReclaimReason.STABLE_CONTENT_REMOVED:
                    assert not prompt
                    assert app.theme_preview is None
                    for reference in references:
                        widget = reference()
                        assert widget is None or (not widget.is_attached and not widget.is_running)
                request_reclaim(reason=reason, prompt=prompt, requested_at=requested_at)

            with patch.object(coordinator, "request_reclaim", autospec=True, side_effect=reclaimed) as reclaim:
                await click_when_settled(pilot, "#theme-close")
                if dirty:
                    await wait_for_confirmation(pilot)
                    await pilot.press("escape")
                    await wait_for(lambda: app.screen is main, pilot=pilot)
                    assert main.theme_editor is panel and panel.editor.document.unsaved
                    reclaim.assert_not_called()
                    await click_when_settled(pilot, "#theme-close")
                    await wait_for_confirmation(pilot)
                    await pilot.click("#confirm-yes")
                await wait_for(lambda: main.theme_editor is None and coordinator._idle_reclaim_pending, pilot=pilot)
                reclaim.assert_called_once()
                assert coordinator._idle_reclaim_reasons == {GcReclaimReason.STABLE_CONTENT_REMOVED}
                assert app.theme_preview is None and app.current_theme == app.get_theme("chrys-legacy")
                panel = None
                gc.collect()
                assert _alive(references[0])

                drained = asyncio.Event()
                assert app.call_later(drained.set)
                await wait_for(drained.is_set, pilot=pilot)
                offset += 4.0
                coordinator.on_tick()
                await wait_for(lambda: not coordinator._idle_reclaim_pending, pilot=pilot)
                assert coordinator.last_action_metrics is not None
                assert coordinator.last_action_metrics.action == "full"
                assert coordinator.last_action_metrics.idle_reclaim_reasons == ("stable_content_removed",)
                assert coordinator.frozen
                # The dock has no live owners after close. Still-mounted
                # Textual pumps may retain a recent message's editor sender;
                # those references are not unreachable cycles for GC to free.
                assert not _alive(references[0])
