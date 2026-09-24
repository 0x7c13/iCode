# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""User-theme deletion through the editor, with applied selection and draft isolation."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest
from textual.widgets import Button, Label, OptionList

from chrys.app.tui.screens.main import MainScreen
from chrys.app.tui.screens.settings.dialog import SettingsDialog
from chrys.app.tui.screens.themes.editor import ResettableThemeEditor
from chrys.app.tui.theme import CHRYS_LEGACY_THEME
from chrys.app.tui.theme_loader import load_user_themes
from chrys.app.tui.themes.document import copy_theme
from chrys.app.tui.themes.store import ThemeFileRevision, UserThemeStore
from chrys.app.tui.widgets.select import Select
from tests.support.tui_app_harness import EmptyAgentRegistry
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for

from .helpers import _EditorScreen, make_app, wait_for_confirmation, wait_for_editor, wait_for_themes


@pytest.mark.parametrize(("name", "expected"), [("custom", "other"), ("other", "chrys")])
async def test_delete_adopts_replacement_while_settings_covers_editor(
    name: str, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deleting an unapplied theme adopts the applied one; deleting the applied
    # theme adopts the default. Each case runs its own app: chained under one
    # startup, the two Settings mounts and their app-wide style passes made this
    # the slowest test of the Windows shard, whose CPU-bound runs vary enough to
    # reach the per-test timeout.
    store = theme_store(tmp_path, monkeypatch)
    untouched = store.save(copy_theme(CHRYS_LEGACY_THEME, name="untouched"), None)
    app = make_app(tmp_path, "other")
    original_delete = store.delete
    started, release = Event(), Event()

    def blocked_delete(name: str, revision: ThemeFileRevision) -> None:
        started.set()
        if not release.wait(10):
            raise TimeoutError("Test did not release the deletion")
        original_delete(name, revision)

    with (
        patch.object(EmptyAgentRegistry, "list_profiles", return_value=[]),
        patch("chrys.app.tui.app.persist_theme", autospec=True) as persist,
    ):
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            main = app.screen
            assert isinstance(main, MainScreen)
            theme, revision = store.load(name)
            await main.open_theme_editor(theme, store, revision)
            editor = await wait_for_editor(pilot)
            old_document = editor.document
            assert old_document.draft.name == name
            token = old_document.begin("color:primary")
            editor.preview_edit(token, "#123456")
            assert editor.commit_edit(token)
            with (
                patch.object(store, "delete", side_effect=blocked_delete) as delete,
                patch.object(editor, "recompose", wraps=editor.recompose) as recompose,
            ):
                await pilot.click("#theme-delete")
                await wait_for_confirmation(pilot)
                try:
                    app.screen.query_one("#confirm-yes", Button).press()
                    await wait_for(started.is_set)
                    assert editor.disabled and editor.delete_pending
                    assert main.theme_editor is not None
                    main.theme_editor.action_close()
                    editor._confirm_delete()
                    assert main.theme_editor is not None and app.screen is main
                    delete.assert_called_once_with(name, revision)
                    # Pilot would wait on the deliberately blocked editor pump.
                    # Only covering the editor is required here, not querying
                    # Settings controls or completing its deferred focus.
                    await app._press_keys(["f10"])
                    await wait_for(lambda: isinstance(app.screen, SettingsDialog) and app.screen.is_mounted)
                finally:
                    release.set()
                await wait_for(lambda: not editor.delete_pending)
                assert isinstance(app.screen, SettingsDialog)
                assert not revision.path.exists() and name not in app.available_themes
                assert editor.document is not old_document and editor.document.draft.name == expected
                assert not editor.document.unsaved and editor.document.undo_stack == []
                assert editor.revision == (store.load("other")[1] if expected == "other" else None)
                assert editor.disabled
                assert app.theme_preview is not None
                assert app.theme_preview.history == [] and app.theme_preview.baseline.name == expected
                assert app.current_theme.name == app.theme == app._settings.theme == expected
                if expected == "chrys":
                    persist.assert_called_once_with("chrys")
                else:
                    persist.assert_not_called()
                recompose.assert_not_called()
                await pilot.press("escape")
                assert await wait_for_editor(pilot) is editor
                assert not editor.disabled
                recompose.assert_awaited_once()
                assert editor.query_one(Select).value == expected
                assert name not in editor._listed_themes
                assert editor.query_one("#theme-delete", Button).display == (expected == "other")
                assert not editor.commit_edit(token)
            # Applied-theme deletion must pick the default even while another
            # user theme remains available, not merely when deleting the last.
            assert "untouched" in app.available_themes
            assert store.load("untouched")[1] == untouched


def theme_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UserThemeStore:
    directory = tmp_path / "themes"
    monkeypatch.setattr("chrys.app.tui.theme_loader.default_theme_directory", lambda: directory)
    monkeypatch.setattr("chrys.app.tui.themes.store.default_theme_directory", lambda: directory)
    store = UserThemeStore(directory)
    for name in ("custom", "other"):
        store.save(copy_theme(CHRYS_LEGACY_THEME, name=name), None)
    return store


async def test_cancel_then_delete_refreshes_lists_without_changing_the_applied_builtin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The covered-deletion cases exercise deleting the applied theme and an
    # unapplied user theme. Keep cancellation, localization and F9 in one
    # end-to-end scenario instead of repeating them for every selection.
    active = "chrys-legacy"
    store = theme_store(tmp_path, monkeypatch)
    app = make_app(tmp_path, active)
    theme, revision = store.load("custom")
    with patch("chrys.app.tui.app.persist_theme", autospec=True) as persist:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            main = app.screen
            assert isinstance(main, MainScreen)
            await main.open_theme_editor(theme, store, revision)
            editor = main.query_one(ResettableThemeEditor)
            await wait_for(lambda: app.focused is editor._color_buttons["primary"], pilot=pilot)
            delete = editor.query_one("#theme-delete", Button)
            close = editor.query_one("#theme-close", Button)
            assert delete.display and not delete.disabled and delete.variant == "error"
            assert delete.region.right + 1 == close.region.x
            token = editor.document.begin("color:primary")
            editor.preview_edit(token, "#123456")
            assert editor.commit_edit(token)
            draft = editor.document
            before = app.theme_variables.copy()
            await pilot.click(delete)
            await wait_for_confirmation(pilot)
            confirm = app.screen
            assert "custom" in confirm.query_one("#confirm-message").content.plain
            assert confirm.query_one("#confirm-yes", Button).variant == "error"
            app.locale_controller.switch_locale("zh-Hans")
            assert str(confirm.query_one("#confirm-yes", Button).label) == "删除"
            assert "custom" in confirm.query_one("#confirm-message").content.plain
            await pilot.click("#confirm-no")
            await wait_for(lambda: app.screen is main and not editor.delete_pending, pilot=pilot)
            assert editor.document is draft and draft.unsaved and len(draft.undo_stack) == 1
            assert app.theme_variables == before
            assert revision.path.exists()
            persist.assert_not_called()
            assert str(delete.label) == "删除"
            await click_when_settled(pilot, delete)
            await wait_for_confirmation(pilot)
            await pilot.click("#confirm-yes")
            expected = active
            await wait_for(lambda: not editor.delete_pending and editor.document.draft.name == expected, pilot=pilot)
            assert main.theme_editor is not None
            assert not revision.path.exists()
            assert "custom" not in app.available_themes and "custom" not in app._user_theme_names
            assert editor.query_one(Select).value == expected
            assert "custom" not in editor._listed_themes
            assert not editor.document.unsaved and editor.document.undo_stack == []
            assert app.theme_preview is not None and app.theme_preview.history == []
            assert app.theme_preview.baseline.name == expected
            editor.preview_edit(token, "#654321")
            assert not editor.commit_edit(token)
            assert app.theme == app._settings.theme == expected
            persist.assert_not_called()
            assert [item.name for item in load_user_themes(store.directory)[0]] == ["other"]
            assert editor.query_one("#theme-delete", Button).display == (expected == "other")
            assert (
                editor.query_one("#theme-close").region.right == editor.query_one("#theme-editor-body").region.right - 1
            )
            await pilot.click("#theme-close")
            await wait_for(lambda: main.theme_editor is None, pilot=pilot)
            assert app.current_theme.name == expected
            await pilot.press("f9")
            await wait_for_themes(pilot)
            options = app.screen.query_one(OptionList)
            assert all(options.get_option_at_index(i).id != "custom" for i in range(options.option_count))


async def test_failed_delete_retains_applied_theme_file_and_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = theme_store(tmp_path, monkeypatch)
    app = make_app(tmp_path, "custom", component=True)
    with patch("chrys.app.tui.app.persist_theme", autospec=True) as persist:
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            main = app.screen
            assert isinstance(main, _EditorScreen)
            theme, revision = store.load("custom")
            await main.open_theme_editor(theme, store, revision)
            editor = main.query_one(ResettableThemeEditor)
            await wait_for(lambda: app.focused is editor._color_buttons["primary"], pilot=pilot)
            token = editor.document.begin("color:primary")
            editor.preview_edit(token, "#123456")
            assert editor.commit_edit(token)
            draft = editor.document
            await click_when_settled(pilot, "#theme-delete")
            await wait_for_confirmation(pilot)
            revision.path.write_text("primary: blue\n")
            await pilot.click("#confirm-yes")
            error = editor.query_one("#editor-error", Label)
            await wait_for(lambda: error.display and not editor.delete_pending, pilot=pilot)
            assert "before deleting" in error.content.plain
            app.locale_controller.switch_locale("zh-Hans")
            assert "删除" in error.content.plain
            assert editor.document is draft and draft.unsaved and len(draft.undo_stack) == 1
            assert app.current_theme.primary == "#123456"
            assert app.theme == app._settings.theme == "custom"
            assert "custom" in app.available_themes
            assert revision.path.read_text() == "primary: blue\n"
            assert not editor.disabled
            persist.assert_not_called()
