# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real dock entry must coalesce layout without delaying menu interaction."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from unittest.mock import patch

import pytest
from textual.geometry import Region, Size
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import OptionList
from textual.widgets._select import SelectOverlay

from chrys.app.tui.screens.main import MainScreen
from chrys.app.tui.screens.settings.dialog import SettingsDialog
from chrys.app.tui.screens.themes.picker import ThemesScreen
from chrys.app.tui.theme import CHRYS_LEGACY_THEME
from chrys.app.tui.themes.document import copy_theme
from chrys.app.tui.themes.store import ThemeFileRevision, UserThemeStore
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chat.tool_call import ToolCall
from chrys.app.tui.widgets.select import Select
from chrys.app.tui.widgets.sidebar.panel import SidebarPanel
from tests.support.tui_app_harness import EmptyAgentRegistry
from tests.support.waiting import wait_for

from .helpers import NotificationCapture, make_app, open_editor, wait_for_themes


@pytest.mark.parametrize("locale", ["en", "zh-Hans"])
async def test_f9_with_editor_open_shows_localized_toast(tmp_path: Path, locale: str) -> None:
    app = make_app(tmp_path, "chrys-legacy")
    capture = NotificationCapture()
    async with app.run_test(size=(140, 50), notifications=True, message_hook=capture) as pilot:
        editor = await open_editor(app, pilot, tmp_path)
        main = app.screen
        document = editor.document
        app.locale_controller.switch_locale(locale)
        await pilot.press("f9")
        await wait_for(lambda: bool(capture.notifications), pilot=pilot, description="theme warning delivered")
        (toast,) = capture.notifications
        expected = (
            "请先关闭主题编辑器，再切换应用主题。"  # noqa: RUF001
            if locale == "zh-Hans"
            else "Close the theme editor before changing the applied theme."
        )
        assert toast.message == expected and toast.severity == "warning" and not toast.markup
        assert app.screen is main
        assert editor.document is document
        assert app.focused is editor._color_buttons["primary"]
        assert app.theme == app._settings.theme == "chrys-legacy"


# Both entry paths at both widths: compact layout must also follow browsing.
@pytest.mark.parametrize("width", [100, 140])
@pytest.mark.parametrize("browse", [False, True])
async def test_edit_entry_coalesces_layout_and_defers_hidden_menu_geometry(
    tmp_path: Path, browse: bool, width: int
) -> None:
    app = make_app(tmp_path, "chrys-legacy")
    async with app.run_test(size=(width, 50)) as pilot:
        await pilot.pause()
        main = app.screen
        assert isinstance(main, MainScreen)
        chat = main.query_one(ChatPanel)
        # Keep an overflowing real transcript. Assertions check when layout
        # and style work occurs, rather than measuring per-card throughput.
        cards = [ToolCall(f"open-{i}", "read_file", args={"path": f"file-{i}.py"}) for i in range(12)]
        await chat.mount(*cards)
        for card in cards:
            card.set_complete("File contents\n" * 12)
        await wait_for(lambda: chat.virtual_size.height > chat.size.height, pilot=pilot)
        await pilot.press("f9")
        await wait_for_themes(pilot)
        picker = app.screen
        assert isinstance(picker, ThemesScreen)
        options = picker.query_one(OptionList)
        if browse:
            options.highlighted = options.get_option_index("dracula")
            await wait_for(lambda: app.theme == "dracula", pilot=pilot)
        options.highlighted = options.option_count - 1
        await pilot.pause()
        geometry_reads: list[bool] = []
        original_region = Widget.scrollable_content_region.fget
        assert original_region is not None

        def menu_region(overlay: SelectOverlay) -> Region:
            geometry_reads.append(overlay.display)
            return original_region(overlay)

        original_arrange = main._compositor._arrange_root

        def arrange_root(root: Widget, size: Size, visible_only: bool):
            if root is main:
                # There must be no intermediate layout of the full-width chat,
                # nor a geometry query into a partially mounted editor subtree.
                assert app.screen is main
                assert main.theme_editor is not None and main.theme_editor.is_mounted
            return original_arrange(root, size, visible_only)

        with (
            patch.object(SelectOverlay, "scrollable_content_region", property(menu_region)),
            patch.object(main._compositor, "_arrange_root", autospec=True, side_effect=arrange_root) as arrange,
            patch.object(app, "refresh_css", autospec=True, side_effect=app.refresh_css) as refresh,
            patch("chrys.app.tui.app.persist_theme", autospec=True) as persist,
        ):
            await pilot.click(options, offset=(4, options.region.height - 1))
            await wait_for(
                lambda: (
                    main.theme_editor is not None and app.focused is main.theme_editor.editor._color_buttons["primary"]
                ),
                pilot=pilot,
            )
            await pilot.pause()
            panel = main.theme_editor
            assert panel is not None
            assert app.screen is main
            assert panel.editor.document.draft.name == "chrys-legacy"
            assert app.theme == app._settings.theme == "chrys-legacy"
            assert app.current_theme == app.get_theme("chrys-legacy")
            assert refresh.call_count == int(browse)
            persist.assert_not_called()
            assert not geometry_reads
            assert not main._theme_editor_opening
            assert chat.region.x >= panel.region.right
            assert chat.region.width > 0
            assert main.query_one(SidebarPanel).display == (width >= 128)
            # Count actual compositor work, including synchronous geometry reads
            # that bypass Screen._refresh_layout. Opening needs a dock reflow.
            main_layouts = [call for call in arrange.call_args_list if call.args[0] is main]
            assert main_layouts

            selector = panel.editor.query_one(Select)
            await pilot.click(selector)
            overlay = selector.query_one(SelectOverlay)
            await wait_for(lambda: selector.expanded and overlay.region.height > 0, pilot=pilot)
            assert geometry_reads and all(geometry_reads)
            assert overlay.highlighted is not None
            assert str(overlay.highlighted_option.prompt) == "chrys-legacy"
            # The highlighted theme is one row below what the menu shows unscrolled. The menu scrolls to
            # it when it handles Show, and its own message pump gets to that after the layout waited for.
            await wait_for(
                lambda: any("chrys-legacy" in overlay.render_line(y).text for y in range(overlay.content_size.height)),
                pilot=pilot,
                description="the open menu to show the highlighted theme",
            )
            await pilot.press("escape")
            assert not selector.expanded
            geometry_reads.clear()
            # Rebuild choices and move the highlight while collapsed, including
            # after the overlay has a populated line cache from its first open.
            with panel.editor.prevent(Select.Changed):
                selector.set_options([(name, name) for name in panel.editor._listed_themes])
                selector.value = "chrys-legacy"
            await pilot.pause()
            assert not geometry_reads
            await pilot.click(selector)
            await pilot.press("down", "escape")
            assert selector.value == "chrys-legacy"
            assert all(geometry_reads)

        assert main.query_one(ChatPanel) is chat
        assert len(chat.query(ToolCall)) == len(cards)


@pytest.mark.parametrize("while_loading", ["cancel", "cover", "continue"])
async def test_user_theme_loading_keeps_picker_responsive_and_checks_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, while_loading: str
) -> None:
    directory = tmp_path / "themes"
    monkeypatch.setattr("chrys.app.tui.themes.store.default_theme_directory", lambda: directory)
    monkeypatch.setattr("chrys.app.tui.theme_loader.default_theme_directory", lambda: directory)
    monkeypatch.setattr(EmptyAgentRegistry, "list_profiles", lambda self, include_sub_agent_only=False: [])
    UserThemeStore(directory).save(copy_theme(CHRYS_LEGACY_THEME, name="custom"), None)
    app = make_app(tmp_path, "custom")
    started, release = Event(), Event()
    original_load = UserThemeStore.load

    def blocked_load(store: UserThemeStore, name: str) -> tuple[Theme, ThemeFileRevision]:
        started.set()
        if not release.wait(10):
            raise TimeoutError("Test did not release theme loading")
        return original_load(store, name)

    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        main = app.screen
        assert isinstance(main, MainScreen)
        await pilot.press("f9")
        picker = await wait_for_themes(pilot)
        options = picker.query_one(OptionList)
        options.highlighted = options.option_count - 1
        await pilot.pause()
        with patch.object(UserThemeStore, "load", autospec=True, side_effect=blocked_load):
            try:
                await pilot.click(options, offset=(4, options.region.height - 1))
                await wait_for(started.is_set)
                worker = next(worker for worker in app.workers if worker.name == "open-theme-editor")
                assert not main._theme_editor_opening
                assert app._batch_count == 0
                if while_loading == "cancel":
                    await pilot.press("escape")
                    assert app.screen is main
                elif while_loading == "cover":
                    await app.push_screen(
                        SettingsDialog(main._settings_coordinator(), locale_controller=app.locale_controller)
                    )
                    assert isinstance(app.screen, SettingsDialog)
            finally:
                release.set()
            await worker.wait()
        assert not main._theme_editor_opening
        assert app._batch_count == 0
        assert not picker._opening_editor
        if while_loading == "continue":
            await wait_for(lambda: main.theme_editor is not None, pilot=pilot)
            assert main.theme_editor is not None
            assert main.theme_editor.editor.document.draft.name == "custom"
            assert main.theme_editor.editor.revision is not None
        else:
            assert main.theme_editor is None
            assert app.theme_preview is None
