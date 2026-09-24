# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Preview admission, lazy CSS, persistence isolation and real editor transactions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from rich.color import Color as RichColor
from rich.console import Console
from rich.text import Text
from textual.app import ComposeResult
from textual.css.stylesheet import Stylesheet, StylesheetError
from textual.css.tokenizer import TokenError
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Input, Static

from chrys.app.tui.screens.themes.dialogs import _PickerModal
from chrys.app.tui.screens.themes.editor import ResettableThemeEditor
from chrys.app.tui.theme import CHRYS_LEGACY_THEME, TUI_VARIABLE_DEFAULTS
from chrys.app.tui.theme_loader import _theme_from_data
from chrys.app.tui.themes.document import copy_theme
from chrys.app.tui.themes.preview import ThemePreview, validate_theme
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chat.renderers.todo import TodoToolCall
from chrys.app.tui.widgets.color_picker import ColorPicker
from chrys.app.tui.widgets.color_picker.controls import ColorPlane
from chrys.app.tui.widgets.diff_view.inline import InlineUnifiedDiffLines
from chrys.app.tui.widgets.diff_view.palette import LIGHT, DiffLook
from chrys.app.tui.widgets.sidebar.tasks import TasksPanel, TodoListState
from chrys.foundation.models.todos import TodoItem
from tests.support.waiting import wait_for

from .helpers import make_app as _app
from .helpers import open_editor, press_button, wait_for_editor, wait_for_picker


@pytest.mark.parametrize(
    ("variables", "css"),
    [
        ({"footer-background": "not-a-color"}, "background: $footer-background"),
        ({"primary-muted": "initial"}, "border: solid $primary-muted"),
        ({"ansi-foreground": "auto"}, "background: $ansi-foreground"),
    ],
)
def test_loaded_css_validation_rejects_values_the_yaml_loader_accepts(
    tmp_path: Path, variables: dict[str, str], css: str
) -> None:
    theme = _theme_from_data("custom", {"primary": "red", "variables": variables}, tmp_path / "custom.yaml")
    live = Stylesheet()
    live.add_source(f"* {{ {css}; }}")
    original_sources = live.source.copy()
    with pytest.raises((StylesheetError, TokenError)):
        validate_theme(theme, TUI_VARIABLE_DEFAULTS, live)
    assert live.source == original_sources
    assert not live._rules


@pytest.mark.parametrize("field", ["background", "surface", "panel", "boost"])
def test_transparent_base_surfaces_cannot_reach_rich_fallback(field: str) -> None:
    candidate = copy_theme(CHRYS_LEGACY_THEME)
    setattr(candidate, field, "transparent")
    with pytest.raises(ValueError, match=f"{field} requires an opaque color"):
        validate_theme(candidate, TUI_VARIABLE_DEFAULTS, Stylesheet())


class LateColor(Static):
    DEFAULT_CSS = "LateColor { color: $editor-late-color; }"


class LateColorDialog(ModalScreen[None]):
    DEFAULT_CSS = "LateColorDialog { color: $editor-late-color; }"

    def compose(self) -> ComposeResult:
        yield Input("preserve this dialog draft")


class WarningProbe(Static):
    DEFAULT_CSS = "WarningProbe { color: $warning; }"


async def test_first_preview_of_changed_theme_is_validated_even_with_same_name(tmp_path: Path) -> None:
    app = _app(tmp_path, component=True)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        original = copy_theme(app.current_theme)
        invalid = copy_theme(original)
        invalid.variables["footer-background"] = "not-a-color"
        variables = app.theme_variables.copy()
        with patch.object(app, "refresh_css", wraps=app.refresh_css) as refresh:
            assert not app.begin_theme_preview(invalid)
            assert app.theme_preview is not None and app.theme_preview.error
            assert app.current_theme == original
            assert app.theme_variables == variables
            refresh.assert_not_called()
            valid = copy_theme(original)
            valid.primary = "#123456"
            assert app.theme_preview.show(valid)
            assert app.current_theme == valid
            assert app.theme_variables["primary"] == "#123456"
            refresh.assert_called_once_with(animate=False)
        app.end_theme_preview()
        assert app.current_theme == original
        assert app.theme_variables == variables


@pytest.mark.parametrize("modal", [False, True])
async def test_runtime_lazy_css_rejects_preview_without_losing_input_or_theme(tmp_path: Path, modal: bool) -> None:
    app = _app(tmp_path, component=True)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        assert app.begin_theme_preview(copy_theme(app.current_theme))
        assert app.theme_preview is not None
        await wait_for(lambda: app.theme_preview.current is not None, pilot=pilot)
        valid = copy_theme(app.current_theme)
        valid.variables["editor-late-color"] = "#123456"
        assert app.theme_preview.show(valid)
        # A widget or dialog discovers this usage only when it is first opened.
        invalid = copy_theme(valid)
        invalid.variables["editor-late-color"] = "not-a-color"
        assert app.theme_preview.show(invalid, "var:editor-late-color")
        field = Input("preserve this draft")
        await app.screen.mount(field)
        if modal:
            probe = LateColorDialog()
            await app.push_screen(probe)
        else:
            probe = LateColor(Text("late widget"))
            await app.screen.mount(probe)
        await wait_for(lambda: app.current_theme == valid, pilot=pilot)
        assert field.value == "preserve this draft"
        assert app.theme == "chrys"
        assert app.theme_preview.error
        assert probe.styles.color.hex == "#123456"
        if modal:
            assert probe.query_one(Input).value == "preserve this dialog draft"


def test_lazy_css_recovery_retains_baseline_after_preview_history_overflows(tmp_path: Path) -> None:
    """Exercise history eviction with real CSS validation, without mounting an App.

    The widget/dialog tests above own live publication and deferred CSS discovery;
    each extra history entry here only needs to exercise admission and retention.
    """
    app = _app(tmp_path)
    preview = ThemePreview(app)
    valid = copy_theme(app.current_theme)
    valid.variables["editor-late-color"] = "#123456"
    with patch.object(preview, "_refresh", autospec=True):
        assert preview.show(valid)
        for i in range(25):
            invalid = copy_theme(valid)
            invalid.variables["editor-late-color"] = f"not-a-color-{i}"
            assert preview.show(invalid, "var:editor-late-color")
        assert 0 < len(preview.history) < 25
        assert valid not in preview.history
        assert preview.baseline == valid
        late = Stylesheet(variables=preview.variables)
        late.add_source(LateColor.DEFAULT_CSS)
        with pytest.raises(StylesheetError) as rejected:
            late.parse()
        with patch.object(app, "call_next", autospec=True) as schedule:
            assert preview.recover(late, rejected.value)
            schedule.assert_called_once_with(preview._refresh)
        assert preview.current == valid
        assert preview.error
        assert preview.variables["editor-late-color"] == "#123456"
        late.parse()


async def test_name_preview_preserves_palette_without_registering(tmp_path: Path) -> None:
    app = _app(tmp_path, "chrys-legacy", component=True)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        assert app.begin_theme_preview(copy_theme(app.current_theme))
        assert app.theme_preview is not None
        await wait_for(lambda: app.theme_preview.current is not None, pilot=pilot)
        # Not the gray the default theme would draw, were the selection to fall back to it.
        assert app.theme_variables["tui-border-primary"] == "#AF87FF"
        original_variables = app.theme_variables.copy()
        clone = copy_theme(CHRYS_LEGACY_THEME, name="chrys-legacy-copy")
        assert app.theme_preview.show(clone)
        assert app.current_theme.name == "chrys-legacy-copy"
        assert app.theme_variables == original_variables
        assert not app.has_class("-chrys")
        assert app.theme_preview.show(copy_theme(clone, name="my-copy"))
        assert app.theme_variables == original_variables
        assert not app.has_class("-chrys")
        assert "my-copy" not in app.available_themes
        assert "chrys-legacy-copy" not in app.available_themes
        app.end_theme_preview()
        assert app.current_theme.name == "chrys-legacy"
        assert app.theme_variables["tui-border-primary"] == "#AF87FF"


async def test_dialog_confirmation_cancel_and_undo_are_real_transactions(tmp_path: Path) -> None:
    app = _app(tmp_path, "textual-dark", component=True)
    async with app.run_test(size=(140, 50)) as pilot:
        await open_editor(app, pilot, tmp_path)
        await wait_for(lambda: bool(app.screen.query(ResettableThemeEditor)), pilot=pilot)
        editor = app.screen.query_one(ResettableThemeEditor)
        original = editor._theme.primary
        await press_button(pilot, editor._color_buttons["primary"])
        await wait_for_picker(pilot, ColorPicker)
        picker = app.screen.query_one(ColorPicker)
        picker.query_one("#color-expression", Input).value = "#123456"
        await wait_for(lambda: app.current_theme.primary == "#123456", pilot=pilot)
        assert editor._theme.primary == original
        await press_button(pilot, "#picker-confirm")
        await wait_for_editor(pilot)
        assert editor._theme.primary == "#123456"
        assert len(editor.document.undo_stack) == 1
        await press_button(pilot, editor._color_buttons["primary"])
        await wait_for_picker(pilot, ColorPicker)
        app.screen.query_one("#color-expression", Input).value = "#654321"
        await wait_for(lambda: app.current_theme.primary == "#654321", pilot=pilot)
        await pilot.press("escape")
        await wait_for_editor(pilot)
        assert app.current_theme.primary == "#123456"
        assert len(editor.document.undo_stack) == 1
        assert await pilot.click("#theme-undo")
        assert editor._theme.primary == original
        assert await pilot.click("#theme-redo")
        assert editor._theme.primary == "#123456"


async def test_rich_checklist_and_diff_follow_effective_theme(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        assert app.begin_theme_preview(copy_theme(app.current_theme))
        assert app.theme_preview is not None
        await wait_for(lambda: app.theme_preview.current is not None, pilot=pilot)
        tasks = app.screen.query_one(TasksPanel)
        tasks.todo_state = TodoListState(items=(TodoItem(content="work", status="in_progress"),))
        diff = InlineUnifiedDiffLines("a", "a", "old\n", "new\n")
        card = TodoToolCall(
            "theme-preview", "todo_write", args={"todos": [{"content": "work", "status": "in_progress"}]}
        )
        await app.screen.mount(diff, card)
        card.set_complete("Todo list updated.")
        candidate = copy_theme(app.current_theme)
        candidate.warning = "#123456"
        candidate.dark = False
        notifications: list[Theme] = []
        app.theme_changed_signal.subscribe(tasks, notifications.append)
        assert app.theme_preview.show(candidate)
        await wait_for(
            lambda: tasks._theme_style is not None and tasks._theme_style.color == RichColor.parse("#123456"),
            pilot=pilot,
        )
        await wait_for(
            lambda: card._theme_style is not None and card._theme_style.color == RichColor.parse("#123456"), pilot=pilot
        )
        assert app.has_class("-light-mode") and not app.has_class("-dark-mode")
        assert DiffLook.of(diff).palette is LIGHT
        content = tasks.query_one("#tasks-checklist", Static).content
        segments = list(Console(force_terminal=False, _environ={}).render(content))
        assert any(segment.style and segment.style.color == RichColor.parse("#123456") for segment in segments)
        assert notifications[-1].dark is False
        card_content = card.query_one("#tc-body", Static).content
        segments = list(Console(force_terminal=False, _environ={}).render(card_content))
        assert any(segment.style and segment.style.color == RichColor.parse("#123456") for segment in segments)


async def test_invalid_input_and_noop_preview_do_not_restyle_populated_transcript(tmp_path: Path) -> None:
    app = _app(tmp_path, "textual-dark")
    async with app.run_test(size=(140, 50)) as pilot:
        await open_editor(app, pilot, tmp_path)
        await wait_for(lambda: bool(app.screen.query(ResettableThemeEditor)), pilot=pilot)
        editor = app.screen.query_one(ResettableThemeEditor)
        chat = next(screen.query_one(ChatPanel) for screen in app.screen_stack if screen.query(ChatPanel))
        await chat.mount(*(Static(Text(f"Transcript line {i}")) for i in range(60)))
        await wait_for(lambda: chat.virtual_size.height > chat.size.height, pilot=pilot)
        await pilot.click(editor._color_buttons["primary"])
        await wait_for_picker(pilot, ColorPicker)
        field = app.screen.query_one("#color-expression", Input)
        with patch.object(app, "refresh_css", autospec=True, side_effect=app.refresh_css) as refresh:
            field.value = "#12"
            await pilot.pause()
            assert app.theme_preview is not None
            assert app.theme_preview.show(copy_theme(app.current_theme))
            await pilot.pause()
            refresh.assert_not_called()
        assert len(chat.query(Static)) >= 60


async def test_alpha_rich_checklists_match_css_and_reblend_on_background_changes(tmp_path: Path) -> None:
    app = _app(tmp_path)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.pause()
        assert app.begin_theme_preview(copy_theme(app.current_theme))
        assert app.theme_preview is not None
        await wait_for(lambda: app.theme_preview.current is not None, pilot=pilot)
        tasks = app.screen.query_one(TasksPanel)
        items = [{"content": "work", "status": "in_progress"}]
        tasks.todo_state = TodoListState(items=(TodoItem(content="work", status="in_progress"),))
        card = TodoToolCall("alpha-preview", "todo_write", args={"todos": items})
        await app.screen.mount(card)
        card.set_complete("Todo list updated.")
        task_probe, card_probe = WarningProbe(Text("CSS reference")), WarningProbe(Text("CSS reference"))
        await tasks.mount(task_probe)
        await card.mount(card_probe)
        candidate = copy_theme(app.current_theme)
        candidate.variables["warning"] = "#FF000080"
        assert app.theme_preview.show(candidate)
        await wait_for(
            lambda: tasks._theme_style is not None and tasks._theme_style.color == task_probe.rich_style.color,
            pilot=pilot,
        )
        await wait_for(
            lambda: card._theme_style is not None and card._theme_style.color == card_probe.rich_style.color,
            pilot=pilot,
        )
        before = card._theme_style
        assert before is not None and before.color != RichColor.parse("#FF0000")
        # Same warning expression, different background: cached Rich colors must change.
        candidate.background = candidate.surface = candidate.panel = candidate.boost = "#EEEEEE"
        assert app.theme_preview.show(candidate)
        await wait_for(lambda: card._theme_style != before, pilot=pilot)
        for owner, probe in ((tasks, task_probe), (card, card_probe)):
            await wait_for(
                lambda owner=owner, probe=probe: (
                    owner._theme_style is not None and owner._theme_style.color == probe.rich_style.color
                ),
                pilot=pilot,
            )
        assert app.theme_variables["warning"] == "#FF000080"


async def test_picker_event_burst_is_coalesced_and_confirmation_is_one_undo(tmp_path: Path) -> None:
    app = _app(tmp_path, "textual-dark", component=True)
    async with app.run_test(size=(140, 50)) as pilot:
        await open_editor(app, pilot, tmp_path)
        await wait_for(lambda: bool(app.screen.query(ResettableThemeEditor)), pilot=pilot)
        editor = app.screen.query_one(ResettableThemeEditor)
        await press_button(pilot, editor._color_buttons["primary"])
        await wait_for_picker(pilot, ColorPicker)
        picker = app.screen.query_one(ColorPicker)
        plane = picker.query_one("#color-hue", ColorPlane)
        with patch.object(app, "refresh_css", autospec=True, side_effect=app.refresh_css) as refresh:
            for _ in range(60):
                plane.action_step(1, 0, 1)
            await wait_for(lambda: app.current_theme.primary == plane.value.expression, pilot=pilot)
            assert 1 <= refresh.call_count <= 2
        assert editor.document.undo_stack == []
        await pilot.resize_terminal(60, 24)
        # The App dispatches a resize from a debounce timer, which Pilot's
        # settle barrier does not wait for.
        await wait_for(
            lambda: picker.has_class("compact") and app.screen.can_view_entire(app.screen.query_one("#picker-confirm")),
            pilot=pilot,
            description="compact picker with its confirm button in view",
        )
        await press_button(pilot, "#picker-confirm")
        await wait_for(lambda: not isinstance(app.screen, _PickerModal), pilot=pilot)
        assert len(editor.document.undo_stack) == 1
