# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Chrys themes, their palette pins, and the stylesheet scans."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from rich.color import EIGHT_BIT_PALETTE, Color, ColorSystem
from textual.app import App, ComposeResult
from textual.color import Color as TextualColor
from textual.widgets import Button, Checkbox, Input, Static, Tab, TabbedContent, TextArea, Tree

from chrys.app.tui.app import ChrysApp
from chrys.app.tui.theme import (
    CHRYS_ANSI_THEME,
    CHRYS_LEGACY_THEME,
    CHRYS_THEME,
    TuiVariableDefaultsMixin,
)
from chrys.app.tui.themes.document import copy_theme
from chrys.app.tui.widgets.chat.messages import UserMessage
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.app.tui.widgets.sidebar.toc import ConversationToc, TocItem
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.service.state.store import JsonFileStateStore
from tests.support.paths import SRC_ROOT
from tests.support.tui_app_harness import ShutdownOnlyEngine, make_chrys_app

_CHRYS_CSS = SRC_ROOT / "chrys" / "app" / "tui" / "chrys.tcss"
_CHRYS_THEME = SRC_ROOT / "chrys" / "app" / "tui" / "theme.py"


class _TurnHighlightApp(TuiVariableDefaultsMixin, App[None]):
    def compose(self) -> ComposeResult:
        message = UserMessage("Selected turn")
        message.add_class("-highlighted")
        yield message
        yield ConversationToc()


class _SelectedMessageKindsApp(TuiVariableDefaultsMixin, App[None]):
    def compose(self) -> ComposeResult:
        for message_id, compressed, is_injection, selected in (
            ("turn", False, False, True),
            ("compressed-turn", True, False, True),
            ("note", False, True, True),
            ("unselected-note", False, True, False),
        ):
            message = UserMessage("Selected turn", compressed=compressed, is_injection=is_injection)
            message.id = message_id
            if selected:
                message.add_class("-highlighted")
            yield message


@pytest.mark.parametrize("theme_name", ["chrys", "chrys-ansi"])
async def test_selection_wins_over_compression_but_never_borders_an_injected_note(theme_name: str) -> None:
    app = _SelectedMessageKindsApp()
    app.register_theme(CHRYS_THEME)
    app.register_theme(CHRYS_ANSI_THEME)
    app.theme = theme_name
    async with app.run_test() as pilot:
        await pilot.pause()
        turn, compressed_turn, note, unselected_note = (
            app.query_one(f"#{message_id}", UserMessage)
            for message_id in ("turn", "compressed-turn", "note", "unselected-note")
        )

        assert compressed_turn.styles.border_left == turn.styles.border_left
        assert compressed_turn.styles.background == turn.styles.background
        # Selecting a note must not give it a border, which would also shift its text.
        assert note.styles.border_left == unselected_note.styles.border_left
        assert note.content_region.x == unselected_note.content_region.x
        assert note.styles.background == turn.styles.background


@pytest.mark.parametrize("theme_name", ["chrys-ansi", "ansi-dark", "ansi-light", "textual-light", "nord"])
async def test_selected_turn_colors_follow_theme_switches_and_ansi_mode(theme_name: str) -> None:
    app = _TurnHighlightApp()
    app.register_theme(CHRYS_THEME)
    app.register_theme(CHRYS_ANSI_THEME)
    source = app.get_theme(theme_name)
    assert source is not None
    app.register_theme(copy_theme(source, name="user-copy"))
    async with app.run_test() as pilot:
        toc = app.query_one(ConversationToc)
        toc.update_items([TocItem("turn-1", "Selected turn")])
        tree = toc.query_one(Tree)
        for name in ("chrys", theme_name, "user-copy", "chrys"):
            app.theme = name
            app.screen.set_focus(None)
            await pilot.pause()
            variables = app.get_css_variables()
            primary = TextualColor.parse(variables["primary"])
            border_opacity = float(variables["border-opacity"].removesuffix("%")) / 100
            theme = app.current_theme
            background = (
                TextualColor.parse("#303030" if theme.dark else "#DADADA") if theme.ansi else primary.with_alpha(0.18)
            )
            selected = app.query_one(UserMessage)
            assert selected.styles.background == background
            assert tree.get_component_styles("tree--cursor").background == background
            assert selected.styles.border_left[1] == (
                TextualColor.parse("#808080") if theme.ansi else primary.with_alpha(border_opacity)
            )
            tree.focus()
            await pilot.pause()
            cursor = tree.get_component_styles("tree--cursor")
            assert cursor.background == (
                TextualColor.parse("#444444" if theme.dark else "#C6C6C6") if theme.ansi else primary.with_alpha(0.28)
            )
            assert cursor.color == TextualColor.parse(variables["foreground"])


@pytest.mark.parametrize(
    ("theme_name", "background_index", "focus_index"),
    # Rich rounds these light RGB grays down one palette step; pin the emitted indices.
    [("chrys-ansi", 236, 238), ("ansi-dark", 236, 238), ("ansi-light", 252, 250)],
)
async def test_ansi_selected_turns_emit_256_color_grays(
    theme_name: str, background_index: int, focus_index: int
) -> None:
    app = _TurnHighlightApp()
    app.register_theme(CHRYS_ANSI_THEME)
    app.theme = theme_name
    app.console._color_system = ColorSystem.EIGHT_BIT
    async with app.run_test() as pilot:
        toc = app.query_one(ConversationToc)
        toc.update_items([TocItem("turn-1", "Selected turn")])
        await pilot.pause()
        tree = toc.query_one(Tree)
        tree.move_cursor(tree.root.children[0])
        selected = app.query_one(UserMessage)
        for focused, expected_index in ((False, background_index), (True, focus_index)):
            app.screen.set_focus(tree if focused else None)
            await pilot.pause()
            strips = app.screen._compositor.render_strips()
            # Inspect final composited text, through the terminal encoder, rather than CSS alone.
            for row, index in (
                (selected.content_region.y + 1, background_index),
                (tree.content_region.y + tree.cursor_line, expected_index),
            ):
                strip = strips[row]
                assert "Selected turn" in strip.text
                encoded = strip.render(app.console)
                assert f"48;5;{index}" in encoded
                assert "48;2;" not in encoded
                assert "38;2;" not in encoded
                text_segments = [segment for segment in strip if "Selected turn" in segment.text]
                assert text_segments
                for segment in text_segments:
                    assert segment.style is not None and segment.style.bgcolor is not None
                    downgraded = segment.style.bgcolor.downgrade(ColorSystem.EIGHT_BIT)
                    assert downgraded.number == index
                    red, green, blue = downgraded.get_truecolor()
                    assert red == green == blue
                    assert red >= 188 if not app.current_theme.dark else red <= 68


@pytest.mark.parametrize(
    ("theme", "expected_opacity"),
    [
        ("chrys", 0.8),
        ("chrys-legacy", 0.8),
        ("chrys-ansi", 0.8),
        ("textual-dark", 0.8),
        ("ansi-dark", 1.0),
        ("ansi-light", 1.0),
    ],
)
async def test_default_border_opacity_respects_theme_color_mode(
    theme: str,
    expected_opacity: float,
) -> None:
    """Native ANSI tokens remain opaque; RGB-backed themes honor 80% alpha."""

    class BorderApp(TuiVariableDefaultsMixin, App):
        CSS = "Static { border: round $primary $border-opacity; }"

        def __init__(self) -> None:
            super().__init__()
            self.register_theme(CHRYS_THEME)
            self.register_theme(CHRYS_LEGACY_THEME)
            self.register_theme(CHRYS_ANSI_THEME)
            self.theme = theme

        def compose(self) -> ComposeResult:
            yield Static("border", id="border")

    app = BorderApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        border_color = app.query_one("#border", Static).styles.border_top[1]
        assert border_color.a == expected_opacity
        variables = app.get_css_variables()
        expected_tool_group_title = "#949494" if theme == CHRYS_THEME.name else variables["warning"]
        assert variables["tui-tool-group-title"] == expected_tool_group_title


async def test_chrys_uses_gray_borders_without_muting_semantic_colors() -> None:
    class BorderApp(TuiVariableDefaultsMixin, App):
        CSS = """
        #primary { border: round $tui-border-primary $border-opacity; }
        #accent { border: round $tui-border-accent $border-opacity; }
        #warning { border: round $tui-border-warning $border-opacity; }
        #literal { border: round $tui-border-neutral-160 $border-opacity; }
        #user-message { border: round $tui-border-user-message $border-opacity; }
        #agent-message { border: round $tui-border-agent-message $border-opacity; }
        #selected-turn { border: round $tui-border-selected-turn $border-opacity; }
        #titled {
            border: round $tui-border-primary $border-opacity;
            border-title-color: $tui-border-title-primary;
            border-subtitle-color: $tui-border-title-primary;
        }
        """

        def __init__(self) -> None:
            super().__init__()
            self.register_theme(CHRYS_THEME)
            self.theme = CHRYS_THEME.name

        def compose(self) -> ComposeResult:
            yield Static("primary", id="primary")
            yield Static("accent", id="accent")
            yield Static("warning", id="warning")
            yield Static("literal", id="literal")
            yield Static("user-message", id="user-message")
            yield Static("agent-message", id="agent-message")
            yield Static("selected-turn", id="selected-turn")
            titled = Static("titled", id="titled")
            titled.border_title = "Title"
            titled.border_subtitle = "Subtitle"
            yield titled

    app = BorderApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        for widget_id in ("primary", "accent", "warning", "literal"):
            border_color = app.query_one(f"#{widget_id}", Static).styles.border_top[1]
            assert border_color.hex6 == "#5F5F5F"
            assert border_color.a == 0.8
        for widget_id, expected_color in (
            ("user-message", "#FF87D7"),
            ("agent-message", "#5FFF87"),
            ("selected-turn", "#AF87FF"),
        ):
            border_color = app.query_one(f"#{widget_id}", Static).styles.border_top[1]
            assert border_color.hex6 == expected_color
            assert border_color.a == 0.8
        title_color = app.query_one("#titled", Static).styles.border_title_color
        assert title_color.hex6 == "#AF87FF"
        assert title_color.a == 1.0
        subtitle_color = app.query_one("#titled", Static).styles.border_subtitle_color
        assert subtitle_color.hex6 == "#AF87FF"
        assert subtitle_color.a == 1.0
        variables = app.get_css_variables()
        assert variables["scrollbar"] == "#585858"
        assert variables["scrollbar-hover"] == "#767676"
        assert variables["scrollbar-active"] == "#808080"


def test_all_tui_border_declarations_use_theme_aware_colors() -> None:
    border_declaration = re.compile(
        r"^\s*border(?:-(?:top|right|bottom|left))?\s*:\s*(?P<value>[^;]+);",
        re.MULTILINE,
    )
    failures: list[str] = []
    tui_root = SRC_ROOT / "chrys" / "app" / "tui"
    for path in (*tui_root.rglob("*.py"), *tui_root.rglob("*.tcss")):
        source = path.read_text(encoding="utf-8")
        for match in border_declaration.finditer(source):
            value = match.group("value")
            if value != "none" and "$tui-border-" not in value:
                line = source.count("\n", 0, match.start()) + 1
                failures.append(f"{path.relative_to(tui_root)}:{line} {value}")

    assert failures == []


def test_screen_css_private_fork_version_matches_pinned_textual() -> None:
    """A Textual upgrade must explicitly re-audit the private CSS loader fork."""
    from textual import __version__ as textual_version

    from chrys.app.tui import app as tui_app

    assert textual_version == tui_app._TEXTUAL_LOAD_SCREEN_CSS_FORK_VERSION


def test_chrys_legacy_is_registered_and_selectable(tmp_path: Path) -> None:
    app = ChrysApp(
        EventBus(),
        ShutdownOnlyEngine(),  # type: ignore[arg-type]
        settings=Settings(theme=CHRYS_LEGACY_THEME.name),
        state_store=JsonFileStateStore(tmp_path),
    )

    assert CHRYS_LEGACY_THEME.name in app.available_themes
    assert app.theme == CHRYS_LEGACY_THEME.name
    assert "-chrys" not in app.classes


def test_chrys_theme_hex_colors_are_256_color_palette_entries() -> None:
    """Chrys/chrys-ansi theme hex literals should stay inside the 256-color palette.

    DiffView's per-row palettes are intentionally excluded: those colors
    are visually tuned truecolor values. test_file_edit_renderer covers the
    weaker terminal-compatibility invariant that dark diff backgrounds do
    not downgrade to xterm black.
    """
    palette = {f"#{red:02X}{green:02X}{blue:02X}" for red, green, blue in EIGHT_BIT_PALETTE}
    hex_color = re.compile(r"#[0-9A-Fa-f]{6}(?![0-9A-Fa-f])")

    failures: list[str] = []
    for path in (_CHRYS_THEME, _CHRYS_CSS):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in hex_color.finditer(line):
                color = match.group(0).upper()
                if color not in palette:
                    failures.append(f"{path.name}:{line_no} {color}")

    assert failures == []


def test_switching_from_ansi_theme_back_to_chrys_restores_truecolor(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """ANSI theme previews should not leave a global ANSI override behind."""

    persisted: list[str] = []
    monkeypatch.setattr("chrys.app.tui.app.persist_theme", persisted.append)

    app = ChrysApp(
        EventBus(),
        ShutdownOnlyEngine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys"),
        state_store=JsonFileStateStore(tmp_path),
    )

    assert app.ansi_color is None
    assert app.native_ansi_color is False

    app.theme = "ansi-dark"

    assert app.ansi_color is None
    assert app.native_ansi_color is True

    app.theme = "chrys"

    assert app.ansi_color is None
    assert app.native_ansi_color is False
    assert "-chrys" not in app.classes
    assert "-chrys-ansi" not in app.classes
    assert persisted == ["ansi-dark", "chrys"]


def test_chrys_ansi_uses_theme_native_ansi_flag(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The transparent Chrys theme should opt in via Theme.ansi, not App.ansi_color."""

    monkeypatch.setattr("chrys.app.tui.app.persist_theme", lambda _theme: None)

    app = ChrysApp(
        EventBus(),
        ShutdownOnlyEngine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys"),
        state_store=JsonFileStateStore(tmp_path),
    )

    app.theme = "chrys-ansi"

    variables = app.get_css_variables()
    assert app.current_theme.ansi is True
    assert app.ansi_color is None
    assert app.native_ansi_color is True
    assert variables["ansi-background"] == "ansi_black"
    assert variables["ansi-foreground"] == "ansi_white"
    assert variables["button-color-foreground"] == "#000000"
    assert variables["text-muted"] == "ansi_white"
    assert variables["markdown-h1-color"] == "#AF87FF"
    assert variables["markdown-h2-color"] == "#AF87FF"
    assert variables["markdown-h3-color"] == "#AF87FF"
    assert variables["markdown-h4-color"] == "#EEEEEE"
    assert variables["markdown-h5-color"] == "#EEEEEE"
    assert variables["markdown-h6-color"] == "#9E9E9E"
    assert variables["input-selection-background"] == "#5F5F87"
    assert variables["screen-selection-background"] == "#5F5F87"
    assert "-chrys" not in app.classes
    assert "-chrys-ansi" not in app.classes


async def test_chrys_ansi_theme_defines_textual_ansi_css_variables() -> None:
    """Textual's built-in ANSI-mode CSS needs ansi-background/foreground."""

    class ThemeApp(App):
        def __init__(self) -> None:
            super().__init__()
            self.register_theme(CHRYS_ANSI_THEME)
            self.theme = "chrys-ansi"

        def compose(self) -> ComposeResult:
            yield Static("ok")

    app = ThemeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.current_theme.name == "chrys-ansi"


async def test_chrys_ansi_rendered_colors_follow_chrys_without_ansi_fallbacks() -> None:
    """chrys-ansi should stay close to Chrys colors, not Textual ANSI fallbacks."""

    class ThemeParityApp(App):
        CSS_PATH = str(_CHRYS_CSS)

        def __init__(self, theme_name: str) -> None:
            super().__init__()
            self.register_theme(CHRYS_THEME)
            self.register_theme(CHRYS_ANSI_THEME)
            self.theme = theme_name

        def compose(self) -> ComposeResult:
            yield Button("focus target", id="focus-target")
            yield Input("selected text", id="selection-input")
            yield VirtualizedMarkdown(
                "# H1\n\n## H2\n\n### H3\n\n#### H4\n\n##### H5\n\n###### H6",
                id="headings",
            )

    def color_rgb(color: Color | None) -> tuple[int, int, int]:
        assert color is not None
        triplet = color.triplet
        assert triplet is not None
        return triplet.red, triplet.green, triplet.blue

    async def collect(theme_name: str) -> dict[str, tuple[int, int, int]]:
        app = ThemeParityApp(theme_name)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#focus-target", Button).focus()
            await pilot.pause()

            markdown = app.query_one("#headings", VirtualizedMarkdown)
            values = {
                "screen-selection": color_rgb(app.screen.get_component_rich_style("screen--selection").bgcolor),
                "input-selection": color_rgb(
                    app.query_one("#selection-input", Input).get_visual_style("input--selection").rich_style.bgcolor
                ),
            }
            for level in range(1, 7):
                values[f"h{level}"] = color_rgb(
                    markdown.get_visual_style(f"virtualized-markdown--h{level}").rich_style.color
                )
            return values

    ansi_values = await collect("chrys-ansi")
    chrys_values = await collect("chrys")

    for key in ("h1", "h2", "h3", "h4", "h5"):
        assert ansi_values[key] == chrys_values[key]

    # These colors are alpha-derived in truecolor chrys but must be flattened
    # to nearby xterm-256 entries in transparent chrys-ansi.
    assert ansi_values["h6"] == (158, 158, 158)
    assert ansi_values["input-selection"] == (95, 95, 135)
    assert ansi_values["screen-selection"] == (95, 95, 135)
    for key in ("h6", "input-selection", "screen-selection"):
        assert max(abs(ansi - chrys) for ansi, chrys in zip(ansi_values[key], chrys_values[key], strict=True)) <= 24


async def test_chrys_app_runs_with_chrys_ansi_css(tmp_path) -> None:
    """The full Chrys stylesheet should parse with chrys-ansi active."""
    from chrys.app.tui.screens.main.screen import MainScreen
    from chrys.app.tui.widgets.chrome.input_bar import InputBar
    from chrys.app.tui.widgets.chrome.status_bar import StatusBar
    from chrys.app.tui.widgets.sidebar.toc import ConversationToc

    app = make_chrys_app(tmp_path, settings=Settings(theme="chrys-ansi"), gc_freeze_enabled=None)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.current_theme.name == "chrys-ansi"
        assert app.native_ansi_color is True
        screen = next(screen for screen in app.screen_stack if isinstance(screen, MainScreen))
        chat_input = screen.query_one(InputBar).query_one(TextArea)
        placeholder_style = chat_input.get_visual_style("text-area--placeholder").rich_style
        assert placeholder_style.color is not None
        assert placeholder_style.color.number == 8
        assert chat_input.rich_style.bgcolor is not None
        assert chat_input.rich_style.bgcolor.triplet is not None
        chat_input_background = chat_input.rich_style.bgcolor.triplet
        assert (chat_input_background.red, chat_input_background.green, chat_input_background.blue) == (48, 48, 48)
        selection_style = screen.get_component_rich_style("screen--selection")
        assert selection_style.bgcolor is not None
        assert selection_style.bgcolor.number is None
        assert selection_style.bgcolor.triplet is not None
        triplet = selection_style.bgcolor.triplet
        assert (triplet.red, triplet.green, triplet.blue) == (95, 95, 135)
        toc_empty_style = screen.query_one(ConversationToc).query_one(".toc-empty").rich_style
        assert toc_empty_style.color is not None
        assert toc_empty_style.color.number == 8
        input_bar = screen.query_one(InputBar)
        status_bar = screen.query_one(StatusBar)
        status_bar.set_profile("Code Agent")
        await pilot.pause()
        agent_label_style = status_bar.query_one("#agent-label", Static).rich_style
        assert agent_label_style.bgcolor is not None
        assert agent_label_style.bgcolor.number == 0
        assert agent_label_style.color is not None
        assert agent_label_style.color.number == 7
        profile_tag_style = status_bar.query_one("#profile-tag", Static).rich_style
        assert profile_tag_style.bgcolor is not None
        assert profile_tag_style.bgcolor.triplet is not None
        profile_tag_triplet = profile_tag_style.bgcolor.triplet
        assert (profile_tag_triplet.red, profile_tag_triplet.green, profile_tag_triplet.blue) == (76, 40, 64)
        send_button = input_bar.query_one("#send-btn", Button)
        assert send_button.disabled is True
        assert send_button.styles.text_opacity == pytest.approx(0.9)
        new_button = input_bar.query_one("#new-btn", Button)
        assert new_button.rich_style.bgcolor is not None
        assert new_button.rich_style.bgcolor == Color.parse(app.theme_variables["tui-button-primary-background"])


@pytest.mark.parametrize("target_theme", ["textual-dark", "ansi-dark", "ansi-light"])
async def test_theme_switch_to_non_chrys_does_not_crash(tmp_path, target_theme: str) -> None:
    """Switching to reachable non-chrys themes must keep chrys.tcss parseable."""
    from chrys.app.tui.screens.main.screen import MainScreen

    app = make_chrys_app(tmp_path, settings=Settings(theme="chrys"), gc_freeze_enabled=None)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.current_theme.name == "chrys"
        # Sanity: the screen is mounted and CSS parsed for chrys.
        screen = next(s for s in app.screen_stack if isinstance(s, MainScreen))
        assert screen is not None

        # Switch to a builtin non-chrys theme — CSS re-parse must succeed.
        app.theme = target_theme
        await pilot.pause()
        assert app.current_theme.name == target_theme

        # Switch back to chrys; round trip.
        app.theme = "chrys"
        await pilot.pause()
        assert app.current_theme.name == "chrys"


async def test_chrys_ansi_config_inner_borders_are_lighter(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_platform
) -> None:
    """Config screen inner borders should not disappear in chrys-ansi."""
    from chrys.app.tui.screens.agents.config import AgentsConfigScreen
    from chrys.app.tui.screens.models.screen import ModelConfigScreen
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.schema import ModelProfile

    config_dir = tmp_path / "config"
    model_dir = config_dir / "models"
    existing_model_files = set(model_dir.glob("*.yaml")) if model_dir.exists() else set()
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform(config_dir=config_dir))

    def fail_save_profile(_profile: ModelProfile) -> None:
        raise AssertionError("styling test must not write model profiles")

    monkeypatch.setattr("chrys.service.profiles.models.serializer.save_profile", fail_save_profile)

    app = make_chrys_app(tmp_path, settings=Settings(theme="chrys-ansi"), gc_freeze_enabled=None)
    agent_registry = AgentProfileRegistry()
    agent_registry.load_builtins()
    model_registry = ModelProfileRegistry()
    model_profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    model_registry.register(model_profile)

    try:
        async with app.run_test() as pilot:
            await pilot.pause()

            agents_screen = AgentsConfigScreen(agent_registry, current_profile="Code")
            await app.push_screen(agents_screen)
            await pilot.pause()
            _, agent_sidebar_color = agents_screen.query_one("#ac-sidebar").styles.border_right
            assert agent_sidebar_color.a == pytest.approx(0.25)
            active_agent_tab = next(tab for tab in agents_screen.query(Tab) if tab.has_class("-active"))
            assert active_agent_tab.rich_style.bold is True
            unselected_agent_tab = next(tab for tab in agents_screen.query(Tab) if not tab.has_class("-active"))
            assert unselected_agent_tab.rich_style.dim is not True
            agent_unchecked_checkbox_style = agents_screen.query_one(
                "#bc-sub-agent-only", Checkbox
            ).get_component_rich_style("toggle--button")
            agent_checked_checkbox_style = agents_screen.query_one(
                "#bc-model-use-active", Checkbox
            ).get_component_rich_style("toggle--button")
            agents_screen.query_one("#ac-tabs", TabbedContent).active = "tools"
            await pilot.pause()
            agent_tool_checkbox_style = agents_screen.query_one(
                "#tc-cat-filesystem-read", Checkbox
            ).get_component_rich_style("toggle--button")
            assert agent_tool_checkbox_style == agent_checked_checkbox_style

            await app.pop_screen()
            await pilot.pause()

            models_screen = ModelConfigScreen(model_registry, global_default_profile_id=model_profile.id)
            await app.push_screen(models_screen)
            await pilot.pause()
            _, model_sidebar_color = models_screen.query_one("#mc-sidebar").styles.border_right
            option_section_border = models_screen.query_one("#mc-model-options").styles.border
            model_stream = models_screen.query_one("#mc-stream", Checkbox)
            model_stream.value = False
            await pilot.pause()
            model_unchecked_checkbox_style = model_stream.get_component_rich_style("toggle--button")
            model_stream.value = True
            await pilot.pause()
            model_checked_checkbox_style = model_stream.get_component_rich_style("toggle--button")
            assert model_sidebar_color.a == pytest.approx(0.25)
            assert option_section_border.top[1].a == pytest.approx(0.25)
            assert model_unchecked_checkbox_style == agent_unchecked_checkbox_style
            assert model_checked_checkbox_style == agent_checked_checkbox_style
    finally:
        if model_dir.exists():
            for path in model_dir.glob("*.yaml"):
                if path not in existing_model_files:
                    path.unlink()


async def test_chrys_active_config_tab_is_bold(tmp_path) -> None:
    """The base chrys theme should keep selected tabs visually emphasized."""
    from chrys.app.tui.screens.agents.config import AgentsConfigScreen
    from chrys.service.profiles.agents.registry import AgentProfileRegistry

    app = make_chrys_app(tmp_path, settings=Settings(theme="chrys"), gc_freeze_enabled=None)
    agent_registry = AgentProfileRegistry()
    agent_registry.load_builtins()

    async with app.run_test() as pilot:
        await pilot.pause()

        agents_screen = AgentsConfigScreen(agent_registry, current_profile="Code")
        await app.push_screen(agents_screen)
        await pilot.pause()

        active_agent_tab = next(tab for tab in agents_screen.query(Tab) if tab.has_class("-active"))
        assert active_agent_tab.rich_style.bold is True
