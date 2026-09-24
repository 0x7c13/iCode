# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Mounted sidebar checks for pixel animation timing and theme compositing."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.theme import Theme
from textual.widgets import Static

from chrys.app.features.buddy.model import Rarity, Species
from chrys.app.features.buddy.portrait import PORTRAIT_HEIGHT, SHINY_FPS, render_portrait
from chrys.app.tui.util.visibility import is_widget_shown, is_widget_shown_on_active_screen
from chrys.app.tui.widgets.sidebar import buddy as buddy_module
from chrys.app.tui.widgets.sidebar.buddy import BuddyPanel
from tests.support.buddies import a_buddy, turns_to_finish
from tests.support.pilot_barrier import screen_is_settled
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from textual.visual import VisualType

    from chrys.app.features.buddy.model import Buddy


@pytest.mark.parametrize("background", ["ansi_default", "ansi_blue"])
async def test_ansi_buddy_keeps_terminal_background_and_cannot_be_selected(
    monkeypatch: pytest.MonkeyPatch, background: str
) -> None:
    buddy = a_buddy(rarity=Rarity.SSR, species=Species.RABBIT, shiny=True, name="Lucas")
    monkeypatch.setattr(buddy_module, "current_buddy", lambda: buddy)

    class BuddyApp(App):
        CSS = "Screen { background: $ansi-background; }"

        def compose(self) -> ComposeResult:
            yield BuddyPanel()

    app = BuddyApp()
    app.register_theme(
        Theme(name="buddy-ansi", primary="ansi_magenta", ansi=True, variables={"ansi-background": background})
    )
    app.theme = "buddy-ansi"
    async with app.run_test(size=(28, 30)) as pilot:
        panel = app.query_one(BuddyPanel)
        sprite = panel.query_one("#buddy-sprite", Static)
        await wait_for(lambda: is_widget_shown(sprite), pilot=pilot)
        panel._render_sprite()
        content = sprite.content
        assert isinstance(content, Text)
        body = content.split("\n")[1:-2]
        assert any("▄" in row.plain for row in body)
        console = Console(force_terminal=False, _environ={})
        for row in body:
            for x, character in enumerate(row.plain):
                if character in {" ", "▄"}:
                    assert row.get_style_at_offset(console, x).bgcolor is None
        app.copy_to_clipboard("previous clipboard")
        await pilot.mouse_down(sprite, offset=(3, 1))
        await pilot.hover(sprite, offset=(20, 8))
        await pilot.mouse_up(sprite, offset=(20, 8))
        assert not app.screen.selections
        assert not app.screen.get_selected_text()
        await pilot.press("super+c")
        assert app.clipboard == "previous clipboard"
        with patch.object(panel, "pet", autospec=True) as pet:
            assert await pilot.click(sprite, offset=(10, 5))
            pet.assert_called_once()


@pytest.mark.parametrize("tick_method", ["_on_tick", "_animate_shiny"])
async def test_buddy_timer_skips_covered_transcript_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tick_method: str
) -> None:
    """Both timer lanes must avoid portrait work behind a translucent modal."""
    from chrys.app.tui.screens.dialogs.tool_view import ToolDetailModal

    buddy = a_buddy(rarity=Rarity.SSR, species=Species.RABBIT, shiny=True, name="Lucas")
    monkeypatch.setattr(buddy_module, "current_buddy", lambda: buddy)
    capture_next_tick = False
    calls: list[tuple[int, int, int, int]] = []
    tick = BuddyPanel._on_tick if tick_method == "_on_tick" else BuddyPanel._animate_shiny

    def capture_tick(panel: BuddyPanel) -> None:
        nonlocal capture_next_tick
        if not capture_next_tick:
            tick(panel)
            return
        capture_next_tick = False
        sprite = panel.query_one("#buddy-sprite", Static)
        screen = panel.screen
        compositor = screen._compositor
        with (
            patch.object(buddy_module, "render_portrait", autospec=True, side_effect=render_portrait) as render,
            patch.object(sprite, "update", autospec=True, side_effect=sprite.update) as update,
            patch.object(compositor, "_arrange_root", autospec=True, side_effect=compositor._arrange_root) as arrange,
            patch.object(screen, "_refresh_layout", autospec=True, side_effect=screen._refresh_layout) as layout,
            patch.object(compositor, "_full_map_invalidated", True),
        ):
            tick(panel)
            calls.append((render.call_count, update.call_count, arrange.call_count, layout.call_count))

    class BuddyApp(App):
        CSS = "#transcript { width: 1fr; } BuddyPanel { width: 42; }"

        def compose(self) -> ComposeResult:
            with Horizontal():
                with VerticalScroll(id="transcript"):
                    for index in range(200):
                        yield Static(Text(f"Transcript message {index}"))
                yield BuddyPanel()

    with patch.object(BuddyPanel, tick_method, autospec=True, side_effect=capture_tick):
        async with BuddyApp().run_test(size=(100, 30)) as pilot:
            panel = pilot.app.query_one(BuddyPanel)
            sprite = panel.query_one("#buddy-sprite", Static)
            await wait_for(lambda: is_widget_shown(sprite), pilot=pilot, description="buddy portrait is composited")
            underlay = pilot.app.screen
            await pilot.app.push_screen(ToolDetailModal(title="Details", input_widgets=[], output_widgets=[]))
            assert pilot.app.screen is not underlay
            assert is_widget_shown(sprite), "the covered screen retains its own visible cut"
            capture_next_tick = True
            await wait_for(lambda: bool(calls), pilot=pilot, description=f"covered buddy {tick_method} callback")
            assert calls == [(0, 0, 0, 0)]

            await pilot.app.pop_screen()
            await wait_for(lambda: is_widget_shown(sprite), pilot=pilot, description="buddy portrait is restored")
            calls.clear()
            capture_next_tick = True
            await wait_for(lambda: bool(calls), pilot=pilot, description=f"uncovered buddy {tick_method} callback")
            assert calls == [(1, 1, 0, 0)]


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("shiny", [False, True])
@pytest.mark.parametrize("panel_width", [16, 19, 20, 24])
@pytest.mark.parametrize("species", [Species.DUCK, Species.RABBIT])
async def test_sidebar_uses_pixel_petting_frames_and_current_background(
    monkeypatch: pytest.MonkeyPatch, theme: str, shiny: bool, panel_width: int, species: Species
) -> None:
    buddy = a_buddy(rarity=Rarity.N, species=species, shiny=shiny, name="Ducky")
    monkeypatch.setattr(buddy_module, "current_buddy", lambda: buddy)
    now = 10.0
    monkeypatch.setattr(buddy_module, "monotonic", lambda: now)

    class BuddyApp(App):
        def compose(self) -> ComposeResult:
            yield BuddyPanel()

    app = BuddyApp()
    app.theme = theme
    async with app.run_test(size=(panel_width + 2, 30)) as pilot:
        panel = app.query_one(BuddyPanel)
        panel._timer.stop()
        sprite = panel.query_one("#buddy-sprite", Static)
        await wait_for(lambda: sprite.content_size.width == panel_width, pilot=pilot)
        assert sprite.content_size.height == PORTRAIT_HEIGHT
        panel.is_petting = True
        panel._pet_at = now
        console = Console(force_terminal=False, _environ={})

        for phase in range(3):
            now = 10.0 + phase * 0.1 + 0.01
            panel._render_sprite()
            actual = sprite.content
            assert isinstance(actual, Text)
            background = sprite.background_colors[1].rgb
            expected = Text("\n").join(
                render_portrait(
                    buddy.appearance,
                    "Ducky",
                    3 + phase,
                    width=panel_width,
                    effect_tick=int(now * SHINY_FPS),
                    bg_rgb=background,
                )
            )
            assert actual.plain == expected.plain
            assert all(line.cell_len == panel_width for line in actual.split("\n"))
            assert actual.plain.split("\n")[-1].strip() == ("Ducky [N] ✧" if shiny else "Ducky [N]")
            assert [actual.get_style_at_offset(console, i) for i in range(len(actual))] == [
                expected.get_style_at_offset(console, i) for i in range(len(expected))
            ]
            if species == Species.RABBIT:
                # Preserve the rabbit's dark pupils, including the first
                # petting pose's hop and both halves of a terminal character.
                scale_width = min(panel_width, 20)
                scale_height = round(16 * scale_width / 20)
                for pupil_y in (7, 8) if phase == 0 else (8, 9):
                    y = int((pupil_y + 0.5) * scale_height / 16) + (16 - scale_height) // 2
                    row = actual.split("\n")[1 + y // 2]
                    for pupil_x in (7, 12):
                        x = int((pupil_x + 0.5) * scale_width / 20) + (panel_width - scale_width) // 2
                        style = row.get_style_at_offset(console, x)
                        color = style.bgcolor if y % 2 else style.color
                        assert color is not None and color.triplet == (30, 30, 30)


@pytest.mark.asyncio
async def test_short_sidebar_scrolls_details_and_suspends_offscreen_portrait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Narrow and muted, so the details wrap into more rows than the portrait is tall.
    buddy = a_buddy(
        rarity=Rarity.SSR, species=Species.RABBIT, shiny=True, name="Lucas", muted=True, turns=turns_to_finish(1)
    )
    monkeypatch.setattr(buddy_module, "current_buddy", lambda: buddy)

    class BuddyApp(App):
        def compose(self) -> ComposeResult:
            yield BuddyPanel()

    # Short enough that scrolling to the last detail takes the whole portrait off the screen.
    async with BuddyApp().run_test(size=(24, 15)) as pilot:
        panel = pilot.app.query_one(BuddyPanel)
        sprite = panel.query_one("#buddy-sprite", Static)
        level = panel.query_one("#buddy-level", Static)
        status = panel.query_one("#buddy-status", Static)
        compositor = pilot.app.screen._compositor
        await wait_for(lambda: is_widget_shown(sprite), pilot=pilot)
        rows = [strip.text for strip in compositor.render_strips()]

        def row_text(y: int) -> str:
            # The panel's scrollbar shares these rows; read only the columns the details occupy.
            return rows[y][level.region.x : level.region.right].strip()

        assert row_text(level.region.y) == "Level 100"
        assert row_text(status.region.y) == "Click to pet!"
        assert status.region.y == level.region.bottom + 1
        assert not row_text(level.region.bottom)
        assert level.styles.text_align == status.styles.text_align == "center"

        panel.scroll_end(animate=False)
        await wait_for(
            lambda: "✨ Shiny!" in "\n".join(strip.text for strip in compositor.render_strips()),
            pilot=pilot,
            description="last buddy details visible after scrolling",
        )
        assert panel.scroll_y > 0
        assert is_widget_shown(panel)
        assert not is_widget_shown(sprite)

        # Model the stale-map window after hit testing. Neither idle nor shiny
        # ticks may rebuild layout or render an offscreen body/badge.
        _ = compositor.full_map
        with (
            patch.object(buddy_module, "render_portrait", autospec=True) as render,
            patch.object(compositor, "_arrange_root", autospec=True) as arrange,
            patch.object(compositor, "_full_map_invalidated", True),
        ):
            compositor._visible_widgets = None
            compositor._visible_map = None
            panel._on_tick()
            panel._animate_shiny()
            render.assert_not_called()
            arrange.assert_not_called()

        panel.scroll_home(animate=False)
        await wait_for(lambda: is_widget_shown(sprite), pilot=pilot)
        with patch.object(buddy_module, "render_portrait", autospec=True, side_effect=render_portrait) as render:
            panel._animate_shiny()
            render.assert_called_once()


@pytest.mark.asyncio
async def test_visible_shiny_timer_does_not_arrange_populated_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buddy = a_buddy(rarity=Rarity.SSR, species=Species.RABBIT, shiny=True, name="Lucas")
    monkeypatch.setattr(buddy_module, "current_buddy", lambda: buddy)
    capture_next_tick = False
    calls: list[tuple[int, int, int]] = []
    animate_shiny = BuddyPanel._animate_shiny

    def capture_tick(panel: BuddyPanel) -> None:
        nonlocal capture_next_tick
        if not capture_next_tick:
            animate_shiny(panel)
            return
        capture_next_tick = False
        screen = panel.screen
        compositor = screen._compositor
        with (
            patch.object(buddy_module, "render_portrait", autospec=True, side_effect=render_portrait) as render,
            patch.object(compositor, "_arrange_root", autospec=True, side_effect=compositor._arrange_root) as arrange,
            patch.object(screen, "_refresh_layout", autospec=True, side_effect=screen._refresh_layout) as layout,
            patch.object(compositor, "_full_map_invalidated", True),
        ):
            # Scope the dirty-map probe to the real timer callback. The normal
            # compositor may independently rebuild its map when painting later.
            animate_shiny(panel)
            calls.append((arrange.call_count, layout.call_count, render.call_count))

    class BuddyApp(App):
        CSS = """
        #transcript { width: 1fr; }
        BuddyPanel { width: 42; }
        """

        def compose(self) -> ComposeResult:
            with Horizontal():
                with VerticalScroll(id="transcript"):
                    for index in range(200):
                        yield Static(Text(f"Transcript message {index}"))
                yield BuddyPanel()

    with patch.object(BuddyPanel, "_animate_shiny", autospec=True, side_effect=capture_tick):
        async with BuddyApp().run_test(size=(100, 30)) as pilot:
            panel = pilot.app.query_one(BuddyPanel)
            sprite = panel.query_one("#buddy-sprite", Static)
            await wait_for(lambda: is_widget_shown(sprite), pilot=pilot)
            assert len(pilot.app.query("#transcript Static")) == 200
            _ = pilot.app.screen._compositor.full_map
            capture_next_tick = True
            await wait_for(lambda: bool(calls), pilot=pilot, description="visible shiny timer repaint")
            assert calls == [(0, 0, 1)]
            content = sprite.content
            assert isinstance(content, Text)
            assert "Lucas [SSR]" in content.plain


@pytest.mark.asyncio
async def test_looking_again_at_the_save_file_redraws_only_what_changed_and_lays_out_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = {"turns": 0}
    looks = 0

    def current_buddy() -> Buddy:
        nonlocal looks
        looks += 1
        return a_buddy(name="Lucas", **saved)  # a new object every time, as reading the file gives

    monkeypatch.setattr(buddy_module, "current_buddy", current_buddy)
    monkeypatch.setattr(buddy_module, "_RELOAD_SECONDS", 3600.0)  # the test looks again by hand

    class BuddyApp(App):
        CSS = """
        #transcript { width: 1fr; }
        BuddyPanel { width: 42; }
        """

        def compose(self) -> ComposeResult:
            with Horizontal():
                with VerticalScroll(id="transcript"):
                    for index in range(200):
                        yield Static(Text(f"Transcript message {index}"))
                yield BuddyPanel()

    async with BuddyApp().run_test(size=(100, 30)) as pilot:
        panel = pilot.app.query_one(BuddyPanel)
        screen = pilot.app.screen
        sprite = panel.query_one("#buddy-sprite", Static)
        await wait_for(
            lambda: is_widget_shown(sprite) and screen_is_settled(pilot.app, screen),
            pilot=pilot,
            description="the panel is drawn and mounting has no layout left to do",
        )
        labels = {
            selector: panel.query_one(selector, Static)
            for selector in (".buddy-empty", "#buddy-level", "#buddy-info", "#buddy-status")
        }
        updated: list[str] = []
        static_update = Static.update

        def record_update(widget: Static, content: VisualType = "", *, layout: bool = True) -> None:
            updated.extend(selector for selector, label in labels.items() if label is widget)
            static_update(widget, content, layout=layout)

        with (
            patch.object(Static, "update", autospec=True, side_effect=record_update),
            patch.object(screen, "_refresh_layout", autospec=True, side_effect=screen._refresh_layout) as layout,
        ):
            before = looks
            for _ in range(5):
                panel._reload_if_shown()
            await pilot.pause()

            assert looks == before + 5
            assert updated == []
            assert layout.call_count == 0

            # The control: a finished turn changes the level line and nothing else.
            saved["turns"] = 1
            panel._reload_if_shown()
            assert updated == ["#buddy-level"]

        panel.display = False
        await wait_for(
            lambda: not is_widget_shown_on_active_screen(panel), pilot=pilot, description="the panel is out of view"
        )
        before = looks
        panel._reload_if_shown()
        assert looks == before


async def test_buddy_timers_run_only_for_a_composited_buddy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a buddy, or with the panel hidden, neither animation timer sleeps; the badge timer needs a shiny one."""
    saved: list[Buddy | None] = [None]
    monkeypatch.setattr(buddy_module, "current_buddy", lambda: saved[0])

    class BuddyApp(App):
        def compose(self) -> ComposeResult:
            yield BuddyPanel()

    async with BuddyApp().run_test(size=(28, 30)) as pilot:
        panel = pilot.app.query_one(BuddyPanel)
        assert panel._timer is not None and panel._shiny_timer is not None
        await wait_for(lambda: panel._composited, pilot=pilot, description="the empty panel is composited")
        assert not panel._timer._active.is_set()
        assert not panel._shiny_timer._active.is_set()

        saved[0] = a_buddy(rarity=Rarity.N, species=Species.DUCK, shiny=False, name="Ducky")
        panel.reload()
        await wait_for(panel._timer._active.is_set, pilot=pilot, description="a hatched buddy animates")
        assert not panel._shiny_timer._active.is_set()

        saved[0] = a_buddy(rarity=Rarity.SSR, species=Species.RABBIT, shiny=True, name="Lucas")
        panel.reload()
        await wait_for(panel._shiny_timer._active.is_set, pilot=pilot, description="a shiny badge animates")

        panel.display = False
        await wait_for(lambda: not panel._timer._active.is_set(), pilot=pilot, description="Hide parks the idle tick")
        assert not panel._shiny_timer._active.is_set()
        panel.display = True
        await wait_for(panel._timer._active.is_set, pilot=pilot, description="Show restarts the idle tick")
        assert panel._shiny_timer._active.is_set()


async def test_hidden_buddy_finishes_its_petting_burst_before_parking(monkeypatch: pytest.MonkeyPatch) -> None:
    """The idle tick ends a petting burst, so hiding the panel mid-burst must not park it yet."""
    buddy = a_buddy(rarity=Rarity.N, species=Species.DUCK, shiny=False, name="Ducky", muted=True)
    monkeypatch.setattr(buddy_module, "current_buddy", lambda: buddy)
    monkeypatch.setattr(buddy_module, "record_pet", lambda: None)
    now = 10.0
    monkeypatch.setattr(buddy_module, "monotonic", lambda: now)

    class BuddyApp(App):
        def compose(self) -> ComposeResult:
            yield BuddyPanel()

    async with BuddyApp().run_test(size=(28, 30)) as pilot:
        panel = pilot.app.query_one(BuddyPanel)
        assert panel._timer is not None
        await wait_for(panel._timer._active.is_set, pilot=pilot, description="the buddy animates")
        panel.pet()
        assert panel.is_petting
        panel.display = False
        await wait_for(lambda: not panel._composited, pilot=pilot, description="the petted panel is hidden")
        assert panel._timer._active.is_set()

        now = 10.0 + buddy_module._PETTING_SECONDS
        panel._on_tick()
        assert not panel.is_petting
        assert not panel._timer._active.is_set()
