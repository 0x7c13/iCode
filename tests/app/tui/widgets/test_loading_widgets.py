# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for ChrysLoadingIndicator and WelcomeWidget: unattached render, live-locale resolution, and hidden-widget refresh skipping.

Also the component styles an animated widget keeps between its frames.
"""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import patch

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.geometry import Region
from textual.widgets import Static

from chrys.app.tui import i18n as tui_i18n
from chrys.app.tui.i18n import LocaleController, LocaleSwitchStatus
from chrys.app.tui.util.logo import CHAT_LOGO
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator, GatedAnimation
from chrys.app.tui.widgets.welcome import WelcomeWidget
from chrys.foundation.config.settings import Settings
from tests.support.tui_helpers import (
    WidgetApp,
)
from tests.support.waiting import wait_for


def test_loading_indicator_render_unattached_returns_text() -> None:
    indicator = ChrysLoadingIndicator()
    rendered_lines = indicator.render_lines(Region(0, 0, 20, 2))
    rendered = indicator.render()

    assert [strip.cell_length for strip in rendered_lines] == [20, 20]
    assert rendered.plain == "Loading..."


async def test_welcome_stays_plain_while_loading_resolves_live_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)

    class PaintApp(App):
        def __init__(self) -> None:
            self.locale_controller = controller
            super().__init__()

        def compose(self) -> ComposeResult:
            yield WelcomeWidget(CHAT_LOGO, title="Code", cwd="/workspace")
            yield ChrysLoadingIndicator()

    async with PaintApp().run_test(size=(60, 20)) as pilot:
        pilot.app.animation_level = "none"
        welcome = pilot.app.query_one(WelcomeWidget)
        loading = pilot.app.query_one(ChrysLoadingIndicator)

        english = "".join(segment.text for segment in pilot.app.console.render(welcome.render()))
        assert "Code" in english
        assert "Agent:" not in english
        assert loading.render().plain == "Loading..."

        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        chinese = "".join(segment.text for segment in pilot.app.console.render(welcome.render()))
        assert "Code" in chinese
        assert "智能体：" not in chinese  # noqa: RUF001
        assert loading.render().plain == "正在加载..."


async def test_loading_indicator_automatic_refresh_skips_hidden_widget() -> None:
    """The 16/s animation tick must not repaint (nor arrange) while hidden.

    Stock ``DOMNode.automatic_refresh`` gates on ``is_on_screen`` →
    ``Screen.find_widget``, which recomputes the compositor's full map — an
    O(all widgets) arrange per tick on large transcripts even when the
    indicator is invisible.
    """
    from textual.containers import Container

    async with WidgetApp(lambda: Container(ChrysLoadingIndicator())).run_test(size=(40, 10)) as pilot:
        indicator = pilot.app.query_one(ChrysLoadingIndicator)
        container = pilot.app.query_one(Container)

        refreshes: list[bool] = []
        original_refresh = indicator.refresh

        def counting_refresh(*args: object, **kwargs: object) -> None:
            refreshes.append(True)
            original_refresh(*args, **kwargs)

        indicator.refresh = counting_refresh  # type: ignore[method-assign]

        indicator.automatic_refresh()
        assert refreshes, "a visible indicator must keep animating"

        container.display = False
        # visible_widgets is documentedly one-frame stale: a single pause can
        # land before the recomposite that drops the hidden widget from the
        # cut, so wait the compositor state out instead of racing it.
        compositor = pilot.app.screen._compositor
        await wait_for(
            lambda: indicator not in compositor.visible_widgets,
            pilot=pilot,
            description="hidden indicator leaves the composited frame",
        )
        # Clear only now: the indicator's live 16/s auto_refresh timer may
        # legitimately repaint once inside the stale window above (that is the
        # accepted one-frame staleness, not the behavior under test). No await
        # between here and the assert, so the timer cannot interleave.
        refreshes.clear()
        indicator.automatic_refresh()
        assert not refreshes, "a hidden indicator must not repaint on its animation tick"


async def test_loading_indicator_ticks_only_while_composited_and_wanted() -> None:
    """The 16/s timer sleeps nothing for a spinner that is hidden or parked by its owner."""
    async with WidgetApp(lambda: Container(ChrysLoadingIndicator())).run_test(size=(40, 10)) as pilot:
        indicator = pilot.app.query_one(ChrysLoadingIndicator)
        container = pilot.app.query_one(Container)
        timer = indicator._auto_refresh_timer
        assert timer is not None
        await wait_for(timer._active.is_set, pilot=pilot, description="a composited indicator animates")

        container.display = False
        await wait_for(lambda: not timer._active.is_set(), pilot=pilot, description="Hide parks the animation timer")
        container.display = True
        await wait_for(timer._active.is_set, pilot=pilot, description="Show restarts the animation timer")

        # The owner's pause outlives a Hide/Show cycle; only its resume restarts the timer.
        indicator.pause_animation()
        assert not timer._active.is_set()
        container.display = False
        await wait_for(lambda: not indicator._composited, pilot=pilot, description="the parked indicator is hidden")
        container.display = True
        await wait_for(lambda: indicator._composited, pilot=pilot, description="the parked indicator is shown again")
        assert not timer._active.is_set()
        indicator.resume_animation()
        assert timer._active.is_set()


async def test_loading_indicator_mounted_hidden_parks_its_timer_until_shown() -> None:
    """No Show ever reaches a spinner inside a hidden container: its first tick parks the timer."""

    def hidden_container() -> Container:
        container = Container(ChrysLoadingIndicator())
        container.display = False
        return container

    async with WidgetApp(hidden_container).run_test(size=(40, 10)) as pilot:
        indicator = pilot.app.query_one(ChrysLoadingIndicator)
        container = pilot.app.query_one(Container)
        timer = indicator._auto_refresh_timer
        assert timer is not None
        await wait_for(lambda: not timer._active.is_set(), pilot=pilot, description="the hidden spinner parks itself")
        assert not indicator._composited

        container.display = True
        await wait_for(timer._active.is_set, pilot=pilot, description="Show starts the animation")


async def test_loading_indicator_resumed_spinner_animates_after_a_geometry_read_maps_it() -> None:
    """A full map rebuilt lazily between the owner's display flip and the reflow must not hide the spinner's Show.

    Any ``find_widget``/``region`` read in that window rebuilds the map with the
    spinner already in it. Compared with that map, the reflow would announce
    nothing and the timer would stay parked on a visible spinner; the reflow
    reuse patch compares with the map the previous reflow produced instead.
    """

    def owner_hidden_spinner() -> Container:
        indicator = ChrysLoadingIndicator()
        indicator.display = False
        return Container(indicator)

    async with WidgetApp(owner_hidden_spinner).run_test(size=(40, 10)) as pilot:
        indicator = pilot.app.query_one(ChrysLoadingIndicator)
        timer = indicator._auto_refresh_timer
        assert timer is not None
        await wait_for(lambda: not timer._active.is_set(), pilot=pilot, description="the hidden spinner parks itself")

        # The owner's sequence (ToolGroup, sub-agent card): flip display, then resume.
        indicator.display = True
        indicator.resume_animation()
        compositor = pilot.app.screen._compositor
        compositor._full_map_invalidated = True
        assert indicator in compositor.full_map, "the lazy rebuild must already map the spinner"

        await wait_for(timer._active.is_set, pilot=pilot, description="the shown spinner animates")
        assert indicator._composited


class LitRow(GatedAnimation, Static):
    COMPONENT_CLASSES: ClassVar[set[str]] = {"litrow--lit"}

    DEFAULT_CSS = """
    LitRow > .litrow--lit {
        color: $primary;
        background: red;
        text-style: bold;
    }
    """


class LitRowApp(App[None]):
    CSS = """
    .warm LitRow > .litrow--lit {
        color: #c04000;
    }
    LitRow.-cool > .litrow--lit {
        color: #0040c0;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="holder"):
            yield LitRow("row")


async def test_an_animated_row_keeps_its_component_style_between_frames_until_a_restyle_changes_it() -> None:
    async with LitRowApp().run_test() as pilot:
        row = pilot.app.query_one(LitRow)
        holder = pilot.app.query_one("#holder")

        def resolved() -> object:
            return row.get_component_rich_style("litrow--lit").color

        lit = row.text_style_of("litrow--lit")
        assert (lit.color, lit.bgcolor, lit.bold) == (resolved(), None, True)
        with patch.object(
            row, "get_component_rich_style", autospec=True, side_effect=row.get_component_rich_style
        ) as resolve:
            # Each frame repaints the row, which drops Textual's own cache of its component styles.
            row.refresh()
            assert row.text_style_of("litrow--lit") == lit
            assert resolve.call_count == 0

        seen = [lit.color]
        # An ancestor's class that changes the component's rules, not the row's own; then the row's own class.
        for restyle in (lambda: holder.add_class("warm"), lambda: row.add_class("-cool")):
            restyle()
            color = row.text_style_of("litrow--lit").color
            assert color == resolved()
            assert color not in seen
            seen.append(color)

        holder.remove_class("warm")
        row.remove_class("-cool")
        assert row.text_style_of("litrow--lit") == lit
        pilot.app.theme = "textual-light" if pilot.app.theme != "textual-light" else "textual-dark"
        await wait_for(lambda: resolved() not in seen, pilot=pilot, description="the new theme restyles the component")
        assert row.text_style_of("litrow--lit").color == resolved()
