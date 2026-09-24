# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the mount-race-tolerant Select widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Region
from textual.widget import Widget
from textual.widgets import Select as TextualSelect
from textual.widgets._select import SelectCurrent, SelectOverlay

from chrys.app.tui.widgets import Select
from chrys.app.tui.widgets.select import LazySelect
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from textual.pilot import Pilot

_OPTIONS = [("Alpha", "a"), ("Beta", "b")]


class _SelectApp(App):
    def __init__(self, select_type: type[Select] = Select) -> None:
        super().__init__()
        self.select_type = select_type

    def compose(self) -> ComposeResult:
        yield self.select_type(_OPTIONS, allow_blank=True, value="a")


async def _open(pilot: Pilot, select: Select) -> SelectOverlay:
    """Open the dropdown and hand back its overlay once it has rows to read and click."""
    overlay = select.query_one(SelectOverlay)
    await click_when_settled(pilot, select.query_one(SelectCurrent))
    await wait_for(
        lambda: overlay.display and overlay.region.height > 0, pilot=pilot, description="select overlay laid out"
    )
    return overlay


async def test_closed_lazy_overlay_ignores_hit_testing_before_relayout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A queued pointer lookup may still use the open overlay's compositor map."""
    app = _SelectApp(LazySelect)
    async with app.run_test() as pilot:
        select = app.query_one(Select)
        overlay = await _open(pilot, select)
        region = overlay.region
        original_region = Widget.scrollable_content_region.fget
        assert original_region is not None
        geometry_reads: list[bool] = []

        def read_region(widget: SelectOverlay) -> Region:
            geometry_reads.append(widget.display)
            return original_region(widget)

        monkeypatch.setattr(SelectOverlay, "scrollable_content_region", property(read_region))
        overlay.refresh()
        app.screen._compositor.get_style_at(region.x + 1, region.y + 1)
        assert geometry_reads and all(geometry_reads)
        geometry_reads.clear()

        # Force a pending repaint, then dismiss without yielding to layout. The
        # compositor still considers this position part of the old overlay.
        overlay.refresh()
        select.expanded = False
        assert not overlay.display
        app.screen._compositor.get_style_at(region.x + 1, region.y + 1)
        assert not geometry_reads

        assert await _open(pilot, select) is overlay
        assert "Alpha" in "".join(overlay.render_line(y).text for y in range(overlay.content_size.height))
        await pilot.press("down", "enter")
        await wait_for(lambda: select.value == "b", pilot=pilot, description="reopened overlay selects Beta")


@pytest.mark.asyncio
@pytest.mark.parametrize("select_type", [Select, LazySelect])
async def test_select_populates_overlay_on_normal_mount(select_type: type[Select]) -> None:
    """The defensive overrides must not disturb the ordinary mount path."""
    app = _SelectApp(select_type)
    async with app.run_test() as pilot:
        await pilot.pause()
        select = app.query_one(Select)
        overlay = select.query_one(SelectOverlay)

        # allow_blank adds the blank prompt entry ahead of the options.
        assert overlay.option_count == len(_OPTIONS) + 1
        assert select.value == "a"
        assert isinstance(select, TextualSelect)


@pytest.mark.asyncio
@pytest.mark.parametrize("select_type", [Select, LazySelect])
async def test_select_survives_children_missing_at_mount_time(select_type: type[Select]) -> None:
    """Prune racing a fresh mount must not crash the app.

    ``Widget.mount()`` silently skips mounting children while the widget is
    being pruned, but the already-queued ``Mount`` event still dispatches, so
    ``_on_mount`` runs against a Select with no ``SelectOverlay``. Textual's
    base class crashes the whole app with ``NoMatches`` there; the chrys
    subclass defers instead. Re-running ``_on_mount``'s body after stripping
    the children reproduces that state deterministically.
    """
    app = _SelectApp(select_type)
    async with app.run_test() as pilot:
        await pilot.pause()
        select = app.query_one(Select)
        await select.remove_children()

        # Base Select raises NoMatches from _setup_options_renderables here
        # (_watch_value is guarded upstream); ours must survive both.
        select._setup_options_renderables()
        select._watch_value("b")
        await pilot.pause()

        assert select._value == "b"


@pytest.mark.parametrize("allow_blank", [False, True])
@pytest.mark.parametrize("select_type", [Select, LazySelect])
async def test_divider_does_not_become_a_selectable_value(allow_blank: bool, select_type: type[Select]) -> None:
    class GroupedSelectApp(App):
        def compose(self) -> ComposeResult:
            yield select_type(_OPTIONS, separators_before=("b",), value="a", allow_blank=allow_blank)

    app = GroupedSelectApp()
    async with app.run_test() as pilot:
        select = app.query_one(Select)
        overlay = await _open(pilot, select)
        assert overlay.option_count == 2 + allow_blank
        assert overlay.get_option_at_index(int(allow_blank))._divider
        # Read the painted separator row; clicking it must not select the
        # preceding option or close the dropdown.
        line = next(y for y in range(overlay.size.height) if "───" in overlay.render_line(y).text)
        assert await pilot.click(overlay, offset=(3, overlay.gutter.top + line))
        assert select.expanded and select.value == "a"
        await pilot.press("down", "enter")
        await wait_for(
            lambda: select.value == "b" and not select.expanded, pilot=pilot, description="keyboard selects Beta"
        )
        await _open(pilot, select)
        await pilot.press("a", "l", "p", "enter")
        await wait_for(lambda: select.value == "a", pilot=pilot, description="type-ahead selects Alpha")
        # Replacing choices recomputes the divider without shifting the
        # option/value mapping, including the optional blank entry.
        select.set_options([("Beta", "b"), ("Alpha", "a")], separators_before=("a",))
        select.value = "a"
        await _open(pilot, select)
        assert overlay.highlighted == 1 + allow_blank
        await pilot.press("up", "enter")
        await wait_for(lambda: select.value == "b", pilot=pilot, description="divider is skipped going up")
        select.set_options(_OPTIONS)
        assert not any(option._divider for option in overlay.options)
