# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the patch that makes a Textual reflow replay the arrangement of unchanged subtrees."""

from __future__ import annotations

import asyncio
import importlib
import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import textual._compositor as compositor_module
from textual._compositor import Compositor
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.scalar import Scalar
from textual.dom import DOMNode
from textual.geometry import Offset, Region, Size
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Static

from chrys.foundation.patches import textual_reflow_reuse
from chrys.foundation.patches.staged_members import StagedMembers, members_installed
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from textual._arrange import DockArrangeResult
    from textual._compositor import CompositorMap
    from textual.pilot import Pilot


class _Layout(App[None]):
    CSS = """
    #top { dock: top; height: 1; }
    #side { width: 20; }
    #side.wide { width: 30; }
    #overlay { layer: overlay; width: 10; height: 1; }
    Screen { layers: base overlay; }
    """

    def compose(self) -> ComposeResult:
        yield Static("top", id="top")
        with Horizontal():
            with VerticalScroll(id="log"):
                for index in range(40):
                    yield Static(f"line {index}", id=f"line-{index}")
            with Vertical(id="side"):
                for index in range(3):
                    yield Static(f"side {index}", id=f"side-{index}")
        yield Static("overlay", id="overlay")


def _layout_stamps(root: Widget) -> dict[DOMNode, object]:
    """The layout epoch of every node an arrangement of ``root`` can stamp (``None`` when never stamped)."""
    nodes: list[DOMNode] = list(root.ancestors)
    for node in root.walk_children(with_self=True):
        nodes.append(node)
        if isinstance(node, Widget):
            chrome = (node._vertical_scrollbar, node._horizontal_scrollbar, node._scrollbar_corner)
            nodes.extend(part for part in chrome if part is not None)
    return {node: vars(node).get("_layout_dirty_epoch") for node in nodes}


@pytest.fixture
def verify_reuse(monkeypatch: pytest.MonkeyPatch) -> list[Compositor]:
    """Check every reusing arrangement against a from-scratch one; a divergence fails the App.

    So does a check that leaves a node stamped, which would invalidate the records it checks
    and hide a change that was not stamped. Returns the compositor of each check made, so a
    test can prove its steps reused records.
    """
    textual_reflow_reuse.apply_runtime_patch()
    monkeypatch.setattr(compositor_module, "_VERIFY_REUSE", True)
    checks: list[Compositor] = []
    verify = Compositor._verify_reuse

    def record_check(
        self: Compositor,
        root: Widget,
        size: Size,
        map: CompositorMap,
        widgets: set[Widget],
        state: list[tuple[DOMNode, tuple[str, ...], dict[str, object]]],
    ) -> None:
        checks.append(self)
        stamps = _layout_stamps(root)
        try:
            verify(self, root, size, map, widgets, state)
        finally:
            stamped = [node for node, epoch in _layout_stamps(root).items() if stamps.get(node, epoch) != epoch]
            if stamped:
                raise AssertionError(f"the verification stamped {stamped[:5]}")

    monkeypatch.setattr(Compositor, "_verify_reuse", record_check)
    return checks


def test_fragments_match_installed_textual() -> None:
    for patch in textual_reflow_reuse._PATCHES:
        module = importlib.import_module(f"textual.{patch.module_file.removesuffix('.py').replace('/', '.')}")
        assert module.__file__ is not None
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert patch.old_fragment in source or patch.new_fragment in source, patch.description


def test_runtime_patch_installs_every_member() -> None:
    textual_reflow_reuse.apply_runtime_patch()

    for name, members in textual_reflow_reuse._RUNTIME_MEMBERS.items():
        assert members_installed(importlib.import_module(name), members, textual_reflow_reuse._RUNTIME_PATCH_MARKER)
    assert vars(compositor_module)["DOMNode"] is DOMNode


def test_a_drifted_module_installs_no_module(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Reusing records while widgets do not stamp their changes would paint stale placements."""
    installed: list[StagedMembers] = []
    monkeypatch.setattr(textual_reflow_reuse, "members_installed", lambda _module, _members, _marker: False)
    monkeypatch.setattr(StagedMembers, "install", lambda self, _marker: installed.append(self))
    drifted = [
        replace(patch, old_fragment="absent", new_fragment="also absent") if patch.module_file == "widget.py" else patch
        for patch in textual_reflow_reuse._PATCHES
    ]
    monkeypatch.setattr(textual_reflow_reuse, "_PATCHES", tuple(drifted))
    caplog.set_level(logging.WARNING, logger=textual_reflow_reuse.__name__)

    textual_reflow_reuse.apply_runtime_patch()

    # Widgets stage last, so every other module had staged cleanly.
    first_drifted = next(patch for patch in drifted if patch.module_file == "widget.py")
    assert installed == []
    assert [record.getMessage() for record in caplog.records] == [
        f"Skipping Textual reflow reuse runtime patch: fragment drifted: {first_drifted.description}"
    ]


async def test_a_reflow_replays_the_placements_of_unchanged_subtrees(verify_reuse: list[Compositor]) -> None:
    app = _Layout()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        compositor = app.screen._compositor
        before = dict(compositor._full_map)
        lines = list(app.query_one("#log").query(Static))
        side = app.query_one("#side-1", Static)

        side.update("side 1\nnow taller")
        await pilot.pause()

        after = compositor._full_map
        assert after[side] is not before[side]
        assert after[side].region.height == 2
        # Replayed, not rebuilt: the untouched log keeps the very geometry objects it had.
        assert all(after[line] is before[line] for line in lines)


async def test_a_widget_shown_after_a_reflow_reports_its_region_before_the_next_layout(
    verify_reuse: list[Compositor],
) -> None:
    """A reflow keeps a pending full-map rebuild pending, as upstream's does.

    A repaint of a widget outside the visible map marks the full map stale, and the next
    geometry lookup rebuilds it from the current DOM. A widget shown after the reflow then
    reports the region the next layout gives it, so code that sizes content right after showing
    a widget reads its real size, not an empty region.
    """
    app = _Layout()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = app.screen
        compositor = screen._compositor
        side = app.query_one("#side")
        shown = side.region
        # Driven synchronously, so no screen update runs between the steps.
        side.display = False
        screen._refresh_layout()
        compositor.update_widgets({side})
        screen._refresh_layout()
        side.display = True
        verify_reuse.clear()

        assert side.region == shown
        # The lookup rebuilt the full map from records, checked against a from-scratch arrangement.
        assert verify_reuse, "the lookup did not rebuild the full map"


class _Watched(ScrollView):
    """Ten lines whose watcher reads their scrollbar's region when it shows, as a watcher sizing content does."""

    def __init__(self) -> None:
        super().__init__(id="watched")
        self.watched: list[Region] = []

    def on_mount(self) -> None:
        self.virtual_size = Size(20, 10)

    def render_line(self, y: int) -> Strip:
        return Strip.blank(self.size.width)

    def watch_show_vertical_scrollbar(self, shown: bool) -> None:
        if shown:
            self.watched.append(self.vertical_scrollbar.region)


class _Watching(App[None]):
    CSS = "#watched { height: 12; }"

    def compose(self) -> ComposeResult:
        yield _Watched()
        yield Static("hidden", id="hidden")


async def test_a_refreshing_reactive_stamps_its_widget_before_its_watchers_look_up_geometry(
    verify_reuse: list[Compositor],
) -> None:
    """A watcher that reads geometry can rebuild the full map before the reactive's refresh runs.

    A reflow shrinks a scroll view, and its scrollbar shows while the layout refresh updates
    sizes. The watcher's lookup then rebuilds the pending full map from records. The view is
    stamped before its watchers run, so the rebuild arranges it afresh, as a from-scratch
    arrangement would, instead of replaying the record the reflow made without the scrollbar.
    (Upstream's ``_set_dirty`` rebuilds the full map earlier, before the scrollbar shows, so an
    upstream watcher reads the map from before the change.)
    """
    app = _Watching()
    async with app.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        screen = app.screen
        view = app.query_one(_Watched)
        hidden = app.query_one("#hidden")
        assert not view.show_vertical_scrollbar
        # Driven synchronously: a repaint outside the visible map leaves a full-map rebuild pending.
        hidden.display = False
        screen._refresh_layout()
        screen._compositor.update_widgets({hidden})
        view.styles.height = 5
        screen._refresh_layout()
        await pilot.pause()

        scrollbar = view.vertical_scrollbar.region
        assert scrollbar.height == 5
        assert view.watched == [scrollbar]


async def _mount(app: App[None], pilot: Pilot[None]) -> None:
    log = app.query_one("#log", VerticalScroll)
    await log.mount(Static("mounted"))
    log.scroll_end(animate=False)


async def _remove(app: App[None], pilot: Pilot[None]) -> None:
    await app.query_one("#log").children[20].remove()


async def _grow(app: App[None], pilot: Pilot[None]) -> None:
    app.query_one("#line-3", Static).update("line 3\n" * 4)


async def _style(app: App[None], pilot: Pilot[None]) -> None:
    app.query_one("#line-5").styles.margin = (1, 2)


async def _hide_side(app: App[None], pilot: Pilot[None]) -> None:
    side = app.query_one("#side")
    side.display = False
    await pilot.pause()
    side.display = True


async def _toggle_class(app: App[None], pilot: Pilot[None]) -> None:
    app.query_one("#side").add_class("wide")


async def _scroll(app: App[None], pilot: Pilot[None]) -> None:
    app.query_one("#log", VerticalScroll).scroll_to(y=12, animate=False)


async def _resize(app: App[None], pilot: Pilot[None]) -> None:
    await pilot.resize_terminal(100, 30)


async def _absolute_offset(app: App[None], pilot: Pilot[None]) -> None:
    app.query_one("#overlay").absolute_offset = Offset(5, 6)
    # Inside the log, which the next reflow would otherwise replay whole.
    app.query_one("#line-10").absolute_offset = Offset(3, 4)


async def _offset(app: App[None], pilot: Pilot[None]) -> None:
    app.query_one("#side-0").styles.offset = (2, 1)


async def _anchor(app: App[None], pilot: Pilot[None]) -> None:
    log = app.query_one("#log", VerticalScroll)
    log.anchor()
    await pilot.pause()
    await log.mount_all(Static(f"anchored {index}") for index in range(5))


async def _reanchor(app: App[None], pilot: Pilot[None]) -> None:
    log = app.query_one("#log", VerticalScroll)
    log.anchor()
    await pilot.pause()
    log.scroll_to(y=5, animate=False)
    await pilot.pause()
    # A reflow settles the scroll, so only the anchor state is left to change.
    app.query_one("#side-1", Static).update("side 1\nsettled")
    await pilot.pause()
    # As the chat panel re-anchors: only the next arrangement scrolls back to the end.
    assert log._anchor_released
    log._anchor_released = False


class _SlowMount(Static):
    def __init__(self, gate: asyncio.Event) -> None:
        super().__init__("slow")
        self.gate = gate

    async def on_mount(self) -> None:
        await self.gate.wait()


async def _slow_mount(app: App[None], pilot: Pilot[None]) -> None:
    """A widget whose mount completes after a reflow has recorded its parent without it."""
    gate = asyncio.Event()
    slow = _SlowMount(gate)
    mounting = app.query_one("#log").mount(slow, before=0)
    # The pilot drains handlers, so it would wait on the gate: wait for a reflow instead.
    compositor = app.screen._compositor
    side = app.query_one("#side-0", Static)
    height = compositor._full_map[side].region.height
    side.update("side 0" + "\nmore" * height)
    await wait_for(
        lambda: compositor._full_map[side].region.height == height + 1,
        description="a reflow while the widget mounts",
    )
    assert not slow.is_mounted
    gate.set()
    await mounting


async def _raw_rule(app: App[None], pilot: Pilot[None]) -> None:
    """Chrome widgets write style rules directly to skip Textual's layout escalation."""
    line = app.query_one("#line-10")
    line.styles.set_rule("visibility", "hidden" if line.visible else "visible")
    app.query_one("#line-12").styles.set_rule("height", None if line.visible else Scalar.from_number(3))


async def _reorder(app: App[None], pilot: Pilot[None]) -> None:
    log = app.query_one("#log", VerticalScroll)
    log.move_child(app.query_one("#line-30"), before=app.query_one("#line-2"))


_SCENARIOS: dict[str, Callable[[App[None], Pilot[None]], Awaitable[None]]] = {
    "mount": _mount,
    "remove": _remove,
    "grow": _grow,
    "style": _style,
    "hide-side": _hide_side,
    "toggle-class": _toggle_class,
    "scroll": _scroll,
    "resize": _resize,
    "absolute-offset": _absolute_offset,
    "offset": _offset,
    "anchor": _anchor,
    "reanchor": _reanchor,
    "slow-mount": _slow_mount,
    "reorder": _reorder,
    "raw-rule": _raw_rule,
}


@pytest.mark.parametrize("scenario", list(_SCENARIOS))
async def test_reusing_arrangements_match_from_scratch_ones(verify_reuse: list[Compositor], scenario: str) -> None:
    app = _Layout()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        side = app.query_one("#side-2", Static)
        for step in range(2):
            await _SCENARIOS[scenario](app, pilot)
            await pilot.pause()
            # A layout change elsewhere makes the next reflow replay whatever the step left unchanged.
            verify_reuse.clear()
            side.update("side 2" + "\nmore" * (step + 1))
            await pilot.pause()
            assert verify_reuse, "no reflow reused the arrangement after the step"
    # Leaving run_test re-raises a divergence, which takes the App down.


async def test_verification_fails_a_change_that_was_not_stamped(
    verify_reuse: list[Compositor], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The oracle the scenarios rely on goes red when a change skips the layout epoch."""
    monkeypatch.setattr(DOMNode, "_mark_layout_dirty", lambda self: None)

    with pytest.raises(AssertionError, match="compositor reuse diverged"):
        app = _Layout()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await _grow(app, pilot)
            await pilot.pause()


class _Stacked(App[None]):
    """Two widgets on the same cells on different layers, beneath two undeclaring containers."""

    CSS = """
    Screen.stacking { layers: below above; }
    Screen.flipped { layers: above below; }
    #spacer { height: 1; }
    #outer { height: 5; }
    #inner { height: 4; }
    #inner.stacking { layers: below above; }
    #high { layer: above; height: 3; }
    #low { layer: below; height: 3; }
    """

    def compose(self) -> ComposeResult:
        yield Static("spacer", id="spacer")
        with Vertical(id="outer"), Vertical(id="inner"):
            yield Static("high", id="high")
            yield Static("low", id="low")


def _reverse_screen_layers(app: App[None]) -> None:
    app.screen.styles.layers = ("above", "below")


def _flip_screen_class(app: App[None]) -> None:
    app.screen.add_class("flipped")


def _declare_default_on_screen(app: App[None]) -> None:
    """The screen's explicit ``default`` overrides the inner declaration; both resolve to ``("default",)`` above it."""
    app.screen.styles.layers = ("default",)


def _reverse_screen_layers_and_move(app: App[None]) -> None:
    """The inner container moves without resizing, so its subtree is a candidate for translation."""
    _reverse_screen_layers(app)
    app.query_one("#spacer").styles.height = 2


_STACKING_CHANGES: dict[str, tuple[str, Callable[[App[None]], None], str]] = {
    "rule-write": ("Screen", _reverse_screen_layers, "low"),
    "class-toggle": ("Screen", _flip_screen_class, "low"),
    "explicit-default-over-nested-declaration": ("#inner", _declare_default_on_screen, "low"),
    "moved-subtree": ("Screen", _reverse_screen_layers_and_move, "low"),
}


def _top_widget_id(app: App[None]) -> str | None:
    x, y = app.query_one("#low", Static).region.offset
    widget, _ = app.screen.get_widget_at(x, y)
    return widget.id


@pytest.mark.parametrize("change", list(_STACKING_CHANGES))
async def test_a_layers_change_above_a_replayed_subtree_restacks_it(
    verify_reuse: list[Compositor], change: str
) -> None:
    """``Widget.layers`` takes the outermost declaration, so a subtree's stacking follows its ancestors'."""
    declared_on, apply_change, expected_top = _STACKING_CHANGES[change]
    app = _Stacked()
    async with app.run_test(size=(40, 12)) as pilot:
        (app.screen if declared_on == "Screen" else app.query_one(declared_on)).add_class("stacking")
        await pilot.pause()
        assert _top_widget_id(app) == "high"
        verify_reuse.clear()

        apply_change(app)
        await pilot.pause()

        assert verify_reuse, "no reflow reused the arrangement after the change"
        assert _top_widget_id(app) == expected_top


class _SettlingScroll(VerticalScroll):
    """Settles an anchor pinned above its top while it arranges, as the chat panel does."""

    def arrange(self, size: Size, optimal: bool = False) -> DockArrangeResult:
        result = super().arrange(size, optimal=optimal)
        if self.scroll_y < 0 and self._anchored and not self._anchor_released:
            self._anchor_released = True
            self.set_reactive(VerticalScroll.scroll_y, 0.0)
        return result


class _Settling(App[None]):
    def compose(self) -> ComposeResult:
        with _SettlingScroll(id="short"):
            yield Static("short")
        yield Static("below", id="below")


async def _anchor_short_content(verify_reuse: list[Compositor]) -> None:
    app = _Settling()
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()
        verify_reuse.clear()
        # Anchoring pins content shorter than the viewport above its top; only the next
        # arrangement's override settles it, so arranging twice in a row differs.
        app.query_one("#short", _SettlingScroll).anchor()
        app.query_one("#below", Static).update("below\nnow taller")
        await pilot.pause()
        assert verify_reuse, "no reflow reused the arrangement after anchoring"


async def test_verification_starts_from_the_scroll_state_the_reusing_arrangement_started_from(
    verify_reuse: list[Compositor],
) -> None:
    await _anchor_short_content(verify_reuse)


async def test_verification_without_restoring_the_scroll_state_reports_a_false_divergence(
    verify_reuse: list[Compositor], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The restore the previous test relies on is what keeps it green."""
    monkeypatch.setattr(Compositor, "_restore_arrangement_state", staticmethod(lambda _state: None))

    with pytest.raises(AssertionError, match="compositor reuse diverged"):
        await _anchor_short_content(verify_reuse)


class _Lines(ScrollView):
    """A scroll view with no children, which the compositor pins without arranging any."""

    def on_mount(self) -> None:
        self.virtual_size = Size(20, 60)

    def render_line(self, y: int) -> Strip:
        return Strip.blank(self.size.width)


class _Squeezed(App[None]):
    """An anchored widget with a hidden scrollbar above a dock that can grow."""

    CSS = """
    #anchored { height: 1fr; scrollbar-size-vertical: 0; }
    #foot { dock: bottom; height: 2; }
    """

    def __init__(self, anchored: Callable[[], Widget]) -> None:
        super().__init__()
        self._anchored_factory = anchored

    def compose(self) -> ComposeResult:
        yield self._anchored_factory()
        yield Static("foot", id="foot")


def _anchored_container() -> Widget:
    return VerticalScroll(*(Static(f"line {index}") for index in range(60)), id="anchored")


def _anchored_lines() -> Widget:
    return _Lines(id="anchored")


async def _squeeze(anchored: Callable[[], Widget]) -> None:
    """Grow the dock below an anchored widget; only the widget's container size changes.

    The waits end early when the App fails, so its error leaves ``run_test`` at once.
    """
    app = _Squeezed(anchored)
    async with app.run_test(size=(40, 20)) as pilot:
        await pilot.pause()
        widget = app.query_one("#anchored")
        widget.anchor()
        await wait_for(
            lambda: app._exception is not None or 0 < widget.scroll_y == widget.max_scroll_y,
            description="the widget pinned to its end",
        )
        bottom = widget.max_scroll_y

        app.query_one("#foot", Static).styles.height = 6

        await wait_for(
            lambda: app._exception is not None or widget.scroll_y == widget.max_scroll_y == bottom + 4,
            description="the squeezed widget pinned to its new end",
        )


_ANCHORED = pytest.mark.parametrize("anchored", [_anchored_container, _anchored_lines], ids=["container", "leaf"])


@_ANCHORED
async def test_an_anchored_widget_squeezed_by_a_dock_stays_pinned_to_its_end(anchored: Callable[[], Widget]) -> None:
    """The container size is known only after the arrangement that pinned the widget; the oracle is off, as in the app."""
    textual_reflow_reuse.apply_runtime_patch()
    await _squeeze(anchored)


@_ANCHORED
async def test_verifying_a_squeezed_anchored_widget_stamps_nothing(
    verify_reuse: list[Compositor], anchored: Callable[[], Widget]
) -> None:
    """The from-scratch arrangement pins the widget and fires its scroll watchers; its stamps are undone."""
    await _squeeze(anchored)

    assert verify_reuse, "no reflow reused the arrangement"


def _unstamp_scroll_view_size_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave a scroll view's container-size change unstamped, as before the patch stamped it."""
    size_updated = ScrollView._size_updated

    def unstamped(self: ScrollView, size: Size, virtual_size: Size, container_size: Size, layout: bool = True) -> bool:
        vars(self)["_mark_layout_dirty"] = lambda: None
        try:
            return size_updated(self, size, virtual_size, container_size, layout)
        finally:
            del vars(self)["_mark_layout_dirty"]

    monkeypatch.setattr(ScrollView, "_size_updated", unstamped)


async def test_verification_fails_a_scroll_view_pinned_from_a_stale_container_size(
    verify_reuse: list[Compositor], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A widget that scrolls its own content is absent from the map, so only the compared state shows it.

    The from-scratch arrangement pins the widget the reusing one left, and its stamps are undone.
    """
    _unstamp_scroll_view_size_updates(monkeypatch)

    with pytest.raises(AssertionError, match=r"compositor reuse diverged: .*state_changed=\[\(_Lines\(id='anchored'\)"):
        await _squeeze(_anchored_lines)


async def test_a_verification_that_keeps_its_stamps_is_caught(
    verify_reuse: list[Compositor], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture's stamp check, which the previous test relies on, goes red without the epoch restore."""
    _unstamp_scroll_view_size_updates(monkeypatch)
    snapshot = Compositor._arrangement_state

    def without_epochs(root: Widget) -> list[tuple[DOMNode, tuple[str, ...], dict[str, object]]]:
        return [
            (
                node,
                tuple(name for name in names if name != "_layout_dirty_epoch"),
                {name: value for name, value in values.items() if name != "_layout_dirty_epoch"},
            )
            for node, names, values in snapshot(root)
        ]

    monkeypatch.setattr(Compositor, "_arrangement_state", staticmethod(without_epochs))

    with pytest.raises(AssertionError, match="the verification stamped"):
        await _squeeze(_anchored_lines)
