# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chrome flips on a populated transcript replay the transcript instead of arranging it again.

The scroll-to-bottom button, the status bar, the suggestion list and the input bar's buttons
flip visibility or pinned widths dozens of times per turn and resynchronize the compositor
outside a layout pass each time. A from-scratch reflow there arranges every transcript
container again; with the reflow-reuse patch only the flipped widgets' ancestor paths are
arranged, and the reflow-reuse oracle checks the result against a from-scratch one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest
import textual._compositor as compositor_module
from textual._compositor import Compositor, ReflowResult
from textual.widget import Widget

from chrys.app.tui.screens.main.screen import MainScreen
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chrome.input_bar import InputBar
from chrys.app.tui.widgets.chrome.status_bar import StatusBar
from chrys.app.tui.widgets.chrome.suggestion_list import SuggestionList
from chrys.foundation.patches import textual_reflow_reuse
from chrys.service.tools.kinds import KIND_FILESYSTEM_READ
from tests.support.pilot_barrier import screen_is_settled
from tests.support.tui_app_harness import make_chrys_app
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from pathlib import Path

    from textual._arrange import DockArrangeResult
    from textual._compositor import CompositorMap
    from textual.dom import DOMNode
    from textual.geometry import Size
    from textual.pilot import Pilot

    from chrys.app.tui.app import ChrysApp

_TURNS = 12


async def _populate(panel: ChatPanel) -> None:
    panel.set_tool_kinds({"read_file": KIND_FILESYSTEM_READ})
    for turn in range(_TURNS):
        await panel.add_user_message(f"Request {turn}: review module {turn}.")
        call_id = f"read-{turn}"
        await panel.add_tool_start(call_id, "read_file", KIND_FILESYSTEM_READ, args={"path": f"src/module_{turn}.py"})
        await panel.add_tool_result(call_id, "read_file", "\n".join(f"line {index}" for index in range(6)), 12)
        await panel.add_agent_message(
            f"## Turn {turn}\n\n- first point\n- second point\n\n```python\nvalue = {turn}\n```", is_final=True
        )


@asynccontextmanager
async def _populated_main_screen(tmp_path: Path) -> AsyncIterator[tuple[ChrysApp, Pilot[None], MainScreen]]:
    """A settled MainScreen whose transcript overflows the chat panel, scrolled to the bottom."""
    textual_reflow_reuse.apply_runtime_patch()
    app = make_chrys_app(tmp_path)
    async with app.run_test(size=(120, 36)) as pilot:
        screen = app.screen
        assert isinstance(screen, MainScreen)
        panel = screen.query_one(ChatPanel)
        await _populate(panel)
        await wait_for(
            lambda: panel.max_scroll_y > 10 and panel.scroll_y == panel.max_scroll_y,
            pilot=pilot,
            description="transcript overflowing the panel, anchored at its end",
        )
        await wait_for(lambda: screen_is_settled(app, screen), pilot=pilot, description="settled transcript layout")
        yield app, pilot, screen


def _chrome_flips(screen: MainScreen) -> list[tuple[str, Callable[[], None]]]:
    """Each chrome flip through its owner's API, in an order where every step changes geometry."""
    panel = screen.query_one(ChatPanel)
    status = screen.query_one(StatusBar)
    suggestions = screen.query_one(SuggestionList)
    input_bar = screen.query_one(InputBar)

    def scroll_up() -> None:
        panel.scroll_to(y=panel.max_scroll_y - 3, animate=False, immediate=True)

    def scroll_to_end() -> None:
        panel.scroll_to(y=panel.max_scroll_y, animate=False, immediate=True)

    def show_suggestions() -> None:
        suggestions.show("commands", [("/help", "Show help"), ("/new", "Start a new chat")])

    def show_new_button() -> None:
        input_bar.has_messages = True

    def start_run() -> None:
        input_bar.agent_running = True

    return [
        ("scroll-button shown", scroll_up),
        ("scroll-button hidden", scroll_to_end),
        ("status shown", lambda: status.show("Working")),
        ("status flashed", lambda: status.flash("Done")),
        ("status hidden", status.hide),
        ("suggestions shown", show_suggestions),
        ("suggestions hidden", suggestions.hide),
        ("new button shown", show_new_button),
        ("send button relabelled", start_run),
    ]


def _transcript(panel: ChatPanel) -> set[Widget]:
    """Every widget of the transcript entries, without the panel's own chrome."""
    chrome = {panel.scroll_to_bottom_button(), panel.bottom_spacer()}
    return {
        widget
        for entry in panel.children
        if entry not in chrome
        for widget in entry.walk_children(Widget, with_self=True)
    }


@pytest.mark.asyncio
async def test_chrome_flips_on_a_populated_transcript_do_not_arrange_the_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _populated_main_screen(tmp_path) as (_app, _pilot, screen):
        panel = screen.query_one(ChatPanel)
        compositor = screen._compositor
        transcript = _transcript(panel)
        arranged: list[Widget] = []
        reflows: list[Widget] = []
        arrange = Widget.arrange
        reflow = compositor.reflow

        def record_arrange(self: Widget, size: Size, optimal: bool = False) -> DockArrangeResult:
            arranged.append(self)
            return arrange(self, size, optimal)

        def record_reflow(parent: Widget, size: Size, reuse: bool = False) -> ReflowResult:
            reflows.append(parent)
            return reflow(parent, size, reuse=reuse)

        # The oracle's own from-scratch arrangement would show up in the spy.
        monkeypatch.setattr(compositor_module, "_VERIFY_REUSE", False)
        monkeypatch.setattr(Widget, "arrange", record_arrange)
        monkeypatch.setattr(compositor, "reflow", record_reflow)

        for step, flip in _chrome_flips(screen):
            arranged.clear()
            reflows.clear()
            # Synchronous: only the flip's own compositor resync runs in between.
            flip()
            assert reflows, f"{step}: the flip did not resynchronize the compositor"
            assert [widget for widget in arranged if widget in transcript] == [], f"{step}: arranged the transcript"

        # Positive control: the spy sees the transcript containers a from-scratch reflow arranges.
        arranged.clear()
        compositor.reflow(screen, screen.outer_size)
        assert len({widget for widget in arranged if widget in transcript}) >= _TURNS


@pytest.mark.asyncio
async def test_chrome_flips_on_a_populated_transcript_match_a_from_scratch_reflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reflow-reuse oracle checks each flip's reusing arrangement; a divergence raises in the flip."""
    async with _populated_main_screen(tmp_path) as (_app, pilot, screen):
        checks: list[Widget] = []
        verify = Compositor._verify_reuse

        def record_check(
            self: Compositor,
            root: Widget,
            size: Size,
            map: CompositorMap,
            widgets: set[Widget],
            state: list[tuple[DOMNode, tuple[str, ...], dict[str, object]]],
        ) -> None:
            checks.append(root)
            verify(self, root, size, map, widgets, state)

        monkeypatch.setattr(compositor_module, "_VERIFY_REUSE", True)
        monkeypatch.setattr(Compositor, "_verify_reuse", record_check)

        for step, flip in _chrome_flips(screen):
            checks.clear()
            flip()
            assert screen in checks, f"{step}: no reusing arrangement was checked"
            # The layout passes the flip leads to are checked too; a divergence there fails the App.
            await pilot.pause()
