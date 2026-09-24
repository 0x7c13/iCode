# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow navigation stays below the actions and follows the graph viewport."""

from __future__ import annotations

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.geometry import Region
from textual.scrollbar import ScrollBarRender
from textual.widgets import TabbedContent

from chrys.app.tui.theme import CHRYS_ANSI_THEME, CHRYS_THEME, TuiVariableDefaultsMixin
from chrys.app.tui.util.visibility import is_widget_shown_on_active_screen
from chrys.app.tui.widgets.workflow.graph import WorkflowGraph
from chrys.app.tui.widgets.workflow.panel import WorkflowPanel
from chrys.app.tui.widgets.workflow.scrollbar import WorkflowScrollBar
from tests.support.waiting import wait_for


class Harness(TuiVariableDefaultsMixin, App):
    def __init__(self) -> None:
        super().__init__()
        self.register_theme(CHRYS_THEME)
        self.register_theme(CHRYS_ANSI_THEME)

    def compose(self) -> ComposeResult:
        panel = WorkflowPanel()
        panel.display = True
        yield panel


def _manifest(count: int) -> dict:
    return {
        "nodes": [
            {"id": f"node{index}", "kind": "python", "callable": {"name": "process_workflow_stage"}}
            for index in range(count)
        ],
        "edges": [{"src": "node0", "dst": f"node{index}"} for index in range(1, count)],
    }


@pytest.mark.parametrize("theme", ["textual-dark", "textual-light", "chrys-ansi"])
async def test_footer_navigation_drags_pages_and_follows_keyboard_and_resize(theme: str) -> None:
    app = Harness()
    app.theme = theme
    async with app.run_test(size=(60, 24)) as pilot:
        panel = app.query_one(WorkflowPanel)
        panel.remove_class("-empty")
        graph = panel.query_one(WorkflowGraph)
        graph.show_manifest(_manifest(8), [])
        bar = panel.query_one(WorkflowScrollBar)
        controls = panel.query_one("#workflow-controls")
        native = graph.horizontal_scrollbar
        # display and window_size flip before the reflow that places the bar, so wait on the geometry too.
        await wait_for(
            lambda: (
                bar.display
                and bar.window_size == graph.scrollable_content_region.width
                and bar.region.y == controls.region.bottom + 1
                and bar.region.bottom == panel.content_region.bottom
            ),
            pilot=pilot,
            description="workflow scrollbar laid out below the controls",
        )
        assert bar.region.y == controls.region.bottom + 1
        assert bar.region.bottom == panel.content_region.bottom
        assert bar.region.width == graph.scrollable_content_region.width + graph.styles.padding.width
        assert not native.visible
        assert app.screen.get_widget_at(native.region.x, native.region.y)[0] is not native
        assert bar.window_virtual_size == graph.virtual_size.width
        before = controls.region
        rendered, original = bar.render(), native.render()
        assert isinstance(rendered, ScrollBarRender) and isinstance(original, ScrollBarRender)
        assert rendered.style == original.style

        # Clicking the track pages the graph; dragging the thumb uses native capture.
        assert await pilot.click(bar, offset=(bar.size.width - 2, 0))
        await wait_for(lambda: graph.scroll_x > 0 and bar.position == native.position, pilot=pilot)
        graph.scroll_to(x=0, animate=False, immediate=True)
        await wait_for(lambda: bar.position == 0, pilot=pilot)
        assert await pilot.mouse_down(bar, offset=(1, 0))
        await wait_for(lambda: bar.grabbed is not None, pilot=pilot)
        assert await pilot.hover(bar, offset=(12, 0))
        assert await pilot.mouse_up(bar, offset=(12, 0))
        await wait_for(lambda: graph.scroll_x > 0 and bar.grabbed is None, pilot=pilot)
        assert controls.region == before

        graph.focus()
        graph.scroll_to(x=0, animate=False, immediate=True)
        await pilot.press("right")
        await wait_for(lambda: graph.scroll_x > 0 and bar.position == native.position, pilot=pilot)
        graph.scroll_to(x=0, animate=False, immediate=True)
        bar.post_message(
            events.MouseScrollRight(bar, x=1, y=0, delta_x=1, delta_y=0, button=0, shift=False, meta=False, ctrl=False)
        )
        await wait_for(lambda: graph.scroll_x > 0 and bar.position == native.position, pilot=pilot)
        await pilot.resize_terminal(70, 28)
        await wait_for(lambda: bar.window_size == graph.scrollable_content_region.width, pilot=pilot)
        assert bar.region.width == graph.scrollable_content_region.width + graph.styles.padding.width
        assert bar.region.bottom == panel.content_region.bottom

        graph.toggle_layout()
        await wait_for(
            lambda: (
                not graph.show_vertical_scrollbar
                and bar.region.width == graph.scrollable_content_region.width + graph.styles.padding.width
                and bar.window_size == graph.scrollable_content_region.width
            ),
            pilot=pilot,
        )
        assert bar.region.y == controls.region.bottom + 1

        tabs = panel.query_one("#workflow-run", TabbedContent)
        for tab in ("workflow-info-tab", "workflow-code-tab", "workflow-input-tab", "workflow-output-tab"):
            tabs.active = tab
            await pilot.pause()
            assert not is_widget_shown_on_active_screen(bar)
        tabs.active = "workflow-graph-tab"
        graph.show_manifest(_manifest(1), [])
        # Hiding the bar schedules a layout; its display flag changes before
        # the controls have received the two rows the bar and margin occupied.
        await wait_for(
            lambda: not bar.display and controls.region.bottom == panel.content_region.bottom,
            pilot=pilot,
            description="workflow controls fill the viewport after the scrollbar is removed",
        )
        assert graph.max_scroll_x == 0
        assert controls.region.bottom == panel.content_region.bottom


async def test_welcome_has_no_scrollbar_even_after_a_wide_graph() -> None:
    async with Harness().run_test(size=(60, 24)) as pilot:
        panel = pilot.app.query_one(WorkflowPanel)
        graph = panel.query_one(WorkflowGraph)
        graph.show_manifest(_manifest(8), [])
        await pilot.pause()
        assert not panel.query_one(WorkflowScrollBar).display


async def test_original_bar_stays_blank_when_switching_to_and_from_ansi() -> None:
    async with Harness().run_test(size=(60, 24)) as pilot:
        panel = pilot.app.query_one(WorkflowPanel)
        panel.remove_class("-empty")
        graph = panel.query_one(WorkflowGraph)
        graph.show_manifest(_manifest(8), [])
        footer = panel.query_one(WorkflowScrollBar)
        native = graph.horizontal_scrollbar
        for theme in ("chrys", "chrys-ansi", "textual-light", "chrys-ansi", "chrys"):
            pilot.app.theme = theme
            await pilot.pause()
            assert footer.display
            # ANSI default foreground and background are distinct terminal colors:
            # reversed spaces still paint a white bar even at zero opacity.
            hidden_row = native.render_lines(Region(0, 0, native.size.width, 1))[0]
            assert not hidden_row.text.strip()
            assert all(not segment.style or not segment.style.reverse for segment in hidden_row)
            visible_row = footer.render_lines(Region(0, 0, footer.size.width, 1))[0]
            assert any(segment.style and segment.style.reverse for segment in visible_row)
