# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Dragging the workflow graph pans the canvas; the press that panned is not a click."""

from __future__ import annotations

from textual import on
from textual.app import App, ComposeResult
from textual.events import MouseDown, MouseEvent, MouseMove, MouseUp
from textual.geometry import Offset

from chrys.app.tui.widgets.workflow.graph import WorkflowGraph
from tests.support.waiting import wait_for

_MANIFEST = {
    "nodes": [{"id": f"node{index}", "kind": "python", "callable": {"name": "prepare"}} for index in range(8)],
    "edges": [{"src": f"node{index}", "dst": f"node{index + 1}"} for index in range(7)],
}


class _Harness(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.opened: list[str] = []

    def compose(self) -> ComposeResult:
        yield WorkflowGraph()

    def on_mount(self) -> None:
        graph = self.query_one(WorkflowGraph)
        graph.show_manifest(_MANIFEST, [])
        graph.show_nodes({})

    @on(WorkflowGraph.NodeSelected)
    def node_selected(self, event: WorkflowGraph.NodeSelected) -> None:
        self.opened.append(event.node_id)


def _send(app: App[None], event_type: type[MouseEvent], graph: WorkflowGraph, offset: Offset, button: int = 1) -> None:
    """Deliver a pointer event the way the terminal driver does, so the App also synthesizes clicks."""
    x, y = graph.region.offset + offset
    app.post_message(event_type(None, x, y, 0, 0, button, False, False, False, screen_x=x, screen_y=y))


def _node_offset(graph: WorkflowGraph, node_id: str) -> Offset:
    box = graph.geometry[node_id]
    content = graph.scrollable_content_region.offset - graph.region.offset
    return content + graph.diagram_origin - graph.scroll_offset + Offset(box.x + 1, box.y + 1)


def _blank_offset(graph: WorkflowGraph, *, min_x: int) -> Offset:
    region = graph.scrollable_content_region
    for y in range(region.height):
        for x in range(min_x, region.width):
            point = region.offset + Offset(x, y)
            if not graph._node_at(graph._diagram_offset(point)):
                return point - graph.region.offset
    raise AssertionError("no blank canvas in view")


async def _drag(app: _Harness, graph: WorkflowGraph, start: Offset, delta: Offset) -> None:
    _send(app, MouseDown, graph, start)
    _send(app, MouseMove, graph, start + delta)
    await wait_for(
        lambda: graph.styles.pointer == "grabbing" and app.mouse_captured is graph,
        description="graph pan in progress",
    )
    _send(app, MouseUp, graph, start + delta)
    await wait_for(lambda: app.mouse_captured is None, description="graph released the mouse")


async def test_dragging_blank_canvas_pans_and_keeps_the_selection() -> None:
    app = _Harness()
    async with app.run_test(size=(60, 16)) as pilot:
        graph = app.query_one(WorkflowGraph)
        await wait_for(lambda: graph.max_scroll_x > 20, pilot=pilot, description="graph wider than its viewport")
        graph.select_node("node1")
        graph.scroll_to(x=0, y=0, animate=False, force=True, immediate=True)
        graph.focus()

        await _drag(app, graph, _blank_offset(graph, min_x=20), Offset(-12, 0))
        assert graph.scroll_offset == Offset(12, 0)
        assert graph.styles.pointer == "default"
        # The key queues behind the click the App synthesized from the release: a click on blank
        # canvas would have cleared the selection, leaving Enter nothing to open.
        await pilot.press("enter")
        await wait_for(lambda: app.opened == ["node1"], pilot=pilot)
        assert graph.selected_node == "node1"

        # The same press without movement is still a click on blank canvas.
        blank = _blank_offset(graph, min_x=0)
        _send(app, MouseDown, graph, blank)
        _send(app, MouseUp, graph, blank)
        await wait_for(lambda: graph.selected_node == "", pilot=pilot)
        assert graph.scroll_offset == Offset(12, 0)


async def test_dragging_from_a_node_pans_without_opening_it() -> None:
    app = _Harness()
    async with app.run_test(size=(60, 16)) as pilot:
        graph = app.query_one(WorkflowGraph)
        await wait_for(lambda: graph.max_scroll_x > 20, pilot=pilot, description="graph wider than its viewport")
        graph.select_node("node2")
        graph.scroll_to(x=0, y=0, animate=False, force=True, immediate=True)
        graph.focus()

        start = _node_offset(graph, "node1")
        await _drag(app, graph, start, Offset(-6, 0))
        assert graph.scroll_offset == Offset(6, 0)
        # The canvas moved with the pointer, which is still over the node it grabbed.
        assert graph._hovered_node == "node1" and graph.styles.pointer == "pointer"
        await pilot.press("enter")
        await wait_for(lambda: bool(app.opened), pilot=pilot)
        assert app.opened == ["node2"] and graph.selected_node == "node2"

        # Without movement the press opens the node under it.
        _send(app, MouseDown, graph, start)
        _send(app, MouseUp, graph, start)
        await wait_for(lambda: app.opened == ["node2", "node1"], pilot=pilot)


async def test_losing_focus_or_a_right_button_press_does_not_pan() -> None:
    app = _Harness()
    async with app.run_test(size=(60, 16)) as pilot:
        graph = app.query_one(WorkflowGraph)
        await wait_for(lambda: graph.max_scroll_x > 20, pilot=pilot, description="graph wider than its viewport")
        graph.scroll_to(x=0, y=0, animate=False, force=True, immediate=True)
        start = _blank_offset(graph, min_x=20)

        _send(app, MouseDown, graph, start)
        await wait_for(lambda: app.mouse_captured is graph and app.focused is graph, pilot=pilot)
        app.screen.set_focus(None)
        await wait_for(lambda: app.mouse_captured is None, pilot=pilot, description="blur ended the pan")
        _send(app, MouseDown, graph, start, button=3)
        _send(app, MouseMove, graph, start - Offset(12, 0), button=3)
        _send(app, MouseUp, graph, start - Offset(12, 0), button=3)
        # A later left-button click proves every earlier event has been handled.
        graph.select_node("node0")
        graph.scroll_to(x=0, y=0, animate=False, force=True, immediate=True)
        blank = _blank_offset(graph, min_x=0)
        _send(app, MouseDown, graph, blank)
        _send(app, MouseUp, graph, blank)
        await wait_for(lambda: graph.selected_node == "", pilot=pilot)
        assert graph.scroll_offset == Offset(0, 0) and app.mouse_captured is None
