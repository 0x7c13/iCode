# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow manifest geometry, conditional edges, iteration badges and node selection."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual import on
from textual.app import App, ComposeResult

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.widgets.markdown.diagram.layout import compile_ir_with_geometry
from chrys.app.tui.widgets.markdown.diagram.model import Direction, EdgeStyle, NodeShape
from chrys.app.tui.widgets.workflow.graph import WorkflowGraph, manifest_ir
from chrys.app.tui.widgets.workflow.node_view import NodeView, RetryTarget
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.types import WorkflowNodeStateChanged
from chrys.service.workflows.discovery import read_builtin_manifest
from chrys.service.workflows.transcript import NodeUsage
from tests.support.waiting import wait_for


@pytest.mark.parametrize("count", [1, 4, 201], ids=["centered", "scrolled", "list"])
@pytest.mark.parametrize("locale", ["en", "zh-Hans"])
async def test_retry_link_uses_cell_coordinates_and_preserves_node_selection(count: int, locale: str) -> None:
    node_id = f"node{count - 1}"
    failure = WorkflowNodeStateChanged(
        run_id="run", node_id=node_id, activation_id=f"{node_id}@iter#2", attempt=2, state="awaiting_retry"
    )
    manifest = {
        "nodes": [
            {"id": f"node{index}", "kind": "python", "callable": {"name": "读取" + "x" * (50 if count > 1 else 1)}}
            for index in range(count)
        ],
        "edges": [{"src": f"node{index}", "dst": f"node{index + 1}"} for index in range(count - 1)],
    }
    selected: list[str] = []
    retries: list[RetryTarget] = []

    class Harness(App):
        def compose(self) -> ComposeResult:
            yield WorkflowGraph()

        def on_mount(self) -> None:
            graph = self.query_one(WorkflowGraph)
            graph.show_manifest(manifest, [], locale=LocaleController(Settings(locale=locale)))
            graph.show_nodes(
                {
                    node_id: NodeView(
                        "awaiting_retry",
                        retry=RetryTarget(failure.run_id, failure.activation_id, failure.attempt),
                        usage=NodeUsage(3, 1000),
                    )
                }
            )

        @on(WorkflowGraph.NodeSelected)
        def node_selected(self, event: WorkflowGraph.NodeSelected) -> None:
            selected.append(event.node_id)

        @on(WorkflowGraph.RetryRequested)
        def retry_requested(self, event: WorkflowGraph.RetryRequested) -> None:
            retries.append(event.target)

    async with Harness().run_test(size=(120, 20) if count == 1 else (38, 10)) as pilot:
        graph = pilot.app.query_one(WorkflowGraph)
        graph.select_node(node_id)
        region = graph._retry_regions[node_id]
        graph.scroll_to(x=max(0, region.x - 10), y=max(0, region.y - 4), animate=False, force=True, immediate=True)
        diagram, geometry, scroll = graph.diagram, graph.geometry, graph.scroll_offset
        x = region.x - round(scroll.x) + graph.diagram_origin.x
        y = region.y - round(scroll.y) + graph.diagram_origin.y
        row = graph.render_line(y)
        label = "Retry" if locale == "en" else "重试"
        assert label in row.text
        assert any(segment.style and segment.style.underline for segment in row)
        assert await pilot.click(graph, offset=(x + graph.gutter.left, y + graph.gutter.top))
        await wait_for(lambda: len(retries) == 1, pilot=pilot)
        assert retries == [RetryTarget(failure.run_id, failure.activation_id, failure.attempt)] and selected == []
        graph.show_nodes({node_id: replace(graph._views[node_id], retry_pending=True)})
        graph.focus()
        await pilot.press("r")
        assert retries == [RetryTarget(failure.run_id, failure.activation_id, failure.attempt)]
        graph.show_nodes({node_id: NodeView("running")})
        assert not graph._retry_regions
        assert graph.diagram is diagram and graph.geometry is geometry
        # The action key cannot retry a running or terminal node.
        await pilot.press("r", "enter")
        assert retries == [RetryTarget(failure.run_id, failure.activation_id, failure.attempt)] and selected == [
            node_id
        ]
        assert graph.selected_node == node_id
        next_failure = replace(failure, attempt=3)
        graph.show_nodes(
            {
                node_id: NodeView(
                    "awaiting_retry",
                    retry=RetryTarget(next_failure.run_id, next_failure.activation_id, next_failure.attempt),
                )
            }
        )
        graph.select_node(node_id)
        await pilot.press("r")
        await wait_for(lambda: len(retries) == 2, pilot=pilot)
        assert retries[-1].attempt == 3


def test_manifest_expands_loops_and_preserves_join_and_switch_semantics() -> None:
    manifest = read_builtin_manifest("demo-workflow")
    assert manifest is not None
    ir = manifest_ir(manifest)
    nodes = {node.node_id: node for node in ir.nodes}
    assert nodes["refine"].shape is NodeShape.ROUNDED
    assert nodes["join:refine"].sections == (("carry_state",),)
    assert nodes["read_request"].sections == (("py · read_request",),)
    assert nodes["architecture"].sections == (("QA · default model",),)
    edges = {(edge.source, edge.target): edge for edge in ir.edges}
    assert ("refine", "open_round") in edges
    assert edges["review", "refine"].style is EdgeStyle.DOTTED
    # Leaving the loop: the plain edge is solid, the conditional one dotted, like both arms of the switch.
    assert edges["review", "render_tour"].style is EdgeStyle.SOLID
    assert edges["review", "tour_text"].style is EdgeStyle.DOTTED
    assert edges["plan", "fan_out"].style is EdgeStyle.DOTTED
    assert edges["plan", "hand_over"].style is EdgeStyle.DOTTED
    resolved = [{"node_id": "architecture", "agent_display_name": "Reviewer", "model_id": "mock-model"}]
    # A run names what was resolved and reserves the statistics row of every agent up front.
    refreshed = manifest_ir(manifest, resolved, reserve_usage=True)
    assert next(node for node in refreshed.nodes if node.node_id == "architecture").sections == (
        ("Reviewer · mock-model",),
        ("",),
    )


@pytest.mark.parametrize("direction", list(Direction))
@pytest.mark.parametrize("reverse_declarations", [False, True])
def test_builtin_graph_preserves_parallel_branches_and_join_order(
    direction: Direction, reverse_declarations: bool
) -> None:
    manifest = read_builtin_manifest("demo-workflow")
    assert manifest is not None
    if reverse_declarations:
        manifest = {**manifest, "nodes": list(reversed(manifest["nodes"]))}
    ir = manifest_ir(manifest, direction=direction)
    diagram, placed = compile_ir_with_geometry("", ir)
    assert not diagram.diagnostics
    horizontal = direction in {Direction.LEFT_RIGHT, Direction.RIGHT_LEFT}
    reverse = direction in {Direction.BOTTOM_UP, Direction.RIGHT_LEFT}
    positions = {node_id: (box.x if horizontal else box.y) * (-1 if reverse else 1) for node_id, box in placed.items()}
    readers = ("architecture", "entry_points", "conventions")
    assert len({positions[node_id] for node_id in readers}) == 1
    assert len({placed[node_id].y if horizontal else placed[node_id].x for node_id in readers}) == 3
    assert positions["fan_out"] < positions[readers[0]] < positions["join:merge_findings"] < positions["merge_findings"]
    feedback = ("review", "refine")
    for edge in ir.edges:
        if (edge.source, edge.target) != feedback:
            assert positions[edge.source] < positions[edge.target]
    expected_edges = {(edge.target, edge.source) if reverse else (edge.source, edge.target) for edge in ir.edges}
    assert {(route.edge.source, route.edge.target) for route in diagram.routed_edges} == expected_edges
    feedback_route = next(
        route
        for route in diagram.routed_edges
        if (route.edge.source, route.edge.target) == (feedback[::-1] if reverse else feedback)
    )
    assert feedback_route.edge.style == EdgeStyle.DOTTED
    assert (feedback_route.source_marker_at if reverse else feedback_route.arrow_at) is not None


@pytest.mark.parametrize("locale", ["en", "zh-Hans"])
async def test_workflow_graph_keyboard_click_and_iteration_overlay(locale: str) -> None:
    manifest = read_builtin_manifest("demo-workflow")
    assert manifest is not None
    selected: list[str] = []

    class Harness(App):
        def compose(self) -> ComposeResult:
            yield WorkflowGraph()

        def on_mount(self) -> None:
            self.query_one(WorkflowGraph).show_manifest(manifest, [], locale=LocaleController(Settings(locale=locale)))

        @on(WorkflowGraph.NodeSelected)
        def node_selected(self, event: WorkflowGraph.NodeSelected) -> None:
            selected.append(event.node_id)

    app = Harness()
    async with app.run_test(size=(120, 30)) as pilot:
        graph = app.query_one(WorkflowGraph)
        graph.focus()
        assert not graph.selected_node
        await pilot.press("enter")
        assert not selected
        await pilot.press("j", "enter")
        assert selected == [next(iter(graph.geometry))]
        assert graph.selected_node == selected[-1]
        await pilot.press("k", "enter")
        assert selected[-1] == list(graph.geometry)[-1]
        graph.select_node("refine")
        box = graph.geometry["refine"]
        await wait_for(lambda: graph.scrollable_content_region.height > 0, pilot=pilot)
        x = box.x + 1 - round(graph.scroll_offset.x) + graph.diagram_origin.x
        y = box.y + 1 - round(graph.scroll_offset.y) + graph.diagram_origin.y
        await pilot.click(graph, offset=(x + graph.gutter.left, y + graph.gutter.top))
        assert selected[-1] == "refine"
        assert graph.selected_node == "refine"
        graph.select_node("refine")
        diagram = graph.diagram
        geometry = graph.geometry
        scroll = graph.scroll_offset
        graph.show_iterations({"refine": (2, 3)})
        label = "Iteration 2/3" if locale == "en" else "迭代 2/3"
        assert graph.diagram is diagram and graph.geometry is geometry and graph.scroll_offset == scroll
        _, badge_y, _ = graph._badge_positions["refine"]
        row = graph.render_line(badge_y - round(graph.scroll_offset.y) + graph.diagram_origin.y)
        assert label in row.text
        for direction in (Direction.TOP_DOWN, Direction.LEFT_RIGHT):
            graph.toggle_layout()
            assert graph.direction == direction and graph.selected_node == "refine"
            _, badge_y, _ = graph._badge_positions["refine"]
            await wait_for(
                lambda badge_y=badge_y: (
                    label in graph.render_line(badge_y - round(graph.scroll_offset.y) + graph.diagram_origin.y).text
                ),
                pilot=pilot,
                description="selected loop badge visible after layout switch",
            )
            row = graph.render_line(badge_y - round(graph.scroll_offset.y) + graph.diagram_origin.y)
            assert label in row.text


async def test_wide_workflow_uses_selectable_list_when_diagram_exceeds_canvas_limit() -> None:
    agents = [f"review_{index:03}" for index in range(190)]
    manifest = {
        "nodes": [
            {"id": "prepare", "kind": "python", "callable": {"name": "prepare"}},
            *({"id": node_id, "kind": "agent", "agent": {"profile": "QA"}} for node_id in agents),
            {"id": "report", "kind": "join", "callable": {"name": "report"}},
        ],
        "edges": [
            edge
            for node_id in agents
            for edge in ({"src": "prepare", "dst": node_id}, {"src": node_id, "dst": "report"})
        ],
    }
    selected: list[str] = []

    class Harness(App):
        def compose(self) -> ComposeResult:
            yield WorkflowGraph()

        def on_mount(self) -> None:
            graph = self.query_one(WorkflowGraph)
            # In a vertical flow these parallel branches exceed the canvas width.
            graph.direction = Direction.TOP_DOWN
            graph.show_manifest(manifest, [])

        @on(WorkflowGraph.NodeSelected)
        def node_selected(self, event: WorkflowGraph.NodeSelected) -> None:
            selected.append(event.node_id)

    app = Harness()
    async with app.run_test(size=(120, 30)) as pilot:
        graph = app.query_one(WorkflowGraph)
        assert len(manifest["nodes"]) < 200
        assert graph.list_fallback
        assert set(graph.geometry) == {node["id"] for node in manifest["nodes"]}
        graph.show_nodes({agents[0]: NodeView("awaiting_retry")})
        graph.focus()
        await pilot.press("j", "j", "enter", "j", "enter")
        assert selected == [agents[0], agents[1]]
        box = graph.geometry[agents[0]]
        assert await pilot.click(
            graph,
            offset=(
                box.x + 1 + graph.diagram_origin.x + graph.gutter.left,
                box.y + graph.diagram_origin.y + graph.gutter.top,
            ),
        )
        assert selected == [agents[0], agents[1], agents[0]]
