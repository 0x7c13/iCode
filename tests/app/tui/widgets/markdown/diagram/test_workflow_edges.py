# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Connection state, directed flow and viewport-local repainting."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import create_autospec

import pytest
from rich.cells import cell_len
from rich.style import Style
from textual.app import App, ComposeResult

from chrys.app.tui.widgets.markdown.diagram.canvas import styled_cell_segments
from chrys.app.tui.widgets.markdown.diagram.layout import compile_ir_with_geometry
from chrys.app.tui.widgets.markdown.diagram.model import Direction
from chrys.app.tui.widgets.workflow.edges import connection_state, edge_cells, ink_spans, pulse_strength
from chrys.app.tui.widgets.workflow.graph import WorkflowGraph, manifest_ir
from chrys.app.tui.widgets.workflow.node_view import NodeView


def _manifest() -> dict:
    return {
        "nodes": [
            {"id": name, "kind": "python", "callable": {"name": name}}
            for name in ("prepare", "review", "skip", "report")
        ],
        "edges": [
            {"src": "prepare", "dst": "review", "conditional": True, "predicate": "需要审查"},
            {"src": "prepare", "dst": "skip", "conditional": True, "predicate": "skip"},
            {"src": "review", "dst": "report"},
        ],
    }


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
def test_routes_follow_heading_and_trim_without_painting_labels_or_nodes(direction: Direction) -> None:
    ir = manifest_ir(_manifest(), direction=direction)
    plain, _ = compile_ir_with_geometry("", ir)
    headed, boxes = compile_ir_with_geometry("", replace(ir, title="Workflow"))
    shift = headed.height - plain.height
    assert shift > 0
    assert headed.routed_edges == tuple(route.translated(0, shift) for route in plain.routed_edges)
    diagram = headed
    rows = edge_cells(diagram, boxes)
    assert rows
    for y, cells in rows.items():
        for cell in cells:
            assert 0 <= cell.x < diagram.width and 0 <= y < diagram.height
            assert not any(
                box.x <= cell.x < box.x + box.width and box.y <= y < box.y + box.height for box in boxes.values()
            )
            for route in diagram.routed_edges:
                if route.label_at is not None and route.label_at.y == y:
                    assert not route.label_at.x <= cell.x < route.label_at.x + cell_len(route.edge.label)
    for route in diagram.routed_edges:
        assert route.arrow_at is not None
        assert diagram.crop_plain_row(route.arrow_at.y, route.arrow_at.x, 1) == route.arrow


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("running", "pending", "pending"),
        ("completed", "running", "running"),
        ("completed", "completed", "completed"),
        ("completed", "awaiting_retry", "awaiting_retry"),
        ("completed", "cancelled", "cancelled"),
        ("completed", "skipped", "skipped"),
        ("pending", "running", "pending"),
    ],
)
def test_connections_do_not_animate_undecided_outgoing_branches(source: str, target: str, expected: str) -> None:
    assert connection_state(source, target) == expected


def test_pulse_moves_toward_target_and_repeats_without_a_phase_jump() -> None:
    for frame in range(80):
        assert pulse_strength(frame, frame) == 3
        assert pulse_strength(frame - 3, frame) == 2
        assert pulse_strength(frame - 5, frame) == 1
        assert pulse_strength(frame - 9, frame) == 0
    assert [pulse_strength(x, 80) for x in range(32)] == [pulse_strength(x, 0) for x in range(32)]


def test_long_connection_ink_merges_without_coloring_gaps() -> None:
    active, idle = Style(color="magenta"), Style(color="grey50")
    spans = ink_spans({**dict.fromkeys(range(1000), active), 1002: idle, 1003: idle})
    assert len(spans) == 2
    segments = styled_cell_segments("─" * 1000 + "  ──", spans, 998, 6, Style())
    assert [(segment.text, segment.style) for segment in segments] == [("──", active), ("  ", Style()), ("──", idle)]


@pytest.mark.parametrize("direction", [Direction.LEFT_RIGHT, Direction.TOP_DOWN])
async def test_animation_preserves_geometry_and_repaints_only_visible_rows(
    direction: Direction, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Harness(App):
        def compose(self) -> ComposeResult:
            yield WorkflowGraph()

    async with Harness().run_test(size=(64, 16)) as pilot:
        graph = pilot.app.query_one(WorkflowGraph)
        graph.direction = direction
        graph.show_manifest(_manifest(), [])
        graph.show_nodes({"prepare": NodeView("completed"), "review": NodeView("running"), "skip": NodeView("skipped")})
        graph.select_node("review")
        await pilot.pause()
        diagram, geometry, scroll = graph.diagram, graph.geometry, graph.scroll_offset
        assert graph._edge_states == ["running", "skipped", "pending"]
        before = {y: graph._pulse_spans(y) for y in graph._flow_rows}
        refresh = create_autospec(graph.refresh, side_effect=graph.refresh)
        monkeypatch.setattr(graph, "refresh", refresh)
        graph.advance_animation()
        assert any(graph._pulse_spans(y) != spans for y, spans in before.items())
        assert refresh.call_count
        for call in refresh.call_args_list:
            assert not call.kwargs.get("layout")
            assert call.args and all(
                region.height == 1 and 0 <= region.y < graph.scrollable_content_region.height for region in call.args
            )
        assert graph.diagram is diagram and graph.geometry is geometry and graph.scroll_offset == scroll

        pilot.app.animation_level = "none"
        refresh.reset_mock()
        frame = graph._animation_frame
        graph.advance_animation()
        assert graph._animation_frame == frame
        assert all(not graph._pulse_spans(y) for y in graph._flow_rows)
        refresh.assert_not_called()

        pilot.app.animation_level = "full"
        graph.display = False
        await pilot.pause()
        refresh.reset_mock()
        graph.advance_animation()
        refresh.assert_not_called()
