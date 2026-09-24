# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow status colors follow the active theme without rebuilding the graph."""

from __future__ import annotations

import pytest
from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.color import Color
from textual.theme import Theme

from chrys.app.tui.widgets.workflow.graph import WorkflowGraph
from chrys.app.tui.widgets.workflow.node_view import NodeView, RetryTarget
from chrys.app.tui.widgets.workflow.output import WorkflowStatusOutput
from chrys.app.tui.widgets.workflow.projector import ObservedRun
from chrys.foundation.events import types as events
from tests.support.waiting import wait_for


@pytest.mark.parametrize("ansi", [False, True], ids=["rgb", "ansi"])
async def test_node_and_output_colors_follow_theme_changes(ansi: bool) -> None:
    states = {
        "done": "completed",
        "failed": "failed",
        "waiting": "awaiting_retry",
        "cancelled": "cancelled",
        "retrying": "retrying",
        "active": "running",
    }
    tokens = {
        "done": "success",
        "failed": "error",
        "waiting": "error",
        "cancelled": "warning",
        "retrying": "warning",
        "active": "accent",
    }
    colors = (
        {"success": "ansi_blue", "error": "ansi_magenta", "warning": "ansi_cyan", "accent": "ansi_white"}
        if ansi
        else {"success": "#225588", "error": "#dd6699", "warning": "#bb8844", "accent": "#6677aa"}
    )
    manifest = {
        "nodes": [{"id": node, "kind": "python", "callable": {"name": node}} for node in states],
        "edges": [{"src": a, "dst": b} for a, b in zip(states, list(states)[1:], strict=False)],
    }
    failure = events.WorkflowNodeStateChanged(
        run_id="run", node_id="waiting", activation_id="waiting", state="awaiting_retry", error="Failed check"
    )
    run = ObservedRun(events.WorkflowRunStarted(run_id="run", title="Test", manifest=manifest), facts=[failure])
    run.notices["notice"] = events.WorkflowRunNotice(run_id="run", code="notice", message="Check configuration")

    class Harness(App):
        def compose(self) -> ComposeResult:
            yield WorkflowGraph()
            yield WorkflowStatusOutput(None)

        def on_mount(self) -> None:
            graph = self.query_one(WorkflowGraph)
            graph.show_manifest(manifest, [])
            graph.show_nodes(
                {
                    node_id: NodeView(
                        state,
                        retry=RetryTarget(failure.run_id, failure.activation_id, failure.attempt)
                        if node_id == "waiting"
                        else None,
                    )
                    for node_id, state in states.items()
                }
            )
            self.query_one(WorkflowStatusOutput).show_run(run)

    app = Harness()
    async with app.run_test(size=(75, 20)) as pilot:
        graph = app.query_one(WorkflowGraph)
        output = app.query_one(WorkflowStatusOutput)
        graph.select_node("failed")
        diagram, geometry, scroll = graph.diagram, graph.geometry, graph.scroll_offset

        def node_style(node: str) -> Style:
            box = graph.geometry[node]
            return next(span.style for span in graph.cell_styles[box.y] if span.start == box.x)

        def output_color(message: str) -> str | None:
            content = output.content
            assert isinstance(content, Text)
            color = content.get_style_at_offset(app.console, content.plain.index(message)).color
            return color.name if color else None

        app.register_theme(
            Theme(
                name="workflow-test",
                primary="ansi_green" if ansi else "#4488bb",
                ansi=ansi,
                accent=colors["accent"],
                success=colors["success"],
                error=colors["error"],
                warning=colors["warning"],
            )
        )
        app.theme = "workflow-test"
        await wait_for(lambda: node_style("done").color == Color.parse(colors["success"]).rich_color, pilot=pilot)
        for node, token in tokens.items():
            if node == graph.selected_node:
                token = "primary"
            assert node_style(node).color == Color.parse(app.theme_variables[token]).rich_color
        assert not node_style("failed").reverse
        assert node_style("failed").bgcolor == node_style("done").bgcolor
        box = graph.geometry["failed"]
        fill = graph.get_component_rich_style("workflow-node--selected").bgcolor
        for y in range(box.y + 1, box.y + box.height - 1):
            assert any(
                span.start == box.x + 1 and span.end == box.x + box.width - 1 and span.style.bgcolor == fill
                for span in graph.cell_styles[y]
            )
        assert node_style("failed").bold
        title_y = box.y + 1 + graph.diagram_origin.y - round(graph.scroll_offset.y)
        title = graph.render_line(title_y)
        assert any(
            segment.text == "failed"
            and segment.style
            and segment.style.bold
            and segment.style.color == Color.parse(app.theme_variables["primary"]).rich_color
            and segment.style.bgcolor == fill
            for segment in title
        )
        assert all(not segment.style.reverse for segment in title if segment.style)
        region = graph._retry_regions["waiting"]
        assert any(span.style.underline for span in graph.cell_styles[region.y])
        await wait_for(
            lambda: output_color("Failed check") == Color.parse(app.theme_variables["error"]).rich_color.name,
            pilot=pilot,
        )
        assert output_color("Check configuration") == Color.parse(app.theme_variables["warning"]).rich_color.name
        assert graph.diagram is diagram and graph.geometry is geometry
        assert graph.scroll_offset == scroll

        # Refreshing a preview (for example after choosing another directory)
        # must restore its styles even when the theme itself hasn't changed.
        graph.show_manifest(manifest, [])
        first = graph.geometry["done"]
        title_y = first.y + 1 + graph.diagram_origin.y - round(graph.scroll_offset.y)
        assert any(
            segment.text == "done"
            and segment.style
            and segment.style.bold
            and segment.style.color == Color.parse(app.theme_variables["primary"]).rich_color
            for segment in graph.render_line(title_y)
        )
        assert node_style("done").color == graph.get_component_rich_style("workflow-node--pending").color

        for state, token in (
            ("running", "primary"),
            ("completed", "success"),
            ("cancelled", "warning"),
            ("retrying", "warning"),
            ("awaiting_retry", "error"),
            ("failed", "error"),
            ("skipped", "primary"),
            ("pending", "primary"),
        ):
            graph.show_nodes({"done": NodeView(state)})
            title = next(segment.style for segment in graph.render_line(title_y) if segment.text == "done")
            assert title is not None and title.bold and not title.reverse
            assert title.color == Color.parse(app.theme_variables[token]).rich_color
            if token != "primary":
                assert title.color == node_style("done").color
