# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow-only recordings remain readable across the trajectory dashboard's tabs."""

from __future__ import annotations

from pathlib import Path

from chrys.app.tui.widgets.trajectory import DashboardTab
from chrys.app.tui.widgets.trajectory.text_view import TrajectoryTextView
from chrys.app.tui.widgets.trajectory.timeline import _family_category
from tests.app.tui.widgets._trajectory_fixtures import _NS, open_dashboard, page_text, plain_look
from tests.service.analytics._events import EventLog


async def test_dashboard_accepts_workflow_run_and_node_events_without_a_chat_turn(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    log.span(
        "workflow.run",
        "a" * 32,
        0,
        4 * _NS,
        turn_id=None,
        start_payload={"workflow": "demo-workflow", "run_id": "a" * 32},
        finish_payload={"outcome": "completed"},
    )
    for attempt in (1, 2):
        log.span(
            "workflow.node",
            str(attempt) * 32,
            attempt * _NS,
            (attempt + 1) * _NS,
            turn_id=None,
            parent_operation_id="a" * 32,
            start_payload={"node": "research", "activation": "research@iter#1", "attempt": attempt, "kind": "agent"},
            finish_payload={"outcome": "failed" if attempt == 1 else "completed"},
        )
    log.write(path)
    async with open_dashboard(path, size=(150, 40)) as (dashboard, pilot):
        analysis = dashboard._analysis
        assert analysis is not None
        assert not analysis.diagnostics.unsupported_event_count
        assert not analysis.diagnostics.corrupt_lines
        assert not analysis.turns
        assert _family_category(plain_look(), "workflow.run") == "Workflow"
        assert _family_category(plain_look(), "workflow.node") == "Workflow"
        for tab in (DashboardTab.TIMELINE, DashboardTab.INSIGHTS, DashboardTab.OVERVIEW):
            await pilot.click(f"#{tab}")
            assert page_text(dashboard.query_one(TrajectoryTextView))
