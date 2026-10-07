# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow status messages end on their last fact, with no blank row before the outputs."""

from __future__ import annotations

from collections import deque

import pytest
from rich.style import Style
from textual.app import App, ComposeResult
from textual.widgets import Static

from chrys.app.tui.widgets.workflow.output import (
    WorkflowOutputText,
    WorkflowOutputView,
    WorkflowStatusOutput,
    status_output,
)
from chrys.app.tui.widgets.workflow.projector import ObservedRun
from chrys.foundation.events import types as events
from tests.support.waiting import wait_for


def _finished_run() -> ObservedRun:
    manifest = {"nodes": [{"id": "fn", "kind": "python", "callable": {"name": "fn"}}], "edges": []}
    run = ObservedRun(
        events.WorkflowRunStarted(run_id="run", title="Example", manifest=manifest),
        facts=deque(
            [
                events.WorkflowNodeStateChanged(run_id="run", node_id="fn", activation_id="fn", state="completed"),
                events.WorkflowRunFinished(run_id="run", outcome="failed", error="Traceback line\n"),
            ]
        ),
    )
    run.notices["notice"] = events.WorkflowRunNotice(run_id="run", code="notice", message="Check configuration")
    return run


@pytest.mark.parametrize("finished", [True, False], ids=["run", "idle"])
async def test_status_messages_sit_one_row_from_their_neighbors(finished: bool) -> None:
    run = _finished_run() if finished else None
    last_line = "Traceback line" if finished else "idle"

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield WorkflowOutputView(None)

        def on_mount(self) -> None:
            view = self.query_one(WorkflowOutputView)
            view.show_iterations({"loop": (1, 3)})
            view.query_one(WorkflowStatusOutput).show_run(run)
            view.show_outputs((WorkflowOutputText("fn", "hello"),))

    app = Harness()
    async with app.run_test(size=(100, 30)) as pilot:
        iterations = app.query_one("#workflow-iterations", Static)
        status = app.query_one(WorkflowStatusOutput)
        outputs = app.query_one("#workflow-outputs", Static)
        await wait_for(lambda: str(status.content).endswith(last_line), pilot=pilot)
        rows = len(str(status.content).splitlines())

        # Every rendered row is a line of text: no trailing blank row that selection can land on.
        await wait_for(lambda: status.region.height == rows, pilot=pilot)
        assert status.region.y == iterations.region.bottom + 1
        assert outputs.region.y == status.region.bottom + 1


def test_the_run_title_is_shown_without_terminal_controls() -> None:
    started = events.WorkflowRunStarted(
        run_id="run", title="Review\x1b[2JInjected", manifest={"nodes": [], "edges": []}
    )

    shown = status_output(ObservedRun(started), None, warning=Style(), error=Style()).plain

    assert "Review\ufffd[2JInjected" in shown and "\x1b" not in shown
