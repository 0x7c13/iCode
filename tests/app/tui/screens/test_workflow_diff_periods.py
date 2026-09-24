# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Persisted Workflow diffs retain Run IDs through rendering."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from textual.app import App
from textual.widgets import Tab

from chrys.app.tui.screens.diff.screen import DiffScreen, load_diff_entries_by_period
from chrys.foundation.models.mutation_scope import WorkflowRunScope
from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import MutationOp, MutationSource
from chrys.service.state.store import JsonFileStateStore
from chrys.service.state.workflow import WorkflowSessionState
from tests.support.waiting import wait_for
from tests.support.workflow_history import workflow_state


async def test_persisted_workflow_diff_renders_run_tabs_with_stable_targets(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path / "sessions")
    session_id = str(uuid4())
    tracker = MutationTracker(SnapshotStore(store.session_dir(session_id)))
    file = tmp_path / "review.txt"
    file.write_text("before")
    for run_id, text in [("stable-first", "first"), ("stable-last", "last")]:
        tracker.start_workflow_run(run_id)
        mutation = tracker.record(str(file), MutationOp.MODIFY, MutationSource.WRITE_FILE, "write")
        assert mutation is not None
        file.write_text(text)
        tracker.record_after(mutation)
    state = WorkflowSessionState.decode(workflow_state(tmp_path, run_count=2, latest_run_id="stable-last"))
    state.mutations = tracker.serialize()
    await store.save_workflow_session(session_id, state)
    result = load_diff_entries_by_period(store, session_id, str(tmp_path))
    assert result.scopes == {1: WorkflowRunScope("stable-first"), 2: WorkflowRunScope("stable-last")}
    assert result.total_periods == 2
    assert [(entry.before_text, entry.after_text) for entry in result.per_period_entries[1]] == [("before", "first")]
    assert [(entry.before_text, entry.after_text) for entry in result.per_period_entries[2]] == [("first", "last")]

    class DiffApp(App):
        def on_mount(self) -> None:
            self.push_screen(
                DiffScreen(
                    result.per_period_entries, cwd=str(tmp_path), all_entries=result.all_entries, scopes=result.scopes
                )
            )

    async with DiffApp().run_test() as pilot:
        screen = pilot.app.screen
        assert isinstance(screen, DiffScreen)
        await wait_for(lambda: screen._content_ready, pilot=pilot, description="workflow diff to be rendered")
        assert screen.query_one("#--content-tab-turn-1", Tab).label.plain == "Run 1"
        assert screen.query_one("#--content-tab-turn-2", Tab).label.plain == "Run 2"
