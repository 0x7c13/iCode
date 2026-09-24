# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Run identities, historical diffs and file rollback never masquerade as Chat turns."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chrys.foundation.models.mutation_scope import ChatTurnScope, WorkflowRunScope
from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import MutationOp, MutationSource, RunMutations


def _write(tracker: MutationTracker, path: Path, text: str) -> None:
    record = tracker.record(str(path), MutationOp.MODIFY, MutationSource.WRITE_FILE, "write")
    assert record is not None
    path.write_text(text)
    tracker.record_after(record)


def test_run_queries_and_file_rollback_use_real_ids(tmp_path: Path) -> None:
    file = tmp_path / "file.txt"
    file.write_text("before")
    tracker = MutationTracker(SnapshotStore(tmp_path / "session"))
    tracker.start_workflow_run("first")
    _write(tracker, file, "attempt")
    tracker.reset_file_cache()
    _write(tracker, file, "first result")
    tracker.start_workflow_run("empty")
    tracker.start_workflow_run("last")
    _write(tracker, file, "last result")

    restored = MutationTracker.deserialize(tracker.serialize(), tracker.store)
    assert [period.scope for period in restored.get_all_periods()] == [
        WorkflowRunScope("first"),
        WorkflowRunScope("empty"),
        WorkflowRunScope("last"),
    ]
    first = restored.get_period_file_summary(WorkflowRunScope("first"))[str(file)]
    assert restored.store.read_blob(first.before) == b"before"
    assert restored.store.read_blob(first.after) == b"first result"
    assert restored.get_period_file_summary(WorkflowRunScope("empty")) == {}
    plan = restored.get_rollback_plan_for_periods({WorkflowRunScope("empty"), WorkflowRunScope("last")})
    assert plan.paths == {str(file)}
    before = restored.serialize()
    results = restored.restore_files(plan)
    assert len(results) == 1 and results[0].ok
    assert file.read_text() == "first result"
    assert restored.serialize() == before  # File restoration does not erase execution evidence.


def test_run_ledger_has_no_synthetic_turn_fields(tmp_path: Path) -> None:
    file = tmp_path / "file.txt"
    file.write_text("before")
    tracker = MutationTracker(SnapshotStore(tmp_path / "session"))
    tracker.start_workflow_run("run-id")
    _write(tracker, file, "after")
    encoded = tracker.serialize()
    assert "runs" in encoded and "turns" not in encoded
    assert "turn_id" not in json.dumps(encoded)
    period = tracker.current_period
    assert isinstance(period, RunMutations) and period.run_id == "run-id"
    snapshot = tracker.get_period_snapshot(str(file), period.scope)
    assert snapshot is not None and snapshot.period_index == period.period_index
    for scope in [WorkflowRunScope("1"), WorkflowRunScope("missing"), ChatTurnScope(1)]:
        with pytest.raises(KeyError):
            tracker.get_rollback_plan_for_periods({scope})
    with pytest.raises(ValueError, match="already exists"):
        tracker.start_workflow_run("run-id")
    with pytest.raises(TypeError):
        tracker.get_all_turns()
    with pytest.raises(TypeError):
        tracker.rollback_turns({1})
    with pytest.raises(TypeError):
        tracker.start_turn(2)
    assert file.read_text() == "after"


def test_legacy_chat_snapshots_still_load(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    file = tmp_path / "file.txt"
    file.write_text("before")
    snapshot = store.save(str(file), 7).to_dict()
    snapshot["turn_id"] = snapshot.pop("period_index")
    tracker = MutationTracker.deserialize(
        {"turns": [{"turn_id": 7, "mutations": []}], "snapshots": {f"{file}::7": snapshot}}, store
    )
    assert tracker.get_all_turns()[0].turn_id == 7
    assert tracker.get_snapshot(str(file), 7) is not None
    with pytest.raises(ValueError, match="Workflow mutation ledger"):
        tracker.start_workflow_run("run-id")


@pytest.mark.parametrize("runs", [[("one", 1), ("one", 2)], [("one", 1), ("two", 1)], [("one", 2), ("two", 1)]])
def test_run_ledger_rejects_ambiguous_identity(runs: list[tuple[str, int]], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Duplicate or unordered"):
        MutationTracker.deserialize(
            {"runs": [{"run_id": run, "period_index": index} for run, index in runs]}, SnapshotStore(tmp_path)
        )


@pytest.mark.parametrize("created_in_run", [False, True])
@pytest.mark.parametrize("move_back", [False, True])
def test_run_summary_accounts_for_both_move_endpoints(tmp_path: Path, created_in_run: bool, move_back: bool) -> None:
    old, new = tmp_path / "old.txt", tmp_path / "new.txt"
    if not created_in_run:
        old.write_text("contents")
    tracker = MutationTracker(SnapshotStore(tmp_path / "session"))
    tracker.start_workflow_run("rename")
    if created_in_run:
        _write(tracker, old, "contents")
    moves = [(old, new), (new, old)] if move_back else [(old, new)]
    for source, destination in moves:
        tracker.pre_snapshot([str(source), str(destination)])
        source.rename(destination)
        tracker.record(str(destination), MutationOp.MOVE, MutationSource.SHELL, "mv", old_path=str(source))
    summary = tracker.get_period_file_summary(WorkflowRunScope("rename"))
    assert set(summary) == {str(old), str(new)}
    assert (summary[str(old)].before is not None) == (not created_in_run)
    assert (summary[str(old)].after is not None) == move_back
    assert summary[str(new)].before is None
    assert (summary[str(new)].after is not None) == (not move_back)
    assert tracker.get_session_file_summary() == {path: diff for path, diff in summary.items() if not diff.is_net_zero}
