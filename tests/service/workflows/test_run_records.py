# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Immutable run artifacts, complete replay and bounded owner-verified readers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import WorkflowOutputSummary, WorkflowRunFinished
from chrys.foundation.events.workflow import WORKFLOW_RUN_EVENTS
from chrys.foundation.platform.files import atomic_write_basename_budget
from chrys.service.workflows import store as store_module
from chrys.service.workflows.journal import WorkflowJournal
from chrys.service.workflows.layout import OUTPUT_INDEX_FILE
from chrys.service.workflows.orphans import read_run_terminal, reconcile_orphaned_runs
from chrys.service.workflows.outcomes import RunOutcome
from chrys.service.workflows.records import decode_run_event, read_run_started
from chrys.service.workflows.scheduler import AttemptRef
from chrys.service.workflows.store import (
    INPUT_EXCERPT_CHARS,
    MAX_HEADER_BYTES,
    RunRecord,
    RunSpec,
    WorkflowRunStore,
    node_emits_path,
    node_value_path,
    read_node_emits,
    read_node_value,
    read_run_events,
    read_run_header,
    read_run_spec,
)
from tests.service.workflows.test_store import header
from tests.support.event_capture import capture_event_sequence
from tests.support.secure_files import plant_owner_only_bytes


async def test_large_input_and_spec_leave_a_small_write_once_listing_header(tmp_path: Path) -> None:
    full_input = "\x00汉字" * 200_000
    definition = {"nodes": [{"id": f"n{i}", "kind": "python"} for i in range(3000)]}
    identity = replace(header(), input_excerpt=full_input)
    directory = tmp_path / "workflows" / identity.run_id
    store = WorkflowRunStore.open(
        directory, header=identity, spec=RunSpec(definition, {}), input_text=full_input, source=b""
    )
    before = (directory / "run.json").read_bytes()
    try:
        assert len(before) < MAX_HEADER_BYTES
        assert set(read_run_header(directory)).isdisjoint({"manifest", "environment", "resolved_nodes", "input_text"})
        assert read_run_header(directory)["input_excerpt"] == full_input[:INPUT_EXCERPT_CHARS]
        assert read_run_spec(directory)["manifest"] == definition
        assert read_run_started(directory).input_text == full_input
        await store.finish("completed", {"outputs": []})
    finally:
        await store.close()
    assert reconcile_orphaned_runs(tmp_path).already_finished == (directory.name,)
    assert (directory / "run.json").read_bytes() == before
    assert not (directory / ".reconcile.lock").exists()
    assert read_run_terminal(directory).finished_at


async def test_codec_preserves_attempt_iteration_notice_question_and_answer_identity(tmp_path: Path) -> None:
    identity = header()
    store = WorkflowRunStore.open(
        tmp_path, header=identity, spec=RunSpec({"nodes": []}, {}), input_text="go", source=b""
    )
    bus = EventBus()
    journal = WorkflowJournal(store, bus, session_id=identity.session_id)
    ref = AttemptRef(identity.run_id, "loop-node", "opaque-activation", 3)
    try:
        async with capture_event_sequence(bus, *WORKFLOW_RUN_EVENTS) as live:
            await journal.run_started()
            await journal.node_state(ref, "running", iteration=5, failure_phase="until")
            await journal.node_output(ref, "emit", 7, "partial")
            await journal.node_ask(ref, "question", "Continue?")
            await journal.node_answer(ref, "question", "Yes")
            await journal.loop_iteration(ref, 5, "exit")
            await journal.run_notice(ref.node_id, ref.activation_id, "notice", "Visible")
            await journal.finish(RunOutcome.NODE_FAILED, node_id=ref.node_id, error="failure", reason="reason")
        replay = [decode_run_event(record, directory=tmp_path) for record in read_run_events(tmp_path).events]
        replay = [event for event in replay if event is not None]
        assert len(replay) == len(live)
        for observed, restored in zip(live, replay, strict=True):
            assert replace(restored, timestamp=observed.timestamp, event_id=observed.event_id) == observed
    finally:
        await store.close()


@pytest.mark.parametrize("damage", ["", "missing", "count", "attempt"])
async def test_omitted_outputs_replay_from_exact_output_index(tmp_path: Path, damage: str) -> None:
    store = WorkflowRunStore.open(tmp_path, header=header(), spec=RunSpec({"nodes": []}, {}), input_text="", source=b"")
    journal = WorkflowJournal(store, None, session_id=store.header.session_id)
    outputs = [WorkflowOutputSummary(f"n{i}", f"activation-{i}", attempt=2 + i % 3) for i in range(100)]
    try:
        await journal.finish(RunOutcome.COMPLETED, outputs=outputs)
    finally:
        await store.close()
    terminal = next(
        record for record in read_run_events(tmp_path).events if record.event_type == RunRecord.RUN_FINISHED
    )
    assert terminal.payload["outputs"] == [] and terminal.payload["outputs_omitted"] == 100
    assert len(json.dumps(dict(terminal.payload)).encode()) < 3072
    path = tmp_path / OUTPUT_INDEX_FILE
    if damage == "missing":
        path.unlink()
    elif damage:
        payload = json.loads(path.read_bytes())
        if damage == "count":
            payload["outputs"].pop()
        else:
            payload["outputs"][0]["attempt"] = "invalid"
        path.write_text(json.dumps(payload))
    if damage:
        with pytest.raises(ValueError, match=r"output (index|identity)"):
            decode_run_event(terminal, directory=tmp_path)
    else:
        restored = decode_run_event(terminal, directory=tmp_path)
        assert isinstance(restored, WorkflowRunFinished)
        assert restored.outputs == outputs


@pytest.mark.parametrize("kind", ["python", "join", "agent"])
def test_node_record_readers_bound_serialized_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    for constant in ("MAX_PYTHON_VALUE_BYTES", "MAX_JOIN_VALUE_BYTES", "MAX_AGENT_VALUE_BYTES"):
        monkeypatch.setattr(store_module, constant, 128)
    (tmp_path / "nodes").mkdir()
    # Planted as the product writes them: a plain write is owned by the Administrators group on Windows.
    plant_owner_only_bytes(node_value_path(tmp_path, "node", 1, "output"), b"x" * 100_001)
    with pytest.raises(ValueError, match="ceiling"):
        read_node_value(tmp_path, "node", 1, "output", node_kind=kind)
    monkeypatch.setattr(store_module, "MAX_EMITS_BYTES", 64)
    plant_owner_only_bytes(node_emits_path(tmp_path, "node", 1), b"x" * 65)
    with pytest.raises(ValueError, match="ceiling"):
        read_node_emits(tmp_path, "node", 1)


def test_node_readers_refuse_links_and_reserve_the_atomic_basename_budget(tmp_path: Path) -> None:
    (tmp_path / "nodes").mkdir()
    foreign = tmp_path / "foreign"
    foreign.write_text("{}")
    path = node_value_path(tmp_path, "node", 1, "output")
    try:
        path.symlink_to(foreign)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    with pytest.raises(OSError):
        read_node_value(tmp_path, "node", 1, "output")
    emit = node_emits_path(tmp_path, "node", 1)
    emit.symlink_to(foreign)
    with pytest.raises(OSError):
        read_node_emits(tmp_path, "node", 1)
    long_path = node_value_path(tmp_path, "汉" * 230, 1, "output")
    assert len(long_path.name.encode()) <= atomic_write_basename_budget()
