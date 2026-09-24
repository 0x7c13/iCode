# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Archive atomicity, bounded reads and shared compressed-history occurrence identity."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping
from pathlib import Path
from threading import Event
from typing import Any
from unittest.mock import create_autospec

import pytest

import chrys.orchestration.workflows.agent_archive as archive_module
from chrys.foundation.trajectory.metadata import ensure_analytics_item_id
from chrys.kernel import Message
from chrys.orchestration.workflows.agent_archive import AgentNodeArchive
from chrys.service.context.providers.history import CompressedBlock
from chrys.service.session.sub_agent_logs import SubAgentLogStats
from chrys.service.workflows import transcript as transcript_module
from chrys.service.workflows.store import (
    NODE_RECORD_SESSION,
    RunSpec,
    WorkflowRunStore,
    WorkflowStorageFailed,
    node_value_path,
)
from chrys.service.workflows.transcript import read_node_transcript
from tests.service.workflows.test_store import header
from tests.support.waiting import wait_for


async def test_compressed_archive_deduplicates_identity_not_equal_content(tmp_path: Path) -> None:
    store = WorkflowRunStore.open(
        spec=RunSpec(manifest={}, environment={}, resolved_nodes=()),
        input_text="go",
        run_dir=tmp_path / "run",
        header=header(),
        source=b"",
    )
    archive = AgentNodeArchive(store, activation_id="a@iter#1", invocation_id="inv", profile_name="QA")
    original = Message("assistant", ["Repeated output"], message_id="msg_1")
    repeated = Message("assistant", ["Repeated output"], message_id="msg_1")
    for message in (original, repeated):
        ensure_analytics_item_id(message.additional_properties)
    state = {
        "messages": [original, repeated],
        "compressed_msgs": [
            CompressedBlock(compressed_context_id="block", messages=[copy.deepcopy(original)], summary_text="summary")
        ],
    }
    try:
        await archive.write(
            attempt=1, status="completed", error="", state=state, acp_state=None, stats=SubAgentLogStats()
        )
    finally:
        await store.close()
    loaded = read_node_transcript(store.run_dir, "a@iter#1", 1)
    assert loaded is not None
    assert len(loaded.replay.messages) == 2
    assert [message["contents"][0]["text"] for message in loaded.replay.messages] == ["Repeated output"] * 2


async def test_cancellation_drains_atomic_write_before_releasing_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = WorkflowRunStore.open(
        spec=RunSpec(manifest={}, environment={}, resolved_nodes=()),
        input_text="go",
        run_dir=tmp_path / "run",
        header=header(),
        source=b"",
    )
    archive = AgentNodeArchive(store, activation_id="a@iter#1", invocation_id="inv", profile_name="QA")
    started, release = Event(), Event()
    write = store.write_node_value

    def delayed(activation_id: str, attempt: int, kind: str, payload: Mapping[str, Any]) -> Path:
        started.set()
        assert release.wait(timeout=10)
        return write(activation_id, attempt, kind, payload)

    monkeypatch.setattr(store, "write_node_value", create_autospec(write, side_effect=delayed))
    task = asyncio.create_task(
        archive.write(
            attempt=1,
            status="cancelled",
            error="",
            state={"messages": [Message("user", ["input"])]},
            acp_state=None,
            stats=SubAgentLogStats(),
        )
    )
    try:
        await wait_for(lambda: started.is_set() or task.done())
        if task.done():
            await task
        assert started.is_set()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        loaded = read_node_transcript(store.run_dir, "a@iter#1", 1)
        assert loaded is not None and loaded.status == "cancelled"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()


@pytest.mark.parametrize("invalid", ["identity", "oversized", "json"])
async def test_invalid_archive_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str) -> None:
    store = WorkflowRunStore.open(
        spec=RunSpec(manifest={}, environment={}, resolved_nodes=()),
        input_text="go",
        run_dir=tmp_path / "run",
        header=header(),
        source=b"",
    )
    archive = AgentNodeArchive(store, activation_id="a@iter#1", invocation_id="inv", profile_name="QA")
    try:
        await archive.write(attempt=1, status="completed", error="", state={}, acp_state=None, stats=SubAgentLogStats())
    finally:
        await store.close()
    path = node_value_path(store.run_dir, "a@iter#1", 1, NODE_RECORD_SESSION)
    if invalid == "oversized":
        monkeypatch.setattr(transcript_module, "MAX_SUB_AGENT_AUDIT_BYTES", 32)
    elif invalid == "json":
        path.write_bytes(b"{")
    else:
        path.write_text(path.read_text().replace('"attempt": 1', '"attempt": 2'))
    with pytest.raises(ValueError):
        read_node_transcript(store.run_dir, "a@iter#1", 1)


async def test_running_checkpoint_failure_can_be_repaired_by_terminal_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = WorkflowRunStore.open(
        spec=RunSpec(manifest={}, environment={}, resolved_nodes=()),
        input_text="go",
        run_dir=tmp_path / "run",
        header=header(),
        source=b"",
    )
    archive = AgentNodeArchive(store, activation_id="a@iter#1", invocation_id="inv", profile_name="QA")
    write = store.write_node_value
    fail_checkpoint = True

    def faulty_write(activation_id: str, attempt: int, kind: str, payload: Mapping[str, Any]) -> Path:
        if fail_checkpoint:
            raise WorkflowStorageFailed("temporary archive failure")
        return write(activation_id, attempt, kind, payload)

    monkeypatch.setattr(store, "write_node_value", create_autospec(write, side_effect=faulty_write))
    try:
        await archive.write(attempt=1, status="running", error="", state={}, acp_state=None, stats=SubAgentLogStats())
        fail_checkpoint = False
        await archive.write(attempt=1, status="completed", error="", state={}, acp_state=None, stats=SubAgentLogStats())
        loaded = read_node_transcript(store.run_dir, "a@iter#1", 1)
        assert loaded is not None and loaded.status == "completed"
    finally:
        await store.close()


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
async def test_terminal_serialization_failure_is_a_storage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    store = WorkflowRunStore.open(
        spec=RunSpec(manifest={}, environment={}, resolved_nodes=()),
        input_text="go",
        run_dir=tmp_path / "run",
        header=header(),
        source=b"",
    )
    archive = AgentNodeArchive(store, activation_id="a@iter#1", invocation_id="inv", profile_name="QA")
    failure = TypeError("cannot serialize transcript")
    monkeypatch.setattr(
        archive_module,
        "serialize_state",
        create_autospec(archive_module.serialize_state, side_effect=failure),
    )
    try:
        with pytest.raises(WorkflowStorageFailed, match=r"a@iter#1 attempt 1.*cannot serialize transcript") as caught:
            await archive.write(attempt=1, status=status, error="", state={}, acp_state=None, stats=SubAgentLogStats())
        assert caught.value.__cause__ is failure
    finally:
        await store.close()


async def test_checkpoint_coalesces_updates_and_terminal_close_drains_writer() -> None:
    from chrys.orchestration.workflows.agent_archive import CoalescedCheckpoint

    entered, release = asyncio.Event(), asyncio.Event()
    snapshots = []
    value = 0

    async def write():
        snapshots.append(value)
        entered.set()
        await release.wait()

    checkpoint = CoalescedCheckpoint(write, interval=0.01)
    close = None
    try:
        for index in range(100):
            value = index
            checkpoint.changed()
        await wait_for(entered.is_set)
        assert snapshots == [99]
        value = 100
        checkpoint.changed()
        close = asyncio.create_task(checkpoint.close())
        await wait_for(checkpoint._closed.is_set)
        assert not close.done()
    finally:
        release.set()
        await checkpoint.close()
        if close is not None:
            await close
    # Closing discarded the timer but drained the old writer. The caller's
    # terminal snapshot now takes precedence, including the last dirty update.
    snapshots.append(value)
    checkpoint.changed()
    assert snapshots == [99, 100]
