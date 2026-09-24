# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow checkpoints and Chat-only operations enforce the storage boundary."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from chrys.foundation.models.workflow_session import WorkingDirSnapshot, WorkspaceSnapshot
from chrys.service.session.runtime_metadata import SessionUsageMetadata
from chrys.service.state.store import JsonFileStateStore, SessionForkError
from chrys.service.state.workflow import WorkflowSessionState
from tests.support.workflow_history import workflow_state


async def test_workflow_checkpoint_contains_only_its_state_and_shared_metadata(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path / "sessions")
    identity = str(uuid4())
    state = WorkflowSessionState.decode(workflow_state(tmp_path))
    state.workspace = WorkspaceSnapshot(str(tmp_path), (WorkingDirSnapshot(str(tmp_path / "shared"), "shared"),))
    state.runtime = SessionUsageMetadata()
    state.runtime.accumulate_invocation_usage(21, input_tokens=13, output_tokens=8, cache_hit_tokens=4)
    checkpoint = await store.save_workflow_session(identity, state, title="Review")
    assert checkpoint is not None
    envelope = json.loads((store.session_dir(identity) / "session.json").read_text(encoding="utf-8"))
    assert not set(envelope["meta"]) & {
        "agent_profile",
        "model_profile_id",
        "model_provider",
        "message_count",
        "service_session_id",
        "parent_session_id",
    }
    assert not set(envelope["state"]) & {
        "messages",
        "compressed_msgs",
        "turn_counter",
        "last_usage",
        "context_calibration",
    }
    assert envelope["meta"]["primary_cwd"] == state.workspace.primary_cwd
    assert envelope["meta"]["working_dirs"] == [str(tmp_path / "shared")]
    loaded = await store.load_workflow_session(identity)
    assert loaded == state
    await store.update_session_titles(identity, custom_title="My review")
    state.run_count, state.latest_run_id = 1, uuid4().hex
    await store.save_workflow_session(identity, state)
    meta = await store.load_session_meta(identity)
    assert meta is not None and meta.custom_title == "My review" and meta.title == "Review"


async def test_fork_rejects_workflow_before_creating_a_destination(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path / "sessions")
    identity = str(uuid4())
    await store.save_workflow_session(identity, WorkflowSessionState.decode(workflow_state(tmp_path)))
    directory = store.session_dir(identity).parent
    before = set(directory.iterdir())
    with pytest.raises(SessionForkError, match="Only Chat sessions"):
        store.fork_session(identity)
    assert set(directory.iterdir()) == before


async def test_typed_workflow_checkpoint_refuses_chat_identity(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path / "sessions")
    identity = str(uuid4())
    await store.save_session(identity, {"messages": []})
    before = (store.session_dir(identity) / "session.json").read_bytes()
    with pytest.raises(ValueError, match="cannot change"):
        await store.save_workflow_session(identity, WorkflowSessionState.decode(workflow_state(tmp_path)))
    assert (store.session_dir(identity) / "session.json").read_bytes() == before
