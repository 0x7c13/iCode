# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow settings cannot cross unfinished restore or resource-close boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from chrys.service.state.store import JsonFileStateStore
from chrys.service.state.workflow import WorkflowSessionState
from tests.orchestration.workflows._hosting import make_host
from tests.support.workflow_history import workflow_state


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("identity", None, "identity must be an object"),
    ],
)
async def test_explicit_restore_preserves_workflow_validation_error(
    tmp_path: Path, field: str, value: object, reason: str
) -> None:
    host = make_host(tmp_path, project=tmp_path)
    store = JsonFileStateStore(tmp_path / "sessions")
    session_id = str(uuid4())
    await store.save_workflow_session(
        session_id, WorkflowSessionState.decode(workflow_state(tmp_path, run_count=1, latest_run_id="run-1"))
    )
    path = store.session_dir(session_id) / "session.json"
    envelope = json.loads(path.read_text())
    envelope["state"][field] = value
    path.write_text(json.dumps(envelope))
    try:
        with pytest.raises(ValueError, match=reason):
            await host.load_workflow_session(session_id)
        assert await store.list_sessions(kind="workflow") == []
    finally:
        await host.shutdown()


async def test_explicit_restore_distinguishes_missing_wrong_kind_and_unreadable(tmp_path: Path) -> None:
    host = make_host(tmp_path, project=tmp_path)
    store = JsonFileStateStore(tmp_path / "sessions")
    session_id = str(uuid4())
    try:
        with pytest.raises(ValueError, match="No workflow session"):
            await host.load_workflow_session(session_id)
        await store.save_session(session_id, {"messages": []})
        with pytest.raises(ValueError, match="Chat session"):
            await host.load_workflow_session(session_id)
        directory = store.session_dir(session_id)
        for path in directory.glob("session.json*"):
            path.write_text("{")
        with pytest.raises(ValueError, match="no readable checkpoint"):
            await host.load_workflow_session(session_id)
    finally:
        await host.shutdown()


async def test_restore_matches_complete_identity_and_accepts_normalized_short_ids(tmp_path: Path) -> None:
    host = make_host(tmp_path, project=tmp_path)
    store = JsonFileStateStore(tmp_path / "sessions")
    session_id = str(uuid4())
    normalized = session_id.replace("-", "")
    await store.save_workflow_session(
        session_id, WorkflowSessionState.decode(workflow_state(tmp_path, run_count=1, latest_run_id="run-1"))
    )
    try:
        for requested in (session_id, normalized, f" {session_id.upper()} ", normalized[:12].upper()):
            await host.load_workflow_session(requested)
            assert host.workflow_session_id == session_id
        target = host.workflow_target("archived")
        wrong_id = normalized[:-1] + ("0" if normalized[-1] != "0" else "1")
        with pytest.raises(ValueError, match="No workflow session"):
            await host.load_workflow_session(wrong_id)
        assert host.workflow_target("archived") == target
    finally:
        await host.shutdown()
