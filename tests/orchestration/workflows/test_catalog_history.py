# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow history readers share the log-owned terminal without rewriting artifacts."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from chrys.service.workflows.artifacts import session_runs
from chrys.service.workflows.history import read_workflow_meta
from chrys.service.workflows.store import read_run_header
from tests.support.workflow_history import record_workflow_run


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize(
    "outcome,status", [("completed", "completed"), ("cancelled", "cancelled"), ("storage_failed", "failed")]
)
async def test_catalog_and_session_browser_resolve_the_same_durable_terminal(
    tmp_path: Path, active: bool, outcome: str, status: str
) -> None:
    session_id, run_id = str(uuid4()), uuid4().hex
    directory = tmp_path / "session" / "workflows" / run_id
    await record_workflow_run(directory, session_id=session_id, title="Review", outcome=outcome)
    original = read_run_header(directory)
    before = {path: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    (record,) = session_runs(directory.parent.parent)
    assert record.terminal is not None
    assert record.terminal.outcome == outcome
    assert record.terminal.last_seq == 2
    assert record.terminal.finished_at
    meta = read_workflow_meta(directory, active=active)
    assert meta is not None and meta.status == status
    assert read_run_header(directory) == original
    assert {path: path.read_bytes() for path in before} == before
