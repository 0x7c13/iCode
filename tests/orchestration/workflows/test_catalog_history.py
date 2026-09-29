# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow history readers share the log-owned terminal without rewriting artifacts."""

from __future__ import annotations

import errno
from pathlib import Path
from uuid import uuid4

import pytest

from chrys.service.workflows import history as history_module
from chrys.service.workflows.artifacts import session_runs
from chrys.service.workflows.history import read_workflow_meta, read_workflow_run
from chrys.service.workflows.layout import HEADER_FILE
from chrys.service.workflows.store import read_run_header
from tests.support.secure_files import plant_owner_only_bytes
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


def _raise(error: Exception):
    def reader(_directory: Path) -> object:
        raise error

    return reader


@pytest.mark.parametrize(
    ("fault", "listed", "settled"),
    [
        ("none", True, True),
        ("header missing", False, True),
        ("header corrupt", False, True),
        ("header I/O error", False, False),
        ("terminal corrupt", True, True),
        ("terminal I/O error", True, False),
    ],
)
async def test_a_run_read_is_settled_unless_an_io_error_shaped_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, listed: bool, settled: bool
) -> None:
    """The session browser caches settled reads on the run files' signatures; an I/O error may clear."""
    directory = tmp_path / "session" / "workflows" / uuid4().hex
    await record_workflow_run(directory, session_id=str(uuid4()), title="Review", outcome="completed")
    if fault == "header missing":
        (directory / HEADER_FILE).unlink()
    elif fault == "header corrupt":
        (directory / HEADER_FILE).unlink()
        plant_owner_only_bytes(directory / HEADER_FILE, b"{")
    elif fault == "header I/O error":
        monkeypatch.setattr(history_module, "read_run_header", _raise(OSError(errno.EIO, "I/O error")))
    elif fault == "terminal corrupt":
        monkeypatch.setattr(history_module, "read_run_terminal", _raise(ValueError("tail exceeds the ceiling")))
    elif fault == "terminal I/O error":
        monkeypatch.setattr(history_module, "read_run_terminal", _raise(OSError(errno.EIO, "I/O error")))

    read = read_workflow_run(directory, active=False)

    assert (read.meta is not None, read.settled) == (listed, settled)
