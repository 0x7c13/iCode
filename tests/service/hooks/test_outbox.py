# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the durable outbox."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from chrys.service.hooks.outbox import DONE, FAILED, PENDING, Outbox
from tests.support.windows_replace import briefly_locked_rename


def test_creates_subdirs(tmp_path: Path) -> None:
    Outbox(tmp_path / "outbox")
    assert (tmp_path / "outbox" / PENDING).is_dir()
    assert (tmp_path / "outbox" / DONE).is_dir()
    assert (tmp_path / "outbox" / FAILED).is_dir()


def test_pending_write_then_done(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox")
    job = outbox.write_pending(hook_id="h1", event="after_turn", payload={"x": 1})
    pending_path = tmp_path / "outbox" / PENDING / f"{job.job_id}.json"
    assert pending_path.exists()

    outbox.mark_started(job)
    outbox.mark_done(job, exit_code=0)
    assert not pending_path.exists()
    done_path = tmp_path / "outbox" / DONE / f"{job.job_id}.json"
    assert done_path.exists()
    data = json.loads(done_path.read_text())
    assert data["state"] == "done"
    assert data["exit_code"] == 0
    assert data["retries"] == 1


def test_mark_started_retries_a_briefly_locked_pending_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outbox = Outbox(tmp_path / "outbox")
    job = outbox.write_pending(hook_id="h1", event="after_turn", payload={"x": 1})
    rename = briefly_locked_rename(monkeypatch, failures=2)

    outbox.mark_started(job)

    assert len(rename.attempts) == 3
    pending_path = tmp_path / "outbox" / PENDING / f"{job.job_id}.json"
    assert json.loads(pending_path.read_text(encoding="utf-8"))["retries"] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits are not portable on Windows")
def test_outbox_files_are_owner_only(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox")
    job = outbox.write_pending(hook_id="h1", event="after_turn", payload={"prompt": "secret"})
    pending_path = tmp_path / "outbox" / PENDING / f"{job.job_id}.json"
    assert stat.S_IMODE(os.stat(pending_path).st_mode) == 0o600

    outbox.mark_done(job, exit_code=0)
    done_path = tmp_path / "outbox" / DONE / f"{job.job_id}.json"
    assert stat.S_IMODE(os.stat(done_path).st_mode) == 0o600


def test_pending_then_failed(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox")
    job = outbox.write_pending(hook_id="h", event="session_end", payload={})
    outbox.mark_started(job)
    outbox.mark_failed(job, exit_code=2, error="boom", stderr_tail="x" * 2000)
    failed_path = tmp_path / "outbox" / FAILED / f"{job.job_id}.json"
    data = json.loads(failed_path.read_text())
    assert data["state"] == "failed"
    assert data["exit_code"] == 2
    assert data["error"] == "boom"
    # Tail is capped.
    assert len(data["stderr_tail"]) == 1024


def test_list_stale_pending(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox")
    job = outbox.write_pending(hook_id="h", event="session_end", payload={})

    # Fresh: created_at is now, so retry_age_seconds=60 should NOT include it.
    fresh = outbox.list_stale_pending(retry_age_seconds=60.0)
    assert fresh == []

    # Backdate created_at.
    path = tmp_path / "outbox" / PENDING / f"{job.job_id}.json"
    data = json.loads(path.read_text())
    data["created_at"] = "2000-01-01T00:00:00Z"
    path.write_text(json.dumps(data))

    stale = outbox.list_stale_pending(retry_age_seconds=60.0)
    assert len(stale) == 1
    assert stale[0].job_id == job.job_id


def test_unknown_fields_round_trip_through_extra(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox")
    job = outbox.write_pending(hook_id="h", event="session_end", payload={})
    path = tmp_path / "outbox" / PENDING / f"{job.job_id}.json"
    data = json.loads(path.read_text())
    data["future_field"] = "from a newer chrys"
    path.write_text(json.dumps(data))

    stale = outbox.list_stale_pending(retry_age_seconds=0.0)
    assert stale[0].extra == {"future_field": "from a newer chrys"}


def test_force_fail(tmp_path: Path) -> None:
    outbox = Outbox(tmp_path / "outbox")
    job = outbox.write_pending(hook_id="h", event="session_end", payload={})
    outbox.force_fail(job, reason="exceeded retries")
    failed_path = tmp_path / "outbox" / FAILED / f"{job.job_id}.json"
    assert failed_path.exists()
    data = json.loads(failed_path.read_text())
    assert data["error"] == "exceeded retries"
