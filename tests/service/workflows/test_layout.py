# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The session-directory predicate every cleanup, listing, and fork site shares."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import chrys.service.workflows.layout as layout
from chrys.service.workflows.layout import HEADER_FILE, WORKFLOWS_DIR, iter_run_dirs, run_dir


def _run(session_dir: Path, name: str) -> Path:
    path = run_dir(session_dir, name)
    path.mkdir(parents=True)
    (path / HEADER_FILE).write_text("{}", encoding="utf-8")
    return path


def test_only_directories_holding_a_header_count(tmp_path: Path) -> None:
    session_dir = tmp_path / "s"
    assert iter_run_dirs(session_dir) == ([], False)
    workflows = session_dir / WORKFLOWS_DIR
    workflows.mkdir(parents=True)
    (workflows / "aborted").mkdir()
    (workflows / ".tmp-run").mkdir()
    (workflows / ".tmp-run" / HEADER_FILE).write_text("{}", encoding="utf-8")
    (workflows / "file").write_text("", encoding="utf-8")
    second = _run(session_dir, "b" * 32)
    first = _run(session_dir, "a" * 32)
    assert iter_run_dirs(session_dir) == ([first, second], False)


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs a privilege on Windows")
def test_a_symlinked_run_directory_is_not_followed(tmp_path: Path) -> None:
    session_dir = tmp_path / "s"
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / HEADER_FILE).write_text("{}", encoding="utf-8")
    (session_dir / WORKFLOWS_DIR).mkdir(parents=True)
    os.symlink(real, session_dir / WORKFLOWS_DIR / "linked", target_is_directory=True)
    assert iter_run_dirs(session_dir) == ([], False)


def test_the_scan_stops_at_the_cap_and_reports_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session_dir = tmp_path / "s"
    for index in range(4):
        _run(session_dir, f"{index:032d}")
    monkeypatch.setattr(layout, "MAX_SCANNED_RUN_DIRS", 2)
    found, truncated = iter_run_dirs(session_dir)
    assert truncated is True
    assert len(found) == 2
