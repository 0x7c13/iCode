# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for session environment capture."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.models.workspace import WorkingDir, Workspace
from chrys.foundation.platform import get_platform


def _surrogateescaped_dir(tmp_path: Path, name: bytes) -> tuple[bytes, str]:
    if not get_platform().is_linux:
        pytest.skip("invalid-UTF-8 filesystem paths are Linux-specific")
    raw_path = os.path.join(os.fsencode(tmp_path), name)
    os.mkdir(raw_path)
    return raw_path, os.fsdecode(raw_path)


def test_capture_keeps_raw_surrogateescaped_workspace_text() -> None:
    raw_cwd = "/work/pro\udcffject"
    raw_extra = "/root/\udcfe"
    runtime = SessionEnvironment.capture(
        workspace=Workspace(
            primary_cwd=raw_cwd,
            working_dirs=[WorkingDir(path=raw_extra)],
        )
    )

    assert runtime.cwd == raw_cwd
    assert runtime.working_dirs[0].path == raw_extra


def test_capture_preserves_surrogateescaped_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_bytes, raw_cwd = _surrogateescaped_dir(tmp_path, b"process-\xff")
    monkeypatch.chdir(raw_cwd)

    runtime = SessionEnvironment.capture()

    assert runtime.cwd == raw_cwd
    assert os.fsencode(runtime.cwd) == raw_bytes
    assert os.path.isdir(runtime.cwd)


def test_capture_preserves_surrogateescaped_workspace_paths(tmp_path: Path) -> None:
    cwd_bytes, raw_cwd = _surrogateescaped_dir(tmp_path, b"primary-\xff")
    extra_bytes, raw_extra = _surrogateescaped_dir(tmp_path, b"extra-\xfe")
    workspace = Workspace(
        primary_cwd=raw_cwd,
        working_dirs=[
            WorkingDir(path=raw_cwd, label="primary", is_primary=True),
            WorkingDir(path=raw_extra, label="extra"),
        ],
    )

    runtime = SessionEnvironment.capture(workspace=workspace)

    runtime_paths = [working_dir.path for working_dir in runtime.working_dirs]
    assert runtime.cwd == raw_cwd
    assert runtime_paths == [raw_cwd, raw_extra]
    assert os.fsencode(runtime.cwd) == cwd_bytes
    assert [os.fsencode(path) for path in runtime_paths] == [cwd_bytes, extra_bytes]
    assert all(os.path.isdir(path) for path in [runtime.cwd, *runtime_paths])
    assert workspace.primary_cwd == raw_cwd
    assert [working_dir.path for working_dir in workspace.working_dirs] == [raw_cwd, raw_extra]
