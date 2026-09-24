# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for packaged-runtime PATH demotion helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from chrys.foundation.platform import runtime_paths


def _exe(path: Path) -> Path:
    if sys.platform == "win32" and not path.suffix:
        return path.with_name(path.name + ".exe")
    return path


def _make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    path.chmod(0o755)
    return path


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    python: Path,
    prefix: Path,
    scripts: Path,
    platform: str | None = None,
) -> None:
    monkeypatch.setattr(runtime_paths.sys, "executable", str(python))
    monkeypatch.setattr(runtime_paths.sys, "prefix", str(prefix))
    monkeypatch.setattr(runtime_paths.sys, "exec_prefix", str(prefix))
    if platform is not None:
        monkeypatch.setattr(runtime_paths.sys, "platform", platform)
    monkeypatch.setattr(
        runtime_paths.sysconfig,
        "get_path",
        lambda name: str(scripts) if name == "scripts" else "",
    )


def test_non_frozen_path_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_bin = tmp_path / "runtime" / "bin"
    system_bin = tmp_path / "system" / "bin"
    _patch_runtime(
        monkeypatch,
        python=_exe(runtime_bin / "python"),
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
    )
    original = os.pathsep.join([str(runtime_bin), "", str(system_bin), str(runtime_bin)])
    env = {"PATH": original}

    runtime_paths.reorder_path_demoting_runtime(env)

    assert env["PATH"] == original


def test_frozen_path_demotes_runtime_dirs_with_stable_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_prefix = tmp_path / "runtime"
    runtime_bin = runtime_prefix / "bin"
    system_bin = tmp_path / "system" / "bin"
    _patch_runtime(
        monkeypatch,
        python=_exe(runtime_bin / "python"),
        prefix=runtime_prefix,
        scripts=runtime_bin,
    )
    env = {
        "PYAPP": str(tmp_path / "chrys"),
        "PATH": os.pathsep.join([str(runtime_bin), str(system_bin), str(runtime_bin), str(runtime_prefix)]),
    }

    runtime_paths.reorder_path_demoting_runtime(env)

    assert env["PATH"].split(os.pathsep) == [
        str(system_bin),
        str(runtime_bin),
        str(runtime_bin),
        str(runtime_prefix),
    ]


def test_frozen_path_strips_empty_entries_without_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bin = tmp_path / "runtime" / "bin"
    system_bin = tmp_path / "system" / "bin"
    _patch_runtime(
        monkeypatch,
        python=_exe(runtime_bin / "python"),
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
    )
    env = {
        "PYAPP": "1",
        "PATH": os.pathsep.join(["", str(runtime_bin), "", str(system_bin), ""]),
    }

    runtime_paths.reorder_path_demoting_runtime(env)

    assert env["PATH"].split(os.pathsep) == [str(system_bin), str(runtime_bin)]


def test_frozen_path_with_only_runtime_dirs_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_prefix = tmp_path / "runtime"
    runtime_bin = runtime_prefix / "bin"
    _patch_runtime(
        monkeypatch,
        python=_exe(runtime_bin / "python"),
        prefix=runtime_prefix,
        scripts=runtime_bin,
    )
    original = os.pathsep.join([str(runtime_bin), str(runtime_prefix)])
    env = {"PYAPP": "1", "PATH": original}

    runtime_paths.reorder_path_demoting_runtime(env)

    assert env["PATH"] == original


def test_path_key_casing_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_bin = tmp_path / "runtime" / "bin"
    system_bin = tmp_path / "system" / "bin"
    _patch_runtime(
        monkeypatch,
        python=_exe(runtime_bin / "python"),
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
    )
    env = {
        "PYAPP": "1",
        "Path": os.pathsep.join([str(runtime_bin), str(system_bin)]),
    }

    runtime_paths.reorder_path_demoting_runtime(env)

    assert "Path" in env
    assert "PATH" not in env
    assert env["Path"].split(os.pathsep) == [str(system_bin), str(runtime_bin)]


def test_windows_duplicate_path_keys_collapse_to_later_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bin = tmp_path / "runtime" / "Scripts"
    system_bin = tmp_path / "system" / "bin"
    _patch_runtime(
        monkeypatch,
        python=tmp_path / "runtime" / "python.exe",
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
        platform="win32",
    )
    env = {
        "PYAPP": "1",
        "Path": str(runtime_bin),
        "PATH": os.pathsep.join([str(runtime_bin), str(system_bin)]),
    }

    runtime_paths.reorder_path_demoting_runtime(env)

    assert "Path" not in env
    assert env["PATH"].split(os.pathsep) == [str(system_bin), str(runtime_bin)]


def test_windows_duplicate_path_keys_collapse_when_selected_path_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bin = tmp_path / "runtime" / "Scripts"
    _patch_runtime(
        monkeypatch,
        python=tmp_path / "runtime" / "python.exe",
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
        platform="win32",
    )
    env = {
        "PYAPP": "1",
        "Path": str(runtime_bin),
        "PATH": "",
    }

    runtime_paths.reorder_path_demoting_runtime(env)

    assert "Path" not in env
    assert env["PATH"] == ""


def test_missing_path_key_is_noop() -> None:
    env = {"PYAPP": "1"}

    runtime_paths.reorder_path_demoting_runtime(env)

    assert env == {"PYAPP": "1"}


def test_strip_python_runtime_overrides_is_unconditional_and_case_insensitive() -> None:
    env = {
        "PYTHONHOME": "/bad/home",
        "pythonpath": "/bad/path",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "PATH": "/usr/bin",
    }

    result = runtime_paths.strip_python_runtime_overrides(env)

    assert result is env
    assert "PYTHONHOME" not in env
    assert "pythonpath" not in env
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PATH"] == "/usr/bin"


def test_strip_python_runtime_overrides_can_preserve_pythonpath() -> None:
    env = {
        "PYTHONHOME": "/bad/home",
        "PYTHONPATH": "/shared/helpers",
    }

    runtime_paths.strip_python_runtime_overrides(env, strip_pythonpath=False)

    assert "PYTHONHOME" not in env
    assert env["PYTHONPATH"] == "/shared/helpers"


def test_windows_scripts_dir_is_demoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_prefix = tmp_path / "runtime"
    runtime_scripts = runtime_prefix / "Scripts"
    runtime_library_bin = runtime_prefix / "Library" / "bin"
    system_bin = tmp_path / "system" / "bin"
    _patch_runtime(
        monkeypatch,
        python=runtime_prefix / "python.exe",
        prefix=runtime_prefix,
        scripts=runtime_scripts,
        platform="win32",
    )
    env = {
        "PYAPP": "1",
        "PATH": os.pathsep.join([str(runtime_scripts), str(system_bin), str(runtime_library_bin), str(runtime_prefix)]),
    }

    runtime_paths.reorder_path_demoting_runtime(env)

    assert env["PATH"].split(os.pathsep) == [
        str(system_bin),
        str(runtime_scripts),
        str(runtime_library_bin),
        str(runtime_prefix),
    ]


def test_which_excluding_runtime_prefers_system_then_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bin = tmp_path / "runtime" / "bin"
    system_bin = tmp_path / "system" / "bin"
    runtime_uv = _make_executable(_exe(runtime_bin / "uv"))
    system_uv = _make_executable(_exe(system_bin / "uv"))
    _patch_runtime(
        monkeypatch,
        python=_exe(runtime_bin / "python"),
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
    )
    if sys.platform == "win32":
        monkeypatch.setenv("PATHEXT", ".exe")
    env = {
        "PYAPP": "1",
        "PATH": os.pathsep.join([str(runtime_bin), str(system_bin)]),
    }

    assert runtime_paths.which_excluding_runtime("uv", env=env) == str(system_uv)

    env["PATH"] = str(runtime_bin)
    assert runtime_paths.which_excluding_runtime("uv", env=env) == str(runtime_uv)


def test_which_excluding_runtime_honors_env_pathext_on_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bin = tmp_path / "runtime" / "Scripts"
    system_bin = tmp_path / "system" / "bin"
    system_tool = _make_executable(system_bin / "tool.cmd")
    _patch_runtime(
        monkeypatch,
        python=runtime_bin / "python.exe",
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
        platform="win32",
    )
    monkeypatch.delenv("PATHEXT", raising=False)
    env = {
        "PYAPP": "1",
        "PATH": os.pathsep.join([str(runtime_bin), str(system_bin)]),
        "PATHEXT": ".cmd",
    }

    assert runtime_paths.which_excluding_runtime("tool", env=env) == str(system_tool)


def test_which_excluding_runtime_preserves_windows_file_casing_with_default_pathext(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bin = tmp_path / "runtime" / "Scripts"
    system_bin = tmp_path / "system" / "bin"
    system_uv = _make_executable(system_bin / "uv.exe")
    _patch_runtime(
        monkeypatch,
        python=runtime_bin / "python.exe",
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
        platform="win32",
    )
    env = {
        "PYAPP": "1",
        "PATH": os.pathsep.join([str(runtime_bin), str(system_bin)]),
    }

    assert runtime_paths.which_excluding_runtime("uv", env=env) == str(system_uv)
