# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for packaged-runtime PATH demotion helpers."""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chrys.foundation.platform import runtime_paths
from tests.support.app_execution_aliases import (
    IO_REPARSE_TAG_APPEXECLINK,
    STORE_PYTHON,
    STORE_PYTHON_REDIRECTOR,
    STORE_PYTHON_STUB,
    alias_reparse_data,
)
from tests.support.symlinks import junction_or_skip


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
        # Off Windows, nothing is an App Execution Alias, and lstat reports no reparse tag to ask.
        monkeypatch.setattr(runtime_paths, "is_app_execution_alias", lambda path: False)
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


def _declared_length(reparse: bytes, length: int) -> bytes:
    return reparse[:4] + struct.pack("<H", length) + reparse[6:]


def _alias_with_strings_data(data: bytes) -> bytes:
    """An alias whose strings are the raw *data*, with the header and prefix an alias has."""
    payload = struct.pack("<I", 3) + data
    return struct.pack("<IHH", IO_REPARSE_TAG_APPEXECLINK, len(payload), 0) + payload


_REDIRECTOR_STRING = f"{STORE_PYTHON_REDIRECTOR}\0".encode("utf-16-le")


@pytest.mark.parametrize(
    ("reparse", "is_stub"),
    [
        (STORE_PYTHON_STUB, True),
        (_alias_with_strings_data(_REDIRECTOR_STRING), True),
        (STORE_PYTHON, False),
        (b"", False),
        (STORE_PYTHON_STUB[:7], False),
        # Another kind of reparse point with the same strings.
        (alias_reparse_data(STORE_PYTHON_REDIRECTOR, tag=0xA000000C), False),
        # Declaring more data than was returned, or an odd number of bytes.
        (_declared_length(STORE_PYTHON_STUB, len(STORE_PYTHON_STUB) - 6), False),
        (_alias_with_strings_data(_REDIRECTOR_STRING + b"\x00"), False),
        # The redirector without its NUL is no complete string.
        (_alias_with_strings_data(_REDIRECTOR_STRING[:-2]), False),
        # A lone surrogate is no UTF-16 text.
        (_alias_with_strings_data(b"\x00\xd8A\x00\x00\x00" + _REDIRECTOR_STRING), False),
    ],
    ids=[
        "stub",
        "only_the_redirector",
        "store_python",
        "empty",
        "short",
        "symlink_tag",
        "overlong",
        "odd",
        "unterminated",
        "bad_utf16",
    ],
)
def test_only_the_store_python_redirector_alias_counts_as_the_placeholder(reparse: bytes, is_stub: bool) -> None:
    assert runtime_paths._names_store_python_redirector(reparse) is is_stub


def test_lookup_without_any_path_still_passes_over_the_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    placeholder = r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\python3.exe"
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("PATH")
    monkeypatch.setattr(runtime_paths, "shutil", SimpleNamespace(which=lambda name: placeholder))
    monkeypatch.setattr(runtime_paths, "is_windows_store_python_stub", lambda path: path == placeholder)
    monkeypatch.setattr(runtime_paths, "normalize_path", os.fspath)

    assert runtime_paths.which_excluding_runtime("python3") == ""
    monkeypatch.setattr(runtime_paths, "is_windows_store_python_stub", lambda path: False)
    assert runtime_paths.which_excluding_runtime("python3") == placeholder


class _CountedKernel32:
    """The real kernel32 calls, counting the handles opened and closed."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.opened: list[int] = []
        self.closed: list[int] = []

    def CreateFileW(self, *args: Any) -> int:
        handle = self._real.CreateFileW(*args)
        if handle != runtime_paths._INVALID_HANDLE_VALUE:
            self.opened.append(handle)
        return handle

    def DeviceIoControl(self, *args: Any) -> int:
        return self._real.DeviceIoControl(*args)

    def CloseHandle(self, handle: int) -> int:
        self.closed.append(handle)
        return self._real.CloseHandle(handle)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows reparse points only")
def test_reading_a_reparse_point_returns_its_data_and_closes_every_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    junction = tmp_path / "junction"
    junction_or_skip(junction, target)
    regular = tmp_path / "regular.exe"
    regular.write_bytes(b"MZ")
    kernel32 = _CountedKernel32(runtime_paths._windows_kernel32())
    monkeypatch.setattr(runtime_paths, "_windows_kernel32", lambda: kernel32)

    data = runtime_paths._read_reparse_point(str(junction))
    assert struct.unpack_from("<I", data) == (0xA0000003,)  # IO_REPARSE_TAG_MOUNT_POINT
    # A file that is no reparse point, and one that does not exist.
    assert runtime_paths._read_reparse_point(str(regular)) == b""
    assert runtime_paths._read_reparse_point(str(tmp_path / "missing.exe")) == b""

    assert len(kernel32.opened) == 2
    assert kernel32.closed == kernel32.opened
    assert not runtime_paths.is_app_execution_alias(str(regular))
    assert not runtime_paths.is_windows_store_python_stub(str(regular))


@pytest.mark.parametrize(
    ("lstat", "is_alias"),
    [
        (SimpleNamespace(st_reparse_tag=IO_REPARSE_TAG_APPEXECLINK), True),
        (SimpleNamespace(st_reparse_tag=0xA000000C), False),
        (SimpleNamespace(st_reparse_tag=0), False),
        (None, False),
    ],
    ids=["alias", "symlink", "plain_file", "unreadable"],
)
def test_an_app_execution_alias_is_known_by_its_reparse_tag(
    monkeypatch: pytest.MonkeyPatch, lstat: SimpleNamespace | None, is_alias: bool
) -> None:
    def fake_lstat(path: str) -> SimpleNamespace:
        if lstat is None:
            raise PermissionError(path)
        return lstat

    monkeypatch.setattr(runtime_paths.sys, "platform", "win32")
    monkeypatch.setattr(runtime_paths, "os", SimpleNamespace(lstat=fake_lstat))

    assert runtime_paths.is_app_execution_alias(r"C:\apps\python3.exe") is is_alias


@pytest.mark.skipif(sys.platform != "win32", reason="Windows App Execution Aliases only")
def test_the_aliases_this_windows_has_are_read_as_aliases() -> None:
    apps = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WindowsApps"
    # Found without the code under test, so that code being wrong cannot skip this test.
    aliases = [
        str(path) for path in sorted(apps.glob("*.exe")) if os.lstat(path).st_reparse_tag == IO_REPARSE_TAG_APPEXECLINK
    ]
    if not aliases:
        pytest.skip("This Windows has no App Execution Aliases")

    for alias in aliases:
        assert runtime_paths.is_app_execution_alias(alias), alias
        data = runtime_paths._app_execution_alias_data(alias)
        assert struct.unpack_from("<I", data) == (IO_REPARSE_TAG_APPEXECLINK,)
        # The header and the prefix take 12 bytes, so the strings stay aligned.
        names_redirector = "appinstallerpythonredirector.exe" in data.decode("utf-16-le", "ignore").casefold()
        assert runtime_paths.is_windows_store_python_stub(alias) is names_redirector, alias
