# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A spawn into a deleted working directory is told apart from a missing executable."""

from __future__ import annotations

import asyncio
import errno
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import chrys.foundation.platform as platform_mod
from chrys.foundation.platform import process as process_mod
from chrys.foundation.platform.process import (
    ManagedStdioProcess,
    MissingWorkingDirectoryError,
    managed_subprocess,
    raise_if_missing_cwd,
    spawn_managed_stdio_process,
)

_POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX spawn path; the Windows branch is stubbed in its own test"
)


def _deleted_dir(tmp_path: Path) -> Path:
    gone = tmp_path / "gone"
    gone.mkdir()
    gone.rmdir()
    return gone


def _windows_directory_error() -> NotADirectoryError:
    """What CreateProcess reports for a missing cwd (``WinError 267``)."""
    return NotADirectoryError(errno.ENOTDIR, "The directory name is invalid")


def test_missing_working_directory_error_is_an_oserror_but_not_a_file_not_found_error() -> None:
    error = MissingWorkingDirectoryError("/work/gone")

    assert isinstance(error, OSError)
    assert not isinstance(error, FileNotFoundError)
    assert error.errno == errno.ENOENT
    assert error.path == "/work/gone"
    assert error.filename == "/work/gone"
    assert str(error) == "working directory no longer exists: /work/gone"


@pytest.mark.parametrize("as_pathlike", [False, True])
def test_raise_if_missing_cwd_reclassifies_a_spawn_into_a_deleted_directory(tmp_path: Path, as_pathlike: bool) -> None:
    gone = _deleted_dir(tmp_path)
    original = FileNotFoundError(errno.ENOENT, "No such file or directory", "sh")

    with pytest.raises(MissingWorkingDirectoryError) as raised:
        raise_if_missing_cwd(gone if as_pathlike else str(gone), original)

    assert raised.value.path == str(gone)
    assert raised.value.__cause__ is original


@pytest.mark.parametrize("case", ["existing", "none", "empty", "bytes", "already_classified"])
def test_raise_if_missing_cwd_leaves_other_failures_to_the_caller(tmp_path: Path, case: str) -> None:
    gone = str(_deleted_dir(tmp_path))
    error: OSError = FileNotFoundError(errno.ENOENT, "No such file or directory", "sh")
    cwd: object = {
        "existing": str(tmp_path),
        "none": None,
        "empty": "",
        "bytes": os.fsencode(gone),
        "already_classified": gone,
    }[case]
    if case == "already_classified":
        error = MissingWorkingDirectoryError(gone)

    raise_if_missing_cwd(cwd, error)  # returns, so the caller re-raises *error* unchanged


async def test_managed_subprocess_reports_a_deleted_cwd(tmp_path: Path) -> None:
    gone = _deleted_dir(tmp_path)

    with pytest.raises(MissingWorkingDirectoryError) as raised:
        async with managed_subprocess(sys.executable, "-c", "pass", cwd=str(gone)):
            pytest.fail("the child must not start in a deleted directory")

    assert raised.value.path == str(gone)
    assert isinstance(raised.value.__cause__, OSError)


async def test_managed_subprocess_keeps_a_missing_executable_a_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        async with managed_subprocess("__nonexistent_binary_xyz__", cwd=str(tmp_path)):
            pytest.fail("a missing executable must not start")


@pytest.mark.parametrize("cwd_exists", [False, True])
async def test_managed_subprocess_classifies_a_windows_shaped_spawn_error_by_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cwd_exists: bool
) -> None:
    cwd = tmp_path if cwd_exists else _deleted_dir(tmp_path)
    failure = _windows_directory_error()

    async def refuse_spawn(*_args: Any, **_kwargs: Any) -> asyncio.subprocess.Process:
        raise failure

    shadow = ModuleType("asyncio")
    shadow.__dict__.update(vars(asyncio), create_subprocess_exec=refuse_spawn)
    monkeypatch.setattr(process_mod, "asyncio", shadow)

    with pytest.raises(OSError) as raised:
        async with managed_subprocess("agent.exe", cwd=str(cwd)):
            pytest.fail("the stubbed spawn never starts a child")

    if cwd_exists:
        assert raised.value is failure
    else:
        assert isinstance(raised.value, MissingWorkingDirectoryError)
        assert raised.value.__cause__ is failure


@_POSIX_ONLY
async def test_spawn_managed_stdio_process_reports_a_deleted_cwd(tmp_path: Path) -> None:
    gone = _deleted_dir(tmp_path)

    with pytest.raises(MissingWorkingDirectoryError) as raised:
        await spawn_managed_stdio_process(
            sys.executable, "-c", "pass", cwd=str(gone), env=dict(os.environ), limit=64 * 1024
        )

    assert raised.value.path == str(gone)


@_POSIX_ONLY
async def test_spawn_managed_stdio_process_keeps_a_missing_executable_a_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await spawn_managed_stdio_process(
            str(tmp_path / "no-such-agent"), cwd=str(tmp_path), env=dict(os.environ), limit=64 * 1024
        )


@pytest.mark.parametrize("cwd_exists", [False, True])
async def test_windows_stdio_spawn_classifies_its_failure_by_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cwd_exists: bool
) -> None:
    cwd = tmp_path if cwd_exists else _deleted_dir(tmp_path)
    failure = _windows_directory_error()

    async def refuse_spawn(
        command: str,
        args: tuple[str, ...],
        *,
        cwd: str,
        env: dict[str, str],
        limit: int,
        parent_env: dict[str, str] | None,
    ) -> ManagedStdioProcess:
        raise failure

    monkeypatch.setattr(platform_mod, "get_platform", lambda: SimpleNamespace(is_windows=True))
    monkeypatch.setattr(process_mod, "_spawn_windows_managed_stdio_process", refuse_spawn)

    with pytest.raises(OSError) as raised:
        await spawn_managed_stdio_process("agent.exe", cwd=str(cwd), env={}, limit=1024)

    if cwd_exists:
        assert raised.value is failure
    else:
        assert isinstance(raised.value, MissingWorkingDirectoryError)
        assert raised.value.path == str(cwd)
        assert raised.value.__cause__ is failure
