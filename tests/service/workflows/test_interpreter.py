# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Interpreter probe: facts from a real interpreter, the floor, and unusable executables."""

from __future__ import annotations

import asyncio
import platform
import sys
from pathlib import Path

import psutil
import pytest

from chrys.service.workflows import interpreter as probe_module
from chrys.service.workflows.interpreter import InterpreterError, InterpreterProbe, probe_interpreter
from tests.support.waiting import wait_for


async def test_probe_reports_the_running_interpreter() -> None:
    probe = await probe_interpreter(sys.executable)
    assert probe.version_tuple == tuple(sys.version_info[:3])
    assert probe.implementation == sys.implementation.name.capitalize().replace("Cpython", "CPython")
    assert probe.platform == sys.platform
    assert (probe.machine, probe.libc) == (platform.machine(), platform.libc_ver()[0])


@pytest.mark.parametrize(
    ("python_version", "expected"),
    [("3.14.0", (3, 14, 0)), ("3.9.1rc1", (3, 9, 1)), ("3.15.0a1", (3, 15, 0)), ("3.12.4+", (3, 12, 4))],
)
def test_version_tuple_keeps_the_patch_number_of_a_pre_release(python_version: str, expected: tuple[int, ...]) -> None:
    probe = InterpreterProbe("python", python_version, "CPython", "linux", "x86_64", "glibc")
    assert probe.version_tuple == expected


async def test_probe_rejects_an_interpreter_below_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probe_module, "PYTHON_FLOOR", (3, 99))
    with pytest.raises(InterpreterError, match="or newer"):
        await probe_interpreter(sys.executable)


async def test_probe_rejects_a_missing_executable(tmp_path: Path) -> None:
    with pytest.raises(InterpreterError, match="Cannot start"):
        await probe_interpreter(str(tmp_path / "missing-python"))


async def test_probe_rejects_a_non_python_executable(tmp_path: Path) -> None:
    with pytest.raises(InterpreterError, match="failed the version probe"):
        await probe_interpreter("git")


async def test_cancelled_probe_reaps_its_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    stall = f"import os,time;f=open({str(pid_file)!r},'w');f.write(str(os.getpid()));f.close();time.sleep(60)"
    monkeypatch.setattr(probe_module, "PROBE_SOURCE", stall)
    task = asyncio.create_task(probe_interpreter(sys.executable))
    await wait_for(
        lambda: task.done() or bool(pid_file.exists() and pid_file.read_text()), description="probe child wrote its pid"
    )
    if task.done():
        await task
    # Identity, not the number: once the reaped child's handle is closed, Windows can hand
    # its pid to the next process another test worker spawns, and pid_exists() reports that one.
    child = psutil.Process(int(pid_file.read_text()))

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_for(lambda: not child.is_running(), description="cancelled probe reaped its child")
