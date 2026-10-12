# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A shell command that outlives its timeout moves to the session's background tasks."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import psutil
import pytest

from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.tool_result_metadata import (
    SHELL_BACKGROUND_TASK_METADATA_KEY,
    SHELL_TIMED_OUT_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
)
from chrys.service.tools.background import BackgroundTasks, stop_command
from chrys.service.tools.background import registry as registry_mod
from chrys.service.tools.builtins.shell import ShellTools, shell_result_metadata
from chrys.service.tools.builtins.shell_filter import ShellCommandFilter
from chrys.service.tools.result_metadata import tool_result_metadata
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

_WAIT_FOR_RELEASE = """\
import pathlib, sys, time
print("started", flush=True)
release = pathlib.Path(sys.argv[1])
while not release.exists():
    time.sleep(0.05)
print("finished", flush=True)
"""

_SILENT_FAMILY = """\
import os, pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
pathlib.Path(sys.argv[1]).write_text(f"{os.getpid()} {child.pid}", encoding="utf-8")
while True:
    time.sleep(0.05)
"""


@pytest.fixture
async def tasks(tmp_path: Path) -> AsyncIterator[BackgroundTasks]:
    registry = BackgroundTasks(tmp_path / "session")
    yield registry
    await registry.close()


@pytest.fixture
def metadata() -> Iterator[dict[str, Any]]:
    values: dict[str, Any] = {}
    shell_token = shell_result_metadata.set(values)
    tool_token = tool_result_metadata.set(values)
    yield values
    tool_result_metadata.reset(tool_token)
    shell_result_metadata.reset(shell_token)


def _python_command(script: Path, *args: Path) -> str:
    # PowerShell runs a quoted program path only through its call operator.
    prefix = "& " if sys.platform == "win32" else ""
    return " ".join([f'{prefix}"{sys.executable}"', f'"{script}"', *(f'"{arg}"' for arg in args)])


async def _family(pids: Path, call: asyncio.Task[str] | None = None) -> list[psutil.Process]:
    """The command's processes, once its script wrote their pids; *call* runs the command in the foreground."""
    await wait_for(
        lambda: pids.exists() or (call is not None and call.done()),
        timeout=ENGINE_TURN_TIMEOUT,
        description="pids written",
    )
    if call is not None and call.done():
        pytest.fail(f"the call ended before its command started: {call.result()}")
    await wait_for(lambda: pids.read_text(encoding="utf-8").count(" ") == 1, description="both pids written")
    return [psutil.Process(int(pid)) for pid in pids.read_text(encoding="utf-8").split()]


async def _gone(processes: list[psutil.Process]) -> None:
    await wait_for(
        lambda: not any(process.is_running() for process in processes),
        timeout=ENGINE_TURN_TIMEOUT,
        description="every process of the command exited",
    )


async def _start_waiting_command(
    shell: ShellTools, tasks: BackgroundTasks, tmp_path: Path
) -> tuple[str, registry_mod.BackgroundTask, Path]:
    script = tmp_path / "wait_for_release.py"
    script.write_text(_WAIT_FOR_RELEASE, encoding="utf-8")
    release = tmp_path / "release"
    result = await shell.execute(_python_command(script, release), reason="test", timeout=1)
    [task] = tasks.running
    return result, task, release


async def test_timed_out_command_keeps_running_and_logs_to_its_end(
    tasks: BackgroundTasks, metadata: dict[str, Any], tmp_path: Path
) -> None:
    shell = ShellTools(SessionEnvironment.capture(), background_tasks=tasks)

    result, task, release = await _start_waiting_command(shell, tasks, tmp_path)

    assert "[Still running in the background: timeout of 1s reached]" in result
    assert f"Task {task.id} (pid {task.pid}) keeps running" in result
    assert str(task.log_path) in result
    assert stop_command(shell.shell.name, task.stop_path) in result
    assert "[exit_code:" not in result
    assert metadata[SHELL_BACKGROUND_TASK_METADATA_KEY] == {
        "id": task.id,
        "pid": task.pid,
        "trigger": "timeout",
        "log": str(task.log_path),
    }
    assert metadata[TOOL_FAILED_METADATA_KEY] is False
    assert SHELL_TIMED_OUT_METADATA_KEY not in metadata
    # What the tool call kept is in its result; the command may run on for hours without it.
    assert task.process.captured() == ()

    release.touch()
    await wait_for(task.ended.is_set, timeout=ENGINE_TURN_TIMEOUT, description="the command finished")

    log = task.log_path.read_text(encoding="utf-8")
    assert log.endswith("finished\n[icode] process exited with code 0\n")
    assert log.count("started") == 1
    assert task.exit_code == 0


async def test_stop_command_stops_a_silent_command_with_every_process_it_started(
    tasks: BackgroundTasks, metadata: dict[str, Any], tmp_path: Path
) -> None:
    # A filter that blocks every way of creating the file still lets the stop command through.
    blocking = ShellCommandFilter(blacklist=frozenset({"touch", "New-Item", "type"}), allow_redirections=False)
    shell = ShellTools(SessionEnvironment.capture(), command_filter=blocking, background_tasks=tasks)
    script = tmp_path / "family.py"
    script.write_text(_SILENT_FAMILY, encoding="utf-8")
    pids = tmp_path / "pids"
    await shell.execute(_python_command(script, pids), reason="test", timeout=1)
    [task] = tasks.running
    # Nothing was printed, yet the log and its directory exist for the stop command.
    assert task.log_path.exists()
    processes = await _family(pids)
    other = await shell.execute(stop_command(shell.shell.name, tmp_path / "other.stop"), reason="test")
    assert other.startswith("Error: command blocked")

    result = await shell.execute(stop_command(shell.shell.name, task.stop_path), reason="stop the task")

    assert f"[Stopped background task {task.id}]" in result
    assert result.endswith("[exit_code: 0]")
    assert task.log_path.read_text(encoding="utf-8").endswith("[icode] process stopped by agent\n")
    await _gone(processes)
    assert tasks.running == []


async def test_cancelled_call_still_kills_its_command(
    tasks: BackgroundTasks, metadata: dict[str, Any], tmp_path: Path
) -> None:
    shell = ShellTools(SessionEnvironment.capture(), background_tasks=tasks)
    script = tmp_path / "family.py"
    script.write_text(_SILENT_FAMILY, encoding="utf-8")
    pids = tmp_path / "pids"
    run = asyncio.create_task(shell.execute(_python_command(script, pids), reason="test", timeout=60))
    processes = await _family(pids, run)

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    await _gone(processes)
    assert tasks.running == []


async def test_full_session_kills_the_timed_out_command(
    tasks: BackgroundTasks, metadata: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry_mod, "MAX_RUNNING_TASKS", 0)
    shell = ShellTools(SessionEnvironment.capture(), background_tasks=tasks)
    script = tmp_path / "family.py"
    script.write_text(_SILENT_FAMILY, encoding="utf-8")
    pids = tmp_path / "pids"
    run = asyncio.create_task(shell.execute(_python_command(script, pids), reason="test", timeout=3))
    processes = await _family(pids, run)

    result = await run

    assert result.startswith("Error: command timed out after 3 seconds.")
    assert "stop one first" in result
    assert metadata[SHELL_TIMED_OUT_METADATA_KEY] is True
    await _gone(processes)
    assert tasks.running == []


def test_description_offers_the_background_only_with_a_registry(tmp_path: Path) -> None:
    runtime = SessionEnvironment.capture()

    def description(shell: ShellTools) -> str:
        [tool] = [tool for tool in shell.tools() if tool.name == shell.shell.name]
        return tool.description

    without = description(ShellTools(runtime))
    with_tasks = description(ShellTools(runtime, background_tasks=BackgroundTasks(tmp_path)))
    session_wide = description(
        ShellTools(runtime, background_tasks=BackgroundTasks(tmp_path), session_wide_background=True)
    )

    assert "No asynchronous task handle is provided." in without
    assert "moves to the background" not in without
    assert "No asynchronous task handle is provided." not in with_tasks
    assert "moves to the background" in with_tasks
    assert "stopped when you finish" in with_tasks
    assert "stopped when the session ends" not in with_tasks
    assert "stopped when the session ends" in session_wide
    assert "stopped when you finish" not in session_wide
