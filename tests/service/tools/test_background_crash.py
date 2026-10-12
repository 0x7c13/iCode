# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The record of running background commands a launch that crashed leaves, and what a later launch does with it."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from chrys.service.tools.background import BackgroundTasks, crash
from chrys.service.tools.background.crash import (
    RunningRecord,
    find_left_behind,
    forget_left_behind,
    stop_left_behind,
)
from tests.support.background_tasks import HeldProcess
from tests.support.waiting import wait_for

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="on Windows a command's job ends with its launch")

_HOLD_LOCK = """
import fcntl, os, sys, time
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print("held", flush=True)
time.sleep(60)
"""


@pytest.fixture
def command() -> Iterator[subprocess.Popen[bytes]]:
    """A command running in its own session, as a background command does."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True, stdin=subprocess.DEVNULL
    )
    yield process
    process.kill()
    process.wait()


@pytest.fixture
def sessions(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


type MakeRecord = Callable[..., Awaitable[RunningRecord]]


@pytest.fixture
async def make_record(sessions: Path, command: subprocess.Popen[bytes]) -> AsyncIterator[MakeRecord]:
    """Write the record a launch keeps for *command*; one that is gone, unless not *crashed*."""
    records: list[RunningRecord] = []

    async def make(*, surface: str = "tui", crashed: bool = True) -> RunningRecord:
        record = RunningRecord(sessions / "abc" / "background", surface=surface)
        records.append(record)
        task = SimpleNamespace(id="t1", pid=command.pid, command="npm run dev", cwd="/repo")
        await record.update([task])  # type: ignore[list-item]
        if crashed:
            # What the system does as the launch ends.
            assert record._held is not None
            record._held.release()
            record._held = None
        return record

    yield make
    for record in records:
        await record.update([])


async def test_a_crashed_launch_leaves_its_commands_and_stopping_ends_them(
    sessions: Path, command: subprocess.Popen[bytes], make_record: MakeRecord
) -> None:
    record = await make_record()

    left = find_left_behind(sessions)

    assert [entry.record for entry in left] == [record.path]
    [found] = left[0].commands
    assert (found.command, found.cwd, found.session) == ("npm run dev", "/repo", command.pid)
    assert command.pid in [pid for pid, _created in found.members]
    stop_left_behind(left)
    assert await asyncio.to_thread(command.wait, 5) == -signal.SIGTERM
    assert list(record.path.parent.iterdir()) == []


async def test_ignoring_forgets_the_record_and_leaves_the_commands_running(
    sessions: Path, command: subprocess.Popen[bytes], make_record: MakeRecord
) -> None:
    record = await make_record()

    forget_left_behind(find_left_behind(sessions))

    assert list(record.path.parent.iterdir()) == []
    assert command.poll() is None
    assert find_left_behind(sessions) == []


async def test_a_launch_still_running_keeps_its_record(sessions: Path, make_record: MakeRecord) -> None:
    record = await make_record(crashed=False)

    assert find_left_behind(sessions) == []
    assert record.path.exists()


async def test_a_killed_launch_leaves_its_record_before_it_is_reaped(sessions: Path, make_record: MakeRecord) -> None:
    record = await make_record()
    launch = await asyncio.to_thread(
        subprocess.Popen,
        [sys.executable, "-c", _HOLD_LOCK, str(record.path.with_suffix(".lock"))],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
    )
    assert launch.stdout is not None
    try:
        assert await asyncio.to_thread(launch.stdout.readline) == b"held\n"
        assert find_left_behind(sessions) == []

        launch.kill()
        await wait_for(lambda: psutil.Process(launch.pid).status() == psutil.STATUS_ZOMBIE, description="unreaped")
        assert [entry.record for entry in find_left_behind(sessions)] == [record.path]
    finally:
        launch.kill()
        await asyncio.to_thread(launch.wait)
        launch.stdout.close()


async def test_only_the_asked_surface_is_found(sessions: Path, make_record: MakeRecord) -> None:
    record = await make_record(surface="tui")

    assert find_left_behind(sessions, surface="acp") == []
    assert record.path.exists()
    assert [entry.record for entry in find_left_behind(sessions, surface="tui")] == [record.path]


async def test_a_corrected_clock_still_finds_the_command(
    sessions: Path, command: subprocess.Popen[bytes], make_record: MakeRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded = ((command.pid, psutil.Process(command.pid).create_time() + 1),)
    monkeypatch.setattr(crash, "_members", lambda _session: recorded)
    record = await make_record()

    assert [entry.record for entry in find_left_behind(sessions)] == [record.path]


@pytest.mark.parametrize("members", ["pid-reused", "left-its-session"])
async def test_a_record_whose_processes_are_not_the_command_is_removed(
    sessions: Path,
    command: subprocess.Popen[bytes],
    make_record: MakeRecord,
    monkeypatch: pytest.MonkeyPatch,
    members: str,
) -> None:
    if members == "pid-reused":
        # Also what a reboot looks like: every process now is younger.
        recorded = ((command.pid, psutil.Process(command.pid).create_time() - 60),)
    else:
        recorded = ((os.getpid(), psutil.Process().create_time()),)
    monkeypatch.setattr(crash, "_members", lambda _session: recorded)
    record = await make_record()

    assert find_left_behind(sessions) == []
    assert list(record.path.parent.iterdir()) == []
    assert command.poll() is None


async def test_an_unreadable_record_is_removed(sessions: Path) -> None:
    path = sessions / "abc" / "background" / "running-0000.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"tasks": [1]}', encoding="utf-8")

    assert find_left_behind(sessions) == []
    assert list(path.parent.iterdir()) == []


async def test_the_registry_keeps_the_record_while_a_command_runs(
    tmp_path: Path, command: subprocess.Popen[bytes]
) -> None:
    tasks = BackgroundTasks(tmp_path / "abc")
    try:
        process = HeldProcess()
        process.pid = command.pid
        task = tasks.adopt(process, command="npm run dev", trigger="timeout")
        assert task is not None

        def records() -> list[Path]:
            return list(tasks.directory.glob("running-*"))

        await wait_for(lambda: len(records()) == 2, description="the record and its lock written")
        assert find_left_behind(tmp_path) == []

        process.exit(0)
        await wait_for(lambda: not records(), description="the record and its lock removed")
    finally:
        await tasks.close()
