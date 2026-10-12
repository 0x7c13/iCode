# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The registry of a session's background commands and the logs they write."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from chrys.foundation.platform.output_capture import CapturedOutput
from chrys.service.tools.background import BackgroundTasks, stop_command
from chrys.service.tools.background import log as log_mod
from chrys.service.tools.background import registry as registry_mod
from chrys.service.tools.background.log import TaskLog
from tests.support.waiting import wait_for


class _Process:
    """A command the registry adopts; the test decides when and how it ends."""

    def __init__(self, *, ignores: tuple[bool, ...] = ()) -> None:
        self.pid = 4242
        self.sink: Callable[[int, bytes], None] | None = None
        self.signals: list[bool] = []
        self.closed = False
        self._ignores = ignores
        self._code: asyncio.Future[int] = asyncio.get_running_loop().create_future()

    def hand_off(self, sink: Callable[[int, bytes], None]) -> tuple[CapturedOutput, ...]:
        self.sink = sink
        return (CapturedOutput(b"before\n", b"", 7),)

    async def wait(self) -> int:
        return await asyncio.shield(self._code)

    async def terminate(self, *, force: bool) -> None:
        self.signals.append(force)
        if force not in self._ignores:
            self.exit(-9 if force else -15)

    async def close(self) -> None:
        self.closed = True
        self.exit(-9)

    def exit(self, code: int) -> None:
        if not self._code.done():
            self._code.set_result(code)


@pytest.fixture
async def tasks(tmp_path: Path) -> AsyncIterator[BackgroundTasks]:
    registry = BackgroundTasks(tmp_path)
    yield registry
    await registry.close()


def _log(path: Path) -> str:
    return path.read_text(encoding="utf-8")


async def test_adopted_command_logs_its_output_and_exit_code(tasks: BackgroundTasks) -> None:
    process = _Process()
    task = tasks.adopt(process, command="make serve", trigger="timeout")
    assert task is not None and process.sink is not None
    assert tasks.running == [task]

    process.sink(0, b"line one\n")
    process.sink(0, b"\x1b[31mred\x1b[0m\n")
    process.exit(3)
    await wait_for(task.ended.is_set, description="the task ended")

    assert _log(task.log_path) == "before\nline one\nred\n[icode] process exited with code 3\n"
    assert task.exit_code == 3
    assert task.stopped_by is None
    assert process.closed
    assert tasks.running == []


async def test_stop_file_stops_the_task_and_is_removed(tasks: BackgroundTasks) -> None:
    process = _Process()
    task = tasks.adopt(process, command="npm run dev", trigger="timeout")
    assert task is not None
    assert await tasks.stop_requested() == []

    task.stop_path.touch()
    assert await tasks.stop_requested() == [task]

    assert task.ended.is_set()
    assert process.signals == [False]
    assert _log(task.log_path).endswith("[icode] process stopped by agent\n")
    assert task.exit_code is None
    await wait_for(lambda: not task.stop_path.exists(), description="the stop file was removed")


async def test_stop_kills_a_command_that_ignores_it_then_stops_reading(
    tasks: BackgroundTasks, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry_mod, "_TERM_GRACE", 0.01)
    monkeypatch.setattr(registry_mod, "_KILL_GRACE", 0.01)
    process = _Process(ignores=(False, True))
    task = tasks.adopt(process, command="stubborn", trigger="timeout")
    assert task is not None

    await tasks.stop(task, "user")

    assert process.signals == [False, True]
    assert process.closed
    assert _log(task.log_path).endswith("[icode] process stopped by user\n")


async def test_close_stops_every_task_and_takes_no_more(tmp_path: Path) -> None:
    tasks = BackgroundTasks(tmp_path)
    first = tasks.adopt(_Process(), command="one", trigger="timeout")
    second = tasks.adopt(_Process(), command="two", trigger="timeout")
    assert first is not None and second is not None

    await tasks.close()

    for task in (first, second):
        assert task.ended.is_set()
        assert _log(task.log_path).endswith("[icode] process stopped at session end\n")
    assert tasks.adopt(_Process(), command="three", trigger="timeout") is None


async def test_a_full_session_takes_no_more_tasks(tasks: BackgroundTasks) -> None:
    for index in range(registry_mod.MAX_RUNNING_TASKS):
        assert tasks.adopt(_Process(), command=f"task {index}", trigger="timeout") is not None
    assert tasks.full

    process = _Process()
    assert tasks.adopt(process, command="one more", trigger="timeout") is None
    assert process.sink is None


async def test_task_ids_never_repeat_a_used_id_or_an_existing_file(
    tasks: BackgroundTasks, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks.directory.mkdir(parents=True)
    (tasks.directory / "bg-aaaaaa.stop").touch()
    tokens = iter(["aaaaaa", "bbbbbb", "bbbbbb", "cccccc"])
    monkeypatch.setattr(registry_mod, "secrets", SimpleNamespace(token_hex=lambda _n: next(tokens)))

    first = tasks.adopt(_Process(), command="one", trigger="timeout")
    second = tasks.adopt(_Process(), command="two", trigger="timeout")

    assert first is not None and first.id == "bg-bbbbbb"
    assert second is not None and second.id == "bg-cccccc"


async def test_only_the_newest_finished_logs_are_kept(tasks: BackgroundTasks, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_mod, "_KEPT_FINISHED_LOGS", 2)
    tasks.directory.mkdir(parents=True)
    old = time.time() - 100
    for index, name in enumerate(("bg-old001", "bg-old002", "bg-old003")):
        for suffix in (".log", ".log.1"):
            path = tasks.directory / f"{name}{suffix}"
            path.touch()
            os.utime(path, (old + index, old + index))
    running = tasks.adopt(_Process(), command="still running", trigger="timeout")
    process = _Process()
    finished = tasks.adopt(process, command="done", trigger="timeout")
    assert running is not None and finished is not None
    await wait_for(running.log_path.exists, description="the running task's log was written")

    process.exit(0)

    kept = {f"{finished.id}.log", f"{running.id}.log", "bg-old003.log", "bg-old003.log.1"}
    await wait_for(lambda: {path.name for path in tasks.directory.iterdir()} == kept, description="old logs pruned")


@pytest.mark.parametrize(
    ("shell_name", "expected"),
    [
        ("zsh", "touch '/tmp/it'\"'\"'s/bg-1.stop'"),
        ("git_bash", "touch '/tmp/it'\"'\"'s/bg-1.stop'"),
        ("pwsh", "New-Item -ItemType File -Force -Path '/tmp/it''s/bg-1.stop' | Out-Null"),
        ("cmd", 'type nul > "/tmp/it\'s/bg-1.stop"'),
    ],
)
def test_stop_command_quotes_the_path_for_each_shell(shell_name: str, expected: str) -> None:
    assert stop_command(shell_name, Path("/tmp/it's/bg-1.stop")) == expected


async def test_only_the_exact_stop_command_of_a_task_counts(tasks: BackgroundTasks) -> None:
    task = tasks.adopt(_Process(), command="serve", trigger="timeout")
    ended_process = _Process()
    ended = tasks.adopt(ended_process, command="build", trigger="timeout")
    assert task is not None and ended is not None
    ended_process.exit(0)
    await wait_for(ended.ended.is_set, description="the build ended")
    command = stop_command("zsh", task.stop_path)

    assert tasks.is_stop_command(f"  {command}\n", "zsh")
    # The model may not have seen yet that the task ended on its own.
    assert tasks.is_stop_command(stop_command("zsh", ended.stop_path), "zsh")
    assert not tasks.is_stop_command(f"{command} && rm -rf build", "zsh")
    assert not tasks.is_stop_command(command, "pwsh")
    assert not tasks.is_stop_command(stop_command("zsh", tasks.directory / "bg-other.stop"), "zsh")


# ───────────────────────── task log ─────────────────────────


async def _written(log: TaskLog, text: str) -> None:
    await wait_for(lambda: log.path.exists() and _log(log.path) == text, description=f"log reads {text!r}")


async def test_log_merges_streams_line_by_line(tmp_path: Path) -> None:
    log = TaskLog(tmp_path / "bg-1.log")
    log.start(())
    log.feed(0, b"out1\nou")
    log.feed(1, b"err1\n")
    log.feed(0, b"t2\n")
    await log.close("[icode] process exited with code 0")

    assert _log(log.path) == "out1\nerr1\nout2\n[icode] process exited with code 0\n"


async def test_log_keeps_a_character_split_between_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(log_mod, "_FLUSH_DELAY", 0)
    character = "你".encode()
    log = TaskLog(tmp_path / "bg-1.log")
    log.start(())
    log.feed(0, b"a\n" + character[:2])
    await _written(log, "a\n")
    log.feed(0, character[2:] + b"\r\n")
    await log.close("[icode] process exited with code 0")

    assert _log(log.path) == "a\n你\n[icode] process exited with code 0\n"


async def test_log_counts_output_it_could_not_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(log_mod, "_QUEUE_LIMIT", 10)
    log = TaskLog(tmp_path / "bg-1.log")
    log.start(())
    log.feed(0, b"kept\n")
    log.feed(0, b"x" * 20)
    log.feed(0, b"tail\n")
    await log.close("[icode] process stopped by agent")

    assert _log(log.path) == "kept\ntail\n[... 20 bytes of output dropped ...]\n[icode] process stopped by agent\n"


async def test_log_turns_utf16_output_into_utf8_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(log_mod, "subprocess_output_utf16_codec", lambda _raw: "utf-16-le")
    data = "结果: 成功\r\n第二行\r\n".encode("utf-16-le")
    log = TaskLog(tmp_path / "bg-1.log")
    log.start(())
    # Split inside a code unit: each half alone is no valid UTF-16.
    log.feed(0, data[:7])
    log.feed(0, data[7:])
    await log.close("[icode] process exited with code 0")

    assert _log(log.path) == "结果: 成功\n第二行\n[icode] process exited with code 0\n"


async def test_log_rotates_past_its_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(log_mod, "LOG_LIMIT_BYTES", 40)
    log = TaskLog(tmp_path / "bg-1.log")
    log.start((CapturedOutput(b"x" * 30 + b"\n", b"", 31),))
    await _written(log, "x" * 30 + "\n")
    log.feed(0, b"after rotation\n")
    await log.close("[icode] process exited with code 0")

    assert _log(tmp_path / "bg-1.log.1") == "x" * 30 + "\n"
    assert _log(log.path) == "after rotation\n[icode] process exited with code 0\n"
