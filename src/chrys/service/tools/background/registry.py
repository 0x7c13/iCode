# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The shell commands of one session that kept running after their tool call returned.

The registry owns each command from the moment it moves to the background: it
keeps reading the output into the task's log, writes the line that ends the
log, and stops the command when asked — by the stop file the model creates,
or when the session ends. A turn that is cancelled leaves these commands alone.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import secrets
import shlex
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal, Protocol

from chrys.service.tools.background.log import TaskLog

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from pathlib import Path

    from chrys.foundation.platform.output_capture import CapturedOutput

logger = logging.getLogger(__name__)

BACKGROUND_DIR_NAME = "background"
"""The session subdirectory holding each task's log and stop file."""

MAX_RUNNING_TASKS = 8
"""Background commands a session runs at once; past it, a timed-out command is killed."""

STOP_WAIT_SECONDS = 5.0
"""How long a shell call waits for a task it stopped to end before it returns."""

_KEPT_FINISHED_LOGS = 20

_TERM_GRACE = 3.0
"""Seconds between asking a command to stop and killing it."""

_KILL_GRACE = 2.0
"""Seconds a killed command gets to close its output before it is no longer read."""

_STOP_FILE_POLL = 2.0

StopCause = Literal["agent", "user", "session_end"]

_STOPPED_LINES: dict[StopCause, str] = {
    "agent": "[icode] process stopped by agent",
    "user": "[icode] process stopped by user",
    "session_end": "[icode] process stopped at session end",
}


class BackgroundProcess(Protocol):
    """A running command the registry can take over."""

    @property
    def pid(self) -> int: ...

    def hand_off(self, sink: Callable[[int, bytes], None]) -> tuple[CapturedOutput, ...]:
        """Send all further output to *sink*, returning what was kept until now."""
        ...

    async def wait(self) -> int:
        """Wait for the command to end and return its shell's exit code."""
        ...

    async def terminate(self, *, force: bool) -> None:
        """Ask every process of the command to stop, or kill them all when *force* is set."""
        ...

    async def close(self) -> None:
        """Stop reading the output and release the process, killing what still runs."""
        ...


def stop_command(shell_name: str, path: Path) -> str:
    """A command for the shell named *shell_name* that creates the stop file at *path*."""
    if shell_name in ("pwsh", "powershell"):
        literal = str(path).replace("'", "''")
        return f"New-Item -ItemType File -Force -Path '{literal}' | Out-Null"
    if shell_name == "cmd":
        return f'type nul > "{path}"'
    if shell_name == "git_bash":
        return f"touch {shlex.quote(path.as_posix())}"
    return f"touch {shlex.quote(str(path))}"


class BackgroundTask:
    """One command running in the background, from its hand-off to its end."""

    def __init__(
        self, task_id: str, command: str, trigger: str, process: BackgroundProcess, log: TaskLog, stop_path: Path
    ) -> None:
        self.id = task_id
        self.command = command
        self.trigger = trigger
        self.pid = process.pid
        self.log_path = log.path
        self.stop_path = stop_path
        self.exit_code: int | None = None
        """The shell's exit code, once the command ended without being stopped."""
        self.stopped_by: StopCause | None = None
        self.ended = asyncio.Event()
        """Set once the command ended and its log was closed."""
        self.process = process
        self.log = log
        self.stopping: asyncio.Task[None] | None = None


class BackgroundTasks:
    """The background commands of one session."""

    def __init__(self, session_dir: Path) -> None:
        self.directory = session_dir / BACKGROUND_DIR_NAME
        self._running: dict[str, BackgroundTask] = {}
        self._used_ids: set[str] = set()
        self._workers: set[asyncio.Task[None]] = set()
        """Each task's watcher and stopper; closing waits for them all."""
        self._sampler: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def running(self) -> list[BackgroundTask]:
        return list(self._running.values())

    @property
    def full(self) -> bool:
        return len(self._running) >= MAX_RUNNING_TASKS

    def adopt(self, process: BackgroundProcess, *, command: str, trigger: str) -> BackgroundTask | None:
        """Take *process* over, or return None when the session takes no more or has no log for it.

        Synchronous, so the output *process* kept just before this call is
        exactly what the log starts with.
        """
        if self._closed or self.full:
            return None
        task_id = self._new_id()
        log = TaskLog(self.directory / f"{task_id}.log")
        try:
            log.open()
        except OSError:
            logger.warning("Could not create the log of background task %s", task_id, exc_info=True)
            return None
        log.start(process.hand_off(log.feed))
        task = BackgroundTask(task_id, command, trigger, process, log, self._stop_path(task_id))
        self._running[task_id] = task
        self._spawn(self._watch(task), self._workers)
        if self._sampler is None:
            self._sampler = self._spawn(self._sample())
        return task

    def is_stop_command(self, command: str, shell_name: str) -> bool:
        """Whether *command* is exactly the command that stops one of this session's tasks, ended ones too."""
        command = command.strip()
        return any(command == stop_command(shell_name, self._stop_path(task_id)) for task_id in self._used_ids)

    async def stop_requested(self) -> list[BackgroundTask]:
        """Stop the tasks whose stop file exists and wait a little for them to end."""
        tasks = self._apply_stop_files()
        if tasks:
            await asyncio.wait([self.stop(task, "agent") for task in tasks], timeout=STOP_WAIT_SECONDS)
        return tasks

    def stop(self, task: BackgroundTask, by: StopCause) -> asyncio.Task[None]:
        """Start stopping *task*, unless that already began; the returned task ends once *task* did."""
        if task.stopping is None:
            if task.stopped_by is None and not task.ended.is_set():
                task.stopped_by = by
            task.stopping = self._spawn(self._stop(task), self._workers)
        return task.stopping

    async def close(self) -> None:
        """Stop every task at session end and take no more."""
        self._closed = True
        if self._sampler is not None:
            self._sampler.cancel()
        for task in self.running:
            self.stop(task, "session_end")
        if self._workers:
            await asyncio.wait(set(self._workers))

    def _stop_path(self, task_id: str) -> Path:
        return self.directory / f"{task_id}.stop"

    def _new_id(self) -> str:
        # Unique among the files already here too: an old result naming bg-x
        # must never reach a later task.
        while True:
            task_id = f"bg-{secrets.token_hex(3)}"
            if task_id in self._used_ids:
                continue
            names = (f"{task_id}.log", f"{task_id}.log.1", f"{task_id}.stop")
            if not any((self.directory / name).exists() for name in names):
                self._used_ids.add(task_id)
                return task_id

    def _apply_stop_files(self) -> list[BackgroundTask]:
        tasks = [task for task in self._running.values() if task.stop_path.exists()]
        for task in tasks:
            self.stop(task, "agent")
        return tasks

    async def _sample(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(_STOP_FILE_POLL)
                self._apply_stop_files()
        finally:
            self._sampler = None

    async def _stop(self, task: BackgroundTask) -> None:
        for force, grace in ((False, _TERM_GRACE), (True, _KILL_GRACE)):
            try:
                await task.process.terminate(force=force)
            except Exception:
                logger.debug("Could not signal background task %s", task.id, exc_info=True)
            with suppress(TimeoutError):
                await asyncio.wait_for(task.ended.wait(), grace)
                return
        # A process outside the command's reach still holds the output open: stop reading it.
        await task.process.close()
        await task.ended.wait()

    async def _watch(self, task: BackgroundTask) -> None:
        try:
            code = await task.process.wait()
            if task.stopped_by is None:
                task.exit_code = code
                last_line = f"[icode] process exited with code {code}"
            else:
                last_line = _STOPPED_LINES[task.stopped_by]
            await task.log.close(last_line)
        finally:
            await task.process.close()
            self._running.pop(task.id, None)
            task.ended.set()
        await asyncio.to_thread(_tidy, self.directory, task.stop_path, set(self._running))

    def _spawn[T](self, coro: Coroutine[Any, Any, T], owner: set[asyncio.Task[T]] | None = None) -> asyncio.Task[T]:
        # A fresh context: the task belongs to the session, not to the tool call that started it.
        task = asyncio.get_running_loop().create_task(coro, context=contextvars.Context())
        if owner is not None:
            owner.add(task)
            task.add_done_callback(owner.discard)
        return task


def _tidy(directory: Path, stop_path: Path, running: set[str]) -> None:
    """Remove an ended task's stop file and all but the newest finished logs."""
    with suppress(OSError):
        stop_path.unlink(missing_ok=True)
    try:
        logs = [log for log in directory.glob("*.log") if log.stem not in running]
    except OSError:
        return
    logs.sort(key=_modified, reverse=True)
    for log in logs[_KEPT_FINISHED_LOGS:]:
        for path in (log, log.with_name(f"{log.name}.1"), log.with_suffix(".stop")):
            with suppress(OSError):
                path.unlink(missing_ok=True)


def _modified(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
