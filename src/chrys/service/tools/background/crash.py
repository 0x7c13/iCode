# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The background commands a launch of iCode left running when it ended without stopping them.

While a session has background commands, its registry keeps a record naming
them in the session's background folder, and removes it once none runs. The
launch holds a lock beside the record while it exists, which the system
releases when the launch ends in any way; a record whose lock is free was left
behind. A later launch finds it, and stops the commands still running or lets
them be.

Only on POSIX: on Windows a command's job ends with the process holding it.
A command counts as still running only while one of the processes recorded as
it moved to the background — same pid, same creation time — is still in its
session; anything else may be an unrelated process that reuses the number, also
after a reboot. A record that proves nothing is removed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chrys.foundation.platform.files import atomic_write_text
from chrys.foundation.platform.process import session_processes, signal_process_session
from chrys.foundation.trajectory.keys import ensure_owner_only_directory
from chrys.foundation.util.lock import FileLock

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from chrys.service.tools.background.registry import BackgroundTask

logger = logging.getLogger(__name__)

_RECORD_PREFIX = "running-"

_SAME_START = 2.0
"""Seconds two readings of a process's creation time may differ by: it follows the wall clock, which gets corrected."""

_TERM_GRACE = 3.0
"""Seconds between asking the commands to stop and killing them, as the registry gives its own."""

_POLL = 0.1


@dataclass(frozen=True, slots=True)
class LeftCommand:
    """A command a launch that is gone left running."""

    command: str
    cwd: str
    session: int
    """The command's POSIX session, which carries its shell's pid."""
    members: tuple[tuple[int, float], ...]
    """Its processes as it moved to the background: pid and creation time."""


@dataclass(frozen=True, slots=True)
class LeftBehind:
    """The commands one record names that still run."""

    record: Path
    commands: tuple[LeftCommand, ...]


@dataclass(frozen=True, slots=True)
class _Record:
    surface: str
    commands: tuple[LeftCommand, ...]


class RunningRecord:
    """The record of one registry's running commands, kept for the launch after a crash."""

    def __init__(self, directory: Path, *, surface: str) -> None:
        self.path = _new_record_path(directory)
        self._surface = surface
        self._held: FileLock | None = None
        self._members: dict[str, tuple[tuple[int, float], ...]] = {}
        self._named: frozenset[str] = frozenset()
        self._lock = asyncio.Lock()

    async def update(self, running: Iterable[BackgroundTask]) -> None:
        """Name the *running* commands, or remove the record when none runs."""
        if sys.platform == "win32":
            return
        tasks = {task.id: task for task in running}
        async with self._lock:
            if tasks.keys() == self._named:
                return
            for task in tasks.values():
                if task.id not in self._members:
                    self._members[task.id] = await asyncio.to_thread(_members, task.pid)
            self._members = {task_id: self._members[task_id] for task_id in tasks}
            entries: list[dict[str, object]] = [
                {
                    "command": task.command,
                    "cwd": task.cwd,
                    "session": task.pid,
                    "members": self._members[task.id],
                }
                for task in tasks.values()
                if self._members[task.id]
            ]
            try:
                await asyncio.to_thread(self._write, entries)
            except OSError:
                logger.warning("Could not update %s", self.path, exc_info=True)
                return
            self._named = frozenset(tasks)

    def _write(self, entries: list[dict[str, object]]) -> None:
        # The lock outlives the record on both ends, so a record never looks left behind while this launch runs.
        if not entries:
            self.path.unlink(missing_ok=True)
            if self._held is not None:
                self._held.release()
                self._held = None
                _lock_path(self.path).unlink(missing_ok=True)
            return
        if self._held is None:
            # A new name each time: a launch checking the old one may be about to remove it.
            self.path = _new_record_path(self.path.parent)
            ensure_owner_only_directory(self.path.parent)
            held = FileLock(_lock_path(self.path), timeout=0)
            held.acquire()
            self._held = held
        atomic_write_text(self.path, json.dumps({"surface": self._surface, "tasks": entries}))


def find_left_behind(sessions_dir: Path, *, surface: str | None = None) -> list[LeftBehind]:
    """The commands still running that launches now gone left under *sessions_dir*; on *surface* only, if given.

    Records of launches still running are left alone; records that prove
    nothing are removed. Blocking: run it in a thread.
    """
    if sys.platform == "win32":
        return []
    # Here, not at the top: the registry imports this module.
    from chrys.service.tools.background.registry import BACKGROUND_DIR_NAME

    try:
        paths = sorted(sessions_dir.glob(f"*/{BACKGROUND_DIR_NAME}/{_RECORD_PREFIX}*.json"))
    except OSError:
        return []
    found: list[LeftBehind] = []
    for path in paths:
        record = _read(path)
        if (record is not None and surface not in (None, record.surface)) or _launch_runs(path):
            continue
        commands = ()
        if record is not None:
            commands = tuple(command for command in record.commands if _still_runs(command))
        if commands:
            found.append(LeftBehind(path, commands))
        else:
            _remove(path)
    return found


def stop_left_behind(left: Iterable[LeftBehind]) -> None:
    """Stop the commands of *left* still running, then forget them. Blocking: run it in a thread."""
    left = list(left)
    if sys.platform != "win32":
        import signal

        running = {command.session for record in left for command in record.commands if _still_runs(command)}
        for session in running:
            signal_process_session(session, signal.SIGTERM)
        deadline = time.monotonic() + _TERM_GRACE
        # A session once empty stays out: its number may already name another.
        while (running := {session for session in running if session_processes(session)}) and (
            time.monotonic() < deadline
        ):
            time.sleep(_POLL)
        for session in running:
            signal_process_session(session, signal.SIGKILL)
    forget_left_behind(left)


def forget_left_behind(left: Iterable[LeftBehind]) -> None:
    """Remove the records of *left*, leaving their commands running. Blocking."""
    for record in left:
        _remove(record.record)


def _members(session: int) -> tuple[tuple[int, float], ...]:
    """The processes of the POSIX session *session*, each with its creation time."""
    if sys.platform == "win32":
        return ()
    import psutil

    found: list[tuple[int, float]] = []
    for pid in psutil.pids():
        try:
            if os.getsid(pid) == session:
                found.append((pid, psutil.Process(pid).create_time()))
        except OSError, psutil.Error:
            continue
    return tuple(found)


def _new_record_path(directory: Path) -> Path:
    return directory / f"{_RECORD_PREFIX}{secrets.token_hex(4)}.json"


def _lock_path(record: Path) -> Path:
    return record.with_suffix(".lock")


def _launch_runs(record: Path) -> bool:
    """Whether the launch that wrote *record* still holds its lock."""
    try:
        with FileLock(_lock_path(record), timeout=0):
            return False
    except OSError:  # Held (TimeoutError), or not to be checked: let it be.
        return True


def _still_runs(command: LeftCommand) -> bool:
    """Whether a process recorded for *command* still runs in its session: the session is still its."""
    if sys.platform == "win32":
        return False
    import psutil

    for pid, created in command.members:
        try:
            if abs(psutil.Process(pid).create_time() - created) < _SAME_START and os.getsid(pid) == command.session:
                return True
        except OSError, psutil.Error:
            continue
    return False


def _read(path: Path) -> _Record | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _Record(
            surface=str(data["surface"]),
            commands=tuple(
                LeftCommand(
                    command=str(task["command"]),
                    cwd=str(task["cwd"]),
                    session=int(task["session"]),
                    members=tuple((int(pid), float(created)) for pid, created in task["members"]),
                )
                for task in data["tasks"]
            ),
        )
    except OSError, ValueError, KeyError, TypeError, IndexError:
        return None


def _remove(record: Path) -> None:
    for path in (record, _lock_path(record)):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Could not remove %s", path, exc_info=True)
