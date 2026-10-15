# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""What the TUI shows of the main agent's background shell commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackgroundTaskView:
    """One background command of the main agent, as it is at one moment."""

    id: str
    command: str
    reason: str
    cwd: str
    trigger: str
    """Why it moved to the background: "timeout", "idle", "shell_exited" or "user"."""
    turn: int
    """The main agent's turn it started in."""
    pids: tuple[int, ...]
    """The shell's pid, or the processes that outlived the shell."""
    log_path: str
    started_at: float
    """When it moved to the background, in seconds since the epoch."""
    ended_at: float | None = None
    exit_code: int | None = None
    """The shell's exit code, once the command ended with it without being stopped."""
    shell_exit_code: int | None = None
    """The shell's exit code, once it exited while processes it started ran on."""
    stopped_by: str | None = None
    """Who stopped it: "agent", "user", "agent_finished", "rollback" or "session_end"."""

    @property
    def running(self) -> bool:
        return self.ended_at is None
