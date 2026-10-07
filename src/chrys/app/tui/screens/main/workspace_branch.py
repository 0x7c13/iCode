# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The git branch shown beside the workspace cwd."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from chrys.app.tui.util.git_branch import GIT_BRANCH_POLL_INTERVAL_SECONDS, GitBranchMonitor, GitBranchSnapshot

if TYPE_CHECKING:
    from textual.timer import Timer

    from chrys.app.tui.screens.main.state import WorkspaceViewState

logger = logging.getLogger(__name__)

_REFRESH_DEBOUNCE_SECONDS = 0.1

type _Operation = Literal["start", "configure", "refresh"]


class WorkspaceBranchController:
    """Keep the displayed git branch in step with the workspace cwd.

    Monitor calls run off the event loop one at a time; requests made while one
    is in flight collapse into a single pending operation. A snapshot applies
    only while its cwd is still both the workspace cwd and the displayed cwd.
    When the display lags the workspace (a restore moves the workspace before
    the chat border follows), the snapshot is dropped and read again once the
    display catches up. File-change notifications from the monitor's watcher
    are debounced; without a native watcher, a poll timer refreshes instead.
    """

    def __init__(
        self,
        *,
        workspace: WorkspaceViewState,
        workspace_cwd: Callable[[], str],
        displayed_cwd: Callable[[], str],
        show_branch: Callable[[str], None],
        set_timer: Callable[[float, Callable[[], None]], Timer],
        set_interval: Callable[[float, Callable[[], None]], Timer],
        call_from_thread: Callable[[Callable[[], None]], object],
    ) -> None:
        self._workspace = workspace
        self._workspace_cwd = workspace_cwd
        self._displayed_cwd = displayed_cwd
        self._show_branch = show_branch
        self._set_timer = set_timer
        self._set_interval = set_interval
        self._call_from_thread = call_from_thread
        self._monitor = GitBranchMonitor(self._on_files_changed)
        self._refresh_timer: Timer | None = None
        self._poll_timer: Timer | None = None
        self._task: asyncio.Task[None] | None = None
        self._pending: tuple[_Operation, str] | None = None
        self._retry_cwd_on_display_sync: str | None = None
        self._closed = False

    def start(self, cwd: str) -> None:
        """Begin monitoring *cwd* (screen mount)."""
        self._closed = False
        self._queue("start", cwd)

    def configure(self, cwd: str) -> None:
        """Follow a workspace cwd change."""
        self._queue("configure", cwd)

    def displayed_cwd_changed(self, old_cwd: str, cwd: str) -> None:
        """Clear the stale branch and re-read one dropped while the display lagged."""
        if old_cwd == cwd:
            return
        self._show(branch="")
        # Retry only after a snapshot was dropped because the displayed cwd
        # lagged the workspace cwd. Normal cwd updates must not force a second
        # git read while the first configure can still apply.
        if cwd and cwd == self._workspace_cwd() and cwd == self._retry_cwd_on_display_sync and not self._closed:
            self.configure(cwd)

    def schedule_refresh(self) -> None:
        """Re-read the branch shortly, coalescing bursts of change notifications."""
        if not self._monitor.active or self._closed:
            return
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
        self._refresh_timer = self._set_timer(_REFRESH_DEBOUNCE_SECONDS, self._refresh)

    async def close(self) -> None:
        """Stop the timers, wait out the operation in flight, then stop the monitor."""
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None
        self._stop_poll_timer()
        self._closed = True
        self._pending = None
        task = self._task
        if task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self._task = None
        await asyncio.to_thread(self._monitor.stop)

    def _on_files_changed(self) -> None:
        # Called on the monitor's watcher thread.
        with contextlib.suppress(Exception):
            self._call_from_thread(self.schedule_refresh)

    def _refresh(self) -> None:
        self._refresh_timer = None
        self._poll()

    def _poll(self) -> None:
        if self._monitor.active and not self._closed:
            self._queue("refresh", "")

    def _queue(self, operation: _Operation, cwd: str) -> None:
        if self._closed:
            return
        if operation != "refresh":
            self._retry_cwd_on_display_sync = None
        if self._pending is not None:
            pending_operation, pending_cwd = self._pending
            if operation == "refresh" and pending_operation != "refresh":
                return
            if operation == "configure" and pending_operation == "start":
                if pending_cwd == cwd:
                    return
                operation = "start"
        self._pending = (operation, cwd)
        task = self._task
        if task is not None and not task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("no running loop for git branch refresh task", exc_info=True)
            return
        self._task = loop.create_task(self._drain())

    async def _drain(self) -> None:
        try:
            while self._pending is not None and not self._closed:
                operation, cwd = self._pending
                self._pending = None
                snapshot = await self._run_operation(operation, cwd)
                if snapshot is None or self._closed:
                    continue
                if snapshot.cwd != self._workspace_cwd():
                    continue
                if snapshot.cwd != self._displayed_cwd():
                    self._retry_cwd_on_display_sync = snapshot.cwd
                    continue
                self._retry_cwd_on_display_sync = None
                with contextlib.suppress(Exception):
                    self._show(snapshot.branch)
                    self._sync_poll_timer()
        finally:
            if self._task is asyncio.current_task():
                self._task = None

    async def _run_operation(self, operation: _Operation, cwd: str) -> GitBranchSnapshot | None:
        try:
            if operation == "start":
                return await asyncio.to_thread(self._monitor.start, cwd)
            if operation == "configure":
                return await asyncio.to_thread(self._monitor.configure, cwd)
            return await asyncio.to_thread(self._monitor.refresh)
        except Exception:
            logger.debug("git branch monitor operation failed: %s", operation, exc_info=True)
            return None

    def _show(self, branch: str) -> None:
        if branch == self._workspace.current_git_branch:
            return
        self._workspace.current_git_branch = branch
        self._show_branch(branch)

    def _sync_poll_timer(self) -> None:
        if not self._monitor.active or self._closed:
            return
        if self._monitor.watching:
            self._stop_poll_timer()
        elif self._poll_timer is None:
            self._poll_timer = self._set_interval(GIT_BRANCH_POLL_INTERVAL_SECONDS, self._poll)

    def _stop_poll_timer(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None
