# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the git branch shown beside the workspace cwd."""

from __future__ import annotations

import asyncio
import gc
from collections.abc import Awaitable, Callable

from pytest import WarningsRecorder

from chrys.app.tui.screens.main.state import WorkspaceViewState
from chrys.app.tui.screens.main.workspace_branch import WorkspaceBranchController
from chrys.app.tui.util.git_branch import GIT_BRANCH_POLL_INTERVAL_SECONDS, GitBranchSnapshot
from tests.support.waiting import wait_for

type _RunOperation = Callable[[str, str], Awaitable[GitBranchSnapshot | None]]


class _Timer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _Monitor:
    def __init__(self, *, active: bool = True, watching: bool = False) -> None:
        self.active = active
        self.watching = watching
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _Harness:
    """A controller over a fake monitor, timers and cwd sources the test moves."""

    def __init__(
        self,
        run_operation: _RunOperation | None = None,
        *,
        current_cwd: str = "",
        displayed_cwd: str | None = None,
        monitor: _Monitor | None = None,
    ) -> None:
        self.current_cwd = current_cwd
        self.displayed_cwd = displayed_cwd if displayed_cwd is not None else current_cwd
        self.shown: list[str] = []
        self.timers: list[tuple[float, Callable[[], None], _Timer]] = []
        self.intervals: list[tuple[float, Callable[[], None], _Timer]] = []
        self.workspace = WorkspaceViewState(current_cwd=current_cwd)
        self.monitor = monitor or _Monitor()
        self.controller = WorkspaceBranchController(
            workspace=self.workspace,
            workspace_cwd=lambda: self.current_cwd,
            displayed_cwd=lambda: self.displayed_cwd,
            show_branch=self.shown.append,
            set_timer=self._set_timer,
            set_interval=self._set_interval,
            call_from_thread=lambda callback: callback(),
        )
        # The real monitor runs git on worker threads; these tests drive the
        # queue with a scripted monitor and operation instead.
        self.controller._monitor = self.monitor  # type: ignore[assignment]
        if run_operation is not None:
            self.controller._run_operation = run_operation  # type: ignore[method-assign]

    def _set_timer(self, delay: float, callback: Callable[[], None]) -> _Timer:
        timer = _Timer()
        self.timers.append((delay, callback, timer))
        return timer

    def _set_interval(self, delay: float, callback: Callable[[], None]) -> _Timer:
        timer = _Timer()
        self.intervals.append((delay, callback, timer))
        return timer

    def move_to(self, cwd: str) -> None:
        self.current_cwd = cwd
        self.displayed_cwd = cwd

    async def drained(self) -> None:
        task = self.controller._task
        if task is not None:
            await task

    async def operation_started(self, started: asyncio.Event) -> None:
        """Wait for the queued operation to reach *started*, surfacing its error if it ends first."""
        task = self.controller._task
        assert task is not None, "no git branch operation was queued"
        await wait_for(lambda: started.is_set() or task.done(), description="git branch operation started")
        if task.done():
            await task
        assert started.is_set()


def _snapshot_for_operation(calls: list[tuple[str, str]]) -> _RunOperation:
    async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
        calls.append((operation, cwd))
        return GitBranchSnapshot(cwd=cwd, branch=f"{operation}-branch")

    return run_operation


def test_schedule_refresh_debounces_the_existing_timer() -> None:
    harness = _Harness()
    controller = harness.controller

    controller.schedule_refresh()
    controller.schedule_refresh()

    assert [delay for delay, _callback, _timer in harness.timers] == [0.1, 0.1]
    assert harness.timers[0][2].stopped is True
    assert harness.timers[1][2].stopped is False


def test_schedule_refresh_is_ignored_until_the_monitor_is_active() -> None:
    harness = _Harness(monitor=_Monitor(active=False))

    harness.controller.schedule_refresh()

    assert harness.timers == []


def test_file_change_notifications_schedule_a_refresh_on_the_loop_thread() -> None:
    hops: list[Callable[[], None]] = []
    harness = _Harness()
    harness.controller._call_from_thread = hops.append

    harness.controller._on_files_changed()

    assert harness.timers == []
    hops[0]()
    assert len(harness.timers) == 1


def test_refresh_timer_queues_a_refresh() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        harness = _Harness(_snapshot_for_operation(calls), current_cwd="/repo")
        harness.controller.schedule_refresh()

        _delay, refresh, _timer = harness.timers[0]
        refresh()
        await harness.drained()

        assert calls == [("refresh", "")]
        assert harness.controller._refresh_timer is None

    asyncio.run(run())


def test_poll_timer_stops_while_the_native_watcher_runs_and_restarts_when_it_stops() -> None:
    async def run() -> None:
        monitor = _Monitor(watching=False)
        harness = _Harness(_snapshot_for_operation([]), current_cwd="/repo", monitor=monitor)
        harness.controller.configure("/repo")
        await harness.drained()
        assert [delay for delay, _callback, _timer in harness.intervals] == [GIT_BRANCH_POLL_INTERVAL_SECONDS]

        monitor.watching = True
        harness.controller.configure("/repo")
        await harness.drained()

        assert harness.intervals[0][2].stopped is True
        assert len(harness.intervals) == 1

        monitor.watching = False
        harness.controller.configure("/repo")
        await harness.drained()

        assert [timer.stopped for _delay, _callback, timer in harness.intervals] == [True, False]

    asyncio.run(run())


def test_poll_timer_refreshes_while_the_native_watcher_is_unavailable() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        harness = _Harness(_snapshot_for_operation(calls), current_cwd="/repo")
        harness.controller.configure("/repo")
        await harness.drained()

        _delay, poll, _timer = harness.intervals[0]
        poll()
        await harness.drained()

        assert calls == [("configure", "/repo"), ("refresh", "")]
        assert len(harness.intervals) == 1

    asyncio.run(run())


def test_queue_without_running_loop_keeps_pending_operation_without_coroutine_warning(
    recwarn: WarningsRecorder,
) -> None:
    harness = _Harness(_snapshot_for_operation([]), current_cwd="/repo")

    harness.controller.configure("/repo")
    gc.collect()

    assert harness.controller._task is None
    assert harness.controller._pending == ("configure", "/repo")
    runtime_warnings = [warning for warning in recwarn if issubclass(warning.category, RuntimeWarning)]
    assert runtime_warnings == []


def test_configure_same_cwd_does_not_overwrite_pending_start() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        harness = _Harness(_snapshot_for_operation(calls), current_cwd="/repo")

        harness.controller.start("/repo")
        harness.controller.configure("/repo")
        await harness.drained()

        assert calls == [("start", "/repo")]
        assert harness.shown == ["start-branch"]

    asyncio.run(run())


def test_configure_new_cwd_preserves_pending_start_semantics() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        harness = _Harness(_snapshot_for_operation(calls), current_cwd="/repo-a")

        harness.controller.start("/repo-a")
        harness.move_to("/repo-b")
        harness.controller.configure("/repo-b")
        await harness.drained()

        assert calls == [("start", "/repo-b")]
        assert harness.shown == ["start-branch"]

    asyncio.run(run())


def test_queue_drops_stale_configure_before_applying_newer_cwd() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            calls.append((operation, cwd))
            if cwd == "/repo-a":
                first_started.set()
                await release_first.wait()
                return GitBranchSnapshot(cwd=cwd, branch="branch-a")
            return GitBranchSnapshot(cwd=cwd, branch="branch-b")

        harness = _Harness(run_operation, current_cwd="/repo-a")

        harness.controller.configure("/repo-a")
        await harness.operation_started(first_started)
        harness.move_to("/repo-b")
        harness.controller.configure("/repo-b")
        release_first.set()
        await harness.drained()

        assert calls == [("configure", "/repo-a"), ("configure", "/repo-b")]
        assert harness.shown == ["branch-b"]
        assert harness.workspace.current_git_branch == "branch-b"
        assert len(harness.intervals) == 1
        assert harness.controller._task is None

    asyncio.run(run())


def test_queue_skips_refresh_when_configure_is_pending() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        harness = _Harness(_snapshot_for_operation(calls), current_cwd="/repo")

        harness.controller.configure("/repo")
        harness.controller._poll()
        await harness.drained()

        assert calls == [("configure", "/repo")]
        assert harness.shown == ["configure-branch"]

    asyncio.run(run())


def test_queue_drops_stale_refresh_when_workspace_changes() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        refresh_started = asyncio.Event()
        release_refresh = asyncio.Event()

        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            calls.append((operation, cwd))
            if operation == "refresh":
                refresh_started.set()
                await release_refresh.wait()
                return GitBranchSnapshot(cwd="/repo-a", branch="old-branch")
            return GitBranchSnapshot(cwd=cwd, branch="new-branch")

        harness = _Harness(run_operation, current_cwd="/repo-a")

        harness.controller._poll()
        await harness.operation_started(refresh_started)
        harness.move_to("/repo-b")
        harness.controller.configure("/repo-b")
        release_refresh.set()
        await harness.drained()

        assert calls == [("refresh", ""), ("configure", "/repo-b")]
        assert harness.shown == ["new-branch"]

    asyncio.run(run())


def test_queue_drops_result_when_displayed_cwd_differs() -> None:
    async def run() -> None:
        harness = _Harness(_snapshot_for_operation([]), current_cwd="/real/repo", displayed_cwd="/repo/chrys")

        harness.controller.configure("/real/repo")
        await harness.drained()

        assert harness.shown == []
        assert harness.intervals == []

    asyncio.run(run())


def test_queue_retries_when_displayed_cwd_catches_up() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []

        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            calls.append((operation, cwd))
            return GitBranchSnapshot(cwd=cwd, branch="restored-branch")

        harness = _Harness(run_operation, current_cwd="/restored/repo", displayed_cwd="/old/repo")
        harness.workspace.current_git_branch = "old-branch"

        harness.controller.configure("/restored/repo")
        await harness.drained()

        assert calls == [("configure", "/restored/repo")]
        assert harness.shown == []
        assert harness.controller._retry_cwd_on_display_sync == "/restored/repo"

        harness.displayed_cwd = "/restored/repo"
        harness.controller.displayed_cwd_changed("/old/repo", "/restored/repo")
        await harness.drained()

        assert calls == [("configure", "/restored/repo"), ("configure", "/restored/repo")]
        assert harness.shown == ["", "restored-branch"]
        assert len(harness.intervals) == 1
        assert harness.controller._retry_cwd_on_display_sync is None

    asyncio.run(run())


def test_queue_does_not_duplicate_configure_when_displayed_cwd_catches_up_before_result() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        operation_started = asyncio.Event()
        release_operation = asyncio.Event()

        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            calls.append((operation, cwd))
            operation_started.set()
            await release_operation.wait()
            return GitBranchSnapshot(cwd=cwd, branch="restored-branch")

        harness = _Harness(run_operation, current_cwd="/restored/repo", displayed_cwd="/old/repo")
        harness.workspace.current_git_branch = "old-branch"

        harness.controller.configure("/restored/repo")
        await harness.operation_started(operation_started)

        harness.displayed_cwd = "/restored/repo"
        harness.controller.displayed_cwd_changed("/old/repo", "/restored/repo")
        release_operation.set()
        await harness.drained()

        assert calls == [("configure", "/restored/repo")]
        assert harness.shown == ["", "restored-branch"]
        assert harness.controller._retry_cwd_on_display_sync is None

    asyncio.run(run())


def test_displayed_cwd_change_does_not_reconfigure_for_unmatched_displayed_cwd() -> None:
    harness = _Harness(current_cwd="/real/repo")
    harness.workspace.current_git_branch = "branch"
    harness.controller._retry_cwd_on_display_sync = "/repo/chrys"

    harness.controller.displayed_cwd_changed("/real/repo", "/repo/chrys")

    assert harness.shown == [""]
    assert harness.controller._pending is None


def test_displayed_cwd_change_reconfigures_when_monitor_not_yet_active() -> None:
    harness = _Harness(current_cwd="/restored/repo", monitor=_Monitor(active=False))
    harness.workspace.current_git_branch = "branch"
    harness.controller._retry_cwd_on_display_sync = "/restored/repo"

    harness.controller.displayed_cwd_changed("/old/repo", "/restored/repo")

    assert harness.shown == [""]
    assert harness.controller._pending == ("configure", "/restored/repo")


def test_queue_drops_result_after_close() -> None:
    async def run() -> None:
        operation_started = asyncio.Event()
        release_operation = asyncio.Event()

        async def run_operation(_operation: str, cwd: str) -> GitBranchSnapshot:
            operation_started.set()
            await release_operation.wait()
            return GitBranchSnapshot(cwd=cwd, branch="late-branch")

        harness = _Harness(run_operation, current_cwd="/repo")

        harness.controller.configure("/repo")
        await harness.operation_started(operation_started)
        close_task = asyncio.create_task(harness.controller.close())
        await asyncio.sleep(0)
        assert harness.controller._closed is True
        release_operation.set()
        await close_task

        assert harness.shown == []
        assert harness.intervals == []
        assert harness.controller._task is None

    asyncio.run(run())


def test_close_waits_for_in_flight_operation_before_stopping_monitor() -> None:
    async def run() -> None:
        order: list[str] = []
        task_started = asyncio.Event()
        release_task = asyncio.Event()

        class _OrderedMonitor(_Monitor):
            def stop(self) -> None:
                order.append("stop")

        async def run_operation(_operation: str, cwd: str) -> GitBranchSnapshot:
            order.append("task-start")
            task_started.set()
            await release_task.wait()
            order.append("task-end")
            return GitBranchSnapshot(cwd=cwd, branch="branch")

        harness = _Harness(run_operation, current_cwd="/repo", monitor=_OrderedMonitor())
        harness.controller.configure("/repo")
        await harness.operation_started(task_started)
        harness.controller._poll()

        close_task = asyncio.create_task(harness.controller.close())
        await asyncio.sleep(0)

        assert order == ["task-start"]
        assert harness.controller._closed is True
        assert harness.controller._pending is None

        release_task.set()
        await close_task

        assert order == ["task-start", "task-end", "stop"]
        assert harness.controller._task is None

    asyncio.run(run())


def test_close_stops_pending_timers_and_ignores_later_requests() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        harness = _Harness(_snapshot_for_operation(calls), current_cwd="/repo")
        harness.controller.configure("/repo")
        await harness.drained()
        harness.controller.schedule_refresh()

        await harness.controller.close()
        harness.controller.configure("/repo")
        harness.controller.schedule_refresh()

        assert harness.timers[0][2].stopped is True
        assert harness.intervals[0][2].stopped is True
        assert len(harness.timers) == 1
        assert harness.monitor.stopped is True
        assert calls == [("configure", "/repo")]

    asyncio.run(run())


def test_start_reopens_a_closed_controller() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        harness = _Harness(_snapshot_for_operation(calls), current_cwd="/repo")
        await harness.controller.close()

        harness.controller.start("/repo")
        await harness.drained()

        assert calls == [("start", "/repo")]
        assert harness.shown == ["start-branch"]

    asyncio.run(run())
