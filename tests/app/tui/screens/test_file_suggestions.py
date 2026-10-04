# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for @ file suggestions: cold scans, index builds, debounced queries, and their races."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field

import pytest

from chrys.app.tui.screens.main import suggestions as suggestions_module
from chrys.app.tui.screens.main.suggestions import SuggestionHandler
from chrys.app.tui.widgets.chrome.file_scanner import ProjectPathScanResult, ProjectPathSuggestion
from chrys.app.tui.widgets.chrome.suggestion_list import SuggestionItem
from tests.support.tui_helpers import (
    DeferredWorker,
    SuggestionScreen,
    make_suggestion_handler,
    make_suggestion_screen,
    scan_result,
    suggestion_values,
    wait_for_file_query,
)

_SCAN_TARGET = "chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths"

type _ScanFake = Callable[[str], Awaitable[ProjectPathScanResult]]


def _patch_scan(
    monkeypatch: pytest.MonkeyPatch,
    source: Mapping[str, Sequence[str]] | _ScanFake,
    *,
    roots: list[str] | None = None,
) -> None:
    """Point the project-path scanner at a fake.

    ``source`` is either a coroutine function taking the scan root, or a
    mapping from root to the file paths that root yields. Every scanned root
    is appended to ``roots`` when one is supplied.
    """
    if isinstance(source, Mapping):
        paths_by_root = source

        async def scan(root: str) -> ProjectPathScanResult:
            return scan_result(root, [ProjectPathSuggestion(path=path, kind="file") for path in paths_by_root[root]])
    else:
        scan = source

    async def record(root: str) -> ProjectPathScanResult:
        if roots is not None:
            roots.append(root)
        return await scan(root)

    monkeypatch.setattr(_SCAN_TARGET, record)


@dataclass(slots=True)
class _ColdScanGate:
    """A cold scan held open on the event loop until the test releases it."""

    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)
    roots: list[str] = field(default_factory=list)


def _hold_cold_scan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    paths: Sequence[str] = ("stale.py",),
) -> _ColdScanGate:
    """Block the cold scan until ``gate.release`` is set."""
    gate = _ColdScanGate()

    async def scan(root: str) -> ProjectPathScanResult:
        gate.started.set()
        await gate.release.wait()
        return scan_result(root, [ProjectPathSuggestion(path=path, kind="file") for path in paths])

    _patch_scan(monkeypatch, scan, roots=gate.roots)
    return gate


@dataclass(slots=True)
class _IndexBuildGate:
    """An index build paused on its worker thread until the test releases it."""

    started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    builds: int = 0


def _hold_index_build(
    monkeypatch: pytest.MonkeyPatch,
    handler: SuggestionHandler,
    *,
    first_build_only: bool = False,
) -> _IndexBuildGate:
    """Block the off-loop index build until ``gate.release`` is set."""
    gate = _IndexBuildGate()
    original_build = handler._build_file_index

    def build(scan: ProjectPathScanResult):
        gate.builds += 1
        if not first_build_only or gate.builds == 1:
            gate.started.set()
            gate.release.wait(timeout=1)
        return original_build(scan)

    monkeypatch.setattr(handler, "_build_file_index", build)
    return gate


def test_file_suggestions_rescan_when_cwd_changes_without_explicit_invalidation(monkeypatch) -> None:
    """The @ file cache is scoped by cwd, not just by prior scan existence."""
    screen = make_suggestion_screen()
    handler = make_suggestion_handler(screen)
    screen._set_workspace_cwd("/repo-a")
    scans: list[str] = []

    def fake_getcwd() -> str:
        raise AssertionError("tracked workspace cwd should be used before process cwd")

    monkeypatch.setattr("chrys.foundation.platform.safe_getcwd", fake_getcwd)
    _patch_scan(monkeypatch, {"/repo-a": ["a.py"], "/repo-b": ["b.py"]}, roots=scans)

    handler._suggestion_mode = "files"
    asyncio.run(handler.show_file_suggestions_async())
    assert suggestion_values(screen.suggestion_list.last_items) == ["a.py"]

    screen._set_workspace_cwd("/repo-b")
    asyncio.run(handler.show_file_suggestions_async())

    assert scans == ["/repo-a", "/repo-b"]
    assert suggestion_values(screen.suggestion_list.last_items) == ["b.py"]


async def test_file_trigger_opens_loading_popup_before_cold_scan_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)
    gate = _hold_cold_scan(monkeypatch, paths=["ready.py"])

    handler.on_file_triggered()
    assert screen.suggestion_list.last_mode == "files"
    assert screen.suggestion_list.last_title == "Files under /repo"
    assert screen.suggestion_list.is_loading is True
    assert screen.suggestion_list.last_items == []

    task = asyncio.create_task(handler.show_file_suggestions_async())
    await gate.started.wait()
    assert screen.suggestion_list.is_loading is True

    gate.release.set()
    await task

    assert gate.roots == ["/repo"]
    assert screen.suggestion_list.is_loading is False
    assert suggestion_values(screen.suggestion_list.last_items) == ["ready.py"]


def test_file_suggestions_show_disabled_truncation_row(monkeypatch) -> None:
    """A bounded scan advertises truncation instead of silently omitting files."""
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)

    async def truncated_scan(root: str) -> ProjectPathScanResult:
        return scan_result(
            root,
            [ProjectPathSuggestion(path="a.py", kind="file")],
            truncated=True,
            file_budget=1,
            suggestion_budget=1,
            source_truncations={"rg": True},
        )

    _patch_scan(monkeypatch, truncated_scan)

    handler._suggestion_mode = "files"
    asyncio.run(handler.show_file_suggestions_async())

    assert suggestion_values(screen.suggestion_list.last_items) == ["a.py", "__chrys_file_results_truncated__"]
    truncation_row = screen.suggestion_list.last_items[-1]
    assert isinstance(truncation_row, SuggestionItem)
    assert truncation_row.disabled is True
    assert truncation_row.kind == "status"
    assert truncation_row.disabled_reason == "1 files / 1 rows indexed"


def test_file_suggestions_show_no_file_rows_for_empty_scan(monkeypatch) -> None:
    """A session from another machine may point at a workspace with no local files."""
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("Z:\\Fake\\MissingWorkspace")
    handler = make_suggestion_handler(screen)

    _patch_scan(monkeypatch, {"Z:\\Fake\\MissingWorkspace": []})

    handler._suggestion_mode = "files"
    asyncio.run(handler.show_file_suggestions_async())

    assert screen.suggestion_list.mode == "files"
    assert screen.suggestion_list.last_items == []
    assert handler.file_cache == []


def test_file_suggestion_enter_without_selection_does_not_submit_text() -> None:
    screen = make_suggestion_screen()
    screen.input_bar.value = "@missing"
    screen.suggestion_list.is_visible = True
    handler = make_suggestion_handler(screen)
    handler._suggestion_mode = "files"

    assert handler.on_suggestion_select(execute=True) is True

    assert screen.input_bar.value == "@missing"
    assert screen.submitted == []
    assert screen.suggestion_list.is_visible is True


def test_file_suggestions_discard_scan_when_cwd_changes_mid_scan(monkeypatch) -> None:
    """A scan result is cached only if its root is still current when it returns."""
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo-a")
    handler = make_suggestion_handler(screen)
    scans: list[str] = []

    def fake_getcwd() -> str:
        raise AssertionError("tracked workspace cwd should be used before process cwd")

    async def scan_and_chdir(root: str) -> ProjectPathScanResult:
        if root == "/repo-a":
            screen._set_workspace_cwd("/repo-b")
            return scan_result(root, [ProjectPathSuggestion(path="stale.py", kind="file")])
        return scan_result(root, [ProjectPathSuggestion(path="fresh.py", kind="file")])

    monkeypatch.setattr("chrys.foundation.platform.safe_getcwd", fake_getcwd)
    _patch_scan(monkeypatch, scan_and_chdir, roots=scans)

    handler._suggestion_mode = "files"
    asyncio.run(handler.show_file_suggestions_async())

    assert scans == ["/repo-a", "/repo-b"]
    assert suggestion_values(screen.suggestion_list.last_items) == ["fresh.py"]


async def test_file_suggestions_filter_directory_query() -> None:
    """A directory-shaped query should keep the matching directory entry."""
    screen = make_suggestion_screen()
    handler = make_suggestion_handler(screen)
    handler.file_cache = [
        ProjectPathSuggestion(path="src/chrys/app/tui/", kind="directory"),
        ProjectPathSuggestion(path="src/chrys/app/tui/widgets/chrome/file_scanner.py", kind="file"),
        ProjectPathSuggestion(path="src/chrys/orchestration/engine/engine.py", kind="file"),
    ]
    handler._suggestion_mode = "files"

    handler.on_text_changed("@src/chrys/app/tui/")
    await wait_for_file_query(handler)

    assert suggestion_values(screen.suggestion_list.last_items)[0] == "src/chrys/app/tui/"
    assert isinstance(screen.suggestion_list.last_items[0], SuggestionItem)
    assert screen.suggestion_list.last_items[0].kind == "directory"


async def test_file_suggestions_filter_full_relative_path_from_injected_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct file-cache injection remains a cheap test path for file suggestions."""
    screen = make_suggestion_screen()
    handler = make_suggestion_handler(screen)
    injected = [
        ProjectPathSuggestion(path="src/chrys/app/tui/widgets/chrome/", kind="directory"),
        ProjectPathSuggestion(path="src/chrys/app/tui/widgets/chrome/file_scanner.py", kind="file"),
        ProjectPathSuggestion(path="src/chrys/orchestration/engine/engine.py", kind="file"),
    ]
    handler.file_cache = injected
    handler._suggestion_mode = "files"

    def fail_fuzzy_filter(*_args, **_kwargs) -> list[str]:
        raise AssertionError("file suggestions should query ProjectPathIndex")

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.fuzzy_filter", fail_fuzzy_filter)

    handler.on_text_changed("@chrome")
    await wait_for_file_query(handler)

    assert handler.file_cache is injected
    assert suggestion_values(screen.suggestion_list.last_items) == [
        "src/chrys/app/tui/widgets/chrome/",
        "src/chrys/app/tui/widgets/chrome/file_scanner.py",
    ]
    assert isinstance(screen.suggestion_list.last_items[0], SuggestionItem)
    assert screen.suggestion_list.last_items[0].kind == "directory"


async def test_file_cache_assignment_invalidates_file_index() -> None:
    screen = make_suggestion_screen()
    handler = make_suggestion_handler(screen)
    handler.file_cache = [ProjectPathSuggestion(path="src/chrome/file.py", kind="file")]
    handler._suggestion_mode = "files"

    handler.on_text_changed("@chrome")
    await wait_for_file_query(handler)
    assert handler._file_index is not None

    handler.file_cache = None

    assert handler.file_cache is None
    assert handler._file_index is None


async def test_dismissed_suggestions_do_not_receive_stale_file_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suggestions_module, "_FILE_QUERY_DEBOUNCE_SECONDS", 0)
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)
    handler.file_cache = [ProjectPathSuggestion(path="src/chrome/file.py", kind="file")]
    handler._suggestion_mode = "files"

    handler.on_text_changed("@chrome")
    handler.dismiss_suggestions()
    await wait_for_file_query(handler)

    assert screen.suggestion_list.is_visible is False
    assert screen.suggestion_list.last_items == []


async def test_switching_modes_invalidates_pending_file_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suggestions_module, "_FILE_QUERY_DEBOUNCE_SECONDS", 0)
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)
    handler.file_cache = [ProjectPathSuggestion(path="src/chrome/file.py", kind="file")]
    handler._suggestion_mode = "files"

    handler.on_text_changed("@chrome")
    handler._show_suggestions("commands", [])
    await wait_for_file_query(handler)

    assert screen.suggestion_list.mode == "commands"
    assert suggestion_values(screen.suggestion_list.last_items) == []


async def test_rapid_file_typing_keeps_one_active_query_and_runs_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suggestions_module, "_FILE_QUERY_DEBOUNCE_SECONDS", 0)
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)
    handler._suggestion_mode = "files"
    handler.file_cache = [ProjectPathSuggestion(path="placeholder.py", kind="file")]

    lock = threading.Lock()
    active = 0
    max_active = 0
    calls: list[str] = []

    class SlowIndex:
        truncated = False
        file_count = 0
        suggestion_count = 0

        def query(self, query: str, *, limit: int) -> list[ProjectPathSuggestion]:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                calls.append(query)
                time.sleep(0.03)
                return [ProjectPathSuggestion(path=f"{query}.py", kind="file")]
            finally:
                with lock:
                    active -= 1

    handler._file_index = SlowIndex()  # type: ignore[assignment]
    handler._file_index_root = "/repo"

    handler.on_text_changed("@alpha")
    await asyncio.sleep(0.01)
    handler.on_text_changed("@beta")
    handler.on_text_changed("@gamma")
    await wait_for_file_query(handler)

    assert max_active == 1
    assert calls[-1] == "gamma"
    assert "beta" not in calls
    assert suggestion_values(screen.suggestion_list.last_items) == ["gamma.py"]


async def test_typing_file_query_before_cold_scan_completion_replays_latest_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suggestions_module, "_FILE_QUERY_DEBOUNCE_SECONDS", 0)
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)
    gate = _hold_cold_scan(monkeypatch, paths=["alpha.py", "src/foo.py"])

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    await gate.started.wait()
    handler.on_text_changed("@foo")
    gate.release.set()
    await warmup
    await wait_for_file_query(handler)

    assert gate.roots == ["/repo"]
    assert suggestion_values(screen.suggestion_list.last_items) == ["src/foo.py"]


async def test_cache_invalidation_during_index_build_discards_stale_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)
    scans: list[str] = []

    async def stale_then_fresh_scan(root: str) -> ProjectPathScanResult:
        path = "stale.py" if len(scans) == 1 else "fresh.py"
        return scan_result(root, [ProjectPathSuggestion(path=path, kind="file")])

    _patch_scan(monkeypatch, stale_then_fresh_scan, roots=scans)
    gate = _hold_index_build(monkeypatch, handler, first_build_only=True)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    await asyncio.to_thread(gate.started.wait, 1)
    handler.file_cache = None
    gate.release.set()
    await warmup

    assert scans == ["/repo", "/repo"]
    assert handler.file_cache == [ProjectPathSuggestion(path="fresh.py", kind="file")]
    assert suggestion_values(screen.suggestion_list.last_items) == ["fresh.py"]


@dataclass(frozen=True, slots=True)
class _WarmupHold:
    """A warmup paused at one point of the cold-scan then index-build pipeline."""

    wait_until_held: Callable[[], Awaitable[None]]
    release: Callable[[], None]
    roots: list[str]


def _hold_at_cold_scan(monkeypatch: pytest.MonkeyPatch, _handler: SuggestionHandler) -> _WarmupHold:
    gate = _hold_cold_scan(monkeypatch)

    async def wait_until_held() -> None:
        await gate.started.wait()

    return _WarmupHold(wait_until_held=wait_until_held, release=gate.release.set, roots=gate.roots)


def _hold_at_index_build(monkeypatch: pytest.MonkeyPatch, handler: SuggestionHandler) -> _WarmupHold:
    roots: list[str] = []
    _patch_scan(monkeypatch, {"/repo": ["stale.py"]}, roots=roots)
    gate = _hold_index_build(monkeypatch, handler)

    async def wait_until_held() -> None:
        await asyncio.to_thread(gate.started.wait, 1)

    return _WarmupHold(wait_until_held=wait_until_held, release=gate.release.set, roots=roots)


def _dismiss_suggestions(_screen: SuggestionScreen, handler: SuggestionHandler) -> None:
    handler.dismiss_suggestions()


def _switch_to_commands(_screen: SuggestionScreen, handler: SuggestionHandler) -> None:
    handler._show_suggestions("commands", [])


def _detach_screen(screen: SuggestionScreen, _handler: SuggestionHandler) -> None:
    screen.is_attached = False


@dataclass(frozen=True, slots=True)
class _Interruption:
    """One way a warmup is cut short, and the popup state that must survive it."""

    hold: Callable[[pytest.MonkeyPatch, SuggestionHandler], _WarmupHold]
    interrupt: Callable[[SuggestionScreen, SuggestionHandler], None]
    expected_mode: str
    expected_visible: bool


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _Interruption(_hold_at_cold_scan, _dismiss_suggestions, "", False),
            id="dismiss-during-cold-scan",
        ),
        pytest.param(
            _Interruption(_hold_at_index_build, _switch_to_commands, "commands", True),
            id="mode-switch-during-index-build",
        ),
        pytest.param(
            _Interruption(_hold_at_cold_scan, _detach_screen, "", False),
            id="detach-during-cold-scan",
        ),
    ],
)
async def test_interrupting_a_file_warmup_never_reopens_file_suggestions(
    monkeypatch: pytest.MonkeyPatch,
    case: _Interruption,
) -> None:
    """Dismiss, mode switch, and detach all discard the in-flight warmup: the
    cache stays empty, no follow-up scan is queued, and the popup is not reopened.
    """
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)
    hold = case.hold(monkeypatch, handler)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    await hold.wait_until_held()
    case.interrupt(screen, handler)
    hold.release()
    await warmup

    assert hold.roots == ["/repo"]
    assert handler.file_cache is None
    assert handler._file_warmup_requested_root is None
    assert screen.suggestion_list.last_items == []
    assert screen.suggestion_list.mode == case.expected_mode
    assert screen.suggestion_list.is_visible is case.expected_visible


async def test_dismiss_before_warmup_starts_does_not_restore_file_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)
    scans: list[str] = []
    _patch_scan(monkeypatch, {"/repo": ["stale.py"]}, roots=scans)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    handler.dismiss_suggestions()
    await warmup

    assert scans == []
    assert handler.suggestion_mode is None
    assert screen.suggestion_list.is_visible is False


async def test_mode_switch_before_warmup_worker_scans_skips_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)
    scans: list[str] = []
    deferred_workers: list[DeferredWorker] = []

    def run_worker(work, **_kwargs) -> DeferredWorker:
        worker = DeferredWorker(work)
        deferred_workers.append(worker)
        return worker

    monkeypatch.setattr(screen, "run_worker", run_worker)
    _patch_scan(monkeypatch, {"/repo": ["stale.py"]}, roots=scans)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    await asyncio.sleep(0)
    assert deferred_workers

    handler._show_suggestions("commands", [])
    await warmup

    assert scans == []
    assert handler.file_cache is None
    assert screen.suggestion_list.mode == "commands"


async def test_detached_screen_does_not_receive_file_query_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suggestions_module, "_FILE_QUERY_DEBOUNCE_SECONDS", 0)
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)
    handler.file_cache = [ProjectPathSuggestion(path="src/chrome/file.py", kind="file")]
    handler._suggestion_mode = "files"

    handler.on_text_changed("@chrome")
    screen.is_attached = False
    await wait_for_file_query(handler)

    assert screen.suggestion_list.last_items == []


async def test_index_warmup_build_does_not_starve_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = make_suggestion_screen()
    screen._set_workspace_cwd("/repo")
    handler = make_suggestion_handler(screen)
    original_build = handler._build_file_index

    async def large_scan(root: str) -> ProjectPathScanResult:
        return scan_result(
            root,
            [ProjectPathSuggestion(path=f"src/file_{index:04d}.py", kind="file") for index in range(1000)],
        )

    loop_kept_ticking = threading.Event()

    def slow_build_file_index(scan: ProjectPathScanResult):
        # Block the worker until the event loop has demonstrably kept
        # ticking underneath the build.  If the build ran on the loop
        # thread instead, this could never be set — the timeout path
        # finishes the warmup with the tick count still at ~0 and the
        # assertion fails.  A handshake instead of a fixed sleep: tick
        # counting against wall-clock flakes on Windows' coarse timer.
        loop_kept_ticking.wait(timeout=5.0)
        return original_build(scan)

    _patch_scan(monkeypatch, large_scan)
    monkeypatch.setattr(handler, "_build_file_index", slow_build_file_index)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    ticks = 0
    while not warmup.done():
        ticks += 1
        if ticks > 3:
            loop_kept_ticking.set()
        await asyncio.sleep(0.005)
    await warmup

    assert ticks > 3
    assert suggestion_values(screen.suggestion_list.last_items)[:1] == ["src/file_0000.py"]


def test_selecting_directory_suggestion_inserts_trailing_slash() -> None:
    screen = make_suggestion_screen()
    handler = make_suggestion_handler(screen)

    handler.on_suggestion_selected("files", "src/chrys", execute=False, kind="directory")

    assert screen.input_bar.replacements == [("@", "@src/chrys/ ")]


def test_selecting_file_status_suggestion_is_ignored() -> None:
    screen = make_suggestion_screen()
    handler = make_suggestion_handler(screen)

    handler.on_suggestion_selected("files", "__chrys_file_results_truncated__", execute=False, kind="status")

    assert screen.input_bar.replacements == []
    assert screen.suggestion_list.is_visible is False
