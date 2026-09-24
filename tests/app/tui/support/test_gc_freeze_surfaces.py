# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for TUI surfaces participating in GC freeze coordination."""

from __future__ import annotations

import asyncio
import gc
import weakref
from contextlib import suppress
from threading import Event
from types import SimpleNamespace

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.strip import Strip

from chrys.app.tui.screens.main import screen as screen_module
from chrys.app.tui.screens.main.screen import MainScreen
from chrys.app.tui.screens.main.suggestions import SuggestionHandler
from chrys.app.tui.support.gc_freeze import (
    DetachedFifoCache,
    DetachedLruCache,
    GcAbsorbRequested,
    GcFreezeBlockReason,
    GcFreezeCoordinator,
    GcReclaimRequested,
    after_textual_screen_gc_freeze,
    prepare_textual_screen_for_gc,
)
from chrys.app.tui.terminal.panel import ShellPanel
from chrys.app.tui.terminal.widget import Terminal
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chat.session_json import SessionJsonPanel
from chrys.app.tui.widgets.chat.tool_call import ToolCall, ToolGroup
from chrys.app.tui.widgets.chrome.file_index import ProjectPathIndex
from chrys.app.tui.widgets.chrome.file_scanner import ProjectPathScanResult, ProjectPathSuggestion
from tests.support.waiting import wait_for


class _Participant:
    def __init__(self, reason: GcFreezeBlockReason | None = None) -> None:
        self.reason = reason
        self.calls: list[str] = []

    def gc_freeze_block_reason(self) -> GcFreezeBlockReason | None:
        self.calls.append("block")
        return self.reason

    def prepare_for_gc_freeze(self) -> None:
        self.calls.append("prepare")

    def after_gc_freeze(self) -> None:
        self.calls.append("after")

    def abort_gc_freeze(self) -> None:
        self.calls.append("abort")


class _FailingParticipant(_Participant):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label

    def after_gc_freeze(self) -> None:
        self.calls.append("after")
        raise RuntimeError(self.label)


def _weakref_is_alive(reference: weakref.ReferenceType[object]) -> bool:
    """Inspect a weakref without retaining its target in the async test frame."""
    return reference() is not None


def _tool_call_weakref(group: ToolGroup, call_id: str) -> weakref.ReferenceType[ToolCall]:
    """Build a weakref without retaining the widget in an async test frame."""
    descendant = group.get_tool(call_id)
    assert isinstance(descendant, ToolCall)
    return weakref.ref(descendant)


@pytest.mark.parametrize(
    ("loading", "running", "scrolling", "expected"),
    [
        (True, False, False, GcFreezeBlockReason.AGENT_LOADING),
        (False, True, False, GcFreezeBlockReason.AGENT_RUNNING),
        (False, False, True, GcFreezeBlockReason.SCROLL_GC_PAUSED),
    ],
)
def test_main_screen_own_freeze_gates(
    monkeypatch: pytest.MonkeyPatch,
    loading: bool,
    running: bool,
    scrolling: bool,
    expected: GcFreezeBlockReason,
) -> None:
    participant = _Participant(GcFreezeBlockReason.PARTICIPANT)
    screen = SimpleNamespace(
        _agent_loading=loading,
        _agent_running=running,
        _gc_freeze_participants=(participant,),
    )
    monkeypatch.setattr(screen_module, "scroll_gc_paused", lambda: scrolling)

    assert MainScreen.gc_freeze_block_reason(screen) is expected
    assert participant.calls == []


def test_main_screen_delegates_gate_and_hooks_in_registration_order(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _Participant()
    second = _Participant(GcFreezeBlockReason.PARTICIPANT)
    third = _Participant(GcFreezeBlockReason.SHELL_VISIBLE)
    screen = SimpleNamespace(
        _agent_loading=False,
        _agent_running=False,
        _gc_freeze_participants=(first, second, third),
    )
    monkeypatch.setattr(screen_module, "scroll_gc_paused", lambda: False)
    screen_cache_calls: list[str] = []
    monkeypatch.setattr(
        screen_module,
        "prepare_textual_screen_for_gc",
        lambda _screen: screen_cache_calls.append("prepare"),
    )
    monkeypatch.setattr(
        screen_module,
        "after_textual_screen_gc_freeze",
        lambda _screen: screen_cache_calls.append("after"),
    )

    assert MainScreen.gc_freeze_block_reason(screen) is GcFreezeBlockReason.PARTICIPANT
    MainScreen.prepare_for_gc_freeze(screen)
    MainScreen.after_gc_freeze(screen)

    assert first.calls == ["block", "prepare", "after"]
    assert second.calls == ["block", "prepare", "after"]
    assert third.calls == ["prepare", "after"]
    assert screen_cache_calls == ["prepare", "after"]


def test_main_screen_reports_every_gc_renewal_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _FailingParticipant("first participant")
    second = _FailingParticipant("second participant")
    screen = SimpleNamespace(_gc_freeze_participants=(first, second))

    def _fail_screen_cache(_screen: object) -> None:
        raise RuntimeError("screen cache")

    monkeypatch.setattr(screen_module, "after_textual_screen_gc_freeze", _fail_screen_cache)

    with pytest.raises(ExceptionGroup) as raised:
        MainScreen.after_gc_freeze(screen)

    assert [str(error) for error in raised.value.exceptions] == [
        "screen cache",
        "first participant",
        "second participant",
    ]


@pytest.mark.asyncio
async def test_textual_screen_cache_hooks_clear_compositor_and_renew_cyclic_lrus() -> None:
    class _ScreenCacheApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    app = _ScreenCacheApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        old_box_cache = screen._box_model_cache
        old_query_cache = screen._query_one_cache
        old_arrangement_cache = screen._arrangement_cache
        old_box_cache["box"] = object()  # type: ignore[index]
        old_query_cache[("query",)] = screen  # type: ignore[index]
        assert screen._compositor.widgets

        prepare_textual_screen_for_gc(screen)
        assert not screen._compositor.widgets
        assert isinstance(screen._box_model_cache, DetachedLruCache)
        assert isinstance(screen._query_one_cache, DetachedLruCache)
        assert isinstance(screen._arrangement_cache, DetachedFifoCache)

        after_textual_screen_gc_freeze(screen)
        assert screen._box_model_cache is not old_box_cache
        assert screen._query_one_cache is not old_query_cache
        assert screen._arrangement_cache is not old_arrangement_cache
        assert screen._layout_required is True


def test_chat_panel_delegates_gc_hooks_to_dynamic_render_cache_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _Participant()
    second = _Participant()
    panel = ChatPanel()
    monkeypatch.setattr(panel, "_gc_freeze_render_cache_surfaces", lambda: [first, second])

    assert panel.gc_freeze_block_reason() is None
    panel.prepare_for_gc_freeze()
    panel.after_gc_freeze()

    assert first.calls == ["prepare", "after"]
    assert second.calls == ["prepare", "after"]


def test_chat_panel_reports_every_gc_renewal_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    first = _FailingParticipant("first surface")
    second = _FailingParticipant("second surface")
    panel = ChatPanel()
    monkeypatch.setattr(panel, "_gc_freeze_render_cache_surfaces", lambda: [first, second])

    with pytest.raises(ExceptionGroup) as raised:
        panel.after_gc_freeze()

    assert [str(error) for error in raised.value.exceptions] == ["first surface", "second surface"]


def test_session_json_hide_releases_content_and_after_hook_renews_lru() -> None:
    panel = SessionJsonPanel()
    panel.display = True
    panel._formatted_json = "x" * 1_000
    panel._file_mtime = 10.0
    panel._last_theme_key = (True, "#ffffff")
    panel._text_lines = [Text("value")]
    panel._plain_lines = ["value"]
    panel._line_widths = [5]
    panel._max_width = 5
    panel._strip_cache[0] = Strip.blank(5)
    panel._cache_scroll_x = 2
    panel._cache_width = 5
    old_cache = panel._strip_cache
    old_generation = panel._content_generation

    assert panel.gc_freeze_block_reason() is GcFreezeBlockReason.SESSION_JSON_VISIBLE
    panel.hide_session_json()

    assert panel.gc_freeze_block_reason() is None
    assert panel._formatted_json == ""
    assert panel._file_mtime == 0.0
    assert panel._last_theme_key is None
    assert panel._text_lines == []
    assert panel._plain_lines == []
    assert panel._line_widths == []
    assert panel._max_width == 0
    assert len(panel._strip_cache) == 0
    assert panel._cache_scroll_x == -1
    assert panel._cache_width == -1
    assert panel.virtual_size.height == 0
    assert panel._content_generation == old_generation + 1

    panel.prepare_for_gc_freeze()
    assert isinstance(panel._strip_cache, DetachedLruCache)
    assert panel._content_generation == old_generation + 2
    panel.after_gc_freeze()
    assert panel._strip_cache is not old_cache
    assert panel._strip_cache.maxsize == old_cache.maxsize


def test_session_json_highlight_descendants_are_acyclic_while_frozen() -> None:
    class _Sentinel:
        pass

    payload = SessionJsonPanel._highlight('{"value": [1, 2, 3]}', True, None)
    sentinel = _Sentinel()
    sentinel_ref = weakref.ref(sentinel)
    descendant = payload[0][0]
    descendant._text.append(sentinel)  # type: ignore[arg-type]

    gc.collect()
    gc.freeze()
    try:
        del sentinel
        del descendant
        del payload
        assert sentinel_ref() is None
    finally:
        gc.unfreeze()
        gc.collect()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True], ids=["completed", "cancelled"])
async def test_real_session_json_to_thread_payload_drops_while_frozen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    cancelled: bool,
) -> None:
    class _Sentinel:
        pass

    path = tmp_path / "session.json"
    path.write_text('{"value": [1, 2, 3]}', encoding="utf-8")
    original_highlight = SessionJsonPanel._highlight
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = asyncio.Event()
    release = Event()
    sentinel_refs: list[weakref.ReferenceType[_Sentinel]] = []

    def _blocked_highlight(json_text: str, dark: bool, gutter_color: str | None):
        payload = original_highlight(json_text, dark, gutter_color)
        sentinel = _Sentinel()
        sentinel_refs.append(weakref.ref(sentinel))
        payload[0][0]._text.append(sentinel)  # type: ignore[arg-type]
        loop.call_soon_threadsafe(started.set)
        release.wait()
        loop.call_soon_threadsafe(finished.set)
        return payload

    monkeypatch.setattr(SessionJsonPanel, "_highlight", staticmethod(_blocked_highlight))
    panel = SessionJsonPanel()
    panel.display = True
    panel._content_generation = 1
    task = asyncio.create_task(panel._load_worker(path, True, None, 1))
    await started.wait()

    gc.collect()
    gc.freeze()
    try:
        panel.hide_session_json()
        if cancelled:
            task.cancel()
        release.set()
        with suppress(asyncio.CancelledError):
            await task
        await finished.wait()
        del task
        del panel
        del _blocked_highlight
        # A worker Future may retain its prior result after the coroutine has
        # exited. Turn over every pool thread before inspecting the payload.
        await asyncio.gather(*(asyncio.to_thread(lambda: None) for _ in range(32)))
        await asyncio.sleep(0)

        assert len(sentinel_refs) == 1
        assert sentinel_refs[0]() is None
    finally:
        release.set()
        gc.unfreeze()
        gc.collect()


@pytest.mark.asyncio
async def test_session_json_shell_suspension_allows_in_flight_load_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from chrys.app.tui.widgets.chat import session_json as session_json_module

    panel = SessionJsonPanel()
    panel.display = True
    panel._content_generation = 7
    monkeypatch.setattr(panel, "scroll_home", lambda *, animate: None)
    started = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def _to_thread(_function, *_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            started.set()
            await release.wait()
            return '{"value": 1}'
        return [Text("new")], ["new"], [3], 3

    monkeypatch.setattr(session_json_module.asyncio, "to_thread", _to_thread)
    task = asyncio.create_task(panel._load_worker(tmp_path / "session.json", True, None, 7))
    await started.wait()

    panel.suspend_for_shell_mode()
    panel.prepare_for_gc_freeze()
    assert panel.display is False
    assert panel.gc_freeze_block_reason() is GcFreezeBlockReason.SESSION_JSON_VISIBLE
    assert panel._content_generation == 7

    release.set()
    await task

    assert panel._formatted_json == '{\n  "value": 1\n}'
    assert panel._status == ""
    assert panel._plain_lines == ["new"]
    assert panel._content_generation == 7

    panel.finish_shell_mode(restore=True)
    assert panel.display is True
    assert panel._shell_mode_suspended is False
    assert panel._plain_lines == ["new"]


def test_session_json_shell_finish_without_restore_cancels_retained_content_once() -> None:
    panel = SessionJsonPanel()
    panel.display = True
    panel._formatted_json = '{"value": 1}'
    panel._plain_lines = ["value"]
    panel._content_generation = 7

    panel.suspend_for_shell_mode()
    panel.finish_shell_mode(restore=False)

    assert panel.display is False
    assert panel._shell_mode_suspended is False
    assert panel.gc_freeze_block_reason() is None
    assert panel._formatted_json == ""
    assert panel._plain_lines == []
    assert panel._content_generation == 8

    panel.finish_shell_mode(restore=False)
    assert panel._content_generation == 8


@pytest.mark.asyncio
async def test_session_json_hide_during_load_prevents_stale_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from chrys.app.tui.widgets.chat import session_json as session_json_module

    panel = SessionJsonPanel()
    panel.display = True
    panel._content_generation = 7
    started = asyncio.Event()
    release = asyncio.Event()
    call_count = 0

    async def _to_thread(_function, *_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            started.set()
            await release.wait()
            return '{"value": 1}'
        return [Text("new")], ["new"], [3], 3

    monkeypatch.setattr(session_json_module.asyncio, "to_thread", _to_thread)
    task = asyncio.create_task(panel._load_worker(tmp_path / "session.json", True, None, 7))
    await started.wait()

    panel.hide_session_json()
    release.set()
    await task

    assert panel._formatted_json == ""
    assert panel._text_lines == []
    assert panel._plain_lines == []
    assert panel.virtual_size.height == 0


@pytest.mark.asyncio
async def test_session_json_hide_during_rehighlight_prevents_stale_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.app.tui.widgets.chat import session_json as session_json_module

    panel = SessionJsonPanel()
    panel.display = True
    panel._content_generation = 4
    panel._formatted_json = "{}"
    panel._text_lines = [Text("old")]
    started = asyncio.Event()
    release = asyncio.Event()

    async def _to_thread(_function, *_args, **_kwargs):
        started.set()
        await release.wait()
        return [Text("new")], ["new"], [3], 3

    monkeypatch.setattr(session_json_module.asyncio, "to_thread", _to_thread)
    task = asyncio.create_task(panel._rehighlight_worker(False, None, 4))
    await started.wait()

    panel.hide_session_json()
    release.set()
    await task

    assert panel._formatted_json == ""
    assert panel._text_lines == []
    assert panel._plain_lines == []


def test_shell_participant_blocks_visible_mode_and_renews_render_lru() -> None:
    panel = ShellPanel()
    terminal = Terminal(size=(20, 4))
    panel._terminal = terminal
    terminal._strip_cache[0] = Strip.blank(20)
    old_cache = terminal._strips
    old_capacity = old_cache.maxsize

    assert panel.gc_freeze_block_reason() is None
    panel.prepare_for_gc_freeze()
    assert isinstance(terminal._strips, DetachedLruCache)

    panel.after_gc_freeze()
    assert terminal._strips is not old_cache
    assert terminal._strips.maxsize == old_capacity

    panel.display = True
    assert panel.gc_freeze_block_reason() is GcFreezeBlockReason.SHELL_VISIBLE


@pytest.mark.parametrize("surface", ["json", "terminal"])
def test_post_freeze_lru_renewal_keeps_refilled_links_collectible(surface: str) -> None:
    class _Sentinel:
        pass

    panel: SessionJsonPanel | ShellPanel
    if surface == "json":
        json_panel = SessionJsonPanel()
        json_panel.display = False
        json_panel.prepare_for_gc_freeze()
        panel = json_panel
    else:
        shell_panel = ShellPanel()
        shell_panel._terminal = Terminal(size=(20, 4))
        shell_panel.prepare_for_gc_freeze()
        panel = shell_panel

    gc.collect()
    gc.freeze()
    try:
        panel.after_gc_freeze()
        cache = (
            panel._strip_cache if isinstance(panel, SessionJsonPanel) else panel._terminal._strips  # type: ignore[union-attr]
        )
        sentinel = _Sentinel()
        sentinel_ref = weakref.ref(sentinel)
        cache["key"] = sentinel  # type: ignore[index]
        del sentinel
        cache.clear()
        gc.collect()

        assert sentinel_ref() is None
    finally:
        gc.unfreeze()
        gc.collect()


def test_suggestion_participant_blocks_popup_without_releasing_warm_cache() -> None:
    handler = object.__new__(SuggestionHandler)
    warm_cache = [ProjectPathSuggestion("src/chrys/app.py", "file")]
    warm_index = object()
    handler._view = SimpleNamespace(suggestions_active=True)
    handler._file_cache = warm_cache
    handler._file_index = warm_index

    assert handler.gc_freeze_block_reason() is GcFreezeBlockReason.SUGGESTIONS_VISIBLE
    handler.prepare_for_gc_freeze()
    handler.after_gc_freeze()
    assert handler._file_cache is warm_cache
    assert handler._file_index is warm_index

    handler._view.suggestions_active = False
    assert handler.gc_freeze_block_reason() is None


def test_frozen_project_path_index_dies_by_refcount_without_unfreeze() -> None:
    scan = ProjectPathScanResult.from_suggestions(
        root="/repo",
        paths=[ProjectPathSuggestion("src/chrys/app.py", "file")],
    )
    index = ProjectPathIndex.build(scan)
    index_ref = weakref.ref(index)

    gc.collect()
    gc.freeze()
    try:
        del index
        assert index_ref() is None
    finally:
        gc.unfreeze()
        gc.collect()


@pytest.mark.asyncio
async def test_cancelled_inflight_index_to_thread_drops_payload_while_frozen() -> None:
    scan = ProjectPathScanResult.from_suggestions(
        root="/repo",
        paths=[ProjectPathSuggestion(f"src/file_{path_number}.py", "file") for path_number in range(1_000)],
    )
    index = ProjectPathIndex.build(scan)
    index_ref = weakref.ref(index)
    payload_box = [index]
    del index
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = asyncio.Event()
    release = Event()

    def _return_index() -> ProjectPathIndex:
        loop.call_soon_threadsafe(started.set)
        release.wait()
        value = payload_box[0]
        loop.call_soon_threadsafe(finished.set)
        return value

    task = asyncio.create_task(asyncio.to_thread(_return_index))
    await started.wait()
    gc.collect()
    gc.freeze()
    try:
        task.cancel()
        release.set()
        with suppress(asyncio.CancelledError):
            await task
        await finished.wait()
        del task
        del _return_index
        payload_box.clear()
        # Turn over the pool after the cancelled worker exits; its Future may
        # otherwise retain the prior return value even though no task/frame does.
        await asyncio.gather(*(asyncio.to_thread(lambda: None) for _ in range(32)))
        await asyncio.sleep(0)

        assert index_ref() is None
    finally:
        release.set()
        gc.unfreeze()
        gc.collect()


@pytest.mark.asyncio
async def test_frozen_completed_tool_subtree_dies_on_its_idle_full_reclaim() -> None:
    class _Clock:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            return self.value

        def advance(self, seconds: float) -> None:
            self.value += seconds

    class _ToolGroupGcApp(App):
        def __init__(self) -> None:
            self.clock = _Clock()
            self.coordinator = GcFreezeCoordinator(self, enabled=True, clock=self.clock)
            super().__init__()

        def compose(self) -> ComposeResult:
            yield ChatPanel()

        def freeze_block_reason(self) -> GcFreezeBlockReason | None:
            return None

        def prepare_for_gc_freeze(self) -> None:
            return

        def after_gc_freeze(self) -> None:
            return

        def on_gc_absorb_requested(self, event: GcAbsorbRequested) -> None:
            self.coordinator.request_absorb(
                reason=event.reason,
                terminal_boundary=event.terminal_boundary,
                requested_at=event.time,
            )

        def on_gc_reclaim_requested(self, event: GcReclaimRequested) -> None:
            self.coordinator.request_reclaim(
                reason=event.reason,
                prompt=event.prompt,
                requested_at=event.time,
            )

        def on_unmount(self) -> None:
            self.coordinator.close()

    app = _ToolGroupGcApp()
    async with app.run_test() as pilot:
        panel = app.query_one(ChatPanel)
        await panel.add_user_message("tools")
        await panel.add_tool_start("tool1", "plain_tool", "", args={"value": 1})
        await panel.add_tool_result("tool1", "plain_tool", "done", 10)
        group = panel.query_one(ToolGroup)
        descendant_ref = _tool_call_weakref(group, "tool1")

        app.coordinator.start()
        await wait_for(
            lambda: app.coordinator.frozen,
            pilot=pilot,
            description="GC coordinator freezes mounted content",
        )
        assert app.coordinator.frozen is True

        group.collapsed = True
        await wait_for(
            lambda: not group._content_mounted and app.coordinator._idle_reclaim_pending,
            pilot=pilot,
            description="collapsed tool subtree is removed and reclaim requested",
        )
        assert group._content_mounted is False
        assert app.coordinator._idle_reclaim_pending is True
        gc.collect()
        assert _weakref_is_alive(descendant_ref)

        # The real idle window spans many event-loop turns. Drain removal and
        # InvokeLater bookkeeping before advancing the deterministic clock so
        # the full reclaim measures the detached subtree, not the test's await.
        removal_drained = asyncio.Event()
        assert group.call_later(removal_drained.set)
        await wait_for(removal_drained.is_set, pilot=pilot, description="tool group removal callback drains")

        app.clock.advance(4.0)
        app.coordinator.on_tick()
        await wait_for(
            lambda: not app.coordinator._idle_reclaim_pending,
            pilot=pilot,
            description="idle GC reclaim completes",
        )

        assert app.coordinator._idle_reclaim_pending is False
        assert app.coordinator.frozen is True
        assert not _weakref_is_alive(descendant_ref)
