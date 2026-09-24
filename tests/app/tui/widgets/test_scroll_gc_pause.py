# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the chat-panel scroll controller's GC pause: owner counting, resume debounce, scrollbar grabs, and release on clear/unmount."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from textual import events as textual_events
from textual.geometry import Offset
from textual.scrollbar import ScrollTo

from chrys.app.tui.widgets.chat.panel import ChatPanel
from tests.app.tui.widgets._scroll_gc import install_fake_chat_panel_gc
from tests.support.tui_helpers import ChatPanelApp
from tests.support.waiting import wait_for


async def _wait_for_chat_panel_gc_resume(pilot: object, panel: ChatPanel) -> None:
    """Wait until the scroll-GC debounce timer has resumed the panel state."""
    await wait_for(
        lambda: not panel._manual_scroll_gc_paused,
        pilot=pilot,
        description="scroll-GC pause released by the resume debounce",
    )


async def test_chat_panel_pauses_gc_while_manual_scroll_is_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual scroll pauses cyclic GC and restores it after the debounce window."""
    _, _FakeGC = install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        cp._pause_gc_for_manual_scroll()
        assert _FakeGC.enabled is False
        assert _FakeGC.disable_calls == 1

        cp._pause_gc_for_manual_scroll()
        assert _FakeGC.disable_calls == 1

        await _wait_for_chat_panel_gc_resume(pilot, cp)

        assert _FakeGC.enabled is True
        assert _FakeGC.enable_calls == 1
        assert cp._manual_scroll_gc_paused is False

        _FakeGC.enabled = False
        cp._pause_gc_for_manual_scroll()
        await _wait_for_chat_panel_gc_resume(pilot, cp)

        assert _FakeGC.enabled is False
        assert _FakeGC.enable_calls == 1


def test_scroll_gc_paused_uses_shared_owner_count(monkeypatch: pytest.MonkeyPatch) -> None:
    scroll_controller_module, _FakeGC = install_fake_chat_panel_gc(monkeypatch)

    assert scroll_controller_module.scroll_gc_paused() is False
    scroll_controller_module._claim_scroll_gc_pause()
    scroll_controller_module._claim_scroll_gc_pause()
    assert scroll_controller_module.scroll_gc_paused() is True
    assert _FakeGC.disable_calls == 1

    scroll_controller_module._release_scroll_gc_pause()
    assert scroll_controller_module.scroll_gc_paused() is True
    scroll_controller_module._release_scroll_gc_pause()
    assert scroll_controller_module.scroll_gc_paused() is False
    assert _FakeGC.enable_calls == 1


async def test_chat_panel_resume_collects_gen0_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume restores GC and drains the paused-scroll gen0 backlog right away."""
    scroll_controller_module, _FakeGC = install_fake_chat_panel_gc(monkeypatch)
    monkeypatch.setattr(scroll_controller_module, "_MANUAL_SCROLL_GC_RESUME_SECONDS", 0.01)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        cp._pause_gc_for_manual_scroll()
        assert _FakeGC.collect_generations == []
        await _wait_for_chat_panel_gc_resume(pilot, cp)

        assert _FakeGC.enabled is True
        assert _FakeGC.collect_generations == [0]


async def test_chat_panel_resume_skips_collect_while_another_owner_holds_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No gen0 collect when GC is still disabled by another panel's pause claim."""
    scroll_controller_module, _FakeGC = install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        scroll_controller_module._claim_scroll_gc_pause()
        cp._pause_gc_for_manual_scroll()

        cp._resume_gc_after_manual_scroll()

        assert cp._manual_scroll_gc_paused is False
        assert _FakeGC.enabled is False
        assert _FakeGC.collect_generations == []

        scroll_controller_module._release_scroll_gc_pause()
        assert _FakeGC.enabled is True


async def test_chat_panel_scrollbar_grab_holds_gc_pause_without_resume_debounce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grabbed thumb holds the pause open; ticks never arm the resume timer."""
    _, _FakeGC = install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        timers: list[object] = []
        real_set_timer = cp.set_timer

        def set_timer_spy(delay: float, callback: Callable[[], None]) -> object:
            timer = real_set_timer(delay, callback)
            timers.append(timer)
            return timer

        monkeypatch.setattr(cp, "set_timer", set_timer_spy)

        cp._scroll_controller.on_scrollbar_grab()
        assert cp._manual_scroll_gc_paused is True
        assert _FakeGC.enabled is False
        assert cp._manual_scroll_gc_timer is None
        assert timers == []

        cp._pause_gc_for_manual_scroll()
        assert cp._manual_scroll_gc_timer is None
        assert timers == []

        cp._scroll_controller.on_scrollbar_release()
        assert cp._manual_scroll_gc_paused is True
        assert cp._manual_scroll_gc_timer is not None
        assert len(timers) == 1

        await _wait_for_chat_panel_gc_resume(pilot, cp)
        assert _FakeGC.enabled is True
        assert _FakeGC.collect_generations == [0]


async def test_chat_panel_resume_never_fires_while_thumb_still_grabbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: a stray resume call during an active grab keeps GC paused."""
    _, _FakeGC = install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        cp._scroll_controller.on_scrollbar_grab()
        cp._resume_gc_after_manual_scroll()

        assert cp._manual_scroll_gc_paused is True
        assert _FakeGC.enabled is False
        assert _FakeGC.collect_generations == []


async def test_chat_panel_scrollbar_grabbed_reactive_drives_gc_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mounted panel watches ScrollBar.grabbed: grab pauses, release re-arms."""
    _, _FakeGC = install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await pilot.pause()

        cp.vertical_scrollbar.grabbed = Offset(0, 0)
        await pilot.pause()
        assert cp._scroll_controller.scrollbar_grabbed is True
        assert cp._manual_scroll_gc_paused is True
        assert _FakeGC.enabled is False
        assert cp._manual_scroll_gc_timer is None

        cp.vertical_scrollbar.grabbed = None
        await pilot.pause()
        assert cp._scroll_controller.scrollbar_grabbed is False
        assert cp._manual_scroll_gc_timer is not None

        await _wait_for_chat_panel_gc_resume(pilot, cp)
        assert _FakeGC.enabled is True


async def test_chat_panel_mouse_wheel_pauses_gc_before_base_scroll_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wheel input should pause GC before Textual's base scroll handler runs once."""
    _, _FakeGC = install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await pilot.pause()
        scroll_down_calls: list[dict[str, object]] = []
        scroll_down_gc_states: list[bool] = []

        def allow_vertical_scroll(_panel: ChatPanel) -> bool:
            return True

        def scroll_down_spy(**kwargs: object) -> bool:
            scroll_down_calls.append(kwargs)
            scroll_down_gc_states.append(_FakeGC.enabled)
            return True

        monkeypatch.setattr(ChatPanel, "allow_vertical_scroll", property(allow_vertical_scroll))
        monkeypatch.setattr(cp, "_scroll_down_for_pointer", scroll_down_spy)

        await cp._on_message(
            textual_events.MouseScrollDown(
                cp, x=1, y=1, delta_x=0, delta_y=1, button=0, shift=False, meta=False, ctrl=False
            )
        )

        assert scroll_down_calls == [{"animate": False}]
        assert scroll_down_gc_states == [False]
        assert _FakeGC.enabled is False
        assert _FakeGC.disable_calls == 1
        assert cp._manual_scroll_gc_paused is True


async def test_chat_panel_scrollbar_drag_pauses_gc_before_base_scroll_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrollbar ScrollTo messages should pause GC before Textual scrolls once."""
    _, _FakeGC = install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await pilot.pause()
        scroll_to_calls: list[tuple[float | None, float | None, object, object, bool]] = []

        def allow_scroll(_panel: ChatPanel) -> bool:
            return True

        def scroll_to_spy(x: float | None = None, y: float | None = None, **kwargs: object) -> None:
            scroll_to_calls.append((x, y, kwargs.get("animate"), kwargs.get("duration"), _FakeGC.enabled))

        monkeypatch.setattr(ChatPanel, "_allow_scroll", property(allow_scroll))
        monkeypatch.setattr(cp, "scroll_to", scroll_to_spy)

        await cp._on_message(ScrollTo(y=10, animate=False))

        assert scroll_to_calls == [(None, 10, False, 0.1, False)]
        assert _FakeGC.enabled is False
        assert _FakeGC.disable_calls == 1
        assert cp._manual_scroll_gc_paused is True


async def test_chat_panel_clear_and_unmount_release_gc_pause_without_collect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear and unmount stop a pending resume timer and release the pause, never collecting."""
    _, _FakeGC = install_fake_chat_panel_gc(monkeypatch)

    class FakeTimer:
        def __init__(self, label: str, callback: Callable[[], None]) -> None:
            self.label = label
            self.callback = callback
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    timers: list[FakeTimer] = []

    def set_timer_spy(_delay: float, callback: Callable[[], None]) -> FakeTimer:
        timer = FakeTimer(callback.__name__, callback)
        timers.append(timer)
        return timer

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        monkeypatch.setattr(cp, "set_timer", set_timer_spy)

        cp._pause_gc_for_manual_scroll()
        gc_timer_for_clear = timers[-1]
        assert gc_timer_for_clear.label == "resume_gc_after_manual_scroll"

        await cp.clear()

        assert gc_timer_for_clear.stop_calls == 1
        assert cp._manual_scroll_gc_timer is None
        assert cp._manual_scroll_gc_paused is False
        assert _FakeGC.enabled is True
        assert _FakeGC.collect_generations == []

        cp._pause_gc_for_manual_scroll()
        gc_timer_for_unmount = timers[-1]
        assert gc_timer_for_unmount.label == "resume_gc_after_manual_scroll"

        cp.on_unmount()

        assert gc_timer_for_unmount.stop_calls == 1
        assert cp._manual_scroll_gc_timer is None
        assert cp._manual_scroll_gc_paused is False
        assert _FakeGC.enabled is True
        assert _FakeGC.collect_generations == []


async def test_chat_panel_clear_mid_grab_releases_gesture_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear during an active thumb grab must still restore GC (restorability parity)."""
    _, _FakeGC = install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        cp._scroll_controller.on_scrollbar_grab()
        assert _FakeGC.enabled is False

        await cp.clear()

        assert cp._scroll_controller.scrollbar_grabbed is False
        assert cp._manual_scroll_gc_paused is False
        assert _FakeGC.enabled is True
        assert _FakeGC.collect_generations == []
