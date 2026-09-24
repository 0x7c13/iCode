# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for ChatPanel scrolling: bottom spacer, scroll-to-bottom affordance, anchor/autoscroll contract."""

from __future__ import annotations

import pytest
from textual.geometry import Size
from textual.widget import Widget
from textual.widgets import Static

from chrys.app.tui.widgets.chat.panel import ChatPanel, _ChatBottomSpacer, _ScrollToBottomButton
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from tests.app.tui.widgets._scroll_gc import install_fake_chat_panel_gc
from tests.support.tui_helpers import (
    ChatPanelApp,
    _simulate_chat_panel_user_scroll_y,
)
from tests.support.waiting import wait_for


async def _fill_eight_turns(cp: ChatPanel, pilot: object) -> None:
    """Add eight user/agent turns so an 80x20 panel overflows, then let layout settle."""
    body = "\n".join(f"line {i}" for i in range(8))
    for i in range(8):
        await cp.add_user_message(f"message {i}")
        await cp.add_agent_message(body)
    await pilot.pause()


async def _scrolled_to_bottom(cp: ChatPanel, pilot: object) -> None:
    """Fill the panel with eight turns and settle it at the exact bottom."""
    await _fill_eight_turns(cp, pilot)
    cp.scroll_end(immediate=True, animate=False)
    await pilot.pause()


async def test_chat_panel_bottom_spacer_lifecycle() -> None:
    """Spacer is hidden during welcome, shown after it's dismissed, and always trails content.

    The spacer's ``height: 1fr`` has an effective ``min-height: 1`` in Textual —
    leaving it visible while the welcome widget (``height: 100%``) is present
    produces a phantom 1-row scrollbar.  After the first user message the
    welcome is removed and the spacer becomes the flexible tail that gives
    ``scroll_visible(top=True)`` room to scroll into.
    """
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        spacer = cp.query_one(_ChatBottomSpacer)
        scroll_button = cp.query_one(_ScrollToBottomButton)

        # Welcome state: spacer exists but is hidden.
        assert spacer.display is False
        # Spacer must trail content so new mounts land before it.
        assert cp.children[-2] is spacer
        assert cp.children[-1] is scroll_button

        await cp.add_user_message("hello")
        # Welcome dismissed → spacer is now visible.
        assert spacer.display is True
        # Spacer is still immediately before the floating affordance.
        assert cp.children[-2] is spacer
        assert cp.children[-1] is scroll_button

        await cp.add_agent_message("hi there")
        await pilot.pause()
        assert cp.children[-2] is spacer
        assert cp.children[-1] is scroll_button

        # clear() recreates the spacer (fresh welcome) and re-hides it.
        await cp.clear()
        spacer2 = cp.query_one(_ChatBottomSpacer)
        scroll_button2 = cp.query_one(_ScrollToBottomButton)
        assert spacer2.display is False
        assert cp.children[-2] is spacer2
        assert cp.children[-1] is scroll_button2


async def test_chat_panel_scroll_to_bottom_affordance_visibility_and_jump() -> None:
    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        button = cp.query_one(_ScrollToBottomButton)

        await cp._dismiss_welcome()
        for index in range(60):
            await cp.mount(Static(f"line {index}"))
        await pilot.pause()

        assert cp.max_scroll_y > 0

        cp._set_scroll_y_programmatically(cp.max_scroll_y - 0.4)
        cp._sync_scroll_to_bottom_button()
        assert button.visible is False, "sub-cell scroll positions that round to bottom should hide the affordance"

        cp._set_scroll_y_programmatically(cp.max_scroll_y - 1)
        cp._sync_scroll_to_bottom_button()
        assert button.visible is True

        cp._auto_scroll_paused_by_user = True
        cp._anchor_released = True
        cp.jump_to_bottom()
        await pilot.pause()

        assert cp._auto_scroll_paused_by_user is False
        assert cp._anchor_released is False
        assert round(cp.scroll_y) == cp.max_scroll_y
        assert button.visible is False
        # The affordance is toggled via visibility, never display: a display
        # flip would force a full reflow + compositor full-map rebuild on
        # every bottom-boundary crossing while scrolling a large transcript.
        assert button.display is True


async def test_chat_panel_mount_survives_detached_spacer_reference() -> None:
    """``mount()`` must not crash if ``_bottom_spacer`` points at a detached widget.

    Regression: on Windows session restore a race between ``clear()`` (awaiting
    ``remove_children``) and an incoming event-handler mount caused
    ``MountError: Unable to find relative location of _ChatBottomSpacer``.
    The override now guards on ``spacer.parent is self``; this test forces
    that exact shape by swapping in an unattached ``_ChatBottomSpacer``
    instance and then mounting a normal widget.
    """
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        # Simulate the race: a stale reference to a widget that is not our
        # child.  A freshly constructed spacer has ``parent is None``, which
        # is the same state as a just-removed child during the async yield.
        detached = _ChatBottomSpacer()
        cp._bottom_spacer = detached
        assert detached.parent is None, "test setup precondition"

        survivor = Static("survivor", id="survivor")
        # Must not raise MountError.
        await cp.mount(survivor)

        # The new widget landed (appended) and the detached spacer was
        # correctly ignored rather than used as a ``before=`` target.
        assert survivor.parent is cp
        assert detached.parent is None, "detached spacer must not have been attached by the mount path"


async def test_chat_panel_bottom_spacer_reshows_after_shrink() -> None:
    """Spacer re-appears when natural content shrinks back below the viewport.

    Regression for a bug where ``watch_virtual_size`` only hid the spacer and
    never restored it: once a tool group auto-collapsed its first expanded
    tool (triggered by a second tool start), the shrunk canvas was aligned
    to the viewport bottom by the bottom-pinning anchor, leaving empty
    padding above the current turn's user message.
    """
    from textual.geometry import Size

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        spacer = cp.query_one(_ChatBottomSpacer)

        # Dismiss welcome so the spacer enters its normal lifecycle.
        await cp.add_user_message("hello")
        await pilot.pause()
        assert spacer.display is True

        viewport_h = cp.size.height
        assert viewport_h > 0, "test harness must give the panel a real viewport"

        # Simulate natural content growing past the viewport: watch_virtual_size
        # sees new_h == natural + spacer_h, and the branch compares the
        # computed natural_h against viewport_h.  With the spacer collapsed
        # to ~0 here (natural alone is already tall), new_h >= viewport_h.
        cp.watch_virtual_size(Size(cp.size.width, 0), Size(cp.size.width, viewport_h + 20))
        assert spacer.display is False, "spacer should hide once natural content fills viewport"

        # Now simulate natural content shrinking back below the viewport
        # (tool-group collapse).  With spacer hidden, new_h == natural_h.
        cp.watch_virtual_size(
            Size(cp.size.width, viewport_h + 20),
            Size(cp.size.width, max(1, viewport_h - 5)),
        )
        assert spacer.display is True, "spacer must re-show so the user message is not bottom-aligned"


async def test_chat_panel_dismiss_welcome_preserves_hidden_spacer_after_first_turn() -> None:
    """After welcome is gone, dismissing it again must not re-show a hidden spacer."""

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        spacer = cp.query_one(_ChatBottomSpacer)
        await pilot.pause()

        await cp.add_user_message("hello")
        await pilot.pause()
        assert cp._welcome is None
        assert spacer.display is True

        spacer.display = False
        await cp._dismiss_welcome()
        assert spacer.display is False


async def test_chat_panel_arrange_defuses_short_canvas_anchor_pin() -> None:
    """When the arranged canvas is shorter than the container, ``arrange`` must
    release the bottom-anchor + reset ``scroll_y`` BEFORE the compositor reads
    the flag in the compositor's anchored-pin branch.

    Regression for a bug where, whenever the scrollbar wasn't at the top,
    collapsing tool groups left content docked at the viewport bottom with a
    blank band at the top.  Textual's compositor pin writes
    ``scroll_y = total_region.bottom - container_h`` via ``set_reactive``,
    bypassing the [0, max_scroll_y] clamp — a NEGATIVE value when
    virtual_h < container_h, which renders content offset down within the
    viewport.  ``watch_virtual_size`` runs too late to defuse it; the hook
    has to live in ``arrange`` itself.
    """
    from textual.geometry import Size

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        spacer = cp.query_one(_ChatBottomSpacer)

        # Dismiss welcome so the normal spacer lifecycle is in effect.
        await cp.add_user_message("hello")
        await pilot.pause()

        container_h = cp.size.height
        assert container_h > 0

        # Put the panel in the "spacer hidden, canvas short" state we hit
        # after a tool group auto-collapses back below the viewport.
        spacer.display = False
        await pilot.pause()

        # Simulate exactly what the compositor's anchored-pin branch does: write a
        # negative scroll_y via ``set_reactive`` (which bypasses the
        # validator's [0, max_scroll_y] clamp) while the anchor is still
        # engaged.  A plain ``cp.scroll_y = -5`` would get clamped to 0
        # on write and wouldn't reproduce the bug.
        from textual.widget import Widget

        cp._anchor_released = False
        cp.set_reactive(Widget.scroll_y, -5.0)
        assert cp.scroll_y == -5.0, "set_reactive must bypass validator (sanity check)"

        # Arrange against a container taller than the (now-tiny) content.
        cp.arrange(Size(cp.size.width, container_h))

        assert cp._anchor_released is True, (
            "anchor must be released inside arrange() so the compositor skips the pin block on this same pass"
        )
        assert cp.scroll_y == 0, "stale pinned scroll_y must be reset before compositor paints"


async def test_chat_panel_arrange_scroll_reset_does_not_re_engage_anchor() -> None:
    """The scroll_y reset in ``arrange`` must NOT go through the reactive setter.

    ``watch_scroll_y`` calls ``_check_anchor`` which re-engages the anchor
    whenever ``scroll_y >= max_scroll_y``.  When the canvas fits the
    viewport, both are 0, so the check would trivially re-engage the anchor
    right after we just released it — bringing the negative compositor pin
    straight back on the next layout.  ``arrange`` must use
    ``set_reactive`` to bypass the watcher.
    """
    from textual.geometry import Size

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        spacer = cp.query_one(_ChatBottomSpacer)

        await cp.add_user_message("hello")
        await pilot.pause()

        spacer.display = False
        await pilot.pause()

        container_h = cp.size.height
        cp._anchor_released = False
        cp.set_reactive(Widget.scroll_y, -3.0)

        cp.arrange(Size(cp.size.width, container_h))

        assert cp._anchor_released is True, (
            "anchor must STAY released after scroll_y reset — a reactive setter "
            "would trigger _check_anchor and re-engage with scroll_y >= max_scroll_y (both 0)"
        )


async def test_chat_panel_manual_scroll_schedules_no_repaint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual scrolling must not refresh the panel or arm any repaint timer.

    The old debounced cleanup repaint fired mid-gesture on slow scrollbar
    drags.  The scroll fast path repaints every changed cell on its own
    (pinned by ``test_chat_scroll_fastpath.py``), so the only timer a manual
    scroll tick may arm is the GC resume debounce.
    """
    scroll_controller_module, _FakeGC = install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        for i in range(10):
            await cp.add_user_message(f"message {i}")
        await pilot.pause()

        assert cp.max_scroll_y > 1, "test setup must produce scrollable content"
        cp.scroll_y = 0
        await pilot.pause()

        calls = 0
        original_refresh = cp.refresh

        def refresh_spy(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return original_refresh(*args, **kwargs)

        monkeypatch.setattr(cp, "refresh", refresh_spy)

        timers: list[tuple[float, object]] = []

        class FakeTimer:
            def stop(self) -> None:
                pass

        def set_timer_spy(delay: float, callback: object) -> FakeTimer:
            timers.append((delay, callback))
            return FakeTimer()

        monkeypatch.setattr(cp, "set_timer", set_timer_spy)

        cp._anchor_released = True
        cp.scroll_y = 1

        assert calls == 0, "released-anchor scroll must not repaint the panel"
        gc_resume = scroll_controller_module._MANUAL_SCROLL_GC_RESUME_SECONDS
        non_gc_timers = [timer for timer in timers if timer[0] != gc_resume]
        assert non_gc_timers == [], "manual scroll must only arm the GC resume debounce"

        cp._anchor_released = False
        cp.scroll_y = 2

        assert calls == 0, "anchored auto-follow must keep Textual's scroll fast path"
        non_gc_timers = [timer for timer in timers if timer[0] != gc_resume]
        assert non_gc_timers == []


async def test_chat_panel_user_pause_blocks_growth_reanchor() -> None:
    """New tool/intermediate growth must not override an explicit user scroll-up."""
    from textual.geometry import Size

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = True
        cp.set_reactive(Widget.scroll_y, 5.0)

        cp.watch_virtual_size(Size(cp.size.width, 20), Size(cp.size.width, 80))

        assert cp._anchor_released is True
        assert cp._auto_scroll_paused_by_user is True


async def test_chat_panel_soft_release_still_reanchors_on_growth() -> None:
    """The user-message soft-release flow should still reanchor on real growth."""
    from textual.geometry import Size

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = False
        cp.set_reactive(Widget.scroll_y, 0.0)

        cp.watch_virtual_size(Size(cp.size.width, 20), Size(cp.size.width, 80))

        assert cp._anchor_released is False


async def test_chat_panel_check_anchor_reengages_at_exact_bottom_only() -> None:

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        for i in range(12):
            await cp.add_user_message(f"message {i}")
        await pilot.pause()
        assert cp.max_scroll_y > 1

        near_bottom = cp.max_scroll_y - 1

        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = False
        cp.set_reactive(Widget.scroll_y, near_bottom)
        cp._check_anchor()
        assert cp._anchor_released is True

        cp._auto_scroll_paused_by_user = True
        cp._check_anchor()
        assert cp._anchor_released is True
        assert cp._auto_scroll_paused_by_user is True

        cp.set_reactive(Widget.scroll_y, cp.max_scroll_y)
        cp._check_anchor()
        assert cp._anchor_released is False
        assert cp._auto_scroll_paused_by_user is False


async def test_chat_panel_second_upward_scroll_stays_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await _scrolled_to_bottom(cp, pilot)
        bottom_y = cp.scroll_y
        assert bottom_y == cp.max_scroll_y > 6

        cp.agent_running = True
        _simulate_chat_panel_user_scroll_y(cp, bottom_y - 3)
        after_first_scroll = cp.scroll_y

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True
        assert bottom_y - after_first_scroll == 3

        _simulate_chat_panel_user_scroll_y(cp, after_first_scroll - 3)

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True
        assert cp.scroll_y < after_first_scroll
        assert cp.scroll_y < cp.max_scroll_y


async def test_chat_panel_tool_growth_follows_after_exact_bottom_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrolling back to the exact bottom resumes autoscroll, and later tool growth keeps following."""
    install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await _scrolled_to_bottom(cp, pilot)
        assert cp.max_scroll_y > 6

        cp.agent_running = True
        _simulate_chat_panel_user_scroll_y(cp, cp.max_scroll_y - 6)

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True
        assert cp.max_scroll_y - cp.scroll_y == 6

        cp.scroll_end(immediate=True, animate=False)
        await pilot.pause()

        assert cp._auto_scroll_paused_by_user is False
        assert cp._anchor_released is False
        assert cp.scroll_y == cp.max_scroll_y

        await cp.add_tool_start("call-1", "zsh", "shell", '{"cmd":"echo hi"}')
        await pilot.pause()

        assert cp.scroll_y == cp.max_scroll_y


async def test_chat_panel_tool_growth_stays_detached_after_user_scroll_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detached state should block tool anchor-sync and growth reanchor."""
    from textual.geometry import Size

    install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await _fill_eight_turns(cp, pilot)

        assert cp.max_scroll_y > 6
        cp._set_scroll_y_programmatically(cp.max_scroll_y)
        cp.agent_running = True

        _simulate_chat_panel_user_scroll_y(cp, cp.max_scroll_y - 6)
        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True

        # The setup mounts arm a coalesced anchor-sync via call_after_refresh;
        # whether it has drained by now is refresh-timing dependent (flaky on
        # Windows CI). Reset to a known baseline so this isolates the detached
        # guard: while the anchor is released, _schedule_anchor_sync must not arm.
        cp._anchor_sync_scheduled = False
        cp._schedule_anchor_sync()
        assert cp._anchor_sync_scheduled is False

        cp.watch_virtual_size(Size(cp.size.width, 20), Size(cp.size.width, 80))

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True


async def test_chat_panel_anchor_sync_is_coalesced(monkeypatch: pytest.MonkeyPatch) -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        calls: list[object] = []

        def call_after_refresh_spy(callback: object, *args: object, **kwargs: object) -> bool:
            del args, kwargs
            calls.append(callback)
            return True

        monkeypatch.setattr(cp, "call_after_refresh", call_after_refresh_spy)

        cp._anchor_released = False
        cp._schedule_anchor_sync()
        cp._schedule_anchor_sync()
        cp.update_tool_progress("missing", ["ignored"])
        cp.update_tool_args("missing", {"ignored": True})

        assert len(calls) == 1
        assert cp._anchor_sync_scheduled is True

        callback = calls[0]
        assert callable(callback)
        callback()

        assert cp._anchor_sync_scheduled is False
        cp._schedule_anchor_sync()
        assert len(calls) == 2

        cp._anchor_sync_scheduled = False
        cp._auto_scroll_paused_by_user = True
        cp._schedule_anchor_sync()
        assert len(calls) == 2
        cp._auto_scroll_paused_by_user = False

        def call_after_refresh_closed(_callback: object, *args: object, **kwargs: object) -> bool:
            del args, kwargs
            return False

        monkeypatch.setattr(cp, "call_after_refresh", call_after_refresh_closed)
        cp._anchor_sync_scheduled = False
        cp._schedule_anchor_sync()
        assert cp._anchor_sync_scheduled is False


async def test_chat_panel_bottom_clamp_does_not_mark_user_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """A layout clamp to exact bottom is not a user scroll-up gesture."""

    install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await _fill_eight_turns(cp, pilot)

        assert cp.max_scroll_y > 0
        cp.agent_running = True
        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = False
        cp.set_reactive(Widget.scroll_y, float(cp.max_scroll_y))

        cp.watch_scroll_y(float(cp.max_scroll_y + 1), float(cp.max_scroll_y))

        assert cp._anchor_released is False
        assert cp._auto_scroll_paused_by_user is False


async def test_chat_panel_intermediate_message_does_not_resume_user_pause() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = True
        cp._final_response_started = False

        await cp.add_agent_message("checking files", is_intermediate=True)

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True
        assert cp._final_response_started is False


async def test_chat_panel_final_response_resumes_once() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = True
        cp._final_response_started = False

        await cp.add_agent_message("final starts", is_final=False)

        # The mount's layout frame can transiently release the anchor with a
        # deferred re-anchor, so wait until the resume state lands.
        await wait_for(
            lambda: cp._auto_scroll_paused_by_user is False and cp._anchor_released is False,
            pilot=pilot,
            description="final-response autoscroll resume",
        )
        assert cp._final_response_started is True

        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = True

        await cp.add_agent_message("final complete", is_final=True)

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True


async def test_chat_panel_final_response_scrolls_detached_to_bottom(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await _scrolled_to_bottom(cp, pilot)
        assert cp.max_scroll_y > 6

        cp.agent_running = True
        _simulate_chat_panel_user_scroll_y(cp, cp.max_scroll_y - 6)
        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True

        await cp.add_agent_message("final starts", is_final=False)

        # Same deferred re-anchor settle as above: one pause is not enough on
        # slow runners, so wait until the yank to the bottom lands.
        await wait_for(
            lambda: (
                cp._auto_scroll_paused_by_user is False
                and cp._anchor_released is False
                and cp.scroll_y == cp.max_scroll_y
            ),
            pilot=pilot,
            description="final-response bottom yank",
        )


@pytest.mark.parametrize(
    ("status_kind", "running_when_status"),
    [
        ("error", False),
        ("interrupted", False),
        ("retry", True),
    ],
)
async def test_chat_panel_status_messages_scroll_detached_to_bottom(
    monkeypatch: pytest.MonkeyPatch,
    status_kind: str,
    running_when_status: bool,
) -> None:
    install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await _scrolled_to_bottom(cp, pilot)
        assert cp.max_scroll_y > 6

        cp.agent_running = True
        _simulate_chat_panel_user_scroll_y(cp, cp.max_scroll_y - 6)
        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True

        cp.agent_running = running_when_status
        if status_kind == "error":
            await cp.add_error("boom")
        elif status_kind == "interrupted":
            await cp.add_interrupted()
        else:
            await cp.add_retry("retrying", 1, 2, 0)

        # The status-message yank re-anchors to the bottom, but on a shrink
        # frame ``arrange`` transiently releases the anchor and defers the
        # re-anchor to ``_reanchor_after_settle`` via ``call_after_refresh``.
        # On slower CI runners (Windows) that settle can span more than one
        # frame, so pump the loop until it lands instead of asserting after a
        # single pause.
        await wait_for(
            lambda: (
                cp._anchor_released is False
                and cp._auto_scroll_paused_by_user is False
                and cp.scroll_y == cp.max_scroll_y
            ),
            pilot=pilot,
            description="cp._anchor_released is False and cp._auto_scroll_paused_by_user is False and cp.scroll_y == cp.max_scroll_y",
        )

        assert cp._auto_scroll_paused_by_user is False
        assert cp._anchor_released is False
        assert cp.scroll_y == cp.max_scroll_y


async def test_chat_panel_agent_running_resets_final_response_autoscroll_gate() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        await cp.add_agent_message("first run starts", is_final=False)

        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = True
        await cp.add_agent_message("first run final", is_final=True)
        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True

        cp.agent_running = False
        cp.agent_running = True
        await cp.add_agent_message("retry starts", is_final=False)

        # The mount's layout frame can transiently release the anchor and
        # defer the re-anchor via ``call_after_refresh`` (see the settle loop
        # in the sibling test above), so wait until the reset state lands.
        await wait_for(
            lambda: cp._auto_scroll_paused_by_user is False and cp._anchor_released is False,
            pilot=pilot,
            description="rerun autoscroll gate reset",
        )
        assert cp._final_response_started is True


async def test_chat_panel_agent_running_post_run_disables_running_only_scroll_state() -> None:
    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await _scrolled_to_bottom(cp, pilot)

        cp.agent_running = True
        cp.agent_running = False
        cp._auto_scroll_paused_by_user = False
        cp._anchor_released = True

        _simulate_chat_panel_user_scroll_y(cp, max(0, cp.scroll_y - 5))

        assert cp._agent_running is False
        assert cp._auto_scroll_paused_by_user is False

        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = False
        cp.watch_virtual_size(Size(80, 10), Size(80, 200))

        assert cp._anchor_released is True


async def test_chat_panel_programmatic_upward_scroll_does_not_pause() -> None:

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        cp.set_reactive(Widget.scroll_y, 10.0)

        cp._set_scroll_y_programmatically(5.0)

        assert cp._auto_scroll_paused_by_user is False


async def test_chat_panel_scroll_to_turn_pauses_running_autoscroll() -> None:
    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("first")
        await cp.add_agent_message("one")
        await cp.add_user_message("second")
        await pilot.pause()

        cp.agent_running = True
        cp.scroll_to_turn("turn-1")

        assert cp._auto_scroll_paused_by_user is True


async def test_chat_panel_toc_navigation_running_state_post_run_does_not_pause_autoscroll() -> None:
    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("first")
        await cp.add_agent_message("one")
        await cp.add_user_message("second")
        await pilot.pause()

        cp.agent_running = True
        cp.agent_running = False
        cp._auto_scroll_paused_by_user = False

        cp.scroll_to_turn("turn-1")

        assert cp._agent_running is False
        assert cp._auto_scroll_paused_by_user is False


async def test_chat_panel_ignores_focus_center_scroll() -> None:
    """Focus restoration must not yank chat back to an old markdown child.

    Textual schedules ``scroll_to_center`` for focused widgets.  After a
    modal dismisses, a stale focused markdown descendant can otherwise center
    the first assistant body and leave the first user message at the top.
    """

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await _scrolled_to_bottom(cp, pilot)
        bottom_y = cp.scroll_y
        assert bottom_y == cp.max_scroll_y > 0, "test setup must produce a bottom-scrolled panel"

        first_markdown = cp.query(VirtualizedMarkdown).first()
        cp.scroll_to_widget(first_markdown, center=True, animate=False, immediate=True)
        await pilot.pause()

        # Tolerate a 1-row drift: on Windows Textual's headless harness can
        # recompute virtual height on the layout pass triggered by
        # ``scroll_to_widget`` and clamp scroll_y down by one even though
        # ``scroll_to_region`` returned Offset() early.  A real yank from
        # focus-centring would move the transcript by many rows.
        assert abs(cp.scroll_y - bottom_y) <= 1, "focus-centering a chat descendant must not move the transcript"
        assert cp._anchor_released is False, "ignored focus scroll must not release bottom-follow"

        cp.scroll_to_widget(first_markdown, top=True, animate=False, immediate=True)
        await pilot.pause()
        assert cp.scroll_y < bottom_y, "explicit top navigation must still work"


async def test_chat_panel_textual_anchor_contract() -> None:
    """Regression test: guard the private Textual APIs ChatPanel reaches into.

    ``panel.py`` touches four pieces of Textual's internal anchor/scroll state:

    * ``widget._anchored`` — bool flag set by ``widget.anchor()``
    * ``widget._anchor_released`` — bool toggled on user scroll-away / restored on scroll-to-bottom
    * ``widget._container_size`` — ``Size`` mutated inside our ``arrange()`` override so the compositor's anchor pin reads the fresh value
    * ``widget.anchor()`` / ``widget.release_anchor()`` / ``scroll_visible(top=...)``

    All four are either private (``_``-prefixed) or depend on private state, so a
    Textual upgrade could rename / remove them silently.  This test asserts the
    contract ``ChatPanel`` depends on so such a regression surfaces immediately
    rather than producing invisible scroll/flash glitches at runtime.

    If this test fails after a Textual bump, re-audit the corresponding code
    in ``src/chrys/app/tui/widgets/chat/panel.py`` (``arrange`` override +
    ``watch_virtual_size``) — the scroll behaviour likely needs to be
    ported to whatever replaced the private API.
    """
    import inspect

    from textual.geometry import Size

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        # --- Attribute surface ------------------------------------------------
        # Must exist as bool flags we can read/write.
        assert isinstance(cp._anchored, bool), "Textual removed/renamed Widget._anchored"
        assert isinstance(cp._anchor_released, bool), "Textual removed/renamed Widget._anchor_released"
        # _container_size must be a Size (we mutate it with a Size literal).
        assert isinstance(cp._container_size, Size), "Textual changed _container_size type"
        assert hasattr(cp._container_size, "width") and hasattr(cp._container_size, "height")

        # --- Method surface ---------------------------------------------------
        # anchor() / release_anchor() are the public entry points we call.
        assert callable(getattr(cp, "anchor", None)), "Textual removed Widget.anchor()"
        assert callable(getattr(cp, "release_anchor", None)), "Textual removed Widget.release_anchor()"
        # arrange(size, optimal=False) is the method we override.
        arrange_params = list(inspect.signature(cp.arrange).parameters.keys())
        assert arrange_params[:2] == ["size", "optimal"], (
            f"Textual changed Widget.arrange signature (got {arrange_params!r}); our override must match"
        )

        # --- Behavioural contract --------------------------------------------
        # on_mount() calls self.anchor() → _anchored must become True.
        # (ChatPanel.on_mount ran during compose; re-assert for clarity.)
        cp.anchor()
        assert cp._anchored is True, "anchor() must set _anchored=True"

        # release_anchor() flips _anchor_released True — this is what
        # scroll_visible(top=True) triggers, and what watch_virtual_size
        # flips back to False to re-engage auto-follow.
        cp._anchor_released = False
        cp.release_anchor()
        assert cp._anchor_released is True, "release_anchor() must set _anchor_released=True"

        # We manually clear _anchor_released to re-engage — verify that's
        # still a plain attribute and not e.g. a read-only property.
        cp._anchor_released = False
        assert cp._anchor_released is False, (
            "Textual made _anchor_released read-only; watch_virtual_size's manual re-engage will break"
        )

        # _container_size must be writable — we reassign it inside arrange().
        original = cp._container_size
        cp._container_size = Size(original.width, original.height + 1)
        assert cp._container_size.height == original.height + 1, (
            "Textual made _container_size read-only; our arrange() override will break"
        )
        cp._container_size = original  # restore


# ---------------------------------------------------------------------------
# InputBar (enhanced)
# ---------------------------------------------------------------------------
