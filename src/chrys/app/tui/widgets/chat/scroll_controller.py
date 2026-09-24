# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Scroll-follow and Textual anchor policy for chat transcripts."""

from __future__ import annotations

import gc
from typing import Any

from textual._arrange import DockArrangeResult
from textual.geometry import Offset, Region, Size
from textual.timer import Timer
from textual.widget import Widget

from chrys.app.tui.util.visibility import set_widget_visibility_without_layout
from chrys.app.tui.widgets.chat.ports import ChatScrollHost

_MANUAL_SCROLL_GC_RESUME_SECONDS = 0.25
_SCROLL_GC_PAUSE_OWNERS = 0
_SCROLL_GC_WAS_ENABLED = False


def scroll_gc_paused() -> bool:
    """Return whether any chat panel currently owns the shared GC pause."""
    return _SCROLL_GC_PAUSE_OWNERS > 0


def _claim_scroll_gc_pause() -> None:
    """Disable cyclic GC while at least one chat panel is actively scrolling."""
    global _SCROLL_GC_PAUSE_OWNERS, _SCROLL_GC_WAS_ENABLED
    if _SCROLL_GC_PAUSE_OWNERS == 0:
        _SCROLL_GC_WAS_ENABLED = gc.isenabled()
        if _SCROLL_GC_WAS_ENABLED:
            gc.disable()
    _SCROLL_GC_PAUSE_OWNERS += 1


def _release_scroll_gc_pause() -> None:
    """Restore cyclic GC when the last chat-panel scroll pause ends."""
    global _SCROLL_GC_PAUSE_OWNERS, _SCROLL_GC_WAS_ENABLED
    if _SCROLL_GC_PAUSE_OWNERS <= 0:
        return
    _SCROLL_GC_PAUSE_OWNERS -= 1
    if _SCROLL_GC_PAUSE_OWNERS == 0:
        was_enabled = _SCROLL_GC_WAS_ENABLED
        _SCROLL_GC_WAS_ENABLED = False
        if was_enabled:
            gc.enable()


class ManualScrollGcGuard:
    """Gesture-scoped GC pause for transcript surfaces without follow state."""

    def __init__(self, host: Widget) -> None:
        self._host = host
        self._timer: Timer | None = None
        self._paused = False
        self._scrollbar_grabbed = False

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        """Hold cyclic GC until the current scroll cadence settles."""
        if not self._paused:
            _claim_scroll_gc_pause()
            self._paused = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if not self._scrollbar_grabbed:
            self._timer = self._host.set_timer(_MANUAL_SCROLL_GC_RESUME_SECONDS, self.resume)

    def set_scrollbar_grabbed(self, grabbed: bool) -> None:
        """Hold the pause continuously across a scrollbar-thumb drag."""
        if grabbed:
            self._scrollbar_grabbed = True
            self.pause()
            return
        if not self._scrollbar_grabbed:
            return
        self._scrollbar_grabbed = False
        if self._paused:
            self.pause()

    def resume(self, *, schedule_collect: bool = True) -> None:
        """Release the shared pause after a quiet cadence interval."""
        self._timer = None
        if self._scrollbar_grabbed or not self._paused:
            return
        self._paused = False
        _release_scroll_gc_pause()
        if schedule_collect and gc.isenabled():
            gc.collect(0)

    def stop(self) -> None:
        """Release timers and shared ownership during surface teardown."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._scrollbar_grabbed = False
        self.resume(schedule_collect=False)


class ChatScrollController:
    """Own chat scroll-follow state, timers, and bottom-anchor policy."""

    def __init__(self, host: ChatScrollHost) -> None:
        self._host = host
        self.held_shrink_height: int = -1
        self.manual_scroll_gc_timer: Timer | None = None
        self.manual_scroll_gc_paused: bool = False
        self.scrollbar_grabbed: bool = False
        self.auto_scroll_paused_by_user: bool = False
        self.final_response_started: bool = False
        self.programmatic_scroll: bool = False
        self.anchor_sync_scheduled: bool = False
        self.agent_running: bool = False

    def set_agent_running(self, running: bool) -> None:
        """Mirror current run state and reset the final gate only on run start."""
        if running and not self.agent_running:
            self.final_response_started = False
        self.agent_running = running

    def on_user_turn_started(self) -> None:
        """Reset live-turn scroll gates."""
        self.auto_scroll_paused_by_user = False
        self.final_response_started = False

    def on_replay_started(self) -> None:
        """Reset replay scroll gates after welcome dismissal."""
        self.auto_scroll_paused_by_user = False
        self.final_response_started = False

    def scroll_user_message_to_top(self, widget: Widget) -> None:
        """Scroll a newly mounted user message to the viewport top."""
        self.programmatic_scroll = True
        try:
            widget.scroll_visible(top=True, immediate=True, animate=False)
        finally:
            self.programmatic_scroll = False

    def on_final_response_started(self) -> None:
        """Resume bottom-follow once when the agent starts its final answer."""
        if self.final_response_started:
            return
        self.final_response_started = True
        self.auto_scroll_paused_by_user = False
        if self._host.is_anchored():
            self._host.set_anchor_released(False)
        self.schedule_anchor_sync()

    def on_status_message_mounted(self) -> None:
        """Force a mounted status message to the bottom."""
        self.auto_scroll_paused_by_user = False
        if self._host.is_anchored():
            self._host.set_anchor_released(False)
        self._host.scroll_end(animate=False)

    def scroll_inline_prompt_to_top_after_refresh(self, widget: Widget) -> None:
        """Schedule inline ask_user prompt scroll-to-top after refresh."""
        self.auto_scroll_paused_by_user = False
        self._host.call_after_refresh(self._host.scroll_to_widget_top, widget)

    def after_replay_history_mounted(self) -> None:
        """Schedule final replay scroll-to-bottom."""
        self._host.call_after_refresh(self._host.scroll_end, animate=False)

    def scroll_to_region(self, region: Region, **kwargs: Any) -> Offset:
        """Ignore focus-restoration center scrolls inside the transcript."""
        if kwargs.get("center") and not kwargs.get("top"):
            on_complete = kwargs.get("on_complete")
            if callable(on_complete):
                self._host.call_after_refresh(on_complete)
            return Offset()
        return self._host.call_super_scroll_to_region(region, **kwargs)

    def check_anchor(self) -> None:
        """Re-engage bottom-follow only at exact bottom."""
        if (
            self._host.is_anchored()
            and self._host.is_anchor_released()
            and self._host.scroll_y >= self._host.max_scroll_y
        ):
            self._host.set_anchor_released(False)
            self.auto_scroll_paused_by_user = False

    def arrange(self, size: Size, *, optimal: bool = False) -> DockArrangeResult:
        """Defuse Textual anchor pins while chat layout is settling."""
        if self._host.is_anchored() and not self._host.is_anchor_released():
            new_container_h = size.height + self._host.scrollbar_size_horizontal
            container_size = self._host.get_container_size()
            if container_size.height != new_container_h:
                self._host.set_container_size(Size(container_size.width, new_container_h))
        result = self._host.call_super_arrange(size, optimal=optimal)
        self._host.place_fixed_scroll_button(result, size)
        virtual_h = result.spatial_map.total_region.bottom
        if virtual_h < size.height and self._host.scroll_y < 0:
            if self._host.is_anchored() and not self._host.is_anchor_released():
                self._host.set_anchor_released(True)
            self._host.set_scroll_y_reactive(0.0)
            self._host.set_scroll_target_y_reactive(0.0)
        elif (
            virtual_h < self._host.virtual_size.height
            and self._host.is_anchored()
            and not self._host.is_anchor_released()
        ):
            self._host.set_anchor_released(True)
            self._host.call_after_refresh(self.reanchor_after_settle)
        return result

    def reanchor_after_settle(self) -> None:
        """Re-engage the anchor and pin to settled ``max_scroll_y``."""
        if not self._host.is_anchored():
            return
        self._host.set_anchor_released(False)
        new_y = self._host.max_scroll_y
        if self._host.scroll_y != new_y:
            self.set_scroll_y_programmatically(new_y)
        self.sync_scroll_to_bottom_button()

    def set_scroll_y_programmatically(self, y: float) -> None:
        """Set vertical scroll without treating the movement as a user pause."""
        self.programmatic_scroll = True
        try:
            self._host.scroll_y = y
            self._host.set_scroll_target_y(y)
        finally:
            self.programmatic_scroll = False

    def jump_to_bottom(self) -> None:
        """Scroll to the current bottom and resume bottom-follow."""
        self.auto_scroll_paused_by_user = False
        if self._host.is_anchored():
            self._host.set_anchor_released(False)
        self.set_scroll_y_programmatically(self._host.max_scroll_y)
        self.schedule_anchor_sync()
        self.sync_scroll_to_bottom_button()

    def on_resize(self) -> None:
        """Refresh the bottom-jump affordance when viewport height changes."""
        self._host.call_after_refresh(self.sync_scroll_to_bottom_button)

    def size_updated(self, size: Size, virtual_size: Size, container_size: Size, layout: bool = True) -> bool:
        """Hold one anchored shrink frame before validator clamping."""
        if (
            self._host.is_anchored()
            and 0 < virtual_size.height < self._host.virtual_size.height
            and self.held_shrink_height != virtual_size.height
        ):
            self.held_shrink_height = virtual_size.height
            changed = self._host.apply_held_shrink_size_update(size, container_size)
            self._host.call_after_refresh(self.release_size_hold)
            return changed
        self.held_shrink_height = -1
        return self._host.call_super_size_updated(size, virtual_size, container_size, layout)

    def release_size_hold(self) -> None:
        """Force a layout pass that either recovers or accepts the held shrink."""
        if self.held_shrink_height < 0:
            return
        self._host.refresh(layout=True)

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        """Track manual scroll ticks: GC pause, follow-state, bottom button."""
        old_line = round(old_value)
        new_line = round(new_value)
        self._host.call_super_watch_scroll_y(old_value, new_value)
        if self._host.is_anchored() and not self._host.is_anchor_released() and new_value >= self._host.max_scroll_y:
            self.auto_scroll_paused_by_user = False
        if (
            self.agent_running
            and not self.programmatic_scroll
            and self._host.is_anchored()
            and self._host.is_anchor_released()
            and new_line < old_line
        ):
            self.auto_scroll_paused_by_user = True
        manual_scroll = old_line != new_line and not self.programmatic_scroll
        if manual_scroll:
            self.pause_gc_for_manual_scroll()
        self.sync_scroll_to_bottom_button()

    def sync_scroll_to_bottom_button(self) -> None:
        """Show the bottom-jump affordance only when there is unseen content below.

        Toggles ``visibility``, never ``display``: a display flip marks the
        screen layout-required and invalidates the compositor full map, which
        turns every bottom-boundary crossing during a scroll into an
        O(all widgets) reflow hitch on large transcripts.

        The rule is written directly instead of assigning ``button.visible``:
        Textual declares the ``visibility`` style property ``layout=True``, so
        the setter escalates every flip to that same full reflow. The
        compositor applies visibility at map-build time (``add_widget`` reads
        the current rule), and a flip can only happen because ``scroll_y`` or
        ``max_scroll_y`` changed — both of which already schedule the cheap
        visible-only re-arrange that refreshes map membership. The explicit
        ``refresh()`` covers repaint dirtiness for the button's region.
        """
        button = self._host.scroll_to_bottom_button()
        if button is None:
            return
        visible = self._host.max_scroll_y > 0 and round(self._host.scroll_y) < self._host.max_scroll_y
        set_widget_visibility_without_layout(button, visible)

    def pause_gc_for_manual_scroll(self) -> None:
        """Temporarily disable cyclic GC while user scrolling is active.

        While the scrollbar thumb is grabbed the pause is held open by the
        gesture itself: no resume debounce is armed, so a slow drag with long
        gaps between mouse moves can never re-enable GC (and eat the backlog
        collection) in the middle of the gesture.
        """
        if not self.manual_scroll_gc_paused:
            _claim_scroll_gc_pause()
            self.manual_scroll_gc_paused = True
        if self.manual_scroll_gc_timer is not None:
            self.manual_scroll_gc_timer.stop()
            self.manual_scroll_gc_timer = None
        if self.scrollbar_grabbed:
            return
        self.manual_scroll_gc_timer = self._host.set_timer(
            _MANUAL_SCROLL_GC_RESUME_SECONDS,
            self.resume_gc_after_manual_scroll,
        )

    def on_scrollbar_grab(self) -> None:
        """Hold the GC pause for the whole scrollbar drag gesture."""
        self.scrollbar_grabbed = True
        self.pause_gc_for_manual_scroll()

    def on_scrollbar_release(self) -> None:
        """Return to the cadence debounce once the drag gesture ends.

        Trailing scroll animation frames keep restarting the debounce, so
        resume still happens a quiet interval after the last movement.
        """
        if not self.scrollbar_grabbed:
            return
        self.scrollbar_grabbed = False
        if self.manual_scroll_gc_paused:
            self.pause_gc_for_manual_scroll()

    def resume_gc_after_manual_scroll(self, *, schedule_collect: bool = True) -> None:
        """Restore cyclic GC after manual scrolling has settled.

        Immediately drains the paused-scroll allocation backlog (gen0) at
        this known-quiet moment: the gen0 counter keeps growing while GC is
        disabled, so the next organic trigger would otherwise fire on
        whatever allocation comes first — historically the middle of the
        next drag movement.  Skipped while another panel still holds the
        pause (GC still disabled) or during teardown paths that pass
        ``schedule_collect=False``.
        """
        self.manual_scroll_gc_timer = None
        if self.scrollbar_grabbed:
            return
        if not self.manual_scroll_gc_paused:
            return
        self.manual_scroll_gc_paused = False
        _release_scroll_gc_pause()
        if schedule_collect and gc.isenabled():
            gc.collect(0)

    def schedule_anchor_sync(self) -> None:
        """Sync scroll position to the bottom after the next refresh."""
        if (
            self._host.is_anchored()
            and not self._host.is_anchor_released()
            and not self.auto_scroll_paused_by_user
            and not self.anchor_sync_scheduled
            and self._host.call_after_refresh(self.sync_anchor_bottom)
        ):
            self.anchor_sync_scheduled = True

    def sync_anchor_bottom(self) -> None:
        """Push ``scroll_y`` to the current ``max_scroll_y`` via setter."""
        self.anchor_sync_scheduled = False
        if not (
            self._host.is_anchored() and not self._host.is_anchor_released() and not self.auto_scroll_paused_by_user
        ):
            self.sync_scroll_to_bottom_button()
            return
        new_y = self._host.max_scroll_y
        if self._host.scroll_y != new_y:
            self.set_scroll_y_programmatically(new_y)
        self.sync_scroll_to_bottom_button()

    def watch_virtual_size(self, old: object, new: object) -> None:
        """Post-layout hook: re-engage anchor, sync scrollbar, toggle spacer."""
        old_h = getattr(old, "height", 0)
        new_h = getattr(new, "height", 0)
        if (
            self.agent_running
            and new_h > old_h
            and self._host.is_anchored()
            and self._host.is_anchor_released()
            and not self.auto_scroll_paused_by_user
        ):
            max_y = max(0, new_h - self._host.size.height)
            if max_y > self._host.scroll_y + 2:
                self._host.set_anchor_released(False)
        if new_h > old_h and self._host.is_anchored() and not self._host.is_anchor_released():
            new_y = self._host.max_scroll_y
            if self._host.scroll_y != new_y:
                self.set_scroll_y_programmatically(new_y)
        spacer = self._host.bottom_spacer()
        if spacer is not None:
            viewport_h = self._host.size.height
            if spacer.display:
                natural_h = new_h - spacer.size.height
                if natural_h >= viewport_h:
                    spacer.display = False
            elif new_h < viewport_h:
                spacer.display = True
        self.sync_scroll_to_bottom_button()

    def reset_for_clear(self) -> None:
        """Reset transcript scroll state and release the GC pause without collecting."""
        if self.manual_scroll_gc_timer is not None:
            self.manual_scroll_gc_timer.stop()
            self.manual_scroll_gc_timer = None
        self.scrollbar_grabbed = False
        self.resume_gc_after_manual_scroll(schedule_collect=False)
        self.held_shrink_height = -1
        self.agent_running = False
        self.auto_scroll_paused_by_user = False
        self.final_response_started = False
        self.programmatic_scroll = False
        self.anchor_sync_scheduled = False
        self._host.anchor()

    def stop(self) -> None:
        """Shutdown timer/GC state for widget unmount."""
        if self.manual_scroll_gc_timer is not None:
            self.manual_scroll_gc_timer.stop()
            self.manual_scroll_gc_timer = None
        self.scrollbar_grabbed = False
        self.resume_gc_after_manual_scroll(schedule_collect=False)
