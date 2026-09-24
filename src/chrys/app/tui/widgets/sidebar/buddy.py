# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Buddy panel widget for the sidebar."""

from __future__ import annotations

import asyncio
import contextlib
from time import monotonic
from typing import TYPE_CHECKING, Self

from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from chrys.app.features.buddy.actions import current_buddy, record_pet
from chrys.app.features.buddy.animation import get_idle_frame, get_pet_frame
from chrys.app.features.buddy.portrait import (
    PORTRAIT_HEIGHT,
    PORTRAIT_WIDTH,
    RARITY_COLORS,
    SHINY_FPS,
    render_portrait,
)
from chrys.app.tui.buddy_messages import BUDDY_TITLE
from chrys.app.tui.buddy_reply import PetReplyFlow
from chrys.app.tui.i18n import render_str
from chrys.app.tui.util.visibility import is_widget_shown_on_active_screen
from chrys.app.tui.widgets.sidebar.empty_state import SidebarEmptyStateLabel
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from textual.events import Click
    from textual.geometry import Region
    from textual.timer import Timer

    from chrys.app.features.buddy.model import Buddy
    from chrys.app.tui.i18n import LocaleController

_IDLE_TICK_SECONDS = 0.4
_PETTING_SECONDS = 2.0
_PETTING_FPS = 10
# The save file changes behind the panel's back: every finished turn credits it, and so does
# every other running instance. A shown panel looks again this often.
_RELOAD_SECONDS = 4.0

_BUDDY_EMPTY = msg("tui.sidebar.buddy.empty", fallback="No buddy hatched yet. Try /buddy hatch")
_BUDDY_LEVEL = msg("tui.sidebar.buddy.level", fallback="Level {level}")
_BUDDY_PROGRESS = msg("tui.sidebar.buddy.progress", fallback="{xp}/{needed} XP")
_BUDDY_CLICK_TO_PET = msg("tui.sidebar.buddy.click_to_pet", fallback="Click to pet!")
_BUDDY_PETTING = msg("tui.sidebar.buddy.petting", fallback="Petting...")
_BUDDY_SPECIES = msg("tui.sidebar.buddy.species", fallback="Species: {species}")
_BUDDY_RARITY = msg("tui.sidebar.buddy.rarity", fallback="Rarity: {rarity}")
_BUDDY_SHINY = msg("tui.sidebar.buddy.shiny", fallback="✨ Shiny!")
_BUDDY_NOTIFICATIONS_MUTED = msg("tui.sidebar.buddy.notifications_muted", fallback="🔇 Notifications muted")

_CONTENT_IDS = ("#buddy-sprite", "#buddy-level", "#buddy-info", "#buddy-status")


def _run_timer(timer: Timer, running: bool) -> None:
    """Pause or resume a repeating timer; a paused timer sleeps nothing until resumed."""
    if running:
        timer.resume()
    else:
        timer.pause()


class _BuddySprite(Static):
    """Fixed portrait chrome whose repaint path uses cached geometry."""

    ALLOW_SELECT = False

    def refresh(
        self,
        *regions: Region,
        repaint: bool = True,
        layout: bool = False,
        recompose: bool = False,
    ) -> Self:
        if not regions and repaint and not layout and not recompose and self.is_mounted:
            # Static.update(layout=False) otherwise dirties via Widget.size,
            # which resolves the full compositor map. Explicit regions use
            # only cached geometry; include borders/padding in this full repaint.
            regions = (self.outer_size.region - self.content_offset,)
        return super().refresh(*regions, repaint=repaint, layout=layout, recompose=recompose)


class BuddyPanel(Widget):
    """The buddy's corner of the sidebar: its animated portrait, how it is doing, and a click to pet it."""

    DEFAULT_CSS = """
    BuddyPanel {
        width: 100%;
        height: 100%;
        /* Padding lies outside the scrollbar, so the right side has none: the one-cell bar
           sits on the panel's edge, and its always-reserved column mirrors the left padding. */
        padding: 0 0 0 1;
        /* The portrait and its details sit in the middle of a tall sidebar; once they are
           taller than the panel, Textual pins them to the top and the panel scrolls. */
        align-vertical: middle;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        scrollbar-gutter: stable;
    }
    BuddyPanel #buddy-sprite {
        /* compose() fixes height to PORTRAIT_HEIGHT; animation only repaints. */
        width: 100%;
        content-align: center top;
        margin: 0 0 1 0;
    }
    BuddyPanel #buddy-level {
        height: auto;
        width: 100%;
        color: $accent;
        text-style: bold;
        text-align: center;
    }
    BuddyPanel #buddy-info {
        height: auto;
        width: 100%;
        color: $text-muted;
        margin: 1 0;
    }
    BuddyPanel #buddy-status {
        height: auto;
        width: 100%;
        color: $accent;
        text-align: center;
        margin: 1 0;
    }
    BuddyPanel .buddy-empty {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
    }
    """

    buddy: reactive[Buddy | None] = reactive(None)

    def __init__(self, *, locale_controller: LocaleController | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._locale_controller = locale_controller
        self.is_petting = False
        self._tick_count = 0
        self._composited = False
        self._timer: Timer | None = None
        self._shiny_timer: Timer | None = None
        self._pet_timer: Timer | None = None
        self._pet_at = 0.0
        self._replies = PetReplyFlow(self._toast, on_answered=self.reload)
        self._shown: dict[str, str] = {}
        self._status_message: MessageRef = _BUDDY_CLICK_TO_PET.bind()
        self._status_style: str | None = "green"

    def compose(self) -> ComposeResult:
        yield SidebarEmptyStateLabel(Text(self._render_message(_BUDDY_EMPTY.bind())), classes="buddy-empty")
        sprite = _BuddySprite("", id="buddy-sprite")
        sprite.styles.height = PORTRAIT_HEIGHT
        yield sprite
        yield Static("", id="buddy-level")
        yield Static("", id="buddy-status")
        yield Static("", id="buddy-info")

    def on_mount(self) -> None:
        """Start the animation timers and show the saved buddy."""
        self._timer = self.set_interval(_IDLE_TICK_SECONDS, self._on_tick)
        self._shiny_timer = self.set_interval(1 / SHINY_FPS, self._animate_shiny)
        self.set_interval(_RELOAD_SECONDS, self._reload_if_shown)
        self.reload()
        # Textual marks a widget mounted only after this handler, so the watcher has drawn nothing yet.
        self._redraw()
        self._sync_timers()

    async def on_unmount(self) -> None:
        await self._replies.shutdown()

    def reload(self) -> None:
        """Show the buddy as it is saved now. An unchanged buddy redraws nothing."""
        self.buddy = current_buddy()

    def on_show(self) -> None:
        self._composited = True
        self.reload()
        self._sync_timers()

    def on_hide(self) -> None:
        self._composited = False
        self._sync_timers()

    def _sync_timers(self) -> None:
        """Tick only while there is a buddy to animate on a composited panel.

        A petting burst keeps the idle tick running until it has ended it, and the
        badge timer runs only for a shiny buddy. The reload timer stays: it is what
        notices a buddy hatched or credited by another instance.
        """
        if self._timer is None or self._shiny_timer is None:
            return
        buddy = self.buddy
        animating = buddy is not None and (self._composited or self.is_petting)
        _run_timer(self._timer, animating)
        _run_timer(self._shiny_timer, animating and buddy is not None and buddy.shiny)

    def _reload_if_shown(self) -> None:
        if is_widget_shown_on_active_screen(self):
            self.reload()

    def refresh_localization(self) -> None:
        """Retranslate the visible empty, level, and interaction status chrome."""
        if self.is_mounted:
            self._show(".buddy-empty", Text(self._render_message(_BUDDY_EMPTY.bind())))
            self._redraw()

    def watch_buddy(self, _buddy: Buddy | None) -> None:
        if self.is_mounted:
            self._redraw()
            self._sync_timers()

    def _redraw(self) -> None:
        has_buddy = self.buddy is not None
        self.query_one(".buddy-empty", Static).display = not has_buddy
        for selector in _CONTENT_IDS:
            self.query_one(selector, Static).display = has_buddy
        if has_buddy:
            self._update_info()
            self._update_status(self._status_message, style=self._status_style)
            self._render_sprite()

    def _show(self, selector: str, text: Text) -> None:
        """Put *text* on a label, unless it is there already: an update lays the whole screen out again."""
        if self._shown.get(selector) != text.markup:
            self._shown[selector] = text.markup
            self.query_one(selector, Static).update(text)

    def _render_message(self, reference: MessageRef) -> str:
        controller = self._locale_controller
        if controller is None:
            return format_message(reference)
        return render_str(controller.localizer, reference)

    def _toast(self, message: MessageRef | str, timeout: float) -> None:
        self.screen.notify(
            message if isinstance(message, str) else self._render_message(message),
            title=self._render_message(BUDDY_TITLE.bind()),
            timeout=timeout,
            markup=False,
        )

    def _on_tick(self) -> None:
        """Advance the idle animation, and end a petting burst that has run its course."""
        self._tick_count += 1
        if self.is_petting and monotonic() - self._pet_at >= _PETTING_SECONDS:
            self.is_petting = False
            if self._pet_timer is not None:
                self._pet_timer.stop()
                self._pet_timer = None
            self._update_status(_BUDDY_CLICK_TO_PET.bind(), style="green")
            self._sync_timers()
        self._render_sprite()

    def _animate_shiny(self) -> None:
        """Refresh the badge independently of the slower idle body animation."""
        if self.buddy is not None and self.buddy.shiny and not self.is_petting:
            self._render_sprite()

    def _render_sprite(self) -> None:
        """Render the current sprite frame."""
        buddy = self.buddy
        if buddy is None:
            return
        sprite = self.query_one("#buddy-sprite", Static)
        if not is_widget_shown_on_active_screen(sprite):
            # Skip hidden, offscreen, or covered sprites; the next tick
            # resumes rendering when this screen is active again. Do not
            # use is_on_screen here — it forces a full compositor arrange.
            return

        now = monotonic()
        if self.is_petting:
            frame, blink = get_pet_frame(int((now - self._pet_at) * _PETTING_FPS)), False
        else:
            frame, blink = get_idle_frame(buddy.species, self._tick_count)

        # Keep the corners and nameplate anchored while the body animates.
        _base_background, background = sprite.background_colors
        # content_size resolves region through the compositor and can arrange
        # the entire transcript. outer_size is the latest cached layout size.
        width = max(0, sprite.outer_size.width - sprite.styles.gutter.width)
        lines = render_portrait(
            buddy.appearance,
            buddy.display_name,
            frame,
            blink,
            width=width or PORTRAIT_WIDTH,
            effect_tick=int(now * SHINY_FPS),
            bg_rgb=None if self.app.current_theme.ansi else background.rgb,
        )
        # The portrait height is fixed at compose time (width 100%); frames only
        # repaint. layout=True here would force a full-screen arrange per
        # animation tick, which is O(all transcript widgets) in long chats.
        sprite.update(Text("\n").join(lines), layout=False)

    def _update_info(self) -> None:
        buddy = self.buddy
        if buddy is None:
            return
        # The stylesheet makes this label bold and accented; only the XP line opts out.
        level = Text(self._render_message(_BUDDY_LEVEL.bind(level=buddy.level)))
        if not buddy.progress.maxed:
            xp = _BUDDY_PROGRESS.bind(xp=buddy.progress.xp_into_level, needed=buddy.progress.xp_for_level)
            level.append("\n" + self._render_message(xp), style="not bold dim")
        self._show("#buddy-level", level)

        info = Text(self._render_message(_BUDDY_SPECIES.bind(species=buddy.species.value)))
        info.append(
            "\n" + self._render_message(_BUDDY_RARITY.bind(rarity=buddy.rarity.value)),
            style=f"bold {RARITY_COLORS[buddy.rarity]}",
        )
        info.append(f"\n“{self._render_message(buddy.persona)}”", style="italic")
        if buddy.shiny:
            info.append("\n" + self._render_message(_BUDDY_SHINY.bind()), style="bold gold1")
        self._show("#buddy-info", info)

    def _update_status(self, message: MessageRef, *, style: str | None) -> None:
        self._status_message = message
        self._status_style = style
        text = Text(self._render_message(message), style=style or "")
        if self.buddy is not None and self.buddy.muted:
            text.append("\n\n" + self._render_message(_BUDDY_NOTIFICATIONS_MUTED.bind()), style="dim")
        self._show("#buddy-status", text)

    def pet(self) -> None:
        """Pet the buddy: play the burst, ask for an answer unless muted, and count it off the event loop."""
        buddy = current_buddy()
        if buddy is None:
            self.buddy = None
            return
        if not buddy.muted and not self._replies.start(buddy):
            return
        # Counting waits for the save file's lock, which another instance may hold for seconds. A worker
        # keeps that off the event loop, and the panel's unmount cancels the wait.
        self.run_worker(self._count_pet, group="buddy-count")
        self._pet_at = monotonic()
        self.is_petting = True
        if self._pet_timer is None:
            self._pet_timer = self.set_interval(1 / _PETTING_FPS, self._render_sprite)
        self._status_message, self._status_style = _BUDDY_PETTING.bind(), None
        self.buddy = buddy
        # The watcher has drawn the status already, unless the buddy came back unchanged.
        self._update_status(self._status_message, style=self._status_style)
        self._sync_timers()

    async def _count_pet(self) -> None:
        # A save file that cannot be written, or that another instance is wedged on, costs the count, not the pet.
        with contextlib.suppress(OSError):
            await asyncio.to_thread(record_pet)
            self.reload()

    def on_click(self, event: Click) -> None:
        """Pet the buddy on click."""
        if self.buddy is not None:
            self.pet()
            event.stop()
