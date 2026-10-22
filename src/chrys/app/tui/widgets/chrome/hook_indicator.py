# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""HookIndicator — the hook run the user waits on, at the right end of the footer row."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar

from rich.style import Style
from rich.text import Text
from textual.message import Message

from chrys.app.tui.i18n import render_str
from chrys.app.tui.util.spinner import SPINNER_FRAMES
from chrys.app.tui.util.visibility import is_widget_shown_on_active_screen
from chrys.app.tui.widgets.hook_activity import HOOK_INDICATOR_IDLE, hook_activity_text, indicator_activity
from chrys.app.tui.widgets.selection import NonSelectableStatic
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from textual.events import Click, Leave, MouseMove
    from textual.geometry import Offset
    from textual.timer import Timer

    from chrys.app.tui.i18n import LocaleController
    from chrys.foundation.i18n import MessageRef
    from chrys.foundation.models.hook_executions import HookExecutionView

HOOK_INDICATOR_WIDTH = 30
"""The longest the indicator gets. Fixed, so a changing text repaints the cell
range without a layout pass; the text sits at its right end, in the screen's corner."""
# The label's own padding cell on each side, the glyph and the space after it.
_CHROME_CELLS = 4
# The spinner's full dot grid: at rest when idle, and the still glyph of a
# run when animations are off, so both states draw from one glyph family.
_REST_GLYPH = "⠿"
_SPIN_INTERVAL_SECONDS = 0.1


class HookIndicator(NonSelectableStatic):
    """``⠿ Hooks`` when no waited-on hook is in flight; else a spinner and the latest started one's id.

    The glyph cell holds the resting dot grid or the spinner frame, right before
    the text. A long id keeps its head and tail. The label carries its own
    padding, so hover highlights, and clicks hit, only the label's cells: the
    box itself keeps its width, and a changing label never needs a layout pass.
    """

    COMPONENT_CLASSES: ClassVar[set[str]] = {
        "hook-indicator--idle",
        "hook-indicator--running",
        "hook-indicator--hover",
    }

    DEFAULT_CSS = f"""
    HookIndicator {{
        width: {HOOK_INDICATOR_WIDTH};
        /* On a terminal narrower than the box it takes the whole row. */
        max-width: 100%;
        height: 1;
        text-align: right;
        /* Transparent on the footer row's background, as a footer key is. */
        background: $footer-item-background;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }}
    HookIndicator > .hook-indicator--idle {{
        color: $text-muted;
    }}
    HookIndicator > .hook-indicator--running {{
        color: $accent;
    }}
    /* The label's background on hover, as the app styles a footer key (chrys.tcss). */
    HookIndicator > .hook-indicator--hover {{
        background: $control-background;
    }}
    """

    class Clicked(Message):
        """The indicator was clicked: show the session's hooks."""

    def __init__(self, *, locale_controller: LocaleController | None = None, **kwargs: Any) -> None:
        super().__init__("", markup=False, **kwargs)
        self._locale_controller = locale_controller
        self._executions: tuple[HookExecutionView, ...] = ()
        self._timer: Timer | None = None
        self._frame = 0
        self._shown: Text | None = None
        # Where the mouse is over the box, for hovering the label only.
        self._mouse_x: int | None = None
        self._hot = False

    def on_mount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.register_surface(self)
        # The colors are resolved into the painted text, so a theme change repaints it.
        self.app.theme_changed_signal.subscribe(self, self._theme_changed)
        # Runs only while waited-on runs are in flight and the indicator shows:
        # it turns the spinner and moves the 1 s threshold on.
        self._timer = self.set_interval(_SPIN_INTERVAL_SECONDS, self._tick, pause=True)
        self._paint()

    def on_unmount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.unregister_surface(self)
        self.app.theme_changed_signal.unsubscribe(self)
        if self._timer is not None:
            self._timer.stop()

    def on_resize(self) -> None:
        # Narrower than the box only on a narrow terminal: the id is cut to fit.
        self._shown = None
        self._paint()

    def on_show(self) -> None:
        self._sync_timer()

    def screen_resumed(self) -> None:
        """Catch up once an overlay over the screen closes.

        A snapshot that arrived under the overlay paused the timer, and closing
        it resumes the screen without showing this widget again. The mouse may
        have left while the overlay took its events, with no Leave sent here:
        the hover is read again from where the mouse is.
        """
        if self.is_mounted:
            self._mouse_x = self._mouse_x_at(self.app.mouse_position)
            self._sync_timer()
            self._paint()

    def _mouse_x_at(self, position: Offset) -> int | None:
        region = self.region
        return position.x - region.x if region.contains_point(position) else None

    def on_hide(self) -> None:
        if self._timer is not None:
            self._timer.pause()
        # Hidden widgets get no Leave: drop the hover now, not on the next Show.
        self._mouse_x = None
        self._paint()

    def set_executions(self, executions: tuple[HookExecutionView, ...]) -> None:
        """Show the latest of *executions* someone waits on, or ``Hooks``."""
        self._executions = executions
        if self.is_mounted:
            self._sync_timer()
            self._paint()

    def refresh_localization(self) -> None:
        if not self.is_mounted:
            return
        self._shown = None
        self._paint()

    def on_mouse_move(self, event: MouseMove) -> None:
        self._mouse_x = event.x
        self._paint()

    def on_leave(self, _event: Leave) -> None:
        self._mouse_x = None
        self._paint()

    def on_click(self, event: Click) -> None:
        if not self._over_label(event.x):
            return
        event.prevent_default()
        event.stop()
        self.post_message(self.Clicked())

    def _over_label(self, x: int | None) -> bool:
        if x is None or self._shown is None:
            return False
        return x >= self.size.width - self._shown.cell_len

    def _sync_timer(self) -> None:
        if self._timer is None:
            return
        # Only waited-on runs can show. Hidden with the footer row (shell,
        # interactive mode): nothing to paint until Show.
        waited_on = any(view.waited_at != "none" for view in self._executions)
        if waited_on and is_widget_shown_on_active_screen(self):
            self._timer.resume()
        else:
            self._timer.pause()

    def _theme_changed(self, _theme: object) -> None:
        if self.is_mounted:
            self._shown = None
            self._paint()

    def _tick(self) -> None:
        self._frame += 1
        self._paint()

    def _paint(self) -> None:
        activity = indicator_activity(self._executions, now=time.monotonic())
        if activity is None:
            component = "hook-indicator--idle"
            label = f" {_REST_GLYPH} {self._render_message(HOOK_INDICATOR_IDLE.bind())} "
        else:
            component = "hook-indicator--running"
            if self.app.animation_level == "none":
                glyph = _REST_GLYPH
            else:
                glyph = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
            name = hook_activity_text(activity, max_cells=max(0, self.content_size.width - _CHROME_CELLS))
            label = f" {glyph} {name} "
        width = Text(label).cell_len
        # Hover follows the label as it changes under a still mouse.
        hot = self._mouse_x is not None and self._mouse_x >= self.size.width - width
        color = self.get_component_rich_style(component, partial=False).color
        bgcolor = self.get_component_rich_style("hook-indicator--hover", partial=False).bgcolor if hot else None
        text = Text(label, style=Style(color=color, bgcolor=bgcolor))
        # ``Text`` equality ignores the base style, which carries the colors.
        shown = self._shown
        if shown is None or text.plain != shown.plain or text.style != shown.style:
            self._shown = text
            self.update(text, layout=False)
        if hot != self._hot:
            self._hot = hot
            # Inline, so only this widget repaints; the screen reads it for the cursor.
            self.styles.pointer = "pointer" if hot else "default"
            if self.screen.is_current:
                self.screen.update_pointer_shape()

    def _render_message(self, message: MessageRef) -> str:
        controller = self._locale_controller
        if controller is None:
            return format_message(message)
        return render_str(controller.localizer, message)
