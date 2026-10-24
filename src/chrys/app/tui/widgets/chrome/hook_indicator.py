# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""HookIndicator — the hook run the user waits on, at the right end of the footer row."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar

from rich.style import Style
from rich.text import Text
from textual.css.scalar import Scalar
from textual.message import Message
from textual.widget import Widget

from chrys.app.tui.i18n import render_str
from chrys.app.tui.util.spinner import SPINNER_FRAMES
from chrys.app.tui.util.visibility import is_widget_shown_on_active_screen, resync_compositor_regions
from chrys.app.tui.widgets.hook_activity import HOOK_INDICATOR_IDLE, hook_activity_text, indicator_activity
from chrys.app.tui.widgets.selection import NonSelectableStatic
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from textual.events import Click, Enter, Leave, MouseMove
    from textual.timer import Timer

    from chrys.app.tui.i18n import LocaleController
    from chrys.foundation.i18n import MessageRef
    from chrys.foundation.models.hook_executions import HookExecutionView

HOOK_INDICATOR_WIDTH = 30
"""The most cells the indicator takes; a longer hook id is cut in the middle."""
# The label's own padding cell on each side, the glyph and the space after it.
_CHROME_CELLS = 4
# The spinner's full dot grid: at rest when idle, and the still glyph of a
# run when animations are off, so both states draw from one glyph family.
_REST_GLYPH = "⠿"
_SPIN_INTERVAL_SECONDS = 0.1


class HookIndicator(NonSelectableStatic):
    """``⠿ Hooks`` when no waited-on hook is in flight; else a spinner and the latest started one's id.

    The glyph cell holds the resting dot grid or the spinner frame, right before
    the text. A long id keeps its head and tail. The box is exactly as wide as
    its label, padding included, so the footer keys keep every other cell of
    the row: hover highlights, and clicks hit, the whole box. Its width is
    written as a raw rule whenever the label's width changes, and only the
    footer row is remapped; a label of the same width (each spinner frame)
    repaints without either. A run's id is cut to the row on every frame, so
    the next one fits it to a resized terminal.
    """

    COMPONENT_CLASSES: ClassVar[set[str]] = {
        "hook-indicator--idle",
        "hook-indicator--running",
        "hook-indicator--hover",
    }

    DEFAULT_CSS = """
    HookIndicator {
        /* Until the first paint pins it to the label. */
        width: auto;
        /* On a terminal narrower than the label it takes the whole row. */
        max-width: 100%;
        height: 1;
        /* Transparent on the footer row's background, as a footer key is. */
        background: $footer-item-background;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    HookIndicator > .hook-indicator--idle {
        color: $text-muted;
    }
    HookIndicator > .hook-indicator--running {
        color: $accent;
    }
    /* The label's background on hover, as the app styles a footer key (chrys.tcss). */
    HookIndicator > .hook-indicator--hover {
        background: $control-background;
    }
    """

    class Clicked(Message):
        """The indicator was clicked: show the run it shows, or the hooks when it shows none."""

        def __init__(self, run_id: str) -> None:
            super().__init__()
            self.run_id = run_id
            """The run shown when clicked; empty when idle."""

    def __init__(self, *, locale_controller: LocaleController | None = None, **kwargs: Any) -> None:
        super().__init__("", markup=False, **kwargs)
        self._locale_controller = locale_controller
        self._executions: tuple[HookExecutionView, ...] = ()
        self._timer: Timer | None = None
        self._frame = 0
        self._shown: Text | None = None
        self._activity: HookExecutionView | None = None
        """The run the label shows; None when it reads ``Hooks``."""
        # The mouse's screen column while it is over the box, for the hover.
        self._mouse_x: int | None = None
        self._hot = False

    def on_mount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.register_surface(self)
        # The colors are resolved into the painted text, so a theme change repaints it.
        self.app.theme_changed_signal.subscribe(self, self._theme_changed)
        # Runs only while waited-on runs are in flight and the indicator shows
        # on the active screen: it turns the spinner and moves the 1 s threshold on.
        self._timer = self.set_interval(_SPIN_INTERVAL_SECONDS, self._tick, pause=True)
        self._paint()

    def on_unmount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.unregister_surface(self)
        self.app.theme_changed_signal.unsubscribe(self)
        if self._timer is not None:
            self._timer.stop()

    def on_show(self) -> None:
        self._sync_timer()

    def screen_resumed(self) -> None:
        """Catch up once an overlay over the screen closes.

        A snapshot that arrived under the overlay, or the overlay itself,
        paused the timer, and closing it resumes the screen without showing
        this widget again. The mouse may have left while the overlay took its
        events, with no Leave sent here: the hover is read again from where
        the mouse is.
        """
        if self.is_mounted:
            self._mouse_x = self._mouse_x_over_box()
            self._sync_timer()
            self._paint()

    def _mouse_x_over_box(self) -> int | None:
        position = self.app.mouse_position
        return position.x if self.region.contains_point(position) else None

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

    def on_enter(self, _event: Enter) -> None:
        # Also sent when a wider label grows under a mouse that stays put.
        self._mouse_x = self._mouse_x_over_box()
        self._paint()

    def on_mouse_move(self, event: MouseMove) -> None:
        self._mouse_x = event.screen_x
        self._paint()

    def on_leave(self, _event: Leave) -> None:
        # A Leave sent as an overlay covered the label can arrive after the
        # screen resumed and read the hover again: a mouse still resting on
        # the label keeps it.
        if self.screen.is_current and self.region.contains_point(self.app.mouse_position):
            return
        self._mouse_x = None
        self._paint()

    def on_click(self, event: Click) -> None:
        event.prevent_default()
        event.stop()
        # The run the user saw, not one picked again at the click.
        self.post_message(self.Clicked(self._activity.run_id if self._activity is not None else ""))

    def _sync_timer(self) -> None:
        if self._timer is None:
            return
        # Only waited-on runs can show. Hidden with the footer row (shell,
        # interactive mode) or covered by an overlay: nothing to turn until
        # Show or the screen's resume.
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
        # An overlay opened (or the row hid) since the timer started: nothing
        # sends this widget a Hide then, so the first tick stops the timer.
        if not is_widget_shown_on_active_screen(self):
            if self._timer is not None:
                self._timer.pause()
            return
        self._frame += 1
        self._paint()

    def _room(self) -> int:
        """The most cells the label may take: its cap, or the whole row on a narrower terminal."""
        row = self.parent
        width = row.size.width if isinstance(row, Widget) and row.size.width else self.app.size.width
        return min(HOOK_INDICATOR_WIDTH, width)

    def _paint(self) -> None:
        activity = indicator_activity(self._executions, now=time.monotonic())
        self._activity = activity
        if activity is None:
            component = "hook-indicator--idle"
            label = f" {_REST_GLYPH} {self._render_message(HOOK_INDICATOR_IDLE.bind())} "
        else:
            component = "hook-indicator--running"
            if self.app.animation_level == "none":
                glyph = _REST_GLYPH
            else:
                glyph = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
            name = hook_activity_text(activity, max_cells=max(0, self._room() - _CHROME_CELLS))
            label = f" {glyph} {name} "
        self._fit(Text(label).cell_len)
        # After the fit: a box that grew or shrank under a still mouse decides
        # the hover. The region is read only with the mouse over the box: a
        # read can rebuild the screen's whole map, and the spinner paints ten
        # times a second.
        mouse_x = self._mouse_x
        hot = False
        if mouse_x is not None:
            region = self.region
            hot = region.x <= mouse_x < region.right
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
            # Inline, so only this widget repaints; the setter has the screen read it again for the cursor.
            self.styles.pointer = "pointer" if hot else "default"

    def _fit(self, width: int) -> None:
        """Pin the box to *width* cells, remapping only the footer row when that changes it."""
        if not self.styles.set_rule("width", Scalar.from_number(width)):
            return
        row = self.parent
        if isinstance(row, Widget):
            row._clear_arrangement_cache()
            resync_compositor_regions(row)

    def _render_message(self, message: MessageRef) -> str:
        controller = self._locale_controller
        if controller is None:
            return format_message(message)
        return render_str(controller.localizer, message)
