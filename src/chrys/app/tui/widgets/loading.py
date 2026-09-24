# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared loading indicator widgets."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, cast

from rich.style import Style
from rich.text import Text
from textual.color import Color, Gradient
from textual.geometry import Region
from textual.strip import Strip
from textual.widgets import LoadingIndicator

from chrys.app.tui.i18n import render_text
from chrys.app.tui.util.visibility import is_widget_shown_on_active_screen
from chrys.foundation.i18n import msg
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from chrys.app.tui.i18n import LocaleController

_LOADING = msg("tui.loading.label", fallback="Loading...")


class _LocaleControllerApp(Protocol):
    @property
    def locale_controller(self) -> LocaleController: ...


class ChrysLoadingIndicator(LoadingIndicator):
    """LoadingIndicator with a brighter gradient floor for transparent themes.

    Its 16/s refresh timer ticks only while the indicator is composited, between
    Textual's Show and Hide events, and its owner wants it animated
    (``resume_animation``). The compositor maps neither ``display: none`` nor
    ``visibility: hidden`` widgets, and Chrys's layout-free visibility flips post
    the same Show/Hide as a reflow, so a spinner inside a hidden container, a
    collapsed tool group or an idle status bar costs no timer ticks at all; on
    Windows every tick used to be an executor-thread hop.
    """

    _composited = False
    """Whether the compositor maps this widget: set by Show, cleared by Hide."""
    _animating = True
    """The owner's intent: ``pause_animation`` parks the timer even while composited."""

    def pause_animation(self) -> None:
        """Park the refresh timer while the owner shows the indicator idle."""
        self._animating = False
        self._sync_timer()

    def resume_animation(self) -> None:
        """Let the refresh timer run again once the owner shows activity."""
        self._animating = True
        self._sync_timer()

    def on_show(self) -> None:
        self._composited = True
        self._sync_timer()

    def on_hide(self) -> None:
        self._composited = False
        self._sync_timer()

    def _sync_timer(self) -> None:
        timer = self._auto_refresh_timer
        if timer is None:
            return
        if self._animating and self._composited:
            timer.resume()
        else:
            timer.pause()

    def automatic_refresh(self) -> None:
        """Animate only on the active screen, without arranging the tree.

        ``DOMNode.automatic_refresh`` gates on ``is_on_screen``, which goes
        through ``Screen.find_widget`` and recomputes the compositor's full
        map whenever it was invalidated. With ``auto_refresh`` at 16/s that
        turns every invalidation (scroll ticks, any layout change) into an
        O(all mounted widgets) arrange pass — even while this indicator is
        hidden inside a collapsed status section.
        """
        if is_widget_shown_on_active_screen(self):
            self.refresh()
        if not (self._animating and self._composited):
            # No Show is coming for a spinner mounted inside a hidden container,
            # and a timer may start after the owner's pause: the first tick parks it.
            # Painting is gated above on the compositor's cut, not on Show having
            # been processed, so a freshly shown spinner still paints this tick.
            self._sync_timer()

    def render_lines(self, crop: Region) -> list[Strip]:
        if not self.is_attached:
            return [Strip.blank(crop.width) for _ in crop.line_range]
        return super().render_lines(crop)

    def render(self) -> Text:
        if not self.is_attached:
            return Text(format_message(_LOADING.bind()))
        if self.app.animation_level == "none":
            try:
                localizer = cast("_LocaleControllerApp", self.app).locale_controller.localizer
            except AttributeError, RuntimeError:
                return Text(format_message(_LOADING.bind()))
            return render_text(localizer, _LOADING.bind())
        elapsed = time.time() - self._start_time
        speed = 0.8
        dot = "\u25cf"
        _, _, _bg, color = self.colors
        # Dim end: desaturate towards grey for a muted appearance.
        dim = color.blend(Color(50, 50, 50), 0.75)
        gradient = Gradient(
            (0.0, dim),
            (0.7, color),
            (1.0, color.lighten(0.1)),
        )
        blends = [(elapsed * speed - i / 8) % 1 for i in range(5)]
        dots = [(f"{dot} ", Style.from_color(gradient.get_color((1 - b) ** 2).rich_color)) for b in blends]
        indicator = Text.assemble(*dots)
        indicator.rstrip()
        return indicator
