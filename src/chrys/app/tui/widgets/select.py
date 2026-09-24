# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Select widget that tolerates Textual's mount-time child-lookup races."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.css.query import NoMatches
from textual.widgets import Select as TextualSelect
from textual.widgets._select import NoSelection, SelectCurrent, SelectOverlay, SelectType
from textual.widgets.option_list import Option

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rich.console import RenderableType
    from textual.app import ComposeResult
    from textual.geometry import Region
    from textual.strip import Strip


class Select(TextualSelect[SelectType]):
    """Select variant that tolerates Textual's mount-time child-lookup races.

    Textual's ``Select._on_mount`` queries the composed ``SelectOverlay``
    unguarded, which crashes the whole app with ``NoMatches`` in two race
    flavors (both seen on CI under xdist load):

    * the children have not finished mounting when the ``Mount`` event
      dispatches (Windows CI) — recovered by a deferred retry once the
      overlay exists;
    * the children never mount at all because the widget is already being
      pruned: ``Widget.mount()`` silently no-ops while ``_pruning`` is set,
      but the already-queued ``Mount`` event still dispatches (Linux CI,
      app teardown racing a fresh panel remount) — the deferred retries
      then no-op harmlessly against their cap.

    Textual itself guards the same lookup in ``_watch_value`` and
    ``_watch_expanded`` but not in ``_on_mount``; re-check this class when
    upgrading Textual.
    """

    def __init__(self, *args: Any, separators_before: Iterable[SelectType] = (), **kwargs: Any) -> None:
        self._separators_before = frozenset(separators_before)
        super().__init__(*args, **kwargs)
        self._deferred_setup_options: bool = False
        self._setup_options_retries: int = 0
        self._deferred_visual_sync: bool = False
        self._visual_sync_retries: int = 0

    def set_options(
        self,
        options: Iterable[tuple[RenderableType, SelectType]],
        *,
        separators_before: Iterable[SelectType] = (),
    ) -> None:
        """Replace choices and dividers without adding selectable values.

        Omitting ``separators_before`` clears all existing dividers.
        """
        self._separators_before = frozenset(separators_before)
        super().set_options(options)

    def _populate_options(self) -> None:
        if not self._separators_before:
            super()._setup_options_renderables()
            return
        overlay = self.query_one(SelectOverlay)
        options: list[Option | None] = []
        for prompt, value in self._options:
            if options and value in self._separators_before:
                options.append(None)
            options.append(Option(Text(self.prompt, style="dim") if value is self.NULL else prompt))
        # OptionList dividers consume display rows, never option indices. This
        # preserves Select's value mapping, keyboard search and mouse messages.
        overlay.clear_options()
        overlay.add_options(options)

    def _setup_options_renderables(self) -> None:
        # Wraps Textual's Select._setup_options_renderables so that the SelectOverlay-
        # not-yet-mounted race during _on_mount is recovered via a deferred retry
        # instead of crashing the app's exception handler.  Leaves _on_mount and
        # _watch_value's existing flow intact; once the retry succeeds, _watch_value
        # re-runs to populate the visual.
        try:
            self._populate_options()
        except NoMatches:
            self._defer_setup_options()
            return
        self._setup_options_retries = 0

    def _defer_setup_options(self) -> None:
        if self._deferred_setup_options or self._setup_options_retries >= 3:
            return
        self._deferred_setup_options = True
        self._setup_options_retries += 1
        self.call_after_refresh(self._sync_deferred_setup_options)

    def _sync_deferred_setup_options(self) -> None:
        self._deferred_setup_options = False
        try:
            self._populate_options()
        except NoMatches:
            self._defer_setup_options()
            return
        self._setup_options_retries = 0
        # Re-fire the watcher so the visual catches up now that the overlay exists.
        self._watch_value(self._value)

    def _watch_value(self, value: SelectType | NoSelection) -> None:
        """Update current value, deferring visual sync if children are still mounting."""
        # Mirrors Textual's Select._watch_value, with only the current-label update deferred on mount races.
        self._value = value
        try:
            select_current = self.query_one(SelectCurrent)
        except NoMatches:
            return

        if value == self.NULL:
            if self._try_update_current(select_current, self.NULL):
                self.post_message(self.Changed(self, value))
            return

        for index, (prompt, option_value) in enumerate(self._options):
            if option_value != value:
                continue
            try:
                select_overlay = self.query_one(SelectOverlay)
            except NoMatches:
                self._defer_visual_sync()
                return
            select_overlay.highlighted = index
            if self._try_update_current(select_current, prompt):
                self.post_message(self.Changed(self, value))
            return

        self.post_message(self.Changed(self, value))

    def _try_update_current(self, select_current: SelectCurrent, label: RenderableType | NoSelection) -> bool:
        try:
            select_current.update(label)
        except NoMatches:
            self._defer_visual_sync()
            return False
        self._visual_sync_retries = 0
        return True

    def _defer_visual_sync(self) -> None:
        if self._deferred_visual_sync or self._visual_sync_retries >= 3:
            return
        self._deferred_visual_sync = True
        self._visual_sync_retries += 1
        self.call_after_refresh(self._sync_deferred_visual)

    def _sync_deferred_visual(self) -> None:
        self._deferred_visual_sync = False
        self._watch_value(self._value)


class _LazySelectOverlay(SelectOverlay):
    """Keep options ready without querying screen geometry while collapsed."""

    def render_lines(self, crop: Region) -> list[Strip]:
        # The compositor keeps the open overlay in its map until the next
        # layout, so a pointer move in between still hit-tests this widget. A
        # closed overlay has no hover style to report, and rendering one would
        # read its geometry while collapsed.
        return super().render_lines(crop) if self.display else []

    def _update_lines(self) -> None:
        # OptionList calls this on mount, option changes and highlight changes.
        # Even an empty region lookup can synchronously lay out the whole Screen.
        # Its normal Show/render paths populate the cache once displayed.
        if self.display:
            super()._update_lines()

    def scroll_to_highlight(self, top: bool = False) -> None:
        # A previously opened overlay may still have cached line positions.
        # Do not use them to look up geometry while the menu is closed again.
        if self.display:
            super().scroll_to_highlight(top)


class LazySelect(Select[SelectType]):
    """Select for dynamically mounted panels that defer hidden menu layout.

    When upgrading Textual, re-check ``compose`` and the overlay's
    ``render_lines`` / ``_update_lines`` / ``scroll_to_highlight`` overrides against its internal
    Select composition and OptionList display/cache lifecycle.
    """

    def compose(self) -> ComposeResult:
        yield SelectCurrent(self.prompt)
        yield _LazySelectOverlay(type_to_search=self._type_to_search).data_bind(compact=TextualSelect.compact)
