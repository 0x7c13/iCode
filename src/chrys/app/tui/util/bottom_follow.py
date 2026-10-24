# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bottom-follow for scroll views whose content grows at the end, as the chat follows."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    from textual.widget import Widget


class BottomFollow:
    """Keep a scroll view at its bottom while the user leaves it there.

    The view follows while it is at the bottom: content that lands below, or a
    viewport that shrinks, takes it to the new bottom. A scroll up stops the
    following, and the view then stays where the user put it until a scroll
    takes it back to the bottom, or a size change leaves it there.

    Every scroll the view is asked for (keys, wheel, scrollbar, code) starts in
    Textual's ``_scroll_to``, which the view runs inside :meth:`scroll`: a
    scroll decides as it starts, by where it heads. The view's other moves are
    judged by where it goes (:meth:`scrolled`): a selection dragged past its
    edge scrolls it a row at a time without asking for a scroll.

    Textual runs some scrolls after the next refresh (a ``VerticalScroll``'s
    ``scroll_to``), toward a row worked out as they were asked for: the view
    tells the follow of each (:meth:`defer`), which then leaves the view where
    it is until the scroll has run.

    The view wires its hooks: ``_scroll_to`` runs inside :meth:`scroll`,
    ``watch_scroll_y`` calls :meth:`scrolled`, and ``watch_virtual_size`` and
    ``on_resize`` call :meth:`resized`.
    """

    def __init__(self, view: Widget) -> None:
        self._view = view
        self._following = True
        self._destination = 0.0
        self._settle_scheduled = False
        self._pinning = False
        self._deferred = 0
        self._settle_after_deferred = False

    @property
    def following(self) -> bool:
        """Whether the view keeps to its bottom."""
        return self._following

    @property
    def pinning(self) -> bool:
        """Whether the scroll under way is this follow's own move to the bottom, not the user's."""
        return self._pinning

    @contextmanager
    def scroll(self, y: float | None) -> Generator[None]:
        """Run one scroll the view was asked for, and decide by where it heads.

        A scroll that heads to the bottom follows; one that heads down from
        where the view is keeps a following view following, so content that
        lands below it first is followed; any other stops the following. The
        scroll still animating lands first, as Textual would land it, so where
        the view is means where the user last sent it.
        """
        if y is None:
            yield
            return
        view = self._view
        view.app.animator.force_stop_animation(view, "scroll_y")
        start = view.scroll_y
        yield
        # Textual's own target: clamped to the bottom and rounded, as the scroll animates to it.
        self._destination = target = view.scroll_target_y
        self._following = target >= view.max_scroll_y or (self._following and target > start)

    def defer(self) -> None:
        """Note a scroll Textual runs after the next refresh; :meth:`deferred_ran` once it has run."""
        self._deferred += 1

    def deferred_ran(self) -> None:
        """A scroll noted by :meth:`defer` has run: settle the view, if a size change waited for it."""
        self._deferred -= 1
        if not self._deferred and self._settle_after_deferred:
            self._settle_after_deferred = False
            self.resized()

    def scrolled(self, old: float, new: float) -> None:
        """A move up a row stops the following; a move down that shows the bottom row follows.

        By the row Textual shows (it rounds), since a selection's auto-scroll
        moves part of a row a frame. A view above its bottom is one a size
        change just clamped there (the bottom moved up past it): :meth:`_settle`
        decides once that change has laid out, since a scroll may still be on
        its way up.
        """
        bottom = self._view.max_scroll_y
        if old > bottom:
            return
        if new > old:
            if round(new) >= bottom:
                self._following = True
        elif round(new) < round(old):
            self._following = False

    def resized(self) -> None:
        """Settle the view once the size change has laid out: :meth:`_settle`."""
        if not self._settle_scheduled and self._view.call_after_refresh(self._settle):
            self._settle_scheduled = True

    def _settle(self) -> None:
        """A view the size change left at its bottom follows; a following one goes to its new bottom.

        Not while a scroll animates toward a row above the bottom: the size
        change clamped it on its way up, and it goes on up. A following view
        waits for the scrolls :meth:`defer` noted, which were worked out from
        where it is.
        """
        self._settle_scheduled = False
        view = self._view
        if not view.is_attached:
            return
        bottom = view.max_scroll_y
        if view.scroll_y >= bottom:
            heading_up = view.app.animator.is_being_animated(view, "scroll_y") and self._destination < bottom
            if not heading_up:
                self._following = True
            return
        if not self._following:
            return
        if self._deferred:
            self._settle_after_deferred = True
            return
        self._pinning = True
        try:
            view.scroll_end(animate=False, immediate=True, x_axis=False)
        finally:
            self._pinning = False
