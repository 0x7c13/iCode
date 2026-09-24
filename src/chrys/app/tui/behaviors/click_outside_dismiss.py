# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Mixin that dismisses a ModalScreen when the user clicks the backdrop."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual import events
from textual.await_complete import AwaitComplete
from textual.errors import NoWidget

if TYPE_CHECKING:
    from textual.screen import Screen

    _ClickOutsideDismissBase = Screen[Any]
else:
    _ClickOutsideDismissBase = object


class ClickOutsideDismissMixin(_ClickOutsideDismissBase):
    """Mixin for ``ModalScreen`` that dismisses on backdrop click.

    When a left-click lands on the screen itself (the dimmed overlay area
    outside any child widget), ``dismiss()`` is called.  This mirrors the
    common UX convention where clicking outside a dialog closes it.

    Put this mixin before ``ModalScreen`` in the class bases::

        class MyScreen(ClickOutsideDismissMixin, ModalScreen[None]): ...

    Compatible with ``RightClickScreenCopyMixin`` — that mixin intercepts
    right-click via ``_forward_event``; this one uses ``on_click`` for
    left-click only.

    Subclasses can customise backdrop-dismiss via three hooks:

    * ``_allow_click_outside_dismiss()`` — return ``False`` to suppress
      dismiss in certain states.
    * ``_default_dismiss_result()`` — a *pure* getter for the value passed
      to ``dismiss()`` (e.g. ``ConfirmDialog`` returns ``False``).
    * ``_dismiss_clicked_outside()`` — override the whole dismiss action
      when a backdrop cancel needs side effects (e.g. ``ThemesScreen``
      restores its previewed theme); reuse the screen's existing cancel
      path here rather than mutating state inside the value getter.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._chrys_dismiss_await_complete: AwaitComplete | None = None
        super().__init__(*args, **kwargs)

    def dismiss(self, result: object | None = None) -> AwaitComplete:
        """Dismiss once.

        Textual completes a modal result future inside ``dismiss()`` and
        raises if a later event tries to complete that same modal again.
        Most modal close paths are fed by queued UI messages, so make the
        operation idempotent for every modal using this mixin.
        """
        existing = self._chrys_dismiss_await_complete
        if existing is not None:
            return existing
        await_complete = super().dismiss(result)  # type: ignore[misc]
        self._chrys_dismiss_await_complete = await_complete
        return await_complete

    def on_click(self, event: events.Click) -> None:
        # Only left-click (button 1).
        if event.button != 1:
            return
        try:
            widget, _ = self.get_widget_at(event.screen_x, event.screen_y)  # type: ignore[attr-defined]
        except NoWidget:
            return
        # If the topmost widget under the cursor is the screen itself,
        # the click landed on the backdrop — dismiss.
        if widget is self and self._allow_click_outside_dismiss():
            self._dismiss_clicked_outside()

    def _allow_click_outside_dismiss(self) -> bool:
        """Hook for subclasses to conditionally suppress dismiss."""
        return True

    def _dismiss_clicked_outside(self) -> None:
        """Perform the dismissal. Override for cancel paths with side effects."""
        self.dismiss(self._default_dismiss_result())  # type: ignore[attr-defined]

    def _default_dismiss_result(self):
        """Pure getter for the dismiss value; override to customise."""
        return
