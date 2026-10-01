# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: keep Textual's terminal cursor anchored to focused text inputs.

Problem
-------
Textual uses ``App.cursor_position`` to place the hidden terminal cursor, which
terminals use as the anchor for OS IME candidate windows. Textual's ``Input``
and ``TextArea`` update that value when focus, selection, or scroll changes, but
not when layout moves the focused widget without a cursor event. The visible
Textual cursor can then be correct while the terminal cursor remains at an old
screen coordinate, making IME candidate windows appear away from the typed text
or jump during repeated layout changes.

Solution
--------
Before each terminal update, refresh ``App.cursor_position`` from the focused
widget's ``cursor_screen_offset`` when that focused widget exposes one. This
runs after layout has resolved and covers both Textual ``Input`` and
``TextArea`` without changing their editing behavior.
"""

from __future__ import annotations

from typing import Any

from chrys.foundation.patches.patcher import FilePatch, register

_OLD_BEGIN_UPDATE = """\
    def _begin_update(self) -> None:
        if self._sync_available and self._driver is not None:
            self._driver.write(SYNC_START)"""

_NEW_BEGIN_UPDATE = """\
    def _sync_focused_widget_cursor_position(self) -> None:
        \"\"\"Keep OS IME candidate windows anchored to the focused text cursor.\"\"\"
        focused = self.focused
        cursor_screen_offset = getattr(focused, "cursor_screen_offset", None) if focused is not None else None
        if isinstance(cursor_screen_offset, Offset):
            self.cursor_position = cursor_screen_offset

    def _begin_update(self) -> None:
        self._sync_focused_widget_cursor_position()
        if self._sync_available and self._driver is not None:
            self._driver.write(SYNC_START)"""

_RUNTIME_PATCH_MARKER = "_chrys_ime_cursor_anchor"


def _sync_focused_widget_cursor_position(app: Any) -> None:
    from textual.geometry import Offset

    focused = app.focused
    cursor_screen_offset = getattr(focused, "cursor_screen_offset", None) if focused is not None else None
    if isinstance(cursor_screen_offset, Offset):
        app.cursor_position = cursor_screen_offset


def apply_runtime_patch() -> None:
    """Patch ``App._begin_update`` in the current process."""
    try:
        from textual.app import App
    except ImportError:
        return

    if getattr(App._begin_update, _RUNTIME_PATCH_MARKER, False):
        return
    if getattr(App, "_sync_focused_widget_cursor_position", None) is not None:
        return

    original_begin_update = App._begin_update

    def _begin_update(self: Any) -> None:
        _sync_focused_widget_cursor_position(self)
        original_begin_update(self)

    setattr(_begin_update, _RUNTIME_PATCH_MARKER, True)
    App._sync_focused_widget_cursor_position = _sync_focused_widget_cursor_position
    App._begin_update = _begin_update


register(
    FilePatch(
        package="textual",
        module_file="app.py",
        old_fragment=_OLD_BEGIN_UPDATE,
        new_fragment=_NEW_BEGIN_UPDATE,
        description="Keep IME cursor anchor synced after focused text input layout shifts",
    )
)
