# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Patch: hand the callbacks a removed screen still holds to the screen that replaced it.

Problem
-------
Textual 8.2.7 runs ``call_after_refresh`` callbacks on whichever screen is on top when the
caller handles its ``InvokeLater``, not on the caller's own screen, and that screen runs them only
when it goes idle with nothing left to lay out. A popped screen is sent ``ScreenSuspend`` and then
``Prune``; unless its queue runs empty between the two it closes without idling again, and whatever
it held is dropped. A widget under a short-lived modal loses work it scheduled while the modal was
up: a tab bar clicked just before a loading dialog opens never moves its underline to the new tab.

Solution
--------
Once ``App._replace_screen`` has removed a screen, move the callbacks it never ran to the front
of the current screen's queue, as they were scheduled before anything that screen already holds.
Callbacks scheduled from inside the removed screen went with it and are dropped as before, and a
screen that stays installed or stacked keeps its own. Wrapping the method duplicates no upstream
body.
"""

from __future__ import annotations

import functools
from typing import Any

_RUNTIME_PATCH_MARKER = "_chrys_removed_screen_callbacks"


def apply_runtime_patch() -> None:
    """Patch ``App._replace_screen`` in the current process."""
    try:
        from textual.app import App
    except ImportError:
        return

    original = App._replace_screen
    if getattr(original, _RUNTIME_PATCH_MARKER, False):
        return

    @functools.wraps(original)
    async def _replace_screen(self: Any, screen: Any) -> Any:
        replaced = await original(self, screen)
        if screen._callbacks and not screen.is_attached:
            stranded = [(callback, sender) for callback, sender in screen._callbacks if sender.is_attached]
            screen._callbacks.clear()
            if stranded and self._screen_stack:
                current = self.screen
                current._callbacks[:0] = stranded
                current.check_idle()
        return replaced

    setattr(_replace_screen, _RUNTIME_PATCH_MARKER, True)
    App._replace_screen = _replace_screen
