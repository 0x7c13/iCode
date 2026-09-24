# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Release each Windows watchdog directory handle only once.

watchdog 6.0.0 leaves the closed handle on its emitter. Both the observer and
the emitter itself can stop it, so a later stop may close an unrelated handle
that Windows has allocated with the same integer value. See watchdog #1132.
Detach ownership under a separate lock before calling the native close: the
emitter's own lock is already held when a deleted directory stops itself.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from chrys.foundation.platform import get_platform

_RUNTIME_PATCH_WATCHDOG_VERSION = "6.0.0"
_RUNTIME_PATCH_MARKER = "_chrys_closes_directory_handle_once"
logger = logging.getLogger(__name__)


def apply_runtime_patch() -> None:
    """Guard the pinned Windows emitter without importing Win32 on other OSes."""
    if not get_platform().is_windows:
        return
    try:
        from watchdog.version import VERSION_STRING
    except ImportError:
        return
    if VERSION_STRING != _RUNTIME_PATCH_WATCHDOG_VERSION:
        logger.warning("Skipping Windows directory handle patch for unsupported watchdog %s", VERSION_STRING)
        return

    from watchdog.observers import read_directory_changes

    emitter_type = read_directory_changes.WindowsApiEmitter
    if getattr(emitter_type.on_thread_stop, _RUNTIME_PATCH_MARKER, False):
        return
    ownership_lock = Lock()

    def on_thread_stop(self: Any) -> None:
        with ownership_lock:
            handle = self._whandle
            self._whandle = None
        if handle:
            read_directory_changes.close_directory_handle(handle)

    setattr(on_thread_stop, _RUNTIME_PATCH_MARKER, True)
    emitter_type.on_thread_stop = on_thread_stop
