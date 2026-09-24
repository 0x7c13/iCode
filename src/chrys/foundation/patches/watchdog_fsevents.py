# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Let a stopped macOS watchdog emitter be collected.

watchdog 6.0.0's FSEvents extension takes a reference to the callback given to
``add_watch`` and never drops it: the stream context it creates has no release
callback. The callback is the emitter's bound ``events_callback``, so every
emitter ever started stays alive, and so does everything its thread object
holds: ``Thread.__init__`` keeps the ``sys.stderr`` of that moment, which under
a running Textual App is the App's print capture, and through it the App.
Hand the extension a forwarder that reaches the emitter through a weak
reference, so what it keeps is only that forwarder.
"""

from __future__ import annotations

import logging
import time
import weakref
from collections.abc import Callable
from typing import Any

from chrys.foundation.platform import get_platform

_RUNTIME_PATCH_WATCHDOG_VERSION = "6.0.0"
_RUNTIME_PATCH_MARKER = "_chrys_passes_a_weak_events_callback"
logger = logging.getLogger(__name__)


def apply_runtime_patch() -> None:
    """Patch the pinned macOS emitter without importing FSEvents on other OSes."""
    if not get_platform().is_macos:
        return
    try:
        from watchdog.version import VERSION_STRING
    except ImportError:
        return
    if VERSION_STRING != _RUNTIME_PATCH_WATCHDOG_VERSION:
        logger.warning("Skipping FSEvents callback patch for unsupported watchdog %s", VERSION_STRING)
        return

    from watchdog.observers import fsevents

    emitter_type = fsevents.FSEventsEmitter
    if getattr(emitter_type.run, _RUNTIME_PATCH_MARKER, False):
        return

    # watchdog 6.0.0's run(), passing the forwarder in place of the bound method.
    def run(self: Any) -> None:
        self.pathnames = [self.watch.path]
        self._start_time = time.monotonic()
        try:
            fsevents._fsevents.add_watch(self, self.watch, _weak_events_callback(self), self.pathnames)
            fsevents._fsevents.read_events(self)
        except Exception:
            fsevents.logger.exception("Unhandled exception in FSEventsEmitter")

    setattr(run, _RUNTIME_PATCH_MARKER, True)
    emitter_type.run = run


def _weak_events_callback(emitter: Any) -> Callable[[list[bytes], list[int], list[int], list[int]], None]:
    events_callback = weakref.WeakMethod(emitter.events_callback)

    def forward(paths: list[bytes], inodes: list[int], flags: list[int], ids: list[int]) -> None:
        callback = events_callback()
        if callback is not None:
            callback(paths, inodes, flags, ids)

    return forward
