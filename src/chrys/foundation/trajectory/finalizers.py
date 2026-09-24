# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pending operation settlement owned by one recorder, across producer replacements."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class RecordingFinalizers:
    """Retain unfinished operations until their result arrives or the recorder closes.

    Registration may run in the writer's sequence callback on another thread.
    Close runs callbacks outside the lock so they can queue their final events.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._callbacks: set[Callable[[], None]] = set()
        self._closed = False

    def add(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Trajectory recorder closed before operation start")
            self._callbacks.add(callback)

    def discard(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._callbacks.discard(callback)

    def close(self) -> None:
        """Settle each remaining operation once, before the writer freezes."""
        with self._lock:
            self._closed = True
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.warning("Trajectory operation settlement failed", exc_info=True)
