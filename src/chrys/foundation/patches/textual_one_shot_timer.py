# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Patch: let a one-shot Textual timer fire late instead of never.

Problem
-------
Textual 8.2.7's ``Timer._run`` reads the clock once to start and again before its first sleep.
With ``skip`` on, the default, it takes any gap between the two reads that is longer than the
interval for missed ticks and counts past them. A one-shot timer (``set_timer`` builds one with
``repeat=0``) then counts past its only tick, and the loop ends without calling back. A garbage
collection or the OS descheduling the process between the two reads for longer than the delay is
enough. ``Tabs`` moves its underline from ``set_timer(0.02, ...)``, so a clicked tab can keep the
underline under the tab that was active before.

Solution
--------
Build one-shot timers with ``skip`` off. A single tick has no later tick to catch up to, so the
only difference is that a late timer still fires, as late as it is; repeating timers keep
skipping the ticks they missed. Wrapping the constructor duplicates no upstream body.
"""

from __future__ import annotations

import functools
from typing import Any

_RUNTIME_PATCH_MARKER = "_chrys_one_shot_timer_fires"


def apply_runtime_patch() -> None:
    """Patch ``Timer.__init__`` in the current process."""
    try:
        from textual.timer import Timer
    except ImportError:
        return

    original = Timer.__init__
    if getattr(original, _RUNTIME_PATCH_MARKER, False):
        return

    @functools.wraps(original)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("repeat") == 0:
            kwargs["skip"] = False
        original(self, *args, **kwargs)

    setattr(__init__, _RUNTIME_PATCH_MARKER, True)
    Timer.__init__ = __init__
