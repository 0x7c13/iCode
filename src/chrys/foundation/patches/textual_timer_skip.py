# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: a late Textual timer tick skips only the ticks that have passed.

Problem
-------
When a repeating timer wakes after its next tick is already due, Textual 8.2.7's ``Timer._run``
skips ahead with ``count = int((now - start) / interval + 1)``, which counts one boundary too
many: the next tick lands an extra interval later than the first one still ahead. Every
screen's refresh timer (``Screen._update_timer``, 1/60 s) hits this after each idle pause and
after every frame that overruns its interval, so a busy streaming transcript paints at half the
rate its frames would allow.

Solution
--------
Skip to the last tick that has passed, so the next tick is the first one still ahead
(``count + 1`` keeps the loop advancing if float rounding lands on a passed tick). One-shot
timers keep ``textual_one_shot_timer``'s ``skip=False``.
"""

from __future__ import annotations

import logging

from chrys.foundation.patches.patcher import FilePatch, register
from chrys.foundation.patches.staged_members import install_patched_members

_RUNTIME_PATCH_MARKER = "_chrys_timer_skips_passed_ticks"
_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
logger = logging.getLogger(__name__)

_OLD = """\
            if self._skip and next_timer < now:
                count = int((now - start) / _interval + 1)
                continue"""

_NEW = """\
            if self._skip and next_timer < now:
                # Skip the ticks whose time has passed; the next tick is the first one still ahead.
                # (`count + 1` keeps the loop advancing if float rounding lands on a passed tick.)
                count = max(count + 1, int((now - start) / _interval))
                continue"""

_PATCH = FilePatch(
    package="textual",
    module_file="timer.py",
    old_fragment=_OLD,
    new_fragment=_NEW,
    description="Skip only the timer ticks that have passed",
)

register(_PATCH)


def apply_runtime_patch() -> None:
    """Install the patched ``Timer._run`` in the current process."""
    try:
        import textual
        import textual.timer as timer_mod
    except ImportError:
        return
    if textual.__version__ != _RUNTIME_PATCH_TEXTUAL_VERSION:
        logger.warning(
            "Skipping Textual timer runtime patch: loaded Textual is not the pinned %s that the patch targets.",
            _RUNTIME_PATCH_TEXTUAL_VERSION,
        )
        return
    install_patched_members(timer_mod, [_PATCH], {"Timer": ["_run"]}, marker=_RUNTIME_PATCH_MARKER, label="timer")
