# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Recognizing the outer terminal's own reports when they arrive as typed keys.

The terminal the app runs in answers focus and mouse tracking with escape sequences. When the
app's input parser gives up on one partway (Windows Terminal splits them under load), the rest
arrives as ordinary key presses: ``[``, ``I``. Forwarded to the embedded program they would be
typed at its prompt, so the widget holds the keys that follow an Escape for as long as they could
still be such a report, and drops them once they turn out to be one.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Final

_FOCUS_REPORTS: Final = ("\x1b[I", "\x1b[O")
# SGR and urxvt mouse reports are numeric; X10 is ``CSI M`` and three raw bytes.
_NUMERIC_MOUSE_REPORT = re.compile(r"\x1b\[<?-?\d+(?:;-?\d+){1,2}[mM]")
_NUMERIC_MOUSE_PREFIX = re.compile(r"\x1b\[<?[-\d;]*")
_X10_MOUSE_PREFIX: Final = "\x1b[M"
_X10_MOUSE_LENGTH: Final = len(_X10_MOUSE_PREFIX) + 3

MAX_HELD_LENGTH: Final = 32
"""A would-be report longer than this is not one; the keys are released rather than held forever."""


class ReportMatch(Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    NONE = "none"


def match_outer_report(sequence: str) -> ReportMatch:
    """Whether ``sequence``, which starts with ESC, is or may yet become a focus or mouse report."""
    if sequence in _FOCUS_REPORTS or _NUMERIC_MOUSE_REPORT.fullmatch(sequence):
        return ReportMatch.COMPLETE
    if sequence.startswith(_X10_MOUSE_PREFIX):
        return ReportMatch.COMPLETE if len(sequence) == _X10_MOUSE_LENGTH else ReportMatch.PARTIAL
    if len(sequence) > MAX_HELD_LENGTH:
        return ReportMatch.NONE
    could_grow = any(report.startswith(sequence) for report in (*_FOCUS_REPORTS, _X10_MOUSE_PREFIX))
    return ReportMatch.PARTIAL if could_grow or _NUMERIC_MOUSE_PREFIX.fullmatch(sequence) else ReportMatch.NONE
