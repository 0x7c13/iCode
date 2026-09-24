# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Patch: keep Textual selection extraction inside the widget's text.

Problem
-------
Textual 8.2.7's ``Selection.extract`` splits a widget's text with
``str.splitlines()``, which drops the empty line after a trailing newline. That
line is still rendered and still carries selection offsets, so a selection whose
first row is that blank line (a drag that starts on it, or one dragged up onto
it) indexes one past the last line and raises ``IndexError`` when the selection
is copied, crashing the app.

Solution
--------
Return an empty string when the selection starts below the last line of text,
where every rendered row is blank; hand every other selection to the upstream
method unchanged. Wrapping the method on the class reaches consumers that
imported ``Selection`` before startup patching and duplicates no upstream body.
"""

from __future__ import annotations

import functools
from typing import Any

_RUNTIME_PATCH_MARKER = "_chrys_selection_extract_bounds"


def apply_runtime_patch() -> None:
    """Patch ``Selection.extract`` in the current process."""
    try:
        from textual.selection import Selection
    except ImportError:
        return

    original = Selection.extract
    if getattr(original, _RUNTIME_PATCH_MARKER, False):
        return

    @functools.wraps(original)
    def extract(self: Any, text: str) -> str:
        if self.start is not None and self.start.y >= len(text.splitlines()):
            return ""
        return original(self, text)

    setattr(extract, _RUNTIME_PATCH_MARKER, True)
    Selection.extract = extract
