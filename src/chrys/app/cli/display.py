# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Values from the user's files, shown on one line of command-line output."""

from __future__ import annotations

import os
from pathlib import Path

from chrys.foundation.i18n.formatting import sanitize_legacy_scalar
from chrys.foundation.platform.files import surrogate_safe_text


def one_line(text: str) -> str:
    """*text* as one display line: lone surrogates made safe, controls and Unicode line breaks replaced."""
    return sanitize_legacy_scalar(surrogate_safe_text(text)).replace("\u2028", "\ufffd").replace("\u2029", "\ufffd")


def counted(number: int, noun: str) -> str:
    """*number* and *noun*, plural unless *number* is 1: ``1 file``, ``2 files``."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def shown_path(path: str, workspace: Path) -> str:
    """``./`` and the path below the current directory, or the absolute path, on one line."""
    try:
        relative = Path(path).relative_to(workspace)
    except ValueError:
        shown = path
    else:
        shown = "." if not relative.parts else f".{os.sep}{relative}"
    return one_line(shown)
