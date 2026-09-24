# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Skill path normalization helpers for agent profile validation."""

from __future__ import annotations

import posixpath
from collections.abc import Callable
from pathlib import Path

from chrys.app.tui.screens.agents.validation_messages import PATHS_MATCH_CASE_INSENSITIVE, PATHS_MATCH_NORMALIZED
from chrys.foundation.i18n import MessageRef
from chrys.foundation.i18n.formatting import format_message


def normalize_skill_path_for_compare(path: str) -> str:
    """Normalize a skill path for duplicate detection."""
    from chrys.foundation.platform import get_platform

    expanded = str(Path(path.strip()).expanduser()).replace("\\", "/")
    normalized = posixpath.normpath(expanded)
    platform_info = get_platform()
    if platform_info.is_macos or platform_info.is_windows:
        return normalized.lower()
    return normalized


def skill_path_duplicate_match_description(
    render_message: Callable[[MessageRef], str] = format_message,
) -> str:
    """Describe path matching semantics for duplicate validation errors."""
    from chrys.foundation.platform import get_platform

    platform_info = get_platform()
    definition = (
        PATHS_MATCH_CASE_INSENSITIVE if platform_info.is_macos or platform_info.is_windows else PATHS_MATCH_NORMALIZED
    )
    return render_message(definition.bind())
