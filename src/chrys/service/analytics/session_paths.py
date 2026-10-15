# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Recognize session-owned trajectory logs, including relocated session stores."""

from __future__ import annotations

from pathlib import Path

from chrys.foundation.config.settings import resolve_sessions_dir


def session_directory_for_events(path: Path) -> Path | None:
    """Return the owning session directory, or None for an unrelated loose log.

    Conventional ``sessions/<id>/trajectory/events.jsonl`` trees remain readable
    outside the active store. Its resolved directory may have any name when
    ``<root>/sessions`` is a link, so also recognize the configured store by
    path or filesystem identity. Resolving it must never create directories.
    """
    if len(path.parents) < 3 or path.name != "events.jsonl" or path.parent.name != "trajectory":
        return None
    sessions = path.parents[2]
    if sessions.name != "sessions":
        try:
            configured = resolve_sessions_dir(create=False)
            if sessions != configured and not sessions.samefile(configured):
                return None
        except OSError, ValueError:
            return None
    return path.parents[1]
