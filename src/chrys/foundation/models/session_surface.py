# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The user-facing surface (TUI, CLI or ACP) that last worked in a session."""

from __future__ import annotations

from enum import StrEnum


class SessionSurface(StrEnum):
    """Where a session was last used for a conversation turn or a workflow run.

    Persisted as its value. A session with no recorded surface predates the
    field and is treated as TUI by the session browser.
    """

    TUI = "tui"
    CLI = "cli"
    ACP = "acp"


def parse_session_surface(value: object) -> SessionSurface | None:
    """Read a persisted surface; missing and unknown values are ``None``."""
    if not isinstance(value, str):
        return None
    try:
        return SessionSurface(value)
    except ValueError:
        return None
