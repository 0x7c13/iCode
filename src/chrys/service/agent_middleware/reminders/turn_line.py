# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The turn line: when the turn started, and how full the context was after the previous one.

Snapshotted when a turn starts fresh and sent once per turn; a retry that
preserves the turn's reminders keeps it byte-identical.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from chrys.foundation.platform.files import surrogate_safe_text

from .context_usage import format_usage_line

if TYPE_CHECKING:
    from collections.abc import Mapping

    from chrys.foundation.models.session_env import SessionEnvironment


class TurnLineSource:
    """The line one middleware opens each turn with."""

    def __init__(self, runtime: SessionEnvironment | None, *, max_context_tokens: int) -> None:
        self._runtime = runtime
        self._max_context_tokens = max_context_tokens

    def snapshot(self, usage: Mapping[str, Any]) -> str | None:
        """The line for a turn starting now: the time with a runtime, then the previous turn's *usage* if any."""
        lines: list[str] = []
        if self._runtime is not None:
            lines.append(self.clock())
        if usage:
            lines.append(format_usage_line(usage, max_context_tokens=self._max_context_tokens))
        return "\n".join(lines) if lines else None

    @staticmethod
    def clock() -> str:
        """The current local and UTC time."""
        now_utc = datetime.now(tz=UTC)
        now_local = now_utc.astimezone()
        return surrogate_safe_text(
            f"[Turn Start] Local time: {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
            f"UTC time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
