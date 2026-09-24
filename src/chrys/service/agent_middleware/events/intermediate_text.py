# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Intermediate assistant text buffering for tool-event ordering."""

from __future__ import annotations


class IntermediateTextBuffer:
    """Buffer for intermediate text detected by result_hook.

    The result_hook (sync, cannot await) stores text here.
    ToolEventMiddleware drains it and publishes before ToolCallStart,
    ensuring correct event ordering (text -> tool call) in the TUI.

    Also tracks a ``batch_id`` counter that increments once per LLM
    response.  The middleware reads this to tag session messages with
    the correct batch grouping.
    """

    def __init__(self) -> None:
        self._pending: list[str] = []
        self.batch_id: int = 0

    def new_batch(self) -> None:
        """Signal a new LLM response (batch boundary)."""
        self.batch_id += 1

    def store(self, text: str) -> None:
        """Store intermediate text (called from sync result_hook)."""
        self.new_batch()
        self._pending.append(text)

    def drain(self) -> list[str]:
        """Drain all pending text (called from ToolEventMiddleware)."""
        items = self._pending.copy()
        self._pending.clear()
        return items
