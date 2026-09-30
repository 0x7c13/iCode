# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workspace file changes: the notice drained from the workspace change tracker.

The tracker is drained when the turn is prepared, so a notice no request
carried goes back to it (``SystemReminderMiddleware.take_undelivered_file_change``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from chrys.kernel import Message

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FileChangeTurn:
    """One turn's file-change notice."""

    text: str
    delivered: bool = False
    """Whether the notice reached at least one established provider request.

    Set by a ``request_message_observers`` callback at the final-handler
    boundary, not when ``process()`` enriches the messages: on lazy streaming
    paths ``call_next()`` returns a proxy and the request is only established
    at first stream consumption. A pass that ends with this still False
    (Stop, load/hook failure, cancellation before the first request) must
    requeue the notice — the tracker was already drained at ``prepare_turn``.
    """

    def mark_delivered(self, _messages: Sequence[Message]) -> None:
        """The request-message observer that records a delivery."""
        self.delivered = True


class FileChangeSource:
    """The file-change notices one middleware offers, drained from the workspace change tracker."""

    def __init__(self, provider: Callable[[], str | None] | None) -> None:
        self._provider = provider

    def drain(self) -> FileChangeTurn | None:
        """Drain the advisory provider without allowing it to break a turn.

        Unlike the other sources' reads, this consumes the tracker's notice:
        only ``prepare_turn`` calls it.
        """
        if self._provider is None:
            return None
        try:
            text = self._provider() or None
        except Exception:
            logger.debug("Workspace file-change reminder provider failed", exc_info=True)
            return None
        return FileChangeTurn(text) if text is not None else None
