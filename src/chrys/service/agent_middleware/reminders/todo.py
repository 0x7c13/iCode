# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The ``todo`` catalog: the session's todo list when the turn starts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class TodoSource:
    """The todo list one middleware offers, read from the session's todo provider."""

    name = "todo"
    """The catalog's record name."""
    withdrawn = "Current todo list: empty. Earlier todo lists no longer apply."
    """Replaces the list once the provider offers none (the list emptied)."""

    def __init__(self, provider: Callable[[], str | None] | None) -> None:
        self._provider = provider

    def snapshot(self) -> str | None:
        """The todo-list reminder text; None when the list is empty, there is no provider or it failed."""
        if self._provider is None:
            return None
        try:
            return self._provider() or None
        except Exception:
            logger.exception("Failed to render todo-list reminder")
            return None
