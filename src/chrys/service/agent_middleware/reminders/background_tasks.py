# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The ``background_tasks`` catalog: the main agent's background commands when the turn starts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class BackgroundTasksSource:
    """The background commands one middleware lists, read from the session's registry."""

    name = "background_tasks"
    """The catalog's record name."""
    withdrawn = "There are no background commands anymore. Earlier lists of background commands no longer apply."
    """Replaces the list once the registry offers none."""

    def __init__(self, provider: Callable[[], str | None] | None) -> None:
        self._provider = provider

    def snapshot(self) -> str | None:
        """The list's reminder text; None when there is nothing to list, no provider, or it failed."""
        if self._provider is None:
            return None
        try:
            return self._provider() or None
        except Exception:
            logger.exception("Failed to render background-task reminder")
            return None
