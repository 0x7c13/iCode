# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The ``skills`` catalog: the runtime skills available to the agent in this workspace.

The one catalog a turn may refresh mid-turn (a skill added by a user
injection or retry is a real capability change); ``system_reminder`` owns
those refresh points and their scope checks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class SkillsSource:
    """The runtime skill catalog one middleware offers, read from the skill provider."""

    name = "skills"
    """The catalog's record name."""
    withdrawn = (
        "<available_skills>\n"
        "  <none>No runtime skills are available for the current agent and workspace.</none>\n"
        "</available_skills>"
    )
    """Replaces the catalog once no runtime skill is available."""

    def __init__(self, provider: Callable[[], str | None] | None) -> None:
        self._provider = provider

    def snapshot(self) -> str | None:
        """The latest runtime skill catalog; None without a provider or when it failed."""
        if self._provider is None:
            return None
        try:
            return self._provider()
        except Exception:
            logger.exception("Failed to render runtime skill catalog reminder")
            return None
