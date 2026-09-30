# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The ``mcp`` catalog: the instructions the agent's MCP servers publish."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class McpSource:
    """The MCP server instructions one middleware offers, read from the agent's MCP adapter."""

    name = "mcp"
    """The catalog's record name."""
    withdrawn = (
        "<mcp_instructions>\n"
        "  <none>No MCP server instructions apply to the current agent; earlier ones no longer apply.</none>\n"
        "</mcp_instructions>"
    )
    """Replaces the instructions once the current agent's servers publish none."""

    def __init__(self, provider: Callable[[], str | None] | None) -> None:
        self._provider = provider

    def snapshot(self) -> str | None:
        """The current MCP server instructions block; None without a provider or when it failed."""
        if self._provider is None:
            return None
        try:
            return self._provider()
        except Exception:
            logger.exception("Failed to render MCP server instructions reminder")
            return None
