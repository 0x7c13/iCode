# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Internal tool-expansion seam consulted by ``validate_tools``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import ToolTypes

type ToolExpander = Callable[["ToolTypes"], Awaitable[list["ToolTypes"] | None]]

_tool_expander: ToolExpander | None = None


def _set_tool_expander(expander: ToolExpander | None) -> None:
    """Set the process-global tool expander used by ``validate_tools``."""
    global _tool_expander
    _tool_expander = expander


def _get_tool_expander() -> ToolExpander | None:
    """Return the currently registered tool expander, if any."""
    return _tool_expander
