# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tool-specific body renderers for approval dialogs."""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard

if TYPE_CHECKING:
    from textual.widget import Widget

    from chrys.foundation.i18n import MessageRef

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ApprovalBypass:
    """Immediate approval response for requests that should not show a dialog."""

    approved: bool = True
    reason: str = ""
    debug_reason: str = ""


@dataclass(slots=True)
class ApprovalBody:
    """Custom body content for an approval dialog."""

    detail: MessageRef | str | None = None
    widgets: list[Widget] = field(default_factory=list)
    hidden_arg_keys: frozenset[str] = field(default_factory=frozenset)
    bypass: ApprovalBypass | None = None
    modified_args: Callable[[], dict[str, Any] | None] | None = None


type ApprovalBodyResult = ApprovalBody | None
type ApprovalBodyBuilder = Callable[
    [str, dict[str, Any], str | None], ApprovalBodyResult | Awaitable[ApprovalBodyResult]
]

_REGISTRY: dict[str, ApprovalBodyBuilder] = {}
_KIND_REGISTRY: dict[str, ApprovalBodyBuilder] = {}
_loaded = False


def _is_approval_body_awaitable(
    result: ApprovalBodyResult | Awaitable[ApprovalBodyResult],
) -> TypeGuard[Awaitable[ApprovalBodyResult]]:
    """Preserve the builder result type across ``inspect.isawaitable``."""
    return inspect.isawaitable(result)


def register_approval_body(tool_name: str, builder: ApprovalBodyBuilder) -> None:
    """Register a custom approval body builder for a tool name."""
    _REGISTRY[tool_name] = builder


def register_approval_kind_body(tool_kind: str, builder: ApprovalBodyBuilder) -> None:
    """Register a custom approval body builder for every tool of a kind."""
    _KIND_REGISTRY[tool_kind] = builder


async def create_approval_body(
    tool_name: str,
    tool_kind: str,
    args: dict[str, Any],
    *,
    workspace_cwd: str | None = None,
) -> ApprovalBody | None:
    """Return a custom approval body for *tool_name*, if one is registered."""
    _ensure_loaded()
    name_builder = _REGISTRY.get(tool_name)
    kind_builder = _KIND_REGISTRY.get(tool_kind)
    builders = [builder for builder in (name_builder, kind_builder) if builder is not None]
    if len(builders) == 2 and builders[0] is builders[1]:
        # Same callable registered under both name and kind.
        builders.pop()

    for builder in builders:
        try:
            candidate = builder(tool_kind, args, workspace_cwd)
            if _is_approval_body_awaitable(candidate):
                result: object = await candidate
            else:
                result = candidate
            if result is None:
                continue
            if not isinstance(result, ApprovalBody):
                msg = f"Approval body builder returned {type(result).__name__}, expected ApprovalBody or None"
                raise TypeError(msg)
            return result
        except Exception:
            logger.debug("Approval body builder failed for %s", tool_name, exc_info=True)
    return None


def _ensure_loaded() -> None:
    """Import built-in approval body modules on first use."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    importlib.import_module("chrys.app.tui.screens.dialogs.approval.bodies.file_edit")
    importlib.import_module("chrys.app.tui.screens.dialogs.approval.bodies.sub_agent")
