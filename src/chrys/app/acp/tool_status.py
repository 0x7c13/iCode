# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Helpers for mapping Chrys tool results to ACP statuses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chrys.foundation.tool_kinds import KIND_MCP, KIND_SHELL
from chrys.foundation.tool_result_metadata import (
    legacy_error_text_applies,
    result_text_exit_code,
    tool_result_metadata_failure_state,
)
from chrys.service.tools.builtins.shell import SHELL_TOOL_NAMES


def tool_result_failed(
    result_text: str,
    metadata: Mapping[str, Any] | None,
    *,
    tool_kind: str = "",
    tool_name: str = "",
    has_exception: bool = False,
) -> bool:
    """Return whether ACP should mark a tool result as failed."""
    metadata_state = tool_result_metadata_failure_state(metadata)
    if metadata_state is not None:
        return metadata_state
    if has_exception:
        return True
    # The name fallback only fires while the persisted kind is empty (old
    # sessions predating kind stamping) — an MCP tool legitimately named
    # e.g. "bash" must not be judged by shell exit-code text.  Mirrors
    # ``legacy_error_text_applies``.
    if tool_kind == KIND_SHELL or (not tool_kind and tool_name in SHELL_TOOL_NAMES):
        exit_code = result_text_exit_code(result_text)
        if exit_code is not None:
            return exit_code != 0
    # ACP clients have no dedicated MCP renderer, so preserve the legacy
    # isError-style signal here. The TUI stays looser for MCP text payloads.
    return (
        tool_kind == KIND_MCP or legacy_error_text_applies(tool_kind, tool_name)
    ) and result_text.lstrip().startswith("Error:")
