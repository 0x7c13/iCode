# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for ACP tool-result failure mapping (``tool_result_failed``).

Pins the name-vs-kind contract: the persisted kind is authoritative, and the
``SHELL_TOOL_NAMES`` fallback only applies while the kind is empty (legacy
sessions predating kind stamping).
"""

from __future__ import annotations

from chrys.app.acp.tool_status import tool_result_failed
from chrys.foundation.tool_kinds import KIND_MCP, KIND_SHELL
from chrys.foundation.tool_result_metadata import SHELL_EXIT_CODE_METADATA_KEY


def test_shell_kind_parses_exit_code_suffix() -> None:
    assert tool_result_failed("boom\n[exit_code: 7]", None, tool_kind=KIND_SHELL, tool_name="mksh")
    assert not tool_result_failed("ok\n[exit_code: 0]", None, tool_kind=KIND_SHELL, tool_name="mksh")


def test_legacy_blank_kind_falls_back_to_shell_names() -> None:
    assert tool_result_failed("boom\n[exit_code: 7]", None, tool_kind="", tool_name="bash")
    assert not tool_result_failed("ok\n[exit_code: 0]", None, tool_kind="", tool_name="bash")


def test_mcp_tool_named_after_shell_is_not_exit_code_parsed() -> None:
    """An MCP tool legitimately named ``bash`` succeeded; text that happens to
    contain a shell-style exit suffix must not flip it to failed."""
    assert not tool_result_failed("payload\n[exit_code: 7]", None, tool_kind=KIND_MCP, tool_name="bash")


def test_mcp_error_text_signal_is_preserved() -> None:
    assert tool_result_failed("Error: upstream broke", None, tool_kind=KIND_MCP, tool_name="bash")


def test_structured_metadata_wins_over_result_text() -> None:
    metadata = {SHELL_EXIT_CODE_METADATA_KEY: 3}
    assert tool_result_failed("ok\n[exit_code: 0]", metadata, tool_kind=KIND_SHELL, tool_name="zsh")


def test_exception_marks_failed_when_metadata_is_silent() -> None:
    assert tool_result_failed("partial", None, tool_kind=KIND_SHELL, tool_name="zsh", has_exception=True)
