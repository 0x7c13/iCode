# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared canonical tool-content display helpers."""

from __future__ import annotations

from chrys.kernel import Content


def tool_call_name(content: Content) -> str | None:
    """Return the display name for any canonical local or hosted call."""
    if content.type == "function_call":
        return content.name
    if content.type in {"hosted_tool_call", "mcp_server_tool_call", "search_tool_call"}:
        return content.tool_name
    if content.type == "code_interpreter_tool_call":
        return "code_interpreter"
    if content.type == "image_generation_tool_call":
        return "image_generation"
    if content.type == "shell_tool_call":
        return "shell"
    return None
