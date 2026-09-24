# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""MCP initialization failures remain visible through cancellation and teardown."""

from __future__ import annotations

import asyncio
import sys

import pytest

from chrys.kernel.exceptions import ToolException
from chrys.service.mcp._stdio_transport import _SafeStdioTool
from chrys.service.mcp.errors import MCPConnectionError
from chrys.service.mcp.owned import _describe_error, _describe_with_cleanup


async def test_invalid_utf8_stdio_failure_reaches_user_diagnostic() -> None:
    # Wait for initialize so the reader failure cancels a live request.
    script = "import sys; sys.stdin.buffer.readline(); sys.stdout.buffer.write(b'\\xff\\n'); sys.stdout.buffer.flush()"
    tool = _SafeStdioTool(name="invalid-utf8", command=sys.executable, args=["-c", script], request_timeout=5)
    try:
        with pytest.raises(ToolException) as raised:
            await tool.connect()
        displayed = str(MCPConnectionError("invalid-utf8", "stdio", raised.value))
        assert "utf-8" in displayed.lower()
        assert "decode" in displayed.lower()
        assert "Cancelled via cancel scope" not in displayed
    finally:
        await tool.close()


def test_diagnostic_unwrap_is_bounded_and_prefers_cleanup_for_cancelled_initialize() -> None:
    cancelled = asyncio.CancelledError("Cancelled via cancel scope")
    cleanup = ExceptionGroup("task group", [ValueError("bad protocol bytes")])
    assert _describe_with_cleanup(cancelled, cleanup) == "bad protocol bytes"
    original = ValueError("original")
    assert _describe_with_cleanup(original, cleanup) == "original"
    original.__cause__ = original
    assert _describe_error(original) == "original"
