# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Minimal MCP echo server for integration tests.

Run with: python -m tests.service.mcp.mcp_echo_server
Exposes two tools over stdio: echo (returns input) and add (sums two numbers).
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

ECHO_SERVER_INSTRUCTIONS = "Use echo to repeat text and add to sum two integers."

mcp = FastMCP("echo-server", instructions=ECHO_SERVER_INSTRUCTIONS)


@mcp.tool()
def echo(message: str) -> str:
    """Echo back the input message."""
    return message


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


if __name__ == "__main__":
    mcp.run(transport="stdio")
