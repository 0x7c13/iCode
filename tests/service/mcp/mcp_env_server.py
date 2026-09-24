# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Minimal MCP server that exposes selected process environment values."""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("env-server")


@mcp.tool()
def get_env(name: str) -> str:
    """Return an environment variable value from the MCP server process."""
    return os.environ.get(name, "<missing>")


if __name__ == "__main__":
    mcp.run(transport="stdio")
