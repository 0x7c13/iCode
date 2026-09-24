# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""MCP echo server that emits a banner before speaking JSON-RPC.

Used by integration tests to verify ``tolerant_stdio_client`` ignores
non-JSON-RPC stdout chatter.  Three modes via argv[1]:

* ``banner-then-serve`` — print banner, then run the normal MCP server.
* ``banner-then-hang`` — print banner, then sleep forever (no MCP).
* ``malformed-json`` — print a malformed JSON-RPC message, then sleep.
* ``stderr-then-exit`` — emit startup failure details on stderr and exit 23.
"""

from __future__ import annotations

import sys
import time

from mcp.server.fastmcp import FastMCP


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "banner-then-serve"

    if mode == "banner-then-serve":
        sys.stdout.write("FooBarServer v1.0 (build 42)\n")
        sys.stdout.write("Initializing backend...\n")
        sys.stdout.write("\n")  # blank line — should be ignored silently
        sys.stdout.flush()

        mcp = FastMCP("banner-server")

        @mcp.tool()
        def echo(message: str) -> str:
            """Echo back the input message."""
            return message

        mcp.run(transport="stdio")
        return

    if mode == "banner-then-hang":
        sys.stdout.write("FooBarServer v1.0\n")
        sys.stdout.write("Waiting for connection...\n")
        sys.stdout.flush()
        while True:
            time.sleep(60)

    if mode == "malformed-json":
        # Looks like JSON (starts with `{`) but is not valid JSON-RPC.
        sys.stdout.write('{"not": "a valid jsonrpc message"}\n')
        sys.stdout.flush()
        while True:
            time.sleep(60)

    if mode == "stderr-then-exit":
        sys.stderr.write("Failed to start MCP server\n")
        # Exceed typical pipe capacity so the integration test also proves
        # Chrys drains stderr concurrently instead of deadlocking the child.
        sys.stderr.write("dependency resolver detail\n" * 5_000)
        sys.stderr.write("ModuleNotFoundError: No module named 'missing_mcp_dependency'\n")
        sys.stderr.flush()
        raise SystemExit(23)

    raise SystemExit(f"unknown mode: {mode!r}")


if __name__ == "__main__":
    main()
