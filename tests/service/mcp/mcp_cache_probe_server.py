# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Stateful MCP probe server used by cache E2E tests.

The same script can run as either a stdio MCP server or a Streamable HTTP MCP
server.  It records process starts and tool calls to a JSONL state file so
tests can verify whether Chrys reused a live MCP connection or spawned/created
a new one.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import uvicorn
from mcp.server.fastmcp import FastMCP


def _append_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _build_server(*, state_file: Path, transport: str, host: str, port: int) -> FastMCP:
    call_count = 0
    process_id = os.getpid()
    mcp = FastMCP(
        "cache-probe-server",
        host=host,
        port=port,
        streamable_http_path="/mcp",
        log_level="ERROR",
    )

    @mcp.tool()
    def probe(label: str = "") -> str:
        """Return state proving which process and call sequence handled the request."""
        nonlocal call_count
        call_count += 1
        payload = {
            "event": "probe",
            "pid": process_id,
            "transport": transport,
            "label": label,
            "call_count": call_count,
            "cwd": os.getcwd(),
            "token": os.environ.get("CHRYS_E2E_TOKEN", ""),
        }
        _append_event(state_file, payload)
        return json.dumps(payload, sort_keys=True)

    return mcp


def _run_http_server(server: FastMCP, *, host: str, port: int, port_file: Path | None) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen()
    actual_port = int(listener.getsockname()[1])
    if port_file is not None:
        port_file.write_text(str(actual_port), encoding="utf-8")

    config = uvicorn.Config(server.streamable_http_app(), log_level="error")
    uvicorn.Server(config).run(sockets=[listener])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "http"], required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file")
    parser.add_argument("--startup-delay", type=float, default=0)
    args = parser.parse_args()

    state_file = Path(args.state_file)
    _append_event(
        state_file,
        {
            "event": "start",
            "pid": os.getpid(),
            "transport": args.transport,
            "cwd": os.getcwd(),
            "token": os.environ.get("CHRYS_E2E_TOKEN", ""),
        },
    )
    if args.startup_delay > 0:
        time.sleep(args.startup_delay)
    server = _build_server(
        state_file=state_file,
        transport=args.transport,
        host=args.host,
        port=args.port,
    )
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        _run_http_server(
            server,
            host=args.host,
            port=args.port,
            port_file=Path(args.port_file) if args.port_file else None,
        )


if __name__ == "__main__":
    main()
