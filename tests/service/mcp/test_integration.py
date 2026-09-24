# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Integration tests for MCP adapter with a real MCP server (echo-server)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from chrys.kernel import FunctionTool as ChrysFunctionTool
from chrys.service.mcp.adapter import MCPAdapter
from chrys.service.profiles.agents.schema import MCPServerConfig

SERVER_DIR = Path(__file__).parent


@pytest.fixture
def stdio_config() -> MCPServerConfig:
    """Config pointing at the echo server in this test directory."""
    return MCPServerConfig(
        name="echo-server",
        transport="stdio",
        command=sys.executable,
        args=[str(SERVER_DIR / "mcp_echo_server.py")],
    )


@pytest.mark.integration
async def test_connect_and_list_tools(stdio_config: MCPServerConfig) -> None:
    adapter = MCPAdapter()
    try:
        tools = await adapter.connect(stdio_config)
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"echo", "add"}
        assert adapter.server_names == ["echo-server"]
        # MCPAdapter delivers tools out of the MCP domain as chrys
        # FunctionTool clones (the lease funnel constructs the subclass).
        assert all(type(t) is ChrysFunctionTool for t in tools)
    finally:
        await adapter.disconnect_all()


@pytest.mark.integration
async def test_call_echo_tool(stdio_config: MCPServerConfig) -> None:
    adapter = MCPAdapter()
    try:
        tools = await adapter.connect(stdio_config)
        echo_tool = next(t for t in tools if t.name == "echo")
        result = await echo_tool.invoke(arguments={"message": "hello"})
        # Result is a list of Content objects; extract text
        assert any("hello" in c.text for c in result)
    finally:
        await adapter.disconnect_all()


@pytest.mark.integration
async def test_call_add_tool(stdio_config: MCPServerConfig) -> None:
    adapter = MCPAdapter()
    try:
        tools = await adapter.connect(stdio_config)
        add_tool = next(t for t in tools if t.name == "add")
        result = await add_tool.invoke(arguments={"a": 3, "b": 7})
        assert any("10" in c.text for c in result)
    finally:
        await adapter.disconnect_all()


@pytest.mark.integration
async def test_server_instructions_captured_from_real_handshake(stdio_config: MCPServerConfig) -> None:
    """InitializeResult.instructions flows through connect() into the rendered reminder."""
    from tests.service.mcp.mcp_echo_server import ECHO_SERVER_INSTRUCTIONS

    adapter = MCPAdapter()
    try:
        await adapter.connect(stdio_config)
        assert adapter.get_server_instructions_map() == {"echo-server": ECHO_SERVER_INSTRUCTIONS}
        reminder = adapter.render_instructions_reminder()
        assert reminder is not None
        assert '<server name="echo-server">' in reminder
        assert ECHO_SERVER_INSTRUCTIONS in reminder
    finally:
        await adapter.disconnect_all()


@pytest.mark.integration
async def test_connect_idempotent(stdio_config: MCPServerConfig) -> None:
    """Connecting twice to the same server returns cached tools."""
    adapter = MCPAdapter()
    try:
        tools1 = await adapter.connect(stdio_config)
        tools2 = await adapter.connect(stdio_config)
        assert len(tools1) == len(tools2)
        assert adapter.server_names == ["echo-server"]
    finally:
        await adapter.disconnect_all()


@pytest.mark.integration
async def test_disconnect_and_reconnect(stdio_config: MCPServerConfig) -> None:
    adapter = MCPAdapter()
    try:
        await adapter.connect(stdio_config)
        await adapter.disconnect("echo-server")
        assert adapter.server_names == []

        tools = await adapter.connect(stdio_config)
        assert len(tools) == 2
    finally:
        await adapter.disconnect_all()


# ---------------------------------------------------------------------------
# tolerant_stdio_client + connect timeout — real subprocesses
# ---------------------------------------------------------------------------


def _banner_config(name: str, mode: str, *, request_timeout: int | None = None) -> MCPServerConfig:
    """Build a config that runs ``mcp_banner_server.py`` in the given mode."""
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=sys.executable,
        args=[str(SERVER_DIR / "mcp_banner_server.py"), mode],
        request_timeout=request_timeout,
    )


def _env_config(name: str = "env-server", *, env: dict[str, str] | None = None) -> MCPServerConfig:
    """Build a config that reports its subprocess environment."""
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=sys.executable,
        args=[str(SERVER_DIR / "mcp_env_server.py")],
        env=env or {},
    )


async def _read_child_env(config: MCPServerConfig, name: str) -> str:
    adapter = MCPAdapter()
    try:
        tools = await adapter.connect(config)
        get_env = next(t for t in tools if t.name == "get_env")
        result = await get_env.invoke(arguments={"name": name})
        return "\n".join(c.text for c in result)
    finally:
        await adapter.disconnect_all()


@pytest.mark.integration
async def test_stdio_inherits_parent_lowercase_no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A parent-process lowercase ``no_proxy`` reaches the real stdio child.

    This reproduces the upstream SDK allowlist failure mode: if Chrys used
    ``stdio_client`` directly, the child would only receive the SDK's small
    default environment and this value would be missing.
    """
    value = "localhost,.corp.example"
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setenv("no_proxy", value)

    assert await _read_child_env(_env_config(), "no_proxy") == value
    assert await _read_child_env(_env_config(), "NO_PROXY") == value


@pytest.mark.integration
async def test_stdio_mirrors_parent_uppercase_no_proxy_to_lowercase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canonical ``NO_PROXY`` is also exposed as lowercase for stdio commands.

    Some spawned commands check only the lowercase proxy convention, so the
    stdio env builder mirrors the inherited alias at the subprocess boundary.
    """
    value = "127.0.0.1,localhost"
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", value)

    assert await _read_child_env(_env_config(), "no_proxy") == value


@pytest.mark.integration
async def test_stdio_per_server_no_proxy_overrides_parent_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-server MCP env wins over inherited aliases and is mirrored."""
    monkeypatch.setenv("no_proxy", "stale-parent.example")

    value = "localhost,.server.example"
    config = _env_config(env={"NO_PROXY": value})

    assert await _read_child_env(config, "no_proxy") == value


@pytest.mark.integration
async def test_banner_then_serve_connects_successfully() -> None:
    """A server that prints a banner then speaks JSON-RPC connects fine.

    The tolerant reader must drop the banner lines without breaking the
    initialize handshake.
    """
    from chrys.service.mcp._stdio_transport import _SafeStdioTool

    adapter = MCPAdapter()
    config = _banner_config("banner-server", "banner-then-serve")
    try:
        tools = await adapter.connect(config)
        assert len(tools) == 1
        assert tools[0].name == "echo"
        # Banner lines were captured for diagnostics, even on success.
        tool = adapter._servers["banner-server"]
        assert isinstance(tool, _SafeStdioTool)
        assert any("FooBarServer" in line for line in tool.dropped_banner_lines)
    finally:
        await adapter.disconnect_all()


@pytest.mark.integration
async def test_stdio_early_exit_surfaces_stderr_and_process_exit_code() -> None:
    """A child that exits before initialize reports its real startup failure."""
    from chrys.service.mcp.adapter import MCPConnectionError

    adapter = MCPAdapter()
    config = _banner_config("early-exit", "stderr-then-exit")

    with pytest.raises(MCPConnectionError) as info:
        await asyncio.wait_for(adapter.connect(config), timeout=10)

    err = info.value
    assert err.process_exit_code == 23
    assert err.stderr_dropped_bytes > 0
    assert "missing_mcp_dependency" in err.stderr_tail
    assert "Server process exit code: 23" in str(err)
    assert "ModuleNotFoundError" in str(err)


@pytest.mark.integration
async def test_banner_then_hang_times_out_with_diagnostics() -> None:
    """A server that prints a banner and then hangs surfaces a timeout + banner.

    Verifies the timeout floor in MCPAdapter.test_connection — without it,
    this server would block forever (regression that motivated this fix).
    """
    from chrys.service.mcp.adapter import MCPConnectionError

    adapter = MCPAdapter()
    config = _banner_config("hung-server", "banner-then-hang", request_timeout=2)

    with pytest.raises(MCPConnectionError) as info:
        await asyncio.wait_for(adapter.test_connection(config), timeout=10)

    err = info.value
    assert isinstance(err.cause, TimeoutError)
    assert any("FooBarServer" in line for line in err.banner_lines)
    assert "non-JSON output" in str(err)


@pytest.mark.integration
async def test_banner_then_hang_uses_default_floor_when_no_request_timeout() -> None:
    """The default timeout floor must protect a hung server even with no ``request_timeout``.

    This is the path that ``asyncio.wait_for`` could not reach safely:
    MCPTool._run_lifecycle_owner runs in a separate
    ``asyncio.Task`` with anyio cancel scopes pinned to it, so the
    timeout has to fire from inside via the SDK's ``read_timeout_seconds``.
    The fix injects ``DEFAULT_TEST_TIMEOUT_SECONDS`` into the tool's
    ``request_timeout`` when the user didn't set one.

    Without this, the test would hang indefinitely.  With the (previous)
    ``asyncio.wait_for`` approach, it raised
    ``RuntimeError("Attempted to exit cancel scope in a different task ...")``
    instead of a clean ``MCPConnectionError``.
    """
    from chrys.service.mcp.adapter import MCPConnectionError

    adapter = MCPAdapter()
    # No request_timeout — must rely on the default floor.
    config = _banner_config("hung-no-cfg", "banner-then-hang")

    # Wrap in a generous outer timeout so a regression manifests as a
    # test failure rather than a wedged CI run.
    with (
        patch("chrys.service.mcp.adapter.DEFAULT_TEST_TIMEOUT_SECONDS", 2),
        pytest.raises(MCPConnectionError) as info,
    ):
        await asyncio.wait_for(adapter.test_connection(config), timeout=10)

    err = info.value
    assert isinstance(err.cause, TimeoutError)
    assert any("FooBarServer" in line for line in err.banner_lines)


@pytest.mark.integration
async def test_connect_all_uses_default_floor_for_hung_server_without_request_timeout() -> None:
    """``connect_all`` must isolate a hung server even when no per-server timeout is set."""
    adapter = MCPAdapter()
    bad = _banner_config("bad-no-cfg", "banner-then-hang")  # no request_timeout
    good = MCPServerConfig(
        name="good",
        transport="stdio",
        command=sys.executable,
        args=[str(SERVER_DIR / "mcp_echo_server.py")],
    )

    try:
        # Keep this comfortably above normal MCP startup under xdist load:
        # the assertion is that the hung server is bounded by the default
        # floor, not that a healthy peer can always initialize within 2s on
        # a saturated CI worker.
        with patch("chrys.service.mcp._connection.DEFAULT_CONNECT_TIMEOUT_SECONDS", 5):
            tools = await asyncio.wait_for(adapter.connect_all([bad, good]), timeout=20)
    finally:
        await adapter.disconnect_all()

    names = {t.name for t in tools}
    assert names == {"echo", "add"}
    assert "bad-no-cfg" in adapter.failures


@pytest.mark.integration
async def test_malformed_jsonrpc_surfaces_under_timeout_floor() -> None:
    """A server that emits valid JSON but invalid JSON-RPC eventually surfaces.

    Distinguishes "banner chatter" from "wrong protocol".  The protocol
    error is pushed onto the read stream but the SDK's ``_handle_incoming``
    swallows it — so the pending ``initialize()`` only fails once
    ``read_timeout_seconds`` fires.  Our adapter's ``request_timeout``
    floor turns that into a bounded ``MCPConnectionError`` instead of an
    indefinite hang.
    """
    from chrys.service.mcp.adapter import MCPConnectionError

    adapter = MCPAdapter()
    config = _banner_config("malformed-server", "malformed-json", request_timeout=3)

    with pytest.raises(MCPConnectionError):
        await asyncio.wait_for(adapter.test_connection(config), timeout=10)


@pytest.mark.integration
async def test_connect_all_isolates_one_hung_server() -> None:
    """One hung server does not block another working server in connect_all."""
    adapter = MCPAdapter()
    bad = _banner_config("bad", "banner-then-hang", request_timeout=2)
    good = MCPServerConfig(
        name="good",
        transport="stdio",
        command=sys.executable,
        args=[str(SERVER_DIR / "mcp_echo_server.py")],
    )
    try:
        tools = await asyncio.wait_for(adapter.connect_all([bad, good]), timeout=20)
        # The good server's tools must be present; the bad one is recorded
        # as a failure but does not abort the build.
        names = {t.name for t in tools}
        assert names == {"echo", "add"}
        assert "bad" in adapter.failures
        assert "good" in adapter.server_names
    finally:
        await adapter.disconnect_all()


@pytest.mark.integration
async def test_delivered_tools_carry_static_provenance_context(stdio_config: MCPServerConfig) -> None:
    """owned.py stamps server identity at creation and the cache clone preserves it."""
    from chrys.foundation.tool_call_context import get_tool_context

    adapter = MCPAdapter()
    try:
        tools = await adapter.connect(stdio_config)
        echo_tool = next(t for t in tools if t.name == "echo")
        assert get_tool_context(echo_tool) == {
            "server_name": "echo-server",
            "remote_name": "echo",
            "normalized_name": "echo",
        }
    finally:
        await adapter.disconnect_all()
