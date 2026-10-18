# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The patched MCP POST path preserves the SDK's redirect security boundary."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from mcp.shared.message import SessionMessage
from mcp.types import CONNECTION_CLOSED, JSONRPCError, JSONRPCMessage, JSONRPCRequest, JSONRPCResponse

from chrys.service.mcp._http_transport import _chrys_streamable_http_client
from chrys.service.mcp.owned import LOCAL_HTTP_FAILURE_ERROR_DATA


@pytest.mark.parametrize("follow_redirects", [False, True], ids=["default-client", "redirecting-client"])
@pytest.mark.parametrize(
    ("origin", "location", "status", "allowed"),
    [
        ("https://origin.test/mcp", "/mcp/", 307, True),
        ("https://origin.test/mcp", "/mcp/", 308, True),
        ("http://origin.test/mcp", "https://origin.test/mcp/", 307, True),
        ("https://origin.test/mcp", "https://foreign.test/mcp", 307, False),
        ("https://origin.test/mcp", "https://origin.test:8443/mcp", 307, False),
        ("https://origin.test/mcp", "http://origin.test/mcp", 307, False),
        ("https://origin.test/mcp", "https://user:password@origin.test/mcp/", 307, False),
        ("https://origin.test/mcp", "/mcp/", 301, False),
        ("https://origin.test/mcp", "/mcp/", 302, False),
        ("https://origin.test/mcp", "/mcp/", 303, False),
    ],
    ids=["slash-307", "slash-308", "https-upgrade", "host", "port", "downgrade", "userinfo", "301", "302", "303"],
)
async def test_post_redirects_preserve_origin_method_and_timeout(
    origin: str,
    location: str,
    status: int,
    allowed: bool,
    follow_redirects: bool,
) -> None:
    """Unsafe redirects return a local error without disclosing the request to the target."""
    sent: list[httpx.Request] = []
    payload = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "echo", "arguments": {"key": "s"}}}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if str(request.url) == origin:
            return httpx.Response(status, headers={"Location": location})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 7, "result": {"ok": True}})

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            headers={"X-Key": "s"},
            follow_redirects=follow_redirects,
        ) as client,
        _chrys_streamable_http_client(origin, http_client=client, terminate_on_close=False, request_timeout=2.0) as (
            read_stream,
            write_stream,
            _get_session_id,
        ),
    ):
        await write_stream.send(SessionMessage(JSONRPCMessage(JSONRPCRequest.model_validate(payload))))
        received = await asyncio.wait_for(read_stream.receive(), timeout=5.0)

    assert isinstance(received, SessionMessage)
    result = received.message.root
    if allowed:
        assert isinstance(result, JSONRPCResponse)
        assert result.id == 7
        assert result.result == {"ok": True}
        assert [str(request.url) for request in sent] == [origin, str(httpx.URL(origin).join(location))]
    else:
        assert isinstance(result, JSONRPCError)
        assert result.id == 7
        assert result.error.code == CONNECTION_CLOSED
        assert result.error.data == LOCAL_HTTP_FAILURE_ERROR_DATA
        assert "not followed" in result.error.message
        assert "user:password" not in result.error.message
        assert [str(request.url) for request in sent] == [origin]

    for request in sent:
        assert request.method == "POST"
        assert request.headers["x-key"] == "s"
        assert json.loads(request.content) == payload
        assert request.extensions["timeout"] == {"connect": 1.75, "read": 1.75, "write": 1.75, "pool": 1.75}
