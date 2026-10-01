# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Failures through an HTTPS proxy: what the proxy answered, and what that proves about the first hop.

A loopback proxy answers ``CONNECT`` as each case scripts it. Only a TCP
answer from the proxy itself (refused) is first-hop evidence: a tunnel that
opened and then broke says nothing about which hop failed.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import struct
import sys
from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Any

import anthropic
import httpx
import openai
import pytest

from chrys.foundation.errors import ErrorKind, Origin, classify_error
from chrys.service.llm.clients import create_client
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.network_faults import INJECTED_V4, network_faults
from tests.support.provider_errors import API_HOST

_TARGET = Origin("https", API_HOST, 443)
_CHAT = [{"role": "user", "content": "hi"}]
_UNREACHABLE_PROXY_HOST = "proxy.example.test"


async def _connect_head(reader: asyncio.StreamReader) -> None:
    head = await reader.readuntil(b"\r\n\r\n")
    assert head.startswith(f"CONNECT {API_HOST}:443 ".encode())


def _answer_then_close(status_line: bytes, *headers: bytes) -> Callable[..., Any]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await _connect_head(reader)
            writer.write(status_line + b"\r\n" + b"".join(h + b"\r\n" for h in headers) + b"Content-Length: 0\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()

    return handle


async def _open_tunnel_and_read_client_hello(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Open the tunnel, then read the client's whole first TLS record.

    Closing while that record may still be in flight lets this side's kernel
    answer it with a reset, so the client would see an EOF or a reset
    depending on timing.  Once it is read, the client sends nothing more
    until the server answers, and the hang-up is exactly the one scripted.
    """
    await _connect_head(reader)
    writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
    await writer.drain()
    header = await reader.readexactly(5)
    await reader.readexactly(int.from_bytes(header[3:5]))


async def _tunnel_then_close(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        await _open_tunnel_and_read_client_hello(reader, writer)
    finally:
        writer.close()


async def _recv_at_least(conn: socket.socket, received: bytearray, size: int) -> None:
    loop = asyncio.get_running_loop()
    while len(received) < size:
        chunk = await loop.sock_recv(conn, 4096)
        if not chunk:
            raise ConnectionError("the client hung up")
        received += chunk


@contextlib.asynccontextmanager
async def _resetting_proxy() -> AsyncIterator[Origin]:
    """A proxy that opens the tunnel, reads the ClientHello, then resets the connection.

    It serves one raw socket, not an asyncio transport: closing a proactor
    transport (Windows) shuts the socket down first, which sends a FIN ahead
    of the reset.
    """
    loop = asyncio.get_running_loop()

    async def serve(listener: socket.socket) -> None:
        conn, _address = await loop.sock_accept(listener)
        with conn:
            conn.setblocking(False)
            received = bytearray()
            while b"\r\n\r\n" not in received:
                await _recv_at_least(conn, received, len(received) + 1)
            head, _, record = bytes(received).partition(b"\r\n\r\n")
            assert head.startswith(f"CONNECT {API_HOST}:443 ".encode())
            await loop.sock_sendall(conn, b"HTTP/1.1 200 Connection established\r\n\r\n")
            # As in _open_tunnel_and_read_client_hello: nothing may be in flight at the reset.
            hello = bytearray(record)
            await _recv_at_least(conn, hello, 5)
            await _recv_at_least(conn, hello, 5 + int.from_bytes(hello[3:5]))
            # A zero linger timeout makes close() send a reset instead of a FIN.
            linger = struct.pack("HH", 1, 0) if sys.platform == "win32" else struct.pack("ii", 1, 0)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.setblocking(False)
        task = asyncio.create_task(serve(listener))
        try:
            yield Origin("http", "127.0.0.1", listener.getsockname()[1])
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def _tunnel_then_stall(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        await _connect_head(reader)
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        # Swallow the TLS ClientHello and never answer, until the client gives up.
        while await reader.read(4096):
            pass
    except ConnectionError:
        pass
    finally:
        writer.close()


@contextlib.asynccontextmanager
async def _proxy(handle: Callable[..., Any]) -> AsyncIterator[Origin]:
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield Origin("http", "127.0.0.1", server.sockets[0].getsockname()[1])
    finally:
        server.close()
        server.close_clients()
        await server.wait_closed()


def _profile(provider: str) -> ModelProfile:
    return ModelProfile(
        id="p",
        name="p",
        provider=provider,
        model_id="test-model",
        api_key="sk-test",
        base_url=f"https://{API_HOST}" if provider == "anthropic" else f"https://{API_HOST}/v1",
        http_max_retries=0,
        http_connect_timeout=0.2,
    )


async def _fail_through(proxy: Origin, provider: str, monkeypatch: pytest.MonkeyPatch) -> BaseException:
    monkeypatch.setenv("HTTPS_PROXY", f"{proxy.scheme}://{proxy.host}:{proxy.port}")
    stack = await create_client(_profile(provider))
    raw = stack.inner.inner
    try:
        if provider == "anthropic":
            await raw.sdk_client.messages.create(model="test-model", max_tokens=8, messages=_CHAT)
        else:
            await raw.sdk_client.chat.completions.create(model="test-model", messages=_CHAT)
    except (openai.APIError, anthropic.APIError, httpx.HTTPError) as exc:
        return exc
    finally:
        await stack.aclose()
    raise AssertionError("the request did not fail")


@pytest.fixture
def proxy_env(monkeypatch: pytest.MonkeyPatch, clear_proxy_env: Callable[[], None]) -> pytest.MonkeyPatch:
    clear_proxy_env()
    return monkeypatch


_PROVIDERS = pytest.mark.parametrize("provider", ["openai", "anthropic"])


@_PROVIDERS
@pytest.mark.parametrize(
    ("serve_proxy", "kind", "retryable"),
    [
        pytest.param(
            partial(_proxy, _answer_then_close(b"HTTP/1.1 502 Bad Gateway")),
            ErrorKind.PROXY_REJECTED,
            True,
            id="connect-502",
        ),
        pytest.param(
            partial(
                _proxy, _answer_then_close(b"HTTP/1.1 407 Proxy Authentication Required", b"Proxy-Authenticate: Basic")
            ),
            ErrorKind.PROXY_AUTH_FAILED,
            False,
            id="connect-407",
        ),
        # A hang-up mid-handshake failed the connect; a reset is a lost connection in any phase.
        pytest.param(partial(_proxy, _tunnel_then_close), ErrorKind.CONNECTION_FAILED, True, id="tunnel-then-close"),
        pytest.param(_resetting_proxy, ErrorKind.CONNECTION_LOST, True, id="tunnel-then-reset"),
        pytest.param(partial(_proxy, _tunnel_then_stall), ErrorKind.CONNECT_TIMEOUT, True, id="tunnel-then-stall"),
    ],
)
async def test_proxy_answers_are_not_first_hop_evidence(
    provider: str,
    serve_proxy: Callable[[], contextlib.AbstractAsyncContextManager[Origin]],
    kind: ErrorKind,
    retryable: bool,
    proxy_env: pytest.MonkeyPatch,
) -> None:
    async with serve_proxy() as proxy:
        exc = await _fail_through(proxy, provider, proxy_env)

    result = classify_error(exc)
    assert (result.kind, result.retryable, result.failed_at_first_hop) == (kind, retryable, False)
    assert result.route is not None
    assert (result.route.target, result.route.proxy) == (_TARGET, proxy)


@_PROVIDERS
async def test_a_proxy_that_refuses_the_connection_is_the_failed_first_hop(
    provider: str, proxy_env: pytest.MonkeyPatch
) -> None:
    # Injected: macOS drops a SYN to a closed loopback port instead of refusing it.
    proxy = Origin("http", _UNREACHABLE_PROXY_HOST, 3128)
    with network_faults() as faults:
        faults.resolve_to(_UNREACHABLE_PROXY_HOST, INJECTED_V4)
        faults.refuse(INJECTED_V4)
        exc = await _fail_through(proxy, provider, proxy_env)

    result = classify_error(exc)
    assert (result.kind, result.retryable, result.failed_at_first_hop) == (ErrorKind.CONNECTION_REFUSED, True, True)
    assert result.route is not None
    assert (result.route.target, result.route.proxy, result.route.first_hop) == (_TARGET, proxy, proxy)
