# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Validated address fallback without replaying requests after transmission."""

from __future__ import annotations

import asyncio
import errno
import socket
import ssl
from unittest.mock import patch

import httpx
import pytest

from chrys.service.tools.builtins.web.http import (
    FALLBACK_CONNECT_SECONDS,
    DestinationPolicy,
    ValidatedAsyncTransport,
    WebError,
)
from chrys.service.tools.builtins.web.tls_identity import tls_target
from tests.service.tools.web._support import Chunks


async def test_real_connection_reaches_second_validated_address():
    """Keep the refusing endpoint bound so the failure cannot race port reuse."""
    tasks = set()
    received = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        tasks.add(task)
        try:
            received.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    target = f"http://fallback.invalid:{port}"
    grants = frozenset({target})
    policy = DestinationPolicy(http_origins=grants, private_origins=grants)
    try:
        with (
            socket.socket() as refusing,
            patch(
                "chrys.service.tools.builtins.web.http.resolve_addresses",
                autospec=True,
                return_value=("127.0.0.2", "127.0.0.1"),
            ) as resolve,
        ):
            try:
                refusing.bind(("127.0.0.2", port))
            except OSError as err:
                if err.errno == errno.EADDRNOTAVAIL:
                    pytest.skip("This host does not provide a second loopback address")
                raise
            async with httpx.AsyncClient(transport=ValidatedAsyncTransport(policy), trust_env=False) as client:
                response = await client.get(target)
            assert response.text == "ok"
            resolve.assert_awaited_once_with("fallback.invalid", port)
            assert len(received) == 1
            assert f"Host: fallback.invalid:{port}".encode() in received[0]
    finally:
        server.close()
        await server.wait_closed()
        if tasks:
            await asyncio.gather(*tasks)


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ConnectTimeout])
@pytest.mark.parametrize("proxy_url", ["", "http://proxy.invalid:8080", "https://proxy.invalid:8443"])
async def test_connection_fallback_preserves_post_body_and_tls_identity(error_type, proxy_url):
    transport = ValidatedAsyncTransport(DestinationPolicy(), proxy_url=proxy_url)
    seen = []
    body_reads = []

    async def body():
        body_reads.append(True)
        yield b"search query"

    async def send(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers["Host"], request.extensions["sni_hostname"], tls_target.get()))
        if len(seen) == 1:
            raise error_type("unreachable")
        assert await request.aread() == b"search query"
        return httpx.Response(200, stream=Chunks(b"ok"))

    try:
        with (
            patch(
                "chrys.service.tools.builtins.web.http.resolve_addresses",
                autospec=True,
                return_value=("2606:4700:4700::1111", "1.0.0.1", "1.1.1.1"),
            ) as resolve,
            patch.object(transport.inner, "handle_async_request", autospec=True, side_effect=send),
        ):
            async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
                response = await client.post("https://example.com/search", content=body())
        assert response.text == "ok"
        assert body_reads == [True]
        assert resolve.await_count == 1
        sni = "proxy.invalid" if proxy_url else "example.com"
        # Through a proxy the IPv6 address is never dialed: httpcore would write it unbracketed.
        dialed = ("1.0.0.1", "1.1.1.1") if proxy_url else ("2606:4700:4700::1111", "1.0.0.1")
        assert seen == [
            (address, "example.com", sni, (address, "example.com", proxy_url.startswith("https:")))
            for address in dialed
        ]
        assert tls_target.get() is None
    finally:
        await transport.aclose()


async def test_a_proxy_is_never_handed_an_ipv6_only_destination():
    transport = ValidatedAsyncTransport(DestinationPolicy(), proxy_url="http://proxy.invalid:8080")
    try:
        with (
            patch(
                "chrys.service.tools.builtins.web.http.resolve_addresses",
                autospec=True,
                return_value=("2606:4700:4700::1111",),
            ),
            patch.object(transport.inner, "handle_async_request", autospec=True) as send,
            pytest.raises(WebError, match=r"^proxy_ipv6_unsupported$") as error,
        ):
            await transport.handle_async_request(httpx.Request("GET", "https://example.com"))
        assert error.value.retryable is False
        send.assert_not_called()
    finally:
        await transport.aclose()


async def test_only_the_last_address_waits_the_full_connect_timeout():
    transport = ValidatedAsyncTransport(DestinationPolicy())
    connect_timeouts = []

    async def send(request: httpx.Request) -> httpx.Response:
        connect_timeouts.append(request.extensions["timeout"]["connect"])
        raise httpx.ConnectTimeout("no answer")

    try:
        with (
            patch(
                "chrys.service.tools.builtins.web.http.resolve_addresses",
                autospec=True,
                return_value=("2606:4700:4700::1111", "1.0.0.1", "1.1.1.1"),
            ),
            patch.object(transport.inner, "handle_async_request", autospec=True, side_effect=send),
            pytest.raises(WebError, match=r"^connection_failed$"),
        ):
            async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=30) as client:
                await client.get("https://example.com")
        assert connect_timeouts == [FALLBACK_CONNECT_SECONDS, FALLBACK_CONNECT_SECONDS, 30]
    finally:
        await transport.aclose()


@pytest.mark.parametrize(
    "error_type",
    [httpx.ReadError, httpx.WriteError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.ProxyError],
)
async def test_post_connection_errors_do_not_try_another_address(error_type):
    transport = ValidatedAsyncTransport(DestinationPolicy())
    try:
        with (
            patch(
                "chrys.service.tools.builtins.web.http.resolve_addresses",
                autospec=True,
                return_value=("1.1.1.1", "8.8.8.8"),
            ),
            patch.object(
                transport.inner, "handle_async_request", autospec=True, side_effect=error_type("failed")
            ) as send,
            pytest.raises(WebError, match="connection_failed"),
        ):
            await transport.handle_async_request(httpx.Request("POST", "https://example.com", content=b"query"))
        assert send.await_count == 1
        assert tls_target.get() is None
    finally:
        await transport.aclose()


async def test_certificate_failure_does_not_try_another_address():
    transport = ValidatedAsyncTransport(DestinationPolicy())

    async def send(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("certificate failed") from ssl.SSLCertVerificationError("wrong hostname")

    try:
        with (
            patch(
                "chrys.service.tools.builtins.web.http.resolve_addresses",
                autospec=True,
                return_value=("1.1.1.1", "8.8.8.8"),
            ),
            patch.object(transport.inner, "handle_async_request", autospec=True, side_effect=send) as mocked,
            pytest.raises(WebError, match="connection_failed") as error,
        ):
            await transport.handle_async_request(httpx.Request("GET", "https://example.com"))
        assert error.value.retryable is False
        assert mocked.await_count == 1
    finally:
        await transport.aclose()


async def test_all_addresses_fail_once_and_error_stays_safe():
    transport = ValidatedAsyncTransport(DestinationPolicy())
    try:
        with (
            patch(
                "chrys.service.tools.builtins.web.http.resolve_addresses",
                autospec=True,
                return_value=("1.1.1.1", "8.8.8.8"),
            ) as resolve,
            patch.object(
                transport.inner, "handle_async_request", autospec=True, side_effect=httpx.ConnectError("private detail")
            ) as send,
            pytest.raises(WebError, match=r"^connection_failed$") as error,
        ):
            await transport.handle_async_request(httpx.Request("GET", "https://example.com"))
        assert error.value.retryable is True
        assert [call.args[0].url.host for call in send.await_args_list] == ["1.1.1.1", "8.8.8.8"]
        assert resolve.await_count == 1
        assert tls_target.get() is None
    finally:
        await transport.aclose()


async def test_outer_deadline_stops_address_fallback():
    transport = ValidatedAsyncTransport(DestinationPolicy())
    seen = []
    deadline = asyncio.timeout(None)

    async def send(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if len(seen) == 1:
            raise httpx.ConnectError("unreachable")
        deadline.reschedule(asyncio.get_running_loop().time())
        await asyncio.get_running_loop().create_future()
        raise AssertionError("deadline must cancel the connection")

    try:
        with (
            patch(
                "chrys.service.tools.builtins.web.http.resolve_addresses",
                autospec=True,
                return_value=("1.1.1.1", "8.8.8.8", "8.8.4.4"),
            ),
            patch.object(transport.inner, "handle_async_request", autospec=True, side_effect=send),
            pytest.raises(TimeoutError),
        ):
            async with deadline:
                await transport.handle_async_request(httpx.Request("GET", "https://example.com"))
        assert seen == ["1.1.1.1", "8.8.8.8"]
        assert tls_target.get() is None
    finally:
        await transport.aclose()
