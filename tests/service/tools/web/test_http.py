# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Connection-layer DNS pinning and bounded raw decompression contracts."""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import socket
import threading
import zlib
from unittest.mock import patch

import httpx
import pytest

from chrys.foundation.net.url import normalize_url
from chrys.service.tools.builtins.web.http import (
    BODY_LIMIT,
    DestinationPolicy,
    ValidatedAsyncTransport,
    WebError,
    bounded_body,
    resolve_addresses,
    run_off_loop,
)
from tests.service.tools.web._support import Chunks
from tests.support.waiting import wait_for, wait_until


@pytest.mark.parametrize(
    "addresses", [("127.0.0.1",), ("8.8.8.8", "10.0.0.1"), ("::ffff:127.0.0.1",), ("100.100.100.200",)]
)
async def test_all_dns_addresses_checked_before_transport(addresses):
    transport = ValidatedAsyncTransport(DestinationPolicy())
    try:
        with (
            patch("chrys.service.tools.builtins.web.http.resolve_addresses", autospec=True, return_value=addresses),
            patch.object(transport.inner, "handle_async_request", autospec=True) as send,
        ):
            with pytest.raises(WebError, match="private_address_blocked"):
                await transport.handle_async_request(httpx.Request("GET", "https://example.com"))
            send.assert_not_called()
    finally:
        await transport.aclose()


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("93.184.215.14", "93.184.215.14"),
        ("10.0.0.1", "10.0.0.1"),
        ("2606:2800:0220:0001:0000:0000:0000:0001", "2606:2800:220:1::1"),
    ],
)
async def test_an_address_literal_is_never_looked_up(host, expected):
    loop = asyncio.get_running_loop()
    with patch.object(loop, "getaddrinfo", autospec=True, side_effect=AssertionError("no lookup expected")) as lookup:
        assert await resolve_addresses(host, 443) == (expected,)
    lookup.assert_not_called()


def _record(address: str) -> tuple:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sockaddr = (address, 443, 0, 0) if family == socket.AF_INET6 else (address, 443)
    return (family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)


async def test_a_lookup_keeps_every_distinct_address_in_order():
    loop = asyncio.get_running_loop()
    records = [_record("93.184.215.14"), _record("2606:2800:220:1::1"), _record("93.184.215.14")]
    with patch.object(loop, "getaddrinfo", autospec=True, return_value=records) as lookup:
        assert await resolve_addresses("example.com", 443) == ("93.184.215.14", "2606:2800:220:1::1")
    lookup.assert_awaited_once_with("example.com", 443, type=socket.SOCK_STREAM)


@pytest.mark.parametrize(
    "outcome",
    [
        pytest.param(socket.gaierror(socket.EAI_NONAME, "Name or service not known"), id="lookup-error"),
        pytest.param([], id="no-records"),
    ],
)
async def test_a_failed_lookup_is_a_dns_failure(outcome):
    loop = asyncio.get_running_loop()
    behavior = {"side_effect": outcome} if isinstance(outcome, BaseException) else {"return_value": outcome}
    with (
        patch.object(loop, "getaddrinfo", autospec=True, **behavior) as lookup,
        pytest.raises(WebError, match="dns_failure") as failure,
    ):
        await resolve_addresses("missing.example", 443)
    lookup.assert_awaited_once_with("missing.example", 443, type=socket.SOCK_STREAM)
    assert failure.value.retryable is False


async def test_a_private_address_literal_is_blocked_without_a_lookup():
    """Classification applies to literal hosts too, through the real resolver."""
    transport = ValidatedAsyncTransport(DestinationPolicy())
    loop = asyncio.get_running_loop()
    try:
        with (
            patch.object(loop, "getaddrinfo", autospec=True, side_effect=AssertionError("no lookup expected")),
            patch.object(transport.inner, "handle_async_request", autospec=True) as send,
        ):
            with pytest.raises(WebError, match="private_address_blocked"):
                await transport.handle_async_request(httpx.Request("GET", "https://10.0.0.1/admin"))
            send.assert_not_called()
    finally:
        await transport.aclose()


async def test_numeric_tcp_target_original_host_and_sni():
    transport = ValidatedAsyncTransport(DestinationPolicy())
    seen = []

    async def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, stream=Chunks(b"ok"))

    try:
        with (
            patch(
                "chrys.service.tools.builtins.web.http.resolve_addresses", autospec=True, return_value=("8.8.8.8",)
            ) as resolve,
            patch.object(transport.inner, "handle_async_request", autospec=True, side_effect=handle),
        ):
            await transport.handle_async_request(httpx.Request("GET", "https://example.com/path"))
        assert resolve.await_count == 1
        assert seen[0].url.host == "8.8.8.8"
        assert seen[0].headers["Host"] == "example.com"
        assert seen[0].extensions["sni_hostname"] == "example.com"
    finally:
        await transport.aclose()


async def test_real_connection_uses_pinned_literal():
    """A fake logical DNS name reaches a real loopback HTTP listener exactly once."""
    received = asyncio.get_running_loop().create_future()
    tasks = set()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        tasks.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            received.set_result(headers)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    target = f"http://pin-test.invalid:{port}"
    policy = DestinationPolicy(http_origins=frozenset({target}), private_origins=frozenset({target}))
    try:
        with patch(
            "chrys.service.tools.builtins.web.http.resolve_addresses", autospec=True, return_value=("127.0.0.1",)
        ) as resolve:
            async with httpx.AsyncClient(transport=ValidatedAsyncTransport(policy), trust_env=False) as client:
                result = await client.get(target)
        assert result.text == "ok"
        assert resolve.await_count == 1
        assert f"Host: pin-test.invalid:{port}".encode() in await received
    finally:
        server.close()
        await server.wait_closed()
        if tasks:
            await asyncio.gather(*tasks)


async def test_gzip_bomb_rejected_during_decompression():
    raw = gzip.compress(b"a" * (20 * 1024 * 1024))
    response = httpx.Response(200, headers={"content-encoding": "gzip"}, stream=Chunks(raw))
    try:
        with pytest.raises(WebError, match="response_too_large"):
            await bounded_body(response, decompress=True)
    finally:
        await response.aclose()


@pytest.mark.parametrize("encoding", ["deflate", "br"])
async def test_only_the_advertised_encoding_is_decoded(encoding):
    """Requests advertise gzip alone, so any other encoding is refused rather than guessed at."""
    response = httpx.Response(200, headers={"content-encoding": encoding}, stream=Chunks(zlib.compress(b"text")))
    try:
        with pytest.raises(WebError, match="unsupported_content_encoding"):
            await bounded_body(response, decompress=True)
    finally:
        await response.aclose()


@pytest.mark.parametrize("decompress", [False, True])
async def test_raw_size_bound(decompress):
    response = httpx.Response(200, stream=Chunks(b"a" * BODY_LIMIT, b"b"))
    try:
        with pytest.raises(WebError, match="response_too_large"):
            await bounded_body(response, decompress=decompress)
    finally:
        await response.aclose()


async def test_cancelled_off_loop_work_is_drained_and_the_cancel_still_propagates():
    """A job that fails after the cancel must not replace the cancellation with its error."""
    started = threading.Event()
    release = threading.Event()

    def job() -> None:
        started.set()
        release.wait(10)
        raise WebError("conversion_failed")

    task = asyncio.create_task(run_off_loop(job))
    try:
        await wait_for(started.is_set, description="the executor job started")
        task.cancel()
        # The job still owns its thread, so the caller keeps waiting for it.
        assert not await wait_until(task.done, timeout=0.2)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


def test_url_normalization_preserves_query_and_removes_fragment():
    assert normalize_url("https://EXAMPLE.com:443/a?b=2&a=1#frag") == "https://example.com/a?b=2&a=1"
    assert normalize_url("https://example.com。/") == "https://example.com/"
    for value in [
        "http://user:pass@example.com/",
        "file:///tmp/a",
        "https://example.com:0",
        "https://example.com\\@evil.com",
    ]:
        with pytest.raises(ValueError):
            normalize_url(value)


@pytest.mark.parametrize(
    "use_proxy,proxy_dns", [(False, "local"), (True, "local"), (True, "remote")], ids=["direct", "proxy", "proxy-dns"]
)
async def test_tls_checks_original_hostname_with_numeric_connection(tmp_path, monkeypatch, use_proxy, proxy_dns):
    """TLS verifies the logical hostname directly, through a numeric CONNECT tunnel, and through a named one.

    Directly, httpcore's ``sni_hostname`` extension carries the name. Its proxy
    tunnel ignores that extension, so only the transport's ``IdentityBackend``
    keeps the tunnelled TLS handshake on the logical name. With remote DNS the
    tunnel is opened to the name itself, and httpcore's own check applies.
    """
    import ssl
    from datetime import UTC, datetime, timedelta

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "tls-pin.invalid")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(days=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("tls-pin.invalid")]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(cert_path, key_path)
    sni = []
    server_context.set_servername_callback(lambda socket, name, context: sni.append(name))
    done = asyncio.get_running_loop().create_future()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
            if not done.done():
                done.set_result(None)

    server = await asyncio.start_server(serve, "127.0.0.1", 0, ssl=server_context)
    port = server.sockets[0].getsockname()[1]
    target = f"https://tls-pin.invalid:{port}"
    proxy_server = None
    proxy_tasks = set()
    connect_targets = []

    async def proxy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        proxy_tasks.add(task)
        remote_writer = None
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            connect_targets.append(header.split(b"\r\n", 1)[0])
            remote_reader, remote_writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await writer.drain()

            async def relay(source: asyncio.StreamReader, destination: asyncio.StreamWriter) -> None:
                try:
                    while data := await source.read(65536):
                        destination.write(data)
                        await destination.drain()
                except ConnectionError:
                    # A tunnel peer may reset after receiving the full HTTP response.
                    # The client response and SNI assertions still verify delivery.
                    pass
                finally:
                    # Preserve the reverse relay long enough to forward TLS close_notify.
                    # Closing the whole socket here races that relay on macOS.
                    if destination.can_write_eof():
                        with contextlib.suppress(ConnectionError):
                            destination.write_eof()

            async with asyncio.TaskGroup() as group:
                group.create_task(relay(reader, remote_writer))
                group.create_task(relay(remote_reader, writer))
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
            if remote_writer is not None:
                remote_writer.close()
                with contextlib.suppress(ConnectionError):
                    await remote_writer.wait_closed()
            proxy_tasks.discard(task)

    proxy_url = ""
    if use_proxy:
        proxy_server = await asyncio.start_server(proxy, "127.0.0.1", 0)
        proxy_url = f"http://127.0.0.1:{proxy_server.sockets[0].getsockname()[1]}"
    # The production transport keeps environment CA configuration (trust_env=True), so
    # the test CA arrives the way a corporate CA would, before the transport is built.
    # Nothing here rebuilds its inner transport or its network backend.
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_path))
    transport = ValidatedAsyncTransport(
        DestinationPolicy(private_origins=frozenset({target})), proxy_url=proxy_url, proxy_dns=proxy_dns
    )
    # With remote DNS the name never reaches a local resolver; the test proxy dials the server itself.
    resolved = {"side_effect": AssertionError("resolved locally")} if proxy_dns == "remote" else {}
    try:
        with patch(
            "chrys.service.tools.builtins.web.http.resolve_addresses",
            autospec=True,
            return_value=("127.0.0.1",),
            **resolved,
        ):
            async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
                result = await client.get(target)
        assert result.text == "ok"
        assert sni == ["tls-pin.invalid"]
        if use_proxy:
            tunnel_host = "tls-pin.invalid" if proxy_dns == "remote" else "127.0.0.1"
            assert connect_targets == [f"CONNECT {tunnel_host}:{port} HTTP/1.1".encode()]
        await done
    finally:
        await transport.aclose()
        server.close()
        await server.wait_closed()
        if proxy_server is not None:
            proxy_server.close()
            await proxy_server.wait_closed()
        if proxy_tasks:
            await asyncio.gather(*proxy_tasks)
