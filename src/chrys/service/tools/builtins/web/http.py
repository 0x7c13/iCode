# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared bounded HTTP egress with validated numeric TCP targets and original TLS SNI.

Clients ignore ambient proxy variables. The transport retains environment CA
configuration, but only uses an explicitly configured proxy. By default a proxy
receives the validated numeric target (including CONNECT), never an unvalidated
name. With ``proxy_dns="remote"`` the proxy resolves the name instead, and so
owns the final-address check; the transport still blocks the addresses it can
see without DNS (IP literals and ``localhost``) and verifies the target's TLS
identity as usual.
Each execution owns its transport/client; the runtime owns only the shared gate.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import zlib
from collections.abc import Callable
from contextlib import suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpcore
import httpx

from chrys.foundation.errors import is_retryable
from chrys.foundation.net.address_class import is_restricted_address, literal_address, names_this_host
from chrys.foundation.net.url import normalize_origin, normalize_url, origin
from chrys.service.tools.builtins.web.tls_identity import IdentityBackend, tls_target

BODY_LIMIT = 2 * 1024 * 1024
# The connect timeout for every address but the last: an address that does not
# answer (a black-holed IPv6 route) must leave the deadline to the next one.
FALLBACK_CONNECT_SECONDS = 3.0
# RFC 2544 benchmarking space, which fake-IP proxies (Clash, sing-box, Surge)
# hand out as stand-in answers for every name they intercept.
_FAKE_IP_RANGE = ipaddress.ip_network("198.18.0.0/15")
FAKE_IP_HINT = (
    "the address is in a range fake-IP proxies use; with such a proxy, set an explicit proxy URL "
    "and let the proxy resolve names (tools.web_egress.proxy_dns: remote)"
)
_private_http: ContextVar[bool] = ContextVar("private_web_http", default=False)


class _WebLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _private_http.get()


_log_filter = _WebLogFilter()
for _logger_name in (
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
):
    logging.getLogger(_logger_name).addFilter(_log_filter)


class _PrivateWebClient(httpx.AsyncClient):
    """Suppress SDK request/header logging only within a web execution context."""

    _log_token: Token[bool] | None = None

    async def __aenter__(self):
        self._log_token = _private_http.set(True)
        try:
            return await super().__aenter__()
        except BaseException:
            _private_http.reset(self._log_token)
            self._log_token = None
            raise

    async def __aexit__(self, exc_type=None, exc_value=None, traceback=None):
        try:
            return await super().__aexit__(exc_type, exc_value, traceback)
        finally:
            if self._log_token is not None:
                _private_http.reset(self._log_token)
                self._log_token = None


class WebError(Exception):
    """Safe, bounded diagnostic; never includes request secrets or response bodies."""

    def __init__(self, code: str, *, retryable: bool = False, status: int | None = None, hint: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.status = status
        self.hint = hint

    def text(self) -> str:
        """The model-facing diagnostic: the code, the HTTP status when there is one, and any hint."""
        text = f"{self.code} (HTTP {self.status})" if self.status is not None else self.code
        return f"{text}: {self.hint}" if self.hint else text


@dataclass(frozen=True)
class DestinationPolicy:
    """A single capability's immutable origin grants."""

    private_origins: frozenset[str] = frozenset()
    http_origins: frozenset[str] = frozenset()
    allowed_origins: frozenset[str] = frozenset()
    denied_origins: frozenset[str] = frozenset()

    def check(self, url: str) -> str:
        """Apply explicit deny before scheme and optional allow rules."""
        target = origin(url)
        if target in self.denied_origins:
            raise WebError("origin_denied")
        if urlsplit(url).scheme != "https" and target not in self.http_origins:
            raise WebError("http_not_allowed")
        if self.allowed_origins and target not in self.allowed_origins:
            raise WebError("origin_not_allowed")
        return target


async def resolve_addresses(host: str, port: int) -> tuple[str, ...]:
    """Resolve once, preserving every A/AAAA address for the classification gate."""
    try:
        return (str(ipaddress.ip_address(host)),)
    except ValueError:
        pass
    try:
        records = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as err:
        raise WebError("dns_failure") from err
    addresses = tuple(dict.fromkeys(str(record[4][0]) for record in records))
    if not addresses:
        raise WebError("dns_failure")
    return addresses


class ValidatedAsyncTransport(httpx.AsyncBaseTransport):
    """Pin each request to a classified IP while preserving Host and TLS identity.

    httpcore 1.0.9's sni_hostname extension controls both SNI and certificate
    hostname checking. Connection pooling is local to an execution; the numeric
    destination is its pool key. Redirects are never handled by this transport.
    With a proxy and ``proxy_dns="remote"``, requests keep their hostname and the
    proxy resolves it.
    """

    def __init__(self, policy: DestinationPolicy, *, proxy_url: str = "", proxy_dns: str = "local") -> None:
        self.policy = policy
        if proxy_url:
            proxy_url = normalize_origin(proxy_url)
        self.proxy_url = proxy_url
        self.remote_dns = bool(proxy_url) and proxy_dns == "remote"
        self.inner = httpx.AsyncHTTPTransport(
            proxy=proxy_url or None,
            trust_env=True,
            retries=0,
            limits=httpx.Limits(max_connections=3, max_keepalive_connections=0),
        )
        # httpx has no public network_backend argument. These two private pool
        # members are pinned by connection/TLS tests and the direct dependency.
        pool = self.inner._pool
        if not isinstance(pool, httpcore.AsyncConnectionPool):
            raise TypeError("The httpx transport no longer exposes the expected httpcore connection pool.")
        pool._network_backend = IdentityBackend(pool._network_backend)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            url = normalize_url(str(request.url), limit=16384)
        except ValueError as err:
            # httpx percent-encodes the URL, so one the tool accepted can still
            # outgrow the limit here.
            raise WebError("invalid_url") from err
        target = self.policy.check(url)
        # normalize_url IDNA-encodes the authority: DNS resolution and TLS SNI
        # need the A-label form, while request.url.host may keep the U-label.
        host = urlsplit(url).hostname
        if host is None:
            raise WebError("invalid_url")
        if self.remote_dns:
            return await self._send_by_name(request, host, granted=target in self.policy.private_origins)
        addresses = await resolve_addresses(host, request.url.port or (443 if request.url.scheme == "https" else 80))
        if target not in self.policy.private_origins and any(is_restricted_address(ip) for ip in addresses):
            fake_ip = any(ipaddress.ip_address(ip) in _FAKE_IP_RANGE for ip in addresses)
            raise WebError("private_address_blocked", hint=FAKE_IP_HINT if fake_ip else "")
        if self.proxy_url:
            # httpcore writes the proxy an IPv6 target without brackets
            # (``CONNECT 2001:db8::1:443``), which no proxy can parse.
            addresses = tuple(ip for ip in addresses if ipaddress.ip_address(ip).version == 4)
            if not addresses:
                raise WebError("proxy_ipv6_unsupported")
        # Build a separate request: user-visible URL and cookies remain attached
        # to the logical origin, while the connection sees only the pinned IP.
        extensions = dict(request.extensions)
        extensions["sni_hostname"] = urlsplit(self.proxy_url).hostname if self.proxy_url else host
        for index, address in enumerate(addresses):
            pinned = httpx.Request(
                request.method,
                request.url.copy_with(host=address),
                headers=request.headers,
                stream=request.stream,
                extensions=extensions if index + 1 == len(addresses) else _capped_connect(extensions),
            )
            token = tls_target.set((address, host, self.proxy_url.startswith("https:")))
            try:
                return await self.inner.handle_async_request(pinned)
            except httpx.TransportError as err:
                retryable = is_retryable(err)
                # Only connection establishment failures precede transmission
                # of the application request. Never replay a body after a
                # write/read/protocol failure, or bypass certificate failures.
                # The caller's attempt/execution deadlines enclose this loop.
                if (
                    isinstance(err, httpx.ConnectError | httpx.ConnectTimeout)
                    and retryable
                    and index + 1 < len(addresses)
                ):
                    continue
                raise WebError("connection_failed", retryable=retryable) from err
            finally:
                tls_target.reset(token)
        raise WebError("dns_failure")

    async def _send_by_name(self, request: httpx.Request, host: str, *, granted: bool) -> httpx.Response:
        """Hand the name to the proxy, blocking only the destinations visible without DNS."""
        address = literal_address(host)
        if address is not None and ipaddress.ip_address(address).version == 6:
            raise WebError("proxy_ipv6_unsupported")
        if not granted and (names_this_host(host) or (address is not None and is_restricted_address(address))):
            raise WebError("private_address_blocked")
        # Unchanged, the request tunnels to its own hostname: httpcore checks the
        # target certificate against it and the proxy's against the proxy host.
        try:
            return await self.inner.handle_async_request(request)
        except httpx.TransportError as err:
            raise WebError("connection_failed", retryable=is_retryable(err)) from err

    async def aclose(self) -> None:
        await self.inner.aclose()


def _capped_connect(extensions: dict) -> dict:
    timeout = dict(extensions.get("timeout") or {})
    connect = timeout.get("connect")
    timeout["connect"] = FALLBACK_CONNECT_SECONDS if connect is None else min(connect, FALLBACK_CONNECT_SECONDS)
    return {**extensions, "timeout": timeout}


class WebEgressRuntime:
    """One build's shared concurrency limit; never a global connection pool."""

    def __init__(self, *, proxy_url: str = "", proxy_dns: str = "local") -> None:
        self.proxy_url = normalize_origin(proxy_url) if proxy_url else ""
        self.proxy_dns = proxy_dns
        self.gate = asyncio.Semaphore(3)

    def client(self, policy: DestinationPolicy, *, timeout: float) -> httpx.AsyncClient:
        """Create an execution-owned client without ambient auth/proxy state.

        ``timeout`` bounds each network operation; the caller's deadline still
        bounds the whole execution.
        """
        return _PrivateWebClient(
            transport=ValidatedAsyncTransport(policy, proxy_url=self.proxy_url, proxy_dns=self.proxy_dns),
            trust_env=False,
            follow_redirects=False,
            timeout=timeout,
            headers={"User-Agent": "Chrys-Web/1.0"},
        )


async def run_off_loop[T](func: Callable[..., T], *args: object) -> T:
    """Run bounded CPU work in the default executor, keeping ownership until it ends.

    The job cannot be interrupted, so a cancellation first waits for it to
    finish, whatever its outcome, and then propagates.
    """
    job = asyncio.get_running_loop().run_in_executor(None, func, *args)
    try:
        return await asyncio.shield(job)
    except asyncio.CancelledError:
        with suppress(Exception):
            await asyncio.shield(job)
        raise


async def bounded_body(response: httpx.Response, *, decompress: bool) -> bytes:
    """Bound both wire bytes and zlib output before allocating the decoded body."""
    try:
        if int(response.headers.get("content-length", "0")) > BODY_LIMIT:
            raise WebError("response_too_large")
    except ValueError as err:
        raise WebError("protocol_error") from err
    encoding = response.headers.get("content-encoding", "identity").strip().lower()
    # Requests advertise gzip alone: deflate is ambiguous between zlib-wrapped and raw streams.
    if encoding not in ({"identity", "gzip"} if decompress else {"identity"}):
        raise WebError("unsupported_content_encoding")
    decoder = None if encoding == "identity" else zlib.decompressobj(31)
    data = bytearray()
    wire_size = 0
    try:
        async for chunk in response.aiter_raw():
            wire_size += len(chunk)
            if wire_size > BODY_LIMIT:
                raise WebError("response_too_large")
            decoded = chunk if decoder is None else decoder.decompress(chunk, BODY_LIMIT - len(data) + 1)
            data.extend(decoded)
            if len(data) > BODY_LIMIT or (decoder is not None and decoder.unconsumed_tail):
                raise WebError("response_too_large")
        if decoder is not None and (not decoder.eof or decoder.unused_data):
            raise WebError("protocol_error")
    except zlib.error as err:
        raise WebError("protocol_error") from err
    return bytes(data)


def check_status(response: httpx.Response) -> None:
    """Never disclose non-success response bodies."""
    status = response.status_code
    if not 200 <= status < 300:
        raise WebError(
            "http_5xx" if status >= 500 else "http_4xx" if status >= 400 else "redirect_not_allowed",
            retryable=status in {408, 429} or 500 <= status <= 599,
            status=status,
        )
