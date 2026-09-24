# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""TLS identity adapter for numeric CONNECT targets in pinned httpcore 1.0.9.

Its proxy tunnel does not honor sni_hostname. Wrap the network stream so target
TLS always checks the logical hostname, while TLS to an HTTPS proxy checks the
proxy's own hostname. No certificate verification is disabled.
"""

from __future__ import annotations

import ssl
from collections.abc import Iterable
from contextvars import ContextVar
from typing import Any

import httpcore

tls_target: ContextVar[tuple[str, str, bool] | None] = ContextVar("web_tls_target", default=None)


class IdentityStream(httpcore.AsyncNetworkStream):
    def __init__(self, inner: httpcore.AsyncNetworkStream, target: tuple[str, str, bool]) -> None:
        self.inner = inner
        self.target = target

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return await self.inner.read(max_bytes, timeout)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        await self.inner.write(buffer, timeout)

    async def aclose(self) -> None:
        await self.inner.aclose()

    async def start_tls(
        self, ssl_context: ssl.SSLContext, server_hostname: str | None = None, timeout: float | None = None
    ) -> httpcore.AsyncNetworkStream:
        numeric, logical, proxy_tls = self.target
        hostname = logical if not proxy_tls and server_hostname == numeric else server_hostname
        stream = await self.inner.start_tls(ssl_context, hostname, timeout)
        return IdentityStream(stream, (numeric, logical, False))

    def get_extra_info(self, info: str) -> Any:
        return self.inner.get_extra_info(info)


class IdentityBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, inner: httpcore.AsyncNetworkBackend) -> None:
        self.inner = inner

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[tuple[int, int, int] | tuple[int, int, bytes | bytearray] | tuple[int, int, None, int]]
        | None = None,
    ) -> httpcore.AsyncNetworkStream:
        stream = await self.inner.connect_tcp(host, port, timeout, local_address, socket_options)
        target = tls_target.get()
        return IdentityStream(stream, target) if target is not None else stream

    async def sleep(self, seconds: float) -> None:
        await self.inner.sleep(seconds)
