# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Where names are resolved: locally with a pinned address, or by the proxy with the checks DNS cannot bypass."""

from __future__ import annotations

import ssl
from dataclasses import replace
from unittest.mock import patch

import httpx
import pytest

from chrys.foundation.config.settings import Settings
from chrys.service.tools.builtins.web.build import build_web_tools
from chrys.service.tools.builtins.web.http import FAKE_IP_HINT, DestinationPolicy, ValidatedAsyncTransport, WebError
from chrys.service.tools.builtins.web.tls_identity import tls_target
from tests.service.tools.web._support import Chunks, socket_network

_PROXY = "http://proxy.invalid:8080"
_RESOLVE = "chrys.service.tools.builtins.web.http.resolve_addresses"


def _remote(policy: DestinationPolicy | None = None) -> ValidatedAsyncTransport:
    return ValidatedAsyncTransport(policy or DestinationPolicy(), proxy_url=_PROXY, proxy_dns="remote")


async def test_remote_dns_hands_the_proxy_the_name_and_resolves_nothing():
    transport = _remote()
    sent: list[tuple[httpx.Request, object]] = []

    async def send(request: httpx.Request) -> httpx.Response:
        sent.append((request, tls_target.get()))
        return httpx.Response(200)

    try:
        with (
            patch(_RESOLVE, autospec=True, side_effect=AssertionError("resolved locally")),
            patch.object(transport.inner, "handle_async_request", autospec=True, side_effect=send),
        ):
            response = await transport.handle_async_request(httpx.Request("GET", "https://example.com/a?q=1"))
    finally:
        await transport.aclose()
    assert response.status_code == 200
    [(request, identity)] = sent
    assert str(request.url) == "https://example.com/a?q=1"
    # httpcore verifies the target certificate against the URL host, the proxy's against the proxy host.
    assert "sni_hostname" not in request.extensions
    assert identity is None


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/",
        "https://10.0.0.1/",
        "https://198.18.0.1/",
        # The proxy's resolver would read these numeric forms as 127.0.0.1.
        "https://2130706433/",
        "https://0x7f.1/",
        "https://127.1/",
        "https://localhost/",
        "https://LOCALHOST./",
        "https://api.localhost/",
    ],
)
async def test_remote_dns_still_blocks_what_is_private_without_a_lookup(url):
    transport = _remote()
    try:
        with (
            patch(_RESOLVE, autospec=True, side_effect=AssertionError("resolved locally")),
            patch.object(transport.inner, "handle_async_request", autospec=True) as send,
            pytest.raises(WebError, match=r"^private_address_blocked$") as blocked,
        ):
            await transport.handle_async_request(httpx.Request("GET", url))
    finally:
        await transport.aclose()
    send.assert_not_called()
    # The fake-IP hint recommends remote DNS; repeating it here would send the user in a circle.
    assert blocked.value.hint == ""


@pytest.mark.parametrize("url", ["https://[::1]/", "https://[2606:4700:4700::1111]/"])
async def test_remote_dns_cannot_hand_the_proxy_an_ipv6_literal(url):
    transport = _remote()
    try:
        with (
            patch.object(transport.inner, "handle_async_request", autospec=True) as send,
            pytest.raises(WebError, match=r"^proxy_ipv6_unsupported$"),
        ):
            await transport.handle_async_request(httpx.Request("GET", url))
    finally:
        await transport.aclose()
    send.assert_not_called()


async def test_a_private_origin_grant_lets_a_local_name_through_the_proxy():
    transport = _remote(DestinationPolicy(private_origins=frozenset({"https://localhost:8443"})))
    try:
        with patch.object(
            transport.inner, "handle_async_request", autospec=True, return_value=httpx.Response(204)
        ) as send:
            response = await transport.handle_async_request(httpx.Request("GET", "https://localhost:8443/"))
    finally:
        await transport.aclose()
    assert response.status_code == 204
    assert send.call_args.args[0].url.host == "localhost"


def _certificate_rejected() -> httpx.ConnectError:
    error = httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    error.__cause__ = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    return error


@pytest.mark.parametrize(
    "error,retryable", [(httpx.ConnectError("refused"), True), (_certificate_rejected(), False)], ids=["refused", "tls"]
)
async def test_a_proxy_transport_failure_is_a_connection_failure(error, retryable):
    transport = _remote()
    try:
        with (
            patch.object(transport.inner, "handle_async_request", autospec=True, side_effect=error),
            pytest.raises(WebError, match=r"^connection_failed$") as failure,
        ):
            await transport.handle_async_request(httpx.Request("GET", "https://example.com/"))
    finally:
        await transport.aclose()
    assert failure.value.retryable is retryable


async def test_remote_dns_without_a_proxy_still_resolves_locally():
    transport = ValidatedAsyncTransport(DestinationPolicy(), proxy_dns="remote")
    try:
        with (
            patch(_RESOLVE, autospec=True, return_value=("10.0.0.1",)) as resolve,
            pytest.raises(WebError, match="private_address_blocked"),
        ):
            await transport.handle_async_request(httpx.Request("GET", "https://example.com/"))
    finally:
        await transport.aclose()
    resolve.assert_awaited_once()


@pytest.mark.parametrize(
    "addresses,hinted",
    [(("198.18.0.7",), True), (("198.19.255.1",), True), (("10.0.0.1",), False), (("198.20.0.1", "10.0.0.1"), False)],
)
async def test_a_blocked_fake_ip_answer_names_the_remote_dns_setting(addresses, hinted):
    transport = ValidatedAsyncTransport(DestinationPolicy(), proxy_url=_PROXY)
    try:
        with (
            patch(_RESOLVE, autospec=True, return_value=addresses),
            pytest.raises(WebError, match=r"^private_address_blocked$") as blocked,
        ):
            await transport.handle_async_request(httpx.Request("GET", "https://example.com/"))
    finally:
        await transport.aclose()
    # The code stays the same either way: the hint explains, it never lets the request through.
    assert blocked.value.code == "private_address_blocked"
    expected = f"private_address_blocked: {FAKE_IP_HINT}" if hinted else "private_address_blocked"
    assert blocked.value.text() == expected


async def test_the_setting_reaches_the_clients_the_build_creates():
    settings = replace(Settings(), web_egress_proxy_url=_PROXY, web_egress_proxy_dns="remote")
    fetch = build_web_tools(["web_fetch"], settings, None, None).fetch
    assert fetch is not None
    hosts: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, stream=Chunks(b"page body"))

    # No DNS answers exist: a local lookup would fail the fetch.
    with socket_network(handle, {}) as network:
        result = await fetch.web_fetch("https://example.com/page", "the body")
    assert "page body" in result
    assert hosts == ["example.com"]
    network.resolve.assert_not_called()
