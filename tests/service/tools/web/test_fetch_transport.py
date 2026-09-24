# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""web_fetch through the real egress client: redirects, cookies, retries and deadlines.

Only DNS and the socket-level transport are replaced, so every hop still passes
through ``ValidatedAsyncTransport`` and its destination checks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import contextmanager
from unittest.mock import patch

import httpx

from chrys.service.tools.builtins.web.fetch.tool import WebFetchTools
from chrys.service.tools.builtins.web.http import DestinationPolicy, WebEgressRuntime
from chrys.service.tools.result_metadata import tool_result_metadata
from tests.service.tools.web._support import Chunks

_PUBLIC = {"example.com": "93.184.215.14", "www.example.com": "93.184.215.14"}

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def _page(text: str = "page body", status: int = 200, **headers: str) -> httpx.Response:
    return httpx.Response(status, headers={"Content-Type": "text/plain", **headers}, stream=Chunks(text.encode()))


def _redirect(location: str, **headers: str) -> httpx.Response:
    return httpx.Response(301, headers={"Location": location, **headers}, stream=Chunks(b""))


@contextmanager
def _network(handle: Handler, addresses: Mapping[str, str]):
    """Resolve fixed names and answer at the socket layer, beneath the validated transport."""

    async def resolve(host: str, port: int) -> tuple[str, ...]:
        return (addresses[host],)

    async def send(_transport: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        return await handle(request)

    with (
        patch("chrys.service.tools.builtins.web.http.resolve_addresses", autospec=True, side_effect=resolve),
        patch.object(httpx.AsyncHTTPTransport, "handle_async_request", autospec=True, side_effect=send),
    ):
        yield


async def _fetch(
    url: str, handle: Handler, *, addresses: Mapping[str, str] = _PUBLIC, timeout_seconds: int = 60
) -> tuple[str, dict]:
    fetch = WebFetchTools(WebEgressRuntime(), DestinationPolicy(), timeout_seconds=timeout_seconds)
    metadata: dict = {}
    token = tool_result_metadata.set(metadata)
    try:
        with _network(handle, addresses):
            result = await fetch.web_fetch(url, "Read")
    finally:
        tool_result_metadata.reset(token)
    return result, metadata


async def test_a_same_site_hop_to_a_private_address_is_blocked():
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _redirect("https://www.example.com/admin")

    result, metadata = await _fetch(
        "https://example.com/start", handle, addresses={**_PUBLIC, "www.example.com": "10.0.0.8"}
    )
    assert result == "Error: private_address_blocked"
    assert metadata["tool_error_code"] == "private_address_blocked"
    # The second hop is refused before any connection, not after.
    assert [request.headers["Host"] for request in requests] == ["example.com"]


async def test_a_same_site_redirect_is_followed():
    async def handle(request: httpx.Request) -> httpx.Response:
        return _redirect("https://www.example.com/next") if request.headers["Host"] == "example.com" else _page()

    result, metadata = await _fetch("https://example.com/start", handle)
    assert "page body" in result
    assert "Redirected from https://example.com/start." in result
    assert metadata["web_fetch_final_url"] == "https://www.example.com/next"


async def test_the_redirect_chain_is_bounded():
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _redirect(f"https://example.com/{len(requests)}")

    result, _ = await _fetch("https://example.com/0", handle)
    assert result == "Error: too_many_redirects"
    assert len(requests) == 3


async def test_a_cookie_set_on_one_hop_is_never_sent_on_the_next():
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            # Same host and path scope, so only the per-hop clear keeps it off the next request.
            return _redirect("https://example.com/next", **{"Set-Cookie": "session=secret; Path=/"})
        return _page()

    result, _ = await _fetch("https://example.com/start", handle)
    assert "page body" in result
    assert len(requests) == 2
    assert "cookie" not in requests[1].headers


async def test_a_retry_after_response_is_retried_once():
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _page("busy", 503, **{"Retry-After": "0"}) if len(requests) == 1 else _page()

    result, _ = await _fetch("https://example.com/", handle)
    assert "page body" in result
    assert len(requests) == 2
    assert all(request.headers["Accept-Encoding"] == "gzip" for request in requests)


async def test_a_page_that_stays_unavailable_is_retried_once_then_reported():
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _page("busy", 503, **{"Retry-After": "0"})

    result, metadata = await _fetch("https://example.com/", handle)
    assert result == "Error: http_5xx (HTTP 503)"
    assert metadata["tool_error_code"] == "http_5xx"
    assert metadata["tool_error_retryable"] is True
    # The body of an error response is never returned.
    assert "busy" not in result
    assert len(requests) == 2


async def test_a_stalled_page_ends_at_the_fetch_deadline():
    never = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        await never.wait()
        return _page()

    result, metadata = await _fetch("https://example.com/", handle, timeout_seconds=1)
    assert result == "Error: Web fetch deadline exceeded"
    assert metadata["tool_error_code"] == "fetch_timeout"


async def test_a_url_that_outgrows_the_limit_once_encoded_is_an_invalid_url():
    # 2000 CJK characters pass the tool's own length check, but percent-encoding
    # makes the request URL about 18 KB.
    async def handle(request: httpx.Request) -> httpx.Response:
        raise AssertionError("an over-long URL must not reach the network")

    result, metadata = await _fetch("https://example.com/" + "中" * 2000, handle)

    assert result == "Error: invalid_url"
    assert metadata["tool_error_code"] == "invalid_url"
