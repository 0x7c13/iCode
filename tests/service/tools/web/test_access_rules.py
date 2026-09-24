# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""URL access rules, from user settings through ``build_web_tools`` to the socket seam.

Each tool is built the way an agent build builds it, so the origin grants reach
``DestinationPolicy`` through the real settings wiring. Only DNS and the inner
socket transport are replaced; a refused URL must never reach either.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import replace

import httpx
import pytest

from chrys.foundation.config.settings import Settings
from chrys.service.tools.builtins.web.build import build_web_tools
from chrys.service.tools.builtins.web.config import WebSearchConfigPatch, parse_search
from chrys.service.tools.builtins.web.fetch.tool import WebFetchTools
from chrys.service.tools.builtins.web.search.tool import WebSearchTools
from chrys.service.tools.result_metadata import tool_result_metadata
from tests.service.tools.web._support import Chunks, SocketNetwork, socket_network

_PUBLIC = "93.184.215.14"
_ADDRESSES = {
    "example.com": _PUBLIC,
    "www.example.com": _PUBLIC,
    "allowed.example": _PUBLIC,
    "other.example": _PUBLIC,
    "intranet.example": "10.0.0.8",
    "10.0.0.5": "10.0.0.5",
    "search.corp.example": "10.0.0.5",
    "public-search.example": _PUBLIC,
}


def _page(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200, headers={"Content-Type": "text/plain"}, stream=Chunks(f"body of {request.headers['Host']}".encode())
    )


async def _serve_page(request: httpx.Request) -> httpx.Response:
    return _page(request)


async def _unreachable(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no request may reach the socket layer: {request.method} {request.headers['Host']}")


def _fetch_tools(**settings: str) -> WebFetchTools:
    built = build_web_tools(["web_fetch"], replace(Settings(), **settings), None, None)
    assert built.fetch is not None
    return built.fetch


async def _run(call: Callable[[], Awaitable[str]]) -> tuple[str, dict]:
    metadata: dict = {}
    token = tool_result_metadata.set(metadata)
    try:
        return await call(), metadata
    finally:
        tool_result_metadata.reset(token)


def _assert_untouched(network: SocketNetwork) -> None:
    network.resolve.assert_not_called()
    network.send.assert_not_called()


@pytest.mark.parametrize(
    ("settings", "url", "code"),
    [
        pytest.param({}, "http://example.com/", "http_not_allowed", id="plain-http-refused-by-default"),
        pytest.param(
            {"web_fetch_allowed_origins": '["https://allowed.example"]'},
            "https://other.example/doc",
            "origin_not_allowed",
            id="outside-the-allow-list",
        ),
        pytest.param(
            {"web_fetch_denied_origins": '["https://EXAMPLE.com:443"]'},
            "https://example.com/doc",
            "origin_denied",
            id="deny-list-matches-the-normalized-origin",
        ),
        pytest.param(
            {
                "web_fetch_denied_origins": '["https://example.com"]',
                "web_fetch_allowed_origins": '["https://example.com"]',
            },
            "https://example.com/doc",
            "origin_denied",
            id="deny-beats-allow",
        ),
        pytest.param(
            {"web_fetch_denied_origins": '["http://example.com"]', "web_fetch_http_origins": '["http://example.com"]'},
            "http://example.com/doc",
            "origin_denied",
            id="deny-beats-http-grant",
        ),
        pytest.param(
            {"web_fetch_allowed_origins": '["http://example.com"]'},
            "http://example.com/doc",
            "http_not_allowed",
            id="allow-list-does-not-grant-plain-http",
        ),
        pytest.param(
            {"web_fetch_http_origins": '["http://example.com"]'},
            "http://example.com:8080/doc",
            "http_not_allowed",
            id="http-grant-is-one-exact-origin",
        ),
        pytest.param(
            {"web_search_http_origins": '["http://example.com"]'},
            "http://example.com/doc",
            "http_not_allowed",
            id="search-http-grant-does-not-reach-fetch",
        ),
    ],
)
async def test_a_refused_fetch_never_resolves_or_connects(settings, url, code):
    fetch = _fetch_tools(**settings)
    with socket_network(_unreachable, _ADDRESSES) as network:
        result, metadata = await _run(lambda: fetch.web_fetch(url, "Read"))
    assert result == f"Error: {code}"
    assert metadata["tool_error_code"] == code
    _assert_untouched(network)
    assert not fetch.cache.entries


@pytest.mark.parametrize(
    ("settings", "url"),
    [
        pytest.param(
            {"web_fetch_allowed_origins": '["https://allowed.example"]'},
            "https://allowed.example/doc",
            id="inside-the-allow-list",
        ),
        pytest.param({"web_fetch_http_origins": '["http://example.com"]'}, "http://example.com/doc", id="http-grant"),
        pytest.param(
            {"web_fetch_private_origins": '["https://intranet.example"]'},
            "https://intranet.example/doc",
            id="private-grant",
        ),
        pytest.param(
            {
                "web_fetch_http_origins": '["http://10.0.0.5:8080"]',
                "web_fetch_private_origins": '["http://10.0.0.5:8080"]',
            },
            "http://10.0.0.5:8080/doc",
            id="http-and-private-grant-for-an-address-literal",
        ),
    ],
)
async def test_a_granted_origin_is_fetched(settings, url):
    fetch = _fetch_tools(**settings)
    with socket_network(_serve_page, _ADDRESSES) as network:
        result, metadata = await _run(lambda: fetch.web_fetch(url, "Read"))
    host = httpx.URL(url).netloc.decode()
    assert f"body of {host}" in result, result
    assert metadata["web_fetch_final_url"] == url
    assert network.send.call_count == 1
    assert network.send.call_args.args[1].headers["Host"] == host


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({}, id="no-grant"),
        pytest.param({"web_fetch_private_origins": '["https://example.com"]'}, id="another-origins-grant"),
        pytest.param({"web_search_private_origins": '["https://intranet.example"]'}, id="search-grant-only"),
    ],
)
async def test_a_private_address_needs_this_origins_fetch_grant(settings):
    fetch = _fetch_tools(**settings)
    with socket_network(_unreachable, _ADDRESSES) as network:
        result, metadata = await _run(lambda: fetch.web_fetch("https://intranet.example/doc", "Read"))
    assert result == "Error: private_address_blocked"
    assert metadata["tool_error_code"] == "private_address_blocked"
    # The name is resolved to classify it, and then no connection is made.
    assert [call.args[0] for call in network.resolve.call_args_list] == ["intranet.example"]
    network.send.assert_not_called()


@pytest.mark.parametrize(
    ("settings", "code"),
    [
        pytest.param({"web_fetch_denied_origins": '["https://www.example.com"]'}, "origin_denied", id="denied-hop"),
        pytest.param(
            {"web_fetch_allowed_origins": '["https://example.com"]'}, "origin_not_allowed", id="hop-outside-allow-list"
        ),
    ],
)
async def test_a_same_site_redirect_hop_is_checked_against_the_access_rules(settings, code):
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers["Host"] == "example.com":
            return httpx.Response(301, headers={"Location": "https://www.example.com/admin"}, stream=Chunks(b""))
        return _page(request)

    fetch = _fetch_tools(**settings)
    with socket_network(handle, _ADDRESSES) as network:
        result, metadata = await _run(lambda: fetch.web_fetch("https://example.com/start", "Read"))
    assert result == f"Error: {code}"
    assert metadata["tool_error_code"] == code
    # The refused hop is neither resolved nor requested.
    assert [request.headers["Host"] for request in requests] == ["example.com"]
    assert [call.args[0] for call in network.resolve.call_args_list] == ["example.com"]


def _custom_search(endpoint: str) -> WebSearchConfigPatch:
    return parse_search(
        {
            "mode": "provider",
            "provider": "corp",
            "providers": {
                "corp": {
                    "type": "custom_http",
                    "endpoint": endpoint,
                    "request": {"query_params": {"q": {"$input": "query"}}},
                    "response": {"results_pointer": "/results", "url_pointer": "/url", "title_pointer": "/title"},
                }
            },
        }
    )


def _search_tools(endpoint: str, **settings: str) -> WebSearchTools:
    granted = replace(Settings(), web_search_custom_endpoints=json.dumps([{"url": endpoint}]), **settings)
    built = build_web_tools(["web_search"], granted, _custom_search(endpoint), None)
    assert built.search is not None
    return built.search


async def _search_results(request: httpx.Request) -> httpx.Response:
    body = {"results": [{"title": "Found", "url": "https://example.com/a"}]}
    return httpx.Response(200, headers={"Content-Type": "application/json"}, stream=Chunks(json.dumps(body).encode()))


@pytest.mark.parametrize(
    ("settings", "admitted"),
    [
        pytest.param({}, False, id="no-grant"),
        pytest.param({"web_fetch_private_origins": '["https://search.corp.example"]'}, False, id="fetch-grant-only"),
        pytest.param({"web_search_private_origins": '["https://search.corp.example"]'}, True, id="search-grant"),
    ],
)
async def test_a_private_search_endpoint_needs_the_search_private_grant(settings, admitted):
    search = _search_tools("https://search.corp.example/search", **settings)
    with socket_network(_search_results if admitted else _unreachable, _ADDRESSES) as network:
        result, metadata = await _run(lambda: search.web_search("query"))
    if admitted:
        assert json.loads(result.split("\nExternal", 1)[0])["results"][0]["url"] == "https://example.com/a"
        assert network.send.call_args.args[1].headers["Host"] == "search.corp.example"
    else:
        assert result == "Error: private_address_blocked"
        assert metadata["tool_error_code"] == "private_address_blocked"
        network.send.assert_not_called()


async def test_a_plain_http_search_endpoint_is_admitted_by_the_search_http_grant():
    endpoint = "http://public-search.example:8080/search"
    search = _search_tools(endpoint, web_search_http_origins='["http://public-search.example:8080"]')
    with socket_network(_search_results, _ADDRESSES) as network:
        result, _ = await _run(lambda: search.web_search("query"))
    assert not result.startswith("Error:"), result
    request = network.send.call_args.args[1]
    assert request.url.scheme == "http"
    assert request.headers["Host"] == "public-search.example:8080"


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param({}, id="no-grant"),
        pytest.param({"web_fetch_http_origins": '["http://public-search.example:8080"]'}, id="fetch-grant-only"),
    ],
)
def test_a_plain_http_search_endpoint_without_the_search_grant_fails_the_build(settings):
    with pytest.raises(ValueError, match=r"grant http://public-search\.example:8080 in user setting tools\.web_search"):
        _search_tools("http://public-search.example:8080/search", **settings)
