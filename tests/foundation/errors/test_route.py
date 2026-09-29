# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Route facts read back from a failed request."""

from __future__ import annotations

import httpx
import pytest

from chrys.foundation.errors import ROUTE_EXTENSION_KEY, Origin, RouteFacts, origin_of, route_of
from chrys.foundation.errors._walk import iter_explicit_graph
from tests.support.provider_errors import raised_from

_A = RouteFacts(Origin("https", "a.example", 443), None, first_hop_reached=True)
_B = RouteFacts(Origin("https", "b.example", 443), Origin("http", "proxy.example", 3128), first_hop_reached=False)


def _request(url: str, facts: object | None) -> httpx.Request:
    request = httpx.Request("POST", url)
    if facts is not None:
        request.extensions = {**request.extensions, ROUTE_EXTENSION_KEY: facts}
    return request


def test_route_of_takes_the_deepest_snapshot() -> None:
    # After a redirect the inner transport error holds the redirected request.
    inner = httpx.ConnectError("failed", request=_request("https://b.example/v1", _B))
    outer = raised_from(httpx.ConnectError("failed", request=_request("https://a.example/v1", _A)), inner)

    assert route_of(iter_explicit_graph(outer)) == _B


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(httpx.ConnectError("failed", request=_request("https://a.example/v1", None)), id="no-snapshot"),
        pytest.param(
            httpx.ConnectError("failed", request=_request("https://a.example/v1", {"target": "a"})), id="wrong-type"
        ),
        pytest.param(httpx.ConnectError("failed"), id="no-request"),
        pytest.param(OSError("failed"), id="not-an-http-error"),
    ],
)
def test_route_of_is_none_without_a_route_snapshot(exc: BaseException) -> None:
    assert route_of(iter_explicit_graph(exc)) is None


@pytest.mark.parametrize(
    ("url", "origin"),
    [
        ("https://API.Example.com/v1", Origin("https", "api.example.com", 443)),
        ("http://api.example.com:8080/v1", Origin("http", "api.example.com", 8080)),
        ("http://user:secret@proxy.example:3128", Origin("http", "proxy.example", 3128)),
        ("socks5://proxy.example", Origin("socks5", "proxy.example", 1080)),
        ("https://[2001:DB8::1]/v1", Origin("https", "2001:db8::1", 443)),
        (httpx.URL("https://api.example.com/v1"), Origin("https", "api.example.com", 443)),
    ],
)
def test_origin_of_normalizes_scheme_host_and_port(url: object, origin: Origin) -> None:
    assert origin_of(url) == origin


@pytest.mark.parametrize("url", ["", "/v1/chat", "ftp://files.example/x", "https://api.example.com:notaport/"])
def test_origin_of_is_none_without_a_full_origin(url: str) -> None:
    assert origin_of(url) is None


def test_first_hop_is_the_proxy_when_there_is_one() -> None:
    assert (_A.first_hop, _B.first_hop) == (_A.target, _B.proxy)
