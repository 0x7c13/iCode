# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``ProxyRouter`` names the same proxy the client's own transport lookup picks.

The router rebuilds httpx's proxy table instead of reading the client's
transports, so this contract is what keeps the two from drifting apart across
an httpx or SDK bump.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from chrys.foundation.errors import Origin
from chrys.service.llm.clients import _build_profile_http_client
from chrys.service.llm.proxy_route import ProxyRouter
from chrys.service.profiles.models.schema import ModelProfile

_DEFAULT_PORTS = {"http": 80, "https": 443, "socks5": 1080, "socks5h": 1080}

# Each environment is a set of proxy variables; unset ones are cleared first.
# "system" sets none, so both sides fall back to the OS proxy settings alike.
_ENVIRONMENTS: dict[str, dict[str, str]] = {
    "system": {},
    "https-only": {"HTTPS_PROXY": "http://https-proxy.example:3128"},
    "http-only": {"HTTP_PROXY": "http://http-proxy.example:8080"},
    "lowercase": {"https_proxy": "https-proxy.example:3128", "no_proxy": "a.example"},
    "all-with-no-proxy": {
        "ALL_PROXY": "http://all-proxy.example:8080",
        "HTTPS_PROXY": "http://user:secret@https-proxy.example:3128",
        "NO_PROXY": "a.example,.internal.example,10.0.0.1,::1,localhost,http://plain.example",
    },
    "socks": {"ALL_PROXY": "socks5://socks-proxy.example:1080"},
    "no-proxy-star": {"HTTPS_PROXY": "http://https-proxy.example:3128", "NO_PROXY": "b.example,*"},
}
_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
_URLS = (
    "https://a.example/v1",
    "http://a.example/v1",
    "https://sub.a.example/v1",
    "https://xa.example/v1",
    "https://b.internal.example/v1",
    "https://internal.example/v1",
    "https://10.0.0.1/v1",
    "https://10.0.0.2/v1",
    "http://localhost:8080/v1",
    "https://[::1]/v1",
    "http://plain.example/v1",
    "https://plain.example/v1",
    "https://api.other.example:8443/v1",
)


def _profile(provider: str, *, bypass_proxy: bool) -> ModelProfile:
    return ModelProfile(
        id="p",
        name="p",
        provider=provider,
        model_id="test-model",
        api_key="sk-test",
        base_url="https://api.other.example/v1",
        http_max_retries=0,
        bypass_proxy=bypass_proxy,
    )


def _transport_proxy(client: httpx.AsyncClient, url: httpx.URL) -> Origin | None:
    """The proxy the client's own lookup picks for *url*, read off httpcore's pool."""
    pool: Any = client._transport_for_url(url)._pool  # ty: ignore[unresolved-attribute]
    proxy_url = getattr(pool, "_proxy_url", None)
    if proxy_url is None:
        return None
    scheme = proxy_url.scheme.decode()
    return Origin(scheme, proxy_url.host.decode().lower(), proxy_url.port or _DEFAULT_PORTS[scheme])


@pytest.fixture
def set_environment(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    def apply(name: str) -> None:
        for variable in (*_PROXY_VARIABLES, *(variable.lower() for variable in _PROXY_VARIABLES)):
            monkeypatch.delenv(variable, raising=False)
        for variable, value in _ENVIRONMENTS[name].items():
            monkeypatch.setenv(variable, value)

    return apply


@pytest.mark.parametrize("bypass_proxy", [False, True], ids=["env", "bypass"])
@pytest.mark.parametrize("environment", sorted(_ENVIRONMENTS))
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_router_matches_the_client_transport_lookup(
    provider: str, environment: str, bypass_proxy: bool, set_environment: Callable[[str], None]
) -> None:
    set_environment(environment)
    profile = _profile(provider, bypass_proxy=bypass_proxy)
    client = _build_profile_http_client(profile, httpx.Timeout(5.0))
    router = ProxyRouter.from_client_config(bypass_proxy=bypass_proxy)
    try:
        routes = {url: router.proxy_for(httpx.URL(url)) for url in _URLS}
        transports = {url: _transport_proxy(client, httpx.URL(url)) for url in _URLS}
    finally:
        await client.aclose()

    assert routes == transports
    if bypass_proxy:
        assert set(routes.values()) == {None}


def test_the_contract_covers_proxied_and_direct_routes(set_environment: Callable[[str], None]) -> None:
    set_environment("all-with-no-proxy")
    router = ProxyRouter.from_client_config(bypass_proxy=False)

    assert {url: router.proxy_for(httpx.URL(url)) for url in _URLS} == {
        "https://a.example/v1": None,
        "http://a.example/v1": None,
        "https://sub.a.example/v1": None,
        "https://xa.example/v1": Origin("http", "https-proxy.example", 3128),
        "https://b.internal.example/v1": None,
        "https://internal.example/v1": Origin("http", "https-proxy.example", 3128),
        "https://10.0.0.1/v1": None,
        "https://10.0.0.2/v1": Origin("http", "https-proxy.example", 3128),
        "http://localhost:8080/v1": None,
        "https://[::1]/v1": None,
        "http://plain.example/v1": None,
        "https://plain.example/v1": Origin("http", "https-proxy.example", 3128),
        "https://api.other.example:8443/v1": Origin("http", "https-proxy.example", 3128),
    }
