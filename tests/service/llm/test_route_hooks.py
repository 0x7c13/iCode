# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Route hooks: each request's own snapshot, and reach recorded per first hop."""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING

import httpx
import pytest

from chrys.foundation.errors import (
    ROUTE_EXTENSION_KEY,
    ErrorKind,
    Origin,
    RouteFacts,
    classify_error,
    is_deterministic_connection_error,
)
from chrys.service.llm import raw_http_log
from chrys.service.llm.clients import _build_profile_http_client
from chrys.service.llm.proxy_route import ProxyRouter
from chrys.service.llm.route_facts import build_route_hooks, has_reached
from chrys.service.profiles.models.schema import ModelProfile

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import LocalHTTPServer

_PROXY = Origin("http", "proxy.example", 3128)
_A = Origin("https", "a.example", 443)
_B = Origin("https", "b.example", 443)


def _profile(*, bypass_proxy: bool = False, base_url: str = "https://a.example/v1") -> ModelProfile:
    return ModelProfile(
        id="p",
        name="p",
        provider="openai",
        model_id="test-model",
        api_key="sk-test",
        base_url=base_url,
        http_max_retries=0,
        bypass_proxy=bypass_proxy,
    )


@pytest.fixture
def proxy_env(monkeypatch: pytest.MonkeyPatch, clear_proxy_env: Callable[[], None]) -> pytest.MonkeyPatch:
    clear_proxy_env()
    monkeypatch.setenv("ALL_PROXY", "http://proxy.example:3128")
    return monkeypatch


@pytest.fixture
def direct_env(monkeypatch: pytest.MonkeyPatch, clear_proxy_env: Callable[[], None]) -> pytest.MonkeyPatch:
    """No proxy for any host. With no proxy env at all, ``urllib.request.getproxies``
    falls back to the macOS or Windows system proxy settings, as httpx does."""
    clear_proxy_env()
    monkeypatch.setenv("NO_PROXY", "*")
    return monkeypatch


def _gaierror_noname() -> socket.gaierror:
    return socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known")


def _mock_client(handler: Callable[[httpx.Request], httpx.Response], *, bypass_proxy: bool) -> httpx.AsyncClient:
    """A client whose route hooks are Chrys's and whose wire is *handler*."""
    router = ProxyRouter.from_client_config(bypass_proxy=bypass_proxy)
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), event_hooks=build_route_hooks(router), follow_redirects=True
    )


def _connect_failure(leaf: BaseException) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(str(leaf), request=request) from leaf

    return handler


class _Stop(Exception):
    """Raised by a recording hook so the request never leaves the process."""


async def _stamped_route(client: httpx.AsyncClient, url: str) -> RouteFacts:
    seen: list[object] = []

    async def record(request: httpx.Request) -> None:
        seen.append(request.extensions.get(ROUTE_EXTENSION_KEY))
        raise _Stop

    client.event_hooks["request"].append(record)
    with pytest.raises(_Stop):
        await client.post(url)
    [facts] = seen
    assert isinstance(facts, RouteFacts)
    return facts


async def test_same_host_direct_and_proxied_clients_keep_their_own_route(proxy_env: pytest.MonkeyPatch) -> None:
    timeout = httpx.Timeout(5.0)
    proxied = _build_profile_http_client(_profile(), timeout)
    direct = _build_profile_http_client(_profile(bypass_proxy=True), timeout)
    try:
        via_proxy = await _stamped_route(proxied, "https://a.example/v1/chat/completions")
        straight = await _stamped_route(direct, "https://a.example/v1/chat/completions")
    finally:
        await proxied.aclose()
        await direct.aclose()

    assert via_proxy == RouteFacts(_A, _PROXY, first_hop_reached=False)
    assert straight == RouteFacts(_A, None, first_hop_reached=False)


async def test_proxied_reach_does_not_make_direct_noname_retryable(proxy_env: pytest.MonkeyPatch) -> None:
    async with _mock_client(lambda request: httpx.Response(200), bypass_proxy=False) as proxied:
        await proxied.post("https://a.example/v1")
    assert has_reached(_PROXY)

    async with _mock_client(_connect_failure(_gaierror_noname()), bypass_proxy=True) as direct:
        with pytest.raises(httpx.ConnectError) as info:
            await direct.post("https://a.example/v1")

    assert classify_error(info.value).route == RouteFacts(_A, None, first_hop_reached=False)
    assert is_deterministic_connection_error(info.value) is True


async def test_401_response_records_first_hop_then_noname_retries(direct_env: pytest.MonkeyPatch) -> None:
    answers: list[Callable[[httpx.Request], httpx.Response]] = [
        lambda request: httpx.Response(401, json={"error": {"message": "bad key"}}),
        _connect_failure(_gaierror_noname()),
    ]

    def next_answer(request: httpx.Request) -> httpx.Response:
        return answers.pop(0)(request)

    async with _mock_client(next_answer, bypass_proxy=False) as client:
        assert (await client.post("https://a.example/v1")).status_code == 401
        with pytest.raises(httpx.ConnectError) as info:
            await client.post("https://a.example/v1")

    result = classify_error(info.value)
    assert (result.kind, result.retryable) == (ErrorKind.DNS_FAILED, True)
    assert result.route == RouteFacts(_A, None, first_hop_reached=True)
    assert is_deterministic_connection_error(info.value) is False


async def test_cross_host_redirect_failure_reads_inner_request_route(
    proxy_env: pytest.MonkeyPatch,
) -> None:
    proxy_env.setenv("NO_PROXY", "a.example")
    # Each request's extensions dict itself: had the redirect shared the
    # first request's dict, its entry would now hold the second snapshot.
    extensions: dict[str, Mapping[str, object]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        extensions[request.url.host] = request.extensions
        if request.url.host == "a.example":
            return httpx.Response(302, headers={"Location": "https://b.example/v1"})
        raise httpx.ConnectError("refused", request=request) from ConnectionRefusedError(61, "Connect call failed")

    async with _mock_client(handler, bypass_proxy=False) as client:
        with pytest.raises(httpx.ConnectError) as info:
            await client.post("https://a.example/v1")

    assert classify_error(info.value).route == RouteFacts(_B, _PROXY, first_hop_reached=False)
    # The redirect got a fresh snapshot; the first request's is untouched.
    assert {host: stamped[ROUTE_EXTENSION_KEY] for host, stamped in extensions.items()} == {
        "a.example": RouteFacts(_A, None, first_hop_reached=False),
        "b.example": RouteFacts(_B, _PROXY, first_hop_reached=False),
    }


class _StallingStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"partial": true}\n\n'
        raise httpx.ReadTimeout("timed out")


async def test_streamed_read_timeout_via_proxy_carries_route(proxy_env: pytest.MonkeyPatch) -> None:
    def stalling(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_StallingStream())

    async with (
        _mock_client(stalling, bypass_proxy=False) as client,
        client.stream("POST", "https://a.example/v1") as response,
    ):
        with pytest.raises(httpx.ReadTimeout) as info:
            async for _chunk in response.aiter_bytes():
                pass

    result = classify_error(info.value)
    assert (result.kind, result.route) == (ErrorKind.READ_TIMEOUT, RouteFacts(_A, _PROXY, first_hop_reached=False))


async def test_proxy_credentials_are_stripped(
    monkeypatch: pytest.MonkeyPatch, clear_proxy_env: Callable[[], None]
) -> None:
    clear_proxy_env()
    monkeypatch.setenv("HTTPS_PROXY", "http://user:s3cret@proxy.example:3128")
    client = _build_profile_http_client(_profile(), httpx.Timeout(5.0))
    try:
        facts = await _stamped_route(client, "https://a.example/v1/chat/completions")
    finally:
        await client.aclose()

    assert facts.proxy == _PROXY
    assert "s3cret" not in repr(facts)
    assert "user" not in repr(facts)


async def test_route_hooks_run_before_raw_log_hooks(
    monkeypatch: pytest.MonkeyPatch,
    direct_env: pytest.MonkeyPatch,
    local_http_server: Callable[..., AsyncIterator[LocalHTTPServer]],
    tmp_path: Path,
) -> None:
    seen_by_raw_log: list[object] = []
    real_session_id = raw_http_log._request_session_id

    def spy(request: httpx.Request, *, fallback: str | None) -> str | None:
        seen_by_raw_log.append(request.extensions.get(ROUTE_EXTENSION_KEY))
        return real_session_id(request, fallback=fallback)

    monkeypatch.setattr(raw_http_log, "_request_session_id", spy)
    async with local_http_server() as server:
        client = _build_profile_http_client(
            _profile(base_url=server.url), httpx.Timeout(5.0), raw_http_log_path=tmp_path / "raw.jsonl"
        )
        try:
            assert (await client.get(f"{server.url}/v1")).status_code == 200
        finally:
            await client.aclose()

    [facts] = seen_by_raw_log
    assert isinstance(facts, RouteFacts)
    assert facts.proxy is None


async def test_route_recorded_even_if_raw_log_hook_raises(
    monkeypatch: pytest.MonkeyPatch,
    direct_env: pytest.MonkeyPatch,
    local_http_server: Callable[..., AsyncIterator[LocalHTTPServer]],
    tmp_path: Path,
) -> None:

    class _RawLogBroke(Exception):
        pass

    def broken_tee(**_kwargs: object) -> httpx.AsyncByteStream:
        raise _RawLogBroke

    monkeypatch.setattr(raw_http_log, "_RawLogAsyncByteStream", broken_tee)
    async with local_http_server() as server:
        target = Origin("http", "127.0.0.1", httpx.URL(server.url).port or 0)
        client = _build_profile_http_client(
            _profile(base_url=server.url), httpx.Timeout(5.0), raw_http_log_path=tmp_path / "raw.jsonl"
        )
        try:
            with pytest.raises(_RawLogBroke):
                await client.get(f"{server.url}/v1")
        finally:
            await client.aclose()

    assert has_reached(target)
