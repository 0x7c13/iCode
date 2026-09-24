# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""web_fetch's build-local cache: 15-minute lifetime, 64 entries, 16 MiB, and no redirects.

The cache reads ``time.monotonic`` through its own module's ``time`` name, so
the clock is replaced there alone; the event loop keeps the real clock.
"""

from __future__ import annotations

import types
from collections.abc import Callable

import httpx
import pytest

from chrys.service.tools.builtins.web.fetch import cache as cache_module
from chrys.service.tools.builtins.web.fetch.cache import FetchCache, FetchedPage
from chrys.service.tools.builtins.web.fetch.tool import WebFetchTools
from chrys.service.tools.builtins.web.http import DestinationPolicy, WebEgressRuntime
from chrys.service.tools.result_metadata import tool_result_metadata
from tests.service.tools.web._support import Chunks, socket_network

_LIFETIME = 15 * 60
_MAX_ENTRIES = 64
_MAX_BYTES = 16 * 1024 * 1024
_ADDRESSES = {"example.com": "93.184.215.14", "www.example.com": "93.184.215.14"}


class _Clock:
    """A settable stand-in for ``time.monotonic``."""

    def __init__(self) -> None:
        self.now = 10_000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    shadow = types.ModuleType("time")
    shadow.monotonic = clock.monotonic  # type: ignore[attr-defined]
    monkeypatch.setattr(cache_module, "time", shadow)
    return clock


def _page(url: str, text: str = "text", *, redirect_to: str | None = None) -> FetchedPage:
    return FetchedPage(url, url, 200, "text/plain", len(text), text, redirect_to)


def _size(key: str, page: FetchedPage) -> int:
    """The cache's own accounting for one entry."""
    return len(page.text.encode()) + len(key.encode()) + len(page.final_url.encode()) + 512


def test_an_entry_lives_fifteen_minutes_from_its_fetch_even_when_read(clock):
    cache = FetchCache()
    page = _page("https://example.com/")
    cache.put("https://example.com/", page)
    clock.now += _LIFETIME / 2
    # A read keeps the entry recent for eviction, but does not extend its lifetime.
    assert cache.get("https://example.com/") is page
    clock.now += _LIFETIME / 2 - 0.001
    assert cache.get("https://example.com/") is page
    clock.now += 0.001
    assert cache.get("https://example.com/") is None
    assert not cache.entries
    assert cache.bytes == 0


def test_the_least_recently_used_entry_leaves_beyond_64_entries(clock):
    cache = FetchCache()
    keys = [f"https://example.com/{index}" for index in range(_MAX_ENTRIES + 1)]
    for key in keys[:_MAX_ENTRIES]:
        cache.put(key, _page(key))
    assert len(cache.entries) == _MAX_ENTRIES
    # Reading the oldest entry makes the second-oldest the eviction candidate.
    assert cache.get(keys[0]) is not None
    cache.put(keys[-1], _page(keys[-1]))
    assert len(cache.entries) == _MAX_ENTRIES
    assert keys[0] in cache.entries
    assert keys[1] not in cache.entries
    assert cache.bytes == sum(_size(key, page) for key, (_, page, _) in cache.entries.items())


def test_the_byte_budget_evicts_the_oldest_entries(clock):
    cache = FetchCache()
    text = "a" * (5 * 1024 * 1024)
    keys = [f"https://example.com/{index}" for index in range(4)]
    for key in keys[:3]:
        cache.put(key, _page(key, text))
    assert list(cache.entries) == keys[:3]
    cache.put(keys[3], _page(keys[3], text))
    assert list(cache.entries) == keys[1:]
    assert cache.bytes == sum(_size(key, _page(key, text)) for key in keys[1:])
    assert cache.bytes <= _MAX_BYTES


def test_a_page_larger_than_the_byte_budget_is_never_cached(clock):
    cache = FetchCache()
    kept = "https://example.com/kept"
    cache.put(kept, _page(kept))
    key = "https://example.com/big"
    overhead = _size(key, _page(key, ""))
    exactly = _page(key, "a" * (_MAX_BYTES - overhead))
    assert _size(key, exactly) == _MAX_BYTES
    cache.put(key, exactly)
    # A page of exactly the budget fits once everything else is evicted.
    assert list(cache.entries) == [key]
    assert cache.bytes == _MAX_BYTES

    cache = FetchCache()
    cache.put(kept, _page(kept))
    cache.put(key, _page(key, "a" * (_MAX_BYTES - overhead + 1)))
    # One byte over is refused outright, without evicting anything.
    assert list(cache.entries) == [kept]
    assert cache.bytes == _size(kept, _page(kept))


def test_a_redirect_notice_is_never_cached(clock):
    cache = FetchCache()
    cache.put("https://example.com/", _page("https://example.com/", "", redirect_to="https://other.example/"))
    assert not cache.entries
    assert cache.bytes == 0


async def _fetch_twice(
    fetch: WebFetchTools, url: str, handle, *, between: Callable[[], None] = lambda: None
) -> list[tuple[str, dict]]:
    outcomes = []
    with socket_network(handle, _ADDRESSES):
        for index in range(2):
            if index:
                between()
            metadata: dict = {}
            token = tool_result_metadata.set(metadata)
            try:
                outcomes.append((await fetch.web_fetch(url, "Read"), metadata))
            finally:
                tool_result_metadata.reset(token)
    return outcomes


def _text(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers={"Content-Type": "text/plain"}, stream=Chunks(b"page body"))


async def test_a_fetch_after_fifteen_minutes_goes_back_to_the_network(clock):
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _text(request)

    def later() -> None:
        clock.now += _LIFETIME

    fetch = WebFetchTools(WebEgressRuntime(), DestinationPolicy())
    outcomes = await _fetch_twice(fetch, "https://example.com/", handle, between=later)
    assert len(requests) == 2
    assert [metadata["web_fetch_cache_hit"] for _, metadata in outcomes] == [False, False]
    assert all("served from cache" not in result for result, _ in outcomes)


async def test_a_fetch_within_fifteen_minutes_is_served_from_the_cache(clock):
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _text(request)

    def later() -> None:
        clock.now += _LIFETIME - 1

    fetch = WebFetchTools(WebEgressRuntime(), DestinationPolicy())
    outcomes = await _fetch_twice(fetch, "https://example.com/", handle, between=later)
    assert len(requests) == 1
    assert [metadata["web_fetch_cache_hit"] for _, metadata in outcomes] == [False, True]
    assert "served from cache" in outcomes[1][0]


async def test_a_cross_site_redirect_is_fetched_again_every_time(clock):
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(301, headers={"Location": "https://other.example/moved"}, stream=Chunks(b""))

    fetch = WebFetchTools(WebEgressRuntime(), DestinationPolicy())
    outcomes = await _fetch_twice(fetch, "https://example.com/", handle)
    assert len(requests) == 2
    assert [metadata["web_fetch_cache_hit"] for _, metadata in outcomes] == [False, False]
    assert all(metadata["web_fetch_redirect_to"] == "https://other.example/moved" for _, metadata in outcomes)
    assert not fetch.cache.entries


async def test_a_followed_same_site_redirect_caches_the_final_page_under_the_requested_url(clock):
    requests: list[httpx.Request] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers["Host"] == "example.com":
            return httpx.Response(301, headers={"Location": "https://www.example.com/next"}, stream=Chunks(b""))
        return _text(request)

    fetch = WebFetchTools(WebEgressRuntime(), DestinationPolicy())
    outcomes = await _fetch_twice(fetch, "https://example.com/start", handle)
    assert [request.headers["Host"] for request in requests] == ["example.com", "www.example.com"]
    assert [metadata["web_fetch_cache_hit"] for _, metadata in outcomes] == [False, True]
    assert outcomes[1][1]["web_fetch_final_url"] == "https://www.example.com/next"
    assert list(fetch.cache.entries) == ["https://example.com/start"]
