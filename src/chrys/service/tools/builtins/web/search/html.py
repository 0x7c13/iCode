# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Credential-free HTML search adapters using the shared validated transport."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from chrys.foundation.net.url import normalize_url
from chrys.service.tools.builtins.web.config import HTML_SEARCH_ENDPOINTS
from chrys.service.tools.builtins.web.http import WebError, bounded_body, check_status, run_off_loop
from chrys.service.tools.builtins.web.search.providers import ENVIRONMENT_FAILURES, site_filtered
from chrys.service.tools.builtins.web.search.types import SearchHit, SearchRequest, SearchResponse

_REDIRECT_HOSTS = {
    "bing_html": frozenset({"www.bing.com", "cn.bing.com"}),
    "duckduckgo_html": frozenset({"html.duckduckgo.com"}),
}
# A 4xx from a search page nobody holds a key for is a bot block, not our mistake.
_FALLS_THROUGH = ENVIRONMENT_FAILURES | {"http_4xx"}


def result_url(href: str, provider_type: str) -> str:
    """Decode result link wrappers locally, without requesting tracking URLs."""
    if len(href) > 16384:
        raise ValueError("Result link too long")
    absolute = urljoin(HTML_SEARCH_ENDPOINTS[provider_type], href)
    parsed = urlsplit(absolute)
    params = parse_qs(parsed.query)
    if provider_type == "bing_html" and parsed.hostname in _REDIRECT_HOSTS[provider_type] and parsed.path == "/ck/a":
        encoded = params.get("u", [""])[0]
        if not encoded.startswith("a1"):
            raise ValueError("Unknown Bing link wrapper")
        try:
            absolute = base64.b64decode(
                encoded[2:] + "=" * (-len(encoded[2:]) % 4), altchars=b"-_", validate=True
            ).decode()
        except (binascii.Error, UnicodeError) as err:
            raise ValueError("Invalid Bing link wrapper") from err
    elif provider_type == "duckduckgo_html" and parsed.hostname in {"duckduckgo.com", "html.duckduckgo.com"}:
        if parsed.path.rstrip("/") == "/l":
            absolute = params.get("uddg", [""])[0]
    return normalize_url(absolute)


def parse_html(raw: bytes, provider_type: str) -> tuple[SearchHit, ...]:
    """Extract organic results only; unknown layouts and challenges fail closed."""
    soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
    if soup.select_one(
        '#challenge-form, #anomaly-modal, .anomaly-modal, #b_captcha, [id*="captcha"], iframe[src*="captcha"]'
    ):
        raise WebError("search_challenge")
    if soup.select_one('form[action*="anomaly"], form[action*="challenge"]'):
        raise WebError("search_challenge")
    for node in soup.select("script, style, template, noscript"):
        node.decompose()
    bing = provider_type == "bing_html"
    blocks = (
        soup.select("li.b_algo")
        if bing
        else soup.select("div.result:not(.result--ad):not(.result--silver):not(.result--sponsored)")
    )
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for block in blocks:
        anchor = block.select_one("h2 a[href]" if bing else "a.result__a[href]")
        if anchor is None:
            continue
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        try:
            url = result_url(href, provider_type)
        except ValueError:
            continue
        if url in seen:
            continue
        title = anchor.get_text(" ", strip=True)
        if not title:
            continue
        snippet = block.select_one(".b_caption p" if bing else ".result__snippet")
        hits.append(SearchHit(title[:1000], url, snippet.get_text(" ", strip=True)[:10000] if snippet else ""))
        seen.add(url)
    if hits:
        return tuple(hits)
    empty = soup.select_one("#b_results .b_no" if bing else ".no-results")
    if empty is not None and not soup.select("li.b_algo h2 a[href]" if bing else "a.result__a[href]"):
        return ()
    # Login pages, consent forms, changed layouts and malformed links are not
    # authoritative evidence that a query has no results.
    raise WebError("search_parse_failed")


@dataclass(frozen=True)
class HtmlSearchProvider:
    """A search engine's plain HTML results page, read without a key or tracking redirects."""

    id: str
    type: str
    """A key of HTML_SEARCH_ENDPOINTS."""

    @property
    def endpoint(self) -> str:
        return HTML_SEARCH_ENDPOINTS[self.type]

    def falls_through(self, failure: WebError) -> bool:
        return failure.code in _FALLS_THROUGH

    async def search(self, request: SearchRequest, *, http: httpx.AsyncClient) -> SearchResponse:
        current = str(httpx.URL(self.endpoint, params={"q": site_filtered(request).query}))
        if len(current) > 16384:
            raise WebError("request_too_large")
        for hop in range(3):
            http.cookies.clear()
            async with http.stream(
                "GET", current, headers={"Accept": "text/html", "Accept-Encoding": "gzip"}
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    current = self._redirect_target(current, response.headers.get("location"), last_hop=hop == 2)
                    continue
                if response.status_code == 202:
                    raise WebError("search_challenge")
                check_status(response)
                if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "text/html":
                    raise WebError("protocol_error")
                raw = await bounded_body(response, decompress=True)
            return SearchResponse(self.id, await run_off_loop(parse_html, raw, self.type))
        raise WebError("redirect_not_allowed")

    def _redirect_target(self, current: str, location: str | None, *, last_hop: bool) -> str:
        """Follow only the engine's own regional hosts, and only to its search path."""
        if last_hop or not location:
            raise WebError("redirect_not_allowed")
        try:
            target = normalize_url(urljoin(current, location), limit=16384)
        except ValueError as err:
            raise WebError("redirect_not_allowed") from err
        parsed = urlsplit(target)
        if (
            parsed.scheme != "https"
            or parsed.port not in {None, 443}
            or parsed.hostname not in _REDIRECT_HOSTS[self.type]
            or parsed.path != urlsplit(self.endpoint).path
        ):
            raise WebError("redirect_not_allowed")
        return target
