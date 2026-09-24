# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The contract between the search coordinator and its provider adapters.

The coordinator owns the call budget, deadlines, retries, the fallback order
and the authoritative domain filter. An adapter owns everything particular to
its service: adapting the request, the wire protocol, and which of its
failures leave the query to the next provider.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import httpx

from chrys.service.tools.builtins.web.http import WebError
from chrys.service.tools.builtins.web.search.types import SearchRequest, SearchResponse

# Provider-environment failures: another configured provider may still serve
# the query, so the chain advances instead of failing the whole call. A request
# level error (invalid_request) is the caller's and fails every provider alike.
# A connection failure that is not worth retrying (a certificate rejected by an
# inspecting proxy, a DNS filter answering with 0.0.0.0) still belongs to one
# provider's host, and a request too large for one provider (a long CJK query
# in a GET URL) may still fit another's POST body.
ENVIRONMENT_FAILURES = frozenset(
    {
        "dns_failure",
        "connection_failed",
        "private_address_blocked",
        "proxy_ipv6_unsupported",
        "redirect_not_allowed",
        "unsupported_content_encoding",
        "response_too_large",
        "request_too_large",
        "search_challenge",
        "search_parse_failed",
        "protocol_error",
    }
)


class SearchProvider(Protocol):
    """One configured search service, frozen for a build."""

    @property
    def id(self) -> str: ...

    @property
    def endpoint(self) -> str:
        """The URL requests go to; the tool description names its origin."""
        ...

    async def search(self, request: SearchRequest, *, http: httpx.AsyncClient) -> SearchResponse:
        """Search with the caller's request, adapted to what this service understands."""
        ...

    def falls_through(self, failure: WebError) -> bool:
        """Whether the next provider may still serve a query this one failed with *failure*."""
        ...


def site_filtered(request: SearchRequest) -> SearchRequest:
    """Push domain filters into the query, for services without native filtering.

    The server-side filter widens the candidate pool; the local filter stays authoritative.
    """
    if not (request.allowed_domains or request.blocked_domains):
        return request
    parts = [request.query]
    if request.allowed_domains:
        sites = " OR ".join(f"site:{domain}" for domain in request.allowed_domains)
        parts.append(f"({sites})" if len(request.allowed_domains) > 1 else sites)
    parts.extend(f"-site:{domain}" for domain in request.blocked_domains)
    return replace(request, query=" ".join(parts))
