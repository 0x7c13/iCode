# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Anonymous Exa search over bounded, stateless MCP HTTP requests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from chrys.foundation.net.url import normalize_url
from chrys.service.tools.builtins.web.config import ANONYMOUS_SEARCH_ENDPOINTS
from chrys.service.tools.builtins.web.http import WebError, bounded_body, check_status
from chrys.service.tools.builtins.web.search.providers import ENVIRONMENT_FAILURES
from chrys.service.tools.builtins.web.search.types import SearchHit, SearchRequest, SearchResponse

_RESULT_HEADER = re.compile(r"(?:\A|\n---\n+)Title: ([^\n]+)\nURL: ([^\n]+)\n")
# Every request to this anonymous service is ours, so these are the service
# refusing, not a caller or configuration error: a 4xx (typically a bot block),
# a JSON-RPC error, or a tool result flagged isError (its free-tier rate limit,
# an upstream API error, a timeout). An empty search is a plain result, never
# flagged, so it still ends the chain.
_FALLS_THROUGH = ENVIRONMENT_FAILURES | {"http_4xx", "exa_rpc_error", "exa_search_failed"}
_OBJECTIVE = (
    "Find original pages relevant to the full query. Respect any dates and constraints in the query. "
    "Extract relevant facts and publication dates; prefer primary sources."
)


def rpc_result(raw: bytes, mime: str) -> dict[str, Any]:
    """Accept JSON or complete SSE frames; never interpret errors as empty hits."""
    try:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        if mime == "text/event-stream":
            messages = []
            for frame in text.split("\n\n"):
                data = []
                for line in frame.split("\n"):
                    if line.startswith("data:"):
                        value = line[5:]
                        data.append(value.removeprefix(" "))
                if any(data):
                    messages.append(json.loads("\n".join(data)))
        elif mime == "application/json" or mime.endswith("+json"):
            messages = [json.loads(text)]
        else:
            raise WebError("protocol_error")
    except ValueError, UnicodeError, RecursionError:
        raise WebError("protocol_error") from None
    replies = []
    for message in messages:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            raise WebError("protocol_error")
        if "id" not in message and isinstance(message.get("method"), str):
            continue  # Unrelated MCP progress notifications carry no search results.
        if type(message.get("id")) is not int or message["id"] != 1:
            raise WebError("protocol_error")
        replies.append(message)
    if len(replies) != 1:
        raise WebError("protocol_error")
    reply = replies[0]
    if "error" in reply:
        raise WebError("exa_rpc_error")
    result = reply.get("result")
    if not isinstance(result, dict):
        raise WebError("protocol_error")
    if result.get("isError", False) is not False:
        raise WebError("exa_search_failed")
    return result


def parse_results(result: dict[str, Any]) -> tuple[SearchHit, ...]:
    """Normalize Exa's title/URL/highlights text, retaining publication metadata."""
    content = result.get("content")
    if not isinstance(content, list):
        raise WebError("protocol_error")
    hits = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str):
            raise WebError("protocol_error")
        # Exa represents an empty search as this single text block, not an empty list.
        if len(content) == 1 and block["text"] == "No search results found. Please try a different query.":
            return ()
        text = block["text"].replace("\r\n", "\n").strip() + "\n"
        headers = list(_RESULT_HEADER.finditer(text))
        if not headers or headers[0].start() != 0:
            raise WebError("search_parse_failed")
        for index, header in enumerate(headers):
            try:
                url = normalize_url(header[2].strip())
            except ValueError:
                continue
            end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
            hits.append(SearchHit(header[1].strip()[:1000], url, text[header.end() : end].strip()))
        if not hits:
            # Headers matched but every URL was rejected; do not disguise that as an empty search.
            raise WebError("search_parse_failed")
    return tuple(hits)


@dataclass(frozen=True)
class ExaMcpProvider:
    """Exa's public MCP search: no key, a query and a result count, filters only as prose."""

    id: str
    endpoint: str = ANONYMOUS_SEARCH_ENDPOINTS["exa_mcp"]

    def falls_through(self, failure: WebError) -> bool:
        return failure.code in _FALLS_THROUGH

    async def search(self, request: SearchRequest, *, http: httpx.AsyncClient) -> SearchResponse:
        objective = _OBJECTIVE
        filters = ""
        if request.allowed_domains:
            filters = " Only include these domains and their subdomains: " + ", ".join(request.allowed_domains)
        elif request.blocked_domains:
            filters = " Exclude these domains and their subdomains: " + ", ".join(request.blocked_domains)
        # The hosted tool limits objective to 4096 characters. The coordinator
        # also enforces every domain filter locally, including filters too long
        # to send, so a filtered search asks for more candidates to keep.
        if len(objective) + len(filters) <= 4096:
            objective += filters
        limit = min(request.limit * 3, 20) if filters else request.limit
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_search_exa",
                "arguments": {"query": request.query, "numResults": limit, "objective": objective},
            },
        }
        http.cookies.clear()
        async with http.stream(
            "POST",
            self.endpoint,
            headers={"Accept": "application/json, text/event-stream", "Accept-Encoding": "identity"},
            json=payload,
        ) as response:
            check_status(response)
            mime = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            raw = await bounded_body(response, decompress=False)
        return SearchResponse(self.id, parse_results(rpc_result(raw, mime)))
