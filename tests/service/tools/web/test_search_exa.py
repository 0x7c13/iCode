# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Anonymous Exa wire contract and fresh-install search defaults."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.tool_kinds import get_tool_kind
from chrys.service.profiles.agents.loader import load_profile_from_yaml
from chrys.service.tools.builtins.web.build import build_web_tools
from chrys.service.tools.builtins.web.config import parse_provider, parse_search
from chrys.service.tools.builtins.web.http import WebError
from chrys.service.tools.builtins.web.search.exa_mcp import parse_results, rpc_result
from chrys.service.tools.registry import ToolRegistry
from tests.service.tools.web._support import Chunks, assemble
from tests.support.paths import REPO_ROOT

_TEXT = (
    "Title: 中文科技新闻\nURL: https://example.com/news\nPublished: 2026-09-13\n"
    "Author: Editor\nHighlights:\nOriginal news.\n\n---\n\n"
    "Title: Another page\nURL: https://other.example/news\nPublished: N/A\nHighlights:\nOther news."
)


def envelope(**changes):
    return {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": _TEXT}]}, **changes}


@pytest.mark.parametrize("name", ["Code", "QA", "General", "Explore"])
def test_shipped_profiles_leave_web_tools_off(name):
    """Web tools send queries off the machine, so only a profile the user edits turns them on."""
    profile = load_profile_from_yaml(REPO_ROOT / "src/chrys/service/profiles/agents/builtins" / f"{name}.yaml")
    assert not {"web_search", "web_fetch"}.intersection(profile.tools.builtins)
    assert profile.tools.web_search is None and profile.tools.web_fetch is None


def test_enabling_web_search_without_configuration_uses_keyless_exa():
    with patch(
        "chrys.service.tools.builtins.web.build.resolve_search_credentials",
        autospec=True,
        side_effect=AssertionError("secret lookup"),
    ):
        registry = ToolRegistry()
        registry.load_builtins(["web_search"], settings=Settings(), web=assemble(["web_search"]))
        tools = registry.get_all()
    assert [tool.name for tool in tools] == ["web_search"]
    assert get_tool_kind(tools[0]) == "web_search"
    assert "exa (https://mcp.exa.ai)" in tools[0].description


def test_disabled_and_missing_category_do_not_create_search():
    assert build_web_tools([], Settings(), None, None).search is None
    assert build_web_tools(["web_search"], Settings(web_search_mode="off"), None, None).search is None
    assert build_web_tools(["web_search"], Settings(), parse_search({"mode": "off"}), None).search is None


@pytest.mark.parametrize("config", [{"fallback_chain": []}, {"providers": {}}, {"provider": "missing"}])
def test_explicit_incomplete_configuration_is_not_replaced_by_default(config):
    with pytest.raises(ValueError):
        build_web_tools(["web_search"], Settings(), parse_search(config), None)


@pytest.mark.parametrize("extra", [{"api_key_env": "SECRET"}, {"endpoint": "https://evil.example/"}])
def test_anonymous_exa_rejects_credential_and_endpoint_overrides(extra):
    with pytest.raises(ValueError, match="do not accept"):
        parse_provider({"type": "exa_mcp", **extra})


@pytest.mark.parametrize("mime", ["application/json", "text/event-stream"])
async def test_native_exa_preserves_full_query_and_filters_results(mime):
    search = build_web_tools(["web_search"], Settings(), None, None).search
    seen = []

    def handle(request):
        seen.append(request)
        assert request.url == "https://mcp.exa.ai/mcp"
        assert request.method == "POST"
        assert not {"authorization", "x-api-key", "cookie", "mcp-session-id"}.intersection(request.headers)
        payload = json.loads(request.content)
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "web_search_exa"
        arguments = payload["params"]["arguments"]
        assert arguments["query"] == '今日科技新闻 2026-09-13 "AI"'
        # Exa's public search cannot filter by domain, so it is asked for extra
        # candidates and the local filter keeps the allowed ones.
        assert arguments["numResults"] == 6
        assert "example.com" in arguments["objective"]
        raw = json.dumps(envelope(), ensure_ascii=False).encode()
        if mime == "text/event-stream":
            raw = b": heartbeat\r\n\r\nevent: message\r\ndata: " + raw + b"\r\n\r\n"
        return httpx.Response(200, headers={"Content-Type": mime}, stream=Chunks(raw[:80], raw[80:]))

    with patch(
        "chrys.service.tools.builtins.web.http.ValidatedAsyncTransport",
        autospec=True,
        return_value=httpx.MockTransport(handle),
    ):
        output = await search.web_search('今日科技新闻 2026-09-13 "AI"', num_results=2, allowed_domains=["example.com"])
    assert len(seen) == 1
    payload = json.loads(output.split("\nExternal search")[0])
    assert payload["provider"] == "exa"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["title"] == "中文科技新闻"
    assert "Published: 2026-09-13" in payload["results"][0]["snippet"]


def test_sse_multiline_json_and_progress():
    notification = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress", "params": {}})
    reply = json.dumps(envelope(), indent=2)
    raw = (f"data: {notification}\n\n" + "\n".join("data: " + line for line in reply.splitlines()) + "\n\n").encode()
    assert len(parse_results(rpc_result(raw, "text/event-stream"))) == 2


@pytest.mark.parametrize(
    "reply",
    [envelope(id=2), envelope(id=True), envelope(jsonrpc="1.0"), [], {}, envelope(result=None)],
)
def test_invalid_envelopes_are_protocol_errors(reply):
    with pytest.raises(WebError, match="protocol_error"):
        rpc_result(json.dumps(reply).encode(), "application/json")


def test_duplicate_sse_responses_are_rejected():
    raw = ("data: " + json.dumps(envelope()) + "\n\n").encode() * 2
    with pytest.raises(WebError, match="protocol_error"):
        rpc_result(raw, "text/event-stream")


@pytest.mark.parametrize(
    "reply,code",
    [
        (envelope(error={"code": -32603, "message": "private diagnostic"}), "exa_rpc_error"),
        (
            envelope(result={"isError": True, "content": [{"type": "text", "text": "private diagnostic"}]}),
            "exa_search_failed",
        ),
    ],
)
def test_remote_errors_are_not_returned_as_search_content(reply, code):
    with pytest.raises(WebError, match=code) as error:
        rpc_result(json.dumps(reply).encode(), "application/json")
    assert "private diagnostic" not in str(error.value)


@pytest.mark.parametrize("text", ["Please log in", "", "Title: Bad\nURL: javascript:alert(1)\nHighlights: bad"])
def test_unknown_text_and_invalid_urls_fail_closed(text):
    with pytest.raises(WebError, match="search_parse_failed"):
        parse_results({"content": [{"type": "text", "text": text}]})


def test_explicit_empty_content_is_empty():
    assert parse_results({"content": []}) == ()


@pytest.mark.parametrize("mime", ["application/json", "text/event-stream"])
async def test_exa_no_results_response_stops_without_retry_or_fallback(mime):
    """Exa reports an empty search as an unflagged text block: an answer, so the chain ends there."""
    config = parse_search(
        {
            "mode": "auto",
            "fallback_chain": ["exa", "backup"],
            "providers": {"exa": {"type": "exa_mcp"}, "backup": {"type": "bing_html"}},
        }
    )
    search = build_web_tools(["web_search"], Settings(), config, None).search
    assert search is not None
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        reply = envelope(
            result={"content": [{"type": "text", "text": "No search results found. Please try a different query."}]}
        )
        raw = json.dumps(reply).encode()
        if mime == "text/event-stream":
            raw = b"data: " + raw + b"\n\n"
        return httpx.Response(200, headers={"Content-Type": mime}, stream=Chunks(raw))

    with patch(
        "chrys.service.tools.builtins.web.http.ValidatedAsyncTransport",
        autospec=True,
        return_value=httpx.MockTransport(handle),
    ):
        output = await search.web_search("query with no results")
    assert [request.url.host for request in seen] == ["mcp.exa.ai"]
    assert json.loads(output.split("\nExternal search")[0]) == {
        "query": "query with no results",
        "provider": "exa",
        "results": [],
        "truncated": False,
    }


@pytest.mark.parametrize(
    "blocks",
    [
        [{"type": "text", "text": "No search results found. Please try a different query. Extra diagnostic"}],
        [
            {"type": "text", "text": "No search results found. Please try a different query."},
            {"type": "text", "text": _TEXT},
        ],
        [
            {"type": "text", "text": _TEXT},
            {"type": "text", "text": "No search results found. Please try a different query."},
        ],
    ],
)
def test_no_results_marker_does_not_hide_unknown_or_mixed_content(blocks):
    with pytest.raises(WebError, match="search_parse_failed"):
        parse_results({"content": blocks})


@pytest.mark.parametrize(
    "status,retries,error",
    [(429, 2, "http_4xx (HTTP 429)"), (503, 2, "http_5xx (HTTP 503)"), (401, 1, "http_4xx (HTTP 401)")],
)
async def test_default_exa_http_failure_is_bounded_and_reported_without_its_body(status, retries, error):
    """A fresh install's chain holds Exa alone, so its failure is the call's result."""
    search = build_web_tools(["web_search"], Settings(), None, None).search
    assert search is not None
    assert [provider.id for provider in search.providers] == ["exa"]
    seen = []

    def handle(request):
        seen.append(request)
        return httpx.Response(status, stream=Chunks(b"sensitive diagnostic"))

    with patch(
        "chrys.service.tools.builtins.web.http.ValidatedAsyncTransport",
        autospec=True,
        return_value=httpx.MockTransport(handle),
    ):
        output = await search.web_search("query")
    assert [request.url.host for request in seen] == ["mcp.exa.ai"] * retries
    assert output == f"Error: {error}"
    assert "sensitive diagnostic" not in output


@pytest.mark.parametrize("status,retries", [(429, 2), (503, 2), (401, 1)])
async def test_exa_http_failure_falls_back_to_the_next_configured_provider(status, retries):
    """Retryable failures are retried once; an anonymous refusal is not retried. Both then fall back."""
    config = parse_search(
        {
            "mode": "auto",
            "fallback_chain": ["exa", "backup"],
            "providers": {"exa": {"type": "exa_mcp"}, "backup": {"type": "bing_html"}},
        }
    )
    search = build_web_tools(["web_search"], Settings(), config, None).search
    assert search is not None
    seen = []

    def handle(request):
        seen.append(request)
        if request.url.host == "mcp.exa.ai":
            return httpx.Response(status, stream=Chunks(b"sensitive diagnostic"))
        body = (
            b'<ol id="b_results"><li class="b_algo"><h2><a href="https://example.com/a">Backup hit</a></h2>'
            b'<div class="b_caption"><p>From the fallback.</p></div></li></ol>'
        )
        return httpx.Response(200, headers={"Content-Type": "text/html"}, stream=Chunks(body))

    with patch(
        "chrys.service.tools.builtins.web.http.ValidatedAsyncTransport",
        autospec=True,
        return_value=httpx.MockTransport(handle),
    ):
        output = await search.web_search("query")
    assert [request.url.host for request in seen] == ["mcp.exa.ai"] * retries + ["www.bing.com"]
    payload = json.loads(output.split("\nExternal search")[0])
    assert payload["provider"] == "backup"
    assert [result["url"] for result in payload["results"]] == ["https://example.com/a"]
    assert "sensitive diagnostic" not in output


_EXA_REFUSALS = [
    pytest.param(
        {
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "You've hit Exa's free MCP rate limit. private diagnostic"}],
            }
        },
        "exa_search_failed",
        id="tool-error-result",
    ),
    pytest.param(
        {"error": {"code": -32603, "message": "private diagnostic"}},
        "exa_rpc_error",
        id="json-rpc-error",
    ),
]


def _exa_refusal_handler(changes, seen):
    def handle(request):
        seen.append(request)
        if request.url.host == "mcp.exa.ai":
            raw = json.dumps(envelope(**changes)).encode()
            return httpx.Response(200, headers={"Content-Type": "application/json"}, stream=Chunks(raw))
        body = (
            b'<ol id="b_results"><li class="b_algo"><h2><a href="https://example.com/a">Backup hit</a></h2>'
            b'<div class="b_caption"><p>From the fallback.</p></div></li></ol>'
        )
        return httpx.Response(200, headers={"Content-Type": "text/html"}, stream=Chunks(body))

    return handle


@pytest.mark.parametrize("changes,code", _EXA_REFUSALS)
async def test_exa_service_error_falls_back_to_the_next_configured_provider(changes, code):
    """Exa's rate limit and upstream errors arrive over HTTP 200; the next provider still gets the query."""
    config = parse_search(
        {
            "mode": "auto",
            "fallback_chain": ["exa", "backup"],
            "providers": {"exa": {"type": "exa_mcp"}, "backup": {"type": "bing_html"}},
        }
    )
    search = build_web_tools(["web_search"], Settings(), config, None).search
    assert search is not None
    seen = []
    with patch(
        "chrys.service.tools.builtins.web.http.ValidatedAsyncTransport",
        autospec=True,
        return_value=httpx.MockTransport(_exa_refusal_handler(changes, seen)),
    ):
        output = await search.web_search("query")
    # Not retried: an immediate second call would meet the same rate limit.
    assert [request.url.host for request in seen] == ["mcp.exa.ai", "www.bing.com"]
    payload = json.loads(output.split("\nExternal search")[0])
    assert payload["provider"] == "backup"
    assert [result["url"] for result in payload["results"]] == ["https://example.com/a"]
    assert "private diagnostic" not in output


@pytest.mark.parametrize("changes,code", _EXA_REFUSALS)
async def test_exa_service_error_is_the_result_when_exa_is_the_last_provider(changes, code):
    search = build_web_tools(["web_search"], Settings(), None, None).search
    assert search is not None
    seen = []
    with patch(
        "chrys.service.tools.builtins.web.http.ValidatedAsyncTransport",
        autospec=True,
        return_value=httpx.MockTransport(_exa_refusal_handler(changes, seen)),
    ):
        output = await search.web_search("query")
    assert [request.url.host for request in seen] == ["mcp.exa.ai"]
    assert output == f"Error: {code}"


async def test_exa_response_body_limit():
    search = build_web_tools(["web_search"], Settings(), None, None).search

    def handle(request):
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=Chunks(b"x" * (2 * 1024 * 1024 + 1))
        )

    with patch(
        "chrys.service.tools.builtins.web.http.ValidatedAsyncTransport",
        autospec=True,
        return_value=httpx.MockTransport(handle),
    ):
        output = await search.web_search("query")
    assert output.startswith("Error:")
    assert "response_too_large" in output
