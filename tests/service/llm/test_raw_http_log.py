# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for raw LLM HTTP request/response logging."""

from __future__ import annotations

import asyncio
import json

import pytest

from chrys.foundation.config.settings import SESSION_ROOT_DIR_ENV_VAR
from chrys.foundation.util.chrys_headers import X_SESSION_ID_HEADER
from chrys.foundation.util.session_ids import session_short_id
from chrys.service.llm.raw_http_log import (
    RAW_HTTP_LOG_ENV,
    RAW_HTTP_LOG_FILE,
    build_raw_http_event_hooks,
    raw_http_log_path,
    raw_http_logging_enabled,
)
from chrys.service.profiles.models.schema import ModelProfile


def _profile() -> ModelProfile:
    return ModelProfile(id="profile-1", name="Debug Profile", provider="openai", model_id="gpt-debug")


def test_raw_http_logging_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RAW_HTTP_LOG_ENV, raising=False)

    assert RAW_HTTP_LOG_ENV == "CHRYS_DEBUG_LLM_RAW_HTTP_LOG"
    assert raw_http_logging_enabled() is False
    assert raw_http_log_path("sess-1", None) is None


def test_raw_http_log_path_uses_session_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv(RAW_HTTP_LOG_ENV, "1")

    assert raw_http_log_path("sess-1", tmp_path) == tmp_path / RAW_HTTP_LOG_FILE


def test_raw_http_log_path_uses_session_root_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    custom_root = tmp_path / "custom-root"
    monkeypatch.setenv(RAW_HTTP_LOG_ENV, "1")
    monkeypatch.setenv(SESSION_ROOT_DIR_ENV_VAR, str(custom_root))

    assert (
        raw_http_log_path("sess-1", None) == custom_root / "sessions" / session_short_id("sess-1") / RAW_HTTP_LOG_FILE
    )


@pytest.mark.asyncio
async def test_raw_http_event_hooks_write_full_request_and_response(tmp_path) -> None:
    import httpx

    log_path = tmp_path / RAW_HTTP_LOG_FILE
    hooks = build_raw_http_event_hooks(log_path=log_path, profile=_profile(), session_id="sess-raw")

    async def _handler(request: httpx.Request) -> httpx.Response:
        assert await request.aread() == b'{"prompt":"hello"}'
        return httpx.Response(
            200,
            headers={"X-Request-Id": "resp-123", "Content-Type": "application/json"},
            content=b'{"ok":true}',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler), event_hooks=hooks) as client:
        response = await client.post(
            "https://example.test/v1/chat/completions",
            headers={"Authorization": "Bearer raw-secret"},
            content=b'{"prompt":"hello"}',
        )

    assert response.text == '{"ok":true}'
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["request", "response"]
    assert records[0]["exchange_id"] == records[1]["exchange_id"]
    assert records[0]["session_id"] == "sess-raw"
    assert records[0]["model_profile"] == {
        "id": "profile-1",
        "name": "Debug Profile",
        "provider": "openai",
        "model_id": "gpt-debug",
    }

    request_headers = dict(records[0]["request"]["headers"])
    assert request_headers["authorization"] == "Bearer raw-secret"
    assert records[0]["request"]["method"] == "POST"
    assert records[0]["request"]["url"] == "https://example.test/v1/chat/completions"
    assert records[0]["request"]["body"] == {
        "size_bytes": 18,
        "encoding": "utf-8",
        "json": {"prompt": "hello"},
    }

    response_headers = dict(records[1]["response"]["headers"])
    assert response_headers["x-request-id"] == "resp-123"
    assert records[1]["response"]["status_code"] == 200
    assert records[1]["response"]["body"] == {
        "size_bytes": 11,
        "encoding": "utf-8",
        "json": {"ok": True},
    }


@pytest.mark.asyncio
async def test_raw_http_event_hooks_use_wire_session_header_for_log_session_id(tmp_path) -> None:
    import httpx

    log_path = tmp_path / RAW_HTTP_LOG_FILE
    hooks = build_raw_http_event_hooks(log_path=log_path, profile=_profile(), session_id="client-session")

    async def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"ok":true}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler), event_hooks=hooks) as client:
        await client.post(
            "https://example.test/v1/chat/completions",
            headers={X_SESSION_ID_HEADER: "wire-session"},
            content=b"{}",
        )

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["session_id"] for record in records] == ["wire-session", "wire-session"]


@pytest.mark.asyncio
async def test_raw_http_event_hooks_use_wire_session_header_for_streaming_response(tmp_path) -> None:
    import httpx

    class _TwoChunkStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"hello "
            yield b"world"

    log_path = tmp_path / RAW_HTTP_LOG_FILE
    hooks = build_raw_http_event_hooks(log_path=log_path, profile=_profile(), session_id="client-session")

    async def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_TwoChunkStream())

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(_handler), event_hooks=hooks) as client,
        client.stream("GET", "https://example.test/stream", headers={X_SESSION_ID_HEADER: "wire-session"}) as response,
    ):
        _ = [chunk async for chunk in response.aiter_bytes()]

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    # The streaming response record is appended lazily as bytes are consumed; it
    # must still carry the per-request wire session id, not the client default.
    assert [record["event"] for record in records] == ["request", "response"]
    assert [record["session_id"] for record in records] == ["wire-session", "wire-session"]


@pytest.mark.asyncio
async def test_raw_http_event_hooks_keep_concurrent_request_session_ids_distinct(tmp_path) -> None:
    import httpx

    log_path = tmp_path / RAW_HTTP_LOG_FILE
    hooks = build_raw_http_event_hooks(log_path=log_path, profile=_profile(), session_id="client-session")

    async def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"ok":true}')

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler), event_hooks=hooks) as client:
        await asyncio.gather(
            client.post(
                "https://example.test/v1/chat/completions",
                headers={X_SESSION_ID_HEADER: "route-a"},
                content=b"{}",
            ),
            client.post(
                "https://example.test/v1/chat/completions",
                headers={X_SESSION_ID_HEADER: "route-b"},
                content=b"{}",
            ),
        )

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    # Correlation is per-``httpx.Request`` (via request.extensions), so two
    # concurrent requests sharing one client must not cross-contaminate: each
    # request+response pair shares exactly one wire session id, and the two
    # pairs carry the two distinct ids.
    session_ids_by_exchange: dict[str, set[str]] = {}
    for record in records:
        session_ids_by_exchange.setdefault(record["exchange_id"], set()).add(record["session_id"])

    assert len(session_ids_by_exchange) == 2
    assert all(len(ids) == 1 for ids in session_ids_by_exchange.values())
    assert {next(iter(ids)) for ids in session_ids_by_exchange.values()} == {"route-a", "route-b"}


@pytest.mark.asyncio
async def test_raw_http_event_hooks_preserve_non_utf8_bodies(tmp_path) -> None:
    import httpx

    log_path = tmp_path / RAW_HTTP_LOG_FILE
    hooks = build_raw_http_event_hooks(log_path=log_path, profile=_profile(), session_id="sess-raw")

    async def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\xff\x00")

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler), event_hooks=hooks) as client:
        await client.post("https://example.test/binary", content=b"\xfe")

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["request"]["body"] == {"size_bytes": 1, "encoding": "base64", "base64": "/g=="}
    assert records[1]["response"]["body"] == {"size_bytes": 2, "encoding": "base64", "base64": "/wA="}


@pytest.mark.asyncio
async def test_raw_http_event_hooks_do_not_preconsume_streaming_response(tmp_path) -> None:
    import httpx

    class _TwoChunkStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"hello "
            yield b"world"

    log_path = tmp_path / RAW_HTTP_LOG_FILE
    hooks = build_raw_http_event_hooks(log_path=log_path, profile=_profile(), session_id="sess-raw")

    async def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_TwoChunkStream())

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(_handler), event_hooks=hooks) as client,
        client.stream("GET", "https://example.test/stream") as response,
    ):
        chunks = [chunk async for chunk in response.aiter_bytes()]

    assert chunks == [b"hello ", b"world"]
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["request", "response"]
    assert records[1]["response"]["body"] == {
        "size_bytes": 11,
        "encoding": "utf-8",
        "text": "hello world",
    }


@pytest.mark.asyncio
async def test_raw_http_event_hooks_do_not_classify_plain_event_line_as_sse(tmp_path) -> None:
    import httpx

    payload = b"event: deprecated_api\n"
    log_path = tmp_path / RAW_HTTP_LOG_FILE
    hooks = build_raw_http_event_hooks(log_path=log_path, profile=_profile(), session_id="sess-raw")

    async def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler), event_hooks=hooks) as client:
        await client.get("https://example.test/error")

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records[1]["response"]["body"] == {
        "size_bytes": len(payload),
        "encoding": "utf-8",
        "text": payload.decode("utf-8"),
    }


@pytest.mark.asyncio
async def test_raw_http_event_hooks_expand_sse_json_data(tmp_path) -> None:
    import httpx

    class _SseStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"id":"chunk-1","choices":[{"delta":{"content":"Hi"}}]}\n\n'
            yield b"event: usage\n"
            yield b'data: {"total_tokens":12}\n\n'
            yield b"data: [DONE]\n\n"

    log_path = tmp_path / RAW_HTTP_LOG_FILE
    hooks = build_raw_http_event_hooks(log_path=log_path, profile=_profile(), session_id="sess-raw")

    async def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=_SseStream())

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(_handler), event_hooks=hooks) as client,
        client.stream("GET", "https://example.test/stream") as response,
    ):
        chunks = [chunk async for chunk in response.aiter_bytes()]

    assert chunks == [
        b'data: {"id":"chunk-1","choices":[{"delta":{"content":"Hi"}}]}\n\n',
        b"event: usage\n",
        b'data: {"total_tokens":12}\n\n',
        b"data: [DONE]\n\n",
    ]
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records[1]["response"]["body"] == {
        "size_bytes": sum(len(chunk) for chunk in chunks),
        "encoding": "utf-8",
        "sse": [
            {
                "data_json": {
                    "id": "chunk-1",
                    "choices": [{"delta": {"content": "Hi"}}],
                }
            },
            {"event": "usage", "data_json": {"total_tokens": 12}},
            {"data": "[DONE]"},
        ],
    }
