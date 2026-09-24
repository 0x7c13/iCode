# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for shared chat-response helpers."""

from __future__ import annotations

import asyncio

import pytest

from chrys.kernel import Message
from chrys.service.llm.responses import get_final_response


class _Response:
    text = "done"


class _HangingStream:
    def __init__(self) -> None:
        self.cleanup_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(10)
        raise AssertionError("unreachable after timeout")

    async def get_final_response(self):
        raise AssertionError("final response should not be reached")

    async def aclose(self) -> None:
        self.cleanup_calls += 1


class _StreamingClient:
    def __init__(self) -> None:
        self.stream = _HangingStream()

    async def get_response(self, _messages, *, stream=False, **_kwargs):
        assert stream is True
        return self.stream


class _BlockingClient:
    def __init__(self) -> None:
        self.streams: list[bool] = []

    async def get_response(self, _messages, *, stream=False, **_kwargs):
        self.streams.append(stream)
        assert stream is False
        return _Response()


@pytest.mark.asyncio
async def test_get_final_response_returns_blocking_response_without_timeout() -> None:
    client = _BlockingClient()
    response = await get_final_response(
        client,
        [Message("user", ["hello"])],
        stream=False,
        timeout=0.01,
    )

    assert response.text == "done"
    assert client.streams == [False]


@pytest.mark.asyncio
async def test_get_final_response_times_out_hung_stream_and_runs_cleanup() -> None:
    client = _StreamingClient()

    with pytest.raises(TimeoutError, match=r"LLM stream update timed out after 0\.01s"):
        await get_final_response(
            client,
            [Message("user", ["hello"])],
            stream=True,
            timeout=0.01,
        )

    assert client.stream.cleanup_calls == 1
