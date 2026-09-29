# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Errors an Anthropic stream reports after its 200, through the main turn's wire lane."""

from __future__ import annotations

import json

import httpx
import pytest

from chrys.foundation.events.types import Error, InvocationMessage
from tests.support.mock_provider_turns import mock_provider_profile, run_mock_provider_turn
from tests.support.provider_errors import anthropic_error_event


def _sse(*events: dict[str, object]) -> bytes:
    return b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


_TEXT_STREAM = _sse(
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "test-model",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 2},
    },
    {"type": "message_stop"},
)


def _error_stream(error_type: str) -> bytes:
    return f"event: error\ndata: {anthropic_error_event(error_type, 'stream failed')}\n\n".encode()


@pytest.mark.parametrize(("error_type", "retried"), [("overloaded_error", True), ("invalid_request_error", False)])
async def test_post_200_stream_error_retries_only_when_transient(
    error_type: str, retried: bool, agent_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = [_error_stream(error_type), _TEXT_STREAM]

    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(bodies.pop(0))
        )

    turn = await run_mock_provider_turn(
        agent_engine, monkeypatch, mock_provider_profile("anthropic", stream=True), respond
    )

    if retried:
        assert len(turn.requests) == 2
        assert [retry.scope for retry in turn.retries] == ["wire"]
        assert isinstance(turn.terminal, InvocationMessage)
        assert turn.terminal.text == "ok"
    else:
        assert len(turn.requests) == 1
        assert turn.retries == []
        assert isinstance(turn.terminal, Error)
