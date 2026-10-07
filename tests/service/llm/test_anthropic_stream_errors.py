# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Errors an Anthropic stream reports after its 200, through the main turn's wire lane."""

from __future__ import annotations

import json

import httpx
import pytest

from chrys.foundation.events.types import Error, InvocationMessage
from tests.support.mock_provider_turns import mock_provider_profile, run_mock_provider_turn
from tests.support.provider_errors import anthropic_error_event, anthropic_thinking_binding_body
from tests.support.wire_cases._kit import anth_message, anth_replies, anth_text


def _sse(*events: dict[str, object]) -> bytes:
    return b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


_MESSAGE_START = {
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
}

_TEXT_STREAM = _sse(
    _MESSAGE_START,
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


def _local_call_cut_off(*hosted: dict[str, object]) -> bytes:
    """A stream that ends without its message delta or stop, behind a local call."""
    return _sse(
        _MESSAGE_START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "zsh", "input": {}},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"comm'}},
        *hosted,
    )


def _respond_with(bodies: list[bytes]):
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(bodies.pop(0))
        )

    return respond


async def test_a_stream_cut_off_behind_a_call_is_sent_again_without_running_it(
    agent_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    turn = await run_mock_provider_turn(
        agent_engine,
        monkeypatch,
        mock_provider_profile("anthropic", stream=True),
        _respond_with([_local_call_cut_off(), _TEXT_STREAM]),
    )

    assert len(turn.requests) == 2
    assert [retry.scope for retry in turn.retries] == ["wire"]
    assert b"tool_result" not in turn.requests[1].content
    assert isinstance(turn.terminal, InvocationMessage)
    assert turn.terminal.text == "ok"


async def test_a_stream_cut_off_after_hosted_work_behind_a_call_is_not_sent_again(
    agent_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The held MCP call already ran on the provider: sending the request again would run it twice."""
    mcp = (
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "mcp_tool_use",
                "id": "mcptoolu_1",
                "name": "deploy",
                "server_name": "ops",
                "input": {},
            },
        },
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": 1},
    )
    turn = await run_mock_provider_turn(
        agent_engine,
        monkeypatch,
        mock_provider_profile("anthropic", stream=True),
        _respond_with([_local_call_cut_off(*mcp), _TEXT_STREAM]),
    )

    assert len(turn.requests) == 1
    assert turn.retries == []
    assert isinstance(turn.terminal, Error)


@pytest.mark.parametrize("stream", [False, True], ids=["send", "stream"])
async def test_thinking_refused_as_bound_elsewhere_is_resent_without_it_and_no_retry(
    stream: bool, agent_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client resends the refused request itself: the turn sees one answer and announces no retry."""
    call = anth_message(
        message_id="msg_call",
        content=[
            {"type": "thinking", "thinking": "Run it.", "signature": "sig-call"},
            {"type": "tool_use", "id": "toolu_1", "name": "zsh", "input": {}},
        ],
        stop_reason="tool_use",
    )
    first, done = anth_replies([call, anth_text("done", message_id="msg_done")], stream=stream)
    responses = [
        httpx.Response(first.status, headers=list(first.headers), content=first.body),
        httpx.Response(400, json=anthropic_thinking_binding_body()),
        httpx.Response(done.status, headers=list(done.headers), content=done.body),
    ]

    turn = await run_mock_provider_turn(
        agent_engine, monkeypatch, mock_provider_profile("anthropic", stream=stream), lambda _: responses.pop(0)
    )

    assert len(turn.requests) == 3
    assert turn.retries == []
    assert isinstance(turn.terminal, InvocationMessage)
    assert turn.terminal.text == "done"
    refused, resent = (json.loads(request.content) for request in turn.requests[1:])
    assert [block["type"] for block in refused["messages"][1]["content"]] == ["thinking", "tool_use"]
    assert [block["type"] for block in resent["messages"][1]["content"]] == ["tool_use"]
    assert resent["messages"][2] == refused["messages"][2]
    assert resent["messages"][2]["content"][0]["type"] == "tool_result"
