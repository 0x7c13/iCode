# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tool calls a Chat Completions answer sends without a function name.

Such an answer is never run: response validation fails the turn with its
reason, and the request is never sent again.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from chrys.kernel import Message, ResponseStream, tool
from chrys.kernel.middleware import ChatMiddlewareLayer
from chrys.service.agent_middleware.response_validation import (
    ResponseValidationMiddleware,
    TerminalResponseValidationError,
)
from chrys.service.agent_middleware.validators import UNNAMED_TOOL_CALL_REASON
from chrys.service.llm.chat_completions import ChatCompletionsClient
from tests.support.scripted_wire import ScriptedWire
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer
from tests.support.wire_cases import Reply
from tests.support.wire_cases._kit import json_reply, sse_reply

_CREATED = 1_717_171_717


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": _CREATED,
        "model": "test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _streamed(*chunks: dict[str, Any], done: bool = True) -> Reply:
    events: list[tuple[str | None, Any]] = [(None, chunk) for chunk in chunks]
    if done:
        events.append((None, "[DONE]"))
    return sse_reply(events)


def _completion(message: dict[str, Any], finish_reason: str) -> Reply:
    return json_reply(
        {
            "id": "completion-1",
            "object": "chat.completion",
            "created": _CREATED,
            "model": "test",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        }
    )


def _call(index: int, call_id: str, arguments: str, name: str | None = None) -> dict[str, Any]:
    function: dict[str, Any] = {"arguments": arguments}
    if name is not None:
        function["name"] = name
    return {"index": index, "id": call_id, "type": "function", "function": function}


def _calls_streamed(*calls: dict[str, Any]) -> Reply:
    return _streamed(_chunk({"role": "assistant", "tool_calls": list(calls)}), _chunk({}, finish_reason="tool_calls"))


def _calls_completed(*calls: dict[str, Any]) -> Reply:
    message_calls = [{key: value for key, value in call.items() if key != "index"} for call in calls]
    return _completion({"role": "assistant", "content": None, "tool_calls": message_calls}, "tool_calls")


def _text(text: str, *, stream: bool) -> Reply:
    if stream:
        return _streamed(_chunk({"role": "assistant", "content": text}, finish_reason="stop"))
    return _completion({"role": "assistant", "content": text}, "stop")


def _read(path: str, *, stream: bool) -> Reply:
    call = _call(0, "call_read", f'{{"path": "{path}"}}', name="read_file")
    return _calls_streamed(call) if stream else _calls_completed(call)


@asynccontextmanager
async def _tool_loop(*replies: Reply) -> AsyncIterator[tuple[InvariantCheckedToolLoopLayer, ScriptedWire, list[str]]]:
    """A tool loop over *replies* with a ``read_file`` tool, and the paths it read."""
    reads: list[str] = []
    wire = ScriptedWire(replies)
    # No proxy from the environment may route around the scripted transport.
    http_client = httpx.AsyncClient(transport=wire.transport, trust_env=False)
    sdk_client = AsyncOpenAI(api_key="sk-test", base_url="https://api.test/v1", max_retries=0, http_client=http_client)
    client = ChatCompletionsClient(model="test", sdk_client=sdk_client)
    validation = ResponseValidationMiddleware()
    try:
        yield InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(client, middleware=[validation])), wire, reads
    finally:
        await client.aclose()


async def _run(layer: InvariantCheckedToolLoopLayer, reads: list[str], *, stream: bool) -> str:
    @tool
    def read_file(path: str) -> str:
        reads.append(path)
        return "contents"

    result = layer.get_response([Message("user", ["go"])], stream=stream, options={"tools": [read_file]})
    response = await result.get_final_response() if isinstance(result, ResponseStream) else await result
    return response.text


_NAMELESS = _call(0, "call_1", '{"path": "nameless"}')

_FIRST_ANSWERS = [
    pytest.param(_calls_streamed(_NAMELESS), True, id="streamed"),
    pytest.param(
        _calls_streamed(_call(0, "call_0", '{"path": "beside"}', name="read_file"), {**_NAMELESS, "index": 1}),
        True,
        id="streamed_beside_a_named_call",
    ),
    pytest.param(
        _calls_streamed(_NAMELESS, _call(0, "call_2", '{"path": "named"}', name="read_file")),
        True,
        id="streamed_before_a_named_call_reusing_its_index",
    ),
    pytest.param(
        _streamed(_chunk({"role": "assistant", "tool_calls": [_NAMELESS]})), True, id="streamed_without_finish_reason"
    ),
    pytest.param(
        _streamed(
            _chunk({"role": "assistant", "content": "Reading it."}),
            _chunk({"tool_calls": [_NAMELESS]}),
            _chunk({}, finish_reason="tool_calls"),
        ),
        True,
        id="streamed_after_text",
    ),
    pytest.param(_calls_completed(_NAMELESS), False, id="blocking"),
    pytest.param(
        _calls_completed(_call(0, "call_1", '{"path": "nameless"}', name="")), False, id="blocking_empty_name"
    ),
]


@pytest.mark.parametrize(("first", "stream"), _FIRST_ANSWERS)
async def test_an_answer_with_a_call_without_a_name_fails_the_turn_and_none_of_it_runs(
    first: Reply, stream: bool
) -> None:
    async with _tool_loop(first, _read("a", stream=stream)) as (layer, wire, reads):
        with pytest.raises(TerminalResponseValidationError) as raised:
            await _run(layer, reads, stream=stream)

    assert str(raised.value) == UNNAMED_TOOL_CALL_REASON
    assert raised.value.__cause__ is None
    assert reads == []
    assert len(wire.requests) == 1


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "blocking"])
async def test_a_nameless_call_cut_off_at_the_length_limit_is_dropped_and_the_answer_kept(stream: bool) -> None:
    if stream:
        cut_off = _streamed(
            _chunk({"role": "assistant", "content": "Partial"}),
            _chunk({"tool_calls": [_NAMELESS]}, finish_reason="length"),
        )
    else:
        nameless = {key: value for key, value in _NAMELESS.items() if key != "index"}
        cut_off = _completion({"role": "assistant", "content": "Partial", "tool_calls": [nameless]}, "length")

    async with _tool_loop(cut_off) as (layer, wire, reads):
        text = await _run(layer, reads, stream=stream)

    assert text == "Partial"
    assert (reads, len(wire.requests)) == ([], 1)
