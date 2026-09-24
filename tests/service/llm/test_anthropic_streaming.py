# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Regression tests for Anthropic streaming response assembly."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace

import pytest
from anthropic.types.beta import (
    BetaInputJSONDelta,
    BetaMessageDeltaUsage,
    BetaRawContentBlockDeltaEvent,
    BetaRawContentBlockStartEvent,
    BetaRawContentBlockStopEvent,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStopEvent,
    BetaRawMessageStreamEvent,
    BetaTextBlock,
    BetaTextDelta,
    BetaToolUseBlock,
)

from chrys.kernel import ChatResponse, ChatResponseUpdate, Content, Message, ResponseStream
from chrys.service.llm.anthropic_chat import RawAnthropicClient


class _FakeMessages:
    def __init__(self, events: Sequence[BetaRawMessageStreamEvent]) -> None:
        self._events = events
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> AsyncIterator[BetaRawMessageStreamEvent]:
        self.calls.append(dict(kwargs))

        async def _stream() -> AsyncIterator[BetaRawMessageStreamEvent]:
            for event in self._events:
                yield event

        return _stream()


@pytest.mark.asyncio
async def test_anthropic_adapter_closes_sdk_stream_after_consumption() -> None:
    class _SdkStream:
        def __init__(self) -> None:
            self.closed = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self) -> None:
            self.closed += 1

    sdk_stream = _SdkStream()

    class _Messages:
        async def create(self, **_kwargs: object):
            return sdk_stream

    client = RawAnthropicClient(
        model="claude-test",
        anthropic_client=SimpleNamespace(beta=SimpleNamespace(messages=_Messages())),  # type: ignore[arg-type]
    )
    response_stream = client._inner_get_response(
        messages=[Message("user", ["hi"])],
        options={},
        stream=True,
    )

    assert isinstance(response_stream, ResponseStream)
    assert [update async for update in response_stream] == []
    assert sdk_stream.closed == 1


async def _stream_response(
    events: Sequence[BetaRawMessageStreamEvent],
) -> tuple[list[ChatResponseUpdate], ChatResponse]:
    anthropic_client = SimpleNamespace(beta=SimpleNamespace(messages=_FakeMessages(events)))
    client = RawAnthropicClient(model="kimi-k3", anthropic_client=anthropic_client)  # type: ignore[arg-type]
    stream = client._inner_get_response(
        messages=[Message("user", ["Explore the repository"])],
        options={},
        stream=True,
    )

    assert isinstance(stream, ResponseStream)
    updates = [update async for update in stream]
    response = await stream.get_final_response()
    return updates, response


def _function_calls(response: ChatResponse) -> list[Content]:
    return [content for message in response.messages for content in message.contents if content.type == "function_call"]


async def _stream_function_calls(events: Sequence[BetaRawMessageStreamEvent]) -> list[Content]:
    _, response = await _stream_response(events)
    return _function_calls(response)


@pytest.mark.asyncio
async def test_additional_beta_flags_are_forwarded_only_in_betas() -> None:
    messages_client = _FakeMessages([])
    anthropic_client = SimpleNamespace(beta=SimpleNamespace(messages=messages_client))
    client = RawAnthropicClient(model="claude-test", anthropic_client=anthropic_client)  # type: ignore[arg-type]
    stream = client._inner_get_response(
        messages=[Message("user", ["hi"])],
        options={"additional_beta_flags": ["x-beta"]},
        stream=True,
        additional_beta_flags=["must-not-leak"],
    )

    assert isinstance(stream, ResponseStream)
    assert [update async for update in stream] == []
    assert len(messages_client.calls) == 1
    request_kwargs = messages_client.calls[0]
    betas = request_kwargs["betas"]
    assert isinstance(betas, set)
    assert "x-beta" in betas
    assert "additional_beta_flags" not in request_kwargs


@pytest.mark.asyncio
async def test_additional_beta_flags_from_chat_options_survive_public_get_response() -> None:
    """Model-profile ``chat_options`` enter through the public ``get_response``
    boundary (options mapping in, ``messages.create`` kwargs out): the flag is
    folded into ``betas`` and the raw key never reaches the provider call,
    whether carried in options or in client kwargs."""
    messages_client = _FakeMessages([])
    anthropic_client = SimpleNamespace(beta=SimpleNamespace(messages=messages_client))
    client = RawAnthropicClient(model="claude-test", anthropic_client=anthropic_client)  # type: ignore[arg-type]
    stream = client.get_response(
        [Message("user", ["hi"])],
        stream=True,
        options={"additional_beta_flags": ["x-beta"]},
        client_kwargs={"additional_beta_flags": ["must-not-leak"]},
    )

    assert [update async for update in stream] == []
    assert len(messages_client.calls) == 1
    request_kwargs = messages_client.calls[0]
    betas = request_kwargs["betas"]
    assert isinstance(betas, set)
    assert "x-beta" in betas
    # kwargs-borne flags are filtered out, not folded (no real caller uses them).
    assert "must-not-leak" not in betas
    assert "additional_beta_flags" not in request_kwargs


@pytest.mark.asyncio
async def test_interleaved_parallel_tool_calls_are_assembled_by_content_block_index() -> None:
    """Kimi K3 may interleave Anthropic input deltas from parallel tool-use blocks."""
    events: list[BetaRawMessageStreamEvent] = [
        BetaRawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=BetaToolUseBlock(type="tool_use", id="call-a", name="zsh", input={}),
        ),
        BetaRawContentBlockStartEvent(
            type="content_block_start",
            index=1,
            content_block=BetaToolUseBlock(type="tool_use", id="call-b", name="glob", input={}),
        ),
        BetaRawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=BetaInputJSONDelta(type="input_json_delta", partial_json='{"command":'),
        ),
        BetaRawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=BetaInputJSONDelta(type="input_json_delta", partial_json='{"pattern":'),
        ),
        BetaRawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=BetaInputJSONDelta(type="input_json_delta", partial_json='"ls"}'),
        ),
        BetaRawContentBlockStopEvent(type="content_block_stop", index=0),
        BetaRawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=BetaInputJSONDelta(type="input_json_delta", partial_json='"*.py"}'),
        ),
        BetaRawContentBlockStopEvent(type="content_block_stop", index=1),
        BetaRawMessageStopEvent(type="message_stop"),
    ]
    calls = await _stream_function_calls(events)

    assert [(call.call_id, call.name, call.parse_arguments()) for call in calls] == [
        ("call-a", "zsh", {"command": "ls"}),
        ("call-b", "glob", {"pattern": "*.py"}),
    ]
    assert all(call.name for call in calls)


@pytest.mark.asyncio
async def test_parallel_tool_calls_preserve_block_order_when_stops_are_reversed() -> None:
    """Tool execution order follows content-block indices, not interleaved stop-event order."""
    events: list[BetaRawMessageStreamEvent] = [
        BetaRawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=BetaToolUseBlock(type="tool_use", id="call-a", name="zsh", input={}),
        ),
        BetaRawContentBlockStartEvent(
            type="content_block_start",
            index=1,
            content_block=BetaToolUseBlock(type="tool_use", id="call-b", name="glob", input={}),
        ),
        BetaRawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=BetaInputJSONDelta(type="input_json_delta", partial_json='{"command":"ls"}'),
        ),
        BetaRawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=BetaInputJSONDelta(type="input_json_delta", partial_json='{"pattern":"*.py"}'),
        ),
        BetaRawContentBlockStopEvent(type="content_block_stop", index=1),
        BetaRawContentBlockStopEvent(type="content_block_stop", index=0),
        BetaRawMessageStopEvent(type="message_stop"),
    ]

    calls = await _stream_function_calls(events)

    assert [(call.call_id, call.name) for call in calls] == [
        ("call-a", "zsh"),
        ("call-b", "glob"),
    ]


@pytest.mark.asyncio
async def test_tool_calls_are_emitted_before_terminal_message_delta() -> None:
    """The standard Anthropic tail exposes calls before its terminal finish update."""
    events: list[BetaRawMessageStreamEvent] = [
        BetaRawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=BetaToolUseBlock(type="tool_use", id="call-a", name="zsh", input={}),
        ),
        BetaRawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=BetaInputJSONDelta(type="input_json_delta", partial_json='{"command":"ls"}'),
        ),
        BetaRawContentBlockStopEvent(type="content_block_stop", index=0),
        BetaRawMessageDeltaEvent(
            type="message_delta",
            delta={"stop_reason": "tool_use", "stop_sequence": None},
            usage=BetaMessageDeltaUsage(output_tokens=12),
        ),
        BetaRawMessageStopEvent(type="message_stop"),
    ]

    updates, response = await _stream_response(events)
    calls = _function_calls(response)
    call_update_index = next(
        index
        for index, update in enumerate(updates)
        if any(content.type == "function_call" for content in update.contents)
    )
    finish_update_index = next(index for index, update in enumerate(updates) if update.finish_reason == "tool_calls")

    assert call_update_index < finish_update_index
    assert [(call.call_id, call.name, call.parse_arguments()) for call in calls] == [
        ("call-a", "zsh", {"command": "ls"})
    ]


@pytest.mark.asyncio
async def test_local_tool_call_preserves_order_before_later_text_block() -> None:
    """Deferred local calls retain their indexed position relative to later content blocks."""
    events: list[BetaRawMessageStreamEvent] = [
        BetaRawContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=BetaToolUseBlock(type="tool_use", id="call-a", name="zsh", input={}),
        ),
        BetaRawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=BetaInputJSONDelta(type="input_json_delta", partial_json='{"command":"ls"}'),
        ),
        BetaRawContentBlockStopEvent(type="content_block_stop", index=0),
        BetaRawContentBlockStartEvent(
            type="content_block_start",
            index=1,
            content_block=BetaTextBlock(type="text", text="", citations=None),
        ),
        BetaRawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=BetaTextDelta(type="text_delta", text="Done"),
        ),
        BetaRawContentBlockStopEvent(type="content_block_stop", index=1),
        BetaRawMessageDeltaEvent(
            type="message_delta",
            delta={"stop_reason": "tool_use", "stop_sequence": None},
            usage=BetaMessageDeltaUsage(output_tokens=16),
        ),
        BetaRawMessageStopEvent(type="message_stop"),
    ]

    updates, response = await _stream_response(events)
    contents = response.messages[0].contents
    tool_boundary_updates = [
        update for update in updates if any(content.type == "function_call" for content in update.contents)
    ]

    assert [content.type for content in contents] == ["function_call", "text"]
    assert contents[1].text == "Done"
    assert len(tool_boundary_updates) == 1
    assert any(content.type == "text" and content.text == "Done" for content in tool_boundary_updates[0].contents)
