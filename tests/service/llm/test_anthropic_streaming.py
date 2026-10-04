# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Regression tests for Anthropic streaming response assembly."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace

import pytest
from anthropic.types.beta import (
    BetaInputJSONDelta,
    BetaMCPToolUseBlock,
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

from chrys.foundation.errors import ProviderResponseError
from chrys.foundation.hosted_tools import HeldHostedEvidence
from chrys.kernel import ChatResponse, ChatResponseUpdate, Content, Message, ResponseStream
from chrys.service.llm.anthropic_messages import AnthropicMessagesClient
from chrys.service.llm.anthropic_messages.stream import StreamState


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
            self._events = [_message_stop()]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._events:
                raise StopAsyncIteration
            return self._events.pop(0)

        async def close(self) -> None:
            self.closed += 1

    sdk_stream = _SdkStream()

    class _Messages:
        async def create(self, **_kwargs: object):
            return sdk_stream

    client = AnthropicMessagesClient(
        model="claude-test",
        sdk_client=SimpleNamespace(
            base_url="https://api.anthropic.com", default_headers={}, beta=SimpleNamespace(messages=_Messages())
        ),  # type: ignore[arg-type]
    )
    response_stream = client._inner_get_response(
        messages=[Message("user", ["hi"])],
        options={},
        stream=True,
    )

    assert isinstance(response_stream, ResponseStream)
    assert [update async for update in response_stream] == []
    assert sdk_stream.closed == 1


def _open_stream(events: Sequence[BetaRawMessageStreamEvent]) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
    anthropic_client = SimpleNamespace(
        base_url="https://api.anthropic.com", default_headers={}, beta=SimpleNamespace(messages=_FakeMessages(events))
    )
    client = AnthropicMessagesClient(model="kimi-k3", sdk_client=anthropic_client)  # type: ignore[arg-type]
    stream = client._inner_get_response(
        messages=[Message("user", ["Explore the repository"])],
        options={},
        stream=True,
    )
    assert isinstance(stream, ResponseStream)
    return stream


async def _stream_response(
    events: Sequence[BetaRawMessageStreamEvent],
) -> tuple[list[ChatResponseUpdate], ChatResponse]:
    stream = _open_stream(events)
    updates = [update async for update in stream]
    response = await stream.get_final_response()
    return updates, response


def _function_calls(response: ChatResponse) -> list[Content]:
    return [content for message in response.messages for content in message.contents if content.type == "function_call"]


async def _stream_function_calls(events: Sequence[BetaRawMessageStreamEvent]) -> list[Content]:
    _, response = await _stream_response(events)
    return _function_calls(response)


_DEFAULT_BETA_HEADER = "mcp-client-2025-04-04,code-execution-2025-08-25"


@pytest.mark.asyncio
async def test_beta_options_are_written_into_the_anthropic_beta_header() -> None:
    messages_client = _FakeMessages([_message_stop()])
    anthropic_client = SimpleNamespace(
        base_url="https://api.anthropic.com", default_headers={}, beta=SimpleNamespace(messages=messages_client)
    )
    client = AnthropicMessagesClient(model="claude-test", sdk_client=anthropic_client)  # type: ignore[arg-type]
    stream = client._inner_get_response(
        messages=[Message("user", ["hi"])],
        options={"additional_beta_flags": ["x-beta"], "betas": ["y-beta"]},
        stream=True,
        additional_beta_flags=["must-not-leak"],
        betas=["must-not-leak-either"],
    )

    assert isinstance(stream, ResponseStream)
    assert [update async for update in stream] == []
    assert len(messages_client.calls) == 1
    request_kwargs = messages_client.calls[0]
    assert "betas" not in request_kwargs
    assert "additional_beta_flags" not in request_kwargs
    extra_headers = request_kwargs["extra_headers"]
    assert isinstance(extra_headers, dict)
    assert extra_headers["anthropic-beta"] == f"{_DEFAULT_BETA_HEADER},x-beta,y-beta"


@pytest.mark.asyncio
async def test_beta_options_from_chat_options_survive_public_get_response() -> None:
    """Model-profile ``chat_options`` enter through the public ``get_response``
    boundary (options mapping in, ``messages.create`` kwargs out): the betas
    are written into the ``anthropic-beta`` header and the raw keys never reach
    the provider call, whether carried in options or in client kwargs."""
    messages_client = _FakeMessages([_message_stop()])
    anthropic_client = SimpleNamespace(
        base_url="https://api.anthropic.com", default_headers={}, beta=SimpleNamespace(messages=messages_client)
    )
    client = AnthropicMessagesClient(model="claude-test", sdk_client=anthropic_client)  # type: ignore[arg-type]
    stream = client.get_response(
        [Message("user", ["hi"])],
        stream=True,
        options={"additional_beta_flags": ["x-beta"], "betas": ["y-beta"]},
        client_kwargs={"additional_beta_flags": ["must-not-leak"], "betas": ["must-not-leak-either"]},
    )

    assert [update async for update in stream] == []
    assert len(messages_client.calls) == 1
    request_kwargs = messages_client.calls[0]
    # kwargs-borne betas are filtered out, not folded (no real caller uses them).
    assert "betas" not in request_kwargs
    assert "additional_beta_flags" not in request_kwargs
    extra_headers = request_kwargs["extra_headers"]
    assert isinstance(extra_headers, dict)
    assert extra_headers["anthropic-beta"] == f"{_DEFAULT_BETA_HEADER},x-beta,y-beta"


def _tool_start(index: int, call_id: str, name: str) -> BetaRawContentBlockStartEvent:
    block = BetaToolUseBlock(type="tool_use", id=call_id, name=name, input={})
    return BetaRawContentBlockStartEvent(type="content_block_start", index=index, content_block=block)


def _text_start(index: int) -> BetaRawContentBlockStartEvent:
    block = BetaTextBlock(type="text", text="", citations=None)
    return BetaRawContentBlockStartEvent(type="content_block_start", index=index, content_block=block)


def _json_delta(index: int, partial_json: str) -> BetaRawContentBlockDeltaEvent:
    delta = BetaInputJSONDelta(type="input_json_delta", partial_json=partial_json)
    return BetaRawContentBlockDeltaEvent(type="content_block_delta", index=index, delta=delta)


def _text_delta(index: int, text: str) -> BetaRawContentBlockDeltaEvent:
    delta = BetaTextDelta(type="text_delta", text=text)
    return BetaRawContentBlockDeltaEvent(type="content_block_delta", index=index, delta=delta)


def _block_stop(index: int) -> BetaRawContentBlockStopEvent:
    return BetaRawContentBlockStopEvent(type="content_block_stop", index=index)


def _message_stop() -> BetaRawMessageStopEvent:
    return BetaRawMessageStopEvent(type="message_stop")


def _message_delta(stop_reason: str | None, *, output_tokens: int) -> BetaRawMessageDeltaEvent:
    return BetaRawMessageDeltaEvent(
        type="message_delta",
        delta={"stop_reason": stop_reason, "stop_sequence": None},  # type: ignore[typeddict-item]
        usage=BetaMessageDeltaUsage(output_tokens=output_tokens),
    )


def _tool_use_tail(output_tokens: int) -> list[BetaRawMessageStreamEvent]:
    """The closing events of a message that stopped to call tools."""
    return [_message_delta("tool_use", output_tokens=output_tokens), _message_stop()]


def _mcp_start(index: int, call_id: str) -> BetaRawContentBlockStartEvent:
    block = BetaMCPToolUseBlock(type="mcp_tool_use", id=call_id, name="search", server_name="docs", input={})
    return BetaRawContentBlockStartEvent(type="content_block_start", index=index, content_block=block)


async def _read_until_failure(
    events: Sequence[BetaRawMessageStreamEvent],
) -> tuple[list[ChatResponseUpdate], ProviderResponseError]:
    updates: list[ChatResponseUpdate] = []
    with pytest.raises(ProviderResponseError) as raised:
        async for update in _open_stream(events):
            updates.append(update)
    return updates, raised.value


def _sent_calls(updates: Sequence[ChatResponseUpdate]) -> list[Content]:
    return [content for update in updates for content in update.contents if content.type == "function_call"]


@pytest.mark.parametrize(
    ("tail", "output_tokens"),
    [([], None), ([_message_delta(None, output_tokens=7)], 7)],
    ids=["eof", "delta_without_stop_reason"],
)
async def test_a_stream_that_ends_before_saying_how_its_message_ends_is_a_retryable_truncation(
    tail: list[BetaRawMessageStreamEvent], output_tokens: int | None
) -> None:
    """The held call may be cut short: it is never sent, and the next attempt asks again."""
    events = [_tool_start(0, "call-a", "zsh"), _json_delta(0, '{"command":"rm -rf bu'), *tail]

    updates, failure = await _read_until_failure(events)

    assert (failure.code, failure.retryable) == ("stream_truncated", True)
    assert (failure.usage_details or {}).get("output_token_count") == output_tokens
    assert _sent_calls(updates) == []


async def test_a_message_stop_alone_ends_the_stream() -> None:
    events = [_tool_start(0, "call-a", "zsh"), _json_delta(0, '{"command":"ls"}'), _message_stop()]

    assert [call.call_id for call in await _stream_function_calls(events)] == ["call-a"]


async def test_a_message_refused_with_calls_raises_before_any_call_is_sent() -> None:
    events = [
        _tool_start(0, "call-a", "zsh"),
        _json_delta(0, '{"command":"ls"}'),
        _block_stop(0),
        _message_delta("refusal", output_tokens=9),
        _message_stop(),
    ]

    updates, failure = await _read_until_failure(events)

    assert (failure.code, failure.retryable) == ("content_filter", False)
    assert (failure.usage_details or {}).get("output_token_count") == 9
    assert _sent_calls(updates) == []


async def test_a_message_refused_without_calls_ends_as_filtered() -> None:
    events = [
        _text_start(0),
        _text_delta(0, "I can't help."),
        _block_stop(0),
        _message_delta("refusal", output_tokens=4),
    ]

    _, response = await _stream_response([*events, _message_stop()])

    assert response.finish_reason == "content_filter"
    assert response.text == "I can't help."


def test_hosted_work_held_behind_a_call_is_reported_once_as_it_arrives() -> None:
    """The retry gates count it at once; the contents still go out only in block order."""
    state = StreamState()
    events = [
        _tool_start(0, "call-a", "zsh"),
        _json_delta(0, '{"command":"ls"}'),
        _mcp_start(1, "mcptoolu_1"),
        _json_delta(1, '{"q":'),
        _json_delta(1, '"x"}'),
        _block_stop(1),
        _block_stop(0),
    ]

    held = [update for event in events for update in state.updates_for(event)]

    assert [update.contents for update in held if update.contents] == []
    evidence = [
        update.raw_representation for update in held if isinstance(update.raw_representation, HeldHostedEvidence)
    ]
    assert [[content.call_id for content in item.contents] for item in evidence] == [["mcptoolu_1"]]  # type: ignore[attr-defined]

    released = [update for event in _tool_use_tail(output_tokens=5) for update in state.updates_for(event)]
    state.finish()
    contents = ChatResponse.from_updates([*held, *released]).messages[0].contents
    assert [(content.type, content.call_id) for content in contents] == [
        ("function_call", "call-a"),
        ("mcp_server_tool_call", "mcptoolu_1"),
    ]
    assert contents[1] is evidence[0].contents[0]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_interleaved_parallel_tool_calls_are_assembled_by_content_block_index() -> None:
    """Kimi K3 may interleave Anthropic input deltas from parallel tool-use blocks."""
    events: list[BetaRawMessageStreamEvent] = [
        _tool_start(0, "call-a", "zsh"),
        _tool_start(1, "call-b", "glob"),
        _json_delta(0, '{"command":'),
        _json_delta(1, '{"pattern":'),
        _json_delta(0, '"ls"}'),
        _block_stop(0),
        _json_delta(1, '"*.py"}'),
        _block_stop(1),
        _message_stop(),
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
        _tool_start(0, "call-a", "zsh"),
        _tool_start(1, "call-b", "glob"),
        _json_delta(0, '{"command":"ls"}'),
        _json_delta(1, '{"pattern":"*.py"}'),
        _block_stop(1),
        _block_stop(0),
        _message_stop(),
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
        _tool_start(0, "call-a", "zsh"),
        _json_delta(0, '{"command":"ls"}'),
        _block_stop(0),
        *_tool_use_tail(output_tokens=12),
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
        _tool_start(0, "call-a", "zsh"),
        _json_delta(0, '{"command":"ls"}'),
        _block_stop(0),
        _text_start(1),
        _text_delta(1, "Done"),
        _block_stop(1),
        *_tool_use_tail(output_tokens=16),
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
