# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Reasoning-dialect capture/replay pins for the generic Chat Completions client.

The base client owns established ``reasoning_details`` (OpenRouter style)
and ``reasoning_content`` (DeepSeek/GLM style) dialects, plus vLLM's fallback
``reasoning`` field, with precedence-aware capture, marker-routed replay,
per-field aggregation, and reasoning-aware assistant coalescing. DeepSeek
narrows only the replay policy (tool-gated) and keeps its own strict assembler.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta as ChunkChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from chrys.kernel import (
    OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY,
    ChatMiddlewareLayer,
    ChatResponse,
    Content,
    FunctionTool,
    Message,
    ResponseStream,
)
from chrys.service.llm.chat_completions import ChatCompletionsClient, DeepSeekChatCompletionsClient
from chrys.service.llm.chat_completions.client import DEEPSEEK, OPENAI
from chrys.service.llm.chat_completions.decode import decode_completion
from chrys.service.llm.chat_completions.history import encode_message, encode_messages
from chrys.service.llm.openai_timestamps import openai_created_at_iso
from tests.support.openai_chat_wire import parse_stream_chunks, scripted_openai
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer


class _UnusedCompletions:
    async def create(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("parser/assembler tests should not call the SDK")


class _UnusedChat:
    def __init__(self) -> None:
        self.completions = _UnusedCompletions()


class _UnusedAsyncOpenAI:
    base_url = "https://api.test"

    def __init__(self) -> None:
        self.chat = _UnusedChat()


def _client() -> ChatCompletionsClient:
    return ChatCompletionsClient(model="glm-5.2", sdk_client=_UnusedAsyncOpenAI())


@pytest.mark.asyncio
async def test_a_stream_closed_after_its_first_update_releases_the_body() -> None:
    """The SDK closes a body it read to the end itself; one left early is the client's to close."""
    chunks = [
        _chunk(ChunkChoiceDelta.model_construct(role="assistant", content="hi")),
        _chunk(ChunkChoiceDelta.model_construct(content=" there")),
    ]
    async with scripted_openai([chunks]) as wire:
        raw_client = ChatCompletionsClient(model="glm-5.2", sdk_client=wire.client)
        response_stream = raw_client._inner_get_response(
            messages=[Message("user", ["hi"])],
            options={},
            stream=True,
        )
        assert isinstance(response_stream, ResponseStream)
        first = await anext(response_stream)
        assert first.text == "hi"
        assert [stream.closed for stream in wire.streams] == [False]

        await response_stream.aclose()

        assert [stream.closed for stream in wire.streams] == [True]


def _deepseek() -> DeepSeekChatCompletionsClient:
    return DeepSeekChatCompletionsClient(model="deepseek-reasoner", sdk_client=_UnusedAsyncOpenAI())


def _completion(message: ChatCompletionMessage, created: int = 1234567890) -> ChatCompletion:
    return ChatCompletion(
        id="resp-1",
        object="chat.completion",
        created=created,
        model="glm-5.2",
        choices=[Choice(index=0, message=message, finish_reason="stop")],
    )


def _chunk(delta: ChunkChoiceDelta, created: int = 1234567890) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk-1",
        object="chat.completion.chunk",
        created=created,
        model="glm-5.2",
        choices=[ChunkChoice(index=0, delta=delta, finish_reason=None)],
    )


def _content_reasoning(text: str | None = None, protected: Any = None) -> Content:
    return Content.from_text_reasoning(
        text=text,
        protected_data=json.dumps(protected) if protected is not None else None,
        additional_properties={"openai_reasoning_format": "reasoning_content"},
    )


def _details_reasoning(payload: list[Any]) -> Content:
    return Content.from_text_reasoning(
        protected_data=json.dumps(payload),
        additional_properties={"openai_reasoning_format": "reasoning_details"},
    )


def _vllm_reasoning(text: str | None = None, protected: Any = None) -> Content:
    return Content.from_text_reasoning(
        text=text,
        protected_data=json.dumps(protected) if protected is not None else None,
        additional_properties={"openai_reasoning_format": "reasoning"},
    )


# ───────────────────────── capture (§A.3/§A.4) ─────────────────────────


def test_base_nonstreaming_captures_reasoning_content_with_marker_and_props() -> None:
    created_ms = 1_717_171_717_123
    parsed = decode_completion(
        _completion(
            ChatCompletionMessage(role="assistant", content="Answer.", reasoning_content="GLM thinking"),
            created=created_ms,
        ),
        {},
        variant=OPENAI,
    )

    message = parsed.messages[0]
    assert message.contents[0].text == "Answer."
    reasoning = message.contents[1]
    assert reasoning.type == "text_reasoning"
    assert reasoning.protected_data == json.dumps("GLM thinking")
    assert reasoning.additional_properties["openai_reasoning_format"] == "reasoning_content"
    assert message.additional_properties["reasoning_content"] == "GLM thinking"
    assert message.additional_properties["openai_reasoning_format"] == "reasoning_content"
    assert parsed.created_at == openai_created_at_iso(created_ms)


def test_base_nonstreaming_captures_vllm_reasoning_with_marker_props_and_exact_replay() -> None:
    parsed = decode_completion(
        _completion(ChatCompletionMessage.model_construct(role="assistant", content="Answer.", reasoning="thinking")),
        {},
        variant=OPENAI,
    )

    message = parsed.messages[0]
    reasoning = message.contents[1]
    assert reasoning.type == "text_reasoning"
    assert reasoning.protected_data == json.dumps("thinking")
    assert reasoning.additional_properties["openai_reasoning_format"] == "reasoning"
    assert message.additional_properties == {
        "reasoning": "thinking",
        "openai_reasoning_format": "reasoning",
    }

    (prepared,) = encode_message(message, variant=OPENAI)
    assert prepared["reasoning"] == "thinking"
    assert "reasoning_content" not in prepared
    assert "reasoning_details" not in prepared


def test_vllm_reasoning_survives_openai_sdk_wire_validation_for_blocking_and_streaming() -> None:
    completion = ChatCompletion.model_validate(
        {
            "id": "resp-vllm",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Answer.", "reasoning": "blocking chain"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }
    )
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "chunk-vllm",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "qwen",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {"role": "assistant", "reasoning": "streaming chain"},
                }
            ],
        }
    )

    client = _client()
    blocking = decode_completion(completion, {}, variant=client.VARIANT)
    (streaming,) = parse_stream_chunks(client, chunk)

    assert blocking.messages[0].contents[1].protected_data == json.dumps("blocking chain")
    assert streaming.contents[0].text == "streaming chain"
    assert streaming.contents[0].additional_properties["openai_reasoning_format"] == "reasoning"


def test_vllm_explicit_null_reasoning_is_absence_not_empty_reasoning() -> None:
    """A verbatim thinking-disabled Qwen3.6 blocking reply from vLLM 0.19.2.

    vLLM serialises the field as ``"reasoning": null`` rather than omitting
    it; that must read as "no reasoning" — no phantom reasoning content and
    no message-level ``reasoning`` prop that replay would echo back.
    """
    completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-a448ad50fd1bff2b",
            "object": "chat.completion",
            "created": 1_777_044_630,
            "model": "qwen3.6-35b-nvfp4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "12", "reasoning": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 35, "total_tokens": 38, "completion_tokens": 3},
        }
    )

    parsed = decode_completion(completion, {}, variant=OPENAI)

    message = parsed.messages[0]
    assert [content.type for content in message.contents] == ["text"]
    assert message.contents[0].text == "12"
    assert message.additional_properties == {}
    (prepared,) = encode_message(message, variant=OPENAI)
    assert "reasoning" not in prepared


def test_vllm_reasoning_session_round_trip_preserves_exact_replay_dialect() -> None:
    original = Message(
        "assistant",
        [Content.from_text("Answer."), _vllm_reasoning(protected="thinking")],
        additional_properties={"reasoning": "thinking", "openai_reasoning_format": "reasoning"},
    )

    serialized = original.to_dict()
    restored = Message.from_dict(serialized)

    assert serialized["contents"][1]["protected_data"] == json.dumps("thinking")
    assert serialized["contents"][1]["additional_properties"]["openai_reasoning_format"] == "reasoning"
    assert serialized["additional_properties"]["reasoning"] == "thinking"
    (prepared,) = encode_message(restored, variant=OPENAI)
    assert prepared["reasoning"] == "thinking"
    assert "reasoning_content" not in prepared


@pytest.mark.asyncio
async def test_vllm_reasoning_replay_reaches_actual_openai_sdk_http_body() -> None:
    request_body: dict[str, Any] = {}

    def handle_request(request: httpx.Request) -> httpx.Response:
        request_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp-vllm",
                "object": "chat.completion",
                "created": 1,
                "model": "qwen",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Next answer."},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_request)) as http_client:
        sdk_client = AsyncOpenAI(
            api_key="sk-test",
            base_url="https://vllm.test/v1",
            http_client=http_client,
        )
        client = ChatCompletionsClient(model="qwen", sdk_client=sdk_client)
        await client.get_response(
            [
                Message(
                    "assistant",
                    [Content.from_text("Answer."), _vllm_reasoning(protected="thinking")],
                ),
                Message("user", [Content.from_text("Continue.")]),
            ]
        )

    assistant_message = request_body["messages"][0]
    assert assistant_message["reasoning"] == "thinking"
    assert "reasoning_content" not in assistant_message
    assert "reasoning_details" not in assistant_message


def test_base_nonstreaming_captures_reasoning_details_stamped() -> None:
    details = [{"type": "reasoning.text", "text": "chain"}]
    parsed = decode_completion(
        _completion(ChatCompletionMessage.model_construct(role="assistant", content="A", reasoning_details=details)),
        {},
        variant=OPENAI,
    )

    message = parsed.messages[0]
    reasoning = message.contents[1]
    assert reasoning.protected_data == json.dumps(details)
    assert reasoning.additional_properties["openai_reasoning_format"] == "reasoning_details"
    assert message.additional_properties["reasoning_details"] == details
    assert message.additional_properties["openai_reasoning_format"] == "reasoning_details"


def test_base_nonstreaming_dual_field_capture_details_first() -> None:
    details = [{"type": "reasoning.encrypted", "data": "opaque"}]
    parsed = decode_completion(
        _completion(
            ChatCompletionMessage.model_construct(
                role="assistant",
                content="A",
                reasoning_details=details,
                reasoning_content="visible chain",
            )
        ),
        {},
        variant=OPENAI,
    )

    message = parsed.messages[0]
    markers = [
        content.additional_properties["openai_reasoning_format"]
        for content in message.contents
        if content.type == "text_reasoning"
    ]
    assert markers == ["reasoning_details", "reasoning_content"]
    assert message.additional_properties["reasoning_details"] == details
    assert message.additional_properties["reasoning_content"] == "visible chain"
    # Details lead the dual-field stash.
    assert message.additional_properties["openai_reasoning_format"] == "reasoning_details"


@pytest.mark.parametrize(
    ("established_fields", "expected_markers"),
    [
        ({"reasoning_details": [{"type": "reasoning.text", "text": "OpenRouter"}]}, ["reasoning_details"]),
        ({"reasoning_content": "legacy vLLM"}, ["reasoning_content"]),
        (
            {
                "reasoning_details": [{"type": "reasoning.text", "text": "OpenRouter"}],
                "reasoning_content": "visible chain",
            },
            ["reasoning_details", "reasoning_content"],
        ),
    ],
)
def test_base_nonstreaming_reasoning_is_fallback_to_established_fields(
    established_fields: dict[str, Any],
    expected_markers: list[str],
) -> None:
    message = ChatCompletionMessage.model_construct(
        role="assistant",
        content="Answer.",
        reasoning="mirrored plaintext",
        **established_fields,
    )

    parsed = decode_completion(_completion(message), {}, variant=OPENAI)

    kernel_message = parsed.messages[0]
    reasoning_contents = [content for content in kernel_message.contents if content.type == "text_reasoning"]
    assert [content.additional_properties["openai_reasoning_format"] for content in reasoning_contents] == (
        expected_markers
    )
    assert "reasoning" not in kernel_message.additional_properties


def test_base_streaming_captures_reasoning_content_as_visible_text() -> None:
    created_ms = 1_717_171_717_123
    (update,) = parse_stream_chunks(
        _client(),
        _chunk(ChunkChoiceDelta.model_construct(role="assistant", reasoning_content="streamed chain"), created_ms),
    )

    reasoning = next(content for content in update.contents if content.type == "text_reasoning")
    assert reasoning.text == "streamed chain"
    assert reasoning.protected_data is None
    assert reasoning.additional_properties["openai_reasoning_format"] == "reasoning_content"
    assert update.created_at == openai_created_at_iso(created_ms)


def test_base_streaming_captures_vllm_reasoning_as_visible_text() -> None:
    (update,) = parse_stream_chunks(
        _client(), _chunk(ChunkChoiceDelta.model_construct(role="assistant", reasoning="streamed chain"))
    )

    (reasoning,) = [content for content in update.contents if content.type == "text_reasoning"]
    assert reasoning.text == "streamed chain"
    assert reasoning.protected_data is None
    assert reasoning.additional_properties["openai_reasoning_format"] == "reasoning"


@pytest.mark.parametrize(
    ("established_fields", "expected_markers"),
    [
        ({"reasoning_details": [{"type": "reasoning.text", "text": "OpenRouter"}]}, ["reasoning_details"]),
        ({"reasoning_content": "legacy vLLM"}, ["reasoning_content"]),
    ],
)
def test_base_streaming_reasoning_is_fallback_to_established_fields(
    established_fields: dict[str, Any],
    expected_markers: list[str],
) -> None:
    (update,) = parse_stream_chunks(
        _client(),
        _chunk(
            ChunkChoiceDelta.model_construct(
                role="assistant",
                reasoning="mirrored plaintext",
                **established_fields,
            )
        ),
    )

    reasoning_contents = [content for content in update.contents if content.type == "text_reasoning"]
    assert [content.additional_properties["openai_reasoning_format"] for content in reasoning_contents] == (
        expected_markers
    )


def test_base_streaming_captures_reasoning_details_as_protected_json() -> None:
    details = [{"type": "reasoning.text", "text": "delta"}]
    (update,) = parse_stream_chunks(
        _client(), _chunk(ChunkChoiceDelta.model_construct(role="assistant", reasoning_details=details))
    )

    reasoning = next(content for content in update.contents if content.type == "text_reasoning")
    assert reasoning.text is None
    assert reasoning.protected_data == json.dumps(details)
    assert reasoning.additional_properties["openai_reasoning_format"] == "reasoning_details"


# ───────────────────────── replay policy (§A.5) ─────────────────────────


def test_base_replays_marked_reasoning_without_tool_interaction() -> None:
    """GLM preserved thinking: replay happens on every multi-turn request."""
    prepared = encode_messages(
        [
            Message("user", ["First question"]),
            Message("assistant", [Content.from_text("Answer"), _content_reasoning(protected="no-tool chain")]),
            Message("user", ["Follow-up"]),
        ],
        variant=OPENAI,
    )

    assistant_message = prepared[1]
    assert assistant_message["content"] == "Answer"
    assert assistant_message["reasoning_content"] == "no-tool chain"


def test_streamed_text_and_protected_json_shapes_replay_identically() -> None:
    streamed = Message("assistant", [Content.from_text("A"), _content_reasoning(text="chain")])
    protected = Message("assistant", [Content.from_text("A"), _content_reasoning(protected="chain")])

    for message in (streamed, protected):
        (prepared,) = encode_message(message, variant=OPENAI)
        assert prepared["reasoning_content"] == "chain"


def test_streamed_vllm_reasoning_fragments_concatenate_and_replay_exact_field() -> None:
    message = Message(
        "assistant",
        [
            _vllm_reasoning(text="first "),
            _vllm_reasoning(text="second"),
            Content.from_text("Answer"),
        ],
    )

    (prepared,) = encode_message(message, variant=OPENAI)

    assert prepared["reasoning"] == "first second"
    assert "reasoning_content" not in prepared
    assert "reasoning_details" not in prepared


def test_dual_field_replay_attaches_both_fields_once() -> None:
    details = [{"type": "reasoning.encrypted", "data": "opaque"}]
    message = Message(
        role="assistant",
        contents=[
            Content.from_text("A"),
            _details_reasoning(details),
            _content_reasoning(protected="visible chain"),
        ],
        additional_properties={
            "reasoning_details": details,
            "reasoning_content": "visible chain",
            "openai_reasoning_format": "reasoning_details",
        },
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 1
    assert prepared[0]["reasoning_details"] == details
    assert prepared[0]["reasoning_content"] == "visible chain"


def test_parse_replay_round_trip_emits_each_field_value_exactly_once() -> None:
    """Dual storage (content + message props) must not double a field's value."""
    parsed = decode_completion(
        _completion(ChatCompletionMessage(role="assistant", content="A", reasoning_content="X")),
        {},
        variant=OPENAI,
    )

    (prepared,) = encode_message(parsed.messages[0], variant=OPENAI)

    assert prepared["reasoning_content"] == "X"


def test_deepseek_still_gates_replay_on_tool_interaction() -> None:
    messages = [
        Message("user", ["Q"]),
        Message("assistant", [Content.from_text("Answer"), _content_reasoning(protected="chain")]),
        Message("user", ["Follow-up"]),
    ]

    deepseek_prepared = encode_messages(messages, variant=DEEPSEEK)
    assert "reasoning_content" not in deepseek_prepared[1]

    base_prepared = encode_messages(messages, variant=OPENAI)
    assert base_prepared[1]["reasoning_content"] == "chain"


# ───────────────────────── coalescing (§A.7) ─────────────────────────


def test_nonstreaming_text_reasoning_tool_calls_coalesce_to_one_message() -> None:
    message = Message(
        role="assistant",
        contents=[
            Content.from_text("Let me check."),
            Content.from_function_call(call_id="call_1", name="get_weather", arguments='{"city":"Seattle"}'),
            _content_reasoning(protected="tool chain"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 1
    assert prepared[0]["content"] == "Let me check."
    assert prepared[0]["reasoning_content"] == "tool chain"
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_streaming_accumulated_reasoning_coalesces_to_one_message() -> None:
    chunks = [
        _chunk(ChunkChoiceDelta.model_construct(role="assistant", reasoning_content="first ")),
        _chunk(ChunkChoiceDelta.model_construct(role="assistant", reasoning_content="second")),
        _chunk(ChunkChoiceDelta.model_construct(role="assistant", content="Answer")),
        _chunk(
            ChunkChoiceDelta.model_construct(
                role="assistant",
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        type="function",
                        function=ChoiceDeltaToolCallFunction(name="get_weather", arguments='{"city":"Seattle"}'),
                    )
                ],
            )
        ),
    ]
    async with scripted_openai([chunks]) as wire:
        stream = ChatCompletionsClient(model="glm-5.2", sdk_client=wire.client)._inner_get_response(
            messages=[Message("user", ["hi"])], options={}, stream=True
        )
        assert isinstance(stream, ResponseStream)
        response = await stream.get_final_response()

    prepared = encode_message(response.messages[0], variant=OPENAI)

    assert len(prepared) == 1
    assert prepared[0]["content"] == "Answer"
    assert prepared[0]["reasoning_content"] == "first second"
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_no_reasoning_assistant_message_sends_one_tool_call_carrier() -> None:
    """Without replayable reasoning, the text and tool-call fragments go out as one message."""
    message = Message(
        role="assistant",
        contents=[
            Content.from_text("Preface"),
            Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
        ],
    )

    prepared = encode_messages([message], variant=OPENAI)

    assert prepared == [{"role": "assistant", "content": "Preface", "tool_calls": prepared[0]["tool_calls"]}]
    assert [call["function"]["name"] for call in prepared[0]["tool_calls"]] == ["lookup"]


def test_text_image_reasoning_tool_calls_aggregate_excludes_image() -> None:
    """Assistant content arrays admit only text/refusal parts; images stay standalone."""
    message = Message(
        role="assistant",
        contents=[
            Content.from_text("Look at this."),
            Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
            _content_reasoning(protected="image chain"),
            Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 2
    image_message, aggregate = prepared
    assert image_message["content"][0]["type"] == "image_url"
    assert "reasoning_content" not in image_message
    assert aggregate["content"] == "Look at this."
    assert aggregate["reasoning_content"] == "image chain"
    assert aggregate["tool_calls"][0]["function"]["name"] == "lookup"
    assert not isinstance(aggregate["content"], list)


@pytest.mark.parametrize(
    "contents_order",
    [
        ["text", "function_call", "image"],
        ["image", "text", "function_call"],
    ],
)
def test_aggregate_emits_last_for_tool_result_adjacency(contents_order: list[str]) -> None:
    """The tool_calls-carrying aggregate must sit immediately before its results."""
    parts = {
        "text": Content.from_text("Preface"),
        "function_call": Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
        "image": Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
    }
    messages = [
        Message("user", ["Q"]),
        Message(
            role="assistant",
            contents=[*(parts[name] for name in contents_order), _content_reasoning(protected="chain")],
        ),
        Message("tool", [Content.from_function_result(call_id="call_1", result="found")]),
    ]

    prepared = encode_messages(messages, variant=OPENAI)

    roles = [message["role"] for message in prepared]
    assert roles == ["user", "assistant", "assistant", "tool"]
    image_message, aggregate, tool_result = prepared[1], prepared[2], prepared[3]
    assert image_message["content"][0]["type"] == "image_url"
    assert aggregate["content"] == "Preface"
    assert aggregate["reasoning_content"] == "chain"
    assert aggregate["tool_calls"][0]["function"]["name"] == "lookup"
    assert tool_result["tool_call_id"] == "call_1"


def test_zero_text_aggregate_keeps_empty_string_content() -> None:
    message = Message(
        role="assistant",
        contents=[
            Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
            _content_reasoning(protected="chain"),
            Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 2
    image_message, aggregate = prepared
    assert image_message["content"][0]["type"] == "image_url"
    assert aggregate["content"] == ""
    assert aggregate["reasoning_content"] == "chain"
    assert aggregate["tool_calls"][0]["function"]["name"] == "lookup"


# ───────────────────────── aggregation (§A.5a) ─────────────────────────


def test_interleaved_dual_field_fragments_aggregate_per_field() -> None:
    details_1 = [{"step": 1}]
    details_2 = [{"step": 2}]
    message = Message(
        role="assistant",
        contents=[
            _details_reasoning(details_1),
            _content_reasoning(text="C1"),
            _details_reasoning(details_2),
            _content_reasoning(text="C2"),
            Content.from_text("Answer"),
        ],
    )

    for client in (_client(), _deepseek()):
        (prepared,) = encode_message(message, variant=client.VARIANT)
        assert prepared["reasoning_content"] == "C1C2"
        assert prepared["reasoning_details"] == [{"step": 1}, {"step": 2}]


def test_separator_split_details_replay_both_blocks() -> None:
    """`D1, text, D2, function_call` must replay BOTH detail blocks in one message."""
    message = Message(
        role="assistant",
        contents=[
            _details_reasoning([{"block": "D1"}]),
            Content.from_text("thinking aloud"),
            _details_reasoning([{"block": "D2"}]),
            Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
        ],
    )

    for client in (_client(), _deepseek()):
        (prepared,) = encode_message(message, variant=client.VARIANT)
        assert prepared["reasoning_details"] == [{"block": "D1"}, {"block": "D2"}]
        assert prepared["content"] == "thinking aloud"
        assert prepared["tool_calls"][0]["function"]["name"] == "lookup"


def test_streamed_protected_detail_blocks_survive_and_replay_extended() -> None:
    client = _client()
    updates = parse_stream_chunks(
        client,
        _chunk(ChunkChoiceDelta.model_construct(role="assistant", reasoning_details=[{"block": "D1"}])),
        _chunk(ChunkChoiceDelta.model_construct(role="assistant", reasoning_details=[{"block": "D2"}])),
        _chunk(ChunkChoiceDelta.model_construct(role="assistant", content="Answer")),
    )

    response = ChatResponse.from_updates(updates)

    reasoning_contents = [content for content in response.messages[0].contents if content.type == "text_reasoning"]
    assert [content.protected_data for content in reasoning_contents] == [
        json.dumps([{"block": "D1"}]),
        json.dumps([{"block": "D2"}]),
    ]

    (prepared,) = encode_message(response.messages[0], variant=client.VARIANT)
    assert prepared["reasoning_details"] == [{"block": "D1"}, {"block": "D2"}]


def test_mixed_type_reasoning_content_contributions_last_win() -> None:
    message = Message(
        role="assistant",
        contents=[
            _content_reasoning(text="R"),
            _content_reasoning(protected=[2]),
            Content.from_text("Answer"),
        ],
    )

    (prepared,) = encode_message(message, variant=OPENAI)

    assert prepared["reasoning_content"] == [2]


# ───────────────────────── predicate / role matrix (§A.5b) ─────────────────────────


@pytest.mark.parametrize("role", ["system", "developer"])
def test_system_and_developer_reasoning_is_never_replayed(role: str) -> None:
    message = Message(
        role=role,
        contents=[Content.from_text("Instructions"), _content_reasoning(text="never send")],
        additional_properties={"reasoning_content": "never send"},
    )

    prepared = encode_message(message, variant=OPENAI)

    assert prepared == [{"role": role, "content": "Instructions"}]


def test_user_reasoning_attaches_to_next_emitted_fragment() -> None:
    message = Message(
        role="user",
        contents=[
            _details_reasoning([1]),
            Content.from_text("Q"),
            Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 2
    assert prepared[0]["content"] == "Q"
    assert prepared[0]["reasoning_details"] == [1]
    assert "reasoning_details" not in prepared[1]
    assert prepared[1]["content"][0]["type"] == "image_url"


@pytest.mark.parametrize("role", ["user", "tool"])
def test_props_only_contentless_non_assistant_message_emits_nothing(role: str) -> None:
    message = Message(role=role, contents=[], additional_properties={"reasoning_content": "stash"})

    assert encode_message(message, variant=OPENAI) == []


def test_props_only_contentless_assistant_message_emits_empty_content_carrier() -> None:
    message = Message(
        role="assistant",
        contents=[],
        additional_properties={"reasoning_content": "props chain", "openai_reasoning_format": "reasoning_content"},
    )

    prepared = encode_message(message, variant=OPENAI)

    assert prepared == [{"role": "assistant", "content": "", "reasoning_content": "props chain"}]


def test_malformed_marker_reasoning_drops_without_coalescing() -> None:
    message = Message(
        role="assistant",
        contents=[
            Content.from_text_reasoning(text="mystery", additional_properties={"openai_reasoning_format": "bogus"})
        ],
    )

    assert encode_message(message, variant=OPENAI) == []


def test_unparseable_protected_data_suppresses_marked_text() -> None:
    """Protected data is authoritative: a foreign/undecodable payload silences the content."""
    foreign = Content.from_text_reasoning(
        text="visible but foreign",
        protected_data="opaque-not-json",
        additional_properties={"openai_reasoning_format": "reasoning_content"},
    )
    message = Message(role="assistant", contents=[foreign, Content.from_text("A")])

    prepared = encode_message(message, variant=OPENAI)

    assert prepared == [{"role": "assistant", "content": "A"}]


def test_legacy_unmarked_json_protected_data_replays_as_details() -> None:
    legacy = Content.from_text_reasoning(protected_data=json.dumps([{"block": "legacy"}]))
    message = Message(
        role="assistant",
        contents=[legacy, Content.from_function_call(call_id="call_1", name="lookup", arguments="{}")],
    )

    (prepared,) = encode_message(message, variant=OPENAI)

    assert prepared["reasoning_details"] == [{"block": "legacy"}]
    assert prepared["content"] == ""
    assert prepared["tool_calls"][0]["function"]["name"] == "lookup"


# ─────────────── result-carrying carriers & marker authority (§A.5b/§A.7) ───────────────


def _foreign_reasoning(text: str | None = None, protected: Any = None) -> Content:
    return Content.from_text_reasoning(
        text=text,
        protected_data=json.dumps(protected) if protected is not None else None,
        additional_properties={"openai_reasoning_format": "future_reasoning_v2"},
    )


def test_result_carrying_assistant_attaches_aggregate_to_tool_calls_message() -> None:
    """Result records stay standalone in call→result order; the full aggregate rides the tool-calls carrier."""
    message = Message(
        role="assistant",
        contents=[
            _content_reasoning(text="C1"),
            Content.from_text("Working."),
            _content_reasoning(text="C2"),
            Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
            Content.from_function_result(call_id="call_1", result="found"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 2
    aggregate, result_record = prepared
    assert aggregate["content"] == "Working."
    assert aggregate["tool_calls"][0]["function"]["name"] == "lookup"
    assert aggregate["reasoning_content"] == "C1C2"
    assert result_record == {"role": "tool", "tool_call_id": "call_1", "content": "found"}


def test_per_run_reasoning_ownership_survives_result_separators() -> None:
    """Each run's thinking rides that run's tool calls; it is never pooled onto a later run."""
    message = Message(
        role="assistant",
        contents=[
            _content_reasoning(text="R1"),
            Content.from_function_call(call_id="call_1", name="alpha", arguments="{}"),
            Content.from_function_result(call_id="call_1", result="one"),
            _content_reasoning(text="R2"),
            Content.from_function_call(call_id="call_2", name="beta", arguments="{}"),
            Content.from_function_result(call_id="call_2", result="two"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 4
    first_call, first_result, second_call, second_result = prepared
    assert first_call["tool_calls"][0]["function"]["name"] == "alpha"
    assert first_call["reasoning_content"] == "R1"
    assert first_result == {"role": "tool", "tool_call_id": "call_1", "content": "one"}
    assert second_call["tool_calls"][0]["function"]["name"] == "beta"
    assert second_call["reasoning_content"] == "R2"
    assert second_result == {"role": "tool", "tool_call_id": "call_2", "content": "two"}


def test_deepseek_per_run_reasoning_ownership_survives_result_separators() -> None:
    message = Message(
        role="assistant",
        contents=[
            _content_reasoning(text="R1"),
            Content.from_function_call(call_id="call_1", name="alpha", arguments="{}"),
            Content.from_function_result(call_id="call_1", result="one"),
            _content_reasoning(text="R2"),
            Content.from_function_call(call_id="call_2", name="beta", arguments="{}"),
            Content.from_function_result(call_id="call_2", result="two"),
        ],
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert len(prepared) == 4
    first_call, first_result, second_call, second_result = prepared
    assert first_call["tool_calls"][0]["function"]["name"] == "alpha"
    assert first_call["reasoning_content"] == "R1"
    assert first_result["tool_call_id"] == "call_1"
    assert "reasoning_content" not in first_result
    assert second_call["tool_calls"][0]["function"]["name"] == "beta"
    assert second_call["reasoning_content"] == "R2"
    assert second_result["tool_call_id"] == "call_2"
    assert "reasoning_content" not in second_result


def test_text_separated_tool_calls_coalesce_into_one_aggregate_before_results() -> None:
    """A run without result separators lands as ONE aggregate — never a text/tool_calls split."""
    message = Message(
        role="assistant",
        contents=[
            _content_reasoning(text="R"),
            Content.from_text("First."),
            Content.from_function_call(call_id="call_1", name="alpha", arguments="{}"),
            Content.from_text("Second."),
            Content.from_function_call(call_id="call_2", name="beta", arguments="{}"),
            Content.from_function_result(call_id="call_1", result="one"),
            Content.from_function_result(call_id="call_2", result="two"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 3
    aggregate, result_1, result_2 = prepared
    assert aggregate["content"] == "First.\nSecond."
    assert [call["function"]["name"] for call in aggregate["tool_calls"]] == ["alpha", "beta"]
    assert aggregate["reasoning_content"] == "R"
    assert result_1 == {"role": "tool", "tool_call_id": "call_1", "content": "one"}
    assert result_2 == {"role": "tool", "tool_call_id": "call_2", "content": "two"}


def test_wire_text_segments_split_at_calls_without_splitting_aggregate() -> None:
    message = Message(
        "assistant",
        [
            _content_reasoning(text="R1"),
            Content.from_text("回"),
            _content_reasoning(text="R2"),
            Content.from_text("退"),
            Content.from_function_call(call_id="call_1", name="alpha", arguments="{}"),
            Content.from_text(""),
            Content.from_function_call(call_id="call_2", name="beta", arguments="{}"),
            Content.from_text("核对"),
        ],
    )

    prepared = encode_messages([message], variant=OPENAI)

    assert len(prepared) == 1
    aggregate = prepared[0]
    assert aggregate["content"] == "回退\n核对"
    assert aggregate["reasoning_content"] == "R1R2"
    assert [call["id"] for call in aggregate["tool_calls"]] == ["call_1", "call_2"]


@pytest.mark.parametrize(
    ("hosted_call", "summary"),
    [
        (
            Content.from_mcp_server_tool_call("hosted_1", "lookup", hosted_provider="anthropic"),
            "[Provider-hosted tool context]\nTool: lookup\nFamily: mcp\nStatus: interrupted",
        ),
        (
            Content.from_search_tool_call("hosted_1", tool_name="web_search", hosted_provider="anthropic"),
            "[Provider-hosted tool context]\nTool: web_search\nFamily: search\nStatus: interrupted",
        ),
    ],
    ids=["mcp", "search"],
)
def test_reasoning_history_preserves_text_boundary_at_degraded_hosted_calls(hosted_call: Content, summary: str) -> None:
    message = Message(
        "assistant",
        [_content_reasoning(text="R"), Content.from_text("A"), hosted_call, Content.from_text("B")],
    )
    original = message.to_dict()

    prepared = encode_messages([message], variant=OPENAI)

    assert prepared == [{"role": "assistant", "content": f"A\n{summary}\nB", "reasoning_content": "R"}]
    assert message.to_dict() == original


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        ([("回", None), ("退\n把代码核对完毕", None)], "回退\n把代码核对完毕"),
        ([("基线\n", None), ("核对完毕", None)], "基线\n核对完毕"),
        ([("First.", None), ("\n\n", None), ("Second.", None)], "First.\n\nSecond."),
        ([("回", "a"), ("退", "a"), ("Next.", "b")], "回退\nNext."),
        ([("One.", "a"), ("Two.", "b"), ("Three.", "a")], "One.\nTwo.\nThree."),
        ([("One.", None), ("Two.", "a"), ("Three.", None)], "One.\nTwo.\nThree."),
        ([("One.", "a"), ("", "b"), ("Three.", "a")], "One.\nThree."),
        ([("One.", "a"), ("", None), ("Three.", "a")], "One.\nThree."),
    ],
)
def test_reasoning_history_reconstructs_text_within_item_boundaries(
    parts: list[tuple[str, str | None]], expected: str
) -> None:
    contents: list[Content] = []
    for index, (text, block_id) in enumerate(parts):
        properties = {}
        if block_id is not None:
            # Mutable envelope fields must not split fragments of the same ID.
            properties[OPENAI_OUTPUT_MESSAGE_ENVELOPE_KEY] = {
                "id": block_id,
                "status": "in_progress" if index == 0 else "completed",
                "phase": "commentary" if index == 0 else "final_answer",
            }
        contents.extend([_content_reasoning(text="R"), Content.from_text(text, additional_properties=properties)])
    contents.append(Content.from_function_call(call_id="call_1", name="read_file", arguments="{}"))
    message = Message.from_dict(Message("assistant", contents).to_dict())

    prepared = encode_messages([message], variant=OPENAI)

    assert len(prepared) == 1
    assert prepared[0]["content"] == expected
    assert prepared[0]["reasoning_content"] == "R" * len(parts)
    assert prepared[0]["tool_calls"][0]["id"] == "call_1"


def test_trailing_reasoning_after_last_result_emits_positional_carrier() -> None:
    """Reasoning captured after the final result stays in its own run — never reassigned to an earlier call."""
    message = Message(
        role="assistant",
        contents=[
            Content.from_function_call(call_id="call_1", name="alpha", arguments="{}"),
            Content.from_function_result(call_id="call_1", result="one"),
            Content.from_function_call(call_id="call_2", name="beta", arguments="{}"),
            Content.from_function_result(call_id="call_2", result="two"),
            _content_reasoning(text="R"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 5
    first_call, first_result, second_call, second_result, carrier = prepared
    assert first_call["tool_calls"][0]["function"]["name"] == "alpha"
    assert "reasoning_content" not in first_call
    assert first_result == {"role": "tool", "tool_call_id": "call_1", "content": "one"}
    assert second_call["tool_calls"][0]["function"]["name"] == "beta"
    assert "reasoning_content" not in second_call
    assert second_result == {"role": "tool", "tool_call_id": "call_2", "content": "two"}
    assert carrier == {"role": "assistant", "content": "", "reasoning_content": "R"}


def test_reasoning_never_rides_a_multimodal_carrier() -> None:
    """Reasoning fields ride string-content messages only; an image record is never the carrier."""
    message = Message(
        role="assistant",
        contents=[
            _content_reasoning(text="R"),
            Content.from_text("caption"),
            Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
            Content.from_function_result(call_id="call_1", result="found"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 3
    image_message, aggregate, result_record = prepared
    assert image_message["content"][0]["type"] == "image_url"
    assert "reasoning_content" not in image_message
    assert aggregate["content"] == "caption"
    assert aggregate["reasoning_content"] == "R"
    assert result_record == {"role": "tool", "tool_call_id": "call_1", "content": "found"}


def test_image_only_result_message_reasoning_defers_carrier_past_result() -> None:
    """A reasoning-only run never splits a result block; its carrier defers past the results."""
    message = Message(
        role="assistant",
        contents=[
            _content_reasoning(text="R"),
            Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
            Content.from_function_result(call_id="call_1", result="found"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 3
    image_message, result_record, carrier = prepared
    assert "reasoning_content" not in image_message
    assert result_record == {"role": "tool", "tool_call_id": "call_1", "content": "found"}
    assert carrier == {"role": "assistant", "content": "", "reasoning_content": "R"}


def test_reasoning_only_run_never_splits_a_parallel_result_block() -> None:
    message = Message(
        role="assistant",
        contents=[
            Content.from_function_call(call_id="call_1", name="alpha", arguments="{}"),
            Content.from_function_call(call_id="call_2", name="beta", arguments="{}"),
            Content.from_function_result(call_id="call_1", result="one"),
            _content_reasoning(text="R"),
            Content.from_function_result(call_id="call_2", result="two"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 4
    aggregate, result_1, result_2, carrier = prepared
    assert [call["function"]["name"] for call in aggregate["tool_calls"]] == ["alpha", "beta"]
    assert "reasoning_content" not in aggregate
    assert result_1 == {"role": "tool", "tool_call_id": "call_1", "content": "one"}
    assert result_2 == {"role": "tool", "tool_call_id": "call_2", "content": "two"}
    assert carrier == {"role": "assistant", "content": "", "reasoning_content": "R"}


def test_mid_block_reasoning_joins_the_next_run_aggregate() -> None:
    message = Message(
        role="assistant",
        contents=[
            Content.from_function_result(call_id="call_1", result="one"),
            _content_reasoning(text="R"),
            Content.from_function_result(call_id="call_2", result="two"),
            Content.from_text("Conclusion."),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 3
    result_1, result_2, aggregate = prepared
    assert result_1 == {"role": "tool", "tool_call_id": "call_1", "content": "one"}
    assert result_2 == {"role": "tool", "tool_call_id": "call_2", "content": "two"}
    assert aggregate == {"role": "assistant", "content": "Conclusion.", "reasoning_content": "R"}


def test_deepseek_reasoning_never_rides_a_multimodal_carrier() -> None:
    """DeepSeek shares the reasoning-bearing encoder: string aggregate carries reasoning, image stays standalone."""
    message = Message(
        role="assistant",
        contents=[
            _content_reasoning(text="R"),
            Content.from_text("caption"),
            Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
            Content.from_function_result(call_id="call_1", result="found"),
        ],
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert len(prepared) == 3
    image_message, aggregate, result_record = prepared
    assert image_message["content"][0]["type"] == "image_url"
    assert "reasoning_content" not in image_message
    assert aggregate["content"] == "caption"
    assert aggregate["reasoning_content"] == "R"
    assert result_record == {"role": "tool", "tool_call_id": "call_1", "content": "found"}


def test_deepseek_image_then_tool_call_reasoning_rides_string_aggregate() -> None:
    """The tool_calls carrier stays string-content; the image is never merged into it."""
    message = Message(
        role="assistant",
        contents=[
            _content_reasoning(text="R"),
            Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
            Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
        ],
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert len(prepared) == 2
    image_message, aggregate = prepared
    assert image_message["content"][0]["type"] == "image_url"
    assert "reasoning_content" not in image_message
    assert "tool_calls" not in image_message
    assert aggregate["content"] == ""
    assert aggregate["tool_calls"][0]["function"]["name"] == "lookup"
    assert aggregate["reasoning_content"] == "R"


def test_result_only_assistant_reasoning_appends_carrier_after_result() -> None:
    message = Message(
        role="assistant",
        contents=[
            Content.from_function_result(call_id="call_1", result="found"),
            _content_reasoning(text="R"),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 2
    result_record, carrier = prepared
    assert result_record == {"role": "tool", "tool_call_id": "call_1", "content": "found"}
    assert carrier == {"role": "assistant", "content": "", "reasoning_content": "R"}


def test_deepseek_result_carrying_assistant_reasoning_rides_tool_calls_message() -> None:
    """DeepSeek's contract: replayed reasoning_content rides the tool_calls carrier, never a result record."""
    message = Message(
        role="assistant",
        contents=[
            Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
            _content_reasoning(text="R"),
            Content.from_function_result(call_id="call_1", result="found"),
        ],
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert len(prepared) == 2
    tool_calls_message, result_record = prepared
    assert tool_calls_message["tool_calls"][0]["function"]["name"] == "lookup"
    assert tool_calls_message["content"] == ""
    assert tool_calls_message["reasoning_content"] == "R"
    assert result_record["tool_call_id"] == "call_1"
    assert "reasoning_content" not in result_record


def test_deepseek_text_call_result_reasoning_rides_merged_tool_calls_message() -> None:
    message = Message(
        role="assistant",
        contents=[
            _content_reasoning(text="R"),
            Content.from_text("Working."),
            Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
            Content.from_function_result(call_id="call_1", result="found"),
        ],
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert len(prepared) == 2
    merged, result_record = prepared
    assert merged["content"] == "Working."
    assert merged["tool_calls"][0]["function"]["name"] == "lookup"
    assert merged["reasoning_content"] == "R"
    assert "reasoning_content" not in result_record


@pytest.mark.parametrize("role", ["user", "tool"])
def test_deepseek_props_only_non_assistant_message_emits_nothing(role: str) -> None:
    message = Message(role=role, contents=[], additional_properties={"reasoning_content": "stash"})

    assert encode_message(message, variant=DEEPSEEK) == []


def test_deepseek_props_only_assistant_message_emits_empty_content_carrier() -> None:
    message = Message(role="assistant", contents=[], additional_properties={"reasoning_content": "stash"})

    prepared = encode_message(message, variant=DEEPSEEK)

    assert prepared == [{"role": "assistant", "content": "", "reasoning_content": "stash"}]


def test_deepseek_tool_result_message_never_carries_reasoning_props() -> None:
    message = Message(
        role="tool",
        contents=[Content.from_function_result(call_id="call_1", result="found")],
        additional_properties={"reasoning_content": "stash"},
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert len(prepared) == 1
    assert prepared[0]["tool_call_id"] == "call_1"
    assert "reasoning_content" not in prepared[0]


def test_unknown_format_marker_never_replays_through_the_aggregate_path() -> None:
    """The stamp is authoritative: an unknown stamp drops even with a decodable protected payload."""
    message = Message(
        role="assistant",
        contents=[
            Content.from_text("Answer."),
            _foreign_reasoning(protected=[{"future": "payload"}]),
        ],
    )

    prepared = encode_message(message, variant=OPENAI)

    assert len(prepared) == 1
    assert prepared[0]["content"] == "Answer."
    assert "reasoning_details" not in prepared[0]
    assert "reasoning_content" not in prepared[0]


def test_unknown_format_marker_never_replays_through_the_legacy_walk() -> None:
    user_message = Message(
        role="user",
        contents=[_foreign_reasoning(protected=[{"future": "payload"}]), Content.from_text("Q")],
    )

    prepared = encode_message(user_message, variant=OPENAI)

    assert len(prepared) == 1
    assert prepared[0]["content"] == "Q"
    assert "reasoning_details" not in prepared[0]
    assert "reasoning_content" not in prepared[0]


def test_unknown_format_marker_only_assistant_message_emits_nothing() -> None:
    message = Message(role="assistant", contents=[_foreign_reasoning(protected=[{"future": "payload"}])])

    assert encode_message(message, variant=OPENAI) == []


# ───────────────────────── streaming tool loop (§F(e)) ─────────────────────────


@pytest.mark.asyncio
async def test_streaming_tool_loop_accumulates_and_replays_reasoning_content() -> None:
    def _tool_call_chunks() -> list[ChatCompletionChunk]:
        return [
            _chunk(ChunkChoiceDelta.model_construct(role="assistant", reasoning_content="accumulated ")),
            _chunk(ChunkChoiceDelta.model_construct(role="assistant", reasoning_content="chain")),
            _chunk(
                ChunkChoiceDelta.model_construct(
                    role="assistant",
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_abc",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(name="read_file", arguments='{"path":"foo.py"}'),
                        )
                    ],
                )
            ),
            ChatCompletionChunk(
                id="chunk-1",
                object="chat.completion.chunk",
                created=1234567890,
                model="glm-5.2",
                choices=[ChunkChoice(index=0, delta=ChunkChoiceDelta(role="assistant"), finish_reason="tool_calls")],
            ),
        ]

    def _final_chunks() -> list[ChatCompletionChunk]:
        return [
            _chunk(ChunkChoiceDelta.model_construct(role="assistant", content="Done.")),
            ChatCompletionChunk(
                id="chunk-2",
                object="chat.completion.chunk",
                created=1234567891,
                model="glm-5.2",
                choices=[ChunkChoice(index=0, delta=ChunkChoiceDelta(role="assistant"), finish_reason="stop")],
            ),
        ]

    def read_file(path: str) -> str:
        return f"contents of {path}"

    async with scripted_openai([_tool_call_chunks(), _final_chunks()]) as wire:
        client = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(ChatCompletionsClient(model="glm-5.2", sdk_client=wire.client))
        )
        tool = FunctionTool(name="read_file", description="Read a file", func=read_file)

        stream = client.get_response([Message("user", ["inspect foo.py"])], stream=True, options={"tools": [tool]})
        async for _update in stream:
            pass

    assert len(wire.requests) == 2
    assistant_message = wire.requests[1]["messages"][1]
    assert assistant_message["reasoning_content"] == "accumulated chain"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "read_file"
    tool_message = wire.requests[1]["messages"][2]
    assert tool_message["tool_call_id"] == "call_abc"
    assert tool_message["content"] == "contents of foo.py"


@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("marker", ["reasoning", "reasoning_content"])
def test_only_reasoning_non_assistant_message_emits_no_carrier(role: str, marker: str) -> None:
    """Reasoning replay is assistant-scoped: a user/tool message whose ONLY content
    is reasoning must emit no wire message — never a synthesized
    ``{role: <non-assistant>, content: '', <field>: ...}`` record. For role='tool'
    such a record would be schema-invalid (tool messages require ``tool_call_id``).
    """
    message = Message(
        role=role,
        contents=[Content.from_text_reasoning(text="chain", additional_properties={"openai_reasoning_format": marker})],
    )

    assert encode_message(message, variant=OPENAI) == []
