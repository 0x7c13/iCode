# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The wire codecs themselves carry their provider-compatibility repairs.

Every client stack wraps one of these loop-free codec classes, so a repair
pinned here holds for the main agent, judges, sub-agents and workflow nodes
alike.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

from chrys.kernel import ChatClientException, Content, Message
from chrys.service.llm.anthropic_messages.decode import decode_usage
from chrys.service.llm.anthropic_messages.history import encode_messages
from chrys.service.llm.chat_completions import ChatCompletionsClient, DeepSeekChatCompletionsClient
from chrys.service.llm.chat_completions import history as chat_history
from chrys.service.llm.chat_completions.client import OPENAI
from chrys.service.llm.chat_completions.decode import decode_completion
from tests.support.openai_chat_wire import parse_stream_chunks


def _chat_completions(
    cls: type[ChatCompletionsClient] = ChatCompletionsClient,
) -> ChatCompletionsClient:
    return cls(model="gpt-test", sdk_client=AsyncOpenAI(api_key="sk-test"))


@pytest.mark.parametrize("cls", [ChatCompletionsClient, DeepSeekChatCompletionsClient])
def test_chat_completions_sends_text_and_a_repaired_tool_call_as_one_message(
    cls: type[ChatCompletionsClient],
) -> None:
    wire = chat_history.encode_messages(
        [
            Message(
                "assistant",
                [
                    Content.from_text("I'll read it."),
                    Content.from_function_call(call_id="call_1", name="read_file", arguments="{"),
                ],
            )
        ],
        variant=cls.VARIANT,
    )

    assert len(wire) == 1
    assert wire[0]["content"] == "I'll read it."
    assert wire[0]["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    ]


def test_chat_completions_rejects_a_completion_without_choices() -> None:
    gateway_error = ChatCompletion.model_construct(
        id="r",
        choices=None,
        created=0,
        model="gpt-test",
        object="chat.completion",
        error={"message": "gateway oops"},
    )

    with pytest.raises(ChatClientException, match="gateway oops"):
        decode_completion(gateway_error, {}, variant=OPENAI)


def test_chat_completions_update_keeps_a_seconds_copy_of_a_millisecond_created() -> None:
    created_ms = 1_717_171_717_123
    chunk = ChatCompletionChunk(
        id="chunk-1",
        object="chat.completion.chunk",
        created=created_ms,
        model="gpt-test",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant", content="hi"), finish_reason=None)],
    )

    (update,) = parse_stream_chunks(_chat_completions(), chunk)

    assert isinstance(update.raw_representation, ChatCompletionChunk)
    assert update.raw_representation.created == created_ms / 1000
    assert chunk.created == created_ms


def test_anthropic_drops_unsigned_thinking_and_the_messages_it_empties() -> None:
    wire = encode_messages(
        [
            Message("user", ["before"]),
            Message(
                "assistant",
                [
                    Content.from_text_reasoning(text="unsigned"),
                    Content.from_text_reasoning(text="signed", protected_data="sig-1"),
                ],
            ),
            Message("assistant", [Content.from_text_reasoning(text="unsigned only")]),
            Message("user", ["after"]),
        ]
    )

    assert wire == [
        {"role": "user", "content": [{"type": "text", "text": "before"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "signed", "signature": "sig-1"}]},
        {"role": "user", "content": [{"type": "text", "text": "after"}]},
    ]


def test_anthropic_counts_cache_tokens_as_input_once() -> None:
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=20,
        cache_read_input_tokens=30,
    )

    details = decode_usage(usage)  # type: ignore[arg-type]

    assert details is not None
    assert details["input_token_count"] == 150
    assert details["cache_creation_input_token_count"] == 20
    assert details["cache_read_input_token_count"] == 30
