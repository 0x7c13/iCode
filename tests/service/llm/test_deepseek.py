# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import json
from typing import Any, Literal

import pytest
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta as ChunkChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall, Function

from chrys.kernel import ChatMiddlewareLayer, ChatResponse, Content, FunctionTool, Message
from chrys.service.llm.chat_completions import DeepSeekChatCompletionsClient
from chrys.service.llm.chat_completions.client import DEEPSEEK
from chrys.service.llm.chat_completions.decode import decode_completion
from chrys.service.llm.chat_completions.history import encode_message, encode_messages
from chrys.service.llm.openai_timestamps import openai_created_at_iso
from tests.service.llm._wire_stacks import make_chat_client
from tests.support.openai_chat_wire import parse_stream_chunks, scripted_openai
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer


class _UnusedCompletions:
    async def create(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("DeepSeek parser tests should not call the SDK")


class _UnusedChat:
    def __init__(self) -> None:
        self.completions = _UnusedCompletions()


class _UnusedAsyncOpenAI:
    base_url = "https://api.deepseek.test"

    def __init__(self) -> None:
        self.chat = _UnusedChat()


def _client() -> DeepSeekChatCompletionsClient:
    return DeepSeekChatCompletionsClient(model="deepseek-reasoner", sdk_client=_UnusedAsyncOpenAI())


def test_parse_reasoning_content_from_response() -> None:
    response = ChatCompletion(
        id="resp-1",
        object="chat.completion",
        created=1234567890,
        model="deepseek-reasoner",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Need to inspect the file.",
                    reasoning_content="Step-by-step thinking...",
                ),
                finish_reason="stop",
            )
        ],
    )

    parsed = decode_completion(response, {}, variant=DEEPSEEK)

    assert len(parsed.messages) == 1
    assert parsed.messages[0].contents[0].text == "Need to inspect the file."
    assert parsed.messages[0].contents[1].protected_data == json.dumps("Step-by-step thinking...")
    assert parsed.messages[0].contents[1].additional_properties["openai_reasoning_format"] == "reasoning_content"
    assert parsed.messages[0].additional_properties["reasoning_content"] == "Step-by-step thinking..."
    # Content-level metadata drives replay; message-level metadata preserves the raw provider field.
    assert parsed.messages[0].additional_properties["openai_reasoning_format"] == "reasoning_content"


def test_decode_completion_accepts_millisecond_created_timestamp() -> None:
    created_ms = 1_717_171_717_123
    response = ChatCompletion(
        id="resp-1",
        object="chat.completion",
        created=created_ms,
        model="deepseek-reasoner",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Done."),
                finish_reason="stop",
            )
        ],
    )

    parsed = decode_completion(response, {}, variant=DEEPSEEK)

    assert parsed.created_at == openai_created_at_iso(created_ms)


def test_stream_update_accepts_millisecond_created_timestamp() -> None:
    client = _client()
    created_ms = 1_717_171_717_123
    chunk = ChatCompletionChunk(
        id="chunk-1",
        object="chat.completion.chunk",
        created=created_ms,
        model="deepseek-reasoner",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChunkChoiceDelta(role="assistant", content="Done."),
                finish_reason=None,
            )
        ],
    )

    (parsed,) = parse_stream_chunks(client, chunk)

    assert parsed.created_at == openai_created_at_iso(created_ms)


def test_stream_update_skips_null_delta_finish_chunk() -> None:
    """Some OpenAI-compatible gateways send ``delta: null`` on finish chunks.

    Mirrors the guard in ``StreamState.update_for``.
    """
    client = _client()
    chunk = ChatCompletionChunk.model_construct(
        id="chunk-1",
        object="chat.completion.chunk",
        created=1_717_171_717,
        model="deepseek-reasoner",
        choices=[ChunkChoice.model_construct(index=0, delta=None, finish_reason="stop")],
    )

    (parsed,) = parse_stream_chunks(client, chunk)

    assert parsed.finish_reason == "stop"
    assert all(c.type != "text" for c in parsed.contents)
    assert all(c.type != "function_call" for c in parsed.contents)


def test_encode_message_with_reasoning_content_before_function_call() -> None:
    message = Message(
        role="assistant",
        contents=[
            Content.from_text_reasoning(
                text=None,
                protected_data=json.dumps("Analyzing before tool call"),
                additional_properties={"openai_reasoning_format": "reasoning_content"},
            ),
            Content.from_function_call(call_id="call_abc", name="get_weather", arguments='{"city":"Seattle"}'),
        ],
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert len(prepared) == 1
    assert prepared[0]["content"] == ""
    assert prepared[0]["reasoning_content"] == "Analyzing before tool call"
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "reasoning_details" not in prepared[0]


def test_encode_message_with_message_level_reasoning_content() -> None:
    message = Message(
        role="assistant",
        contents=[
            Content.from_function_call(call_id="call_abc", name="get_weather", arguments='{"city":"Seattle"}'),
        ],
        additional_properties={
            "reasoning_content": "Analyzing before tool call",
            "openai_reasoning_format": "reasoning_content",
        },
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert len(prepared) == 1
    assert prepared[0]["content"] == ""
    assert prepared[0]["reasoning_content"] == "Analyzing before tool call"
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_encode_message_preserves_empty_message_level_reasoning_content() -> None:
    message = Message(
        role="assistant",
        contents=[
            Content.from_function_call(call_id="call_abc", name="get_weather", arguments='{"city":"Seattle"}'),
        ],
        additional_properties={
            "reasoning_content": "",
            "openai_reasoning_format": "reasoning_content",
        },
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert len(prepared) == 1
    assert prepared[0]["content"] == ""
    assert "reasoning_content" in prepared[0]
    assert prepared[0]["reasoning_content"] == ""
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_encode_message_with_text_function_call_and_reasoning_content_stays_single_message() -> None:
    message = Message(
        role="assistant",
        contents=[
            Content.from_text(text="Let me check that."),
            Content.from_function_call(call_id="call_abc", name="get_weather", arguments='{"city":"Seattle"}'),
            Content.from_text_reasoning(
                text=None,
                protected_data=json.dumps("Deciding to call a function"),
                additional_properties={"openai_reasoning_format": "reasoning_content"},
            ),
        ],
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert len(prepared) == 1
    assert prepared[0]["content"] == "Let me check that."
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert prepared[0]["reasoning_content"] == "Deciding to call a function"


def test_encode_message_multiple_user_text_contents_stays_single_message() -> None:
    message = Message("user", ["hello", "world"])

    prepared = encode_message(message, variant=DEEPSEEK)

    assert prepared == [{"role": "user", "content": "hello\nworld"}]


def test_encode_message_user_multimodal_contents_stays_single_message() -> None:
    message = Message(
        role="user",
        contents=[
            Content.from_text("Describe this image."),
            Content.from_uri(uri="https://example.com/image.png", media_type="image/png"),
        ],
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "user"
    assert isinstance(prepared[0]["content"], list)
    assert prepared[0]["content"][0]["type"] == "text"
    assert prepared[0]["content"][0]["text"] == "Describe this image."
    assert prepared[0]["content"][1]["type"] == "image_url"


def test_encode_message_tool_results_stay_split() -> None:
    message = Message(
        role="tool",
        contents=[
            Content.from_function_result(call_id="call_1", result="one"),
            Content.from_function_result(call_id="call_2", result="two"),
        ],
    )

    prepared = encode_message(message, variant=DEEPSEEK)

    assert prepared == [
        {"role": "tool", "tool_call_id": "call_1", "content": "one"},
        {"role": "tool", "tool_call_id": "call_2", "content": "two"},
    ]


def test_parse_empty_tool_call_content_replays_content_field() -> None:
    response = ChatCompletion(
        id="resp-1",
        object="chat.completion",
        created=1234567890,
        model="deepseek-reasoner",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="",
                    reasoning_content="Need to call the tool",
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id="call_abc",
                            type="function",
                            function=Function(name="get_weather", arguments='{"city":"Seattle"}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    parsed = decode_completion(response, {}, variant=DEEPSEEK)
    prepared = encode_messages(
        [
            Message("user", ["weather?"]),
            parsed.messages[0],
            Message("tool", [Content.from_function_result(call_id="call_abc", result="rain")]),
        ],
        variant=DEEPSEEK,
    )

    assistant_message = prepared[1]
    assert assistant_message["content"] == ""
    assert assistant_message["reasoning_content"] == "Need to call the tool"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "get_weather"


def test_encode_messages_omits_reasoning_content_without_tool_interaction() -> None:
    messages = [
        Message("user", ["First question"]),
        Message(
            role="assistant",
            contents=[
                Content.from_text("Final answer"),
                Content.from_text_reasoning(
                    text=None,
                    protected_data=json.dumps("No tool reasoning"),
                    additional_properties={"openai_reasoning_format": "reasoning_content"},
                ),
            ],
            additional_properties={
                "reasoning_content": "No tool reasoning",
                "openai_reasoning_format": "reasoning_content",
            },
        ),
        Message("user", ["Follow-up"]),
    ]

    prepared = encode_messages(messages, variant=DEEPSEEK)

    assistant_message = prepared[1]
    assert assistant_message["content"] == "Final answer"
    assert "reasoning_content" not in assistant_message
    assert "reasoning_details" not in assistant_message


def test_encode_messages_preserves_reasoning_content_after_tool_interaction() -> None:
    messages = [
        Message("user", ["Inspect the file"]),
        Message(
            role="assistant",
            contents=[
                Content.from_text("I will inspect it."),
                Content.from_function_call(call_id="call_abc", name="read_file", arguments='{"path":"foo.py"}'),
                Content.from_text_reasoning(
                    text=None,
                    protected_data=json.dumps("Tool planning"),
                    additional_properties={"openai_reasoning_format": "reasoning_content"},
                ),
            ],
        ),
        Message("tool", [Content.from_function_result(call_id="call_abc", result="file contents")]),
        Message(
            role="assistant",
            contents=[
                Content.from_text("Final answer"),
                Content.from_text_reasoning(
                    text=None,
                    protected_data=json.dumps("Final tool-based reasoning"),
                    additional_properties={"openai_reasoning_format": "reasoning_content"},
                ),
            ],
        ),
        Message("user", ["Follow-up"]),
    ]

    prepared = encode_messages(messages, variant=DEEPSEEK)

    assert prepared[1]["reasoning_content"] == "Tool planning"
    assert prepared[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert prepared[3]["content"] == "Final answer"
    assert prepared[3]["reasoning_content"] == "Final tool-based reasoning"


def test_streaming_reasoning_content_accumulates_and_replays_on_tool_call_message() -> None:
    client = _client()
    first_chunk = ChatCompletionChunk(
        id="chunk-1",
        object="chat.completion.chunk",
        created=1234567890,
        model="deepseek-reasoner",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChunkChoiceDelta(role="assistant", reasoning_content="first "),
                finish_reason=None,
            )
        ],
    )
    second_chunk = ChatCompletionChunk(
        id="chunk-1",
        object="chat.completion.chunk",
        created=1234567890,
        model="deepseek-reasoner",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChunkChoiceDelta(role="assistant", reasoning_content="second"),
                finish_reason=None,
            )
        ],
    )

    response = ChatResponse.from_updates(parse_stream_chunks(client, first_chunk, second_chunk))

    tool_call_message = Message(
        role="assistant",
        contents=[
            *response.messages[0].contents,
            Content.from_function_call(call_id="call_abc", name="get_weather", arguments='{"city":"Seattle"}'),
        ],
    )

    prepared = encode_message(tool_call_message, variant=client.VARIANT)

    assert len(prepared) == 1
    assert prepared[0]["reasoning_content"] == "first second"
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"


def _tool_loop_client(sdk_client: AsyncOpenAI) -> InvariantCheckedToolLoopLayer:
    return InvariantCheckedToolLoopLayer(
        ChatMiddlewareLayer(DeepSeekChatCompletionsClient(model="deepseek-reasoner", sdk_client=sdk_client))
    )


def _deepseek_completion(
    message: ChatCompletionMessage,
    *,
    finish_reason: Literal["stop", "tool_calls"],
    response_id: str = "resp-1",
    created: int = 1234567890,
) -> ChatCompletion:
    return ChatCompletion(
        id=response_id,
        object="chat.completion",
        created=created,
        model="deepseek-reasoner",
        choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
    )


def _read_file_call(call_id: str = "call_abc") -> ChatCompletionMessageFunctionToolCall:
    return ChatCompletionMessageFunctionToolCall(
        id=call_id,
        type="function",
        function=Function(name="read_file", arguments='{"path":"foo.py"}'),
    )


def _done() -> ChatCompletion:
    return _deepseek_completion(
        ChatCompletionMessage(role="assistant", content="Done."),
        finish_reason="stop",
        response_id="resp-2",
        created=1234567891,
    )


def _read_file(path: str) -> str:
    return f"contents of {path}"


@pytest.mark.asyncio
async def test_tool_loop_replays_reasoning_content_after_function_result() -> None:
    tool_call = _deepseek_completion(
        ChatCompletionMessage(
            role="assistant",
            content="Need to inspect the file.",
            reasoning_content="Step-by-step thinking...",
            tool_calls=[_read_file_call()],
        ),
        finish_reason="tool_calls",
    )
    tool = FunctionTool(name="read_file", description="Read a file", func=_read_file)

    async with scripted_openai([tool_call, _done()]) as wire:
        response = await _tool_loop_client(wire.client).get_response(
            [Message("user", ["inspect foo.py"])], options={"tools": [tool]}
        )

    assert response.text == "Need to inspect the file.\n\nDone."
    assert len(wire.requests) == 2
    second_request_messages = wire.requests[1]["messages"]
    assert second_request_messages[1]["reasoning_content"] == "Step-by-step thinking..."
    assert second_request_messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert second_request_messages[2]["tool_call_id"] == "call_abc"
    assert second_request_messages[2]["content"] == "contents of foo.py"


@pytest.mark.asyncio
async def test_tool_loop_replays_empty_reasoning_content_after_function_result() -> None:
    tool_call = _deepseek_completion(
        ChatCompletionMessage(
            role="assistant",
            content="Need to inspect the file.",
            reasoning_content="",
            tool_calls=[_read_file_call()],
        ),
        finish_reason="tool_calls",
    )
    tool = FunctionTool(name="read_file", description="Read a file", func=_read_file)

    async with scripted_openai([tool_call, _done()]) as wire:
        await _tool_loop_client(wire.client).get_response(
            [Message("user", ["inspect foo.py"])], options={"tools": [tool]}
        )

    assert len(wire.requests) == 2
    second_request_messages = wire.requests[1]["messages"]
    assert "reasoning_content" in second_request_messages[1]
    assert second_request_messages[1]["reasoning_content"] == ""
    assert second_request_messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert second_request_messages[2]["tool_call_id"] == "call_abc"


@pytest.mark.asyncio
async def test_tool_loop_replays_reasoning_content_after_parallel_function_results() -> None:
    tool_calls = _deepseek_completion(
        ChatCompletionMessage(
            role="assistant",
            content="Need to inspect two files.",
            reasoning_content="Parallel planning...",
            tool_calls=[
                _read_file_call("call_read"),
                ChatCompletionMessageFunctionToolCall(
                    id="call_grep",
                    type="function",
                    function=Function(name="grep", arguments='{"pattern":"needle"}'),
                ),
            ],
        ),
        finish_reason="tool_calls",
    )

    def grep(pattern: str) -> str:
        return f"matches for {pattern}"

    tools = [
        FunctionTool(name="read_file", description="Read a file", func=_read_file),
        FunctionTool(name="grep", description="Search text", func=grep),
    ]

    async with scripted_openai([tool_calls, _done()]) as wire:
        await _tool_loop_client(wire.client).get_response(
            [Message("user", ["inspect foo.py and grep"])], options={"tools": tools}
        )

    assert len(wire.requests) == 2
    second_request_messages = wire.requests[1]["messages"]
    assistant_message = second_request_messages[1]
    assert assistant_message["reasoning_content"] == "Parallel planning..."
    assert [call["function"]["name"] for call in assistant_message["tool_calls"]] == ["read_file", "grep"]

    tool_results = {message["tool_call_id"]: message["content"] for message in second_request_messages[2:]}
    assert tool_results == {
        "call_read": "contents of foo.py",
        "call_grep": "matches for needle",
    }


def _foreign_tool_history() -> list[Message]:
    """A tool exchange another model wrote: no reasoning on either assistant message."""
    return [
        Message("user", ["Inspect the file"]),
        Message(
            "assistant",
            [
                Content.from_text("I will inspect it."),
                Content.from_function_call(call_id="call_abc", name="read_file", arguments='{"path":"foo.py"}'),
            ],
        ),
        Message("tool", [Content.from_function_result(call_id="call_abc", result="file contents")]),
        Message("assistant", ["It defines foo."]),
        Message("user", ["Follow-up"]),
    ]


def test_encode_messages_sends_empty_reasoning_content_for_another_models_tool_history() -> None:
    messages = _foreign_tool_history()

    prepared = encode_messages(messages, variant=DEEPSEEK)

    assistants = [message for message in prepared if message["role"] == "assistant"]
    assert [message["reasoning_content"] for message in assistants] == ["", ""]
    assert assistants[0]["tool_calls"][0]["function"]["name"] == "read_file"
    assert all("reasoning_content" not in message for message in prepared if message["role"] != "assistant")
    assert all("reasoning_content" not in message.additional_properties for message in messages)


def test_encode_messages_fills_only_assistant_messages_missing_reasoning_content() -> None:
    messages = _foreign_tool_history()
    messages[3] = Message(
        "assistant",
        [
            Content.from_text("It defines foo."),
            Content.from_text_reasoning(
                text=None,
                protected_data=json.dumps("DeepSeek reasoning"),
                additional_properties={"openai_reasoning_format": "reasoning_content"},
            ),
        ],
    )

    prepared = encode_messages(messages, variant=DEEPSEEK)

    assert [message["reasoning_content"] for message in prepared if message["role"] == "assistant"] == [
        "",
        "DeepSeek reasoning",
    ]


def test_encode_messages_adds_no_reasoning_content_without_tool_interaction() -> None:
    messages = [Message("user", ["Hi"]), Message("assistant", ["Hello"]), Message("user", ["Again"])]

    prepared = encode_messages(messages, variant=DEEPSEEK)

    assert all("reasoning_content" not in message for message in prepared)


def _answered_without_tool_call() -> ChatCompletion:
    return _deepseek_completion(
        ChatCompletionMessage(role="assistant", content="Done.", reasoning_content="ok"),
        finish_reason="stop",
    )


async def test_request_after_another_models_tool_history_carries_empty_reasoning_content() -> None:
    async with scripted_openai([_answered_without_tool_call()]) as wire:
        await _tool_loop_client(wire.client).get_response(_foreign_tool_history())

    [request] = wire.requests
    assistants = [message for message in request["messages"] if message["role"] == "assistant"]
    assert [message["reasoning_content"] for message in assistants] == ["", ""]


def test_production_client_keeps_the_empty_reasoning_content() -> None:
    """The production client canonicalizes each message first; the fill must survive it."""
    chat_client: Any = make_chat_client(chat_client_cls=DeepSeekChatCompletionsClient)

    prepared = encode_messages(_foreign_tool_history(), variant=chat_client.VARIANT)

    assert [message["reasoning_content"] for message in prepared if message["role"] == "assistant"] == ["", ""]


def _history_without_tool_calls() -> list[Message]:
    """Turns answered without a tool call: one by DeepSeek with reasoning, one by another model."""
    return [
        Message("user", ["First question"]),
        Message(
            "assistant",
            [
                Content.from_text("First answer"),
                Content.from_text_reasoning(
                    text=None,
                    protected_data=json.dumps("DeepSeek reasoning"),
                    additional_properties={"openai_reasoning_format": "reasoning_content"},
                ),
            ],
        ),
        Message("user", ["Second question"]),
        Message("assistant", ["Second answer"]),
        Message("user", ["Follow-up"]),
    ]


def test_encode_messages_replays_reasoning_content_for_a_request_with_tools() -> None:
    """With tools, thinking mode wants every turn's reasoning back, tool call or not."""
    prepared = encode_messages(_history_without_tool_calls(), request_has_tools=True, variant=DEEPSEEK)

    assert [message.get("reasoning_content") for message in prepared if message["role"] == "assistant"] == [
        "DeepSeek reasoning",
        "",
    ]


@pytest.mark.parametrize(
    ("with_tools", "expected"),
    [(True, ["DeepSeek reasoning", ""]), (False, [None, None])],
    ids=["with-tools", "without-tools"],
)
async def test_request_replays_reasoning_content_when_it_carries_tools(
    with_tools: bool, expected: list[str | None]
) -> None:
    tool = FunctionTool(name="read_file", description="Read a file", func=_read_file)

    async with scripted_openai([_answered_without_tool_call()]) as wire:
        await _tool_loop_client(wire.client).get_response(
            _history_without_tool_calls(), options={"tools": [tool]} if with_tools else None
        )

    [request] = wire.requests
    assert ("tools" in request) is with_tools
    assert [message.get("reasoning_content") for message in request["messages"] if message["role"] == "assistant"] == (
        expected
    )
