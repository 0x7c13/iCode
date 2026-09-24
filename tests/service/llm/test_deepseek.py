# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import json
from typing import Any

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta as ChunkChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall, Function

from chrys.kernel import ChatMiddlewareLayer, ChatResponse, Content, FunctionTool, Message
from chrys.service.llm.deepseek import DeepSeekChatCompletionClient
from chrys.service.llm.openai_timestamps import openai_created_at_iso
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


def _client() -> DeepSeekChatCompletionClient:
    return DeepSeekChatCompletionClient(model="deepseek-reasoner", async_client=_UnusedAsyncOpenAI())


def test_parse_reasoning_content_from_response() -> None:
    client = _client()
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

    parsed = client._parse_response_from_openai(response, {})

    assert len(parsed.messages) == 1
    assert parsed.messages[0].contents[0].text == "Need to inspect the file."
    assert parsed.messages[0].contents[1].protected_data == json.dumps("Step-by-step thinking...")
    assert parsed.messages[0].contents[1].additional_properties["openai_reasoning_format"] == "reasoning_content"
    assert parsed.messages[0].additional_properties["reasoning_content"] == "Step-by-step thinking..."
    # Content-level metadata drives replay; message-level metadata preserves the raw provider field.
    assert parsed.messages[0].additional_properties["openai_reasoning_format"] == "reasoning_content"


def test_parse_response_accepts_millisecond_created_timestamp() -> None:
    client = _client()
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

    parsed = client._parse_response_from_openai(response, {})

    assert parsed.created_at == openai_created_at_iso(created_ms)


def test_parse_response_update_accepts_millisecond_created_timestamp() -> None:
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

    parsed = client._parse_response_update_from_openai(chunk)

    assert parsed.created_at == openai_created_at_iso(created_ms)


def test_parse_response_update_skips_null_delta_finish_chunk() -> None:
    """Some OpenAI-compatible gateways send ``delta: null`` on finish chunks.

    Mirrors the guard in
    ``RawOpenAIChatCompletionClient._parse_response_update_from_openai``.
    """
    client = _client()
    chunk = ChatCompletionChunk.model_construct(
        id="chunk-1",
        object="chat.completion.chunk",
        created=1_717_171_717,
        model="deepseek-reasoner",
        choices=[ChunkChoice.model_construct(index=0, delta=None, finish_reason="stop")],
    )

    parsed = client._parse_response_update_from_openai(chunk)

    assert parsed.finish_reason == "stop"
    assert all(c.type != "text" for c in parsed.contents)
    assert all(c.type != "function_call" for c in parsed.contents)


def test_prepare_message_with_reasoning_content_before_function_call() -> None:
    client = _client()
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

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    assert prepared[0]["content"] == ""
    assert prepared[0]["reasoning_content"] == "Analyzing before tool call"
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert "reasoning_details" not in prepared[0]


def test_prepare_message_with_message_level_reasoning_content() -> None:
    client = _client()
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

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    assert prepared[0]["content"] == ""
    assert prepared[0]["reasoning_content"] == "Analyzing before tool call"
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_prepare_message_preserves_empty_message_level_reasoning_content() -> None:
    client = _client()
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

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    assert prepared[0]["content"] == ""
    assert "reasoning_content" in prepared[0]
    assert prepared[0]["reasoning_content"] == ""
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_prepare_message_with_text_function_call_and_reasoning_content_stays_single_message() -> None:
    client = _client()
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

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    assert prepared[0]["content"] == "Let me check that."
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert prepared[0]["reasoning_content"] == "Deciding to call a function"


def test_prepare_message_multiple_user_text_contents_stays_single_message() -> None:
    client = _client()
    message = Message("user", ["hello", "world"])

    prepared = client._prepare_message_for_openai(message)

    assert prepared == [{"role": "user", "content": "hello\nworld"}]


def test_prepare_message_user_multimodal_contents_stays_single_message() -> None:
    client = _client()
    message = Message(
        role="user",
        contents=[
            Content.from_text("Describe this image."),
            Content.from_uri(uri="https://example.com/image.png", media_type="image/png"),
        ],
    )

    prepared = client._prepare_message_for_openai(message)

    assert len(prepared) == 1
    assert prepared[0]["role"] == "user"
    assert isinstance(prepared[0]["content"], list)
    assert prepared[0]["content"][0]["type"] == "text"
    assert prepared[0]["content"][0]["text"] == "Describe this image."
    assert prepared[0]["content"][1]["type"] == "image_url"


def test_prepare_message_tool_results_stay_split() -> None:
    client = _client()
    message = Message(
        role="tool",
        contents=[
            Content.from_function_result(call_id="call_1", result="one"),
            Content.from_function_result(call_id="call_2", result="two"),
        ],
    )

    prepared = client._prepare_message_for_openai(message)

    assert prepared == [
        {"role": "tool", "tool_call_id": "call_1", "content": "one"},
        {"role": "tool", "tool_call_id": "call_2", "content": "two"},
    ]


def test_parse_empty_tool_call_content_replays_content_field() -> None:
    client = _client()
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

    parsed = client._parse_response_from_openai(response, {})
    prepared = client._prepare_messages_for_openai(
        [
            Message("user", ["weather?"]),
            parsed.messages[0],
            Message("tool", [Content.from_function_result(call_id="call_abc", result="rain")]),
        ]
    )

    assistant_message = prepared[1]
    assert assistant_message["content"] == ""
    assert assistant_message["reasoning_content"] == "Need to call the tool"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "get_weather"


def test_prepare_messages_omits_reasoning_content_without_tool_interaction() -> None:
    client = _client()
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

    prepared = client._prepare_messages_for_openai(messages)

    assistant_message = prepared[1]
    assert assistant_message["content"] == "Final answer"
    assert "reasoning_content" not in assistant_message
    assert "reasoning_details" not in assistant_message


def test_prepare_messages_preserves_reasoning_content_after_tool_interaction() -> None:
    client = _client()
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

    prepared = client._prepare_messages_for_openai(messages)

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

    response = ChatResponse.from_updates(
        [
            client._parse_response_update_from_openai(first_chunk),
            client._parse_response_update_from_openai(second_chunk),
        ]
    )

    tool_call_message = Message(
        role="assistant",
        contents=[
            *response.messages[0].contents,
            Content.from_function_call(call_id="call_abc", name="get_weather", arguments='{"city":"Seattle"}'),
        ],
    )

    prepared = client._prepare_message_for_openai(tool_call_message)

    assert len(prepared) == 1
    assert prepared[0]["reasoning_content"] == "first second"
    assert prepared[0]["tool_calls"][0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_tool_loop_replays_reasoning_content_after_function_result() -> None:
    captured_requests: list[dict[str, Any]] = []

    class _FakeCompletions:
        def __init__(self) -> None:
            self._call_count = 0

        async def create(self, stream: bool = False, **kwargs: Any) -> ChatCompletion:
            captured_requests.append(kwargs)
            self._call_count += 1
            if self._call_count == 1:
                return ChatCompletion(
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
                                tool_calls=[
                                    ChatCompletionMessageFunctionToolCall(
                                        id="call_abc",
                                        type="function",
                                        function=Function(name="read_file", arguments='{"path":"foo.py"}'),
                                    )
                                ],
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
            return ChatCompletion(
                id="resp-2",
                object="chat.completion",
                created=1234567891,
                model="deepseek-reasoner",
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content="Done."),
                        finish_reason="stop",
                    )
                ],
            )

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    def read_file(path: str) -> str:
        return f"contents of {path}"

    client = InvariantCheckedToolLoopLayer(
        ChatMiddlewareLayer(DeepSeekChatCompletionClient(model="deepseek-reasoner", async_client=_FakeAsyncOpenAI()))
    )
    tool = FunctionTool(name="read_file", description="Read a file", func=read_file)

    response = await client.get_response([Message("user", ["inspect foo.py"])], options={"tools": [tool]})

    assert response.text == "Need to inspect the file.\n\nDone."
    assert len(captured_requests) == 2
    second_request_messages = captured_requests[1]["messages"]
    assert second_request_messages[1]["reasoning_content"] == "Step-by-step thinking..."
    assert second_request_messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert second_request_messages[2]["tool_call_id"] == "call_abc"
    assert second_request_messages[2]["content"] == "contents of foo.py"


@pytest.mark.asyncio
async def test_tool_loop_replays_empty_reasoning_content_after_function_result() -> None:
    captured_requests: list[dict[str, Any]] = []

    class _FakeCompletions:
        def __init__(self) -> None:
            self._call_count = 0

        async def create(self, stream: bool = False, **kwargs: Any) -> ChatCompletion:
            captured_requests.append(kwargs)
            self._call_count += 1
            if self._call_count == 1:
                return ChatCompletion(
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
                                reasoning_content="",
                                tool_calls=[
                                    ChatCompletionMessageFunctionToolCall(
                                        id="call_abc",
                                        type="function",
                                        function=Function(name="read_file", arguments='{"path":"foo.py"}'),
                                    )
                                ],
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
            return ChatCompletion(
                id="resp-2",
                object="chat.completion",
                created=1234567891,
                model="deepseek-reasoner",
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content="Done."),
                        finish_reason="stop",
                    )
                ],
            )

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    def read_file(path: str) -> str:
        return f"contents of {path}"

    client = InvariantCheckedToolLoopLayer(
        ChatMiddlewareLayer(DeepSeekChatCompletionClient(model="deepseek-reasoner", async_client=_FakeAsyncOpenAI()))
    )
    tool = FunctionTool(name="read_file", description="Read a file", func=read_file)

    await client.get_response([Message("user", ["inspect foo.py"])], options={"tools": [tool]})

    assert len(captured_requests) == 2
    second_request_messages = captured_requests[1]["messages"]
    assert "reasoning_content" in second_request_messages[1]
    assert second_request_messages[1]["reasoning_content"] == ""
    assert second_request_messages[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert second_request_messages[2]["tool_call_id"] == "call_abc"


@pytest.mark.asyncio
async def test_tool_loop_replays_reasoning_content_after_parallel_function_results() -> None:
    captured_requests: list[dict[str, Any]] = []

    class _FakeCompletions:
        def __init__(self) -> None:
            self._call_count = 0

        async def create(self, stream: bool = False, **kwargs: Any) -> ChatCompletion:
            captured_requests.append(kwargs)
            self._call_count += 1
            if self._call_count == 1:
                return ChatCompletion(
                    id="resp-1",
                    object="chat.completion",
                    created=1234567890,
                    model="deepseek-reasoner",
                    choices=[
                        Choice(
                            index=0,
                            message=ChatCompletionMessage(
                                role="assistant",
                                content="Need to inspect two files.",
                                reasoning_content="Parallel planning...",
                                tool_calls=[
                                    ChatCompletionMessageFunctionToolCall(
                                        id="call_read",
                                        type="function",
                                        function=Function(name="read_file", arguments='{"path":"foo.py"}'),
                                    ),
                                    ChatCompletionMessageFunctionToolCall(
                                        id="call_grep",
                                        type="function",
                                        function=Function(name="grep", arguments='{"pattern":"needle"}'),
                                    ),
                                ],
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
            return ChatCompletion(
                id="resp-2",
                object="chat.completion",
                created=1234567891,
                model="deepseek-reasoner",
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content="Done."),
                        finish_reason="stop",
                    )
                ],
            )

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    def read_file(path: str) -> str:
        return f"contents of {path}"

    def grep(pattern: str) -> str:
        return f"matches for {pattern}"

    client = InvariantCheckedToolLoopLayer(
        ChatMiddlewareLayer(DeepSeekChatCompletionClient(model="deepseek-reasoner", async_client=_FakeAsyncOpenAI()))
    )
    tools = [
        FunctionTool(name="read_file", description="Read a file", func=read_file),
        FunctionTool(name="grep", description="Search text", func=grep),
    ]

    await client.get_response([Message("user", ["inspect foo.py and grep"])], options={"tools": tools})

    assert len(captured_requests) == 2
    second_request_messages = captured_requests[1]["messages"]
    assistant_message = second_request_messages[1]
    assert assistant_message["reasoning_content"] == "Parallel planning..."
    assert [call["function"]["name"] for call in assistant_message["tool_calls"]] == ["read_file", "grep"]

    tool_results = {message["tool_call_id"]: message["content"] for message in second_request_messages[2:]}
    assert tool_results == {
        "call_read": "contents of foo.py",
        "call_grep": "matches for needle",
    }
