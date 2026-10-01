# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chat-compatible tool-result image lowering tests."""

from __future__ import annotations

from chrys.kernel import Content, Message
from chrys.service.llm.anthropic_messages.history import encode_messages
from chrys.service.llm.chat_completions import history as chat_history
from chrys.service.llm.chat_completions.client import DEEPSEEK, OPENAI
from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.replay import encode_input


def _image_result(call_id: str, text: str = "caption") -> Content:
    return Content.from_function_result(
        call_id,
        result=[
            Content.from_text(text),
            Content.from_data(b"image-bytes", "image/png"),
        ],
    )


def test_openai_chat_lowers_single_tool_result_image_after_tool_message() -> None:
    prepared = chat_history.encode_messages([Message("tool", [_image_result("call_1")])], variant=OPENAI)

    assert len(prepared) == 2
    assert prepared[0] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "caption\n(see following user message for image)",
    }
    assert all("_chrys_pending_image_parts" not in message for message in prepared)
    assert prepared[1]["role"] == "user"
    content = prepared[1]["content"]
    assert content[0] == {"type": "text", "text": "Image from tool call call_1, item 2:"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_chat_buffers_multiple_tool_result_images_into_one_user_message() -> None:
    prepared = chat_history.encode_messages(
        [Message("tool", [_image_result("call_1", "one"), _image_result("call_2", "two")])], variant=OPENAI
    )

    assert [message["role"] for message in prepared] == ["tool", "tool", "user"]
    assert prepared[0]["tool_call_id"] == "call_1"
    assert prepared[1]["tool_call_id"] == "call_2"
    user_content = prepared[2]["content"]
    assert user_content[0] == {"type": "text", "text": "Image from tool call call_1, item 2:"}
    assert user_content[2] == {"type": "text", "text": "Image from tool call call_2, item 2:"}


def test_openai_chat_buffers_parallel_mixed_tool_results_after_all_tools() -> None:
    assistant = Message(
        "assistant",
        [
            Content.from_function_call("call_image", "view_image", arguments='{"path":"plot.png"}'),
            Content.from_function_call("call_text", "read_file", arguments='{"path":"notes.txt"}'),
        ],
    )
    tool_results = Message(
        "tool",
        [
            _image_result("call_image", "image caption"),
            Content.from_function_result("call_text", result="file contents"),
        ],
    )

    prepared = chat_history.encode_messages([assistant, tool_results], variant=OPENAI)

    assert [message["role"] for message in prepared] == ["assistant", "tool", "tool", "user"]
    assert len(prepared[0]["tool_calls"]) == 2
    assert prepared[1] == {
        "role": "tool",
        "tool_call_id": "call_image",
        "content": "image caption\n(see following user message for image)",
    }
    assert prepared[2] == {"role": "tool", "tool_call_id": "call_text", "content": "file contents"}
    user_content = prepared[3]["content"]
    assert user_content[0] == {"type": "text", "text": "Image from tool call call_image, item 2:"}
    assert user_content[1]["type"] == "image_url"


def test_openai_chat_flushes_image_message_before_following_non_tool_message() -> None:
    assistant = Message(
        "assistant",
        [Content.from_function_call("call_image", "view_image", arguments='{"path":"plot.png"}')],
    )
    tool_results = Message("tool", [_image_result("call_image", "image caption")])
    followup = Message("user", [Content.from_text("next user turn")])

    prepared = chat_history.encode_messages([assistant, tool_results, followup], variant=OPENAI)

    assert [message["role"] for message in prepared] == ["assistant", "tool", "user", "user"]
    assert prepared[1]["tool_call_id"] == "call_image"
    synthetic_content = prepared[2]["content"]
    assert synthetic_content[0] == {"type": "text", "text": "Image from tool call call_image, item 2:"}
    assert synthetic_content[1]["type"] == "image_url"
    assert prepared[3] == {"role": "user", "content": "next user turn"}


def test_openai_chat_does_not_lower_non_image_rich_tool_results() -> None:
    result = Content.from_function_result(
        "call_1",
        result=[Content.from_text("audio"), Content.from_data(b"audio-bytes", "audio/wav")],
    )

    prepared = chat_history.encode_messages([Message("tool", [result])], variant=OPENAI)

    assert prepared == [{"role": "tool", "tool_call_id": "call_1", "content": "audio"}]


def test_openai_chat_omits_unknown_rich_tool_results_without_crashing() -> None:
    result = Content.from_function_result(
        "call_1",
        result=[Content.from_text("unknown"), Content.from_uri("https://example.com/blob")],
    )

    prepared = chat_history.encode_messages([Message("tool", [result])], variant=OPENAI)

    assert prepared == [{"role": "tool", "tool_call_id": "call_1", "content": "unknown"}]


def test_deepseek_calls_shared_post_pass_and_preserves_reasoning_fields() -> None:
    tool_call = Content.from_function_call("call_1", "view_image", arguments="{}")
    assistant = Message(
        "assistant",
        [tool_call],
        additional_properties={"reasoning_content": "look first", "openai_reasoning_format": "reasoning_content"},
    )
    tool = Message("tool", [_image_result("call_1")])

    prepared = chat_history.encode_messages([assistant, tool], variant=DEEPSEEK)

    assert prepared[0]["role"] == "assistant"
    assert prepared[0]["reasoning_content"] == "look first"
    assert prepared[1]["role"] == "tool"
    assert prepared[2]["role"] == "user"
    assert prepared[2]["content"][0] == {"type": "text", "text": "Image from tool call call_1, item 2:"}
    assert all("_chrys_pending_image_parts" not in message for message in prepared)


def test_responses_keeps_native_rich_function_output_shape() -> None:
    prepared = encode_input([Message("tool", [_image_result("call_1")])], service_side=False, variant=OPENAI_RESPONSES)

    assert len(prepared) == 1
    output = prepared[0]["output"]
    assert prepared[0]["type"] == "function_call_output"
    assert output[0] == {"type": "input_text", "text": "caption"}
    assert output[1]["type"] == "input_image"
    assert output[1]["image_url"].startswith("data:image/png;base64,")
    assert all(message.get("role") != "user" for message in prepared)


def test_anthropic_keeps_native_tool_result_image_block() -> None:
    prepared = encode_messages([Message("tool", [_image_result("call_1")])])

    assert len(prepared) == 1
    tool_result = prepared[0]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"
    assert tool_result["content"][0] == {"type": "text", "text": "caption"}
    assert tool_result["content"][1]["type"] == "image"
    assert tool_result["content"][1]["source"]["type"] == "base64"
    assert tool_result["content"][1]["source"]["media_type"] == "image/png"
