# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""How each wire protocol sends images, in user messages and tool results."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx
import pytest

from chrys.foundation.events.types import InvocationMessage, UserMessage
from chrys.kernel import Content, Message
from chrys.service.llm.anthropic_messages.history import encode_messages
from chrys.service.llm.chat_completions import history as chat_history
from chrys.service.llm.chat_completions.client import DEEPSEEK, OPENAI
from chrys.service.llm.images import UNSUPPORTED_IMAGE_TEXT
from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.replay import encode_input
from tests.support.images import image_bytes
from tests.support.mock_provider_turns import mock_provider_profile, run_mock_provider_turn


def _image_result(call_id: str, text: str = "caption") -> Content:
    return Content.from_function_result(
        call_id,
        result=[
            Content.from_text(text),
            Content.from_data(image_bytes("PNG"), "image/png"),
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


_ENCODERS: dict[str, Callable[[list[Message]], list[dict[str, Any]]]] = {
    "chat_completions": lambda messages: chat_history.encode_messages(messages, variant=OPENAI),
    "responses": lambda messages: encode_input(messages, service_side=False, variant=OPENAI_RESPONSES),
    "anthropic": encode_messages,
}


@pytest.mark.parametrize("protocol", list(_ENCODERS))
@pytest.mark.parametrize("media_type", [None, "application/pdf", "audio/wav"])
def test_a_link_older_sessions_kept_is_left_out(protocol: str, media_type: str | None) -> None:
    """Older sessions kept MCP links as their URL, some with no media type; they still encode."""
    link = Content.from_uri("https://example.com/report", media_type=media_type)
    message = Message("tool", [Content.from_function_result("call_1", result=[Content.from_text("report"), link])])

    wire = json.dumps(_ENCODERS[protocol]([message]))

    assert "report" in wire
    assert "https://example.com/report" not in wire


def _sent_images(wire: Any) -> list[str]:
    """Each image part of *wire*: a data image as its media type, a URL image as its URL."""
    if isinstance(wire, list):
        return [image for item in wire for image in _sent_images(item)]
    if not isinstance(wire, dict):
        return []
    match wire.get("type"):
        case "image_url":
            uri = wire["image_url"]["url"]
        case "input_image":
            uri = wire["image_url"]
        case "image":
            source = wire["source"]
            return [source["media_type"] if source["type"] == "base64" else source["url"]]
        case _:
            return [image for value in wire.values() for image in _sent_images(value)]
    return [uri.split(";", 1)[0].removeprefix("data:") if uri.startswith("data:") else uri]


@pytest.mark.parametrize("protocol", list(_ENCODERS))
@pytest.mark.parametrize("in_tool_result", [False, True], ids=["user", "tool_result"])
@pytest.mark.parametrize(
    ("make_image", "sent"),
    [
        pytest.param(lambda: Content.from_data(image_bytes("PNG"), "image/png"), "image/png", id="png"),
        pytest.param(lambda: Content.from_data(image_bytes("GIF"), "image/gif"), "image/gif", id="gif"),
        pytest.param(lambda: Content.from_data(image_bytes("WEBP"), "image/webp"), "image/webp", id="webp"),
        pytest.param(lambda: Content.from_data(image_bytes("JPEG"), "image/png"), "image/jpeg", id="bytes-name-type"),
        pytest.param(lambda: Content.from_data(image_bytes("BMP"), "image/png"), None, id="unsupported-bytes"),
        pytest.param(
            lambda: Content.from_uri("https://example.com/a.jpg", media_type="image/jpg"),
            "https://example.com/a.jpg",
            id="url-type-alias",
        ),
        pytest.param(
            lambda: Content.from_uri("https://example.com/a.bmp", media_type="image/bmp"), None, id="url-unsupported"
        ),
        pytest.param(
            lambda: Content.from_uri("file:///tmp/shot.png", media_type="image/png"), None, id="url-not-fetchable"
        ),
    ],
)
def test_images_go_out_as_a_format_the_api_reads_or_as_a_placeholder(
    protocol: str, in_tool_result: bool, make_image: Callable[[], Content], sent: str | None
) -> None:
    image = make_image()
    stored = (image.uri, image.media_type)
    message = (
        Message("tool", [Content.from_function_result("call_1", result=[Content.from_text("caption"), image])])
        if in_tool_result
        else Message("user", ["look", image])
    )

    wire = _ENCODERS[protocol]([message])

    if sent is None:
        assert _sent_images(wire) == []
        assert UNSUPPORTED_IMAGE_TEXT in json.dumps(wire)
        assert "see following user message" not in json.dumps(wire)
    else:
        assert _sent_images(wire) == [sent]
        assert UNSUPPORTED_IMAGE_TEXT not in json.dumps(wire)
    # History keeps the image as it is.
    assert (image.uri, image.media_type) == stored


def _answer_ok(_request: httpx.Request) -> httpx.Response:
    events = [
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
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    body = b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=httpx.ByteStream(body))


async def test_an_image_the_api_cannot_read_goes_out_as_a_placeholder(
    agent_engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BMP a message holds as a PNG would fail every request of the session; it goes out as text instead."""
    image = Content.from_data(image_bytes("BMP"), "image/png")

    turn = await run_mock_provider_turn(
        agent_engine,
        monkeypatch,
        replace(mock_provider_profile("anthropic", stream=True), vision=True),
        _answer_ok,
        user_message=UserMessage(text="look", prepared_contents=["look", image]),
    )

    assert isinstance(turn.terminal, InvocationMessage)
    (request,) = turn.requests
    blocks = json.loads(request.content)["messages"][0]["content"]
    assert {"type": "text", "text": UNSUPPORTED_IMAGE_TEXT} in blocks
    assert all(block["type"] != "image" for block in blocks)
