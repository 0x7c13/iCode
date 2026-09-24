# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Refusal text survives response parsing and remains ordinary replay text."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.responses import ResponseRefusalDeltaEvent, ResponseRefusalDoneEvent

from chrys.kernel import ChatResponse, Message
from chrys.kernel.exceptions import ChatClientInvalidRequestException
from chrys.service.agent_middleware.validators import DefaultResponseValidator
from chrys.service.llm.openai_chat_completion import RawOpenAIChatCompletionClient
from chrys.service.llm.openai_responses import RawOpenAIChatClient


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("content", "refusal", "expected"),
    [
        (None, "I cannot continue.", "I cannot continue."),
        ("Partial answer. ", "I cannot continue.", "Partial answer. I cannot continue."),
        ([{"type": "thinking", "text": "private"}], "I cannot continue.", "I cannot continue."),
        ({"type": "text", "text": "unsupported"}, "I cannot continue.", "I cannot continue."),
        ([{"type": "thinking", "text": "private"}], None, ""),
        ("Answer.", {"text": "invalid refusal"}, "Answer."),
        (None, {"text": "invalid refusal"}, ""),
    ],
)
def test_chat_completion_captures_refusal_without_relaxing_content_contract(stream, content, refusal, expected) -> None:
    client = RawOpenAIChatCompletionClient(model="test", async_client=SimpleNamespace(base_url="https://api.test"))
    if stream:
        choice = ChunkChoice.model_construct(
            index=0,
            finish_reason=None,
            delta=ChoiceDelta.model_construct(content=content, refusal=refusal),
        )
        chunk = ChatCompletionChunk.model_construct(id="r1", created=1, model="test", choices=[choice], usage=None)
        update = client._parse_response_update_from_openai(chunk)
        response = ChatResponse.from_updates([update])
    else:
        choice = Choice.model_construct(
            index=0,
            finish_reason="stop",
            message=ChatCompletionMessage.model_construct(role="assistant", content=content, refusal=refusal),
        )
        raw = ChatCompletion.model_construct(id="r1", created=1, model="test", choices=[choice], usage=None)
        response = client._parse_response_from_openai(raw, {})
    assert "".join(content.text or "" for message in response.messages for content in message.contents) == expected
    if expected:
        assert DefaultResponseValidator().validate(response).ok
        wire = client._prepare_messages_for_openai(response.messages)
        assert all("refusal" not in item for item in wire)


def test_responses_refusal_deltas_assemble_once_with_message_provenance() -> None:
    client = RawOpenAIChatClient(model="test", async_client=SimpleNamespace(base_url="https://api.test"))
    events = [
        ResponseRefusalDeltaEvent(
            type="response.refusal.delta",
            item_id="msg1",
            output_index=0,
            content_index=0,
            delta=text,
            sequence_number=index,
        )
        for index, text in enumerate(["I cannot ", "continue."])
    ]
    events.append(
        ResponseRefusalDoneEvent(
            type="response.refusal.done",
            item_id="msg1",
            output_index=0,
            content_index=0,
            refusal="I cannot continue.",
            sequence_number=2,
        )
    )
    updates = [client._parse_chunk_from_openai(event, {}, {}) for event in events]
    response = ChatResponse.from_updates(updates)
    assert response.text == "I cannot continue."
    assert DefaultResponseValidator().validate(response).ok
    wire = client._prepare_messages_for_openai(response.messages, request_uses_service_side_storage=False)
    assert wire[0]["content"][0]["type"] == "output_text"
    assert wire[0]["content"][0]["text"] == "I cannot continue."


@pytest.mark.parametrize("options", [{"n": 2}, {"extra_body": {"n": 2}}, {"n": 1, "extra_body": {"n": 2}}])
def test_chat_completions_rejects_multiple_choices_at_request_boundary(options) -> None:
    client = RawOpenAIChatCompletionClient(model="test", async_client=SimpleNamespace(base_url="https://api.test"))
    with pytest.raises(ChatClientInvalidRequestException, match="only n=1"):
        client._prepare_options([Message("user", ["hi"])], options)


@pytest.mark.parametrize("options", [{}, {"n": 1}, {"extra_body": {"n": 1}}])
def test_chat_completions_accepts_single_choice(options) -> None:
    client = RawOpenAIChatCompletionClient(model="test", async_client=SimpleNamespace(base_url="https://api.test"))
    assert client._prepare_options([Message("user", ["hi"])], options)["messages"]
