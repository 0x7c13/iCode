# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A request past the end of a ``scripted_openai`` script fails the test.

The SDK wraps the transport's ``AssertionError`` into a connection error the
client reports as ``ChatClientException``, which a test expecting a client
error would otherwise accept.
"""

from __future__ import annotations

import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice, ChoiceDelta

from chrys.kernel import ChatResponse, Message, ResponseStream
from chrys.kernel.exceptions import ChatClientException
from chrys.service.llm.chat_completions import ChatCompletionsClient
from tests.support.openai_chat_wire import ScriptedOpenAI, scripted_openai


def _reply(text: str) -> list[ChatCompletionChunk]:
    delta = ChoiceDelta.model_construct(role="assistant", content=text)
    return [
        ChatCompletionChunk(
            id="chunk-1",
            object="chat.completion.chunk",
            created=1,
            model="glm-5.2",
            choices=[Choice(index=0, delta=delta, finish_reason=None)],
        )
    ]


async def _ask(wire: ScriptedOpenAI, prompt: str) -> ChatResponse:
    client = ChatCompletionsClient(model="glm-5.2", sdk_client=wire.client)
    stream = client._inner_get_response(messages=[Message("user", [prompt])], options={}, stream=True)
    assert isinstance(stream, ResponseStream)
    return await stream.get_final_response()


@pytest.mark.asyncio
async def test_scripted_requests_are_answered_in_order() -> None:
    async with scripted_openai([_reply("one"), _reply("two")]) as wire:
        assert [(await _ask(wire, "a")).text, (await _ask(wire, "b")).text] == ["one", "two"]

    assert [request["messages"][0]["content"] for request in wire.requests] == ["a", "b"]
    assert [stream.closed for stream in wire.streams] == [True, True]


@pytest.mark.asyncio
async def test_an_overrun_caught_inside_the_block_fails_on_exit() -> None:
    with pytest.raises(AssertionError, match=r"unscripted request\(s\) \[2\]"):
        async with scripted_openai([_reply("one")]) as wire:
            await _ask(wire, "a")
            with pytest.raises(ChatClientException):
                await _ask(wire, "b")


@pytest.mark.asyncio
async def test_an_overrun_escaping_the_block_is_not_taken_for_a_client_error() -> None:
    with pytest.raises(AssertionError, match=r"unscripted request\(s\) \[1\]") as caught:
        async with scripted_openai([]) as wire:
            await _ask(wire, "a")

    assert isinstance(caught.value.__context__, ChatClientException)
