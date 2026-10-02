# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chat Completions finish reasons: spellings, failure reasons, refused calls and streams that end without one."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall, Function

from chrys.foundation.errors import (
    ErrorKind,
    ProviderResponseError,
    classify_error,
    invalidates_continuation_token,
    is_context_overflow,
)
from chrys.kernel import ChatResponse, Content, Message, ResponseStream, tool
from chrys.kernel.middleware import ChatMiddlewareLayer
from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware
from chrys.service.llm.chat_completions import ChatCompletionsClient
from chrys.service.profiles.models.options import STREAM_REQUIRES_FINISH_REASON_OPTION
from tests.support.openai_chat_wire import ChatReply, scripted_openai
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer

_STREAM_LOGGER = "chrys.service.llm.chat_completions.stream"
_NO_FINISH_WARNING = "ended without a finish reason"


def _chunk(delta: ChoiceDelta | None, *, finish_reason: str | None = None) -> ChatCompletionChunk:
    choice = ChunkChoice.model_construct(index=0, delta=delta, finish_reason=finish_reason)
    return ChatCompletionChunk.model_construct(
        id="chunk-1", object="chat.completion.chunk", created=1_717_171_717, model="test", choices=[choice], usage=None
    )


def _text(text: str, *, finish_reason: str | None = None) -> ChatCompletionChunk:
    return _chunk(ChoiceDelta.model_construct(role="assistant", content=text), finish_reason=finish_reason)


def _refusal(text: str) -> ChatCompletionChunk:
    return _chunk(ChoiceDelta.model_construct(role="assistant", refusal=text))


def _fragment(arguments: str, *, index: int = 0, name: str | None = "read_file") -> ChoiceDeltaToolCall:
    return ChoiceDeltaToolCall.model_construct(
        index=index,
        id=f"call_{index}",
        type="function",
        function=ChoiceDeltaToolCallFunction.model_construct(name=name, arguments=arguments),
    )


def _call(arguments: str, *, index: int = 0, finish_reason: str | None = None) -> ChatCompletionChunk:
    fragment = _fragment(arguments, index=index)
    return _chunk(ChoiceDelta.model_construct(role="assistant", tool_calls=[fragment]), finish_reason=finish_reason)


def _completion(
    *, content: str | None = None, refusal: str | None = None, calls: int = 0, finish_reason: str | None = "stop"
) -> ChatCompletion:
    tool_calls = [
        ChatCompletionMessageFunctionToolCall(
            id=f"call_{index}", type="function", function=Function(name="read_file", arguments='{"path": "a"}')
        )
        for index in range(calls)
    ]
    message = ChatCompletionMessage.model_construct(
        role="assistant", content=content, refusal=refusal, tool_calls=tool_calls or None
    )
    choice = Choice.model_construct(index=0, message=message, finish_reason=finish_reason)
    return ChatCompletion.model_construct(
        id="completion-1", object="chat.completion", created=1_717_171_717, model="test", choices=[choice], usage=None
    )


async def _respond(
    reply: ChatReply, *, options: dict[str, Any] | None = None, done: bool = True
) -> tuple[ChatResponse, list[dict[str, Any]]]:
    """The response *reply* decodes to, and the request bodies sent for it."""
    stream = not isinstance(reply, ChatCompletion)
    async with scripted_openai([reply], done=done) as wire:
        client = ChatCompletionsClient(model="test", sdk_client=wire.client)
        result = client._inner_get_response(
            messages=[Message("user", ["hi"])], options=dict(options or {}), stream=stream
        )
        if isinstance(result, ResponseStream):
            _ = [update async for update in result]
            return await result.get_final_response(), wire.requests
        return await result, wire.requests


def _calls(response: ChatResponse) -> list[Content]:
    return [content for message in response.messages for content in message.contents if content.type == "function_call"]


# ---------------------------------------------------------------------------
# Spellings and failure reasons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
@pytest.mark.parametrize(
    ("sent", "read"),
    [("stop", "stop"), ("end", "stop"), ("sensitive", "content_filter"), ("vendor_reason", "vendor_reason")],
)
async def test_finish_reasons_are_read_in_the_kernels_spelling(stream: bool, sent: str, read: str) -> None:
    reply: ChatReply = [_text("Hi", finish_reason=sent)] if stream else _completion(content="Hi", finish_reason=sent)

    response, _ = await _respond(reply)

    assert response.finish_reason == read
    assert response.text == "Hi"


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
@pytest.mark.parametrize(
    ("reason", "kind"),
    [("network_error", ErrorKind.STREAM_TRUNCATED), ("insufficient_system_resource", ErrorKind.OVERLOADED)],
)
async def test_a_failed_completion_raises_a_retryable_provider_error(
    stream: bool, reason: str, kind: ErrorKind
) -> None:
    reply: ChatReply = (
        [_text("Partial", finish_reason=reason)] if stream else _completion(content="Partial", finish_reason=reason)
    )

    with pytest.raises(ProviderResponseError) as raised:
        await _respond(reply)

    assert raised.value.code == reason
    assert (classify_error(raised.value).kind, classify_error(raised.value).retryable) == (kind, True)
    assert invalidates_continuation_token(raised.value) is False


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_a_context_overflow_finish_reason_fails_without_retry(stream: bool) -> None:
    reason = "model_context_window_exceeded"
    reply: ChatReply = [_text("", finish_reason=reason)] if stream else _completion(finish_reason=reason)

    with pytest.raises(ProviderResponseError) as raised:
        await _respond(reply)

    assert is_context_overflow(raised.value)
    assert classify_error(raised.value).retryable is False
    assert invalidates_continuation_token(raised.value) is True


# ---------------------------------------------------------------------------
# Refused or filtered responses never run their calls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param([_call('{"path": "a"}', finish_reason="content_filter")], id="filtered_in_the_calls_chunk"),
        pytest.param([_call('{"path": "a"}', finish_reason="sensitive")], id="sensitive_in_the_calls_chunk"),
        pytest.param([_call('{"path": "a"}'), _text("", finish_reason="content_filter")], id="filtered_after_calls"),
        pytest.param([_refusal("I can't."), _call('{"path": "a"}', finish_reason="tool_calls")], id="refusal_first"),
        pytest.param(
            [_call('{"path": "a"}'), _refusal("I can't."), _text("", finish_reason="tool_calls")], id="refusal_after"
        ),
        pytest.param([_refusal("I can't."), _call('{"path": "a"}')], id="refusal_then_eof"),
    ],
)
async def test_a_refused_stream_runs_none_of_its_calls(chunks: Sequence[ChatCompletionChunk]) -> None:
    emitted: list[Content] = []
    async with scripted_openai([chunks]) as wire:
        client = ChatCompletionsClient(model="test", sdk_client=wire.client)
        stream = client._inner_get_response(messages=[Message("user", ["hi"])], options={}, stream=True)
        assert isinstance(stream, ResponseStream)
        with pytest.raises(ProviderResponseError) as raised:
            async for update in stream:
                emitted.extend(update.contents)

    assert [content for content in emitted if content.type == "function_call"] == []
    assert raised.value.code == "content_filter"
    assert classify_error(raised.value).kind is ErrorKind.CONTENT_FILTERED
    assert classify_error(raised.value).retryable is False
    assert invalidates_continuation_token(raised.value) is True


@pytest.mark.parametrize(
    "completion",
    [
        pytest.param(_completion(refusal="I can't.", calls=1, finish_reason="tool_calls"), id="refusal"),
        pytest.param(_completion(calls=1, finish_reason="content_filter"), id="filtered"),
        pytest.param(_completion(calls=1, finish_reason="sensitive"), id="sensitive"),
    ],
)
async def test_a_refused_completion_runs_none_of_its_calls(completion: ChatCompletion) -> None:
    with pytest.raises(ProviderResponseError) as raised:
        await _respond(completion)

    assert raised.value.code == "content_filter"
    assert invalidates_continuation_token(raised.value) is True


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_a_refusal_without_calls_is_an_ordinary_answer(stream: bool) -> None:
    reply: ChatReply = (
        [_refusal("I can't help with that."), _text("", finish_reason="stop")]
        if stream
        else _completion(refusal="I can't help with that.")
    )

    response, _ = await _respond(reply)

    assert response.text == "I can't help with that."


async def test_calls_of_an_unrefused_stream_still_run() -> None:
    response, _ = await _respond([_text("Reading."), _call('{"path": "a"}', finish_reason="tool_calls")])

    assert [call.arguments for call in _calls(response)] == ['{"path": "a"}']


def _on_choice(chunk: ChatCompletionChunk, index: int) -> ChatCompletionChunk:
    [choice] = chunk.choices
    choice.index = index
    return chunk


async def _tool_runs(chunks: Sequence[ChatCompletionChunk]) -> tuple[list[str], int, BaseException | None]:
    """Drive *chunks* through the tool loop: the tool runs, the requests sent and what the run raised."""
    runs: list[str] = []

    @tool
    def read_file(path: str) -> str:
        runs.append(path)
        return "contents"

    async with scripted_openai([chunks, [_text("done", finish_reason="stop")]]) as wire:
        client = ChatCompletionsClient(model="test", sdk_client=wire.client)
        layer = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(client, middleware=[ResponseValidationMiddleware(backoff_schedule=(0,))])
        )
        result = layer.get_response([Message("user", ["go"])], stream=True, options={"tools": [read_file]})
        assert isinstance(result, ResponseStream)
        try:
            await result.get_final_response()
        except ProviderResponseError as error:
            return runs, len(wire.requests), error
        return runs, len(wire.requests), None


@pytest.mark.parametrize(
    "late",
    [
        pytest.param(_refusal("I can't."), id="refusal"),
        pytest.param(_text("", finish_reason="content_filter"), id="filtered"),
        pytest.param(_on_choice(_refusal("I can't."), 1), id="refusal_on_another_choice"),
    ],
)
async def test_a_refusal_after_released_calls_still_runs_none_of_them(late: ChatCompletionChunk) -> None:
    runs, requests, raised = await _tool_runs([_call('{"path": "a"}', finish_reason="tool_calls"), late])

    assert (runs, requests) == ([], 1)
    assert isinstance(raised, ProviderResponseError)
    assert raised.code == "content_filter"


async def test_released_calls_run_when_no_refusal_follows() -> None:
    runs, requests, raised = await _tool_runs(
        [_call('{"path": "a"}', finish_reason="tool_calls"), _on_choice(_text("", finish_reason="stop"), 1)]
    )

    assert (runs, requests, raised) == (["a"], 2, None)


def _cut_off(*fragments: ChoiceDeltaToolCall, refusal: str | None = None) -> ChatCompletionChunk:
    """A chunk ending its choice at the length limit with *fragments*, and *refusal* when given."""
    delta = ChoiceDelta.model_construct(role="assistant", refusal=refusal, tool_calls=list(fragments))
    return _chunk(delta, finish_reason="length")


_NAMELESS = _fragment("{", name=None)


@pytest.mark.parametrize(
    "chunks",
    [
        pytest.param([_cut_off(_NAMELESS, refusal="I can't.")], id="same_chunk"),
        pytest.param([_refusal("I can't."), _cut_off(_NAMELESS)], id="refusal_first"),
        pytest.param([_cut_off(_NAMELESS), _refusal("I can't.")], id="refusal_after"),
    ],
)
async def test_a_refusal_beside_a_dropped_nameless_call_stays_an_answer(chunks: Sequence[ChatCompletionChunk]) -> None:
    response, _ = await _respond(chunks)

    assert response.text == "I can't."
    assert _calls(response) == []


async def test_a_named_call_beside_a_dropped_nameless_one_is_still_refused() -> None:
    named = _fragment('{"path": "a"}', index=1)
    runs, requests, raised = await _tool_runs([_cut_off(_NAMELESS, named, refusal="I can't.")])

    assert (runs, requests) == ([], 1)
    assert isinstance(raised, ProviderResponseError)
    assert raised.code == "content_filter"


# ---------------------------------------------------------------------------
# Streams that end without a finish reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("done", [True, False], ids=["done", "eof"])
@pytest.mark.parametrize("arguments", ["", "  ", "{}", '{"path": "a"}'])
async def test_calls_with_whole_arguments_are_released_at_the_end(done: bool, arguments: str) -> None:
    response, _ = await _respond([_call(arguments)], done=done)

    assert [call.call_id for call in _calls(response)] == ["call_0"]


@pytest.mark.parametrize("done", [True, False], ids=["done", "eof"])
@pytest.mark.parametrize("arguments", ['{"path": ', "[]", "null", '"text"'])
async def test_a_call_cut_off_at_the_end_fails_the_whole_response(done: bool, arguments: str) -> None:
    emitted: list[Content] = []
    chunks = [_call('{"path": "a"}', index=0), _call(arguments, index=1)]
    async with scripted_openai([chunks], done=done) as wire:
        client = ChatCompletionsClient(model="test", sdk_client=wire.client)
        stream = client._inner_get_response(messages=[Message("user", ["hi"])], options={}, stream=True)
        assert isinstance(stream, ResponseStream)
        with pytest.raises(ProviderResponseError) as raised:
            async for update in stream:
                emitted.extend(update.contents)

    # The whole call batch is withheld, the complete call included.
    assert [content for content in emitted if content.type == "function_call"] == []
    assert raised.value.code == "stream_truncated"
    assert classify_error(raised.value).retryable is True
    assert invalidates_continuation_token(raised.value) is False


_UNFINISHED_ENDS = [
    pytest.param([_text("Hi")], id="no_finish_reason"),
    pytest.param([_text("Hi", finish_reason="")], id="empty_finish_reason"),
    pytest.param([_text("Hi"), _chunk(None)], id="null_delta"),
]


@pytest.mark.parametrize("done", [True, False], ids=["done", "eof"])
@pytest.mark.parametrize("chunks", _UNFINISHED_ENDS)
async def test_an_unfinished_text_stream_is_kept_with_a_warning(
    chunks: Sequence[ChatCompletionChunk], done: bool, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=_STREAM_LOGGER):
        response, _ = await _respond(chunks, done=done)

    assert response.text == "Hi"
    assert response.finish_reason is None
    assert [record.getMessage() for record in caplog.records if _NO_FINISH_WARNING in record.getMessage()] == [
        "Chat Completions stream ended without a finish reason; the answer may be incomplete"
    ]


@pytest.mark.parametrize("done", [True, False], ids=["done", "eof"])
@pytest.mark.parametrize("chunks", _UNFINISHED_ENDS)
async def test_an_unfinished_text_stream_fails_when_the_profile_requires_a_finish_reason(
    chunks: Sequence[ChatCompletionChunk], done: bool
) -> None:
    with pytest.raises(ProviderResponseError) as raised:
        await _respond(chunks, options={STREAM_REQUIRES_FINISH_REASON_OPTION: True}, done=done)

    assert raised.value.code == "stream_truncated"
    assert classify_error(raised.value).retryable is True


async def test_a_finished_stream_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=_STREAM_LOGGER):
        response, _ = await _respond(
            [_text("Hi"), _text("", finish_reason="stop")], options={STREAM_REQUIRES_FINISH_REASON_OPTION: True}
        )

    assert response.text == "Hi"
    assert [record for record in caplog.records if _NO_FINISH_WARNING in record.getMessage()] == []


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_the_finish_reason_requirement_is_never_sent(stream: bool) -> None:
    reply: ChatReply = [_text("Hi", finish_reason="stop")] if stream else _completion(content="Hi")

    _, requests = await _respond(reply, options={STREAM_REQUIRES_FINISH_REASON_OPTION: True})

    assert [STREAM_REQUIRES_FINISH_REASON_OPTION in request for request in requests] == [False]
