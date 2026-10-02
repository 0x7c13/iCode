# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Token usage of responses the adapter fails, counted by the usage middleware in the tool loop.

The Chat Completions and Responses clients and the OpenAI SDK are real; only
the HTTP answers are scripted.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest
from openai.types import CompletionUsage
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

from chrys.foundation.errors import ProviderResponseError
from chrys.kernel import BaseChatClient, Message, ResponseStream
from chrys.kernel.loop import StallExhaustedAction
from chrys.kernel.middleware import ChatMiddlewareLayer
from chrys.orchestration.invoker.attempts import WireRetryPolicyAdapter
from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware
from chrys.service.context.middleware.usage import UsageTrackingMiddleware
from chrys.service.llm.chat_completions import ChatCompletionsClient
from chrys.service.profiles.models.options import STREAM_REQUIRES_FINISH_REASON_OPTION
from tests.service.llm._responses_wire import Script, blocking, call_item, responses_client
from tests.support.openai_chat_wire import ChatReply, scripted_openai
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer
from tests.support.wire_cases import Reply
from tests.support.wire_cases._kit import resp_message

# What every scripted response reports: total, input and output tokens.
_REPORTED = (79, 70, 9)
_FAILURE = {"code": "server_error", "message": "The model failed."}


def _completion(
    *, refusal: str | None = None, calls: int = 0, finish_reason: str, usage: bool = True
) -> ChatCompletion:
    tool_calls = [
        ChatCompletionMessageFunctionToolCall(
            id=f"call_{index}", type="function", function=Function(name="read_file", arguments='{"path": "a"}')
        )
        for index in range(calls)
    ]
    message = ChatCompletionMessage.model_construct(
        role="assistant", content="Partial", refusal=refusal, tool_calls=tool_calls or None
    )
    return ChatCompletion.model_construct(
        id="completion-1",
        object="chat.completion",
        created=1_717_171_717,
        model="test",
        choices=[Choice.model_construct(index=0, message=message, finish_reason=finish_reason)],
        usage=CompletionUsage(prompt_tokens=70, completion_tokens=9, total_tokens=79) if usage else None,
    )


def _chunk(
    *, finish_reason: str | None = None, usage: bool = False, output_tokens: int = 9, **delta: Any
) -> ChatCompletionChunk:
    """One streamed chunk; with no *delta* fields and no finish reason, it has no choice at all.

    With *usage*, the chunk reports the usage so far: 70 input tokens and *output_tokens*.
    """
    choices = []
    if delta or finish_reason:
        choices = [
            ChunkChoice.model_construct(
                index=0, delta=ChoiceDelta.model_construct(role="assistant", **delta), finish_reason=finish_reason
            )
        ]
    return ChatCompletionChunk.model_construct(
        id="chunk-1",
        object="chat.completion.chunk",
        created=1_717_171_717,
        model="test",
        choices=choices,
        usage=CompletionUsage(prompt_tokens=70, completion_tokens=output_tokens, total_tokens=70 + output_tokens)
        if usage
        else None,
    )


def _call(arguments: str = '{"path": "a"}') -> list[ChoiceDeltaToolCall]:
    function = ChoiceDeltaToolCallFunction.model_construct(name="read_file", arguments=arguments)
    return [ChoiceDeltaToolCall.model_construct(index=0, id="call_1", type="function", function=function)]


@asynccontextmanager
async def _chat_completions(*replies: ChatReply) -> AsyncIterator[BaseChatClient]:
    async with scripted_openai(replies) as wire:
        yield ChatCompletionsClient(model="test", sdk_client=wire.client)


@asynccontextmanager
async def _responses(*replies: Reply) -> AsyncIterator[BaseChatClient]:
    async with responses_client(*replies) as (client, _):
        yield client


def _policy(validation: ResponseValidationMiddleware) -> WireRetryPolicyAdapter:
    async def no_sleep(_seconds: int) -> bool:
        return False

    async def publish(_message: str, _attempt: int, _total: int, _delay: int, _error: BaseException) -> None:
        return None

    return WireRetryPolicyAdapter(
        max_retries=2,
        stall_timeout_seconds=None,
        stall_max_retries=0,
        stall_exhausted_action=StallExhaustedAction.BLOCKING_FALLBACK,
        backoff_schedule=(0,),
        interrupted=lambda: False,
        interruptible_sleep=no_sleep,
        publish_retry=publish,
        hosted_commits_in_flight=validation.hosted_commits_in_flight,
    )


async def _counted(
    client: BaseChatClient, *, stream: bool, wire_retry: bool = False, options: Mapping[str, Any] | None = None
) -> list[tuple[int, int, int]]:
    """Run one turn through the usage and validation middleware; the usage counted per call."""
    counted: list[tuple[int, int, int]] = []

    def on_usage(total: int, input_tokens: int, output_tokens: int, *_calibration: object) -> None:
        counted.append((total, input_tokens, output_tokens))

    validation = ResponseValidationMiddleware(backoff_schedule=(0,))
    layer = InvariantCheckedToolLoopLayer(
        ChatMiddlewareLayer(client, middleware=[UsageTrackingMiddleware(on_usage=on_usage), validation])
    )
    result = layer.get_response(
        [Message("user", ["hi"])],
        stream=stream,
        options={"store": False, **(options or {})},
        client_kwargs={"wire_retry_policy": _policy(validation)} if wire_retry else {},
    )
    with contextlib.suppress(ProviderResponseError):
        await (result.get_final_response() if isinstance(result, ResponseStream) else result)
    return counted


_REFUSED_CALL = (call_item("fc_1", "call_1"),)

_FAILED_WITH_USAGE: list[Any] = [
    pytest.param(lambda: _responses(Script().started().failed().reply()), True, id="responses_failed_stream"),
    pytest.param(lambda: _responses(blocking(status="failed", error=_FAILURE)), False, id="responses_failed"),
    pytest.param(
        lambda: _responses(
            Script().started().call(0, "fc_1", "call_1").finished(*_REFUSED_CALL, incomplete="content_filter").reply()
        ),
        True,
        id="responses_refused_stream",
    ),
    pytest.param(
        lambda: _responses(blocking(*_REFUSED_CALL, status="incomplete", incomplete="content_filter")),
        False,
        id="responses_refused",
    ),
    pytest.param(
        lambda: _chat_completions([_chunk(content="Partial", finish_reason="network_error", usage=True)]),
        True,
        id="chat_completions_failed_stream",
    ),
    pytest.param(
        lambda: _chat_completions([_chunk(content="Partial", usage=True), _chunk(finish_reason="network_error")]),
        True,
        id="chat_completions_failed_after_its_usage",
    ),
    pytest.param(
        lambda: _chat_completions(
            [
                _chunk(content="Partial", usage=True, output_tokens=1),
                _chunk(content=" answer", usage=True),
                _chunk(finish_reason="network_error"),
            ]
        ),
        True,
        id="chat_completions_failed_after_growing_usage",
    ),
    # The usage comes in a chunk of its own after the finish reason.
    pytest.param(
        lambda: _chat_completions(
            [_chunk(content="Partial"), _chunk(finish_reason="network_error"), _chunk(usage=True)]
        ),
        True,
        id="chat_completions_failed_then_its_usage",
    ),
    pytest.param(
        lambda: _chat_completions([_chunk(finish_reason="model_context_window_exceeded"), _chunk(usage=True)]),
        True,
        id="chat_completions_context_overflow_then_its_usage",
    ),
    pytest.param(
        lambda: _chat_completions(
            [_chunk(refusal="I can't.", tool_calls=_call()), _chunk(finish_reason="network_error"), _chunk(usage=True)]
        ),
        True,
        id="chat_completions_refused_then_failed_then_its_usage",
    ),
    pytest.param(
        lambda: _chat_completions(
            [_chunk(refusal="I can't.", tool_calls=_call()), _chunk(finish_reason="tool_calls"), _chunk(usage=True)]
        ),
        True,
        id="chat_completions_refused_then_its_usage",
    ),
    pytest.param(
        lambda: _chat_completions(
            [_chunk(tool_calls=_call()), _chunk(finish_reason="content_filter"), _chunk(usage=True)]
        ),
        True,
        id="chat_completions_filtered_then_its_usage",
    ),
    # Streams without a finish reason: the usage came in a chunk of its own.
    pytest.param(
        lambda: _chat_completions([_chunk(refusal="I can't.", tool_calls=_call()), _chunk(usage=True)]),
        True,
        id="chat_completions_refused_stream",
    ),
    pytest.param(
        lambda: _chat_completions([_chunk(tool_calls=_call('{"path": ')), _chunk(usage=True)]),
        True,
        id="chat_completions_call_cut_off",
    ),
    pytest.param(
        lambda: _chat_completions(_completion(finish_reason="network_error")), False, id="chat_completions_failed"
    ),
    pytest.param(
        lambda: _chat_completions(_completion(refusal="I can't.", calls=1, finish_reason="tool_calls")),
        False,
        id="chat_completions_refused",
    ),
]


@pytest.mark.parametrize(("client", "stream"), _FAILED_WITH_USAGE)
async def test_a_failed_response_counts_the_usage_it_reported(client: Callable[[], Any], stream: bool) -> None:
    async with client() as chat_client:
        assert await _counted(chat_client, stream=stream) == [_REPORTED]


@pytest.mark.parametrize(
    ("client", "stream"),
    [
        pytest.param(lambda: _responses(Script().started().error("server_error").reply()), True, id="error_event"),
        pytest.param(lambda: _responses(Script().started().reply()), True, id="cut_off"),
        pytest.param(
            lambda: _chat_completions(_completion(finish_reason="network_error", usage=False)),
            False,
            id="no_usage_reported",
        ),
        pytest.param(
            lambda: _chat_completions([_chunk(content="Partial"), _chunk(finish_reason="network_error")]),
            True,
            id="no_usage_streamed",
        ),
    ],
)
async def test_a_failed_response_without_usage_counts_nothing(client: Callable[[], Any], stream: bool) -> None:
    async with client() as chat_client:
        assert await _counted(chat_client, stream=stream) == []


@pytest.mark.parametrize("stream", [True, False], ids=["streaming", "blocking"])
async def test_each_attempt_of_a_wire_retry_counts_its_usage(stream: bool) -> None:
    if stream:
        failed = Script().started().failed().reply()
        recovered = Script().started().text(0, "msg_1", "Sunny.").finished(resp_message("msg_1", "Sunny.")).reply()
    else:
        failed = blocking(status="failed", error=_FAILURE)
        recovered = blocking(resp_message("msg_1", "Sunny."))

    async with _responses(failed, recovered) as client:
        assert await _counted(client, stream=stream, wire_retry=True) == [_REPORTED, _REPORTED]


@pytest.mark.parametrize(
    "failed",
    [
        pytest.param([_chunk(content="Partial"), _chunk(usage=True)], id="cut_off_after_its_usage"),
        pytest.param(
            [_chunk(content="Partial", usage=True), _chunk(finish_reason="network_error")], id="failed_after_its_usage"
        ),
        pytest.param(
            [_chunk(content="Partial"), _chunk(finish_reason="network_error"), _chunk(usage=True)],
            id="failed_then_its_usage",
        ),
    ],
)
async def test_each_attempt_of_a_chat_completions_retry_counts_its_usage(failed: list[ChatCompletionChunk]) -> None:
    recovered = [_chunk(content="Sunny.", finish_reason="stop"), _chunk(usage=True)]

    async with _chat_completions(failed, recovered) as client:
        counted = await _counted(
            client, stream=True, wire_retry=True, options={STREAM_REQUIRES_FINISH_REASON_OPTION: True}
        )

    assert counted == [_REPORTED, _REPORTED]
