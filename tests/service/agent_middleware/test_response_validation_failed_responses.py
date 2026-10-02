# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Responses the provider filtered or failed: the terminal verdict, and the hosted work they showed."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

import pytest

from chrys.foundation.errors import ErrorKind, ProviderResponseError, classify_error, invalidates_continuation_token
from chrys.foundation.hosted_tools import HeldHostedEvidence, HostedToolFamily, HostedToolPhase
from chrys.foundation.retry import RetryAttemptInfo
from chrys.foundation.trajectory.context import TRAJECTORY_EXCHANGE_KWARG, ExchangeTrace
from chrys.foundation.trajectory.event_types import EventType, ValidationReason
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.kernel import ChatResponse, ChatResponseUpdate, Content, Message, ResponseStream
from chrys.kernel.middleware import ChatContext
from chrys.service.agent_middleware.response_validation import (
    ResponseValidationMiddleware,
    TerminalResponseValidationError,
)
from chrys.service.agent_middleware.validators import CONTENT_FILTERED_REASON
from tests.service.agent_middleware._response_validation_fakes import (
    _assistant,
    _assistant_truncated,
    _FakeCallNext,
    _make_context,
    _ObservationHook,
    _service_context,
)
from tests.service.trajectory._fakes import FakeSink, make_context


def _exchange(sink: FakeSink) -> ExchangeTrace:
    return ExchangeTrace(make_context(sink).with_cycle(new_analytics_id()).with_exchange(new_analytics_id()))


def _context(*, stream: bool, service_side: bool, exchange: ExchangeTrace) -> ChatContext:
    if service_side:
        return _service_context(stream, exchange)
    return ChatContext(
        client=None,
        messages=[Message("user", ["hi"])],
        options=None,
        stream=stream,
        kwargs={"client_kwargs": {TRAJECTORY_EXCHANGE_KWARG: exchange}},
    )


async def _run(
    middleware: ResponseValidationMiddleware, context: ChatContext, call_next: Callable[[], Awaitable[None]]
) -> ChatResponse:
    """One pass through *middleware*; a streamed result is drained to its final response."""
    await middleware.process(context, call_next)
    if isinstance(context.result, ResponseStream):
        return await context.result.get_final_response()
    assert isinstance(context.result, ChatResponse)
    return context.result


def _evidence(family: HostedToolFamily) -> Content:
    """Finished hosted work that gathers evidence for an answer but is none itself."""
    return Content.from_hosted_tool_result(
        "evidence_1",
        tool_name="provider_tool",
        hosted_family=family,
        status="completed",
        provider_phase=HostedToolPhase.TERMINAL,
        provider_status="completed",
        result={"evidence": "found"},
    )


def _filtered(contents: list[Content]) -> ChatResponse:
    return ChatResponse(messages=[Message(role="assistant", contents=contents)], finish_reason="content_filter")


_EVIDENCE = [
    pytest.param([], id="nothing"),
    pytest.param([_evidence(HostedToolFamily.SEARCH)], id="search_evidence"),
    pytest.param([_evidence(HostedToolFamily.FETCH)], id="fetch_evidence"),
    pytest.param([_evidence(HostedToolFamily.TOOL_DISCOVERY)], id="tool_discovery_evidence"),
]


@pytest.mark.parametrize("contents", _EVIDENCE)
@pytest.mark.parametrize("service_side", [False, True], ids=["local_storage", "service_storage"])
@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_a_filtered_response_without_an_answer_fails_as_filtered(
    stream: bool, service_side: bool, contents: list[Content]
) -> None:
    sink = FakeSink()
    filtered = _filtered(contents)
    fake = _FakeCallNext([filtered, _assistant([Content.from_text("must stay unused")])], stream=stream)
    context = _context(stream=stream, service_side=service_side, exchange=_exchange(sink))
    fake.bind(context)
    retries: list[RetryAttemptInfo] = []

    async def _on_retry(info: RetryAttemptInfo) -> None:
        retries.append(info)

    middleware = ResponseValidationMiddleware(publish_retry=_on_retry, backoff_schedule=[0.0])
    with pytest.raises(TerminalResponseValidationError, match="content filter") as raised:
        await _run(middleware, context, fake)

    assert fake.call_count == 1
    assert retries == []
    cause = raised.value.__cause__
    assert isinstance(cause, ProviderResponseError)
    assert (cause.code, cause.retryable, cause.invalidates_continuation_token) == ("content_filter", False, True)
    classification = classify_error(raised.value)
    assert (classification.kind, classification.retryable) == (ErrorKind.CONTENT_FILTERED, False)
    # The filtered response is finished: a retry must not poll it.
    assert invalidates_continuation_token(raised.value) is True
    [finished] = sink.of_type(EventType.MODEL_VALIDATION_FINISHED)
    assert finished.payload["reason_code"] == ValidationReason.CONTENT_FILTERED
    assert finished.payload["gave_up"] is True
    assert str(raised.value) == CONTENT_FILTERED_REASON


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param([Content.from_text("Part of the answer")], id="visible_text"),
        pytest.param(
            [_evidence(HostedToolFamily.SEARCH), Content.from_text("The answer.")], id="answer_after_evidence"
        ),
        pytest.param(
            [
                Content.from_image_generation_tool_result(
                    image_id="image_1",
                    outputs=["data:image/png;base64,AA=="],
                    provider_phase=HostedToolPhase.TERMINAL,
                    provider_status="completed",
                )
            ],
            id="image",
        ),
    ],
)
@pytest.mark.parametrize("service_side", [False, True], ids=["local_storage", "service_storage"])
@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_an_answer_the_filter_cut_short_is_kept(
    stream: bool, service_side: bool, contents: list[Content]
) -> None:
    fake = _FakeCallNext([_filtered(contents)], stream=stream)
    context = _context(stream=stream, service_side=service_side, exchange=_exchange(FakeSink()))
    fake.bind(context)

    response = await _run(ResponseValidationMiddleware(backoff_schedule=[0.0]), context, fake)

    assert fake.call_count == 1
    assert response.finish_reason == "content_filter"
    assert [content.type for message in response.messages for content in message.contents] == [
        content.type for content in contents
    ]


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_other_terminal_verdicts_keep_their_own_classification(stream: bool) -> None:
    fake = _FakeCallNext([_assistant_truncated([])], stream=stream)
    context = _make_context(stream=stream)
    fake.bind(context)

    with pytest.raises(TerminalResponseValidationError, match="output token limit") as raised:
        await _run(ResponseValidationMiddleware(backoff_schedule=[0.0]), context, fake)

    assert raised.value.__cause__ is None
    assert classify_error(raised.value).kind is not ErrorKind.CONTENT_FILTERED
    assert invalidates_continuation_token(raised.value) is False


def _hosted_work() -> tuple[Content, Content]:
    return (
        Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github"),
        Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("created #42")]),
    )


def _failing_call_next(
    context: ChatContext, error: ProviderResponseError, *, stream: bool
) -> Callable[[], Awaitable[None]]:
    """A provider call that fails with *error*: at once, or from its stream after one text update."""

    async def _updates() -> AsyncIterator[ChatResponseUpdate]:
        yield ChatResponseUpdate(contents=[Content.from_text("partial")], role="assistant")
        raise error

    async def _call_next() -> None:
        if not stream:
            raise error
        context.result = ResponseStream(_updates(), finalizer=ChatResponse.from_updates)

    return _call_next


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_hosted_work_a_failed_response_showed_counts_as_executed(stream: bool) -> None:
    error = ProviderResponseError(
        "content_filter",
        "filtered",
        retryable=False,
        invalidates_continuation_token=True,
        observed_contents=_hosted_work(),
    )
    context = _make_context(stream=stream)
    hook = _ObservationHook()
    middleware = ResponseValidationMiddleware(observation_hook=hook, backoff_schedule=[0.0])

    with pytest.raises(ProviderResponseError) as raised:
        await _run(middleware, context, _failing_call_next(context, error, stream=stream))

    assert raised.value is error
    assert middleware.hosted_commits_in_flight() == middleware.hosted_commits_observed()
    assert len(middleware.hosted_commits_observed()) == 1
    assert ("contents", True, ("mcp_server_tool_call", "mcp_server_tool_result")) in hook.events


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_a_failed_response_without_hosted_work_commits_nothing(stream: bool) -> None:
    error = ProviderResponseError("network_error", "dropped", retryable=True)
    context = _make_context(stream=stream)
    hook = _ObservationHook()
    middleware = ResponseValidationMiddleware(observation_hook=hook, backoff_schedule=[0.0])

    with pytest.raises(ProviderResponseError):
        await _run(middleware, context, _failing_call_next(context, error, stream=stream))

    assert middleware.hosted_commits_observed() == ()
    assert [event for event in hook.events if event[0] == "contents" and event[1] is True] == []


@pytest.mark.parametrize("released", [False, True], ids=["dropped_while_held", "released"])
async def test_held_hosted_work_counts_at_once_and_is_shown_only_when_released(released: bool) -> None:
    hosted = _hosted_work()
    context = _make_context(stream=True)

    async def _updates() -> AsyncIterator[ChatResponseUpdate]:
        yield ChatResponseUpdate(contents=[], role="assistant", raw_representation=HeldHostedEvidence(hosted))
        if not released:
            raise ProviderResponseError("network_error", "dropped", retryable=True)
        call = Content.from_function_call(call_id="call_1", name="lookup", arguments="{}")
        yield ChatResponseUpdate(contents=[call, *hosted, Content.from_text("Filed #42.")], role="assistant")

    async def _call_next() -> None:
        context.result = ResponseStream(_updates(), finalizer=ChatResponse.from_updates)

    hook = _ObservationHook()
    middleware = ResponseValidationMiddleware(observation_hook=hook, backoff_schedule=[0.0])
    if released:
        response = await _run(middleware, context, _call_next)
        # In the order the stream released it, behind the call it waited for.
        assert [content.type for message in response.messages for content in message.contents] == [
            "function_call",
            "mcp_server_tool_call",
            "mcp_server_tool_result",
            "text",
        ]
    else:
        with pytest.raises(ProviderResponseError, match="dropped"):
            await _run(middleware, context, _call_next)

    assert len(middleware.hosted_commits_observed()) == 1
    # Shown once as the stream releases it and once in the final response; never while held.
    shown = [event[1] for event in hook.events if event[0] == "contents" and "mcp_server_tool_call" in event[2]]
    assert shown == ([False, True] if released else [])
