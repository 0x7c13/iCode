# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Usage completeness across real provider calls, validation and retry lanes."""

from __future__ import annotations

import asyncio
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.kernel import Agent, ChatResponse, Content, ContextProvider, Message, ResponseStream, tool
from chrys.kernel.types import ChatResponseUpdate
from chrys.orchestration.invoker.contracts import Failed, Ok, RunIntent, RunRequest, UsageDelta
from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware
from chrys.service.llm.mock import MockChatClient, MockResponse
from tests.orchestration.sub_agents._controller_fixtures import _make_controller
from tests.support.scripted_clients import ErrorMockChatClient

_USAGE = {"input_token_count": 7, "output_token_count": 3, "total_token_count": 10}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("first", ["reported", "missing", "exception", "validation"])
async def test_pass_usage_observes_every_response(executor, monkeypatch, stream, store, first):
    @tool
    async def counted() -> str:
        return "done"

    initial = (
        ConnectionError("wire failed")
        if first == "exception"
        else MockResponse(text="", usage_details=_USAGE)
        if first == "validation"
        else MockResponse(tool_calls=[("counted", "c1", {})], usage_details=_USAGE if first == "reported" else None)
    )
    client = ErrorMockChatClient([initial, MockResponse(text="done", usage_details=_USAGE)])
    executor._stream = stream
    executor._chat_options = {"store": store}
    executor._max_retries_override = 1
    monkeypatch.setattr(executor, "_BACKOFF_SCHEDULE", (0,))
    executor._agent.client = client
    executor._agent.default_options["tools"] = [counted]
    executor._agent.middleware = [ResponseValidationMiddleware(backoff_schedule=[0])]

    outcome = await executor.backend.run(executor.inputs.fresh_request(["work"]))

    assert isinstance(outcome, Ok)
    assert client.call_count == 2
    reported = first in ("reported", "validation")
    assert outcome.usage.input_tokens == (14 if reported else 7)
    assert outcome.usage.output_tokens == (6 if reported else 3)
    assert outcome.usage.total_tokens == (20 if reported else 10)
    assert outcome.usage.complete is reported
    assert outcome.usage.unreported == (0 if reported else 1)

    # A new pass does not inherit old missing reports or old spend.
    client = MockChatClient(responses=[MockResponse(text="next", usage_details=_USAGE)])
    executor._agent.client = client
    executor.inputs.begin_invocation()
    next_outcome = await executor.backend.run(executor.inputs.fresh_request(["next"]))
    assert next_outcome.usage.total_tokens == 10
    assert next_outcome.usage.complete is True
    assert next_outcome.usage.unreported == 0


@pytest.mark.parametrize("stream", [False, True])
async def test_child_backend_observes_missing_usage(stream):
    @tool
    async def counted() -> str:
        return "done"

    client = MockChatClient(
        responses=[MockResponse(tool_calls=[("counted", "c1", {})]), MockResponse(text="done", usage_details=_USAGE)]
    )
    agent = Agent(client=client, tools=[counted])
    controller = _make_controller(agent, EventBus(), stream=stream)
    try:
        for index in range(2):
            if index:
                agent.client = MockChatClient(responses=[MockResponse(text="next", usage_details=_USAGE)])
            outcome = await controller.policy.backend.run(
                RunRequest([Message("user", ["work"])], RunIntent.FRESH, controller.origin)
            )
            assert isinstance(outcome, Ok)
            assert outcome.usage.total_tokens == 10
            assert outcome.usage.unreported == (0 if index else 1)
            assert outcome.usage.complete is bool(index)
        assert "middleware" not in controller.policy._run_kwargs
    finally:
        await controller.policy.backend.owner.aclose()


@pytest.mark.parametrize("store", [False, True])
async def test_stalled_stream_usage_survives_blocking_fallback(executor, monkeypatch, store):
    client = MockChatClient(responses=[MockResponse(text="done", usage_details=_USAGE)])
    original = client._inner_get_response
    calls = []

    def response(*, messages, stream, options, **kwargs):
        calls.append(stream)
        if stream:

            async def updates():
                yield ChatResponseUpdate(contents=[Content.from_usage(_USAGE)])
                await asyncio.Event().wait()

            return ResponseStream(updates(), finalizer=ChatResponse.from_updates)
        return original(messages=messages, stream=stream, options=options, **kwargs)

    monkeypatch.setattr(client, "_inner_get_response", create_autospec(original, side_effect=response))
    executor._agent.client = client
    executor._stream = True
    executor._chat_options = {"store": store}
    executor._max_retries_override = 0
    executor._stream_attempt_timeout = 0.02
    monkeypatch.setattr(executor, "_BACKOFF_SCHEDULE", (0,))

    outcome = await executor.backend.run(executor.inputs.fresh_request(["work"]))

    assert isinstance(outcome, Ok)
    assert calls == [True, False]
    assert outcome.usage.total_tokens == 20
    assert outcome.usage.complete is False
    assert outcome.usage.unreported == 1


@pytest.mark.parametrize("stream", [False, True])
async def test_child_run_validation_counts_rejected_usage(stream):
    client = MockChatClient(
        responses=[MockResponse(text="", usage_details=_USAGE), MockResponse(text="valid", usage_details=_USAGE)]
    )
    middleware = [ResponseValidationMiddleware(backoff_schedule=[0])]
    controller = _make_controller(
        Agent(client=client), EventBus(), stream=stream, run_kwargs={"middleware": middleware}
    )
    try:
        result = await controller.policy.backend.run(
            RunRequest([Message("user", ["work"])], RunIntent.FRESH, controller.origin)
        )
        assert isinstance(result, Ok)
        assert client.call_count == 2
        assert result.usage == UsageDelta(14, 6, 20, True, 0)
        assert controller.policy._run_kwargs["middleware"] is middleware
    finally:
        await controller.policy.backend.owner.aclose()


class _AfterRunFailure(ContextProvider):
    """Fail after provider usage arrives, before Agent.run returns its response."""

    def __init__(self):
        super().__init__("fail-after-run")

    async def after_run(self, *, agent, session, context, state):
        assert context.response is not None
        raise ValueError("after-run failure")


@pytest.mark.parametrize("stream", [False, True])
async def test_after_run_failure_usage_is_incomplete(executor, stream):
    executor._stream = stream
    executor._agent.context_providers.append(_AfterRunFailure())
    executor._agent.client = MockChatClient(responses=[MockResponse(text="done", usage_details=_USAGE)])
    result = await executor.backend.run(executor.inputs.fresh_request(["work"]))
    assert isinstance(result, Failed)
    assert result.usage == UsageDelta(7, 3, 10, False, 0)


@pytest.mark.parametrize("validation", [False, True])
async def test_stream_preserves_raw_finalizer_hooks_and_cache(executor, monkeypatch, validation):
    client = MockChatClient(
        responses=([MockResponse(text="", usage_details=_USAGE)] if validation else [])
        + [MockResponse(text="valid", usage_details=_USAGE)]
    )
    original = client._inner_get_response
    log, raw, finals = [], [], []

    def get_response(*, messages, stream, options, **kwargs):
        result = original(messages=messages, stream=stream, options=options, **kwargs)
        assert isinstance(result, ResponseStream)
        ordinal = len(raw)

        async def finalize(updates):
            log.append(("finalize", ordinal))
            response = ChatResponse.from_updates(updates)
            # Provider totals are authoritative even when input + output differs.
            response.usage_details = {**_USAGE, "total_token_count": 99}
            finals.append(response)
            return response

        def hook(response):
            log.append(("hook", ordinal))
            assert response is finals[ordinal]

        wrapped = ResponseStream(result, finalizer=finalize, result_hooks=[hook])
        raw.append(wrapped)
        return wrapped

    monkeypatch.setattr(client, "_inner_get_response", create_autospec(original, side_effect=get_response))
    executor._agent.client = client
    executor._stream = True
    if validation:
        executor._agent.middleware = [ResponseValidationMiddleware(backoff_schedule=[0])]
    result = await executor.backend.run(executor.inputs.fresh_request(["work"]))
    assert isinstance(result, Ok)
    assert result.usage.total_tokens == (198 if validation else 99)
    assert result.usage.complete
    expected = [("finalize", 0), ("finalize", 1), ("hook", 1)] if validation else [("finalize", 0), ("hook", 0)]
    assert log == expected
    for index, stream in enumerate(raw):
        assert await stream.get_final_response() is finals[index]
    assert log == expected
