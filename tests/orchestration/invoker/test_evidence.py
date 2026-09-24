# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Invocation evidence algebra and the hosted shell reporting boundary."""

from __future__ import annotations

import pytest

from chrys.kernel import Content
from chrys.orchestration.invoker.evidence import (
    UNKNOWN_COUNT,
    Completeness,
    Count,
    InvocationEvidence,
    PassEvidence,
    WireSnapshot,
    hosted_count,
)
from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware
from tests.service.agent_middleware._response_validation_fakes import (
    _assistant,
    _FakeCallNext,
    _final_response,
    _make_context,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (Count(2, Completeness.EXACT), Count(3, Completeness.EXACT), Count(5, Completeness.EXACT)),
        (Count(2, Completeness.EXACT), UNKNOWN_COUNT, Count(2, Completeness.LOWER_BOUND)),
        (Count(0, Completeness.EXACT), UNKNOWN_COUNT, UNKNOWN_COUNT),
        (Count(0, Completeness.LOWER_BOUND), UNKNOWN_COUNT, Count(0, Completeness.LOWER_BOUND)),
        (Count(2, Completeness.LOWER_BOUND), Count(3, Completeness.EXACT), Count(5, Completeness.LOWER_BOUND)),
        (Count(2, Completeness.LOWER_BOUND), Count(3, Completeness.LOWER_BOUND), Count(5, Completeness.LOWER_BOUND)),
        (UNKNOWN_COUNT, UNKNOWN_COUNT, UNKNOWN_COUNT),
    ],
)
def test_count_merge_is_symmetric_and_preserves_observed_lower_bound(
    left: Count, right: Count, expected: Count
) -> None:
    assert left + right == expected
    assert right + left == expected


@pytest.mark.parametrize("left", [False, True, None])
@pytest.mark.parametrize("right", [False, True, None])
def test_stateful_three_valued_or(left: bool | None, right: bool | None) -> None:
    first = PassEvidence("inv", "1", UNKNOWN_COUNT, UNKNOWN_COUNT, UNKNOWN_COUNT, left)
    second = PassEvidence("inv", "2", UNKNOWN_COUNT, UNKNOWN_COUNT, UNKNOWN_COUNT, right)
    total = InvocationEvidence("inv").add(first).add(second)
    expected = True if left is True or right is True else False if left is False and right is False else None
    assert total.external_stateful is expected


def test_converged_passes_accumulate_once_across_retry_and_cancellation() -> None:
    first = PassEvidence("inv", "1", Count(2, Completeness.EXACT), UNKNOWN_COUNT, hosted_count(("shell",)), False)
    total = InvocationEvidence("inv").add(first)
    with pytest.raises(ValueError, match="already accumulated"):
        total.add(first)
    with pytest.raises(TypeError, match="Only pass"):
        total.add(WireSnapshot("inv", "1", 0, hosted_count(("shell",))))  # type: ignore[arg-type]
    cancelled = PassEvidence("inv", "2", UNKNOWN_COUNT, UNKNOWN_COUNT, hosted_count(()), False)
    retried = total.add(cancelled)
    assert total.passes == ("1",)
    assert retried.passes == ("1", "2")
    assert retried.local_dispatched == Count(2, Completeness.LOWER_BOUND)
    assert retried.hosted_observed == Count(1, Completeness.LOWER_BOUND)
    with pytest.raises(ValueError, match="Foreign"):
        retried.add(PassEvidence("foreign", "3", UNKNOWN_COUNT, UNKNOWN_COUNT, UNKNOWN_COUNT, None))


@pytest.mark.parametrize("stream", [False, True])
async def test_hosted_shell_call_and_result_produce_one_lower_bound(stream: bool) -> None:
    response = _assistant(
        [
            Content.from_shell_tool_call(call_id="sh1", commands=["touch output"]),
            Content.from_shell_tool_result(call_id="sh1", outputs=[]),
            Content.from_text("done"),
        ]
    )
    context = _make_context(stream=stream)
    fake = _FakeCallNext([response], stream=stream)
    fake.bind(context)
    middleware = ResponseValidationMiddleware(backoff_schedule=[0.0])
    await middleware.process(context, fake)
    await _final_response(context, stream=stream)
    labels = middleware.hosted_commits_observed()
    assert labels == ("shell", "shell_tool_result")
    assert hosted_count(labels) == Count(1, Completeness.LOWER_BOUND)
    assert hosted_count(None) == UNKNOWN_COUNT
    assert hosted_count(()) == Count(0, Completeness.LOWER_BOUND)


async def test_real_loop_recorder_positive_pass_counts_accumulate_once(tmp_path, monkeypatch, agent_engine):
    from chrys.foundation.events.types import UserMessage, UserRetry
    from chrys.kernel import tool
    from chrys.orchestration.invoker.contracts import Failed, Ok
    from chrys.service.llm.mock import MockResponse
    from tests.orchestration.invoker._build_fixtures import build_recipe_engine

    calls = []

    @tool
    async def counted() -> str:
        calls.append("committed")
        return "done"

    engine, main, _ = await build_recipe_engine(
        agent_engine,
        monkeypatch,
        tmp_path,
        main=[
            MockResponse(tool_calls=[("counted", "a", {})]),
            ValueError("failed after one tool"),
            MockResponse(tool_calls=[("counted", "b", {}), ("counted", "c", {})]),
            MockResponse(text="done"),
        ],
        child=[],
    )
    executor = engine.current.loaded.bindings
    executor._agent.default_options["tools"] = [counted]
    try:
        await engine.event_bus.publish(UserMessage(text="work"))
        await engine.wait_for_run_task()
        first = executor.inputs.outcome
        assert isinstance(first, Failed)
        assert executor._loop_recorder.committed_count == 1
        assert first.effects.local_answered == Count(1, Completeness.EXACT)
        await engine.event_bus.publish(UserRetry())
        await engine.wait_for_run_task()
        second = executor.inputs.outcome
        assert isinstance(second, Ok)
        assert executor._loop_recorder.committed_count == 2
        assert second.effects.local_answered == Count(2, Completeness.EXACT)
        assert executor.inputs.evidence.local_answered == Count(3, Completeness.EXACT)
        assert executor.inputs.evidence.local_dispatched == UNKNOWN_COUNT
        assert executor.inputs.evidence.passes == (first.handle.pass_id, second.handle.pass_id)
        assert calls == ["committed"] * 3
        assert main.call_count == 4
    finally:
        await engine.shutdown()


@pytest.mark.parametrize("stream", [False, True])
async def test_direct_backend_passes_own_recorder_reset(executor, stream):
    from chrys.kernel import Message, tool
    from chrys.orchestration.invoker.contracts import Ok, RunIntent, RunRequest
    from chrys.service.llm.mock import MockChatClient, MockResponse

    calls = []

    @tool
    async def counted() -> str:
        calls.append("committed")
        return "done"

    executor._stream = stream
    executor._agent.default_options["tools"] = [counted]
    executor._agent.client = MockChatClient(
        responses=[
            MockResponse(tool_calls=[("counted", "a", {})]),
            MockResponse(text="first"),
            MockResponse(tool_calls=[("counted", "b", {}), ("counted", "c", {})]),
            MockResponse(text="second"),
        ]
    )
    origin = executor.inputs.origin
    total = InvocationEvidence(origin.invocation_id)
    for count in (1, 2):
        outcome = await executor.backend.run(RunRequest([Message("user", ["work"])], RunIntent.FRESH, origin))
        assert isinstance(outcome, Ok)
        assert outcome.effects.local_answered == Count(count, Completeness.EXACT)
        total = total.add(outcome.effects)
    assert total.local_answered == Count(3, Completeness.EXACT)
    assert len(total.passes) == 2
    assert calls == ["committed"] * 3
