# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A strict context overflow is resent once, in place, when the strategy says compacting can help.

The strategy holds the note, so the resend's client preparation compacts
first; these doubles do not compact, so the tests count notes, announced
retries and wire calls. The resend is outside the transient and stall budgets
and happens at most once per logical call; a run without a wire retry policy
(service-side storage) only keeps the note.
"""

from __future__ import annotations

from typing import Any

import pytest

from chrys.foundation.retry import StreamStall
from chrys.foundation.trajectory.context import TRAJECTORY_CONTEXT_KWARG
from chrys.foundation.trajectory.event_types import EventType, RetryMode, RetryReason
from chrys.kernel import ChatResponse
from tests.kernel._fakes import (
    _call_response,
    _call_update,
    _final_response,
    _make_tool,
    _OverflowSink,
    _stack,
    _text_response,
    _text_update,
    _user,
    _WireRetryPolicy,
)
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.provider_errors import anthropic_thinking_binding_rejection, openai_context_overflow


def _answer(*, stream: bool) -> Any:
    return [_text_update("done")] if stream else _text_response("done")


async def _run(
    turns: list[Any],
    *,
    stream: bool,
    sink: _OverflowSink,
    policy: _WireRetryPolicy | None,
    **client_kwargs: Any,
) -> tuple[ChatResponse, int]:
    """Run one turn; return its final response and how many wire calls it made."""
    layer, wire = _stack(turns)
    if policy is not None:
        client_kwargs["wire_retry_policy"] = policy
    response = await _final_response(
        layer,
        [_user()],
        stream=stream,
        options={"tools": [_make_tool()]},
        compaction_strategy=sink,
        client_kwargs=client_kwargs,
    )
    return response, len(wire.calls)


@pytest.mark.parametrize(
    ("stream", "mid_stream"),
    [
        pytest.param(False, False, id="blocking"),
        pytest.param(True, False, id="streaming"),
        pytest.param(True, True, id="streaming_after_output"),
    ],
)
async def test_an_overflow_is_resent_once_without_using_the_retry_budget(stream: bool, mid_stream: bool) -> None:
    error = await openai_context_overflow()
    sink = _OverflowSink()
    policy = _WireRetryPolicy(max_retries=0)
    trajectory = FakeSink()
    rejected = [_text_update("partial"), error] if mid_stream else error

    response, wire_calls = await _run(
        [rejected, _answer(stream=stream)],
        stream=stream,
        sink=sink,
        policy=policy,
        **{TRAJECTORY_CONTEXT_KWARG: make_context(trajectory)},
    )

    assert response.text == "done"
    assert wire_calls == 2
    assert sink.notes == [error]
    assert policy.before_retry_calls == 1
    assert policy.retries == [(1, 1, 0, error)]
    [scheduled] = trajectory.of_type(EventType.RETRY_SCHEDULED)
    assert (scheduled.payload["retry_mode"], scheduled.payload["reason_code"], scheduled.payload["delay_ms"]) == (
        RetryMode.CONTEXT_OVERFLOW,
        RetryReason.TRANSIENT_ERROR,
        0,
    )
    [started] = trajectory.of_type(EventType.RETRY_STARTED)
    assert started.payload["retry_mode"] == RetryMode.CONTEXT_OVERFLOW


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_a_second_overflow_in_the_same_call_fails(stream: bool) -> None:
    first, second = await openai_context_overflow(), await openai_context_overflow()
    sink = _OverflowSink()
    policy = _WireRetryPolicy()

    with pytest.raises(type(second)) as raised:
        await _run([first, second], stream=stream, sink=sink, policy=policy)

    assert raised.value is second
    assert sink.notes == [first, second]
    assert [retry[3] for retry in policy.retries] == [first]


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_a_thinking_binding_refusal_after_the_resend_fails_without_a_note(stream: bool) -> None:
    overflow = await openai_context_overflow()
    refusal = await anthropic_thinking_binding_rejection(names_the_window=True)
    sink = _OverflowSink()
    policy = _WireRetryPolicy()

    with pytest.raises(type(refusal)) as raised:
        await _run([overflow, refusal], stream=stream, sink=sink, policy=policy)

    assert raised.value is refusal
    assert sink.notes == [overflow]
    assert [retry[3] for retry in policy.retries] == [overflow]


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_after_the_resend_a_dropped_connection_keeps_its_retries(stream: bool) -> None:
    overflow = await openai_context_overflow()
    dropped = ConnectionError("peer closed connection")
    policy = _WireRetryPolicy(max_retries=1)

    response, wire_calls = await _run(
        [overflow, dropped, _answer(stream=stream)], stream=stream, sink=_OverflowSink(), policy=policy
    )

    assert response.text == "done"
    assert wire_calls == 3
    assert policy.retries == [(1, 1, 0, overflow), (1, 1, 0, dropped)]


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_each_model_call_of_a_turn_may_resend_once(stream: bool) -> None:
    first, second = await openai_context_overflow(), await openai_context_overflow()
    call: Any = [_call_update("c1", "echo", {"text": "x"})] if stream else _call_response(("c1", "echo", {"text": "x"}))
    sink = _OverflowSink()

    response, wire_calls = await _run(
        [first, call, second, _answer(stream=stream)], stream=stream, sink=sink, policy=_WireRetryPolicy()
    )

    assert response.text == "done"
    assert wire_calls == 4
    assert sink.notes == [first, second]


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_hosted_work_the_provider_already_ran_is_not_resent(stream: bool) -> None:
    error = await openai_context_overflow()
    sink = _OverflowSink()
    policy = _WireRetryPolicy(hosted_in_flight=("mcp",))

    with pytest.raises(type(error)) as raised:
        await _run([error], stream=stream, sink=sink, policy=policy)

    assert raised.value is error
    assert sink.notes == [error]
    assert policy.retries == []


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_a_run_without_a_wire_retry_policy_only_notes(stream: bool) -> None:
    """Service-side storage forbids an in-place resend."""
    error = await openai_context_overflow()
    sink = _OverflowSink()

    with pytest.raises(type(error)):
        await _run([error], stream=stream, sink=sink, policy=None)

    assert sink.notes == [error]


async def test_the_blocking_fallback_after_a_resend_does_not_resend_again() -> None:
    first, second = await openai_context_overflow(), await openai_context_overflow()
    stall = StreamStall("Streaming response produced no progress")
    sink = _OverflowSink()
    policy = _WireRetryPolicy(stall_max_retries=0)

    with pytest.raises(type(second)) as raised:
        await _run([first, stall, second], stream=True, sink=sink, policy=policy)

    assert raised.value is second
    assert sink.notes == [first, second]
    assert [retry[3] for retry in policy.retries] == [first, stall]
