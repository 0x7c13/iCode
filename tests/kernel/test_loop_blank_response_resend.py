# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A response that showed nothing is sampled once more, in place.

Validation rejects a response with no answer text and no call; when it is
empty, blank or reasoning-only, the wire caller resends the request at once,
outside the transient and stall budgets and at most once per logical call.
A rejection of a response that showed something, or that a resend cannot
fix, fails the call as before; so does any rejection in a run without a wire
retry policy (service-side storage). The tool loop's exhaustion tail is the
exception: a response that showed nothing passes there, and the loop answers
it with its fallback text.
"""

from __future__ import annotations

from typing import Any

import pytest

from chrys.foundation.retry import StreamStall
from chrys.foundation.trajectory.context import TRAJECTORY_CONTEXT_KWARG
from chrys.foundation.trajectory.event_types import EventType, RetryMode, RetryReason
from chrys.kernel import EXHAUSTION_TAIL_KWARG, ChatResponse, ChatResponseUpdate, Content, Message
from chrys.kernel import loop as loop_module
from chrys.kernel.loop import _MAX_ITERATIONS_FALLBACK_TEXT
from chrys.kernel.middleware import ChatMiddlewareLayer
from chrys.service.agent_middleware.response_validation import (
    ResponseValidationMiddleware,
    TerminalResponseValidationError,
)
from tests.kernel._fakes import (
    _call_response,
    _call_update,
    _final_response,
    _make_tool,
    _ScriptedClient,
    _text_response,
    _text_update,
    _user,
    _WireRetryPolicy,
)
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer

_MODES = pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])


def _reply(*contents: Any, stream: bool, finish_reason: str | None = None) -> Any:
    if stream:
        return [ChatResponseUpdate(contents=list(contents), role="assistant", finish_reason=finish_reason)]
    return ChatResponse(messages=[Message(role="assistant", contents=list(contents))], finish_reason=finish_reason)


def _blank(shape: str, *, stream: bool) -> Any:
    if shape == "empty":
        return _reply(stream=stream)
    if shape == "whitespace":
        return _reply(Content.from_text("  \n"), stream=stream)
    if shape == "reasoning":
        return _reply(Content.from_text_reasoning(text="thinking"), stream=stream)
    return _reply(Content.from_text_reasoning(text="thinking"), stream=stream, finish_reason="length")


def _answer(*, stream: bool) -> Any:
    return [_text_update("done")] if stream else _text_response("done")


async def _run(
    turns: list[Any],
    *,
    stream: bool,
    policy: _WireRetryPolicy | None,
    options: dict[str, Any] | None = None,
    **client_kwargs: Any,
) -> tuple[ChatResponse, int]:
    """Run one turn through the validation middleware; return its final response and its wire calls."""
    wire = _ScriptedClient(turns)
    layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire, middleware=[ResponseValidationMiddleware()]))
    if policy is not None:
        client_kwargs["wire_retry_policy"] = policy
    response = await _final_response(
        layer, [_user()], stream=stream, options=options or {"tools": [_make_tool()]}, client_kwargs=client_kwargs
    )
    return response, len(wire.calls)


async def _run_to_tail(
    turns: list[Any], *, stream: bool, policy: _WireRetryPolicy | None, service_side: bool = False
) -> tuple[ChatResponse, _ScriptedClient]:
    """Run a turn whose one tool batch exhausts the loop: the model call after it is the exhaustion tail."""
    wire = _ScriptedClient(list(turns))
    layer = InvariantCheckedToolLoopLayer(
        ChatMiddlewareLayer(wire, middleware=[ResponseValidationMiddleware()]), max_iterations=1
    )
    client_kwargs: dict[str, Any] = {} if policy is None else {"wire_retry_policy": policy}
    options: dict[str, Any] = {"tools": [_make_tool()], **({"store": True} if service_side else {})}
    response = await _final_response(layer, [_user()], stream=stream, options=options, client_kwargs=client_kwargs)
    return response, wire


def _tool_call(*, stream: bool) -> Any:
    return [_call_update("c1", "echo", {"text": "x"})] if stream else _call_response(("c1", "echo", {"text": "x"}))


@pytest.mark.parametrize("shape", ["empty", "whitespace", "reasoning", "reasoning_exhausted"])
@_MODES
async def test_a_response_that_showed_nothing_is_sampled_once_more(stream: bool, shape: str) -> None:
    policy = _WireRetryPolicy(max_retries=0)
    trajectory = FakeSink()

    response, wire_calls = await _run(
        [_blank(shape, stream=stream), _answer(stream=stream)],
        stream=stream,
        policy=policy,
        **{TRAJECTORY_CONTEXT_KWARG: make_context(trajectory)},
    )

    # The rejected attempt leaves nothing in the response: not even its whitespace.
    assert response.text == "done"
    assert wire_calls == 2
    [(attempt, max_attempts, delay, error)] = policy.retries
    assert (attempt, max_attempts, delay) == (1, 1, 0)
    assert isinstance(error, TerminalResponseValidationError)
    assert policy.before_retry_calls == 1
    [scheduled] = trajectory.of_type(EventType.RETRY_SCHEDULED)
    assert (scheduled.payload["retry_mode"], scheduled.payload["reason_code"], scheduled.payload["delay_ms"]) == (
        RetryMode.VALIDATION,
        RetryReason.VALIDATION_REJECTED,
        0,
    )


@_MODES
async def test_a_second_blank_response_in_the_same_call_fails(stream: bool) -> None:
    policy = _WireRetryPolicy()

    with pytest.raises(TerminalResponseValidationError, match="empty"):
        await _run([_blank("empty", stream=stream), _blank("whitespace", stream=stream)], stream=stream, policy=policy)

    assert len(policy.retries) == 1


@pytest.mark.parametrize(
    ("finish_reason", "text"),
    [
        pytest.param("length", "", id="input_filled_the_window"),
        pytest.param("content_filter", "", id="content_filter"),
    ],
)
@_MODES
async def test_a_rejection_a_resend_cannot_fix_fails_at_once(stream: bool, finish_reason: str, text: str) -> None:
    policy = _WireRetryPolicy()
    contents = [Content.from_text(text)] if text else []

    with pytest.raises(TerminalResponseValidationError):
        await _run([_reply(*contents, stream=stream, finish_reason=finish_reason)], stream=stream, policy=policy)

    assert policy.retries == []


@_MODES
async def test_hosted_work_the_provider_already_ran_is_not_resent(stream: bool) -> None:
    policy = _WireRetryPolicy(hosted_in_flight=("mcp",))

    with pytest.raises(TerminalResponseValidationError):
        await _run([_blank("empty", stream=stream)], stream=stream, policy=policy)

    assert policy.retries == []


@_MODES
async def test_a_run_without_a_wire_retry_policy_fails_at_once(stream: bool) -> None:
    """Service-side storage forbids an in-place resend."""
    with pytest.raises(TerminalResponseValidationError):
        await _run([_blank("empty", stream=stream), _answer(stream=stream)], stream=stream, policy=None)


@_MODES
async def test_each_model_call_of_a_turn_may_resend_once(stream: bool) -> None:
    call: Any = [_call_update("c1", "echo", {"text": "x"})] if stream else _call_response(("c1", "echo", {"text": "x"}))

    response, wire_calls = await _run(
        [_blank("empty", stream=stream), call, _blank("reasoning", stream=stream), _answer(stream=stream)],
        stream=stream,
        policy=_WireRetryPolicy(),
    )

    assert response.text == "done"
    assert wire_calls == 4


async def test_the_blocking_fallback_after_a_resend_does_not_resend_again() -> None:
    stall = StreamStall("Streaming response produced no progress")
    policy = _WireRetryPolicy(stall_max_retries=0)

    with pytest.raises(TerminalResponseValidationError):
        await _run([_blank("empty", stream=True), [stall], _blank("empty", stream=False)], stream=True, policy=policy)

    assert [type(retry[3]) for retry in policy.retries] == [TerminalResponseValidationError, StreamStall]


# ---------------------------------------------------------------------------
# The exhaustion tail
# ---------------------------------------------------------------------------


_RUNS = pytest.mark.parametrize(
    ("with_policy", "service_side"),
    [(True, False), (False, False), (True, True)],
    ids=["wire_policy", "no_wire_policy", "service_side_storage"],
)


@pytest.mark.parametrize("shape", ["empty", "whitespace", "reasoning", "reasoning_exhausted"])
@_RUNS
@_MODES
async def test_a_blank_exhaustion_tail_gets_the_fallback_text(
    stream: bool, with_policy: bool, service_side: bool, shape: str
) -> None:
    # Service-side storage drops the wire policy it is given.
    policy = _WireRetryPolicy() if with_policy else None

    response, wire = await _run_to_tail(
        [_tool_call(stream=stream), _blank(shape, stream=stream)],
        stream=stream,
        policy=policy,
        service_side=service_side,
    )

    assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT
    # The tail's fallback answers it: no resend.
    assert len(wire.calls) == 2
    assert policy is None or policy.retries == []
    assert [EXHAUSTION_TAIL_KWARG in call["kwargs"]["client_kwargs"] for call in wire.calls] == [False, True]
    stored = True if service_side else None
    assert [call["options"].get("store") for call in wire.calls] == [stored, stored]


@_MODES
async def test_the_tail_poll_of_a_background_response_gets_the_fallback_text(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop_module, "CONTINUATION_POLL_INTERVAL_SECONDS", 0.0)
    token = {"response_id": "background"}
    pending: Any = (
        [ChatResponseUpdate(contents=[], role="assistant", continuation_token=token)]
        if stream
        else ChatResponse(messages=[], continuation_token=token)
    )

    response, wire = await _run_to_tail(
        [_tool_call(stream=stream), pending, _blank("empty", stream=stream)], stream=stream, policy=_WireRetryPolicy()
    )

    assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT
    assert [EXHAUSTION_TAIL_KWARG in call["kwargs"]["client_kwargs"] for call in wire.calls] == [False, True, True]


async def test_the_tail_keeps_its_fallback_through_the_stall_fallback() -> None:
    stall = StreamStall("Streaming response produced no progress")
    policy = _WireRetryPolicy(stall_max_retries=0)

    response, wire = await _run_to_tail(
        [_tool_call(stream=True), [stall], _blank("empty", stream=False)], stream=True, policy=policy
    )

    assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT
    assert [type(retry[3]) for retry in policy.retries] == [StreamStall]
    assert [EXHAUSTION_TAIL_KWARG in call["kwargs"]["client_kwargs"] for call in wire.calls] == [False, True, True]


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
@_MODES
async def test_a_tail_rejection_the_fallback_cannot_answer_still_fails(stream: bool, finish_reason: str) -> None:
    with pytest.raises(TerminalResponseValidationError):
        await _run_to_tail(
            [_tool_call(stream=stream), _reply(stream=stream, finish_reason=finish_reason)],
            stream=stream,
            policy=_WireRetryPolicy(),
        )


@_MODES
async def test_an_ordinary_call_asking_for_no_tools_still_resends_a_blank_response(stream: bool) -> None:
    policy = _WireRetryPolicy()

    response, wire_calls = await _run(
        [_blank("empty", stream=stream), _answer(stream=stream)],
        stream=stream,
        policy=policy,
        options={"tools": [_make_tool()], "tool_choice": "none"},
    )

    assert response.text == "done"
    assert wire_calls == 2
    assert len(policy.retries) == 1
