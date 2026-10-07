# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The loop tells the compaction strategy when the provider found the context window full.

The note makes the strategy compact before the next request. Only a strict
context overflow counts, and it is noted with or without a wire retry policy
(service-side storage runs have none). The one resend the note allows is
covered in ``test_loop_context_overflow_recovery``.
"""

from __future__ import annotations

from typing import Any

import pytest

from chrys.kernel import ContextOverflowSink, Message
from tests.kernel._fakes import (
    _final_response,
    _OverflowSink,
    _stack,
    _text_response,
    _text_update,
    _user,
    _WireRetryPolicy,
)
from tests.support.provider_errors import (
    anthropic_thinking_binding_rejection,
    openai_context_overflow,
    openai_status,
)

_NOT_OVERFLOW = {
    "payload_too_large": (
        413,
        {"error": {"type": "invalid_request_error", "code": "request_too_large", "message": "Request too large."}},
    ),
    # A gateway may wrap an overflow in a 5xx; the main turn keeps that a retry.
    "server_error_naming_the_window": (
        500,
        {"error": {"type": "server_error", "message": "This model's maximum context length is 128000 tokens."}},
    ),
    "bad_request": (
        400,
        {"error": {"type": "invalid_request_error", "message": "Invalid 'messages[0].content': string too long."}},
    ),
}


class _NoNote:
    """A compaction strategy without the note, as a test double or third party has."""

    max_context_tokens = 1_000_000
    last_included_tokens = 0
    system_overhead_tokens = 0
    calibration_ratio = 1.0

    async def __call__(self, messages: list[Message], context: Any = None) -> bool:
        return False


async def _fail(
    error: BaseException, *, strategy: Any, stream: bool, policy: _WireRetryPolicy | None, mid_stream: bool = False
) -> int:
    """Run one logical call that fails with *error*; return how many wire calls it made."""
    layer, wire = _stack([[_text_update("partial"), error] if mid_stream else error])
    client_kwargs = {"wire_retry_policy": policy} if policy is not None else {}
    with pytest.raises(type(error)) as raised:
        await _final_response(
            layer, [_user()], stream=stream, compaction_strategy=strategy, client_kwargs=client_kwargs
        )
    assert raised.value is error
    return len(wire.calls)


_CALL_SHAPES = [
    pytest.param(False, False, id="blocking"),
    pytest.param(True, False, id="streaming"),
    pytest.param(True, True, id="streaming_after_output"),
]


@pytest.mark.parametrize("with_policy", [True, False], ids=["local_retry", "no_wire_policy"])
@pytest.mark.parametrize(("stream", "mid_stream"), _CALL_SHAPES)
async def test_a_context_overflow_is_noted_once_and_the_call_still_fails(
    with_policy: bool, stream: bool, mid_stream: bool
) -> None:
    """The strategy says a resend cannot help, so even a local run fails at once."""
    error = await openai_context_overflow()
    sink = _OverflowSink(resend_helps=False)
    policy = _WireRetryPolicy() if with_policy else None

    wire_calls = await _fail(error, strategy=sink, stream=stream, policy=policy, mid_stream=mid_stream)

    assert sink.notes == [error]
    assert wire_calls == 1
    assert policy is None or policy.retries == []


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
@pytest.mark.parametrize("name", sorted(_NOT_OVERFLOW))
async def test_other_rejections_are_not_noted(stream: bool, name: str) -> None:
    error = await openai_status(*_NOT_OVERFLOW[name])
    sink = _OverflowSink()

    await _fail(error, strategy=sink, stream=stream, policy=_WireRetryPolicy())

    assert sink.notes == []


@pytest.mark.parametrize("with_policy", [True, False], ids=["local_retry", "no_wire_policy"])
@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_a_thinking_binding_refusal_naming_the_window_is_not_noted(stream: bool, with_policy: bool) -> None:
    """Compacting cannot fix thinking bound to another conversation; the Anthropic client handles it."""
    error = await anthropic_thinking_binding_rejection(names_the_window=True)
    sink = _OverflowSink()
    policy = _WireRetryPolicy() if with_policy else None

    wire_calls = await _fail(error, strategy=sink, stream=stream, policy=policy)

    assert sink.notes == []
    assert wire_calls == 1
    assert policy is None or policy.retries == []


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_a_transient_failure_that_recovers_is_not_noted(stream: bool) -> None:
    sink = _OverflowSink()
    policy = _WireRetryPolicy()
    success: Any = [_text_update("done")] if stream else _text_response("done")
    layer, wire = _stack([ConnectionError("peer closed connection"), success])

    response = await _final_response(
        layer,
        [_user()],
        stream=stream,
        compaction_strategy=sink,
        client_kwargs={"wire_retry_policy": policy},
    )

    assert response.text == "done"
    assert len(wire.calls) == 2
    assert sink.notes == []


@pytest.mark.parametrize("strategy", [None, _NoNote()], ids=["no_strategy", "strategy_without_the_note"])
@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_without_a_sink_the_overflow_fails_unchanged(strategy: Any, stream: bool) -> None:
    assert not isinstance(strategy, ContextOverflowSink)
    error = await openai_context_overflow()

    assert await _fail(error, strategy=strategy, stream=stream, policy=_WireRetryPolicy()) == 1
