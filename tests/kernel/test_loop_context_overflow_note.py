# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The loop tells the compaction strategy when the provider found the context window full.

The request still fails; the note makes the strategy compact before the next
one. Only a strict context overflow counts, and it is noted with or without a
wire retry policy (service-side storage runs have none).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from chrys.foundation.retry import StreamStall
from chrys.kernel import ContextOverflowSink, Message, StallExhaustedAction
from tests.kernel._fakes import _final_response, _stack, _text_response, _text_update, _user
from tests.support.provider_errors import openai_status

_OVERFLOW = (
    400,
    {
        "error": {
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "message": "This model's maximum context length is 131072 tokens. "
            "However, your messages resulted in 140000 tokens.",
        }
    },
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


class _Sink:
    """A compaction strategy reduced to the overflow note."""

    def __init__(self) -> None:
        self.notes: list[BaseException | None] = []

    async def __call__(self, messages: list[Message], context: Any = None) -> bool:
        return False

    def note_context_overflow(self, exc: BaseException | None = None) -> bool:
        self.notes.append(exc)
        return True


class _NoNote:
    """A compaction strategy without the note, as a test double or third party has."""

    max_context_tokens = 1_000_000
    last_included_tokens = 0
    system_overhead_tokens = 0
    calibration_ratio = 1.0

    async def __call__(self, messages: list[Message], context: Any = None) -> bool:
        return False


@dataclass
class _Policy:
    """Local-storage wire retry that retries nothing a provider rejected."""

    max_retries: int = 2
    stall_timeout_seconds: float | None = None
    stall_max_retries: int = 0
    stall_exhausted_action: StallExhaustedAction = StallExhaustedAction.BLOCKING_FALLBACK
    retries: list[BaseException] = field(default_factory=list)

    def backoff_seconds(self, _attempt: int) -> int:
        return 0

    def is_retryable(self, exc: BaseException) -> bool:
        return isinstance(exc, ConnectionError | StreamStall)

    def is_interrupted(self) -> bool:
        return False

    async def sleep(self, seconds: int) -> bool:
        return False

    async def on_retry(self, message: str, attempt: int, max_attempts: int, delay: int, exc: BaseException) -> None:
        self.retries.append(exc)

    def before_retry(self) -> None:
        pass


async def _fail(
    error: BaseException, *, strategy: Any, stream: bool, policy: _Policy | None, mid_stream: bool = False
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
    error = await openai_status(*_OVERFLOW)
    sink = _Sink()
    policy = _Policy() if with_policy else None

    wire_calls = await _fail(error, strategy=sink, stream=stream, policy=policy, mid_stream=mid_stream)

    assert sink.notes == [error]
    assert wire_calls == 1
    assert policy is None or policy.retries == []


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
@pytest.mark.parametrize("name", sorted(_NOT_OVERFLOW))
async def test_other_rejections_are_not_noted(stream: bool, name: str) -> None:
    error = await openai_status(*_NOT_OVERFLOW[name])
    sink = _Sink()

    await _fail(error, strategy=sink, stream=stream, policy=_Policy())

    assert sink.notes == []


@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_a_transient_failure_that_recovers_is_not_noted(stream: bool) -> None:
    sink = _Sink()
    policy = _Policy()
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
    error = await openai_status(*_OVERFLOW)

    assert await _fail(error, strategy=strategy, stream=stream, policy=_Policy()) == 1
