# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Anthropic stop reasons reach both validation and telemetry on every parse path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from anthropic.types.beta import BetaMessage, BetaUsage

from chrys.kernel import ChatResponse
from chrys.service.agent_middleware.validators import DefaultResponseValidator, ValidationReason
from chrys.service.llm.anthropic_messages.decode import decode_message
from chrys.service.llm.anthropic_messages.stream import StreamState


@pytest.mark.parametrize(
    "reason, expected", [("model_context_window_exceeded", "length"), ("future_reason", "future_reason")]
)
@pytest.mark.parametrize("mode", ["blocking", "message_start", "message_delta"])
def test_anthropic_stop_reason_survives_all_response_paths(reason, expected, mode) -> None:
    message = BetaMessage.model_construct(
        id="m1",
        model="test",
        role="assistant",
        content=[],
        stop_reason=reason,
        usage=BetaUsage(input_tokens=100, output_tokens=0),
    )
    if mode == "blocking":
        response = decode_message(message, response_format=None)
    else:
        event = SimpleNamespace(type=mode, message=message, delta=SimpleNamespace(stop_reason=reason), usage=None)
        (update,) = StreamState().updates_for(event)  # type: ignore[arg-type]
        response = ChatResponse.from_updates([update])
    assert response.finish_reason == expected
    result = DefaultResponseValidator().validate(response)
    if expected == "length":
        assert result.code == ValidationReason.OUTPUT_TRUNCATED
        assert not result.retryable
    else:
        assert result.retryable
