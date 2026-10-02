# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Anthropic stop reasons reach both validation and telemetry on every parse path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from anthropic.types.beta import BetaMCPToolUseBlock, BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaUsage

from chrys.foundation.errors import ProviderResponseError
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


def _refused(*content: object) -> BetaMessage:
    return BetaMessage.model_construct(
        id="m1",
        model="test",
        role="assistant",
        content=list(content),
        stop_reason="refusal",
        usage=BetaUsage(input_tokens=10, output_tokens=3),
    )


def test_a_blocking_message_refused_with_calls_raises_with_the_hosted_work_it_showed() -> None:
    message = _refused(
        BetaMCPToolUseBlock(type="mcp_tool_use", id="mcptoolu_1", name="search", server_name="docs", input={}),
        BetaToolUseBlock(type="tool_use", id="toolu_1", name="zsh", input={"command": "ls"}),
    )

    with pytest.raises(ProviderResponseError) as raised:
        decode_message(message, response_format=None)

    assert (raised.value.code, raised.value.retryable) == ("content_filter", False)
    assert [content.call_id for content in raised.value.observed_contents] == ["mcptoolu_1"]  # type: ignore[attr-defined]
    assert (raised.value.usage_details or {}).get("output_token_count") == 3


def test_a_blocking_message_refused_without_calls_ends_as_filtered() -> None:
    response = decode_message(_refused(BetaTextBlock(type="text", text="I can't help.")), response_format=None)

    assert (response.finish_reason, response.text) == ("content_filter", "I can't help.")
