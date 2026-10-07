# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Anthropic's refusal of replayed thinking bound to a different conversation.

Only one provider signal carrying all three facts counts: HTTP 400,
``invalid_request_error`` and the binding sentence.
"""

from __future__ import annotations

from typing import NoReturn

import httpx
import pytest

from chrys.foundation.errors import is_context_overflow, is_retryable, is_thinking_binding_rejection
from tests.support.provider_errors import (
    ANTHROPIC_THINKING_BINDING_MESSAGE,
    anthropic_error_event,
    anthropic_status,
    anthropic_stream_error,
    anthropic_thinking_binding_rejection,
    raised_from,
    raised_while_handling,
)


def _body(error_type: str, message: str) -> dict[str, object]:
    return {"type": "error", "error": {"type": error_type, "message": message}}


async def test_the_binding_refusal_is_recognized_and_not_retried() -> None:
    exc = await anthropic_thinking_binding_rejection()

    assert is_thinking_binding_rejection(exc) is True
    assert is_retryable(exc) is False
    assert is_context_overflow(exc) is False


async def test_a_wrapper_raised_from_the_refusal_is_recognized() -> None:
    exc = raised_from(RuntimeError("the call failed"), await anthropic_thinking_binding_rejection())

    assert is_thinking_binding_rejection(exc) is True


@pytest.mark.parametrize(
    ("status", "error_type", "message"),
    [
        pytest.param(400, "invalid_request_error", "messages: at least one message is required", id="other_400"),
        pytest.param(
            400,
            "invalid_request_error",
            "messages.1.content.2: Invalid `signature` in `thinking` block",
            id="corrupted_signature",
        ),
        pytest.param(400, "api_error", ANTHROPIC_THINKING_BINDING_MESSAGE, id="other_error_type"),
        pytest.param(429, "rate_limit_error", ANTHROPIC_THINKING_BINDING_MESSAGE, id="rate_limited"),
        pytest.param(500, "api_error", ANTHROPIC_THINKING_BINDING_MESSAGE, id="server_error"),
    ],
)
async def test_other_rejections_are_not_the_binding_refusal(status: int, error_type: str, message: str) -> None:
    exc = await anthropic_status(status, _body(error_type, message))

    assert is_thinking_binding_rejection(exc) is False


async def test_a_stream_error_after_the_200_is_not_the_binding_refusal() -> None:
    exc = await anthropic_stream_error(
        anthropic_error_event("invalid_request_error", ANTHROPIC_THINKING_BINDING_MESSAGE)
    )

    assert is_thinking_binding_rejection(exc) is False


async def test_a_refusal_left_only_in_the_implicit_context_is_not_recognized() -> None:
    stale = await anthropic_thinking_binding_rejection()

    def raise_fresh() -> NoReturn:
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    assert is_thinking_binding_rejection(raised_while_handling(stale, raise_fresh)) is False


@pytest.mark.parametrize("first", ["plain_400", "sentence_in_a_500"])
async def test_facts_spread_over_two_signals_are_not_recognized(first: str) -> None:
    plain_400 = await anthropic_status(400, _body("invalid_request_error", "messages: invalid content"))
    sentence_in_a_500 = await anthropic_status(500, _body("api_error", ANTHROPIC_THINKING_BINDING_MESSAGE))
    members = [plain_400, sentence_in_a_500] if first == "plain_400" else [sentence_in_a_500, plain_400]

    assert is_thinking_binding_rejection(ExceptionGroup("both", members)) is False
