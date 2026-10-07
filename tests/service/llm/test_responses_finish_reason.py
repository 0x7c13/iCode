# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""OpenAI Responses adapter maps an output-token cutoff to finish_reason='length'.

The Responses API signals a truncated response via ``status: "incomplete"`` +
``incomplete_details.reason: "max_output_tokens"`` rather than the Chat
Completions ``finish_reason: "length"``. The adapter normalises it so downstream
truncation handling (response validation, tool-arg parsing) behaves identically
across providers. A response the content filter stopped reads as
``content_filter``, as on Chat Completions.
"""

from __future__ import annotations

from types import SimpleNamespace

from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.decode import decode_response
from chrys.service.llm.openai_responses.stream import StreamState


def _stream() -> StreamState:
    return StreamState({}, model="gpt-test", variant=OPENAI_RESPONSES)


def _event(event_type: str, *, reason: str | None, status: str | None = None) -> SimpleNamespace:
    incomplete_details = SimpleNamespace(reason=reason) if reason is not None else None
    response_status = status if status is not None else event_type.removeprefix("response.")
    response = SimpleNamespace(
        id="resp_1",
        created_at=0,
        model="gpt-test",
        usage=None,
        conversation=None,
        status=response_status,
        incomplete_details=incomplete_details,
    )
    return SimpleNamespace(type=event_type, response=response)


def _response(*, status: str, reason: str | None) -> SimpleNamespace:
    incomplete_details = SimpleNamespace(reason=reason) if reason is not None else None
    return SimpleNamespace(
        id="resp_1",
        created_at=0,
        model="gpt-test",
        metadata={},
        output=[],
        usage=None,
        conversation=None,
        status=status,
        incomplete_details=incomplete_details,
    )


def test_streaming_incomplete_max_output_tokens_maps_to_length() -> None:
    update = _stream().update_for(_event("response.incomplete", reason="max_output_tokens"))
    assert update.finish_reason == "length"


def test_streaming_completed_has_no_length_finish_reason() -> None:
    # A normally-completed response must not be labelled truncated.
    update = _stream().update_for(_event("response.completed", reason=None))
    assert update.finish_reason is None


def test_streaming_completed_with_incomplete_details_is_not_length() -> None:
    # Be conservative with OpenAI-compatible gateways: incomplete_details alone
    # is not enough unless the response status is also incomplete.
    update = _stream().update_for(_event("response.completed", reason="max_output_tokens"))
    assert update.finish_reason is None


def test_streaming_incomplete_content_filter_maps_to_content_filter() -> None:
    update = _stream().update_for(_event("response.incomplete", reason="content_filter"))
    assert update.finish_reason == "content_filter"


def test_streaming_incomplete_other_reason_has_no_finish_reason() -> None:
    update = _stream().update_for(_event("response.incomplete", reason="something_new"))
    assert update.finish_reason is None


def test_non_streaming_incomplete_max_output_tokens_maps_to_length() -> None:
    response = decode_response(
        _response(status="incomplete", reason="max_output_tokens"),
        {},
        variant=OPENAI_RESPONSES,
    )
    assert response.finish_reason == "length"


def test_non_streaming_completed_with_incomplete_details_is_not_length() -> None:
    response = decode_response(
        _response(status="completed", reason="max_output_tokens"),
        {},
        variant=OPENAI_RESPONSES,
    )
    assert response.finish_reason is None


def test_non_streaming_incomplete_content_filter_maps_to_content_filter() -> None:
    response = decode_response(
        _response(status="incomplete", reason="content_filter"),
        {},
        variant=OPENAI_RESPONSES,
    )
    assert response.finish_reason == "content_filter"


def test_non_streaming_incomplete_other_reason_has_no_finish_reason() -> None:
    response = decode_response(
        _response(status="incomplete", reason="something_new"),
        {},
        variant=OPENAI_RESPONSES,
    )
    assert response.finish_reason is None
