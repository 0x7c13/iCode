# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""OpenAI Responses adapter maps an output-token cutoff to finish_reason='length'.

The Responses API signals a truncated response via ``status: "incomplete"`` +
``incomplete_details.reason: "max_output_tokens"`` rather than the Chat
Completions ``finish_reason: "length"``. The adapter normalises it so downstream
truncation handling (response validation, tool-arg parsing) behaves identically
across providers.
"""

from __future__ import annotations

from types import SimpleNamespace

from chrys.service.llm.openai_responses import RawOpenAIChatClient


class _FakeAsyncOpenAI:
    base_url = "https://api.test"


def _client() -> RawOpenAIChatClient:
    return RawOpenAIChatClient(model="gpt-test", async_client=_FakeAsyncOpenAI())


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
    update = _client()._parse_chunk_from_openai(_event("response.incomplete", reason="max_output_tokens"), {}, {})
    assert update.finish_reason == "length"


def test_streaming_completed_has_no_length_finish_reason() -> None:
    # A normally-completed response must not be labelled truncated.
    update = _client()._parse_chunk_from_openai(_event("response.completed", reason=None), {}, {})
    assert update.finish_reason is None


def test_streaming_completed_with_incomplete_details_is_not_length() -> None:
    # Be conservative with OpenAI-compatible gateways: incomplete_details alone
    # is not enough unless the response status is also incomplete.
    update = _client()._parse_chunk_from_openai(_event("response.completed", reason="max_output_tokens"), {}, {})
    assert update.finish_reason is None


def test_streaming_incomplete_other_reason_is_not_length() -> None:
    # Only an output-token cutoff maps to "length"; content_filter etc. do not.
    update = _client()._parse_chunk_from_openai(_event("response.incomplete", reason="content_filter"), {}, {})
    assert update.finish_reason is None


def test_non_streaming_incomplete_max_output_tokens_maps_to_length() -> None:
    response = _client()._parse_response_from_openai(
        _response(status="incomplete", reason="max_output_tokens"),
        {},
    )
    assert response.finish_reason == "length"


def test_non_streaming_completed_with_incomplete_details_is_not_length() -> None:
    response = _client()._parse_response_from_openai(
        _response(status="completed", reason="max_output_tokens"),
        {},
    )
    assert response.finish_reason is None


def test_non_streaming_incomplete_other_reason_is_not_length() -> None:
    response = _client()._parse_response_from_openai(
        _response(status="incomplete", reason="content_filter"),
        {},
    )
    assert response.finish_reason is None
