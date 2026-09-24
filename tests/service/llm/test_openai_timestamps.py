# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for OpenAI-compatible timestamp normalization."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from chrys.service.llm.openai_timestamps import (
    normalize_openai_created_payload,
    normalize_openai_created_timestamp,
    openai_created_at_iso,
)


def test_normalize_openai_created_timestamp_leaves_standard_seconds_unchanged() -> None:
    created = 1_717_171_717

    assert normalize_openai_created_timestamp(created) == created


def test_openai_created_at_iso_converts_thirteen_digit_milliseconds() -> None:
    created_ms = 1_717_171_717_123

    assert openai_created_at_iso(created_ms) == datetime.fromtimestamp(
        1_717_171_717.123,
        tz=UTC,
    ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def test_normalize_openai_created_payload_copies_openai_sdk_models() -> None:
    created_ms = 1_717_171_717_123
    chunk = ChatCompletionChunk(
        id="chunk-1",
        object="chat.completion.chunk",
        created=created_ms,
        model="gpt-test",
        choices=[],
    )

    normalized = normalize_openai_created_payload(chunk)

    assert normalized is not chunk
    assert normalized.created == 1_717_171_717.123
    assert chunk.created == created_ms


def test_normalize_openai_created_payload_leaves_standard_seconds_object_unchanged() -> None:
    payload = SimpleNamespace(created=1_717_171_717)

    assert normalize_openai_created_payload(payload) is payload
