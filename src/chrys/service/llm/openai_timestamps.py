# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Timestamp helpers for OpenAI-compatible chat-completion payloads."""

from __future__ import annotations

from copy import copy
from datetime import UTC, datetime
from typing import Any

_MILLISECOND_TIMESTAMP_MIN = 1_000_000_000_000
_MILLISECOND_TIMESTAMP_MAX = 10_000_000_000_000


def normalize_openai_created_timestamp(value: Any) -> Any:
    """Return seconds for OpenAI-compatible ``created`` timestamps.

    The OpenAI Chat Completions schema uses Unix seconds, but some compatible
    providers return Unix milliseconds.  Limit conversion to current 13-digit
    integer timestamps so ordinary 10-digit OpenAI seconds stay byte-for-byte
    on the fast path.
    """
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and _MILLISECOND_TIMESTAMP_MIN <= value < _MILLISECOND_TIMESTAMP_MAX
    ):
        return value / 1000
    return value


def openai_created_at_iso(value: Any) -> str:
    """Format an OpenAI-compatible ``created`` value as an ISO timestamp."""
    created = normalize_openai_created_timestamp(value)
    return datetime.fromtimestamp(created, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def normalize_openai_created_payload(payload: Any) -> Any:
    """Return a payload copy with normalized ``created`` when needed.

    Chrys reads ``payload.created`` internally, so it normalizes
    only the copied value that reaches the parser.  The original SDK object is
    left untouched for callers that still need to inspect the raw provider
    payload.
    """
    created = payload.created
    normalized = normalize_openai_created_timestamp(created)
    if normalized == created:
        return payload

    model_copy = getattr(payload, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"created": normalized})

    cloned = copy(payload)
    cloned.created = normalized
    return cloned
