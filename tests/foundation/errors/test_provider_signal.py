# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Provider signals: read from one node, never from stale context."""

from __future__ import annotations

from typing import Any, NoReturn

import httpx
import pytest

from chrys.foundation.errors import (
    ContinuationVerdictError,
    ErrorKind,
    ProviderResponseError,
    classify_error,
    invalidates_continuation_token,
)
from chrys.foundation.errors import classify as classify_module
from chrys.service.context.compaction.last_words import LastWordsGenerationError
from tests.foundation.errors._corpus import STALE_PROVIDER_ERRORS
from tests.support.provider_errors import openai_status, raised_from, raised_while_handling


class _BareStreamError(Exception):
    """The shape of a bare ``openai.APIError``: ``body`` and ``code``, no status."""

    def __init__(self, message: str, body: Any, code: str | None) -> None:
        super().__init__(message)
        self.body = body
        self.code = code


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int, response: httpx.Response | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response


async def test_stale_context_quota_error_does_not_classify_new_read_timeout() -> None:
    stale = await STALE_PROVIDER_ERRORS["429-insufficient-quota"]()

    def raise_fresh() -> NoReturn:
        raise httpx.ReadTimeout("timed out")

    result = classify_error(raised_while_handling(stale, raise_fresh))

    assert result.signal is None
    assert result.kind is ErrorKind.READ_TIMEOUT
    assert result.retryable is True


def test_signal_fields_come_from_one_node() -> None:
    inner = _StatusError("Error code: 429", 429)
    outer = raised_from(_BareStreamError("stream failed", {"message": "stream failed"}, None), inner)

    signal = classify_error(outer).signal

    assert signal is not None
    assert signal.source is outer
    assert (signal.status_code, signal.code, signal.message) == (None, None, "stream failed")


async def test_signal_reads_code_type_message_and_retry_after_from_the_sdk_error() -> None:
    exc = await openai_status(
        429, {"error": {"type": "insufficient_quota", "code": "insufficient_quota", "message": "Quota exceeded"}}
    )

    signal = classify_error(exc).signal

    assert signal is not None
    assert signal.source is exc
    assert (signal.status_code, signal.code, signal.error_type, signal.message) == (
        429,
        "insufficient_quota",
        "insufficient_quota",
        "Quota exceeded",
    )
    assert signal.retry_after is None


def test_retry_after_header_is_read_in_seconds() -> None:
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    response = httpx.Response(429, headers={"retry-after": "7"}, request=request)
    exc = _StatusError("Error code: 429", 429, response)

    signal = classify_error(exc).signal

    assert signal is not None
    assert signal.retry_after == 7.0


def test_system_exit_code_is_not_a_provider_signal() -> None:
    assert classify_error(SystemExit(2)).signal is None
    assert classify_error(_BareStreamError("no body", None, "insufficient_quota")).signal is None


@pytest.mark.parametrize(
    ("error", "kind", "retryable"),
    [
        (
            ProviderResponseError("stream_truncated", "no message_stop", retryable=True),
            ErrorKind.STREAM_TRUNCATED,
            True,
        ),
        (
            ProviderResponseError("overloaded_error", "busy", retryable=False, kind=ErrorKind.OVERLOADED),
            ErrorKind.OVERLOADED,
            False,
        ),
        (ProviderResponseError("vendor_specific", "?", retryable=True), ErrorKind.UNKNOWN, True),
        (ProviderResponseError("network_error", "?", retryable=True), ErrorKind.STREAM_TRUNCATED, True),
        (ProviderResponseError("insufficient_system_resource", "?", retryable=True), ErrorKind.OVERLOADED, True),
    ],
    ids=["truncated-retryable", "explicit-kind-not-retryable", "unknown-code", "network-error", "no-resources"],
)
def test_provider_response_error_states_kind_and_retry(
    error: ProviderResponseError, kind: ErrorKind, retryable: bool
) -> None:
    result = classify_error(error)

    assert (result.kind, result.retryable) == (kind, retryable)
    assert str(error) == f"{error.code}: {error.provider_message}"


def test_owner_terminal_veto_outranks_a_retryable_provider_response_error() -> None:
    inner = ProviderResponseError("stream_truncated", "no message_stop", retryable=True)

    result = classify_error(raised_from(LastWordsGenerationError("note failed"), inner))

    assert result.kind is ErrorKind.STREAM_TRUNCATED
    assert result.retryable is False


def test_continuation_token_invalidation_is_found_below_a_wrapper() -> None:
    inner = ProviderResponseError("stream_truncated", "?", retryable=True, invalidates_continuation_token=True)

    assert invalidates_continuation_token(inner) is True
    assert invalidates_continuation_token(raised_from(RuntimeError("wrapper"), inner)) is True
    assert invalidates_continuation_token(ProviderResponseError("stream_truncated", "?", retryable=True)) is False


def test_only_a_typed_continuation_verdict_invalidates_the_token() -> None:
    class Judged(ContinuationVerdictError):
        invalidates_continuation_token = True

    class LookAlike(Exception):
        invalidates_continuation_token = True

    assert invalidates_continuation_token(raised_from(RuntimeError("wrapper"), Judged())) is True
    assert invalidates_continuation_token(ContinuationVerdictError()) is False
    assert invalidates_continuation_token(LookAlike()) is False


def test_the_continuation_verdict_never_runs_the_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every failed wire call asks before the retry policy classifies; a
    # classifier failure here would replace the provider's own error.
    class Judged(ContinuationVerdictError):
        invalidates_continuation_token = True

    def refuse(_exc: BaseException) -> NoReturn:
        raise AssertionError("classified")

    monkeypatch.setattr(classify_module, "classify_error", refuse)

    assert invalidates_continuation_token(raised_from(RuntimeError("wrapper"), Judged())) is True
    assert invalidates_continuation_token(RuntimeError("plain")) is False
