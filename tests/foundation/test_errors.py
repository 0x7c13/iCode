# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for :mod:`chrys.foundation.errors`."""

from __future__ import annotations

import asyncio
import ssl
from typing import Any

import pytest

from chrys.foundation.errors import clean_error_message


def test_single_member_exception_groups_expose_the_leaf() -> None:
    error = ExceptionGroup("task group", [ExceptionGroup("nested", [ValueError("invalid UTF-8")])])
    error.__suppress_context__ = True
    assert clean_error_message(error) == "invalid UTF-8"
    assert clean_error_message(_make_chained(error)) == "invalid UTF-8"


def test_multiple_member_exception_group_is_not_arbitrarily_reduced() -> None:
    error = ExceptionGroup("two failures", [ValueError("first"), ValueError("second")])
    assert clean_error_message(_make_chained(error)) == str(error)


@pytest.mark.parametrize("message", ["MCP initialize failed: invalid UTF-8", "Background export stopped: disk full"])
@pytest.mark.parametrize("cancellation_detail", ["", "Cancelled via cancel scope"])
def test_cancelled_inner_keeps_diagnostic_wrapper(message: str, cancellation_detail: str) -> None:
    error = RuntimeError(message)
    error.__cause__ = asyncio.CancelledError(cancellation_detail)
    assert clean_error_message(error) == str(error)


def test_bare_nested_cancellation_keeps_the_generic_diagnostic() -> None:
    inner = asyncio.CancelledError()
    outer = asyncio.CancelledError()
    outer.__cause__ = inner
    assert clean_error_message(outer) == clean_error_message(inner)
    assert outer.__cause__ is inner


def _make_status_error(
    message: str,
    *,
    body: object,
    status_code: int = 400,
    response_text: str | None = None,
) -> Exception:
    cls = type("BadRequestError", (Exception,), {})
    exc = cls(message)
    exc.status_code = status_code  # type: ignore[attr-defined]
    exc.body = body  # type: ignore[attr-defined]
    if response_text is not None:
        exc.response = type("Response", (), {"text": response_text})()  # type: ignore[attr-defined]
    return exc


def _make_chained(cause: Exception, wrapper_msg: str = "service failed") -> Exception:
    try:
        raise RuntimeError(wrapper_msg) from cause
    except RuntimeError as exc:
        return exc


def _make_named_chained(cls_name: str, message: str, cause: Exception) -> Exception:
    cls = type(cls_name, (Exception,), {})
    try:
        raise cls(message) from cause
    except Exception as exc:
        return exc


@pytest.mark.parametrize(
    ("cls_name", "expected"),
    [
        ("ConnectError", "Connection failed (ConnectError)"),
        ("ConnectionResetError", "Connection reset (ConnectionResetError)"),
        ("ReadError", "Read failed (ReadError)"),
        ("ReadTimeout", "Read timed out (ReadTimeout)"),
        ("RemoteProtocolError", "Remote protocol error (RemoteProtocolError)"),
        ("LocalProtocolError", "Local protocol error (LocalProtocolError)"),
        ("ProxyError", "Proxy error (ProxyError)"),
    ],
)
def test_clean_error_message_falls_back_for_empty_http_transport_errors(cls_name: str, expected: str) -> None:
    cls = type(cls_name, (Exception,), {})
    exc = cls(TimeoutError())
    assert str(exc) == ""
    assert clean_error_message(exc) == expected


def test_clean_error_message_uses_empty_cause_type_over_wrapper_noise() -> None:
    class ReadTimeout(Exception):
        pass

    try:
        raise RuntimeError("service failed to complete the prompt") from ReadTimeout(TimeoutError())
    except RuntimeError as exc:
        assert clean_error_message(exc) == "Read timed out (ReadTimeout)"


def test_clean_error_message_uses_empty_context_type_over_wrapper_noise() -> None:
    class RemoteProtocolError(Exception):
        pass

    try:
        raise RemoteProtocolError(TimeoutError())
    except RemoteProtocolError:
        try:
            raise RuntimeError("service failed to complete the prompt")
        except RuntimeError as exc:
            assert clean_error_message(exc) == "Remote protocol error (RemoteProtocolError)"


def test_clean_error_message_exposes_tls_cause_hidden_by_sdk_connection_error() -> None:
    tls_error = ssl.SSLCertVerificationError(
        1,
        "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate (_ssl.c:1016)",
    )
    transport_error = _make_named_chained("ConnectError", str(tls_error), tls_error)
    sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)
    wrapped = _make_chained(sdk_error, "service failed to complete the prompt: Connection error.")

    assert clean_error_message(wrapped) == (
        "Connection error: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
        "self-signed certificate (_ssl.c:1016)"
    )


def test_clean_error_message_preserves_generic_sdk_connection_error_without_cause() -> None:
    sdk_error = type("APIConnectionError", (Exception,), {})("Connection error.")

    assert clean_error_message(sdk_error) == "Connection error."


def test_clean_error_message_exposes_invalid_url_hidden_by_sdk_connection_error() -> None:
    url_error = type("InvalidURL", (Exception,), {})("Invalid port: 'not-a-port'")
    sdk_error = _make_named_chained("APIConnectionError", "Connection error.", url_error)
    wrapped = _make_chained(sdk_error, "service failed to complete the prompt: Connection error.")

    assert clean_error_message(wrapped) == "Connection error: Invalid port: 'not-a-port'"


def test_clean_error_message_ignores_stale_context_below_transport_error() -> None:
    connect_error_cls = type("ConnectError", (Exception,), {})
    stale_transport_error: Exception | None = None
    try:
        raise ValueError("unrelated prior failure")
    except ValueError:
        try:
            raise connect_error_cls("Connection refused")
        except Exception as transport_error:
            stale_transport_error = transport_error

    assert stale_transport_error is not None
    assert isinstance(stale_transport_error.__context__, ValueError)
    sdk_error = _make_named_chained("APIConnectionError", "Connection error.", stale_transport_error)
    wrapped = _make_chained(sdk_error, "service failed to complete the prompt: Connection error.")

    assert clean_error_message(wrapped) == "Connection error: Connection refused"


def test_clean_error_message_respects_suppressed_context() -> None:
    try:
        raise TimeoutError
    except TimeoutError:
        try:
            raise RuntimeError("not retryable") from None
        except RuntimeError as exc:
            assert exc.__suppress_context__ is True
            assert clean_error_message(exc) == "not retryable"


@pytest.mark.parametrize(
    ("body", "expected_detail"),
    [
        ({"message": "prompt is too long"}, "prompt is too long"),
        ({"error": {"message": "invalid tool schema"}}, "invalid tool schema"),
        ({"detail": [{"msg": "model does not support tools"}]}, "model does not support tools"),
    ],
)
def test_clean_error_message_includes_provider_status_body(body: Any, expected_detail: str) -> None:
    exc = _make_status_error("Error code: 400", body=body)

    assert clean_error_message(exc) == f"Error code: 400 - {expected_detail}"


def test_clean_error_message_uses_provider_body_from_wrapped_status_error() -> None:
    cause = _make_status_error("Error code: 400", body={"message": "chat template rejected tool call"})
    wrapped = _make_chained(
        cause,
        "<class 'chrys.service.llm.openai_chat_completion.RawOpenAIChatCompletionClient'> service failed to complete the prompt: "
        "Error code: 400",
    )

    assert clean_error_message(wrapped) == "Error code: 400 - chat template rejected tool call"


def test_clean_error_message_uses_response_text_when_status_body_is_empty() -> None:
    exc = _make_status_error("Error code: 400", body=None, response_text="plain gateway rejection")

    assert clean_error_message(exc) == "Error code: 400 - plain gateway rejection"


def test_clean_error_message_preserves_status_code_when_provider_body_is_empty() -> None:
    exc = _make_status_error("Error code: 400", body="")

    assert clean_error_message(exc) == "Error code: 400"


def test_clean_error_message_handles_circular_provider_body() -> None:
    body: dict[str, object] = {}
    body["error"] = body
    exc = _make_status_error("Error code: 400", body=body)

    assert clean_error_message(exc) == "Error code: 400 - {'error': {...}}"


def test_clean_error_message_keeps_cause_text_over_outer_status_metadata() -> None:
    wrapped = _make_chained(ValueError("specific provider failure"), "Error code: 400")
    wrapped.status_code = 400  # type: ignore[attr-defined]
    wrapped.body = {"message": "less specific wrapper body"}  # type: ignore[attr-defined]

    assert clean_error_message(wrapped) == "specific provider failure"
