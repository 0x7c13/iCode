# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for is_retryable — root-cause exclusions plus transient classification.

Layer 0: Deterministic connection root cause (takes precedence over wrappers)
Layer 1: Exception type (Python builtins + SDK class names)
Layer 2: HTTP status code (via status_code attribute)
Layer 3: String matching (transport-level error phrases)
"""

from __future__ import annotations

import socket
import ssl

import pytest

from chrys.foundation.errors import is_deterministic_connection_error, is_retryable
from chrys.service.agent_middleware.response_validation import TerminalResponseValidationError
from chrys.service.context.compaction.last_words import LastWordsGenerationError

# ── helpers ──────────────────────────────────────────────────────────


def _make_sdk_exception(cls_name: str, message: str, *, status_code: int | None = None) -> Exception:
    """Create a fake SDK exception with the given class name and optional status_code."""
    cls = type(cls_name, (Exception,), {})
    exc = cls(message)
    if status_code is not None:
        exc.status_code = status_code  # type: ignore[attr-defined]
    return exc


def _make_chained(cause: Exception, wrapper_msg: str = "service failed") -> Exception:
    """Wrap *cause* in a generic exception via ``raise ... from cause``."""
    try:
        raise Exception(wrapper_msg) from cause
    except Exception as e:
        return e


def _make_named_chained(cls_name: str, message: str, cause: Exception) -> Exception:
    cls = type(cls_name, (Exception,), {})
    try:
        raise cls(message) from cause
    except Exception as exc:
        return exc


# ── Layer 1: exception types ────────────────────────────────────────


class TestRetryableByType:
    """Python built-in and SDK exception types detected via isinstance / class name."""

    def test_connection_error(self) -> None:
        assert is_retryable(ConnectionError("Connection refused"))

    def test_timeout_error(self) -> None:
        assert is_retryable(TimeoutError("deadline exceeded"))

    @pytest.mark.parametrize(
        "cls_name",
        [
            "CloseError",
            "ConnectError",
            "DecodingError",
            "NetworkError",
            "ProxyError",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
            "WriteError",
        ],
    )
    def test_http_transport_error_with_empty_message_is_retryable(self, cls_name: str) -> None:
        cls = type(cls_name, (Exception,), {})
        exc = cls(TimeoutError())
        assert str(exc) == ""
        assert is_retryable(exc)

    @pytest.mark.parametrize(
        "cls_name",
        [
            "LocalProtocolError",
            "UnsupportedProtocol",
        ],
    )
    def test_non_transient_http_transport_error_with_empty_message_not_retryable(self, cls_name: str) -> None:
        cls = type(cls_name, (Exception,), {})
        exc = cls(TimeoutError())
        assert str(exc) == ""
        assert not is_retryable(exc)

    def test_os_error_subclass(self) -> None:
        # ConnectionResetError is ConnectionError subclass
        assert is_retryable(ConnectionResetError("reset by peer"))

    @pytest.mark.parametrize(
        "cls_name",
        [
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
            "OverloadedError",
            "ServiceUnavailableError",
        ],
    )
    def test_sdk_exception_by_class_name(self, cls_name: str) -> None:
        exc = _make_sdk_exception(cls_name, "some error")
        assert is_retryable(exc)

    def test_sdk_exception_wrapped_in_cause_chain(self) -> None:
        """is_retryable follows __cause__ so wrapping an SDK error preserves retryability."""
        cause = _make_sdk_exception("RateLimitError", "Error code: 429")
        wrapped = _make_chained(cause)
        assert is_retryable(wrapped)

    def test_ssl_certificate_verification_root_cause_is_not_retryable(self) -> None:
        tls_error = ssl.SSLCertVerificationError(1, "certificate verify failed: self-signed certificate")
        transport_error = _make_named_chained(
            "ConnectError",
            "certificate verify failed: self-signed certificate",
            tls_error,
        )
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)
        wrapped = _make_chained(sdk_error, "Chat Completions request failed: Connection error.")

        assert not is_retryable(wrapped)

    @pytest.mark.parametrize(
        "cls_name",
        [
            "InvalidURL",
            "LocalProtocolError",
            "TooManyRedirects",
            "UnsupportedProtocol",
        ],
    )
    def test_sdk_connection_wrapper_does_not_mask_deterministic_transport_error(self, cls_name: str) -> None:
        transport_error = _make_sdk_exception(cls_name, "invalid connection configuration")
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)

        assert not is_retryable(sdk_error)

    @pytest.mark.parametrize("cls_name", ["CloseError", "DecodingError", "ProxyError"])
    def test_sdk_connection_wrapper_preserves_transient_transport_retry(self, cls_name: str) -> None:
        transport_error = _make_sdk_exception(cls_name, "temporary transport failure")
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)

        assert is_retryable(sdk_error)

    @pytest.mark.parametrize(
        "message",
        [
            "407 Proxy Authentication Required",
            "Invalid username/password",
            "Requested USERNAME/PASSWORD from proxy server, but got NO ACCEPTABLE METHODS.",
            "Requested NO AUTHENTICATION REQUIRED from proxy server, but got USERNAME/PASSWORD.",
            "Requested NO AUTHENTICATION REQUIRED from proxy server, but got NO ACCEPTABLE METHODS.",
        ],
    )
    def test_proxy_authentication_error_is_not_retryable(self, message: str) -> None:
        proxy_error = _make_sdk_exception("ProxyError", message)
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", proxy_error)

        assert is_deterministic_connection_error(sdk_error)
        assert not is_retryable(sdk_error)

    def test_httpcore_tls_argument_cause_is_not_retryable(self) -> None:
        """Pinned httpcore may retain the TLS error only as a wrapper argument."""
        import httpcore

        tls_error = ssl.SSLCertVerificationError(1, "certificate verify failed: self-signed certificate")
        core_error = httpcore.ConnectError(tls_error)
        core_error.__context__ = tls_error
        core_error.__suppress_context__ = True
        transport_error = _make_named_chained("ConnectError", str(core_error), core_error)
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)

        assert core_error.__cause__ is None
        assert is_deterministic_connection_error(sdk_error)
        assert not is_retryable(sdk_error)

    @pytest.mark.parametrize(
        "message",
        [
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:1000)",
            "[SSL: WRONG_VERSION_NUMBER] wrong version number (_ssl.c:1000)",
            "[SSL: NO_CIPHERS_AVAILABLE] no ciphers available (_ssl.c:1000)",
            "[SSL: UNSUPPORTED_PROTOCOL] unsupported protocol (_ssl.c:1000)",
            "[SSL: NO_PROTOCOLS_AVAILABLE] no protocols available (_ssl.c:1000)",
        ],
    )
    def test_deterministic_tls_configuration_error_is_not_retryable(self, message: str) -> None:
        tls_error = ssl.SSLError(1, message)
        transport_error = _make_named_chained("ConnectError", message, tls_error)
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)

        assert not is_retryable(sdk_error)

    @pytest.mark.parametrize(
        "message",
        [
            "[SSL: TLSV1_ALERT_PROTOCOL_VERSION] tlsv1 alert protocol version (_ssl.c:1000)",
            "[SSL: TLSV13_ALERT_CERTIFICATE_REQUIRED] tlsv13 alert certificate required (_ssl.c:1000)",
            "[SSL: TLSV1_ALERT_UNKNOWN_CA] tlsv1 alert unknown ca (_ssl.c:1000)",
            "[SSL: TLSV1_ALERT_INSUFFICIENT_SECURITY] tlsv1 alert insufficient security (_ssl.c:1000)",
            "[SSL: SSLV3_ALERT_HANDSHAKE_FAILURE] sslv3 alert handshake failure (_ssl.c:1000)",
        ],
    )
    def test_peer_tls_alert_remains_retryable(self, message: str) -> None:
        tls_error = ssl.SSLError(1, message)
        transport_error = _make_named_chained("ConnectError", message, tls_error)
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)

        assert is_retryable(sdk_error)

    def test_transient_tls_error_remains_retryable(self) -> None:
        tls_error = ssl.SSLError(1, "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")
        transport_error = _make_named_chained("ConnectError", str(tls_error), tls_error)
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)

        assert is_retryable(sdk_error)

    def test_structured_tls_reason_code_does_not_depend_on_message_language(self) -> None:
        tls_error = ssl.SSLError(1, "localized TLS failure text")
        tls_error.reason = "WRONG_VERSION_NUMBER"
        transport_error = _make_named_chained("ConnectError", str(tls_error), tls_error)
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)

        assert not is_retryable(sdk_error)

    def test_unknown_dns_host_is_not_retryable(self) -> None:
        dns_error = socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        transport_error = _make_named_chained("ConnectError", str(dns_error), dns_error)
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)

        assert not is_retryable(sdk_error)

    def test_non_recoverable_dns_failure_is_not_retryable(self) -> None:
        dns_error = socket.gaierror(socket.EAI_FAIL, "Non-recoverable failure in name resolution")
        transport_error = _make_named_chained("ConnectError", str(dns_error), dns_error)
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)

        assert not is_retryable(sdk_error)

    def test_temporary_dns_failure_remains_retryable(self) -> None:
        dns_error = socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
        transport_error = _make_named_chained("ConnectError", str(dns_error), dns_error)
        sdk_error = _make_named_chained("APIConnectionError", "Connection error.", transport_error)

        assert is_retryable(sdk_error)

    def test_http_status_stops_deterministic_scan_before_polluted_context(self) -> None:
        tls_error = ssl.SSLCertVerificationError(1, "certificate verify failed")
        rate_limit_cls = type("RateLimitError", (Exception,), {})
        polluted_error: Exception | None = None
        try:
            raise tls_error
        except ssl.SSLCertVerificationError:
            try:
                raise rate_limit_cls("rate limited")
            except Exception as rate_limit_error:
                rate_limit_error.status_code = 429  # type: ignore[attr-defined]
                polluted_error = rate_limit_error

        assert polluted_error is not None
        assert polluted_error.__context__ is tls_error
        assert not is_deterministic_connection_error(polluted_error)
        assert is_retryable(polluted_error)

    def test_httpx_response_status_stops_deterministic_scan_before_polluted_context(self) -> None:
        response = type("Response", (), {"status_code": 429})()
        tls_error = ssl.SSLCertVerificationError(1, "certificate verify failed")
        status_error_cls = type("HTTPStatusError", (Exception,), {})
        polluted_error: Exception | None = None
        try:
            raise tls_error
        except ssl.SSLCertVerificationError:
            try:
                raise status_error_cls("429 Too Many Requests")
            except Exception as status_error:
                status_error.response = response  # type: ignore[attr-defined]
                polluted_error = status_error

        assert polluted_error is not None
        assert polluted_error.__context__ is tls_error
        assert not is_deterministic_connection_error(polluted_error)

    def test_transient_connection_error_ignores_deterministic_implicit_context(self) -> None:
        tls_error = ssl.SSLCertVerificationError(1, "certificate verify failed")
        connect_error_cls = type("ConnectError", (Exception,), {})
        polluted_error: Exception | None = None
        try:
            raise tls_error
        except ssl.SSLCertVerificationError:
            try:
                raise connect_error_cls("connection refused")
            except Exception as connect_error:
                polluted_error = connect_error

        assert polluted_error is not None
        assert polluted_error.__cause__ is None
        assert polluted_error.__context__ is tls_error
        assert not is_deterministic_connection_error(polluted_error)
        assert is_retryable(polluted_error)

    def test_sdk_exception_wrapped_in_context_chain(self) -> None:
        """Bare ``raise`` inside an except block stores transport errors on __context__."""
        cause = _make_sdk_exception("ReadTimeout", "")
        try:
            raise cause
        except Exception:
            try:
                raise RuntimeError("service failed")
            except RuntimeError as wrapped:
                assert wrapped.__context__ is cause
                assert is_retryable(wrapped)

    def test_suppressed_context_is_not_retryable(self) -> None:
        """``raise ... from None`` intentionally hides the previous transient error."""
        try:
            raise TimeoutError
        except TimeoutError:
            try:
                raise RuntimeError("not retryable") from None
            except RuntimeError as wrapped:
                assert wrapped.__suppress_context__ is True
                assert not is_retryable(wrapped)

    def test_unknown_exception_type_not_retryable(self) -> None:
        assert not is_retryable(ValueError("bad input"))

    def test_runtime_error_not_retryable(self) -> None:
        assert not is_retryable(RuntimeError("something broke"))

    def test_key_error_not_retryable(self) -> None:
        assert not is_retryable(KeyError("missing"))

    def test_last_words_generation_error_owns_its_retry_budget(self) -> None:
        try:
            try:
                raise ConnectionError("connection dropped")
            except ConnectionError as exc:
                raise LastWordsGenerationError("compaction failed") from exc
        except LastWordsGenerationError as wrapped:
            assert not is_retryable(wrapped)


# ── Layer 2: HTTP status codes ──────────────────────────────────────


class TestRetryableByStatusCode:
    """Exceptions with a status_code attribute (SDK status errors)."""

    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504, 529])
    def test_retryable_status_code(self, code: int) -> None:
        exc = _make_sdk_exception("SomeError", f"Error code: {code}", status_code=code)
        assert is_retryable(exc)

    @pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 422])
    def test_non_retryable_status_code(self, code: int) -> None:
        exc = _make_sdk_exception("SomeError", f"Error code: {code}", status_code=code)
        assert not is_retryable(exc)

    def test_status_code_on_cause(self) -> None:
        """Status code on __cause__ should be detected."""
        cause = _make_sdk_exception("APIStatusError", "Error code: 429", status_code=429)
        wrapped = _make_chained(cause)
        assert is_retryable(wrapped)


# ── Layer 3: string matching (transport-level fallback) ─────────────


class TestRetryableByStringMatch:
    """Fallback string matching for transport errors that lose type info."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "peer closed connection without sending complete message body",
            "incomplete chunked read",
            "RemoteProtocolError: peer closed connection",
            "connection reset by peer",
            "connection closed while sending",
            "server disconnected",
            "socket hang up",
        ],
    )
    def test_transport_error_phrases(self, phrase: str) -> None:
        # Use a generic Exception so only string matching triggers
        exc = Exception(phrase)
        assert is_retryable(exc)

    def test_case_insensitive(self) -> None:
        exc = Exception("PEER CLOSED CONNECTION")
        assert is_retryable(exc)

    def test_non_matching_string(self) -> None:
        exc = Exception("invalid JSON in request body")
        assert not is_retryable(exc)

    @pytest.mark.parametrize("reason", ["server disconnected", "connection reset"])
    def test_terminal_response_validation_reason_never_reenters_retry_lane(self, reason: str) -> None:
        assert not is_retryable(TerminalResponseValidationError(reason))

    def test_wrapped_terminal_response_validation_error_is_not_retryable(self) -> None:
        terminal = TerminalResponseValidationError("server disconnected")
        assert not is_retryable(_make_chained(terminal, "connection reset"))


# ── Real-world error messages ───────────────────────────────────────


class TestRealWorldErrors:
    """Reproduce actual error messages seen from OpenAI/Anthropic SDKs."""

    def test_openai_connection_error(self) -> None:
        exc = _make_sdk_exception("APIConnectionError", "Connection error.")
        assert is_retryable(exc)

    def test_openai_timeout(self) -> None:
        exc = _make_sdk_exception("APITimeoutError", "Request timed out.")
        assert is_retryable(exc)

    def test_openai_rate_limit(self) -> None:
        exc = _make_sdk_exception(
            "RateLimitError",
            "Error code: 429 - {'error': {'message': 'Rate limit exceeded'}}",
            status_code=429,
        )
        assert is_retryable(exc)

    def test_openai_internal_server_error(self) -> None:
        exc = _make_sdk_exception(
            "InternalServerError",
            "Error code: 500 - {'error': {'message': 'Internal server error'}}",
            status_code=500,
        )
        assert is_retryable(exc)

    def test_anthropic_overloaded(self) -> None:
        exc = _make_sdk_exception(
            "OverloadedError",
            "Error code: 529 - {'error': {'message': 'Overloaded'}}",
            status_code=529,
        )
        assert is_retryable(exc)

    def test_anthropic_timeout(self) -> None:
        exc = _make_sdk_exception(
            "APITimeoutError",
            "Request timed out or interrupted. This could be due to a network timeout.",
        )
        assert is_retryable(exc)

    def test_client_wrapped_rate_limit(self) -> None:
        """Chrys chat clients wrap OpenAI errors as ChatClientException."""
        cause = _make_sdk_exception("RateLimitError", "Error code: 429", status_code=429)
        wrapper = _make_chained(cause, "Chat Completions request failed: Error code: 429")
        assert is_retryable(wrapper)

    def test_client_wrapped_connection_error(self) -> None:
        cause = _make_sdk_exception("APIConnectionError", "Connection error.")
        wrapper = _make_chained(cause, "Chat Completions request failed: Connection error.")
        assert is_retryable(wrapper)

    def test_openai_streaming_sse_error(self) -> None:
        """Bare openai.APIError from an SSE ``error`` event mid-stream is retryable."""
        exc = _make_sdk_exception("APIError", "'list' object has no attribute 'lower'")
        assert is_retryable(exc)

    def test_bad_request_not_retryable(self) -> None:
        """400 errors should NOT be retried."""
        exc = _make_sdk_exception(
            "BadRequestError",
            "Error code: 400 - {'error': {'message': 'Invalid request'}}",
            status_code=400,
        )
        assert not is_retryable(exc)

    def test_auth_error_not_retryable(self) -> None:
        exc = _make_sdk_exception(
            "AuthenticationError",
            "Error code: 401 - {'error': {'message': 'Invalid API key'}}",
            status_code=401,
        )
        assert not is_retryable(exc)
