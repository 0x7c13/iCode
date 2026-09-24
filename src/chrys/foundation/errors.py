# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared error formatting and retryability classification."""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

_PROVIDER_ERROR_DETAIL_LIMIT = 2000
_PROVIDER_ERROR_MAX_DEPTH = 20
_MISSING_PROVIDER_ERROR_VALUE = object()
_GENERIC_CONNECTION_ERROR_MESSAGES = frozenset({"connection error"})
_NON_RETRYABLE_CONNECTION_TYPE_NAMES = frozenset(
    {
        "InvalidURL",
        "LocalProtocolError",
        "TooManyRedirects",
        "UnsupportedProtocol",
    }
)
_NON_RETRYABLE_DNS_ERROR_CODES = frozenset(
    {
        socket.EAI_BADFLAGS,
        socket.EAI_FAIL,
        socket.EAI_FAMILY,
        socket.EAI_NONAME,
        socket.EAI_SERVICE,
        socket.EAI_SOCKTYPE,
    }
)
# Keep this allowlist deliberately narrow.  Python exposes OpenSSL ``reason``
# mnemonics, but neither Python, OpenSSL, nor the TLS RFCs assign retry policy
# to them.  Only failures that identify a stable local/endpoint configuration
# mismatch belong here.  In particular, peer ``*_ALERT_*`` reasons describe
# only the failed handshake and can be ambiguous (or vary across endpoints),
# so they must retain the SDK's normal retry behavior.
_DETERMINISTIC_TLS_CONFIGURATION_REASON_CODES = frozenset(
    {
        "CERTIFICATE_VERIFY_FAILED",
        "WRONG_VERSION_NUMBER",
        "NO_CIPHERS_AVAILABLE",
        "UNSUPPORTED_PROTOCOL",
        "NO_PROTOCOLS_AVAILABLE",
    }
)
_NON_RETRYABLE_PROXY_ERROR_MESSAGES = frozenset({"invalid username/password"})
_PROXY_AUTH_NEGOTIATION_PREFIX = "requested "
_PROXY_AUTH_NEGOTIATION_SEPARATOR = " from proxy server, but got "


@dataclass(frozen=True, slots=True)
class _ExceptionCauseRule:
    """Named matcher used by exception-chain disposition policies."""

    name: str
    matcher: Callable[[BaseException], bool]

    def matches(self, exc: BaseException) -> bool:
        return bool(self.matcher(exc))


EMPTY_EXCEPTION_MESSAGES = {
    "APITimeoutError": "Request timed out",
    "CloseError": "Connection close failed",
    "ConnectionAbortedError": "Connection aborted",
    "ConnectionError": "Connection error",
    "ConnectionRefusedError": "Connection refused",
    "ConnectionResetError": "Connection reset",
    "ConnectError": "Connection failed",
    "ConnectTimeout": "Connection timed out",
    "DecodingError": "Response decoding failed",
    "LocalProtocolError": "Local protocol error",
    "NetworkError": "Network error",
    "PoolTimeout": "Connection pool timed out",
    "ProtocolError": "Protocol error",
    "ProxyError": "Proxy error",
    "ReadError": "Read failed",
    "ReadTimeout": "Read timed out",
    "RemoteProtocolError": "Remote protocol error",
    "TimeoutError": "Operation timed out",
    "TimeoutException": "Request timed out",
    "TransportError": "Transport error",
    "UnsupportedProtocol": "Unsupported protocol",
    "WriteError": "Write failed",
    "WriteTimeout": "Write timed out",
}


def _fallback_exception_message(exc: BaseException) -> str:
    """Return a readable label for exceptions whose ``str(exc)`` is empty."""
    name = type(exc).__name__
    label = EMPTY_EXCEPTION_MESSAGES.get(name)
    if label is not None:
        return f"{label} ({name})"
    return name


def _iter_exception_chain(exc: BaseException, *, unwrap_single_groups: bool = False) -> Iterator[BaseException]:
    """Yield an exception and its explicit/implicit cause chain."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        yield cur
        seen.add(id(cur))
        if unwrap_single_groups and isinstance(cur, BaseExceptionGroup) and len(cur.exceptions) == 1:
            cur = cur.exceptions[0]
        else:
            cur = (
                cur.__cause__ if cur.__cause__ is not None else (None if cur.__suppress_context__ else cur.__context__)
            )


def _clean_exception_text(exc: BaseException) -> str:
    """Return ``str(exc)`` normalized for client wrapper prefixes."""
    msg = str(exc).strip()
    if msg.startswith("<class "):
        msg = msg.split("> ", 1)[-1]
    return msg


def _normalize_provider_error_detail(detail: str) -> str:
    """Return a compact one-line provider error detail."""
    return " ".join(detail.strip().split())


def _truncate_provider_error_detail(detail: str) -> str:
    if len(detail) <= _PROVIDER_ERROR_DETAIL_LIMIT:
        return detail
    return detail[: _PROVIDER_ERROR_DETAIL_LIMIT - len("...[truncated]")] + "...[truncated]"


def _stringify_provider_error_detail(value: Any) -> str:
    try:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError, ValueError, RecursionError:
            text = str(value)
    except Exception:
        text = type(value).__name__
    return _truncate_provider_error_detail(_normalize_provider_error_detail(text))


def _provider_mapping_value(value: Mapping[Any, Any], key: str) -> Any:
    try:
        return value[key]
    except Exception:
        return _MISSING_PROVIDER_ERROR_VALUE


def _extract_provider_error_detail(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> str:
    """Extract a provider error message from SDK ``body`` values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return _truncate_provider_error_detail(_normalize_provider_error_detail(value))
    if _depth >= _PROVIDER_ERROR_MAX_DEPTH:
        return _stringify_provider_error_detail(value)
    if isinstance(value, Mapping):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return _stringify_provider_error_detail(value)
        _seen.add(value_id)
        try:
            for key in ("message", "msg", "detail", "error", "errors"):
                nested = _provider_mapping_value(value, key)
                if nested is _MISSING_PROVIDER_ERROR_VALUE:
                    continue
                detail = _extract_provider_error_detail(nested, _seen=_seen, _depth=_depth + 1)
                if detail:
                    return detail
            return _stringify_provider_error_detail(value)
        finally:
            _seen.discard(value_id)
    if isinstance(value, list):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return _stringify_provider_error_detail(value)
        _seen.add(value_id)
        try:
            parts = [_extract_provider_error_detail(item, _seen=_seen, _depth=_depth + 1) for item in value]
        except Exception:
            return _stringify_provider_error_detail(value)
        finally:
            _seen.discard(value_id)
        return _truncate_provider_error_detail("; ".join(part for part in parts if part))
    return _stringify_provider_error_detail(value)


def _provider_response_detail(exc: BaseException) -> str:
    """Best-effort fallback for SDKs that expose response text separately."""
    response = getattr(exc, "response", None)
    if response is None:
        return ""
    try:
        text = response.text
    except Exception:
        return ""
    if not isinstance(text, str):
        return ""
    return _extract_provider_error_detail(text)


def _provider_status_error_message(exc: BaseException) -> str:
    """Return a concise status-code message with SDK body details when present."""
    status = getattr(exc, "status_code", None)
    if status is None:
        return ""

    detail = _extract_provider_error_detail(getattr(exc, "body", None)) or _provider_response_detail(exc)
    if not detail:
        return ""

    prefix = f"Error code: {status}"
    raw = _clean_exception_text(exc)
    if raw == detail or raw == f"{prefix} - {detail}":
        return raw
    return f"{prefix} - {detail}"


def _is_generic_connection_error(exc: BaseException, message: str) -> bool:
    """Return whether an SDK connection wrapper has no useful detail."""
    return (
        type(exc).__name__ == "APIConnectionError"
        and message.casefold().rstrip(".") in _GENERIC_CONNECTION_ERROR_MESSAGES
    )


def _iter_explicit_exception_causes(exc: BaseException) -> Iterator[BaseException]:
    """Yield only explicitly chained causes below *exc*, guarding cycles."""
    seen = {id(exc)}
    current = exc.__cause__
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__


def _generic_connection_root_detail(exc: BaseException) -> str:
    """Extract the deepest useful detail from an SDK wrapper's explicit causes."""
    fallback = ""
    for candidate in reversed(list(_iter_explicit_exception_causes(exc))):
        message = _clean_exception_text(candidate)
        if message and message.casefold().rstrip(".") not in _GENERIC_CONNECTION_ERROR_MESSAGES:
            return _truncate_provider_error_detail(_normalize_provider_error_detail(message))
        if not fallback:
            fallback = _fallback_exception_message(candidate)
    return fallback


def _matches_non_retryable_connection_type(exc: BaseException) -> bool:
    return type(exc).__name__ in _NON_RETRYABLE_CONNECTION_TYPE_NAMES


def _matches_non_retryable_dns_error(exc: BaseException) -> bool:
    return isinstance(exc, socket.gaierror) and exc.errno in _NON_RETRYABLE_DNS_ERROR_CODES


def _matches_non_retryable_proxy_error(exc: BaseException) -> bool:
    """Match only explicit proxy-authentication failures."""
    if type(exc).__name__ != "ProxyError":
        return False
    message = _clean_exception_text(exc).casefold().strip().rstrip(".")
    status_token = message.partition(" ")[0]
    return (
        status_token == "407"
        or message in _NON_RETRYABLE_PROXY_ERROR_MESSAGES
        or (message.startswith(_PROXY_AUTH_NEGOTIATION_PREFIX) and _PROXY_AUTH_NEGOTIATION_SEPARATOR in message)
    )


def _ssl_reason_code(exc: ssl.SSLError) -> str:
    """Return OpenSSL's stable reason code without depending on prose text."""
    reason = getattr(exc, "reason", None)
    if isinstance(reason, str) and reason:
        return reason.upper()

    message = _clean_exception_text(exc).upper()
    prefix = "[SSL: "
    start = message.find(prefix)
    if start < 0:
        return ""
    start += len(prefix)
    end = message.find("]", start)
    return message[start:end] if end >= 0 else ""


def _matches_non_retryable_tls_error(exc: BaseException) -> bool:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return True
    return isinstance(exc, ssl.SSLError) and _ssl_reason_code(exc) in _DETERMINISTIC_TLS_CONFIGURATION_REASON_CODES


_NON_RETRYABLE_CONNECTION_CAUSE_RULES = (
    _ExceptionCauseRule("deterministic_transport_type", _matches_non_retryable_connection_type),
    _ExceptionCauseRule("dns_configuration", _matches_non_retryable_dns_error),
    _ExceptionCauseRule("proxy_authentication", _matches_non_retryable_proxy_error),
    _ExceptionCauseRule("tls_configuration", _matches_non_retryable_tls_error),
)


def _non_retryable_connection_cause_rule(exc: BaseException) -> _ExceptionCauseRule | None:
    """Return the first deterministic connection rule matching *exc*."""
    return next((rule for rule in _NON_RETRYABLE_CONNECTION_CAUSE_RULES if rule.matches(exc)), None)


def _has_http_response_status(exc: BaseException) -> bool:
    """Return whether *exc* proves that its request received an HTTP response."""
    if getattr(exc, "status_code", None) is not None:
        return True
    response = getattr(exc, "response", None)
    return response is not None and getattr(response, "status_code", None) is not None


def _httpcore_transport_argument_cause(exc: BaseException) -> BaseException | None:
    """Recover the TLS cause retained in pinned httpcore's wrapper argument."""
    exc_type = type(exc)
    if exc_type.__module__.partition(".")[0] != "httpcore" or exc_type.__name__ != "ConnectError":
        return None
    if len(exc.args) != 1 or not isinstance(exc.args[0], BaseException):
        return None
    return exc.args[0]


def _iter_deterministic_connection_chain(exc: BaseException) -> Iterator[BaseException]:
    """Follow explicit causes plus the one known httpcore transport wrapper."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ if current.__cause__ is not None else _httpcore_transport_argument_cause(current)


def _contains_deterministic_connection_error(chain: Iterable[BaseException]) -> bool:
    """Classify a connection-cause chain without trusting implicit context."""
    for exc in chain:
        if _has_http_response_status(exc):
            # A response proves that this request connected successfully.
            # Deeper entries belong to response processing or to an older
            # implicit ``__context__`` and cannot be this request's connection
            # root cause.
            return False
        if _non_retryable_connection_cause_rule(exc) is not None:
            return True
    return False


def is_deterministic_connection_error(e: BaseException) -> bool:
    """Return whether an exception chain contains a deterministic connection failure."""
    return _contains_deterministic_connection_error(_iter_deterministic_connection_chain(e))


def clean_error_message(e: BaseException) -> str:
    """Extract a clean error message, stripping framework class path prefixes."""
    chain = list(_iter_exception_chain(e, unwrap_single_groups=True))
    for candidate in chain[1:]:
        if message := _provider_status_error_message(candidate):
            return message
    display_index = 1 if len(chain) > 1 else 0
    display_exc = chain[display_index]
    while isinstance(display_exc, BaseExceptionGroup) and len(display_exc.exceptions) == 1:
        if display_index + 1 >= len(chain):
            break
        display_index += 1
        display_exc = chain[display_index]
    if isinstance(display_exc, asyncio.CancelledError) and display_index > 0:
        # A wrapper may explain why its operation was cancelled; a bare
        # cancellation usually has no actionable detail. Prefer that wrapper
        # across callers, including MCP initialization and teardown failures.
        display_exc = chain[display_index - 1]
    if message := _provider_status_error_message(display_exc):
        return message
    message = _clean_exception_text(display_exc) or _fallback_exception_message(display_exc)
    if _is_generic_connection_error(display_exc, message) and (detail := _generic_connection_root_detail(display_exc)):
        return f"{message.rstrip('.')}: {detail}"
    return message


# HTTP status codes that are transient and worth retrying after SDK
# retries are exhausted.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504, 529})

# SDK exception class names (checked by name to avoid hard imports).
RETRYABLE_TYPE_NAMES = frozenset(
    {
        # OpenAI SDK
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        # Bare ``openai.APIError`` is only raised in ``openai/_streaming.py``
        # when the upstream emits an in-band SSE ``error`` event mid-stream.
        # ``type(exc).__name__`` returns the most-derived class, so this name
        # match catches only the base class — subclasses (APIStatusError,
        # BadRequestError, etc.) keep their existing classification.
        "APIError",
        # httpx / httpcore transport failures.  These often carry no
        # message when wrapping a bare OS/timeout exception on Windows
        # (e.g. ``ReadTimeout(TimeoutError())``), so string fallback alone
        # cannot classify them.
        "CloseError",
        "ConnectError",
        "ConnectTimeout",
        "DecodingError",
        "NetworkError",
        "PoolTimeout",
        "ProxyError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutException",
        "WriteError",
        "WriteTimeout",
        # Anthropic SDK
        "OverloadedError",
        "ServiceUnavailableError",
    }
)

# Fallback string patterns for transport-level errors that may not carry
# a typed exception (e.g. httpx errors surfacing as generic exceptions
# through framework wrapping).
RETRYABLE_PHRASES = (
    "peer closed connection",
    "incomplete chunked read",
    "remoteprotocolerror",
    "connection reset",
    "connection closed",
    "server disconnected",
    "socket hang up",
)


def is_retryable(e: BaseException) -> bool:
    """Check if an exception is a transient error worth retrying.

    Deterministic non-retryable root causes take precedence over the
    three-layer transient-error strategy:

    1. Exception type — Python builtins and SDK types (via class name)
    2. HTTP status code — ``status_code`` attribute on SDK exceptions
    3. String matching — fallback for transport-level errors
    """
    chain = tuple(_iter_exception_chain(e))
    non_retryable_type_names = {"LastWordsGenerationError", "TerminalResponseValidationError"}
    if any(type(exc).__name__ in non_retryable_type_names for exc in chain):
        # These failures have already exhausted or terminally concluded
        # their owning component's policy, so an outer retry would replay
        # a larger unit of work without making the failure recoverable.
        return False
    if is_deterministic_connection_error(e):
        # SDKs wrap deterministic transport, protocol, DNS, and TLS failures
        # in retryable APIConnectionError/ConnectError types. The root cause
        # must win: repeating the same request cannot fix it.
        return False
    for exc in chain:
        # Python built-in transient exceptions
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        # SDK exception types (by name to avoid hard imports)
        if type(exc).__name__ in RETRYABLE_TYPE_NAMES:
            return True
        # HTTP status code (works for both OpenAI and Anthropic SDKs)
        status = getattr(exc, "status_code", None)
        if status is not None and status in RETRYABLE_STATUS_CODES:
            return True
    # Fallback: string matching for transport-level errors
    for exc in chain:
        error_msg = _clean_exception_text(exc).lower()
        if any(phrase in error_msg for phrase in RETRYABLE_PHRASES):
            return True
    return False
