# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Raw English error text for the model, logs, ACP and headless output."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import socket
import ssl
import sys
from collections.abc import Iterator, Mapping
from typing import Any

from ._text import _clean_exception_text
from ._walk import iter_full_chain
from .network import deciding_leaf

_PROVIDER_ERROR_DETAIL_LIMIT = 2000
_PROVIDER_ERROR_MAX_DEPTH = 20
_MISSING_PROVIDER_ERROR_VALUE = object()
_GENERIC_CONNECTION_ERROR_MESSAGES = frozenset({"connection error", "all connection attempts failed"})
# SDK and transport wrappers whose generic text hides the socket error below.
_GENERIC_CONNECTION_ERROR_TYPES = frozenset({"APIConnectionError", "ConnectError"})
# Winsock's error codes, which Windows socket errors carry as ``errno``.
_WINSOCK_CODES = range(10000, 12000)


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
    """Return whether an SDK or transport connection wrapper has no useful detail."""
    return (
        type(exc).__name__ in _GENERIC_CONNECTION_ERROR_TYPES
        and message.casefold().rstrip(".") in _GENERIC_CONNECTION_ERROR_MESSAGES
    )


def _system_words(code: int) -> str:
    """The system's words for an ``errno``; on Windows, a Winsock code's too."""
    if sys.platform == "win32" and code in _WINSOCK_CODES:
        # The C runtime's strerror knows no Winsock code ("Unknown error").
        import ctypes

        return ctypes.FormatError(code).strip()
    return os.strerror(code)


def _socket_error_text(leaf: OSError) -> str:
    """``str(leaf)`` with the system's words in place of the raiser's message.

    asyncio's own message also names the peer, ``Connect call failed
    ('10.1.2.3', 8443)``; this text reaches the model, so it never does.
    A file name stays, as ``str()`` shows it.  Windows errors with a
    ``winerror`` already carry the system's words.
    """
    winerror = getattr(leaf, "winerror", None)
    if isinstance(winerror, int):
        words = leaf.strerror if isinstance(leaf.strerror, str) else ""
        text = f"[WinError {winerror}] {words}".rstrip()
    elif isinstance(leaf, ConnectionError) and leaf.errno == errno.EINVAL:
        # Windows' proactor re-raises a dropped connection (ERROR_NETNAME_DELETED,
        # ERROR_OPERATION_ABORTED) as ``ConnectionResetError(*exc.args)``: the
        # ``winerror`` is gone and ``errno`` is the C runtime's stand-in EINVAL,
        # which names nothing; ``strerror`` still holds the system's words.
        text = leaf.strerror if isinstance(leaf.strerror, str) else ""
    elif isinstance(leaf.errno, int):
        text = f"[Errno {leaf.errno}] {_system_words(leaf.errno)}"
    else:
        return ""
    if leaf.filename is None:
        return text
    if leaf.filename2 is None:
        return f"{text}: {leaf.filename!r}"
    return f"{text}: {leaf.filename!r} -> {leaf.filename2!r}"


def _cause_text(cause: BaseException) -> str:
    """A cause's own text; a socket error's from its code (resolver and TLS messages name no peer)."""
    socket_error = isinstance(cause, OSError) and not isinstance(cause, (socket.gaierror, socket.herror, ssl.SSLError))
    if socket_error and (text := _socket_error_text(cause)):
        return text
    return _clean_exception_text(cause)


def _explicit_causes(exc: BaseException) -> Iterator[BaseException]:
    """Yield only explicitly chained causes below *exc*, guarding cycles."""
    seen = {id(exc)}
    current = exc.__cause__
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__


def _generic_connection_root_detail(exc: BaseException) -> str:
    """Extract the detail an SDK or transport wrapper's generic text hides.

    That is the leaf the network classifier read, so the raw text names the
    same failure as the display: in a group of attempts, the one that got
    furthest.  httpcore keeps it in ``args`` below ``All connection attempts
    failed``.  Without such a leaf, the deepest useful ``__cause__``.
    """
    leaf = deciding_leaf(exc)
    candidates = [leaf] if leaf is not None and leaf is not exc else reversed(list(_explicit_causes(exc)))
    fallback = ""
    for candidate in candidates:
        message = _cause_text(candidate)
        if message and message.casefold().rstrip(".") not in _GENERIC_CONNECTION_ERROR_MESSAGES:
            return _truncate_provider_error_detail(_normalize_provider_error_detail(message))
        if not fallback:
            fallback = _fallback_exception_message(candidate)
    return fallback


def clean_error_message(e: BaseException) -> str:
    """Extract a clean error message, stripping framework class path prefixes."""
    chain = list(iter_full_chain(e, unwrap_single_groups=True))
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
