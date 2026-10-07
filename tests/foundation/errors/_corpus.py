# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A corpus of real exception chains for classifier tests.

The chains come from the real SDKs through :mod:`tests.support.provider_errors`.
Each case records the kind and retry decision the classifier must produce, and
whether that decision deliberately differs from the legacy rules.
"""

from __future__ import annotations

import errno
import socket
import ssl
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import NoReturn

import httpx
import openai

from chrys.foundation.errors import ErrorKind
from chrys.foundation.retry import StreamStall
from chrys.kernel.exceptions import ChatClientContentFilterException, ChatClientException
from chrys.service.agent_middleware.response_validation import TerminalResponseValidationError
from chrys.service.context.compaction.last_words import LastWordsGenerationError
from tests.support.network_faults import INJECTED_V4, NetworkFaults, os_error
from tests.support.provider_errors import (
    API_HOST,
    anthropic_error_event,
    anthropic_network,
    anthropic_status,
    anthropic_stream_error,
    api_request,
    httpcore_connect_failure,
    openai_network,
    openai_status,
    openai_stream_error,
    openai_transport_error,
    raised_from,
)


@dataclass(frozen=True, slots=True)
class Case:
    """One corpus entry: how to build the exception and what it must classify as."""

    id: str
    build: Callable[[], Awaitable[BaseException]]
    kind: ErrorKind
    retryable: bool
    # True when ``retryable`` deliberately differs from the legacy rules.
    changed: bool = False


def _dns(code: int) -> Callable[[NetworkFaults], None]:
    return lambda faults: faults.fail_resolve(API_HOST, code)


def _connect(arrange: Callable[[NetworkFaults, str], None]) -> Callable[[NetworkFaults], None]:
    def install(faults: NetworkFaults) -> None:
        faults.resolve_to(API_HOST, INJECTED_V4)
        arrange(faults, INJECTED_V4)

    return install


def _raise_connect_error() -> NoReturn:
    raise httpx.ConnectError("All connection attempts failed", request=api_request())


def _raise_api_connection_error() -> NoReturn:
    request = api_request()
    try:
        raise httpx.ConnectError("All connection attempts failed", request=request)
    except httpx.ConnectError as err:
        raise openai.APIConnectionError(request=request) from err


def _raise_read_timeout() -> NoReturn:
    raise httpx.ReadTimeout("", request=api_request())


def _raise_remote_protocol_error() -> NoReturn:
    raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=api_request())


FRESH_FAILURES: dict[str, Callable[[], NoReturn]] = {
    "httpx-connect-error": _raise_connect_error,
    "api-connection-error": _raise_api_connection_error,
    "httpx-read-timeout": _raise_read_timeout,
    "remote-protocol-error": _raise_remote_protocol_error,
}

STALE_PROVIDER_ERRORS: dict[str, Callable[[], Awaitable[BaseException]]] = {
    "413-request-too-large": lambda: openai_status(
        413, {"error": {"type": "invalid_request_error", "code": "request_too_large", "message": "Request too large"}}
    ),
    "429-insufficient-quota": lambda: openai_status(
        429, {"error": {"type": "insufficient_quota", "code": "insufficient_quota", "message": "Quota exceeded"}}
    ),
    "400-context-length": lambda: openai_status(
        400,
        {
            "error": {
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
                "message": "This model's maximum context length is 8192 tokens.",
            }
        },
    ),
    "401-invalid-key": lambda: openai_status(
        401, {"error": {"type": "invalid_request_error", "code": "invalid_api_key", "message": "Bad key"}}
    ),
}


def _sync(build: Callable[[], BaseException]) -> Callable[[], Awaitable[BaseException]]:
    async def run() -> BaseException:
        return build()

    return run


def _wrapped(
    build: Callable[[], Awaitable[BaseException]], wrap: Callable[[BaseException], BaseException]
) -> Callable[[], Awaitable[BaseException]]:
    async def run() -> BaseException:
        inner = await build()
        return raised_from(wrap(inner), inner)

    return run


def _chat_client_error(inner: BaseException) -> BaseException:
    return ChatClientException(f"Chat Completions request failed: {inner}", inner_exception=inner)


def _rate_limited() -> Awaitable[BaseException]:
    return openai_status(
        429, {"error": {"type": "requests", "code": "rate_limit_exceeded", "message": "Rate limit reached"}}
    )


CORPUS: tuple[Case, ...] = (
    # ── provider responses ─────────────────────────────────────────────
    Case("openai-429-rate-limit", _rate_limited, ErrorKind.RATE_LIMITED, True),
    Case(
        "openai-429-insufficient-quota",
        STALE_PROVIDER_ERRORS["429-insufficient-quota"],
        ErrorKind.QUOTA_EXHAUSTED,
        False,
        changed=True,
    ),
    Case(
        "openai-429-too-many-tokens-please-wait",
        lambda: openai_status(
            429,
            {
                "error": {
                    "type": "tokens",
                    "code": "rate_limit_exceeded",
                    "message": "Too many tokens, please wait before trying again.",
                }
            },
        ),
        ErrorKind.RATE_LIMITED,
        True,
    ),
    Case(
        "glm-429-1113",
        lambda: openai_status(429, {"error": {"code": "1113", "message": "余额不足或无可用资源包,请充值。"}}),
        ErrorKind.QUOTA_EXHAUSTED,
        False,
        changed=True,
    ),
    Case(
        "glm-400-1113",
        lambda: openai_status(400, {"error": {"code": "1113", "message": "余额不足或无可用资源包,请充值。"}}),
        ErrorKind.REQUEST_REJECTED,
        False,
    ),
    Case(
        "openai-402",
        lambda: openai_status(402, {"error": {"message": "Payment required"}}),
        ErrorKind.QUOTA_EXHAUSTED,
        False,
    ),
    Case(
        "openai-500",
        lambda: openai_status(500, {"error": {"type": "server_error", "message": "boom"}}),
        ErrorKind.SERVER_ERROR,
        True,
    ),
    Case("openai-400-context-length", STALE_PROVIDER_ERRORS["400-context-length"], ErrorKind.CONTEXT_OVERFLOW, False),
    Case(
        "openai-413-request-too-large",
        STALE_PROVIDER_ERRORS["413-request-too-large"],
        ErrorKind.PAYLOAD_TOO_LARGE,
        False,
    ),
    Case("openai-401", STALE_PROVIDER_ERRORS["401-invalid-key"], ErrorKind.AUTH_FAILED, False),
    Case(
        "anthropic-529-overloaded",
        lambda: anthropic_status(
            529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        ),
        ErrorKind.OVERLOADED,
        True,
    ),
    Case(
        "anthropic-400-prompt-too-long",
        lambda: anthropic_status(
            400,
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "prompt is too long: 210000 tokens"},
            },
        ),
        ErrorKind.CONTEXT_OVERFLOW,
        False,
    ),
    # ── errors a stream reports in-band ───────────────────────────────
    Case(
        "openai-stream-error-no-code",
        lambda: openai_stream_error({"message": "The server had an error while processing your request."}),
        ErrorKind.UNKNOWN,
        True,
    ),
    Case(
        "openai-stream-error-server-type",
        lambda: openai_stream_error({"type": "server_error", "message": "upstream failed"}),
        ErrorKind.SERVER_ERROR,
        True,
    ),
    Case(
        "openai-stream-error-quota",
        lambda: openai_stream_error(
            {"type": "insufficient_quota", "code": "insufficient_quota", "message": "You exceeded your quota."}
        ),
        ErrorKind.QUOTA_EXHAUSTED,
        False,
        changed=True,
    ),
    Case(
        "openai-stream-error-overflow",
        lambda: openai_stream_error(
            {"message": "This model's maximum context length is 128000 tokens. Your messages resulted in 130000."}
        ),
        ErrorKind.CONTEXT_OVERFLOW,
        False,
        changed=True,
    ),
    *(
        Case(
            f"anthropic-stream-overloaded{suffix}",
            lambda as_text=as_text: anthropic_stream_error(
                anthropic_error_event("overloaded_error", "Overloaded", as_json_text=as_text)
            ),
            ErrorKind.OVERLOADED,
            True,
            changed=True,
        )
        for suffix, as_text in (("", False), ("-json-text", True))
    ),
    *(
        Case(
            f"anthropic-stream-invalid-request{suffix}",
            lambda as_text=as_text: anthropic_stream_error(
                anthropic_error_event("invalid_request_error", "tools: bad schema", as_json_text=as_text)
            ),
            ErrorKind.REQUEST_REJECTED,
            False,
        )
        for suffix, as_text in (("", False), ("-json-text", True))
    ),
    Case(
        "anthropic-stream-raw-text",
        lambda: anthropic_stream_error("upstream hiccup"),
        ErrorKind.UNKNOWN,
        True,
        changed=True,
    ),
    Case(
        "chat-client-wrapped-429",
        _wrapped(_rate_limited, _chat_client_error),
        ErrorKind.RATE_LIMITED,
        True,
    ),
    Case(
        "content-filter-wrapper",
        _wrapped(
            lambda: openai_status(400, {"error": {"code": "content_filter", "message": "filtered"}}),
            lambda inner: ChatClientContentFilterException("content error", inner_exception=inner),
        ),
        ErrorKind.CONTENT_FILTERED,
        False,
    ),
    Case(
        "last-words-over-429",
        _wrapped(_rate_limited, lambda _inner: LastWordsGenerationError("progress note failed")),
        ErrorKind.RATE_LIMITED,
        False,
    ),
    # ── network, through the real SDK and httpx stack ─────────────────
    Case(
        "openai-read-timeout",
        lambda: openai_transport_error(lambda request: httpx.ReadTimeout("timed out", request=request)),
        ErrorKind.READ_TIMEOUT,
        True,
    ),
    Case(
        "openai-write-timeout",
        lambda: openai_transport_error(lambda request: httpx.WriteTimeout("timed out", request=request)),
        ErrorKind.WRITE_TIMEOUT,
        True,
    ),
    Case("openai-dns-noname", lambda: openai_network(_dns(socket.EAI_NONAME)), ErrorKind.DNS_FAILED, False),
    Case("openai-dns-again", lambda: openai_network(_dns(socket.EAI_AGAIN)), ErrorKind.DNS_FAILED, True),
    Case("openai-dns-fail", lambda: openai_network(_dns(socket.EAI_FAIL)), ErrorKind.DNS_FAILED, False),
    Case("anthropic-dns-noname", lambda: anthropic_network(_dns(socket.EAI_NONAME)), ErrorKind.DNS_FAILED, False),
    Case(
        "openai-connect-refused",
        lambda: openai_network(_connect(lambda faults, ip: faults.refuse(ip))),
        ErrorKind.CONNECTION_REFUSED,
        True,
    ),
    Case(
        "openai-connect-no-route",
        lambda: openai_network(_connect(lambda faults, ip: faults.fail_connect(ip, os_error(errno.ENETUNREACH)))),
        ErrorKind.NO_ROUTE,
        True,
    ),
    Case(
        "openai-connect-timeout",
        lambda: openai_network(_connect(lambda faults, ip: faults.hang_connect(ip)), connect_timeout=0.05),
        ErrorKind.CONNECT_TIMEOUT,
        True,
    ),
    Case(
        "tls-certificate-verify-failed",
        _sync(
            lambda: httpcore_connect_failure(
                ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
            )
        ),
        ErrorKind.TLS_FAILED,
        False,
    ),
    Case(
        "proxy-407",
        _sync(
            lambda: raised_from(
                openai.APIConnectionError(request=api_request()),
                httpx.ProxyError("407 Proxy Authentication Required", request=api_request()),
            )
        ),
        ErrorKind.PROXY_AUTH_FAILED,
        False,
    ),
    Case(
        "invalid-url",
        _sync(lambda: raised_from(openai.APIConnectionError(request=api_request()), httpx.InvalidURL("bad"))),
        ErrorKind.INVALID_ENDPOINT,
        False,
    ),
    Case(
        "unsupported-scheme",
        _sync(
            lambda: raised_from(
                openai.APIConnectionError(request=api_request()),
                httpx.UnsupportedProtocol("Request URL has an unsupported protocol 'ftp://'."),
            )
        ),
        ErrorKind.INVALID_ENDPOINT,
        False,
    ),
    # Never retried, but nothing names the endpoint as the cause: the raw text explains it.
    Case(
        "local-protocol-error",
        _sync(
            lambda: raised_from(
                openai.APIConnectionError(request=api_request()), httpx.LocalProtocolError("Illegal header value")
            )
        ),
        ErrorKind.UNKNOWN,
        False,
    ),
    Case(
        "redirect-loop",
        _sync(
            lambda: raised_from(
                openai.APIConnectionError(request=api_request()),
                httpx.TooManyRedirects("Exceeded maximum allowed redirects.", request=api_request()),
            )
        ),
        ErrorKind.UNKNOWN,
        False,
    ),
    # ── runtime ────────────────────────────────────────────────────────
    Case("stream-stall", _sync(lambda: StreamStall("no chunk for 60s")), ErrorKind.STREAM_STALLED, False),
    Case(
        "terminal-validation",
        _sync(lambda: TerminalResponseValidationError("empty response")),
        ErrorKind.INVALID_RESPONSE,
        False,
    ),
    Case("builtin-connection-error", _sync(lambda: ConnectionError("reset")), ErrorKind.UNKNOWN, True),
    Case("builtin-value-error", _sync(lambda: ValueError("bad input")), ErrorKind.UNKNOWN, False),
)
