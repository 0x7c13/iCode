# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The closed set of error kinds the classifier assigns."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

TimeoutPhase = Literal["connect", "write", "read", "pool"]


class ErrorKind(StrEnum):
    """What went wrong, as far as explicit evidence shows.

    The values are a stable contract: display text, logs and future wire
    fields key on them.
    """

    # Network: the request never got a response.
    NO_ROUTE = "no_route"
    HOST_UNREACHABLE = "host_unreachable"
    DNS_FAILED = "dns_failed"
    CONNECTION_REFUSED = "connection_refused"
    CONNECT_TIMEOUT = "connect_timeout"
    CONNECTION_FAILED = "connection_failed"
    CONNECTION_LOST = "connection_lost"
    READ_TIMEOUT = "read_timeout"
    WRITE_TIMEOUT = "write_timeout"
    PROXY_REJECTED = "proxy_rejected"
    PROXY_AUTH_FAILED = "proxy_auth_failed"
    TLS_FAILED = "tls_failed"
    INVALID_ENDPOINT = "invalid_endpoint"
    # Provider: the service answered with an error.
    RATE_LIMITED = "rate_limited"
    QUOTA_EXHAUSTED = "quota_exhausted"
    OVERLOADED = "overloaded"
    SERVER_ERROR = "server_error"
    CONTEXT_OVERFLOW = "context_overflow"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    AUTH_FAILED = "auth_failed"
    REQUEST_REJECTED = "request_rejected"
    CONTENT_FILTERED = "content_filtered"
    STREAM_TRUNCATED = "stream_truncated"
    # Runtime: Chrys's own policies concluded the attempt.
    STREAM_STALLED = "stream_stalled"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


NETWORK_KINDS = frozenset(
    {
        ErrorKind.NO_ROUTE,
        ErrorKind.HOST_UNREACHABLE,
        ErrorKind.DNS_FAILED,
        ErrorKind.CONNECTION_REFUSED,
        ErrorKind.CONNECT_TIMEOUT,
        ErrorKind.CONNECTION_FAILED,
        ErrorKind.CONNECTION_LOST,
        ErrorKind.READ_TIMEOUT,
        ErrorKind.WRITE_TIMEOUT,
        ErrorKind.PROXY_REJECTED,
        ErrorKind.PROXY_AUTH_FAILED,
        ErrorKind.TLS_FAILED,
        ErrorKind.INVALID_ENDPOINT,
    }
)
