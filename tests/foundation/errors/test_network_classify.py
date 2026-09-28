# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Connection failures: attempt groups, first-hop evidence and route-aware DNS verdicts."""

from __future__ import annotations

import errno
import socket
import ssl
import sys
from collections.abc import Callable

import httpx
import pytest

from chrys.foundation.errors import (
    ROUTE_EXTENSION_KEY,
    ErrorKind,
    Origin,
    RouteFacts,
    classify_error,
    is_deterministic_connection_error,
    is_retryable,
)
from chrys.foundation.errors.network import NetworkCodes, classify_connection_failure, codes_for
from tests.support.provider_errors import API_HOST, api_request, httpcore_connect_failure, raised_from

pytestmark = pytest.mark.skipif(codes_for(sys.platform) is None, reason="no code table for this OS")

_CODES = codes_for(sys.platform)
_TARGET = Origin("https", API_HOST, 443)


def _codes() -> NetworkCodes:
    assert _CODES is not None
    return _CODES


def _os_error(code: int) -> OSError:
    return OSError(code, "Connect call failed")


def _attempts(*leaves: BaseException) -> BaseException:
    """What anyio's ``connect_tcp`` raises after several failed attempts."""
    group = ExceptionGroup("multiple connection attempts failed", list(leaves))
    return raised_from(OSError("All connection attempts failed"), group)


def _noname() -> socket.gaierror:
    return socket.gaierror(_codes().eai_noname, "nodename nor servname provided, or not known")


def test_a_mixed_group_is_named_by_the_attempt_that_got_furthest() -> None:
    codes = _codes()
    exc = httpcore_connect_failure(_attempts(_os_error(codes.enetunreach), _os_error(codes.econnrefused)))

    result = classify_error(exc)

    assert (result.kind, result.retryable, result.failed_at_first_hop) == (
        ErrorKind.CONNECTION_REFUSED,
        True,
        True,
    )


def test_no_route_names_a_group_only_when_every_attempt_had_none() -> None:
    codes = _codes()
    all_no_route = httpcore_connect_failure(_attempts(_os_error(codes.enetunreach), _os_error(codes.enetunreach)))
    some_timeout = httpcore_connect_failure(_attempts(_os_error(codes.enetunreach), _os_error(codes.etimedout)))

    assert classify_error(all_no_route).kind is ErrorKind.NO_ROUTE
    assert classify_error(some_timeout).kind is ErrorKind.CONNECT_TIMEOUT


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_an_absent_address_family_abstains_on_every_platform(platform: str) -> None:
    codes = codes_for(platform)
    assert codes is not None

    def absent() -> OSError:
        return OSError(codes.eafnosupport, "Address family not supported by protocol")

    mixed = classify_connection_failure(
        ExceptionGroup("attempts", [absent(), OSError(codes.enetunreach, "Connect call failed")]), None, codes=codes
    )
    only_absent = classify_connection_failure(ExceptionGroup("attempts", [absent(), absent()]), None, codes=codes)
    unrecognized = classify_connection_failure(
        ExceptionGroup("attempts", [absent(), OSError(errno.EACCES, "Permission denied")]), None, codes=codes
    )

    assert mixed is not None and (mixed.kind, mixed.first_hop_evidence) == (ErrorKind.NO_ROUTE, True)
    assert only_absent is not None and (only_absent.kind, only_absent.first_hop_evidence) == (ErrorKind.NO_ROUTE, True)
    # An unrecognized attempt still votes, and proves nothing about the first hop.
    assert unrecognized is not None
    assert (unrecognized.kind, unrecognized.first_hop_evidence) == (ErrorKind.CONNECTION_FAILED, False)


def test_a_group_vetoes_a_retry_only_when_every_attempt_is_deterministic() -> None:
    codes = _codes()
    mixed = httpcore_connect_failure(_attempts(_noname(), _os_error(codes.econnrefused)))

    assert classify_error(mixed).retryable is True
    assert is_deterministic_connection_error(mixed) is False


def test_group_behind_httpcore_arg_now_vetoed_without_reach() -> None:
    exc = httpcore_connect_failure(_attempts(_noname(), _noname()))

    result = classify_error(exc)

    assert (result.kind, result.retryable, result.failed_at_first_hop) == (ErrorKind.DNS_FAILED, False, True)
    assert is_retryable(exc) is False


def test_a_group_has_first_hop_evidence_only_when_every_attempt_does() -> None:
    codes = _codes()
    mixed = httpcore_connect_failure(_attempts(_os_error(codes.econnrefused), TimeoutError()))

    result = classify_error(mixed)

    # The furthest attempt refused, but the other one proves nothing about the first hop.
    assert (result.kind, result.failed_at_first_hop) == (ErrorKind.CONNECTION_REFUSED, False)


@pytest.mark.parametrize(
    ("leaf", "kind"),
    [
        pytest.param(lambda codes: _os_error(codes.econnrefused), ErrorKind.CONNECTION_REFUSED, id="refused"),
        pytest.param(lambda codes: _os_error(codes.ehostunreach), ErrorKind.HOST_UNREACHABLE, id="unreachable"),
        pytest.param(lambda codes: _os_error(codes.enetunreach), ErrorKind.NO_ROUTE, id="no-route"),
        pytest.param(lambda codes: socket.gaierror(codes.eai_again, "try again"), ErrorKind.DNS_FAILED, id="dns"),
    ],
)
def test_socket_and_dns_answers_are_first_hop_evidence(
    leaf: Callable[[NetworkCodes], BaseException], kind: ErrorKind
) -> None:
    result = classify_error(httpcore_connect_failure(leaf(_codes())))

    assert (result.kind, result.failed_at_first_hop) == (kind, True)


def test_a_connect_timeout_is_not_first_hop_evidence() -> None:
    exc = raised_from(
        httpx.ConnectTimeout("timed out", request=api_request()), httpcore_connect_failure(TimeoutError())
    )

    result = classify_error(exc)

    assert (result.kind, result.timeout_phase, result.failed_at_first_hop) == (
        ErrorKind.CONNECT_TIMEOUT,
        "connect",
        False,
    )


def test_a_timeout_outside_a_connect_wrapper_is_not_a_network_fact() -> None:
    exc = raised_from(RuntimeError("tool deadline"), TimeoutError())

    assert classify_connection_failure(exc, None) is None


def test_tls_eof_during_connect_is_a_failed_connection_not_a_tls_verdict() -> None:
    eof = ssl.SSLEOFError(8, "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol")

    result = classify_error(httpcore_connect_failure(eof))

    assert (result.kind, result.retryable, result.failed_at_first_hop) == (ErrorKind.CONNECTION_FAILED, True, False)


@pytest.mark.parametrize(
    ("message", "kind", "retryable"),
    [
        ("407 Proxy Authentication Required", ErrorKind.PROXY_AUTH_FAILED, False),
        ("502 Bad Gateway", ErrorKind.PROXY_REJECTED, True),
    ],
)
def test_proxy_errors_are_named_by_the_proxy_answer(message: str, kind: ErrorKind, retryable: bool) -> None:
    exc = raised_from(RuntimeError("Connection error."), httpx.ProxyError(message, request=api_request()))

    result = classify_error(exc)

    assert (result.kind, result.retryable, result.failed_at_first_hop) == (kind, retryable, False)


def _request_via(route: RouteFacts | None) -> httpx.Request:
    request = api_request()
    if route is not None:
        request.extensions = {**request.extensions, ROUTE_EXTENSION_KEY: route}
    return request


def _noname_on(request: httpx.Request) -> BaseException:
    transport = raised_from(httpx.ConnectError("nodename nor servname provided", request=request), _noname())
    return raised_from(RuntimeError("Connection error."), transport)


@pytest.mark.parametrize(
    ("route", "retryable"),
    [
        pytest.param(None, False, id="no-route-facts"),
        pytest.param(RouteFacts(_TARGET, None, first_hop_reached=False), False, id="never-reached"),
        pytest.param(RouteFacts(_TARGET, None, first_hop_reached=True), True, id="reached-before"),
    ],
)
def test_noname_vetoes_a_retry_unless_the_first_hop_was_reached_before(
    route: RouteFacts | None, retryable: bool
) -> None:
    exc = _noname_on(_request_via(route))

    result = classify_error(exc)

    assert (result.kind, result.retryable, result.route) == (ErrorKind.DNS_FAILED, retryable, route)
    assert is_deterministic_connection_error(exc) is not retryable


def test_a_route_below_a_response_is_never_read() -> None:
    class _StatusError(Exception):
        status_code = 401

    reached = RouteFacts(_TARGET, None, first_hop_reached=True)
    hidden = httpx.ConnectError("stale", request=_request_via(reached))
    exc = raised_from(_StatusError("unauthorized"), hidden)

    assert classify_error(exc).route is None
