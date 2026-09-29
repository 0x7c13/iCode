# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The literal per-platform code tables, and every platform's leaves classified on any OS."""

from __future__ import annotations

import dataclasses
import errno
import socket
import sys
from collections.abc import Callable
from types import ModuleType

import httpx
import pytest

from chrys.foundation.errors import ErrorKind, network
from chrys.foundation.errors.network import (
    _DARWIN,
    _LINUX,
    _WIN32,
    NetworkCodes,
    NetworkFailure,
    classify_connection_failure,
    codes_for,
)
from tests.support.network_faults import SimulatedWindowsError
from tests.support.provider_errors import api_request, raised_from

# Each table field and the symbolic constant it spells out.
_SYMBOLS: dict[str, tuple[object, str]] = {
    "eai_noname": (socket, "EAI_NONAME"),
    "eai_nodata": (socket, "EAI_NODATA"),
    "eai_again": (socket, "EAI_AGAIN"),
    "eai_fail": (socket, "EAI_FAIL"),
    "enetunreach": (errno, "ENETUNREACH"),
    "enetdown": (errno, "ENETDOWN"),
    "ehostunreach": (errno, "EHOSTUNREACH"),
    "ehostdown": (errno, "EHOSTDOWN"),
    "econnrefused": (errno, "ECONNREFUSED"),
    "etimedout": (errno, "ETIMEDOUT"),
    "econnreset": (errno, "ECONNRESET"),
    "econnaborted": (errno, "ECONNABORTED"),
    "eaddrnotavail": (errno, "EADDRNOTAVAIL"),
    "eafnosupport": (errno, "EAFNOSUPPORT"),
}
_EAI_CONFIG_SYMBOLS = ("EAI_BADFLAGS", "EAI_FAMILY", "EAI_SERVICE", "EAI_SOCKTYPE")
# Windows-only fields: neither ``errno`` nor ``socket`` exports them.
_UNEXPORTED_FIELDS = {"windows", "wsano_data"} | {
    field.name for field in dataclasses.fields(NetworkCodes) if field.name.startswith("win_")
}


def test_every_table_field_is_either_checked_against_a_constant_or_known_unexported() -> None:
    names = {field.name for field in dataclasses.fields(NetworkCodes)}

    assert names == {*_SYMBOLS, "eai_config", *_UNEXPORTED_FIELDS}


@pytest.mark.skipif(codes_for(sys.platform) is None, reason="no code table for this OS")
def test_literal_table_matches_platform_constants() -> None:
    table = codes_for(sys.platform)
    assert table is not None

    assert {field: getattr(table, field) for field in _SYMBOLS} == {
        field: getattr(module, name) for field, (module, name) in _SYMBOLS.items()
    }
    assert table.eai_config == frozenset(getattr(socket, name) for name in _EAI_CONFIG_SYMBOLS)
    assert table.windows is (sys.platform == "win32")


def test_codes_for_names_each_supported_platform() -> None:
    assert (codes_for("darwin"), codes_for("linux"), codes_for("win32"), codes_for("freebsd14")) == (
        _DARWIN,
        _LINUX,
        _WIN32,
        None,
    )


def _connect_failure(leaf: BaseException) -> BaseException:
    """``leaf`` below httpx's connect wrapper, as a failed connect raises it."""
    return raised_from(httpx.ConnectError(str(leaf), request=api_request()), leaf)


def _posix(field: str) -> Callable[[NetworkCodes], BaseException]:
    return lambda codes: _connect_failure(OSError(getattr(codes, field), "Connect call failed"))


def _iocp(field: str) -> Callable[[NetworkCodes], BaseException]:
    # IOCP connect failures carry EINVAL in errno and the real code in winerror.
    return lambda codes: _connect_failure(
        SimulatedWindowsError(errno.EINVAL, "Connect call failed", getattr(codes, field))
    )


def _dns(field: str) -> Callable[[NetworkCodes], BaseException]:
    return lambda codes: _connect_failure(socket.gaierror(getattr(codes, field), "lookup failed"))


def _dns_code(code: Callable[[NetworkCodes], int]) -> Callable[[NetworkCodes], BaseException]:
    return lambda codes: _connect_failure(socket.gaierror(code(codes), "lookup failed"))


_REFUSED = NetworkFailure(ErrorKind.CONNECTION_REFUSED, first_hop_evidence=True)
_NO_ROUTE = NetworkFailure(ErrorKind.NO_ROUTE, first_hop_evidence=True)
_UNREACHABLE = NetworkFailure(ErrorKind.HOST_UNREACHABLE, first_hop_evidence=True)
_TIMEOUT = NetworkFailure(ErrorKind.CONNECT_TIMEOUT, timeout_phase="connect")
_RESET = NetworkFailure(ErrorKind.CONNECTION_LOST)
_NONAME = NetworkFailure(ErrorKind.DNS_FAILED, deterministic=True, first_hop_evidence=True)
_AGAIN = NetworkFailure(ErrorKind.DNS_FAILED, first_hop_evidence=True)
_MALFORMED_LOOKUP = NetworkFailure(ErrorKind.INVALID_ENDPOINT, deterministic=True)

# (leaf name, builder, expected) for every table; evidence is log-only and not compared.
_COMMON_LEAVES: tuple[tuple[str, Callable[[NetworkCodes], BaseException], NetworkFailure], ...] = (
    ("refused", _posix("econnrefused"), _REFUSED),
    ("net-unreachable", _posix("enetunreach"), _NO_ROUTE),
    ("net-down", _posix("enetdown"), _NO_ROUTE),
    ("host-unreachable", _posix("ehostunreach"), _UNREACHABLE),
    ("host-down", _posix("ehostdown"), _UNREACHABLE),
    ("timed-out", _posix("etimedout"), _TIMEOUT),
    ("reset", _posix("econnreset"), _RESET),
    ("aborted", _posix("econnaborted"), _RESET),
    ("dns-noname", _dns("eai_noname"), _NONAME),
    ("dns-nodata", _dns("eai_nodata"), _NONAME),
    ("dns-again", _dns("eai_again"), _AGAIN),
    ("dns-fail", _dns("eai_fail"), NetworkFailure(ErrorKind.DNS_FAILED, deterministic=True, first_hop_evidence=True)),
    ("dns-badflags", _dns_code(lambda codes: min(codes.eai_config)), _MALFORMED_LOOKUP),
)
_POSIX_ONLY_LEAVES = (("addr-not-available", _posix("eaddrnotavail"), _UNREACHABLE),)
_WINDOWS_ONLY_LEAVES = (
    ("iocp-refused", _iocp("win_refused"), _REFUSED),
    ("iocp-net-unreachable", _iocp("win_netunreach"), _NO_ROUTE),
    ("iocp-host-unreachable", _iocp("win_hostunreach"), _UNREACHABLE),
    ("iocp-timeout", _iocp("win_timeout"), _TIMEOUT),
    ("iocp-reset", _iocp("win_reset"), _RESET),
    ("iocp-aborted", _iocp("win_aborted"), _RESET),
    ("wsano-data", _dns_code(lambda codes: codes.wsano_data or 0), _NONAME),
)
_CASES = [
    pytest.param(codes, build, expected, id=f"{platform}-{leaf}")
    for platform, codes, extra in (
        ("darwin", _DARWIN, _POSIX_ONLY_LEAVES),
        ("linux", _LINUX, _POSIX_ONLY_LEAVES),
        ("win32", _WIN32, _WINDOWS_ONLY_LEAVES),
    )
    for leaf, build, expected in (*_COMMON_LEAVES, *extra)
]


@pytest.mark.parametrize(("codes", "build", "expected"), _CASES)
def test_classify_leaf_with_each_platform_table(
    codes: NetworkCodes, build: Callable[[NetworkCodes], BaseException], expected: NetworkFailure
) -> None:
    failure = classify_connection_failure(build(codes), None, codes=codes)

    assert failure is not None
    assert dataclasses.replace(failure, evidence="") == expected


def test_windows_reads_winerror_before_errno() -> None:
    # errno 10061 alone would say refused; the IOCP winerror says unreachable.
    leaf = SimulatedWindowsError(_WIN32.econnrefused, "Connect call failed", _WIN32.win_hostunreach)

    failure = classify_connection_failure(_connect_failure(leaf), None, codes=_WIN32)

    assert failure is not None
    assert (failure.kind, failure.evidence) == (ErrorKind.HOST_UNREACHABLE, "winerror 1232")


def test_posix_tables_ignore_winerror() -> None:
    leaf = SimulatedWindowsError(_LINUX.econnrefused, "Connect call failed", _WIN32.win_hostunreach)

    failure = classify_connection_failure(_connect_failure(leaf), None, codes=_LINUX)

    assert failure is not None
    assert (failure.kind, failure.evidence) == (ErrorKind.CONNECTION_REFUSED, "errno 111")


def test_an_unsupported_os_still_names_dns_and_refused_leaves_by_type(monkeypatch: pytest.MonkeyPatch) -> None:
    shadow_sys = ModuleType("sys")
    shadow_sys.__dict__.update(vars(sys), platform="freebsd14")
    monkeypatch.setattr(network, "sys", shadow_sys)

    dns = classify_connection_failure(_connect_failure(socket.gaierror(8, "lookup failed")), None)
    refused = classify_connection_failure(_connect_failure(ConnectionRefusedError(61, "Connect call failed")), None)

    # Without a table no code is trusted: DNS never vetoes, refused comes from the type.
    assert dns == NetworkFailure(ErrorKind.DNS_FAILED, first_hop_evidence=True, evidence="gaierror 8")
    assert refused is not None
    assert (refused.kind, refused.deterministic) == (ErrorKind.CONNECTION_REFUSED, False)
