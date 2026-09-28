# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Connection failures: what the socket, DNS, TLS and proxy layers said, on every OS.

:func:`classify_connection_failure` searches the explicit graph for the first
leaf a rule recognizes: proxy errors, TLS errors, ``getaddrinfo`` codes,
socket codes (``winerror`` first on Windows, then ``errno``, then the
exception type) and httpx/httpcore transport types.  Codes come from literal
per-platform tables, never from message text.  The attempts of one connect
(dual stack, several addresses) arrive as an exception group; the attempt
that got furthest names the failure, an attempt in an address family the
machine lacks abstains, and only a group whose every voting member is
deterministic vetoes a retry.
"""

from __future__ import annotations

import socket
import ssl
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from ._text import _clean_exception_text
from ._walk import explicit_children, has_response_status, iter_explicit_graph
from .kinds import ErrorKind, TimeoutPhase
from .route import RouteFacts, route_of


@dataclass(frozen=True, slots=True)
class NetworkCodes:
    """One platform's ``getaddrinfo`` and socket error codes, as literals."""

    eai_noname: int
    eai_nodata: int
    eai_again: int
    eai_fail: int
    # BADFLAGS, FAMILY, SERVICE, SOCKTYPE: the lookup itself was malformed.
    eai_config: frozenset[int]
    enetunreach: int
    enetdown: int
    ehostunreach: int
    ehostdown: int
    econnrefused: int
    etimedout: int
    econnreset: int
    econnaborted: int
    eaddrnotavail: int
    # The machine has no stack for the socket's address family (no IPv6).
    eafnosupport: int
    # Windows reads ``winerror`` before ``errno``: an IOCP connect failure
    # carries its real code there and EINVAL in ``errno``.
    windows: bool = False
    # Neither ``socket`` nor ``errno`` exports this one: Winsock's
    # "valid name, no data record".
    wsano_data: int | None = None
    win_refused: int = 1225
    win_netunreach: int = 1231
    win_hostunreach: int = 1232
    win_timeout: int = 121
    win_reset: int = 64
    win_aborted: int = 1236


_DARWIN = NetworkCodes(
    eai_noname=8,
    eai_nodata=7,
    eai_again=2,
    eai_fail=4,
    eai_config=frozenset({3, 5, 9, 10}),
    enetunreach=51,
    enetdown=50,
    ehostunreach=65,
    ehostdown=64,
    econnrefused=61,
    etimedout=60,
    econnreset=54,
    econnaborted=53,
    eaddrnotavail=49,
    eafnosupport=47,
)
_LINUX = NetworkCodes(
    eai_noname=-2,
    eai_nodata=-5,
    eai_again=-3,
    eai_fail=-4,
    eai_config=frozenset({-1, -6, -8, -7}),
    enetunreach=101,
    enetdown=100,
    ehostunreach=113,
    ehostdown=112,
    econnrefused=111,
    etimedout=110,
    econnreset=104,
    econnaborted=103,
    eaddrnotavail=99,
    eafnosupport=97,
)
# Winsock: EAI_NODATA is EAI_NONAME (WSAHOST_NOT_FOUND); socket errno values
# are the WSA codes.
_WIN32 = NetworkCodes(
    eai_noname=11001,
    eai_nodata=11001,
    eai_again=11002,
    eai_fail=11003,
    eai_config=frozenset({10022, 10047, 10109, 10044}),
    enetunreach=10051,
    enetdown=10050,
    ehostunreach=10065,
    ehostdown=10064,
    econnrefused=10061,
    etimedout=10060,
    econnreset=10054,
    econnaborted=10053,
    eaddrnotavail=10049,
    eafnosupport=10047,
    windows=True,
    wsano_data=11004,
)


def codes_for(platform: str) -> NetworkCodes | None:
    """Return the code table for a ``sys.platform`` value, or None for an unsupported OS."""
    if platform == "darwin":
        return _DARWIN
    if platform == "win32":
        return _WIN32
    if platform.startswith("linux"):
        return _LINUX
    return None


@dataclass(frozen=True, slots=True)
class NetworkFailure:
    """What a failed connection's leaf says."""

    kind: ErrorKind
    # No retry of the same request can succeed.
    deterministic: bool = False
    timeout_phase: TimeoutPhase | None = None
    # The leaf is a TCP or DNS answer about the first hop (refused, name not
    # found, unreachable): the first hop itself failed.  Timeouts, TLS EOF and
    # proxy errors prove nothing about which hop failed.
    first_hop_evidence: bool = False
    # Log-only, e.g. ``"gaierror 8"`` or ``"winerror 1231"``.
    evidence: str = ""


# How far an attempt got before it failed; the furthest attempt names a group.
# An attempt in an address family the machine lacks never left it: it abstains.
_REACH_ABSENT = 0
_REACH_NO_ROUTE = 1
_REACH_OTHER = 2
_REACH_ADDR_NOT_AVAILABLE = 3
_REACH_HOST_UNREACHABLE = 4
_REACH_TIMEOUT = 5
_REACH_RESET = 6
_REACH_REFUSED = 7


@dataclass(frozen=True, slots=True)
class _Hit:
    failure: NetworkFailure
    reach: int = _REACH_OTHER
    # The exception that named the failure; set by the search, not the rules.
    leaf: BaseException | None = None


_TRANSPORT_MODULES = frozenset({"httpx", "httpcore"})
_CONNECT_WRAPPER_NAMES = frozenset({"ConnectError", "ConnectTimeout"})
# Failures no retry of the same request fixes.  Only a malformed URL or an
# unsupported scheme names the endpoint as the cause; a local protocol
# violation or a redirect loop stays unnamed, and its raw text explains it.
_INVALID_ENDPOINT_TYPE_NAMES = frozenset({"InvalidURL", "UnsupportedProtocol"})
_DETERMINISTIC_TRANSPORT_TYPE_NAMES = _INVALID_ENDPOINT_TYPE_NAMES | {"LocalProtocolError", "TooManyRedirects"}
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
_TLS_EOF_REASON_CODES = frozenset({"UNEXPECTED_EOF_WHILE_READING", "EOF"})
_NON_RETRYABLE_PROXY_ERROR_MESSAGES = frozenset({"invalid username/password"})
_PROXY_AUTH_NEGOTIATION_PREFIX = "requested "
_PROXY_AUTH_NEGOTIATION_SEPARATOR = " from proxy server, but got "
# Fallback string patterns for transport-level errors that may not carry
# a typed exception (e.g. httpx errors surfacing as generic exceptions
# through framework wrapping).  The legacy transient layer retries on them;
# here they only name the kind.
RETRYABLE_PHRASES = (
    "peer closed connection",
    "incomplete chunked read",
    "remoteprotocolerror",
    "connection reset",
    "connection closed",
    "server disconnected",
    "socket hang up",
)


def _transport_type_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__ if cls.__module__.partition(".")[0] in _TRANSPORT_MODULES}


def _is_connect_wrapper(exc: BaseException) -> bool:
    return bool(_transport_type_names(exc) & _CONNECT_WRAPPER_NAMES)


def _is_proxy_authentication_failure(exc: BaseException) -> bool:
    message = _clean_exception_text(exc).casefold().strip().rstrip(".")
    return (
        message.partition(" ")[0] == "407"
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


def _lost(connect_phase: bool, evidence: str) -> _Hit:
    """An unnamed hang-up: a failed connect under a connect wrapper, else a lost connection."""
    kind = ErrorKind.CONNECTION_FAILED if connect_phase else ErrorKind.CONNECTION_LOST
    return _Hit(NetworkFailure(kind, evidence=evidence), _REACH_RESET)


def _reset(evidence: str) -> _Hit:
    return _Hit(NetworkFailure(ErrorKind.CONNECTION_LOST, evidence=evidence), _REACH_RESET)


def _proxy_leaf(exc: BaseException) -> _Hit | None:
    if type(exc).__name__ != "ProxyError":
        return None
    if _is_proxy_authentication_failure(exc):
        return _Hit(NetworkFailure(ErrorKind.PROXY_AUTH_FAILED, deterministic=True, evidence="proxy authentication"))
    return _Hit(NetworkFailure(ErrorKind.PROXY_REJECTED, evidence="proxy error"))


def _tls_leaf(exc: BaseException, connect_phase: bool) -> _Hit | None:
    if isinstance(exc, ssl.SSLCertVerificationError):
        return _Hit(NetworkFailure(ErrorKind.TLS_FAILED, deterministic=True, evidence="tls certificate"))
    if not isinstance(exc, ssl.SSLError):
        return None
    reason = _ssl_reason_code(exc)
    if isinstance(exc, ssl.SSLEOFError) or reason in _TLS_EOF_REASON_CODES:
        # The peer (often a proxy or middlebox) hung up mid-handshake.
        return _lost(connect_phase, "tls eof")
    deterministic = reason in _DETERMINISTIC_TLS_CONFIGURATION_REASON_CODES
    return _Hit(NetworkFailure(ErrorKind.TLS_FAILED, deterministic=deterministic, evidence=f"tls {reason}".rstrip()))


def _dns_leaf(exc: BaseException, codes: NetworkCodes | None, route: RouteFacts | None) -> _Hit | None:
    if not isinstance(exc, socket.gaierror):
        return None
    code = exc.errno
    evidence = f"gaierror {code}"
    if codes is None or not isinstance(code, int):
        return _Hit(NetworkFailure(ErrorKind.DNS_FAILED, first_hop_evidence=True, evidence=evidence))
    if code in codes.eai_config:
        return _Hit(NetworkFailure(ErrorKind.INVALID_ENDPOINT, deterministic=True, evidence=evidence))
    if code in (codes.eai_noname, codes.eai_nodata, codes.wsano_data):
        # "No such name" usually means a typo — unless this process has
        # already been answered through the same first hop, in which case the
        # resolver (or the network under it) is failing transiently.
        deterministic = route is None or not route.first_hop_reached
    else:
        deterministic = code == codes.eai_fail
    return _Hit(NetworkFailure(ErrorKind.DNS_FAILED, deterministic, first_hop_evidence=True, evidence=evidence))


def _code_hit(code: int, codes: NetworkCodes, connect_phase: bool, evidence: str, *, windows: bool) -> _Hit | None:
    refused = (codes.win_refused,) if windows else (codes.econnrefused,)
    no_route = (codes.win_netunreach,) if windows else (codes.enetunreach, codes.enetdown)
    host_unreachable = (codes.win_hostunreach,) if windows else (codes.ehostunreach, codes.ehostdown)
    timed_out = (codes.win_timeout,) if windows else (codes.etimedout,)
    reset = (codes.win_reset, codes.win_aborted) if windows else (codes.econnreset, codes.econnaborted)
    if code == codes.eafnosupport:
        # Alone, no stack for the address family is no route to the address.
        return _Hit(NetworkFailure(ErrorKind.NO_ROUTE, first_hop_evidence=True, evidence=evidence), _REACH_ABSENT)
    if code in refused:
        return _Hit(
            NetworkFailure(ErrorKind.CONNECTION_REFUSED, first_hop_evidence=True, evidence=evidence), _REACH_REFUSED
        )
    if code in no_route:
        return _Hit(NetworkFailure(ErrorKind.NO_ROUTE, first_hop_evidence=True, evidence=evidence), _REACH_NO_ROUTE)
    if code in host_unreachable:
        return _Hit(
            NetworkFailure(ErrorKind.HOST_UNREACHABLE, first_hop_evidence=True, evidence=evidence),
            _REACH_HOST_UNREACHABLE,
        )
    if not windows and code == codes.eaddrnotavail:
        return _Hit(
            NetworkFailure(ErrorKind.HOST_UNREACHABLE, first_hop_evidence=True, evidence=evidence),
            _REACH_ADDR_NOT_AVAILABLE,
        )
    if code in timed_out:
        if connect_phase:
            return _Hit(
                NetworkFailure(ErrorKind.CONNECT_TIMEOUT, timeout_phase="connect", evidence=evidence), _REACH_TIMEOUT
            )
        return _Hit(NetworkFailure(ErrorKind.CONNECTION_LOST, evidence=evidence), _REACH_TIMEOUT)
    if code in reset:
        return _reset(evidence)
    return None


_TYPE_LEAVES: tuple[tuple[type[BaseException], Callable[[bool], _Hit]], ...] = (
    (
        ConnectionRefusedError,
        lambda _phase: _Hit(
            NetworkFailure(ErrorKind.CONNECTION_REFUSED, first_hop_evidence=True, evidence="ConnectionRefusedError"),
            _REACH_REFUSED,
        ),
    ),
    # The peer tore down a connection that existed, in any phase.
    (ConnectionResetError, lambda _phase: _reset("ConnectionResetError")),
    (ConnectionAbortedError, lambda _phase: _reset("ConnectionAbortedError")),
    (BrokenPipeError, lambda _phase: _reset("BrokenPipeError")),
)


def _socket_leaf(exc: BaseException, codes: NetworkCodes | None, connect_phase: bool) -> _Hit | None:
    if not isinstance(exc, OSError):
        return None
    if codes is not None:
        winerror = getattr(exc, "winerror", None) if codes.windows else None
        if isinstance(winerror, int) and (
            hit := _code_hit(winerror, codes, connect_phase, f"winerror {winerror}", windows=True)
        ):
            return hit
        if isinstance(exc.errno, int) and (
            hit := _code_hit(exc.errno, codes, connect_phase, f"errno {exc.errno}", windows=False)
        ):
            return hit
    if connect_phase and isinstance(exc, TimeoutError):
        # Only a timeout under a connect wrapper is a network fact; elsewhere
        # it may be any ``asyncio.timeout`` in Chrys itself.
        return _Hit(
            NetworkFailure(ErrorKind.CONNECT_TIMEOUT, timeout_phase="connect", evidence="TimeoutError"), _REACH_TIMEOUT
        )
    return next((build(connect_phase) for cls, build in _TYPE_LEAVES if isinstance(exc, cls)), None)


_TIMEOUT_TYPES: dict[str, tuple[ErrorKind, TimeoutPhase]] = {
    "ConnectTimeout": (ErrorKind.CONNECT_TIMEOUT, "connect"),
    "ReadTimeout": (ErrorKind.READ_TIMEOUT, "read"),
    "WriteTimeout": (ErrorKind.WRITE_TIMEOUT, "write"),
    "PoolTimeout": (ErrorKind.UNKNOWN, "pool"),
}
_LOST_TYPE_NAMES = frozenset({"RemoteProtocolError", "ReadError", "WriteError", "CloseError"})


def _transport_type_leaf(exc: BaseException, connect_phase: bool) -> _Hit | None:
    names = _transport_type_names(exc)
    # By name alone, as the legacy veto always matched them.
    if (type_name := type(exc).__name__) in _DETERMINISTIC_TRANSPORT_TYPE_NAMES:
        verdict = ErrorKind.INVALID_ENDPOINT if type_name in _INVALID_ENDPOINT_TYPE_NAMES else ErrorKind.UNKNOWN
        return _Hit(NetworkFailure(verdict, deterministic=True, evidence=f"type {type_name}"))
    for name, (kind, phase) in _TIMEOUT_TYPES.items():
        if name in names:
            return _Hit(NetworkFailure(kind, timeout_phase=phase, evidence=f"type {name}"), _REACH_TIMEOUT)
    if names & _LOST_TYPE_NAMES:
        return _lost(connect_phase, f"type {type(exc).__name__}")
    anyio_names = {cls.__name__ for cls in type(exc).__mro__ if cls.__module__.partition(".")[0] == "anyio"}
    if "EndOfStream" in anyio_names or ("BrokenResourceError" in anyio_names and exc.__cause__ is None):
        return _lost(connect_phase, f"type {type(exc).__name__}")
    return None


def _leaf(exc: BaseException, codes: NetworkCodes | None, route: RouteFacts | None, connect_phase: bool) -> _Hit | None:
    return (
        _proxy_leaf(exc)
        or _tls_leaf(exc, connect_phase)
        or _dns_leaf(exc, codes, route)
        or _socket_leaf(exc, codes, connect_phase)
        or _transport_type_leaf(exc, connect_phase)
    )


def _aggregate(members: Sequence[BaseException], hits: Sequence[_Hit | None]) -> _Hit | None:
    """Name a group of attempts by the one that got furthest.

    An unrecognized attempt still votes, as a failed connection that is its
    own leaf.  Attempts in an absent address family vote only when every
    attempt was one.
    """
    if not any(hits):
        return None
    attempts = [
        hit
        if hit is not None
        else _Hit(NetworkFailure(ErrorKind.CONNECTION_FAILED, evidence="unrecognized attempt"), leaf=member)
        for member, hit in zip(members, hits, strict=True)
    ]
    voters = [hit for hit in attempts if hit.reach != _REACH_ABSENT] or attempts
    # No route ranks lowest, so it names the group only when every voter had none.
    furthest = max(voters, key=lambda hit: hit.reach)
    failure = NetworkFailure(
        furthest.failure.kind,
        deterministic=all(hit.failure.deterministic for hit in voters),
        timeout_phase=furthest.failure.timeout_phase,
        first_hop_evidence=all(hit.failure.first_hop_evidence for hit in voters),
        evidence=f"group[{', '.join(hit.failure.evidence for hit in attempts)}]",
    )
    return _Hit(failure, furthest.reach, furthest.leaf)


def _search(
    exc: BaseException,
    codes: NetworkCodes | None,
    route: RouteFacts | None,
    connect_phase: bool,
    seen: set[int],
) -> _Hit | None:
    if id(exc) in seen:
        return None
    seen.add(id(exc))
    if hit := _leaf(exc, codes, route, connect_phase):
        return replace(hit, leaf=exc)
    if has_response_status(exc):
        # This request got a response: nothing deeper is a connection failure.
        return None
    phase = connect_phase or _is_connect_wrapper(exc)
    if isinstance(exc, BaseExceptionGroup):
        if exc.__cause__ is not None and (hit := _search(exc.__cause__, codes, route, phase, seen)):
            return hit
        return _aggregate(exc.exceptions, [_search(member, codes, route, phase, seen) for member in exc.exceptions])
    for child in explicit_children(exc):
        if hit := _search(child, codes, route, phase, seen):
            return hit
    return None


def _named_connection_loss(explicit: Iterable[BaseException]) -> bool:
    return any(any(phrase in _clean_exception_text(node).lower() for phrase in RETRYABLE_PHRASES) for node in explicit)


def classify_connection_failure(
    exc: BaseException, route: RouteFacts | None, *, codes: NetworkCodes | None = None
) -> NetworkFailure | None:
    """Return what *exc* says about a failed connection, or None when it isn't one.

    *codes* defaults to the table of the platform this interpreter was built
    for (the C library's constants follow ``sys.platform``); tests pass
    another platform's table.
    """
    table = codes if codes is not None else codes_for(sys.platform)
    if hit := _search(exc, table, route, False, set()):
        return hit.failure
    explicit = tuple(iter_explicit_graph(exc))
    if any(_is_connect_wrapper(node) for node in explicit):
        return NetworkFailure(ErrorKind.CONNECTION_FAILED, evidence="connect phase")
    if _named_connection_loss(explicit):
        return NetworkFailure(ErrorKind.CONNECTION_LOST, evidence="phrase")
    return None


def deciding_leaf(exc: BaseException) -> BaseException | None:
    """Return the exception below *exc* that names its connection failure.

    In a group of attempts it is the attempt that got furthest, the one whose
    kind names the group.  The route never changes which leaf decides, so the
    search runs without one.
    """
    hit = _search(exc, codes_for(sys.platform), None, False, set())
    return hit.leaf if hit is not None else None


def is_deterministic_connection_error(e: BaseException) -> bool:
    """Return whether *e* is a connection failure no retry of the same request can fix.

    Reads the route snapshot on the failed request, so the SDK's own retry
    guard and Chrys's outer lanes reach the same verdict.
    """
    failure = classify_connection_failure(e, route_of(iter_explicit_graph(e)))
    return failure is not None and failure.deterministic
