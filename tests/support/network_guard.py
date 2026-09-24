# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Loopback-only network policy used by the pytest suite.

Guarded seams (installed at ``pytest_configure`` time so module-level code
imported during collection is covered too): ``socket.socket.connect``,
``socket.socket.connect_ex``, ``socket.socket.sendto``,
``socket.socket.sendmsg`` (where available), ``socket.create_connection``,
and name resolution via ``socket.getaddrinfo`` / ``socket.gethostbyname`` /
``socket.gethostbyname_ex``.

Known limits — this is a tripwire, not a jail:
- Windows asyncio's ProactorEventLoop connects through IOCP ``ConnectEx``
  without calling ``socket.socket.connect``, so asyncio-initiated egress is
  not intercepted there.
- Child processes (MCP stdio servers, pytester subprocesses) inherit no
  patches and are unguarded.
- Reverse DNS (``getnameinfo``/``gethostbyaddr``) and raw sockets are not
  wrapped.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

# Plain module state, deliberately NOT a ContextVar: asyncio resolves names by
# running socket.getaddrinfo in the default executor, and run_in_executor does
# not propagate context, so a ContextVar allowance would read as False inside
# those threads and block marked integration tests. One test runs at a time
# per xdist worker process, so process scope is the correct scope — and it is
# visible to every thread the test spawns.
_INTEGRATION_EGRESS_ALLOWED: bool = False


def begin_test_network_scope(*, integration: bool) -> bool:
    """Set the current test's integration escape hatch; returns a restore token."""
    global _INTEGRATION_EGRESS_ALLOWED
    previous = _INTEGRATION_EGRESS_ALLOWED
    _INTEGRATION_EGRESS_ALLOWED = integration
    return previous


def end_test_network_scope(token: bool) -> None:
    """Restore the preceding network-policy state."""
    global _INTEGRATION_EGRESS_ALLOWED
    _INTEGRATION_EGRESS_ALLOWED = token


def _as_host_ip(host: Any) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Parse *host* as a numeric address, tolerating bytes and scope ids."""
    if isinstance(host, (bytes, bytearray)):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return None
    if not isinstance(host, str):
        return None
    try:
        return ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return None


def _is_local_host(host: Any) -> bool:
    """Return whether *host* names this machine without external routing."""
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(host, str):
        return False
    if host == "" or host.casefold() in {"localhost", "localhost."}:
        return True
    ip = _as_host_ip(host)
    if ip is None:
        return False
    if ip.is_loopback or ip.is_unspecified:
        return True
    return isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None and ip.ipv4_mapped.is_loopback


def is_loopback_destination(family: int | None, address: Any) -> bool:
    """Return whether *address* is an explicitly allowed local destination."""
    af_unix = getattr(socket, "AF_UNIX", None)
    if af_unix is not None and family == af_unix:
        return True
    if not isinstance(address, tuple) or not address:
        return False
    return _is_local_host(address[0])


def require_allowed_destination(family: int | None, address: Any) -> None:
    """Raise before a syscall for non-loopback, non-integration egress."""
    if _INTEGRATION_EGRESS_ALLOWED or is_loopback_destination(family, address):
        return
    raise RuntimeError(
        f"Network egress blocked for destination {address!r}; "
        "mark the test with @pytest.mark.integration if external network access is intentional"
    )


def require_allowed_resolution(host: Any) -> None:
    """Raise before non-local NAME resolution; numeric hosts pass untouched.

    A numeric address never generates DNS traffic — the connect/sendto guards
    police whether it may actually be reached.
    """
    if _INTEGRATION_EGRESS_ALLOWED or _is_local_host(host) or _as_host_ip(host) is not None:
        return
    raise RuntimeError(
        f"DNS resolution blocked for host {host!r}; "
        "mark the test with @pytest.mark.integration if external network access is intentional"
    )


def guarded_socket_connect(original: Any):
    """Wrap ``socket.socket.connect`` or ``connect_ex`` with the policy."""

    def _guard(sock: socket.socket, address: Any):
        require_allowed_destination(sock.family, address)
        return original(sock, address)

    return _guard


def guarded_sendto(original: Any):
    """Wrap ``socket.socket.sendto`` — unconnected UDP egress."""

    def _guard(sock: socket.socket, *args: Any):
        # sendto(data, address) or sendto(data, flags, address)
        if len(args) >= 2:
            require_allowed_destination(sock.family, args[-1])
        return original(sock, *args)

    return _guard


def guarded_sendmsg(original: Any):
    """Wrap ``socket.socket.sendmsg`` — unconnected UDP egress."""

    def _guard(sock: socket.socket, *args: Any):
        # sendmsg(buffers[, ancdata[, flags[, address]]])
        if len(args) >= 4 and args[3] is not None:
            require_allowed_destination(sock.family, args[3])
        return original(sock, *args)

    return _guard


def guarded_create_connection(original: Any):
    """Wrap ``socket.create_connection`` with the policy."""

    def _guard(address: Any, *args: Any, **kwargs: Any):
        require_allowed_destination(None, address)
        return original(address, *args, **kwargs)

    return _guard


def guarded_getaddrinfo(original: Any):
    """Wrap ``socket.getaddrinfo`` — blocks non-local name lookups."""

    def _guard(host: Any, port: Any, *args: Any, **kwargs: Any):
        require_allowed_resolution(host)
        return original(host, port, *args, **kwargs)

    return _guard


def guarded_hostname_resolution(original: Any):
    """Wrap ``socket.gethostbyname``/``gethostbyname_ex`` with the policy."""

    def _guard(hostname: Any):
        require_allowed_resolution(hostname)
        return original(hostname)

    return _guard
