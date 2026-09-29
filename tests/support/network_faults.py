# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Network fault injection on the running event loop.

httpx's default transport resolves and connects through anyio, whose asyncio
backend calls the running loop's ``getaddrinfo`` and, via
``create_connection``, its ``sock_connect``.  :func:`network_faults` shadows
those two methods on the running loop instance only — stdlib modules stay
untouched — so a real SDK client produces the exact exception chain a real
DNS or connect failure would, without any packet leaving the machine.

Hosts and addresses without a rule fall through to the real loop methods
(and so to the suite's loopback-only egress guard).
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import socket
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

# Documentation addresses (RFC 5737 / RFC 3849): only ever reached through a rule.
INJECTED_V4 = "198.51.100.7"
INJECTED_V4_B = "198.51.100.8"
INJECTED_V6 = "2001:db8::7"
_ABSENT = object()

_GAI_MESSAGES = {
    socket.EAI_NONAME: "nodename nor servname provided, or not known",
    socket.EAI_AGAIN: "Temporary failure in name resolution",
    socket.EAI_FAIL: "Non-recoverable failure in name resolution",
}


def gaierror(code: int) -> socket.gaierror:
    """Return the ``gaierror`` the resolver raises for *code*."""
    return socket.gaierror(code, _GAI_MESSAGES.get(code, "name resolution failed"))


def socket_error_text(code: int) -> str:
    """``[Errno N] words``, with the words this system has for *code*.

    On Windows, ``errno``'s socket names (``ECONNREFUSED``, ``ENETUNREACH`` …)
    are Winsock codes, which only ``FormatError`` knows: the C runtime's
    strerror calls them "Unknown error".
    """
    if sys.platform == "win32" and 10000 <= code < 12000:
        import ctypes

        return f"[Errno {code}] {ctypes.FormatError(code).strip()}"
    return f"[Errno {code}] {os.strerror(code)}"


def refused(address: tuple[Any, ...]) -> OSError:
    return ConnectionRefusedError(errno.ECONNREFUSED, f"Connect call failed {address!r}")


def os_error(code: int) -> Callable[[tuple[Any, ...]], OSError]:
    """Return a connect-failure factory raising ``OSError(code)``."""

    def build(address: tuple[Any, ...]) -> OSError:
        return OSError(code, f"Connect call failed {address!r}")

    return build


class SimulatedWindowsError(OSError):
    """An ``OSError`` carrying ``winerror`` on any OS, as Windows sockets raise it."""

    def __init__(self, errno_code: int, message: str, winerror: int) -> None:
        super().__init__(errno_code, message)
        self.winerror = winerror


def win_proactor_reset(winerror: int, words: str) -> Callable[[tuple[Any, ...]], OSError]:
    """Return a factory for the reset Windows' proactor re-raises: ``ConnectionResetError(*exc.args)``.

    For ERROR_NETNAME_DELETED and ERROR_OPERATION_ABORTED it drops the
    ``winerror``; what stays is the C runtime's stand-in ``errno`` EINVAL and
    the system's *words*.
    """

    def build(_address: tuple[Any, ...]) -> OSError:
        return ConnectionResetError(*SimulatedWindowsError(errno.EINVAL, words, winerror).args)

    return build


def win_iocp_error(winerror: int) -> Callable[[tuple[Any, ...]], OSError]:
    """Return a factory for the IOCP shape: ``errno`` EINVAL, the real code in ``winerror``."""

    def build(address: tuple[Any, ...]) -> OSError:
        return SimulatedWindowsError(errno.EINVAL, f"Connect call failed {address!r}", winerror)

    return build


@dataclass
class NetworkFaults:
    """Rules and call records for one :func:`network_faults` scope."""

    resolve_rules: dict[str, Callable[[], list[tuple[Any, ...]] | BaseException]] = field(default_factory=dict)
    connect_rules: dict[str, Callable[[tuple[Any, ...]], BaseException | None]] = field(default_factory=dict)
    hanging: set[str] = field(default_factory=set)
    resolve_calls: list[str] = field(default_factory=list)
    connect_attempts: list[tuple[Any, ...]] = field(default_factory=list)

    def fail_resolve(self, host: str, code: int) -> None:
        self.resolve_rules[host] = lambda: gaierror(code)

    def resolve_to(self, host: str, *addresses: str) -> None:
        def answer() -> list[tuple[Any, ...]]:
            rows: list[tuple[Any, ...]] = []
            for address in addresses:
                if ":" in address:
                    rows.append((socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 0, 0, 0)))
                else:
                    rows.append((socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 0)))
            return rows

        self.resolve_rules[host] = answer

    def fail_connect(self, address: str, build: Callable[[tuple[Any, ...]], BaseException]) -> None:
        self.connect_rules[address] = build

    def refuse(self, address: str) -> None:
        self.connect_rules[address] = refused

    def hang_connect(self, address: str) -> None:
        """Make connects to *address* wait until cancelled (e.g. by a connect timeout)."""
        self.hanging.add(address)


def _with_port(rows: list[tuple[Any, ...]], port: Any) -> list[tuple[Any, ...]]:
    number = int(port) if port not in (None, "") else 0
    return [(*row[:4], (row[4][0], number, *row[4][2:])) for row in rows]


@contextlib.contextmanager
def network_faults() -> Iterator[NetworkFaults]:
    """Install fault rules on the running loop for the ``with`` body."""
    loop = asyncio.get_running_loop()
    faults = NetworkFaults()
    real_getaddrinfo = loop.getaddrinfo
    real_sock_connect = loop.sock_connect

    async def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        name = host.decode() if isinstance(host, bytes) else host
        rule = faults.resolve_rules.get(name) if isinstance(name, str) else None
        if rule is None:
            return await real_getaddrinfo(host, port, *args, **kwargs)
        faults.resolve_calls.append(name)
        answer = rule()
        if isinstance(answer, BaseException):
            raise answer
        return _with_port(answer, port)

    async def sock_connect(sock: socket.socket, address: Any) -> None:
        host = address[0]
        if host in faults.hanging:
            faults.connect_attempts.append(address)
            await asyncio.Event().wait()
        rule = faults.connect_rules.get(host)
        if rule is None:
            return await real_sock_connect(sock, address)
        faults.connect_attempts.append(address)
        failure = rule(address)
        if failure is not None:
            raise failure
        return None

    shadows = {"getaddrinfo": getaddrinfo, "sock_connect": sock_connect}
    saved = {name: vars(loop).get(name, _ABSENT) for name in shadows}
    vars(loop).update(shadows)
    try:
        yield faults
    finally:
        for name, value in saved.items():
            if value is _ABSENT:
                vars(loop).pop(name, None)
            else:
                vars(loop)[name] = value
