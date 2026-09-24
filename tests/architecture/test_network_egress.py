# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Counterexample tests for the loopback-only network guard."""

from __future__ import annotations

import asyncio
import socket
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.support.ci import CI_LINUX_ONLY
from tests.support.network_guard import (
    begin_test_network_scope,
    end_test_network_scope,
    guarded_sendmsg,
    is_loopback_destination,
    require_allowed_resolution,
)

# Platform-independent guard self-checks: the Linux CI job covers them. The
# collection-time probe below still runs on every platform — skip markers
# gate test execution, not module import.
pytestmark = CI_LINUX_ONLY

_EXTERNAL_TEST_ADDRESS = ("203.0.113.1", 80)

# Executed at import — i.e. during collection — proving the guard is already
# installed by pytest_configure before any fixture machinery runs. Without the
# guard this would be a real (fast-failing) egress attempt.
_COLLECTION_TIME_BLOCK: BaseException | None
try:
    socket.create_connection(_EXTERNAL_TEST_ADDRESS, timeout=0.001)
    _COLLECTION_TIME_BLOCK = None
except BaseException as exc:  # record whatever interception produced
    _COLLECTION_TIME_BLOCK = exc
_LOCAL_DESTINATIONS: list[tuple[int, object]] = [
    (socket.AF_INET, ("127.255.1.2", 1)),
    (socket.AF_INET6, ("::1", 1, 0, 0)),
    (socket.AF_INET6, ("::ffff:127.8.9.10", 1, 0, 0)),
]
_AF_UNIX = getattr(socket, "AF_UNIX", None)
if _AF_UNIX is not None:
    _LOCAL_DESTINATIONS.append((_AF_UNIX, "/tmp/chrys-test.sock"))


def test_direct_non_loopback_socket_connect_guard_goes_red() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match=r"Network egress blocked.*203.0.113.1"):
            sock.connect(_EXTERNAL_TEST_ADDRESS)
    finally:
        sock.close()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="ProactorEventLoop connects via IOCP ConnectEx, bypassing socket.socket.connect entirely",
)
@pytest.mark.asyncio
async def test_asyncio_non_loopback_open_connection_guard_goes_red() -> None:
    with pytest.raises(RuntimeError, match=r"Network egress blocked.*203.0.113.1"):
        await asyncio.open_connection(*_EXTERNAL_TEST_ADDRESS)


def test_non_loopback_connect_ex_guard_goes_red() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match=r"Network egress blocked.*203.0.113.1"):
            sock.connect_ex(_EXTERNAL_TEST_ADDRESS)
    finally:
        sock.close()


def test_non_loopback_create_connection_guard_goes_red() -> None:
    with pytest.raises(RuntimeError, match=r"Network egress blocked.*203.0.113.1"):
        socket.create_connection(_EXTERNAL_TEST_ADDRESS)


def test_loopback_connect_reaches_the_os_instead_of_the_guard() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        try:
            sock.connect(("127.0.0.1", 9))
        except RuntimeError as exc:
            pytest.fail(f"loopback was blocked: {exc}")
        except OSError:
            # Expected when no discard server is listening (or this sandbox
            # denies sockets): reaching OSError proves the guard allowed it.
            pass
    finally:
        sock.close()


@pytest.mark.parametrize(
    ("family", "address"),
    _LOCAL_DESTINATIONS,
)
def test_loopback_and_unix_allowlist_is_green(family: int, address: object) -> None:
    assert is_loopback_destination(family, address)


def test_collection_time_egress_is_blocked() -> None:
    assert isinstance(_COLLECTION_TIME_BLOCK, RuntimeError), f"collection-time attempt saw {_COLLECTION_TIME_BLOCK!r}"
    assert "Network egress blocked" in str(_COLLECTION_TIME_BLOCK)


def test_udp_sendto_non_loopback_guard_goes_red() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(RuntimeError, match=r"Network egress blocked.*203.0.113.1"):
            sock.sendto(b"probe", _EXTERNAL_TEST_ADDRESS)
    finally:
        sock.close()


def test_udp_sendto_loopback_reaches_the_os_instead_of_the_guard() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        try:
            sock.sendto(b"probe", ("127.0.0.1", 9))
        except RuntimeError as exc:
            pytest.fail(f"loopback UDP was blocked: {exc}")
        except OSError:
            # Sandboxes may deny the syscall; reaching it proves allowance.
            pass
    finally:
        sock.close()


def test_sendmsg_guard_factory_blocks_external_destination() -> None:
    guard = guarded_sendmsg(lambda sock, *args: "sent")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        with pytest.raises(RuntimeError, match=r"Network egress blocked"):
            guard(sock, [b"probe"], [], 0, _EXTERNAL_TEST_ADDRESS)
        assert guard(sock, [b"probe"]) == "sent", "connected sends are policed by the connect guard, not sendmsg"
    finally:
        sock.close()


def test_non_local_dns_resolution_guard_goes_red() -> None:
    with pytest.raises(RuntimeError, match=r"DNS resolution blocked.*chrys-egress-guard\.invalid"):
        socket.getaddrinfo("chrys-egress-guard.invalid", 80)


@pytest.mark.asyncio
async def test_dns_guard_blocks_in_executor_threads_too() -> None:
    """asyncio resolves names via run_in_executor — the guard must hold there."""
    loop = asyncio.get_running_loop()
    with pytest.raises(RuntimeError, match=r"DNS resolution blocked"):
        await loop.run_in_executor(None, socket.getaddrinfo, "chrys-egress-guard.invalid", 80)


def test_integration_allowance_is_visible_across_threads() -> None:
    """run_in_executor does not propagate ContextVars; the allowance must not
    depend on them (a marked integration test's asyncio DNS lookups run in
    executor threads)."""
    token = begin_test_network_scope(integration=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(require_allowed_resolution, "chrys-egress-guard.invalid").result()
    finally:
        end_test_network_scope(token)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(require_allowed_resolution, "chrys-egress-guard.invalid")
        with pytest.raises(RuntimeError, match=r"DNS resolution blocked"):
            future.result()


@pytest.mark.integration
def test_integration_marker_allows_resolution_from_executor_threads() -> None:
    """Full chain: marker → protocol wrapper → process flag → worker thread.
    Deselected by CI's marker expression; runs in the unfiltered local suite.
    No real DNS happens — only the policy check runs in the thread."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(require_allowed_resolution, "chrys-egress-guard.invalid").result()


def test_gethostbyname_guard_goes_red() -> None:
    with pytest.raises(RuntimeError, match=r"DNS resolution blocked"):
        socket.gethostbyname("chrys-egress-guard.invalid")


def test_numeric_external_host_resolution_is_not_dns() -> None:
    # Numeric addresses generate no DNS traffic; whether they may be REACHED
    # is the connect guard's job (proven red above).
    assert socket.getaddrinfo("203.0.113.1", 80, type=socket.SOCK_STREAM)


def test_localhost_resolution_stays_green() -> None:
    assert socket.getaddrinfo("localhost", 80, type=socket.SOCK_STREAM)
