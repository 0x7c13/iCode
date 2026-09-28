# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The default-route probe: False only when both families say "no route", None whenever it can't tell."""

from __future__ import annotations

import errno
import socket
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import pytest

from chrys.foundation.errors.network import NetworkCodes, codes_for
from chrys.foundation.net import route_probe
from chrys.foundation.net.route_probe import default_route_available, is_local_target
from tests.support.network_faults import SimulatedWindowsError


def _table(platform: str) -> NetworkCodes:
    table = codes_for(platform)
    if table is None:
        raise LookupError(platform)
    return table


_CODES = _table(sys.platform)
_NO_ROUTE_FIELDS = ("enetunreach", "enetdown", "ehostunreach", "eaddrnotavail")


def _no_route(field_name: str = "enetunreach") -> OSError:
    return OSError(getattr(_CODES, field_name), "no route")


@dataclass
class _Kernel:
    """Answers each family's UDP ``connect()``: None connects, an exception is raised."""

    answers: dict[int, BaseException | None]
    # An exception raised when the socket is created instead of at connect().
    create_failures: dict[int, BaseException] = field(default_factory=dict)
    probed: list[tuple[int, Any]] = field(default_factory=list)

    def socket(self, family: int, kind: int) -> _Socket:
        assert kind == socket.SOCK_DGRAM
        if family in self.create_failures:
            raise self.create_failures[family]
        return _Socket(self, family)


@dataclass
class _Socket:
    kernel: _Kernel
    family: int

    def __enter__(self) -> _Socket:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def connect(self, address: Any) -> None:
        self.kernel.probed.append((self.family, address))
        if (answer := self.kernel.answers[self.family]) is not None:
            raise answer


@pytest.fixture
def kernel(monkeypatch: pytest.MonkeyPatch) -> _Kernel:
    fake = _Kernel({})
    shadow = ModuleType("socket")
    shadow.__dict__.update(vars(socket), socket=fake.socket)
    monkeypatch.setattr(route_probe, "socket", shadow)
    return fake


@pytest.mark.parametrize("v6_field", _NO_ROUTE_FIELDS)
@pytest.mark.parametrize("v4_field", _NO_ROUTE_FIELDS)
def test_no_route_on_both_families_is_offline(kernel: _Kernel, v4_field: str, v6_field: str) -> None:
    kernel.answers.update({socket.AF_INET: _no_route(v4_field), socket.AF_INET6: _no_route(v6_field)})

    assert default_route_available() is False
    # Documentation addresses only: nothing real is ever named.
    assert kernel.probed == [(socket.AF_INET, ("192.0.2.1", 9)), (socket.AF_INET6, ("2001:db8::1", 9))]


def test_an_ipv4_route_answers_without_asking_ipv6(kernel: _Kernel) -> None:
    kernel.answers.update({socket.AF_INET: None, socket.AF_INET6: _no_route()})

    assert default_route_available() is True
    assert [family for family, _ in kernel.probed] == [socket.AF_INET]


def test_an_ipv6_route_alone_is_enough(kernel: _Kernel) -> None:
    kernel.answers.update({socket.AF_INET: _no_route(), socket.AF_INET6: None})

    assert default_route_available() is True


@pytest.mark.parametrize(
    "v6_failure",
    [
        pytest.param(OSError(errno.EACCES, "denied"), id="other-oserror"),
        pytest.param(RuntimeError("egress blocked"), id="runtime-error"),
    ],
)
def test_any_other_answer_is_unknown(kernel: _Kernel, v6_failure: BaseException) -> None:
    kernel.answers.update({socket.AF_INET: _no_route(), socket.AF_INET6: v6_failure})

    assert default_route_available() is None


@pytest.mark.parametrize("at_connect", [False, True], ids=["socket", "connect"])
def test_a_family_the_machine_lacks_leaves_the_other_to_decide(kernel: _Kernel, at_connect: bool) -> None:
    # A host without IPv6: its IPv4 answer alone decides.
    absent = OSError(_CODES.eafnosupport, "Address family not supported by protocol")
    if at_connect:
        kernel.answers[socket.AF_INET6] = absent
    else:
        kernel.create_failures[socket.AF_INET6] = absent

    for v4_answer, verdict in ((_no_route(), False), (None, True), (OSError(errno.EACCES, "denied"), None)):
        kernel.answers[socket.AF_INET] = v4_answer
        assert default_route_available() is verdict


def test_no_family_at_all_is_unknown(kernel: _Kernel) -> None:
    absent = OSError(_CODES.eafnosupport, "Address family not supported by protocol")
    kernel.create_failures.update({socket.AF_INET: absent, socket.AF_INET6: absent})

    assert default_route_available() is None


def test_a_family_whose_socket_fails_otherwise_is_unknown(kernel: _Kernel) -> None:
    kernel.answers[socket.AF_INET] = _no_route()
    kernel.create_failures[socket.AF_INET6] = OSError(errno.EMFILE, "Too many open files")

    assert default_route_available() is None


def test_windows_reads_a_missing_family_from_winerror(kernel: _Kernel, monkeypatch: pytest.MonkeyPatch) -> None:
    shadow_sys = ModuleType("sys")
    shadow_sys.__dict__.update(vars(sys), platform="win32")
    monkeypatch.setattr(route_probe, "sys", shadow_sys)
    win32 = _table("win32")
    kernel.answers[socket.AF_INET] = SimulatedWindowsError(errno.EINVAL, "no route", win32.win_netunreach)
    kernel.create_failures[socket.AF_INET6] = SimulatedWindowsError(errno.EINVAL, "no IPv6", win32.eafnosupport)

    assert default_route_available() is False


def test_the_test_egress_guard_makes_the_real_probe_unknown() -> None:
    # No shadow: the loopback-only guard raises from connect(), as any guard would.
    assert default_route_available() is None


def test_windows_reads_winerror(kernel: _Kernel, monkeypatch: pytest.MonkeyPatch) -> None:
    shadow_sys = ModuleType("sys")
    shadow_sys.__dict__.update(vars(sys), platform="win32")
    monkeypatch.setattr(route_probe, "sys", shadow_sys)
    win32 = _table("win32")
    kernel.answers.update(
        {
            socket.AF_INET: SimulatedWindowsError(errno.EINVAL, "no route", win32.win_netunreach),
            socket.AF_INET6: OSError(win32.enetunreach, "no route"),
        }
    )

    assert default_route_available() is False


def test_an_unsupported_os_is_unknown(kernel: _Kernel, monkeypatch: pytest.MonkeyPatch) -> None:
    shadow_sys = ModuleType("sys")
    shadow_sys.__dict__.update(vars(sys), platform="freebsd14")
    monkeypatch.setattr(route_probe, "sys", shadow_sys)
    kernel.answers.update({socket.AF_INET: _no_route(), socket.AF_INET6: _no_route()})

    assert default_route_available() is None
    assert kernel.probed == []


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "api.localhost",
        "printer.local",
        "Printer.Local.",
        "127.0.0.1",
        "10.0.0.5",
        "172.16.3.4",
        "192.168.1.2",
        "169.254.1.1",
        "100.64.0.1",
        "::1",
        "fd00::1",
        "fe80::1",
        "2130706433",
    ],
)
def test_local_targets(host: str) -> None:
    assert is_local_target(host) is True


@pytest.mark.parametrize("host", ["api.openai.com", "local.example.com", "8.8.8.8", "2001:4860:4860::8888", "intranet"])
def test_public_targets(host: str) -> None:
    assert is_local_target(host) is False
