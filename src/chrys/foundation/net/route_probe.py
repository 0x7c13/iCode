# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Whether this machine's routing table has a way out at all.

Only adds a "seems offline" hint to a network error the classifier already
named; it never changes a kind or a retry verdict.  A UDP ``connect()`` asks
the kernel for a route and sends nothing, so the probe costs two syscalls
and may run on the event loop.
"""

from __future__ import annotations

import logging
import socket
import sys
from typing import Literal

from chrys.foundation.errors.network import NetworkCodes, codes_for

from .address_class import is_restricted_address, literal_address, names_this_host

logger = logging.getLogger(__name__)

# RFC 5737 and RFC 3849 documentation addresses: routed like any public
# address, owned by no one.
_PROBES: tuple[tuple[socket.AddressFamily, tuple[str, int]], ...] = (
    (socket.AF_INET, ("192.0.2.1", 9)),
    (socket.AF_INET6, ("2001:db8::1", 9)),
)


# A family this machine has no stack for (IPv6 on a host without it) says
# nothing about the network.
type _Verdict = bool | Literal["absent"] | None


def _no_route_codes(codes: NetworkCodes) -> frozenset[int]:
    found = {codes.enetunreach, codes.enetdown, codes.ehostunreach, codes.eaddrnotavail}
    if codes.windows:
        found |= {codes.win_netunreach, codes.win_hostunreach}
    return frozenset(found)


def _family_has_route(family: socket.AddressFamily, address: tuple[str, int], codes: NetworkCodes) -> _Verdict:
    try:
        with socket.socket(family, socket.SOCK_DGRAM) as probe:
            probe.connect(address)
    except OSError as exc:
        answered = {exc.errno, getattr(exc, "winerror", None)}
        if codes.eafnosupport in answered:
            return "absent"
        return False if answered & _no_route_codes(codes) else None
    return True


def default_route_available() -> bool | None:
    """Return whether either address family has a route to the internet; None when that can't be told.

    False needs every family this machine has to answer "no route" (network
    or host unreachable, network down, no usable source address); a family
    it has no stack for doesn't count, and with none left the answer is
    None.  Any other failure, including one from a test's egress guard, is
    None.  A full
    tunnel (VPN, TUN-mode proxy) makes this True while offline: the hint is
    then missing, never wrong.
    """
    try:
        codes = codes_for(sys.platform)
        if codes is None:
            return None
        verdicts: list[_Verdict] = []
        for family, address in _PROBES:
            verdict = _family_has_route(family, address, codes)
            if verdict is True:
                return True
            verdicts.append(verdict)
    except Exception:
        logger.debug("Default route probe failed", exc_info=True)
        return None
    present = [verdict for verdict in verdicts if verdict != "absent"]
    return False if present and all(verdict is False for verdict in present) else None


def is_local_target(host: str) -> bool:
    """Return whether *host* is this machine or on a private network, where the internet's route says nothing.

    Covers ``localhost`` names, mDNS ``.local`` names, and loopback, private,
    link-local, ULA and other non-global address literals.
    """
    name = host.rstrip(".").lower()
    if names_this_host(name) or name.endswith(".local"):
        return True
    address = literal_address(name)
    return address is not None and is_restricted_address(address)
