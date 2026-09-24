# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Conservative public-address classification, shared by web capabilities."""

from __future__ import annotations

import socket
from ipaddress import IPv4Address, IPv6Address, ip_address, ip_network

_PROTOCOL_ASSIGNMENTS = ip_network("192.0.0.0/24")
# RFC 6052's well-known NAT64 prefix, which carries only global IPv4 addresses.
# The local-use 64:ff9b:1::/48 stays restricted: its IPv4 half may be private.
_NAT64_WELL_KNOWN = ip_network("64:ff9b::/96")


def is_restricted_address(value: str) -> bool:
    """Reject non-global, multicast and IPv4-mapped restricted destinations."""
    address = ip_address(value)
    if isinstance(address, IPv4Address) and address in _PROTOCOL_ASSIGNMENTS:
        return True
    if isinstance(address, IPv6Address):
        if address.ipv4_mapped is not None:
            return is_restricted_address(str(address.ipv4_mapped))
        # DNS64 answers with this prefix on an IPv6-only network, where every
        # site resolves into it: it is as public as the IPv4 address it embeds.
        if address in _NAT64_WELL_KNOWN:
            return is_restricted_address(str(IPv4Address(int(address) & 0xFFFFFFFF)))
        # Transition addresses must not smuggle a private IPv4 destination.
        if address.sixtofour is not None or address.teredo is not None:
            return True
        # Deprecated site-local space (fec0::/10) still routes on some networks,
        # yet Python classifies it as global.
        if address.is_site_local:
            return True
    return not address.is_global or address.is_multicast or address.is_reserved


def literal_address(host: str) -> str | None:
    """Return the address *host* names without DNS, or None for a name DNS must answer.

    Besides IP literals this reads the numeric IPv4 forms a resolver accepts in
    place of a name (``2130706433``, ``0x7f.1``, ``127.1``), so a name handed
    to someone else's resolver cannot reach an address it hides.
    """
    try:
        return str(ip_address(host))
    except ValueError:
        pass
    try:
        return socket.inet_ntoa(socket.inet_aton(host))
    except OSError, UnicodeError, ValueError:
        return None


def names_this_host(host: str) -> bool:
    """RFC 6761: ``localhost`` and every name under it resolve to loopback."""
    return host == "localhost" or host.endswith(".localhost")
