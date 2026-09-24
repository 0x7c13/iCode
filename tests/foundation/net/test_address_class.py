# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pure address classification covers IPv4, IPv6, transition encodings and hosts named without DNS."""

from __future__ import annotations

import pytest

from chrys.foundation.net.address_class import is_restricted_address, literal_address, names_this_host


@pytest.mark.parametrize(
    "address",
    [
        "0.1.2.3",
        "10.0.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "100.100.100.200",
        "172.16.0.1",
        "192.168.1.1",
        "192.0.0.1",
        "198.18.0.1",
        "::",
        "::1",
        "fe80::1",
        "fc00::1",
        "fec0::1",
        "::ffff:127.0.0.1",
        "224.0.0.1",
    ],
)
def test_restricted_addresses(address):
    assert is_restricted_address(address)


def test_protocol_assignment_exceptions_are_still_restricted():
    assert is_restricted_address("192.0.0.9")
    assert is_restricted_address("192.0.0.10")


@pytest.mark.parametrize("address", ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"])
def test_public_addresses(address):
    assert not is_restricted_address(address)


@pytest.mark.parametrize(
    ("address", "restricted"),
    [
        # DNS64's well-known prefix is as public as the IPv4 address it embeds.
        ("64:ff9b::5db8:d70e", False),  # 93.184.215.14
        ("64:ff9b::a00:1", True),  # 10.0.0.1
        ("64:ff9b::7f00:1", True),  # 127.0.0.1
        ("64:ff9b::a9fe:a9fe", True),  # 169.254.169.254
        # The local-use prefix may embed a private address.
        ("64:ff9b:1::5db8:d70e", True),
    ],
)
def test_a_nat64_address_is_classified_by_its_ipv4_half(address, restricted):
    assert is_restricted_address(address) is restricted


@pytest.mark.parametrize(
    "host,address",
    [
        ("127.0.0.1", "127.0.0.1"),
        ("::1", "::1"),
        ("2606:4700:4700:0:0:0:0:1111", "2606:4700:4700::1111"),
        # inet_aton forms: what a proxy's resolver makes of a numeric host.
        ("2130706433", "127.0.0.1"),
        ("0x7f.1", "127.0.0.1"),
        ("127.1", "127.0.0.1"),
        ("017700000001", "127.0.0.1"),
        ("example.com", None),
        ("1.2.3.4.example", None),
        ("localhost", None),
        ("xn--fsq.example", None),
    ],
)
def test_literal_address_reads_every_form_that_needs_no_dns(host, address):
    assert literal_address(host) == address


@pytest.mark.parametrize(
    "host,local", [("localhost", True), ("api.localhost", True), ("localhost.example", False), ("notlocalhost", False)]
)
def test_localhost_and_every_name_under_it_name_this_host(host, local):
    assert names_this_host(host) is local
