# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hostname and URL normalization shared by the web settings and the web tools."""

from __future__ import annotations

import pytest

from chrys.foundation.net.url import hostname, normalize_origin, normalize_url


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Example.COM.", "example.com"),
        ("faß.de", "xn--fa-hia.de"),
        ("corp_proxy", "corp_proxy"),
        ("_svc.Corp-Net.example", "_svc.corp-net.example"),
        ("::1", "::1"),
    ],
)
def test_hostname_normalizes_dns_names_including_underscore_labels(value, expected):
    assert hostname(value) == expected


@pytest.mark.parametrize(
    "value", ["-bad_label.example", "bad_label-.example", "ünï_code.example", "a_" + "b" * 62 + ".example"]
)
def test_hostname_rejects_malformed_underscore_labels(value):
    with pytest.raises(ValueError):
        hostname(value)


@pytest.mark.parametrize("control", ["\x00", "\x1b", "\x7f", "\x85", "\x9b"])
def test_normalize_url_rejects_c0_and_c1_controls(control):
    """C1 controls act as terminal escapes once a URL reaches a card or hyperlink."""
    with pytest.raises(ValueError, match="characters"):
        normalize_url(f"https://example.com/{control}31m")


@pytest.mark.parametrize("url", ["https://example.com:0/", "https://example.com:65536/"])
def test_normalize_url_rejects_out_of_range_ports(url):
    with pytest.raises(ValueError):
        normalize_url(url)


def test_normalize_origin_accepts_underscore_proxy_hosts():
    assert normalize_origin("http://corp_proxy:3128") == "http://corp_proxy:3128"


_RLO, _ZWSP, _IDEOGRAPHIC_SPACE, _LINE_SEPARATOR = chr(0x202E), chr(0x200B), chr(0x3000), chr(0x2028)


@pytest.mark.parametrize("invisible", [_RLO, _ZWSP, _IDEOGRAPHIC_SPACE, _LINE_SEPARATOR])
def test_normalize_url_percent_encodes_characters_that_reorder_or_hide_text(invisible):
    """A bidi override or an invisible space would change how the URL reads wherever it is shown."""
    encoded = "".join(f"%{byte:02X}" for byte in invisible.encode())

    url = normalize_url(f"https://example.com/a{invisible}b?q={invisible}")

    assert url == f"https://example.com/a{encoded}b?q={encoded}"


def test_normalize_url_keeps_visible_non_ascii_spelling():
    assert normalize_url("https://example.com/中文?q=ü") == "https://example.com/中文?q=ü"
