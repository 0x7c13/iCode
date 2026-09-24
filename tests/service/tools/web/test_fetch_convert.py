# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Decoding a fetched body by the charset its response declares."""

from __future__ import annotations

import pytest

from chrys.service.tools.builtins.web.fetch.convert import convert


def test_a_declared_charset_decodes_the_body() -> None:
    assert convert("café".encode("cp1252"), "text/plain; charset=windows-1252") == "café"


_BACKSLASH = bytes([0x5C])


@pytest.mark.parametrize(
    ("charset", "body"),
    [
        # Decoding punycode costs time quadratic in the body.
        ("punycode", ("\u00fc" * 50).encode("punycode")),
        # The escape codecs would turn an escape written on the page into the character it names.
        ("unicode_escape", b"see " + _BACKSLASH + b"x41 here"),
        ("raw_unicode_escape", b"see " + _BACKSLASH + b"u0041 here"),
    ],
    ids=["punycode", "unicode_escape", "raw_unicode_escape"],
)
def test_a_codec_that_is_not_a_charset_is_not_used_to_decode(charset: str, body: bytes) -> None:
    assert body.decode(charset) != body.decode("ascii")

    assert convert(body, f"text/plain; charset={charset}") == body.decode("ascii")


def test_an_unknown_charset_falls_back_to_detection() -> None:
    assert convert(b"plain text", "text/plain; charset=no-such-charset") == "plain text"
