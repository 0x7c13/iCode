# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Deterministic HTML conversion; run outside the async event loop."""

from __future__ import annotations

import codecs
from contextlib import suppress
from email.message import Message

from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
from markdownify import markdownify

from chrys.service.tools.builtins.web.http import WebError

# Codecs Python decodes bytes with that no page is written in. A declared
# ``charset=punycode`` would otherwise cost time quadratic in the body (about a
# minute for 2 MiB, and the job cannot be cancelled), and the escape codecs
# rewrite the text rather than decode it.
_NOT_A_CHARSET = frozenset({"punycode", "idna", "unicode-escape", "raw-unicode-escape", "undefined"})


def _declared_codec(charset: str | None) -> str | None:
    if not charset:
        return None
    try:
        name = codecs.lookup(charset).name
    except LookupError:
        return None
    return None if name in _NOT_A_CHARSET else name


def convert(raw: bytes, content_type: str) -> str:
    header = Message()
    header["content-type"] = content_type
    mime = header.get_content_type()
    if not (mime.startswith("text/") or mime in {"application/json", "application/xml", "application/xhtml+xml"}):
        raise WebError("unsupported_content_type")
    charset = _declared_codec(header.get_content_charset())
    text = None
    if charset:
        with suppress(LookupError, UnicodeError):
            text = raw.decode(charset)
    if text is None:
        detected = from_bytes(raw).best()
        text = str(detected) if detected is not None else raw.decode("utf-8", errors="replace")
    if mime not in {"text/html", "application/xhtml+xml"}:
        return text
    try:
        soup = BeautifulSoup(text, "html.parser")
        for node in soup.find_all(["script", "style", "noscript", "iframe", "svg", "form", "template"]):
            node.decompose()
        return markdownify(str(soup), heading_style="ATX")
    except Exception as err:
        raise WebError("convert_failed") from err
