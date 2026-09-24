# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared external URL policy for Markdown clicks and terminal hyperlinks."""

from __future__ import annotations

from urllib.parse import unquote, urlsplit

_MAX_TERMINAL_LINK_BYTES = 2048


def external_link_target(raw_href: str) -> str | None:
    """Return the encoded URL only when it may be handed to an external opener.

    OSC 8 clicks belong to the terminal and never reach our click handler, so
    destinations must pass the same policy before they enter rendered output.
    Decoding is only for validation: changing encoded delimiters or spaces
    would change the target. Anchors and relative paths stay inside the viewer.
    """
    if any(char.isspace() for char in raw_href):
        return None
    if any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in unquote(raw_href)):
        return None
    try:
        parts = urlsplit(raw_href)
        scheme = parts.scheme.lower()
        if scheme in {"http", "https"}:
            host = unquote(parts.hostname or "")
            if not host or any(char.isspace() or char in "/\\?#@" for char in host):
                return None
            # Accessing port validates both its syntax and its numeric range.
            _ = parts.port
    except ValueError:
        return None
    return raw_href if scheme in {"http", "https", "mailto"} else None


def terminal_link_target(raw_href: str) -> str | None:
    """Bound OSC 8 output without limiting the ordinary click destination."""
    if len(raw_href) > _MAX_TERMINAL_LINK_BYTES:
        return None
    try:
        if len(raw_href.encode("utf-8")) > _MAX_TERMINAL_LINK_BYTES:
            return None
    except UnicodeEncodeError:
        return None
    return external_link_target(raw_href)
