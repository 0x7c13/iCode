# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Strict URL and hostname normalization for web egress and its settings.

The settings validators and the web tools share these functions, so a value
the settings panel accepts is exactly a value the tools can dial.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from urllib.parse import quote, urlsplit, urlunsplit

import idna

# DNS permits underscores (service records, docker-compose names, some bucket
# hosts) and browsers accept them; IDNA's STD3 rules do not.
_UNDERSCORE_LABEL = re.compile(r"(?!-)[a-z0-9_-]{1,63}(?<!-)\Z")
# Format characters (bidi overrides, zero-width spaces) and non-ASCII spaces
# reorder or hide part of a URL wherever it is shown or linked. Encoding them
# changes nothing on the wire, where the client percent-encodes them anyway.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Zs", "Zl", "Zp"})


def _encode_invisible(text: str) -> str:
    if text.isascii():
        return text
    return "".join(quote(c, safe="") if unicodedata.category(c) in _INVISIBLE_CATEGORIES else c for c in text)


def hostname(value: str) -> str:
    """Normalize an exact DNS hostname or IP literal; never accept URL syntax."""
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError("Invalid hostname")
    if any(c in value for c in "/@?#\\%*"):
        raise ValueError("Invalid hostname")
    value = value.rstrip(".").lower()
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        pass
    if "_" not in value:
        return idna.encode(value, uts46=True, std3_rules=True).decode("ascii").rstrip(".")
    if not value.isascii():
        raise ValueError("Invalid hostname")
    labels = []
    for label in value.split("."):
        if "_" in label:
            if _UNDERSCORE_LABEL.fullmatch(label) is None:
                raise ValueError("Invalid hostname")
            labels.append(label)
        else:
            labels.append(idna.encode(label, uts46=True, std3_rules=True).decode("ascii"))
    result = ".".join(labels)
    if len(result) > 253:
        raise ValueError("Invalid hostname")
    return result


def normalize_url(value: str, *, limit: int = 2048, endpoint: bool = False) -> str:
    """Keep path/query spelling, normalize the authority, and remove fragments."""
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError("Invalid URL length")
    # C1 controls are rejected with C0: a URL reaches terminal hyperlinks and
    # display text, where U+0080-U+009F act as escape sequences.
    if any(ord(c) <= 32 or 127 <= ord(c) <= 159 for c in value) or "\\" in value:
        raise ValueError("Invalid URL characters")
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname or parts.username is not None:
        raise ValueError("An absolute HTTP(S) URL without credentials is required")
    host = hostname(parts.hostname)
    port = parts.port
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("Invalid port")
    if endpoint and (parts.query or parts.fragment):
        raise ValueError("Endpoints cannot contain query strings or fragments")
    authority = f"[{host}]" if ":" in host else host
    if port is not None and port != (443 if parts.scheme.lower() == "https" else 80):
        authority += f":{port}"
    path, query = _encode_invisible(parts.path), _encode_invisible(parts.query)
    return urlunsplit((parts.scheme.lower(), authority, path or "/", query, ""))


def origin(url: str) -> str:
    """Return the normalized scheme/authority identity."""
    parts = urlsplit(normalize_url(url, limit=16384))
    return f"{parts.scheme}://{parts.netloc}"


def normalize_origin(value: str) -> str:
    """Validate an exact origin, without paths, query strings or fragments."""
    url = normalize_url(value, endpoint=True)
    if urlsplit(url).path != "/":
        raise ValueError("An origin cannot contain a path")
    return origin(url)


def same_site(left: str, right: str) -> bool:
    """Permit only same scheme/port and an optional www prefix change."""
    a, b = urlsplit(left), urlsplit(right)
    return (a.scheme, a.port, (a.hostname or "").removeprefix("www.")) == (
        b.scheme,
        b.port,
        (b.hostname or "").removeprefix("www."),
    )
