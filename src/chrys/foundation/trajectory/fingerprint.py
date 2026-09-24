# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Keyed, domain-separated fingerprints for content-minimized events.

``argument_fingerprint`` / ``content_fingerprint`` / segment ``value_hash`` must
never be bare digests: the fingerprinted value may be user content, and a bare
SHA leaves an enumerable fingerprint behind. Every fingerprint is an HMAC over
``domain || 0x00 || data`` with a per-installation secret key, so equal inputs
in different domains never collide and nothing is recoverable without the key.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import Any, Final

FINGERPRINT_KEY_BYTES: Final = 32

DOMAIN_TOOL_ARGUMENTS: Final = "chrys.trajectory.tool_arguments"
DOMAIN_TOOL_CONTENT: Final = "chrys.trajectory.tool_content"
DOMAIN_SEGMENT_VALUE: Final = "chrys.trajectory.segment_value"
DOMAIN_WORKSPACE: Final = "chrys.trajectory.workspace"
DOMAIN_MEMBERSHIP: Final = "chrys.trajectory.context_membership"


def keyed_fingerprint(key: bytes, domain: str, data: bytes) -> str:
    """Return the hex HMAC-SHA256 of *data* under *key*, separated by *domain*."""
    if not key:
        raise ValueError("Fingerprint key must not be empty.")
    if not domain or "\x00" in domain:
        raise ValueError("Fingerprint domain must be a non-empty string without NUL bytes.")
    mac = hmac.new(key, domain.encode("utf-8") + b"\x00" + data, hashlib.sha256)
    return mac.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any] | list[Any] | str | int | float | bool | None) -> bytes:
    """Encode a JSON value canonically (sorted keys, ASCII, compact) for hashing.

    ``ensure_ascii=True`` keeps the encoding injective for surrogate-bearing
    strings: a raw ``\\udcXX`` and a literal ``\\\\udcXX`` never alias.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str).encode("ascii")


def fingerprint_json(
    key: bytes, domain: str, value: Mapping[str, Any] | list[Any] | str | int | float | bool | None
) -> str:
    """Fingerprint a JSON-compatible value under *domain*."""
    return keyed_fingerprint(key, domain, canonical_json_bytes(value))


def fingerprint_text(key: bytes, domain: str, text: str) -> str:
    """Fingerprint a text value under *domain* (lone surrogates encoded, never raised).

    ``surrogatepass`` is total like ``backslashreplace`` but also injective:
    an escape sequence a user typed never encodes to the same bytes as the
    surrogate it spells, so two different values never share a fingerprint.
    """
    return keyed_fingerprint(key, domain, text.encode("utf-8", errors="surrogatepass"))
