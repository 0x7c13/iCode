# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Filesystem-safe filename helpers."""

from __future__ import annotations

import hashlib
import re

# Characters forbidden on Windows file systems + POSIX separators + control bytes.
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_fs_name(name: str) -> str:
    """Replace characters that are unsafe in file/directory names with ``_``."""
    return _UNSAFE_FILENAME_RE.sub("_", name)


def safe_platform_basename(name: str, *, default: str = "unnamed") -> str:
    """Return a non-empty, platform-safe basename."""
    safe = safe_fs_name(name).rstrip(" .")
    if not safe or safe in {".", ".."}:
        safe = default
    if _is_windows_reserved_basename(safe):
        safe = f"_{safe}"
    return safe


def bounded_safe_stem(
    name: str,
    *,
    prefix: str = "",
    suffix: str = "",
    max_filename_bytes: int = 255,
    temp_prefix: str = "",
    temp_suffix: str = "",
    default: str = "unnamed",
    digest_chars: int = 8,
) -> str:
    """Return a safe basename stem that fits a final filename budget.

    ``prefix`` and ``suffix`` are final filename affixes around the stem.
    ``temp_prefix`` and ``temp_suffix`` reserve room for atomic-write temporary
    filenames derived from the final basename.
    """
    safe = safe_platform_basename(name, default=default)
    reserved = _utf8_len(prefix) + _utf8_len(suffix) + _utf8_len(temp_prefix) + _utf8_len(temp_suffix)
    budget = max(1, max_filename_bytes - reserved)
    if _utf8_len(safe) <= budget:
        return safe

    digest = hashlib.sha256(name.encode("utf-8", errors="surrogatepass")).hexdigest()[: max(1, digest_chars)]
    marker = f"_{digest}"
    marker_len = _utf8_len(marker)
    if marker_len >= budget:
        truncated = _truncate_utf8(digest, budget)
        return safe_platform_basename(truncated, default=default)

    prefix = _truncate_utf8(safe, budget - marker_len).rstrip(" ._")
    if not prefix:
        prefix = default
    stem = f"{prefix}{marker}"
    if _is_windows_reserved_basename(stem):
        stem = f"_{stem}"
    return _truncate_utf8(stem, budget).rstrip(" .") or default


def _is_windows_reserved_basename(name: str) -> bool:
    base = name.split(".", 1)[0].upper()
    return base in _WINDOWS_RESERVED_BASENAMES


def _utf8_len(value: str) -> int:
    return len(value.encode("utf-8", errors="surrogatepass"))


def _truncate_utf8(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    used = 0
    out: list[str] = []
    for char in value:
        size = _utf8_len(char)
        if used + size > max_bytes:
            break
        out.append(char)
        used += size
    return "".join(out)
