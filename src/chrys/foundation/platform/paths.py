# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cross-platform path helpers."""

from __future__ import annotations

import ntpath
import os
import posixpath
import unicodedata
from collections.abc import Iterator
from pathlib import Path

# Limit on whitespace positions to substitute in a single basename. This prevents
# model-controlled malformed paths with thousands of spaces from generating
# quadratic numbers of candidate strings and exhausting memory.
_MAX_WHITESPACE_SUBSTITUTION_POSITIONS = 100


def is_absolute_path(p: str) -> bool:
    """Return whether *p* is absolute under POSIX or Windows path rules."""
    return posixpath.isabs(p) or ntpath.isabs(p)


def resolve_cross_platform_path(raw: str, *, base: str | Path | None = None) -> str:
    """Resolve native paths while preserving foreign-platform absolute paths.

    Relative paths are resolved against *base* when provided, otherwise against
    the current process directory. Windows absolute paths rendered on POSIX
    hosts (and other foreign absolute forms) are returned unchanged so they are
    not accidentally interpreted as relative local paths.
    """
    expanded = os.path.expanduser(raw.strip())
    path = Path(expanded)
    if path.is_absolute():
        try:
            return str(path.resolve())
        except OSError:
            return str(path.absolute())
    if is_absolute_path(expanded):
        return expanded

    if base is not None:
        path = Path(base) / expanded
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def resolve_workspace_path(raw: str, *, base_cwd: str | os.PathLike[str] | None = None) -> str:
    """Resolve *raw* relative to an explicit workspace cwd, not process cwd."""
    expanded = os.path.expanduser(raw)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    if is_absolute_path(expanded):
        return expanded
    if not base_cwd:
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(os.fspath(base_cwd), expanded))


def display_relpath(path: str, cwd: str) -> str:
    """Display an entry relative to either spelling of an aliased cwd.

    Prefer the user's spelling, then try the physical cwd if the entry
    appears external. Resolve only the cwd: a deleted entry or a symlink
    must retain its own name. Genuine external paths keep their old label.
    """
    try:
        relative = os.path.relpath(path, cwd)
    except ValueError:
        relative = path
    else:
        if relative != os.pardir and not relative.startswith(os.pardir + os.sep):
            return relative
    try:
        physical_relative = os.path.relpath(path, os.path.realpath(cwd))
    except OSError, ValueError:
        return relative
    if physical_relative != os.pardir and not physical_relative.startswith(os.pardir + os.sep):
        return physical_relative
    return relative


def resolve_existing_path(
    raw: str,
    *,
    base_cwd: str | os.PathLike[str] | None = None,
) -> str | None:
    """Resolve *raw* and return a path that actually exists on disk.

    If the exact resolved path does not exist, this function tries canonical
    Unicode normalization forms (NFC, NFD) and a conservative set of
    whitespace substitutions. The most common trigger is macOS screenshot
    filenames, which use U+202F (narrow no-break space) before AM/PM, while
    models often reproduce the path with a regular U+0020 space.

    Compatibility normalization forms (NFKC/NFKD) are intentionally excluded
    because they can change character identity (e.g. a fullwidth dot ``U+FF0E``
    becomes a regular ``.``), which could bypass path-based approval checks or
    create directory traversal sequences.

    If more than one candidate exists, the match is considered ambiguous and
    ``None`` is returned. This helper is intended for read-only filesystem
    tools: if no unique match exists, ``None`` is returned and the caller can
    surface a ``not found`` error.
    """
    resolved = resolve_workspace_path(raw, base_cwd=base_cwd)
    if os.path.exists(resolved):
        return resolved

    # Base candidates from canonical normalization only. NFC/NFD
    # reorder/decompose characters but do not change identity; NFKC/NFKD can
    # map ``U+FF0E env`` to ``.env`` or ``U+FF0E U+FF0E U+FF0F secret.txt`` to
    # ``../secret.txt``, which would bypass approval classification performed on
    # the raw argument before the tool runs.
    base_candidates: set[str] = {resolved}
    for form in ("NFC", "NFD"):
        candidate = unicodedata.normalize(form, resolved)
        if candidate != resolved:
            base_candidates.add(candidate)

    # Collect unique existing files by (dev, inode) so that normalization-
    # insensitive filesystems (e.g. macOS) do not count the same physical file as
    # multiple ambiguous matches.
    unique_files: dict[tuple[int, int], str] = {}
    seen_paths: set[str] = set()

    for base in base_candidates:
        if not _try_add_existing_candidate(base, unique_files, seen_paths):
            return None

        for candidate in _iter_whitespace_substitution_candidates(base):
            if not _try_add_existing_candidate(candidate, unique_files, seen_paths):
                return None

    if len(unique_files) == 1:
        return next(iter(unique_files.values()))
    return None


def _try_add_existing_candidate(
    candidate: str,
    unique_files: dict[tuple[int, int], str],
    seen_paths: set[str],
) -> bool:
    """Try to add *candidate* to *unique_files*.

    Returns ``True`` if processing can continue (candidate is missing, already
    seen, or refers to a file already in the map). Returns ``False`` once a
    second distinct file has been found, signalling ambiguity.
    """
    if candidate in seen_paths:
        return True
    seen_paths.add(candidate)
    if not os.path.exists(candidate):
        return True
    try:
        st = os.stat(candidate)
    except OSError:
        return True
    key = (st.st_dev, st.st_ino)
    if key in unique_files:
        return True
    unique_files[key] = candidate
    return len(unique_files) <= 1


def _iter_whitespace_substitution_candidates(resolved: str) -> Iterator[str]:
    """Yield candidate paths produced by substituting one whitespace character at a time.

    macOS screenshot filenames use U+202F (narrow no-break space) only before
    AM/PM, while other spaces in the name remain regular U+0020. A naive global
    replacement would miss the target, so we vary each whitespace position
    independently and let the caller resolve ambiguities.

    If the basename contains too many whitespace positions, no candidates are
    yielded to avoid quadratic memory allocation from model-controlled input.
    """
    dirname, basename = os.path.split(resolved)
    special_spaces = ("\u202f", "\u00a0", "\u2009")

    whitespace_count = sum(1 for ch in basename if ch == " " or ch in special_spaces)
    if whitespace_count > _MAX_WHITESPACE_SUBSTITUTION_POSITIONS:
        return

    for idx, ch in enumerate(basename):
        if ch == " ":
            for special in special_spaces:
                yield os.path.join(dirname, basename[:idx] + special + basename[idx + 1 :])
        elif ch in special_spaces:
            yield os.path.join(dirname, basename[:idx] + " " + basename[idx + 1 :])
