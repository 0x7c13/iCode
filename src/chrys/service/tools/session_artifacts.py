# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Strict-UTF-8 handles and shared roots for session-owned tool artifacts."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from chrys.foundation.config.settings import resolve_sessions_dir
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import (
    atomic_write_owner_only_bytes,
    secure_open_owner_verified_binary,
    secure_open_regular_binary,
)
from chrys.foundation.util.session_ids import session_short_id

_DOCUMENT_ARTIFACT_PREFIX: Final[str] = "chrys-session-document:"
_MAX_FILENAME_BYTES: Final[int] = 1024
_INVALID_FILENAME_CHARS: Final[frozenset[str]] = frozenset('<>:"/\\|?*')
_DOCUMENT_MARKDOWN_SUFFIXES: Final[frozenset[str]] = frozenset({".md"})
_DOCUMENT_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".png", ".jpg", ".jpeg", ".webp"})
DOCUMENT_IMAGE_ARTIFACT_MAX_FILES: Final[int] = 8192
DOCUMENT_IMAGE_ARTIFACT_MAX_FILE_BYTES: Final[int] = 50 * 1024 * 1024
DOCUMENT_IMAGE_ARTIFACT_MAX_TOTAL_BYTES: Final[int] = 512 * 1024 * 1024


class DocumentImageArtifactLimitError(OSError):
    """Raised when a session's document-image artifact set exceeds its bounds."""


def iter_document_image_artifacts(
    artifact_root: Path,
    *,
    max_files: int = DOCUMENT_IMAGE_ARTIFACT_MAX_FILES,
    max_file_bytes: int = DOCUMENT_IMAGE_ARTIFACT_MAX_FILE_BYTES,
    max_total_bytes: int = DOCUMENT_IMAGE_ARTIFACT_MAX_TOTAL_BYTES,
) -> Iterator[tuple[Path, int]]:
    """Yield bounded regular image artifacts without materializing the directory."""
    file_count = 0
    total_bytes = 0
    with os.scandir(artifact_root) as entries:
        for entry in entries:
            if os.path.splitext(entry.name)[1].lower() not in _DOCUMENT_IMAGE_SUFFIXES:
                continue
            file_count += 1
            if file_count > max_files:
                raise DocumentImageArtifactLimitError("session document-image file count exceeds its limit")
            info = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise OSError("session document-image artifact is not a regular file")
            if info.st_size > max_file_bytes:
                raise DocumentImageArtifactLimitError("session document-image artifact exceeds its file-size limit")
            total_bytes += info.st_size
            if total_bytes > max_total_bytes:
                raise DocumentImageArtifactLimitError("session document-image artifacts exceed their byte limit")
            yield Path(entry.path), info.st_size


def _verified_document_artifact_root(session_dir: Path) -> Path | None:
    """Return a session's link-free artifact root, or ``None`` when it has none."""
    artifact_root = session_dir / "doc_converter"
    try:
        info = artifact_root.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode) or os.path.isjunction(artifact_root):
        raise OSError("session document-image artifact root is not a regular directory")
    return artifact_root


def reharden_document_image_artifacts(
    source_session_dir: Path, destination_session_dir: Path, *, legacy_import: bool = False
) -> None:
    """Re-publish copied document images with a protected owner-only Windows DACL.

    Sources are read owner-verified, as fork reads the user's own session.
    ``legacy_import`` (session migration) reads them as the legacy reader
    does instead: an older release running elevated owned its images by the
    Administrators group, and the chosen legacy folder may sit behind a parent
    alias such as a mapped drive.
    """
    if not get_platform().is_windows:
        return

    # Judge the source first: a junction there is dropped by the copy ignore,
    # so checking the destination ahead of it would report the resulting
    # absence instead of the redirected root that caused it.
    source_root = _verified_document_artifact_root(source_session_dir)
    if source_root is None:
        return
    destination_root = _verified_document_artifact_root(destination_session_dir)
    if destination_root is None:
        raise OSError("session document-image artifact root is missing from the copy")

    read_source = secure_open_regular_binary if legacy_import else secure_open_owner_verified_binary
    for destination_path, expected_size in list(iter_document_image_artifacts(destination_root)):
        source_path = source_root / destination_path.name
        with read_source(source_path) as source:
            payload = source.read(DOCUMENT_IMAGE_ARTIFACT_MAX_FILE_BYTES + 1)
        if len(payload) != expected_size:
            raise OSError("session document-image artifact changed while being copied")
        atomic_write_owner_only_bytes(destination_path, payload)


def _validate_document_filename(
    filename: str,
    *,
    allowed_suffixes: frozenset[str] = _DOCUMENT_MARKDOWN_SUFFIXES,
) -> bytes:
    """Return strict UTF-8 filename bytes or raise for an unsafe handle target."""
    if not filename or filename in {".", ".."} or any(char in _INVALID_FILENAME_CHARS for char in filename):
        raise ValueError("document artifact filename must be one path component")
    suffix = os.path.splitext(filename)[1].lower()
    if suffix not in allowed_suffixes:
        allowed = ", ".join(sorted(allowed_suffixes))
        raise ValueError(f"document artifact filename must use one of these suffixes: {allowed}")
    if any(ord(char) < 0x20 or 0xD800 <= ord(char) <= 0xDFFF for char in filename):
        raise ValueError("document artifact filename contains unsupported characters")
    encoded = filename.encode("utf-8")
    if len(encoded) > _MAX_FILENAME_BYTES:
        raise ValueError("document artifact filename is too long")
    return encoded


def make_document_artifact_handle(filename: str) -> str:
    """Return a readable UTF-8 handle scoped to the session document directory."""
    _validate_document_filename(filename)
    return _DOCUMENT_ARTIFACT_PREFIX + filename


def make_document_image_artifact_handle(filename: str) -> str:
    """Return a readable UTF-8 handle for a normalized document image."""
    _validate_document_filename(filename, allowed_suffixes=_DOCUMENT_IMAGE_SUFFIXES)
    return _DOCUMENT_ARTIFACT_PREFIX + filename


def is_document_artifact_handle(reference: str) -> bool:
    """Return whether *reference* uses the reserved document-handle namespace."""
    return reference.startswith(_DOCUMENT_ARTIFACT_PREFIX)


def resolve_tool_session_dir(
    config_dir: Path,
    *,
    session_id: str = "",
    session_dir: Path | None = None,
) -> Path | None:
    """Return the one session directory shared by artifact producers and readers."""
    if session_dir is not None:
        return session_dir
    if not session_id:
        return None
    return resolve_sessions_dir(config_dir, create=False) / session_short_id(session_id)


def resolve_document_markdown_artifact_handle(reference: str, session_dir: Path | None) -> str | None:
    """Resolve a Markdown document artifact for ``read_file``."""
    return _resolve_document_artifact_handle(reference, session_dir, allowed_suffixes=_DOCUMENT_MARKDOWN_SUFFIXES)


def resolve_document_image_artifact_handle(reference: str, session_dir: Path | None) -> str | None:
    """Resolve a normalized document image artifact for ``view_image``."""
    return _resolve_document_artifact_handle(reference, session_dir, allowed_suffixes=_DOCUMENT_IMAGE_SUFFIXES)


def _resolve_document_artifact_handle(
    reference: str,
    session_dir: Path | None,
    *,
    allowed_suffixes: frozenset[str],
) -> str | None:
    """Resolve one policy-scoped session artifact handle."""
    if not is_document_artifact_handle(reference):
        return None
    if session_dir is None:
        raise ValueError("document artifact handle is unavailable without a session directory")

    filename = reference.removeprefix(_DOCUMENT_ARTIFACT_PREFIX)
    _validate_document_filename(filename, allowed_suffixes=allowed_suffixes)

    session_root = os.path.realpath(os.fspath(session_dir))
    artifact_root = os.path.join(session_root, "doc_converter")
    if os.path.normcase(os.path.realpath(artifact_root)) != os.path.normcase(artifact_root):
        raise ValueError("session document directory must not be redirected")

    resolved = os.path.join(artifact_root, filename)
    if os.path.dirname(resolved) != artifact_root:
        raise ValueError("document artifact handle resolves outside the session document directory")
    if os.path.normcase(os.path.realpath(resolved)) != os.path.normcase(resolved):
        raise ValueError("document artifact file must not be redirected")
    return resolved
