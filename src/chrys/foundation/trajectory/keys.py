# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Per-installation fingerprint key for trajectory events.

Mirrors the ACP audit key precedent (``~/.chrys/acp-audit-hmac.key``): an
owner-only 32-byte secret published with a no-replace ``link`` so a racing
creator can never overwrite a key another process already used.
"""

from __future__ import annotations

import contextlib
import os
import secrets
import stat
from pathlib import Path

from chrys.foundation.platform.files import (
    SecureFileError,
    secure_open_owner_only,
    secure_open_owner_only_binary,
    secure_unlink_owner_verified,
)
from chrys.foundation.trajectory.fingerprint import FINGERPRINT_KEY_BYTES
from chrys.foundation.util.lock import FileLock

TRAJECTORY_KEY_FILE_NAME = "trajectory-hmac.key"


def ensure_owner_only_directory(path: Path) -> None:
    """Create *path* privately (0700) and tighten an existing POSIX mode."""
    from chrys.foundation.platform import get_platform

    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if get_platform().is_windows:
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise OSError(f"Owner-only path is not a directory: {path}")
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


def load_or_create_fingerprint_key(config_dir: Path, *, timeout: float = 5.0) -> bytes:
    """Validate or no-replace-create the installation trajectory fingerprint key."""
    config_dir.mkdir(parents=True, exist_ok=True)
    key_path = config_dir / TRAJECTORY_KEY_FILE_NAME
    lock_path = config_dir / f"{TRAJECTORY_KEY_FILE_NAME}.lock"
    with FileLock(lock_path, timeout=timeout):
        if key_path.exists() or key_path.is_symlink():
            return _read_key(key_path)
        temp_path = config_dir / f".{TRAJECTORY_KEY_FILE_NAME}.{secrets.token_hex(8)}.tmp"
        fd = secure_open_owner_only(temp_path, write=True, create=True)
        try:
            key = secrets.token_bytes(FINGERPRINT_KEY_BYTES)
            with os.fdopen(fd, "wb") as file:
                fd = -1
                file.write(key)
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temp_path, key_path, follow_symlinks=False)
            except FileExistsError:
                return _read_key(key_path)
            _fsync_parent(config_dir)
            return _read_key(key_path)
        finally:
            if fd >= 0:
                os.close(fd)
            secure_unlink_owner_verified(temp_path)


def _read_key(path: Path) -> bytes:
    try:
        with secure_open_owner_only_binary(path) as file:
            key = file.read(FINGERPRINT_KEY_BYTES + 1)
    except OSError as exc:
        raise SecureFileError("The trajectory fingerprint key failed secure verification.") from exc
    if len(key) != FINGERPRINT_KEY_BYTES:
        raise SecureFileError("The trajectory fingerprint key has an invalid length.")
    return key


def _fsync_parent(path: Path) -> None:
    from chrys.foundation.platform import get_platform

    if get_platform().is_windows:
        return
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
