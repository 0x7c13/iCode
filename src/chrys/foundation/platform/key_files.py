# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Complete, immutable installation keys on filesystems without hard links.

Keys normally remain plain files. If neither a hard link nor an exclusive
rename is supported (macOS exFAT supports neither), publish a directory
containing the completed key instead.
An ordinary directory rename cannot replace a file or a nonempty directory, so
even a creator ignoring the caller's lock cannot replace an already used key.
Readers accept both layouts, including after the config directory moves disks.
"""

from __future__ import annotations

import contextlib
import errno
import os
import tempfile
from pathlib import Path

from chrys.foundation.platform.files import (
    fsync_directory,
    publish_file_no_replace,
    secure_open_owner_only_binary,
    secure_unlink_owner_verified,
    strip_inherited_directory_acl,
)

_KEY_PAYLOAD = "key"
_UNSUPPORTED_PUBLICATION = frozenset({errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EINVAL})


def read_owner_only_key(path: Path, *, max_bytes: int) -> bytes:
    """Read a bounded key from either layout, applying the same secure opens."""
    payload = path / _KEY_PAYLOAD if path.is_dir() else path
    with secure_open_owner_only_binary(payload) as file:
        return file.read(max_bytes)


def publish_owner_only_key(source: Path, target: Path) -> None:
    """Publish a flushed, owner-only key, retaining every existing valid key.

    Callers hold their key's FileLock across existence checks, creation, and
    reading. They own syncing the parent and cleaning any remaining source.
    Failed attempts leave only randomly named temporary entries; there is no
    empty final key to recover or guess whether it was ever used.
    """
    try:
        publish_file_no_replace(source, target)
        return
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_PUBLICATION:
            raise

    # Reject even invalid existing entries, including an empty directory. The
    # rename itself also protects any complete key published after this check.
    try:
        target.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(errno.EEXIST, "The installation key already exists.", str(target))

    directory = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    payload = directory / _KEY_PAYLOAD
    try:
        strip_inherited_directory_acl(directory)
        os.rename(source, payload)
        fsync_directory(directory)
        try:
            os.rename(directory, target)
        except OSError as exc:
            if exc.errno in {errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR, errno.EISDIR}:
                raise FileExistsError(errno.EEXIST, "The installation key already exists.", str(target)) from exc
            raise
    finally:
        # Never traverse/delete a published container or recursively remove
        # unknown entries if something interfered with our staging directory.
        secure_unlink_owner_verified(payload)
        with contextlib.suppress(OSError):
            directory.rmdir()
