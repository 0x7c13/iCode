# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session writer lease: one trajectory writer per session, across processes.

The lease is an OS-level file lock on ``<session>/trajectory/writer.lock``. It
is bound to the *writer worker*, not to the "currently active session": the UI
may switch sessions while a stuck worker still holds the descriptor, and the
lease keeps following that worker until it exits (or the OS reclaims it when
the process dies — at which point the stuck thread is gone too).

Acquisition never blocks: a held lease means "this session already has a
writer somewhere", and the caller opens the session trajectory-disabled.
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from typing import Any, cast

from chrys.foundation.platform.files import SecureFileError, secure_open_owner_only_append

WRITER_LEASE_FILE_NAME = "writer.lock"

_HELD_LEASES: dict[str, WriterLease] = {}
_HELD_GUARD = threading.Lock()


def _lease_key(path: Path) -> str:
    return os.path.normcase(str(path.absolute()))


def _try_lock_fd(fd: int) -> bool:
    from chrys.foundation.platform import get_platform

    if get_platform().is_windows:
        import msvcrt

        locking = cast(Any, msvcrt)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            locking.locking(fd, locking.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    # Same shape as the Windows branch above: the members exist only on the
    # platform the branch runs on, so the type checker is told the module is
    # opaque rather than asked to resolve them for the other one.
    flock = cast(Any, fcntl)
    try:
        flock.flock(fd, flock.LOCK_EX | flock.LOCK_NB)
    except BlockingIOError:
        return False
    except OSError:
        return False
    return True


def _unlock_fd(fd: int) -> None:
    from chrys.foundation.platform import get_platform

    with contextlib.suppress(OSError):
        if get_platform().is_windows:
            import msvcrt

            locking = cast(Any, msvcrt)
            os.lseek(fd, 0, os.SEEK_SET)
            locking.locking(fd, locking.LK_UNLCK, 1)
        else:
            import fcntl

            flock = cast(Any, fcntl)
            flock.flock(fd, flock.LOCK_UN)


class WriterLease:
    """A held writer lease; release it only when the worker has exited."""

    def __init__(self, path: Path, fd: int) -> None:
        self._path = path
        self._fd = fd
        self._released = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return not self._released

    @classmethod
    def try_acquire(cls, path: Path) -> WriterLease | None:
        """Acquire the lease at *path* without waiting; ``None`` when held elsewhere.

        The in-process registry is the fast path; the OS lock is the truth
        across processes. The lock file is owner-only and opened with
        delete-sharing on Windows so the session directory stays renamable.
        The parent directory must already exist; open failures other than
        contention propagate as ``SecureFileError``/``OSError``.
        """
        key = _lease_key(path)
        with _HELD_GUARD:
            if key in _HELD_LEASES:
                return None
            handle = secure_open_owner_only_append(path)
            if not _try_lock_fd(handle.fd):
                os.close(handle.fd)
                return None
            lease = cls(path, handle.fd)
            _HELD_LEASES[key] = lease
            return lease

    @classmethod
    def is_held_elsewhere(cls, path: Path) -> bool:
        """Return whether another holder owns the lease at *path* right now.

        Used by the tombstone sweep: a tombstone whose lease is still held may
        still be written by a stuck worker in another process and must be left
        alone. A missing lock file means no holder.
        """
        if not path.is_file():
            return False
        try:
            probe = cls.try_acquire(path)
        except SecureFileError, OSError:
            # Cannot even open the lock file: fail closed, treat it as held.
            return True
        if probe is None:
            return True
        probe.release()
        return False

    @classmethod
    def relocate(cls, old_path: Path, new_path: Path) -> None:
        """Re-key a held lease whose lock file moved with its directory.

        The lock lives on the inode, so the holder does not change — only the
        name the in-process registry answers under. Without this a session
        recreated at the old path is refused a writer that nothing holds.
        """
        old_key = _lease_key(old_path)
        with _HELD_GUARD:
            lease = _HELD_LEASES.pop(old_key, None)
            if lease is None or lease._released:
                # A released lease holds nothing; re-keying one would leave the
                # registry answering "held" under the new name forever.
                return
            lease._path = new_path
            _HELD_LEASES[_lease_key(new_path)] = lease

    def release(self) -> None:
        """Release the lease (idempotent)."""
        # Whole body under the registry guard: the key to remove is derived
        # from ``_path``, and ``relocate`` rewrites that field. Reading it
        # outside lets a relocation land in between, so the removal looks for
        # the old name while the registry now holds this lease under the new
        # one — leaving a phantom holder that refuses every later writer.
        with _HELD_GUARD:
            if self._released:
                return
            self._released = True
            key = _lease_key(self._path)
            if _HELD_LEASES.get(key) is self:
                del _HELD_LEASES[key]
            _unlock_fd(self._fd)
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = -1
