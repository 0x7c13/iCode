# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session migration — copy sessions between two session roots.

Pure filesystem service with no UI dependency: :func:`plan_session_migration`
enumerates what a source root holds and :func:`run_session_migration` copies
each session into the destination root under the same locks the store uses,
never deleting from the source and never overwriting an existing destination.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from chrys.foundation.util.lock import FileLock
from chrys.service.state.store import (
    legacy_session_files,
    make_junction_dropping_ignore,
    session_active_lock_path,
    session_dir_candidates,
    session_write_lock_path,
)
from chrys.service.tools.session_artifacts import reharden_document_image_artifacts

# Per-session wait for the source/destination write locks. Longer than a
# single envelope write, far shorter than the store's own writer timeout so a
# busy session is reported instead of stalling the whole migration.
MIGRATION_WRITE_LOCK_TIMEOUT_SECONDS = 2.0

_LINKED_ENTRY_REASON = "linked entry"
_SOURCE_CHANGED_REASON = "source changed since planning"


class SessionMigrationError(ValueError):
    """Raised when the source/destination pair cannot be migrated at all."""


@dataclass(frozen=True, slots=True)
class MigrationItem:
    """One session the planner intends to copy."""

    session_id: str
    """Short id: the session folder name or the legacy file stem."""
    source: Path
    """Session folder or legacy flat ``<short>.json`` file under the source root."""
    destination: Path
    """``<destination root>/<name>`` — same folder or file name as the source."""
    legacy_file: bool
    already_present: bool
    """The destination path already existed at plan time (never overwritten)."""


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    source_dir: Path
    destination_dir: Path
    items: tuple[MigrationItem, ...]
    rejected: tuple[tuple[Path, str], ...]
    """Entries the planner refuses to copy: ``(path, reason)``."""


@dataclass(frozen=True, slots=True)
class MigrationReport:
    copied: tuple[str, ...]
    skipped_present: tuple[str, ...]
    skipped_active: tuple[str, ...]
    skipped_busy: tuple[str, ...]
    failed: tuple[tuple[Path, str], ...]
    """``(path, reason)`` — planner rejections first, then per-session copy failures."""


def _canonical_key(path: Path) -> Path:
    """Best-effort canonical form for same/ancestor comparisons (missing paths allowed)."""
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    return Path(os.path.normcase(str(resolved)))


def _same_directory(left: Path, right: Path) -> bool:
    """Whether two existing paths name one directory (device/inode identity)."""
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _existing_ancestor_names(path: Path, other: Path) -> bool:
    """Whether an existing strict ancestor of *path* is the directory *other*.

    String comparison misses case-insensitive volumes (macOS APFS: ``resolve``
    keeps the caller's casing and ``normcase`` is a no-op there), so the parts
    that exist on disk are compared by filesystem identity instead. *path* may
    not exist yet; only ancestors that do are consulted.
    """
    if not other.exists():
        return False
    return any(ancestor.exists() and _same_directory(ancestor, other) for ancestor in path.absolute().parents)


def _is_linked_root(path: Path) -> bool:
    """Whether the candidate root itself is a symlink or an NT junction."""
    return path.is_symlink() or os.path.isjunction(path)


def plan_session_migration(source_dir: Path, destination_dir: Path) -> MigrationPlan:
    """Enumerate the sessions under *source_dir* to copy into *destination_dir*.

    Validates the root pair first (nothing is created on disk): the roots
    must differ and neither may contain the other, and the source must be an
    existing directory. The destination may not exist yet. Candidate roots
    that are links (symlink/junction) and legacy candidates that are not
    plain regular files land in :attr:`MigrationPlan.rejected`.
    """
    source = Path(source_dir)
    destination = Path(destination_dir)
    source_key = _canonical_key(source)
    destination_key = _canonical_key(destination)
    if source_key == destination_key or (
        source.exists() and destination.exists() and _same_directory(source, destination)
    ):
        raise SessionMigrationError(f"Source and destination are the same directory: {source}")
    if source_key in destination_key.parents or _existing_ancestor_names(destination, source):
        raise SessionMigrationError(f"Destination {destination} lies inside source {source}")
    if destination_key in source_key.parents or _existing_ancestor_names(source, destination):
        raise SessionMigrationError(f"Source {source} lies inside destination {destination}")
    if not source.is_dir():
        raise SessionMigrationError(f"Source is not a directory: {source}")

    items: list[MigrationItem] = []
    rejected: list[tuple[Path, str]] = []

    for candidate in session_dir_candidates(source):
        if _is_linked_root(candidate):
            rejected.append((candidate, _LINKED_ENTRY_REASON))
            continue
        target = destination / candidate.name
        items.append(
            MigrationItem(
                session_id=candidate.name,
                source=candidate,
                destination=target,
                legacy_file=False,
                already_present=os.path.lexists(target),
            )
        )

    for legacy in legacy_session_files(source):
        try:
            mode = os.lstat(legacy).st_mode
        except OSError as exc:
            rejected.append((legacy, str(exc)))
            continue
        if stat.S_ISDIR(mode):
            # A real directory that merely ends in ``.json`` is judged as a
            # session folder above, never as a flat file.
            continue
        if legacy.is_symlink() or not stat.S_ISREG(mode):
            rejected.append((legacy, _LINKED_ENTRY_REASON))
            continue
        target = destination / legacy.name
        items.append(
            MigrationItem(
                session_id=legacy.stem,
                source=legacy,
                destination=target,
                legacy_file=True,
                already_present=os.path.lexists(target),
            )
        )

    return MigrationPlan(
        source_dir=source,
        destination_dir=destination,
        items=tuple(items),
        rejected=tuple(rejected),
    )


def _copy_session_dir(item: MigrationItem) -> None:
    """Copy a session folder via an exclusively created partial dir, then rename into place.

    The partial comes from ``mkdtemp`` (random dot-prefixed name, created by
    us), so nothing pre-existing at a predictable path is ever removed or
    written through; ``copystat`` at the end of the copy restores the source
    folder's own mode over the temporary directory's.
    """
    partial = Path(tempfile.mkdtemp(prefix=f".partial-{item.session_id}-", dir=item.destination.parent))
    try:
        shutil.copytree(item.source, partial, symlinks=True, ignore=make_junction_dropping_ignore(), dirs_exist_ok=True)
        reharden_document_image_artifacts(item.source, partial)
        os.replace(partial, item.destination)
    finally:
        if os.path.lexists(partial):
            shutil.rmtree(partial, ignore_errors=True)


def _source_changed_reason(item: MigrationItem) -> str | None:
    """Why *item*'s source no longer is what the plan saw, or ``None`` if it still is.

    The plan filtered links and non-sessions, but the copy runs later — after
    the user confirmed the dialog and the locks were taken — so the entry is
    inspected again with ``lstat`` (never following) right before it is read.
    """
    try:
        mode = os.lstat(item.source).st_mode
    except OSError as exc:
        return str(exc)
    # An NT junction reads as a plain directory to ``lstat``: ask for it by name.
    if stat.S_ISLNK(mode) or os.path.isjunction(item.source):
        return _LINKED_ENTRY_REASON
    if item.legacy_file and not stat.S_ISREG(mode):
        return _SOURCE_CHANGED_REASON
    if not item.legacy_file and not stat.S_ISDIR(mode):
        return _SOURCE_CHANGED_REASON
    return None


def _open_source_file(path: Path) -> int:
    """Open a legacy session file for reading without following a link at its leaf.

    ``O_NOFOLLOW`` makes the check and the open one step on POSIX; the
    ``fstat`` guard covers Windows (no such flag; the ``lstat`` re-check
    before the copy is what rejects a planted link there) and FIFOs/devices.
    """
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(_SOURCE_CHANGED_REASON)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _copy_legacy_file(item: MigrationItem) -> None:
    """Copy a legacy flat session file via an exclusively created partial, then rename into place.

    The partial is created by ``mkstemp`` (random name, ``O_EXCL``) and written
    through its descriptor, so a pre-planted path can neither be followed nor
    overwritten; the source is read through a no-follow descriptor and the copy
    keeps that descriptor's mode and timestamps.
    """
    fd, name = tempfile.mkstemp(prefix=f"{item.destination.name}.partial-", dir=item.destination.parent)
    partial = Path(name)
    try:
        with os.fdopen(fd, "wb") as out, os.fdopen(_open_source_file(item.source), "rb") as src:
            shutil.copyfileobj(src, out)
            source_stat = os.fstat(src.fileno())
        os.chmod(partial, stat.S_IMODE(source_stat.st_mode))
        os.utime(partial, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
        os.replace(partial, item.destination)
    finally:
        with contextlib.suppress(OSError):
            partial.unlink()


def run_session_migration(plan: MigrationPlan) -> MigrationReport:
    """Copy every planned session; blocking, meant to run in a worker thread.

    Per session: an already-present destination is skipped without locking;
    the source active lock is taken non-blocking (held by a live session →
    ``skipped_active``) and HELD through the copy; the source and destination
    write locks are taken with a short timeout (``skipped_busy``); the source
    entry is re-checked with ``lstat`` (swapped for a link → ``failed``); the
    copy lands in a partial path and is renamed into place. A failing session is
    reported and never stops the others. The source is never modified beyond
    ensuring its ``.locks`` folder exists.
    """
    source_root = plan.source_dir
    destination_root = plan.destination_dir
    (source_root / ".locks").mkdir(parents=True, exist_ok=True)
    (destination_root / ".locks").mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    skipped_present: list[str] = []
    skipped_active: list[str] = []
    skipped_busy: list[str] = []
    failed: list[tuple[Path, str]] = list(plan.rejected)

    for item in plan.items:
        if item.already_present or os.path.lexists(item.destination):
            skipped_present.append(item.session_id)
            continue
        with contextlib.ExitStack() as locks:
            # ``TimeoutError`` is an ``OSError``: the timeout branches must come
            # first. Any other failure to take a lock (an unwritable ``.locks``
            # folder, a lock path occupied by a directory or a symlink) is
            # that session's failure, not the migration's.
            try:
                locks.enter_context(FileLock(session_active_lock_path(source_root, item.session_id), timeout=0))
            except TimeoutError:
                skipped_active.append(item.session_id)
                continue
            except OSError as exc:
                failed.append((item.source, str(exc)))
                continue
            try:
                locks.enter_context(
                    FileLock(
                        session_write_lock_path(source_root, item.session_id),
                        timeout=MIGRATION_WRITE_LOCK_TIMEOUT_SECONDS,
                    )
                )
                locks.enter_context(
                    FileLock(
                        session_write_lock_path(destination_root, item.session_id),
                        timeout=MIGRATION_WRITE_LOCK_TIMEOUT_SECONDS,
                    )
                )
            except TimeoutError:
                skipped_busy.append(item.session_id)
                continue
            except OSError as exc:
                failed.append((item.source, str(exc)))
                continue
            # Another migration may have landed the same session while this
            # one waited for the destination lock; never overwrite it.
            if os.path.lexists(item.destination):
                skipped_present.append(item.session_id)
                continue
            # The entry may have been swapped for a link since the plan looked.
            reason = _source_changed_reason(item)
            if reason is not None:
                failed.append((item.source, reason))
                continue
            try:
                if item.legacy_file:
                    _copy_legacy_file(item)
                else:
                    _copy_session_dir(item)
            except Exception as exc:
                failed.append((item.source, str(exc)))
            else:
                copied.append(item.session_id)

    return MigrationReport(
        copied=tuple(copied),
        skipped_present=tuple(skipped_present),
        skipped_active=tuple(skipped_active),
        skipped_busy=tuple(skipped_busy),
        failed=tuple(failed),
    )
