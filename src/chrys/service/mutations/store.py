# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SnapshotStore — content-addressable blob persistence for file snapshots.

Stores original file content blobs under the session directory, keyed
by SHA-256 hash.  Identical files (across different turns or paths)
share storage automatically.

Blob writes are gated by a :class:`SnapshotPolicy` — oversized or
binary (non-text) file contents are not backed up.  The mutation
metadata is still recorded upstream; only the content blob is withheld
(marked via :class:`SnapshotSkipReason`).

Layout::

    {session_dir}/mutations/{sha256_hex}
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import logging
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from chrys.foundation.text.encoding import EncodingDetector
from chrys.service.mutations.types import FileSnapshot, RestoreOutcome, RestoreResult, SnapshotSkipReason

if TYPE_CHECKING:
    from chrys.foundation.config.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_SNAPSHOT_FILE_BYTES: Final[int] = 50 * 1024 * 1024
"""Default per-file cap (50 MiB) on mutation backup blobs."""


def on_disk_link_kind(path: str) -> bool | None:
    """The file-vs-directory kind of the link *itself*, where links are typed.

    Windows reparse points carry the kind (dangling links included);
    POSIX links are untyped, so there is nothing to report — ``None``,
    never a value inferred from the current target.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return None
    attrs = getattr(st, "st_file_attributes", None)
    if attrs is None:
        return None
    return bool(attrs & stat.FILE_ATTRIBUTE_DIRECTORY)


def link_targets_directory(path: str) -> bool:
    """Whether ``path`` is a symlink of the directory kind.

    Windows types the link itself (``FILE_ATTRIBUTE_DIRECTORY`` on the
    reparse point, present even when the link dangles) and demands the
    same answer back at creation time.  POSIX links are untyped, so the
    fallback consults the resolved target — a dangling link reads as
    False there, which is harmless because POSIX creation ignores the
    flag entirely.
    """
    if not os.path.islink(path):
        return False
    kind = on_disk_link_kind(path)
    if kind is not None:
        return kind
    return os.path.isdir(path)


def link_target_kind(link_path: str, target_text: str) -> bool | None:
    """Best-effort file-vs-directory kind of a link's *target text*.

    Anchors a relative target at the link's parent (the way the platform
    resolves it) and reports what sits there now: True for a directory,
    False for any other entry, None when the destination is absent — the
    kind is then unprovable and must not be asserted either way.
    """
    resolved = target_text if os.path.isabs(target_text) else os.path.join(os.path.dirname(link_path), target_text)
    if os.path.isdir(resolved):
        return True
    if os.path.lexists(resolved):
        return False
    return None


@dataclass(frozen=True)
class SnapshotPolicy:
    """Decides which file contents may be persisted as backup blobs.

    Applied by :class:`SnapshotStore` at every blob-write entry point,
    so all mutation sources (file tools, shell scan, git calibration)
    are gated uniformly.  Skipped files keep their mutation *records*;
    they just have no content backup (no diff rendering, no rollback
    restore for that side).
    """

    max_file_bytes: int = DEFAULT_MAX_SNAPSHOT_FILE_BYTES
    """Per-file size cap in bytes.  ``<= 0`` disables the size cap."""

    skip_binary: bool = True
    """Whether binary (non-text) file contents are excluded from backup."""

    @classmethod
    def from_settings(cls, settings: Settings) -> SnapshotPolicy:
        """Build the policy from application :class:`Settings`."""
        return cls(
            max_file_bytes=settings.mutation_snapshot_max_file_mb * 1024 * 1024,
            skip_binary=settings.mutation_snapshot_skip_binary,
        )

    def classify_size(self, size: int) -> SnapshotSkipReason | None:
        """Size-only gate — usable from a bare ``stat`` before any read."""
        if self.max_file_bytes > 0 and size > self.max_file_bytes:
            return SnapshotSkipReason.TOO_LARGE
        return None

    def classify_sample(self, sample: bytes) -> SnapshotSkipReason | None:
        """Content gate on an in-memory sample (prefix is sufficient)."""
        if self.skip_binary and EncodingDetector.looks_binary(sample):
            return SnapshotSkipReason.BINARY
        return None

    def classify_data(self, data: bytes) -> SnapshotSkipReason | None:
        """Full gate for content already held in memory."""
        return self.classify_size(len(data)) or self.classify_sample(data)


@dataclass(frozen=True)
class BlobSaveResult:
    """Outcome of a blob-write attempt.

    ``content_hash`` is ``None`` when nothing was persisted — either the
    file was confirmed missing (``skip_reason`` is ``None`` too), or the
    content was withheld/unreadable (``skip_reason`` says why).
    """

    content_hash: str | None
    skip_reason: SnapshotSkipReason | None = None


@dataclass(frozen=True)
class FileProbe:
    """An endpoint's existence and content identity, without storing a backup blob."""

    existed: bool
    content_hash: str | None = None
    skip_reason: SnapshotSkipReason | None = None


class SnapshotStore:
    """Persists original file content blobs to disk.

    Content-addressable: blobs are named by their SHA-256 hash, so
    identical files deduplicate automatically.
    """

    def __init__(self, session_dir: Path, *, policy: SnapshotPolicy | None = None) -> None:
        self._dir = session_dir / "mutations"
        self._policy = policy if policy is not None else SnapshotPolicy()

    @property
    def mutations_dir(self) -> Path:
        """The directory where blobs are stored."""
        return self._dir

    @property
    def policy(self) -> SnapshotPolicy:
        """The blob-write gate applied by this store."""
        return self._policy

    @staticmethod
    def content_hash(data: bytes) -> str:
        """SHA-256 hex digest of raw bytes."""
        return hashlib.sha256(data).hexdigest()

    def probe(self, path: str) -> FileProbe:
        """Hash the current endpoint with bounded reads and this store's snapshot policy.

        Symlinks hash their target text, as snapshots do. Unreadable endpoints
        conservatively count as existing; only confirmed absence allows deletion.
        """
        try:
            entry = os.lstat(path)
        except ValueError:
            return FileProbe(False)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                return FileProbe(False)
            return FileProbe(True, skip_reason=SnapshotSkipReason.UNREADABLE)
        try:
            if stat.S_ISLNK(entry.st_mode):
                return FileProbe(True, self.content_hash(os.fsencode(os.readlink(path))))
            if not stat.S_ISREG(entry.st_mode):
                return FileProbe(True, skip_reason=SnapshotSkipReason.UNREADABLE)
            skip = self._policy.classify_size(entry.st_size)
            if skip is not None:
                return FileProbe(True, skip_reason=skip)
            digest = hashlib.sha256()
            size = 0
            with Path(path).open("rb") as stream:
                while block := stream.read(64 * 1024):
                    skip = self._policy.classify_sample(block) if size == 0 else None
                    size += len(block)
                    skip = self._policy.classify_size(size) or skip
                    if skip is not None:
                        return FileProbe(True, skip_reason=skip)
                    digest.update(block)
            return FileProbe(True, digest.hexdigest())
        except OSError:
            return FileProbe(True, skip_reason=SnapshotSkipReason.UNREADABLE)

    def save(self, path: str, period_index: int) -> FileSnapshot:
        """Snapshot a file's current content before mutation.

        Reads the file, stores its content as a blob, and returns
        a :class:`FileSnapshot`.  If the file does not exist, returns
        a snapshot with ``existed=False`` (no blob written).  If the
        content is withheld by :class:`SnapshotPolicy` (too large /
        binary), returns ``existed=True`` with ``content_hash=None``
        and ``skip_reason`` set — oversized files are rejected from a
        bare ``stat`` without ever being read into memory. Unreadable or
        unsupported endpoints use ``UNREADABLE`` and conservatively keep
        ``existed=True`` so rollback can never turn a failed probe into deletion.

        A symlink entry (including a dangling one or a link to a
        directory) is snapshotted as the link itself: the blob holds
        its target text and ``is_symlink`` is set, so restore can
        recreate the link rather than materializing its destination.
        """
        norm = os.path.normpath(os.path.abspath(path))
        try:
            entry = os.lstat(norm)
        except ValueError:
            # Invalid filesystem identities (e.g. embedded NUL) cannot exist.
            return FileSnapshot(path=norm, period_index=period_index, existed=False)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                return FileSnapshot(path=norm, period_index=period_index, existed=False)
            logger.warning("Failed to stat file for snapshot: %s — %s", norm, exc)
            return self._skipped_snapshot(norm, period_index, 0, SnapshotSkipReason.UNREADABLE)
        if stat.S_ISLNK(entry.st_mode):
            try:
                data = os.fsencode(os.readlink(norm))
            except OSError as exc:
                logger.warning("Failed to read symlink for snapshot: %s — %s", norm, exc)
                return self._skipped_snapshot(norm, period_index, entry.st_size, SnapshotSkipReason.UNREADABLE)
            h = self.content_hash(data)
            blob_path = self._dir / h
            if not blob_path.exists():
                self._dir.mkdir(parents=True, exist_ok=True)
                blob_path.write_bytes(data)
            return FileSnapshot(
                path=norm,
                period_index=period_index,
                existed=True,
                content_hash=h,
                size=len(data),
                is_symlink=True,
                symlink_target_is_dir=link_targets_directory(norm),
            )
        if not stat.S_ISREG(entry.st_mode):
            return self._skipped_snapshot(norm, period_index, entry.st_size, SnapshotSkipReason.UNREADABLE)
        size = entry.st_size

        skip = self._policy.classify_size(size)
        if skip is not None:
            return self._skipped_snapshot(norm, period_index, size, skip)

        try:
            data = Path(norm).read_bytes()
        except OSError as exc:
            logger.warning("Failed to read file for snapshot: %s — %s", norm, exc)
            return self._skipped_snapshot(norm, period_index, size, SnapshotSkipReason.UNREADABLE)

        skip = self._policy.classify_data(data)
        if skip is not None:
            return self._skipped_snapshot(norm, period_index, len(data), skip)

        h = self.content_hash(data)
        blob_path = self._dir / h
        if not blob_path.exists():
            self._dir.mkdir(parents=True, exist_ok=True)
            blob_path.write_bytes(data)

        return FileSnapshot(
            path=norm,
            period_index=period_index,
            existed=True,
            content_hash=h,
            size=len(data),
        )

    def _skipped_snapshot(self, norm: str, period_index: int, size: int, skip: SnapshotSkipReason) -> FileSnapshot:
        """Build an ``existed=True`` snapshot whose content was withheld."""
        logger.debug("Snapshot content withheld for %s (%s, %d bytes)", norm, skip.value, size)
        return FileSnapshot(path=norm, period_index=period_index, existed=True, size=size, skip_reason=skip)

    def read_content(self, snapshot: FileSnapshot) -> bytes | None:
        """Read stored blob content for a snapshot.

        Returns ``None`` if the file didn't exist at snapshot time or
        the blob is missing from disk.
        """
        if not snapshot.existed or snapshot.content_hash is None:
            return None
        blob_path = self._dir / snapshot.content_hash
        if not blob_path.exists():
            return None
        return blob_path.read_bytes()

    def restore(self, snapshot: FileSnapshot) -> RestoreResult:
        """Restore a file to its snapshotted pre-turn state.

        CONTRACT (agreed design, keep this intact unless you're renegotiating it):

        The rollback modal already shows the user exactly what each
        checked path will look like after the revert (diff-view "after"
        side = snapshot target state).  The user confirms by pressing
        the discard button.  From that point on, revert is unconditional:
        bring each selected file to the target state regardless of what
        sits on disk now.  We deliberately do NOT drift-check or prompt
        — if the user modified the file manually after the agent's turn,
        they either accept the overwrite (file stays checked) or they
        de-select it in the change-list tree.  Simpler logic, guaranteed
        outcome.  A pre-press drift *warning* in the modal (badge only,
        doesn't block) is planned as a follow-up; it belongs in the UI,
        not here.

        Behaviour by snapshot shape:

        - ``existed=False``: the file was created during the rolled-back
          turn → target state is "absent".  Unlink if present; no-op if
          already absent.  Refuses to ``rmtree`` a directory that now
          sits at the path (returns :data:`RestoreOutcome.FAILED`) — the
          blast radius of that operation is far beyond the single path
          the user consented to.
        - ``existed=True``: the file existed at turn start → target
          state is the stored blob content.  Overwrites whatever is at
          the path; creates missing parent directories.  If the path is
          currently a symlink we unlink it first so the replacement is
          a real file (``write_bytes`` would otherwise follow the link
          and corrupt its target).  A directory at the path is likewise
          treated as FAILED.
        - ``is_symlink=True``: the entry was a symlink at turn start →
          target state is a link pointing at the stored target text.
          The replacement link is built at a temporary sibling path and
          swapped over whatever sits at the path (file or link) with
          ``os.replace``, so a failed creation (Windows without
          privilege, filesystems without symlink support) reports
          FAILED with the existing entry preserved — never degraded to
          a regular file, never destroyed.  The temporary name is
          unique per call; an existing entry at a candidate name is
          never unlinked.  A directory at the path is FAILED.
          ``symlink_target_is_dir`` is forwarded as
          ``target_is_directory`` so Windows directory links recreate
          with the right kind (POSIX ignores it); on platforms whose
          links are typed, NOOP additionally requires the on-disk
          link's kind to match the recorded one — POSIX links are
          untyped, so the text match alone is NOOP there.  An
          unprovable recorded kind (``None``) is inferred from what
          the target text resolves to at restore time.

        Returns a :class:`RestoreResult` — APPLIED when we mutated disk,
        NOOP when it was already in target state, FAILED otherwise with
        ``reason`` populated.
        """
        target = Path(snapshot.path)

        # -- Delete branch: target state is "file does not exist".
        if not snapshot.existed:
            if target.is_symlink():
                # Symlinks (even broken ones) are removed cleanly by
                # ``unlink`` — and their ``exists()`` may be False when
                # the link is dangling, so handle them first.
                try:
                    target.unlink()
                    logger.debug(
                        "Rollback: removed symlink %s (created after turn %d)", snapshot.path, snapshot.period_index
                    )
                    return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.APPLIED)
                except OSError as exc:
                    logger.warning("Rollback: failed to remove symlink %s — %s", snapshot.path, exc)
                    return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=str(exc))
            if not target.exists():
                return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.NOOP)
            if target.is_dir():
                reason = "refusing to remove directory at path (expected file)"
                logger.warning("Rollback: %s for %s", reason, snapshot.path)
                return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=reason)
            try:
                target.unlink()
                logger.debug("Rollback: deleted %s (created after turn %d)", snapshot.path, snapshot.period_index)
                return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.APPLIED)
            except OSError as exc:
                logger.warning("Rollback: failed to delete %s — %s", snapshot.path, exc)
                return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=str(exc))

        # -- Write branch: target state is the snapshotted blob content.
        content = self.read_content(snapshot)
        if content is None:
            # Rollback plans exclude non-restorable snapshots, so a
            # policy-skipped snapshot reaching here is defense-in-depth.
            if snapshot.skip_reason is not None:
                reason = f"snapshot content not stored ({snapshot.skip_reason.value})"
            else:
                reason = f"blob missing (hash={snapshot.content_hash})"
            logger.warning("Rollback: %s for %s", reason, snapshot.path)
            return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=reason)

        # -- Symlink branch: target state is "a link pointing at the
        # stored target text" — never a regular file with that text.
        if snapshot.is_symlink:
            target_text = os.fsdecode(content)
            recorded_kind = snapshot.symlink_target_is_dir
            if target.is_symlink():
                try:
                    if os.readlink(target) == target_text:
                        # Where links are typed (Windows), the same text
                        # can exist as either kind — NOOP also requires
                        # the on-disk kind to match the recorded one.
                        # POSIX links are untyped: the text match is the
                        # whole identity, and rebuilding could not change
                        # anything anyway.
                        disk_kind = on_disk_link_kind(str(target))
                        if recorded_kind is None or disk_kind is None or disk_kind == recorded_kind:
                            return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.NOOP)
                except OSError:
                    pass
            elif target.is_dir():
                reason = "refusing to overwrite directory at path (expected symlink)"
                logger.warning("Rollback: %s for %s", reason, snapshot.path)
                return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=reason)
            if recorded_kind is None:
                # Kind unprovable at record time: infer from whatever the
                # target text resolves to now (POSIX ignores it anyway).
                recorded_kind = link_target_kind(str(target), target_text)
            # Build the replacement link beside the target and swap it in
            # atomically — a failed creation (Windows without privilege,
            # filesystems without symlink support) must leave whatever
            # currently sits at the path intact.  The temporary name is
            # unique per call: a colliding name belongs to someone else
            # and is never unlinked.
            tmp: Path | None = None
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                for _ in range(10):
                    candidate = target.parent / f".{target.name}.{uuid.uuid4().hex}.chrys-link-tmp"
                    try:
                        os.symlink(target_text, candidate, target_is_directory=bool(recorded_kind))
                    except FileExistsError:
                        continue
                    tmp = candidate
                    break
            except OSError as exc:
                logger.warning("Rollback: failed to recreate symlink %s — %s", snapshot.path, exc)
                return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=str(exc))
            if tmp is None:
                reason = "could not allocate a unique temporary link name"
                logger.warning("Rollback: %s for %s", reason, snapshot.path)
                return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=reason)
            try:
                os.replace(tmp, target)
                logger.debug("Rollback: recreated symlink %s -> %s", snapshot.path, target_text)
            except OSError as exc:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                logger.warning("Rollback: failed to swap in symlink %s — %s", snapshot.path, exc)
                return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=str(exc))
            return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.APPLIED)

        # Guard against a directory sitting where we'd write the file —
        # ``write_bytes`` would raise ``IsADirectoryError`` and we do
        # NOT want to rmtree a user directory as collateral damage.
        if target.is_dir() and not target.is_symlink():
            reason = "refusing to overwrite directory at path (expected file)"
            logger.warning("Rollback: %s for %s", reason, snapshot.path)
            return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=reason)

        # No-op short-circuit: if disk already matches the target blob,
        # skip the write.  Avoids unnecessary mtime churn and makes
        # repeated rollbacks cheap.  Symlinks always take the slow path
        # since a link pointing elsewhere isn't "the same content" even
        # if it resolves to matching bytes — we still want to replace
        # the link with a real file below.
        if not target.is_symlink() and target.is_file():
            try:
                if self.content_hash(target.read_bytes()) == snapshot.content_hash:
                    return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.NOOP)
            except OSError:
                # Fall through to the write path — it will report a
                # cleaner FAILED if the write itself can't proceed.
                pass

        # If the path is currently a symlink, unlink it so the write
        # below lands on a real file at ``target`` rather than writing
        # through the link into some unrelated destination.
        if target.is_symlink():
            try:
                target.unlink()
            except OSError as exc:
                logger.warning("Rollback: failed to remove symlink before restore: %s — %s", snapshot.path, exc)
                return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=str(exc))

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            logger.debug("Rollback: restored %s to turn %d state", snapshot.path, snapshot.period_index)
        except OSError as exc:
            logger.warning("Rollback: failed to restore %s — %s", snapshot.path, exc)
            return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=str(exc))
        return RestoreResult(path=snapshot.path, outcome=RestoreOutcome.APPLIED)

    def save_blob(self, path: str) -> BlobSaveResult:
        """Read a file and save its content as a blob.

        Returns a :class:`BlobSaveResult` — ``content_hash=None`` if the
        file is confirmed missing. Unreadable content carries ``UNREADABLE``;
        content withheld by :class:`SnapshotPolicy` also sets ``skip_reason`` (the
        size gate runs on a bare ``stat`` so oversized files are never
        read into memory).  Unlike :meth:`save`, this does not create a
        ``FileSnapshot`` record — it only stores the blob for later
        retrieval via :meth:`read_blob`.

        A symlink entry hashes as its target text (the Git symlink-blob
        representation), never as the destination's content — the
        entry's identity is the link.
        """
        norm = os.path.normpath(os.path.abspath(path))
        try:
            entry = os.lstat(norm)
        except ValueError:
            return BlobSaveResult(content_hash=None)
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                return BlobSaveResult(content_hash=None)
            logger.warning("Failed to stat file for blob: %s — %s", norm, exc)
            return BlobSaveResult(content_hash=None, skip_reason=SnapshotSkipReason.UNREADABLE)
        if stat.S_ISLNK(entry.st_mode):
            try:
                return self.save_data_as_blob(os.fsencode(os.readlink(norm)))
            except OSError as exc:
                logger.warning("Failed to read symlink for blob: %s — %s", norm, exc)
                return BlobSaveResult(content_hash=None, skip_reason=SnapshotSkipReason.UNREADABLE)
        if not stat.S_ISREG(entry.st_mode):
            return BlobSaveResult(content_hash=None, skip_reason=SnapshotSkipReason.UNREADABLE)
        size = entry.st_size
        skip = self._policy.classify_size(size)
        if skip is not None:
            logger.debug("Blob content withheld for %s (%s, %d bytes)", norm, skip.value, size)
            return BlobSaveResult(content_hash=None, skip_reason=skip)
        try:
            data = Path(norm).read_bytes()
        except OSError as exc:
            logger.warning("Failed to read file for blob: %s — %s", norm, exc)
            return BlobSaveResult(content_hash=None, skip_reason=SnapshotSkipReason.UNREADABLE)
        skip = self._policy.classify_data(data)
        if skip is not None:
            logger.debug("Blob content withheld for %s (%s, %d bytes)", norm, skip.value, len(data))
            return BlobSaveResult(content_hash=None, skip_reason=skip)
        h = self.content_hash(data)
        blob_path = self._dir / h
        if not blob_path.exists():
            self._dir.mkdir(parents=True, exist_ok=True)
            blob_path.write_bytes(data)
        return BlobSaveResult(content_hash=h)

    def save_data_as_blob(self, data: bytes) -> BlobSaveResult:
        """Store arbitrary bytes as a blob, subject to :class:`SnapshotPolicy`.

        Used by git calibration to store HEAD content for files that
        were clean at turn start but became dirty during the turn.
        """
        skip = self._policy.classify_data(data)
        if skip is not None:
            logger.debug("Blob content withheld for in-memory data (%s, %d bytes)", skip.value, len(data))
            return BlobSaveResult(content_hash=None, skip_reason=skip)
        h = self.content_hash(data)
        blob_path = self._dir / h
        if not blob_path.exists():
            self._dir.mkdir(parents=True, exist_ok=True)
            blob_path.write_bytes(data)
        return BlobSaveResult(content_hash=h)

    def read_blob(self, content_hash: str) -> bytes | None:
        """Read blob content by its SHA-256 hash.

        Returns ``None`` if the blob is not found on disk.
        """
        blob_path = self._dir / content_hash
        if not blob_path.exists():
            return None
        return blob_path.read_bytes()

    # -- Cleanup --

    def remove_blobs(self, hashes: set[str]) -> None:
        """Delete specific blob files by their content hashes.

        Silently skips hashes whose blob files do not exist on disk.
        """
        for h in hashes:
            blob_path = self._dir / h
            if blob_path.is_file():
                try:
                    blob_path.unlink()
                except OSError as exc:
                    logger.warning("Failed to delete blob %s — %s", h, exc)

    def remove_all(self) -> None:
        """Delete the entire mutations blob directory.

        Called when a session is deleted to reclaim disk space.
        """
        if self._dir.is_dir():
            shutil.rmtree(self._dir, ignore_errors=True)
