# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Per-shell-command implicit change detection using bounded Git state."""

from __future__ import annotations

import errno
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import PurePath

from chrys.service.mutations.git_state import (
    GIT_TIMEOUT_SECONDS,
    GitDeltaEntry,
    _run_git_nul_stream,
    canonical_git_pathspecs,
    git_path_exists,
    read_git_blob,
    read_git_entry_meta,
    read_git_head_delta,
    read_git_head_oid,
    read_git_target_kind,
    repo_relative_git_path,
)
from chrys.service.mutations.store import DEFAULT_MAX_SNAPSHOT_FILE_BYTES
from chrys.service.mutations.types import MutationOp, SnapshotSkipReason

logger = logging.getLogger(__name__)

MAX_DIRTY_FILES = 500
"""Skip status calibration when a repo exceeds this dirty-path count."""

_GIT_TIMEOUT = GIT_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class _FileStat:
    """Lightweight file stat for change detection."""

    mtime_ns: int
    size: int
    mode: int


@dataclass(frozen=True, slots=True)
class ImplicitChange:
    """A file change detected by status or a scoped HEAD delta.

    ``before_skip``/``before_size`` mark a before endpoint whose content
    the snapshot policy withheld (oversized old blob): the mutation event
    is still recorded, only the backup is missing.  ``before_symlink``
    marks an old endpoint that was a symlink (Git mode 120000), whose
    blob is the link's target text; ``before_symlink_dir`` is that
    link's file-vs-directory kind proven from the old tree (None =
    unprovable — never inferred from the post-command filesystem).
    """

    path: str
    operation: MutationOp
    before_data: bytes | None
    before_skip: SnapshotSkipReason | None = None
    before_size: int | None = None
    before_symlink: bool = False
    before_symlink_dir: bool | None = None


class GitDiffCalibrator:
    """Collect per-command Git observations without mutating the tracker."""

    def __init__(
        self,
        repo_root: str,
        *,
        scope_paths: list[str] | None = None,
        max_snapshot_file_bytes: int = DEFAULT_MAX_SNAPSHOT_FILE_BYTES,
    ) -> None:
        self._root = os.path.normpath(os.path.abspath(repo_root))
        self._scope_pathspecs = self._normalize_scope_pathspecs(scope_paths)
        self._max_snapshot_file_bytes = max_snapshot_file_bytes
        self._before_stats: dict[str, _FileStat] = {}
        self._before_paths: set[str] = set()
        self._old_head: str | None = None
        self._old_head_known = False
        self._active = True
        self._detection_truncated = False

    @property
    def detection_truncated(self) -> bool:
        """Whether required status, delta, or old-content data was unavailable."""
        return self._detection_truncated

    def mark_detection_truncated(self) -> None:
        """Record an advisory failure raised outside the calibrator core."""
        self._detection_truncated = True

    @staticmethod
    def find_repo_root(path: str) -> str | None:
        """Walk upward to a repository or linked-worktree ``.git`` entry."""
        current = os.path.abspath(path)
        while True:
            if os.path.exists(os.path.join(current, ".git")):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent

    def capture_before(self) -> list[str]:
        """Capture old HEAD plus dirty path stats before a shell command."""
        self._detection_truncated = False
        try:
            self._old_head = read_git_head_oid(self._root)
            self._old_head_known = True
        except OSError:
            self._old_head = None
            self._old_head_known = False
            self._detection_truncated = True

        self._before_paths, status_failed = self._git_changed_files()
        if status_failed or len(self._before_paths) > MAX_DIRTY_FILES:
            self._detection_truncated = True
            self._active = False
            self._before_stats = {}
            return []

        self._active = True
        try:
            self._before_stats = {
                path: file_stat for path in self._before_paths if (file_stat := self._stat(path)) is not None
            }
        except OSError:
            self._before_stats = {}
            self._active = False
            self._detection_truncated = True
        return list(self._before_paths)

    def detect_implicit_changes(self, already_tracked: set[str]) -> list[ImplicitChange]:
        """Return per-command candidates, excluding only same-call detector paths."""
        changes: dict[str, ImplicitChange] = {}
        normalized_skip = {os.path.normpath(os.path.abspath(path)) for path in already_tracked}

        if self._active:
            after_paths, status_failed = self._git_changed_files()
            if status_failed or len(after_paths) > MAX_DIRTY_FILES:
                self._detection_truncated = True
            else:
                try:
                    self._collect_status_changes(after_paths, normalized_skip, changes)
                except OSError:
                    changes.clear()
                    self._detection_truncated = True

        if not self._old_head_known:
            # Failed/missing capture is not an empty repository. Keep only
            # status changes with pre-command stat evidence; never enumerate
            # the entire new HEAD as additions from an unknown endpoint.
            self._detection_truncated = True
            return list(changes.values())

        try:
            new_head = read_git_head_oid(self._root)
        except OSError:
            new_head = self._old_head
            self._detection_truncated = True
        if new_head != self._old_head:
            delta = read_git_head_delta(
                self._root,
                self._old_head,
                new_head,
                self._scope_pathspecs,
                max_paths=MAX_DIRTY_FILES,
            )
            if delta.failed or delta.truncated:
                self._detection_truncated = True
            else:
                self._collect_head_delta(delta.entries, normalized_skip, changes)
        return list(changes.values())

    def _collect_status_changes(
        self,
        after_paths: set[str],
        same_call_paths: set[str],
        changes: dict[str, ImplicitChange],
    ) -> None:
        after_stats = {path: file_stat for path in after_paths if (file_stat := self._stat(path)) is not None}
        for path, after_stat in after_stats.items():
            norm = os.path.normpath(path)
            if norm in same_call_paths:
                continue
            before_stat = self._before_stats.get(path)
            if before_stat is None:
                candidate = self._candidate_from_old_head(norm, present_after=True)
                if candidate is not None:
                    changes[norm] = candidate
            elif before_stat != after_stat:
                changes[norm] = ImplicitChange(norm, MutationOp.MODIFY, None)

        for path in after_paths - set(after_stats):
            norm = os.path.normpath(path)
            if norm in same_call_paths:
                continue
            if path in self._before_stats:
                changes[norm] = ImplicitChange(norm, MutationOp.DELETE, None)
            elif path not in self._before_paths:
                candidate = self._candidate_from_old_head(norm, present_after=False)
                if candidate is not None and candidate.operation is not MutationOp.CREATE:
                    changes[norm] = replace_operation(candidate, MutationOp.DELETE)

        for path in self._before_paths - after_paths:
            norm = os.path.normpath(path)
            if norm in same_call_paths or path not in self._before_stats:
                continue
            after_stat = self._stat(path)
            if after_stat is None:
                changes[norm] = ImplicitChange(norm, MutationOp.DELETE, None)
            elif after_stat != self._before_stats[path]:
                changes[norm] = ImplicitChange(norm, MutationOp.MODIFY, None)

    def _collect_head_delta(
        self,
        entries: tuple[GitDeltaEntry, ...],
        same_call_paths: set[str],
        changes: dict[str, ImplicitChange],
    ) -> None:
        for entry in entries:
            status = entry.status[:1]
            destination = os.path.normpath(os.path.join(self._root, entry.path))
            if status == "R" and entry.old_path is not None:
                source = os.path.normpath(os.path.join(self._root, entry.old_path))
                self._add_delta_candidate(source, MutationOp.DELETE, same_call_paths, changes)
                self._add_old_endpoint_destination(destination, same_call_paths, changes)
            elif status == "C":
                self._add_old_endpoint_destination(destination, same_call_paths, changes)
            elif status == "D":
                self._add_delta_candidate(destination, MutationOp.DELETE, same_call_paths, changes)
            elif status == "A":
                self._add_old_endpoint_destination(destination, same_call_paths, changes)
            else:
                self._add_delta_candidate(destination, MutationOp.MODIFY, same_call_paths, changes)

    def _add_old_endpoint_destination(
        self,
        path: str,
        same_call_paths: set[str],
        changes: dict[str, ImplicitChange],
    ) -> None:
        # A path another detector already covered wins outright — probing the
        # old HEAD for it wastes subprocesses and can flag truncation for a
        # row that is already complete.
        norm = os.path.normpath(path)
        if norm in same_call_paths or norm in changes:
            return
        candidate = self._candidate_from_old_head(norm, present_after=True)
        if candidate is not None:
            changes[norm] = candidate

    def _add_delta_candidate(
        self,
        path: str,
        operation: MutationOp,
        same_call_paths: set[str],
        changes: dict[str, ImplicitChange],
    ) -> None:
        norm = os.path.normpath(path)
        if norm in same_call_paths or norm in changes:
            return
        candidate = self._candidate_from_old_head(norm, present_after=operation is not MutationOp.DELETE)
        if candidate is not None:
            changes[norm] = replace_operation(candidate, operation)

    def _candidate_from_old_head(self, path: str, *, present_after: bool) -> ImplicitChange | None:
        if not self._old_head_known:
            self._detection_truncated = True
            return None
        if self._old_head is None:
            return ImplicitChange(path, MutationOp.CREATE, None)
        rel = repo_relative_git_path(self._root, path)
        if rel is None:
            self._detection_truncated = True
            return None
        existed = git_path_exists(self._root, self._old_head, rel)
        if existed is False:
            return ImplicitChange(path, MutationOp.CREATE, None)
        if existed is None:
            self._detection_truncated = True
            return None
        meta = read_git_entry_meta(self._root, self._old_head, rel)
        if meta is None or meta.size is None:
            self._detection_truncated = True
            return None
        operation = MutationOp.MODIFY if present_after else MutationOp.DELETE
        if self._max_snapshot_file_bytes > 0 and meta.size > self._max_snapshot_file_bytes:
            # The policy withholds the content backup; the mutation event
            # itself is still recorded — a content skip is not detection
            # truncation, and the oversized blob is never read.
            return ImplicitChange(
                path,
                operation,
                None,
                before_skip=SnapshotSkipReason.TOO_LARGE,
                before_size=meta.size,
                before_symlink=meta.symlink,
            )
        before_data = read_git_blob(self._root, self._old_head, rel)
        if before_data is None:
            self._detection_truncated = True
            return None
        kind = self._old_target_kind(self._old_head, path, before_data) if meta.symlink else None
        return ImplicitChange(path, operation, before_data, before_symlink=meta.symlink, before_symlink_dir=kind)

    def _old_target_kind(self, head: str, link_path: str, target_blob: bytes) -> bool | None:
        """Old-tree kind of a symlink's target — the only evidence that
        predates the command; the live filesystem may have been rewritten
        by it.  Outside-repo or untracked targets stay unprovable."""
        target_text = os.fsdecode(target_blob)
        resolved = target_text if os.path.isabs(target_text) else os.path.join(os.path.dirname(link_path), target_text)
        rel = repo_relative_git_path(self._root, resolved)
        if rel is None:
            return None
        return read_git_target_kind(self._root, head, rel)

    def _git_changed_files(self) -> tuple[set[str], bool]:
        files: set[str] = set()
        pathspec_args = self._pathspec_args()
        commands = (
            ["diff", "--name-only", "-z", *([] if self._old_head is None else ["HEAD"]), *pathspec_args],
            ["diff", "--name-only", "-z", "--cached", *pathspec_args],
            ["ls-files", "-z", "--others", "--exclude-standard", *pathspec_args],
        )
        failed = False
        for args in commands:
            if not self._run_git_namelist(args, files):
                failed = True
            if len(files) > MAX_DIRTY_FILES:
                break  # Callers give up past the limit; the rest would only be read and dropped.
        return files, failed

    def _pathspec_args(self) -> list[str]:
        return ["--", *self._scope_pathspecs] if self._scope_pathspecs else []

    def _normalize_scope_pathspecs(self, scope_paths: list[str] | None) -> tuple[str, ...]:
        if not scope_paths:
            return ()
        root = os.path.abspath(self._root)
        specs: list[str] = []
        for raw in scope_paths:
            if not raw:
                continue
            abs_path = os.path.abspath(raw)
            try:
                if os.path.commonpath([root, abs_path]) != root:
                    continue
            except ValueError:
                continue
            rel = PurePath(os.path.relpath(abs_path, root)).as_posix()
            specs.append("." if rel == "." else rel)
        return canonical_git_pathspecs(specs)

    def _run_git_namelist(self, args: list[str], out: set[str]) -> bool:
        """Add the paths *args* lists to *out*, stopping Git once *out* holds more than ``MAX_DIRTY_FILES``."""

        def _consume(field: bytes) -> bool:
            if field:
                out.add(os.path.normpath(os.path.join(self._root, os.fsdecode(field))))
            return len(out) <= MAX_DIRTY_FILES

        try:
            stream = _run_git_nul_stream(self._root, args, timeout=_GIT_TIMEOUT, consume=_consume)
        except OSError:
            logger.debug("GitDiffCalibrator: git command failed: %s", args, exc_info=True)
            return False
        return stream.stopped or (stream.returncode == 0 and not stream.malformed)

    @staticmethod
    def _stat(path: str) -> _FileStat | None:
        try:
            value = os.lstat(path)
            return _FileStat(value.st_mtime_ns, value.st_size, stat.S_IMODE(value.st_mode))
        except OSError as exc:
            if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                return None
            raise


def replace_operation(change: ImplicitChange, operation: MutationOp) -> ImplicitChange:
    """Return *change* with an operation selected by the observed endpoint."""
    return ImplicitChange(
        change.path,
        operation,
        change.before_data,
        before_skip=change.before_skip,
        before_size=change.before_size,
        before_symlink=change.before_symlink,
        before_symlink_dir=change.before_symlink_dir,
    )
