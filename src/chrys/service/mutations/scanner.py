# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""WorkspaceScanner — before/after directory comparison for shell commands.

Detects file changes by comparing directory state (mtime + size) before
and after shell command execution.  Supports filtering via:

1. A built-in set of well-known directories to skip (node_modules, .git, ...)
2. Caller-supplied gitignore rules (via ``pathspec.GitIgnoreSpec``)

**Scalability**: full-tree scanning is O(n) in files.  For large repos
(100K+ files), use :meth:`WorkspaceScanner.scan_paths` to scan only
specific files/directories identified by :class:`ShellMutationDetector`,
or set ``max_files`` to cap the walk.

Cross-platform: uses ``os.walk`` + ``os.lstat`` which work identically
on Windows, macOS, and Linux.  Path separators are normalized internally.
"""

from __future__ import annotations

import errno
import logging
import os
import stat
import time
from dataclasses import dataclass
from pathlib import PurePath

import pathspec

from chrys.service.mutations.types import MutationOp

logger = logging.getLogger(__name__)


def _raise_scan_error(error: OSError) -> None:
    """Only a missing entry is absence; discard scans with unreadable coverage."""
    if error.errno not in (errno.ENOENT, errno.ENOTDIR):
        raise error


# ---------------------------------------------------------------------------
# Default exclude directories — well-known generated / VCS / dependency dirs
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDES: frozenset[str] = frozenset(
    {
        # Version control
        ".git",
        ".svn",
        ".hg",
        ".bzr",
        # JavaScript / Node
        "node_modules",
        "bower_components",
        ".next",
        ".nuxt",
        ".turbo",
        ".parcel-cache",
        # Python
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".eggs",
        "*.egg-info",  # handled via _matches_any_pattern
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".pytype",
        "htmlcov",
        ".coverage",
        # Rust / Java / .NET
        "target",
        ".gradle",
        ".mvn",
        "bin",
        "obj",
        # Go / PHP
        "vendor",
        # Ruby
        ".bundle",
        # General build output
        "build",
        "dist",
        "out",
        # IDE / editor
        ".idea",
        ".vscode",
        ".vs",
        ".eclipse",
        # Caches
        ".cache",
        ".sass-cache",
        ".eslintcache",
        ".stylelintcache",
        # OS artifacts
        ".DS_Store",
        "Thumbs.db",
        # Containers / infra
        ".terraform",
        ".serverless",
    }
)


# ---------------------------------------------------------------------------
# Gitignore rule matcher (powered by pathspec)
# ---------------------------------------------------------------------------


class GitignoreFilter:
    """Gitignore-style path filter using ``pathspec.GitIgnoreSpec``.

    Supports the full gitignore specification including:

    - ``*.ext`` -- match by extension
    - ``dirname/`` -- match directories by name
    - ``path/to/file`` -- match specific relative paths
    - ``**/pattern`` -- match in any subdirectory
    - ``a/**/b`` -- match across directory levels
    - ``[charset]`` character ranges
    - ``!pattern`` -- negate a previous rule
    - ``#`` comments and blank lines are ignored
    """

    def __init__(self, rules: list[str] | None = None) -> None:
        self._spec = pathspec.GitIgnoreSpec.from_lines(rules or [])

    @classmethod
    def from_file(cls, path: str) -> GitignoreFilter:
        """Load rules from a ``.gitignore`` file."""
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            return cls()
        return cls(lines)

    @classmethod
    def from_text(cls, text: str) -> GitignoreFilter:
        """Load rules from a gitignore-formatted string."""
        return cls(text.splitlines())

    def is_ignored(self, rel_path: str, *, is_dir: bool = False) -> bool:
        """Test whether a relative path should be ignored.

        Args:
            rel_path: Path relative to the scan root, using ``/`` separators.
            is_dir: Whether the path is a directory.

        Returns:
            True if the path matches an ignore rule (and is not negated).
        """
        # ``pathspec`` expects POSIX separators; ``PurePath.as_posix()`` is
        # platform-aware (converts ``\\`` on Windows, no-op on POSIX).
        rel_path = PurePath(rel_path).as_posix().strip("/")
        # pathspec needs trailing slash to identify directories
        if is_dir:
            rel_path = rel_path.rstrip("/") + "/"
        return self._spec.match_file(rel_path)


# ---------------------------------------------------------------------------
# File stat snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileStat:
    """Lightweight file stat for change detection."""

    mtime_ns: int
    size: int
    mode: int = 0


# ---------------------------------------------------------------------------
# Workspace scanner
# ---------------------------------------------------------------------------

# Safety cap: if the tree walk encounters more files than this,
# stop early and log a warning.  Prevents unbounded memory/time
# on giant monorepos.  Callers should prefer scan_paths() for
# targeted scanning when the command's targets are known.
_DEFAULT_MAX_FILES = 50_000


class WorkspaceScanner:
    """Detects file changes by comparing directory state before/after execution.

    Two scanning modes:

    1. **Full scan** (:meth:`scan`): walks the workspace tree.  Best for
       small-to-medium repos.  Capped by ``max_files`` to avoid runaway
       on giant trees.

    2. **Targeted scan** (:meth:`scan_paths`): stats only specific files
       or shallow directories.  O(targets) regardless of repo size.  Use
       this when :class:`ShellMutationDetector` identifies the likely
       targets.

    Typical usage::

        # Option A — targeted (preferred for large repos)
        targets = ShellMutationDetector.detect_targets(command, cwd)
        target_paths = [p for p, _ in targets]
        before = scanner.scan_paths(target_paths)
        # ... execute ...
        after = scanner.scan_paths(target_paths)

        # Option B — full workspace scan (small repos only)
        before = scanner.scan()
        # ... execute ...
        after = scanner.scan()

        changed = WorkspaceScanner.diff(before, after)
    """

    def __init__(
        self,
        root: str,
        *,
        excludes: frozenset[str] | None = None,
        gitignore: GitignoreFilter | None = None,
        max_depth: int = 10,
        max_files: int = _DEFAULT_MAX_FILES,
        exclude_roots: frozenset[str] | set[str] | None = None,
        deadline: float | None = None,
    ) -> None:
        """
        Args:
            root: Workspace root directory to scan.
            excludes: Directory / file names to skip. Defaults to
                :data:`DEFAULT_EXCLUDES`.
            gitignore: Optional gitignore filter for additional exclusion.
            max_depth: Maximum directory depth to walk (0 = root only).
            max_files: Safety cap on the number of files to stat.  If
                exceeded, the scan stops early and logs a warning.
                Set to 0 for unlimited (not recommended for unknown repos).
        """
        self._root = os.path.normpath(os.path.abspath(root))
        self._excludes = excludes if excludes is not None else DEFAULT_EXCLUDES
        self._gitignore = gitignore
        self._max_depth = max_depth
        self._max_files = max_files
        self._exclude_roots = frozenset(
            os.path.normcase(os.path.realpath(os.path.abspath(path))) for path in (exclude_roots or ())
        )
        self._deadline = deadline

    def scan(self, *, since_ns: int = 0) -> dict[str, FileStat]:
        """Walk the workspace and return ``{abs_path: stat}`` for every file.

        Skips directories in the exclude set, respects max_depth and
        max_files, and applies gitignore filtering.  Symlinks are not
        followed to avoid cycles.

        If the file count exceeds ``max_files``, the scan stops early
        and a warning is logged.  Use :meth:`scan_paths` for targeted
        scanning when this happens.

        Args:
            since_ns: If nonzero, only include files whose ``mtime_ns``
                is >= this value.  Useful for "after" scans where only
                recently-touched files matter.  Files from the "before"
                scan that were NOT modified will be absent from the
                result — ``diff()`` treats these as unchanged (they
                appear in both ``before`` and ``after`` as-is via the
                ``before`` dict passed to ``diff()``).
        """
        result: dict[str, FileStat] = {}
        root = self._root
        root_len = len(root) + 1  # +1 for the separator
        cap = self._max_files

        for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=_raise_scan_error):
            if self._deadline is not None and time.monotonic() >= self._deadline:
                return result
            # Enforce max depth
            depth = dirpath[len(root) :].count(os.sep)
            if depth >= self._max_depth:
                dirnames.clear()
                continue

            # Prune excluded directories (in-place to prevent os.walk descent)
            dirnames[:] = [
                d
                for d in dirnames
                if not self._is_excluded_dir(d, dirpath[root_len:] + os.sep + d if dirpath != root else d)
                and os.path.normcase(os.path.realpath(os.path.join(dirpath, d))) not in self._exclude_roots
            ]

            for fname in filenames:
                if self._deadline is not None and time.monotonic() >= self._deadline:
                    dirnames.clear()
                    return result
                # Budget check
                if cap and len(result) >= cap:
                    logger.warning(
                        "WorkspaceScanner: hit max_files cap (%d) — scan truncated. "
                        "Use scan_paths() for targeted scanning on large repos.",
                        cap,
                    )
                    # Stop the entire walk
                    dirnames.clear()
                    return result

                fpath = os.path.join(dirpath, fname)
                rel = fpath[root_len:] if len(fpath) > root_len else fname

                # Check gitignore
                if self._gitignore and self._gitignore.is_ignored(rel, is_dir=False):
                    continue

                # Check filename against excludes (e.g. .DS_Store, Thumbs.db)
                if fname in self._excludes:
                    continue

                try:
                    st = os.lstat(fpath)
                    if not stat.S_ISREG(st.st_mode):
                        continue
                    if since_ns and st.st_mtime_ns < since_ns:
                        continue
                    result[fpath] = FileStat(
                        mtime_ns=st.st_mtime_ns,
                        size=st.st_size,
                        mode=stat.S_IMODE(st.st_mode),
                    )
                except OSError as exc:
                    _raise_scan_error(exc)

        return result

    def scan_paths(self, paths: list[str], *, shallow_depth: int = 1) -> dict[str, FileStat]:
        """Stat specific files or shallow-scan specific directories.

        Unlike :meth:`scan`, this does NOT walk the full workspace tree.
        It only examines the given *paths*:

        - If a path is a file, stat it directly.
        - If a path is a directory, walk it up to *shallow_depth* levels.
        - If a path doesn't exist, skip it (it may be a deletion target).

        This is O(len(paths)) for files, making it safe for any repo size.

        Args:
            paths: Absolute or relative paths to scan. Relative paths
                are resolved against the scanner's root.
            shallow_depth: When a path is a directory, how many levels
                deep to walk (default 1 = immediate children only).

        Returns:
            ``{abs_path: stat}`` for all discovered regular files.
        """
        result: dict[str, FileStat] = {}

        for raw_path in paths:
            abs_path = os.path.normpath(raw_path if os.path.isabs(raw_path) else os.path.join(self._root, raw_path))

            try:
                entry = os.lstat(abs_path)
            except ValueError:
                # An embedded NUL is invalid, not an unreadable live endpoint.
                continue
            except OSError as exc:
                _raise_scan_error(exc)
                continue
            if stat.S_ISLNK(entry.st_mode):
                # Preserve explicit directory aliases as scan roots. A file
                # link is still excluded by _stat_file; nested directory
                # links remain excluded by walk(followlinks=False).
                try:
                    entry = os.stat(abs_path)
                except OSError as exc:
                    _raise_scan_error(exc)
                    continue
            if stat.S_ISREG(entry.st_mode):
                fs = self._stat_file(abs_path)
                if fs is not None:
                    result[abs_path] = fs

            elif stat.S_ISDIR(entry.st_mode):
                for dirpath, dirnames, filenames in os.walk(abs_path, followlinks=False, onerror=_raise_scan_error):
                    depth = dirpath[len(abs_path) :].count(os.sep)
                    if depth >= shallow_depth:
                        dirnames.clear()
                        continue
                    # Still respect excludes within the shallow walk
                    dirnames[:] = [d for d in dirnames if d not in self._excludes]
                    for fname in filenames:
                        if fname in self._excludes:
                            continue
                        fpath = os.path.join(dirpath, fname)
                        fs = self._stat_file(fpath)
                        if fs is not None:
                            result[fpath] = fs

            # else: path doesn't exist — skip (may be a deletion target)

        return result

    @staticmethod
    def _stat_file(fpath: str) -> FileStat | None:
        """Stat a regular file; return None only for absent or excluded entries."""
        try:
            st = os.lstat(fpath)
            if not stat.S_ISREG(st.st_mode):
                return None
            return FileStat(
                mtime_ns=st.st_mtime_ns,
                size=st.st_size,
                mode=stat.S_IMODE(st.st_mode),
            )
        except OSError as exc:
            _raise_scan_error(exc)
            return None

    def _is_excluded_dir(self, dirname: str, rel_path: str) -> bool:
        """Check if a directory should be skipped."""
        # Exact name match (e.g. ".git", "node_modules")
        if dirname in self._excludes:
            return True
        # Pattern match for entries like "*.egg-info"
        if any(dirname.endswith(pat.removeprefix("*")) for pat in self._excludes if pat.startswith("*")):
            return True
        # Gitignore check — ``is_ignored`` normalises separators internally.
        return bool(self._gitignore and self._gitignore.is_ignored(rel_path, is_dir=True))

    @staticmethod
    def diff(
        before: dict[str, FileStat],
        after: dict[str, FileStat],
    ) -> list[tuple[str, MutationOp]]:
        """Compare two scan results and return detected changes.

        Returns a list of ``(abs_path, operation)`` tuples:

        - Files in *after* but not *before* -> ``CREATE``
        - Files in *before* but not *after* -> ``DELETE``
        - Files in both but with changed mtime or size -> ``MODIFY``

        Note: MOVE is not directly detectable from stat comparison.
        A move appears as ``DELETE`` + ``CREATE``.  Higher-level logic
        can correlate these using content hashes if needed.
        """
        changes: list[tuple[str, MutationOp]] = []

        for path, after_stat in after.items():
            before_stat = before.get(path)
            if before_stat is None:
                changes.append((path, MutationOp.CREATE))
            elif (
                before_stat.mtime_ns != after_stat.mtime_ns
                or before_stat.size != after_stat.size
                or before_stat.mode != after_stat.mode
            ):
                changes.append((path, MutationOp.MODIFY))

        changes.extend((path, MutationOp.DELETE) for path in before if path not in after)

        return changes
