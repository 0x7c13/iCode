# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Async project path discovery and fuzzy filtering.

The scanner supports million-scale workspaces by streaming subprocess sources
to bounded file budgets and returning explicit truncation metadata instead of
silently stopping.  CPU-heavy fallback walking and file-to-directory suggestion
materialization must stay off the Textual event loop when called from the TUI
warmup path.  Keep ``--sort path`` unless a measured change intentionally
updates empty-query ordering expectations.
"""

from __future__ import annotations

import asyncio
import ntpath
import os
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from chrys.foundation.platform import get_platform
from chrys.foundation.platform.process import (
    MissingWorkingDirectoryError,
    decode_subprocess_output,
    managed_subprocess,
)
from chrys.foundation.vendor import find_rg

ProjectPathKind = Literal["file", "directory"]
ProjectPathOrdering = Literal["path-sorted", "discovery"]

DEFAULT_FILE_DISCOVERY_BUDGET = 1_000_000
DEFAULT_SUGGESTION_BUDGET = 2_000_000


@dataclass(frozen=True, slots=True)
class ProjectPathSuggestion:
    """A referenceable project path shown in the ``@`` suggestion menu."""

    path: str
    kind: ProjectPathKind


@dataclass(frozen=True, slots=True)
class ProjectPathScanResult:
    """Project path suggestions plus scan budget/truncation metadata."""

    root: str
    paths: list[ProjectPathSuggestion]
    file_count: int
    suggestion_count: int
    truncated: bool
    file_budget: int
    suggestion_budget: int
    source_truncations: dict[str, bool] = field(default_factory=dict)
    ordering: ProjectPathOrdering = "path-sorted"

    @classmethod
    def from_suggestions(
        cls,
        *,
        root: str,
        paths: list[ProjectPathSuggestion],
        file_budget: int | None = None,
        suggestion_budget: int | None = None,
        truncated: bool = False,
        source_truncations: dict[str, bool] | None = None,
        ordering: ProjectPathOrdering = "path-sorted",
    ) -> ProjectPathScanResult:
        """Wrap a prebuilt suggestion list in scan-result metadata."""
        file_count = sum(1 for path in paths if path.kind == "file")
        suggestion_count = len(paths)
        return cls(
            root=root,
            paths=paths,
            file_count=file_count,
            suggestion_count=suggestion_count,
            truncated=truncated,
            file_budget=file_budget if file_budget is not None else file_count,
            suggestion_budget=suggestion_budget if suggestion_budget is not None else suggestion_count,
            source_truncations=dict(source_truncations or {}),
            ordering=ordering,
        )


@dataclass(frozen=True, slots=True)
class _PathSourceResult:
    """Bounded file paths from one discovery source."""

    paths: list[str]
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class _SuggestionBuildResult:
    """Bounded file/directory suggestions derived from file paths."""

    suggestions: list[ProjectPathSuggestion]
    truncated: bool = False


_NON_REFERENCEABLE_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".7z",
        ".a",
        ".app",
        ".avi",
        ".bin",
        ".bz2",
        ".class",
        ".db",
        ".db3",
        ".dll",
        ".dmg",
        ".dylib",
        ".ear",
        ".eot",
        ".exe",
        ".feather",
        ".gz",
        ".jar",
        ".mov",
        ".mp3",
        ".mp4",
        ".npy",
        ".npz",
        ".o",
        ".obj",
        ".otf",
        ".parquet",
        ".pickle",
        ".pkl",
        ".pyc",
        ".pyd",
        ".rar",
        ".so",
        ".sqlite",
        ".sqlite3",
        ".tar",
        ".tgz",
        ".ttf",
        ".war",
        ".wasm",
        ".webm",
        ".whl",
        ".woff",
        ".woff2",
        ".xz",
        ".zip",
    }
)
_MAX_SUPPLEMENTAL_IGNORED_FILES = 5000
_NOISE_TOPLEVEL_DIRS: frozenset[str] = frozenset(
    {
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "htmlcov",
        "node_modules",
        "venv",
    }
)
_PLATFORM = get_platform()


def _rg_path() -> str:
    """Best-effort rg path: vendored copy → PATH."""
    return find_rg() or "rg"


async def scan_project_paths(
    root: str,
    max_paths: int | None = None,
    *,
    file_budget: int = DEFAULT_FILE_DISCOVERY_BUDGET,
    suggestion_budget: int = DEFAULT_SUGGESTION_BUDGET,
) -> ProjectPathScanResult:
    """Return relative file and directory paths from *root* with scan metadata.

    Strategy:
    1. ``rg --files`` — fast, .gitignore-aware, lists all non-hidden files
       on the filesystem (tracked AND untracked).
    2. ``git ls-files --cached`` — complements ripgrep with tracked files
       it omits because they are hidden or ignored by Git, ``.ignore``,
       ``.rgignore``, or ripgrep-specific configuration.
    3. ``git ls-files`` — tracked + untracked-but-not-ignored files so that
       newly created files aren't silently dropped.
    4. ``os.walk`` — plain directory walk (no ignore rules, skips dotfiles).

    Ignored visible files are appended by a supplemental pass so user-owned
    private docs like ``drafts/notes.md`` remain referenceable.  That pass
    still rejects dot-prefixed paths, high-noise top-level directories, and
    noisy non-referenceable file extensions.

    Directory suggestions are derived from the final file list.  That means
    folders inherit the exact same ignore/noise behavior as files: empty
    folders and folders containing only non-referenceable files are omitted.
    Directory paths include a trailing slash and the repo root is not included.
    The file discovery budget is counted before these derived directories are
    added; ``suggestion_budget`` bounds the combined file plus directory rows.

    Dot files/folders (e.g. ``.github/``, ``.claude/``) are handled by a
    supplementary pass — see :func:`_append_dot_entries` — and appended to
    the primary result with de-duplication.  This is layered on top
    rather than folded into the primary commands because:

    - ``rg --files --hidden`` also pulls in ``.git/`` (rg's default
      gitignore-aware mode does not drop ``.git/``; confirmed locally),
      so it would need a separate ``--glob '!.git'`` anyway.
    - The ``git ls-files`` tier already includes dot entries, so it's
      untouched; the dedup keeps the output stable regardless of which
      tier produced the primary list.
    """
    if max_paths is not None:
        file_budget = max_paths
        suggestion_budget = max_paths
    file_budget = max(0, file_budget)
    suggestion_budget = max(0, suggestion_budget)
    if not os.path.isdir(root):
        return ProjectPathScanResult(
            root=root,
            paths=[],
            file_count=0,
            suggestion_count=0,
            truncated=False,
            file_budget=file_budget,
            suggestion_budget=suggestion_budget,
            source_truncations={},
        )
    primary: list[str] = []
    source_truncations: dict[str, bool] = {}
    ordering: ProjectPathOrdering = "path-sorted"
    rg_truncated = False

    # Try ripgrep. ``--path-separator /`` forces POSIX separators on Windows
    # so downstream fuzzy matching and string compares behave identically
    # across platforms.
    rg_result = await _scan_rg_files(root, file_budget)
    if rg_result.paths:
        primary = rg_result.paths
        rg_truncated = rg_result.truncated
        if rg_truncated:
            source_truncations["rg"] = True

    # Add tracked files that ripgrep can omit when a path is hidden or ignored.
    if primary and file_budget > 0 and not (rg_truncated and len(primary) >= file_budget):
        seen = set(primary)
        if await _append_git_cached_files(root, primary, seen, file_budget=file_budget):
            source_truncations["git_cached"] = True

    # Try git ls-files (tracked + untracked-but-not-ignored)
    if not primary:
        git_result = await _scan_git_files(root, file_budget)
        if git_result.paths:
            primary = git_result.paths
            if git_result.truncated:
                source_truncations["git_files"] = True

    # Fallback: os.walk
    if not primary:
        walk_result = await asyncio.to_thread(_scan_os_walk_files, root, file_budget)
        primary = walk_result.paths
        ordering = "discovery"
        if walk_result.truncated:
            source_truncations["os_walk"] = True

    # Append supplemental paths. Dedup preserves primary order and keeps
    # lower-priority hidden/ignored entries after the normal project files.
    seen = set(primary)
    if file_budget > 0 and await _append_dot_entries(root, primary, seen, file_budget=file_budget):
        source_truncations["dot_entries"] = True

    if len(primary) < file_budget:
        ignored_result = await _scan_ignored_visible_files(root, file_budget - len(primary))
        if ignored_result.truncated:
            source_truncations["ignored_visible"] = True
        if _append_unique_paths(primary, seen, ignored_result.paths, file_budget=file_budget):
            source_truncations["ignored_visible"] = True
    elif file_budget > 0:
        ignored_result = await _scan_ignored_visible_files(root, 1)
        if ignored_result.truncated or any(path not in seen for path in ignored_result.paths):
            source_truncations["ignored_visible"] = True

    suggestions = await asyncio.to_thread(_project_path_suggestions, primary, suggestion_budget=suggestion_budget)
    if suggestions.truncated:
        source_truncations["suggestions"] = True
    return ProjectPathScanResult(
        root=root,
        paths=suggestions.suggestions,
        file_count=len(primary),
        suggestion_count=len(suggestions.suggestions),
        truncated=any(source_truncations.values()),
        file_budget=file_budget,
        suggestion_budget=suggestion_budget,
        source_truncations=source_truncations,
        ordering=ordering,
    )


def _append_unique_paths(paths: list[str], seen: set[str], candidates: list[str], *, file_budget: int) -> bool:
    """Append unseen candidates up to *file_budget* and report dropped unique paths."""
    for path in candidates:
        if path in seen:
            continue
        if len(paths) >= file_budget:
            return True
        seen.add(path)
        paths.append(path)
    return False


def _project_path_suggestions(
    paths: list[str],
    *,
    suggestion_budget: int | None = None,
    max_paths: int | None = None,
) -> _SuggestionBuildResult:
    """Return file suggestions plus parent-directory suggestions in input order."""
    if suggestion_budget is None:
        suggestion_budget = max_paths if max_paths is not None else DEFAULT_SUGGESTION_BUDGET
    suggestion_budget = max(0, suggestion_budget)
    if suggestion_budget <= 0:
        return _SuggestionBuildResult([], truncated=bool(paths))
    suggestions: list[ProjectPathSuggestion] = []
    seen: set[str] = set()

    def append(path: str, kind: ProjectPathKind) -> bool:
        if path in seen:
            return False
        seen.add(path)
        suggestions.append(ProjectPathSuggestion(path=path, kind=kind))
        return len(suggestions) >= suggestion_budget

    for raw_index, raw_path in enumerate(paths):
        path = raw_path.strip("/")
        if not path:
            continue
        parts = [part for part in path.split("/") if part]
        for index in range(1, len(parts)):
            directory = "/".join(parts[:index]) + "/"
            if append(directory, "directory"):
                return _SuggestionBuildResult(suggestions, truncated=True)
        if append(path, "file"):
            return _SuggestionBuildResult(suggestions, truncated=raw_index < len(paths) - 1)
    return _SuggestionBuildResult(suggestions)


async def _scan_rg_files(root: str, max_files: int) -> _PathSourceResult:
    """Return non-hidden ripgrep file paths from *root*."""
    if max_files <= 0:
        return _PathSourceResult([], truncated=False)
    try:
        async with managed_subprocess(
            _rg_path(),
            "--files",
            "--sort",
            "path",
            "--path-separator",
            "/",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        ) as proc:
            result = await _read_limited_stdout_lines(proc, max_files)
            if result.truncated or proc.returncode == 0:
                return result
    except FileNotFoundError, NotADirectoryError, MissingWorkingDirectoryError:
        pass
    return _PathSourceResult([])


async def _scan_git_files(root: str, max_files: int) -> _PathSourceResult:
    """Return tracked plus untracked-but-not-ignored git files from *root*."""
    return await _scan_git_ls_files(
        root,
        "--cached",
        "--others",
        "--exclude-standard",
        max_files=max_files,
    )


async def _append_git_cached_files(root: str, paths: list[str], seen: set[str], *, file_budget: int) -> bool:
    """Append unseen tracked files and report whether more unseen files remain."""
    try:
        async with managed_subprocess(
            "git",
            "-c",
            "core.quotePath=false",
            "ls-files",
            "--cached",
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        ) as proc:
            return await _append_unseen_stdout_lines(proc, paths, seen, file_budget=file_budget)
    except FileNotFoundError, NotADirectoryError, MissingWorkingDirectoryError:
        return False


async def _scan_git_ls_files(root: str, *args: str, max_files: int) -> _PathSourceResult:
    """Return bounded ``git ls-files`` output for *args*."""
    if max_files <= 0:
        return _PathSourceResult([])
    try:
        async with managed_subprocess(
            "git",
            "-c",
            "core.quotePath=false",
            "ls-files",
            *args,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        ) as proc:
            result = await _read_limited_stdout_lines(proc, max_files)
            if result.truncated or proc.returncode == 0:
                return result
    except FileNotFoundError, NotADirectoryError, MissingWorkingDirectoryError:
        pass
    return _PathSourceResult([])


def _scan_os_walk_files(root: str, max_files: int) -> _PathSourceResult:
    """Return a bounded plain directory walk of visible files."""
    if max_files <= 0:
        return _PathSourceResult([])
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname for dirname in dirnames if not dirname.startswith(".") and _is_referenceable_walk_name(dirname)
        ]
        for fname in filenames:
            if fname.startswith(".") or not _is_referenceable_walk_name(fname):
                continue
            rel = _walk_relative_path(root, dirpath, fname)
            if len(paths) >= max_files:
                return _PathSourceResult(paths, truncated=True)
            paths.append(rel)
    return _PathSourceResult(paths)


async def _append_dot_entries(root: str, paths: list[str], seen: set[str], *, file_budget: int) -> bool:
    """Append useful top-level dot-entry files and report whether more unseen files remain."""
    if file_budget <= 0:
        return False
    try:
        dot_entries = sorted(e for e in os.listdir(root) if e.startswith(".") and e != ".git")
    except OSError:
        return False
    if not dot_entries:
        return False

    try:
        async with managed_subprocess(
            "git",
            "-c",
            "core.quotePath=false",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *dot_entries,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        ) as proc:
            truncated = await _append_unseen_stdout_lines(proc, paths, seen, file_budget=file_budget)
            if truncated or proc.returncode == 0:
                return truncated
    except FileNotFoundError, NotADirectoryError, MissingWorkingDirectoryError:
        pass

    return await asyncio.to_thread(
        _append_unseen_dot_entries_from_walk,
        root,
        dot_entries,
        paths,
        seen,
        file_budget=file_budget,
    )


def _append_unseen_dot_entries_from_walk(
    root: str,
    dot_entries: list[str],
    paths: list[str],
    seen: set[str],
    *,
    file_budget: int,
) -> bool:
    """Append unseen dot-entry fallback walk paths and report omitted unseen paths."""
    for entry in dot_entries:
        full = os.path.join(root, entry)
        if os.path.isfile(full):
            if entry in seen:
                continue
            if len(paths) >= file_budget:
                return True
            seen.add(entry)
            paths.append(entry)
            continue
        if not os.path.isdir(full):
            continue
        for dirpath, dirnames, filenames in os.walk(full):
            dirnames[:] = [dirname for dirname in dirnames if _is_referenceable_walk_name(dirname)]
            for fname in filenames:
                if not _is_referenceable_walk_name(fname):
                    continue
                rel = _walk_relative_path(root, dirpath, fname)
                if rel in seen:
                    continue
                if len(paths) >= file_budget:
                    return True
                seen.add(rel)
                paths.append(rel)
    return False


def _is_referenceable_walk_name(name: str) -> bool:
    """Return whether *name* can be referenced as a filesystem component on this platform."""
    return not (_PLATFORM.is_windows and ntpath.isreserved(name))


def _walk_relative_path(root: str, dirpath: str, filename: str) -> str:
    """Return an ``os.walk`` file path relative to *root* without resolving device names.

    ``os.walk`` preserves its root spelling as the lexical prefix of every
    yielded directory.  Slicing that prefix avoids ``os.path.relpath`` and its
    Windows ``GetFullPathName`` normalization, which rewrites reserved final
    components such as ``nul`` into the ``\\\\.\\nul`` device mount.
    """
    full_path = os.path.join(dirpath, filename)
    return full_path[len(root) :].lstrip(os.sep).replace(os.sep, "/")


async def _scan_ignored_visible_files(root: str, max_files: int) -> _PathSourceResult:
    """Return gitignored visible files that are still useful for ``@`` references."""
    budget = min(max_files, _MAX_SUPPLEMENTAL_IGNORED_FILES)
    if budget <= 0:
        return _PathSourceResult([])

    exclude_pathspecs = [":(exclude).*", *(f":(exclude){entry}" for entry in sorted(_NOISE_TOPLEVEL_DIRS))]
    paths: list[str] = []
    try:
        async with managed_subprocess(
            "git",
            "-c",
            "core.quotePath=false",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            *exclude_pathspecs,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        ) as proc:
            if proc.stdout is None:
                await proc.wait()
                return _PathSourceResult([])
            lines_read = 0
            while True:
                line = await proc.stdout.readline()
                if not line:
                    await proc.wait()
                    break
                lines_read += 1
                path = decode_subprocess_output(line).rstrip("\r\n")
                if _include_ignored_visible_path(path) and not _has_non_referenceable_file_extension(path):
                    paths.append(path)
                    if len(paths) >= budget:
                        extra = await proc.stdout.readline()
                        return _PathSourceResult(paths, truncated=bool(extra))
                if lines_read >= _MAX_SUPPLEMENTAL_IGNORED_FILES:
                    extra = await proc.stdout.readline()
                    return _PathSourceResult(paths, truncated=bool(extra))
            return _PathSourceResult(paths)
    except FileNotFoundError, NotADirectoryError, MissingWorkingDirectoryError:
        pass
    return _PathSourceResult([])


async def _read_limited_stdout_lines(proc: asyncio.subprocess.Process, max_lines: int) -> _PathSourceResult:
    """Read up to *max_lines* from process stdout and detect one extra line."""
    if max_lines <= 0 or proc.stdout is None:
        await proc.wait()
        return _PathSourceResult([])
    paths: list[str] = []
    while len(paths) < max_lines:
        line = await proc.stdout.readline()
        if not line:
            await proc.wait()
            return _PathSourceResult(paths)
        paths.append(decode_subprocess_output(line).rstrip("\r\n"))
    extra = await proc.stdout.readline()
    if extra:
        return _PathSourceResult(paths, truncated=True)
    await proc.wait()
    return _PathSourceResult(paths)


async def _append_unseen_stdout_lines(
    proc: asyncio.subprocess.Process,
    paths: list[str],
    seen: set[str],
    *,
    file_budget: int,
) -> bool:
    """Append unseen stdout lines up to *file_budget* and report omitted unseen lines."""
    if proc.stdout is None:
        await proc.wait()
        return False
    while True:
        line = await proc.stdout.readline()
        if not line:
            await proc.wait()
            return False
        path = decode_subprocess_output(line).rstrip("\r\n")
        if path in seen:
            continue
        if len(paths) >= file_budget:
            return True
        seen.add(path)
        paths.append(path)


def _include_ignored_visible_path(path: str) -> bool:
    """Return whether an ignored path should still be offered in ``@`` suggestions."""
    parts = [part for part in path.replace("\\", "/").split("/") if part and part != "."]
    if not parts:
        return False
    if any(part.startswith(".") for part in parts):
        return False
    return parts[0] not in _NOISE_TOPLEVEL_DIRS


def _has_non_referenceable_file_extension(path: str) -> bool:
    """Return whether *path* has a noisy extension that is poor ``@`` material."""
    _, ext = os.path.splitext(path)
    return ext.casefold() in _NON_REFERENCEABLE_FILE_EXTENSIONS


def _normalize_for_search(value: str) -> str:
    """Normalize path/query text for ``@`` search comparisons."""
    return unicodedata.normalize("NFC", value).casefold()


def fuzzy_filter(query: str, candidates: list[str], max_results: int = 30) -> list[str]:
    """Case-insensitive substring match, sorted by match position then length.

    Normalises both query and candidates to NFC so that CJK characters
    (Korean Hangul, Japanese dakuten, etc.) match regardless of whether the
    filesystem returns NFD paths (macOS) or NFC paths (Linux/Windows).
    """
    if not query:
        return candidates[:max_results]
    q = _normalize_for_search(query)
    scored: list[tuple[int, int, str]] = []
    for c in candidates:
        idx = _normalize_for_search(c).find(q)
        if idx >= 0:
            scored.append((idx, len(c), c))
    scored.sort()
    return [s[2] for s in scored[:max_results]]
