# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Persisted MRU index of recently used workspace directories.

A small global JSON file (``<config_dir>/workspace_mru.json``) that quick-access
surfaces — first consumer: the Change Directory picker's recent paths — read
instead of scanning every saved session.  The index is created once per
session root by :func:`ensure_workspace_mru_index` (seeded from existing
session metadata) and afterwards maintained incrementally by
:func:`schedule_workspace_mru_touches` from confirmed workspace events; it
never scans sessions itself and deliberately does not import ``StateStore``.

Concurrency follows the prompt-history model: every read and write holds the
sidecar :class:`~chrys.foundation.util.lock.FileLock` (reads too, so a
Windows ``os.replace`` cannot race an open read handle) and writes go through
:func:`~chrys.foundation.platform.files.atomic_write_text`.

Invariants:

- Only :func:`ensure_workspace_mru_index` may create the file or mark a
  session root as backfilled; event-driven touches no-op while the index is
  missing so they can never pre-empt the one-time seed.
- A corrupt index still counts as "exists" and as "backfilled" — corruption
  must never re-trigger a full session scan; the next touch rewrites a valid
  file from empty.  Rebuilt files carry the wildcard backfill marker
  (``"*"``) so that promise survives the repair — without it, the repaired
  file would report every root as never-backfilled and re-run the scan.
- Entry order is decided by ``last_used_at`` (descending), never by
  background-task completion order; stale touches cannot downgrade a newer
  timestamp for the same path.
- Default favorites, missing directories, and relative paths are filtered on
  both write and read so they never occupy ``max_entries`` slots.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NotRequired, TypedDict

from chrys.app.tui.support.path_shortcuts import get_default_favorite_paths
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import atomic_write_text
from chrys.foundation.util.lock import FileLock

logger = logging.getLogger(__name__)

WORKSPACE_MRU_FILENAME = "workspace_mru.json"
WORKSPACE_MRU_LOCK_FILENAME = "workspace_mru.json.lock"
_SCHEMA_VERSION = 1
_BACKFILL_WILDCARD = "*"
"""Marks every session root as backfilled; real keys are ``sha256:``-prefixed."""


@dataclass(frozen=True, slots=True)
class WorkspaceMruEntry:
    """One recorded workspace directory."""

    path: str
    last_used_at: datetime
    session_id: str = ""


class _WorkspaceMruIndex(TypedDict):
    """Validated mutable shape retained from the user-editable JSON index."""

    version: NotRequired[object]
    created_at: NotRequired[object]
    updated_at: NotRequired[object]
    backfilled_session_roots: list[object]
    paths: list[object]


def workspace_mru_path() -> Path:
    """Return the global workspace MRU index path."""
    return get_platform().config_dir / WORKSPACE_MRU_FILENAME


def workspace_mru_exists() -> bool:
    """Return whether the MRU index file exists (corrupt files count)."""
    return workspace_mru_path().exists()


def session_root_key(session_root: Path | str) -> str:
    """Return the stable backfill key for a resolved session storage root.

    The file stores a hash rather than the raw local path so the schema does
    not leak filesystem layout and stays comparable across path spellings.
    """
    resolved = Path(session_root).expanduser().resolve(strict=False)
    normalized = os.path.normcase(str(resolved))
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def workspace_mru_backfilled_for(root_key: str) -> bool:
    """Return whether *root_key* already had its one-time session backfill.

    A corrupt or unreadable index reports ``True`` ("already handled") so a
    damaged file can never re-trigger the expensive full session scan.
    """
    path = workspace_mru_path()
    if not path.exists():
        return False
    try:
        with FileLock(_lock_path(path)):
            data = _read_index(path)
    except Exception:
        logger.debug("Failed to read workspace MRU index %s", path, exc_info=True)
        return True
    if data is None:
        return True
    return _root_backfilled(data, root_key)


def load_workspace_mru(max_entries: int) -> list[str]:
    """Return recorded workspace paths, most recent first.

    Filters default favorites and missing directories before trimming so a
    reduced ``max_entries`` takes effect immediately on read.  Corrupt files
    yield an empty list without raising.
    """
    if max_entries <= 0:
        return []
    path = workspace_mru_path()
    if not path.exists():
        return []
    try:
        with FileLock(_lock_path(path)):
            data = _read_index(path)
    except Exception:
        logger.debug("Failed to load workspace MRU index %s", path, exc_info=True)
        return []
    if data is None:
        return []
    entries = _usable_entries(_entries_from(data))
    return [entry.path for entry in entries[:max_entries]]


def ensure_workspace_mru_index(
    seed: Sequence[tuple[str, datetime]],
    *,
    max_entries: int,
    root_key: str,
) -> list[str]:
    """Create the index and/or apply the one-time backfill for *root_key*.

    This is the only API allowed to create ``workspace_mru.json`` or mark a
    session root as backfilled.  *seed* pairs each candidate path with the
    timestamp it was last used (session ``updated_at``), so merged seeds slot
    truthfully between entries recorded by live workspace events.

    Re-checks state under the lock: when the root is already backfilled the
    passed seed is ignored and the existing content is returned, so
    concurrent first-time startups cannot overwrite each other.

    Returns the resulting paths, most recent first, trimmed to
    ``max_entries``.
    """
    if max_entries <= 0:
        return []
    path = workspace_mru_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    with FileLock(_lock_path(path)):
        existed = path.exists()
        data = _read_index(path) if existed else None
        rebuilt_from_corrupt = existed and data is None
        if data is None:
            # Missing or corrupt — start from a fresh, valid structure.
            data = _new_index(now)
            if rebuilt_from_corrupt:
                data["backfilled_session_roots"].append(_BACKFILL_WILDCARD)
        entries = _entries_from(data)
        if not _root_backfilled(data, root_key):
            favorites = get_default_favorite_paths()
            for raw_path, used_at in seed:
                prepared = _prepare_path(raw_path)
                if prepared is None or not _path_usable(prepared, favorites):
                    continue
                _merge_entry(entries, prepared, used_at, session_id="")
            data["backfilled_session_roots"].append(root_key)
            entries = _write_index(path, data, entries, max_entries=max_entries, now=now)
        elif rebuilt_from_corrupt:
            # Wildcard says "backfilled" so the seed is skipped, but the
            # repaired structure must still replace the corrupt bytes.
            entries = _write_index(path, data, entries, max_entries=max_entries, now=now)
        else:
            entries = _usable_entries(entries)
    return [entry.path for entry in entries[:max_entries]]


def touch_workspace_mru(
    path: str,
    *,
    max_entries: int,
    session_id: str = "",
    used_at: datetime | None = None,
) -> bool:
    """Record one confirmed workspace directory use.

    No-ops (returning ``False``) when the index does not exist yet — event
    touches must never create it ahead of the one-time backfill — or when the
    path is relative, a default favorite, or missing on disk.  Relative paths
    are dropped rather than resolved against the process cwd.
    """
    return record_workspace_mru_uses(
        [path],
        max_entries=max_entries,
        session_id=session_id,
        used_at=used_at,
    )


def record_workspace_mru_uses(
    paths: Sequence[str],
    *,
    max_entries: int,
    session_id: str = "",
    used_at: datetime | None = None,
) -> bool:
    """Record several confirmed uses from one event in a single locked write.

    List order is authoritative for equal timestamps: each later path gets a
    microsecond-earlier synthetic ``last_used_at`` so the primary cwd stays
    ahead of secondary working dirs regardless of write interleaving.
    """
    if max_entries <= 0 or not paths:
        return False
    base_used_at = used_at if used_at is not None else datetime.now(UTC)
    index_path = workspace_mru_path()
    favorites = get_default_favorite_paths()
    prepared: list[str] = []
    for raw_path in paths:
        candidate = _prepare_path(raw_path)
        if candidate is None:
            logger.debug("Dropping non-absolute workspace MRU path: %r", raw_path)
            continue
        if _path_usable(candidate, favorites) and candidate not in prepared:
            prepared.append(candidate)
    if not prepared:
        return False
    if not index_path.exists():
        # Fast path; also keeps the lock file from being created in a config
        # dir that does not exist yet (FileLock requires the parent dir).
        return False
    with FileLock(_lock_path(index_path)):
        if not index_path.exists():
            return False
        data = _read_index(index_path)
        if data is None:
            # Corrupt content: the file exists, so rebuild a valid index from
            # empty rather than failing every subsequent touch.  The wildcard
            # keeps every root "backfilled" across the repair — corruption
            # must never re-trigger the full session scan.
            data = _new_index(base_used_at)
            data["backfilled_session_roots"].append(_BACKFILL_WILDCARD)
        entries = _entries_from(data)
        for offset, candidate in enumerate(prepared):
            _merge_entry(
                entries,
                candidate,
                base_used_at - timedelta(microseconds=offset),
                session_id=session_id,
            )
        _write_index(index_path, data, entries, max_entries=max_entries, now=datetime.now(UTC))
    return True


def schedule_workspace_mru_touches(
    paths: Iterable[str],
    *,
    max_entries: int,
    session_id: str = "",
    used_at: datetime | None = None,
) -> None:
    """Fire-and-forget MRU recording for a confirmed workspace event.

    Runs the locked read-modify-write in a worker thread via a detached task
    so EventBus handlers (which are awaited inline by ``publish``) never block
    on file locks.  Skipped silently while the index does not exist — that
    protects the one-time backfill semantics and is not an error.  Failures
    are logged at debug level and never surface to the UI.
    """
    if max_entries <= 0:
        return
    candidates = [path for path in dict.fromkeys(paths) if path]
    if not candidates or not workspace_mru_exists():
        return
    task = asyncio.create_task(
        _record_uses_in_background(
            candidates,
            max_entries=max_entries,
            session_id=session_id,
            used_at=used_at if used_at is not None else datetime.now(UTC),
        )
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


# Strong references so fire-and-forget touch tasks cannot be GC'd mid-flight.
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


async def _record_uses_in_background(
    paths: list[str],
    *,
    max_entries: int,
    session_id: str,
    used_at: datetime,
) -> None:
    try:
        await asyncio.to_thread(
            record_workspace_mru_uses,
            paths,
            max_entries=max_entries,
            session_id=session_id,
            used_at=used_at,
        )
    except Exception:
        logger.debug("Failed to update workspace MRU", exc_info=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _lock_path(index_path: Path) -> Path:
    return index_path.with_name(WORKSPACE_MRU_LOCK_FILENAME)


def _root_backfilled(data: _WorkspaceMruIndex, root_key: str) -> bool:
    roots = data["backfilled_session_roots"]
    return _BACKFILL_WILDCARD in roots or root_key in roots


def _new_index(now: datetime) -> _WorkspaceMruIndex:
    stamp = _format_timestamp(now)
    return {
        "version": _SCHEMA_VERSION,
        "created_at": stamp,
        "updated_at": stamp,
        "backfilled_session_roots": [],
        "paths": [],
    }


def _read_index(path: Path) -> _WorkspaceMruIndex | None:
    """Parse the index file; ``None`` means unreadable/invalid content."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(raw, dict):
        return None
    paths = raw.get("paths")
    roots = raw.get("backfilled_session_roots")
    if not isinstance(paths, list) or not isinstance(roots, list):
        return None
    data: _WorkspaceMruIndex = raw
    return data


def _entries_from(data: _WorkspaceMruIndex) -> list[WorkspaceMruEntry]:
    entries: list[WorkspaceMruEntry] = []
    seen: set[str] = set()
    for item in data.get("paths", []):
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path")
        used_at = _parse_timestamp(item.get("last_used_at"))
        if not isinstance(raw_path, str) or not raw_path or used_at is None:
            continue
        key = os.path.normcase(raw_path)
        if key in seen:
            continue
        seen.add(key)
        raw_session_id = item.get("session_id")
        entries.append(
            WorkspaceMruEntry(
                path=raw_path,
                last_used_at=used_at,
                session_id=raw_session_id if isinstance(raw_session_id, str) else "",
            )
        )
    return entries


def _merge_entry(entries: list[WorkspaceMruEntry], path: str, used_at: datetime, *, session_id: str) -> None:
    """Insert or refresh *path*, never downgrading a newer ``last_used_at``."""
    key = os.path.normcase(path)
    for i, existing in enumerate(entries):
        if os.path.normcase(existing.path) != key:
            continue
        if used_at > existing.last_used_at:
            entries[i] = WorkspaceMruEntry(
                path=path,
                last_used_at=used_at,
                session_id=session_id or existing.session_id,
            )
        return
    entries.append(WorkspaceMruEntry(path=path, last_used_at=used_at, session_id=session_id))


def _write_index(
    path: Path,
    data: _WorkspaceMruIndex,
    entries: list[WorkspaceMruEntry],
    *,
    max_entries: int,
    now: datetime,
) -> list[WorkspaceMruEntry]:
    """Filter, order, trim, and atomically persist *entries*; return the kept list."""
    kept = _usable_entries(entries)[:max_entries]
    data["paths"] = [
        {
            "path": entry.path,
            "last_used_at": _format_timestamp(entry.last_used_at),
            **({"session_id": entry.session_id} if entry.session_id else {}),
        }
        for entry in kept
    ]
    data["updated_at"] = _format_timestamp(now)
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return kept


def _usable_entries(entries: list[WorkspaceMruEntry]) -> list[WorkspaceMruEntry]:
    """Order by recency and drop favorites/missing dirs (they never hold slots)."""
    favorites = get_default_favorite_paths()
    ordered = sorted(entries, key=lambda entry: entry.last_used_at, reverse=True)
    return [entry for entry in ordered if _path_usable(entry.path, favorites)]


def _path_usable(path: str, favorite_keys: set[str]) -> bool:
    if os.path.normcase(path) in favorite_keys:
        return False
    return os.path.isdir(path)


def _prepare_path(path: str) -> str | None:
    """Normalize an input path; ``None`` drops relative/empty input.

    Never resolves relative paths against the process cwd — the process cwd
    is unrelated to the workspace the event described.
    """
    if not path:
        return None
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        return None
    return os.path.normpath(expanded)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
