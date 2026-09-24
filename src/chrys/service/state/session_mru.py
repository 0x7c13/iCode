# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Per-root MRU index of the most recently updated sessions.

``<sessions_root>/session_mru.json`` is a small derived cache — at most
:data:`DEFAULT_SESSION_MRU_MAX_ENTRIES` ``(session_id, last_updated_at)``
pairs ordered by ``meta.updated_at`` — that lets ``/resume`` find the newest
session without parsing every envelope under the root.  Session files remain
the only source of truth: the store maintains the index incrementally on
save / recovery save / fork / delete, and a missing, corrupt or incomplete
index only costs one full listing scan followed by :meth:`SessionMruIndex.rebuild`.

Concurrency follows the workspace-MRU model: every read and write holds the
sidecar :class:`~chrys.foundation.util.lock.FileLock` and writes go through
:func:`~chrys.foundation.platform.files.atomic_write_text`.  The lock is a
leaf: the store takes it *inside* a session write lock (record before the
envelope commit, so a crash can only leave the index ahead of disk — never
behind it) and nothing else is acquired while it is held.  Every access is
bounded by :data:`SESSION_MRU_LOCK_TIMEOUT_SECONDS`: a stuck peer must
never stall session persistence or ``/resume`` — writers raise
``TimeoutError`` and the store carries on (the store's post-lookup sweep of
recently modified session folders is the safety net for anything that
bypassed the index), ``load()`` reads as "no usable index" and the caller
falls back to a full scan.  ``record()`` never lets a stale writer
downgrade a newer timestamp, and ``horizon`` keeps an upper bound on
everything trimmed out of the index so a lookup can tell when a downgraded
winner might have fallen below an unindexed session.  Any structurally
invalid content — including a single malformed entry — reads as "no
index", because a complete index that silently lost an entry would hide
that session for good.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chrys.foundation.platform.files import atomic_write_text
from chrys.foundation.util.lock import FileLock

logger = logging.getLogger(__name__)

DEFAULT_SESSION_MRU_MAX_ENTRIES = 20
SESSION_MRU_FILE_NAME = "session_mru.json"
SESSION_MRU_LOCK_FILE_NAME = "session_mru.lock"
SESSION_MRU_LOCK_TIMEOUT_SECONDS = 2.0
_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SessionMruEntry:
    """One indexed session: full canonical id + its ``meta.updated_at``."""

    session_id: str
    last_updated_at: datetime


@dataclass(frozen=True, slots=True)
class SessionMruSnapshot:
    """Parsed index content; ``sessions`` is deduplicated and sorted newest-first.

    ``horizon`` is an upper bound on the recorded ``updated_at`` of every
    session that was trimmed out of the index (``None`` when nothing was).
    """

    complete: bool
    sessions: tuple[SessionMruEntry, ...]
    horizon: datetime | None = None


_UTC_MIN = datetime.min.replace(tzinfo=UTC)
_UTC_MAX = datetime.max.replace(tzinfo=UTC)


def coerce_utc(value: datetime) -> datetime:
    """Return *value* as an aware UTC datetime (naive input is taken as UTC).

    Total: an offset that would push a stamp within a day of ``datetime.min``
    / ``datetime.max`` out of range clamps to that bound instead of raising.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    try:
        return value.astimezone(UTC)
    except OverflowError, ValueError:
        return _UTC_MIN if value.year == _UTC_MIN.year else _UTC_MAX


def sort_entries(entries: Iterable[SessionMruEntry]) -> list[SessionMruEntry]:
    """Newest first; equal timestamps fall back to session id for a stable order."""
    by_id = sorted(entries, key=lambda entry: entry.session_id)
    return sorted(by_id, key=lambda entry: entry.last_updated_at, reverse=True)


class SessionMruIndex:
    """Locked read-modify-write access to one sessions root's MRU index."""

    def __init__(self, sessions_root: Path, *, max_entries: int = DEFAULT_SESSION_MRU_MAX_ENTRIES) -> None:
        self._root = Path(sessions_root)
        self._max_entries = max_entries

    @property
    def path(self) -> Path:
        return self._root / SESSION_MRU_FILE_NAME

    @property
    def lock_path(self) -> Path:
        return self._root / ".locks" / SESSION_MRU_LOCK_FILE_NAME

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load(self) -> SessionMruSnapshot | None:
        """Return the parsed index, or ``None`` when missing/unreadable/invalid.

        A lock timeout or an unreadable file simply reads as "no usable
        index" so the caller falls back to a full scan.
        """
        if not self.path.exists():
            return None
        try:
            with self._lock():
                data = self._read()
        except Exception:
            logger.debug("Failed to read session MRU index %s", self.path, exc_info=True)
            return None
        if data is None:
            return None
        return SessionMruSnapshot(
            complete=data["complete"],
            sessions=tuple(self._entries_from(data)),
            horizon=_parse_timestamp(data.get("horizon")),
        )

    def record(self, session_id: str, updated_at: datetime) -> None:
        """Insert or refresh *session_id* at *updated_at*.

        A missing or corrupt file is recreated with ``complete=false`` so the
        next ``/resume`` backfills it from a full scan.  An older timestamp
        never downgrades a newer entry, and a write is skipped when the
        resulting content would be unchanged.
        """
        stamp = coerce_utc(updated_at)
        with self._lock():
            data = self._read()
            if data is None:
                data = self._new_index(complete=False)
            entries = self._entries_from(data)
            for i, existing in enumerate(entries):
                if existing.session_id != session_id:
                    continue
                if stamp <= existing.last_updated_at:
                    return
                entries[i] = SessionMruEntry(session_id, stamp)
                break
            else:
                entries.append(SessionMruEntry(session_id, stamp))
            self._write(data, entries)

    def remove(self, session_id: str) -> None:
        """Drop *session_id* from the index; a no-op when absent."""
        with self._lock():
            data = self._read()
            if data is None:
                return
            entries = self._entries_from(data)
            kept = [entry for entry in entries if entry.session_id != session_id]
            if len(kept) == len(entries):
                return
            self._write(data, kept)

    def rebuild(self, entries: Sequence[SessionMruEntry]) -> tuple[SessionMruEntry, ...]:
        """Merge full-scan results into the index, mark it complete, return the merged ranking.

        *entries* is the complete listing (not pre-trimmed): the write trims
        it and raises ``horizon`` over what was dropped.  The current content
        is re-read under the lock and, per session, the newer of the scanned
        and the already-recorded timestamp wins — so a save that landed while
        the scan ran is neither lost nor overwritten by the scan's older
        value.  The horizon is likewise never lowered: a record that landed
        during the scan and was trimmed straight away survives only there.
        The return value is the whole merged ranking (sorted, NOT trimmed —
        the file keeps only the top ``max_entries``), so a caller can rank
        the post-merge state instead of the scan's, including a concurrent
        record that the cap trimmed straight away.
        """
        with self._lock():
            data = self._read()
            if data is None:
                data = self._new_index(complete=True)
            merged = {entry.session_id: entry for entry in self._entries_from(data)}
            for entry in entries:
                stamped = SessionMruEntry(entry.session_id, coerce_utc(entry.last_updated_at))
                current = merged.get(stamped.session_id)
                if current is None or stamped.last_updated_at > current.last_updated_at:
                    merged[stamped.session_id] = stamped
            data["complete"] = True
            ordered = sort_entries(merged.values())
            self._write(data, ordered)
            return tuple(ordered)

    def invalidate(self) -> None:
        """Best-effort removal of the index so the next lookup rebuilds it.

        Takes the MRU lock like every other access so a concurrent locked
        writer cannot re-materialize a stale-complete index around the unlink.
        """
        try:
            with self._lock():
                self.path.unlink()
        except FileNotFoundError:
            return
        except Exception:
            logger.warning("Failed to invalidate session MRU index %s", self.path, exc_info=True)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _lock(self) -> FileLock:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(self.lock_path, timeout=SESSION_MRU_LOCK_TIMEOUT_SECONDS)

    def _read(self) -> dict[str, Any] | None:
        """Parse the index; ``None`` means missing or (even partially) invalid."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except OSError, ValueError:
            logger.debug("Invalid session MRU index %s", self.path, exc_info=True)
            return None
        if (
            not isinstance(raw, dict)
            or raw.get("version") != _SCHEMA_VERSION
            or not isinstance(raw.get("complete"), bool)
            or not isinstance(raw.get("sessions"), list)
        ):
            return None
        horizon = raw.get("horizon")
        if horizon is not None and _parse_timestamp(horizon) is None:
            return None
        for item in raw["sessions"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("session_id"), str)
                or not item["session_id"]
                or _parse_timestamp(item.get("last_updated_at")) is None
            ):
                logger.debug("Malformed entry in session MRU index %s: %r", self.path, item)
                return None
        return raw

    @staticmethod
    def _new_index(*, complete: bool) -> dict[str, Any]:
        stamp = _format_timestamp(datetime.now(UTC))
        return {
            "version": _SCHEMA_VERSION,
            "complete": complete,
            "created_at": stamp,
            "updated_at": stamp,
            "horizon": None,
            "sessions": [],
        }

    @staticmethod
    def _entries_from(data: dict[str, Any]) -> list[SessionMruEntry]:
        """Deduplicated (newest wins) entries of a ``_read()``-validated index, newest first."""
        by_id: dict[str, SessionMruEntry] = {}
        for item in data["sessions"]:
            session_id = item["session_id"]
            stamp = _parse_timestamp(item["last_updated_at"])
            if stamp is None:  # cannot happen after _read(); keep the loop total
                continue
            current = by_id.get(session_id)
            if current is None or stamp > current.last_updated_at:
                by_id[session_id] = SessionMruEntry(session_id, stamp)
        return sort_entries(by_id.values())

    def _write(self, data: dict[str, Any], entries: list[SessionMruEntry]) -> list[SessionMruEntry]:
        """Trim, raise ``horizon`` over anything dropped, persist; return the kept entries."""
        ordered = sort_entries(entries)
        kept, dropped = ordered[: self._max_entries], ordered[self._max_entries :]
        if dropped:
            horizon = _parse_timestamp(data.get("horizon"))
            newest_dropped = dropped[0].last_updated_at
            if horizon is None or newest_dropped > horizon:
                horizon = newest_dropped
            data["horizon"] = _format_timestamp(horizon)
        data["sessions"] = [
            {"session_id": entry.session_id, "last_updated_at": _format_timestamp(entry.last_updated_at)}
            for entry in kept
        ]
        data["updated_at"] = _format_timestamp(datetime.now(UTC))
        atomic_write_text(self.path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return kept


def _format_timestamp(value: datetime) -> str:
    return coerce_utc(value).isoformat().replace("+00:00", "Z")


def _parse_timestamp(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return coerce_utc(datetime.fromisoformat(raw))
    except ValueError:
        return None
