# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Global persisted prompt history for TUI input."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import replace_with_retry
from chrys.foundation.util.lock import FileLock

logger = logging.getLogger(__name__)

HISTORY_FILENAME = "prompt_history.jsonl"
HISTORY_LOCK_FILENAME = "prompt_history.jsonl.lock"
DEFAULT_MAX_ENTRIES = 1000
_COMPACT_MULTIPLIER = 1.1
_WRITER_DRAIN_TIMEOUT_SECONDS = 0.5
_HISTORY_READ_LOCK_TIMEOUT_SECONDS = 3.0
_HISTORY_WRITE_LOCK_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True, slots=True)
class _PromptHistoryWrite:
    """One ordered prompt-history append request."""

    text: str
    session_id: str | None
    instance_id: str | None
    cwd: str | None


class PromptHistoryWriter:
    """Serialize prompt-history appends through one background worker."""

    def __init__(self, append: Callable[..., None] | None = None) -> None:
        self._append = append or append_history
        self._queue: asyncio.Queue[_PromptHistoryWrite | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def enqueue(
        self,
        text: str,
        *,
        session_id: str | None = None,
        instance_id: str | None = None,
        cwd: str | None = None,
    ) -> None:
        """Queue one append without performing file I/O on the caller thread."""
        if self._closed:
            return
        request = _PromptHistoryWrite(
            text=text,
            session_id=session_id,
            instance_id=instance_id,
            cwd=cwd,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Synchronous callers have no event loop to stall; preserve the
            # helper's standalone behavior instead of silently losing history.
            self._append(
                request.text,
                session_id=request.session_id,
                instance_id=request.instance_id,
                cwd=request.cwd,
            )
            return
        self._queue.put_nowait(request)
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._run(), name="prompt-history-writer")

    async def flush(self) -> None:
        """Wait for queued appends, or return if their worker stops before draining."""
        task = self._task
        if self._closed or task is None or task.done():
            return
        drained = asyncio.create_task(self._queue.join(), name="prompt-history-flush")
        try:
            # A close timeout or external cancellation can stop the worker
            # with queued writes. Waiting only on join would then never end.
            await asyncio.wait((drained, task), return_when=asyncio.FIRST_COMPLETED)
        finally:
            drained.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drained

    async def close(self, *, timeout_seconds: float | None = _WRITER_DRAIN_TIMEOUT_SECONDS) -> None:
        """Best-effort drain queued writes and stop the background worker."""
        if self._closed:
            return
        self._closed = True
        task = self._task
        if task is None or task.done():
            return
        self._queue.put_nowait(None)
        try:
            if timeout_seconds is None:
                await asyncio.shield(task)
            else:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _run(self) -> None:
        """Drain queued writes in submission order on worker threads."""
        while True:
            request = await self._queue.get()
            try:
                if request is None:
                    return
                await asyncio.to_thread(
                    self._append,
                    request.text,
                    session_id=request.session_id,
                    instance_id=request.instance_id,
                    cwd=request.cwd,
                )
            except Exception:
                logger.warning("Failed to persist queued prompt history", exc_info=True)
            finally:
                self._queue.task_done()


def history_path() -> Path:
    """Return the global prompt history path."""
    return get_platform().config_dir / HISTORY_FILENAME


def load_history(
    max_entries: int = DEFAULT_MAX_ENTRIES,
    *,
    instance_id: str | None = None,
) -> list[str]:
    """Load the most recent valid prompt history entries.

    Acquires the sidecar file lock so a concurrent ``_compact`` cannot
    rename the file out from under the read on Windows (where an open
    read handle blocks ``os.replace``).
    """
    if _history_disabled() or max_entries <= 0:
        return []
    path = history_path()
    if not path.exists():
        return []
    lock_path = path.with_name(HISTORY_LOCK_FILENAME)
    try:
        with FileLock(lock_path, timeout=_HISTORY_READ_LOCK_TIMEOUT_SECONDS):
            if not path.exists():
                return []
            entries = [
                record["text"]
                for record in _iter_valid_records(path)
                if instance_id is None or record.get("instance_id") == instance_id
            ]
        return entries[-max_entries:]
    except Exception as err:
        logger.warning("Failed to load prompt history from %s: %s", path, err)
        return []


def append_history(
    text: str,
    *,
    session_id: str | None = None,
    instance_id: str | None = None,
    cwd: str | None = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> None:
    """Append one prompt history entry, tolerating all persistence failures."""
    if _history_disabled():
        return
    text = text.strip()
    if not text or max_entries <= 0:
        return

    path = history_path()
    lock_path = path.with_name(HISTORY_LOCK_FILENAME)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(lock_path, timeout=_HISTORY_WRITE_LOCK_TIMEOUT_SECONDS):
            records = list(_iter_valid_records(path)) if path.exists() else []
            if records and _is_consecutive_duplicate(
                records[-1],
                text=text,
                session_id=session_id,
                instance_id=instance_id,
                cwd=cwd,
            ):
                return

            record: dict[str, str] = {
                "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "text": text,
            }
            if session_id is not None:
                record["session_id"] = session_id
            if instance_id is not None:
                record["instance_id"] = instance_id
            if cwd is not None:
                record["cwd"] = cwd

            _ensure_append_boundary(path)
            with path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")

            records.append(record)
            if len(records) > _compact_threshold(max_entries):
                _compact(path, records[-max_entries:])
    except Exception as err:
        logger.warning("Failed to append prompt history to %s: %s", path, err)


def _history_disabled() -> bool:
    from chrys.foundation.config.process_settings import process_settings

    return not process_settings().prompt_history_enabled


def _is_consecutive_duplicate(
    record: dict[str, str],
    *,
    text: str,
    session_id: str | None,
    instance_id: str | None,
    cwd: str | None,
) -> bool:
    return (
        record.get("text") == text
        and record.get("session_id") == session_id
        and record.get("instance_id") == instance_id
        and record.get("cwd") == cwd
    )


def _compact_threshold(max_entries: int) -> int:
    """Return the record count at which append-only history should compact."""
    return max(max_entries + 1, int(max_entries * _COMPACT_MULTIPLIER))


def _iter_valid_records(path: Path) -> list[dict[str, str]]:
    """Return valid records, skipping corrupt rows without dropping the whole file."""
    records: list[dict[str, str]] = []
    with path.open("rb") as fp:
        for raw_line in fp:
            # Decode per line so one bad byte does not wipe all prompt history.
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                continue
            try:
                raw: Any = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            text = raw.get("text")
            if not isinstance(text, str):
                continue
            records.append({k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)})
    return records


def _ensure_append_boundary(path: Path) -> None:
    """Separate a torn final row from the next append so the new record stays parseable."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as fp:
        fp.seek(-1, os.SEEK_END)
        if fp.read(1) != b"\n":
            fp.write(b"\n")


def _compact(path: Path, records: list[dict[str, str]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fp:
            for record in records:
                fp.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
        replace_with_retry(tmp_path, path)
    finally:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
