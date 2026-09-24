# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded, process-safe file logging for the TUI debug log."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from dataclasses import dataclass
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, ClassVar

from chrys.foundation.util.lock import FileLock

_DEBUG_LOG_MAX_BYTES = 16 * 1024 * 1024
_DEBUG_LOG_BACKUP_COUNT = 1
_DEBUG_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
_DEBUG_LOG_TMP_SUFFIX = "tmp"
_DEBUG_LOG_QUEUE_MAX_RECORDS = 4096
_DEBUG_LOG_LOCK_TIMEOUT_SECONDS = 0.25
_DEBUG_LOG_STOP_BUDGET_SECONDS = 0.5


@dataclass(frozen=True)
class DebugLogRuntime:
    """Runtime pieces needed to flush and tear down queued debug logging."""

    queue_handler: QueueHandler
    listener: QueueListener
    file_handler: RotatingFileHandler


class _BoundedDebugLogHandler(QueueHandler):
    """Queue log records without allowing debug logging to grow memory unboundedly."""

    queue: Queue[logging.LogRecord | None]

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
            return
        except Full:
            pass

        with contextlib.suppress(Empty):
            dropped = self.queue.get_nowait()
            self.queue.task_done()
            if dropped is None:
                _enqueue_debug_log_sentinel(self.queue, dropped)
                return
        with contextlib.suppress(Full):
            self.queue.put_nowait(record)


if TYPE_CHECKING:

    class _QueueListenerBase(QueueListener):
        _sentinel: ClassVar[None]

else:
    _QueueListenerBase = QueueListener


class _BoundedDebugLogListener(_QueueListenerBase):
    """Queue listener that can always enqueue its shutdown sentinel."""

    queue: Queue[logging.LogRecord | None]
    _stop_deadline: float | None = None

    def stop(self) -> None:
        self._stop_deadline = time.monotonic() + _DEBUG_LOG_STOP_BUDGET_SECONDS
        try:
            super().stop()
        finally:
            self._stop_deadline = None

    def handle(self, record: logging.LogRecord) -> None:
        deadline = self._stop_deadline
        if deadline is not None and time.monotonic() > deadline:
            return
        super().handle(record)

    def enqueue_sentinel(self) -> None:
        while True:
            try:
                self.queue.put_nowait(self._sentinel)
                return
            except Full:
                with contextlib.suppress(Empty):
                    dropped = self.queue.get_nowait()
                    self.queue.task_done()
                    if dropped is self._sentinel:
                        _enqueue_debug_log_sentinel(self.queue, dropped)
                        return


class _DebugRotatingFileHandler(RotatingFileHandler):
    """Rotating debug log handler that does not keep the shared log open."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with FileLock(_debug_log_lock_path(Path(self.baseFilename)), timeout=_DEBUG_LOG_LOCK_TIMEOUT_SECONDS):
                super().emit(record)
        except OSError:
            return
        finally:
            self._close_stream()

    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.backupCount <= 0:
            self._open_after_rollover()
            return

        temp_path = self._rollover_temp_path()
        with contextlib.suppress(OSError):
            os.remove(temp_path)

        try:
            self.rotate(self.baseFilename, temp_path)
        except OSError:
            # Another process may still hold debug.log open, or it may have
            # disappeared between the rollover check and rename. Leave backups
            # intact and retry on a later record.
            self._open_after_rollover()
            return

        try:
            self._replace_backups_from_temp(temp_path)
        except OSError:
            self._restore_rollover_temp(temp_path)
        finally:
            self._open_after_rollover()

    def handleError(self, record: logging.LogRecord) -> None:
        """Keep best-effort debug logging failures off the TUI stderr."""
        return

    def _replace_backups_from_temp(self, temp_path: str) -> None:
        if self.backupCount > 1:
            for i in range(self.backupCount - 1, 0, -1):
                source = self.rotation_filename(f"{self.baseFilename}.{i}")
                dest = self.rotation_filename(f"{self.baseFilename}.{i + 1}")
                if os.path.exists(source):
                    os.replace(source, dest)
        os.replace(temp_path, self.rotation_filename(f"{self.baseFilename}.1"))

    def _restore_rollover_temp(self, temp_path: str) -> None:
        if os.path.exists(self.baseFilename):
            return
        with contextlib.suppress(OSError):
            os.replace(temp_path, self.baseFilename)

    def _rollover_temp_path(self) -> str:
        base_path = Path(self.baseFilename)
        return str(base_path.with_name(f".{base_path.name}.{os.getpid()}.{id(self)}.{_DEBUG_LOG_TMP_SUFFIX}"))

    def _open_after_rollover(self) -> None:
        if not self.delay:
            self.stream = self._open()

    def _close_stream(self) -> None:
        stream = self.stream
        if stream is None:
            return
        self.stream = None
        with contextlib.suppress(OSError):
            stream.close()


def _enqueue_debug_log_sentinel(queue: Queue[logging.LogRecord | None], sentinel: None) -> None:
    """Put the shutdown sentinel back even if concurrent log producers refill the queue."""
    while True:
        try:
            queue.put_nowait(sentinel)
            return
        except Full:
            with contextlib.suppress(Empty):
                queue.get_nowait()
                queue.task_done()


def _debug_log_lock_path(log_path: Path) -> Path:
    """Return the sidecar lock path for shared debug log rotation."""
    return log_path.with_name(f".{log_path.name}.lock")


def configure_debug_logging(log_path: Path) -> DebugLogRuntime:
    """Install bounded debug-file logging without doing file I/O on caller threads."""
    with (
        contextlib.suppress(OSError),
        FileLock(_debug_log_lock_path(log_path), timeout=_DEBUG_LOG_LOCK_TIMEOUT_SECONDS),
    ):
        _normalize_debug_log_files(log_path)

    queue: Queue[logging.LogRecord | None] = Queue(maxsize=_DEBUG_LOG_QUEUE_MAX_RECORDS)
    queue_handler = _BoundedDebugLogHandler(queue)

    file_handler = _DebugRotatingFileHandler(
        log_path,
        maxBytes=_DEBUG_LOG_MAX_BYTES,
        backupCount=_DEBUG_LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setFormatter(logging.Formatter(_DEBUG_LOG_FORMAT))

    root_logger = logging.getLogger()
    for existing in root_logger.handlers[:]:
        root_logger.removeHandler(existing)
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(queue_handler)

    listener = _BoundedDebugLogListener(queue, file_handler, respect_handler_level=True)
    listener.start()
    return DebugLogRuntime(queue_handler=queue_handler, listener=listener, file_handler=file_handler)


def stop_debug_logging(runtime: DebugLogRuntime) -> None:
    """Flush queued debug logs and close the backing file handler."""
    root_logger = logging.getLogger()
    if runtime.queue_handler in root_logger.handlers:
        root_logger.removeHandler(runtime.queue_handler)
    runtime.listener.stop()
    runtime.queue_handler.close()
    runtime.file_handler.close()


def _normalize_debug_log_files(log_path: Path) -> None:
    """Bring existing debug logs under the configured cap before the handler opens them."""
    backup_path = log_path.with_name(f"{log_path.name}.1")
    if _is_oversized(log_path):
        tail = _read_file_tail(log_path, _DEBUG_LOG_MAX_BYTES)
        if tail is not None and _replace_file_contents(backup_path, tail):
            _truncate_file(log_path)
        return
    if _is_oversized(backup_path):
        tail = _read_file_tail(backup_path, _DEBUG_LOG_MAX_BYTES)
        if tail is not None:
            _replace_file_contents(backup_path, tail)


def _is_oversized(path: Path) -> bool:
    try:
        return path.stat().st_size > _DEBUG_LOG_MAX_BYTES
    except OSError:
        return False


def _read_file_tail(path: Path, max_bytes: int) -> bytes | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read(max_bytes)
    except OSError:
        return None

    if size > max_bytes:
        newline = data.find(b"\n")
        if newline != -1 and newline + 1 < len(data):
            data = data[newline + 1 :]
    # Scrub a partial UTF-8 sequence at the seek boundary before rewriting the tail.
    return data.decode("utf-8", errors="ignore").encode("utf-8")


def _replace_file_contents(path: Path, data: bytes) -> bool:
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{_DEBUG_LOG_TMP_SUFFIX}")
    try:
        with temp_path.open("wb") as f:
            f.write(data)
        temp_path.replace(path)
        return True
    except OSError:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        return False


def _truncate_file(path: Path) -> None:
    try:
        with path.open("wb"):
            pass
    except OSError:
        return
