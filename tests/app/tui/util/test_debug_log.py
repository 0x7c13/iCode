# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the bounded debug-log queue, handler, and rollover."""

from __future__ import annotations

import logging
from logging.handlers import QueueHandler
from pathlib import Path
from queue import Queue

import pytest

from chrys.app.tui.util import debug_log


def test_debug_logging_is_bounded_and_queue_backed(tmp_path: Path) -> None:
    """File logging should be capped and moved off the caller thread."""

    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level
    for handler in previous_handlers:
        root_logger.removeHandler(handler)

    runtime = None
    log_path = tmp_path / "debug.log"
    try:
        runtime = debug_log.configure_debug_logging(log_path)

        assert root_logger.handlers == [runtime.queue_handler]
        assert isinstance(runtime.queue_handler, QueueHandler)
        assert runtime.queue_handler.queue.maxsize == debug_log._DEBUG_LOG_QUEUE_MAX_RECORDS
        assert runtime.file_handler.maxBytes == debug_log._DEBUG_LOG_MAX_BYTES
        assert runtime.file_handler.backupCount == 1
        assert runtime.file_handler.stream is None

        logging.getLogger("chrys.test").warning("bounded log entry")
    finally:
        if runtime is not None:
            debug_log.stop_debug_logging(runtime)
        root_logger.handlers.clear()
        root_logger.setLevel(previous_level)
        for handler in previous_handlers:
            root_logger.addHandler(handler)

    assert "bounded log entry" in log_path.read_text(encoding="utf-8")


def test_debug_log_handler_closes_shared_file_after_emit(tmp_path: Path) -> None:
    """The shared debug log should not stay open between queued writes."""

    log_path = tmp_path / "debug.log"
    handler = debug_log._DebugRotatingFileHandler(
        log_path,
        maxBytes=debug_log._DEBUG_LOG_MAX_BYTES,
        backupCount=1,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        record = logging.LogRecord("chrys.test", logging.WARNING, __file__, 1, "one line", None, None)
        handler.emit(record)
    finally:
        handler.close()

    assert handler.stream is None
    assert log_path.read_text(encoding="utf-8") == "one line\n"


def test_debug_log_handler_still_rotates_when_unlocked(tmp_path: Path) -> None:
    """The shared debug log handler should retain normal rotation behavior."""

    log_path = tmp_path / "debug.log"
    backup_path = tmp_path / "debug.log.1"
    log_path.write_text("old line\n", encoding="utf-8")
    handler = debug_log._DebugRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        record = logging.LogRecord("chrys.test", logging.WARNING, __file__, 1, "new line", None, None)
        handler.emit(record)
    finally:
        handler.close()

    assert handler.stream is None
    assert backup_path.read_text(encoding="utf-8") == "old line\n"
    assert log_path.read_text(encoding="utf-8") == "new line\n"


def test_debug_log_rollover_permission_error_keeps_logging(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A denied rollover rename should keep the existing backup and stay quiet."""

    log_path = tmp_path / "debug.log"
    backup_path = tmp_path / "debug.log.1"
    log_path.write_text("existing\n", encoding="utf-8")
    backup_path.write_text("previous backup\n", encoding="utf-8")
    handler = debug_log._DebugRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    def _blocked_rotate(_source: str, _dest: str) -> None:
        raise PermissionError("locked by another process")

    handler.rotate = _blocked_rotate
    try:
        record = logging.LogRecord("chrys.test", logging.WARNING, __file__, 1, "after lock", None, None)
        handler.emit(record)
    finally:
        handler.close()

    assert "after lock" in log_path.read_text(encoding="utf-8")
    assert backup_path.read_text(encoding="utf-8") == "previous backup\n"
    assert capsys.readouterr().err == ""


def test_debug_log_rollover_restore_temp_when_backup_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If replacing debug.log.1 fails, restore the active log from rollover temp."""

    log_path = tmp_path / "debug.log"
    backup_path = tmp_path / "debug.log.1"
    log_path.write_text("old active\n", encoding="utf-8")
    backup_path.write_text("locked backup\n", encoding="utf-8")
    original_replace = debug_log.os.replace

    def _replace_unless_backup(source: str, dest: str) -> None:
        if Path(dest) == backup_path:
            raise PermissionError("backup locked by another process")
        original_replace(source, dest)

    monkeypatch.setattr(debug_log.os, "replace", _replace_unless_backup)
    handler = debug_log._DebugRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        record = logging.LogRecord("chrys.test", logging.WARNING, __file__, 1, "after restore", None, None)
        handler.emit(record)
    finally:
        handler.close()

    assert log_path.read_text(encoding="utf-8") == "old active\nafter restore\n"
    assert backup_path.read_text(encoding="utf-8") == "locked backup\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_debug_log_handler_silences_internal_file_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Debug file logging failures should not leak logging diagnostics to the TUI."""

    log_path = tmp_path / "debug.log"
    handler = debug_log._DebugRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    def _blocked_open() -> object:
        raise PermissionError("log file is unavailable")

    handler._open = _blocked_open
    try:
        record = logging.LogRecord("chrys.test", logging.WARNING, __file__, 1, "lost", None, None)
        handler.emit(record)
    finally:
        handler.close()

    assert capsys.readouterr().err == ""


def test_debug_log_queue_drops_oldest_when_full() -> None:
    """A stalled writer should not let debug logging grow memory without bound."""

    queue: Queue[logging.LogRecord | None] = Queue(maxsize=1)
    handler = debug_log._BoundedDebugLogHandler(queue)
    old_record = logging.LogRecord("chrys.test", logging.INFO, __file__, 1, "old", None, None)
    new_record = logging.LogRecord("chrys.test", logging.INFO, __file__, 2, "new", None, None)

    handler.enqueue(old_record)
    handler.enqueue(new_record)

    assert queue.get_nowait() is new_record


def test_debug_log_listener_can_stop_with_full_queue() -> None:
    """Shutdown should not fail just because the bounded queue is full."""

    queue: Queue[logging.LogRecord | None] = Queue(maxsize=1)
    old_record = logging.LogRecord("chrys.test", logging.INFO, __file__, 1, "old", None, None)
    queue.put_nowait(old_record)

    listener = debug_log._BoundedDebugLogListener(queue)
    listener.enqueue_sentinel()

    assert queue.get_nowait() is listener._sentinel


def test_debug_log_listener_skips_records_after_stop_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown should stop spending time on debug writes after the stop budget."""

    emitted: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            emitted.append(record.getMessage())

    listener = debug_log._BoundedDebugLogListener(Queue(), _Handler())
    record = logging.LogRecord("chrys.test", logging.INFO, __file__, 1, "late", None, None)

    listener._stop_deadline = 1.0
    monkeypatch.setattr(debug_log.time, "monotonic", lambda: 2.0)
    listener.handle(record)
    assert emitted == []

    listener._stop_deadline = 3.0
    listener.handle(record)
    assert emitted == ["late"]


def test_debug_log_queue_preserves_stop_sentinel_when_full() -> None:
    """A late log enqueue must not drop the listener shutdown sentinel."""

    queue: Queue[logging.LogRecord | None] = Queue(maxsize=1)
    handler = debug_log._BoundedDebugLogHandler(queue)
    record = logging.LogRecord("chrys.test", logging.INFO, __file__, 1, "late", None, None)
    queue.put_nowait(None)

    handler.enqueue(record)

    assert queue.get_nowait() is None


def test_debug_log_queue_preserves_stop_sentinel_when_requeue_races() -> None:
    """The shutdown sentinel should survive if another producer refills the freed slot."""

    filler_record = logging.LogRecord("chrys.test", logging.INFO, __file__, 1, "filler", None, None)

    class _RacySentinelQueue(Queue[logging.LogRecord | None]):
        def __init__(self) -> None:
            super().__init__(maxsize=1)
            self._armed = False
            self._filled_once = False

        def arm_race(self) -> None:
            self._armed = True

        def put_nowait(self, item: logging.LogRecord | None) -> None:
            if self._armed and item is None and not self._filled_once and self.empty():
                self._filled_once = True
                super().put_nowait(filler_record)
            super().put_nowait(item)

    queue = _RacySentinelQueue()
    handler = debug_log._BoundedDebugLogHandler(queue)
    record = logging.LogRecord("chrys.test", logging.INFO, __file__, 2, "late", None, None)
    queue.put_nowait(None)
    queue.arm_race()

    handler.enqueue(record)

    assert queue.get_nowait() is None


@pytest.mark.parametrize(
    ("active_text", "backup_text", "expected_active_text", "expected_backup_tail"),
    [
        pytest.param(
            ("old active line\n" * 20) + "active tail\n",
            "stale backup\n",
            "",
            "active tail",
            id="caps-existing-active-log",
        ),
        pytest.param(
            "active\n",
            ("old backup line\n" * 20) + "backup tail\n",
            "active\n",
            "backup tail",
            id="caps-existing-backup-log",
        ),
    ],
)
def test_debug_logging_caps_existing_logs(
    active_text: str,
    backup_text: str,
    expected_active_text: str,
    expected_backup_tail: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An oversized pre-existing debug.log or debug.log.1 must end up capped in the backup."""
    monkeypatch.setattr(debug_log, "_DEBUG_LOG_MAX_BYTES", 80)

    log_path = tmp_path / "debug.log"
    backup_path = tmp_path / "debug.log.1"
    log_path.write_text(active_text, encoding="utf-8")
    backup_path.write_text(backup_text, encoding="utf-8")

    debug_log._normalize_debug_log_files(log_path)

    assert log_path.read_text(encoding="utf-8") == expected_active_text
    assert backup_path.stat().st_size <= 80
    assert expected_backup_tail in backup_path.read_text(encoding="utf-8")
