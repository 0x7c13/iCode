# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session-state holder for the OpenTelemetry pipeline.

Kept deliberately free of ``opentelemetry`` imports so the engine can
import :func:`get_otel_sink` without requiring the SDK to be installed.
The SDK-dependent pieces (``SpanExporter`` / ``LogRecordExporter`` /
``SpanProcessor``) live in :mod:`chrys.foundation.observability.exporters` and
are only loaded from :func:`chrys.foundation.observability.setup.setup_otel` when the
user has enabled OTel and the SDK is available.

The sink holds:

* the active chrys session_id (read by the span processor to stamp
  ``chrys.session_id`` and ``gen_ai.conversation.id`` on every span),
* the file paths JSONL exporters should write to,
* thread-safe bounded buffers that hold spans/logs emitted before a
  session is active, or before the lazily-created session directory has
  been materialized by the first real session writer.

:meth:`activate` is called by :class:`chrys.orchestration.engine.engine.AgentEngine`
once a session id is active; :meth:`deactivate` is called on
session delete.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_BUFFERED_LINES = 1000


class OtelSessionSink:
    """Per-process state bridge between OTel exporters and the chrys session.

    ``write_files`` picks the exporter strategy:

    * ``True`` — Chrys installs JSONL span + log exporters that write
      to ``{session_dir}/otel/`` once :meth:`activate` is called.  Used
      when no ``OTEL_EXPORTER_OTLP_ENDPOINT`` is configured.
    * ``False`` — Span/log data ships via OTLP exporters that ``setup_otel``
      builds from standard OTel env vars. The sink still tracks the
      session_id so the span processor can stamp attributes.
    """

    def __init__(self, *, write_files: bool) -> None:
        self._session_id: str | None = None
        self._write_files = write_files
        self._span_path: Path | None = None
        self._log_path: Path | None = None
        self._span_buffer: list[str] = []
        self._log_buffer: list[str] = []
        self._file_write_failed = False
        self._lock = threading.Lock()

    # --- Session state ---------------------------------------------------

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    @property
    def write_files(self) -> bool:
        return self._write_files

    def activate(self, session_id: str, session_dir: Path | None) -> None:
        """Bind a session.  Flushes any buffered span/log lines to disk."""
        with self._lock:
            if self._session_id is not None and self._session_id != session_id:
                self._span_buffer.clear()
                self._log_buffer.clear()
            self._session_id = session_id
            self._file_write_failed = False
            self._span_path = None
            self._log_path = None
            if not self._write_files:
                self._span_buffer.clear()
                self._log_buffer.clear()
                return
            if session_dir is None:
                self._file_write_failed = True
                self._span_buffer.clear()
                self._log_buffer.clear()
                return
        try:
            otel_dir = session_dir / "otel"
            with self._lock:
                self._span_path = otel_dir / "traces.jsonl"
                self._log_path = otel_dir / "logs.jsonl"
                if self._span_buffer:
                    self._write_or_buffer_unlocked(self._span_path, self._span_buffer, [])
                if self._log_buffer:
                    self._write_or_buffer_unlocked(self._log_path, self._log_buffer, [])
        except OSError:
            logger.warning("Failed to activate OTel session file sink in %s", session_dir, exc_info=True)
            with self._lock:
                self._file_write_failed = True
                self._span_path = None
                self._log_path = None
                self._span_buffer.clear()
                self._log_buffer.clear()

    def deactivate(self) -> None:
        with self._lock:
            self._session_id = None
            self._span_path = None
            self._log_path = None
            self._file_write_failed = True
            self._span_buffer.clear()
            self._log_buffer.clear()

    def flush_pending(self, session_id: str) -> None:
        """Retry buffered file writes after the session root is materialized."""
        with self._lock:
            if not self._write_files:
                return
            if self._file_write_failed:
                return
            if self._session_id != session_id:
                return
            try:
                if self._span_path is not None and self._span_buffer:
                    self._write_or_buffer_unlocked(self._span_path, self._span_buffer, [])
                if self._log_path is not None and self._log_buffer:
                    self._write_or_buffer_unlocked(self._log_path, self._log_buffer, [])
            except OSError:
                logger.warning("Failed to flush buffered OTel file lines for session %s", session_id, exc_info=True)
                self._file_write_failed = True
                self._span_path = None
                self._log_path = None
                self._span_buffer.clear()
                self._log_buffer.clear()

    # --- Exporter write paths --------------------------------------------

    def write_spans(self, lines: list[str]) -> None:
        """Append pre-serialized JSONL lines to the span file (buffer if unset)."""
        if not lines:
            return
        with self._lock:
            if not self._write_files:
                return
            if self._file_write_failed:
                return
            if self._span_path is None:
                self._buffer_lines_unlocked(self._span_buffer, lines)
            else:
                try:
                    self._write_or_buffer_unlocked(self._span_path, self._span_buffer, lines)
                except OSError:
                    logger.warning("Failed to write OTel span file %s", self._span_path, exc_info=True)
                    self._file_write_failed = True
                    self._span_path = None
                    self._log_path = None
                    self._span_buffer.clear()
                    self._log_buffer.clear()

    def write_logs(self, lines: list[str]) -> None:
        """Append pre-serialized JSONL lines to the log file (buffer if unset)."""
        if not lines:
            return
        with self._lock:
            if not self._write_files:
                return
            if self._file_write_failed:
                return
            if self._log_path is None:
                self._buffer_lines_unlocked(self._log_buffer, lines)
            else:
                try:
                    self._write_or_buffer_unlocked(self._log_path, self._log_buffer, lines)
                except OSError:
                    logger.warning("Failed to write OTel log file %s", self._log_path, exc_info=True)
                    self._file_write_failed = True
                    self._span_path = None
                    self._log_path = None
                    self._span_buffer.clear()
                    self._log_buffer.clear()

    def _write_or_buffer_unlocked(self, path: Path, buffer: list[str], lines: list[str]) -> None:
        pending = [*buffer, *lines] if buffer else list(lines)
        if not pending:
            return
        if self._append_lines_unlocked(path, pending):
            buffer.clear()
            return
        buffer.clear()
        self._buffer_lines_unlocked(buffer, pending)

    @staticmethod
    def _buffer_lines_unlocked(buffer: list[str], lines: list[str]) -> None:
        if not lines:
            return
        buffer.extend(lines)
        overflow = len(buffer) - _MAX_BUFFERED_LINES
        if overflow > 0:
            del buffer[:overflow]

    @staticmethod
    def _append_lines_unlocked(path: Path, lines: list[str]) -> bool:
        try:
            path.parent.mkdir(exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.writelines(lines)
            return True
        except FileNotFoundError:
            return False


# --- Module singleton ---------------------------------------------------------

_SINK: OtelSessionSink | None = None


def set_otel_sink(sink: OtelSessionSink | None) -> None:
    """Install the process-wide sink.  Called by :func:`setup_otel`."""
    global _SINK
    _SINK = sink


def get_otel_sink() -> OtelSessionSink | None:
    """Return the installed sink, or ``None`` if OTel is disabled."""
    return _SINK
