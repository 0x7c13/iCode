# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SDK-dependent OpenTelemetry exporters and span processor for Chrys.

This module imports from ``opentelemetry.sdk.*`` at module level and is
ONLY loaded by :func:`chrys.foundation.observability.setup.setup_otel` after confirming
the SDK is available.  Engine code must not import this module directly
— use :func:`chrys.foundation.observability.sink.get_otel_sink` instead, which
has no SDK dependency.

Provides:

* :class:`SessionJsonlSpanExporter` — serialises spans to JSONL and
  hands them to :meth:`OtelSessionSink.write_spans`.
* :class:`SessionJsonlLogExporter` — same for log records.
* :class:`SessionContextSpanProcessor` — stamps ``chrys.session_id``
  and ``gen_ai.conversation.id`` on every span at ``on_start``.  The
  kernel's ``_get_span_attributes`` omits ``None`` values, so the later
  attribute setter preserves injected conversation ids when the provider
  supplies none (the Anthropic / OpenAI Chat Completions path).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.context import Context
    from opentelemetry.sdk._logs import ReadableLogRecord
    from opentelemetry.sdk.trace import ReadableSpan, Span

    from chrys.foundation.observability.sink import OtelSessionSink

logger = logging.getLogger(__name__)


# --- Exporters ----------------------------------------------------------------


class SessionJsonlSpanExporter(SpanExporter):
    """Writes exported spans as one JSONL line each via the session sink."""

    def __init__(self, sink: OtelSessionSink) -> None:
        self._sink = sink

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            lines = [json.dumps(_span_to_dict(s), default=str, ensure_ascii=False) + "\n" for s in spans]
            self._sink.write_spans(lines)
        except Exception:
            logger.exception("SessionJsonlSpanExporter.export failed")
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """No-op — files are opened/closed per write by the sink."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class SessionJsonlLogExporter(LogRecordExporter):
    """Writes exported log records as one JSONL line each via the sink."""

    def __init__(self, sink: OtelSessionSink) -> None:
        self._sink = sink

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        try:
            lines = [json.dumps(_log_to_dict(d), default=str, ensure_ascii=False) + "\n" for d in batch]
            self._sink.write_logs(lines)
        except Exception:
            logger.exception("SessionJsonlLogExporter.export failed")
            return LogRecordExportResult.FAILURE
        return LogRecordExportResult.SUCCESS

    def shutdown(self) -> None:
        """No-op."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# --- Attribute injection ------------------------------------------------------


class SessionContextSpanProcessor(SpanProcessor):
    """Stamp the active session id and a fallback conversation id on each span.

    Injection must run in ``on_start``, before kernel instrumentation calls
    ``span.set_attributes(...)``. ``_get_span_attributes`` omits ``None`` values,
    so the later setter preserves our fallback when the provider supplies no
    conversation id. An explicit provider conversation id overwrites it.
    """

    def __init__(self, sink: OtelSessionSink) -> None:
        self._sink = sink

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        sid = self._sink.session_id
        if sid:
            span.set_attribute("chrys.session_id", sid)
            span.set_attribute("gen_ai.conversation.id", sid)

    def on_end(self, span: ReadableSpan) -> None:
        """No-op — attribute injection happens at start."""


# --- Serialization helpers ----------------------------------------------------


def _span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    """Convert a ReadableSpan to a JSON-safe dict with GenAI fields promoted."""
    ctx = span.get_span_context()
    parent = span.parent
    status = span.status
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "name": span.name,
        "kind": span.kind.name if span.kind else None,
        "trace_id": f"{ctx.trace_id:032x}" if ctx else None,
        "span_id": f"{ctx.span_id:016x}" if ctx else None,
        "parent_span_id": f"{parent.span_id:016x}" if parent else None,
        "start_time_ns": span.start_time,
        "end_time_ns": span.end_time,
        "duration_ns": (span.end_time - span.start_time) if (span.start_time and span.end_time) else None,
        "status": {
            "code": status.status_code.name if status else None,
            "description": status.description if status else None,
        },
        "attributes": dict(span.attributes) if span.attributes else {},
        "events": [
            {
                "name": e.name,
                "timestamp_ns": e.timestamp,
                "attributes": dict(e.attributes) if e.attributes else {},
            }
            for e in (span.events or [])
        ],
        "resource": dict(span.resource.attributes) if span.resource else {},
    }


def _log_to_dict(data: ReadableLogRecord) -> dict[str, Any]:
    """Convert a ReadableLogRecord to a JSON-safe dict.

    ``ReadableLogRecord`` wraps a ``LogRecord`` and an ``InstrumentationScope``;
    we flatten the bits that matter for debugging.
    """
    rec = data.log_record
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "observed_timestamp_ns": rec.observed_timestamp,
        "timestamp_ns": rec.timestamp,
        "severity_text": rec.severity_text,
        "severity_number": rec.severity_number.value if rec.severity_number else None,
        "body": rec.body,
        "attributes": dict(rec.attributes) if rec.attributes else {},
        "trace_id": f"{rec.trace_id:032x}" if rec.trace_id else None,
        "span_id": f"{rec.span_id:016x}" if rec.span_id else None,
        "scope": {
            "name": data.instrumentation_scope.name if data.instrumentation_scope else None,
            "version": data.instrumentation_scope.version if data.instrumentation_scope else None,
        },
    }
