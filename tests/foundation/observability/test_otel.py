# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for ``chrys.foundation.observability.setup`` setup + Settings flags + session sink.

Exporter/env parsing, resource, and providers are built in
``chrys.foundation.observability.setup``, and the chrys
telemetry gate (``chrys.kernel.TELEMETRY_GATE``) flips on success.
``FunctionTool.invoke`` and MCP client spans are Chrys-owned and gate on that
same switch. Each test restores the production revocation path via
``_reset_telemetry_state``.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.observability import setup as otel_module
from chrys.foundation.observability.exporters import (
    SessionContextSpanProcessor,
    SessionJsonlLogExporter,
    SessionJsonlSpanExporter,
)
from chrys.foundation.observability.setup import setup_otel
from chrys.foundation.observability.sink import (
    OtelSessionSink,
    get_otel_sink,
    set_otel_sink,
)
from chrys.kernel import TELEMETRY_GATE, configure_telemetry

_OTLP_ENV_VARS = (
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_METRICS_HEADERS",
    "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"otel_enabled": False, "otel_sensitive_data": False}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _clear_otlp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _OTLP_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _reset_telemetry_state() -> Iterator[None]:
    """Revoke whatever the test attached (gate, sink, root handler, chrys
    logger level — the production revocation path) after each test."""
    yield
    otel_module._revoke_previous_setup()


# --- Settings -----------------------------------------------------------------


def test_settings_otel_fields_disabled_by_default(monkeypatch) -> None:
    for var in ("CHRYS_OTEL", "CHRYS_OTEL_SENSITIVE_DATA"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.otel_enabled is False
    assert s.otel_sensitive_data is False


def test_settings_otel_truthy_variants(monkeypatch) -> None:
    for val in ("1", "true", "TRUE", "yes", "Yes", "on", "ON"):
        monkeypatch.setenv("CHRYS_OTEL", val)
        monkeypatch.setenv("CHRYS_OTEL_SENSITIVE_DATA", val)
        s = Settings.from_env()
        assert s.otel_enabled is True
        assert s.otel_sensitive_data is True


def test_settings_otel_falsey_variants(monkeypatch) -> None:
    for val in ("", "0", "false", "no", "off", "random"):
        monkeypatch.setenv("CHRYS_OTEL", val)
        assert Settings.from_env().otel_enabled is False


# --- setup_otel ---------------------------------------------------------------


def test_setup_otel_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(otel_module, "set_otel_sink", lambda s: None)
    with patch.object(otel_module, "_configure_providers") as mock_configure:
        assert setup_otel(_settings(otel_enabled=False)) is None
    mock_configure.assert_not_called()
    assert TELEMETRY_GATE.enabled is False


def test_setup_otel_uses_file_fallback_when_no_endpoint(monkeypatch) -> None:
    """With no OTLP endpoint set, the JSONL file exporters feed the providers."""
    _clear_otlp_env(monkeypatch)

    captured: dict[str, OtelSessionSink] = {}
    monkeypatch.setattr(otel_module, "set_otel_sink", lambda s: captured.update(sink=s))

    with patch.object(otel_module, "_configure_providers") as mock_configure:
        sink = setup_otel(_settings(otel_enabled=True))

    assert isinstance(sink, OtelSessionSink)
    assert captured["sink"] is sink
    assert sink.write_files is True
    (exporters,) = mock_configure.call_args.args
    assert len(exporters) == 2, "span + log file exporters"
    assert isinstance(exporters[0], SessionJsonlSpanExporter)
    assert isinstance(exporters[1], SessionJsonlLogExporter)
    assert TELEMETRY_GATE.enabled is True
    assert TELEMETRY_GATE.sensitive_data is False


def test_setup_otel_opens_gate_with_sensitive_data(monkeypatch) -> None:
    """Success opens the chrys gate with the sensitive flag."""
    _clear_otlp_env(monkeypatch)
    monkeypatch.setattr(otel_module, "set_otel_sink", lambda s: None)

    with patch.object(otel_module, "_configure_providers"):
        sink = setup_otel(_settings(otel_enabled=True, otel_sensitive_data=True))

    assert isinstance(sink, OtelSessionSink)
    assert TELEMETRY_GATE.enabled is True
    assert TELEMETRY_GATE.sensitive_data is True


def test_owned_mcp_spans_gate_on_the_chrys_switch(monkeypatch) -> None:
    """Regression: owned MCP spans must stop recording after Chrys telemetry revokes."""
    from chrys.kernel.instrumentation import create_mcp_client_span

    _clear_otlp_env(monkeypatch)
    monkeypatch.setattr(otel_module, "set_otel_sink", lambda s: None)

    with patch.object(otel_module, "_configure_providers"):
        setup_otel(_settings(otel_enabled=True))
    assert TELEMETRY_GATE.enabled is True

    otel_module._revoke_previous_setup()
    assert TELEMETRY_GATE.enabled is False
    with create_mcp_client_span("tools/list") as span:
        assert span.is_recording() is False, "no owned MCP span leak once chrys telemetry is off"


def test_setup_otel_skips_file_fallback_when_otlp_endpoint_set(monkeypatch) -> None:
    """With an OTLP endpoint set, env-derived exporters are used — no JSONL files."""
    _clear_otlp_env(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setattr(otel_module, "set_otel_sink", lambda s: None)

    with patch.object(otel_module, "_configure_providers") as mock_configure:
        sink = setup_otel(_settings(otel_enabled=True))

    assert isinstance(sink, OtelSessionSink)
    assert sink.write_files is False
    (exporters,) = mock_configure.call_args.args
    assert len(exporters) == 3, "grpc log + span + metric exporters from the env"
    assert not any(isinstance(e, (SessionJsonlSpanExporter, SessionJsonlLogExporter)) for e in exporters)
    assert TELEMETRY_GATE.enabled is True


def test_setup_otel_endpoint_setting_stands_in_for_the_base_env_var(monkeypatch) -> None:
    """``otel_endpoint`` picks OTLP over the JSONL fallback when the env is silent."""
    _clear_otlp_env(monkeypatch)
    monkeypatch.setattr(otel_module, "set_otel_sink", lambda s: None)
    captured: dict[str, Any] = {}
    monkeypatch.setattr(otel_module, "_create_otlp_exporters", lambda **kw: captured.update(kw) or [])

    with patch.object(otel_module, "_configure_providers"):
        sink = setup_otel(_settings(otel_enabled=True, otel_endpoint=" http://collector:4317 "))

    assert isinstance(sink, OtelSessionSink)
    assert sink.write_files is False
    assert captured["traces_endpoint"] == "http://collector:4317"


def test_setup_otel_standard_env_var_wins_over_the_endpoint_setting(monkeypatch) -> None:
    _clear_otlp_env(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://from-env:4317")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(otel_module, "_create_otlp_exporters", lambda **kw: captured.update(kw) or [])

    assert otel_module._get_exporters_from_env("http://from-settings:4317") == []
    assert captured["logs_endpoint"] == "http://from-env:4317"


def test_setup_otel_disables_everything_when_exporter_import_fails(monkeypatch) -> None:
    import builtins

    monkeypatch.setattr(otel_module, "set_otel_sink", lambda s: None)

    real_import = builtins.__import__

    def fail_exporter_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "chrys.foundation.observability.exporters":
            raise ImportError("opentelemetry-sdk missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_exporter_import)

    assert setup_otel(_settings(otel_enabled=True)) is None
    assert TELEMETRY_GATE.enabled is False, "gate stays closed on failure"


def test_setup_otel_disables_everything_when_logging_handler_import_fails(monkeypatch) -> None:
    _clear_otlp_env(monkeypatch)
    monkeypatch.setattr(otel_module, "set_otel_sink", lambda s: None)
    monkeypatch.setattr(otel_module, "import_module", MagicMock(side_effect=ImportError("handler missing")))

    with patch.object(otel_module, "_configure_providers") as mock_configure:
        assert setup_otel(_settings(otel_enabled=True)) is None

    mock_configure.assert_not_called()
    assert TELEMETRY_GATE.enabled is False


def test_setup_otel_handles_configure_failure(monkeypatch) -> None:
    _clear_otlp_env(monkeypatch)
    monkeypatch.setattr(otel_module, "set_otel_sink", lambda s: None)
    with patch.object(otel_module, "_configure_providers", side_effect=RuntimeError("sdk broke")):
        assert setup_otel(_settings(otel_enabled=True)) is None
    assert TELEMETRY_GATE.enabled is False


def test_setup_otel_disabled_closes_previously_opened_gate(monkeypatch) -> None:
    """The non-success paths must reset authoritatively, not merely stay
    closed: a prior successful call in the same process leaves the gate open
    and a sink installed, and a disabled re-run has to revoke both."""
    configure_telemetry(enabled=True, sensitive_data=True)
    stale = OtelSessionSink(write_files=False)
    stale.activate("sess-stale", session_dir=None)
    set_otel_sink(stale)
    try:
        with patch.object(otel_module, "_configure_providers") as mock_configure:
            assert setup_otel(_settings(otel_enabled=False)) is None
        mock_configure.assert_not_called()
        assert TELEMETRY_GATE.enabled is False, "previously opened gate closed"
        assert TELEMETRY_GATE.sensitive_data is False, "sensitive flag revoked"
        assert get_otel_sink() is None, "stale sink cleared on the disabled path"
        assert stale.session_id is None, "stale sink deactivated, not just unreferenced"
    finally:
        set_otel_sink(None)


def test_setup_otel_failure_closes_previously_opened_gate(monkeypatch) -> None:
    _clear_otlp_env(monkeypatch)
    configure_telemetry(enabled=True, sensitive_data=True)
    stale = OtelSessionSink(write_files=False)
    stale.activate("sess-stale", session_dir=None)
    set_otel_sink(stale)
    try:
        with patch.object(otel_module, "_configure_providers", side_effect=RuntimeError("sdk broke")):
            assert setup_otel(_settings(otel_enabled=True)) is None
        assert TELEMETRY_GATE.enabled is False, "previously opened gate closed"
        assert TELEMETRY_GATE.sensitive_data is False, "sensitive flag revoked"
        assert get_otel_sink() is None, "stale sink cleared on the failure path"
        assert stale.session_id is None, "stale sink deactivated, not just unreferenced"
    finally:
        set_otel_sink(None)


def test_setup_otel_failure_leaves_chrys_logging_level_untouched(monkeypatch) -> None:
    """The chrys logging tree is raised to INFO only on success — a failed
    setup must not leave that global side effect behind."""
    _clear_otlp_env(monkeypatch)
    chrys_logger = logging.getLogger("chrys")
    level_before = chrys_logger.level
    chrys_logger.setLevel(logging.NOTSET)
    try:
        with patch.object(otel_module, "_configure_providers", side_effect=RuntimeError("sdk broke")):
            assert setup_otel(_settings(otel_enabled=True)) is None
        assert chrys_logger.level == logging.NOTSET
    finally:
        chrys_logger.setLevel(level_before)


def test_setup_otel_preserves_more_verbose_chrys_logger_level(monkeypatch) -> None:
    """The chrys-tree raise to INFO is a minimum-verbosity guarantee: a
    pre-existing DEBUG pin (e.g. the TUI log viewer escalation) already admits
    INFO records and must not be clamped down — the per-tool DEBUG log lines
    from ``chrys.kernel.loop`` would otherwise vanish in OTel sessions."""
    _clear_otlp_env(monkeypatch)
    set_otel_sink(None)
    chrys_logger = logging.getLogger("chrys")
    level_before = chrys_logger.level
    chrys_logger.setLevel(logging.DEBUG)
    try:
        with patch.object(otel_module, "_configure_providers"):
            assert setup_otel(_settings(otel_enabled=True)) is not None
        assert chrys_logger.level == logging.DEBUG, "DEBUG pin preserved"

        assert setup_otel(_settings(otel_enabled=False)) is None
        assert chrys_logger.level == logging.DEBUG, "untouched pin not clobbered by the restore"
    finally:
        chrys_logger.setLevel(level_before)
        set_otel_sink(None)


def test_setup_otel_sets_singleton(monkeypatch) -> None:
    _clear_otlp_env(monkeypatch)
    set_otel_sink(None)
    try:
        with patch.object(otel_module, "_configure_providers"):
            sink = setup_otel(_settings(otel_enabled=True))
        assert get_otel_sink() is sink
    finally:
        set_otel_sink(None)


def test_setup_otel_real_providers_install_root_logging_handler(monkeypatch) -> None:
    """Full run without provider patching: the OTel LoggingHandler must land on
    the ROOT logger (the retired framework attached it to its own logger,
    which silently dropped all chrys INFO events — the P2 fix)."""
    from opentelemetry.instrumentation.logging.handler import LoggingHandler

    _clear_otlp_env(monkeypatch)
    monkeypatch.setattr(otel_module, "set_otel_sink", lambda s: None)
    root = logging.getLogger()
    handlers_before = list(root.handlers)

    sink = setup_otel(_settings(otel_enabled=True))
    new_handlers = [h for h in root.handlers if h not in handlers_before]
    try:
        assert isinstance(sink, OtelSessionSink)
        assert TELEMETRY_GATE.enabled is True
        assert any(isinstance(h, LoggingHandler) for h in new_handlers)
    finally:
        for handler in new_handlers:
            root.removeHandler(handler)


def test_setup_otel_partial_failure_revokes_freshly_installed_handler(monkeypatch) -> None:
    """``_configure_providers`` can install the root handler and a *later*
    step inside it still fail — the failure path must revoke the state
    installed by this very call, not just by earlier calls.  The failure is
    injected at ``set_logger_provider``, which runs right after the root
    ``addHandler``, to exercise rollback after handler installation."""
    _clear_otlp_env(monkeypatch)
    root = logging.getLogger()
    handlers_before = list(root.handlers)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("post-handler step broke")

    monkeypatch.setattr("opentelemetry._logs.set_logger_provider", boom)
    try:
        assert setup_otel(_settings(otel_enabled=True)) is None
        assert otel_module._INSTALLED_ROOT_HANDLER is None
        new_handlers = [h for h in root.handlers if h not in handlers_before]
        assert new_handlers == [], "freshly installed root handler revoked on failure"
        assert TELEMETRY_GATE.enabled is False
        assert get_otel_sink() is None
    finally:
        for extra in [h for h in root.handlers if h not in handlers_before]:
            root.removeHandler(extra)


def test_setup_otel_disabled_rerun_revokes_prior_successful_setup(tmp_path: Path, monkeypatch) -> None:
    """Full repro of the review finding: after a *successful* file-fallback
    setup with an activated session, a disabled re-run must tear down the
    chrys-attached logging state — root LoggingHandler detached, chrys logger
    level restored, stale sink deactivated — so chrys records stop reaching
    the old session's ``otel/logs.jsonl`` through stale OTel state."""
    _clear_otlp_env(monkeypatch)
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    chrys_logger = logging.getLogger("chrys")
    level_before = chrys_logger.level
    chrys_logger.setLevel(logging.NOTSET)

    sink: OtelSessionSink | None = None
    try:
        sink = setup_otel(_settings(otel_enabled=True))
        assert isinstance(sink, OtelSessionSink)
        sink.activate("sess-review", session_dir=tmp_path)
        assert chrys_logger.level == logging.INFO

        # Sanity: a chrys INFO record flows root handler -> provider ->
        # JSONL exporter -> session file while enabled.  LoggingHandler.flush
        # is fire-and-forget (unjoined thread), so flush synchronously via
        # the provider.
        logging.getLogger("chrys.review_probe").info("during-enabled")
        structured_mapping = {"method": "notifications/message", "data": {"progress": 1}}
        structured_sequence = ["mcp", {"progress": 2}]
        logging.getLogger("chrys.review_probe").info(structured_mapping)
        logging.getLogger("chrys.review_probe").info(structured_sequence)
        provider = otel_module._INSTALLED_LOGGER_PROVIDER
        assert provider is not None
        provider.force_flush()
        log_file = tmp_path / "otel" / "logs.jsonl"
        assert log_file.exists()
        lines_during = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert any("during-enabled" in line for line in lines_during)
        payloads = [json.loads(line) for line in lines_during]
        probe = next(payload for payload in payloads if payload["body"] == "during-enabled")
        assert {"code.file.path", "code.function.name", "code.line.number"} <= probe["attributes"].keys()
        assert structured_mapping in [payload["body"] for payload in payloads]
        assert structured_sequence in [payload["body"] for payload in payloads]
        handler = otel_module._INSTALLED_ROOT_HANDLER
        assert handler is not None and handler in root.handlers

        assert setup_otel(_settings(otel_enabled=False)) is None

        assert otel_module._INSTALLED_ROOT_HANDLER is None
        assert handler not in root.handlers, "root LoggingHandler revoked"
        assert chrys_logger.level == logging.NOTSET, "chrys logger level restored"
        assert sink.session_id is None, "old sink deactivated"
        assert get_otel_sink() is None
        assert TELEMETRY_GATE.enabled is False

        # WARNING passes any logger-level config — only the detached handler
        # (and the deactivated sink behind it) can stop the export now.
        logging.getLogger("chrys.review_probe").warning("after-disabled")
        provider.force_flush()
        lines_after = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert lines_after == lines_during, "no records exported after the disabled re-run"
    finally:
        chrys_logger.setLevel(level_before)
        for extra in [h for h in root.handlers if h not in handlers_before]:
            root.removeHandler(extra)
        if sink is not None:
            sink.deactivate()
        set_otel_sink(None)


# --- env parsing / resource helpers --------------------------------------------


def test_parse_headers_matrix() -> None:
    assert otel_module._parse_headers("") == {}
    assert otel_module._parse_headers("a=1,b=2") == {"a": "1", "b": "2"}
    assert otel_module._parse_headers(" a = 1 , b = 2 ") == {"a": "1", "b": "2"}
    assert otel_module._parse_headers("a=1,bad,b=2") == {"a": "1", "b": "2"}
    assert otel_module._parse_headers("a=1=extra") == {"a": "1=extra"}, "splits on the first ="


def test_get_exporters_from_env_grpc_base_endpoint(monkeypatch) -> None:
    _clear_otlp_env(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "auth=tok")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS", "extra=1")

    captured: dict[str, Any] = {}

    def fake_create(**kwargs: Any) -> list[Any]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(otel_module, "_create_otlp_exporters", fake_create)
    assert otel_module._get_exporters_from_env() == []
    assert captured["protocol"] == "grpc"
    # gRPC uses the base endpoint as-is for every signal.
    assert captured["traces_endpoint"] == "http://collector:4317"
    assert captured["metrics_endpoint"] == "http://collector:4317"
    assert captured["logs_endpoint"] == "http://collector:4317"
    # Signal headers merge ON TOP of the base headers.
    assert captured["traces_headers"] == {"auth": "tok", "extra": "1"}
    assert captured["logs_headers"] == {"auth": "tok"}


def test_get_exporters_from_env_http_appends_v1_paths(monkeypatch) -> None:
    _clear_otlp_env(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "http://elsewhere:4318/custom/logs")

    captured: dict[str, Any] = {}
    monkeypatch.setattr(otel_module, "_create_otlp_exporters", lambda **kw: captured.update(kw) or [])
    otel_module._get_exporters_from_env()
    # _get_exporters_from_env appends /v1/{signal} to the HTTP base URL;
    # signal-specific endpoints are used verbatim.
    assert captured["traces_endpoint"] == "http://collector:4318/v1/traces"
    assert captured["metrics_endpoint"] == "http://collector:4318/v1/metrics"
    assert captured["logs_endpoint"] == "http://elsewhere:4318/custom/logs"


def test_get_exporters_from_env_empty_when_no_vars(monkeypatch) -> None:
    _clear_otlp_env(monkeypatch)
    assert otel_module._get_exporters_from_env() == []


def test_create_resource_chrys_defaults_and_env_overrides(monkeypatch) -> None:
    for var in ("OTEL_SERVICE_NAME", "OTEL_SERVICE_VERSION", "OTEL_RESOURCE_ATTRIBUTES"):
        monkeypatch.delenv(var, raising=False)
    resource = otel_module._create_resource()
    assert resource.attributes["service.name"] == "chrys"
    assert resource.attributes["service.version"] == importlib.metadata.version("chrys")

    monkeypatch.setenv("OTEL_SERVICE_NAME", "my-svc")
    monkeypatch.setenv("OTEL_SERVICE_VERSION", "9.9.9")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment=prod")
    resource = otel_module._create_resource()
    assert resource.attributes["service.name"] == "my-svc"
    assert resource.attributes["service.version"] == "9.9.9"
    assert resource.attributes["deployment.environment"] == "prod"


# --- OtelSessionSink ----------------------------------------------------------


def test_sink_activate_sets_session_id() -> None:
    sink = OtelSessionSink(write_files=False)
    assert sink.session_id is None
    sink.activate("sess-123", session_dir=None)
    assert sink.session_id == "sess-123"
    sink.deactivate()
    assert sink.session_id is None


def test_sink_activate_defers_otel_dir_until_first_write(tmp_path: Path) -> None:
    sink = OtelSessionSink(write_files=True)
    sink.activate("sess-abc", session_dir=tmp_path)
    assert not (tmp_path / "otel").exists()

    sink.write_logs(['{"post_activate": true}\n'])

    assert (tmp_path / "otel" / "logs.jsonl").exists()


def test_sink_write_does_not_recreate_deleted_session_root(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-deleted"
    session_dir.mkdir()
    sink = OtelSessionSink(write_files=True)
    sink.activate("sess-deleted", session_dir=session_dir)
    session_dir.rmdir()

    sink.write_logs(['{"late_flush": true}\n'])

    assert not session_dir.exists()


def test_sink_drops_missing_old_session_buffer_on_new_activation(tmp_path: Path) -> None:
    old_session_dir = tmp_path / "old"
    new_session_dir = tmp_path / "new"
    old_session_dir.mkdir()
    new_session_dir.mkdir()
    sink = OtelSessionSink(write_files=True)
    sink.activate("old", session_dir=old_session_dir)
    old_session_dir.rmdir()
    sink.write_logs(['{"old_late_flush": true}\n'])

    sink.activate("new", session_dir=new_session_dir)
    sink.write_logs(['{"new_log": true}\n'])

    logs = (new_session_dir / "otel" / "logs.jsonl").read_text(encoding="utf-8")
    assert "new_log" in logs
    assert "old_late_flush" not in logs


def test_sink_buffers_until_session_root_materializes(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-lazy"
    sink = OtelSessionSink(write_files=True)
    sink.write_spans(['{"pre_activate": true}\n'])
    sink.activate("sess-lazy", session_dir=session_dir)
    sink.write_spans(['{"before_materialized": true}\n'])

    assert not session_dir.exists()

    session_dir.mkdir()
    sink.write_spans(['{"after_materialized": true}\n'])

    traces = (session_dir / "otel" / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in traces] == [
        {"pre_activate": True},
        {"before_materialized": True},
        {"after_materialized": True},
    ]


def test_sink_flush_pending_after_session_root_materializes(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess-save"
    sink = OtelSessionSink(write_files=True)
    sink.write_logs(['{"pre_activate": true}\n'])
    sink.activate("sess-save", session_dir=session_dir)
    sink.write_logs(['{"before_save": true}\n'])

    assert not session_dir.exists()

    session_dir.mkdir()
    sink.flush_pending("sess-save")

    logs = (session_dir / "otel" / "logs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in logs] == [
        {"pre_activate": True},
        {"before_save": True},
    ]


def test_sink_activate_without_session_dir_drops_file_writes(tmp_path: Path) -> None:
    sink = OtelSessionSink(write_files=True)
    sink.write_logs(['{"pre_activate": true}\n'])
    sink.activate("sess-no-store", session_dir=None)
    sink.write_logs(['{"dropped": true}\n'])

    session_dir = tmp_path / "sess-with-store"
    session_dir.mkdir()
    sink.activate("sess-with-store", session_dir=session_dir)
    sink.write_logs(['{"kept": true}\n'])

    logs = (session_dir / "otel" / "logs.jsonl").read_text(encoding="utf-8")
    assert "kept" in logs
    assert "pre_activate" not in logs
    assert "dropped" not in logs


def test_sink_write_spans_buffers_then_flushes(tmp_path: Path) -> None:
    """Span lines emitted before activate get buffered, then flushed on activate."""
    sink = OtelSessionSink(write_files=True)
    sink.write_spans(['{"pre_activate": true}\n'])

    sink.activate("sess-1", session_dir=tmp_path)
    traces = tmp_path / "otel" / "traces.jsonl"
    assert traces.exists()
    assert json.loads(traces.read_text().strip())["pre_activate"] is True

    # Subsequent writes go directly to the file
    sink.write_spans(['{"post_activate": true}\n'])
    lines = traces.read_text().strip().split("\n")
    assert len(lines) == 2


def test_sink_activate_handles_file_setup_failure(tmp_path: Path) -> None:
    sink = OtelSessionSink(write_files=True)
    sink.write_spans(['{"pre_activate": true}\n'])
    (tmp_path / "otel").write_text("not a directory", encoding="utf-8")

    sink.activate("sess-bad", session_dir=tmp_path)
    sink.write_spans(['{"post_activate": true}\n'])
    sink.write_logs(['{"post_activate": true}\n'])

    assert sink.session_id == "sess-bad"
    assert not (tmp_path / "otel" / "traces.jsonl").exists()


def test_sink_session_state_uses_lock() -> None:
    class TrackingLock:
        def __init__(self) -> None:
            self.enters = 0

        def __enter__(self):
            self.enters += 1
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    sink = OtelSessionSink(write_files=False)
    lock = TrackingLock()
    sink._lock = lock  # type: ignore[assignment]

    sink.activate("sess-locked", session_dir=None)
    assert sink.session_id == "sess-locked"
    sink.deactivate()
    assert sink.session_id is None
    assert lock.enters == 4


def test_sink_write_spans_noop_when_write_files_false() -> None:
    """write_files=False silently drops writes (OTLP path handles export)."""
    sink = OtelSessionSink(write_files=False)
    # Must not raise even though there's no file path
    sink.write_spans(['{"x": 1}\n'])
    sink.write_logs(['{"x": 1}\n'])


# --- SessionContextSpanProcessor ---------------------------------------------


def test_span_processor_injects_session_id_on_start() -> None:
    sink = OtelSessionSink(write_files=False)
    sink.activate("sess-xyz", session_dir=None)
    proc = SessionContextSpanProcessor(sink)

    mock_span = MagicMock()
    proc.on_start(mock_span, None)

    mock_span.set_attribute.assert_any_call("chrys.session_id", "sess-xyz")
    mock_span.set_attribute.assert_any_call("gen_ai.conversation.id", "sess-xyz")


def test_span_processor_noop_when_no_session() -> None:
    sink = OtelSessionSink(write_files=False)
    proc = SessionContextSpanProcessor(sink)
    mock_span = MagicMock()
    proc.on_start(mock_span, None)
    mock_span.set_attribute.assert_not_called()


# --- SessionJsonlSpanExporter -------------------------------------------------


def test_span_exporter_writes_json_lines(tmp_path: Path) -> None:
    sink = OtelSessionSink(write_files=True)
    sink.activate("sess-x", session_dir=tmp_path)
    exporter = SessionJsonlSpanExporter(sink)

    fake_span = MagicMock()
    fake_span.name = "chat gpt-4"
    fake_span.kind = None
    fake_span.get_span_context.return_value = MagicMock(trace_id=0x1234, span_id=0xABCD)
    fake_span.parent = None
    fake_span.status = MagicMock(status_code=MagicMock(name="OK"), description=None)
    fake_span.attributes = {"gen_ai.conversation.id": "sess-x", "gen_ai.request.model": "gpt-4"}
    fake_span.events = []
    fake_span.resource = MagicMock(attributes={})
    fake_span.start_time = 1
    fake_span.end_time = 2

    exporter.export([fake_span])

    traces = tmp_path / "otel" / "traces.jsonl"
    assert traces.exists()
    payload = json.loads(traces.read_text().strip())
    assert payload["name"] == "chat gpt-4"
    assert payload["attributes"]["gen_ai.conversation.id"] == "sess-x"
