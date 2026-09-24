# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the shared startup bootstrap helper."""

from __future__ import annotations

import dataclasses
import functools
import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from chrys.foundation.config.env_layers import process_env_snapshot
from chrys.foundation.config.runtime_pointer import get_model_pointer
from chrys.foundation.config.settings import Settings
from chrys.foundation.config.spec import SettingOrigin, Source
from chrys.foundation.events.types import Warning as WarningEvent
from chrys.foundation.i18n import MessageRef
from chrys.orchestration import startup


def _isolate_startup(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Stub out side-effecting collaborators so each test is hermetic.

    The working directory moves to ``tmp_path`` because bootstrap reads
    ``<cwd>/.env`` — without the move, a ``.env`` on the developer's machine
    would leak into every test here.
    """
    import chrys.foundation.platform as platform_mod

    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    fake_platform = dataclasses.replace(platform_mod.get_platform(), config_dir=config_dir, data_dir=config_dir)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(startup, "set_process_title", lambda title="chrys": None)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)


def _claim_env_name(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Register ``name`` for restore so injected values never outlive the test."""
    monkeypatch.setenv(name, "claimed")
    monkeypatch.delenv(name)


def test_set_process_title_delegates_to_setproctitle(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "setproctitle", SimpleNamespace(setproctitle=calls.append))

    startup.set_process_title("custom-chrys")

    assert calls == ["custom-chrys"]


def test_set_process_title_ignores_missing_support(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "setproctitle", None)

    startup.set_process_title("custom-chrys")


def test_set_process_title_ignores_platform_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_setproctitle(_title: str) -> None:
        msg = "unsupported platform"
        raise RuntimeError(msg)

    monkeypatch.setitem(sys.modules, "setproctitle", SimpleNamespace(setproctitle=fail_setproctitle))

    startup.set_process_title("custom-chrys")


def test_bootstrap_runtime_sets_process_title_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _isolate_startup(monkeypatch, tmp_path)
    monkeypatch.setattr("chrys.foundation.patches.apply_all", lambda: None)
    monkeypatch.setattr("chrys.foundation.observability.setup.setup_otel", lambda _settings: None)
    calls: list[str] = []
    monkeypatch.setattr(startup, "set_process_title", lambda title="chrys": calls.append(title))

    startup.bootstrap_runtime(dotenv_override=False)

    assert calls == ["chrys"]


def test_bootstrap_runtime_returns_invalid_no_proxy_warning(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _isolate_startup(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("NO_PROXY=[::1]\n", encoding="utf-8")
    monkeypatch.setattr("chrys.foundation.patches.apply_all", lambda: None)
    monkeypatch.setattr("chrys.foundation.observability.setup.setup_otel", lambda _settings: None)

    bootstrap = startup.bootstrap_runtime(dotenv_override=True)

    assert "NO_PROXY" not in os.environ
    assert len(bootstrap.warnings) == 1
    warning = bootstrap.warnings[0]
    assert isinstance(warning, WarningEvent)
    assert warning.code == "invalid_no_proxy"
    assert warning.message == (
        "NO_PROXY is invalid for httpx and was ignored ('[::1]'). Remove or fix this environment value."
    )
    assert isinstance(warning.display_message, MessageRef)
    assert warning.display_message.definition is startup._INVALID_NO_PROXY_ENV
    assert dict(warning.display_message.args) == {"var": "NO_PROXY", "value": "'[::1]'"}


def test_bootstrap_runtime_returns_no_warnings_when_env_clean(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _isolate_startup(monkeypatch, tmp_path)
    monkeypatch.setattr("chrys.foundation.patches.apply_all", lambda: None)
    monkeypatch.setattr("chrys.foundation.observability.setup.setup_otel", lambda _settings: None)

    bootstrap = startup.bootstrap_runtime(dotenv_override=False)

    assert bootstrap.warnings == []


def test_bootstrap_runtime_skips_patches_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _isolate_startup(monkeypatch, tmp_path)
    monkeypatch.setattr("chrys.foundation.observability.setup.setup_otel", lambda _settings: None)
    apply_calls: list[bool] = []
    monkeypatch.setattr("chrys.foundation.patches.apply_all", lambda: apply_calls.append(True))

    startup.bootstrap_runtime(dotenv_override=False, apply_patches=False)

    assert apply_calls == []


def test_bootstrap_runtime_runs_patches_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _isolate_startup(monkeypatch, tmp_path)
    monkeypatch.setattr("chrys.foundation.observability.setup.setup_otel", lambda _settings: None)
    apply_calls: list[bool] = []
    monkeypatch.setattr("chrys.foundation.patches.apply_all", lambda: apply_calls.append(True))

    startup.bootstrap_runtime(dotenv_override=False)

    assert apply_calls == [True]


def test_bootstrap_runtime_skips_telemetry_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _isolate_startup(monkeypatch, tmp_path)
    monkeypatch.setattr("chrys.foundation.patches.apply_all", lambda: None)
    otel_calls: list[Any] = []
    monkeypatch.setattr("chrys.foundation.observability.setup.setup_otel", otel_calls.append)

    startup.bootstrap_runtime(dotenv_override=False, setup_telemetry=False)

    assert otel_calls == []


def test_bootstrap_runtime_invokes_telemetry_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _isolate_startup(monkeypatch, tmp_path)
    monkeypatch.setattr("chrys.foundation.patches.apply_all", lambda: None)
    otel_calls: list[Any] = []
    monkeypatch.setattr("chrys.foundation.observability.setup.setup_otel", otel_calls.append)

    bootstrap = startup.bootstrap_runtime(dotenv_override=False)

    assert otel_calls == [bootstrap.settings]


def test_bootstrap_runtime_dotenv_override_true_replaces_ambient_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _quiet_bootstrap(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("STARTUP_OVERRIDE_PROBE=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("STARTUP_OVERRIDE_PROBE", "from-shell")

    startup.bootstrap_runtime(dotenv_override=True)

    assert os.environ["STARTUP_OVERRIDE_PROBE"] == "from-dotenv"


def test_bootstrap_runtime_dotenv_override_false_keeps_ambient_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _quiet_bootstrap(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("STARTUP_OVERRIDE_PROBE=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("STARTUP_OVERRIDE_PROBE", "from-shell")

    startup.bootstrap_runtime(dotenv_override=False)

    assert os.environ["STARTUP_OVERRIDE_PROBE"] == "from-shell"


def test_bootstrap_runtime_reads_dotenv_from_the_working_directory(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The dotenv root is the workspace cwd, not a ``find_dotenv`` walk."""
    _quiet_bootstrap(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("STARTUP_CWD_PROBE=yes\n", encoding="utf-8")
    _claim_env_name(monkeypatch, "STARTUP_CWD_PROBE")

    startup.bootstrap_runtime(dotenv_override=True)

    assert os.environ["STARTUP_CWD_PROBE"] == "yes"


def test_bootstrap_runtime_reads_explicit_cwd_then_config_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _quiet_bootstrap(monkeypatch, tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".env").write_text("STARTUP_SHARED_PROBE=workspace\nSTARTUP_WORKSPACE_PROBE=yes\n", encoding="utf-8")
    (tmp_path / "config" / ".env").write_text("STARTUP_SHARED_PROBE=config\n", encoding="utf-8")
    for name in ("STARTUP_SHARED_PROBE", "STARTUP_WORKSPACE_PROBE"):
        _claim_env_name(monkeypatch, name)

    startup.bootstrap_runtime(dotenv_override=True, dotenv_cwd=workspace)

    # Same file order as the load_dotenv calls this replaced: the config-dir
    # .env is read second, so with override it wins the overlap.
    assert os.environ["STARTUP_WORKSPACE_PROBE"] == "yes"
    assert os.environ["STARTUP_SHARED_PROBE"] == "config"


def test_bootstrap_runtime_freezes_the_environment_before_injection(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _quiet_bootstrap(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("STARTUP_SNAPSHOT_PROBE=injected\n", encoding="utf-8")
    _claim_env_name(monkeypatch, "STARTUP_SNAPSHOT_PROBE")

    startup.bootstrap_runtime(dotenv_override=True)

    snapshot = process_env_snapshot()
    assert snapshot is not None
    assert "STARTUP_SNAPSHOT_PROBE" not in snapshot.values
    assert os.environ["STARTUP_SNAPSHOT_PROBE"] == "injected"


def test_bootstrap_runtime_never_injects_chrys_keys_from_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Dotenv files cannot mint settings, pointer, or IPC state in the process env."""
    _quiet_bootstrap(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text(
        "CHRYS_MODEL_PROFILE=from-dotenv\nCHRYS_ACP_SUBAGENT_DEPTH=3\nSTARTUP_PLAIN_PROBE=kept\n",
        encoding="utf-8",
    )
    for name in ("CHRYS_MODEL_PROFILE", "CHRYS_ACP_SUBAGENT_DEPTH", "STARTUP_PLAIN_PROBE"):
        _claim_env_name(monkeypatch, name)

    bootstrap = startup.bootstrap_runtime(dotenv_override=True)

    assert os.environ["STARTUP_PLAIN_PROBE"] == "kept"
    assert "CHRYS_MODEL_PROFILE" not in os.environ
    assert "CHRYS_ACP_SUBAGENT_DEPTH" not in os.environ
    assert bootstrap.settings.model_profile == Settings().model_profile


def test_bootstrap_runtime_real_environment_outranks_dotenv_for_chrys_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The shell's export wins over any dotenv line, override flag or not."""
    _quiet_bootstrap(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("CHRYS_MODEL_PROFILE=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "preexisting")

    bootstrap = startup.bootstrap_runtime(dotenv_override=True)

    assert os.environ["CHRYS_MODEL_PROFILE"] == "preexisting"
    assert bootstrap.settings.model_profile == "preexisting"


def test_bootstrap_seeds_the_model_pointer_with_the_layered_origin(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _quiet_bootstrap(monkeypatch, tmp_path)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "shell-model")

    startup.bootstrap_runtime(dotenv_override=True)

    assert get_model_pointer() == ("shell-model", SettingOrigin(layer=Source.ENV))


def test_bootstrap_does_not_seed_an_absent_model_pointer(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _quiet_bootstrap(monkeypatch, tmp_path)
    monkeypatch.delenv("CHRYS_MODEL_PROFILE", raising=False)

    startup.bootstrap_runtime(dotenv_override=True)

    assert "CHRYS_MODEL_PROFILE" not in os.environ
    assert get_model_pointer() == ("", None)


def test_bootstrap_migrates_the_config_dotenv_before_loading(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Legacy chrys lines move into ``settings.yaml`` and still take effect."""
    _quiet_bootstrap(monkeypatch, tmp_path)
    monkeypatch.delenv("CHRYS_THEME", raising=False)
    config_env = tmp_path / "config" / ".env"
    config_env.write_text("CHRYS_THEME=solar\nSTARTUP_PLAIN_PROBE=kept\n", encoding="utf-8")
    _claim_env_name(monkeypatch, "STARTUP_PLAIN_PROBE")

    bootstrap = startup.bootstrap_runtime(dotenv_override=True)

    assert bootstrap.settings.theme == "solar"
    assert bootstrap.loaded.source_for("ui.theme").layer is Source.USER
    remaining = config_env.read_text(encoding="utf-8")
    assert "CHRYS_THEME" not in remaining
    assert "STARTUP_PLAIN_PROBE=kept" in remaining
    assert (tmp_path / "config" / "settings.yaml").is_file()


def test_bootstrap_renames_a_retired_theme_in_the_document(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _quiet_bootstrap(monkeypatch, tmp_path)
    monkeypatch.delenv("CHRYS_THEME", raising=False)
    document = tmp_path / "config" / "settings.yaml"
    document.write_text("ui:\n  theme: chrys-dark\n", encoding="utf-8")

    bootstrap = startup.bootstrap_runtime(dotenv_override=True)

    assert bootstrap.settings.theme == "chrys"
    assert bootstrap.loaded.source_for("ui.theme").layer is Source.USER
    assert yaml.safe_load(document.read_text(encoding="utf-8"))["ui"]["theme"] == "chrys"
    assert bootstrap.warnings == []


def test_bootstrap_draws_the_renamed_theme_even_when_the_document_cannot_be_rewritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The loader follows the rename by itself, so a rewrite that fails is not worth a warning."""
    _quiet_bootstrap(monkeypatch, tmp_path)
    monkeypatch.delenv("CHRYS_THEME", raising=False)
    document = tmp_path / "config" / "settings.yaml"
    document.write_text("ui:\n  theme: chrys-dark\n", encoding="utf-8")

    def explode() -> None:
        raise TimeoutError("lock busy")

    monkeypatch.setattr("chrys.foundation.config.migrations.migrate_renamed_theme", explode)

    bootstrap = startup.bootstrap_runtime(dotenv_override=True)

    assert bootstrap.settings.theme == "chrys"
    assert bootstrap.warnings == []
    assert "chrys-dark" in document.read_text(encoding="utf-8")


def test_bootstrap_reports_a_migration_failure_instead_of_dying(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _quiet_bootstrap(monkeypatch, tmp_path)

    def explode() -> None:
        raise TimeoutError("lock busy")

    monkeypatch.setattr("chrys.foundation.config.migrations.migrate_dotenv_v0", explode)

    bootstrap = startup.bootstrap_runtime(dotenv_override=True)

    (warning,) = [w for w in bootstrap.warnings if w.code == "settings_migration_failed"]
    assert "lock busy" in warning.message


def test_bootstrap_migrates_the_notifications_file_before_loading(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Legacy notification toggles move into ``settings.yaml`` and still take effect."""
    _quiet_bootstrap(monkeypatch, tmp_path)
    legacy = tmp_path / "config" / "notifications.yaml"
    legacy.write_text("enabled: false\nevents:\n  turn_complete: false\n", encoding="utf-8")

    bootstrap = startup.bootstrap_runtime(dotenv_override=True)

    assert bootstrap.settings.notifications_enabled is False
    assert bootstrap.settings.notifications_event_turn_complete is False
    assert bootstrap.loaded.source_for("notifications.enabled").layer is Source.USER
    assert not legacy.exists()
    assert (tmp_path / "config" / "notifications.yaml.migrated").is_file()
    doc = yaml.safe_load((tmp_path / "config" / "settings.yaml").read_text(encoding="utf-8"))
    assert doc["notifications"]["enabled"] is False


def test_bootstrap_runs_later_migrations_after_an_earlier_failure(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """One stuck component must not hold back the other's import."""
    _quiet_bootstrap(monkeypatch, tmp_path)

    def explode() -> None:
        raise TimeoutError("lock busy")

    monkeypatch.setattr("chrys.foundation.config.migrations.migrate_dotenv_v0", explode)
    legacy = tmp_path / "config" / "notifications.yaml"
    legacy.write_text("enabled: false\n", encoding="utf-8")

    bootstrap = startup.bootstrap_runtime(dotenv_override=True)

    (warning,) = [w for w in bootstrap.warnings if w.code == "settings_migration_failed"]
    assert "lock busy" in warning.message
    assert bootstrap.settings.notifications_enabled is False
    assert not legacy.exists()


def test_bootstrap_runtime_configures_stdio_when_requested(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _isolate_startup(monkeypatch, tmp_path)
    monkeypatch.setattr("chrys.foundation.patches.apply_all", lambda: None)
    monkeypatch.setattr("chrys.foundation.observability.setup.setup_otel", lambda _settings: None)
    stdio_calls: list[bool] = []
    monkeypatch.setattr(startup, "configure_utf8_stdio", lambda: stdio_calls.append(True))

    startup.bootstrap_runtime(dotenv_override=False, configure_stdio=True)

    assert stdio_calls == [True]


def test_bootstrap_runtime_does_not_configure_stdio_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _isolate_startup(monkeypatch, tmp_path)
    monkeypatch.setattr("chrys.foundation.patches.apply_all", lambda: None)
    monkeypatch.setattr("chrys.foundation.observability.setup.setup_otel", lambda _settings: None)
    stdio_calls: list[bool] = []
    monkeypatch.setattr(startup, "configure_utf8_stdio", lambda: stdio_calls.append(True))

    startup.bootstrap_runtime(dotenv_override=False)

    assert stdio_calls == []


def test_configure_utf8_stdio_sets_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("PYTHONUTF8", "PYTHONIOENCODING"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)

    startup.configure_utf8_stdio()

    assert os.environ["PYTHONUTF8"] == "1"
    assert os.environ["PYTHONIOENCODING"] == "utf-8"


def test_configure_utf8_stdio_respects_explicit_user_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("PYTHONIOENCODING", "latin-1")

    startup.configure_utf8_stdio()

    assert os.environ["PYTHONUTF8"] == "0"
    assert os.environ["PYTHONIOENCODING"] == "latin-1"


def _quiet_bootstrap(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Bootstrap with everything side-effecting stubbed out."""
    _isolate_startup(monkeypatch, tmp_path)
    monkeypatch.setattr("chrys.foundation.patches.apply_all", lambda: None)
    monkeypatch.setattr("chrys.foundation.observability.setup.setup_otel", lambda _settings: None)


def test_bootstrap_warns_while_raw_http_capture_is_on(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A valid setting, in force, that writes API keys to disk until turned off.

    Never a coercion warning — nothing was rejected — so it has to be composed
    from what the process actually installed, or the one dangerous switch that
    persists across restarts is the one nothing ever mentions again.
    """
    _quiet_bootstrap(monkeypatch, tmp_path)
    monkeypatch.setenv("CHRYS_DEBUG_LLM_RAW_HTTP_LOG", "1")

    bootstrap = startup.bootstrap_runtime(dotenv_override=False)

    assert bootstrap.loaded.warnings == ()
    codes = [warning.code for warning in bootstrap.warnings]
    assert codes == ["raw_http_capture_on"]
    assert "CHRYS_DEBUG_LLM_RAW_HTTP_LOG" in bootstrap.warnings[0].message


def test_bootstrap_stays_quiet_while_raw_http_capture_is_off(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _quiet_bootstrap(monkeypatch, tmp_path)
    monkeypatch.delenv("CHRYS_DEBUG_LLM_RAW_HTTP_LOG", raising=False)

    assert startup.bootstrap_runtime(dotenv_override=False).warnings == []


def test_bootstrap_settles_an_unusable_session_root_before_anything_reads_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The settings a root hands on must describe where sessions really go."""
    from chrys.foundation.config.settings import resolve_session_root_dir

    _quiet_bootstrap(monkeypatch, tmp_path)
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("", encoding="utf-8")
    monkeypatch.setenv("CHRYS_SESSION_ROOT_DIR", str(blocked))

    bootstrap = startup.bootstrap_runtime(dotenv_override=False)

    assert bootstrap.settings.session_root_dir == ""
    assert [warning.key for warning in bootstrap.loaded.warnings] == ["storage.session_root_dir"]
    assert resolve_session_root_dir(tmp_path / "config") == tmp_path / "config"


def test_a_failed_notifications_migration_does_not_claim_the_old_file_still_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # The legacy ``.env`` keeps loading as a layer when its migration fails, so
    # "your existing configuration still applies" is true there. Nothing reads
    # the legacy notifications file any more, so saying it there tells the user
    # the opposite of what happened to the toggles they turned off.
    _quiet_bootstrap(monkeypatch, tmp_path)

    def explode() -> None:
        raise TimeoutError("lock busy")

    monkeypatch.setattr("chrys.foundation.config.migrations.migrate_notifications_v0", explode)
    (tmp_path / "config" / "notifications.yaml").write_text("enabled: false\n", encoding="utf-8")

    bootstrap = startup.bootstrap_runtime(dotenv_override=True)

    (warning,) = [w for w in bootstrap.warnings if w.code == "notifications_migration_failed"]
    assert "lock busy" in warning.message
    assert "still applies" not in warning.message
    # The defaults really are what this start runs on — the sentence has to match.
    assert bootstrap.settings.notifications_enabled is True


def test_bootstrap_survives_a_settings_document_held_by_another_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # A second Chrys must still start while the first is stalled inside the
    # document's lock: the read is on every entry point's startup path.
    from chrys.foundation.config.yaml_store import lock_path_for, read_yaml_doc, update_yaml_doc
    from chrys.foundation.util.lock import FileLock

    _quiet_bootstrap(monkeypatch, tmp_path)
    # Real lock, real contention, real ``TimeoutError`` — only the ten-second
    # wait is shortened, at every point on the boot path that reaches for this
    # document, so the test costs milliseconds instead of stalling a worker for
    # three full production timeouts.
    for target, function in (
        ("chrys.foundation.config.settings_store.read_yaml_doc", read_yaml_doc),
        ("chrys.foundation.config.migrations.update_yaml_doc", update_yaml_doc),
    ):
        monkeypatch.setattr(target, functools.partial(function, lock_timeout=0.05))
    monkeypatch.delenv("CHRYS_THEME", raising=False)
    settings_path = tmp_path / "config" / "settings.yaml"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text("schema_version: 1\nui:\n  theme: solar\n", encoding="utf-8")

    held = FileLock(lock_path_for(settings_path), timeout=0)
    held.acquire()
    try:
        bootstrap = startup.bootstrap_runtime(dotenv_override=True)
    finally:
        held.release()

    # Started, and honest about it: the document was not read, so its value is
    # not in force and provenance does not claim it is.
    assert bootstrap.settings.theme != "solar"
    assert bootstrap.loaded.source_for("ui.theme").layer is Source.DEFAULT
