# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the ``env_bridge`` module — translating ``ModelProfile`` to env updates."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from chrys.foundation.config.runtime_pointer import get_model_pointer
from chrys.foundation.config.spec import Source
from chrys.service.profiles.models.env_bridge import (
    activate_model_profile,
    get_active_profile_id,
    get_global_default_profile_id,
    is_valid_no_proxy,
    profile_to_activation_env_updates,
    sanitize_no_proxy_env,
    set_global_default_profile_id,
)
from chrys.service.profiles.models.schema import ModelProfile


@pytest.fixture(autouse=True)
def config_env_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: SimpleNamespace(config_dir=tmp_path))


def test_openai_profile_with_key_and_base_url() -> None:
    """OpenAI profile activation only emits the active profile pointer."""
    p = ModelProfile(
        id="oa-id",
        name="OAI",
        provider="openai",
        api_key="sk-openai",
        base_url="https://api.openai.example.com",
    )
    updates = profile_to_activation_env_updates(p)

    assert updates == {"CHRYS_MODEL_PROFILE": "oa-id"}


def test_anthropic_profile_with_key_and_base_url() -> None:
    """Anthropic profile activation only emits the active profile pointer."""
    p = ModelProfile(
        id="an-id",
        name="Anth",
        provider="anthropic",
        api_key="sk-anth",
        base_url="https://api.anthropic.example.com",
    )
    updates = profile_to_activation_env_updates(p)

    assert updates == {"CHRYS_MODEL_PROFILE": "an-id"}


def test_deepseek_profile_with_key_and_base_url() -> None:
    """DeepSeek profile activation only emits the active profile pointer."""
    p = ModelProfile(
        id="ds-id",
        name="DS",
        provider="deepseek-openai",
        api_key="sk-ds",
        base_url="https://api.deepseek.example.com",
    )
    updates = profile_to_activation_env_updates(p)

    assert updates == {"CHRYS_MODEL_PROFILE": "ds-id"}


def test_unknown_provider_only_emits_pointer() -> None:
    """Unknown provider activation still only emits the profile pointer."""
    p = ModelProfile(id="x", name="X", provider="unknown")
    updates = profile_to_activation_env_updates(p)
    assert updates == {"CHRYS_MODEL_PROFILE": "x"}


def test_credentials_and_base_urls_are_not_env_updates() -> None:
    """Profile secrets/URLs stay in YAML and are passed directly to the client factory."""
    p = ModelProfile(id="x", name="X", provider="openai", api_key="k", base_url="https://api.example.com")
    updates = profile_to_activation_env_updates(p)
    assert set(updates) == {"CHRYS_MODEL_PROFILE"}


def test_chrys_model_profile_always_present() -> None:
    """CHRYS_MODEL_PROFILE is always in the output."""
    p = ModelProfile(id="abc", name="Abc")
    updates = profile_to_activation_env_updates(p)
    assert updates["CHRYS_MODEL_PROFILE"] == "abc"


def test_no_proxy_env_keys_are_not_profile_updates() -> None:
    """Proxy bypass is now per-client config, not active-profile env output."""
    p = ModelProfile(id="x", name="X", bypass_proxy=True)
    updates = profile_to_activation_env_updates(p)
    assert "NO_PROXY" not in updates
    assert "no_proxy" not in updates


def test_unknown_provider_still_emits_pointer_only() -> None:
    """Unknown provider: only CHRYS_MODEL_PROFILE."""
    p = ModelProfile(id="x", name="X", provider="azure", api_key="k", base_url="https://u")
    updates = profile_to_activation_env_updates(p)
    assert updates == {"CHRYS_MODEL_PROFILE": "x"}


def test_get_active_profile_id_reads_chrys_model_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "new-id")
    assert get_active_profile_id() == "new-id"


def test_get_active_profile_id_ignores_retired_legacy_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-rename ``MODEL_PROFILE`` variable is no longer honoured."""
    monkeypatch.delenv("CHRYS_MODEL_PROFILE", raising=False)
    monkeypatch.setenv("MODEL_PROFILE", "legacy-id")
    assert get_active_profile_id() == ""


def test_get_active_profile_id_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHRYS_MODEL_PROFILE", raising=False)
    assert get_active_profile_id() == ""


def _read_stored_pointer(config_dir: Path) -> object | None:
    settings_path = config_dir / "settings.yaml"
    if not settings_path.exists():
        return None
    doc = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    try:
        return doc["model"]["profile"]["active"]
    except KeyError, TypeError:
        return None


def test_activate_model_profile_updates_only_pointer_and_live_process_without_dotenv_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_original = "# keep\nCHRYS_THEME=file-theme\nCHRYS_MODEL_PROFILE=old-id\n"
    path = tmp_path / ".env"
    path.write_text(env_original, encoding="utf-8")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "old-id")
    monkeypatch.setenv("CHRYS_THEME", "process-theme")

    with patch("dotenv.load_dotenv") as load_dotenv:
        activate_model_profile("new-id")

    assert load_dotenv.call_count == 0
    # The dotenv is user-owned now; the durable pointer lives in the document.
    assert path.read_text(encoding="utf-8") == env_original
    assert _read_stored_pointer(tmp_path) == "new-id"
    assert os.environ["CHRYS_MODEL_PROFILE"] == "new-id"
    assert os.environ["CHRYS_THEME"] == "process-theme"
    origin = get_model_pointer()[1]
    assert origin is not None and origin.layer is Source.PROCESS_RUNTIME


def test_activate_model_profile_leaves_the_dotenv_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even a legacy pointer line is not activation's to rewrite — migration owns it."""
    env_original = "OPENAI_API_KEY=user-owned\nCHRYS_MODEL_PROFILE=old-id\n"
    path = tmp_path / ".env"
    path.write_text(env_original, encoding="utf-8")
    monkeypatch.delenv("CHRYS_MODEL_PROFILE", raising=False)

    activate_model_profile("new-id")

    assert path.read_text(encoding="utf-8") == env_original
    os.environ.pop("CHRYS_MODEL_PROFILE", None)


def test_set_global_default_profile_id_is_file_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "process-id")

    set_global_default_profile_id("file-id")

    assert _read_stored_pointer(tmp_path) == "file-id"
    assert os.environ["CHRYS_MODEL_PROFILE"] == "process-id"
    assert not (tmp_path / ".env").exists()


def test_set_global_default_profile_id_empty_removes_the_stored_pointer(tmp_path: Path) -> None:
    set_global_default_profile_id("file-id")
    assert get_global_default_profile_id() == "file-id"

    set_global_default_profile_id("")

    assert _read_stored_pointer(tmp_path) is None
    assert get_global_default_profile_id() == ""


def test_get_global_default_profile_id_reads_file_instead_of_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "settings.yaml").write_text(
        yaml.safe_dump({"model": {"profile": {"active": "file-id"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "process-id")

    assert get_global_default_profile_id() == "file-id"


def test_get_global_default_profile_id_ignores_the_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leftover dotenv pointer line feeds startup layers, not the stored default."""
    (tmp_path / ".env").write_text("CHRYS_MODEL_PROFILE=legacy-file-id\n", encoding="utf-8")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "legacy-process-id")

    assert get_global_default_profile_id() == ""


# ── NO_PROXY validation / scrub ─────────────────────────────────────


def test_is_valid_no_proxy_accepts_typical_inputs() -> None:
    """Common shapes the placeholder advertises must validate."""
    assert is_valid_no_proxy("") is True
    assert is_valid_no_proxy("localhost") is True
    assert is_valid_no_proxy("127.0.0.1") is True
    assert is_valid_no_proxy("::1") is True  # bare IPv6, no brackets
    assert is_valid_no_proxy("localhost,127.0.0.1,.example.com") is True
    assert is_valid_no_proxy("*.example.com") is True
    assert is_valid_no_proxy("*") is True  # bypass-everything wildcard
    assert is_valid_no_proxy("http://x.y") is True


def test_is_valid_no_proxy_rejects_non_printable() -> None:
    """Non-printable ASCII (tab/LF/control) — fails ``urlparse`` in httpx."""
    assert is_valid_no_proxy("host\twithtab") is False
    assert is_valid_no_proxy("a\nb") is False
    assert is_valid_no_proxy("\x7f") is False  # DEL


def test_is_valid_no_proxy_rejects_bracketed_ipv6() -> None:
    """``[::1]`` is printable but URLPattern rejects it (brackets are added by httpx)."""
    assert is_valid_no_proxy("[::1]") is False


def test_is_valid_no_proxy_rejects_when_any_entry_is_bad() -> None:
    """One bad entry in a comma-separated list invalidates the whole value."""
    assert is_valid_no_proxy("localhost,[::1]") is False
    assert is_valid_no_proxy("good,host\twithtab,also-good") is False


def test_is_valid_no_proxy_wildcard_short_circuits() -> None:
    """``*`` short-circuits httpx's bypass parsing; later entries are ignored.

    ``httpx._utils.get_environment_proxies`` returns ``{}`` immediately on
    ``*``, so malformed entries that follow never reach ``URLPattern`` —
    the validator must mirror that to avoid false rejections.
    """
    assert is_valid_no_proxy("*,[::1]") is True
    assert is_valid_no_proxy("localhost,*,host\twithtab") is True


def test_sanitize_no_proxy_env_noop_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert sanitize_no_proxy_env() == {}


def test_sanitize_no_proxy_env_keeps_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``delenv`` first because Windows ``os.environ`` is case-insensitive —
    # deleting ``no_proxy`` after setting ``NO_PROXY`` would wipe the same key.
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "localhost,*.example.com")
    assert sanitize_no_proxy_env() == {}
    assert os.environ["NO_PROXY"] == "localhost,*.example.com"


def test_sanitize_no_proxy_env_scrubs_uppercase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "host\twithtab")
    assert sanitize_no_proxy_env() == {"NO_PROXY": "host\twithtab"}
    assert "NO_PROXY" not in os.environ


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows env vars are case-insensitive: setting 'no_proxy' actually sets 'NO_PROXY'.",
)
def test_sanitize_no_proxy_env_scrubs_lowercase(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx falls back to lowercase when NO_PROXY is empty — must scrub it too."""
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setenv("no_proxy", "bad\nvalue")
    assert sanitize_no_proxy_env() == {"no_proxy": "bad\nvalue"}
    assert "no_proxy" not in os.environ


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows env vars are case-insensitive: 'NO_PROXY' and 'no_proxy' are the same key.",
)
def test_sanitize_no_proxy_env_scrubs_both_when_both_bad(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_PROXY", "U\tBAD")
    monkeypatch.setenv("no_proxy", "l\tbad")
    scrubbed = sanitize_no_proxy_env()
    assert scrubbed == {"NO_PROXY": "U\tBAD", "no_proxy": "l\tbad"}
    assert "NO_PROXY" not in os.environ
    assert "no_proxy" not in os.environ
