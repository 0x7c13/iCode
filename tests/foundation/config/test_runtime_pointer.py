# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the model-pointer origin registry and its loader attribution."""

from __future__ import annotations

import os

import pytest

import chrys.foundation.config.settings_store as settings_store
from chrys.foundation.config.runtime_pointer import (
    MODEL_POINTER_ENV,
    MODEL_POINTER_KEY,
    get_model_pointer,
    restore_model_pointer,
    set_model_pointer,
)
from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import load_settings
from chrys.foundation.config.spec import SettingOrigin, Source, specs_by_field


def _origin(layer: Source) -> SettingOrigin:
    return SettingOrigin(layer=layer)


def _claim_pointer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the carrier for restore so pointer writes never outlive the test."""
    monkeypatch.setenv(MODEL_POINTER_ENV, "claimed")
    monkeypatch.delenv(MODEL_POINTER_ENV)


def test_pointer_key_matches_the_settings_spec() -> None:
    entry = specs_by_field(Settings)["model_profile"]
    assert entry.key == MODEL_POINTER_KEY
    assert entry.env == MODEL_POINTER_ENV


def test_set_records_value_and_origin_together(monkeypatch: pytest.MonkeyPatch) -> None:
    _claim_pointer_env(monkeypatch)

    set_model_pointer("glm", origin=_origin(Source.PROCESS_RUNTIME))

    assert os.environ[MODEL_POINTER_ENV] == "glm"
    assert get_model_pointer() == ("glm", _origin(Source.PROCESS_RUNTIME))


def test_clearing_the_pointer_clears_the_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_POINTER_ENV, "glm")
    set_model_pointer("glm", origin=_origin(Source.PROCESS_RUNTIME))

    set_model_pointer(None, origin=_origin(Source.PROCESS_RUNTIME))

    assert MODEL_POINTER_ENV not in os.environ
    assert get_model_pointer() == ("", None)


def test_restore_puts_back_value_and_origin_together(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare env write would leave the origin describing the rolled-back writer."""
    monkeypatch.setenv(MODEL_POINTER_ENV, "cli-model")
    set_model_pointer("cli-model", origin=_origin(Source.CLI))

    token = set_model_pointer("restored-model", origin=_origin(Source.SESSION))
    assert get_model_pointer() == ("restored-model", _origin(Source.SESSION))

    restore_model_pointer(token)

    assert get_model_pointer() == ("cli-model", _origin(Source.CLI))


def test_restore_round_trips_an_absent_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    _claim_pointer_env(monkeypatch)

    token = set_model_pointer("restored-model", origin=_origin(Source.SESSION))
    restore_model_pointer(token)

    assert MODEL_POINTER_ENV not in os.environ
    assert get_model_pointer() == ("", None)


def test_restore_declines_when_the_pointer_moved_since_the_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token undoes one write, and is not a licence to overwrite a later one."""
    # The caller unwinding a failed restore has no way to know the user picked
    # a different profile while it was running. Replaying the token there is
    # not a rollback but a silent revert to a selection nobody asked for.
    _claim_pointer_env(monkeypatch)
    token = set_model_pointer("restored-model", origin=_origin(Source.SESSION))
    set_model_pointer("chosen-model", origin=_origin(Source.PROCESS_RUNTIME))

    assert restore_model_pointer(token) is False
    assert get_model_pointer() == ("chosen-model", _origin(Source.PROCESS_RUNTIME))


def test_restore_reports_that_it_put_the_pointer_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """The undone case answers ``True``, so the refusal above is not vacuous."""
    _claim_pointer_env(monkeypatch)
    token = set_model_pointer("restored-model", origin=_origin(Source.SESSION))

    assert restore_model_pointer(token) is True
    assert MODEL_POINTER_ENV not in os.environ


def test_a_restore_is_only_good_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restoring counts as a write, so a replayed token cannot re-fire."""
    monkeypatch.setenv(MODEL_POINTER_ENV, "cli-model")
    set_model_pointer("cli-model", origin=_origin(Source.CLI))
    token = set_model_pointer("restored-model", origin=_origin(Source.SESSION))
    restore_model_pointer(token)
    set_model_pointer("chosen-model", origin=_origin(Source.PROCESS_RUNTIME))

    assert restore_model_pointer(token) is False
    assert get_model_pointer() == ("chosen-model", _origin(Source.PROCESS_RUNTIME))


def test_the_loader_takes_the_pointer_value_and_origin_from_one_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read separately, the two halves can describe two different writes."""
    monkeypatch.setenv(MODEL_POINTER_ENV, "stale-model")
    set_model_pointer("stale-model", origin=_origin(Source.PROCESS_RUNTIME))
    reads: list[bool] = []

    def snapshot() -> tuple[str, SettingOrigin | None]:
        # Nothing here matches what the carrier or the registry say on their
        # own, so a loader consulting either directly is caught.
        reads.append(True)
        return "paired-model", _origin(Source.CLI)

    monkeypatch.setattr(settings_store, "get_model_pointer", snapshot)

    loaded = load_settings()

    assert len(reads) == 1
    assert loaded.settings.model_profile == "paired-model"
    assert loaded.source_for(MODEL_POINTER_KEY) == _origin(Source.CLI)


def test_loader_reports_the_registered_origin_for_the_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_POINTER_ENV, "placeholder")
    set_model_pointer("activated-model", origin=_origin(Source.PROCESS_RUNTIME))

    loaded = load_settings()

    assert loaded.settings.model_profile == "activated-model"
    assert loaded.source_for(MODEL_POINTER_KEY) == _origin(Source.PROCESS_RUNTIME)


def test_loader_blames_the_shell_for_an_unregistered_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """No registered writer means the value really is an export."""
    monkeypatch.setenv(MODEL_POINTER_ENV, "shell-model")

    loaded = load_settings()

    assert loaded.settings.model_profile == "shell-model"
    assert loaded.source_for(MODEL_POINTER_KEY) == _origin(Source.ENV)


def test_injected_test_environments_ignore_the_process_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_POINTER_ENV, "activated-model")
    set_model_pointer("activated-model", origin=_origin(Source.PROCESS_RUNTIME))

    loaded = load_settings(env={MODEL_POINTER_ENV: "hermetic-model"})

    assert loaded.settings.model_profile == "hermetic-model"
    assert loaded.source_for(MODEL_POINTER_KEY) == _origin(Source.ENV)


def test_a_host_pin_outranks_the_registered_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    """One ACP session's activation must not rewrite another session's pin."""
    monkeypatch.setenv(MODEL_POINTER_ENV, "activated-model")
    set_model_pointer("activated-model", origin=_origin(Source.PROCESS_RUNTIME))

    loaded = load_settings(model_profile="pinned-model")

    assert loaded.settings.model_profile == "pinned-model"
    assert loaded.source_for(MODEL_POINTER_KEY) == _origin(Source.SESSION)
