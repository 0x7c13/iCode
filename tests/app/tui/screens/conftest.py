# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared fixtures for screen tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.platform_fakes import platform_with_config_dir
from tests.support.waiting import reset_wait_deadline


@pytest.fixture
def isolated_chrys_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep agent-profile serializer paths inside pytest temp dirs.

    Opt-in via ``pytest.mark.usefixtures`` rather than autouse: the agent-config
    modules share it, while the other screen modules keep their own environment.
    Resetting the shared wait deadline first gives every test a fresh polling
    budget for the ``tests.support.waiting`` helpers.
    """
    reset_wait_deadline()
    fake_platform = platform_with_config_dir(tmp_path)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)


@pytest.fixture
def isolated_model_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep model-profile serializer paths inside pytest temp dirs, and nothing else.

    Deliberately narrower than :func:`isolated_chrys_config_dir`: the stand-in
    carries ``config_dir`` and no other field, so the day the model-config screen
    starts consulting another part of the platform record these tests raise
    ``AttributeError`` instead of quietly reading the developer's real machine.
    The absence is the assertion, so do not widen this to a full ``PlatformInfo``.
    """
    reset_wait_deadline()
    fake_platform = type("P", (), {"config_dir": tmp_path})()
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)


@pytest.fixture
def clear_model_profile_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start each test with no ``CHRYS_MODEL_PROFILE`` process pointer.

    ``setenv`` before ``delenv`` is deliberate rather than redundant: setting
    the key first registers it with monkeypatch even when the developer's shell
    never exported it, so a test that installs its own pointer mid-run is still
    unwound at teardown instead of leaking into the next test.
    """
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "")
    monkeypatch.delenv("CHRYS_MODEL_PROFILE")
