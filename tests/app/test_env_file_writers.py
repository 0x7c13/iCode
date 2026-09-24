# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cross-surface tests for the single Chrys settings write channel."""

from __future__ import annotations

import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

import chrys.foundation.platform as platform_mod
from chrys.foundation.config.settings import (
    persist_approval_mode,
    persist_editor_keymap,
    persist_locale,
    persist_theme,
)
from chrys.foundation.config.settings_store import persist
from chrys.service.profiles.models.env_bridge import set_global_default_profile_id


def test_all_settings_writers_interleave_without_losing_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every persistence surface funnels into one locked document write.

    The dotenv file is not that channel any more: it keeps only the user's own
    variables, and no writer here may touch it or mirror into the process
    environment — a mirrored value would come back as the ``ENV`` layer and
    outrank the very document these writes target.
    """
    fake = dataclasses.replace(platform_mod.get_platform(), config_dir=tmp_path, data_dir=tmp_path)
    monkeypatch.setattr(platform_mod, "get_platform", lambda: fake)
    for key in ("CHRYS_THEME", "CHRYS_LOCALE", "CHRYS_DEFAULT_APPROVAL_MODE", "CHRYS_EDITOR_KEYMAP"):
        monkeypatch.delenv(key, raising=False)
    env_path = tmp_path / ".env"
    env_original = "# preserve\nOPENAI_API_KEY=legacy\nUNRELATED=keep\n"
    env_path.write_text(env_original, encoding="utf-8")
    writers = [
        lambda: persist_theme("midnight"),
        lambda: persist_locale("zh-Hans"),
        lambda: persist_approval_mode("auto"),
        lambda: persist_editor_keymap("vim"),
        lambda: set_global_default_profile_id("model-id"),
        lambda: persist({"rollback.snapshots_keep": 7}),
    ]

    with ThreadPoolExecutor(max_workers=len(writers)) as executor:
        list(executor.map(lambda writer: writer(), writers))

    doc = yaml.safe_load((tmp_path / "settings.yaml").read_text(encoding="utf-8"))
    assert doc["ui"]["theme"] == "midnight"
    assert doc["ui"]["locale"] == "zh-Hans"
    assert doc["ui"]["editor"]["keymap"] == "vim"
    assert doc["approval"]["default_mode"] == "auto"
    assert doc["model"]["profile"]["active"] == "model-id"
    assert doc["rollback"]["snapshots_keep"] == 7
    assert env_path.read_text(encoding="utf-8") == env_original
    for key in ("CHRYS_THEME", "CHRYS_LOCALE", "CHRYS_DEFAULT_APPROVAL_MODE", "CHRYS_EDITOR_KEYMAP"):
        assert key not in os.environ
