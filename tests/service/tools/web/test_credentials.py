# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Search secrets come from the process frozen at bootstrap, then the user dotenv; never a project one."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chrys.foundation.config.env_layers import freeze_process_env
from chrys.service.tools.builtins.web.credentials import resolve_search_credentials


def test_credentials_frozen_process_beats_user_and_live_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_TEST_KEY", "frozen-secret")
    freeze_process_env()
    monkeypatch.setenv("WEB_TEST_KEY", "project-secret")
    dotenv = tmp_path / ".env"
    dotenv.write_text("WEB_TEST_KEY=user-secret\n", encoding="utf-8")
    with patch("chrys.service.tools.builtins.web.credentials.config_env_path", autospec=True, return_value=dotenv):
        credentials = resolve_search_credentials({"WEB_TEST_KEY"})
    assert credentials.values == {"WEB_TEST_KEY": "frozen-secret"}
    assert "frozen-secret" not in repr(credentials)


def test_empty_process_key_does_not_resurrect_user_key(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_TEST_KEY", "")
    freeze_process_env()
    dotenv = tmp_path / ".env"
    dotenv.write_text("WEB_TEST_KEY=old-secret\n", encoding="utf-8")
    with (
        patch("chrys.service.tools.builtins.web.credentials.config_env_path", autospec=True, return_value=dotenv),
        pytest.raises(ValueError, match="Missing"),
    ):
        resolve_search_credentials({"WEB_TEST_KEY"})
