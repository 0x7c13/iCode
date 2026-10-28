# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Startup patch records stay off the stderr of a process that configured no logging."""

from __future__ import annotations

import logging

import pytest

from chrys.foundation.patches import patcher, textual_compositor_cjk


def test_patch_records_stay_off_stderr_where_no_logging_is_configured(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # As a process starts: with no handler on the root logger, Python prints a warning on stderr.
    monkeypatch.setattr(logging.getLogger(), "handlers", [])

    logging.getLogger("chrys.service.profiles.agents.registry").warning("Failed to load agent profile")
    patcher.logger.warning("Patch error: probe — Could not write the file")
    textual_compositor_cjk.logger.warning("Skipping Textual compositor CJK runtime patch")

    err = capsys.readouterr().err
    assert "Failed to load agent profile" in err
    assert "Patch error" not in err
    assert "Skipping" not in err


def test_patch_records_still_reach_a_configured_handler(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        patcher.logger.warning("Patch error: probe — Could not write the file")

    assert [record.getMessage() for record in caplog.records] == ["Patch error: probe — Could not write the file"]
