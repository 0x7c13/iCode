# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Golden reminder scenarios: every call's wire view, record and state match the baseline.

The scenarios live in ``tests/support/reminder_goldens.py``; their golden files in
``reminder_goldens/`` were reviewed by hand and are never regenerated to make a
change pass — a mismatch means the reminders sent, recorded or retained changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.support.reminder_goldens import SCENARIOS, load_golden, run_scenario

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_reminder_golden(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    actual, roots = await run_scenario(name, tmp_path, monkeypatch)
    expected = load_golden(name, roots)

    assert actual["about"] == expected["about"]
    assert [step["step"] for step in actual["steps"]] == [step["step"] for step in expected["steps"]]
    for actual_step, expected_step in zip(actual["steps"], expected["steps"], strict=True):
        assert actual_step == expected_step, actual_step["step"]
    assert actual == expected


def test_every_golden_file_has_a_scenario() -> None:
    from tests.support.reminder_goldens import GOLDEN_DIR

    assert sorted(path.stem for path in GOLDEN_DIR.glob("*.json")) == sorted(SCENARIOS)
