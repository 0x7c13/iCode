# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tool cards keep compact durations; live commands retain seconds through days."""

from __future__ import annotations

import pytest

from chrys.app.tui.widgets.chat.renderers.execute import _fmt_duration
from chrys.app.tui.widgets.chat.tool_call import fmt_duration


@pytest.mark.parametrize(
    ("seconds", "card", "command"),
    [
        (0, "0ms", "0s"),
        (59, "59s", "59s"),
        (60, "1m", "1m"),
        (3599, "59m 59s", "59m 59s"),
        (3600, "1h", "1h"),
        (86399, "23h 59m", "23h 59m 59s"),
        (86400, "1d", "1d"),
        (86401, "1d", "1d 1s"),
        (102050, "1d 4h", "1d 4h 20m 50s"),
        (172800, "2d", "2d"),
    ],
)
def test_tool_and_command_duration_units(seconds: int, card: str, command: str) -> None:
    assert fmt_duration(seconds * 1000) == card
    assert _fmt_duration(seconds) == command
