# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for converting Textual color tokens to Rich styles."""

from __future__ import annotations

import pytest
from rich.style import Style

from chrys.app.tui.util.rich_style import rich_style_from_textual_color


@pytest.mark.parametrize(
    ("color", "expected"),
    [
        ("ansi_yellow", "bold color(3)"),
        ("ansi_default", "bold default"),
        ("#123456", "bold #123456"),
        ("yellow", "bold #ffff00"),
    ],
)
def test_rich_style_from_textual_color(color: str, expected: str) -> None:
    assert str(rich_style_from_textual_color(color, bold=True)) == expected


@pytest.mark.parametrize("color", ["ansi_default 50%", "auto 87%", "none", "reverse"])
def test_rich_style_from_textual_color_falls_back_to_attributes(color: str) -> None:
    assert rich_style_from_textual_color(color, reverse=True) == Style(reverse=True)
