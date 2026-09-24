# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for shared TUI formatting helpers."""

from __future__ import annotations

import pytest

from chrys.app.tui.util.formatting import format_token_count


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1_000, "1.0k"),
        (12_200, "12.2k"),
        (999_949, "999.9k"),
        (999_950, "1.0m"),
        (1_852_300, "1.9m"),
        (999_949_999, "999.9m"),
        (999_950_000, "1.0b"),
        (1_000_000_000, "1.0b"),
    ],
)
def test_format_token_count_uses_compact_units_without_rounded_overflow(value: int, expected: str) -> None:
    assert format_token_count(value) == expected
