# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Textual OptionList stale-click patch."""

from __future__ import annotations

from dataclasses import dataclass

from textual.widgets import OptionList
from textual.widgets.option_list import Option

from chrys.foundation.patches.textual_option_list import apply_runtime_patch


@dataclass
class _Style:
    meta: dict[str, object]


@dataclass
class _Click:
    style: _Style


async def test_runtime_patch_ignores_stale_option_click_metadata() -> None:
    apply_runtime_patch()
    option_list = OptionList(Option("only"))
    option_list.highlighted = None

    await option_list._on_click(_Click(_Style({"option": 2})))

    assert option_list.highlighted is None


async def test_runtime_patch_ignores_negative_option_click_metadata() -> None:
    apply_runtime_patch()
    option_list = OptionList(Option("only"))
    option_list.highlighted = None

    await option_list._on_click(_Click(_Style({"option": -1})))

    assert option_list.highlighted is None
