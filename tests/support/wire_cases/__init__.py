# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Scripted provider exchanges for the client contract tests, by name."""

from __future__ import annotations

from . import grid, named
from ._kit import Case, Reply

CASES = {**grid.CASES, **named.CASES}

__all__ = ["CASES", "Case", "Reply"]
