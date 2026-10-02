# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Typed approval choices and presentation data; neither supplies authorization keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReuseChoice = Literal["", "EXACT_SESSION", "EXACT_PROJECT", "PREFIX_SESSION", "PREFIX_PROJECT"]


@dataclass(frozen=True)
class ApprovalReuseOffer:
    kind: Literal["command", "files"]
    project: str
    targets: tuple[str, ...]
    shell: str = ""
    session: bool = True
    prefix: bool = False
