# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The order a person reads a list of names in."""

from __future__ import annotations

import re

_DIGIT_RUNS = re.compile(r"(\d+)")


def reading_order(name: str) -> tuple[tuple[int, int | str], ...]:
    """Sort key for a name as a person reads it: case aside, and a run of digits as the number it is.

    Character by character, ``GPT-5.10`` sorts ahead of ``GPT-5.9``. The leading tag keeps a
    number from ever being compared with text, and puts it first where the two meet.
    """
    # Splitting on a group alternates: text at the even places, digit runs at the odd ones.
    parts = enumerate(_DIGIT_RUNS.split(name.casefold()))
    return tuple((0, int(part)) if index % 2 else (1, part) for index, part in parts if part)
