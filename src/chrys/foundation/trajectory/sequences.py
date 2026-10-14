# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ranges of event sequence numbers."""

from __future__ import annotations

from typing import NamedTuple


class SequenceRange(NamedTuple):
    """The event sequence numbers from *first* to *last*, both included."""

    first: int
    last: int
