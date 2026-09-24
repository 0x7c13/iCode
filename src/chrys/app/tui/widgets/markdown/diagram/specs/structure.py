# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Explicit grid placement retained from block diagrams."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BlockCell:
    """One block or intentional empty span in source order."""

    node_id: str | None
    span: int = 1


@dataclass(frozen=True, slots=True)
class BlockChart:
    """Fixed columns, independent of graph rank assignment."""

    columns: int
    cells: tuple[BlockCell, ...]
