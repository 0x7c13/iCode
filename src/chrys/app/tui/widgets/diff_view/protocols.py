# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Structural protocols for diff widgets.

Kept dependency-free so capability checks stay importable without pulling
in the heavyweight widget modules of this package.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsPrepare(Protocol):
    """Diff widget that pre-computes expensive render state off-thread.

    Await ``prepare()`` before mounting so ``compose()`` finds cached
    results instantly instead of blocking the event loop on Pygments
    highlighting and ``SequenceMatcher`` diffing. Implemented by
    ``DiffView`` and ``InlineUnifiedDiffLines``.
    """

    async def prepare(self) -> None: ...
