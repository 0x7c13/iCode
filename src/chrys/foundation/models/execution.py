# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Immutable execution lease state shared with frontends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExecutionSnapshot:
    """What holds the lease: nothing, a turn, or a workflow run (``run_id`` set)."""

    kind: Literal["idle", "turn", "workflow"]
    run_id: str = ""
    cancellable: bool = False
    request_id: str = ""
