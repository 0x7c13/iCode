# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Immutable graph inputs, independent of transcript and session projection state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from chrys.service.workflows.transcript import NodeUsage


@dataclass(frozen=True, slots=True)
class RetryTarget:
    run_id: str
    activation_id: str
    attempt: int


@dataclass(frozen=True, slots=True)
class NodeView:
    state: str = "pending"
    elapsed_seconds: float | None = None
    running_since: datetime | None = None
    usage: NodeUsage | None = None
    retry: RetryTarget | None = None
    retry_pending: bool = False
