# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Typed outcomes for strict recovery-sidecar persistence."""

from __future__ import annotations

from enum import Enum


class RecoveryPersistOutcome(Enum):
    """Result of one strict recovery persistence request."""

    PERSISTED = "persisted"
    UNCONFIGURED = "unconfigured"
    NOTHING_TO_PERSIST = "nothing_to_persist"
    FAILED = "failed"
