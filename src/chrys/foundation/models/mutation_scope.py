# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Stable execution identities used by mutation queries and file rollback.

A scope identifies one Chat turn or one Workflow run, never a node/attempt.
Display ordinals are presentation data, not execution identities.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTurnScope:
    """One turn in a Chat session (retries retain this identity)."""

    turn_id: int


@dataclass(frozen=True)
class WorkflowRunScope:
    """One run in a Workflow session (all nodes and retries share it)."""

    run_id: str


type MutationScope = ChatTurnScope | WorkflowRunScope
