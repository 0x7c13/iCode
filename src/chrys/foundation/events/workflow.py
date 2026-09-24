# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared workflow lifecycle event family for subscribers and historical projection."""

from __future__ import annotations

from chrys.foundation.events.types import (
    WorkflowLoopIteration,
    WorkflowNodeAnswered,
    WorkflowNodeAskUser,
    WorkflowNodeOutput,
    WorkflowNodeStateChanged,
    WorkflowRunFinished,
    WorkflowRunNotice,
    WorkflowRunStarted,
)

type WorkflowRunEvent = (
    WorkflowRunStarted
    | WorkflowNodeStateChanged
    | WorkflowNodeOutput
    | WorkflowNodeAskUser
    | WorkflowNodeAnswered
    | WorkflowLoopIteration
    | WorkflowRunNotice
    | WorkflowRunFinished
)

WORKFLOW_RUN_EVENTS = (
    WorkflowRunStarted,
    WorkflowNodeStateChanged,
    WorkflowNodeOutput,
    WorkflowNodeAskUser,
    WorkflowNodeAnswered,
    WorkflowLoopIteration,
    WorkflowRunNotice,
    WorkflowRunFinished,
)
