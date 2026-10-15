# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shell commands that keep running after their tool call returned."""

from __future__ import annotations

from chrys.service.tools.background.registry import (
    BACKGROUND_DIR_NAME,
    MAIN_SCOPE,
    MAX_RUNNING_TASKS,
    BackgroundProcess,
    BackgroundTask,
    BackgroundTasks,
    StopCause,
    TaskScope,
    current_scope,
    stop_command,
)

__all__ = [
    "BACKGROUND_DIR_NAME",
    "MAIN_SCOPE",
    "MAX_RUNNING_TASKS",
    "BackgroundProcess",
    "BackgroundTask",
    "BackgroundTasks",
    "StopCause",
    "TaskScope",
    "current_scope",
    "stop_command",
]
