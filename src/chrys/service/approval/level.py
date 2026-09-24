# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tool base types and approval level definitions."""

from __future__ import annotations

from enum import Enum


class ApprovalLevel(Enum):
    """Approval levels for tool execution."""

    AUTO = "auto"  # Execute without asking
    REQUIRE = "require"  # Pause and ask user
    SKIP = "skip"  # User has opted to skip approval for this session
