# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Notification event identity, shared by settings and the TUI delivery layer.

Pure data on purpose: the persisted ``notifications.events.*`` settings keys
and the app-layer delivery service must agree on one set of event names, and
foundation cannot import the app layer, so the enum lives here. Labels,
descriptions, and delivery behaviour stay in ``chrys.app.tui.notifications``.
"""

from __future__ import annotations

from enum import StrEnum


class NotificationEvent(StrEnum):
    """User-visible notification reasons."""

    APPROVAL_REQUIRED = "approval_required"
    ASK_USER = "ask_user"
    TURN_COMPLETE = "turn_complete"
    TURN_ERROR = "turn_error"
