# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Desktop notification support for the TUI."""

from chrys.app.tui.notifications.service import NotificationService
from chrys.app.tui.notifications.settings import NotificationEvent, NotificationSettings

__all__ = [
    "NotificationEvent",
    "NotificationService",
    "NotificationSettings",
]
