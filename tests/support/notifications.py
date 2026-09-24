# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Capture test notifications without desktop popups, sounds or fallback bells."""

from __future__ import annotations

from chrys.app.tui.notifications.drivers import NotificationDeliveryResult, NotificationPayload


class RecordingNotificationDriver:
    """Acknowledge requested channels while keeping their payloads inspectable."""

    def __init__(self) -> None:
        self.payloads: list[NotificationPayload] = []

    async def send(self, payload: NotificationPayload) -> NotificationDeliveryResult:
        self.payloads.append(payload)
        return NotificationDeliveryResult(desktop_sent=payload.desktop, sound_sent=payload.sound)
