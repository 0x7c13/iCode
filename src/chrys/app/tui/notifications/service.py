# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Runtime notification service for the TUI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from chrys.app.tui.i18n import render_str
from chrys.app.tui.notifications.drivers import (
    NotificationDeliveryResult,
    NotificationDriver,
    NotificationPayload,
    default_notification_driver,
)
from chrys.app.tui.notifications.settings import NOTIFICATION_EVENT_LABELS, NotificationEvent, NotificationSettings
from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from textual.app import App

    from chrys.app.tui.i18n import LocaleController

logger = logging.getLogger(__name__)


class NotificationService:
    """Coordinates focus-aware desktop notifications for the TUI."""

    def __init__(
        self,
        app: App[object],
        settings: NotificationSettings | None = None,
        driver: NotificationDriver | None = None,
        locale_controller: LocaleController | None = None,
    ) -> None:
        self._app = app
        self._settings = settings or NotificationSettings()
        self._driver = driver or default_notification_driver()
        self._locale_controller = locale_controller
        self._focus_known = False
        self._focused = True
        self._tasks: set[asyncio.Task[NotificationDeliveryResult]] = set()

    @property
    def settings(self) -> NotificationSettings:
        """Current live notification settings."""
        return self._settings

    def update_settings(self, settings: NotificationSettings) -> None:
        """Swap live notification settings without reloading the engine."""
        self._settings = settings

    def set_focused(self, focused: bool) -> None:
        """Record app focus state from Textual AppFocus/AppBlur events."""
        self._focus_known = True
        self._focused = focused

    def notify(self, event: NotificationEvent, *, force: bool = False) -> bool:
        """Schedule a notification if current settings allow it.

        Returns True when delivery was accepted and scheduled.
        """
        try:
            settings = self._settings
            if not force and not self._should_notify(event, settings):
                return False
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._deliver(event, settings))
        except RuntimeError:
            return False
        except Exception:
            logger.debug("Notification scheduling failed", exc_info=True)
            return False
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return True

    async def test(self, settings: NotificationSettings | None = None) -> bool:
        """Send a test notification and report whether any requested channel delivered."""
        active_settings = settings or self._settings
        try:
            result = await self._deliver(NotificationEvent.TURN_COMPLETE, active_settings)
        except Exception:
            logger.debug("Test notification delivery failed", exc_info=True)
            return False
        return (active_settings.desktop and result.desktop_sent) or (active_settings.sound and result.sound_sent)

    def _should_notify(self, event: NotificationEvent, settings: NotificationSettings) -> bool:
        if not settings.enabled:
            return False
        if not settings.event_enabled(event):
            return False
        return not (settings.suppress_when_focused and self._focus_known and self._focused)

    async def _deliver(self, event: NotificationEvent, settings: NotificationSettings) -> NotificationDeliveryResult:
        title = APP_DISPLAY_NAME
        reference = NOTIFICATION_EVENT_LABELS[event].bind()
        controller = self._locale_controller
        body = format_message(reference) if controller is None else render_str(controller.localizer, reference)
        payload = NotificationPayload(
            title=title,
            body=body,
            sound=settings.sound,
            desktop=settings.desktop,
        )
        result = await self._driver.send(payload)
        sound_sent = result.sound_sent
        if settings.sound and not result.sound_sent:
            with contextlib.suppress(Exception):
                self._app.bell()
                sound_sent = True
        return NotificationDeliveryResult(desktop_sent=result.desktop_sent, sound_sent=sound_sent)

    def _on_task_done(self, task: asyncio.Task[NotificationDeliveryResult]) -> None:
        self._tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Notification delivery failed", exc_info=True)
