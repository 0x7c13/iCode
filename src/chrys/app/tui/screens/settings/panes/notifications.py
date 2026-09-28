# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The Notifications tab: the former notification dialog body, save-on-change."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.containers import Horizontal, VerticalGroup
from textual.widgets import Button, Static

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.notifications.settings import (
    NOTIFICATION_EVENT_DESCRIPTIONS,
    NOTIFICATION_EVENT_LABELS,
    NOTIFICATIONS_TITLE,
    NotificationEvent,
    NotificationSettings,
)
from chrys.app.tui.screens.settings.rows import HINT_TREE_GLYPH
from chrys.app.tui.widgets import Checkbox
from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.i18n import MessageRef, msg

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.screens.settings.ports import NotificationPorts

logger = logging.getLogger(__name__)

_SAVE_FAILED = msg(
    "tui.notifications.save_failed",
    fallback="Could not save notification settings",
)
_TEST_SENT = msg("tui.notifications.test_sent", fallback="Test notification sent")
_TEST_FAILED = msg(
    "tui.notifications.test_failed",
    fallback="Test notification could not be delivered",
)
_GENERAL = msg("tui.notifications.section.general", fallback="General")
_ENABLE_NOTIFICATIONS = msg("tui.notifications.enable", fallback="Enable Notifications")
_DELIVERY = msg("tui.notifications.section.delivery", fallback="Delivery")
_DESKTOP_POPUP = msg("tui.notifications.desktop_popup", fallback="Desktop popup")
_SOUND = msg("tui.notifications.sound", fallback="Sound")
_SUPPRESS_WHILE_FOCUSED = msg(
    "tui.notifications.suppress_while_focused",
    fallback="Suppress while {app} is focused",
)
_EVENTS = msg("tui.notifications.section.events", fallback="Events")
_TEST_BUTTON = msg("tui.notifications.button.test", fallback="Test")


class NotificationsPane(VerticalGroup):
    """Checkbox form over :class:`NotificationSettings`; every change saves."""

    DEFAULT_CLASSES = "settings-pane"

    def __init__(self, ports: NotificationPorts) -> None:
        super().__init__(id="notifications-content")
        self._ports = ports
        self._auto_save_ready = False
        self._composed_locale: str | None = None

    @property
    def projected(self) -> bool:
        """Whether the checkboxes show the ports' settings; until then there is nothing to repaint."""
        return self._auto_save_ready

    def compose(self) -> ComposeResult:
        self._composed_locale = widget_localizer(self).effective_locale
        settings = self._ports.current()
        with VerticalGroup(classes="settings-section") as general:
            general.border_title = Text(self._render_message(_GENERAL.bind()))
            yield Checkbox(
                self._render_message(_ENABLE_NOTIFICATIONS.bind()),
                value=settings.enabled,
                id="notifications-enabled",
            )
        with VerticalGroup(id="notifications-enabled-settings"):
            with VerticalGroup(classes="settings-section") as delivery:
                delivery.border_title = Text(self._render_message(_DELIVERY.bind()))
                yield Checkbox(
                    self._render_message(_DESKTOP_POPUP.bind()),
                    value=settings.desktop,
                    id="notifications-desktop",
                    classes="notification-delivery-option",
                )
                yield Checkbox(
                    self._render_message(_SOUND.bind()),
                    value=settings.sound,
                    id="notifications-sound",
                    classes="notification-delivery-option",
                )
                yield Checkbox(
                    self._render_message(_SUPPRESS_WHILE_FOCUSED.bind(app=APP_DISPLAY_NAME)),
                    value=settings.suppress_when_focused,
                    id="notifications-focus",
                    classes="notification-delivery-option",
                )
            with VerticalGroup(classes="settings-section") as events:
                events.border_title = Text(self._render_message(_EVENTS.bind()))
                for event in NotificationEvent:
                    yield Checkbox(
                        self._render_message(NOTIFICATION_EVENT_LABELS[event].bind()),
                        value=settings.event_enabled(event),
                        id=f"notifications-event-{event.value}",
                    )
                    with Horizontal(classes="notification-event-description-line"):
                        yield Static(HINT_TREE_GLYPH, classes="settings-row-hint-tree")
                        yield Static(
                            Text(self._render_message(NOTIFICATION_EVENT_DESCRIPTIONS[event].bind())),
                            classes="notification-event-description",
                        )
        with Horizontal(id="notifications-actions"):
            yield Button(
                Text(self._render_message(_TEST_BUTTON.bind())),
                id="notifications-test",
                classes="settings-row-link",
                flat=True,
            )

    def on_mount(self) -> None:
        if self._pruning or not self.is_attached or not self.app.is_running:
            # Removed while mounting, or the App quit mid-mount: Textual never
            # started the checkboxes but still sends Mount, and there is
            # nothing to sync.
            return
        # The dialog repaints only panes that have mounted: catch up with a
        # locale switch or a settings change that landed while this one mounted.
        if widget_localizer(self).effective_locale != self._composed_locale:
            self.refresh_localization()
        self.project()
        self._auto_save_ready = True

    def _sync_enabled_sections(self) -> None:
        self.query_one("#notifications-enabled-settings").display = self.query_one(
            "#notifications-enabled",
            Checkbox,
        ).value

    def _read_settings(self) -> NotificationSettings:
        events: dict[NotificationEvent, bool] = {}
        for event in NotificationEvent:
            events[event] = self.query_one(f"#notifications-event-{event.value}", Checkbox).value
        return NotificationSettings(
            enabled=self.query_one("#notifications-enabled", Checkbox).value,
            desktop=self.query_one("#notifications-desktop", Checkbox).value,
            sound=self.query_one("#notifications-sound", Checkbox).value,
            suppress_when_focused=self.query_one("#notifications-focus", Checkbox).value,
            events=events,
        )

    def project(self) -> None:
        """Repaint the checkboxes from the live settings without saving."""
        settings = self._ports.current()
        pairs: list[tuple[str, bool]] = [
            ("#notifications-enabled", settings.enabled),
            ("#notifications-desktop", settings.desktop),
            ("#notifications-sound", settings.sound),
            ("#notifications-focus", settings.suppress_when_focused),
        ]
        pairs.extend(
            (f"#notifications-event-{event.value}", settings.event_enabled(event)) for event in NotificationEvent
        )
        for selector, value in pairs:
            checkbox = self.query_one(selector, Checkbox)
            with checkbox.prevent(Checkbox.Changed):
                checkbox.value = value
        self._sync_enabled_sections()

    def refresh_localization(self) -> None:
        """Replace every piece of text in place; control state stays."""
        sections = self.query(".settings-section")
        for group, definition in zip(sections, (_GENERAL, _DELIVERY, _EVENTS), strict=True):
            group.border_title = Text(self._render_message(definition.bind()))
        self.query_one("#notifications-enabled", Checkbox).label = self._render_message(_ENABLE_NOTIFICATIONS.bind())
        self.query_one("#notifications-desktop", Checkbox).label = self._render_message(_DESKTOP_POPUP.bind())
        self.query_one("#notifications-sound", Checkbox).label = self._render_message(_SOUND.bind())
        self.query_one("#notifications-focus", Checkbox).label = self._render_message(
            _SUPPRESS_WHILE_FOCUSED.bind(app=APP_DISPLAY_NAME)
        )
        descriptions = list(self.query(".notification-event-description"))
        for index, event in enumerate(NotificationEvent):
            self.query_one(f"#notifications-event-{event.value}", Checkbox).label = self._render_message(
                NOTIFICATION_EVENT_LABELS[event].bind()
            )
            if index < len(descriptions):
                description = descriptions[index]
                if isinstance(description, Static):
                    description.update(Text(self._render_message(NOTIFICATION_EVENT_DESCRIPTIONS[event].bind())))
        self.query_one("#notifications-test", Button).label = Text(self._render_message(_TEST_BUTTON.bind()))

    def _save_current_settings(self) -> bool:
        try:
            return self._ports.save(self._read_settings())
        except Exception:
            logger.warning("Failed to apply notification settings", exc_info=True)
            with contextlib.suppress(Exception):
                self.notify(
                    self._render_message(_SAVE_FAILED.bind()),
                    title=self._render_message(NOTIFICATIONS_TITLE.bind()),
                    severity="error",
                    timeout=5,
                    markup=False,
                )
            return False

    @on(Checkbox.Changed)
    def _on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        self._sync_enabled_sections()
        if self._auto_save_ready:
            self._save_current_settings()

    @on(Button.Pressed, "#notifications-test")
    async def _on_test_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        try:
            sent = await self._ports.test(self._read_settings())
        except Exception:
            logger.warning("Failed to send test notification", exc_info=True)
            sent = False
        if sent:
            self.notify(
                self._render_message(_TEST_SENT.bind()),
                title=self._render_message(NOTIFICATIONS_TITLE.bind()),
                timeout=2,
                markup=False,
            )
        else:
            self.notify(
                self._render_message(_TEST_FAILED.bind()),
                title=self._render_message(NOTIFICATIONS_TITLE.bind()),
                severity="warning",
                timeout=3,
                markup=False,
            )

    def _render_message(self, reference: MessageRef) -> str:
        return render_str(widget_localizer(self), reference)


__all__ = ["NotificationsPane"]
