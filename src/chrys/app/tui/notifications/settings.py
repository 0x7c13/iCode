# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Notification settings view object and event labels.

The persisted model lives in :class:`chrys.foundation.config.settings.Settings`
as eight flattened ``notifications.*`` fields; this module keeps the TUI-facing
view object (:class:`NotificationSettings`) and the event labels. The enum
itself is foundation data, re-exported here for the delivery layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from chrys.foundation.config.notification_events import NotificationEvent
from chrys.foundation.i18n import MessageDef, msg

if TYPE_CHECKING:
    from chrys.foundation.config.settings import Settings

NOTIFICATIONS_TITLE = msg("tui.notifications.title", fallback="Notifications")

_APPROVAL_REQUIRED_LABEL = msg(
    "tui.notifications.event.approval_required.label",
    fallback="Approval required",
)
_ASK_USER_LABEL = msg(
    "tui.notifications.event.ask_user.label",
    fallback="Agent needs input",
)
_TURN_COMPLETE_LABEL = msg(
    "tui.notifications.event.turn_complete.label",
    fallback="Agent finished",
)
_TURN_ERROR_LABEL = msg(
    "tui.notifications.event.turn_error.label",
    fallback="Agent error",
)
_APPROVAL_REQUIRED_DESCRIPTION = msg(
    "tui.notifications.event.approval_required.description",
    fallback="Notify when a tool approval needs your decision",
)
_ASK_USER_DESCRIPTION = msg(
    "tui.notifications.event.ask_user.description",
    fallback="Notify when the agent asks a question",
)
_TURN_COMPLETE_DESCRIPTION = msg(
    "tui.notifications.event.turn_complete.description",
    fallback="Notify when a turn finishes",
)
_TURN_ERROR_DESCRIPTION = msg(
    "tui.notifications.event.turn_error.description",
    fallback="Notify when a running turn errors",
)

NOTIFICATION_EVENT_LABELS: dict[NotificationEvent, MessageDef] = {
    NotificationEvent.APPROVAL_REQUIRED: _APPROVAL_REQUIRED_LABEL,
    NotificationEvent.ASK_USER: _ASK_USER_LABEL,
    NotificationEvent.TURN_COMPLETE: _TURN_COMPLETE_LABEL,
    NotificationEvent.TURN_ERROR: _TURN_ERROR_LABEL,
}

NOTIFICATION_EVENT_DESCRIPTIONS: dict[NotificationEvent, MessageDef] = {
    NotificationEvent.APPROVAL_REQUIRED: _APPROVAL_REQUIRED_DESCRIPTION,
    NotificationEvent.ASK_USER: _ASK_USER_DESCRIPTION,
    NotificationEvent.TURN_COMPLETE: _TURN_COMPLETE_DESCRIPTION,
    NotificationEvent.TURN_ERROR: _TURN_ERROR_DESCRIPTION,
}


def default_event_settings() -> dict[NotificationEvent, bool]:
    """Return per-event default notification settings."""
    return dict.fromkeys(NotificationEvent, True)


@dataclass(slots=True)
class NotificationSettings:
    """TUI notification preferences, projected from :class:`Settings`.

    A view object: the durable values are the flattened ``notifications.*``
    fields on the settings dataclass, and this shape exists so the delivery
    service and the settings dialog keep their existing accessors.
    """

    enabled: bool = True
    desktop: bool = True
    sound: bool = True
    suppress_when_focused: bool = True
    events: dict[NotificationEvent, bool] = field(default_factory=default_event_settings)

    def event_enabled(self, event: NotificationEvent) -> bool:
        """Return True if notifications are enabled for *event*."""
        return bool(self.events.get(event, True))

    @classmethod
    def from_settings(cls, settings: Settings) -> NotificationSettings:
        """Project the flattened settings fields into the view shape."""
        return cls(
            enabled=settings.notifications_enabled,
            desktop=settings.notifications_desktop,
            sound=settings.notifications_sound,
            suppress_when_focused=settings.notifications_suppress_when_focused,
            events={
                NotificationEvent.APPROVAL_REQUIRED: settings.notifications_event_approval_required,
                NotificationEvent.ASK_USER: settings.notifications_event_ask_user,
                NotificationEvent.TURN_COMPLETE: settings.notifications_event_turn_complete,
                NotificationEvent.TURN_ERROR: settings.notifications_event_turn_error,
            },
        )

    def to_settings_fields(self) -> dict[str, bool]:
        """Render this view as ``Settings`` field names, the inverse of :meth:`from_settings`.

        Separate from :meth:`to_settings_patch` because the two ends of a live
        edit speak different spellings: the store persists dotted keys, and the
        runtime overlay on the shared handle is keyed by dataclass field.
        """
        return {
            "notifications_enabled": self.enabled,
            "notifications_desktop": self.desktop,
            "notifications_sound": self.sound,
            "notifications_suppress_when_focused": self.suppress_when_focused,
            "notifications_event_approval_required": self.event_enabled(NotificationEvent.APPROVAL_REQUIRED),
            "notifications_event_ask_user": self.event_enabled(NotificationEvent.ASK_USER),
            "notifications_event_turn_complete": self.event_enabled(NotificationEvent.TURN_COMPLETE),
            "notifications_event_turn_error": self.event_enabled(NotificationEvent.TURN_ERROR),
        }

    def to_settings_patch(self) -> dict[str, bool]:
        """Render this view as the dotted-key patch the settings store persists."""
        return {
            "notifications.enabled": self.enabled,
            "notifications.delivery.desktop": self.desktop,
            "notifications.delivery.sound": self.sound,
            "notifications.suppress_when_focused": self.suppress_when_focused,
            "notifications.events.approval_required": self.event_enabled(NotificationEvent.APPROVAL_REQUIRED),
            "notifications.events.ask_user": self.event_enabled(NotificationEvent.ASK_USER),
            "notifications.events.turn_complete": self.event_enabled(NotificationEvent.TURN_COMPLETE),
            "notifications.events.turn_error": self.event_enabled(NotificationEvent.TURN_ERROR),
        }
