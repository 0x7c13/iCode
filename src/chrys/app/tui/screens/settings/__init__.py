# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Settings dialog: the public face other screen packages import."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chrys.app.tui.screens.settings.dialog import SettingsDialog
from chrys.app.tui.screens.settings.layout import (
    DEFERRED_KEYS,
    GENERAL_TAB_ID,
    NOTIFICATIONS_TAB_ID,
    TAB_IDS,
    TABS,
    Suggestions,
    rendered_keys,
)
from chrys.app.tui.screens.settings.ports import NotificationPorts, SessionStoragePorts, SettingsPanelPorts

__all__ = [
    "DEFERRED_KEYS",
    "GENERAL_TAB_ID",
    "NOTIFICATIONS_TAB_ID",
    "TABS",
    "TAB_IDS",
    "NotificationPorts",
    "SessionStoragePorts",
    "SettingsDialog",
    "SettingsPanelPorts",
    "Suggestions",
    "rendered_keys",
]


def __getattr__(name: str) -> object:
    """Load the settings modal independently of its layout and ports."""
    if name == "SettingsDialog":
        from chrys.app.tui.screens.settings.dialog import SettingsDialog

        globals()[name] = SettingsDialog
        return SettingsDialog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose the settings API before its modal is imported."""
    return sorted({*globals(), *__all__})
