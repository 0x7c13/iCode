# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Publish the hook runs a session's hook manager has in flight."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chrys.foundation.events.types import HookExecutionsChanged

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.service.hooks.manager import HookManager


def wire_hook_reporting(manager: HookManager, bus: EventBus, *, session_id: str) -> None:
    """Publish *manager*'s runs in flight on *bus* for *session_id*, after each change."""

    async def _publish_executions(source: HookManager) -> None:
        await bus.publish(HookExecutionsChanged(session_id=session_id, executions=source.executions()))

    manager.on_change = _publish_executions
