# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Publish the hook runs a session's hook manager has in flight, and record its runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chrys.foundation.events.types import HookExecutionsChanged, HookRunsChanged

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.service.hooks.manager import HookManager
    from chrys.service.hooks.run_log import HookRunLog


def wire_hook_reporting(
    manager: HookManager, bus: EventBus, *, session_id: str, run_log: HookRunLog | None = None
) -> None:
    """Publish *manager*'s runs in flight on *bus* for *session_id*, after each change.

    With *run_log*, every run is recorded there too, and ``HookRunsChanged``
    published after each write.
    """

    async def _publish_executions(source: HookManager) -> None:
        await bus.publish(HookExecutionsChanged(session_id=session_id, executions=source.executions()))

    manager.on_change = _publish_executions
    if run_log is None:
        return

    async def _publish_runs() -> None:
        await bus.publish(HookRunsChanged(session_id=session_id))

    manager.record_runs(run_log, session_id=session_id, on_change=_publish_runs)
