# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which hook run the hook indicator shows."""

from __future__ import annotations

from collections.abc import Iterable

from chrys.app.tui.util.formatting import elide_middle
from chrys.foundation.i18n import msg
from chrys.foundation.i18n.formatting import sanitize_legacy_scalar
from chrys.foundation.models.hook_executions import HookExecutionView

HOOK_SHOW_AFTER_SECONDS = 1.0
"""A run shows only once it has lasted this long: shorter ones would flash the indicator."""

HOOK_INDICATOR_IDLE = msg("tui.hook_indicator.idle", fallback="Hooks")


def indicator_activity(executions: Iterable[HookExecutionView], *, now: float) -> HookExecutionView | None:
    """The run the indicator shows at *now*: the latest started of those someone waits on.

    Runs nobody waits on (``fire_and_forget``, ``user_interrupt`` observers)
    are not shown. A run counts once it has lasted
    :data:`HOOK_SHOW_AFTER_SECONDS`, queueing included; a running one goes
    before a queued one. When the latest one ends, the one started before it
    shows again, since it is now the latest still in flight.
    """
    shown = [
        view for view in executions if view.waited_at != "none" and now - view.started_at >= HOOK_SHOW_AFTER_SECONDS
    ]
    if not shown:
        return None
    running = [view for view in shown if not view.queued]
    return max(running or shown, key=lambda view: view.started_at)


def hook_activity_text(activity: HookExecutionView, *, max_cells: int | None = None) -> str:
    """The shown run's hook id, cut to *max_cells* cells in the middle."""
    text = sanitize_legacy_scalar(activity.hook_id)
    if max_cells is not None:
        text = elide_middle(text, max_cells)
    return text
