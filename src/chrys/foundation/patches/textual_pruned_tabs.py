# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Patch: let a tab bar that is being removed, or mounts during exit, ignore activation.

Problem
-------
Textual 8.2.7's ``Tabs.validate_active`` raises ``ValueError("No Tab with id
...")`` when the requested tab is not under ``#tabs-list``. A ``Tabs`` removed
before its nested compose finishes never gets its ``Tab`` children, because
``mount()`` is a silent no-op while ``_pruning`` is set, yet Textual still
dispatches its ``Mount``. ``Tabs._on_mount`` then activates the constructor's
``active`` tab, the validator raises, and the whole app exits. Any bar built
with ``active=`` (and every ``TabbedContent(initial=...)``, whose ``ContentTabs``
inherits the validator) crashes this way when its screen or owner is removed,
or the app shuts down, within a few loop turns of mounting it. ``App.exit()``
opens the same window before any pruning starts: from then on ``mount_all``,
which a compose mounts through, returns without mounting, while the bar's own
``Mount`` still runs. Opening Settings and quitting at once hit it.

Solution
--------
While the bar is being removed, or the app is exiting and the tab is missing,
keep its current ``active`` value: the reactive sees no change, so no watcher
runs against children that are already gone and no activation message is
posted. A live bar still validates as upstream does.
Wrapping the method on the class covers ``ContentTabs`` and bars created before
startup patching, and duplicates no upstream body.
"""

from __future__ import annotations

import functools
from typing import Any

_RUNTIME_PATCH_MARKER = "_chrys_pruned_tabs_activation"


def apply_runtime_patch() -> None:
    """Patch ``Tabs.validate_active`` in the current process."""
    try:
        from textual.widgets import Tabs
    except ImportError:
        return

    original = Tabs.validate_active
    if getattr(original, _RUNTIME_PATCH_MARKER, False):
        return

    @functools.wraps(original)
    def validate_active(self: Any, active: str) -> str:
        if self._pruning:
            return self.active
        try:
            return original(self, active)
        except ValueError:
            if _app_is_exiting(self):
                return self.active
            raise

    setattr(validate_active, _RUNTIME_PATCH_MARKER, True)
    Tabs.validate_active = validate_active


def _app_is_exiting(node: Any) -> bool:
    # ``is_attached`` already reads False once the app exits, so ask the app.
    try:
        return bool(node.app._exit)
    except RuntimeError:  # NoActiveAppError: a bar with no app is not exiting
        return False
