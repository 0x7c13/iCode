# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: ignore stale Textual OptionList click metadata.

Problem
-------
Textual 8.2.5 stores the rendered option index in each cell's style metadata.
If an ``OptionList`` is cleared, hidden, or rebuilt between render and click
dispatch, the click can still carry an old ``option`` index.  The upstream
handler indexes ``self._options`` directly and raises ``IndexError`` when that
metadata is stale.

Solution
--------
Bounds-check the click metadata before reading ``self._options``.  Stale clicks
are ignored; valid clicks keep the original highlight-and-select behavior.

Some Chrys TUI modules are imported before ``bootstrap_runtime()`` applies
site-package patches, so this module also exposes an in-process monkey patch
for already-imported ``OptionList`` classes.
"""

from __future__ import annotations

from typing import Any

from chrys.foundation.patches.patcher import FilePatch, register

_OLD = """\
    async def _on_click(self, event: events.Click) -> None:
        \"\"\"React to the mouse being clicked on an item.

        Args:
            event: The click event.
        \"\"\"
        clicked_option: int | None = event.style.meta.get(\"option\")
        if clicked_option is not None and not self._options[clicked_option].disabled:
            self.highlighted = clicked_option
            self.action_select()"""

_NEW = """\
    async def _on_click(self, event: events.Click) -> None:
        \"\"\"React to the mouse being clicked on an item.

        Args:
            event: The click event.
        \"\"\"
        clicked_option: int | None = event.style.meta.get(\"option\")
        if clicked_option is None or clicked_option < 0 or clicked_option >= len(self._options):
            return
        if not self._options[clicked_option].disabled:
            self.highlighted = clicked_option
            self.action_select()"""

_RUNTIME_PATCH_MARKER = "_chrys_safe_option_list_click"


def apply_runtime_patch() -> None:
    """Patch ``OptionList._on_click`` in the current process."""
    try:
        from textual.widgets._option_list import OptionList
    except ImportError:
        return

    if getattr(OptionList._on_click, _RUNTIME_PATCH_MARKER, False):
        return

    async def _on_click(self: Any, event: Any) -> None:
        clicked_option = event.style.meta.get("option")
        if clicked_option is None or clicked_option < 0 or clicked_option >= len(self._options):
            return
        if not self._options[clicked_option].disabled:
            self.highlighted = clicked_option
            self.action_select()

    setattr(_on_click, _RUNTIME_PATCH_MARKER, True)
    OptionList._on_click = _on_click


register(
    FilePatch(
        package="textual",
        module_file="widgets/_option_list.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Ignore stale OptionList click metadata",
    ),
)
