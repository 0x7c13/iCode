# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Checkbox widgets with terminal-friendly ASCII state markers."""

from __future__ import annotations

from textual.content import Content
from textual.widgets import Checkbox as TextualCheckbox

CHECKED_MARKER = "[*]"
UNCHECKED_MARKER = "[ ]"


class Checkbox(TextualCheckbox):
    """A Checkbox that avoids using ``X`` as the visual checked marker."""

    @property
    def _button(self) -> Content:
        """Render a stable-width ASCII checkbox marker for the current value."""
        # Textual exposes static BUTTON_* class attributes, but state-specific
        # marker changes currently require overriding this private property.
        # Re-check this implementation when upgrading Textual.
        marker = CHECKED_MARKER if self.value else UNCHECKED_MARKER
        return Content.assemble((marker, self.get_visual_style("toggle--button")))
