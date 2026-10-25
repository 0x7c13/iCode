# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""One display writer for entries with independent content and process visibility."""

from __future__ import annotations

from textual.widget import Widget


class ProcessMember(Widget):
    """A process may hide an entry without overwriting its own visibility."""

    _process_hidden: bool = False
    _content_visible: bool = True

    @property
    def process_hidden(self) -> bool:
        return self._process_hidden

    def set_process_hidden(self, hidden: bool) -> None:
        self._process_hidden = hidden
        self._sync_process_display()

    def set_content_visible(self, visible: bool) -> None:
        self._content_visible = visible
        self._sync_process_display()

    def _sync_process_display(self) -> None:
        visible = self._content_visible and not self._process_hidden
        if self.display != visible:
            self.display = visible
