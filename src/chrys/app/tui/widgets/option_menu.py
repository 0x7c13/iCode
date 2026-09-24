# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Compact menu layout and width-aware option descriptions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from textwrap import fill

from rich.text import Text
from textual.containers import VerticalGroup
from textual.events import Resize
from textual.widgets import OptionList
from textual.widgets.option_list import Option


class OptionMenu(VerticalGroup):
    DEFAULT_CSS = """
    OptionMenu {
        width: 60; max-width: 90%; height: auto; max-height: 90%;
        background: $surface; border: round $tui-border-primary $border-opacity;
        border-title-align: left; border-title-color: $primary; padding: 0;
        overflow-x: hidden;
    }
    """


@dataclass(frozen=True)
class MenuOption:
    label: str
    description: str = ""
    id: str | None = None
    current: bool = False
    disabled: bool = False
    dim: bool = False

    def render(self, width: int) -> Text:
        title = ("◦ " if self.current else "") + self.label
        content = Text(title, style="dim" if self.current or self.dim else "")
        content.stylize("bold", 0, len(title))
        if self.description:
            content.append(
                "\n" + fill(self.description, max(8, width), initial_indent="  - ", subsequent_indent="    ")
            )
        return content


class MenuOptionList(OptionList):
    DEFAULT_CSS = """
    MenuOptionList {
        height: auto; max-height: 100%; border: none;
        padding: 0 0 0 1; scrollbar-size: 1 1;
    }
    MenuOptionList:focus { border: none; }
    """

    def __init__(self, items: Sequence[MenuOption] = (), *, id: str | None = None) -> None:
        self._items = tuple(items)
        self._description_width = 0
        super().__init__(id=id)
        self.set_items(items)

    def set_items(self, items: Sequence[MenuOption]) -> None:
        self._items = tuple(items)
        self._description_width = self.scrollable_content_region.width or 54
        options: list[Option | None] = []
        for item in items:
            if options:
                options.append(None)
            options.append(
                Option(item.render(self._description_width), id=item.id, disabled=item.disabled or item.current)
            )
        self.set_options(options)

    def on_resize(self, _event: Resize) -> None:
        width = self.scrollable_content_region.width
        if width <= 0 or width == self._description_width:
            return
        self._description_width = width
        for index, item in enumerate(self._items):
            self.replace_option_prompt_at_index(index, item.render(width))
