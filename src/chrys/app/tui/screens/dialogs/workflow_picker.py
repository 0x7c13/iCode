# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow selection modal, matching the agent picker."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on

from chrys.app.tui.binding_display import CLOSE_BINDING, localized_binding
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.widgets.option_menu import OptionMenu
from chrys.app.tui.widgets.workflow import text
from chrys.app.tui.widgets.workflow.selection import WorkflowList, WorkflowSelection

if TYPE_CHECKING:
    from textual.app import ComposeResult


class WorkflowPickerDialog(BaseDialog[str | None]):
    DEFAULT_CSS = "WorkflowPickerDialog { align: center middle; }"

    BINDINGS: ClassVar[list] = [localized_binding("escape", "dismiss", CLOSE_BINDING)]

    def __init__(self, selection: WorkflowSelection, *, delete: Callable[[], None]) -> None:
        super().__init__()
        self.selection = selection
        self._delete = delete

    def compose(self) -> ComposeResult:
        with OptionMenu(id="container") as container:
            container.border_title = Text(text.render(text.TITLE.bind(), self.selection.locale_controller))
            yield self.selection

    @on(WorkflowSelection.OpenRequested)
    def open_selected(self, event: WorkflowSelection.OpenRequested) -> None:
        event.stop()
        self.dismiss(event.workflow_id)

    @on(WorkflowList.DeleteRequested)
    def delete_selected(self, event: WorkflowList.DeleteRequested) -> None:
        event.stop()
        self._delete()
