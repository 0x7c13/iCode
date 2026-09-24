# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Review workflow metadata and the exact source in separate, independently scrollable tabs."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.widgets import Button, Static, TabbedContent, TabPane

from chrys.app.tui.binding_display import CANCEL_BINDING, localized_binding
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.widgets import DialogButtonRow, DialogButtonSpec
from chrys.app.tui.widgets.workflow import text
from chrys.app.tui.widgets.workflow.info import INFO_TAB, WorkflowInfo, WorkflowInfoData
from chrys.app.tui.widgets.workflow.source import workflow_source_syntax
from chrys.foundation.i18n import MessageDef
from chrys.orchestration.workflows.preview import WorkflowInspection, WorkflowPreview

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController


class WorkflowConfirmDialog(BaseDialog[bool]):
    CSS_PATH = "workflow_confirm.tcss"
    BINDINGS: ClassVar[list] = [localized_binding("escape", "cancel", CANCEL_BINDING, show=False, priority=True)]

    def __init__(
        self, preview: WorkflowPreview | WorkflowInspection, *, locale_controller: LocaleController | None = None
    ) -> None:
        super().__init__()
        self.preview = preview
        self.locale_controller = locale_controller

    def finish(self) -> None:
        self.dismiss_when_topmost(False)

    def _label(self, message: MessageDef) -> str:
        return text.render(message.bind(), self.locale_controller)

    def compose(self) -> ComposeResult:
        with Vertical(id="workflow-confirm") as container:
            container.border_title = Text(self._label(text.CONFIRM_TITLE))
            with TabbedContent(id="workflow-confirm-tabs"):
                with (
                    TabPane(Content.from_text(self._label(INFO_TAB), markup=False), id="workflow-confirm-info-tab"),
                    VerticalScroll(id="workflow-confirm-info-scroll"),
                ):
                    yield WorkflowInfo(
                        WorkflowInfoData.from_inspection(self.preview)
                        if isinstance(self.preview, WorkflowInspection)
                        else WorkflowInfoData.from_preview(self.preview),
                        locale_controller=self.locale_controller,
                    )
                with (
                    TabPane(
                        Content.from_text(self._label(text.CODE_VIEW), markup=False), id="workflow-confirm-source-tab"
                    ),
                    VerticalScroll(id="workflow-confirm-source-scroll"),
                ):
                    yield Static(
                        workflow_source_syntax(self.preview.source.source),
                        id="workflow-confirm-source",
                        expand=True,
                    )
            yield DialogButtonRow(
                DialogButtonSpec(Text(self._label(text.CONFIRM)), id="workflow-confirm-yes", variant="warning"),
                DialogButtonSpec(Text(self._label(text.CANCEL)), id="workflow-confirm-no"),
            )

    @on(Button.Pressed, "#workflow-confirm-yes")
    def confirm(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(True)

    @on(Button.Pressed, "#workflow-confirm-no")
    def cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def _default_dismiss_result(self) -> bool:
        return False
