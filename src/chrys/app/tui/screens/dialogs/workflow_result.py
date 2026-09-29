# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A finished run's final outputs in a screen-local modal: one output fills it, several get a tab each."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import work
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import TabbedContent, TabPane

from chrys.app.tui.binding_display import CLOSE_BINDING, localized_binding
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.widgets.workflow import text
from chrys.app.tui.widgets.workflow.records import stored_output_value
from chrys.app.tui.widgets.workflow.values import ShownValue, ValueDocument, WorkflowValueView, shown_value
from chrys.foundation.i18n.formatting import sanitize_legacy_block, sanitize_legacy_scalar
from chrys.service.workflows.artifacts import read_node_output

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController
    from chrys.foundation.events.types import WorkflowOutputSummary


class WorkflowResultDialog(BaseDialog[None]):
    """Each output's full value, read from the run store once the dialog has painted.

    Like the Output tab and the CLI's text mode, an output shows its text; one without text shows its data.
    """

    CSS_PATH = "workflow_result.tcss"
    BINDINGS: ClassVar[list] = [localized_binding("escape", "close", CLOSE_BINDING, show=False, priority=True)]

    def __init__(
        self,
        outputs: tuple[WorkflowOutputSummary, ...],
        *,
        kinds: Mapping[str, str],
        directory: Path | None,
        outcome: str,
        locale_controller: LocaleController | None = None,
    ) -> None:
        if not outputs:
            raise ValueError("A workflow result dialog needs at least one output.")
        super().__init__()
        self.outputs = outputs
        self.kinds = dict(kinds)
        self.directory = directory
        self.outcome = outcome
        self.locale_controller = locale_controller

    def compose(self) -> ComposeResult:
        names = [sanitize_legacy_scalar(output.node_id) for output in self.outputs]
        with Vertical(id="workflow-result-frame") as container:
            title = text.RESULT_TITLE.bind(node=names[0]) if len(names) == 1 else text.RESULT.bind()
            container.border_title = Text(text.render(title, self.locale_controller))
            container.border_subtitle = Text(
                sanitize_legacy_scalar(text.state_label(self.outcome, self.locale_controller))
            )
            if len(names) == 1:
                yield self._view(0)
                return
            with TabbedContent():
                for index, name in enumerate(names):
                    # Node ids may hold characters a widget id rejects.
                    with TabPane(Content.from_text(name, markup=False), id=f"workflow-result-tab-{index}"):
                        yield self._view(index)

    def _view(self, index: int) -> WorkflowValueView:
        return WorkflowValueView(self.locale_controller, scroll=True, view_tabs=False, id=f"workflow-result-{index}")

    def on_mount(self) -> None:
        self.call_after_refresh(self.load_outputs)

    @work(exclusive=True, group="workflow-result")
    async def load_outputs(self) -> None:
        documents = await asyncio.to_thread(self._read)
        for index, document in enumerate(documents):
            if not self.is_mounted or self._torn_down():
                return
            await self.query_one(f"#workflow-result-{index}", WorkflowValueView).show(document)

    def _read(self) -> list[ValueDocument]:
        documents: list[ValueDocument] = []
        for output in self.outputs:
            summary = (ShownValue(output.summary_text),) if output.summary_text else ()
            try:
                record = (
                    read_node_output(
                        self.directory,
                        output.activation_id,
                        output.attempt,
                        node_kind=self.kinds.get(output.node_id, ""),
                    )
                    if self.directory is not None
                    else None
                )
                value = None if record is None else stored_output_value(record)
            except (OSError, ValueError) as exc:
                # A damaged result must not hide the other outputs; the live summary still shows under the error.
                error = sanitize_legacy_block(str(exc))
                documents.append(ValueDocument(summary, notice=error) if summary else ValueDocument(placeholder=error))
                continue
            if value is not None:
                documents.append(ValueDocument((shown_value(value),)))
            elif summary:
                documents.append(
                    ValueDocument(summary, notice=text.render(text.OUTPUT_SUMMARY_ONLY.bind(), self.locale_controller))
                )
            else:
                # Archived runs keep no summaries.
                documents.append(ValueDocument(placeholder=text.render(text.NO_RECORD.bind(), self.locale_controller)))
        return documents

    def _torn_down(self) -> bool:
        """App shutdown and removal both start, and drop new mounts, before the screen is detached."""
        return not self.is_attached or not self.app.is_running or self._closing or self._pruning

    def _before_dismiss(self, _result: object | None = None) -> None:
        """Cancel the read's continuation before the asynchronous pop removes its widgets."""
        if self.is_attached:
            self.workers.cancel_group(self, "workflow-result")
        super()._before_dismiss(_result)

    def action_close(self) -> None:
        if self.app.screen is self:
            self.dismiss(None)
