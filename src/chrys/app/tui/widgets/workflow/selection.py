# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow catalog rows and selection widget used by the workflow picker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import OptionList, Static

from chrys.app.tui.binding_display import localized_binding
from chrys.app.tui.widgets.option_menu import MenuOption, MenuOptionList
from chrys.app.tui.widgets.workflow import text
from chrys.service.workflows.discovery import SkippedSource

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController
    from chrys.service.workflows.discovery import WorkflowSource


@dataclass(frozen=True)
class WorkflowRow:
    source: WorkflowSource | SkippedSource
    title: str
    shadowed: bool = False

    @property
    def canonical_path(self) -> str:
        return self.source.path if isinstance(self.source, SkippedSource) else self.source.canonical_path

    @property
    def workflow_id(self) -> str:
        if isinstance(self.source, SkippedSource):
            return self.source.workflow_id or Path(self.source.path).stem
        return self.source.workflow_id

    @property
    def package_folder(self) -> str | None:
        """The folder of a folder workflow, listed or skipped; ``None`` for a single-file workflow."""
        if isinstance(self.source, SkippedSource):
            return self.source.path if self.source.workflow_id == Path(self.source.path).name else None
        return None if self.source.package is None else self.source.package.directory


class WorkflowList(MenuOptionList):
    BINDINGS: ClassVar[list] = [
        localized_binding("j", "cursor_down", text.NEXT_WORKFLOW, show=False),
        localized_binding("k", "cursor_up", text.PREVIOUS_WORKFLOW, show=False),
        localized_binding("d", "delete_workflow", text.DELETE, show=False),
    ]

    class DeleteRequested(Message):
        pass

    def action_delete_workflow(self) -> None:
        self.post_message(self.DeleteRequested())


class WorkflowSelection(Vertical):
    """Catalog list, with selection preserved by the source's canonical path."""

    DEFAULT_CSS = """
    WorkflowSelection { height: auto; max-height: 100%; }
    WorkflowSelection #workflow-warnings { height: auto; max-height: 4; overflow-y: auto; }
    """

    class OpenRequested(Message):
        def __init__(self, workflow_id: str) -> None:
            super().__init__()
            self.workflow_id = workflow_id

    def __init__(
        self, *, locale_controller: LocaleController | None = None, current_source: WorkflowSource | None = None
    ) -> None:
        super().__init__()
        self.locale_controller = locale_controller
        self.current_source = current_source
        self.rows: list[WorkflowRow] = []
        self._warnings = ""
        self.list = WorkflowList(id="workflow-list")
        self._ready = False
        self._warnings_widget = Static(id="workflow-warnings")
        self._warnings_widget.display = False

    def compose(self) -> ComposeResult:
        yield self.list
        yield self._warnings_widget

    def on_mount(self) -> None:
        self._ready = True
        if self.locale_controller is not None:
            self.locale_controller.register_surface(self)
        self.show_rows(self.rows, self._warnings)

    def on_unmount(self) -> None:
        self._ready = False
        if self.locale_controller is not None:
            self.locale_controller.unregister_surface(self)

    def refresh_localization(self) -> None:
        self.show_rows(self.rows, self._warnings, preserve_selection=True)

    def show_rows(self, rows: list[WorkflowRow], warnings: str, *, preserve_selection: bool = False) -> None:
        if not self._ready:
            self.rows, self._warnings = rows, warnings
            return
        picker = self.list
        highlighted = (picker.highlighted or 0) if preserve_selection else 0
        selected = self.selected_row() if preserve_selection else None
        self.rows = rows
        self._warnings = warnings
        options: list[MenuOption] = []
        for row in rows:
            source = text.render(text.SOURCES[row.source.source_kind].bind(), self.locale_controller)
            shadowed = " · " + text.render(text.SHADOWED.bind(), self.locale_controller) if row.shadowed else ""
            current = row.source == self.current_source
            label = text.shown(row.title or row.workflow_id)
            detail = f"{text.shown(row.workflow_id)} · {source}{shadowed}"
            if isinstance(row.source, SkippedSource):
                label = f"⚠ {text.shown(row.workflow_id)} · {source}{shadowed}"
                detail = text.shown(row.source.reason)
            options.append(MenuOption(label, detail, current=current, dim=row.shadowed))
        picker.set_items(options)
        if selected is not None:
            highlighted = next(
                (index for index, row in enumerate(rows) if row.canonical_path == selected.canonical_path), highlighted
            )
        picker.highlighted = min(highlighted, len(rows) - 1) if rows else None
        self._warnings_widget.display = bool(warnings)
        self._warnings_widget.update(Text(text.shown(warnings, block=True)))

    def selected_row(self) -> WorkflowRow | None:
        index = self.list.highlighted
        return self.rows[index] if index is not None and index < len(self.rows) else None

    @on(OptionList.OptionSelected, "#workflow-list")
    def open_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.disabled:
            return
        row = self.rows[event.option_index]
        if not row.shadowed and not isinstance(row.source, SkippedSource):
            self.post_message(self.OpenRequested(row.workflow_id))
