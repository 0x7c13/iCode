# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""TasksPanel — live session todo checklist."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from rich.style import Style
from rich.text import Text
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from chrys.app.tui.i18n import render_str
from chrys.app.tui.util.todo_format import todo_checklist_renderable, todo_done_count
from chrys.app.tui.widgets.sidebar.empty_state import SidebarEmptyStateLabel
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

_TASKS_TITLE = msg("tui.sidebar.tasks.title", fallback="Todos")
_TASKS_COUNTER = msg("tui.sidebar.tasks.counter", fallback="Todos ({done}/{total})")
_TASKS_EMPTY = msg("tui.sidebar.tasks.empty", fallback="No tasks yet")

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController
    from chrys.foundation.models.todos import TodoItem


@dataclass(frozen=True, slots=True)
class TodoListState:
    """View state for the Tasks panel (full-list payload, not a diff)."""

    items: tuple[TodoItem, ...] = ()


class TasksPanel(Widget, can_focus=False):
    """Renders the session todo list with live status markers.

    The checklist lives in a Static (re-rendered at the real content width on
    every layout) rather than a RichLog: RichLog wraps writes at
    ``min_width=78`` and never re-wraps, which cropped long items in the
    narrow sidebar.
    """

    todo_state: reactive[TodoListState | None] = reactive(None, always_update=True)
    _MAX_TODO_SYNC_RETRIES = 3
    COMPONENT_CLASSES: ClassVar[set[str]] = {"tasks--in-progress"}

    DEFAULT_CSS = """
    TasksPanel {
        height: 100%;
        width: 100%;
        padding: 0 1;
    }
    TasksPanel > Static {
        height: auto;
    }
    TasksPanel > .tasks--in-progress {
        color: $warning;
    }
    TasksPanel > .tasks-label {
        text-style: bold;
        color: $text-muted;
    }
    TasksPanel > #tasks-list {
        height: 1fr;
        border: round $tui-border-primary-darken-3 $border-opacity;
        scrollbar-size: 1 1;
        overflow-x: hidden;
        padding: 0 0 0 1;
        display: none;
    }
    TasksPanel #tasks-checklist {
        width: 100%;
        height: auto;
    }
    TasksPanel > #tasks-empty {
        height: 1fr;
        border: round $tui-border-primary-darken-3 $border-opacity;
        color: $text-muted;
        content-align: center middle;
    }
    SidebarPanel.-shell-active TasksPanel > #tasks-list,
    SidebarPanel.-shell-active TasksPanel > #tasks-empty {
        border: round $tui-border-warning-darken-1 $border-opacity;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        locale_controller: LocaleController | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._locale_controller = locale_controller
        self._render_state: TodoListState | None = None
        self._sync_pending = False
        self._sync_retries = 0
        self._theme_style: Style | None = None

    def compose(self) -> ComposeResult:
        yield Static(Text(self._render_message(_TASKS_TITLE.bind())), classes="tasks-label")
        with VerticalScroll(id="tasks-list"):
            yield Static(id="tasks-checklist")
        yield SidebarEmptyStateLabel(Text(self._render_message(_TASKS_EMPTY.bind())), id="tasks-empty")

    def on_mount(self) -> None:
        self.app.theme_changed_signal.subscribe(self, self._theme_colors_changed)
        if self.todo_state is not None:
            self._render_state = self.todo_state
        self._request_todo_sync()

    def on_unmount(self) -> None:
        self.app.theme_changed_signal.unsubscribe(self)

    def _theme_colors_changed(self, _theme: object) -> None:
        if self._in_progress_style() != self._theme_style:
            self._request_todo_sync()

    def _in_progress_style(self) -> Style:
        # Resolve alpha against the live background just as CSS does. Keep only
        # foreground so a Rich row doesn't acquire a painted background rectangle.
        resolved = self.get_component_rich_style("tasks--in-progress", default=Style(color="yellow"))
        return Style(color=resolved.color, bold=True)

    def watch_todo_state(self, state: TodoListState | None) -> None:
        """Apply routed todo state from the screen."""
        if state is None:
            return
        self._render_state = state
        self._sync_retries = 0
        self._sync_todo_state()

    def refresh_localization(self) -> None:
        """Retranslate the current title/counter and empty-state label."""
        if not self.is_mounted:
            return
        self.query_one("#tasks-empty", Static).update(Text(self._render_message(_TASKS_EMPTY.bind())))
        if self._render_state is None:
            self.query_one(".tasks-label", Static).update(Text(self._render_message(_TASKS_TITLE.bind())))
            return
        self._sync_todo_state()

    def _render_message(self, reference: MessageRef) -> str:
        controller = self._locale_controller
        if controller is None:
            return format_message(reference)
        return render_str(controller.localizer, reference)

    def _request_todo_sync(self) -> None:
        if self._sync_pending:
            return
        self._sync_pending = self.call_after_refresh(self._sync_todo_state)

    def _retry_todo_sync(self) -> None:
        if self._sync_retries >= self._MAX_TODO_SYNC_RETRIES:
            return
        self._sync_retries += 1
        self._request_todo_sync()

    def _sync_todo_state(self) -> None:
        self._sync_pending = False
        self._theme_style = self._in_progress_style()
        state = self._render_state
        if state is None:
            return
        try:
            label = self.query_one(".tasks-label", Static)
            scroller = self.query_one("#tasks-list", VerticalScroll)
            checklist = self.query_one("#tasks-checklist", Static)
            empty = self.query_one("#tasks-empty", Static)
        except NoMatches:
            self._retry_todo_sync()
            return
        items = state.items
        empty.update(Text(self._render_message(_TASKS_EMPTY.bind())))
        if not items:
            label.update(Text(self._render_message(_TASKS_TITLE.bind())))
            checklist.update("")
            scroller.display = False
            empty.display = True
            self._sync_retries = 0
            return
        label.update(Text(self._render_message(_TASKS_COUNTER.bind(done=todo_done_count(items), total=len(items)))))
        checklist.update(todo_checklist_renderable(items, in_progress_style=self._theme_style))
        scroller.display = True
        empty.display = False
        self._sync_retries = 0
