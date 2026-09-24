# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SidebarPanel — toggleable tabbed sidebar."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from textual.content import Content
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Tab, TabbedContent, TabPane, Tabs

from chrys.app.tui.i18n import render_content
from chrys.app.tui.widgets.sidebar.buddy import BuddyPanel
from chrys.app.tui.widgets.sidebar.context import ContextPanel, ContextUsageState
from chrys.app.tui.widgets.sidebar.debug import DebugPanel
from chrys.app.tui.widgets.sidebar.tasks import TasksPanel, TodoListState
from chrys.app.tui.widgets.sidebar.toc import ConversationToc
from chrys.foundation.i18n import MessageDef, MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

_SIDEBAR_MESSAGES = msg("tui.sidebar.tab.messages", fallback="Messages")
_SIDEBAR_TASKS = msg("tui.sidebar.tab.tasks", fallback="Tasks")
_SIDEBAR_CONTEXT = msg("tui.sidebar.tab.context", fallback="Context")
_SIDEBAR_DEBUG = msg("tui.sidebar.tab.debug", fallback="Debug")
_SIDEBAR_BUDDY = msg("tui.sidebar.tab.buddy", fallback="Buddy")
_SIDEBAR_TAB_MESSAGES: tuple[MessageDef, ...] = (
    _SIDEBAR_MESSAGES,
    _SIDEBAR_TASKS,
    _SIDEBAR_CONTEXT,
    _SIDEBAR_DEBUG,
    _SIDEBAR_BUDDY,
)
_WORKFLOW_HIDDEN_TABS = ("tab-toc", "tab-tasks", "tab-context")

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController


class SidebarPanel(Widget, can_focus=False):
    """Toggleable sidebar with tabbed panels."""

    context_usage_state: reactive[ContextUsageState | None] = reactive(None, always_update=True)
    todo_state: reactive[TodoListState | None] = reactive(None, always_update=True)

    DEFAULT_CSS = """
    SidebarPanel {
        width: 30%;
        min-width: 42;
        min-height: 3;
        max-width: 60;
        border: round $tui-border-primary $border-opacity;
    }
    SidebarPanel.-shell-active {
        border: round $tui-border-warning $border-opacity;
    }
    SidebarPanel > TabbedContent {
        height: 100%;
    }
    SidebarPanel TabPane {
        height: 1fr;
        padding: 0;
    }
    """

    def __init__(self, *, locale_controller: LocaleController | None = None) -> None:
        super().__init__()
        self._locale_controller = locale_controller
        self._visibility_suppressors: set[str] = set()
        self._display_before_suppression = True
        self._workflow_mode = False
        self._chat_tab = "tab-toc"

    def compose(self) -> ComposeResult:
        with TabbedContent(initial="tab-toc"):
            with TabPane(self._localized_content(_SIDEBAR_MESSAGES.bind()), id="tab-toc"):
                yield ConversationToc(locale_controller=self._locale_controller)
            with TabPane(self._localized_content(_SIDEBAR_TASKS.bind()), id="tab-tasks"):
                yield TasksPanel(locale_controller=self._locale_controller).data_bind(
                    todo_state=SidebarPanel.todo_state
                )
            with TabPane(self._localized_content(_SIDEBAR_CONTEXT.bind()), id="tab-context"):
                yield ContextPanel(locale_controller=self._locale_controller).data_bind(
                    usage_state=SidebarPanel.context_usage_state
                )
            with TabPane(self._localized_content(_SIDEBAR_DEBUG.bind()), id="tab-debug"):
                yield DebugPanel(locale_controller=self._locale_controller)
            with TabPane(self._localized_content(_SIDEBAR_BUDDY.bind()), id="tab-buddy"):
                yield BuddyPanel(locale_controller=self._locale_controller)

    def on_mount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.register_surface(self)
        self.refresh_localization()

    def on_unmount(self) -> None:
        if self._locale_controller is not None:
            self._locale_controller.unregister_surface(self)

    def refresh_localization(self) -> None:
        """Relabel tabs and delegate current-text refreshes to owned panels."""
        if not self.is_mounted:
            return
        tabs = self.query_one(TabbedContent).query_one(Tabs)
        for tab, definition in zip(tabs.query(Tab).results(Tab), _SIDEBAR_TAB_MESSAGES, strict=True):
            tab.label = self._localized_content(definition.bind())
        self.context_panel.refresh_localization()
        self.tasks_panel.refresh_localization()
        self.toc_panel.refresh_localization()
        self.buddy_panel.refresh_localization()
        self.debug_panel.refresh_localization()

    def _localized_content(self, reference: MessageRef) -> Content:
        controller = self._locale_controller
        if controller is None:
            return Content.from_text(format_message(reference), markup=False)
        return render_content(controller.localizer, reference)

    def toggle(self) -> None:
        if not self._visibility_suppressors:
            self.display = not self.display

    def set_workflow_mode(self, workflow: bool) -> None:
        """Keep chat-only panels out of workflow navigation, retaining the chat selection."""
        if workflow == self._workflow_mode:
            return
        self._workflow_mode = workflow
        tabs = self.query_one(TabbedContent)
        if workflow:
            self._chat_tab = tabs.active
            if tabs.active in _WORKFLOW_HIDDEN_TABS:
                tabs.active = "tab-debug"
            for pane in _WORKFLOW_HIDDEN_TABS:
                tabs.hide_tab(pane)
        else:
            for pane in _WORKFLOW_HIDDEN_TABS:
                tabs.show_tab(pane)
            tabs.active = self._chat_tab

    def suppress_visibility(self, owner: Literal["theme-editor", "alternate-screen"], active: bool) -> None:
        """Restore the user's visibility only after every temporary hide ends."""
        if active:
            if owner in self._visibility_suppressors:
                return
            if not self._visibility_suppressors:
                self._display_before_suppression = self.display
            self._visibility_suppressors.add(owner)
            self.display = False
        elif owner in self._visibility_suppressors:
            self._visibility_suppressors.remove(owner)
            if not self._visibility_suppressors:
                self.display = self._display_before_suppression

    @property
    def is_visible(self) -> bool:
        return self.display

    @property
    def debug_panel(self) -> DebugPanel:
        return self.query_one(DebugPanel)

    @property
    def context_panel(self) -> ContextPanel:
        return self.query_one(ContextPanel)

    @property
    def tasks_panel(self) -> TasksPanel:
        return self.query_one(TasksPanel)

    @property
    def toc_panel(self) -> ConversationToc:
        return self.query_one(ConversationToc)

    @property
    def buddy_panel(self) -> BuddyPanel:
        return self.query_one(BuddyPanel)

    def focus_tab(self, tab_id: str) -> None:
        if self._workflow_mode and tab_id in _WORKFLOW_HIDDEN_TABS:
            return
        tc = self.query_one(TabbedContent)
        tc.active = tab_id
