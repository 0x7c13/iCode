# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared helpers for TUI tests."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING

from PIL import Image
from rich.console import Console
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.errors import NoWidget
from textual.events import Click
from textual.message import Message
from textual.messages import Prune
from textual.widget import Widget
from textual.widgets import Static

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.main.commands import SlashCommandActions
from chrys.app.tui.screens.main.diff_controller import LiveDiffTracker
from chrys.app.tui.screens.main.event_handlers import BackendEventCallbacks, BackendEventHandler
from chrys.app.tui.screens.main.live_diff import LiveFileMutation
from chrys.app.tui.screens.main.session_handlers import SessionCallbacks, SessionHandler
from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.app.tui.screens.main.suggestions import SuggestionCallbacks, SuggestionHandler
from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter
from chrys.app.tui.support.gc_freeze import GcAbsorbRequested, GcReclaimRequested
from chrys.app.tui.widgets.chat.agent_transcript_surface import AgentTranscriptSurface
from chrys.app.tui.widgets.chat.panel import ChatPanel, _ChatBottomSpacer, _ScrollToBottomButton
from chrys.app.tui.widgets.chat.ports import TranscriptLocalizationPort
from chrys.app.tui.widgets.chat.renderers.sub_agent import SubAgentToolCall
from chrys.app.tui.widgets.chat.tool_call import BaseToolCard, ToolCardHeader, ToolGroup
from chrys.app.tui.widgets.chrome.file_scanner import ProjectPathScanResult, ProjectPathSuggestion
from chrys.app.tui.widgets.chrome.suggestion_list import SuggestionItem
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import AgentRuntimeDetails
from chrys.foundation.i18n import MessageRef
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.models.ask_user import AskUserAnswer
from chrys.service.approval.policy import ApprovalMode
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    import pytest
    from rich.console import RenderableType
    from rich.segment import Segment
    from textual.pilot import Pilot
    from textual.screen import Screen


def install_trajectory_dashboard_query(screen: object) -> None:
    """Add the dashboard surface expected by the real main-screen view adapter."""
    original_query_one = getattr(screen, "query_one", None)
    if original_query_one is None:
        return
    dashboard = type("_TrajectoryDashboardFake", (), {"foreground": False})()

    def query_one(cls: type):
        if cls.__name__ == "TrajectoryDashboard":
            return dashboard
        return original_query_one(cls)

    screen.query_one = query_one


def make_backend_handler(screen: object, *, locale_controller: LocaleController | None = None) -> BackendEventHandler:
    """Construct a backend event handler around a lightweight screen fake."""
    install_trajectory_dashboard_query(screen)
    state = getattr(screen, "_state", None)
    if not isinstance(state, MainScreenState):
        state = MainScreenState()
    state.run.agent_running = bool(getattr(screen, "_agent_running", state.run.agent_running))
    state.run.agent_loading = bool(getattr(screen, "_agent_loading", state.run.agent_loading))
    state.run.has_messages = bool(getattr(screen, "_has_messages", state.run.has_messages))
    state.session.creating_new_session = bool(
        getattr(screen, "_creating_new_session", state.session.creating_new_session)
    )
    state.session.restoring_session = bool(getattr(screen, "_restoring_session", state.session.restoring_session))
    state.runtime.profile = str(getattr(screen, "_profile", state.runtime.profile))
    state.runtime.details = getattr(screen, "_runtime_details", state.runtime.details)
    state.runtime.main_usage_source_id = str(
        getattr(screen, "_main_usage_source_id", state.runtime.main_usage_source_id)
    )
    state.usage.last_usage_tokens = int(getattr(screen, "_last_usage_tokens", state.usage.last_usage_tokens))
    state.usage.last_total_session_tokens = int(
        getattr(screen, "_last_total_session_tokens", state.usage.last_total_session_tokens)
    )
    state.workspace_marker.original_cwd = getattr(screen, "_chdir_original_cwd", state.workspace_marker.original_cwd)
    state.workspace_marker.current_cwd = str(getattr(screen, "_chdir_current_cwd", state.workspace_marker.current_cwd))
    state.workspace.current_cwd = state.workspace_marker.current_cwd
    state.submit.active = bool(getattr(screen, "_pending_user_submit_active", state.submit.active))
    state.submit.text = str(getattr(screen, "_pending_user_submit_text", state.submit.text))
    state.submit.blocked = bool(getattr(screen, "_pending_user_submit_blocked", state.submit.blocked))
    state.render_gate.active = bool(getattr(screen, "_pending_user_message_render_active", state.render_gate.active))
    screen.context_usage_state = getattr(screen, "context_usage_state", None)

    services = MainScreenServices(
        bus=getattr(screen, "_bus", EventBus()),
        state_store=getattr(screen, "_state_store", None),
        agent_registry=getattr(screen, "_agent_registry", None),
        model_registry=getattr(screen, "_model_registry", None),
        active_model_profile_id=str(getattr(screen, "_active_model_profile_id", "")),
    )
    live_call_paths = getattr(screen, "_live_call_paths", None)
    live_file_mutations = getattr(screen, "_live_file_mutations", None)
    live_diff = LiveDiffTracker(
        call_paths=live_call_paths if isinstance(live_call_paths, MutableMapping) else None,
        file_mutations=live_file_mutations if isinstance(live_file_mutations, MutableMapping) else None,
    )

    def set_agent_running(value: bool) -> None:
        state.run.agent_running = value
        setter = getattr(screen, "_set_agent_running", None)
        if callable(setter):
            setter(value)
        else:
            screen._agent_running = value

    def set_agent_loading(value: bool) -> None:
        state.run.agent_loading = value
        setter = getattr(screen, "_set_agent_loading", None)
        if callable(setter):
            setter(value)
        else:
            screen._agent_loading = value

    def set_has_messages(value: bool) -> None:
        state.run.has_messages = value
        setter = getattr(screen, "_set_has_messages", None)
        if callable(setter):
            setter(value)
        else:
            screen._has_messages = value

    def set_profile_display(value: str) -> None:
        state.runtime.profile = value
        screen._profile = value

    def set_runtime_details(value: object) -> None:
        state.runtime.details = value
        screen._runtime_details = value

    def set_active_model_profile_id(value: str) -> None:
        services.active_model_profile_id = value
        screen._active_model_profile_id = value

    def set_main_usage_source_id(value: str) -> None:
        state.runtime.main_usage_source_id = value
        screen._main_usage_source_id = value

    def set_last_usage_tokens(value: int) -> None:
        state.usage.last_usage_tokens = value
        screen._last_usage_tokens = value

    def set_last_total_session_tokens(value: int) -> None:
        state.usage.last_total_session_tokens = value
        screen._last_total_session_tokens = value

    def set_creating_new_session(value: bool) -> None:
        state.session.creating_new_session = value
        screen._creating_new_session = value

    def set_restoring_session(value: bool) -> None:
        state.session.restoring_session = value
        screen._restoring_session = value

    def set_workspace_cwd(value: str) -> None:
        state.workspace.current_cwd = value
        state.workspace_marker.current_cwd = value
        screen._chdir_current_cwd = value

    def set_workspace_original_cwd(value: str | None) -> None:
        state.workspace_marker.original_cwd = value
        screen._chdir_original_cwd = value

    def update_subtitle() -> None:
        updater = getattr(screen, "_update_subtitle", None)
        if callable(updater):
            updater()

    def update_toc() -> None:
        updater = getattr(screen, "_update_toc", None)
        if callable(updater):
            updater()

    def on_session_fork_error(event: object, message: str, severity: str) -> None:
        sessions = getattr(screen, "_sessions", None)
        if sessions is not None:
            sessions.on_session_fork_error(event, message=message, severity=severity)

    def on_session_clear_error(event: object, message: str) -> None:
        sessions = getattr(screen, "_sessions", None)
        if sessions is not None:
            sessions.on_session_clear_error(event, message=message)

    def block_pending_user_submit() -> None:
        state.submit.block()
        screen._pending_user_submit_blocked = True

    def debug(key: str, message: str = "") -> None:
        debugger = getattr(screen, "_debug", None)
        if callable(debugger):
            debugger(key, message)

    def handle_approval_response(
        request_id: str,
        approved: bool,
        reason: str,
        modified_args: dict[str, object] | None,
    ) -> object | None:
        handler = getattr(screen, "_handle_approval_response", None)
        if callable(handler):
            return handler(request_id, approved, reason, modified_args)
        return None

    def handle_ask_user_response(request_id: str, answers: tuple[AskUserAnswer, ...]) -> None:
        handler = getattr(screen, "_handle_ask_user_response", None)
        if callable(handler):
            handler(request_id, answers)

    def post_gc_message(message: object) -> None:
        messages = getattr(screen, "_gc_messages", None)
        if isinstance(messages, list):
            messages.append(message)

    def refresh_notification_settings() -> None:
        refresher = getattr(screen, "_refresh_notification_settings", None)
        if callable(refresher):
            refresher()

    def refresh_trajectory_verify_commands() -> None:
        refresher = getattr(screen, "_refresh_trajectory_verify_commands", None)
        if callable(refresher):
            refresher()

    return BackendEventHandler(
        state=state,
        services=services,
        locale_controller=locale_controller,
        view=MainScreenViewAdapter(
            screen,  # type: ignore[arg-type]
            state_store=services.state_store,
            locale_controller=locale_controller,
        ),
        callbacks=BackendEventCallbacks(
            set_agent_running=set_agent_running,
            set_agent_loading=set_agent_loading,
            set_has_messages=set_has_messages,
            set_profile_display=set_profile_display,
            set_runtime_details=set_runtime_details,
            set_active_model_profile_id=set_active_model_profile_id,
            set_main_usage_source_id=set_main_usage_source_id,
            set_last_usage_tokens=set_last_usage_tokens,
            set_last_total_session_tokens=set_last_total_session_tokens,
            set_creating_new_session=set_creating_new_session,
            set_restoring_session=set_restoring_session,
            set_workspace_cwd=set_workspace_cwd,
            set_workspace_original_cwd=set_workspace_original_cwd,
            refresh_git_branch=lambda: None,
            update_subtitle=update_subtitle,
            update_toc=update_toc,
            on_session_fork_error=on_session_fork_error,
            on_session_clear_error=on_session_clear_error,
            block_pending_user_submit=block_pending_user_submit,
            handle_approval_response=handle_approval_response,
            handle_ask_user_response=handle_ask_user_response,
            question_inline_preferred=lambda: False,
            post_gc_message=post_gc_message,
            debug=debug,
            refresh_model_indicator=lambda: None,
            refresh_notification_settings=refresh_notification_settings,
            refresh_trajectory_verify_commands=refresh_trajectory_verify_commands,
            settings_reloaded=lambda: None,
        ),
        live_diff=live_diff,
    )


def make_session_handler(
    screen: object,
    *,
    locale_controller: LocaleController | None = None,
) -> SessionHandler:
    install_trajectory_dashboard_query(screen)
    state = getattr(screen, "_state", None)
    if not isinstance(state, MainScreenState):
        state = MainScreenState()
    state.run.agent_running = bool(getattr(screen, "_agent_running", state.run.agent_running))
    state.run.agent_loading = bool(getattr(screen, "_agent_loading", state.run.agent_loading))
    state.run.has_messages = bool(getattr(screen, "_has_messages", state.run.has_messages))
    state.session.restoring_session = bool(getattr(screen, "_restoring_session", state.session.restoring_session))
    state.session.creating_new_session = bool(
        getattr(screen, "_creating_new_session", state.session.creating_new_session)
    )
    state.submit.active = bool(getattr(screen, "_pending_user_submit_active", state.submit.active))
    state.runtime.profile = str(getattr(screen, "_profile", state.runtime.profile))
    state.runtime.details = getattr(screen, "_runtime_details", state.runtime.details)
    state.usage.last_usage_tokens = int(getattr(screen, "_last_usage_tokens", state.usage.last_usage_tokens))
    state.usage.last_total_session_tokens = int(
        getattr(screen, "_last_total_session_tokens", state.usage.last_total_session_tokens)
    )
    state.workspace_marker.original_cwd = getattr(screen, "_chdir_original_cwd", state.workspace_marker.original_cwd)
    current_cwd = getattr(screen, "_chdir_current_cwd", state.workspace_marker.current_cwd)
    workspace_cwd = getattr(screen, "_workspace_cwd", None)
    if current_cwd == state.workspace_marker.current_cwd and callable(workspace_cwd):
        current_cwd = workspace_cwd()
    state.workspace_marker.current_cwd = str(current_cwd)
    state.workspace.current_cwd = state.workspace_marker.current_cwd

    services = MainScreenServices(
        bus=getattr(screen, "_bus", EventBus()),
        state_store=getattr(screen, "_state_store", None),
        active_model_profile_id=str(getattr(screen, "_active_model_profile_id", "")),
        apply_saved_model_on_restore=bool(getattr(screen, "_apply_saved_model_on_restore", True)),
    )

    def set_agent_loading(value: bool) -> None:
        state.run.agent_loading = value
        setter = getattr(screen, "_set_agent_loading", None)
        if callable(setter):
            setter(value)
        else:
            screen._agent_loading = value

    def set_has_messages(value: bool) -> None:
        state.run.has_messages = value
        setter = getattr(screen, "_set_has_messages", None)
        if callable(setter):
            setter(value)
        else:
            screen._has_messages = value

    def set_creating_new_session(value: bool) -> None:
        state.session.creating_new_session = value
        screen._creating_new_session = value

    def set_restoring_session(value: bool) -> None:
        state.session.restoring_session = value
        screen._restoring_session = value

    def set_profile_display(value: str) -> None:
        state.runtime.profile = value
        screen._profile = value

    def set_active_model_profile_id(value: str) -> None:
        services.active_model_profile_id = value
        screen._active_model_profile_id = value

    def set_workspace_cwd(value: str) -> None:
        state.workspace.current_cwd = value
        state.workspace_marker.current_cwd = value
        screen._chdir_current_cwd = value

    def set_workspace_original_cwd(value: str | None) -> None:
        state.workspace_marker.original_cwd = value
        screen._chdir_original_cwd = value

    def update_subtitle() -> None:
        updater = getattr(screen, "_update_subtitle", None)
        if callable(updater):
            updater()

    def update_toc() -> None:
        updater = getattr(screen, "_update_toc", None)
        if callable(updater):
            updater()

    def clear_suggestion_file_cache() -> None:
        suggestions = getattr(screen, "_suggestions", None)
        if suggestions is not None:
            suggestions.file_cache = None

    def start_session_restore(session_id: str) -> object | None:
        restorer = getattr(screen, "_do_session_restore", None)
        if callable(restorer):
            return restorer(session_id)
        return None

    def debug(key: str, message: str = "") -> None:
        debugger = getattr(screen, "_debug", None)
        if callable(debugger):
            debugger(key, message)

    def post_gc_message(message: object) -> None:
        messages = getattr(screen, "_gc_messages", None)
        if isinstance(messages, list):
            messages.append(message)

    class _AgentLoadPort:
        async def begin_session_restore_load(self, session_id: str) -> None:
            events = getattr(screen, "_events", None)
            handler = getattr(events, "begin_session_restore_load", None)
            if callable(handler):
                result = handler(session_id)
                if inspect.isawaitable(result):
                    await result

        def cancel_agent_load(self) -> None:
            events = getattr(screen, "_events", None)
            handler = getattr(events, "cancel_agent_load", None)
            if callable(handler):
                handler()

        def finish_agent_load(self, message: str = "") -> None:
            events = getattr(screen, "_events", None)
            handler = getattr(events, "finish_agent_load", None)
            if callable(handler):
                handler(message)

    return SessionHandler(
        state=state,
        services=services,
        view=MainScreenViewAdapter(screen, state_store=services.state_store),  # type: ignore[arg-type]
        callbacks=SessionCallbacks(
            set_agent_loading=set_agent_loading,
            set_has_messages=set_has_messages,
            set_creating_new_session=set_creating_new_session,
            set_restoring_session=set_restoring_session,
            set_profile_display=set_profile_display,
            set_active_model_profile_id=set_active_model_profile_id,
            set_workspace_cwd=set_workspace_cwd,
            set_workspace_original_cwd=set_workspace_original_cwd,
            update_subtitle=update_subtitle,
            update_toc=update_toc,
            clear_suggestion_file_cache=clear_suggestion_file_cache,
            start_session_restore=start_session_restore,
            post_gc_message=post_gc_message,
            debug=debug,
            refresh_model_indicator=lambda: None,
        ),
        agent_load=_AgentLoadPort(),
        locale_controller=locale_controller,
    )


def status_text(value: MessageRef | str) -> str:
    """Render a status ``MessageRef`` (or plain string) the way the UI would."""
    return value if isinstance(value, str) else format_message(value)


def status_trail(value: MessageRef | str | tuple[MessageRef | str, ...]) -> str:
    """Render a status trail — a ``MessageRef``/string or a tuple of them."""
    if isinstance(value, tuple):
        return " · ".join(status_text(part) for part in value)
    return status_text(value)


def stale_file_cache(*paths: str) -> list[ProjectPathSuggestion]:
    """Build a suggestion file cache standing in for a prior ``@`` scan."""
    return [ProjectPathSuggestion(path=path, kind="file") for path in paths]


def make_live_mutation(
    before_text: str,
    after_text: str,
    operation: str,
    *,
    bytes_changed: bool = True,
    before_hash: str | None = None,
    after_hash: str | None = None,
    source: str = "",
) -> LiveFileMutation:
    return LiveFileMutation(
        before_text=before_text,
        after_text=after_text,
        operation=operation,
        bytes_changed=bytes_changed,
        before_hash=before_hash,
        after_hash=after_hash,
        source=source,
    )


def pending_submit_defaults() -> dict[str, object]:
    """Screen attributes describing an idle pending-submit state."""
    return {
        "_agent_running": False,
        "_pending_user_submit_active": False,
        "_pending_user_submit_text": "",
        "_pending_user_submit_blocked": False,
        "_pending_user_message_render_active": False,
        "_deferred_agent_messages": [],
    }


class BusyWidget(Static):
    """A widget that stays inside a message handler from ``hold()`` until ``release`` is set.

    Mounted inside a widget that a rebuild removes, it keeps that removal waiting: a removed widget
    waits for its children to exit before it detaches. Waits while it holds must not go through
    Pilot, which waits for every message loop to go idle.
    """

    class Hold(Message):
        pass

    def __init__(self) -> None:
        super().__init__("busy")
        self.holding = False
        self.exit_requested = False
        """Set once a removed parent asks this widget to exit; that parent then waits for it."""
        self.release = asyncio.Event()

    def hold(self) -> None:
        self.post_message(self.Hold())

    def post_message(self, message: Message) -> bool:
        if isinstance(message, Prune):
            self.exit_requested = True
        return super().post_message(message)

    @on(Hold)
    async def _hold(self, _: Hold) -> None:
        self.holding = True
        await self.release.wait()


async def interrupt_removal(busy: BusyWidget, begin: Callable[[], object], interrupt: Callable[[], object]) -> None:
    """Hold *busy*, let *begin* start a removal that waits for it, and call *interrupt* while it waits."""
    busy.hold()
    try:
        await wait_for(lambda: busy.holding, description="the busy widget holds its message loop")
        begin()
        await wait_for(lambda: busy.exit_requested, description="a removal waits for the busy widget")
        interrupt()
    finally:
        busy.release.set()


async def assert_app_handles_messages(app: App) -> None:
    """Fail unless the App's message loop still runs."""
    handled = asyncio.Event()
    assert app.call_later(handled.set)
    await wait_for(handled.is_set, description="the App's message loop handles a new message")


class FakeNotificationService:
    """Mock ``App.notification_service`` that records delivered events."""

    def __init__(self) -> None:
        self.events: list[object] = []

    def notify(self, event: object) -> bool:
        self.events.append(event)
        return True


def rich_segment_lines(renderable: RenderableType, *, width: int = 200) -> list[list[Segment]]:
    """Render a Rich renderable (for example a ``Static.content``) at *width*.

    Layout independent: the widget need not be mounted or sized, so live and
    replayed cards can be compared on identical footing.
    """
    console = Console(width=width, force_terminal=False, legacy_windows=False)
    return console.render_lines(renderable, console.options.update_width(width), pad=False)


def rich_plain(renderable: RenderableType, *, width: int = 200) -> str:
    """Plain text of :func:`rich_segment_lines`, each line right-trimmed."""
    return "\n".join(
        "".join(segment.text for segment in line).rstrip() for line in rich_segment_lines(renderable, width=width)
    )


# ---------------------------------------------------------------------------
# Widget harnesses — minimal apps that mount one widget under test
# ---------------------------------------------------------------------------


class LocalizedApp(App):
    """Bare app carrying an English locale controller for widget tests."""

    locale_controller = LocaleController(Settings(locale="en"))


class WidgetApp(App):
    """Mount whatever *factory* returns as the whole app.

    Replaces the one-off ``class _App(App): def compose(self): yield X(...)``
    harness that every widget test used to declare inline; the factory runs at
    compose time, so it may close over widgets built earlier in the test.
    """

    def __init__(self, factory: Callable[[], Widget | Iterable[Widget]]) -> None:
        self._factory = factory
        super().__init__()

    def compose(self) -> ComposeResult:
        produced = self._factory()
        if isinstance(produced, Widget):
            yield produced
        else:
            yield from produced


class LocalizedWidgetApp(WidgetApp):
    """:class:`WidgetApp` with the English locale controller of :class:`LocalizedApp`."""

    locale_controller = LocaleController(Settings(locale="en"))


class PlaceholderHostApp(App):
    """A bare English-locale host app for a screen or dialog pushed onto it.

    Its own composed widget is an inert placeholder: the subject under test is
    whatever the caller pushes afterwards. Unlike :class:`LocalizedWidgetApp`,
    each instance builds its own :class:`LocaleController` rather than sharing
    one at class level, so a test that leaves a locale-aware surface registered
    cannot reach into the next test's app through a shared controller.
    """

    def __init__(self) -> None:
        self.locale_controller = LocaleController(Settings(locale="en"))
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Static("placeholder")


class ChatPanelApp(LocalizedApp):
    """App whose only widget is a :class:`ChatPanel` (optionally with a localization port)."""

    def __init__(self, localization: TranscriptLocalizationPort | None = None) -> None:
        self._localization = localization
        super().__init__()

    def compose(self) -> ComposeResult:
        yield ChatPanel(localization=self._localization)


class GcMessageChatPanelApp(ChatPanelApp):
    """:class:`ChatPanelApp` that records the GC absorb/reclaim requests it receives."""

    def __init__(self) -> None:
        self.gc_messages: list[GcAbsorbRequested | GcReclaimRequested] = []
        super().__init__()

    def on_gc_absorb_requested(self, message: GcAbsorbRequested) -> None:
        self.gc_messages.append(message)

    def on_gc_reclaim_requested(self, message: GcReclaimRequested) -> None:
        self.gc_messages.append(message)


def chat_content_children(panel: ChatPanel) -> list[object]:
    """Return transcript children, excluding persistent chat-panel infrastructure."""
    return [child for child in panel.children if not isinstance(child, (_ChatBottomSpacer, _ScrollToBottomButton))]


def inline_diff_mount_state(card: Widget) -> str:
    """Describe a file tool card's deferred-diff machinery for wait diagnostics."""
    timer = getattr(card, "_diff_timer", None)
    task = getattr(timer, "_task", None)
    if timer is None:
        timer_state = "none"
    elif task is None:
        timer_state = "stopped"
    elif not task.done():
        timer_state = "pending"
    elif task.cancelled():
        timer_state = "cancelled"
    else:
        timer_state = f"done exc={task.exception()!r}"
    try:
        content = card.query_one("#ft-content")
    except Exception:
        children: list[str] | str = "no-content"
    else:
        children = [
            f"{type(child).__name__}(pruning={getattr(child, '_pruning', False)})" for child in content.children
        ]
    return (
        f"status={getattr(card, 'status', None)!r} pending={getattr(card, '_diff_pending', None)} "
        f"mounting={getattr(card, '_diff_mounting', None)} timer={timer_state} "
        f"done_class={card.has_class('-done')} attached={card.is_attached} content={children}"
    )


def exec_panel_text(tool: Widget) -> str:
    """Plain text currently shown in a shell tool card's ``#exec-panel``."""
    content = tool.query_one("#exec-panel", Static).content
    return content.plain if isinstance(content, Text) else str(content)


def png_bytes(color: tuple[int, int, int], *, size: tuple[int, int] = (32, 20)) -> bytes:
    """Encode a solid-*color* PNG of *size* for image-preview tests."""
    image = Image.new("RGB", size, color)
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def make_click(
    widget: Widget, *, x: int = 0, y: int = 0, screen_x: int | None = None, screen_y: int | None = None
) -> Click:
    """Build a plain left-button :class:`Click` at (*x*, *y*) inside *widget*.

    Screen coordinates default to the widget's region offset by (*x*, *y*);
    pass them explicitly to aim at a child widget's region instead.
    """
    return Click(
        widget,
        x=x,
        y=y,
        delta_x=0,
        delta_y=0,
        button=1,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=widget.region.x + x if screen_x is None else screen_x,
        screen_y=widget.region.y + y if screen_y is None else screen_y,
    )


def click_widget(widget: Widget) -> Click:
    """Deliver a click at the widget's origin straight to its ``on_click`` and return the event."""
    event = make_click(widget)
    widget.on_click(event)
    return event


def header_zone_click(header: ToolCardHeader, zone: str) -> Click:
    """Synthesize a click on a revealed ToolCardHeader action zone."""
    actions_width = header._actions_width()
    if not header.copy_action_visible and zone == "view":
        start, end = 1, actions_width - 1
    else:
        zones = {name: (start, end) for name, start, end in ToolCardHeader._ACTION_ZONES}
        start, end = zones[zone]
    x = header.content_size.width - actions_width + (start + end) // 2
    event = make_click(header, x=x, screen_x=header.region.x + x, screen_y=header.region.y)
    header.on_click(event)
    return event


def click_copy_button(header: ToolCardHeader) -> None:
    """Click the copy zone of a tool card header."""
    header_zone_click(header, "copy")


async def click_when_settled(
    pilot: Pilot, target: Widget | str, *, offset: tuple[int, int] = (0, 0), **modifiers: bool
) -> None:
    """Click *target* where the current layout puts it, and require the click to land on it.

    Pilot reads the target's region before it pumps a single message. A reflow that is still
    pending then moves the widget out from under the click: a row that has just appeared above
    it, or a modal that has not been laid out yet, and the click lands on a neighbour or is
    reported out of bounds. So the geometry is only trusted straight after a refresh of the
    screen that owns it, requested once every widget has handed its layout request to that
    screen, and it is looked at again after another refresh while the target cannot be hit.

    What the click sets off (a ``Pressed`` message, a dismissal) is only drained when the Pilot
    wait is the settled one from ``tests/support/pilot_barrier.py``; wait for the state it produces.
    """
    app = pilot.app
    refresh: tuple[Screen, asyncio.Event] | None = None
    found: list[Widget] = []

    def takes_the_click() -> bool:
        nonlocal refresh
        screen = app.screen
        if refresh is not None and refresh[0] is screen:
            if not refresh[1].is_set():
                return False
            widget = target if isinstance(target, Widget) else next(iter(screen.query(target)), None)
            if widget is not None and widget.region.area:
                x, y = widget.region.offset + offset
                with contextlib.suppress(NoWidget):
                    if screen.region.contains(x, y) and app.get_widget_at(x, y)[0] is widget:
                        found[:] = [widget]
                        return True
        refreshed = asyncio.Event()
        refresh = (screen, refreshed)
        screen.call_after_refresh(refreshed.set)
        return False

    # A widget passes its layout request on when it goes idle, which this waits for; the refresh
    # requested next is queued behind those requests.
    await pilot.pause()
    await wait_for(takes_the_click, pilot=pilot, description=f"{target!r} can take a click")
    assert await pilot.click(found[0], offset=offset, **modifiers)


def delay_resize_dispatch(app: App[None], monkeypatch: pytest.MonkeyPatch, delay: float) -> None:
    """Hold back the App's debounced resize dispatch, as a loaded runner does."""
    dispatch = app._check_resize

    def late_dispatch() -> None:
        app._resize_timer = app.set_timer(delay, dispatch)

    monkeypatch.setattr(app, "_check_resize", late_dispatch)


async def resize_when_settled(pilot: Pilot, width: int, height: int) -> None:
    """Resize the terminal and return once the screen is laid out at the new size.

    The App hands a resize on to its screen from a debounce timer, which the Pilot wait does not
    cover, so ``pilot.resize_terminal`` can return with the screen still laid out at the old size.
    ``Screen.size`` reads the App's size, which changes at once; ``outer_size`` changes with layout.
    """
    await pilot.resize_terminal(width, height)
    await wait_for(
        lambda: pilot.app.screen.outer_size == (width, height),
        pilot=pilot,
        description=f"the screen to be laid out at {width}x{height}",
    )
    # Drain the Resize events that layout posted to the widgets.
    await pilot.pause()


def _simulate_chat_panel_user_scroll_y(panel: ChatPanel, y: float) -> None:
    """Move ChatPanel scroll position through the same watcher path as a user scroll."""
    old_y = panel.scroll_y
    panel._anchor_released = True
    panel.set_reactive(Widget.scroll_y, float(y))
    panel.set_reactive(Widget.scroll_target_y, float(y))
    panel.watch_scroll_y(old_y, float(y))


async def mount_sub_agent_detail(card: SubAgentToolCall, pilot: object) -> AgentTranscriptSurface:
    """Mount and return the same full transcript projection used by the modal."""
    surface = card._tool_view_output_widgets()[0]
    assert isinstance(surface, AgentTranscriptSurface)
    await card.app.mount(surface)
    await wait_for(
        lambda: surface.is_mounted,
        pilot=pilot,
        description="sub-agent detail transcript mounted",
    )
    return surface


async def wait_for_sub_agent_inner_tool(
    card: SubAgentToolCall,
    call_id: str,
    pilot: object,
) -> BaseToolCard:
    """Wait for and return a nested tool from a full detail projection."""
    surface = await mount_sub_agent_detail(card, pilot)
    matched: list[BaseToolCard] = []

    def find_tool() -> bool:
        for group in surface.query(ToolGroup):
            group.collapsed = False
            tool = group.get_tool(call_id)
            if isinstance(tool, BaseToolCard):
                matched[:] = [tool]
                return True
        return False

    await wait_for(find_tool, pilot=pilot, description=f"nested tool {call_id}")
    tool = matched[0]
    assert isinstance(tool, BaseToolCard)
    return tool


# --- Suggestion handler doubles -------------------------------------------------
#
# `SuggestionScreen` mirrors the private `MainScreen` attributes the suggestion
# handler reads through `MainScreenViewAdapter`; every new field the handler
# starts reading has to be added here once. `TaskWorker` needs a running event
# loop, so sync tests that use it drive the handler through `asyncio.run(...)`.


class TaskWorker:
    """A Textual worker double that runs its coroutine on the current event loop."""

    def __init__(self, work) -> None:
        self._task = asyncio.create_task(work)

    @property
    def is_finished(self) -> bool:
        return self._task.done()

    async def wait(self):
        return await self._task


class DeferredWorker:
    """A worker double that starts its coroutine only when it is awaited."""

    def __init__(self, work) -> None:
        self._work = work
        self._task: asyncio.Task | None = None

    @property
    def is_finished(self) -> bool:
        return self._task is not None and self._task.done()

    async def wait(self):
        if self._task is None:
            self._task = asyncio.create_task(self._work)
        return await self._task


class SuggestionListStub:
    """Records what the suggestion handler shows without mounting a widget."""

    def __init__(self) -> None:
        self.last_mode: str | None = None
        self.last_items: list[object] = []
        self.last_title: str | None = None
        self.is_visible = False
        self.is_loading = False
        self.select_result = False

    def show(self, mode: str, items=None, *_args, **_kwargs) -> None:
        self.last_mode = mode
        self.last_items = list(items or [])
        self.last_title = _kwargs.get("title")
        self.is_visible = True
        self.is_loading = False

    def show_loading(self, mode: str, *, title: str = "") -> None:
        self.last_mode = mode
        self.last_items = []
        self.last_title = title
        self.is_visible = True
        self.is_loading = True

    def update(self, items=None, *_args, **_kwargs) -> None:
        self.last_items = list(items or [])
        self.is_loading = False
        title = _kwargs.get("title")
        if title is not None:
            self.last_title = title

    def hide(self) -> None:
        self.last_mode = ""
        self.is_visible = False
        self.is_loading = False
        return

    def select_highlighted(self, *, execute: bool = False) -> bool:
        _ = execute
        return self.select_result

    @property
    def mode(self) -> str:
        return self.last_mode or ""


@dataclass(frozen=True, slots=True)
class AgentProfileStub:
    name: str
    display_name: str = ""
    description: str = ""


class AgentRegistryStub:
    """Serves a fixed set of agent profiles to the `#` trigger."""

    def __init__(self, profiles: list[AgentProfileStub]) -> None:
        self._profiles = profiles

    def list_profiles(self) -> list[AgentProfileStub]:
        return list(self._profiles)


class DismissInputBarStub:
    """Input-bar double that records trigger replacements and prompt-history loads."""

    def __init__(self) -> None:
        self.value = ""
        self.replacements: list[tuple[str, str]] = []
        self.prompt_history: list[str] = []
        self.prompt_history_limits: list[int] = []
        self.suggestions_active = False
        self.suggestion_mode: str | None = None

    def set_suggestions_active(self, active: bool, *, mode: str | None = None) -> None:
        self.suggestions_active = active
        self.suggestion_mode = mode if active else None

    def focus_input(self) -> None:
        return

    def replace_trigger_text(self, trigger: str, replacement: str) -> None:
        self.replacements.append((trigger, replacement))

    async def load_prompt_history(self, *, max_entries: int) -> list[str]:
        self.prompt_history_limits.append(max_entries)
        return list(self.prompt_history)


class SuggestionScreen:
    """The main-screen surface the suggestion handler reads through its view adapter."""

    def __init__(self) -> None:
        self.app = type("_App", (), {"available_themes": ["textual-dark"], "theme": "textual-dark"})()
        self.state = MainScreenState()
        self.services = MainScreenServices(bus=EventBus(), state_store=object())
        self._agent_running = False
        self._profile = "Code"
        self._chdir_current_cwd = ""
        self._runtime_details = AgentRuntimeDetails()
        self.is_attached = True
        self.opened: list[str] = []
        self.notifications: list[str] = []
        self.submitted: list[str] = []
        self.picked_models: list[str] = []
        self.fork_requests = 0
        self.clear_requests = 0
        self.title_editor_requests = 0
        self.applied_titles: list[str] = []
        self.suggestion_list = SuggestionListStub()
        self.input_bar = DismissInputBarStub()

    def _debug(self, *_args, **_kwargs) -> None:
        return

    def action_pick_theme(self) -> None:
        return

    def _resume_last_session(self) -> None:
        return

    def _fork_current_session(self) -> None:
        self.fork_requests += 1

    def _open_session_title_editor(self) -> None:
        self.title_editor_requests += 1

    def _apply_session_title_from_command(self, custom_title: str) -> None:
        self.applied_titles.append(custom_title)

    def _create_new_session(self) -> None:
        return

    def _clear_current_session(self) -> None:
        self.clear_requests += 1

    def action_quit(self) -> None:
        return

    def action_sessions(self) -> None:
        return

    def _chdir(self, _arg: str) -> None:
        return

    def _copy_agent_responses(self, _arg: str) -> None:
        return

    def _toggle_fold(self) -> None:
        return

    def action_show_diff(self) -> None:
        return

    def action_show_rollback(self, _arg: str = "") -> None:
        return

    def _set_approval_mode(self, _arg: str) -> None:
        return

    def _open_model_config(self) -> None:
        return

    def _open_agent_config(self) -> None:
        self.opened.append("agent")

    def _open_agent_config_tab(self, tab: str) -> None:
        self.opened.append(tab)

    def action_runtime_details(self) -> None:
        self.opened.append("runtime")

    def _open_settings(self, tab: str) -> None:
        self.opened.append(f"settings:{tab}")

    def _submit_user_text(self, text: str) -> None:
        self.submitted.append(text)

    def notify(self, message: MessageRef | str, **_kwargs) -> None:
        self.notifications.append(format_message(message) if isinstance(message, MessageRef) else message)

    def run_worker(self, work, **_kwargs) -> TaskWorker:
        return TaskWorker(work)

    def query_one(self, cls):
        name = cls.__name__
        if name == "InputBar":
            return self.input_bar
        return self.suggestion_list

    @property
    def _agent_running(self) -> bool:
        return self.state.run.agent_running

    @_agent_running.setter
    def _agent_running(self, value: bool) -> None:
        self.state.run.agent_running = value

    @property
    def _state_store(self) -> object | None:
        return self.services.state_store

    @_state_store.setter
    def _state_store(self, value: object | None) -> None:
        self.services.state_store = value

    @property
    def _profile(self) -> str:
        return self.state.runtime.profile

    @_profile.setter
    def _profile(self, value: str) -> None:
        self.state.runtime.profile = value

    @property
    def _agent_registry(self) -> object | None:
        return self.services.agent_registry

    @_agent_registry.setter
    def _agent_registry(self, value: object | None) -> None:
        self.services.agent_registry = value

    @property
    def _chdir_current_cwd(self) -> str:
        return self.state.workspace_marker.current_cwd

    @_chdir_current_cwd.setter
    def _chdir_current_cwd(self, value: str) -> None:
        self.state.workspace.current_cwd = value
        self.state.workspace_marker.current_cwd = value

    @property
    def _runtime_details(self) -> AgentRuntimeDetails:
        return self.state.runtime.details

    @_runtime_details.setter
    def _runtime_details(self, value: AgentRuntimeDetails) -> None:
        self.state.runtime.details = value


def make_suggestion_screen() -> SuggestionScreen:
    """Build the lightweight main-screen double the suggestion handler drives."""
    return SuggestionScreen()


def make_slash_actions(screen: SuggestionScreen) -> SlashCommandActions:
    """Wire every slash-command action to the recording screen double."""
    return SlashCommandActions(
        list_themes=lambda: sorted(screen.app.available_themes),
        get_theme=lambda: screen.app.theme,
        apply_theme=lambda name: setattr(screen.app, "theme", name),
        pick_theme=screen.action_pick_theme,
        list_languages=list,
        get_language=lambda: "system",
        apply_language=lambda _requested_locale: None,
        pick_language=lambda: None,
        render_unknown_language_warning=lambda requested_locale: f"Unknown /language locale: {requested_locale}",
        debug_event=screen._debug,
        new_session=screen._create_new_session,
        clear_session=screen._clear_current_session,
        quit_app=screen.action_quit,
        resume_session=screen._resume_last_session,
        fork_session=screen._fork_current_session,
        workflow_selection=lambda: None,
        open_guide=lambda: None,
        browse_session_list=screen.action_sessions,
        edit_session_title=screen._open_session_title_editor,
        apply_session_title=screen._apply_session_title_from_command,
        change_directory=screen._chdir,
        copy_conversation=screen._copy_agent_responses,
        fold_tools=screen._toggle_fold,
        open_diff=screen.action_show_diff,
        open_rollback=screen.action_show_rollback,
        get_approval_mode=lambda: ApprovalMode.MANUAL.value,
        change_approval_mode=screen._set_approval_mode,
        configure_model=screen._open_model_config,
        configure_agent=screen._open_agent_config,
        configure_agent_tab=screen._open_agent_config_tab,
        show_runtime_details=screen.action_runtime_details,
        configure_settings=screen._open_settings,
        show_manual_pages=lambda _pages, _start_index: None,
        warn=lambda message, title, timeout: screen.notify(message, title=title, timeout=timeout),
    )


def make_suggestion_handler(
    screen: SuggestionScreen,
    *,
    locale_controller: LocaleController | None = None,
) -> SuggestionHandler:
    """Build a suggestion handler around the screen double."""
    view = MainScreenViewAdapter(screen)  # type: ignore[arg-type]
    return SuggestionHandler(
        state=screen.state,
        services=screen.services,
        view=view,
        command_actions=make_slash_actions(screen),
        callbacks=SuggestionCallbacks(
            notify_warning=lambda message, title, timeout: screen.notify(message, title=title, timeout=timeout),
            show_file_suggestions=lambda: None,
            submit_user_text=screen._submit_user_text,
            start_agent_profile_switch=lambda _profile: None,
            start_model_profile_switch=screen.picked_models.append,
        ),
        buddy_view=view,
        locale_controller=locale_controller,
    )


def suggestion_values(items: list[object]) -> list[str]:
    """Read the values out of suggestion items or plain (value, label) pairs."""
    values: list[str] = []
    for item in items:
        if isinstance(item, SuggestionItem):
            values.append(item.value)
        else:
            value, _label = item  # type: ignore[misc]
            values.append(value)
    return values


def scan_result(root: str, paths: list[ProjectPathSuggestion], **kwargs: object) -> ProjectPathScanResult:
    """Build a project-path scan result for a fake scanner."""
    return ProjectPathScanResult.from_suggestions(root=root, paths=paths, **kwargs)


async def wait_for_file_query(handler: SuggestionHandler) -> None:
    """Await the handler's in-flight `@` file query, if there is one."""
    worker = handler._file_query_worker
    if worker is not None:
        await worker.wait()
