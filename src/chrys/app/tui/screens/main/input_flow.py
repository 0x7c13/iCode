# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""User submit, injection, retry, and interrupt flow for the main screen."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import partial
from typing import TYPE_CHECKING
from uuid import uuid4

from chrys.app.tui.screens.main.ports import InputFlowView, StartWorker
from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.app.tui.support.gc_freeze import GcAbsorbReason, GcAbsorbRequested
from chrys.app.tui.widgets.chrome.input_bar import INPUT_CONTINUE
from chrys.app.tui.widgets.chrome.status_bar import STATUS_RESUMING, STATUS_THINKING
from chrys.foundation.events.types import (
    Error,
    InvocationMessage,
    UserInjectCancel,
    UserInterrupt,
    UserMessage,
    UserRetry,
)

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus


class InputFlowController:
    """Own the temporal contracts around user submits and turn rendering."""

    def __init__(
        self,
        *,
        state: MainScreenState,
        services: MainScreenServices,
        view: InputFlowView,
        start_worker: StartWorker,
        handle_agent_message: Callable[[InvocationMessage], Awaitable[None]],
        handle_error: Callable[[Error], Awaitable[None]],
        set_agent_running: Callable[[bool], None],
        set_has_messages: Callable[[bool], None],
        clear_workspace_marker: Callable[[], None],
        clear_pending_questions: Callable[[], None],
        commit_profile_switch_marker: Callable[[], None],
        post_gc_message: Callable[[GcAbsorbRequested], object],
        debug: Callable[[str, str], None],
    ) -> None:
        self._state = state
        self._services = services
        self._view = view
        self._start_worker = start_worker
        self._handle_agent_message = handle_agent_message
        self._handle_error = handle_error
        self._set_agent_running = set_agent_running
        self._set_has_messages = set_has_messages
        self._clear_workspace_marker = clear_workspace_marker
        self._clear_pending_questions = clear_pending_questions
        self._commit_profile_switch_marker = commit_profile_switch_marker
        self._post_gc_message = post_gc_message
        self._debug = debug

    @property
    def _bus(self) -> EventBus:
        return self._services.bus

    def submit_user_text(self, text: str) -> None:
        """Submit normal user text as a new turn or mid-run injection."""
        if self._state.run.agent_running:
            self._start_worker(partial(self.queue_injection, text))
            return
        self._start_worker(partial(self.send_user_message, text))

    def request_retry(self, text: str = "") -> None:
        """Schedule retry/continue from a sync Textual event handler."""
        event = self._begin_retry(text)
        if event is not None:
            self._start_worker(partial(self._publish_retry, event))

    async def send_user_message(self, text: str) -> None:
        """Publish and render a normal user message."""
        if self._state.run.agent_loading:
            return

        event = UserMessage(text=text)
        self._state.render_gate.begin()
        self._state.submit.begin(text)
        try:
            await self._bus.publish(event)
            if self._state.submit.blocked:
                self._state.render_gate.finish(rendered=False)
                return
        finally:
            self._state.submit.clear()

        self._state.run.started_at = event.timestamp
        self._view.set_terminal_title_for_user_message(text)
        self._set_agent_running(True)
        self._set_has_messages(True)
        self._clear_workspace_marker()
        self._view.add_input_history(text, session_id=self._view.current_chat_session_id() or None)
        self._view.set_retry_mode(False)
        self._view.start_run_status(STATUS_THINKING.bind())
        self._commit_profile_switch_marker()

        rendered = False
        try:
            # EventBus.publish is sequential; the backend fills prepared_contents
            # before returning so this previews exactly what the model receives.
            await self._view.render_user_message(text, created_at=event.timestamp, contents=event.prepared_contents)
            rendered = True
        finally:
            self._state.render_gate.finish(rendered=rendered)

        await self.flush_deferred_agent_messages()
        self._view.update_toc()
        self._debug("UserMessage", text[:60])

    async def queue_injection(self, text: str) -> None:
        """Queue a user message for mid-run injection without chat display."""
        injection_id = uuid4().hex[:12]
        self._view.set_terminal_title_for_user_message(text)
        self._view.lock_input_with_text()
        self._state.pending_injection.begin(injection_id, text)
        self._state.submit.begin(text)
        try:
            await self._bus.publish(UserMessage(text=text, injection_id=injection_id))
            if self._state.submit.blocked:
                self._state.pending_injection.clear()
                return
        finally:
            self._state.submit.clear()
        self._debug("UserMessage (inject)", text[:60])

    def cancel_pending_injection(self) -> bool:
        """Withdraw the queued injection and hand the text back for editing.

        Optimistically unlocks the input right away — the backend cancel is
        best-effort, and a too-late (already consumed) outcome arrives as a
        ``consumed=True`` result that no longer touches the input bar.
        """
        pending = self._state.pending_injection
        if not pending.active:
            return False
        injection_id = pending.injection_id
        pending.clear()
        self._view.unlock_input_keep_if_locked()
        if injection_id is not None:
            self._start_worker(partial(self._publish_inject_cancel, injection_id))
        return True

    async def _publish_inject_cancel(self, injection_id: str) -> None:
        """Publish the backend cancel for a withdrawn injection."""
        await self._bus.publish(UserInjectCancel(injection_id=injection_id))
        self._debug("UserInjectCancel", injection_id)

    async def flush_deferred_agent_messages(self) -> None:
        """Render backend events that arrived while the user bubble was mounting."""
        for event in self._state.render_gate.consume_deferred():
            if isinstance(event, InvocationMessage):
                await self._handle_agent_message(event)
            elif isinstance(event, Error):
                await self._handle_error(event)

    async def publish_interrupt(self) -> None:
        """Publish an interrupt and update local UI state after the backend sees it."""
        if not self._state.run.agent_running:
            return

        await self._bus.publish(UserInterrupt())
        terminal_request = GcAbsorbRequested(GcAbsorbReason.TURN_TERMINAL, terminal_boundary=True)
        try:
            self._clear_pending_questions()
            self._view.flash_interrupted()
            self._view.clear_ask_user_inline_prompts()
            await self._view.render_interrupted()
        finally:
            self._view.clear_terminal_title_result()
            self._set_agent_running(False)
            # Teardown events (late tool results, compaction finishing) that
            # raced the awaits above saw ``agent_running`` still True and may
            # have flipped the status bar back to run mode; now that the gate
            # is closed, re-assert the interrupted flash so nothing leaves a
            # stuck "Thinking" spinner behind.
            self._view.flash_interrupted()
            # The interrupt already unlocks the input with the text kept; drop
            # pending-injection tracking so a later abandoned result (or Esc)
            # cannot act on an entry that no longer belongs to a live turn.
            self._state.pending_injection.clear()
            self._view.unlock_input_keep_if_locked()
        self._post_gc_message(terminal_request)
        self._view.set_retry_mode(True, label=INPUT_CONTINUE.bind())
        self._debug("UserInterrupt", "")

    def _begin_retry(self, text: str) -> UserRetry | None:
        """Reserve admission before scheduling, including same-tick clicks."""
        if self._state.run.agent_running or self._state.submit.active:
            return None

        event = UserRetry(text=text)
        self._state.render_gate.begin()
        self._state.submit.begin(text, is_retry=True)
        self._view.set_retry_pending(True)
        return event

    async def _publish_retry(self, event: UserRetry) -> None:
        text = event.text
        try:
            await self._bus.publish(event)
            if self._state.submit.blocked:
                self._state.render_gate.finish(rendered=False)
                return
        except BaseException:
            self._state.render_gate.finish(rendered=False)
            raise
        finally:
            self._state.submit.clear()
            self._view.set_retry_pending(False)

        self._view.hide_trailing_status_action()
        self._state.run.started_at = event.timestamp
        self._set_agent_running(True)
        self._view.start_run_status(STATUS_RESUMING.bind())

        if text:
            self._view.set_terminal_title_for_user_message(text)
            self._view.add_input_history(text, session_id=self._view.current_chat_session_id() or None)
            rendered = False
            try:
                await self._view.render_user_retry_note(text, created_at=event.timestamp)
                rendered = True
            finally:
                self._state.render_gate.finish(rendered=rendered)
            await self.flush_deferred_agent_messages()
            self._view.update_toc()
        else:
            self._state.render_gate.finish(rendered=True)
            await self.flush_deferred_agent_messages()

        self._debug("UserRetry", text[:60] if text else "")
