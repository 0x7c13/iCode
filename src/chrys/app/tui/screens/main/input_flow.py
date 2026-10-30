# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""User submit, injection, retry, and interrupt flow for the main screen.

Every send to the model claims admission here (``MainScreenState.submit``)
when the user asks for it, and holds the claim until it has published or
handed its text back; a second send meanwhile is refused and its text given
back. A send whose text is prepared first — a prompt shortcut's
``!`command``` placeholders and agent and model switches — claims at Enter
through :meth:`InputFlowController.begin_prepared` and publishes under that
same claim, so no other entry point can publish in between.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from chrys.app.tui.screens.main.ports import InputFlowView, InputRestore, StartWorker
from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState, SubmitKind
from chrys.app.tui.support.gc_freeze import GcAbsorbReason, GcAbsorbRequested
from chrys.app.tui.widgets.chrome.input_bar import INPUT_CONTINUE
from chrys.app.tui.widgets.chrome.status_bar import STATUS_RESUMING, STATUS_THINKING
from chrys.foundation.events.types import (
    Error,
    InvocationContentAbandoned,
    InvocationContentDelta,
    InvocationContentFinished,
    InvocationMessage,
    UserInjectCancel,
    UserInterrupt,
    UserMessage,
    UserRetry,
)

if TYPE_CHECKING:
    from chrys.app.tui.screens.main.state import DeferredTurnEvent
    from chrys.foundation.events.bus import EventBus

type GiveBack = Callable[[], InputRestore | None]
"""A prepared send's one way to return its draft to the composer, once; the proof of that restore, or None."""

type SendIntent = Literal["message", "continue"]
"""What the user asked for: a new message, or to continue the current task.

A message is published as a new turn or, while a turn runs, as an injection,
decided when it is published.
"""


@dataclass(frozen=True, slots=True)
class PreparedSend:
    """A send that claimed admission when the user asked for it and publishes later.

    *text* is the draft as the user wrote it, ``!`command``` placeholders
    included. *claim* is this send's number in ``MainScreenState.submit``.
    Whatever stops it — an abandon, a backend refusal, a check that fails
    right before publishing — gives its text back through the one give-back
    it claimed with, so the draft returns once, with its card.
    """

    claim: int
    text: str
    intent: SendIntent


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
        handle_agent_content: Callable[
            [InvocationContentDelta | InvocationContentFinished | InvocationContentAbandoned], Awaitable[None]
        ],
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
        self._handle_agent_content = handle_agent_content
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
        begun = self._begin_retry(text)
        if begun is not None:
            self._start_worker(partial(self._publish_retry, *begun))
        elif text and self._state.submit.active and self._view.restore_input_draft(text):
            # Another send holds the claim (a prompt shortcut's may hold it
            # while its commands run): the typed note comes back, as for Enter.
            self._view.flash_submit_still_sending()

    # -- prepared sends --------------------------------------------------- #

    def begin_prepared(
        self,
        text: str,
        intent: SendIntent,
        *,
        give_back: GiveBack,
        still_valid: Callable[[], bool],
    ) -> PreparedSend | None:
        """Claim admission now for *text*, to be published later with :meth:`publish_prepared`.

        Nothing else happens yet: no render gate, no running or pending state.
        *give_back* returns the draft to the composer (once, a newer draft
        winning) and hands back the proof of that restore, so a follow-up (a
        refused image's "use paths") can rewrite it; *still_valid* tells
        whether the send may still publish.
        Refused like any submit while another is in flight (and a continue
        while a turn runs): *text* goes back through *give_back*.
        """
        if self._state.submit.active or (intent == "continue" and self._state.run.agent_running):
            if give_back() is not None:
                self._view.flash_submit_still_sending()
            return None
        kind: SubmitKind = "retry" if intent == "continue" else "turn"
        claim = self._state.submit.begin(text, kind=kind, give_back=give_back, still_valid=still_valid)
        return PreparedSend(claim, text, intent)

    def holds(self, prepared: PreparedSend) -> bool:
        """Return whether *prepared* still owns the submit in flight."""
        return self._state.submit.holds(prepared.claim)

    def publish_prepared(self, prepared: PreparedSend, final_text: str) -> bool:
        """Hand *final_text* to the publish under *prepared*'s claim, never claiming again.

        The claim stays the send's until its event is on the bus: the publish
        checks once more, right before publishing, that the send is still
        valid, and gives the draft back instead when it is not. False, with
        nothing scheduled, when the claim is no longer this send's.
        """
        if not self.holds(prepared):
            return False
        claim = prepared.claim
        if prepared.intent == "continue":
            self._start_worker(partial(self._publish_prepared_retry, UserRetry(text=final_text), claim))
        else:
            self._start_worker(partial(self._publish_prepared_message, final_text, claim))
        return True

    def abandon_prepared(self, prepared: PreparedSend) -> bool:
        """Release *prepared*'s claim and give its text back; False when the claim was no longer its own."""
        return self._give_back_and_release(prepared.claim)

    def _give_back_and_release(self, claim: int) -> bool:
        give_back = self._state.submit.give_back
        if not self._state.submit.release(claim):
            return False
        return give_back is not None and give_back() is not None

    def _prepared_may_publish(self, claim: int) -> bool:
        """Whether the prepared send holding *claim* may publish now, its own check included."""
        submit = self._state.submit
        if not submit.holds(claim) or self._state.run.agent_loading:
            return False
        return submit.still_valid is None or submit.still_valid()

    async def _publish_prepared_message(self, text: str, claim: int) -> None:
        # No await between this check and the publish: nothing can change in between.
        if not self._prepared_may_publish(claim):
            self._give_back_and_release(claim)
            return
        if self._state.run.agent_running:
            self._state.submit.kind = "injection"
            await self._inject_claimed(text, claim)
        else:
            await self._send_claimed(text, claim)

    async def _publish_prepared_retry(self, event: UserRetry, claim: int) -> None:
        if not self._prepared_may_publish(claim):
            self._give_back_and_release(claim)
            return
        self._state.render_gate.begin()
        self._view.set_retry_pending(True)
        await self._publish_retry(event, claim)

    # -- claims ----------------------------------------------------------- #

    def _claim_submit(self, text: str, *, kind: SubmitKind) -> int | None:
        """Make *text* the one submit in flight, or hand it back to the composer.

        Called before the worker's first await, so of two submits scheduled in
        the same tick only the first claims: the one in flight owns the submit
        state, and its cleanup would erase a second one's refusal. A running
        turn's Enter never cleared the composer; a new turn's did, so the text
        coming back says why — until the first submit's own outcome shows.
        """
        if self._state.submit.active:
            if self._view.restore_input_draft(text):
                self._view.flash_submit_still_sending()
            return None
        return self._state.submit.begin(text, kind=kind)

    async def _publish_submit(self, event: UserMessage | UserRetry) -> bool:
        """Publish the submit in flight; return whether the backend refused it.

        ``EventBus.publish`` awaits its handlers in order, so a refusal reaches
        the frontend on this task before the publish returns.
        """
        self._state.submit.publish_from_current_task()
        await self._bus.publish(event)
        return self._state.submit.blocked

    async def send_user_message(self, text: str) -> None:
        """Publish and render a normal user message."""
        if self._state.run.agent_loading:
            return
        claim = self._claim_submit(text, kind="turn")
        if claim is None:
            return
        await self._send_claimed(text, claim)

    async def _send_claimed(self, text: str, claim: int) -> None:
        """Publish *text* as a new turn under *claim*, already held."""
        # Read before the claim ends: a prepared send gives back its own draft.
        give_back = self._state.submit.give_back
        try:
            event = UserMessage(text=text)
            self._state.render_gate.begin()
            if await self._publish_submit(event):
                self._state.render_gate.finish(rendered=False)
                return
        finally:
            self._state.submit.release(claim)
        if not event.started_turn:
            # Neither refused nor a new turn: the run seen ending here was still
            # finishing and took the message, or the backend dropped it. The
            # text is the user's again; if that run used it, its outcome shows
            # it in the transcript and clears the composer.
            self._state.render_gate.finish(rendered=False)
            if give_back is not None:
                give_back()
            else:
                self._view.restore_input_draft(text)
            self._debug("UserMessage (no turn)", text[:60])
            return
        await self._open_turn(event)
        self._debug("UserMessage", text[:60])

    async def _open_turn(self, event: UserMessage) -> None:
        """Show *event* as the opener of the turn its publish just started.

        Called right after that publish returns, before the new run's task has
        had a chance to run, with the render gate up: the run's first events
        then find the turn live and wait for the bubble.
        """
        text = event.text
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

    async def queue_injection(self, text: str) -> None:
        """Queue a user message for mid-run injection without chat display."""
        claim = self._claim_submit(text, kind="injection")
        if claim is None:
            return
        await self._inject_claimed(text, claim)

    async def _inject_claimed(self, text: str, claim: int) -> None:
        """Queue *text* for the running turn under *claim*, already held."""
        try:
            injection_id = uuid4().hex[:12]
            self._view.lock_input_with_text()
            self._state.pending_injection.begin(injection_id, text)
            event = UserMessage(text=text, injection_id=injection_id)
            if await self._publish_submit(event):
                self._state.pending_injection.clear()
                return
        finally:
            self._state.submit.release(claim)
        if event.started_turn:
            # The run ended before the backend admitted this message, so it
            # opened a new turn with it: show that turn as a new turn's Enter
            # would. The run's end or an Esc handed the composer back meanwhile,
            # so only the sent text leaves it.
            if self._state.pending_injection.matches(injection_id):
                self._state.pending_injection.clear()
                self._view.unlock_input_clearing(text)
            self._state.render_gate.begin()
            await self._open_turn(event)
            self._debug("UserMessage (inject → new turn)", text[:60])
            return
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
        """Render backend events that arrived while the user bubble was mounting, in arrival order."""
        await self._state.render_gate.drain(self._replay_deferred)

    async def _replay_deferred(self, event: DeferredTurnEvent) -> None:
        if isinstance(event, InvocationMessage):
            await self._handle_agent_message(event)
        elif isinstance(event, InvocationContentDelta | InvocationContentFinished | InvocationContentAbandoned):
            await self._handle_agent_content(event)
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

    def _begin_retry(self, text: str) -> tuple[UserRetry, int] | None:
        """Reserve admission before scheduling, including same-tick clicks; return the event and its claim."""
        if self._state.run.agent_running or self._state.submit.active:
            return None

        event = UserRetry(text=text)
        self._state.render_gate.begin()
        claim = self._state.submit.begin(text, kind="retry")
        self._view.set_retry_pending(True)
        return event, claim

    async def _publish_retry(self, event: UserRetry, claim: int) -> None:
        text = event.text
        try:
            if await self._publish_submit(event):
                self._state.render_gate.finish(rendered=False)
                return
        except BaseException:
            self._state.render_gate.finish(rendered=False)
            raise
        finally:
            self._state.submit.release(claim)
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
            # The backend withdraws the answer it drops to ask for it again (InvocationAnswerWithdrawn).
            await self.flush_deferred_agent_messages()

        self._debug("UserRetry", text[:60] if text else "")
