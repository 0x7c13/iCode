# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The prompt-shortcut card's state, lifecycle and send.

The draft a prompt shortcut expanded into stays that shortcut's draft (a
:class:`ShortcutIntent`: its agent, model and how many ``!`command```
placeholders the draft holds now) for as long as it is edited, whatever the
edit; only clearing or replacing the draft, or sending it, ends it. Every
``!`command``` in such a draft runs when it is sent, and the card is shown
whenever a send would run commands or switch, so what Enter will do is
always on screen. What the card shows, and what a send switches, are both
worked out from that declaration against the session at that moment: the
card shows only what differs. ✕, or the user picking an agent or model
themselves, drops only the switch: the commands still run.

Enter claims the one send in flight (``InputFlowController.begin_prepared``)
and hands the draft to a :class:`PromptSendOperation`, which runs the
commands, switches the agent and then the model (session-wide), and
publishes under that same claim; no other way of sending can publish in
between. A send that fails, is cancelled or goes stale releases its own
claim and gives its text back once. The draft is free meanwhile: a shortcut
expanded during a send gets a card of its own that the send never touches.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from chrys.app.tui.widgets.chrome.prompt_card import CardChoice, CardState
from chrys.foundation.i18n import MessageRef, msg
from chrys.service.prompt_shortcuts import PromptShortcut
from chrys.service.prompt_shortcuts.shell import ShellInterpolationError, interpolate_shell, shell_commands

if TYPE_CHECKING:
    from textual.worker import Worker

    from chrys.app.tui.screens.main.input_flow import GiveBack, PreparedSend, SendIntent
    from chrys.app.tui.screens.main.ports import InputRestore

logger = logging.getLogger(__name__)

PROMPT_SHORTCUT_TITLE = msg("tui.prompt_card.title", fallback="Prompt shortcut")
_BUSY = msg(
    "tui.prompt_card.busy",
    fallback="Send it when the current task ends, or press “✕ Don't switch” to add it to the task as is.",
)
_BUSY_COMMANDS = msg(
    "tui.prompt_card.busy_commands",
    fallback="This prompt runs commands when sent: send it when the current task ends.",
)
_SHELL_TIMEOUT = msg("tui.prompt_card.shell_timeout", fallback="Not sent: {command} did not finish in time")
_SHELL_START_FAILED = msg("tui.prompt_card.shell_start_failed", fallback="Not sent: {command} could not start")
_SHELL_STOPPED = msg("tui.prompt_card.shell_stopped", fallback="Not sent: {command} was stopped")
_SWITCH_FAILED = msg("tui.prompt_card.switch_failed", fallback="Not sent: could not switch to {target}")
_MODEL_LOCKED = msg(
    "tui.prompt_card.model_locked",
    fallback="The agent fixes its own model, so the card's model was not applied.",
)
_CANCELLED = msg("tui.prompt_card.cancelled", fallback="Sending cancelled")
_SESSION_CHANGED = msg(
    "tui.prompt_card.session_changed",
    fallback="Not sent: the session or working folder changed while the prompt was being prepared",
)
_LOADING = msg(
    "tui.prompt_card.loading",
    fallback="Not sent: the agent is still starting; send it again once it is ready",
)
_SEND_FAILED = msg("tui.prompt_card.send_failed", fallback="Not sent: something went wrong; see the logs for details")

_SHELL_REASONS = {"timeout": _SHELL_TIMEOUT, "start_failed": _SHELL_START_FAILED, "stopped": _SHELL_STOPPED}


@dataclass(frozen=True, slots=True)
class Option:
    """A pickable agent or model: what the user sees and what a switch takes."""

    label: str
    value: str
    model_id: str = ""
    provider: str = ""


@dataclass(frozen=True, slots=True)
class PromptCardCallbacks:
    """Screen-owned reads and effects the card needs."""

    show: Callable[[CardState], None]
    hide: Callable[[], None]
    agents: Callable[[], Sequence[Option]]
    """Agents a chat can switch to."""
    models: Callable[[], Sequence[Option]]
    """Selectable model profiles."""
    current_agent: Callable[[], tuple[str, str]]
    """The session agent's profile name and display name."""
    current_model: Callable[[], tuple[str, str]]
    """The session model's profile id and display name."""
    agent_running: Callable[[], bool]
    agent_loading: Callable[[], bool]
    """The agent is starting or rebuilding: nothing can switch or send yet."""
    input_text: Callable[[], str]
    workspace_cwd: Callable[[], str]
    restore_draft: Callable[[str], InputRestore | None]
    """Hand text back to an empty composer, keeping its card; the proof of it, or None when a draft is already there."""
    begin_prepared: Callable[[str, SendIntent, GiveBack, Callable[[], bool]], PreparedSend | None]
    """Claim the one send in flight for a draft, with its give-back and validity check
    (``InputFlowController.begin_prepared``); None when refused."""
    holds: Callable[[PreparedSend], bool]
    publish_prepared: Callable[[PreparedSend, str], bool]
    abandon_prepared: Callable[[PreparedSend], bool]
    """Release the claim and give its text back through its give-back; False when nothing came back."""
    session_id: Callable[[], str | None]
    """The chat session a send belongs to; a send whose session changed stops."""
    switch_agent: Callable[[str], Awaitable[bool]]
    model_switchable: Callable[[str], bool]
    switch_model: Callable[[str], Awaitable[bool]]
    notify: Callable[[MessageRef, MessageRef, str], None]
    """``(message, title, severity)``."""
    run_worker: Callable[[Coroutine[Any, Any, None]], Worker[None]]
    after_refresh: Callable[[Callable[[], None]], object]


def match_model(spec: str, options: Sequence[Option]) -> Option | None:
    """Find the profile a frontmatter ``model`` names, or None when nothing or more than one fits.

    In order: a profile id or name; ``provider/model`` with both parts matching
    (opencode's form); a model id. A model id several profiles share, for
    example the same model behind two gateways, names none of them: the
    request would go to whichever came first, so the user is told instead.
    """
    named = [option for option in options if spec in {option.value, option.label}]
    if len(named) == 1:
        return named[0]
    provider, slash, model = spec.partition("/")
    candidates = [option for option in options if slash and option.provider == provider and option.model_id == model]
    if not candidates:
        # A model id may itself hold a slash (``anthropic/claude-…`` behind a router).
        candidates = [option for option in options if option.model_id == spec]
    return candidates[0] if len(candidates) == 1 else None


def shortcut_model_text(spec: str, options: Sequence[Option]) -> str:
    """A shortcut's ``model`` as listed: without its provider only when ``provider/model`` names a configured model.

    Anything else is shown as written, so a profile whose own name holds a
    slash (``team/fast``) is never cut to look like another model.
    """
    found = match_model(spec, options)
    provider, slash, model_id = spec.partition("/")
    if found is not None and slash and found.provider == provider and found.model_id == model_id:
        return model_id
    return spec


@dataclass(frozen=True, slots=True)
class ShortcutIntent:
    """What an expanded prompt shortcut declares for its draft, kept whatever the session uses now.

    Keeping the declaration, showing it and switching for it are separate: the
    card shows only what differs from the session now, and a send compares
    again when it switches. A shortcut naming the agent already in use keeps
    that name, so a send that changes the agent meanwhile does not lose it.
    """

    name: str
    agent: str
    """The frontmatter ``agent``, as written; empty when it names none."""
    model: str
    """The frontmatter ``model``, as written; empty when it names none."""
    commands: int
    """How many ``!`command``` placeholders the draft holds now."""
    switch_dropped: bool = False
    """✕, or the user's own agent or model pick: the session keeps its agent and model."""


@dataclass(slots=True)
class PromptSendOperation:
    """One send under a card, from Enter to its publish: owns its claim and works from a snapshot.

    Everything it acts on was taken at Enter. Before each step — the commands,
    the agent switch, the model switch, the publish — and after each wait it
    checks it is still current (the claim is its own, the session and working
    folder unchanged); a stale one stops, though what already ran stays done.
    """

    prepared: PreparedSend
    intent: ShortcutIntent
    state: CardState
    session_id: str | None
    cwd: str
    worker: Worker[None] | None = None
    started: bool = False
    """Its worker began running: from then on the worker's own cleanup ends it."""
    work: Coroutine[Any, Any, None] | None = None
    """The send the worker runs; closed unrun when it is cancelled before it starts."""


class PromptCardController:
    """Owns the card for the draft a prompt shortcut expanded into, and any send in progress under one."""

    def __init__(self, callbacks: PromptCardCallbacks) -> None:
        self._callbacks = callbacks
        self._draft: ShortcutIntent | None = None
        self._operation: PromptSendOperation | None = None
        self._shown: CardState | None = None
        """What the card shows now, so a repaint that changes nothing paints nothing."""

    @property
    def active(self) -> bool:
        """Whether the draft is a prompt shortcut's (its card shown or, with nothing to say, unseen)."""
        return self._draft is not None

    @property
    def sending(self) -> bool:
        return self._operation is not None

    # -- the draft's card --------------------------------------------------

    def attach(self, command: PromptShortcut, text: str) -> None:
        """Make *text*, which *command* just expanded into, that shortcut's draft."""
        self._set_draft(ShortcutIntent(command.name, command.agent, command.model, len(shell_commands(text))))

    def clear(self) -> None:
        """End the draft's card; a send in progress keeps its own."""
        self._set_draft(None)

    def on_text_changed(self, text: str) -> None:
        """Keep the command count in step with the draft; an emptied draft ends it after this refresh.

        Any other edit keeps the draft the shortcut's, even one that leaves it no
        commands for now: a command typed or pasted back runs, and the card says so.
        """
        draft = self._draft
        if draft is None:
            return
        if not text:
            # A submit clears the draft just before its UserSubmitted lands: let that claim the card first.
            self._callbacks.after_refresh(self._clear_if_still_empty)
            return
        commands = len(shell_commands(text))
        if commands != draft.commands:
            self._set_draft(replace(draft, commands=commands))

    def _clear_if_still_empty(self) -> None:
        if self._draft is not None and not self._callbacks.input_text():
            self.clear()

    def drop_switch(self) -> None:
        """Keep the session's agent and model (✕); the draft stays the shortcut's, and its commands still run."""
        draft = self._draft
        if draft is None or draft.switch_dropped:
            return
        self._set_draft(replace(draft, switch_dropped=True))

    def on_user_pick(self) -> None:
        """The user chose an agent or model themselves: that wins over the draft's switch.

        The one caller is ``RuntimeConfigCallbacks.user_picked``, which says what
        counts as a choice. Every other change (a settings reload, a restore, a
        settings dialog's fallback, a send) only re-measures the card
        (:meth:`on_session_changed`).
        """
        self.drop_switch()

    def on_session_changed(self) -> None:
        """The session's agent or model changed, by anyone: measure the card again, keeping its declaration."""
        self._repaint()

    def on_draft_replaced(self, *, keep_card: bool) -> None:
        if not keep_card:
            self.clear()

    def _set_draft(self, draft: ShortcutIntent | None) -> None:
        self._draft = draft
        self._repaint()

    def _repaint(self) -> None:
        # A send shows what it is doing; otherwise the draft's card shows what
        # differs from the session now and the commands a send runs. A draft
        # with neither keeps its card unseen.
        operation = self._operation
        state = (
            operation.state if operation is not None else self._view(self._draft) if self._draft is not None else None
        )
        shown = state if state is not None and state.visible else None
        if shown == self._shown:
            return
        self._shown = shown
        if shown is None:
            self._callbacks.hide()
        else:
            self._callbacks.show(shown)

    def _view(self, intent: ShortcutIntent) -> CardState:
        """What the card says for *intent*, measured against the session now."""
        dropped = intent.switch_dropped
        return CardState(
            name=intent.name,
            agent=None if dropped else self._agent_choice(intent.agent),
            model=None if dropped else self._model_choice(intent.model),
            commands=intent.commands,
        )

    # -- choices -----------------------------------------------------------

    def _agent_choice(self, spec: str) -> CardChoice | None:
        if not spec:
            return None
        options = list(self._callbacks.agents())
        current_name, current = self._callbacks.current_agent()
        # The profile name is the agent's identity; display names may repeat.
        found = next((option for option in options if spec == option.value), None) or next(
            (option for option in options if spec == option.label), None
        )
        if found is not None and found.value == current_name:
            return None
        if found is None:
            return CardChoice(value=None, label=spec, current=current)
        return CardChoice(value=found.value, label=found.label, current=current)

    def _model_choice(self, spec: str) -> CardChoice | None:
        if not spec:
            return None
        options = list(self._callbacks.models())
        current_id, current_label = self._callbacks.current_model()
        found = match_model(spec, options)
        if found is not None and found.value == current_id:
            return None
        if found is None:
            return CardChoice(value=None, label=spec, current=current_label)
        return CardChoice(value=found.value, label=found.label, current=current_label)

    # -- send --------------------------------------------------------------

    def submit(self, text: str, *, intent: SendIntent = "message") -> bool:
        """Send *text* under the draft's card; False when it has nothing to do and the caller sends it plainly.

        *intent* is what the user asked for: a new message, or a continue.
        """
        draft = self._draft
        if draft is None:
            return False
        view = self._view(draft)
        switching = any(choice is not None and choice.value is not None for choice in (view.agent, view.model))
        commands = shell_commands(text)
        if not switching and not commands:
            return False
        if self._callbacks.agent_loading():
            # As a plain draft sent now: nothing happens until the agent is ready; the text stays.
            self._callbacks.restore_draft(text)
            return True
        if intent == "message" and self._callbacks.agent_running():
            # The draft is still in the input: a running turn takes text only as an injection.
            busy = _BUSY_COMMANDS if commands else _BUSY
            self._callbacks.notify(busy.bind(), PROMPT_SHORTCUT_TITLE.bind(), "warning")
            return True
        session_id = self._callbacks.session_id()
        cwd = self._callbacks.workspace_cwd()
        # The claim is taken now, at Enter: until this send publishes or gives
        # its text back, every other way of sending is refused.
        prepared = self._callbacks.begin_prepared(
            text,
            intent,
            self._give_back_for(text, draft, session_id, cwd),
            lambda: self._in_place(session_id, cwd),
        )
        if prepared is None:
            return True
        operation = PromptSendOperation(
            prepared=prepared,
            intent=draft,
            state=replace(view, commands=len(commands), running=bool(commands), in_progress=True),
            session_id=session_id,
            cwd=cwd,
        )
        # The draft is consumed: what the user types or expands next gets a card of its own.
        self._draft = None
        self._operation = operation
        self._repaint()
        operation.work = self._send(operation)
        worker = self._callbacks.run_worker(operation.work)
        # A send with nothing to wait for can finish before run_worker returns.
        if self._operation is operation:
            operation.worker = worker
        return True

    def cancel(self) -> bool:
        """Esc during a send: True when the send takes it, False when nothing is sending.

        Only a send that has not started or is running its commands stops; its
        draft and card come back. Once it switches the agent or model, the
        switch's own loading takes over and Esc changes nothing here: a switch
        cut off midway would leave the agent loading with nothing to end it.
        """
        operation = self._operation
        if operation is None or operation.worker is None:
            return False
        if operation.started and not operation.state.running:
            return True
        operation.worker.cancel()
        if not operation.started:
            # Cancelled before it ran, the worker never reaches its own cleanup:
            # end the send here, releasing its claim and giving its draft back.
            if operation.work is not None:
                operation.work.close()
            self._callbacks.notify(_CANCELLED.bind(), PROMPT_SHORTCUT_TITLE.bind(), "information")
            self._finish(operation, published=False)
        return True

    def _current(self, operation: PromptSendOperation) -> bool:
        """Whether *operation* may take its next step: still the send in flight, for the same session and folder."""
        return (
            self._operation is operation and self._callbacks.holds(operation.prepared) and self._same_place(operation)
        )

    async def _send(self, operation: PromptSendOperation) -> None:
        if self._operation is not operation:
            # Already ended (cancelled before it ran).
            return
        operation.started = True
        published = False
        try:
            final = operation.prepared.text
            if shell_commands(final):
                if not self._current(operation):
                    return
                final = await interpolate_shell(final, cwd=operation.cwd)
                # The commands ran: from here the send can no longer be cancelled.
                operation.state = replace(operation.state, commands=0, running=False)
                self._repaint()
            if not await self._switch(operation):
                return
            if not self._current(operation):
                return
            published = self._callbacks.publish_prepared(operation.prepared, final)
        except ShellInterpolationError as error:
            message = _SHELL_REASONS.get(error.reason, _SHELL_START_FAILED)
            self._callbacks.notify(message.bind(command=error.command), PROMPT_SHORTCUT_TITLE.bind(), "error")
        except asyncio.CancelledError:
            self._callbacks.notify(_CANCELLED.bind(), PROMPT_SHORTCUT_TITLE.bind(), "information")
            raise
        except Exception:
            logger.exception("Prompt shortcut send failed")
            self._callbacks.notify(_SEND_FAILED.bind(), PROMPT_SHORTCUT_TITLE.bind(), "error")
        finally:
            self._finish(operation, published=published)

    async def _switch(self, operation: PromptSendOperation) -> bool:
        """Switch the agent, then the model, each compared with the session again; False to stop the send."""
        intent = operation.intent
        if intent.switch_dropped:
            return True
        if not self._current(operation):
            return False
        agent = self._agent_choice(intent.agent)
        if agent is not None and agent.value is not None:
            if not await self._callbacks.switch_agent(agent.value):
                self._callbacks.notify(_SWITCH_FAILED.bind(target=agent.label), PROMPT_SHORTCUT_TITLE.bind(), "error")
                return False
            if not self._current(operation):
                return False
        model = self._model_choice(intent.model)
        if model is None or model.value is None:
            return True
        if not self._callbacks.model_switchable(model.value):
            self._callbacks.notify(_MODEL_LOCKED.bind(), PROMPT_SHORTCUT_TITLE.bind(), "warning")
            return True
        if not await self._callbacks.switch_model(model.value):
            self._callbacks.notify(_SWITCH_FAILED.bind(target=model.label), PROMPT_SHORTCUT_TITLE.bind(), "error")
            return False
        return self._current(operation)

    def _finish(self, operation: PromptSendOperation, *, published: bool) -> None:
        """End *operation*: an unpublished one releases its own claim, which gives its text back once."""
        if self._operation is operation:
            self._operation = None
        if not published:
            self._callbacks.abandon_prepared(operation.prepared)
        self._repaint()

    def _give_back_for(self, text: str, intent: ShortcutIntent, session_id: str | None, cwd: str) -> GiveBack:
        """The one way a send's draft comes back: its text as written, with its card, at most once.

        Every path that ends a send without publishing — a refusal, a failure,
        a cancel, a check that fails right before the publish, the backend
        refusing it — gives the draft back through this, so each says why the
        same way. A draft typed meanwhile wins; for another session or folder
        the text comes back without its card. Returns the proof of the restore,
        so a follow-up can rewrite the text it put back.
        """
        given = False

        def give_back() -> InputRestore | None:
            nonlocal given
            if given:
                return None
            given = True
            moved = not self._in_place(session_id, cwd)
            if moved:
                self._callbacks.notify(_SESSION_CHANGED.bind(), PROMPT_SHORTCUT_TITLE.bind(), "warning")
            elif self._callbacks.agent_loading():
                self._callbacks.notify(_LOADING.bind(), PROMPT_SHORTCUT_TITLE.bind(), "warning")
            restore = self._callbacks.restore_draft(text)
            if restore is None:
                return None
            if not moved:
                self._set_draft(intent)
            return restore

        return give_back

    def _same_place(self, operation: PromptSendOperation) -> bool:
        """Whether the session and working folder are still those *operation* was sent from."""
        return self._in_place(operation.session_id, operation.cwd)

    def _in_place(self, session_id: str | None, cwd: str) -> bool:
        return self._callbacks.session_id() == session_id and self._callbacks.workspace_cwd() == cwd


__all__ = [
    "PROMPT_SHORTCUT_TITLE",
    "Option",
    "PromptCardCallbacks",
    "PromptCardController",
    "PromptSendOperation",
    "ShortcutIntent",
    "match_model",
]
