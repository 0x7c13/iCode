# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Spec tests for the prompt-shortcut card's controller and its model matching, driven through fake screen callbacks."""

from __future__ import annotations

import asyncio
import gc
import inspect
import re
import sys
import warnings
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

import chrys.app.tui.screens.main.prompt_card as prompt_card_module
import chrys.service.prompt_shortcuts.shell as shell_module
from chrys.app.tui.screens.main.input_flow import GiveBack, PreparedSend, SendIntent
from chrys.app.tui.screens.main.ports import InputRestore
from chrys.app.tui.screens.main.prompt_card import (
    Option,
    PromptCardCallbacks,
    PromptCardController,
    match_model,
    shortcut_model_text,
)
from chrys.app.tui.widgets.chrome.prompt_card import CardState
from chrys.foundation.i18n import MessageRef
from chrys.service.prompt_shortcuts import PromptShortcut
from chrys.service.prompt_shortcuts.shell import ShellInterpolationError
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell syntax")


# -- match_model -------------------------------------------------------------


_OPTIONS = (
    Option(label="Fast", value="fast-profile", model_id="gpt-x", provider="openai"),
    Option(label="Smart", value="smart-profile", model_id="claude-y", provider="anthropic"),
)


@pytest.mark.parametrize("spec", ["fast-profile", "Fast"])
def test_match_model_by_profile_id_or_name(spec: str) -> None:
    assert match_model(spec, _OPTIONS) == _OPTIONS[0]


def test_match_model_provider_slash_model_needs_both_parts() -> None:
    a = Option(label="A", value="pa", model_id="same", provider="a")
    b = Option(label="B", value="pb", model_id="same", provider="b")
    assert match_model("b/same", (a, b)) == b
    assert match_model("a/same", (a, b)) == a
    assert match_model("c/same", (a, b)) is None
    assert match_model("openai/claude-y", _OPTIONS) is None


def test_match_model_bare_model_id_must_be_unique() -> None:
    assert match_model("claude-y", _OPTIONS) == _OPTIONS[1]
    a = Option(label="A", value="pa", model_id="same", provider="a")
    b = Option(label="B", value="pb", model_id="same", provider="b")
    assert match_model("same", (a, b)) is None


def test_match_model_model_id_holding_a_slash() -> None:
    routed = Option(label="Routed", value="routed", model_id="anthropic/claude-x", provider="openrouter")
    assert match_model("anthropic/claude-x", (*_OPTIONS, routed)) == routed


def test_match_model_no_match() -> None:
    assert match_model("nothing-like-it", _OPTIONS) is None
    assert match_model("x", ()) is None


@pytest.mark.parametrize(
    ("spec", "listed"),
    [
        # provider/model naming a configured model: the model alone.
        ("anthropic/claude-y", "claude-y"),
        # A profile's own name or id, slash included, is what the file set: as written.
        ("team/fast", "team/fast"),
        ("fast-profile", "fast-profile"),
        # A model id that holds a slash, behind a router: as written.
        ("anthropic/claude-x", "anthropic/claude-x"),
        # Nothing configured by that name, or more than one: as written.
        ("openai/claude-y", "openai/claude-y"),
        ("b/unknown", "b/unknown"),
    ],
)
def test_a_shortcut_model_drops_its_provider_only_when_that_names_a_configured_model(spec: str, listed: str) -> None:
    options = (
        *_OPTIONS,
        Option(label="Team fast", value="team/fast", model_id="gpt-x", provider="openai"),
        Option(label="Routed", value="routed", model_id="anthropic/claude-x", provider="openrouter"),
    )
    assert shortcut_model_text(spec, options) == listed


# -- controller harness ------------------------------------------------------


class _FakeWorker:
    def __init__(self, task: asyncio.Task[None]) -> None:
        self.task = task

    @property
    def is_finished(self) -> bool:
        return self.task.done()

    def cancel(self) -> None:
        self.task.cancel()


class _UnstartedWorker:
    """A worker Textual has scheduled but not yet started: it holds its coroutine and never runs it."""

    def __init__(self, coroutine: Coroutine[Any, Any, None]) -> None:
        self.coroutine = coroutine
        self.cancels = 0

    @property
    def is_finished(self) -> bool:
        return self.cancels > 0

    def cancel(self) -> None:
        self.cancels += 1


class _FinishedWorker:
    """A worker whose coroutine already ran to completion before run_worker returned."""

    is_finished = True

    def cancel(self) -> None:
        raise AssertionError("a finished worker is never cancelled")


class _Screen:
    """Fake screen: records what the controller shows, sends and gives back."""

    def __init__(self) -> None:
        self.log: list[str] = []
        self.shown: list[CardState] = []
        self.visible: CardState | None = None
        self.agent_list = [Option(label="Code", value="Code"), Option(label="QA", value="QA")]
        self.model_list = [
            Option(label="Model A", value="model-a", model_id="ma", provider="p"),
            Option(label="Model B", value="model-b", model_id="mb", provider="p"),
        ]
        self.agent = ("Code", "Code")
        self.model = ("model-a", "Model A")
        self.running = False
        self.draft = ""
        self.restored: list[str] = []
        self.restore_accepts = True
        self.submitted: list[str] = []
        self.session: str | None = "s1"
        self.agent_result = True
        self.model_result = True
        self.switch_gate: asyncio.Event | None = None
        self.switch_entered = asyncio.Event()
        self.notices: list[tuple[MessageRef, str]] = []
        self.workers: list[_FakeWorker] = []
        self.eager_workers = False
        # Workers are scheduled but never started (Textual has not run them yet).
        self.unstarted_workers = False
        self.unstarted: list[_UnstartedWorker] = []
        self.loading = False
        self.agent_error: Exception | None = None
        # The one send in flight, as InputFlowController's claim: its number, or None.
        self.claim: int | None = None
        self.give_back: GiveBack | None = None
        # Counts restores, as the view adapter's generation: each accepted one hands out its own proof.
        self.restores = 0
        self.still_valid: Callable[[], bool] | None = None
        self.claims = 0
        self.refused: list[str] = []
        self.cwd = str(Path.cwd())
        # (claim, keep_card) for every abandon that released its own claim.
        self.abandons: list[int] = []

    def callbacks(self) -> PromptCardCallbacks:
        return PromptCardCallbacks(
            show=self._show,
            hide=self._hide,
            agents=lambda: self.agent_list,
            models=lambda: self.model_list,
            current_agent=lambda: self.agent,
            current_model=lambda: self.model,
            agent_running=lambda: self.running,
            agent_loading=lambda: self.loading,
            input_text=lambda: self.draft,
            workspace_cwd=lambda: self.cwd,
            restore_draft=self._restore,
            begin_prepared=self._begin_prepared,
            holds=lambda prepared: self.claim == prepared.claim,
            publish_prepared=self._publish_prepared,
            abandon_prepared=self._abandon_prepared,
            session_id=lambda: self.session,
            switch_agent=self._switch_agent,
            model_switchable=lambda _value: True,
            switch_model=self._switch_model,
            notify=self._notify,
            run_worker=self._run_worker,
            after_refresh=self._after_refresh,
        )

    def _show(self, state: CardState) -> None:
        self.shown.append(state)
        self.visible = state

    def _hide(self) -> None:
        self.visible = None

    def _restore(self, text: str) -> InputRestore | None:
        # As the view adapter's restore_input_text: a draft already in the input wins,
        # and an accepted restore hands back its proof.
        self.restored.append(text)
        self.restores += 1
        if not self.restore_accepts or self.draft:
            return None
        self.draft = text
        return InputRestore(generation=self.restores, text=text)

    def _begin_prepared(
        self, text: str, intent: SendIntent, give_back: GiveBack, still_valid: Callable[[], bool]
    ) -> PreparedSend | None:
        # As InputFlowController.begin_prepared: the claim keeps the send's own give-back and check.
        if self.claim is not None:
            # Refused like any send while another is in flight: the draft comes back through its give-back.
            self.refused.append(text)
            give_back()
            return None
        self.claims += 1
        self.claim = self.claims
        self.give_back = give_back
        self.still_valid = still_valid
        return PreparedSend(self.claims, text, intent)

    def _release(self, claim: int) -> GiveBack | None:
        if self.claim != claim:
            return None
        give_back = self.give_back
        self.claim = None
        self.give_back = None
        self.still_valid = None
        return give_back

    def _publish_prepared(self, prepared: PreparedSend, final_text: str) -> bool:
        # As InputFlowController.publish_prepared, with its worker run at once:
        # it checks once more right before publishing and gives back instead.
        if self.claim != prepared.claim:
            return False
        if self.loading or (self.still_valid is not None and not self.still_valid()):
            give_back = self._release(prepared.claim)
            self.abandons.append(prepared.claim)
            if give_back is not None:
                give_back()
            return True
        self._release(prepared.claim)
        if prepared.intent == "continue":
            self.log.append(f"continue:{final_text}")
        else:
            self.log.append(f"deliver:{final_text}")
        self.submitted.append(final_text)
        return True

    def _abandon_prepared(self, prepared: PreparedSend) -> bool:
        give_back = self._release(prepared.claim)
        if give_back is None:
            return False
        self.abandons.append(prepared.claim)
        return give_back() is not None

    async def _switch_agent(self, value: str) -> bool:
        self.log.append(f"agent:{value}")
        self.switch_entered.set()
        if self.agent_error is not None:
            raise self.agent_error
        if self.switch_gate is not None:
            await self.switch_gate.wait()
        return self.agent_result

    async def _switch_model(self, value: str) -> bool:
        self.log.append(f"model:{value}")
        self.switch_entered.set()
        if self.switch_gate is not None:
            await self.switch_gate.wait()
        return self.model_result

    def _notify(self, message: MessageRef, _title: MessageRef, severity: str) -> None:
        self.notices.append((message, severity))

    def _run_worker(self, coroutine: Coroutine[Any, Any, None]) -> Any:
        if self.eager_workers:
            with pytest.raises(StopIteration):
                coroutine.send(None)
            return _FinishedWorker()
        if self.unstarted_workers:
            unstarted = _UnstartedWorker(coroutine)
            self.unstarted.append(unstarted)
            return unstarted
        worker = _FakeWorker(asyncio.get_running_loop().create_task(coroutine))
        self.workers.append(worker)
        return worker

    def _after_refresh(self, callback: Callable[[], None]) -> object:
        return asyncio.get_running_loop().call_soon(callback)

    async def drain(self) -> None:
        for worker in self.workers:
            await asyncio.wait_for(asyncio.gather(worker.task, return_exceptions=True), ENGINE_TURN_TIMEOUT)


def _command(*, agent: str = "", model: str = "", name: str = "zz") -> PromptShortcut:
    return PromptShortcut(name=name, template="", path=f"/x/{name}.md", agent=agent, model=model)


@pytest.fixture
def screen() -> _Screen:
    return _Screen()


@pytest.fixture
def posix_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHELL", "/bin/sh")


def _attach(screen: _Screen, command: PromptShortcut, text: str) -> PromptCardController:
    controller = PromptCardController(screen.callbacks())
    screen.draft = text
    controller.attach(command, text)
    return controller


def _submit(screen: _Screen, controller: PromptCardController, text: str) -> bool:
    # The composer clears its draft as it submits.
    screen.draft = ""
    return controller.submit(text)


# -- C1/C2: when the card shows ----------------------------------------------


@pytest.mark.parametrize(
    ("agent", "model"),
    [("QA", ""), ("", "model-b"), ("QA", "Model B"), ("Nope", ""), ("", "no-such-model")],
    ids=["agent", "model", "both", "agent-not-found", "model-not-found"],
)
async def test_card_shows_when_frontmatter_differs_or_is_not_found(screen: _Screen, agent: str, model: str) -> None:
    controller = _attach(screen, _command(agent=agent, model=model), "hello")
    assert controller.active
    assert screen.visible is not None
    assert (screen.visible.agent is not None) == bool(agent)
    assert (screen.visible.model is not None) == bool(model)


@pytest.mark.parametrize(
    ("agent", "model"),
    [("Code", ""), ("", "model-a"), ("Code", "Model A"), ("", "p/ma"), ("", "")],
    ids=["same-agent", "same-model-id", "same-both", "same-model-provider-form", "nothing"],
)
async def test_same_agent_and_model_as_the_session_show_no_card(screen: _Screen, agent: str, model: str) -> None:
    controller = _attach(screen, _command(agent=agent, model=model), "hello")
    assert screen.shown == []
    assert screen.visible is None
    assert not controller.sending
    assert controller.submit("hello") is False


async def test_same_agent_different_model_shows_only_the_model_row(screen: _Screen) -> None:
    _attach(screen, _command(agent="Code", model="model-b"), "hello")
    assert screen.visible is not None
    assert screen.visible.agent is None
    assert screen.visible.model is not None


@posix_only
@pytest.mark.usefixtures("posix_shell")
async def test_commands_only_shows_their_count_and_runs_them_on_send(screen: _Screen) -> None:
    text = "Status: !`echo clean`"
    controller = _attach(screen, _command(), text)
    assert screen.visible is not None
    assert (screen.visible.agent, screen.visible.model, screen.visible.commands) == (None, None, 1)

    assert _submit(screen, controller, text) is True
    await screen.drain()

    assert screen.submitted == ["Status: clean"]
    assert screen.log == ["deliver:Status: clean"]
    assert screen.visible is None
    assert not controller.sending


# -- C3/C4/C11: accepting -----------------------------------------------------


async def test_send_switches_agent_then_model_then_delivers(screen: _Screen) -> None:
    controller = _attach(screen, _command(agent="QA", model="Model B"), "do it")
    assert _submit(screen, controller, "do it") is True
    await screen.drain()

    assert screen.log == ["agent:QA", "model:model-b", "deliver:do it"]
    assert screen.restored == []
    assert not controller.active
    assert screen.visible is None


@posix_only
@pytest.mark.usefixtures("posix_shell")
async def test_send_with_commands_and_switch_delivers_substituted_text(screen: _Screen) -> None:
    text = "Branch: !`echo main`"
    controller = _attach(screen, _command(agent="QA"), text)
    assert _submit(screen, controller, text) is True
    await screen.drain()

    assert screen.log == ["agent:QA", "deliver:Branch: main"]


async def test_switch_uses_the_frontmatter_target_not_the_display_name(screen: _Screen) -> None:
    screen.agent_list = [Option(label="Coder", value="Code"), Option(label="Reviewer", value="QA")]
    screen.agent = ("Code", "Coder")
    controller = _attach(screen, _command(agent="Reviewer", model="p/mb"), "go")
    _submit(screen, controller, "go")
    await screen.drain()

    assert screen.log == ["agent:QA", "model:model-b", "deliver:go"]


@pytest.mark.parametrize(
    ("agent", "model", "expected"),
    [
        ("Nope", "model-b", ["model:model-b", "deliver:go"]),
        ("QA", "no-such", ["agent:QA", "deliver:go"]),
    ],
    ids=["agent-missing", "model-missing"],
)
async def test_items_not_found_are_not_switched_and_the_message_is_sent(
    screen: _Screen, agent: str, model: str, expected: list[str]
) -> None:
    controller = _attach(screen, _command(agent=agent, model=model), "go")
    _submit(screen, controller, "go")
    await screen.drain()

    assert screen.log == expected
    assert not controller.active


async def test_a_card_naming_only_what_is_not_found_is_sent_plainly(screen: _Screen) -> None:
    controller = _attach(screen, _command(agent="Nope", model="no-such"), "go")
    assert screen.visible is not None

    # Nothing to switch and no commands: the caller sends it as any draft, under a claim of its own.
    assert _submit(screen, controller, "go") is False
    assert screen.claims == 0
    assert screen.workers == []
    assert screen.log == []


async def test_continue_intent_switches_then_publishes_a_continue(screen: _Screen) -> None:
    controller = _attach(screen, _command(agent="QA"), "retry me")
    screen.draft = ""
    assert controller.submit("retry me", intent="continue") is True
    await screen.drain()

    assert screen.log == ["agent:QA", "continue:retry me"]
    assert screen.claim is None


async def test_continue_intent_while_a_turn_runs_is_left_to_the_claim(screen: _Screen) -> None:
    """A message while a turn runs is refused by the card; a continue is the claim's to refuse."""
    controller = _attach(screen, _command(agent="QA"), "retry me")
    screen.running = True
    screen.draft = ""
    assert controller.submit("retry me", intent="continue") is True
    # The fake claim takes it; InputFlowController.begin_prepared refuses a continue while a turn runs.
    assert screen.claims == 1
    assert screen.notices == []


# -- C5: failures send nothing and give the draft back ------------------------


def _assert_given_back(screen: _Screen, controller: PromptCardController, text: str, before: CardState) -> None:
    assert screen.submitted == []
    assert screen.restored == [text]
    assert screen.draft == text
    assert not controller.sending
    assert controller.active
    assert screen.visible is not None
    assert not screen.visible.in_progress
    assert not screen.visible.running
    assert screen.visible.commands == before.commands
    assert screen.visible.agent == before.agent
    assert screen.visible.model == before.model


async def test_failed_agent_switch_sends_nothing_and_restores(screen: _Screen) -> None:
    text = "go"
    controller = _attach(screen, _command(agent="QA", model="model-b"), text)
    before = screen.visible
    assert before is not None
    screen.agent_result = False
    _submit(screen, controller, text)
    await screen.drain()

    assert screen.log == ["agent:QA"]
    _assert_given_back(screen, controller, text, before)
    assert any(severity == "error" for _message, severity in screen.notices)


async def test_failed_model_switch_sends_nothing_and_restores(screen: _Screen) -> None:
    text = "go"
    controller = _attach(screen, _command(model="model-b"), text)
    before = screen.visible
    assert before is not None
    screen.model_result = False
    _submit(screen, controller, text)
    await screen.drain()

    assert screen.log == ["model:model-b"]
    _assert_given_back(screen, controller, text, before)


@pytest.mark.usefixtures("posix_shell")
async def test_command_that_cannot_start_sends_nothing_and_restores(
    screen: _Screen, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = str(tmp_path / "no-such-shell")
    monkeypatch.setattr(shell_module, "_shell_argv", lambda command: [missing, "-c", command])
    text = "x !`echo hi`"
    controller = _attach(screen, _command(agent="QA"), text)
    before = screen.visible
    assert before is not None
    _submit(screen, controller, text)
    await screen.drain()

    assert screen.log == []
    _assert_given_back(screen, controller, text, before)
    assert screen.notices


@posix_only
@pytest.mark.usefixtures("posix_shell")
async def test_command_timeout_sends_nothing_and_restores(screen: _Screen, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_module, "SHELL_TIMEOUT_SECONDS", 0.5)
    text = "x !`sleep 30`"
    controller = _attach(screen, _command(agent="QA"), text)
    before = screen.visible
    assert before is not None
    _submit(screen, controller, text)
    await screen.drain()

    assert screen.log == []
    _assert_given_back(screen, controller, text, before)


async def test_cancel_while_the_commands_run_sends_nothing_and_restores(
    monkeypatch: pytest.MonkeyPatch, screen: _Screen
) -> None:
    shell = _spy_shell(monkeypatch, screen)
    shell.gate = asyncio.Event()
    controller = _attach(screen, _command(agent="QA", model="model-b"), _DRAFT)
    before = screen.visible
    assert before is not None
    _submit(screen, controller, _DRAFT)
    await wait_for(shell.entered.is_set, description="send is running its commands")
    assert screen.visible is not None
    assert screen.visible.running

    assert controller.cancel() is True
    await screen.drain()

    assert screen.workers[0].task.cancelled()
    # Stopped before any switch.
    assert screen.log == ["shell"]
    _assert_given_back(screen, controller, _DRAFT, before)


@pytest.mark.parametrize("commands", [True, False], ids=["after-its-commands", "no-commands"])
async def test_esc_while_a_send_switches_is_taken_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, screen: _Screen, commands: bool
) -> None:
    _spy_shell(monkeypatch, screen)
    text = _DRAFT if commands else "go"
    controller = _attach(screen, _command(agent="QA"), text)
    screen.switch_gate = asyncio.Event()
    _submit(screen, controller, text)
    await wait_for(screen.switch_entered.is_set, description="send reached the agent switch")
    # The commands ran: the card no longer offers to cancel them.
    assert screen.visible is not None
    assert (screen.visible.running, screen.visible.commands, screen.visible.in_progress) == (False, 0, True)

    # Esc is the send's, so it reaches nothing behind it, and the switch goes on.
    assert controller.cancel() is True
    assert controller.sending
    assert screen.notices == []
    screen.switch_gate.set()
    await screen.drain()

    assert not screen.workers[0].task.cancelled()
    assert screen.submitted == ["x hi" if commands else "go"]
    assert screen.abandons == []


async def test_cancel_with_nothing_sending_returns_false(screen: _Screen) -> None:
    controller = _attach(screen, _command(agent="QA"), "go")
    assert controller.cancel() is False


async def test_failure_after_user_typed_a_new_draft_keeps_the_new_draft(screen: _Screen) -> None:
    controller = _attach(screen, _command(agent="QA"), "go")
    screen.agent_result = False
    screen.restore_accepts = False
    _submit(screen, controller, "go")
    await screen.drain()

    assert screen.submitted == []
    assert not controller.sending


# -- C6: while a send is in progress -----------------------------------------


async def test_second_submit_while_sending_is_refused_and_given_back(screen: _Screen) -> None:
    controller = _attach(screen, _command(agent="QA", model="model-b"), "first")
    screen.switch_gate = asyncio.Event()
    _submit(screen, controller, "first")
    await wait_for(screen.switch_entered.is_set, description="send reached the agent switch")
    shown_before = list(screen.shown)

    # The draft went with the send: "second" has no card, so the caller sends it plainly, and the claim refuses it.
    assert _submit(screen, controller, "second") is False
    assert screen.claims == 1
    assert len(screen.workers) == 1

    controller.drop_switch()
    assert screen.shown == shown_before
    assert screen.visible is not None
    assert screen.visible.agent is not None
    assert controller.sending

    screen.switch_gate.set()
    await screen.drain()
    assert screen.log == ["agent:QA", "model:model-b", "deliver:first"]
    assert screen.submitted == ["first"]


# -- C7: ✕ ---------------------------------------------------------------------


@posix_only
@pytest.mark.usefixtures("posix_shell")
async def test_drop_switch_keeps_the_commands(screen: _Screen) -> None:
    text = "Now: !`echo ok`"
    controller = _attach(screen, _command(agent="QA", model="model-b"), text)
    assert screen.visible is not None

    controller.drop_switch()
    assert screen.visible is None or (screen.visible.agent is None and screen.visible.model is None)

    assert _submit(screen, controller, text) is True
    await screen.drain()
    assert screen.log == ["deliver:Now: ok"]


async def test_drop_switch_without_commands_hides_the_card_and_keeps_the_draft_the_shortcuts(
    screen: _Screen,
) -> None:
    controller = _attach(screen, _command(agent="QA"), "plain")
    controller.drop_switch()
    assert controller.active
    assert screen.visible is None
    assert controller.submit("plain") is False
    assert screen.log == []


# -- C8: session changed mid-send --------------------------------------------


async def test_session_change_during_send_delivers_nothing(screen: _Screen) -> None:
    text = "go"
    controller = _attach(screen, _command(agent="QA", model="model-b"), text)
    screen.switch_gate = asyncio.Event()
    _submit(screen, controller, text)
    await wait_for(screen.switch_entered.is_set, description="send reached the agent switch")

    screen.session = "s2"
    screen.switch_gate.set()
    await screen.drain()

    assert screen.submitted == []
    assert screen.notices
    assert screen.restored == [text]
    assert not controller.active
    assert not controller.sending
    assert screen.visible is None


@posix_only
@pytest.mark.usefixtures("posix_shell")
async def test_session_change_while_a_command_runs_delivers_nothing(screen: _Screen, tmp_path: Path) -> None:
    go = tmp_path / "go"
    text = f"x !`while [ ! -f '{go}' ]; do sleep 0.05; done; echo ok`"
    controller = _attach(screen, _command(agent="QA"), text)
    _submit(screen, controller, text)
    await wait_for(lambda: screen.visible is not None and screen.visible.in_progress, description="send in progress")

    screen.session = "s2"
    go.write_text("", encoding="utf-8")
    await screen.drain()

    assert screen.submitted == []
    assert "agent:QA" not in screen.log
    assert screen.restored == [text]
    assert not controller.active
    assert screen.visible is None


# -- C9: agent running ---------------------------------------------------------


async def test_submit_while_agent_runs_refuses_with_a_notice(screen: _Screen) -> None:
    controller = _attach(screen, _command(agent="QA"), "go")
    screen.running = True

    assert controller.submit("go") is True

    assert screen.workers == []
    assert not controller.sending
    assert screen.log == []
    assert len(screen.notices) == 1
    assert controller.active


# -- C10: nothing to wait for ---------------------------------------------------


@pytest.mark.parametrize("eager", [False, True], ids=["scheduled", "finished-before-return"])
async def test_send_with_nothing_to_wait_for_does_not_stay_sending(screen: _Screen, eager: bool) -> None:
    screen.eager_workers = eager
    controller = _attach(screen, _command(agent="QA"), "go")
    assert controller.active
    assert _submit(screen, controller, "go") is True
    await screen.drain()

    assert screen.submitted == ["go"]
    assert screen.claim is None
    assert not controller.sending
    assert not controller.active

    controller.attach(_command(agent="QA"), "again")
    assert controller.active
    assert screen.visible is not None
    assert screen.visible.agent is not None


def _notice_keys(screen: _Screen) -> list[str]:
    return [message.definition.key for message, _severity in screen.notices]


# -- agent loading -------------------------------------------------------------


async def test_submit_while_agent_loads_runs_nothing_and_gives_the_text_back(screen: _Screen) -> None:
    controller = _attach(screen, _command(agent="QA", model="model-b"), "go")
    before = screen.visible
    assert before is not None
    screen.loading = True

    assert _submit(screen, controller, "go") is True

    assert screen.workers == []
    assert screen.log == []
    assert screen.submitted == []
    assert screen.restored == ["go"]
    assert screen.draft == "go"
    assert not controller.sending
    assert controller.active
    assert screen.visible == before


@pytest.mark.parametrize("loading", [True, False], ids=["agent-loading", "agent-ready"])
async def test_a_send_ending_unpublished_gives_back_only_through_its_abandon(screen: _Screen, loading: bool) -> None:
    text = "go"
    controller = _attach(screen, _command(agent="QA", model="model-b"), text)
    before = screen.visible
    assert before is not None
    screen.switch_gate = asyncio.Event()
    _submit(screen, controller, text)
    await wait_for(screen.switch_entered.is_set, description="send reached the agent switch")

    # The switch fails; the agent may be starting again by then.
    screen.agent_result = False
    screen.loading = loading
    screen.switch_gate.set()
    await screen.drain()

    # One abandon of its own claim; the text comes back once, through the send's give-back.
    assert screen.abandons == [1]
    assert screen.claim is None
    _assert_given_back(screen, controller, text, before)
    assert ("tui.prompt_card.loading" in _notice_keys(screen)) is loading


async def test_a_sends_give_back_returns_the_draft_at_most_once(screen: _Screen) -> None:
    text = "go"
    controller = _attach(screen, _command(agent="QA"), text)
    screen.switch_gate = asyncio.Event()
    _submit(screen, controller, text)
    await wait_for(screen.switch_entered.is_set, description="send reached the agent switch")
    give_back = screen.give_back
    assert give_back is not None

    # The first call restores the draft and hands back the proof of that restore.
    proof = give_back()
    assert proof is not None
    assert proof.text == text
    assert screen.draft == text
    assert controller.active
    screen.draft = ""
    # Afterwards it gives nothing back, and no proof.
    assert give_back() is None
    assert screen.draft == ""

    # The send's own end gives nothing back a second time.
    screen.agent_result = False
    assert screen.switch_gate is not None
    screen.switch_gate.set()
    await screen.drain()
    assert screen.restored == [text]
    assert screen.abandons == [1]
    assert screen.submitted == []


async def test_a_sends_give_back_hands_no_proof_when_a_draft_typed_meanwhile_wins(screen: _Screen) -> None:
    text = "go"
    controller = _attach(screen, _command(agent="QA"), text)
    screen.switch_gate = asyncio.Event()
    _submit(screen, controller, text)
    await wait_for(screen.switch_entered.is_set, description="send reached the agent switch")
    give_back = screen.give_back
    assert give_back is not None

    # The user typed something new while the send ran: that draft is theirs.
    screen.draft = "typed meanwhile"
    assert give_back() is None
    assert screen.draft == "typed meanwhile"
    # Spent: even once the composer is empty again, nothing comes back.
    screen.draft = ""
    assert give_back() is None
    assert screen.draft == ""
    assert screen.restored == [text]

    screen.agent_result = False
    assert screen.switch_gate is not None
    screen.switch_gate.set()
    await screen.drain()


# -- cancel before the worker starts --------------------------------------------


def test_cancel_before_the_worker_starts_ends_the_send_itself(screen: _Screen) -> None:
    text = "go"
    controller = _attach(screen, _command(agent="QA"), text)
    before = screen.visible
    assert before is not None
    screen.unstarted_workers = True
    _submit(screen, controller, text)
    assert controller.sending
    assert screen.claim == 1
    [worker] = screen.unstarted
    coroutine = worker.coroutine

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert controller.cancel() is True
        assert worker.cancels == 1
        # Ended here: its claim released by one abandon, the draft back once with its card, and the user told.
        assert not controller.sending
        assert screen.claim is None
        assert screen.abandons == [1]
        _assert_given_back(screen, controller, text, before)
        assert _notice_keys(screen) == ["tui.prompt_card.cancelled"]
        # The unrun send is closed, never left to be collected unawaited; it switched nothing.
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
        assert screen.log == []
        del coroutine, worker
        screen.unstarted.clear()
        gc.collect()
    assert not [w for w in caught if "never awaited" in str(w.message)]

    # Nothing to cancel now; a second cancel gives nothing back again.
    assert controller.cancel() is False
    assert screen.restored == [text]


async def test_a_send_after_one_cancelled_before_it_started_goes_out(screen: _Screen) -> None:
    text = "go"
    controller = _attach(screen, _command(agent="QA"), text)
    screen.unstarted_workers = True
    _submit(screen, controller, text)
    assert controller.cancel() is True
    assert screen.draft == text
    screen.unstarted.clear()

    screen.unstarted_workers = False
    assert _submit(screen, controller, text) is True
    await screen.drain()

    assert screen.log == ["agent:QA", f"deliver:{text}"]
    assert screen.submitted == [text]
    assert screen.claims == 2
    assert screen.abandons == [1]
    assert not controller.sending


async def test_cancel_after_the_worker_started_lets_its_own_cleanup_end_it_once(
    monkeypatch: pytest.MonkeyPatch, screen: _Screen
) -> None:
    shell = _spy_shell(monkeypatch, screen)
    shell.gate = asyncio.Event()
    text = _DRAFT
    controller = _attach(screen, _command(agent="QA"), text)
    before = screen.visible
    assert before is not None
    _submit(screen, controller, text)
    await wait_for(shell.entered.is_set, description="send is running its commands")

    assert controller.cancel() is True
    # The cancel reaches the running worker; until it runs, the send is still the worker's to end.
    assert controller.sending
    await screen.drain()

    assert screen.abandons == [1]
    _assert_given_back(screen, controller, text, before)
    assert _notice_keys(screen).count("tui.prompt_card.cancelled") == 1
    assert screen.submitted == []


# -- session changed after the agent switch ------------------------------------


async def test_session_change_after_agent_switch_skips_the_model_switch(screen: _Screen) -> None:
    text = "go"
    controller = _attach(screen, _command(agent="QA", model="model-b"), text)
    screen.switch_gate = asyncio.Event()
    _submit(screen, controller, text)
    await wait_for(screen.switch_entered.is_set, description="send reached the agent switch")

    # The agent switch succeeds, but by the time it returns the chat is another session.
    screen.session = "s2"
    screen.switch_gate.set()
    await screen.drain()

    assert screen.log == ["agent:QA"]
    assert screen.submitted == []
    assert "tui.prompt_card.session_changed" in _notice_keys(screen)


# -- unexpected failure during a send -----------------------------------------


async def test_unexpected_error_during_send_notifies_and_restores(screen: _Screen) -> None:
    text = "go"
    controller = _attach(screen, _command(agent="QA", model="model-b"), text)
    before = screen.visible
    assert before is not None
    screen.agent_error = RuntimeError("boom")
    _submit(screen, controller, text)
    await screen.drain()

    assert screen.log == ["agent:QA"]
    _assert_given_back(screen, controller, text, before)
    assert ("tui.prompt_card.send_failed", "error") in [
        (message.definition.key, severity) for message, severity in screen.notices
    ]


# -- the session's agent or model changed under the card ----------------------


@posix_only
@pytest.mark.usefixtures("posix_shell")
async def test_session_choice_change_drops_switch_lines_and_keeps_commands(screen: _Screen) -> None:
    text = "Now: !`echo ok`"
    controller = _attach(screen, _command(agent="QA", model="model-b"), text)
    assert screen.visible is not None

    # The user picks another agent themselves while the card shows.
    screen.agent = ("Explore", "Explore")
    controller.on_user_pick()
    assert screen.visible is None or (screen.visible.agent is None and screen.visible.model is None)

    assert _submit(screen, controller, text) is True
    await screen.drain()
    assert screen.log == ["deliver:Now: ok"]


async def test_session_choice_change_without_commands_hides_the_card(screen: _Screen) -> None:
    controller = _attach(screen, _command(model="model-b"), "plain")
    assert screen.visible is not None

    screen.model = ("model-c", "Model C")
    controller.on_user_pick()

    assert controller.active
    assert screen.visible is None
    assert controller.submit("plain") is False
    assert screen.log == []


async def test_a_session_change_that_is_not_the_users_pick_keeps_the_switch(screen: _Screen) -> None:
    controller = _attach(screen, _command(agent="QA", model="model-b"), "plain")

    # A settings reload or a restore moves the session to what the card declares: a send has nothing to switch,
    # so the caller sends it plainly; the declaration stays with the draft.
    screen.agent = ("QA", "QA")
    screen.model = ("model-b", "Model B")
    assert controller.submit("plain") is False
    assert controller.active
    assert screen.claims == 0

    # ...and back: the declaration was kept, so a send switches for it again.
    screen.agent = ("Code", "Code")
    screen.model = ("model-a", "Model A")
    assert _submit(screen, controller, "plain") is True
    await screen.drain()
    assert screen.log == ["agent:QA", "model:model-b", "deliver:plain"]


async def test_session_choice_change_while_sending_changes_nothing(screen: _Screen) -> None:
    controller = _attach(screen, _command(agent="QA", model="model-b"), "first")
    screen.switch_gate = asyncio.Event()
    _submit(screen, controller, "first")
    await wait_for(screen.switch_entered.is_set, description="send reached the agent switch")
    shown_before = list(screen.shown)

    # The user picks while a send runs: the send carries its own declaration.
    controller.on_user_pick()

    assert screen.shown == shown_before
    assert screen.visible is not None
    assert screen.visible.agent is not None
    assert controller.sending

    screen.switch_gate.set()
    await screen.drain()
    assert screen.log == ["agent:QA", "model:model-b", "deliver:first"]


@posix_only
@pytest.mark.usefixtures("posix_shell")
async def test_session_choice_change_without_switch_lines_keeps_the_commands(screen: _Screen) -> None:
    text = "Status: !`echo clean`"
    controller = _attach(screen, _command(), text)
    shown_before = list(screen.shown)

    controller.on_user_pick()

    assert screen.shown == shown_before
    assert _submit(screen, controller, text) is True
    await screen.drain()
    assert screen.log == ["deliver:Status: clean"]


# -- a draft expanded while another send is in progress ------------------------


class _ApplyingScreen(_Screen):
    """A fake screen whose switches take effect; as on MainScreen, the card's own switches are no user pick."""

    async def _switch_agent(self, value: str) -> bool:
        result = await super()._switch_agent(value)
        if result:
            self.agent = next((o.value, o.label) for o in self.agent_list if o.value == value)
        return result

    async def _switch_model(self, value: str) -> bool:
        result = await super()._switch_model(value)
        if result:
            self.model = next((o.value, o.label) for o in self.model_list if o.value == value)
        return result


def _applying_controller(screen: _ApplyingScreen) -> PromptCardController:
    return PromptCardController(screen.callbacks())


async def _first_send_held(screen: _Screen, controller: PromptCardController, command: PromptShortcut) -> None:
    """Expand and send a first draft whose agent switch the test holds open."""
    screen.draft = "first"
    controller.attach(command, "first")
    screen.switch_gate = asyncio.Event()
    _submit(screen, controller, "first")
    await wait_for(screen.switch_entered.is_set, description="first send reached the agent switch")
    assert controller.sending


@posix_only
@pytest.mark.usefixtures("posix_shell")
async def test_draft_expanded_during_a_send_keeps_its_card_and_sends_it_after() -> None:
    screen = _ApplyingScreen()
    controller = _applying_controller(screen)
    await _first_send_held(screen, controller, _command(agent="QA", name="first"))

    second = "Second: !`echo two`"
    screen.draft = second
    controller.attach(_command(agent="Code", model="model-b", name="second"), second)

    # The draft has its card, but the card shows the send until it ends.
    assert controller.active
    assert screen.visible is not None
    assert screen.visible.in_progress
    assert screen.visible.name == "first"

    screen.switch_gate.set()  # type: ignore[union-attr]
    await screen.drain()
    assert screen.submitted == ["first"]
    assert not controller.sending

    # The second draft's card, measured against the session the first send left (agent QA).
    assert controller.active
    assert screen.visible is not None
    assert not screen.visible.in_progress
    assert screen.visible.name == "second"
    assert screen.visible.agent is not None
    assert screen.visible.agent.value == "Code"
    assert screen.visible.model is not None
    assert screen.visible.model.value == "model-b"

    screen.switch_gate = None
    screen.log.clear()
    assert _submit(screen, controller, second) is True
    await screen.drain()
    assert screen.log == ["agent:Code", "model:model-b", "deliver:Second: two"]
    assert not controller.active
    assert screen.visible is None


@pytest.mark.parametrize("failure", ["agent", "model"], ids=["agent-switch-failed", "model-switch-failed"])
async def test_failed_send_keeps_the_draft_expanded_during_it(failure: str) -> None:
    screen = _Screen()
    controller = PromptCardController(screen.callbacks())
    await _first_send_held(screen, controller, _command(agent="QA", model="model-b", name="first"))

    screen.draft = "second"
    controller.attach(_command(model="model-b", name="second"), "second")
    if failure == "agent":
        screen.agent_result = False
    else:
        screen.model_result = False
    screen.switch_gate.set()  # type: ignore[union-attr]
    await screen.drain()

    assert screen.submitted == []
    assert not controller.sending
    # The second draft keeps its text and its own card.
    assert screen.draft == "second"
    assert controller.active
    assert screen.visible is not None
    assert screen.visible.name == "second"
    assert screen.visible.agent is None
    assert screen.visible.model is not None
    assert screen.visible.model.value == "model-b"


@posix_only
@pytest.mark.usefixtures("posix_shell")
async def test_failed_command_keeps_the_draft_expanded_during_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_module, "SHELL_TIMEOUT_SECONDS", 0.5)
    screen = _Screen()
    controller = PromptCardController(screen.callbacks())
    first = "x !`sleep 30`"
    screen.draft = first
    controller.attach(_command(agent="QA", name="first"), first)
    _submit(screen, controller, first)
    assert controller.sending

    screen.draft = "second"
    controller.attach(_command(model="model-b", name="second"), "second")
    await screen.drain()

    assert screen.log == []
    assert "tui.prompt_card.shell_timeout" in _notice_keys(screen)
    assert screen.draft == "second"
    assert controller.active
    assert screen.visible is not None
    assert screen.visible.name == "second"
    assert screen.visible.model is not None


async def test_failed_send_with_session_changed_gives_the_text_back_without_a_card(screen: _Screen) -> None:
    text = "go"
    controller = _attach(screen, _command(agent="QA", model="model-b"), text)
    screen.switch_gate = asyncio.Event()
    screen.agent_result = False
    _submit(screen, controller, text)
    await wait_for(screen.switch_entered.is_set, description="send reached the agent switch")

    screen.session = "s2"
    screen.switch_gate.set()
    await screen.drain()

    assert screen.submitted == []
    assert screen.draft == text
    assert not controller.active
    assert screen.visible is None


@pytest.mark.parametrize("with_card", [True, False], ids=["draft-card", "plain-draft"])
async def test_submit_during_a_send_sends_nothing_and_gives_the_text_back(screen: _Screen, with_card: bool) -> None:
    controller = PromptCardController(screen.callbacks())
    await _first_send_held(screen, controller, _command(agent="QA", name="first"))
    if with_card:
        screen.draft = "second"
        controller.attach(_command(model="model-b", name="second"), "second")
        assert controller.active

    # A draft with a card is refused by the claim, which gives it back with its card; a plain one is the
    # caller's to send, and the claim refuses it the same way.
    assert _submit(screen, controller, "second") is with_card
    if with_card:
        assert screen.refused == ["second"]
        assert screen.restored == ["second"]
        assert screen.draft == "second"
    assert screen.claims == 1
    assert len(screen.workers) == 1
    assert controller.active is with_card

    screen.switch_gate.set()  # type: ignore[union-attr]
    await screen.drain()
    assert screen.submitted == ["first"]
    assert screen.log == ["agent:QA", "deliver:first"]


async def test_send_switches_do_not_drop_a_draft_expanded_during_it() -> None:
    screen = _ApplyingScreen()
    controller = _applying_controller(screen)
    await _first_send_held(screen, controller, _command(agent="QA", model="model-b", name="first"))

    # The draft wants what the send is switching to (QA), plus a model the send also switches to.
    screen.draft = "second"
    controller.attach(_command(agent="QA", model="model-a", name="second"), "second")

    screen.switch_gate.set()  # type: ignore[union-attr]
    await screen.drain()
    assert screen.submitted == ["first"]
    assert screen.agent == ("QA", "QA")
    assert screen.model == ("model-b", "Model B")

    # The session already uses QA: that row reads unchanged; the model switch survived.
    assert controller.active
    assert screen.visible is not None
    assert screen.visible.name == "second"
    assert screen.visible.agent is None
    assert screen.visible.model is not None
    assert screen.visible.model.value == "model-a"
    assert screen.visible.model.current == "Model B"


async def test_users_own_pick_after_a_send_still_drops_the_drafts_switch() -> None:
    screen = _ApplyingScreen()
    controller = _applying_controller(screen)
    await _first_send_held(screen, controller, _command(agent="QA", name="first"))
    screen.draft = "second"
    controller.attach(_command(model="model-b", name="second"), "second")
    screen.switch_gate.set()  # type: ignore[union-attr]
    await screen.drain()
    assert screen.visible is not None
    assert screen.visible.model is not None

    # Now no send runs: the user's own model pick wins over the card's switch.
    screen.model = ("model-c", "Model C")
    controller.on_user_pick()

    assert controller.active
    assert screen.visible is None


# -- send operation invariants -------------------------------------------------
#
# The shell runs through a spy standing in for the prompt_card module's
# ``interpolate_shell``, so every command the send starts is recorded with the
# working folder it was given, and a test can hold it or go stale inside it.

_PLACEHOLDER = re.compile(r"!`([^`]*)`")
_DRAFT = "x !`echo hi`"


class _ShellSpy:
    def __init__(self, screen: _Screen) -> None:
        self._screen = screen
        self.calls: list[tuple[str, str]] = []
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()
        self.on_enter: Callable[[], None] | None = None
        self.error: BaseException | None = None

    async def __call__(self, text: str, *, cwd: str) -> str:
        self.calls.append((text, cwd))
        self._screen.log.append("shell")
        self.entered.set()
        if self.on_enter is not None:
            self.on_enter()
        if self.gate is not None:
            await self.gate.wait()
        if self.error is not None:
            raise self.error
        return _PLACEHOLDER.sub(lambda match: match.group(1).removeprefix("echo "), text)


def _spy_shell(monkeypatch: pytest.MonkeyPatch, screen: _Screen) -> _ShellSpy:
    spy = _ShellSpy(screen)
    monkeypatch.setattr(prompt_card_module, "interpolate_shell", spy)
    return spy


class _HookedScreen(_ApplyingScreen):
    """Switches take effect, then run a hook the test sets: where a send can go stale."""

    def __init__(self) -> None:
        super().__init__()
        self.after_agent: Callable[[], None] | None = None
        self.after_model: Callable[[], None] | None = None

    async def _switch_agent(self, value: str) -> bool:
        result = await super()._switch_agent(value)
        if self.after_agent is not None:
            self.after_agent()
        return result

    async def _switch_model(self, value: str) -> bool:
        result = await super()._switch_model(value)
        if self.after_model is not None:
            self.after_model()
        return result


_OTHER_CLAIM = 99
_RAN_BEFORE_STALE = {
    "before_worker": [],
    "during_command": ["shell"],
    "after_agent_switch": ["shell", "agent:QA"],
    "after_model_switch": ["shell", "agent:QA", "model:model-b"],
}


def _go_stale(screen: _Screen, how: str) -> Callable[[], None]:
    def stale() -> None:
        if how == "session":
            screen.session = "s2"
        elif how == "cwd":
            screen.cwd = str(Path(screen.cwd) / "elsewhere")
        else:
            # Another send holds the claim now.
            screen.claim = _OTHER_CLAIM

    return stale


@pytest.mark.parametrize("how", ["session", "cwd", "claim"])
@pytest.mark.parametrize("point", list(_RAN_BEFORE_STALE))
async def test_a_stale_send_takes_no_further_step(monkeypatch: pytest.MonkeyPatch, point: str, how: str) -> None:
    screen = _HookedScreen()
    shell = _spy_shell(monkeypatch, screen)
    sent_from = screen.cwd
    controller = _attach(screen, _command(agent="QA", model="model-b"), _DRAFT)
    stale = _go_stale(screen, how)
    if point == "during_command":
        shell.on_enter = stale
    elif point == "after_agent_switch":
        screen.after_agent = stale
    elif point == "after_model_switch":
        screen.after_model = stale

    assert _submit(screen, controller, _DRAFT) is True
    if point == "before_worker":
        # Between Enter and the send's first step.
        assert screen.workers and not screen.workers[0].task.done()
        stale()
    await screen.drain()

    # Nothing after the point it went stale: no further command, switch or publish.
    assert screen.log == _RAN_BEFORE_STALE[point]
    assert screen.submitted == []
    assert not controller.sending
    # The command ran, if at all, in the folder the send was made from.
    assert shell.calls in ([], [(_DRAFT, sent_from)])
    if how == "claim":
        # Not its claim any more: it releases nothing, gives nothing back and says nothing about a move.
        assert screen.claim == _OTHER_CLAIM
        assert screen.abandons == []
        assert screen.restored == []
        assert "tui.prompt_card.session_changed" not in _notice_keys(screen)
    else:
        # Its own claim, released once; the draft as written comes back once, without its card.
        assert screen.claim is None
        assert screen.abandons == [1]
        assert screen.restored == [_DRAFT]
        assert "tui.prompt_card.session_changed" in _notice_keys(screen)
        assert not controller.active
        assert screen.visible is None


def _set_up_branch(screen: _HookedScreen, shell: _ShellSpy, branch: str) -> None:
    if branch == "shell_failed":
        shell.error = ShellInterpolationError("echo hi", "timeout")
    elif branch == "agent_switch_failed":
        screen.agent_result = False
    elif branch == "model_switch_failed":
        screen.model_result = False
    elif branch == "unexpected_error":
        screen.agent_error = RuntimeError("boom")
    elif branch == "stale":
        screen.after_agent = _go_stale(screen, "session")
    elif branch == "loading_at_publish":

        def start_loading() -> None:
            screen.loading = True

        screen.after_model = start_loading
    elif branch == "cancelled":
        shell.gate = asyncio.Event()
    elif branch == "refused_while_running":
        screen.running = True
    elif branch == "refused_while_loading":
        screen.loading = True
    elif branch == "claim_refused":
        screen.claim = _OTHER_CLAIM


def _reset_branch(screen: _HookedScreen, shell: _ShellSpy) -> None:
    shell.error = None
    shell.gate = None
    screen.agent_result = screen.model_result = True
    screen.agent_error = None
    screen.after_agent = screen.after_model = None
    screen.switch_gate = None
    screen.running = screen.loading = False
    if screen.claim == _OTHER_CLAIM:
        # The other send ends.
        screen.claim = None


_BRANCHES = [
    "shell_failed",
    "agent_switch_failed",
    "model_switch_failed",
    "unexpected_error",
    "cancelled",
    "stale",
    "loading_at_publish",
    "refused_while_running",
    "refused_while_loading",
    "claim_refused",
]


@pytest.mark.parametrize("branch", _BRANCHES)
async def test_a_send_that_did_not_publish_leaves_the_next_send_working(
    monkeypatch: pytest.MonkeyPatch, branch: str
) -> None:
    screen = _HookedScreen()
    shell = _spy_shell(monkeypatch, screen)
    controller = _attach(screen, _command(agent="QA", model="model-b"), _DRAFT)
    _set_up_branch(screen, shell, branch)

    assert _submit(screen, controller, _DRAFT) is True
    if branch == "cancelled":
        await wait_for(shell.entered.is_set, description="send is running its commands")
        assert controller.cancel() is True
    await screen.drain()

    assert screen.submitted == []
    assert not controller.sending
    # Given back at most once, as written; its claim, if it took one, released once.
    assert screen.restored.count(_DRAFT) == (0 if branch == "refused_while_running" else 1)
    assert len(screen.abandons) <= 1
    if branch == "claim_refused":
        assert screen.claim == _OTHER_CLAIM
    else:
        assert screen.claim is None

    _reset_branch(screen, shell)
    screen.draft = ""
    screen.log.clear()
    screen.agent = ("Code", "Code")
    screen.model = ("model-a", "Model A")
    screen.session = "s1"
    controller.attach(_command(agent="QA", model="model-b"), "again !`echo ok`")
    assert _submit(screen, controller, "again !`echo ok`") is True
    await screen.drain()

    assert screen.submitted == ["again ok"]
    assert screen.log == ["shell", "agent:QA", "model:model-b", "deliver:again ok"]
    assert screen.claim is None
    assert not controller.sending


async def test_second_draft_declaring_the_current_agent_switches_back_after_the_first_changes_it() -> None:
    screen = _ApplyingScreen()
    controller = _applying_controller(screen)
    await _first_send_held(screen, controller, _command(agent="QA", name="first"))

    # Expanded while the session still uses Code: the declaration matches the session, so nothing shows for it.
    screen.draft = "second"
    controller.attach(_command(agent="Code", name="second"), "second")
    assert controller.active
    assert screen.visible is not None
    assert screen.visible.name == "first"

    screen.switch_gate.set()  # type: ignore[union-attr]
    await screen.drain()
    assert screen.agent == ("QA", "QA")

    # Measured against the session the first send left: the second now switches back to Code.
    assert screen.visible is not None
    assert screen.visible.name == "second"
    assert screen.visible.agent is not None
    assert screen.visible.agent.value == "Code"
    assert screen.visible.agent.current == "QA"
    screen.switch_gate = None
    screen.log.clear()
    assert _submit(screen, controller, "second") is True
    await screen.drain()
    assert screen.log == ["agent:Code", "deliver:second"]


@pytest.mark.parametrize("failure", ["shell_failed", "agent_switch_failed", "unexpected_error", "cancelled", "stale"])
async def test_a_failed_send_never_takes_over_a_newer_draft(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    screen = _HookedScreen()
    shell = _spy_shell(monkeypatch, screen)
    controller = _attach(screen, _command(agent="QA", name="first"), _DRAFT)
    gate = asyncio.Event()
    shell.gate = gate
    _set_up_branch(screen, shell, failure)
    if failure == "cancelled":
        screen.switch_gate = None
    _submit(screen, controller, _DRAFT)
    await wait_for(shell.entered.is_set, description="first send running its command")

    # The user writes a new draft from another shortcut meanwhile.
    screen.draft = "second"
    controller.attach(_command(model="model-b", name="second"), "second")
    if failure == "cancelled":
        assert controller.cancel() is True
    gate.set()
    await screen.drain()

    assert screen.submitted == []
    assert screen.claim is None
    # The newer draft keeps its text and its own declaration.
    assert screen.draft == "second"
    assert controller.active
    assert screen.visible is not None
    assert screen.visible.name == "second"
    assert screen.visible.agent is None
    assert screen.visible.model is not None
    assert screen.visible.model.value == "model-b"

    _reset_branch(screen, shell)
    screen.session = "s1"
    screen.log.clear()
    assert _submit(screen, controller, "second") is True
    await screen.drain()
    assert screen.log == ["model:model-b", "deliver:second"]
