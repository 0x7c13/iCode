# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Spec tests in a real app: while a prompt shortcut's send prepares, no other way of sending reaches the model.

The send claims the one submit in flight at Enter and keeps it while its
``!`command``` runs and its agent switches. Meanwhile every entry point that
would publish a ``UserMessage`` or ``UserRetry`` is refused (its text comes
back, the status bar says a send is still in flight), and each works again
once the prepared send has ended.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pytest
from textual.widgets import Button
from textual.worker import WorkerCancelled

import chrys.app.tui.screens.main.prompt_card as prompt_card_module
from chrys.app.tui.app import ChrysApp
from chrys.app.tui.screens.agents.config import AgentsConfigScreen
from chrys.app.tui.screens.main.input_flow import PreparedSend
from chrys.app.tui.screens.main.screen import MainScreen
from chrys.app.tui.screens.models.screen import ModelConfigScreen
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chrome.input_bar import InputBar, _ChatTextArea
from chrys.app.tui.widgets.chrome.prompt_card import PromptCard
from chrys.app.tui.widgets.chrome.status_bar import STATUS_STILL_SENDING, StatusBar
from chrys.app.tui.widgets.chrome.suggestion_list import SuggestionList
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentProfileSwitch,
    AgentRuntimeDetails,
    Error,
    ProfileSwitched,
    RuntimeSkillDetails,
    UserMessage,
    UserRetry,
)
from chrys.foundation.i18n import MessageRef
from chrys.foundation.platform import get_platform
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.tui_app_harness import SessionGenerationEngine, make_chrys_app
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for

if TYPE_CHECKING:
    from textual.pilot import Pilot
    from textual.screen import Screen

type Hold = Literal["agent_switch", "command"]
type Entry = Literal[
    "enter", "send_button", "retry_button", "retry_enter", "status_continue", "skill", "slash_fallback"
]

_SKILL = "zzskill"
_AGENT_BODY = "Review this"
_COMMAND_BODY = "Status: !`echo hi`"
_PLACEHOLDER = re.compile(r"!`([^`]*)`")


def _global_shortcut(name: str, text: str) -> None:
    path = get_platform().config_dir / "shortcuts" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _card_name(main: MainScreen) -> str | None:
    draft = main._prompt_card._draft
    return None if draft is None else draft.name


def _notices(app: ChrysApp) -> list[str]:
    return [str(notification.message) for notification in app._notifications]


def _registry(tmp_path: Path) -> AgentProfileRegistry:
    registry = AgentProfileRegistry()
    registry.load_all(user_dir=tmp_path / "agents")
    return registry


class _Published:
    """What reaches the model's side of the bus: every UserMessage and UserRetry, in order."""

    def __init__(self) -> None:
        self.events: list[UserMessage | UserRetry] = []

    async def __call__(self, event: UserMessage | UserRetry) -> None:
        self.events.append(event)

    def described(self) -> list[tuple[str, str]]:
        return [(type(event).__name__, event.text) for event in self.events]


class _Gate:
    """Holds one step of the prompt shortcut's send open until released."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.commands: list[tuple[str, str]] = []


class _AgentSwitchEcho:
    """Stands in for the engine: answers an agent switch with the ProfileSwitched it would publish.

    The card's own (session-scoped) switch waits for the gate first.
    """

    def __init__(self, bus: EventBus, main: MainScreen, gate: _Gate | None) -> None:
        self._bus = bus
        self._main = main
        self._gate = gate

    async def __call__(self, event: AgentProfileSwitch) -> None:
        if self._gate is not None and event.session_scoped:
            self._gate.entered.set()
            await self._gate.release.wait()
        await self._bus.publish(
            ProfileSwitched(
                from_profile=self._main._state.runtime.profile_name,
                to_profile=event.profile_name,
                to_display_name=event.profile_name,
            )
        )


def _gated_shell(gate: _Gate | None, record: list[tuple[str, str]]) -> Callable[..., Awaitable[str]]:
    """A stand-in for ``interpolate_shell``: records each call, holds at *gate*, echoes each command's words."""

    async def interpolate(text: str, *, cwd: str) -> str:
        record.append((text, cwd))
        if gate is not None:
            gate.entered.set()
            await gate.release.wait()
        return _PLACEHOLDER.sub(lambda match: match.group(1).removeprefix("echo "), text)

    return interpolate


async def _ready(pilot: Pilot, monkeypatch: pytest.MonkeyPatch) -> tuple[MainScreen, _ChatTextArea]:
    app = pilot.app
    assert isinstance(app, ChrysApp)

    def ready() -> bool:
        screen = app._main_screen
        return (
            screen is not None
            and app.screen is screen
            and not screen._state.run.agent_loading
            and screen.query_one("#chat-input").has_focus
        )

    await wait_for(ready, pilot=pilot, description="main screen ready with chat input focused")
    main = app._main_screen
    assert main is not None
    main._state.runtime.profile_name = "Code"
    main._state.runtime.profile = "Code"
    main._state.runtime.details = AgentRuntimeDetails(
        skill_details=[RuntimeSkillDetails(name=_SKILL, description="A loaded skill")]
    )
    # The harness has no usable model; every send would stop at the model setup dialog.
    monkeypatch.setattr(main, "_model_unconfigured", lambda: False)
    main._refresh_model_indicator()
    return main, main.query_one("#chat-input", _ChatTextArea)


def _still_sending_flashed(main: MainScreen) -> bool:
    flash = main.query_one(StatusBar)._flash
    return flash is not None and isinstance(flash.text, MessageRef) and flash.text.definition is STATUS_STILL_SENDING


async def _clear_status(pilot: Pilot, main: MainScreen) -> None:
    main.query_one(StatusBar).hide()
    await pilot.pause()


async def _type_and_close_list(pilot: Pilot, main: MainScreen, chat: _ChatTextArea, text: str) -> None:
    chat.insert(text)
    await wait_for(lambda: chat.text == text, pilot=pilot, description="draft typed")
    suggestions = main.query_one(SuggestionList)
    await wait_for(
        lambda: not suggestions.is_visible or not suggestions.is_loading, pilot=pilot, description="list settled"
    )
    if suggestions.is_visible:
        await pilot.press("escape")
        await wait_for(lambda: not suggestions.is_visible, pilot=pilot, description="list dismissed")


async def _start_held_send(
    pilot: Pilot, main: MainScreen, chat: _ChatTextArea, hold: Hold, gate: _Gate
) -> tuple[str, str]:
    """Expand a shortcut and press Enter; return (draft as written, text the send publishes)."""
    name = "zzqa" if hold == "agent_switch" else "zzcmd"
    await wait_for(
        lambda: main._suggestions.resolve_typed_name(f"/{name} x").kind == "shortcut",
        pilot=pilot,
        description="shortcuts loaded",
    )
    chat.insert(f"/{name} x")
    await pilot.press("tab")
    body = _AGENT_BODY if hold == "agent_switch" else _COMMAND_BODY
    draft = f"{body}\n\nx"
    await wait_for(lambda: chat.text == draft, pilot=pilot, description="shortcut expanded")
    await wait_for(lambda: main._prompt_card.active, pilot=pilot, description="draft has its card")
    await pilot.press("enter")
    await wait_for(gate.entered.is_set, pilot=pilot, description="prepared send held")
    assert main._prompt_card.sending
    assert main._state.submit.active
    return draft, _PLACEHOLDER.sub(lambda match: match.group(1).removeprefix("echo "), draft)


async def _use_entry(pilot: Pilot, main: MainScreen, chat: _ChatTextArea, entry: Entry) -> str | None:
    """Use *entry* once, from an empty composer; return the text it consumed (None when it takes none)."""
    input_bar = main.query_one(InputBar)
    if entry in ("retry_button", "retry_enter", "status_continue"):
        input_bar.retry_mode = True
        await wait_for(lambda: input_bar.retry_mode, pilot=pilot, description="continue state")
    if entry == "enter":
        await _type_and_close_list(pilot, main, chat, "another message")
        await pilot.press("enter")
        return "another message"
    if entry == "send_button":
        await _type_and_close_list(pilot, main, chat, "another message")
        await click_when_settled(pilot, "#send-btn")
        return "another message"
    if entry == "retry_button":
        await _type_and_close_list(pilot, main, chat, "a continue note")
        await click_when_settled(pilot, "#send-btn")
        return "a continue note"
    if entry == "retry_enter":
        await _type_and_close_list(pilot, main, chat, "a continue note")
        await pilot.press("enter")
        return "a continue note"
    if entry == "status_continue":
        panel = main.query_one(ChatPanel)
        if not panel.query(".status-action-btn"):
            await panel.add_interrupted()
        await wait_for(lambda: bool(panel.query(".status-action-btn")), pilot=pilot, description="Continue shown")
        # The card's overlay sits over the transcript's last rows; the claim, not the click's landing, is under test.
        panel.query(".status-action-btn").first(Button).press()
        return None
    suggestions = main.query_one(SuggestionList)
    if entry == "skill":
        chat.insert(f"/{_SKILL}")
        await wait_for(
            lambda: (
                suggestions.is_visible
                and not suggestions.is_loading
                and suggestions._highlighted is not None
                and suggestions._contents[suggestions._highlighted].plain.split()[:1] == [f"/{_SKILL}"]
            ),
            pilot=pilot,
            description="skill row highlighted",
        )
        await pilot.press("enter")
        return f"/{_SKILL}"
    # The slash fallback: text the open list matches nothing for goes out as a message.
    chat.insert("/zz")
    await wait_for(
        lambda: suggestions.is_visible and not suggestions.is_loading, pilot=pilot, description="command list open"
    )
    chat.insert("nomatch")
    await wait_for(lambda: chat.text == "/zznomatch", pilot=pilot, description="unmatched text typed")
    await pilot.press("enter")
    return "/zznomatch"


def _expected_publish(entry: Entry, consumed: str | None) -> tuple[str, str]:
    if entry in ("retry_button", "retry_enter", "status_continue"):
        return ("UserRetry", consumed or "")
    assert consumed is not None
    return ("UserMessage", consumed)


async def _app_with_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hold: Hold, *, model_registry: ModelProfileRegistry | None = None
) -> tuple[ChrysApp, EventBus, _Gate, _Published]:
    _global_shortcut("zzqa", f"---\nagent: QA\n---\n{_AGENT_BODY}")
    _global_shortcut("zzcmd", _COMMAND_BODY)
    gate = _Gate()
    if hold == "command":
        monkeypatch.setattr(prompt_card_module, "interpolate_shell", _gated_shell(gate, gate.commands))
    bus = EventBus()
    published = _Published()
    await bus.subscribe(UserMessage, published)
    await bus.subscribe(UserRetry, published)
    app = make_chrys_app(
        tmp_path / "state",
        engine=SessionGenerationEngine(),
        agent_registry=_registry(tmp_path),
        event_bus=bus,
        model_registry=model_registry,
    )
    return app, bus, gate, published


_ENTRIES: list[Entry] = [
    "enter",
    "send_button",
    "retry_button",
    "retry_enter",
    "status_continue",
    "skill",
    "slash_fallback",
]


@pytest.mark.parametrize("hold", ["agent_switch", "command"])
@pytest.mark.parametrize("entry", _ENTRIES)
async def test_no_entry_point_publishes_while_a_prompt_send_prepares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hold: Hold, entry: Entry
) -> None:
    app, bus, gate, published = await _app_with_hold(tmp_path, monkeypatch, hold)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat = await _ready(pilot, monkeypatch)
        await bus.subscribe(AgentProfileSwitch, _AgentSwitchEcho(bus, main, gate if hold == "agent_switch" else None))
        _draft, final = await _start_held_send(pilot, main, chat, hold, gate)
        await _clear_status(pilot, main)

        consumed = await _use_entry(pilot, main, chat, entry)

        if consumed is not None:
            # The text it took comes back to the composer, which says a send is still in flight.
            await wait_for(lambda: chat.text == consumed, pilot=pilot, description="text given back")
            await wait_for(lambda: _still_sending_flashed(main), pilot=pilot, description="still sending shown")
        await pilot.pause()
        assert published.events == []
        assert main._prompt_card.sending

        gate.release.set()
        await wait_for(
            lambda: published.described() == [("UserMessage", final)], pilot=pilot, description="prepared send out"
        )
        await wait_for(lambda: not main._state.submit.active, pilot=pilot, description="claim released")

        # Each way of sending works again once the prepared send has ended.
        chat.clear()
        await wait_for(lambda: chat.text == "", pilot=pilot, description="composer empty")
        consumed = await _use_entry(pilot, main, chat, entry)
        await wait_for(
            lambda: published.described() == [("UserMessage", final), _expected_publish(entry, consumed)],
            pilot=pilot,
            description=f"{entry} published after the prepared send",
        )


async def test_the_user_can_edit_and_expand_another_shortcut_while_a_send_prepares(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, bus, gate, published = await _app_with_hold(tmp_path, monkeypatch, "agent_switch")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat = await _ready(pilot, monkeypatch)
        await bus.subscribe(AgentProfileSwitch, _AgentSwitchEcho(bus, main, gate))
        await _start_held_send(pilot, main, chat, "agent_switch", gate)

        # A second shortcut expands into the free composer and keeps a card of its own.
        chat.insert("/zzcmd y")
        await pilot.press("tab")
        await wait_for(lambda: chat.text == f"{_COMMAND_BODY}\n\ny", pilot=pilot, description="second expanded")
        await wait_for(lambda: main._prompt_card.active, pilot=pilot, description="second draft's card")
        chat.insert(" edited")
        await wait_for(lambda: chat.text.endswith(" edited"), pilot=pilot, description="draft edited")
        assert published.events == []

        gate.release.set()
        await wait_for(
            lambda: published.described() == [("UserMessage", f"{_AGENT_BODY}\n\nx")],
            pilot=pilot,
            description="first send out",
        )
        # The second draft is untouched by the first send's end, and its card survived.
        assert chat.text == f"{_COMMAND_BODY}\n\ny edited"
        assert main._prompt_card.active


async def test_backend_refusal_of_a_prepared_send_is_that_sends_and_gives_back_the_draft_as_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, bus, gate, published = await _app_with_hold(tmp_path, monkeypatch, "command")
    seen_in_publish: list[tuple[bool, str]] = []

    async def refuse(event: UserMessage) -> None:
        main = app._main_screen
        assert main is not None
        # The backend's admission answers inside the message's own publish.
        seen_in_publish.append((main._state.submit.in_publish(), main._state.submit.text))
        await bus.publish(Error(code="hook_blocked", message="hook 'guard' blocked: denied"))

    await bus.subscribe(UserMessage, refuse)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat = await _ready(pilot, monkeypatch)
        await bus.subscribe(AgentProfileSwitch, _AgentSwitchEcho(bus, main, None))
        draft, final = await _start_held_send(pilot, main, chat, "command", gate)
        gate.release.set()

        await wait_for(lambda: len(published.events) == 1, pilot=pilot, description="prepared send published")
        assert published.described() == [("UserMessage", final)]
        # Attributed to this send: inside its publish, with the draft as written (placeholders kept).
        assert seen_in_publish == [(True, draft)]
        await wait_for(lambda: chat.text == draft, pilot=pilot, description="draft as written given back")
        await wait_for(lambda: not main._state.submit.active, pilot=pilot, description="claim released")
        assert not main._state.run.agent_running
        assert main.query_one(PromptCard).state is None or not main.query_one(PromptCard).state.in_progress
        # With its card: the shortcut's declaration is the draft's again, and its commands run on the next send.
        await pilot.pause()
        assert main._prompt_card.active
        assert _card_name(main) == "zzcmd"

        # And nothing stays stuck: a plain message goes out next.
        chat.clear()
        await _type_and_close_list(pilot, main, chat, "next")
        await pilot.press("enter")
        await wait_for(lambda: len(published.events) == 2, pilot=pilot, description="next message published")
        assert published.described()[-1] == ("UserMessage", "next")


# -- the publish checks once more, right before it publishes -------------------


async def _expand_command_shortcut(pilot: Pilot, main: MainScreen, chat: _ChatTextArea) -> str:
    await wait_for(
        lambda: main._suggestions.resolve_typed_name("/zzcmd x").kind == "shortcut",
        pilot=pilot,
        description="shortcuts loaded",
    )
    chat.insert("/zzcmd x")
    await pilot.press("tab")
    draft = f"{_COMMAND_BODY}\n\nx"
    await wait_for(lambda: chat.text == draft, pilot=pilot, description="shortcut expanded")
    await wait_for(lambda: main._prompt_card.active, pilot=pilot, description="draft has its card")
    return draft


def _change_right_after_publish_is_scheduled(
    monkeypatch: pytest.MonkeyPatch, main: MainScreen, change: Callable[[], None]
) -> list[bool]:
    """Run *change* as soon as the send's publish is scheduled, before its worker runs."""
    flow = main._input_flow
    original = flow.publish_prepared
    scheduled: list[bool] = []

    def publish_prepared(prepared: PreparedSend, final_text: str) -> bool:
        result = original(prepared, final_text)
        scheduled.append(result)
        # Synchronously, in the send's own step: the publish worker has not started yet.
        change()
        return result

    monkeypatch.setattr(flow, "publish_prepared", publish_prepared)
    return scheduled


def _count_restores(monkeypatch: pytest.MonkeyPatch, main: MainScreen) -> list[str]:
    input_bar = main.query_one(InputBar)
    original = input_bar.restore_draft
    restored: list[str] = []

    def restore_draft(text: str, *, keep_card: bool = False) -> bool:
        restored.append(text)
        return original(text, keep_card=keep_card)

    monkeypatch.setattr(input_bar, "restore_draft", restore_draft)
    return restored


@pytest.mark.parametrize("intent", ["message", "continue"])
@pytest.mark.parametrize("moved", ["session", "cwd"])
async def test_a_session_or_folder_change_before_the_publish_worker_runs_sends_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, moved: str, intent: str
) -> None:
    _global_shortcut("zzcmd", _COMMAND_BODY)
    monkeypatch.setattr(prompt_card_module, "interpolate_shell", _gated_shell(None, []))
    bus = EventBus()
    published = _Published()
    await bus.subscribe(UserMessage, published)
    await bus.subscribe(UserRetry, published)
    app = make_chrys_app(tmp_path / "state", engine=SessionGenerationEngine(), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat = await _ready(pilot, monkeypatch)
        draft = await _expand_command_shortcut(pilot, main, chat)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        def change() -> None:
            if moved == "session":
                main.chat_session_id = "another-session"
            else:
                main._state.workspace_marker.current_cwd = str(elsewhere)

        scheduled = _change_right_after_publish_is_scheduled(monkeypatch, main, change)
        restored = _count_restores(monkeypatch, main)
        if intent == "continue":
            main.query_one(InputBar).retry_mode = True
        await pilot.press("enter")

        await wait_for(lambda: scheduled == [True], pilot=pilot, description="publish scheduled")
        await wait_for(lambda: not main._state.submit.active, pilot=pilot, description="claim released")
        await wait_for(lambda: chat.text == draft, pilot=pilot, description="draft given back")
        await pilot.pause()

        # Nothing reached the model's side; the draft came back once, without its card, and the user is told why.
        assert published.events == []
        assert restored == [draft]
        assert not main._prompt_card.active
        assert not main._prompt_card.sending
        assert not main._state.run.agent_running
        assert any("session or working folder changed" in notice for notice in _notices(app))


async def test_the_agent_starting_before_the_publish_worker_runs_gives_the_draft_back_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzcmd", _COMMAND_BODY)
    monkeypatch.setattr(prompt_card_module, "interpolate_shell", _gated_shell(None, []))
    bus = EventBus()
    published = _Published()
    await bus.subscribe(UserMessage, published)
    app = make_chrys_app(tmp_path / "state", engine=SessionGenerationEngine(), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat = await _ready(pilot, monkeypatch)
        draft = await _expand_command_shortcut(pilot, main, chat)

        def start_loading() -> None:
            main._state.run.agent_loading = True

        scheduled = _change_right_after_publish_is_scheduled(monkeypatch, main, start_loading)
        restored = _count_restores(monkeypatch, main)
        await pilot.press("enter")

        await wait_for(lambda: scheduled == [True], pilot=pilot, description="publish scheduled")
        await wait_for(lambda: not main._state.submit.active, pilot=pilot, description="claim released")
        await wait_for(lambda: chat.text == draft, pilot=pilot, description="draft given back")
        await pilot.pause()

        assert published.events == []
        assert restored == [draft]
        # Same session and folder: the draft keeps its card.
        assert main._prompt_card.active
        assert _card_name(main) == "zzcmd"
        assert any("the agent is still starting" in notice for notice in _notices(app))


# -- a refused image: "use paths" rewrites the draft as written -----------------


async def test_use_paths_after_a_refused_image_rewrites_the_shortcut_draft_and_keeps_its_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chrys.app.tui.screens.dialogs.vision_unsupported import VisionUnsupportedDialog

    _global_shortcut("zzimg", "Look at @shot.png: !`echo hi`")
    monkeypatch.setattr(prompt_card_module, "interpolate_shell", _gated_shell(None, []))
    bus = EventBus()
    published = _Published()
    await bus.subscribe(UserMessage, published)

    async def refuse(_event: UserMessage) -> None:
        # The backend's image check answers inside the message's own publish.
        await bus.publish(
            Error(code="vision_unsupported", message='The active model profile "Model" does not support image input.')
        )

    await bus.subscribe(UserMessage, refuse)
    app = make_chrys_app(tmp_path / "state", engine=SessionGenerationEngine(), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat = await _ready(pilot, monkeypatch)
        image = Path(main._state.workspace.current_cwd) / "shot.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n")
        await wait_for(
            lambda: main._suggestions.resolve_typed_name("/zzimg x").kind == "shortcut",
            pilot=pilot,
            description="loaded",
        )
        chat.insert("/zzimg x")
        await pilot.press("tab")
        draft = "Look at @shot.png: !`echo hi`\n\nx"
        await wait_for(lambda: chat.text == draft, pilot=pilot, description="shortcut expanded")
        await wait_for(lambda: main._prompt_card.active, pilot=pilot, description="draft has its card")

        await pilot.press("enter")

        await wait_for(
            lambda: isinstance(app.screen, VisionUnsupportedDialog) and app.screen.is_mounted,
            pilot=pilot,
            description="image rejection dialog shown",
        )
        # The send went out with its command's output; what came back is the draft as written.
        assert published.described() == [("UserMessage", "Look at @shot.png: hi\n\nx")]
        assert chat.text == draft
        assert main._prompt_card.active

        await click_when_settled(pilot, "#vision-unsupported-use-paths")

        rewritten = f"Look at {image}: !`echo hi`\n\nx"
        await wait_for(lambda: chat.text == rewritten, pilot=pilot, description="image mention rewritten to a path")
        await pilot.pause()
        # Still the shortcut's draft: its card, and so its command, stay with the rewritten text.
        assert main._prompt_card.active
        assert _card_name(main) == "zzimg"
        assert not main._state.submit.active
        assert len(published.events) == 1


# -- cancelled before its worker starts -----------------------------------------


async def test_a_send_cancelled_before_its_worker_starts_ends_and_the_next_one_goes_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzslow", "Slow: !`sleep 5`")
    commands: list[tuple[str, str]] = []
    monkeypatch.setattr(prompt_card_module, "interpolate_shell", _gated_shell(None, commands))
    bus = EventBus()
    published = _Published()
    await bus.subscribe(UserMessage, published)
    app = make_chrys_app(tmp_path / "state", engine=SessionGenerationEngine(), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat = await _ready(pilot, monkeypatch)
        await wait_for(
            lambda: main._suggestions.resolve_typed_name("/zzslow x").kind == "shortcut",
            pilot=pilot,
            description="loaded",
        )
        chat.insert("/zzslow x")
        await pilot.press("tab")
        draft = "Slow: !`sleep 5`\n\nx"
        await wait_for(lambda: chat.text == draft, pilot=pilot, description="shortcut expanded")
        await wait_for(lambda: main._prompt_card.active, pilot=pilot, description="draft has its card")
        controller = main._prompt_card

        # As Enter does: the composer gives up its draft, then the card takes the send.
        chat.load_text("")
        assert controller.submit(draft) is True
        operation = controller._operation
        assert operation is not None and operation.worker is not None
        worker = operation.worker
        # Synchronously, before Textual has run the worker.
        assert controller.cancel() is True
        assert not controller.sending
        # The worker ends cancelled, never having run the send.
        with pytest.raises(WorkerCancelled):
            await asyncio.wait_for(worker.wait(), ENGINE_TURN_TIMEOUT)

        assert not controller.sending
        assert not main._state.submit.active
        await wait_for(lambda: chat.text == draft, pilot=pilot, description="draft given back")
        await pilot.pause()
        assert main._prompt_card.active
        assert _card_name(main) == "zzslow"
        assert commands == []
        assert published.events == []
        assert sum("cancelled" in notice.casefold() for notice in _notices(app)) == 1

        # A later send of the same draft goes out.
        await pilot.press("enter")
        await wait_for(
            lambda: published.described() == [("UserMessage", "Slow: sleep 5\n\nx")],
            pilot=pilot,
            description="next send published",
        )
        assert [text for text, _cwd in commands] == [draft]


@pytest.mark.parametrize(
    ("key", "dialog", "notice"),
    [("f4", ModelConfigScreen, "#mc-read-only-notice"), ("f2", AgentsConfigScreen, "#ac-read-only-notice")],
    ids=["models", "agents"],
)
async def test_a_config_dialog_opened_while_a_send_prepares_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, dialog: type[Screen[str]], notice: str
) -> None:
    """A dialog reads the session once, as it opens, and the send then switches the model and starts its
    turn under it: left editable, the model just switched to could be deleted with no reload."""
    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-a", name="Model A", model_id="wire-a"))
    app, _bus, gate, published = await _app_with_hold(tmp_path, monkeypatch, "command", model_registry=registry)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat = await _ready(pilot, monkeypatch)

        async def opens_read_only() -> bool:
            await pilot.press(key)
            await wait_for(
                lambda: isinstance(app.screen, dialog) and bool(app.screen.query(notice)),
                pilot=pilot,
                description="dialog open",
            )
            shown = app.screen.query_one(notice).display
            await pilot.press("escape")
            await wait_for(lambda: app.screen is main and chat.has_focus, pilot=pilot, description="dialog closed")
            return shown

        # Nothing in flight: the same dialog is editable.
        assert not await opens_read_only()

        _draft, final = await _start_held_send(pilot, main, chat, "command", gate)
        assert await opens_read_only()

        gate.release.set()
        await wait_for(
            lambda: published.described() == [("UserMessage", final)], pilot=pilot, description="send published"
        )
