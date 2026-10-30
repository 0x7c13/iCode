# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Spec tests for the prompt-shortcut card in a real app: its lifetime with the draft, the ✕ button and name precedence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from textual import events
from textual.widgets import Button

import chrys.app.tui.i18n as tui_i18n
import chrys.app.tui.screens.main.prompt_card as prompt_card_module
from chrys.app.tui.app import ChrysApp
from chrys.app.tui.screens.dialogs.editor import EditorDialog, EditorDialogResult
from chrys.app.tui.screens.main.screen import MainScreen
from chrys.app.tui.widgets.chrome.input_bar import InputBar, _ChatTextArea
from chrys.app.tui.widgets.chrome.prompt_card import CardChoice, CardState, PromptCard
from chrys.app.tui.widgets.chrome.suggestion_list import SuggestionList
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentProfileSwitch,
    AgentRuntimeDetails,
    AgentRuntimeUpdated,
    ProfileSwitched,
    RuntimeSkillDetails,
    SetModelProfile,
    UserMessage,
)
from chrys.foundation.platform import get_platform
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.pilot_barrier import screen_is_settled
from tests.support.tui_app_harness import make_chrys_app
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from textual.pilot import Pilot

    from chrys.foundation.events.types import Event

_SECTION_COMMANDS = "System Commands"
_SECTION_SHORTCUTS = "Prompt Shortcuts"
_SECTION_SKILLS = "Loaded Skills"


def _two_agent_registry(tmp_path: Path) -> AgentProfileRegistry:
    """The builtin agents, whose main ones (Code, QA) a card can switch between."""
    registry = AgentProfileRegistry()
    registry.load_all(user_dir=tmp_path / "agents")
    return registry


def _global_shortcut(name: str, text: str) -> None:
    path = get_platform().config_dir / "shortcuts" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _Sent:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str) -> None:
        self.texts.append(text)


async def _ready(pilot: Pilot, monkeypatch: pytest.MonkeyPatch) -> tuple[MainScreen, _ChatTextArea, _Sent]:
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
    sent = _Sent()
    monkeypatch.setattr(main, "_submit_user_text", sent)
    return main, main.query_one("#chat-input", _ChatTextArea), sent


async def _type(pilot: Pilot, text: str) -> None:
    await pilot.press(*["space" if char == " " else char for char in text])


def _card(main: MainScreen) -> PromptCard:
    return main.query_one(PromptCard)


def _remove_button(main: MainScreen) -> Button:
    return main.query_one("#prompt-card-remove", Button)


async def _expand_with_card(pilot: Pilot, main: MainScreen, chat: _ChatTextArea, name: str, body: str) -> None:
    await _type(pilot, f"/{name} x")
    await pilot.press("tab")
    await wait_for(lambda: chat.text == f"{body}\n\nx", pilot=pilot, description="shortcut expanded")
    await wait_for(lambda: _card(main).state is not None, pilot=pilot, description="card shown")


class _Switches:
    """Records agent and model switches published on the app's bus."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def __call__(self, event: Event) -> None:
        self.events.append(event)


async def _record_switches(bus: EventBus) -> _Switches:
    switches = _Switches()
    await bus.subscribe(AgentProfileSwitch, switches)
    await bus.subscribe(SetModelProfile, switches)
    return switches


async def test_replacing_the_draft_with_a_slash_command_runs_it_and_drops_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    bus = EventBus()
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        switches = await _record_switches(bus)
        dispatched: list[str] = []
        original = main._suggestions.dispatch_slash_command

        def record(text: str) -> bool:
            dispatched.append(text)
            return original(text)

        monkeypatch.setattr(main._suggestions, "dispatch_slash_command", record)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")
        assert main._prompt_card.active

        chat.select_all()
        await _type(pilot, "/help")
        await wait_for(lambda: chat.text == "/help", pilot=pilot, description="draft replaced by the command")
        await pilot.press("escape")
        await wait_for(lambda: not main.query_one(SuggestionList).is_visible, pilot=pilot, description="list dismissed")
        await pilot.press("enter")

        await wait_for(lambda: dispatched == ["/help"], pilot=pilot, description="/help dispatched")
        await wait_for(lambda: _card(main).state is None, pilot=pilot, description="card gone")
        assert not main._prompt_card.active
        assert not main._prompt_card.sending
        assert switches.events == []
        assert sent.texts == []


async def test_full_screen_editor_result_keeps_the_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")
        before = _card(main).state

        main._on_editor_result(
            EditorDialogResult(accepted=True, text="Review this file instead", mode=main._editor_mode),
            opened_with_card=True,
        )

        await wait_for(lambda: chat.text == "Review this file instead", pilot=pilot, description="editor text applied")
        await pilot.pause()
        assert main._prompt_card.active
        assert _card(main).state is not None
        assert before is not None
        assert _card(main).state.agent == before.agent  # type: ignore[union-attr]


async def test_an_editor_opened_before_a_draft_came_back_overwrites_it_and_ends_its_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The editor saves over a draft it never opened on: that draft's shortcut card ends, as any replaced draft's does."""
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this !`git status`")
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        # The input is empty, as while a send is in flight: the editor opens on no shortcut draft.
        await pilot.press("ctrl+o")
        await wait_for(lambda: isinstance(app.screen, EditorDialog), pilot=pilot, description="editor open")
        editor = app.screen
        assert isinstance(editor, EditorDialog)

        # The send fails meanwhile and gives its draft back with its card.
        command = next(c for c in main._suggestions._known_shortcuts() if c.name == "zzqa")
        given_back = "Review this !`git status`"
        assert main.query_one(InputBar).restore_draft(given_back, keep_card=True)
        main._prompt_card.attach(command, given_back)
        assert main._prompt_card.active

        editor.dismiss(EditorDialogResult(accepted=True, text="Something else", mode=main._editor_mode))
        await wait_for(lambda: chat.text == "Something else", pilot=pilot, description="editor text applied")
        await pilot.pause()

        assert not main._prompt_card.active
        assert _card(main).state is None


async def test_plain_draft_replacement_drops_the_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control for the editor test: a replace that is not the same draft ends the card."""
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")

        main.query_one(InputBar).replace_draft("something else")

        await wait_for(lambda: _card(main).state is None, pilot=pilot, description="card gone")
        assert not main._prompt_card.active


def _highlighted_section(main: MainScreen) -> str | None:
    suggestions = main.query_one(SuggestionList)
    rows = [content.plain for content in suggestions._contents]
    index = suggestions._highlighted
    if index is None:
        return None
    for row in reversed(rows[: index + 1]):
        if row in (_SECTION_COMMANDS, _SECTION_SHORTCUTS, _SECTION_SKILLS):
            return row
    return None


def _highlighted_row(main: MainScreen) -> str:
    suggestions = main.query_one(SuggestionList)
    index = suggestions._highlighted
    assert index is not None
    return suggestions._contents[index].plain


async def _open_typed(pilot: Pilot, main: MainScreen, name: str) -> None:
    await _type(pilot, f"/{name}")
    suggestions = main.query_one(SuggestionList)
    await wait_for(
        lambda: (
            suggestions.is_visible
            and not suggestions.is_loading
            and any(row.split()[:1] == [f"/{name}"] for row in (c.plain for c in suggestions._contents))
            and _SECTION_SHORTCUTS in [c.plain for c in suggestions._contents]
        ),
        pilot=pilot,
        description=f"/{name} listed",
    )


async def test_exact_typed_skill_name_highlights_the_skill_over_a_same_named_shortcut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzdup", "SHORTCUT BODY")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, _chat, _sent = await _ready(pilot, monkeypatch)
        main._state.runtime.details = AgentRuntimeDetails(
            skill_details=[RuntimeSkillDetails(name="zzdup", description="Same-named skill")]
        )
        await _open_typed(pilot, main, "zzdup")

        await wait_for(
            lambda: _highlighted_section(main) == _SECTION_SKILLS, pilot=pilot, description="skill row highlighted"
        )
        assert _highlighted_row(main).split()[0] == "/zzdup"


async def test_exact_typed_command_name_highlights_the_command_over_skill_and_shortcut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("help", "SHORTCUT BODY")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, _chat, _sent = await _ready(pilot, monkeypatch)
        main._state.runtime.details = AgentRuntimeDetails(
            skill_details=[RuntimeSkillDetails(name="help", description="Same-named skill")]
        )
        await _open_typed(pilot, main, "help")

        await wait_for(
            lambda: _highlighted_section(main) == _SECTION_COMMANDS, pilot=pilot, description="command row highlighted"
        )
        assert _highlighted_row(main).split()[0] == "/help"


async def test_dont_switch_button_shows_only_when_something_would_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    _global_shortcut("zzmissing", "---\nagent: NoSuchAgent\n---\nOther")
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")
        button = _remove_button(main)
        assert button.display
        assert "Don't switch" in str(button.label)

        chat.clear()
        await wait_for(lambda: _card(main).state is None, pilot=pilot, description="card ended with the draft")
        await _expand_with_card(pilot, main, chat, "zzmissing", "Other")
        state = _card(main).state
        assert state is not None
        assert state.agent is not None
        assert state.agent.value is None
        assert not _remove_button(main).display

        sending = CardState(
            name="zzqa",
            agent=CardChoice(value="QA", label="QA", current="Code"),
            commands=1,
            running=True,
            in_progress=True,
        )
        _card(main).show_state(sending)
        await pilot.pause()
        assert not _remove_button(main).display


async def test_the_card_hides_with_the_input_bar_and_comes_back_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")
        card = _card(main)
        state = card.state
        place = card.region

        # F12: the dashboard takes the screen, and the card goes with the input bar.
        await pilot.press("f12")
        await wait_for(lambda: not main.query_one(InputBar).display, pilot=pilot, description="dashboard shown")
        assert not card.display
        await pilot.pause()
        assert main.get_widget_at(place.x + 2, place.y + 1)[0] not in card.walk_children(with_self=True)

        await pilot.press("f12")
        await wait_for(lambda: main.query_one(InputBar).display, pilot=pilot, description="chat back")
        assert card.display
        assert card.state == state
        assert main._prompt_card.active


async def test_dont_switch_button_drops_the_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    bus = EventBus()
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        switches = await _record_switches(bus)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")

        await click_when_settled(pilot, _remove_button(main))

        await wait_for(lambda: _card(main).state is None, pilot=pilot, description="switch dropped")
        # Still the shortcut's draft: only the switch went.
        assert main._prompt_card.active
        assert chat.text == "Review this\n\nx"
        await pilot.press("enter")
        await wait_for(lambda: sent.texts == ["Review this\n\nx"], pilot=pilot, description="sent as is")
        assert switches.events == []


async def test_editing_a_shown_card_repaints_it_only_when_the_command_count_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The card says how many commands run, not which: typing paints it only when that number changes.

    A new count keeps the card's height, so it never lays the screen out: that costs as much as the transcript is long.
    """
    _global_shortcut("zzqacmd", "---\nagent: QA\n---\nReview !`git status`")
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        await _expand_with_card(pilot, main, chat, "zzqacmd", "Review !`git status`")
        card = _card(main)
        assert card.state is not None
        assert card.state.commands == 1
        await wait_for(lambda: screen_is_settled(app, main), pilot=pilot, description="card laid out")
        painted: list[CardState] = []
        original = card.show_state
        monkeypatch.setattr(card, "show_state", lambda state: (painted.append(state), original(state)))
        layouts: list[None] = []
        monkeypatch.setattr(main, "_refresh_layout", lambda *_args, **_kwargs: layouts.append(None))

        # Inside the command: still one command.
        chat.move_cursor((0, len("Review !`git status")))
        chat.insert(" --short")
        await wait_for(lambda: "--short" in chat.text, pilot=pilot, description="the edit landed")
        assert painted == []

        chat.move_cursor(chat.document.end)
        chat.insert(" !`git log`")
        await wait_for(lambda: len(painted) == 1, pilot=pilot, description="the card took the new count")
        assert painted[0].commands == 2
        await pilot.pause()
        assert layouts == []


def _fake_shell(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Run ``!`echo X``` as X without a shell; every send's text is recorded."""
    ran: list[str] = []

    async def interpolate(text: str, *, cwd: str) -> str:
        ran.append(text)
        return re.sub(r"!`echo ([^`]*)`", r"\1", text)

    monkeypatch.setattr(prompt_card_module, "interpolate_shell", interpolate)
    return ran


async def test_pasting_over_a_commands_only_draft_shows_the_pasted_command_before_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzstatus", "Status: !`echo clean`")
    bus = EventBus()
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent, _switches = await _ready_with_engine_echo(pilot, monkeypatch, bus)
        ran = _fake_shell(monkeypatch)
        await _expand_with_card(pilot, main, chat, "zzstatus", "Status: !`echo clean`")
        assert _card(main).state is not None

        # Ctrl+A, then a paste: one edit that never empties the draft.
        chat.select_all()
        chat.post_message(events.Paste("!`echo pasted`"))
        await wait_for(lambda: chat.text == "!`echo pasted`", pilot=pilot, description="paste replaced the draft")

        state = _card(main).state
        assert state is not None
        assert state.commands == 1
        await pilot.press("enter")
        await wait_for(lambda: sent.texts == ["pasted"], pilot=pilot, description="sent with the command's output")
        assert ran == ["!`echo pasted`"]


async def test_a_command_broken_while_editing_runs_again_once_mended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzstatus", "Status: !`echo clean`")
    bus = EventBus()
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent, _switches = await _ready_with_engine_echo(pilot, monkeypatch, bus)
        ran = _fake_shell(monkeypatch)
        await _expand_with_card(pilot, main, chat, "zzstatus", "Status: !`echo clean`")
        closing = len("Status: !`echo clean")

        # Deleting the closing backtick leaves no command for now: the card hides, the draft stays the shortcut's.
        chat.move_cursor((0, closing + 1))
        await pilot.press("backspace")
        await wait_for(lambda: _card(main).state is None, pilot=pilot, description="no command, no card")
        assert main._prompt_card.active

        chat.insert("`")
        await wait_for(
            lambda: _card(main).state is not None and _card(main).state.commands == 1,
            pilot=pilot,
            description="the mended command shows again",
        )
        await pilot.press("enter")
        await wait_for(lambda: sent.texts == ["Status: clean\n\nx"], pilot=pilot, description="sent with its output")
        assert len(ran) == 1


async def test_locale_switch_repaints_a_showing_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")
        assert "Don't switch" in str(_remove_button(main).label)

        app.locale_controller.switch_locale("zh-Hans")

        await wait_for(
            lambda: "Don't switch" not in str(_remove_button(main).label), pilot=pilot, description="card relocalized"
        )
        assert _card(main).state is not None


def _registry_with_other_agent(tmp_path: Path) -> AgentProfileRegistry:
    """The builtin agents plus a user agent ``Other``, a third main agent to pick."""
    agents = tmp_path / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / "Other.yaml").write_text(
        'name: Other\ndescription: "Another main agent"\ninstructions: |\n  Be helpful.\n', encoding="utf-8"
    )
    registry = AgentProfileRegistry()
    registry.load_all(user_dir=agents)
    assert registry.get("Other") is not None
    return registry


class _AgentSwitchEcho:
    """Stands in for the engine: answers every agent switch with the ProfileSwitched it would publish."""

    def __init__(self, bus: EventBus, main: MainScreen) -> None:
        self._bus = bus
        self._main = main

    async def __call__(self, event: AgentProfileSwitch) -> None:
        current = self._main._state.runtime.profile_name
        await self._bus.publish(
            ProfileSwitched(from_profile=current, to_profile=event.profile_name, to_display_name=event.profile_name)
        )


async def _ready_with_engine_echo(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch, bus: EventBus
) -> tuple[MainScreen, _ChatTextArea, _Sent, _Switches]:
    main, chat, sent = await _ready(pilot, monkeypatch)

    # A card's send publishes under its own claim, not through the screen's plain submit: record what reaches the bus.
    async def published(event: UserMessage) -> None:
        sent(event.text)

    await bus.subscribe(UserMessage, published)
    # The harness has no usable model; every send, a card's included, would stop at the setup dialog.
    monkeypatch.setattr(main, "_model_unconfigured", lambda: False)
    # The screen now knows the session's agent is Code, as a SessionReady would have told it.
    main._refresh_model_indicator()
    switches = await _record_switches(bus)
    await bus.subscribe(AgentProfileSwitch, _AgentSwitchEcho(bus, main))
    return main, chat, sent, switches


def _agent_switches(switches: _Switches) -> list[tuple[str, bool]]:
    return [
        (event.profile_name, event.session_scoped) for event in switches.events if isinstance(event, AgentProfileSwitch)
    ]


async def test_users_own_agent_switch_drops_the_cards_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    bus = EventBus()
    app = make_chrys_app(tmp_path / "state", agent_registry=_registry_with_other_agent(tmp_path), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent, switches = await _ready_with_engine_echo(pilot, monkeypatch, bus)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")

        # The user picks another agent the way the # picker does.
        await main._config_actions.switch_agent_profile("Other")

        await wait_for(lambda: _card(main).state is None, pilot=pilot, description="card's switch dropped")
        assert main._state.runtime.profile_name == "Other"
        assert chat.text == "Review this\n\nx"

        await pilot.press("enter")
        await wait_for(lambda: sent.texts == ["Review this\n\nx"], pilot=pilot, description="sent as is")
        assert _agent_switches(switches) == [("Other", False)]
        assert main._state.runtime.profile_name == "Other"


async def test_runtime_update_that_keeps_agent_and_model_keeps_the_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: only a change of the session's agent or model drops the card's switch."""
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    bus = EventBus()
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent, _switches = await _ready_with_engine_echo(pilot, monkeypatch, bus)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")
        updates: list[AgentRuntimeUpdated] = []

        async def seen(event: AgentRuntimeUpdated) -> None:
            updates.append(event)

        await bus.subscribe(AgentRuntimeUpdated, seen)

        await bus.publish(AgentRuntimeUpdated(runtime_details=main._state.runtime.details))

        await wait_for(lambda: len(updates) == 1, pilot=pilot, description="runtime update handled")
        await pilot.pause()
        assert main._prompt_card.active
        state = _card(main).state
        assert state is not None
        assert state.agent is not None


async def test_the_cards_own_agent_switch_does_not_cancel_its_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    bus = EventBus()
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent, switches = await _ready_with_engine_echo(pilot, monkeypatch, bus)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")

        await pilot.press("enter")
        await wait_for(lambda: sent.texts == ["Review this\n\nx"], pilot=pilot, description="sent after the switch")
        assert _agent_switches(switches) == [("QA", True)]
        assert main._state.runtime.profile_name == "QA"
        await wait_for(lambda: not main._prompt_card.sending, pilot=pilot, description="send finished")
        assert _card(main).state is None


# -- one submit path: Enter and Retry/Continue route a typed shortcut alike ----


async def _ready_for_submit(
    pilot: Pilot, monkeypatch: pytest.MonkeyPatch, *, retry: bool
) -> tuple[MainScreen, _ChatTextArea, _Sent, _Sent]:
    """The main screen, its chat input, what it sends as new turns and what it sends as retry/continue."""
    main, chat, sent = await _ready(pilot, monkeypatch)
    retried = _Sent()
    monkeypatch.setattr(main._input_flow, "request_retry", retried)
    monkeypatch.setattr(main, "_model_unconfigured", lambda: False)
    if retry:
        input_bar = main.query_one(InputBar)
        input_bar.retry_mode = True
        await wait_for(lambda: input_bar.retry_mode, pilot=pilot, description="retry/continue state")
    return main, chat, sent, retried


async def _enter_typed(pilot: Pilot, main: MainScreen, chat: _ChatTextArea, text: str) -> None:
    """Type *text* into an empty draft and press Enter with the suggestion list closed."""
    chat.insert(text)
    await wait_for(lambda: chat.text == text, pilot=pilot, description="draft typed")
    suggestions = main.query_one(SuggestionList)
    await wait_for(
        lambda: not suggestions.is_visible or not suggestions.is_loading, pilot=pilot, description="list settled"
    )
    if suggestions.is_visible:
        await pilot.press("escape")
        await wait_for(lambda: not suggestions.is_visible, pilot=pilot, description="list dismissed")
    await pilot.press("enter")


@pytest.mark.parametrize("retry", [False, True], ids=["normal", "retry-continue"])
@pytest.mark.parametrize(
    ("path", "typed", "body", "arguments"),
    [
        ("zzqa", "/zzqa hello", "Review this", "hello"),
        ("git/review", "/git/review staged", "Review the diff", "staged"),
        ("翻译", "/翻译 hello", "Translate", "hello"),
    ],
    ids=["plain", "nested", "non-ascii"],
)
async def test_typed_shortcut_expands_into_the_draft_and_is_not_sent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry: bool, path: str, typed: str, body: str, arguments: str
) -> None:
    _global_shortcut(path, f"---\nagent: QA\n---\n{body}")
    bus = EventBus()
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent, retried = await _ready_for_submit(pilot, monkeypatch, retry=retry)
        switches = await _record_switches(bus)
        await wait_for(
            lambda: main._suggestions.resolve_typed_name(typed).kind == "shortcut",
            pilot=pilot,
            description="shortcuts loaded",
        )

        await _enter_typed(pilot, main, chat, typed)

        expected = f"{body}\n\n{arguments}"
        await wait_for(lambda: chat.text == expected, pilot=pilot, description="shortcut expanded into the draft")
        await wait_for(lambda: _card(main).state is not None, pilot=pilot, description="card shown")
        state = _card(main).state
        assert state is not None
        assert state.agent is not None
        assert state.agent.value == "QA"
        await pilot.pause()
        assert sent.texts == []
        assert retried.texts == []
        assert switches.events == []
        assert not main._prompt_card.sending


@pytest.mark.parametrize("retry", [False, True], ids=["normal", "retry-continue"])
async def test_typed_shortcut_over_a_card_replaces_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry: bool
) -> None:
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    _global_shortcut("team/other", "---\nagent: Other\n---\nOther work")
    app = make_chrys_app(tmp_path / "state", agent_registry=_registry_with_other_agent(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent, retried = await _ready_for_submit(pilot, monkeypatch, retry=retry)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")
        first = _card(main).state
        assert first is not None
        assert first.name == "zzqa"

        chat.clear()
        await _enter_typed(pilot, main, chat, "/team/other y")

        await wait_for(lambda: chat.text == "Other work\n\ny", pilot=pilot, description="new shortcut expanded")
        await wait_for(
            lambda: _card(main).state is not None and _card(main).state.name == "team/other",  # type: ignore[union-attr]
            pilot=pilot,
            description="new card shown",
        )
        state = _card(main).state
        assert state is not None
        assert state.agent is not None
        assert state.agent.value == "Other"
        assert sent.texts == []
        assert retried.texts == []
        assert not main._prompt_card.sending


async def test_plain_text_in_retry_state_goes_to_the_retry_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    app = make_chrys_app(tmp_path / "state", agent_registry=_two_agent_registry(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent, retried = await _ready_for_submit(pilot, monkeypatch, retry=True)

        await _enter_typed(pilot, main, chat, "keep going")

        await wait_for(lambda: retried.texts == ["keep going"], pilot=pilot, description="sent as the continue note")
        assert sent.texts == []
        assert _card(main).state is None


async def test_a_session_side_agent_change_keeps_the_cards_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A switch the session makes itself (a restore, a reload) is no user pick: the card's switch stays."""
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    bus = EventBus()
    app = make_chrys_app(tmp_path / "state", agent_registry=_registry_with_other_agent(tmp_path), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent, switches = await _ready_with_engine_echo(pilot, monkeypatch, bus)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")

        # The session moves to the declared agent and back, never by the user's pick.
        for agent in ("QA", "Code"):
            await bus.publish(ProfileSwitched(from_profile="", to_profile=agent, to_display_name=agent))
            await wait_for(lambda a=agent: main._state.runtime.profile_name == a, pilot=pilot, description=agent)
            assert main._prompt_card.active

        await pilot.press("enter")
        await wait_for(lambda: sent.texts == ["Review this\n\nx"], pilot=pilot, description="sent after the switch")
        assert _agent_switches(switches) == [("QA", True)]


@pytest.mark.parametrize("to_agent", ["QA", "Other"], ids=["to-the-declared-agent", "to-another-agent"])
async def test_the_card_shows_only_what_differs_from_the_session_now(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, to_agent: str
) -> None:
    """After a session-side agent change the card compares against the new agent: no "QA → QA" row."""
    _global_shortcut("zzqa", "---\nagent: QA\n---\nReview this")
    bus = EventBus()
    app = make_chrys_app(tmp_path / "state", agent_registry=_registry_with_other_agent(tmp_path), event_bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent, _switches = await _ready_with_engine_echo(pilot, monkeypatch, bus)
        await _expand_with_card(pilot, main, chat, "zzqa", "Review this")

        await bus.publish(ProfileSwitched(from_profile="Code", to_profile=to_agent, to_display_name=to_agent))
        await wait_for(
            lambda: main._state.runtime.profile_name == to_agent, pilot=pilot, description="session agent changed"
        )

        if to_agent == "QA":
            await wait_for(
                lambda: _card(main).state is None or _card(main).state.agent is None,  # type: ignore[union-attr]
                pilot=pilot,
                description="no row for the agent the session already uses",
            )
        else:
            await wait_for(
                lambda: (
                    _card(main).state is not None
                    and _card(main).state.agent is not None  # type: ignore[union-attr]
                    and _card(main).state.agent.current == "Other"  # type: ignore[union-attr]
                ),
                pilot=pilot,
                description="row measured against the session's agent now",
            )
        assert main._prompt_card.active


@pytest.mark.parametrize(("named", "row"), [("model-a", False), ("model-p", True)])
async def test_the_cards_model_row_measures_against_the_model_that_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, named: str, row: bool
) -> None:
    """An agent's own model runs while the session's choice waits underneath: a shortcut naming the
    model that runs needs no switch, and one naming another shows what runs as the current one."""
    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-p", name="Model P", model_id="pointer-wire"))
    registry.register(ModelProfile(id="model-a", name="Model A", model_id="agent-wire"))
    _global_shortcut("zzmodel", f"---\nmodel: {named}\n---\nReview this")
    app = make_chrys_app(tmp_path / "state", model_registry=registry)
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        main._services.active_model_profile_id = "model-p"
        main._state.runtime.details_confirmed = True
        main._state.runtime.details.model.profile_id = "model-a"
        main._state.runtime.details.model.selection_source = "agent"

        await _type(pilot, "/zzmodel x")
        await pilot.press("tab")
        await wait_for(lambda: chat.text == "Review this\n\nx", pilot=pilot, description="shortcut expanded")
        await pilot.pause()

        state = _card(main).state
        if row:
            assert state is not None
            assert state.model is not None
            assert state.model.current == "Model A"
        else:
            assert state is None or state.model is None
