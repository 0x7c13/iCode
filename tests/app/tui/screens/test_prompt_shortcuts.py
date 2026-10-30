# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Spec tests for prompt shortcuts in a real app: the / list, expansion into the draft and name resolution."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.color import Color
from textual.widgets import Button, Checkbox, Input, Label

from chrys.app.tui.app import ChrysApp
from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.agents.panels.skills import SkillDirCard
from chrys.app.tui.screens.guides.screen import GuideDialog
from chrys.app.tui.screens.main.screen import MainScreen
from chrys.app.tui.screens.settings import SettingsDialog
from chrys.app.tui.screens.settings.layout import PROMPT_SHORTCUTS_TAB_ID
from chrys.app.tui.screens.settings.panes.prompt_shortcuts import (
    PATHS_KEY,
    USER_FOLDER_KEY,
    WORKING_FOLDER_KEY,
    PromptDirCard,
    PromptShortcutsPane,
)
from chrys.app.tui.widgets.chrome.input_bar import InputBar, _ChatTextArea
from chrys.app.tui.widgets.chrome.suggestion_list import SuggestionList
from chrys.app.tui.widgets.directory_path import DirectoryPathCard
from chrys.app.tui.widgets.hatch import HatchedEmptyState
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.types import AgentRuntimeDetails, RuntimeSkillDetails
from chrys.foundation.platform import get_platform
from tests.app.tui.screens.settings.support import Host, StubPorts, wait_for_every_tab
from tests.support.pilot_barrier import screen_is_settled
from tests.support.tui_app_harness import make_chrys_app
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from textual.pilot import Pilot

_SECTION_COMMANDS = "System Commands"
_SECTION_SHORTCUTS = "Prompt Shortcuts"
_SECTION_SKILLS = "Loaded Skills"

_TRANSLATE = "---\ndescription: Translate a passage\nargument-hint: <tone> <text>\n---\n\nUse a $1 tone for:\n$2\n"
_TRANSLATE_HINTS = "Use a argument 1 tone for:\nargument 2"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _global_shortcut(name: str, text: str) -> Path:
    return _write(get_platform().config_dir / "shortcuts" / f"{name}.md", text)


def _project_shortcut(name: str, text: str) -> Path:
    return _write(Path.cwd() / ".chrys" / "shortcuts" / f"{name}.md", text)


class _Sent:
    """Records what the main screen actually submits as a chat message."""

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
    sent = _Sent()
    monkeypatch.setattr(main, "_submit_user_text", sent)
    return main, main.query_one("#chat-input", _ChatTextArea), sent


def _list(main: MainScreen) -> SuggestionList:
    return main.query_one(SuggestionList)


def _rows(main: MainScreen) -> list[str]:
    return [content.plain for content in _list(main)._contents]


def _list_open(main: MainScreen) -> bool:
    suggestions = _list(main)
    return suggestions.is_visible and not suggestions.is_loading


def _group(main: MainScreen, section: str) -> list[str]:
    """The rows under *section*, up to the next section header."""
    rows = _rows(main)
    if section not in rows:
        return []
    group: list[str] = []
    for row in rows[rows.index(section) + 1 :]:
        if row in (_SECTION_COMMANDS, _SECTION_SHORTCUTS, _SECTION_SKILLS):
            break
        group.append(row)
    return group


def _has_shortcut_row(main: MainScreen, name: str) -> bool:
    return any(row.split()[0] == f"/{name}" for row in _group(main, _SECTION_SHORTCUTS) if row.strip())


async def _type(pilot: Pilot, text: str) -> None:
    await pilot.press(*["space" if char == " " else char for char in text])


async def _open_list(pilot: Pilot, main: MainScreen, filter_text: str = "") -> None:
    await pilot.press("/")
    await _type(pilot, filter_text)
    await wait_for(lambda: _list_open(main), pilot=pilot, description="/ list open")


async def _highlight_shortcut(pilot: Pilot, main: MainScreen, name: str) -> None:
    """Move the list highlight onto the shortcut row for *name*."""
    await wait_for(lambda: _has_shortcut_row(main, name), pilot=pilot, description=f"/{name} shortcut row listed")
    suggestions = _list(main)
    rows = _rows(main)
    start = rows.index(_SECTION_SHORTCUTS)
    target = next(i for i in range(start + 1, len(rows)) if rows[i].split()[0] == f"/{name}")
    for _ in range(len(rows)):
        if suggestions._highlighted == target:
            return
        await pilot.press("down")
    pytest.fail(f"could not highlight /{name}")


async def test_slash_list_groups_shortcuts_between_system_commands_and_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzglobal", "---\ndescription: My own prompt\n---\nGlobal body")
    _project_shortcut("zzproject", "---\ndescription: Team prompt\n---\nProject body")
    _global_shortcut("git/commit-msg", "Write a commit message")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, _chat, _sent = await _ready(pilot, monkeypatch)
        main._state.runtime.details = AgentRuntimeDetails(
            skill_details=[RuntimeSkillDetails(name="zzskill", description="A loaded skill")]
        )

        await _open_list(pilot, main)
        await wait_for(
            lambda: _has_shortcut_row(main, "zzproject") and _has_shortcut_row(main, "zzglobal"),
            pilot=pilot,
            description="shortcut rows listed",
        )

        rows = _rows(main)
        assert rows.index(_SECTION_COMMANDS) < rows.index(_SECTION_SHORTCUTS) < rows.index(_SECTION_SKILLS)
        shortcuts = _group(main, _SECTION_SHORTCUTS)
        assert any(row.startswith("/zzglobal") and "My own prompt" in row for row in shortcuts)
        assert any(row.startswith("/zzproject") and "Team prompt" in row for row in shortcuts)
        assert any(row.startswith("/git/commit-msg") for row in shortcuts)
        assert not any("[project" in row or "[global]" in row for row in rows)


async def test_shortcut_rows_filter_like_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _global_shortcut("zzalpha", "Alpha")
    _global_shortcut("zzbeta", "Beta")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, _chat, _sent = await _ready(pilot, monkeypatch)
        await _open_list(pilot, main, "zzal")

        await wait_for(lambda: _has_shortcut_row(main, "zzalpha"), pilot=pilot, description="matching row listed")
        assert not _has_shortcut_row(main, "zzbeta")


async def test_burst_slash_input_filters_before_selecting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _global_shortcut("explain", "Explain $1")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        await wait_for(
            lambda: any(command.name == "explain" for command in main._suggestions._prompt_shortcuts),
            pilot=pilot,
            description="shortcuts loaded before typing",
        )
        for char in "/ex":
            chat.post_message(events.Key(char, char))
        await wait_for(lambda: _list_open(main) and chat.text == "/ex", pilot=pilot, description="burst handled")

        assert [row.split()[0] for row in _group(main, _SECTION_COMMANDS)] == ["/exit"]
        assert _has_shortcut_row(main, "explain")
        await pilot.press("tab")
        await wait_for(lambda: chat.text == "/exit ", pilot=pilot, description="matching command selected")
        assert sent.texts == []


@pytest.mark.parametrize("key", ["tab", "enter"])
async def test_choosing_a_shortcut_row_expands_it_without_sending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    _global_shortcut("zztranslate", _TRANSLATE)
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        await _open_list(pilot, main, "zztrans")
        await _highlight_shortcut(pilot, main, "zztranslate")

        await pilot.press(key)

        await wait_for(lambda: chat.text == _TRANSLATE_HINTS, pilot=pilot, description="shortcut expanded")
        assert chat.selected_text == "argument 1"
        assert chat.prompt_fields == ((6, 16), (27, 37))
        assert not _list(main).is_visible
        assert sent.texts == []


async def test_expanded_fields_take_typing_and_tab_jumps_and_wraps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zztranslate", _TRANSLATE)
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        await _open_list(pilot, main, "zztrans")
        await _highlight_shortcut(pilot, main, "zztranslate")
        await pilot.press("tab")
        await wait_for(lambda: chat.text == _TRANSLATE_HINTS, pilot=pilot, description="shortcut expanded")

        await pilot.press("tab")
        await wait_for(lambda: chat.selection.end == (1, 10), pilot=pilot, description="second field selected")
        assert chat.selected_text == "argument 2"
        await pilot.press("tab")
        await wait_for(lambda: chat.selection.end == (0, 16), pilot=pilot, description="wrapped to first field")
        assert chat.selected_text == "argument 1"

        await _type(pilot, "dry")
        assert chat.text == "Use a dry tone for:\nargument 2"
        assert chat.prompt_fields == ((20, 30),)
        await pilot.press("tab")
        await wait_for(lambda: chat.selected_text == "argument 2", pilot=pilot, description="remaining field selected")
        assert sent.texts == []


async def test_enter_with_untouched_fields_sends_only_on_second_press(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zztranslate", _TRANSLATE)
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40), notifications=True) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        await _open_list(pilot, main, "zztrans")
        await _highlight_shortcut(pilot, main, "zztranslate")
        await pilot.press("enter")
        await wait_for(lambda: chat.text == _TRANSLATE_HINTS, pilot=pilot, description="shortcut expanded")

        await pilot.press("enter")
        assert sent.texts == []
        assert chat.text == _TRANSLATE_HINTS

        await pilot.press("enter")
        await wait_for(lambda: sent.texts == [_TRANSLATE_HINTS], pilot=pilot, description="second Enter sent")


async def test_an_at_file_picked_after_the_fields_keeps_them(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Picking from the @ list rewrites only the @ token: the unfilled fields stay, so Enter still asks first."""
    Path("notes.txt").write_text("notes", encoding="utf-8")
    _global_shortcut("zztranslate", _TRANSLATE)
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40), notifications=True) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        await _open_list(pilot, main, "zztrans")
        await _highlight_shortcut(pilot, main, "zztranslate")
        await pilot.press("enter")
        await wait_for(lambda: chat.text == _TRANSLATE_HINTS, pilot=pilot, description="shortcut expanded")
        fields = chat.prompt_fields
        assert len(fields) == 2

        chat.move_cursor(chat.document.end)
        await pilot.press("space", "@")
        await wait_for(lambda: _list_open(main), pilot=pilot, description="file list open")
        await pilot.press("enter")
        await wait_for(lambda: chat.text == f"{_TRANSLATE_HINTS} @notes.txt ", pilot=pilot, description="file picked")

        assert chat.prompt_fields == fields
        assert chat.cursor_location == chat.document.end
        await pilot.press("enter")
        assert sent.texts == []


@pytest.mark.parametrize(
    ("text", "cursor", "trigger", "replacement", "expected"),
    [
        # The token runs to the end of the text, wherever the cursor is: nothing typed is left behind.
        ("/hlp", 2, "/", "/help ", "/help "),
        ("@note", 3, "@", "@notes.txt ", "@notes.txt "),
        ("x @no", 0, "@", "@notes.txt ", "x @notes.txt "),
        # An @ token starts at the last @; the text before it stays.
        ("@a x @b", 2, "@", "@notes.txt ", "@a x @notes.txt "),
        # A slash token is the leading one, though a typed shortcut name may hold more slashes.
        ("/git/x", 6, "/", "/gitignore ", "/gitignore "),
        ("/theme da", 9, "/theme ", "/theme name ", "/theme name "),
        # Fullwidth triggers count.
        ("\uff20no", 3, "@", "@notes.txt ", "@notes.txt "),
        ("\uff0fhlp", 4, "/", "/help ", "/help "),
        # No token, no change.
        ("plain text", 3, "@", "@notes.txt ", "plain text"),
        ("x /help", 7, "/", "/help ", "x /help"),
    ],
)
async def test_a_completion_replaces_the_token_the_list_filtered_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    cursor: int,
    trigger: str,
    replacement: str,
    expected: str,
) -> None:
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        bar = main.query_one(InputBar)

        chat.load_text(text)
        chat.move_cursor((0, cursor))
        bar.replace_trigger_text(trigger, replacement)

        assert chat.text == expected
        if expected != text:
            assert chat.cursor_location == chat.document.end


async def test_a_slash_pick_with_the_cursor_inside_the_token_replaces_all_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fixing a typo inside ``/hel`` and picking: the list filtered on the whole token, so the pick replaces it."""
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        await _open_list(pilot, main, "hel")
        suggestions = _list(main)
        target = _rows(main).index(next(row for row in _rows(main) if row.split()[:1] == ["/help"]))
        for _ in range(len(_rows(main))):
            if suggestions._highlighted == target:
                break
            await pilot.press("down")
        assert suggestions._highlighted == target

        chat.move_cursor((0, 2))
        await pilot.press("tab")

        await wait_for(lambda: chat.text == "/help ", pilot=pilot, description="/help picked")
        assert chat.cursor_location == chat.document.end


@pytest.mark.parametrize(("typed_before", "cursor"), [("", (0, 3)), ("x ", (0, 0))])
async def test_an_at_pick_replaces_the_whole_token_wherever_the_cursor_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, typed_before: str, cursor: tuple[int, int]
) -> None:
    Path("notes.txt").write_text("notes", encoding="utf-8")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        await _type(pilot, typed_before)
        await pilot.press("@", "n", "o", "t", "e")
        await wait_for(
            lambda: _list_open(main) and any("notes.txt" in row for row in _rows(main)),
            pilot=pilot,
            description="notes.txt listed",
        )

        chat.move_cursor(cursor)
        await pilot.press("enter")

        await wait_for(lambda: chat.text == f"{typed_before}@notes.txt ", pilot=pilot, description="file picked whole")


@pytest.mark.parametrize("draft", ["/etc/hosts is broken", "/help me"], ids=["path", "command"])
async def test_tab_on_a_slash_draft_that_names_no_shortcut_moves_focus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, draft: str
) -> None:
    """Only a draft a shortcut expands keeps Tab: a path, a command or any other ``/word …`` text
    moves focus as Tab always does."""
    _global_shortcut("zztranslate", _TRANSLATE)
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        # Read: no name is still waiting to turn out a shortcut.
        await wait_for(lambda: main._suggestions._shortcuts() is not None, pilot=pilot, description="shortcuts read")
        chat.insert(draft)
        await wait_for(lambda: chat.text == draft, pilot=pilot, description="draft typed")
        assert not _list(main).is_visible

        await pilot.press("tab")

        await wait_for(lambda: not chat.has_focus, pilot=pilot, description="focus moved on")
        assert main.focused is main.query_one("#send-btn")
        assert chat.text == draft
        assert sent.texts == []


@pytest.mark.parametrize("button", ["#editor-btn", "#send-btn"])
async def test_tab_from_a_button_in_the_bar_moves_focus_over_a_shortcut_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, button: str
) -> None:
    """Tab belongs to the shortcut only in the chat input: from the bar's buttons it moves focus, the draft unexpanded."""
    _global_shortcut("zztranslate", _TRANSLATE)
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        await wait_for(lambda: main._suggestions._shortcuts() is not None, pilot=pilot, description="shortcuts read")
        chat.insert("/zztranslate formal hello")
        await wait_for(lambda: chat.text == "/zztranslate formal hello", pilot=pilot, description="draft typed")
        start = main.query_one(button)
        # Send takes focus once the bar has handled the draft's change.
        await wait_for(lambda: start.focusable, pilot=pilot, description="button can take focus")
        start.focus()
        await wait_for(lambda: main.focused is start, pilot=pilot, description="button focused")

        await pilot.press("tab")

        await wait_for(lambda: main.focused is not start, pilot=pilot, description="focus moved on")
        assert chat.text == "/zztranslate formal hello"
        assert sent.texts == []


@pytest.mark.parametrize("key", ["tab", "enter"])
async def test_typed_name_with_arguments_expands_without_sending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    _global_shortcut("zztranslate", _TRANSLATE)
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        await _type(pilot, '/zztranslate formal "the meeting moved"')
        await wait_for(lambda: not _list(main).is_visible, pilot=pilot, description="list closed after the space")

        await pilot.press(key)

        await wait_for(
            lambda: chat.text == "Use a formal tone for:\nthe meeting moved",
            pilot=pilot,
            description="typed shortcut expanded",
        )
        assert chat.prompt_fields == ()
        assert sent.texts == []
        # The expanding Tab is the shortcut's: focus stays to edit the draft.
        assert chat.has_focus

        await pilot.press("enter")
        await wait_for(
            lambda: sent.texts == ["Use a formal tone for:\nthe meeting moved"],
            pilot=pilot,
            description="expanded text sent on the next Enter",
        )


async def test_escape_codes_in_a_shortcut_never_reach_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file's text is shown in the input bar: control codes become visible marks, tabs and newlines stay."""
    _global_shortcut("zzescape", "Review this change: \x1b[8mhidden\x1b[0m\tthen\x07 $1\nThe next line of the prompt.")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        await _type(pilot, "/zzescape arg")
        await wait_for(lambda: not _list(main).is_visible, pilot=pilot, description="list closed after the space")

        await pilot.press("tab")

        await wait_for(
            lambda: (
                chat.text
                == "Review this change: \ufffd[8mhidden\ufffd[0m\tthen\ufffd arg\nThe next line of the prompt."
            ),
            pilot=pilot,
            description="shortcut expanded without its control codes",
        )


async def test_typed_name_with_too_few_arguments_leaves_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _global_shortcut("zztranslate", _TRANSLATE)
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        _main, chat, sent = await _ready(pilot, monkeypatch)
        await _type(pilot, "/zztranslate formal")
        await pilot.press("enter")

        await wait_for(lambda: chat.text == "Use a formal tone for:\nargument 2", pilot=pilot, description="expanded")
        assert chat.prompt_fields == ((23, 33),)
        assert chat.selected_text == "argument 2"
        assert sent.texts == []


async def test_typed_unknown_name_tab_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _global_shortcut("zztranslate", _TRANSLATE)
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        await _type(pilot, "/zznotashortcut some words")
        await wait_for(lambda: not _list(main).is_visible, pilot=pilot, description="list closed")

        await pilot.press("tab")

        assert chat.text == "/zznotashortcut some words"
        assert chat.prompt_fields == ()
        assert sent.texts == []


async def test_system_command_wins_over_same_named_shortcut_when_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("help", "SHORTCUT BODY for help")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        dispatched: list[str] = []
        original = main._suggestions.dispatch_slash_command

        def record(text: str) -> bool:
            dispatched.append(text)
            return original(text)

        monkeypatch.setattr(main._suggestions, "dispatch_slash_command", record)

        await _open_list(pilot, main, "help")
        await wait_for(lambda: _has_shortcut_row(main, "help"), pilot=pilot, description="shortcut row still listed")
        assert any(row.split()[0] == "/help" for row in _group(main, _SECTION_COMMANDS))
        await pilot.press("escape")
        await wait_for(lambda: not _list(main).is_visible, pilot=pilot, description="list dismissed")

        await _type(pilot, " topic")
        await pilot.press("enter")

        await wait_for(lambda: dispatched == ["/help topic"], pilot=pilot, description="system command dispatched")
        assert "SHORTCUT BODY" not in chat.text
        assert sent.texts == []


async def test_shortcut_row_sharing_a_system_command_name_expands_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("help", "SHORTCUT BODY for help")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        await _open_list(pilot, main, "help")
        await _highlight_shortcut(pilot, main, "help")
        assert not _list(main)._disabled[_list(main)._highlighted or 0]

        await pilot.press("tab")

        await wait_for(lambda: chat.text == "SHORTCUT BODY for help", pilot=pilot, description="shortcut row expanded")
        assert sent.texts == []


async def test_loaded_skill_wins_over_same_named_shortcut_when_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzdup", "SHORTCUT BODY for zzdup")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        main._state.runtime.details = AgentRuntimeDetails(
            skill_details=[RuntimeSkillDetails(name="zzdup", description="Same-named skill")]
        )
        await _open_list(pilot, main, "zzdup")
        await wait_for(lambda: _has_shortcut_row(main, "zzdup"), pilot=pilot, description="shortcut row listed")
        assert any(row.split()[0] == "/zzdup" for row in _group(main, _SECTION_SKILLS))
        await pilot.press("escape")
        await wait_for(lambda: not _list(main).is_visible, pilot=pilot, description="list dismissed")

        await _type(pilot, " some args")
        await pilot.press("enter")

        await wait_for(lambda: sent.texts == ["/zzdup some args"], pilot=pilot, description="skill reference sent")
        assert "SHORTCUT BODY" not in chat.text


async def test_shortcut_row_sharing_a_skill_name_expands_its_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzdup", "SHORTCUT BODY for zzdup")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        main._state.runtime.details = AgentRuntimeDetails(
            skill_details=[RuntimeSkillDetails(name="zzdup", description="Same-named skill")]
        )
        await _open_list(pilot, main, "zzdup")
        await _highlight_shortcut(pilot, main, "zzdup")

        await pilot.press("enter")

        await wait_for(lambda: chat.text == "SHORTCUT BODY for zzdup", pilot=pilot, description="shortcut expanded")
        assert sent.texts == []


async def test_same_named_shortcuts_are_listed_once_and_global_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzsame", "Global body")
    _project_shortcut("zzsame", "Project body")
    _project_shortcut("ZZSAME2", "Project upper")
    _global_shortcut("zzsame2", "Global lower")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        await _open_list(pilot, main, "zzsame")
        await wait_for(lambda: _has_shortcut_row(main, "zzsame"), pilot=pilot, description="shortcut listed")

        shortcuts = [row for row in _group(main, _SECTION_SHORTCUTS) if row.strip()]
        assert [row.split()[0].casefold() for row in shortcuts] == ["/zzsame", "/zzsame2"]
        assert not any("[" in row for row in shortcuts)

        await _highlight_shortcut(pilot, main, "zzsame")
        await pilot.press("tab")
        await wait_for(lambda: chat.text == "Global body", pilot=pilot, description="global file expanded")
        assert sent.texts == []

        chat.clear()
        await _type(pilot, "/zzsame2 x")
        await pilot.press("tab")
        await wait_for(lambda: chat.text == "Global lower\n\nx", pilot=pilot, description="typed name uses global")


async def test_typed_shortcut_resolves_to_global_before_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("zzsame", "Global $1")
    _project_shortcut("zzsame", "Project $1")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        _main, chat, sent = await _ready(pilot, monkeypatch)
        await _type(pilot, "/zzsame value")
        await pilot.press("enter")

        await wait_for(lambda: chat.text == "Global value", pilot=pilot, description="global shortcut expanded")
        assert sent.texts == []


async def test_working_folder_off_hides_project_shortcuts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _global_shortcut("zzglobal", "Global body")
    _project_shortcut("zzproject", "Project body")
    app = make_chrys_app(tmp_path / "state", settings=Settings(prompt_shortcuts_load_working_folder=False))
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        await _open_list(pilot, main, "zz")
        await wait_for(lambda: _has_shortcut_row(main, "zzglobal"), pilot=pilot, description="global row listed")
        assert not _has_shortcut_row(main, "zzproject")
        await pilot.press("escape")
        await wait_for(lambda: not _list(main).is_visible, pilot=pilot, description="list dismissed")
        chat.clear()

        await _type(pilot, "/zzproject arg")
        await pilot.press("tab")
        assert chat.text == "/zzproject arg"
        assert sent.texts == []


async def test_project_shortcuts_load_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().prompt_shortcuts_load_working_folder is True
    _project_shortcut("zzproject", "Project body")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        _main, chat, _sent = await _ready(pilot, monkeypatch)
        await _type(pilot, "/zzproject x")
        await pilot.press("tab")

        await wait_for(lambda: chat.text == "Project body\n\nx", pilot=pilot, description="project shortcut expanded")


async def test_path_like_text_closes_the_list_unless_a_shortcut_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _global_shortcut("git/commit-msg", "Write a commit message")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        await _open_list(pilot, main, "git/")
        await wait_for(
            lambda: _has_shortcut_row(main, "git/commit-msg"), pilot=pilot, description="nested shortcut listed"
        )
        assert _list(main).is_visible

        chat.clear()
        await wait_for(lambda: not _list(main).is_visible, pilot=pilot, description="list closed on clear")
        await _type(pilot, "/Users/me")
        await wait_for(lambda: not _list(main).is_visible, pilot=pilot, description="path closes the list")
        assert chat.text == "/Users/me"


async def test_file_edits_show_the_next_time_the_list_opens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _global_shortcut("zzedit", "---\ndescription: Before edit\n---\nBody")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, _sent = await _ready(pilot, monkeypatch)
        await _open_list(pilot, main, "zz")
        await wait_for(
            lambda: any("Before edit" in row for row in _group(main, _SECTION_SHORTCUTS)),
            pilot=pilot,
            description="first description shown",
        )
        await pilot.press("escape")
        chat.clear()
        await wait_for(lambda: not _list(main).is_visible and chat.text == "", pilot=pilot, description="reset")

        target.write_text("---\ndescription: After edit\n---\nBody", encoding="utf-8")
        _global_shortcut("zzadded", "Added")
        await _open_list(pilot, main, "zz")

        await wait_for(
            lambda: (
                any("After edit" in row for row in _group(main, _SECTION_SHORTCUTS))
                and _has_shortcut_row(main, "zzadded")
            ),
            pilot=pilot,
            description="edited and added shortcuts shown",
        )


# -- Settings → Shortcuts pane ------------------------------------------------


async def _shortcuts_pane(pilot: Pilot, dialog: SettingsDialog) -> PromptShortcutsPane:
    await pilot.app.push_screen(dialog)
    await wait_for_every_tab(dialog, pilot)
    return dialog.query_one(PromptShortcutsPane)


def _dir_cards(pane: PromptShortcutsPane) -> list[PromptDirCard]:
    return list(pane.query(PromptDirCard))


def _card_input(card: PromptDirCard) -> Input:
    return card.query_one(".sk-path-row Input", Input)


def _card_title(card: PromptDirCard) -> str:
    return card.query_one(".config-card-title", Label).render().plain


def _saved_paths(ports: StubPorts) -> list[list[str]]:
    return [json.loads(value) for key, value in ports.live if key == PATHS_KEY]


async def _add_folder_being_typed(pilot: Pilot, pane: PromptShortcutsPane, path: str) -> Input:
    """Add a folder card and type *path* into it without submitting or leaving it."""
    pane.query_one("#ps-add-btn", Button).press()
    await wait_for(lambda: len(_dir_cards(pane)) == 1, pilot=pilot, description="folder card added")
    field = _card_input(_dir_cards(pane)[0])
    field.focus()
    await wait_for(lambda: field.has_focus, pilot=pilot, description="folder field focused")
    field.value = path
    await pilot.pause()
    assert field.has_focus
    return field


async def test_folder_typed_but_not_left_is_saved_by_commit_pending(tmp_path: Path) -> None:
    ports = StubPorts(root=tmp_path)
    folder = str(tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID)
        pane = await _shortcuts_pane(pilot, dialog)
        await _add_folder_being_typed(pilot, pane, folder)
        assert _saved_paths(ports) == []

        dialog.commit_pending()

        assert _saved_paths(ports)[-1:] == [[folder]]


async def test_closing_the_dialog_saves_a_folder_being_typed(tmp_path: Path) -> None:
    ports = StubPorts(root=tmp_path)
    folder = str(tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID)
        pane = await _shortcuts_pane(pilot, dialog)
        await _add_folder_being_typed(pilot, pane, folder)
        assert _saved_paths(ports) == []

        await pilot.press("escape")

        await wait_for(lambda: ports.closed == 1, pilot=pilot, description="dialog closed")
        assert _saved_paths(ports)[-1:] == [[folder]]


async def test_reproject_shows_an_externally_changed_folder_list(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    ports = StubPorts(root=tmp_path)
    ports.desired[PATHS_KEY] = json.dumps([str(first)])
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID)
        pane = await _shortcuts_pane(pilot, dialog)
        await wait_for(
            lambda: [_card_input(card).value for card in _dir_cards(pane)] == [str(first)],
            pilot=pilot,
            description="saved folder shown",
        )

        ports.desired[PATHS_KEY] = json.dumps([str(second), str(first)])
        saves_before = len(_saved_paths(ports))
        dialog.reproject()

        await wait_for(
            lambda: [_card_input(card).value for card in _dir_cards(pane)] == [str(second), str(first)],
            pilot=pilot,
            description="cards rebuilt from the saved list",
        )
        # Nothing writes the old list back over the new one.
        assert all(saved == [str(second), str(first)] for saved in _saved_paths(ports)[saves_before:])


async def test_reproject_leaves_a_folder_card_being_edited_alone(tmp_path: Path) -> None:
    first = tmp_path / "one"
    first.mkdir()
    ports = StubPorts(root=tmp_path)
    ports.desired[PATHS_KEY] = json.dumps([str(first)])
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID)
        pane = await _shortcuts_pane(pilot, dialog)
        await wait_for(lambda: len(_dir_cards(pane)) == 1, pilot=pilot, description="saved folder shown")
        card = _dir_cards(pane)[0]
        field = _card_input(card)
        field.focus()
        await wait_for(lambda: field.has_focus, pilot=pilot, description="folder field focused")
        field.value = str(tmp_path / "half-typed")

        ports.desired[PATHS_KEY] = json.dumps([str(tmp_path / "elsewhere")])
        dialog.reproject()
        await pilot.pause()

        assert _dir_cards(pane) == [card]
        assert field.is_attached
        assert field.value == str(tmp_path / "half-typed")


@pytest.mark.parametrize("locale", ["en", "zh-Hans"])
async def test_the_guide_link_opens_the_shortcuts_page_over_the_settings(tmp_path: Path, locale: str) -> None:
    """The tab links to the user guide's page on shortcuts, in the app's language; closing it returns here."""
    ports = StubPorts(root=tmp_path)
    controller = LocaleController(Settings())
    app = Host(controller)
    async with app.run_test(size=(120, 40)) as pilot:
        if locale != "en":
            controller.switch_locale(locale)
        dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID, locale_controller=controller)
        pane = await _shortcuts_pane(pilot, dialog)

        await click_when_settled(pilot, pane.query_one("#ps-guide-link"))

        await wait_for(
            lambda: isinstance(app.screen, GuideDialog) and app.screen.is_mounted,
            pilot=pilot,
            description="user guide opened",
        )
        guide = app.screen
        assert isinstance(guide, GuideDialog)
        assert guide._active_topic is not None
        assert guide._active_topic.path == "guides/extensions/shortcuts.md"
        assert guide._current_language() == locale

        await pilot.press("escape")
        await wait_for(lambda: app.screen is dialog, pilot=pilot, description="back on the settings")


async def test_language_change_repaints_folder_cards_and_keeps_typed_text(tmp_path: Path) -> None:
    ports = StubPorts(root=tmp_path)
    controller = LocaleController(Settings())
    app = Host(controller)
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID, locale_controller=controller)
        pane = await _shortcuts_pane(pilot, dialog)
        await _add_folder_being_typed(pilot, pane, str(tmp_path / "not-saved-yet"))
        assert "Prompt Shortcut Directory" in _card_title(_dir_cards(pane)[0])

        controller.switch_locale("zh-Hans")

        await wait_for(
            lambda: len(_dir_cards(pane)) == 1 and "快捷指令目录" in _card_title(_dir_cards(pane)[0]),
            pilot=pilot,
            description="folder card in the new language",
        )
        assert _card_input(_dir_cards(pane)[0]).value == str(tmp_path / "not-saved-yet")


async def test_language_change_repaints_the_empty_folder_list(tmp_path: Path) -> None:
    ports = StubPorts(root=tmp_path)
    controller = LocaleController(Settings())
    app = Host(controller)
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID, locale_controller=controller)
        pane = await _shortcuts_pane(pilot, dialog)
        await wait_for(lambda: bool(pane.query(HatchedEmptyState)), pilot=pilot, description="empty state shown")
        assert pane.query_one(HatchedEmptyState).label == "No other folders added"

        controller.switch_locale("zh-Hans")

        await wait_for(
            lambda: [empty.label for empty in pane.query(HatchedEmptyState)] == ["未添加其他目录"],
            pilot=pilot,
            description="empty state in the new language",
        )


@pytest.mark.parametrize(
    ("checkbox_id", "key"),
    [("#ps-load-user-folder", USER_FOLDER_KEY), ("#ps-load-working-folder", WORKING_FOLDER_KEY)],
    ids=["user-folder", "working-folder"],
)
async def test_each_folder_checkbox_saves_its_own_setting(tmp_path: Path, checkbox_id: str, key: str) -> None:
    ports = StubPorts(root=tmp_path)
    ports.desired[USER_FOLDER_KEY] = True
    ports.desired[WORKING_FOLDER_KEY] = True
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID)
        pane = await _shortcuts_pane(pilot, dialog)
        checkbox = pane.query_one(checkbox_id, Checkbox)
        assert checkbox.value is True

        checkbox.toggle()

        await wait_for(lambda: ports.live == [(key, False)], pilot=pilot, description="checkbox saved")


async def test_deleting_a_folder_row_saves_the_shorter_list(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    ports = StubPorts(root=tmp_path)
    ports.desired[PATHS_KEY] = json.dumps([str(first), str(second)])
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID)
        pane = await _shortcuts_pane(pilot, dialog)
        await wait_for(lambda: len(_dir_cards(pane)) == 2, pilot=pilot, description="saved folders shown")

        _dir_cards(pane)[0].query_one(".config-card-delete-btn", Button).press()

        await wait_for(lambda: bool(_saved_paths(ports)), pilot=pilot, description="shorter list saved")
        # The stub never projects what was saved, so a later blur may save again: always the shorter list.
        assert all(saved == [str(second)] for saved in _saved_paths(ports))
        assert [_card_input(card).value for card in _dir_cards(pane)] == [str(second)]


@pytest.mark.parametrize("relative", [False, True], ids=["relative-path-unchecked", "absolute-path-checked"])
async def test_an_invalid_folder_row_blocks_saving(tmp_path: Path, relative: bool) -> None:
    ports = StubPorts(root=tmp_path)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID)
        pane = await _shortcuts_pane(pilot, dialog)
        field = await _add_folder_being_typed(pilot, pane, "")
        card = _dir_cards(pane)[0]
        if relative:
            # "Workspace relative" on, with an absolute path typed.
            card.query_one(".sk-rel-cb", Checkbox).value = True
            field.value = str(tmp_path / "abs")
        else:
            field.value = "relative/prompts"
        await pilot.pause()
        assert card.entry() is None

        dialog.commit_pending()
        await pilot.press("escape")
        await wait_for(lambda: ports.closed == 1, pilot=pilot, description="dialog closed")

        # The row would change the saved list, but nothing is saved while it is invalid.
        assert _saved_paths(ports) == []


# -- the shared folder card: the same preview in Settings and the skills panel -------


class _CardHost(App[None]):
    def __init__(self, cards: list[DirectoryPathCard]) -> None:
        super().__init__()
        self._cards = cards

    def compose(self) -> ComposeResult:
        yield from self._cards


def _preview(card: DirectoryPathCard) -> str:
    label = card.query_one(".sk-preview", Label)
    return str(label.content) if label.display else ""


@pytest.mark.parametrize(
    ("path", "workspace", "expected"),
    [
        ("{existing}", "{ws}", "Current: {existing_resolved}"),
        ("{missing}", "{ws}", "Path does not exist"),
        ("prompts", "{ws}", "Current: {ws_prompts}"),
        ("nowhere", "{ws}", "Missing in current workspace"),
        ("prompts", "", "Resolved at session runtime"),
    ],
    ids=["absolute", "absolute-missing", "relative", "relative-missing", "relative-without-workspace"],
)
async def test_prompt_and_skill_folder_cards_preview_a_path_the_same_way(
    tmp_path: Path, path: str, workspace: str, expected: str
) -> None:
    ws = tmp_path / "ws"
    (ws / "prompts").mkdir(parents=True)
    existing = tmp_path / "existing"
    existing.mkdir()
    values = {
        "ws": str(ws),
        "existing": str(existing),
        "existing_resolved": str(existing.resolve()),
        "missing": str(tmp_path / "missing"),
        "ws_prompts": str((ws / "prompts").resolve()),
    }
    raw = path.format(**values)
    cwd = workspace.format(**values) or None
    prompt_card = PromptDirCard(raw, index=0, workspace_cwd=cwd)
    skill_card = SkillDirCard(raw, index=1, workspace_cwd=cwd)
    async with _CardHost([prompt_card, skill_card]).run_test(size=(120, 40)) as pilot:
        await wait_for(
            lambda: bool(_preview(prompt_card)) and bool(_preview(skill_card)), pilot=pilot, description="previews"
        )

        assert expected.format(**values) in _preview(prompt_card)
        assert _preview(prompt_card) == _preview(skill_card)


@pytest.mark.parametrize("relative", [False, True], ids=["relative-path-unchecked", "absolute-path-checked"])
async def test_prompt_and_skill_folder_cards_explain_a_wrong_relative_choice_the_same_way(
    tmp_path: Path, relative: bool
) -> None:
    raw = str(tmp_path) if relative else "prompts"
    prompt_card = PromptDirCard(raw, index=0, workspace_cwd=str(tmp_path))
    skill_card = SkillDirCard(raw, index=1, workspace_cwd=str(tmp_path))
    async with _CardHost([prompt_card, skill_card]).run_test(size=(120, 40)) as pilot:
        for card in (prompt_card, skill_card):
            card.query_one(".sk-rel-cb", Checkbox).value = relative
        expected = "Turn off Workspace relative" if relative else "Enable Workspace relative"
        await wait_for(
            lambda: expected in _preview(prompt_card) and expected in _preview(skill_card),
            pilot=pilot,
            description="both cards explain the wrong choice",
        )
        assert _preview(prompt_card) == _preview(skill_card)


@pytest.mark.parametrize("raw", [".agents/skills", "prompts"])
async def test_only_the_skill_card_notes_the_working_folders_own_skill_folder(tmp_path: Path, raw: str) -> None:
    ws = tmp_path / "ws"
    (ws / ".agents" / "skills").mkdir(parents=True)
    (ws / "prompts").mkdir()
    prompt_card = PromptDirCard(raw, index=0, workspace_cwd=str(ws))
    skill_card = SkillDirCard(raw, index=1, workspace_cwd=str(ws))
    async with _CardHost([prompt_card, skill_card]).run_test(size=(120, 40)) as pilot:
        await wait_for(
            lambda: bool(_preview(prompt_card)) and bool(_preview(skill_card)), pilot=pilot, description="previews"
        )

        note = "Also auto-loaded"
        assert note not in _preview(prompt_card)
        assert (note in _preview(skill_card)) is (raw == ".agents/skills")
        assert _preview(skill_card).startswith(_preview(prompt_card))


# -- the folder card under an ANSI theme ------------------------------------------


@pytest.fixture
def patched_ansi_buttons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Button CSS as bootstrap leaves it (its ansi block reverted), even with an unpatched install."""
    from chrys.foundation.patches import textual_button_ansi

    monkeypatch.setattr(Button, "DEFAULT_CSS", Button.DEFAULT_CSS)
    monkeypatch.setattr(Button, textual_button_ansi._RUNTIME_PATCH_MARKER, False, raising=False)
    textual_button_ansi.apply_runtime_patch()


def _borderless(widget: Button) -> bool:
    return all(edge[0] in ("", "none", "hidden") for edge in widget.styles.border)


@pytest.mark.usefixtures("patched_ansi_buttons")
@pytest.mark.parametrize("theme", ["chrys", "chrys-ansi"])
async def test_folder_card_browse_button_and_row_stay_compact_and_borderless(tmp_path: Path, theme: str) -> None:
    folder = tmp_path / "prompts"
    folder.mkdir()
    ports = StubPorts(root=tmp_path)
    ports.desired[PATHS_KEY] = json.dumps([str(folder)])
    app = make_chrys_app(tmp_path / "state", settings=Settings(theme=theme))
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for(lambda: app.theme == theme and app._main_screen is not None, pilot=pilot, description="themed")
        assert app.current_theme.ansi is (theme == "chrys-ansi")
        dialog = SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID)
        pane = await _shortcuts_pane(pilot, dialog)
        await wait_for(lambda: len(_dir_cards(pane)) == 1, pilot=pilot, description="folder card shown")
        card = _dir_cards(pane)[0]
        browse = card.query_one("Button.sk-browse-btn", Button)
        field = _card_input(card)
        await wait_for(
            lambda: browse.region.height > 0 and field.region.height > 0 and screen_is_settled(app, app.screen),
            pilot=pilot,
            description="folder row laid out",
        )

        def compact() -> bool:
            return (
                _borderless(browse)
                and browse.region.height == 1
                and "Browse" in browse.render().plain
                and field.region.height == 1
                and browse.region.y == field.region.y
            )

        assert compact()
        if app.current_theme.ansi:
            # The ANSI theme draws every config card on its opaque muted surface, this one included.
            surface = Color.parse(app.theme_variables["control-background-muted"])
            assert card.styles.background == surface
            assert {edge[0] for edge in card.styles.border} == {"solid"}
        # Every state keeps the one-row, borderless look: hover, focus.
        await pilot.hover(browse)
        await wait_for(lambda: browse.mouse_hover, pilot=pilot, description="browse hovered")
        assert compact()
        browse.focus()
        await wait_for(lambda: browse.has_focus, pilot=pilot, description="browse focused")
        assert compact()


async def test_typed_escaped_quotes_reach_the_placeholder_as_quotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """opencode's documented example, typed after the shortcut name: ``\\"`` inside a quoted argument is a quote."""
    _global_shortcut("zzcreate", "Create $1 in $2 with: $3")
    app = make_chrys_app(tmp_path / "state")
    async with app.run_test(size=(120, 40)) as pilot:
        main, chat, sent = await _ready(pilot, monkeypatch)
        typed = r'/zzcreate config.json src "{ \"key\": \"value\" }"'
        await _type(pilot, typed)
        await wait_for(lambda: chat.text == typed, pilot=pilot, description="typed as written")
        await wait_for(lambda: not _list(main).is_visible, pilot=pilot, description="list closed after the space")

        await pilot.press("tab")

        await wait_for(
            lambda: chat.text == 'Create config.json in src with: { "key": "value" }',
            pilot=pilot,
            description="third argument expanded with literal quotes",
        )
        assert sent.texts == []


@pytest.mark.skipif(sys.platform == "win32", reason="Windows file names cannot hold control characters")
@pytest.mark.parametrize("exists", [True, False], ids=["folder-found", "folder-not-yet-made"])
async def test_working_folder_paths_reach_the_screen_without_control_codes(tmp_path: Path, exists: bool) -> None:
    workspace = tmp_path / "ws\x1b[2Jred"
    workspace.mkdir()
    if exists:
        (workspace / ".chrys" / "shortcuts").mkdir(parents=True)
    ports = StubPorts(root=workspace)
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        pane = await _shortcuts_pane(pilot, SettingsDialog(ports, initial_tab=PROMPT_SHORTCUTS_TAB_ID))
        shown = pane.query_one("#ps-working-folder-path", Label).render().plain

    assert "\x1b" not in shown
    assert "ws" in shown and "red" in shown
