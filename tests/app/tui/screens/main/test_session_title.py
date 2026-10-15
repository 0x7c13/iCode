# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the session title on the chat border and the terminal tab."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from chrys.app.tui.screens.dialogs.approval.dialog import ApprovalDialog
from chrys.app.tui.screens.dialogs.session_title import SessionTitleDialog
from chrys.app.tui.screens.main import session_title
from chrys.app.tui.screens.main.screen import MainScreen
from chrys.app.tui.screens.main.session_title import AttentionMark, SessionTitleController, WorkflowTab
from chrys.app.tui.screens.main.state import MainScreenState, RunState
from chrys.app.tui.terminal import title as terminal_title
from chrys.app.tui.util.spinner import SPINNER_FRAMES, STATIC_SPINNER
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import ApprovalCancelled, ApprovalRequest
from tests.support.tui_app_harness import SessionGenerationEngine, make_chrys_app
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for


class _Driver:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, data: str) -> None:
        self.writes.append(data)


class _App:
    def __init__(self) -> None:
        self._driver = _Driver()

    @property
    def titles(self) -> list[str]:
        """The titles written, decoded from their OSC 0 + OSC 2 pairs."""
        titles: list[str] = []
        for write in self._driver.writes:
            osc0, _sep, _osc2 = write.partition("\x07")
            titles.append(osc0.removeprefix("\x1b]0;"))
        return titles


class _Timer:
    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _Harness:
    def __init__(self, *, cwd: str = "/repo", session_id: str = "session-1") -> None:
        self.run = RunState()
        self.app = _App()
        self.cwd = cwd
        self.session_id = session_id
        self.animations = True
        self.attention: AttentionMark = ""
        self.border_titles: list[str] = []
        self.timers: list[_Timer] = []
        self.intervals: list[float] = []
        self.pushed: list[tuple[object, Callable[[str | None], None]]] = []
        self.saves: list[tuple[str, str]] = []
        self.controller = SessionTitleController(
            run=self.run,
            app=lambda: self.app,
            workspace_cwd=lambda: self.cwd,
            show_display_title=self.border_titles.append,
            set_interval=self._set_interval,
            current_session_id=lambda: self.session_id,
            push_screen=lambda screen, callback: self.pushed.append((screen, callback)),
            start_custom_title_save=lambda title, session_id: self.saves.append((title, session_id)),
            session_dir=lambda session_id: Path("/sessions") / session_id,
            animations_enabled=lambda: self.animations,
            attention=lambda: self.attention,
        )

    def _set_interval(self, interval: float, callback: Callable[[], None]) -> _Timer:
        timer = _Timer(callback)
        self.timers.append(timer)
        self.intervals.append(interval)
        return timer

    def tick(self) -> None:
        self.timers[-1].callback()


@pytest.fixture(autouse=True)
def _native_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)


def test_display_title_prefers_custom_then_generated_then_fallback() -> None:
    harness = _Harness()
    controller = harness.controller

    controller.set_session_title_state(fallback="First prompt")
    controller.set_session_title_state(generated="Summary")
    controller.set_session_title_state(custom="Pinned")
    controller.set_session_title_state(custom="")

    assert harness.border_titles == ["First prompt", "Summary", "Pinned", "Summary"]
    assert controller.custom_title == ""
    assert controller.display_title == "Summary"


def test_reset_clears_every_title_and_the_result_mark(tmp_path: Path) -> None:
    harness = _Harness(cwd=str(tmp_path))
    controller = harness.controller
    controller.set_session_title_state(custom="Pinned", generated="Summary", fallback="First")
    controller.mark_terminal_title_completed()

    controller.reset_session_title_state()

    assert controller.display_title == ""
    assert harness.border_titles[-1] == ""
    assert harness.app.titles[-1] == str(tmp_path)


def test_cwd_title_yields_to_pinned_session_title(tmp_path: Path) -> None:
    """cwd updates (workspace changes, restores) must not unpin a session title."""
    pinned = _Harness()
    pinned.controller.set_session_title_state(generated="Login bug fix")
    pinned.app._driver.writes.clear()

    pinned.controller.set_terminal_title_for_cwd(str(tmp_path))

    # The tab already shows the session title, so nothing is rewritten.
    assert pinned.app.titles == []

    unpinned = _Harness()
    unpinned.run.agent_running = True

    unpinned.controller.set_terminal_title_for_cwd(str(tmp_path))

    assert unpinned.app.titles == [f"⠋ {tmp_path}"]


def test_unseeded_cwd_title_falls_back_to_workspace_cwd(tmp_path: Path) -> None:
    harness = _Harness(cwd=str(tmp_path))

    harness.controller.clear_terminal_title_result()
    harness.controller.mark_terminal_title_completed()

    assert harness.app.titles == [f"✓ {tmp_path}"]


def test_first_prompt_seeds_the_fallback_title_and_previews_on_the_tab() -> None:
    harness = _Harness()

    harness.controller.set_terminal_title_for_user_message("  fix \n the  login bug ")
    harness.controller.set_terminal_title_for_user_message("second prompt")

    assert harness.border_titles == ["fix the login bug"]
    assert harness.app.titles[-1] == "second prompt"


def test_custom_title_keeps_the_tab_pinned_over_prompt_previews() -> None:
    harness = _Harness()
    harness.controller.set_session_title_state(custom="Pinned")

    harness.controller.set_terminal_title_for_user_message("a prompt")

    assert harness.app.titles[-1] == "Pinned"


def test_running_animation_preserves_current_prompt_preview() -> None:
    harness = _Harness()
    harness.controller.set_session_title_state(fallback="First task", generated="Old generated title")
    harness.app._driver.writes.clear()

    harness.controller.set_terminal_title_for_user_message("Second task")
    harness.run.agent_running = True
    harness.controller.run_started()
    harness.controller.sync_activity()
    harness.timers[0].callback()

    assert harness.app.titles == ["Second task", "⠋ Second task", "⠙ Second task"]


def test_running_title_cycles_activity_frames_and_settles_on_the_result_mark() -> None:
    harness = _Harness()
    harness.controller.set_session_title_state(custom="Login bug fix")
    harness.app._driver.writes.clear()

    harness.run.agent_running = True
    harness.controller.run_started()
    harness.controller.sync_activity()
    timer = harness.timers[0]
    for _ in range(10):
        timer.callback()
    harness.controller.mark_terminal_title_completed()
    harness.run.agent_running = False
    harness.controller.sync_activity()

    # The workflow node spinner, wrapping back to its first frame.
    assert harness.app.titles == [*(f"{frame} Login bug fix" for frame in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏⠋"), "✓ Login bug fix"]
    assert harness.intervals == [0.3]
    assert timer.stopped is True
    assert len(harness.timers) == 1

    harness.run.agent_running = True
    harness.controller.sync_activity()
    harness.run.agent_running = False
    harness.controller.sync_activity()

    assert [timer.stopped for timer in harness.timers] == [True, True]

    harness.controller.mark_terminal_title_failed()
    harness.controller.clear_terminal_title_result()
    harness.controller.clear_terminal_title_result()

    assert harness.app.titles[-2:] == ["✗ Login bug fix", "Login bug fix"]


def test_a_new_run_drops_the_last_result_mark() -> None:
    harness = _Harness()
    harness.controller.set_session_title_state(custom="Task")
    harness.controller.mark_terminal_title_failed()

    harness.run.agent_running = True
    harness.controller.run_started()
    harness.controller.sync_activity()
    harness.run.agent_running = False
    harness.controller.sync_activity()

    assert harness.app.titles[-2:] == ["⠋ Task", "Task"]


def test_activity_tick_after_the_run_stops_is_ignored() -> None:
    harness = _Harness()
    harness.run.agent_running = True
    harness.controller.sync_activity()
    timer = harness.timers[0]
    harness.run.agent_running = False
    writes = len(harness.app.titles)

    timer.callback()

    assert len(harness.app.titles) == writes


def test_an_unchanged_title_is_not_rewritten() -> None:
    harness = _Harness()
    harness.controller.set_session_title_state(custom="Task")
    harness.controller.set_session_title_state(custom="Task")
    harness.controller.clear_terminal_title_result()
    harness.controller.set_terminal_title_for_user_message("a prompt")

    assert harness.app.titles == ["Task"]


def test_a_title_that_could_not_be_written_is_written_once_the_driver_is_back() -> None:
    harness = _Harness()
    app = harness.app
    harness.app = SimpleNamespace(_driver=None)
    harness.controller.set_session_title_state(custom="Task")
    harness.app = app
    harness.controller.set_session_title_state(custom="Task")
    harness.controller.clear_terminal_title_result()

    assert harness.app.titles == ["Task"]


def test_without_animations_the_spinner_is_a_still_dot_that_keeps_polling_for_waits() -> None:
    harness = _Harness()
    harness.animations = False
    harness.controller.set_session_title_state(custom="Task")
    harness.run.agent_running = True
    harness.controller.run_started()
    harness.controller.sync_activity()

    harness.tick()
    harness.attention = "?"
    harness.tick()
    harness.attention = ""
    harness.tick()

    assert harness.app.titles == ["Task", "● Task", "? Task", "● Task"]


def test_chat_run_waiting_for_the_user_shows_its_mark_instead_of_the_spinner() -> None:
    harness = _Harness()
    harness.controller.set_session_title_state(custom="Task")
    harness.run.agent_running = True
    harness.controller.run_started()
    harness.controller.sync_activity()

    harness.attention = "?"
    harness.tick()
    harness.tick()
    harness.attention = "!"
    harness.tick()
    harness.attention = ""
    harness.tick()
    harness.attention = "?"
    harness.run.agent_running = False
    harness.controller.mark_terminal_title_completed()
    harness.controller.sync_activity()

    # The spinner holds its frame while a mark shows; a finished run ignores marks.
    assert harness.app.titles == ["Task", "⠋ Task", "? Task", "! Task", "⠙ Task", "✓ Task"]
    assert harness.timers[0].stopped is True


def test_a_long_prompt_is_cleaned_once_and_each_tick_only_adds_the_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    cleaned_lengths: list[int] = []
    clean = terminal_title._title_safe_text

    def counting_clean(text: str) -> str:
        cleaned_lengths.append(len(text))
        return clean(text)

    monkeypatch.setattr(terminal_title, "_title_safe_text", counting_clean)
    harness = _Harness()
    harness.controller.set_session_title_state(fallback="First task")
    harness.run.agent_running = True
    harness.controller.run_started()
    prompt = "pasted log line\n" * 1_000

    harness.controller.set_terminal_title_for_user_message(prompt)
    harness.controller.sync_activity()
    for _ in range(5):
        harness.tick()

    assert harness.app.titles[-1] == f"{SPINNER_FRAMES[5]} {('pasted log line ' * 7)[:98]}"
    # The writer still cleans each written title, which is short.
    assert [length for length in cleaned_lengths if length > 100] == [len(prompt)]


def test_a_workflow_run_starts_its_spinner_from_the_first_frame() -> None:
    harness = _Harness()
    controller = harness.controller
    controller.set_workflow_tab(WorkflowTab("Greeting", "/repo", "running"))
    harness.tick()
    harness.tick()
    assert harness.app.titles[-1] == f"{SPINNER_FRAMES[2]} Greeting"

    controller.set_workflow_tab(WorkflowTab("Greeting", "/repo", "completed"))
    controller.set_workflow_tab(WorkflowTab("Greeting", "/repo", "running"))

    assert harness.app.titles[-2:] == ["✓ Greeting", f"{SPINNER_FRAMES[0]} Greeting"]


def test_workflow_tab_shows_each_phase_of_its_run(tmp_path: Path) -> None:
    harness = _Harness()
    controller = harness.controller
    controller.set_session_title_state(custom="Chat task")

    controller.set_workflow_tab(WorkflowTab("Greeting", str(tmp_path)))
    controller.set_workflow_tab(WorkflowTab("Greeting", str(tmp_path), "running"))
    harness.tick()
    controller.set_workflow_tab(WorkflowTab("Greeting", str(tmp_path), "question"))
    harness.tick()
    controller.set_workflow_tab(WorkflowTab("Greeting", str(tmp_path), "awaiting_retry"))
    harness.attention = "?"
    harness.tick()
    controller.set_workflow_tab(WorkflowTab("Greeting", str(tmp_path), "running"))
    harness.attention = ""
    harness.tick()
    controller.set_workflow_tab(WorkflowTab("Greeting", str(tmp_path), "completed"))
    timer = harness.timers[0]
    controller.set_workflow_tab(WorkflowTab("Greeting", str(tmp_path), "failed"))
    controller.set_workflow_tab(WorkflowTab("", str(tmp_path), "failed"))

    assert harness.app.titles == [
        "Chat task",
        "Greeting",
        "⠋ Greeting",
        "⠙ Greeting",
        "? Greeting",
        "! Greeting",
        "? Greeting",
        "⠹ Greeting",
        "✓ Greeting",
        "✗ Greeting",
        f"✗ {tmp_path}",
    ]
    assert len(harness.timers) == 1 and timer.stopped is True


def test_chat_updates_wait_under_the_workflow_tab_until_it_is_cleared() -> None:
    harness = _Harness()
    controller = harness.controller
    controller.set_session_title_state(custom="Chat task")
    controller.set_workflow_tab(WorkflowTab("Greeting", "/repo", "completed"))
    controller.set_workflow_tab(WorkflowTab("Greeting", "/repo", "completed"))

    controller.set_terminal_title_for_cwd("/elsewhere")
    controller.mark_terminal_title_failed()
    controller.set_session_title_state(custom="Renamed")
    harness.attention = "?"
    controller.sync_activity()
    controller.set_workflow_tab(None)

    assert harness.app.titles == ["Chat task", "✓ Greeting", "✗ Renamed"]
    assert harness.timers == []


def test_editor_saves_against_the_session_it_was_opened_for() -> None:
    harness = _Harness(session_id="session-a")
    harness.controller.set_session_title_state(custom="Pinned", generated="Summary", fallback="First prompt")

    harness.controller.open_editor()
    harness.session_id = "session-b"
    dialog, on_result = harness.pushed[0]
    on_result("Renamed")
    on_result(None)

    assert isinstance(dialog, SessionTitleDialog)
    assert (dialog._custom_title, dialog._auto_title) == ("Pinned", "Summary")
    assert (dialog._session_id, dialog._session_folder) == ("sessiona", Path("/sessions/session-a"))
    assert harness.saves == [("Renamed", "session-a")]


def test_editor_and_rename_need_a_session() -> None:
    harness = _Harness(session_id="")

    harness.controller.open_editor()
    harness.controller.apply_custom_title("Renamed")

    assert harness.pushed == []
    assert harness.saves == []


def test_rename_command_saves_against_the_current_session() -> None:
    harness = _Harness(session_id="session-a")

    harness.controller.apply_custom_title("Renamed")

    assert harness.saves == [("Renamed", "session-a")]


def test_new_clear_and_restored_sessions_clear_terminal_title_result() -> None:
    cleared: list[str] = []
    screen = SimpleNamespace(
        _state=MainScreenState(),
        _session_title=SimpleNamespace(clear_terminal_title_result=lambda: cleared.append("clear")),
    )

    # Both /new and confirmed /clear enter the shared creating-new-session path.
    MainScreen._set_creating_new_session(screen, True)
    MainScreen._set_restoring_session(screen, True)
    MainScreen._set_creating_new_session(screen, False)
    MainScreen._set_restoring_session(screen, False)

    assert cleared == ["clear", "clear"]
    assert screen._state.session.creating_new_session is False
    assert screen._state.session.restoring_session is False


@pytest.mark.asyncio
async def test_border_click_opens_the_editor_on_the_store_folder_of_the_full_session_id(tmp_path: Path) -> None:
    session_id = "e9796bf5-2747-4a1b-9c3d-5e6f7a8b9c0d"
    app = make_chrys_app(tmp_path, settings=Settings(locale="en"))
    async with app.run_test(size=(120, 30)) as pilot:
        main = app._main_screen
        assert main is not None
        main.chat_session_id = session_id
        chat_panel = main.query_one(ChatPanel)
        await wait_for(lambda: chat_panel.session_id == session_id, pilot=pilot, description="session id bound")

        await click_when_settled(pilot, chat_panel, offset=(4, 0))
        await wait_for(
            lambda: isinstance(app.screen, SessionTitleDialog) and app.screen.is_mounted,
            pilot=pilot,
            description="session title dialog open",
        )

        dialog = app.screen
        assert isinstance(dialog, SessionTitleDialog)
        assert dialog._session_id == "e9796bf52747"
        assert dialog._session_folder == app._state_store.session_dir(session_id)
        assert dialog._session_folder.name == "e9796bf52747"


@pytest.mark.asyncio
async def test_chat_tab_asks_for_the_user_while_an_approval_dialog_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    titles: list[str] = []

    def write(_app: object, title: str) -> bool:
        titles.append(title)
        return True

    monkeypatch.setattr(session_title, "set_app_terminal_title", write)

    def spinning() -> bool:
        mark, _sep, rest = titles[-1].partition(" ")
        return rest == "Task" and mark in (*SPINNER_FRAMES, STATIC_SPINNER)

    bus = EventBus()
    app = make_chrys_app(tmp_path, engine=SessionGenerationEngine(), event_bus=bus, settings=Settings(locale="en"))
    async with app.run_test(size=(120, 36)) as pilot:
        main = app._main_screen
        assert main is not None
        main._session_title.set_session_title_state(custom="Task")
        main._set_agent_running(True)
        await wait_for(spinning, pilot=pilot, description="chat run spinner")

        await bus.publish(
            ApprovalRequest(request_id="req", tool_name="zsh", tool_kind="shell", args={"command": "ls"}),
            raise_handler_errors=True,
        )
        await wait_for(
            lambda: isinstance(app.screen, ApprovalDialog) and app.screen.is_mounted,
            pilot=pilot,
            description="approval dialog mounted",
        )
        await wait_for(lambda: titles[-1] == "? Task", pilot=pilot, description="tab asks for the user")
        await bus.publish(ApprovalCancelled(request_id="req"), raise_handler_errors=True)
        await wait_for(lambda: app.screen is main and spinning(), pilot=pilot, description="spinner again")
