# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for terminal window title helpers."""

from __future__ import annotations

import pytest

from chrys.app.tui.terminal.title import (
    BASE_TERMINAL_TITLE,
    set_app_terminal_title_for_user_message,
    set_terminal_title,
    set_terminal_title_for_current_cwd,
    set_terminal_title_for_user_message,
    terminal_title_for_cwd,
    terminal_title_for_user_message,
)


class _Driver:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, data: str) -> None:
        self.writes.append(data)


class _App:
    def __init__(self) -> None:
        self._driver = _Driver()


def test_terminal_title_writes_osc_0_and_2(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Native TUI title updates should set both icon/window and window title."""
    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)

    set_terminal_title(BASE_TERMINAL_TITLE)

    assert capsys.readouterr().err == f"\x1b]0;{BASE_TERMINAL_TITLE}\x07\x1b]2;{BASE_TERMINAL_TITLE}\x07"


def test_terminal_title_uses_supplied_writer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live Textual updates should write through the active driver."""
    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    writes: list[str] = []

    set_terminal_title(BASE_TERMINAL_TITLE, writer=writes.append)

    assert writes == [f"\x1b]0;{BASE_TERMINAL_TITLE}\x07\x1b]2;{BASE_TERMINAL_TITLE}\x07"]
    assert capsys.readouterr().err == ""


def test_app_terminal_title_uses_textual_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """App helpers should keep Textual private-driver access in one fail-soft boundary."""
    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    app = _App()

    set_app_terminal_title_for_user_message(app, "hello")

    title = "hello"
    assert app._driver.writes == [f"\x1b]0;{title}\x07\x1b]2;{title}\x07"]


def test_app_terminal_title_without_driver_is_noop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)

    set_app_terminal_title_for_user_message(object(), "hello")

    assert capsys.readouterr().err == ""


def test_terminal_title_skips_textual_web_driver(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """textual-serve captures stderr, so OSC title writes must stay native-only."""
    monkeypatch.setenv("TEXTUAL_DRIVER", "textual.drivers.web_driver:WebDriver")

    set_terminal_title(BASE_TERMINAL_TITLE)

    assert capsys.readouterr().err == ""


def test_user_message_title_caps_prompt_preview_at_100_chars() -> None:
    """The title preview should stay bounded while preserving CJK text."""
    text = "帮" * 120

    assert terminal_title_for_user_message(text) == text[:100]


def test_cwd_title_uses_full_existing_current_directory_path(tmp_path) -> None:
    assert terminal_title_for_cwd(tmp_path) == str(tmp_path)


def test_cwd_title_uses_base_title_for_missing_directory(tmp_path) -> None:
    assert terminal_title_for_cwd(tmp_path / "missing") == BASE_TERMINAL_TITLE


def test_current_cwd_title_uses_base_title_when_getcwd_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[str] = []

    def fail_getcwd() -> str:
        raise OSError

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    monkeypatch.setattr("chrys.app.tui.terminal.title.os.getcwd", fail_getcwd)

    set_terminal_title_for_current_cwd(writer=writes.append)

    assert writes == [f"\x1b]0;{BASE_TERMINAL_TITLE}\x07\x1b]2;{BASE_TERMINAL_TITLE}\x07"]


def test_user_message_title_collapses_whitespace_and_ignores_empty_prompt() -> None:
    assert terminal_title_for_user_message("  hello\n\tworld  ") == "hello world"
    assert terminal_title_for_user_message("\n\t") == BASE_TERMINAL_TITLE


def test_user_message_title_strips_terminal_control_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    """User text must not be able to inject extra terminal controls into OSC title writes."""
    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    writes: list[str] = []
    malicious = "hello\n\x1b]0;bad title\x07 there \x1b[31m red \u202eworld\rnext\tline\x9dignored\x9c\x1b(B"

    set_terminal_title_for_user_message(malicious, writer=writes.append)

    title = "hello there red world next line"
    assert writes == [f"\x1b]0;{title}\x07\x1b]2;{title}\x07"]


def test_session_title_terminal_title_appends_fragment() -> None:
    from chrys.app.tui.terminal.title import terminal_title_for_session_title

    assert terminal_title_for_session_title("Login bug fix") == "Login bug fix"
    assert terminal_title_for_session_title("") == BASE_TERMINAL_TITLE
    assert terminal_title_for_session_title("  spaced \n out  ") == "spaced out"
    assert terminal_title_for_session_title("✓ Login bug fix") == "✓ Login bug fix"
    assert terminal_title_for_session_title("✗ Login bug fix") == "✗ Login bug fix"


def test_cwd_title_yields_to_pinned_session_title(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """cwd updates (workspace changes, restores) must not unpin a session title."""
    from types import SimpleNamespace

    from chrys.app.tui.screens.main.screen import MainScreen

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)

    app = _App()
    pinned = SimpleNamespace(app=app, _session_display_title="Login bug fix")
    pinned._terminal_title_with_activity = lambda title: title
    pinned._render_terminal_title = lambda: MainScreen._render_terminal_title(pinned)
    MainScreen._set_terminal_title_for_cwd(pinned, str(tmp_path))
    title = "Login bug fix"
    assert app._driver.writes == [f"\x1b]0;{title}\x07\x1b]2;{title}\x07"]

    app = _App()
    unpinned = SimpleNamespace(
        app=app,
        _session_display_title="",
        _terminal_title_indicator="◇",
    )
    unpinned._render_terminal_title = lambda: MainScreen._render_terminal_title(unpinned)
    MainScreen._set_terminal_title_for_cwd(unpinned, str(tmp_path))
    title = f"◇ {tmp_path}"
    assert app._driver.writes == [f"\x1b]0;{title}\x07\x1b]2;{title}\x07"]


def test_running_animation_preserves_current_prompt_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    from chrys.app.tui.screens.main.screen import MainScreen

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)

    class FakeTimer:
        def __init__(self, callback) -> None:
            self.callback = callback

        def stop(self) -> None:
            return

    class PromptHarness:
        _terminal_title_indicator = MainScreen._terminal_title_indicator
        _terminal_title_with_activity = MainScreen._terminal_title_with_activity
        _render_terminal_title = MainScreen._render_terminal_title
        _set_terminal_title_for_user_message = MainScreen._set_terminal_title_for_user_message
        _sync_terminal_title_activity = MainScreen._sync_terminal_title_activity
        _advance_terminal_title_activity = MainScreen._advance_terminal_title_activity
        _stop_terminal_title_activity_timer = MainScreen._stop_terminal_title_activity_timer

        def __init__(self) -> None:
            self.app = _App()
            self._agent_running = False
            self._session_custom_title = ""
            self._session_fallback_title = "First task"
            self._terminal_title_activity_frame = 0
            self._terminal_title_activity_timer = None
            self._terminal_title_result = ""
            self._terminal_title_source = "session"
            self._terminal_title_content = "Old generated title"

        def set_interval(self, _interval: float, callback):
            return FakeTimer(callback)

    screen = PromptHarness()
    screen._set_terminal_title_for_user_message("Second task")
    screen._agent_running = True
    screen._sync_terminal_title_activity()
    timer = screen._terminal_title_activity_timer
    assert isinstance(timer, FakeTimer)
    timer.callback()

    assert screen._terminal_title_source == "user_message"
    assert screen._terminal_title_content == "Second task"
    assert screen.app._driver.writes == [
        "\x1b]0;Second task\x07\x1b]2;Second task\x07",
        "\x1b]0;◇ Second task\x07\x1b]2;◇ Second task\x07",
        "\x1b]0;◈ Second task\x07\x1b]2;◈ Second task\x07",
    ]


def test_unseeded_cwd_title_state_falls_back_to_workspace_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from types import SimpleNamespace

    from chrys.app.tui.screens.main.screen import MainScreen

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    app = _App()
    screen = SimpleNamespace(
        app=app,
        _terminal_title_source="cwd",
        _terminal_title_content="",
        _terminal_title_indicator="",
        _workspace_cwd=lambda: str(tmp_path),
    )

    MainScreen._render_terminal_title(screen)

    title = str(tmp_path)
    assert app._driver.writes == [f"\x1b]0;{title}\x07\x1b]2;{title}\x07"]


def test_running_terminal_title_cycles_activity_frames_and_returns_to_plain_title() -> None:
    from chrys.app.tui.screens.main.screen import MainScreen

    class FakeTimer:
        def __init__(self, callback) -> None:
            self.callback = callback
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    class ActivityHarness:
        _terminal_title_indicator = MainScreen._terminal_title_indicator
        _terminal_title_with_activity = MainScreen._terminal_title_with_activity
        _sync_terminal_title_activity = MainScreen._sync_terminal_title_activity
        _advance_terminal_title_activity = MainScreen._advance_terminal_title_activity
        _stop_terminal_title_activity_timer = MainScreen._stop_terminal_title_activity_timer
        _mark_terminal_title_completed = MainScreen._mark_terminal_title_completed
        _mark_terminal_title_failed = MainScreen._mark_terminal_title_failed

        def __init__(self) -> None:
            self._agent_running = False
            self._terminal_title_activity_frame = 0
            self._terminal_title_activity_timer = None
            self._terminal_title_result = ""
            self.indicators: list[str] = []

        def set_interval(self, _interval: float, callback):
            return FakeTimer(callback)

        def _render_terminal_title(self) -> None:
            self.indicators.append(self._terminal_title_indicator)

    screen = ActivityHarness()
    screen._agent_running = True
    assert screen._terminal_title_with_activity("Login bug fix") == "◇ Login bug fix"
    screen._sync_terminal_title_activity()
    timer = screen._terminal_title_activity_timer
    assert isinstance(timer, FakeTimer)

    for _ in range(4):
        timer.callback()

    screen._mark_terminal_title_completed()
    screen._agent_running = False
    screen._sync_terminal_title_activity()

    assert screen.indicators == ["◇", "◈", "◆", "◈", "◇", "✓"]
    assert screen._terminal_title_with_activity("Login bug fix") == "✓ Login bug fix"
    assert timer.stopped is True
    assert screen._terminal_title_activity_timer is None

    screen._mark_terminal_title_failed()
    assert screen._terminal_title_with_activity("Login bug fix") == "✗ Login bug fix"


def test_new_clear_and_restored_sessions_clear_terminal_title_result() -> None:
    from chrys.app.tui.screens.main.screen import MainScreen
    from chrys.app.tui.screens.main.state import MainScreenState

    class ResultHarness:
        _set_creating_new_session = MainScreen._set_creating_new_session
        _set_restoring_session = MainScreen._set_restoring_session
        _clear_terminal_title_result = MainScreen._clear_terminal_title_result

        def __init__(self) -> None:
            self._state = MainScreenState()
            self._creating_new_session = False
            self._restoring_session = False
            self._terminal_title_result = "✓"
            self.refreshes = 0

        def _render_terminal_title(self) -> None:
            self.refreshes += 1

    screen = ResultHarness()
    # Both /new and confirmed /clear enter the shared creating-new-session path.
    screen._set_creating_new_session(True)
    assert screen._terminal_title_result == ""

    screen._terminal_title_result = "✗"
    screen._set_restoring_session(True)
    assert screen._terminal_title_result == ""
    assert screen.refreshes == 2
