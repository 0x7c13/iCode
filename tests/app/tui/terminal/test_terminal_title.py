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
