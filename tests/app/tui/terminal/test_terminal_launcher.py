# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for TUI terminal launcher capability helpers."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chrys.app.tui.terminal import launcher
from chrys.app.tui.terminal.launcher import (
    TerminalLaunchError,
    build_chrys_tui_argv,
    can_access_local_desktop,
    can_open_new_chrys_window,
    is_remote_execution_environment,
    is_remote_ssh_session,
)
from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.i18n import Localizer
from chrys.foundation.i18n.formatting import format_message

_MACOS_WINDOW_SERVER_PROBE = launcher._macos_window_server_session_is_available
_WINDOWS_DESKTOP_PROBE = launcher._windows_desktop_shell_is_visible
_TERMINAL_ENV_MARKERS = (
    "SSH_CONNECTION",
    "SSH_CLIENT",
    "SSH_TTY",
    "CODESPACES",
    "CLOUD_SHELL",
    "GITPOD_WORKSPACE_ID",
    "REPL_ID",
    "TEXTUAL_DRIVER",
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "MIR_SOCKET",
    "PYAPP",
    "TERM",
    "TERM_PROGRAM",
    "LC_TERMINAL",
    "TERMINAL_EMULATOR",
    "GHOSTTY_BIN_DIR",
    "GHOSTTY_RESOURCES_DIR",
    "WEZTERM_EXECUTABLE",
    "WEZTERM_PANE",
    "WT_SESSION",
    "WT_PROFILE_ID",
    "KITTY_LISTEN_ON",
    "GNOME_TERMINAL_SCREEN",
    "GNOME_TERMINAL_SERVICE",
    "KONSOLE_VERSION",
    "KONSOLE_DBUS_SESSION",
    "KONSOLE_DBUS_WINDOW",
)


@pytest.fixture(autouse=True)
def _isolate_terminal_launcher_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _TERMINAL_ENV_MARKERS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(launcher, "_macos_window_server_session_is_available", lambda: True)
    monkeypatch.setattr(launcher, "_windows_desktop_shell_is_visible", lambda: True)


def _mock_platform(monkeypatch: pytest.MonkeyPatch, *, macos: bool = False, windows: bool = False) -> None:
    monkeypatch.setattr(launcher, "get_platform", lambda: SimpleNamespace(is_macos=macos, is_windows=windows))


class _FakeProcess:
    def __init__(self, return_code: int | None = 0) -> None:
        self._return_code = return_code

    def wait(self, timeout: float | None = None) -> int:
        if self._return_code is None:
            raise subprocess.TimeoutExpired("fake-terminal", 0 if timeout is None else timeout)
        return self._return_code


class _FakeCFunction:
    def __init__(self, result: object) -> None:
        self.result = result
        self.argtypes: object = None
        self.restype: object = None
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> object:
        self.calls.append(args)
        if callable(self.result):
            return cast(Any, self.result)(*args)
        return self.result


def test_remote_ssh_session_detection() -> None:
    assert is_remote_ssh_session({"SSH_CONNECTION": "127.0.0.1 50000 127.0.0.1 22"})
    assert is_remote_ssh_session({"SSH_CLIENT": "127.0.0.1 50000 22"})
    assert is_remote_ssh_session({"SSH_TTY": "/dev/pts/1"})
    assert not is_remote_ssh_session({})


@pytest.mark.parametrize(
    "env",
    (
        {"SSH_CONNECTION": "127.0.0.1 50000 127.0.0.1 22"},
        {"TEXTUAL_DRIVER": "textual.drivers.web_driver:WebDriver"},
        {"CODESPACES": "true"},
        {"CLOUD_SHELL": "1"},
        {"GITPOD_WORKSPACE_ID": "workspace-1"},
        {"REPL_ID": "12345678-1234-1234-1234-123456789abc"},
    ),
)
def test_remote_execution_environment_detection(env: dict[str, str]) -> None:
    assert is_remote_execution_environment(env)


def test_remote_execution_environment_ignores_false_flags() -> None:
    assert not is_remote_execution_environment({"CODESPACES": "false", "CLOUD_SHELL": "0"})


def test_local_desktop_capability_uses_linux_display_session(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_platform(monkeypatch)

    assert can_access_local_desktop({"DISPLAY": ":0"})
    assert can_access_local_desktop({"WAYLAND_DISPLAY": "wayland-0"})
    assert not can_access_local_desktop({})


def test_local_desktop_capability_uses_windows_window_station(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_platform(monkeypatch, windows=True)
    monkeypatch.setattr(launcher, "_windows_desktop_shell_is_visible", lambda: True)
    assert can_access_local_desktop({})

    monkeypatch.setattr(launcher, "_windows_desktop_shell_is_visible", lambda: False)
    assert not can_access_local_desktop({})


@pytest.mark.parametrize(
    ("window_station", "info_ok", "flags", "shell_window", "expected"),
    (
        (None, True, 1, 1, False),
        (1, False, 1, 1, False),
        (1, True, 0, 1, False),
        (1, True, 1, None, False),
        (1, True, 1, 1, True),
    ),
)
def test_windows_desktop_probe_truth_table(
    monkeypatch: pytest.MonkeyPatch,
    window_station: object,
    info_ok: bool,
    flags: int,
    shell_window: object,
    expected: bool,
) -> None:
    import ctypes

    def write_flags(_station: object, _index: object, buffer: object, *_args: object) -> bool:
        cast(Any, buffer)._obj.flags = flags
        return info_ok

    user32 = SimpleNamespace(
        GetProcessWindowStation=_FakeCFunction(window_station),
        GetUserObjectInformationW=_FakeCFunction(write_flags),
        GetShellWindow=_FakeCFunction(shell_window),
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: user32, raising=False)

    assert _WINDOWS_DESKTOP_PROBE() is expected


def test_windows_desktop_probe_handles_unavailable_user32(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    def fail_to_load(*_args: object, **_kwargs: object) -> object:
        raise OSError("user32 unavailable")

    monkeypatch.setattr(ctypes, "WinDLL", fail_to_load, raising=False)

    assert not _WINDOWS_DESKTOP_PROBE()


def test_local_desktop_capability_uses_macos_window_server_session(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_platform(monkeypatch, macos=True)
    monkeypatch.setattr(launcher, "_macos_window_server_session_is_available", lambda: True)
    assert can_access_local_desktop({})

    monkeypatch.setattr(launcher, "_macos_window_server_session_is_available", lambda: False)
    assert not can_access_local_desktop({})


@pytest.mark.parametrize(("session", "expected"), ((1234, True), (None, False)))
def test_macos_window_server_probe_releases_session(
    monkeypatch: pytest.MonkeyPatch, session: object, expected: bool
) -> None:
    import ctypes

    copy_session = _FakeCFunction(session)
    release = _FakeCFunction(None)
    libraries = {
        launcher._CORE_GRAPHICS_PATH: SimpleNamespace(CGSessionCopyCurrentDictionary=copy_session),
        launcher._CORE_FOUNDATION_PATH: SimpleNamespace(CFRelease=release),
    }
    monkeypatch.setattr(ctypes, "CDLL", libraries.__getitem__)

    assert _MACOS_WINDOW_SERVER_PROBE() is expected
    assert release.calls == ([(session,)] if expected else [])


def test_macos_window_server_probe_handles_unavailable_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    def fail_to_load(_path: str) -> object:
        raise OSError("framework unavailable")

    monkeypatch.setattr(ctypes, "CDLL", fail_to_load)

    assert not _MACOS_WINDOW_SERVER_PROBE()


def test_new_window_capability_is_disabled_remotely(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_platform(monkeypatch)
    assert not can_open_new_chrys_window({"SSH_CONNECTION": "127.0.0.1 50000 127.0.0.1 22"})
    assert can_open_new_chrys_window({"DISPLAY": ":0"})


def test_build_chrys_tui_argv_uses_python_module(monkeypatch) -> None:
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python3")

    assert build_chrys_tui_argv("session-1") == [
        "/usr/bin/python3",
        "-m",
        "chrys.app.cli.app",
        "__tui_subprocess__",
        "--session",
        "session-1",
    ]


def test_build_chrys_tui_argv_prefers_pyapp(monkeypatch) -> None:
    monkeypatch.setattr(launcher.Path, "is_file", lambda _path: True)
    monkeypatch.setenv("PYAPP", "/Applications/Chrys/chrys")

    assert build_chrys_tui_argv("session-1") == [
        "/Applications/Chrys/chrys",
        "__tui_subprocess__",
        "--session",
        "session-1",
    ]


def test_build_chrys_tui_argv_ignores_pyapp_sentinel(monkeypatch) -> None:
    monkeypatch.setenv("PYAPP", "1")
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python3")

    assert build_chrys_tui_argv("session-1") == [
        "/usr/bin/python3",
        "-m",
        "chrys.app.cli.app",
        "__tui_subprocess__",
        "--session",
        "session-1",
    ]


def test_launch_new_window_rejects_unavailable_environment(monkeypatch) -> None:
    monkeypatch.setenv("SSH_TTY", "/dev/pts/1")

    try:
        launcher.launch_new_chrys_window("session-1")
    except TerminalLaunchError as exc:
        assert str(exc) == f"Opening a new {APP_DISPLAY_NAME} window is not available in the current environment."
        assert exc.display_message is not None
        assert format_message(exc.display_message) == str(exc)
        assert Localizer("zh-Hans").render(exc.display_message) == f"当前环境无法打开新的 {APP_DISPLAY_NAME} 窗口。"
    else:
        raise AssertionError("expected TerminalLaunchError")


def test_macos_launch_prefers_ghostty_when_running_in_ghostty(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    _mock_platform(monkeypatch, macos=True)
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(launcher.shutil, "which", lambda _name: None)

    def fake_popen(argv: list[str], **kwargs: object) -> object:
        calls.append((argv, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

    launcher.launch_new_chrys_window("session-1", cwd="/repo")

    assert calls == [
        (
            [
                "/usr/bin/open",
                "-na",
                "Ghostty",
                "--args",
                "-e",
                "/bin/sh",
                "-lc",
                "cd /repo && /usr/bin/python3 -m chrys.app.cli.app __tui_subprocess__ --session session-1",
            ],
            {
                "cwd": None,
                "stdin": launcher.subprocess.DEVNULL,
                "stdout": launcher.subprocess.DEVNULL,
                "stderr": launcher.subprocess.DEVNULL,
            },
        )
    ]


def test_macos_launch_uses_wezterm_tab_when_running_in_wezterm(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setenv("TERM_PROGRAM", "WezTerm")
    _mock_platform(monkeypatch, macos=True)
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/local/bin/wezterm" if name == "wezterm" else None)

    def fake_popen(argv: list[str], **kwargs: object) -> object:
        calls.append((argv, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

    launcher.launch_new_chrys_window("session-1", cwd="/repo")

    assert calls == [
        (
            [
                "/usr/local/bin/wezterm",
                "cli",
                "spawn",
                "--new-tab",
                "--cwd",
                "/repo",
                "--",
                "/usr/bin/python3",
                "-m",
                "chrys.app.cli.app",
                "__tui_subprocess__",
                "--session",
                "session-1",
            ],
            {
                "cwd": None,
                "stdin": launcher.subprocess.DEVNULL,
                "stdout": launcher.subprocess.DEVNULL,
                "stderr": launcher.subprocess.DEVNULL,
            },
        )
    ]


def test_macos_wezterm_failure_falls_back_to_terminal(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setenv("TERM_PROGRAM", "WezTerm")
    _mock_platform(monkeypatch, macos=True)
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/local/bin/wezterm" if name == "wezterm" else None)

    def fake_popen(argv: list[str], **kwargs: object) -> _FakeProcess:
        calls.append((argv, kwargs))
        if argv[0] == "/usr/local/bin/wezterm":
            return _FakeProcess(1)
        return _FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

    launcher.launch_new_chrys_window("session-1", cwd="/repo")

    assert calls[0][0][:4] == ["/usr/local/bin/wezterm", "cli", "spawn", "--new-tab"]
    assert calls[1][0][:2] == ["/usr/bin/osascript", "-e"]


def test_macos_terminal_fallback_failure_raises(monkeypatch) -> None:
    _mock_platform(monkeypatch, macos=True)
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(launcher.shutil, "which", lambda _name: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs: _FakeProcess(1))

    try:
        launcher.launch_new_chrys_window("session-1", cwd="/repo")
    except TerminalLaunchError as exc:
        assert "launcher exited with status 1" in str(exc)
    else:
        raise AssertionError("expected TerminalLaunchError")


def test_windows_launch_uses_windows_terminal_tab(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setenv("WT_SESSION", "window-session")
    _mock_platform(monkeypatch, windows=True)
    monkeypatch.setattr(launcher.sys, "executable", r"C:\Python\python.exe")
    monkeypatch.setattr(launcher.shutil, "which", lambda name: r"C:\Windows\System32\wt.exe" if name == "wt" else None)

    def fake_popen(argv: list[str], **kwargs: object) -> object:
        calls.append((argv, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

    launcher.launch_new_chrys_window("session-1", cwd=r"C:\repo")

    assert calls == [
        (
            [
                r"C:\Windows\System32\wt.exe",
                "-w",
                "0",
                "new-tab",
                "-d",
                r"C:\repo",
                r"C:\Python\python.exe",
                "-m",
                "chrys.app.cli.app",
                "__tui_subprocess__",
                "--session",
                "session-1",
            ],
            {
                "cwd": None,
                "stdin": launcher.subprocess.DEVNULL,
                "stdout": launcher.subprocess.DEVNULL,
                "stderr": launcher.subprocess.DEVNULL,
            },
        )
    ]


def test_linux_launch_uses_kitty_tab_when_remote_control_is_available(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setenv("KITTY_LISTEN_ON", "unix:/tmp/kitty")
    _mock_platform(monkeypatch)
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/usr/bin/kitty" if name == "kitty" else None)

    def fake_popen(argv: list[str], **kwargs: object) -> object:
        calls.append((argv, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

    launcher.launch_new_chrys_window("session-1", cwd="/repo")

    assert calls == [
        (
            [
                "/usr/bin/kitty",
                "@",
                "launch",
                "--type=tab",
                "--cwd",
                "/repo",
                "--",
                "/usr/bin/python3",
                "-m",
                "chrys.app.cli.app",
                "__tui_subprocess__",
                "--session",
                "session-1",
            ],
            {
                "cwd": None,
                "stdin": launcher.subprocess.DEVNULL,
                "stdout": launcher.subprocess.DEVNULL,
                "stderr": launcher.subprocess.DEVNULL,
            },
        )
    ]


def test_linux_launch_uses_gnome_terminal_tab_when_running_in_gnome_terminal(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setenv("GNOME_TERMINAL_SCREEN", "screen")
    _mock_platform(monkeypatch)
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(
        launcher.shutil,
        "which",
        lambda name: "/usr/bin/gnome-terminal" if name == "gnome-terminal" else None,
    )

    def fake_popen(argv: list[str], **kwargs: object) -> object:
        calls.append((argv, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

    launcher.launch_new_chrys_window("session-1", cwd="/repo")

    assert calls == [
        (
            [
                "/usr/bin/gnome-terminal",
                "--tab",
                "--",
                "/usr/bin/python3",
                "-m",
                "chrys.app.cli.app",
                "__tui_subprocess__",
                "--session",
                "session-1",
            ],
            {
                "cwd": "/repo",
                "stdin": launcher.subprocess.DEVNULL,
                "stdout": launcher.subprocess.DEVNULL,
                "stderr": launcher.subprocess.DEVNULL,
            },
        )
    ]


def test_linux_launch_uses_available_terminal(monkeypatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    _mock_platform(monkeypatch)
    monkeypatch.setattr(launcher.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(
        launcher.shutil,
        "which",
        lambda name: "/usr/bin/gnome-terminal" if name == "gnome-terminal" else None,
    )

    def fake_popen(argv: list[str], **kwargs: object) -> object:
        calls.append((argv, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

    launcher.launch_new_chrys_window("session-1", cwd="/repo")

    assert calls == [
        (
            [
                "/usr/bin/gnome-terminal",
                "--",
                "/usr/bin/python3",
                "-m",
                "chrys.app.cli.app",
                "__tui_subprocess__",
                "--session",
                "session-1",
            ],
            {
                "cwd": "/repo",
                "stdin": launcher.subprocess.DEVNULL,
                "stdout": launcher.subprocess.DEVNULL,
                "stderr": launcher.subprocess.DEVNULL,
            },
        )
    ]
