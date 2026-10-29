# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which apps get terminal graphics at start."""

from __future__ import annotations

import os

import pytest
from rich.color import ColorSystem

from chrys.app.tui.graphics import detect, manager
from chrys.app.tui.graphics.detect import GraphicsProtocol, detect_graphics_protocol
from tests.support.terminal_images import TerminalImagesApp

_GHOSTTY = {"TERM": "xterm-ghostty", "TERM_PROGRAM": "ghostty"}
_WINDOWS_TERMINAL = {"WT_SESSION": "abc"}


def _start(
    app: TerminalImagesApp, monkeypatch: pytest.MonkeyPatch, *, enabled: bool
) -> manager.TerminalGraphics | None:
    """Start graphics as a terminal-backed app would: test apps are headless."""
    with monkeypatch.context() as patch:
        patch.setattr(TerminalImagesApp, "is_headless", property(lambda _app: False))
        return manager.start_terminal_graphics(app, enabled=enabled)


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    """Make ``env`` the only terminal the environment names, whatever terminal runs the tests."""
    detected = ("TERM", "TERM_PROGRAM", "WT_SESSION", "NO_COLOR")
    for name in (*detected, *detect._MULTIPLEXER_VARIABLES, *detect._OTHER_TERMINAL_VARIABLES):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    ("env", "enabled", "expected"),
    [
        pytest.param(_GHOSTTY, True, GraphicsProtocol.KITTY, id="ghostty"),
        pytest.param(_WINDOWS_TERMINAL, True, GraphicsProtocol.SIXEL, id="windows-terminal"),
        pytest.param(_GHOSTTY, False, None, id="setting-off"),
        pytest.param({"TERM": "xterm-256color"}, True, None, id="unknown-terminal"),
    ],
)
async def test_graphics_start_for_known_terminals_when_enabled(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], enabled: bool, expected: GraphicsProtocol | None
) -> None:
    _set_env(monkeypatch, env)
    app = TerminalImagesApp(None)
    async with app.run_test(size=(40, 10)):
        app.console._color_system = ColorSystem.TRUECOLOR
        graphics = _start(app, monkeypatch, enabled=enabled)

        assert (None if graphics is None else graphics.protocol) is expected
        assert manager.graphics_for(app) is graphics
        if graphics is not None:
            await graphics.close()
            assert manager.graphics_for(app) is None


async def test_sixel_needs_the_textual_its_overlay_was_written_against(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, _WINDOWS_TERMINAL)
    assert detect_graphics_protocol(os.environ, truecolor=True) is GraphicsProtocol.SIXEL
    monkeypatch.setattr(manager, "TEXTUAL_VERSION", "0.0.0")
    app = TerminalImagesApp(None)
    async with app.run_test(size=(40, 10)):
        app.console._color_system = ColorSystem.TRUECOLOR
        assert _start(app, monkeypatch, enabled=True) is None


async def test_headless_apps_draw_half_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, _GHOSTTY)
    app = TerminalImagesApp(None)
    async with app.run_test(size=(40, 10)):
        app.console._color_system = ColorSystem.TRUECOLOR
        assert manager.start_terminal_graphics(app, enabled=True) is None
        assert manager.graphics_for(app) is None
