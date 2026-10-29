# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Choosing a graphics protocol from the environment."""

from __future__ import annotations

import pytest

from chrys.app.tui.graphics.detect import GraphicsProtocol, detect_graphics_protocol

KITTY = GraphicsProtocol.KITTY
SIXEL = GraphicsProtocol.SIXEL
BLOCKS = GraphicsProtocol.BLOCKS


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        pytest.param({"TERM": "xterm-ghostty", "TERM_PROGRAM": "ghostty"}, KITTY, id="ghostty"),
        pytest.param({"TERM": "xterm-256color", "TERM_PROGRAM": "ghostty"}, KITTY, id="ghostty-generic-term"),
        pytest.param({"TERM": "xterm-ghostty"}, KITTY, id="ghostty-over-ssh"),
        pytest.param({"TERM": "xterm-kitty", "KITTY_WINDOW_ID": "1"}, KITTY, id="kitty"),
        pytest.param({"TERM": "xterm-256color", "KITTY_WINDOW_ID": "1"}, BLOCKS, id="inherited-kitty-window"),
        pytest.param({"TERM": "xterm-kitty", "TERM_PROGRAM": "vscode"}, BLOCKS, id="vscode-started-from-kitty"),
        pytest.param({"TERM": "alacritty", "TERM_PROGRAM": "ghostty"}, BLOCKS, id="alacritty-started-from-ghostty"),
        pytest.param(
            {"TERM": "xterm", "TERM_PROGRAM": "ghostty", "XTERM_VERSION": "XTerm(390)"}, BLOCKS, id="xterm-from-ghostty"
        ),
        pytest.param(
            {"TERM": "xterm-256color", "TERM_PROGRAM": "ghostty", "VTE_VERSION": "7600"},
            BLOCKS,
            id="gnome-terminal-from-ghostty",
        ),
        pytest.param(
            {"TERM": "xterm-256color", "TERM_PROGRAM": "ghostty", "INSIDE_EMACS": "vterm"},
            BLOCKS,
            id="emacs-vterm-in-ghostty",
        ),
        pytest.param(
            {"TERM": "xterm-256color", "TERM_PROGRAM": "ghostty", "NVIM": "/tmp/nvim.1/0"},
            BLOCKS,
            id="neovim-terminal-in-ghostty",
        ),
        pytest.param(
            {"TERM": "xterm-ghostty", "TERM_PROGRAM": "ghostty", "VTE_VERSION": "7600"},
            KITTY,
            id="ghostty-started-from-gnome-terminal",
        ),
        pytest.param({"TERM": "xterm-256color", "TERM_PROGRAM": "WezTerm"}, BLOCKS, id="wezterm"),
        pytest.param({"WT_SESSION": "abc"}, SIXEL, id="windows-terminal"),
        pytest.param({"WT_SESSION": "abc", "TERM": "xterm-256color"}, SIXEL, id="wsl-in-windows-terminal"),
        pytest.param({"WT_SESSION": "abc", "TERM_PROGRAM": "vscode"}, BLOCKS, id="vscode-started-from-wt"),
        pytest.param({"WT_SESSION": "abc", "TERM": "alacritty"}, BLOCKS, id="alacritty-started-from-wt"),
        pytest.param({"WT_SESSION": "abc", "TERM": "foot"}, BLOCKS, id="foot-started-from-wsl-in-wt"),
        pytest.param(
            {"WT_SESSION": "abc", "TERM": "xterm-256color", "KONSOLE_VERSION": "240802"},
            BLOCKS,
            id="konsole-started-from-wsl-in-wt",
        ),
        pytest.param({"TERM": "xterm-ghostty", "TMUX": "/tmp/tmux,1,0"}, BLOCKS, id="tmux"),
        pytest.param({"TERM": "screen-256color", "WT_SESSION": "abc"}, BLOCKS, id="screen-term"),
        pytest.param({"WT_SESSION": "abc", "ZELLIJ": "0"}, BLOCKS, id="zellij"),
        pytest.param({"TERM": "xterm-ghostty", "NO_COLOR": ""}, BLOCKS, id="no-color"),
        pytest.param({"TERM": "xterm-256color", "TERM_PROGRAM": "Apple_Terminal"}, BLOCKS, id="apple-terminal"),
        pytest.param({}, BLOCKS, id="unknown"),
    ],
)
def test_protocol_follows_the_terminal_the_environment_names(env: dict[str, str], expected: GraphicsProtocol) -> None:
    assert detect_graphics_protocol(env, truecolor=True) is expected


def test_placeholders_need_truecolor_to_name_their_image() -> None:
    env = {"TERM": "xterm-ghostty", "TERM_PROGRAM": "ghostty"}

    assert detect_graphics_protocol(env, truecolor=False) is BLOCKS
    assert detect_graphics_protocol({"WT_SESSION": "abc"}, truecolor=False) is SIXEL
