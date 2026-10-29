# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Choose how the terminal can show images, from the environment alone.

No terminal is asked: Textual's input parser does not know the replies (device attributes, cell size,
graphics query) and would type them into the prompt as keys, which Windows Terminal makes likely by
splitting replies under load. A terminal these rules miss shows the half-block images every terminal
shows.

- Ghostty and kitty: the kitty graphics protocol with Unicode placeholders. ``TERM`` names them;
  ``TERM_PROGRAM`` names Ghostty locally, and over SSH when Ghostty sends it with a generic xterm
  ``TERM``.
- Windows Terminal, which sets ``WT_SESSION`` (also inside WSL) but no ``TERM_PROGRAM``, and no
  ``TERM`` or a generic xterm one (WSL, Git Bash): Sixel. Windows Terminal before 1.22 and conhost
  ignore Sixel, leaving the half-blocks underneath.
- A terminal started from a Ghostty or Windows Terminal shell inherits ``TERM_PROGRAM`` or
  ``WT_SESSION``. One that sets its own ``TERM_PROGRAM`` (VS Code, WezTerm, ...) or ``TERM``
  (Alacritty, foot, ...) is not taken for them, nor one that runs under a generic ``TERM`` but marks
  its shells with a variable of its own (xterm, VTE terminals, Konsole, JetBrains, Emacs, Vim and
  Neovim). One that marks nothing still is.
- Inside tmux, screen or zellij the outer terminal is unknown and placeholders need passthrough:
  half-blocks.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_KITTY_TERMS = frozenset({"xterm-ghostty", "xterm-kitty"})
_KITTY_PROGRAMS = frozenset({"ghostty", "kitty"})
_GENERIC_TERM_PREFIX = "xterm"
"""A ``TERM`` Ghostty may run under (``xterm-256color`` when its own entry is unknown on the host), and
the one Windows Terminal's WSL and Git Bash shells set."""
_OTHER_TERMINAL_VARIABLES = (
    "XTERM_VERSION",
    "VTE_VERSION",
    "KONSOLE_VERSION",
    "TERMINAL_EMULATOR",
    "ALACRITTY_WINDOW_ID",
    "INSIDE_EMACS",
    "VIM_TERMINAL",
    "NVIM",
)
"""Set by terminals that may run under a generic ``TERM``, so that ``TERM`` names none of them."""
_MULTIPLEXER_VARIABLES = ("TMUX", "STY", "ZELLIJ")
_MULTIPLEXER_TERM_PREFIXES = ("screen", "tmux")


class GraphicsProtocol(Enum):
    """How images are drawn."""

    BLOCKS = "blocks"
    """Half-block characters, two pixels per cell; works everywhere."""
    KITTY = "kitty"
    """Kitty graphics protocol with Unicode placeholders."""
    SIXEL = "sixel"
    """Sixel images drawn over the half-blocks after each frame."""


def detect_graphics_protocol(env: Mapping[str, str], *, truecolor: bool) -> GraphicsProtocol:
    """The protocol this terminal is known to draw correctly, judged by ``env``.

    ``truecolor`` is whether the app writes 24-bit colors: placeholder cells name their image by color.
    """
    if "NO_COLOR" in env or any(env.get(name) for name in _MULTIPLEXER_VARIABLES):
        return GraphicsProtocol.BLOCKS
    term = env.get("TERM", "")
    if term.startswith(_MULTIPLEXER_TERM_PREFIXES):
        return GraphicsProtocol.BLOCKS
    term_program = env.get("TERM_PROGRAM", "")
    generic_term = (not term or term.startswith(_GENERIC_TERM_PREFIX)) and not any(
        env.get(name) for name in _OTHER_TERMINAL_VARIABLES
    )
    if term in _KITTY_TERMS or (term_program in _KITTY_PROGRAMS and generic_term):
        if term_program and term_program not in _KITTY_PROGRAMS:
            return GraphicsProtocol.BLOCKS
        return GraphicsProtocol.KITTY if truecolor else GraphicsProtocol.BLOCKS
    if env.get("WT_SESSION") and not term_program and generic_term:
        return GraphicsProtocol.SIXEL
    return GraphicsProtocol.BLOCKS


def terminal_cell_pixels(fd: int) -> tuple[int, int] | None:
    """The terminal's cell size in pixels from ``TIOCGWINSZ``, or None where it is not reported."""
    if sys.platform == "win32":
        return None
    import fcntl
    import struct
    import termios

    try:
        packed = fcntl.ioctl(fd, termios.TIOCGWINSZ, bytes(8))
    except OSError:
        return None
    rows, columns, width, height = struct.unpack("HHHH", packed)
    if not rows or not columns or not width or not height:
        return None
    return max(1, width // columns), max(1, height // rows)
