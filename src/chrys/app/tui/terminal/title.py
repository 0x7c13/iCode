# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Terminal title helpers for native TUI runs."""

from __future__ import annotations

import contextlib
import os
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

BASE_TERMINAL_TITLE = ""

_MESSAGE_TITLE_CHARS = 100
_TEXTUAL_WEB_DRIVER = "textual.drivers.web_driver:WebDriver"
_TitleWriter = Callable[[str], None]
_ESC = "\x1b"
_BEL = "\x07"
_ST = "\x9c"


def set_terminal_title(title: str, *, writer: _TitleWriter | None = None) -> None:
    """Set the terminal window title with OSC 0 and OSC 2."""
    if running_under_textual_web_driver():
        return

    safe_title = _title_safe_text(title)
    sequence = f"\x1b]0;{safe_title}\x07\x1b]2;{safe_title}\x07"
    with contextlib.suppress(Exception):
        if writer is not None:
            writer(sequence)
        else:
            sys.stderr.write(sequence)
            sys.stderr.flush()


def set_app_terminal_title(app: object, title: str) -> bool:
    """Write a built title through a Textual app's driver; False when it has none."""
    writer = _app_terminal_title_writer(app)
    if writer is None:
        return False
    set_terminal_title(title, writer=writer)
    return True


def set_app_terminal_title_for_current_cwd(app: object) -> None:
    """Set the terminal title for a Textual app using the current working directory."""
    writer = _app_terminal_title_writer(app)
    if writer is None:
        return
    set_terminal_title_for_current_cwd(writer=writer)


def set_terminal_title_for_cwd(cwd: str | os.PathLike[str], *, writer: _TitleWriter | None = None) -> None:
    """Set the title to the current directory path."""
    set_terminal_title(terminal_title_for_cwd(cwd), writer=writer)


def set_terminal_title_for_current_cwd(*, writer: _TitleWriter | None = None) -> None:
    """Set the title using ``os.getcwd()``, or the base title if cwd is unavailable."""
    try:
        cwd = os.getcwd()
    except OSError:
        set_terminal_title(BASE_TERMINAL_TITLE, writer=writer)
        return
    set_terminal_title_for_cwd(cwd, writer=writer)


def terminal_title_for_cwd(cwd: str | os.PathLike[str]) -> str:
    """Build the terminal title for a workspace directory."""
    raw_path = os.fspath(cwd)
    if not os.path.isdir(raw_path):
        return BASE_TERMINAL_TITLE
    path = _title_safe_text(raw_path)
    if not path:
        return BASE_TERMINAL_TITLE
    return path


@dataclass(frozen=True, slots=True)
class TerminalTabText:
    """A tab's text, cleaned once when it is set; each repaint only adds a mark.

    ``clipped`` texts (prompt previews and titles) keep the marked title
    within the title length; a directory path is never clipped.
    """

    text: str
    clipped: bool = False

    @classmethod
    def for_cwd(cls, cwd: str | os.PathLike[str]) -> TerminalTabText:
        return cls(terminal_title_for_cwd(cwd))

    @classmethod
    def for_message(cls, text: str) -> TerminalTabText:
        """A submitted prompt's preview, or a session or workflow title."""
        return cls(_title_safe_text(text)[:_MESSAGE_TITLE_CHARS], clipped=True)

    def with_mark(self, mark: str) -> str:
        """The title with ``mark`` (a spinner frame or status mark) before it."""
        if not mark:
            return self.text
        title = f"{mark} {self.text}" if self.text else mark
        return title[:_MESSAGE_TITLE_CHARS] if self.clipped else title


def _title_safe_text(text: str) -> str:
    text = _strip_terminal_sequences(text)
    parts: list[str] = []
    pending_space = False
    for ch in text.strip():
        if ch.isspace():
            pending_space = bool(parts)
            continue
        if unicodedata.category(ch).startswith("C"):
            continue
        if pending_space:
            parts.append(" ")
            pending_space = False
        parts.append(ch)
    return "".join(parts)


def _strip_terminal_sequences(text: str) -> str:
    """Remove ANSI/OSC-style control sequences before title filtering."""
    result: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == _ESC:
            i = _skip_esc_sequence(text, i)
            continue
        if ch == "\x9b":
            i = _skip_until_csi_final(text, i + 1)
            continue
        if ch == "\x9d":
            i = _skip_until_string_terminator(text, i + 1)
            continue
        if ch in {"\x90", "\x98", "\x9e", "\x9f"}:
            i = _skip_until_string_terminator(text, i + 1)
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def _skip_esc_sequence(text: str, start: int) -> int:
    next_index = start + 1
    if next_index >= len(text):
        return next_index

    introducer = text[next_index]
    if introducer == "[":
        return _skip_until_csi_final(text, next_index + 1)
    if introducer == "]":
        return _skip_until_string_terminator(text, next_index + 1)
    if introducer in {"P", "X", "^", "_"}:
        return _skip_until_string_terminator(text, next_index + 1)

    i = next_index
    while i < len(text) and 0x20 <= ord(text[i]) <= 0x2F:
        i += 1
    if i < len(text) and 0x30 <= ord(text[i]) <= 0x7E:
        return i + 1
    return next_index + 1


def _skip_until_csi_final(text: str, start: int) -> int:
    i = start
    while i < len(text):
        if 0x40 <= ord(text[i]) <= 0x7E:
            return i + 1
        i += 1
    return i


def _skip_until_string_terminator(text: str, start: int) -> int:
    i = start
    while i < len(text):
        if text[i] == _BEL or text[i] == _ST:
            return i + 1
        if text[i] == _ESC and i + 1 < len(text) and text[i + 1] == "\\":
            return i + 2
        i += 1
    return i


def _app_terminal_title_writer(app: object) -> _TitleWriter | None:
    # Textual has no public raw terminal writer; keep the private driver
    # boundary here instead of spreading it through screens.
    driver = getattr(app, "_driver", None)
    write = getattr(driver, "write", None)
    if not callable(write):
        return None
    return write


def running_under_textual_web_driver() -> bool:
    """Return true when Chrys is hosted by textual-serve."""
    return os.environ.get("TEXTUAL_DRIVER") == _TEXTUAL_WEB_DRIVER
