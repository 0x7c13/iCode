# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Clipboard helpers for TUI copy actions."""

from __future__ import annotations

import logging
import os
from typing import Protocol

from chrys.foundation import platform as platform_helpers

logger = logging.getLogger(__name__)

BROWSER_CLIPBOARD_META_TYPE = "chrys_browser_clipboard_copy"
_TEXTUAL_WEB_DRIVER = "textual.drivers.web_driver:WebDriver"

OSC52_COPY_MAX_BYTES = 64 * 1024
"""Terminal-clipboard payload cap; larger copies skip OSC52 to avoid truncation."""


class ClipboardApp(Protocol):
    """Application object that can write Textual's terminal clipboard."""

    def copy_to_clipboard(self, text: str) -> None: ...


class ClipboardReadApp(Protocol):
    """Application object exposing Textual's process-local clipboard."""

    @property
    def clipboard(self) -> str: ...


def clipboard_copy(text: str) -> None:
    """Best-effort copy to the host OS clipboard."""
    try:
        platform_helpers.clipboard_copy(text)
    except Exception:
        logger.debug("Host OS clipboard copy failed", exc_info=True)


def _clipboard_safe_text(text: str) -> str:
    """Return text that can be encoded for terminal and OS clipboard paths."""
    return text.encode("utf-8", errors="backslashreplace").decode("utf-8")


def terminal_clipboard_size(text: str) -> int:
    """Return the UTF-8 byte size of text after clipboard sanitization."""
    return len(_clipboard_safe_text(text).encode("utf-8"))


def paste_text_from_clipboards(app: ClipboardReadApp) -> str:
    """Read the freshest paste text without crossing browser sessions.

    Native TUI sessions prefer the host OS clipboard because Textual's local
    cache does not observe copies made outside Chrys. Browser-host sessions
    must not read the server's clipboard: it belongs to the host process, not
    necessarily the browser client or even the same Chrys session.
    """
    if os.environ.get("TEXTUAL_DRIVER") == _TEXTUAL_WEB_DRIVER:
        return app.clipboard
    try:
        text = platform_helpers.clipboard_paste()
    except Exception:
        logger.debug("Host OS clipboard paste failed", exc_info=True)
        text = ""
    return text or app.clipboard


def _copy_text_to_browser_clipboard(app: ClipboardApp, text: str) -> bool:
    """Ask the textual-serve browser frontend to copy *text* to its clipboard."""
    if os.environ.get("TEXTUAL_DRIVER") != _TEXTUAL_WEB_DRIVER:
        return False
    driver = getattr(app, "_driver", None)
    write_meta = getattr(driver, "write_meta", None)
    if not callable(write_meta):
        return False
    try:
        write_meta({"type": BROWSER_CLIPBOARD_META_TYPE, "text": text})
    except Exception:
        logger.debug("Browser clipboard copy dispatch failed", exc_info=True)
        return False
    return True


def copy_text_to_clipboards(app: ClipboardApp, text: str, *, max_terminal_bytes: int | None = None) -> bool:
    """Best-effort copy to Textual's terminal clipboard and the host OS clipboard.

    The host OS clipboard is attempted even when the terminal clipboard is
    skipped or fails. Returns ``True`` only when the terminal clipboard was
    updated.
    """
    text = _clipboard_safe_text(text)
    terminal_copied = max_terminal_bytes is None or len(text.encode("utf-8")) <= max_terminal_bytes
    if terminal_copied:
        try:
            app.copy_to_clipboard(text)
        except Exception:
            logger.debug("Terminal clipboard copy failed", exc_info=True)
            terminal_copied = False
    _copy_text_to_browser_clipboard(app, text)
    try:
        clipboard_copy(text)
    except Exception:
        # Keep this guard even though the default wrapper is best-effort:
        # tests and callers may replace it with a stricter implementation.
        logger.debug("Host OS clipboard copy failed", exc_info=True)
    return terminal_copied
