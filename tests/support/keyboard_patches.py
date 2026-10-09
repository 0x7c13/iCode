# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Isolate keyboard runtime patches without changing unrelated Textual patches."""

from __future__ import annotations

import pytest
from textual import _xterm_parser

from chrys.foundation.platform import get_platform


@pytest.fixture
def isolated_keyboard_patches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore parser methods, driver hooks and iTerm flags on every platform."""
    # Use the same undo stack as the test: its later overrides must unwind
    # before these original methods are restored.
    parser = _xterm_parser.XTermParser
    monkeypatch.setattr(parser, "_sequence_to_key_events", parser._sequence_to_key_events)
    monkeypatch.setattr(parser, "_parse_extended_key", parser._parse_extended_key)
    monkeypatch.setattr(_xterm_parser, "_re_extended_key", _xterm_parser._re_extended_key)
    monkeypatch.setattr(_xterm_parser, "_MAX_SEQUENCE_SEARCH_THRESHOLD", _xterm_parser._MAX_SEQUENCE_SEARCH_THRESHOLD)
    if get_platform().is_windows:
        from textual.drivers import win32
        from textual.drivers.windows_driver import WindowsDriver

        monkeypatch.setattr(win32, "XTermParser", win32.XTermParser)
        drivers = [WindowsDriver]
    else:
        from textual.drivers import linux_driver
        from textual.drivers.linux_inline_driver import LinuxInlineDriver

        monkeypatch.setattr(linux_driver, "KITTY_REPORT_ALL_KEYS", linux_driver.KITTY_REPORT_ALL_KEYS)
        monkeypatch.setattr(linux_driver, "KITTY_REPORT_ASSOCIATED_TEXT", linux_driver.KITTY_REPORT_ASSOCIATED_TEXT)
        drivers = [linux_driver.LinuxDriver, LinuxInlineDriver]
    for driver in drivers:
        monkeypatch.setattr(driver, "write", driver.write)
        monkeypatch.setattr(driver, "close", driver.close)
