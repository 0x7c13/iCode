# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The embedded terminal: an emulator, the widget that shows it, and the shell that runs in it."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chrys.app.tui.terminal.shell import Shell as Shell
    from chrys.app.tui.terminal.widget import Terminal as Terminal

__all__ = ["Shell", "Terminal"]


def __getattr__(name: str) -> object:
    """Keep title and launcher helpers independent of the embedded terminal."""
    if name == "Shell":
        from chrys.app.tui.terminal.shell import Shell

        globals()[name] = Shell
        return Shell
    if name == "Terminal":
        from chrys.app.tui.terminal.widget import Terminal

        globals()[name] = Terminal
        return Terminal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include public exports without loading the terminal implementation."""
    return sorted({*globals(), *__all__})
