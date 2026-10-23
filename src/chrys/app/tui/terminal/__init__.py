# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The embedded terminal: an emulator, the widget that shows it, and the shell that runs in it."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chrys.app.tui.terminal.shell import Shell as Shell
    from chrys.app.tui.terminal.widget import Terminal as Terminal

# Title and launcher helpers import this package without loading the terminal itself.
_EXPORTS = {"Shell": "shell", "Terminal": "widget"}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily resolve the shell or the terminal widget."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the default module attributes plus facade exports."""
    return sorted({*globals(), *__all__})
