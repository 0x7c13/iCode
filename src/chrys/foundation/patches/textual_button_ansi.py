# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: revert the textual 8.2.5 ansi-theme button DEFAULT_CSS block.

Problem
-------
Textual 8.2.5 added a new ``Button:ansi.-style-flat, Button:ansi.-style-default``
CSS block to ``Button.DEFAULT_CSS`` which retargets buttons under any
``ansi=True`` theme (the new ``ansi-dark`` / ``ansi-light`` built-ins, as well
as chrys' ``chrys-ansi``).  The new styling is visually inconsistent with the
chrys ANSI palette.

Solution
--------
Strip the new ansi-targeting block back out of ``Button.DEFAULT_CSS`` so
buttons fall back to the original (pre-8.2.5) rules under ansi themes.

Some Chrys TUI modules import ``textual.widgets.Button`` before
``bootstrap_runtime()`` applies site-package patches, so this module also
exposes an in-process monkey patch for the already-imported ``Button`` class.

The exact 8.2.5 block (whitespace-significant) lives in
``textual_button_ansi_old.txt`` next to this module, so trailing whitespace
in the upstream CSS round-trips verbatim through the patcher's literal
substring match.
"""

from pathlib import Path

from chrys.foundation.patches.patcher import FilePatch, register

_OLD = (Path(__file__).parent / "textual_button_ansi_old.txt").read_text(encoding="utf-8")
_NEW = 'DEFAULT_CSS = """\n    Button {'
_RUNTIME_OLD = _OLD.removeprefix('DEFAULT_CSS = """')
_RUNTIME_NEW = _NEW.removeprefix('DEFAULT_CSS = """')
_RUNTIME_PATCH_MARKER = "_chrys_textual_button_ansi_css"


def apply_runtime_patch() -> None:
    """Patch ``Button.DEFAULT_CSS`` in the current process."""
    try:
        from textual.widgets import Button
    except ImportError:
        return

    if getattr(Button, _RUNTIME_PATCH_MARKER, False):
        return

    if _RUNTIME_OLD in Button.DEFAULT_CSS:
        Button.DEFAULT_CSS = Button.DEFAULT_CSS.replace(_RUNTIME_OLD, _RUNTIME_NEW, 1)

    setattr(Button, _RUNTIME_PATCH_MARKER, True)


register(
    FilePatch(
        package="textual",
        module_file="widgets/_button.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Revert 8.2.5 ansi-theme button styling block",
    ),
)
