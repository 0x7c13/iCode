# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Display messages shared by theme-management screens."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.foundation.i18n import MessageDef, msg

if TYPE_CHECKING:
    from textual.dom import DOMNode

MANAGE = msg("tui.theme_editor.manage", fallback="Edit themes…")
TITLE = msg("tui.theme_editor.title", fallback="Theme editor")
CHOOSE_THEME = msg("tui.theme_editor.choose_theme", fallback="Choose a theme")
CLOSE = msg("tui.theme_editor.close", fallback="Close")
DELETE = msg("tui.theme_editor.delete", fallback="Delete")
DELETE_TITLE = msg("tui.theme_editor.delete_title", fallback="Delete theme?")
DELETE_BODY = msg(
    "tui.theme_editor.delete_body",
    fallback="Delete theme {name}? Its saved file and any unsaved edits will be removed.",
)
DELETED = msg("tui.theme_editor.deleted", fallback="Theme deleted: {name}")
SAVE = msg("tui.theme_editor.save", fallback="Save")
SAVING = msg("tui.theme_editor.saving", fallback="Saving…")
SAVED = msg("tui.theme_editor.saved", fallback="Theme saved: {name}")
SAVE_TITLE = msg("tui.theme_editor.save_title", fallback="Save theme")
NAME = msg("tui.theme_editor.name", fallback="Name")
BUILTIN_SOURCE = msg("tui.theme_editor.builtin_source", fallback="Built-in")
USER_THEME = msg("tui.theme_editor.user_theme", fallback="User theme")
UNDO = msg("tui.theme_editor.undo", fallback="Undo")
REDO = msg("tui.theme_editor.redo", fallback="Redo")
RESET = msg("tui.theme_editor.reset", fallback="Reset all")
RESTORE = msg("tui.theme_editor.restore", fallback="Restore")
CANCEL = msg("tui.theme_editor.cancel", fallback="Cancel")
CONFIRM = msg("tui.theme_editor.confirm", fallback="Confirm")
DARK = msg("tui.theme_editor.dark", fallback="Dark")
LIGHT = msg("tui.theme_editor.light", fallback="Light")
META = msg("tui.theme_editor.meta", fallback="{name} · {mode}")
SAVE_HINT = msg("tui.theme_editor.save_hint", fallback="Saving will apply this theme and remember your choice.")
READ_ONLY_HINT = msg(
    "tui.theme_editor.read_only_hint",
    fallback="Built-in themes cannot be overwritten. New themes must have a unique name.",
)
DISCARD_TITLE = msg("tui.theme_editor.discard_title", fallback="Discard changes?")
DISCARD_BODY = msg("tui.theme_editor.discard_body", fallback="This theme has unsaved changes. Close without saving?")
SWITCH_BODY = msg(
    "tui.theme_editor.switch_body", fallback="This theme has unsaved changes. Discard them and switch themes?"
)
DISCARD = msg("tui.theme_editor.discard", fallback="Discard")
CSS_HINT = msg("tui.theme_editor.css_hint", fallback="CSS value · invalid drafts keep the last valid preview")
OPAQUE = msg("tui.theme_editor.opaque", fallback="This background must remain opaque")
PALETTE_256 = msg("tui.theme_editor.palette_256", fallback="256 colors")
PICK_COLOR = msg("tui.theme_editor.pick_color", fallback="Color: {name}")
PICK_VARIABLE = msg("tui.theme_editor.pick_variable", fallback="Variable: {name}")
CUBE = msg("tui.theme_editor.cube", fallback="Color cube (hue grouped)")
GRAYSCALE = msg("tui.theme_editor.grayscale", fallback="Grayscale")
TRANSPARENT = msg("tui.theme_editor.transparent", fallback="Transparent")

# Keep the technical field table English alongside its CSS variable names.
# Editor actions, metadata and dialogs still follow the application locale.
COLORS = "Colors"
UNSET = "(not set)"
BUTTON_FIELDS = {
    "button-flat-foreground": "Text",
    "button-hover-foreground": "Hover text",
    "button-hover-background": "Hover background",
    "button-disabled-foreground": "Disabled text",
    "button-disabled-background": "Disabled background",
}
GROUPS = {
    "button": "Buttons",
    "border": "Borders",
    "scrollbar": "Scrollbars",
    "block": "Blocks",
    "input": "Inputs",
    "footer": "Footer",
    "markdown": "Markdown headings",
    "muted": "Muted colors",
    "controls": "Controls",
    "misc": "Other",
    "ansi": "ANSI",
}


def text(widget: DOMNode, definition: MessageDef, *, name: str | None = None, mode: str | None = None) -> str:
    if name is None:
        message = definition.bind()
    elif mode is None:
        message = definition.bind(name=name)
    else:
        message = definition.bind(name=name, mode=mode)
    return render_str(widget_localizer(widget), message)
