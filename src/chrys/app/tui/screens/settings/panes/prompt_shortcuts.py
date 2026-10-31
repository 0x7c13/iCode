# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The Prompt Shortcuts tab: where the ``/`` list reads prompt shortcuts from, save-on-change.

Laid out like the agent configuration's Skills tab, but global: the folders
are user settings, and the working folder and workspace-relative entries are
resolved against whatever the workspace is when the ``/`` list opens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on
from textual.containers import Vertical, VerticalGroup
from textual.message import Message
from textual.widgets import Button, Label

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.widgets import Checkbox, ConfigAddButton, HatchedEmptyState
from chrys.app.tui.widgets import EnhancedInput as Input
from chrys.app.tui.widgets.click_affordance import ClickAffordance
from chrys.app.tui.widgets.directory_path import DirectoryPathCard
from chrys.foundation.config.settings import parse_prompt_shortcut_paths
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import sanitize_legacy_scalar
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import surrogate_safe_text
from chrys.foundation.platform.paths import is_absolute_path
from chrys.service.prompt_shortcuts import global_shortcuts_dir, project_shortcuts_dir, working_folder_dirs

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.screens.settings.ports import SettingsPanelPorts

USER_FOLDER_KEY = "prompt_shortcuts.load_user_folder"
WORKING_FOLDER_KEY = "prompt_shortcuts.load_working_folder"
PATHS_KEY = "prompt_shortcuts.paths"

_TITLE = msg("tui.settings.prompt_shortcuts.title", fallback="Prompt Shortcut Directories")
_DESCRIPTION = msg(
    "tui.settings.prompt_shortcuts.description",
    fallback="Configure where the / list reads prompt shortcuts from. Only Markdown (.md) files are supported for now.",
)
_LOAD_USER_FOLDER = msg(
    "tui.settings.prompt_shortcuts.load_user_folder", fallback="Load prompt shortcuts from user folder (if present)"
)
_LOAD_WORKING_FOLDER = msg(
    "tui.settings.prompt_shortcuts.load_working_folder",
    fallback="Load prompt shortcuts from working folder (if present)",
)
_WORKING_FOLDER_SCOPE = msg(
    "tui.settings.prompt_shortcuts.working_folder_scope",
    fallback="{path} and its parent folders up to the repository root",
)
_GUIDE_LINK = msg("tui.settings.prompt_shortcuts.guide_link", fallback="How to write shortcuts →")
_ADD = msg("tui.settings.prompt_shortcuts.add", fallback="+ Add")
_EMPTY = msg("tui.settings.prompt_shortcuts.empty", fallback="No other folders added")
_DIRECTORY = msg("tui.settings.prompt_shortcuts.directory", fallback="Prompt Shortcut Directory")
_PLACEHOLDER_RELATIVE_POSIX = msg(
    "tui.settings.prompt_shortcuts.placeholder.relative_posix", fallback="relative/path/to/prompts"
)
_PLACEHOLDER_RELATIVE_WINDOWS = msg(
    "tui.settings.prompt_shortcuts.placeholder.relative_windows", fallback=r"relative\path\to\prompts"
)
_PLACEHOLDER_ABSOLUTE_MACOS = msg(
    "tui.settings.prompt_shortcuts.placeholder.absolute_macos", fallback="/Users/you/prompts"
)
_PLACEHOLDER_ABSOLUTE_LINUX = msg(
    "tui.settings.prompt_shortcuts.placeholder.absolute_linux", fallback="/home/you/prompts"
)
_PLACEHOLDER_ABSOLUTE_WINDOWS = msg(
    "tui.settings.prompt_shortcuts.placeholder.absolute_windows", fallback=r"C:\path\to\prompts"
)


def _path_text(path: Path) -> str:
    """A folder path as one display line: no control codes, no undecodable bytes."""
    return sanitize_legacy_scalar(surrogate_safe_text(os.fspath(path)))


class PromptDirCard(DirectoryPathCard):
    """One extra prompt shortcut folder: the skill folder card's layout, folder picker and preview."""

    # The skill card's Browse button keeps a border its agent screen strips;
    # in the settings dialog that border would fill its one row and hide the label.
    DEFAULT_CSS = """
    PromptDirCard Button.sk-browse-btn,
    PromptDirCard Button.sk-browse-btn:hover,
    PromptDirCard Button.sk-browse-btn:focus,
    PromptDirCard Button.sk-browse-btn.-active {
        border: none;
        height: 1;
        padding: 0 1;
    }
    """

    _title_label = _DIRECTORY
    _placeholder_relative_posix = _PLACEHOLDER_RELATIVE_POSIX
    _placeholder_relative_windows = _PLACEHOLDER_RELATIVE_WINDOWS
    _placeholder_absolute_macos = _PLACEHOLDER_ABSOLUTE_MACOS
    _placeholder_absolute_linux = _PLACEHOLDER_ABSOLUTE_LINUX
    _placeholder_absolute_windows = _PLACEHOLDER_ABSOLUTE_WINDOWS

    def _open_browse(self) -> None:
        """Open the folder picker for this row."""
        from chrys.app.tui.screens.dialogs.file_picker import open_folder_picker

        open_folder_picker(self.app, *self._browse_from(), self._take_browsed)

    def entry(self) -> str | None:
        """Return the folder to save, ``""`` for a blank row, or None while it is invalid."""
        raw = self.get_path()
        if not raw:
            return ""
        if self.is_relative_checked() == is_absolute_path(os.path.expanduser(raw)):
            return None
        return raw


SHORTCUTS_GUIDE_TOPIC = "shortcuts"
"""The user guide's page on writing and using shortcuts (its id in ``docs/index.yaml``)."""


class ShortcutsGuideLink(ClickAffordance):
    """Opens the user guide at its shortcuts page; the dialog that owns the guide handles the request."""

    class Clicked(Message):
        """Posted when the link is clicked."""

    CLICK_MESSAGE = Clicked


class PromptShortcutsPane(VerticalGroup):
    """Checkboxes and folder cards over the ``prompt_shortcuts.*`` settings; every valid change saves."""

    DEFAULT_CLASSES = "settings-pane"

    # Styled in settings.tcss: the dialog stylesheet's Checkbox rules would beat DEFAULT_CSS.

    _CHECKBOX_KEYS: ClassVar[dict[str, str]] = {
        "ps-load-user-folder": USER_FOLDER_KEY,
        "ps-load-working-folder": WORKING_FOLDER_KEY,
    }

    def __init__(self, ports: SettingsPanelPorts) -> None:
        super().__init__(id="prompt-shortcuts-content")
        self._ports = ports
        self._workspace = ports.workspace_cwd()
        self._paths = self._saved_paths()
        # Per row, the folder last saved for it ("" if none): what the row keeps saved while invalid.
        self._row_saved = list(self._paths)
        self._rebuilding = False
        self._ready = False

    @property
    def projected(self) -> bool:
        """Whether the controls show the ports' settings; until then there is nothing to repaint."""
        return self._ready

    def _saved_paths(self) -> list[str]:
        try:
            return parse_prompt_shortcut_paths(self._ports.projected_value(PATHS_KEY))
        except ValueError, TypeError, RecursionError:
            return []

    def compose(self) -> ComposeResult:
        yield Label(Text(self._render_message(_TITLE.bind())), classes="ps-title", id="ps-title")
        yield Label(Text(self._render_message(_DESCRIPTION.bind())), classes="ps-description", id="ps-description")
        yield ShortcutsGuideLink(Text(self._render_message(_GUIDE_LINK.bind())), id="ps-guide-link")
        yield Checkbox(
            self._render_message(_LOAD_USER_FOLDER.bind()),
            value=bool(self._ports.projected_value(USER_FOLDER_KEY)),
            id="ps-load-user-folder",
        )
        user_dir = global_shortcuts_dir(get_platform().config_dir)
        yield Label(Text(_path_text(user_dir)), classes="ps-folder-path", id="ps-user-folder-path")
        yield Checkbox(
            self._render_message(_LOAD_WORKING_FOLDER.bind()),
            value=bool(self._ports.projected_value(WORKING_FOLDER_KEY)),
            id="ps-load-working-folder",
        )
        yield Label(self._working_folder_text(), classes="ps-folder-path", id="ps-working-folder-path")
        yield ConfigAddButton(self._render_message(_ADD.bind()), id="ps-add-btn", compact=True)
        yield Vertical(id="ps-dirs")

    def _working_folder_text(self) -> Text:
        """The working folder's existing shortcut folders, or the rule when it has none yet."""
        found = working_folder_dirs(self._workspace) if self._workspace else []
        if found:
            return Text("\n".join(_path_text(folder) for folder in found))
        nearest = project_shortcuts_dir(Path(self._workspace) if self._workspace else Path())
        return Text(self._render_message(_WORKING_FOLDER_SCOPE.bind(path=_path_text(nearest))))

    async def on_mount(self) -> None:
        if self._pruning or not self.is_attached or not self.app.is_running:
            return
        await self._rebuild_dirs()
        self._ready = True

    async def _rebuild_dirs(self) -> None:
        container = self.query_one("#ps-dirs", Vertical)
        self._rebuilding = True
        try:
            with self.app.batch_update():
                await container.remove_children()
                if not self._paths:
                    await container.mount(HatchedEmptyState(self._render_message(_EMPTY.bind()), classes="ps-empty"))
                    return
                await container.mount(
                    *(
                        PromptDirCard(path, index=index, workspace_cwd=self._workspace)
                        for index, path in enumerate(self._paths)
                    )
                )
        finally:
            self._rebuilding = False

    def _cards(self) -> list[PromptDirCard]:
        return list(self.query(PromptDirCard))

    def _collect(self) -> list[str]:
        """Current rows, blanks included, so indices still match the cards."""
        if self._rebuilding:
            return list(self._paths)
        cards = self._cards()
        if len(cards) != len(self._paths):
            return list(self._paths)
        return [card.get_path() for card in cards]

    def _save_paths(self) -> None:
        """Save every row as it stands: a valid row its folder, a blank row nothing.

        A row being fixed (invalid, showing why in its preview) keeps the folder
        last saved for it, or nothing if it never had one, so a row removed or
        edited elsewhere is saved meanwhile and an invalid row never drops what
        it held.
        """
        if not self._ready or self._rebuilding:
            return
        entries = [card.entry() for card in self._cards()]
        if len(entries) != len(self._row_saved):
            return
        kept = [saved if entry is None else entry for entry, saved in zip(entries, self._row_saved, strict=True)]
        self._row_saved = kept
        paths = [path for path in kept if path]
        if paths == self._saved_paths():
            # Closing, a language switch or a blur with nothing changed writes nothing.
            return
        self._ports.apply_live(PATHS_KEY, json.dumps(paths, ensure_ascii=False))

    @on(Checkbox.Changed)
    def _on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        key = self._CHECKBOX_KEYS.get(event.checkbox.id or "")
        if key is None:
            # A folder card's "Workspace relative" box: its preview already updated.
            self._save_paths()
            return
        event.stop()
        if self._ready:
            self._ports.apply_live(key, event.value)

    @on(Input.Changed)
    def _on_path_changed(self, event: Input.Changed) -> None:
        # Saving reloads the / list, which walks the folder: a path being typed
        # is saved once it is submitted or left, never each partial path ("/").
        if not event.input.has_focus:
            self._save_paths()

    @on(Input.Submitted)
    @on(Input.Blurred)
    def _on_path_done(self, _event: Input.Submitted | Input.Blurred) -> None:
        self._save_paths()

    @on(Button.Pressed, "#ps-add-btn")
    async def _on_add(self, event: Button.Pressed) -> None:
        event.stop()
        self._paths = self._collect()
        self._paths.insert(0, "")
        self._row_saved.insert(0, "")
        await self._rebuild_dirs()

    @on(DirectoryPathCard.Removed)
    async def _on_remove(self, event: DirectoryPathCard.Removed) -> None:
        event.stop()
        self._paths = self._collect()
        if 0 <= event.index < len(self._paths):
            self._paths.pop(event.index)
            self._row_saved.pop(event.index)
            await self._rebuild_dirs()
            self._save_paths()

    def commit_pending(self) -> None:
        """Save a folder being typed: closing the dialog may come before its field's blur."""
        self._save_paths()

    def project(self) -> None:
        """Repaint from the live settings without saving; a folder being typed keeps its text."""
        for checkbox_id, key in self._CHECKBOX_KEYS.items():
            checkbox = self.query_one(f"#{checkbox_id}", Checkbox)
            with checkbox.prevent(Checkbox.Changed):
                checkbox.value = bool(self._ports.projected_value(key))
        if self._rebuilding or any(card.has_focus_within for card in self._cards()):
            return
        saved = self._saved_paths()
        # Blank rows are not saved: they stay until the folders themselves differ.
        if saved != [path for path in self._collect() if path]:
            self._paths = saved
            self._row_saved = list(saved)
            self.call_later(self._rebuild_dirs)

    def refresh_localization(self) -> None:
        """Replace the pane's text; folder cards are rebuilt from what they hold, in the new language."""
        self.query_one("#ps-title", Label).update(Text(self._render_message(_TITLE.bind())))
        self.query_one("#ps-description", Label).update(Text(self._render_message(_DESCRIPTION.bind())))
        self.query_one("#ps-guide-link", ShortcutsGuideLink).update(Text(self._render_message(_GUIDE_LINK.bind())))
        self.query_one("#ps-load-user-folder", Checkbox).label = self._render_message(_LOAD_USER_FOLDER.bind())
        self.query_one("#ps-load-working-folder", Checkbox).label = self._render_message(_LOAD_WORKING_FOLDER.bind())
        self.query_one("#ps-working-folder-path", Label).update(self._working_folder_text())
        self.query_one("#ps-add-btn", Button).label = self._render_message(_ADD.bind())
        if self._ready:
            self._paths = self._collect()
            self.call_later(self._rebuild_dirs)

    def _render_message(self, reference: MessageRef) -> str:
        return render_str(widget_localizer(self), reference)


__all__ = ["PATHS_KEY", "USER_FOLDER_KEY", "WORKING_FOLDER_KEY", "PromptShortcutsPane"]
