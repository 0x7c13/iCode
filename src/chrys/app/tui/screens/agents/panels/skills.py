# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Skills configuration panel — composable widget for the Skills tab."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Label, Static

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.screens.agents.panels.path_entry import _WORKSPACE_RELATIVE, PathEntryCard
from chrys.app.tui.screens.agents.skill_paths import normalize_skill_path_for_compare
from chrys.app.tui.screens.agents.validation_messages import (
    CONTEXT_ERROR,
    DUPLICATE_SKILL_DIRECTORY,
    FIELD_POSITIVE_INTEGER,
    FIELD_REQUIRED,
    FIELD_VALID_INTEGER,
    PATH_ABSOLUTE_DISABLE,
    PATH_FIELD,
    PATH_RELATIVE_ENABLE,
    PATHS_MATCH_CASE_INSENSITIVE,
    PATHS_MATCH_NORMALIZED,
    SCRIPT_TIMEOUT_FIELD,
    SKILL_DIRECTORY_CONTEXT,
)
from chrys.app.tui.screens.agents.validation_messages import (
    EXECUTION_TIMEOUT as _EXECUTION_TIMEOUT,
)
from chrys.app.tui.screens.agents.validation_messages import (
    SKILL_DIRECTORY as _SKILL_DIRECTORY,
)
from chrys.app.tui.widgets import Checkbox, ConfigActionButton, ConfigAddButton, HatchedEmptyState
from chrys.app.tui.widgets import EnhancedInput as Input
from chrys.foundation.i18n import DisplayBlock, MessageRef, msg

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.service.profiles.agents.schema import SkillsConfig

_ALL_EXTENSIONS = [".py", ".ts", ".js", ".sh", ".rb", ".ps1"]

_PATH_DOES_NOT_EXIST = msg("tui.agent_config.skills.path_missing", fallback="Path does not exist")
_ABSOLUTE_PATH_ERROR = msg(
    "tui.agent_config.skills.absolute_path_error",
    fallback="Absolute path - Turn off Workspace relative",
)
_RESOLVED_AT_RUNTIME = msg(
    "tui.agent_config.skills.resolved_at_runtime",
    fallback="Resolved at session runtime",
)
_MISSING_IN_WORKSPACE = msg(
    "tui.agent_config.skills.missing_in_workspace",
    fallback="Missing in current workspace",
)
_RELATIVE_PATH_ERROR = msg(
    "tui.agent_config.skills.relative_path_error",
    fallback="Relative path - Enable Workspace relative",
)
_PREVIEW_NOTE = msg("tui.agent_config.skills.preview_note", fallback="  · {note}")
_AUTO_LOAD_COVERED = msg(
    "tui.agent_config.skills.auto_load_covered",
    fallback="Also auto-loaded when Load skills from working folder and Load project skills are on",
)
_CURRENT_PATH = msg("tui.agent_config.skills.current_path", fallback="Current: {path}{note}")
_SKILL_DIRECTORIES = msg("tui.agent_config.skills.title", fallback="Skill Directories")
_SKILL_DIRECTORIES_DESCRIPTION = msg(
    "tui.agent_config.skills.description",
    fallback="Configure directories where the agent searches for skills",
)
_LOAD_USER_FOLDER = msg(
    "tui.agent_config.skills.load_user_folder",
    fallback="Load skills from user folder (if present)",
)
_LOAD_USER_FOLDER_TOOLTIP = msg(
    "tui.agent_config.skills.load_user_folder_tooltip",
    fallback="Auto-load skills from the shared cross-tool agents skills directory in your home folder.",
)
_LOAD_WORKING_FOLDER = msg(
    "tui.agent_config.skills.load_working_folder",
    fallback="Load skills from working folder (if present)",
)
_LOAD_WORKING_FOLDER_TOOLTIP = msg(
    "tui.agent_config.skills.load_working_folder_tooltip",
    fallback=(
        "Auto-load skills from the agents skills directory under the current working folder when Load project "
        "skills is on under Settings → Security → Project trust. Reloaded automatically when the workspace cwd "
        "changes (/chdir or file picker)."
    ),
)
_ADD = msg("tui.agent_config.skills.add", fallback="+ Add")
_ALLOWED_EXTENSIONS = msg(
    "tui.agent_config.skills.allowed_extensions",
    fallback="Allowed Script Extensions",
)
_EMPTY = msg("tui.agent_config.skills.empty", fallback="No skill directories configured")
_SKILL_PLACEHOLDER_RELATIVE_POSIX = msg(
    "tui.agent_config.skills.placeholder.relative_posix",
    fallback="relative/path/to/skills",
)
_SKILL_PLACEHOLDER_RELATIVE_WINDOWS = msg(
    "tui.agent_config.skills.placeholder.relative_windows",
    fallback=r"relative\path\to\skills",
)
_SKILL_PLACEHOLDER_ABSOLUTE_MACOS = msg(
    "tui.agent_config.skills.placeholder.absolute_macos",
    fallback="/Users/you/skills",
)
_SKILL_PLACEHOLDER_ABSOLUTE_LINUX = msg(
    "tui.agent_config.skills.placeholder.absolute_linux",
    fallback="/home/you/skills",
)
_SKILL_PLACEHOLDER_ABSOLUTE_WINDOWS = msg(
    "tui.agent_config.skills.placeholder.absolute_windows",
    fallback=r"C:\path\to\skills",
)


def _render_context_error(widget: object, context: str, reference: MessageRef) -> str:
    localizer = widget_localizer(widget)
    return render_str(
        localizer,
        CONTEXT_ERROR.bind(
            context=DisplayBlock(context),
            message=DisplayBlock(render_str(localizer, reference)),
        ),
    )


def _render_duplicate_match_detail(widget: object) -> str:
    from chrys.foundation.platform import get_platform

    definition = (
        PATHS_MATCH_CASE_INSENSITIVE if get_platform().is_macos or get_platform().is_windows else PATHS_MATCH_NORMALIZED
    )
    return render_str(widget_localizer(widget), definition.bind())


# ── Extension toggle chip ────────────────────────────────────────────


class ExtChip(ConfigActionButton):
    """A toggle chip for script extensions."""

    DEFAULT_CSS = """
    ExtChip {
        min-width: 7;
        height: 1;
        border: solid $tui-border-foreground 10%;
        background: $foreground 4%;
        color: $text-muted 50%;
        margin: 0 1 0 0;
    }
    ExtChip.-on {
        border: solid $tui-border-secondary $border-opacity;
        background: $secondary 15%;
        color: $secondary;
    }
    ExtChip:hover {
        border: solid $tui-border-foreground 20%;
        background: $foreground 8%;
        color: $text;
    }
    ExtChip.-on:hover {
        border: solid $tui-border-secondary $border-opacity;
        background: $secondary 25%;
        color: $secondary;
    }
    ExtChip:focus,
    ExtChip.-active {
        border: solid $tui-border-foreground 10%;
        background: $foreground 4%;
        color: $text-muted 50%;
    }
    ExtChip.-on:focus,
    ExtChip.-on.-active {
        border: solid $tui-border-secondary $border-opacity;
        background: $secondary 15%;
        color: $secondary;
    }
    """


# ── Single skill directory card ──────────────────────────────────────


class SkillDirCard(PathEntryCard):
    """A single skill directory entry."""

    DEFAULT_CSS = """
    SkillDirCard .sk-row {
        height: auto;
        margin: 0 0 1 0;
    }
    SkillDirCard .sk-header-row {
        height: auto;
        margin: 0;
    }
    SkillDirCard .sk-label {
        height: 1;
        color: $text-muted;
        margin: 0 0 1 0;
    }
    SkillDirCard .sk-separator {
        display: none;
    }
    SkillDirCard Input {
        height: 1;
        border: none;
        background: $foreground 8%;
        padding: 0 1;
        margin: 0;
    }
    SkillDirCard Input:focus {
        border: none;
        background: $foreground 12%;
    }
    SkillDirCard .sk-field {
        width: 1fr;
        height: auto;
        margin: 1 0 0 0;
    }
    SkillDirCard .sk-ext-row {
        height: auto;
        margin: 0 0 1 0;
    }
    SkillDirCard .sk-path-row {
        height: 1;
        width: 1fr;
        margin: 0;
    }
    SkillDirCard .sk-path-row Input {
        width: 1fr;
        min-width: 18;
    }
    SkillDirCard .sk-browse-btn {
        min-width: 10;
        height: 1;
        border: solid $tui-border-secondary $border-opacity;
        background: $secondary 18%;
        color: $secondary;
        text-style: bold;
        margin: 0 0 0 1;
        content-align: center middle;
        text-align: center;
    }
    SkillDirCard .sk-browse-btn:focus {
        border: solid $tui-border-secondary $border-opacity;
        background: $secondary 18%;
        color: $secondary;
    }
    SkillDirCard .sk-browse-btn:hover {
        border: solid $tui-border-secondary $border-opacity;
        background: $secondary 28%;
        color: $secondary;
    }
    SkillDirCard .sk-browse-btn.-active {
        border: solid $tui-border-secondary $border-opacity;
        background: $secondary 18%;
        color: $secondary;
    }
    SkillDirCard .sk-rel-cb {
        width: auto;
        height: 1;
        border: none;
        background: transparent;
        padding: 0;
        margin: 1 0 0 0;
    }
    SkillDirCard .sk-rel-cb:focus {
        background: $foreground 8%;
    }
    SkillDirCard .sk-rel-cb > .toggle--button {
        color: $foreground 35%;
        background: $foreground 6%;
    }
    SkillDirCard .sk-rel-cb.-on > .toggle--button {
        color: $success;
        background: $secondary 12%;
    }
    SkillDirCard .sk-preview {
        width: 1fr;
        height: auto;
        color: $text-muted;
        margin: 1 0 0 0;
        text-wrap: wrap;
        text-overflow: fold;
    }
    """

    class Removed(Message):
        def __init__(self, index: int) -> None:
            self.index = index
            super().__init__()

    _title_label = _SKILL_DIRECTORY
    _id_prefix = "sk"
    _style_prefix = "sk"
    _placeholder_relative_posix = _SKILL_PLACEHOLDER_RELATIVE_POSIX
    _placeholder_relative_windows = _SKILL_PLACEHOLDER_RELATIVE_WINDOWS
    _placeholder_absolute_macos = _SKILL_PLACEHOLDER_ABSOLUTE_MACOS
    _placeholder_absolute_linux = _SKILL_PLACEHOLDER_ABSOLUTE_LINUX
    _placeholder_absolute_windows = _SKILL_PLACEHOLDER_ABSOLUTE_WINDOWS

    def _refresh_preview(self) -> None:
        """Render the current effective path for this row."""
        from chrys.foundation.platform.paths import is_absolute_path

        label = self.query_one(f"#{self._preview_id}", Label)
        raw = self.query_one(f"#{self._path_input_id}", Input).value.strip()
        if not raw:
            label.update("")
            label.display = False
            return
        label.display = True

        expanded = os.path.expanduser(raw)
        resolved_path: Path | None = None
        resolve_failed = False
        resolved_text = expanded
        localizer = widget_localizer(self)
        missing_note = render_str(localizer, _PATH_DOES_NOT_EXIST.bind())
        if self._is_relative:
            if is_absolute_path(expanded):
                label.update(Text(render_str(localizer, _ABSOLUTE_PATH_ERROR.bind()), style="$error"))
                return
            if not self._workspace_cwd:
                label.update(Text(render_str(localizer, _RESOLVED_AT_RUNTIME.bind()), style="dim"))
                return
            resolved_path, resolve_failed = self._resolve_for_preview(Path(self._workspace_cwd) / expanded)
            resolved_text = str(resolved_path)
            missing_note = render_str(localizer, _MISSING_IN_WORKSPACE.bind())
        else:
            if not is_absolute_path(expanded):
                label.update(Text(render_str(localizer, _RELATIVE_PATH_ERROR.bind()), style="$error"))
                return
            candidate = Path(expanded)
            if candidate.is_absolute():
                resolved_path, resolve_failed = self._resolve_for_preview(candidate)
                resolved_text = str(resolved_path)
            else:
                # Foreign-absolute under the other platform's rules
                # (e.g. ``/foo`` on Windows, ``C:\foo`` on POSIX). Keep
                # the typed text as-is and still report missing so the
                # user knows it won't load on this host.
                resolved_path = candidate
                resolved_text = expanded

        note = ""
        missing = resolved_path is not None and (resolve_failed or not resolved_path.is_dir())
        if missing:
            note = render_str(localizer, _PREVIEW_NOTE.bind(note=missing_note))
        if self._workspace_cwd and resolved_path is not None:
            cwd_default, _default_failed = self._resolve_for_preview(Path(self._workspace_cwd) / ".agents" / "skills")
            if resolved_path == cwd_default:
                note += render_str(
                    localizer,
                    _PREVIEW_NOTE.bind(note=render_str(localizer, _AUTO_LOAD_COVERED.bind())),
                )

        label.update(
            Text(
                render_str(localizer, _CURRENT_PATH.bind(path=resolved_text, note=note)),
                style="$warning" if missing else "dim",
            )
        )

    @staticmethod
    def _resolve_for_preview(path: Path) -> tuple[Path, bool]:
        try:
            return path.resolve(), False
        except OSError:
            return path, True

    def _open_browse(self) -> None:
        """Open the folder picker dialog."""
        from chrys.app.tui.screens.dialogs.file_picker import FilePicker, FilePickerMode

        current = self.query_one(f"#sk-path-{self._index}", Input).value.strip()
        from chrys.foundation.platform import safe_getcwd

        current_expanded = os.path.expanduser(current)
        initial = self._workspace_cwd or safe_getcwd()
        if current and os.path.isdir(current_expanded):
            initial = current_expanded

        def _on_result(result: str | None) -> None:
            if result:
                self.query_one(f"#sk-path-{self._index}", Input).value = result

        self.app.push_screen(
            FilePicker(mode=FilePickerMode.FOLDER, initial_path=initial),
            _on_result,
        )

    def _removed_message(self) -> Message:
        return self.Removed(self._index)


# ── Main skills config panel ─────────────────────────────────────────


class SkillsConfigPanel(VerticalScroll):
    """Composable widget for the Skills configuration tab."""

    DEFAULT_CSS = """
    SkillsConfigPanel {
        height: 1fr;
        padding: 0 0 0 2;
        scrollbar-size-vertical: 1;
    }
    SkillsConfigPanel .sk-section-title {
        color: $secondary;
        text-style: bold;
        height: 1;
    }
    SkillsConfigPanel .sk-section-desc {
        color: $text-muted;
        height: 1;
        margin: 0 0 1 0;
    }
    SkillsConfigPanel .sk-field {
        width: 1fr;
        height: auto;
        margin: 0 1 0 0;
    }
    SkillsConfigPanel #sk-add-btn {
        min-width: 10;
        height: 1;
        margin: 0;
    }
    SkillsConfigPanel #sk-auto-load-agents {
        height: 1;
        border: none;
        background: transparent;
        padding: 0;
        margin: 0;
    }
    SkillsConfigPanel #sk-auto-load-agents:focus {
        background: $foreground 8%;
    }
    SkillsConfigPanel #sk-auto-load-agents > .toggle--button {
        color: $foreground 35%;
        background: $foreground 6%;
    }
    SkillsConfigPanel #sk-auto-load-agents.-on > .toggle--button {
        color: $success;
        background: $secondary 12%;
    }
    SkillsConfigPanel #sk-auto-load-cwd {
        height: 1;
        border: none;
        background: transparent;
        padding: 0;
        margin: 0;
    }
    SkillsConfigPanel #sk-auto-load-cwd:focus {
        background: $foreground 8%;
    }
    SkillsConfigPanel #sk-auto-load-cwd > .toggle--button {
        color: $foreground 35%;
        background: $foreground 6%;
    }
    SkillsConfigPanel #sk-auto-load-cwd.-on > .toggle--button {
        color: $success;
        background: $secondary 12%;
    }
    SkillsConfigPanel .sk-auto-load-path {
        height: 1;
        color: $text-muted;
        text-style: dim;
        padding: 0 0 0 4;
        margin: 0 0 1 0;
    }
    SkillsConfigPanel .sk-header-bar {
        height: auto;
        margin: 0 0 1 0;
    }
    SkillsConfigPanel .sk-section-separator {
        height: auto;
        max-height: 1;
        margin: 0 2 0 0;
        border-top: solid $tui-border-foreground 15%;
    }
    SkillsConfigPanel #sk-dirs {
        height: auto;
    }
    SkillsConfigPanel .sk-empty {
        margin: 0 2 1 0;
    }
    SkillsConfigPanel .sk-label {
        height: 1;
        color: $text-muted;
        margin: 1 0 0 0;
    }
    SkillsConfigPanel .sk-ext-row {
        height: auto;
        margin: 0 0 1 0;
    }
    SkillsConfigPanel .sk-timeout-row {
        height: auto;
        margin: 0 0 1 0;
    }
    SkillsConfigPanel .sk-timeout-hint {
        color: $text-muted;
        width: auto;
        margin: 0 0 0 1;
    }
    SkillsConfigPanel #sk-timeout {
        width: 16;
        height: 1;
        border: none;
        background: $foreground 8%;
        padding: 0 1;
    }
    SkillsConfigPanel #sk-timeout:focus {
        border: none;
        background: $foreground 12%;
    }
    """

    def __init__(
        self,
        skills_config: SkillsConfig | None = None,
        *,
        workspace_cwd: str | None = None,
        read_only: bool = False,
    ) -> None:
        from chrys.service.profiles.agents.schema import SkillsConfig as SC

        self._skills = skills_config or SC()
        self._workspace_cwd = workspace_cwd
        self._read_only = read_only
        # Live source-of-truth for path rows. self._skills.paths is only the
        # initial seed; current values live here and in each card's Input.
        self._paths: list[str] = list(self._skills.paths)
        self._rebuilding_dirs = False
        self._active_extensions: set[str] = set(self._skills.script_extensions)
        super().__init__()

    def compose(self) -> ComposeResult:
        from chrys.foundation.platform import safe_getcwd
        from chrys.service.skills.adapter import user_agents_dir

        localizer = widget_localizer(self)

        # Directories section
        with Vertical(classes="sk-header-bar"):
            yield Label(render_str(localizer, _SKILL_DIRECTORIES.bind()), classes="sk-section-title")
            yield Label(render_str(localizer, _SKILL_DIRECTORIES_DESCRIPTION.bind()), classes="sk-section-desc")
            # Each toggle is a two-line block: short label on the checkbox
            # plus a dimmed resolved path beneath it.  Paths are computed
            # eagerly so the directory does not need to exist yet — the user
            # may be enabling the toggle ahead of time.
            agents_dir = user_agents_dir() / "skills"
            auto_load_cb = Checkbox(
                render_str(localizer, _LOAD_USER_FOLDER.bind()),
                value=self._skills.auto_load_user_agents_skills,
                id="sk-auto-load-agents",
            )
            auto_load_cb.tooltip = render_str(localizer, _LOAD_USER_FOLDER_TOOLTIP.bind())
            auto_load_cb.disabled = self._read_only
            yield auto_load_cb
            yield Label(Text(str(agents_dir)), classes="sk-auto-load-path")

            cwd_for_preview = self._workspace_cwd or safe_getcwd()
            cwd_skills = Path(cwd_for_preview) / ".agents" / "skills"
            cwd_load_cb = Checkbox(
                render_str(localizer, _LOAD_WORKING_FOLDER.bind()),
                value=self._skills.auto_load_cwd_agents_skills,
                id="sk-auto-load-cwd",
            )
            cwd_load_cb.tooltip = render_str(localizer, _LOAD_WORKING_FOLDER_TOOLTIP.bind())
            cwd_load_cb.disabled = self._read_only
            yield cwd_load_cb
            yield Label(Text(str(cwd_skills)), classes="sk-auto-load-path")
            add_button = ConfigAddButton(render_str(localizer, _ADD.bind()), id="sk-add-btn")
            add_button.disabled = self._read_only
            add_button.display = not self._read_only
            yield add_button
        yield Vertical(id="sk-dirs")

        yield Static("", classes="sk-section-separator")

        # Extensions section
        yield Label(render_str(localizer, _ALLOWED_EXTENSIONS.bind()), classes="sk-label")
        with Horizontal(classes="sk-ext-row", id="sk-ext-row"):
            for ext in _ALL_EXTENSIONS:
                chip = ExtChip(ext, id=f"sk-ext-{ext.lstrip('.')}")
                if ext in self._active_extensions:
                    chip.add_class("-on")
                chip.disabled = self._read_only
                yield chip

        # Timeout section
        yield Label(render_str(localizer, _EXECUTION_TIMEOUT.bind()), classes="sk-label")
        with Horizontal(classes="sk-timeout-row"):
            timeout = Input(
                value=str(self._skills.script_timeout),
                placeholder="300",
                id="sk-timeout",
            )
            timeout.disabled = self._read_only
            yield timeout
            yield Static(f"({self._skills.script_timeout}.0s)", id="sk-timeout-hint", classes="sk-timeout-hint")

    async def on_mount(self) -> None:
        await self._rebuild_dirs()

    async def _rebuild_dirs(self) -> None:
        container = self.query_one("#sk-dirs", Vertical)
        self._rebuilding_dirs = True
        try:
            with self.app.batch_update():
                await container.remove_children()
                paths = self._paths
                if not paths:
                    await container.mount(
                        HatchedEmptyState(render_str(widget_localizer(self), _EMPTY.bind()), classes="sk-empty")
                    )
                    return
                await container.mount(
                    *(
                        SkillDirCard(p, index=i, workspace_cwd=self._workspace_cwd, read_only=self._read_only)
                        for i, p in enumerate(paths)
                    )
                )
        finally:
            self._rebuilding_dirs = False

    @on(Button.Pressed, "#sk-add-btn")
    async def _on_add_dir(self, _event: Button.Pressed) -> None:
        if self._read_only:
            return
        self._paths = self._collect_paths()
        self._paths.insert(0, "")
        await self._rebuild_dirs()

    @on(SkillDirCard.Removed)
    async def _on_remove_dir(self, event: SkillDirCard.Removed) -> None:
        if self._read_only:
            return
        self._paths = self._collect_paths()
        if 0 <= event.index < len(self._paths):
            self._paths.pop(event.index)
            await self._rebuild_dirs()

    @on(Button.Pressed)
    def _on_ext_toggle(self, event: Button.Pressed) -> None:
        btn = event.button
        if not isinstance(btn, ExtChip):
            return
        if self._read_only:
            return
        ext = btn.label.plain
        if ext in self._active_extensions:
            self._active_extensions.discard(ext)
            btn.remove_class("-on")
        else:
            self._active_extensions.add(ext)
            btn.add_class("-on")

    @on(Input.Changed, "#sk-timeout")
    def _on_timeout_changed(self, event: Input.Changed) -> None:
        try:
            val = int(event.value)
            self.query_one("#sk-timeout-hint", Static).update(f"({val}.0s)")
        except ValueError:
            self.query_one("#sk-timeout-hint", Static).update("")

    def _collect_paths(self) -> list[str]:
        """Snapshot current path inputs, preserving blanks for row indices."""
        # Old cards disappear individually and may briefly match the new count.
        # Keep the captured edits authoritative until replacement mounting ends.
        if self._rebuilding_dirs:
            return list(self._paths)
        cards = list(self.query(SkillDirCard))
        if len(cards) != len(self._paths):
            return list(self._paths)
        return [card.get_path() for card in cards]

    def get_config(self) -> SkillsConfig:
        """Read current widget state into a SkillsConfig."""
        from chrys.service.profiles.agents.schema import SkillsConfig as SC

        # Collect paths from cards
        paths = [p for p in self._collect_paths() if p]

        # Timeout
        timeout = self._skills.script_timeout
        with contextlib.suppress(ValueError, Exception):
            timeout = int(self.query_one("#sk-timeout", Input).value.strip())

        # Auto-load toggles — fall back to the stored values if the checkboxes
        # can't be queried (e.g. panel never fully mounted).
        auto_load = self._skills.auto_load_user_agents_skills
        with contextlib.suppress(Exception):
            auto_load = bool(self.query_one("#sk-auto-load-agents", Checkbox).value)

        auto_load_cwd = self._skills.auto_load_cwd_agents_skills
        with contextlib.suppress(Exception):
            auto_load_cwd = bool(self.query_one("#sk-auto-load-cwd", Checkbox).value)

        return SC(
            paths=paths,
            inline=self._skills.inline,
            script_timeout=timeout,
            script_extensions=sorted(self._active_extensions),
            auto_load_user_agents_skills=auto_load,
            auto_load_cwd_agents_skills=auto_load_cwd,
        )

    def validate(self) -> list[str]:
        """Validate skills configuration."""
        from chrys.foundation.platform.paths import is_absolute_path

        errors: list[str] = []
        seen_paths: dict[str, int] = {}
        localizer = widget_localizer(self)
        workspace_relative_label = render_str(localizer, _WORKSPACE_RELATIVE.bind())
        duplicate_match_description = _render_duplicate_match_detail(self)
        for i, card in enumerate(self.query(SkillDirCard), 1):
            context = render_str(localizer, SKILL_DIRECTORY_CONTEXT.bind(index=i))
            path = card.get_path()
            if not path:
                errors.append(
                    _render_context_error(
                        self,
                        context,
                        FIELD_REQUIRED.bind(field=render_str(localizer, PATH_FIELD.bind())),
                    )
                )
                continue
            expanded = os.path.expanduser(path)
            is_relative = card.is_relative_checked()
            if is_relative and is_absolute_path(expanded):
                errors.append(
                    _render_context_error(
                        self,
                        context,
                        PATH_ABSOLUTE_DISABLE.bind(
                            path=DisplayBlock(path),
                            label=workspace_relative_label,
                        ),
                    )
                )
                continue
            if not is_relative and not is_absolute_path(expanded):
                errors.append(
                    _render_context_error(
                        self,
                        context,
                        PATH_RELATIVE_ENABLE.bind(
                            path=DisplayBlock(path),
                            label=workspace_relative_label,
                        ),
                    )
                )
                continue
            if is_relative and self._workspace_cwd:
                compare_path = str(Path(self._workspace_cwd) / expanded)
            else:
                compare_path = expanded
            normalized = normalize_skill_path_for_compare(compare_path)
            if normalized in seen_paths:
                errors.append(
                    _render_context_error(
                        self,
                        context,
                        DUPLICATE_SKILL_DIRECTORY.bind(
                            path=DisplayBlock(path),
                            other_index=seen_paths[normalized],
                            detail=DisplayBlock(duplicate_match_description),
                        ),
                    )
                )
            else:
                seen_paths[normalized] = i
        try:
            val = self.query_one("#sk-timeout", Input).value.strip()
            if val:
                t = int(val)
                if t <= 0:
                    errors.append(
                        render_str(
                            localizer,
                            FIELD_POSITIVE_INTEGER.bind(field=render_str(localizer, SCRIPT_TIMEOUT_FIELD.bind())),
                        )
                    )
        except ValueError:
            errors.append(
                render_str(
                    localizer,
                    FIELD_VALID_INTEGER.bind(field=render_str(localizer, SCRIPT_TIMEOUT_FIELD.bind())),
                )
            )
        return errors
