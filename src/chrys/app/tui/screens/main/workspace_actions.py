# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workspace-changing actions for the main screen."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.foundation.i18n import DisplayPath, msg
from chrys.foundation.platform import safe_getcwd
from chrys.foundation.platform.paths import resolve_workspace_path

if TYPE_CHECKING:
    from chrys.app.tui.screens.main.ports import StartWorker, WorkspaceView

_BUSY_TITLE = msg("tui.workspace.title.busy", fallback="Busy")
_INVALID_PATH_TITLE = msg("tui.workspace.title.invalid_path", fallback="Invalid Path")
_CHANGE_DIRECTORY_BUSY = msg(
    "tui.workspace.change_directory_busy",
    fallback="Cannot change directory while agent is busy",
)
_INVALID_DIRECTORY = msg(
    "tui.workspace.invalid_directory",
    fallback="Not a valid directory: {path}",
)
_CHANGE_DIRECTORY = msg("tui.workspace.change_directory", fallback="Change Directory")


@dataclass(frozen=True, slots=True)
class WorkspaceCallbacks:
    """Screen-owned effects required by workspace actions."""

    start_worker: StartWorker
    debug: Callable[[str, str], None]
    allow_change: Callable[[], bool] = lambda: True
    selected_cwd: Callable[[], str] = lambda: ""
    apply_workflow_cwd: Callable[[str], Awaitable[bool]] | None = None


class WorkspaceController:
    """Coordinate working-directory picker and /chdir publishing."""

    def __init__(
        self,
        *,
        state: MainScreenState,
        services: MainScreenServices,
        view: WorkspaceView,
        callbacks: WorkspaceCallbacks,
    ) -> None:
        self._state = state
        self._services = services
        self._view = view
        self._callbacks = callbacks

    def open_working_dir_picker(self) -> None:
        """Open the file dialog when the user clicks the working directory subtitle."""
        if self._can_change_workspace():
            self._push_directory_picker()

    def start_chdir(self, arg: str) -> object:
        """Run :meth:`chdir` in a screen worker."""
        return self._callbacks.start_worker(partial(self.chdir, arg))

    async def chdir(self, arg: str) -> None:
        """Handle /chdir slash command — change the working directory."""
        path = arg.strip()
        if not path:
            self.open_working_dir_picker()
            return

        current_cwd = self.workspace_cwd()
        resolved = resolve_workspace_path(path, base_cwd=current_cwd)

        if not os.path.isdir(resolved):
            self._view.notify(
                _INVALID_DIRECTORY.bind(path=DisplayPath(resolved)),
                title=_INVALID_PATH_TITLE.bind(),
                severity="error",
                timeout=4,
            )
            return

        try:
            if os.path.samefile(resolved, current_cwd):
                return
        except OSError:
            pass

        await self.apply_chdir(resolved)

    def on_chdir_dialog_result(self, result: str | None) -> None:
        """Callback for the file dialog — apply the selected directory."""
        if result and os.path.isdir(result):
            try:
                if os.path.samefile(result, self.workspace_cwd()):
                    return
            except OSError:
                pass
            self._callbacks.start_worker(partial(self.apply_chdir, result))

    async def apply_chdir(self, resolved: str) -> None:
        """Publish a WorkspaceChange for the selected directory."""
        from chrys.foundation.events.types import WorkspaceChange

        if not self._can_change_workspace():
            return
        apply_workflow = self._callbacks.apply_workflow_cwd
        if apply_workflow is None or not await apply_workflow(resolved):
            await self._services.bus.publish(WorkspaceChange(primary_cwd=resolved))
        self._callbacks.debug("Chdir", resolved)

    def _can_change_workspace(self) -> bool:
        if not self._callbacks.allow_change():
            return False
        if self._state.run.agent_running or self._state.run.agent_loading or self._services.execution_busy():
            self._view.notify(_CHANGE_DIRECTORY_BUSY.bind(), title=_BUSY_TITLE.bind(), severity="warning")
            return False
        return True

    def workspace_cwd(self) -> str:
        """Return the current workspace cwd from explicit main-screen state."""
        return (
            self._callbacks.selected_cwd()
            or self._state.workspace_marker.current_cwd
            or self._state.workspace.current_cwd
            or safe_getcwd()
        )

    def _push_directory_picker(self) -> None:
        from chrys.app.tui.screens.dialogs.file_picker import FilePicker, FilePickerMode
        from chrys.app.tui.screens.main.recent_dirs import WorkspaceMruRecentDirs

        # Provider works without a state store too: it reads an existing MRU
        # index but never creates one (creation requires the one-time session
        # backfill, which needs the store).
        self._view.push_screen(
            FilePicker(
                mode=FilePickerMode.FOLDER,
                initial_path=self.workspace_cwd(),
                title=_CHANGE_DIRECTORY.bind(),
                recent_paths=WorkspaceMruRecentDirs(
                    self._services.state_store,
                    max_entries=self._services.workspace_mru_max_entries,
                ),
            ),
            self.on_chdir_dialog_result,
        )
