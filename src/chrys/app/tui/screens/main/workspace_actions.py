# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workspace-changing actions for the main screen.

The working directory can be deleted or moved outside the app. Nothing
watches it: a submit refused for that reason, the end of a turn, or a session
restore asks the user for another folder here, one prompt at a time.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.foundation.i18n import DisplayPath, msg
from chrys.foundation.platform import safe_getcwd
from chrys.foundation.platform.paths import resolve_workspace_path

if TYPE_CHECKING:
    from textual.screen import Screen

    from chrys.app.tui.i18n import LocaleController
    from chrys.app.tui.screens.dialogs.file_picker import FilePicker
    from chrys.app.tui.screens.main.ports import StartWorker, WorkspaceView

type MissingDirReason = Literal["submit", "turn_end", "restore"]

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
_MISSING_TITLE = msg("tui.workspace.missing.title", fallback="Working Folder Not Found")
_MISSING_MESSAGE = msg(
    "tui.workspace.missing.message",
    fallback="{path} was deleted or moved. Choose a folder to continue.",
)
_MISSING_MESSAGE_SUBMIT = msg(
    "tui.workspace.missing.message_submit",
    fallback="{path} was deleted or moved. Choose a folder to continue. Your message is still in the input box.",
)
_MISSING_MESSAGE_RESTORE = msg(
    "tui.workspace.missing.message_restore",
    fallback="The folder of this session, {path}, was deleted or moved. Choose a folder to open the session in.",
)
_MISSING_CHOOSE = msg("tui.workspace.missing.choose", fallback="Choose Folder…")
_MISSING_MESSAGES = {
    "submit": _MISSING_MESSAGE_SUBMIT,
    "turn_end": _MISSING_MESSAGE,
    "restore": _MISSING_MESSAGE_RESTORE,
}


@dataclass(frozen=True, slots=True)
class WorkspaceCallbacks:
    """Screen-owned effects required by workspace actions."""

    start_worker: StartWorker
    debug: Callable[[str, str], None]
    allow_change: Callable[[], bool] = lambda: True
    selected_cwd: Callable[[], str] = lambda: ""
    apply_workflow_cwd: Callable[[str], Awaitable[bool]] | None = None
    workflow_mode: Callable[[], bool] = lambda: False


class WorkspaceController:
    """Coordinate working-directory picker and /chdir publishing."""

    def __init__(
        self,
        *,
        state: MainScreenState,
        services: MainScreenServices,
        view: WorkspaceView,
        callbacks: WorkspaceCallbacks,
        locale_controller: LocaleController | None = None,
    ) -> None:
        self._state = state
        self._services = services
        self._view = view
        self._callbacks = callbacks
        self._locale_controller = locale_controller
        self._missing_prompt_active = False

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

    def prompt_missing_working_dir(self, reason: MissingDirReason) -> None:
        """Ask for another folder when the current one is gone; the chosen one becomes the workspace."""
        if not self._missing_prompt_active and not os.path.isdir(self.workspace_cwd()):
            self._callbacks.start_worker(partial(self._replace_missing_working_dir, reason))

    def check_after_turn(self) -> None:
        """Once the finished turn is saved and its hooks ran, prompt if its folder is gone.

        The folder is checked after that wait: it can disappear while the turn
        saves or its hooks run. A refused submit has no turn to wait for and
        prompts on its own.
        """
        if self._state.submit.active or self._callbacks.workflow_mode():
            return
        lifecycle = self._services.turn_lifecycle_task()
        if (lifecycle is None or lifecycle.done()) and os.path.isdir(self.workspace_cwd()):
            return
        self._callbacks.start_worker(
            partial(
                self._prompt_after_turn,
                lifecycle,
                self._services.session_generation(),
                self._state.run.generation,
            )
        )

    async def choose_replacement_dir(self, missing: str, *, reason: MissingDirReason) -> str | None:
        """Ask for a folder in place of *missing*; ``None`` when the user declines or a prompt is open."""
        if self._missing_prompt_active:
            return None
        self._missing_prompt_active = True
        try:
            from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog

            confirmed = await self._ask(
                ConfirmDialog(
                    title=_MISSING_TITLE.bind(),
                    message=_MISSING_MESSAGES[reason].bind(path=DisplayPath(missing)),
                    confirm_label=_MISSING_CHOOSE.bind(),
                    locale_controller=self._locale_controller,
                )
            )
            if confirmed is not True:
                return None
            chosen = await self._ask(self._directory_picker(missing))
        finally:
            self._missing_prompt_active = False
        return chosen if isinstance(chosen, str) and os.path.isdir(chosen) else None

    async def _prompt_after_turn(
        self, lifecycle: asyncio.Task[None] | None, session_generation: int, run_generation: int
    ) -> None:
        if lifecycle is not None and not lifecycle.done():
            # Waited, not awaited: cancelling this worker must not cancel the turn.
            await asyncio.wait({lifecycle})
        if (
            self._services.session_generation() != session_generation
            or self._state.run.generation != run_generation
            or self._state.run.agent_running
            or self._state.run.agent_loading
            or self._callbacks.workflow_mode()
        ):
            return
        await self._replace_missing_working_dir("turn_end")

    async def _replace_missing_working_dir(self, reason: MissingDirReason) -> None:
        missing = self.workspace_cwd()
        if os.path.isdir(missing):
            return
        chosen = await self.choose_replacement_dir(missing, reason=reason)
        if chosen is not None:
            await self.apply_chdir(chosen)

    async def _ask(self, screen: Screen[Any]) -> object:
        """Push *screen* and wait for its dismissal result."""
        result: asyncio.Future[object] = asyncio.get_running_loop().create_future()

        def settle(value: object) -> None:
            if not result.done():
                result.set_result(value)

        self._view.push_screen(screen, settle)
        return await result

    def _push_directory_picker(self) -> None:
        self._view.push_screen(self._directory_picker(self.workspace_cwd()), self.on_chdir_dialog_result)

    def _directory_picker(self, initial_path: str) -> FilePicker:
        from chrys.app.tui.screens.dialogs.file_picker import FilePicker, FilePickerMode
        from chrys.app.tui.screens.main.recent_dirs import WorkspaceMruRecentDirs

        # Provider works without a state store too: it reads an existing MRU
        # index but never creates one (creation requires the one-time session
        # backfill, which needs the store).
        return FilePicker(
            mode=FilePickerMode.FOLDER,
            initial_path=initial_path,
            title=_CHANGE_DIRECTORY.bind(),
            recent_paths=WorkspaceMruRecentDirs(
                self._services.state_store,
                max_entries=self._services.workspace_mru_max_entries,
            ),
        )
