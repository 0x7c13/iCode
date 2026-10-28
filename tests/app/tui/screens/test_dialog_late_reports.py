# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Progress dialogs buffer reports their operation sends while Textual removes them."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from chrys.app.tui.screens.dialogs.agent_load import AgentLoadDialog
from chrys.app.tui.screens.dialogs.connection_test import ConnectionTestDialog
from chrys.app.tui.screens.dialogs.fork_session import ForkSessionDialog
from chrys.app.tui.screens.dialogs.image_compression import ImageCompressionDialog


def _report_workflow_load(dialog: AgentLoadDialog) -> None:
    dialog.update_title("Loading Workflow echo")
    dialog.update_progress("Loading definition", phase="workflow_definition", status="done")
    dialog.update_finish_progress("Showing workflow")
    dialog.finish("Workflow loaded")


def _report_agent_failure(dialog: AgentLoadDialog) -> None:
    dialog.set_result(False, "Agent failed")


def _report_fork(dialog: ForkSessionDialog) -> None:
    dialog.set_success("abc123")
    dialog.set_error("Fork failed")


def _report_connection(dialog: ConnectionTestDialog) -> None:
    dialog.set_result(True, "Connected")


def _report_compressed(dialog: ImageCompressionDialog) -> None:
    dialog.finish()


_DIALOGS: dict[str, tuple[type[ModalScreen], Callable]] = {
    "agent-load-finish": (AgentLoadDialog, _report_workflow_load),
    "agent-load-result": (AgentLoadDialog, _report_agent_failure),
    "fork-session": (ForkSessionDialog, _report_fork),
    "connection-test": (ConnectionTestDialog, _report_connection),
    "image-compression": (ImageCompressionDialog, _report_compressed),
}


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("placeholder")


@pytest.mark.parametrize("removal", ["pop", "app-exit"])
@pytest.mark.parametrize("kind", list(_DIALOGS))
async def test_reports_during_and_after_removal_are_buffered(kind: str, removal: str) -> None:
    base, report = _DIALOGS[kind]
    reported: list[bool] = []

    class Reporting(base):
        # A relative CSS_PATH resolves against the subclass's module.
        CSS_PATH = Path(inspect.getfile(base)).with_name(str(base.CSS_PATH))

        def on_unmount(self) -> None:
            # Removal has pruned every child; the dialog itself has not unmounted yet.
            reported.append(not self.children)
            report(self)

    dialog = Reporting()
    app = _Host()
    async with app.run_test():
        await app.push_screen(dialog)
        if removal == "pop":
            await app.pop_screen()
    # Leaving run_test re-raises what a report raised on the way out.
    assert reported == [True]
    report(dialog)


async def _push_as_the_app_exits(dialog: ModalScreen, window: str) -> None:
    """Push *dialog* as the app shuts down, at the moment *window* names; leaving re-raises what its mount raised.

    - ``prune``: as the test's last step: the app then closes its screens, which prunes the dialog's children first;
    - ``late``: in the app's last running turn: the dialog mounts once the app stopped, and none of its children do;
    - ``exit``: in the turn the app begins to exit, as the user quits: the dialog is detached and no child mounts.
    """
    if window == "exit":

        class Quitting(_Host):
            def on_ready(self) -> None:
                self.exit()
                self.push_screen(dialog)

        app = Quitting()
        await app.run_async(headless=True)
        if app._exception is not None:
            raise app._exception
        return
    app = _Host()
    async with app.run_test():
        if window == "prune":
            app.push_screen(dialog)
        else:
            # Pushed as leaving run_test yields, before the app shuts down.
            asyncio.get_running_loop().call_soon(app.push_screen, dialog)


# What the dialog sees as it mounts in each window: whether it is pruning, the app running, and it attached.
_MOUNTED_IN = {"prune": (True, False, True), "late": (False, False, True), "exit": (False, True, False)}


@pytest.mark.parametrize("window", list(_MOUNTED_IN))
@pytest.mark.parametrize("reported", [False, True], ids=["unreported", "reported"])
@pytest.mark.parametrize("kind", list(_DIALOGS))
async def test_a_dialog_that_mounts_as_the_app_exits_renders_nothing(kind: str, reported: bool, window: str) -> None:
    """An owner can push its dialog as the app shuts down: its children are pruned before it mounts, or never mount."""
    base, report = _DIALOGS[kind]
    mounted: list[tuple[bool, bool, bool]] = []

    class Mounting(base):
        CSS_PATH = Path(inspect.getfile(base)).with_name(str(base.CSS_PATH))

        def on_mount(self) -> None:
            # Dispatch runs the dialog's own handler after this one.
            mounted.append((self._pruning, self.app.is_running, self.is_attached))

    dialog = Mounting()
    if reported:
        report(dialog)
    await _push_as_the_app_exits(dialog, window)
    assert mounted == [_MOUNTED_IN[window]]
