# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Progress dialogs buffer reports their operation sends while Textual removes them."""

from __future__ import annotations

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
