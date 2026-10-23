# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fresh-process checks for dependencies excluded from TUI startup stages."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from chrys.foundation.platform.process import windows_hidden_subprocess_kwargs


@pytest.mark.parametrize(
    ("module", "excluded"),
    [
        (
            "chrys.app.tui.app",
            [
                "chrys.app.tui.screens.main.screen",
                "chrys.app.tui.screens.settings.dialog",
                "chrys.orchestration.engine.assembly",
                "chrys.orchestration.workflows.runner",
                "chrys.service.acp_client.client",
                "chrys.app.tui.terminal.pty_process",
                "winpty",
            ],
        ),
        ("chrys.app.tui.terminal.title", ["chrys.app.tui.terminal.shell", "winpty"]),
        ("chrys.app.tui.terminal.shell", ["chrys.app.tui.terminal.pty_process", "winpty"]),
        (
            "chrys.app.tui.screens.main",
            [
                "chrys.app.tui.screens.settings.dialog",
                "chrys.app.tui.screens.themes.editor",
                "chrys.app.tui.screens.dialogs.editor",
                "chrys.app.tui.screens.dialogs.app_mode",
            ],
        ),
        (
            "chrys.app.tui.screens.dialogs.confirm",
            ["chrys.app.tui.screens.dialogs.connection_test", "chrys.app.tui.screens.dialogs.session_title"],
        ),
        (
            "chrys.orchestration.engine.assembly",
            [
                "chrys.orchestration.workflows.agent_node_build",
                "chrys.orchestration.workflows.runner",
                "chrys.service.acp_client.client",
                "chrys.orchestration.invoker.acp_protocol",
                "chrys.service.tools.builtins.web.build",
            ],
        ),
    ],
)
def test_startup_import_does_not_load_unused_features(module, excluded, tmp_path) -> None:
    script = """
import importlib, json, sys
importlib.import_module(sys.argv[1])
unexpected = sorted(set(json.loads(sys.argv[2])) & sys.modules.keys())
assert not unexpected, unexpected
"""
    import json

    result = subprocess.run(
        [sys.executable, "-c", script, module, json.dumps(excluded)],
        cwd=tmp_path,
        env=os.environ | {key: str(tmp_path) for key in ("HOME", "USERPROFILE", "APPDATA")},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        check=False,
        **windows_hidden_subprocess_kwargs(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lazy_dialog_exports_preserve_public_api() -> None:
    from chrys.app.tui.screens import dialogs
    from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog

    assert dialogs.ConfirmDialog is ConfirmDialog
    assert "ConfirmDialog" in dir(dialogs)
    with pytest.raises(AttributeError):
        _ = dialogs.missing_dialog


def test_lazy_acp_exports_preserve_public_api() -> None:
    from chrys.service import acp_client
    from chrys.service.acp_client.client import AcpAgentClient

    assert acp_client.AcpAgentClient is AcpAgentClient
    assert "AcpAgentClient" in dir(acp_client)
    with pytest.raises(AttributeError):
        _ = acp_client.missing_client


def test_production_loading_frame_precedes_main_screen_import(tmp_path) -> None:
    # The child quits at its first frame, so the deferred startup never builds the
    # main screen: that work proves nothing here (test_app_startup.py covers the
    # engine order in process) and stretches a child far on loaded runners.
    script = """
import sys
from pathlib import Path
from chrys.app.tui import app as tui
sys.argv = ['icode']
first_frame = []
class Probe(tui.ChrysApp):
    CSS_PATH = [Path(tui.__file__).parent / path for path in tui.ChrysApp.CSS_PATH]
    def _display(self, screen, renderable):
        super()._display(screen, renderable)
        if not first_frame:
            first_frame.append((screen.id, 'chrys.app.tui.screens.main.screen' in sys.modules))
            self.exit()
    def run(self):
        return super().run(headless=True, size=(100, 30))
tui.main(app_cls=Probe)
assert first_frame == [('_default', False)], first_frame
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ
        | {key: str(tmp_path) for key in ("HOME", "USERPROFILE", "APPDATA")}
        | {"CHRYS_UPDATE_CHECK": "0"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
        check=False,
        **windows_hidden_subprocess_kwargs(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
