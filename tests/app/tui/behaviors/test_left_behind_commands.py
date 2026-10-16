# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""At start-up the TUI offers to stop the background commands a launch that crashed left running."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Static

from chrys.app.tui import app as tui_app
from chrys.app.tui.screens.dialogs.confirm import NoticeDialog
from chrys.service.tools.background.crash import LeftBehind, LeftCommand
from tests.support.tui_app_harness import make_chrys_app
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for

_LEFT = [LeftBehind(Path("running-0000.json"), (LeftCommand("npm run dev", "/repo", 42, ((42, 1.0),)),))]


@pytest.mark.parametrize(("button", "action"), [("#confirm-yes", "stop"), ("#confirm-no", "forget")])
async def test_the_answer_stops_or_forgets_the_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, button: str, action: str
) -> None:
    done: list[tuple[str, list[LeftBehind]]] = []
    monkeypatch.setattr(tui_app, "find_left_behind", lambda _sessions: _LEFT)
    monkeypatch.setattr(tui_app, "stop_left_behind", lambda left: done.append(("stop", left)))
    monkeypatch.setattr(tui_app, "forget_left_behind", lambda left: done.append(("forget", left)))
    app = make_chrys_app(tmp_path / "state")

    async with app.run_test(size=(100, 40)) as pilot:
        await wait_for(
            lambda: isinstance(app.screen, NoticeDialog) and app.screen.is_mounted, pilot=pilot, description="asked"
        )
        message = str(app.screen.query_one("#confirm-message", Static).render())
        assert "closed unexpectedly" in message
        await wait_for(
            lambda: "npm run dev" in str(app.screen.query_one(".task-command", Static).render()), pilot=pilot
        )
        assert "npm run dev" in str(app.screen.query_one(".task-command", Static).render())
        assert "/repo" in str(app.screen.query_one(".task-reason", Static).render())

        await click_when_settled(pilot, button)
        await wait_for(lambda: done, pilot=pilot, description="the answer carried out")
        assert done == [(action, _LEFT)]


async def test_nothing_left_behind_asks_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tui_app, "find_left_behind", lambda _sessions: [])
    app = make_chrys_app(tmp_path / "state")

    async with app.run_test(size=(100, 40)) as pilot:
        await wait_for(
            lambda: app._left_behind_task is not None and app._left_behind_task.done(),
            pilot=pilot,
            description="the check done",
        )
        assert not isinstance(app.screen, NoticeDialog)
