# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workflow Info must not mount new content after application shutdown starts."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from textual.widget import Widget
from textual.widgets import Static

from chrys.app.tui.widgets.workflow.info import WorkflowInfo, WorkflowInfoData
from tests.support.tui_app_harness import make_chrys_app
from tests.support.waiting import wait_for


@pytest.mark.parametrize("stage", ["queued", "removing"])
@pytest.mark.parametrize("update", ["data", "locale"])
async def test_shutdown_stops_info_recomposition_before_mounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, update: str
) -> None:
    app = make_chrys_app(tmp_path)
    removing, release, recomposed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    before: tuple[Widget, ...] = ()
    after: tuple[Widget, ...] = ()
    original_close = app._close_all

    class HeldChild(Static):
        async def on_unmount(self) -> None:
            removing.set()
            await release.wait()

    async with app.run_test() as pilot:
        main = app._main_screen
        assert main is not None
        info = main._workflow_panel.query_one(WorkflowInfo)
        original_data = WorkflowInfoData(title="Original", source_kind="file", canonical_path="workflow.py")
        info.data = original_data
        await wait_for(lambda: bool(info.query("#workflow-info-identity")), pilot=pilot)
        if stage == "removing":
            await info.mount(HeldChild())
        await pilot.pause()
        before = tuple(info.children)

        def request_recompose() -> None:
            if update == "data":
                info.data = replace(original_data, title="Updated")
            else:
                info.refresh_localization()
            # This callback runs on Info's pump after its queued recompose.
            info.call_next(recomposed.set)

        async def close_all() -> None:
            nonlocal after
            try:
                # Shutdown leaves underlying screens attached while removing overlays.
                assert not app.is_running and info.is_attached
                if stage == "queued":
                    request_recompose()
                release.set()
                await wait_for(recomposed.is_set)
                after = tuple(info.children)
            finally:
                await original_close()

        monkeypatch.setattr(app, "_close_all", close_all)
        if stage == "removing":
            request_recompose()
            # Keep removal in flight until run_test starts the real shutdown.
            await wait_for(removing.is_set)
            assert not recomposed.is_set()

    assert recomposed.is_set()
    # A queued refresh leaves existing content alone; an in-flight removal drains
    # without replacing its children with widgets that cannot start message loops.
    assert after == (before if stage == "queued" else ())
