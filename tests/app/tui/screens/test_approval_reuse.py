# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real dialog selection reaches the backend and its owner-only grant store."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock, create_autospec

import pytest
from textual.widgets import Select

from chrys.app.tui.screens.dialogs.approval import ApprovalDialog
from chrys.app.tui.screens.main import MainScreen
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import ApprovalRequest, Warning
from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.models.workspace import Workspace
from chrys.kernel.middleware import FunctionInvocationContext
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.approval.judge import ApprovalJudge, JudgeVerdict
from chrys.service.approval.policy import ApprovalMode, ApprovalPolicy
from chrys.service.approval.reuse_binding import ApprovalReuseBinding
from chrys.service.profiles.agents.schema import ApprovalConfig
from chrys.service.tools.builtins.shell import ShellTools
from tests.support.tui_app_harness import make_chrys_app
from tests.support.tui_helpers import click_when_settled
from tests.support.waiting import wait_for


@pytest.mark.parametrize("save_ok", [True, False])
@pytest.mark.parametrize("mode", [ApprovalMode.MANUAL, ApprovalMode.AUTO])
async def test_tui_remember_selection_persists_or_reports_failure(tmp_path, monkeypatch, save_ok, mode):
    runtime = SessionEnvironment.capture("session-a", Workspace.from_cwd(str(tmp_path)))
    runtime = replace(runtime, platform=replace(runtime.platform, config_dir=tmp_path / "config"))
    tools = ShellTools(runtime).tools()
    binding = ApprovalReuseBinding(runtime, tools)
    if not save_ok:
        monkeypatch.setattr(binding.service.store, "add_many", lambda rules: False)
    bus, requests, warnings = EventBus(), [], []

    async def record_request(event):
        requests.append(event)

    async def record_warning(event):
        warnings.append(event)

    await bus.subscribe(ApprovalRequest, record_request)
    await bus.subscribe(Warning, record_warning)
    judge = create_autospec(ApprovalJudge, instance=True)
    judge.evaluate.return_value = JudgeVerdict(approved=False, reason="Human review needed")
    middleware = ApprovalMiddleware(
        ApprovalPolicy(ApprovalConfig(default="require")), bus, reuse=binding, approval_mode=mode, approval_judge=judge
    )
    called = AsyncMock(spec=lambda: None)
    app = make_chrys_app(tmp_path / "ui-sessions", event_bus=bus)
    task = None
    try:
        async with app.run_test(size=(120, 45)) as pilot:
            await wait_for(lambda: isinstance(app.screen, MainScreen), pilot=pilot, description="main screen mounted")
            context = FunctionInvocationContext(tools[0], {"command": "npm run test"})
            task = asyncio.create_task(middleware.process(context, called))
            await wait_for(
                lambda: isinstance(app.screen, ApprovalDialog) or task.done(),
                pilot=pilot,
                description="backend approval dialog",
            )
            if task.done():
                await task
            assert isinstance(app.screen, ApprovalDialog)
            dialog = app.screen
            await wait_for(
                lambda: bool(dialog.query("#reuse-choice #label")),
                pilot=pilot,
                description="remember selection and its label mounted",
            )
            if mode == ApprovalMode.AUTO:
                assert requests[0].judging is True
                assert dialog._flagged == JudgeVerdict(approved=False, reason="Human review needed")
                judge.evaluate.assert_awaited_once()
            else:
                judge.evaluate.assert_not_awaited()
            dialog.query_one("#reuse-choice", Select).value = "EXACT_PROJECT"
            await click_when_settled(pilot, "#approval-yes")
            await wait_for(task.done, pilot=pilot, description="approval saved and call executed")
            await task
            called.assert_awaited_once()
            assert len(requests) == 1
            if save_ok:
                grant = binding.service.rules()[0]
                assert grant.scope == "PROJECT"
                await middleware.process(context, called)
                assert called.await_count == 2 and len(requests) == 1
                assert json.loads(middleware.drain_decisions()[-1]["grant_ids"]) == [grant.id]
                assert context.metadata["approval_grant_ids"] == [grant.id]
                assert not warnings
            else:
                assert binding.service.rules() == []
                assert [event.code for event in warnings] == ["approval_reuse_save_failed"]
                assert "Allowed once" in warnings[0].message
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, record_request)
        await bus.unsubscribe(Warning, record_warning)


async def test_deferred_auto_approval_never_saves_a_reuse_grant(tmp_path):
    runtime = SessionEnvironment.capture("session-a", Workspace.from_cwd(str(tmp_path)))
    runtime = replace(runtime, platform=replace(runtime.platform, config_dir=tmp_path / "config"))
    tools = ShellTools(runtime).tools()
    binding = ApprovalReuseBinding(runtime, tools)
    bus, requests = EventBus(), []

    async def record_request(event):
        requests.append(event)

    await bus.subscribe(ApprovalRequest, record_request)
    judge = create_autospec(ApprovalJudge, instance=True)
    judge.evaluate.return_value = JudgeVerdict(approved=True, reason="Safe command")
    middleware = ApprovalMiddleware(
        ApprovalPolicy(ApprovalConfig(default="require")),
        bus,
        reuse=binding,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
    )
    called = AsyncMock(spec=lambda: None)
    app = make_chrys_app(tmp_path / "ui-sessions", event_bus=bus)
    try:
        async with app.run_test(size=(120, 45)) as pilot:
            await wait_for(lambda: isinstance(app.screen, MainScreen), pilot=pilot, description="main screen mounted")
            screen = app.screen
            for _ in range(2):
                await middleware.process(FunctionInvocationContext(tools[0], {"command": "npm run test"}), called)
                assert app.screen is screen
                assert binding.service.rules() == []
            assert len(requests) == judge.evaluate.await_count == called.await_count == 2
            assert all(request.judging and request.reuse_offer is not None for request in requests)
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, record_request)
