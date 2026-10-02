# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Approval reuse must preserve execution boundaries and unrelated tool behavior."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from threading import get_ident
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from pydantic import create_model

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import ApprovalRequest, ApprovalResponse
from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.models.workspace import Workspace
from chrys.kernel import FunctionTool
from chrys.kernel.middleware import FunctionInvocationContext
from chrys.service.agent_middleware.control import approval
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.approval.policy import ApprovalPolicy
from chrys.service.approval.reuse_binding import ApprovalReuseBinding
from chrys.service.profiles.agents.schema import ApprovalConfig
from chrys.service.tools.builtins.filesystem import FilesystemTools
from chrys.service.tools.builtins.shell import ShellTools
from tests.kernel._fakes import _call_response, _result_contents, _stack, _text_response, _user
from tests.support.symlinks import symlink_or_skip


@pytest.fixture
def runtime(tmp_path):
    env = SessionEnvironment.capture("session-a", Workspace.from_cwd(str(tmp_path)))
    return replace(env, platform=replace(env.platform, config_dir=tmp_path / "config"))


@pytest.mark.parametrize("change", ["cwd", "path", "args"])
def test_session_grant_binds_execution_context(runtime, change):
    tools = ShellTools(runtime).tools()
    binding = ApprovalReuseBinding(runtime, tools)
    args = {"command": "git reset --hard"}
    candidate = binding.candidate(FunctionInvocationContext(tools[0], args))
    assert candidate is not None
    assert binding.service.remember(candidate, "EXACT_SESSION")
    if change == "cwd":
        runtime = replace(runtime, cwd=runtime.cwd + "/other")
    else:
        shell = replace(runtime.platform.shell, **{change: "/other/shell" if change == "path" else ["-l", "-c"]})
        runtime = replace(runtime, platform=replace(runtime.platform, shell=shell))
    tools = ShellTools(runtime).tools()
    rebuilt = ApprovalReuseBinding(runtime, tools)
    assert not rebuilt.service.match(rebuilt.candidate(FunctionInvocationContext(tools[0], args)))


@pytest.mark.parametrize(("annotation", "wire"), [(date, "2026-10-01"), (UUID, "00000000-0000-0000-0000-000000000001")])
@pytest.mark.parametrize("enabled", [False, True])
async def test_custom_typed_tool_keeps_ordinary_approval(runtime, annotation, wire, enabled):
    executed, requests = [], []

    async def consume(value: Any) -> str:
        executed.append(value)
        return "done"

    tool = FunctionTool(name="typed_tool", func=consume, input_model=create_model("Typed", value=(annotation, ...)))
    bus = EventBus()
    binding = ApprovalReuseBinding(runtime, [tool])
    middleware = ApprovalMiddleware(
        ApprovalPolicy(ApprovalConfig(default="require")), bus, reuse=binding if enabled else None
    )

    async def approve(event):
        requests.append(event)
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True))

    await bus.subscribe(ApprovalRequest, approve)
    try:
        layer, _ = _stack(
            [_call_response(("typed", tool.name, {"value": wire})), _text_response()], middleware=middleware
        )
        response = await layer.get_response([_user()], options={"tools": [tool]})
        assert len(requests) == len(executed) == 1
        assert isinstance(executed[0], annotation)
        assert _result_contents(response)[0].result == "done"
        assert not binding.service.rules()
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, approve)


@pytest.mark.parametrize("default", ["require", "auto"])
@pytest.mark.parametrize("explicit_cwd", [False, True])
async def test_read_alias_retarget_requires_approval(runtime, tmp_path, default, explicit_cwd):
    ordinary, secret, alias = tmp_path / "ordinary.txt", tmp_path / ".env", tmp_path / "alias.txt"
    ordinary.write_text("ordinary", encoding="utf-8")
    secret.write_text("test-secret", encoding="utf-8")
    symlink_or_skip(alias, ordinary)
    tools = FilesystemTools(runtime).tools()
    tool = next(tool for tool in tools if tool.name == "read_file")
    bus, requests = EventBus(), []
    binding = ApprovalReuseBinding(runtime, tools)
    middleware = ApprovalMiddleware(
        ApprovalPolicy(ApprovalConfig(default=default)),
        bus,
        reuse=binding,
        workspace_cwd=runtime.cwd if explicit_cwd else None,
    )

    async def approve(event):
        requests.append(event)
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True, remember_choice="EXACT_PROJECT"))

    await bus.subscribe(ApprovalRequest, approve)
    try:
        for index in range(2):
            layer, _ = _stack(
                [_call_response((f"read-{index}", tool.name, {"path": alias.name})), _text_response()],
                middleware=middleware,
            )
            await layer.get_response([_user()], options={"tools": [tool]})
            if index == 0:
                alias.unlink()
                symlink_or_skip(alias, secret)
        assert len(requests) == (2 if default == "require" else 1)
        assert all(event.reuse_offer is None for event in requests)
        assert binding.service.rules() == []
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, approve)


@pytest.mark.parametrize("folder", ["credentials", "cookies"])
@pytest.mark.parametrize("enabled", [False, True])
async def test_resolved_read_checks_are_opt_in_and_off_event_loop(runtime, tmp_path, monkeypatch, folder, enabled):
    workspace = tmp_path / folder
    workspace.mkdir()
    runtime = replace(runtime, cwd=str(workspace))
    tools = FilesystemTools(runtime).tools()
    tool = next(tool for tool in tools if tool.name == "read_file")
    bus, requests = EventBus(), []
    binding = ApprovalReuseBinding(runtime, tools)
    middleware = ApprovalMiddleware(
        ApprovalPolicy(ApprovalConfig(default="auto")),
        bus,
        reuse=binding if enabled else None,
        workspace_cwd=runtime.cwd,
    )
    loop_thread = get_ident()
    resolved = []
    resolve = approval.resolve_workspace_path

    def record_resolution(raw, *, base_cwd=None):
        resolved.append((base_cwd, get_ident()))
        return resolve(raw, base_cwd=base_cwd)

    monkeypatch.setattr(approval, "resolve_workspace_path", record_resolution)

    async def approve(event):
        requests.append(event)
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True))

    called = AsyncMock(spec=lambda: None)
    await bus.subscribe(ApprovalRequest, approve)
    try:
        await middleware.process(FunctionInvocationContext(tool, {"path": "README.md"}), called)
        called.assert_awaited_once()
        assert len(requests) == int(enabled)
        assert len(resolved) == int(enabled)
        assert all(cwd == runtime.cwd and thread != loop_thread for cwd, thread in resolved)
    finally:
        await middleware.close()
        await bus.unsubscribe(ApprovalRequest, approve)
