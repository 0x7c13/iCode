# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared ACP client test helpers."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from acp.schema import PermissionOption, SessionNotification, ToolCallUpdate

from chrys.service.acp_client import AcpAgentClient, AcpAgentSpec, PermissionDecision

_REPO_ROOT = Path(__file__).resolve().parents[3]
_STUB_SCRIPT = (_REPO_ROOT / "tests" / "support" / "acp_stub_agent.py").resolve()
_SOURCE_ROOT = (_REPO_ROOT / "src").resolve()


class CallbackRecorder:
    """Record ordered callbacks and provide deterministic broker answers."""

    def __init__(self) -> None:
        self.updates: list[tuple[int, SessionNotification]] = []
        self.permission_calls: list[tuple[ToolCallUpdate, tuple[PermissionOption, ...]]] = []
        self.ext_calls: list[tuple[str, dict[str, Any]]] = []
        self.fail_updates = False
        self.fail_permission = False
        self.fail_ext = False
        self.permission_gate: asyncio.Event | None = None
        self.permission_decision: PermissionDecision | None = None
        self.update_gate: asyncio.Event | None = None

    async def on_update(self, seq: int, notification: SessionNotification) -> None:
        self.updates.append((seq, notification))
        if self.update_gate is not None:
            await self.update_gate.wait()
        if self.fail_updates:
            raise RuntimeError("update callback failed")

    async def on_permission_request(
        self,
        tool_call: ToolCallUpdate,
        options: tuple[PermissionOption, ...],
    ) -> PermissionDecision:
        self.permission_calls.append((tool_call, options))
        if self.permission_gate is not None:
            await self.permission_gate.wait()
        if self.fail_permission:
            raise RuntimeError("permission callback failed")
        return self.permission_decision or PermissionDecision.allow(options[0].option_id)

    async def on_ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.ext_calls.append((method, params))
        if self.fail_ext:
            raise RuntimeError("request-input callback failed")
        return {"answers": [{"values": ["yes"], "note": ""}], "cancelled": False}


def make_spec(
    tmp_path: Path,
    *,
    scenario: str = "happy",
    idle_timeout_seconds: float = 2.0,
    session_mode: str = "",
    model_id: str = "",
    config_options: dict[str, str | bool] | None = None,
    additional_directories: tuple[str, ...] = (),
    best_effort_options: bool = False,
    depth: int = 0,
    extra_env: dict[str, str] | None = None,
) -> AcpAgentSpec:
    env = {
        "PYTHONPATH": str(_SOURCE_ROOT),
        "CHRYS_ACP_STUB_SCENARIO": scenario,
    }
    if extra_env:
        env.update(extra_env)
    return AcpAgentSpec(
        command=sys.executable,
        args=(str(_STUB_SCRIPT),),
        env=env,
        cwd=str(_REPO_ROOT),
        stderr_log_path=tmp_path.resolve() / "stub.stderr.log",
        session_mode=session_mode,
        model_id=model_id,
        config_options=config_options or {},
        additional_directories=additional_directories,
        best_effort_options=best_effort_options,
        # Covers child interpreter start + pydantic import + initialize; loaded
        # parallel CI runners have blown a 5s budget, and no test waits for
        # this default to expire (stall tests cancel or use idle_timeout).
        handshake_timeout_seconds=20.0,
        idle_timeout_seconds=idle_timeout_seconds,
        depth=depth,
    )


async def connected_client(
    tmp_path: Path,
    *,
    scenario: str = "happy",
    callbacks: CallbackRecorder | None = None,
    **spec_kwargs: Any,
) -> tuple[AcpAgentClient, CallbackRecorder]:
    recorder = callbacks or CallbackRecorder()
    client = AcpAgentClient(make_spec(tmp_path, scenario=scenario, **spec_kwargs), recorder)
    await client.connect()
    await client.open_session()
    return client, recorder


def residual_client_tasks(before: set[asyncio.Task[Any]]) -> list[asyncio.Task[Any]]:
    """Return tasks created since *before* that are still running."""
    current = asyncio.current_task()
    return [task for task in asyncio.all_tasks() if task not in before and task is not current and not task.done()]
