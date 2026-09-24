# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Observe human approval waits without participating in approval decisions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from chrys.service.hooks.events import HookEvent

if TYPE_CHECKING:
    from chrys.service.hooks.manager import HookManager

type ApprovalResult = tuple[bool, str, dict[str, Any] | None]


async def await_approval_with_hooks(
    *,
    future: asyncio.Future[ApprovalResult],
    judge_task: asyncio.Task[None] | None,
    manager: HookManager | None,
    session_id: str | None,
    profile_name: str,
    workspace_cwd: str,
    caller_name: str,
    request_id: str,
    tool_name: str,
    tool_kind: str,
    call_id: str,
    args: dict[str, Any],
    target_operation_id: str | None,
) -> ApprovalResult:
    """Wait for the existing decision future, notifying only actual human waits."""
    if manager is None or not (
        manager.has_hooks_for(HookEvent.APPROVAL_REQUESTED) or manager.has_hooks_for(HookEvent.APPROVAL_RESOLVED)
    ):
        return await future

    if judge_task is not None and not future.done():
        # An AUTO verdict may fulfil the future. Only a still-pending decision
        # after the judge finishes needs human attention.
        await asyncio.wait((future, judge_task), return_when=asyncio.FIRST_COMPLETED)
    if future.done():
        return future.result()

    payload = {
        "session_id": session_id,
        "profile": profile_name,
        "cwd": workspace_cwd,
        "caller_name": caller_name,
        "request_id": request_id,
        "tool": {"name": tool_name, "kind": tool_kind, "call_id": call_id, "args": args},
    }
    # These are observer events. Ignore every decision returned by the manager.
    await manager.fire(HookEvent.APPROVAL_REQUESTED, payload, target_operation_id=target_operation_id)
    result = await future
    await manager.fire(
        HookEvent.APPROVAL_RESOLVED,
        {**payload, "approved": result[0]},
        target_operation_id=target_operation_id,
    )
    return result
