# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared judge-versus-user approval race arbitration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chrys.foundation.events.types import ApprovalAutoFulfillBlocked, ApprovalReviewed

if TYPE_CHECKING:
    from pathlib import Path

    from chrys.foundation.events.bus import EventBus
    from chrys.service.approval.judge import ApprovalJudge


@dataclass(frozen=True, slots=True, kw_only=True)
class ApprovalJudgeInput:
    """Judge-only fields kept separate from spoof-proof presentation fields."""

    tool_name: str
    tool_kind: str
    args: dict[str, Any]
    user_message: str
    user_messages: list[str]
    workspace_roots: list[str]


class ApprovalDecisionArbiter:
    """Single-winner arbitration shared by middleware and ACP broker."""

    def __init__(self, bus: EventBus, *, session_id: str | None = None) -> None:
        self._bus = bus
        self._session_id = session_id
        self._blocked: set[str] = set()
        self._subscribed = False

    async def ensure_subscription(self) -> None:
        if self._subscribed:
            return
        await self._bus.subscribe(ApprovalAutoFulfillBlocked, self._on_blocked)
        self._subscribed = True

    async def judge(
        self,
        *,
        request_id: str,
        judge: ApprovalJudge,
        judge_input: ApprovalJudgeInput,
        decision_future: asyncio.Future[Any],
        approved_value: Any,
        log_dir: Path | None,
    ) -> None:
        """Publish a verdict and fulfil only while no user decision has won."""
        from chrys.service.approval.judge import JudgeVerdict

        try:
            verdict = await judge.evaluate(
                user_message=judge_input.user_message,
                user_messages=list(judge_input.user_messages),
                tool_name=judge_input.tool_name,
                tool_kind=judge_input.tool_kind,
                args=judge_input.args,
                workspace_roots=list(judge_input.workspace_roots),
                request_id=request_id,
                log_dir=log_dir,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            verdict = JudgeVerdict(approved=False, reason=f"Judge evaluation failed: {exc}")

        await self._bus.publish(
            ApprovalReviewed(
                request_id=request_id,
                approved=verdict.approved,
                reason=verdict.reason,
                session_id=self._session_id,
            )
        )
        if verdict.approved:
            blocked = request_id in self._blocked
            self._blocked.discard(request_id)
            if not blocked and not decision_future.done():
                decision_future.set_result(approved_value)

    async def _on_blocked(self, event: ApprovalAutoFulfillBlocked) -> None:
        if event.request_id:
            self._blocked.add(event.request_id)

    def clear(self) -> None:
        self._blocked.clear()

    async def close(self) -> None:
        if not self._subscribed:
            return
        await self._bus.unsubscribe(ApprovalAutoFulfillBlocked, self._on_blocked)
        self._subscribed = False
        self._blocked.clear()
