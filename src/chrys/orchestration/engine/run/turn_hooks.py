# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Lifecycle hook dispatch helpers for main-agent turns."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from chrys.foundation.events.types import Error
from chrys.foundation.i18n import msg
from chrys.foundation.platform import safe_getcwd
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.schema import HookDecision

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.trajectory.context import TrajectoryContext
    from chrys.orchestration.engine.state.active_session import ActiveSession
    from chrys.orchestration.engine.state.current_agent import CurrentAgent
    from chrys.orchestration.engine.state.machine import EngineStateMachine
    from chrys.service.context.compaction import PreCompactInfo
    from chrys.service.hooks.manager import HookManager


logger = logging.getLogger(__name__)

_TURN_HOOKS_PROMPT_BLOCKED = msg(
    "turn_hooks.prompt_blocked",
    fallback="Prompt blocked by hook.",
)


class TurnHookDispatcher:
    """Dispatch non-prompt turn lifecycle hooks."""

    def __init__(
        self,
        *,
        session: ActiveSession,
        current: CurrentAgent,
    ) -> None:
        self._session = session
        self._current = current
        self._pending: set[asyncio.Task[None]] = set()

    async def fire_before_turn(
        self,
        user_text: str,
        *,
        is_retry: bool = False,
        target_operation_id: str | None = None,
    ) -> None:
        """Publish a ``before_turn`` hook event."""
        if self._session.hook_manager is None or not self._session.hook_manager.has_hooks_for(HookEvent.BEFORE_TURN):
            return
        profile_name = self._session.agent_profile.name if self._session.agent_profile is not None else ""
        await self._session.hook_manager.fire(
            HookEvent.BEFORE_TURN,
            {
                "session_id": self._session.session_id,
                "profile": profile_name,
                "cwd": _workspace_cwd(self._session),
                "turn": self._session.turn_number,
                "user_text": user_text,
                "is_retry": is_retry,
            },
            target_operation_id=target_operation_id,
        )

    async def fire_after_turn(self, *, failed: bool) -> None:
        """Publish an ``after_turn`` hook event."""
        if self._session.hook_manager is None or not self._session.hook_manager.has_hooks_for(HookEvent.AFTER_TURN):
            return
        profile_name = self._session.agent_profile.name if self._session.agent_profile is not None else ""
        status = (
            "failed"
            if (self._current.loaded is not None and self._current.loaded.bindings.state.run_failed)
            else (
                "interrupted"
                if (self._current.loaded is not None and self._current.loaded.bindings.state.was_interrupted)
                else "ok"
            )
        )
        await self._session.hook_manager.fire(
            HookEvent.AFTER_TURN,
            {
                "session_id": self._session.session_id,
                "profile": profile_name,
                "cwd": _workspace_cwd(self._session),
                "turn": self._session.turn_number,
                "status": status,
                "failed": failed,
            },
        )

    def schedule_user_interrupt(self) -> None:
        """Schedule observer dispatch with the interrupt's session identity."""
        manager = self._session.hook_manager
        if manager is None or not manager.has_hooks_for(HookEvent.USER_INTERRUPT):
            return
        session_id = self._session.session_id
        profile_name = self._session.agent_profile.name if self._session.agent_profile is not None else ""
        cwd = _workspace_cwd(self._session)
        task = asyncio.create_task(
            self.fire_user_interrupt(manager, session_id=session_id, profile_name=profile_name, cwd=cwd)
        )
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)
        task.add_done_callback(self.log_user_interrupt_hook_error)

    async def fire_user_interrupt(
        self,
        manager: HookManager,
        *,
        session_id: str | None,
        profile_name: str,
        cwd: str,
    ) -> None:
        """Publish an interrupt hook using only its captured manager and identity."""
        await manager.fire(
            HookEvent.USER_INTERRUPT,
            {"session_id": session_id, "profile": profile_name, "cwd": cwd},
            scope="detached",
        )

    @staticmethod
    def log_user_interrupt_hook_error(task: asyncio.Task[None]) -> None:
        """Log a failed background interrupt-hook dispatch."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("user_interrupt hook dispatch failed")

    async def fire_pre_compact(self, info: PreCompactInfo) -> None:
        """Callback from UnifiedContextStrategy — fire pre_compact hooks."""
        if self._session.hook_manager is None:
            return
        from chrys.service.hooks.events import HookEvent

        if not self._session.hook_manager.has_hooks_for(HookEvent.PRE_COMPACT):
            return
        profile_name = self._session.agent_profile.name if self._session.agent_profile is not None else ""
        await self._session.hook_manager.fire(
            HookEvent.PRE_COMPACT,
            {
                "session_id": self._session.session_id,
                "profile": profile_name,
                "cwd": self._session.workspace_cwd(),
                "trigger": info.trigger,
                "usage_pct": info.usage_pct,
                "tokens_before": info.tokens_before,
            },
            target_operation_id=info.trajectory_operation_id,
        )


class PromptSubmitGate:
    """Evaluate ``user_prompt_submit`` hooks and apply accepted reminders."""

    def __init__(
        self,
        *,
        session: ActiveSession,
        current: CurrentAgent,
        bus: EventBus,
        fsm: EngineStateMachine,
    ) -> None:
        self._session = session
        self._current = current
        self._bus = bus
        self._fsm = fsm

    async def fire(self, text: str, *, injected: bool | None = None) -> bool:
        """Return True when a blocking hook denied the prompt."""
        decision = await self.evaluate(text, injected=injected)
        if await self.handle_decision(decision, injected=injected):
            return True
        self.queue_reminders(decision, injected=injected)
        return False

    async def evaluate(
        self,
        text: str,
        *,
        injected: bool | None,
        target_operation_id: str | None = None,
        trajectory_context: TrajectoryContext | None = None,
    ) -> HookDecision | None:
        """Run ``user_prompt_submit`` hooks without applying reminder side effects."""
        if self._session.hook_manager is None or not self._session.hook_manager.has_hooks_for(
            HookEvent.USER_PROMPT_SUBMIT
        ):
            return None
        profile_name = self._session.agent_profile.name if self._session.agent_profile is not None else ""
        is_injected = self._fsm.is_running() if injected is None else injected
        return await self._session.hook_manager.fire(
            HookEvent.USER_PROMPT_SUBMIT,
            {
                "session_id": self._session.session_id,
                "profile": profile_name,
                "cwd": _workspace_cwd(self._session),
                "text": text,
                "injected": is_injected,
            },
            target_operation_id=target_operation_id,
            trajectory_context=trajectory_context,
        )

    async def handle_decision(
        self,
        decision: HookDecision | None,
        *,
        injected: bool | None,
        session_id: str | None = None,
    ) -> bool:
        """Return True after publishing when a prompt-submit hook blocked."""
        _ = injected
        if decision is None:
            return False
        if not decision.blocked:
            return False
        await self._bus.publish(
            Error(
                code="hook_blocked",
                message=decision.block_reason or "Prompt blocked by hook.",
                display_message=None if decision.block_reason else _TURN_HOOKS_PROMPT_BLOCKED.bind(),
                session_id=self._session.session_id if session_id is None else session_id,
            )
        )
        return True

    def queue_reminders(self, decision: HookDecision | None, *, injected: bool | None) -> None:
        """Apply non-blocking prompt-submit hook reminders after prompt validation passes."""
        if decision is None or not decision.system_reminders or self._current.loaded is None:
            return
        is_injected = self._fsm.is_running() if injected is None else injected
        self._current.loaded.reminder_middleware.queue_hook_reminders(
            decision.system_reminders,
            for_next_turn=not is_injected,
        )

    @staticmethod
    def reminder_texts(decision: HookDecision | None) -> list[str]:
        """Return non-empty prompt-hook reminders from an accepted decision."""
        if decision is None:
            return []
        return [reminder for reminder in decision.system_reminders if reminder]


def _workspace_cwd(session: ActiveSession) -> str:
    """Return the current workspace cwd, falling back only before workspace initialization."""
    if session.workspace is not None:
        return session.workspace.primary_cwd
    return safe_getcwd()
