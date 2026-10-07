# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Build the reminder stack by role, and read its state, for the reminder lifecycle tests.

The tests drive LAST_WORDS, profile-switch and archive-pointer calls through
the role that owns them (``stack.last_words``, ``stack.switch``,
``stack.pointer``), restore persisted state through ``restore_phase4`` and
read middleware-private state only through the observers below.  When an
owner moves, only this module changes, never the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Unpack

from chrys.orchestration.invoker.runtime import ReminderInputs, create_reminder, restore_phase4_state

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.service.agent_middleware.reminders.archive_pointer import ArchivePointerSource
    from chrys.service.agent_middleware.reminders.profile_switch import ProfileSwitchSource
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
    from chrys.service.context.compaction.last_words_state import LastWordsState
    from chrys.service.context.compaction.spill import SpillQuota


@dataclass(frozen=True)
class ReminderStack:
    """One reminder middleware and the objects that own its Phase 4, switch and pointer state."""

    middleware: SystemReminderMiddleware
    last_words: LastWordsState
    switch: ProfileSwitchSource
    pointer: ArchivePointerSource


def make_reminder_stack(
    runtime: SessionEnvironment | None = None,
    *,
    max_context_tokens: int,
    shell_tool_enabled: bool = False,
    session_root: Path | None = None,
    file_read_available: bool = False,
    spill_quota: SpillQuota | None = None,
    skill_catalog_provider: Callable[[], str | None] | None = None,
    todo_state_provider: Callable[[], str | None] | None = None,
    mcp_instructions_provider: Callable[[], str | None] | None = None,
    file_change_provider: Callable[[], str | None] | None = None,
) -> ReminderStack:
    """Build a reminder stack wired the way the runtime wires it."""
    middleware, last_words = reminder_pair(
        runtime=runtime,
        max_context_tokens=max_context_tokens,
        warn_threshold_pct=0.50,
        sub_agent_names=None,
        shell_tool_enabled=shell_tool_enabled,
        tool_names=None,
        session_root=session_root,
        file_read_available=file_read_available,
        spill_quota=spill_quota,
        catalog_pointer_enabled=True,
        skill_catalog_provider=skill_catalog_provider,
        todo_state_provider=todo_state_provider,
        mcp_instructions_provider=mcp_instructions_provider,
        file_change_provider=file_change_provider,
    )
    return ReminderStack(
        middleware=middleware,
        last_words=last_words,
        switch=middleware.sources.profile_switch,
        pointer=middleware.sources.archive_pointer,
    )


def reminder_pair(**inputs: Unpack[ReminderInputs]) -> tuple[SystemReminderMiddleware, LastWordsState]:
    """The reminder middleware and the LAST_WORDS state it renders, built by the runtime's factory."""
    return create_reminder(inputs)


def restore_phase4(
    stack: ReminderStack,
    saved: Mapping[str, Any] | None,
    *,
    available_relative_paths: set[str] | None = None,
) -> None:
    """Re-arm persisted LAST_WORDS state and the pointer count on a new stack, as session restore does."""
    restore_phase4_state(stack.middleware, stack.last_words, saved, available_relative_paths=available_relative_paths)


def warning_armed(stack: ReminderStack) -> bool:
    """Whether the next turn at or above the threshold sends the context-usage warning."""
    return stack.middleware.sources.context_warning.armed


def held_catalogs(stack: ReminderStack) -> tuple[str, dict[str, str]] | None:
    """The catalogs a service-side conversation holds, keyed by its handle."""
    return stack.middleware._held_catalogs


def reminder_generation(stack: ReminderStack) -> int:
    """The current-run generation ``prepare_turn`` advances."""
    return stack.middleware._current_run_reminder_generation


def observe(stack: ReminderStack) -> dict[str, Any]:
    """The stack's observable state after a step, for comparing it across steps."""
    held = held_catalogs(stack)
    return {
        "pending_switch": stack.switch.snapshot_pending_switch(),
        "consumed_switch_to": stack.switch.consumed_switch_to,
        "warning_armed": warning_armed(stack),
        "held": [held[0], list(held[1])] if held is not None else None,
        "last_words": stack.last_words.get_last_words(),
        "manifest": [
            [entry["relative_path"], entry["available"]] for entry in stack.last_words.get_last_words_manifest()
        ],
        "breaker": stack.last_words.get_last_words_breaker_state(),
        "pointer_count": stack.pointer.record_count_state(),
    }
