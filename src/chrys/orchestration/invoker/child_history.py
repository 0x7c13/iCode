# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""History recovery and storage decisions shared by child kernel callers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from chrys.foundation.models.history_markers import copy_reminder_record
from chrys.foundation.models.turns import is_continuation_message
from chrys.kernel import Message, resolve_storage_mode_and_handles
from chrys.service.session.history import SessionHistoryManager, stamp_history_item_ids

if TYPE_CHECKING:
    from chrys.kernel import AgentSession, LoopRecorder
    from chrys.kernel.client import BaseChatClient


def service_storage_side(client: BaseChatClient, options: Mapping[str, Any] | None) -> bool:
    """Resolve the client's storage policy without retaining provider handles."""
    return resolve_storage_mode_and_handles(
        options,
        stores_by_default=client.STORES_BY_DEFAULT,
        force_stateless=client.FORCES_STATELESS,
    ).service_side


def active_input_message(active_input: Sequence[Any]) -> Message | None:
    """Return the single input message that can anchor recovered loop work."""
    if len(active_input) != 1:
        return None
    item = active_input[0]
    if isinstance(item, Message):
        return item
    if isinstance(item, str):
        return Message("user", [item])
    return None


class ChildHistory:
    """Read the session's current state each time, including after retry rollback."""

    def __init__(self, session: AgentSession, recorder: LoopRecorder) -> None:
        self.session = session
        self.recorder = recorder

    def state(self) -> dict[str, Any]:
        state = self.session.state.setdefault("chrys_history", {})
        if not isinstance(state, dict):
            state = {}
            self.session.state["chrys_history"] = state
        return state

    def messages(self) -> list[Any]:
        state = self.state()
        messages = state.setdefault("messages", [])
        if not isinstance(messages, list):
            messages = []
            state["messages"] = messages
        return messages

    def repair_after_failure(self, active_input: Sequence[Any], pass_start_index: int) -> None:
        """Keep completed local work and discard the failed service continuation."""
        messages = self.messages()
        if self.recorder.loop_messages:
            input_message = active_input_message(active_input)
            if input_message is not None and not messages:
                messages.append(input_message)
        manager = SessionHistoryManager()
        state = self.state()
        stamp_history_item_ids(state)
        manager.bind(state)
        manager.merge_loop_messages(self.recorder, insert_index=pass_start_index)
        manager.trim_to_last_complete_tool_results()
        self.session.service_session_id = None

    def retry_input(self, seed: Callable[[], list[Any]]) -> list[Any]:
        """Continue completed work, otherwise replay the seed without duplicating its anchor.

        Synthetic continuation nudges cannot become the user anchor for this decision.
        """
        messages = self.messages()
        user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user" and not is_continuation_message(messages[i]):
                user_idx = i
                break
        if user_idx < 0:
            return seed()
        if any(messages[j].role in ("assistant", "tool") for j in range(user_idx + 1, len(messages))):
            return []
        anchor = messages.pop(user_idx)
        seeded = seed()
        # The seed rebuilds the anchor it replaces: it carries the reminders
        # the anchor was sent with, so the replay re-renders them unchanged.
        if len(seeded) == 1 and isinstance(seeded[0], Message):
            copy_reminder_record(anchor.additional_properties, seeded[0].additional_properties)
        return seeded
