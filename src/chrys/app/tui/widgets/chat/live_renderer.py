# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Live transcript rendering for ChatPanel."""

from __future__ import annotations

from typing import Any

from rich.text import Text

from chrys.app.tui.util.removal import finish_shielded, remove_shielded
from chrys.app.tui.widgets.chat.compaction_card import CompactionCard
from chrys.app.tui.widgets.chat.context_fold import ContextFoldWidget
from chrys.app.tui.widgets.chat.messages import (
    AgentMessage,
    ErrorMessage,
    InterruptedMessage,
    RetryMessage,
    SystemMessage,
    UserMessage,
    format_message_created_at,
    process_think_tags,
    resolve_interrupted_message_copy,
)
from chrys.app.tui.widgets.chat.ports import (
    ChatMountPort,
    ChatWidgetQueryPort,
    StatusCleanupPort,
    TranscriptLocalizationPort,
    TranscriptUiPort,
)
from chrys.app.tui.widgets.chat.toc_model import TurnTocModel
from chrys.app.tui.widgets.chat.tool_registry import ToolGroupRegistry
from chrys.foundation.events.types import ProvisionalPresentation
from chrys.foundation.i18n.formatting import format_message


class LiveTranscriptRenderer:
    """Render live user, agent, system, status, and context messages."""

    def __init__(
        self,
        mount: ChatMountPort,
        ui: TranscriptUiPort,
        status: StatusCleanupPort,
        query: ChatWidgetQueryPort,
        toc: TurnTocModel,
        tools: ToolGroupRegistry,
        *,
        localization: TranscriptLocalizationPort | None = None,
    ) -> None:
        self._mount = mount
        self._ui = ui
        self._status = status
        self._query = query
        self._toc = toc
        self._tools = tools
        self._resolve_message = localization.render if localization is not None else format_message
        self._current_agent_msg: AgentMessage | None = None
        self._profile_name: str = ""
        self._compaction_cards: dict[str, CompactionCard] = {}
        self._provisional_agent_messages: dict[str, dict[str, AgentMessage]] = {}

    @property
    def current_agent_msg(self) -> AgentMessage | None:
        """Current streamed agent message cursor."""
        return self._current_agent_msg

    @current_agent_msg.setter
    def current_agent_msg(self, message: AgentMessage | None) -> None:
        self._current_agent_msg = message

    @property
    def profile_name(self) -> str:
        """Current agent profile label for live agent message headers."""
        return self._profile_name

    @profile_name.setter
    def profile_name(self, name: str) -> None:
        self._profile_name = name

    def set_profile(self, name: str) -> None:
        """Set the current agent profile label."""
        self._profile_name = name

    def reset_for_clear(self) -> None:
        """Clear per-transcript cursors while preserving durable metadata."""
        self._current_agent_msg = None
        self._compaction_cards.clear()
        self._provisional_agent_messages.clear()

    def _finalize_current_agent_message(self) -> None:
        """Terminate and release the active streamed-message presentation."""
        if self._current_agent_msg is None:
            return
        self._current_agent_msg.stream_update(
            self._current_agent_msg.format_agent_response_copy(),
            is_final=True,
        )
        self._current_agent_msg = None

    async def add_user_message(
        self,
        text: str,
        *,
        is_injection: bool = False,
        created_at: Any = None,
        contents: list[Any] | None = None,
    ) -> None:
        """Display a live user or injection message."""
        await self._ui.dismiss_welcome_for_content()
        if not is_injection:
            self._ui.on_live_user_turn_started()
            await self._remove_failed_turn()
        self._tools.end_group()
        self._finalize_current_agent_message()
        timestamp = format_message_created_at(created_at)

        if is_injection and self._toc.has_items():
            turn_id, _item = self._toc.add_injection(text)
            widget = UserMessage(text, timestamp=timestamp, is_injection=True, contents=contents)
        else:
            turn_id, _item = self._toc.begin_user_turn(text)
            widget = UserMessage(text, timestamp=timestamp, contents=contents)
        widget.id = turn_id
        await self._mount.mount_transcript_widget(widget)
        self._ui.scroll_user_message_to_top(widget)

    async def _remove_failed_turn(self) -> None:
        """Remove a trailing failed turn status and its preceding prompt."""
        status = self._status.failed_turn_status()
        if not isinstance(status, (ErrorMessage, InterruptedMessage, RetryMessage)):
            return
        # Anchor the prompt to this status before awaiting removal. Notices
        # appended after the failure must not change its predecessor; a notice
        # or output before the status remains a boundary against older input.
        prompt = self._status.preceding_content(status)
        await remove_shielded(status)
        if isinstance(prompt, UserMessage):
            # Drop the entry first: a cancelled removal still takes the prompt away.
            self._toc.remove_failed_prompt(prompt.id)
            await remove_shielded(prompt)
        else:
            self._toc.remember_failed_turn_without_user_removal()

    async def add_agent_message(
        self,
        text: str,
        is_final: bool = True,
        *,
        is_intermediate: bool = False,
        structured_output_completed: bool = False,
        presentation: ProvisionalPresentation | None = None,
        created_at: Any = None,
    ) -> None:
        """Display or update a live agent message."""
        timestamp = format_message_created_at(created_at) if is_final and not is_intermediate else ""
        await self._status.remove_trailing_status()

        if is_intermediate:
            if text and process_think_tags(text, intermediate=True):
                if self._tools.current_tool_group is not None:
                    self._tools.current_tool_group.collapsed = True
                self._tools.end_group()
                self._finalize_current_agent_message()
                msg = AgentMessage(
                    text,
                    is_final=True,
                    profile_name=self._profile_name,
                    is_intermediate=True,
                )
                await self._mount.mount_transcript_widget(msg)
                if presentation is not None:
                    self._provisional_agent_messages.setdefault(presentation.attempt_id, {})[
                        presentation.segment_id
                    ] = msg
                self._ui.schedule_anchor_sync()
            return

        self._ui.on_final_response_started()

        if is_final and self._tools.current_tool_group is not None:
            self._tools.current_tool_group.collapsed = True
        self._tools.end_group()

        if not is_final:
            if self._current_agent_msg is None:
                self._current_agent_msg = AgentMessage(
                    text,
                    is_final=False,
                    profile_name=self._profile_name,
                )
                await self._mount.mount_transcript_widget(self._current_agent_msg)
            else:
                self._current_agent_msg.stream_update(text, is_final=False)
            self._ui.schedule_anchor_sync()
        elif self._current_agent_msg is not None:
            self._current_agent_msg.stream_update(text, is_final=True, timestamp=timestamp)
            self._current_agent_msg = None
            self._ui.schedule_anchor_sync()
        else:
            final_text = text if text.strip() else ""
            if not final_text and not structured_output_completed:
                return
            msg = AgentMessage(
                final_text or "✓",
                is_final=True,
                profile_name=self._profile_name,
                is_structured_completion=structured_output_completed and not final_text,
                timestamp=timestamp,
            )
            await self._mount.mount_transcript_widget(msg)
            self._ui.schedule_anchor_sync()

    async def accept_presentation_attempt(self, attempt_id: str, segment_ids: tuple[str, ...]) -> None:
        """Commit canonical provisional widgets and remove unmatched ones."""
        tracked = self._provisional_agent_messages.pop(attempt_id, {})
        accepted = set(segment_ids)
        rejected = [message for segment_id, message in tracked.items() if segment_id not in accepted]
        if rejected:
            # The attempt no longer tracks these widgets, so a cancelled caller still removes every one.
            await finish_shielded(self._remove_rejected(rejected))

    async def _remove_rejected(self, messages: list[AgentMessage]) -> None:
        for message in messages:
            await self._mount.remove_transcript_widget(message)
        self._ui.schedule_anchor_sync()

    async def reject_presentation_attempt(self, attempt_id: str) -> None:
        """Remove all mounted provisional widgets for a rejected attempt."""
        await self.accept_presentation_attempt(attempt_id, ())

    async def add_context_fold(
        self,
        compressed_context_id: str,
        summary: str,
        freed_messages: int = 0,
        turn_range: tuple[int, int] = (0, 0),
    ) -> None:
        """Display a context compression event as fresh run output."""
        await self._status.remove_trailing_status()
        # A failed stream may leave this cursor pointing at its partial
        # AgentMessage.  The fold is a presentation boundary: recovered text
        # must mount below it instead of updating that stale widget in place.
        # Finalize the visible partial first so its streaming cursor cannot
        # outlive the owner reference discarded below.
        self._finalize_current_agent_message()
        self._tools.end_group()
        widget = ContextFoldWidget(
            compressed_context_id,
            summary,
            freed_messages,
            turn_range,
            resolve_message=self._resolve_message,
        )
        await self._mount.mount_transcript_widget(widget)

    async def add_compaction_start(self, compaction_id: str) -> None:
        """Mount a live Phase-4 compaction card, closing the open tool group.

        Compaction runs between model calls, so every tool in the current
        group has already completed — collapse it (matching the intermediate
        agent-text transition) and let the next tool batch open a fresh
        group below the card.
        """
        await self._status.remove_trailing_status()
        self._finalize_current_agent_message()
        if self._tools.current_tool_group is not None:
            self._tools.current_tool_group.collapsed = True
        self._tools.end_group()
        card = CompactionCard(compaction_id)
        self._compaction_cards[compaction_id] = card
        await self._mount.mount_transcript_widget(card)
        self._ui.schedule_anchor_sync()

    def complete_compaction(
        self,
        compaction_id: str,
        *,
        outcome: str,
        duration_ms: int = 0,
        last_words: str = "",
        format_violation: str = "",
        failure_reason: str = "",
    ) -> None:
        """Finalize the matching live compaction card, if still tracked.

        ``format_violation`` stays in the event payload for surfaces that
        narrate it (sub-agent lines, the ACP bridge) but is deliberately not
        shown on the main card: the compaction succeeded, so the summary
        stands without a lingering warning.
        """
        del format_violation
        card = self._compaction_cards.pop(compaction_id, None)
        if card is None:
            return
        card.set_complete(
            outcome=outcome,
            duration_ms=duration_ms,
            last_words=last_words,
            failure_reason=failure_reason,
        )
        self._ui.schedule_anchor_sync()

    def show_compaction_retry(self, message: str, attempt: int, max_attempts: int, delay_seconds: int) -> None:
        """Route a Phase-4 side-call retry notice onto the running compaction card.

        Retry events carry no compaction id, but main-agent compactions are
        strictly sequential — at most one tracked card is still running.
        With no live card (stale event) the notice is dropped.
        """
        for card in reversed(self._compaction_cards.values()):
            if card.status == "running":
                card.show_retry_notice(message, attempt, max_attempts, delay_seconds)
                return

    def _finalize_dangling_compactions(self) -> None:
        """Mark still-running compaction cards interrupted (run ended first)."""
        for card in self._compaction_cards.values():
            card.set_complete(outcome="canceled")
        self._compaction_cards.clear()

    async def add_system(self, text: str, *, key: str | None = None, warning: bool = False) -> None:
        """Display a system message."""
        self._tools.end_group()
        msg = SystemMessage(text, warning=warning)
        if key is not None:
            msg.id = key
        await self._mount.mount_transcript_widget(msg)

    async def update_system(self, key: str, new_text: str) -> None:
        """Update the first system message matching ``key``."""
        for msg in self._query.direct_children():
            if isinstance(msg, SystemMessage) and msg.id == key:
                msg.update(Text(f"  ◦ {new_text}"))
                break

    async def remove_system(self, key: str) -> None:
        """Remove the first system message matching ``key``."""
        for msg in self._query.direct_children():
            if isinstance(msg, SystemMessage) and msg.id == key:
                await remove_shielded(msg)
                break

    async def add_interrupted(
        self, reason: str = "Execution interrupted", source: str = "user", *, show_action: bool = True
    ) -> None:
        """Display an interruption notice, replacing any trailing status widget."""
        await self.cancel_running_tools()
        await self._status.remove_trailing_status()

        copy = resolve_interrupted_message_copy(reason, source, self._resolve_message)
        label = copy.retry_action if source == "error" else copy.continue_action
        msg = InterruptedMessage(copy.reason, source, action_label=label if show_action else None, header=copy.header)
        await self._mount.mount_transcript_widget(msg)
        self._ui.on_status_message_mounted()

    async def add_error(self, text: str, *, action_label: str | None = "Retry") -> None:
        """Display an error message, replacing any trailing status widget."""
        await self.cancel_running_tools()
        await self._status.remove_trailing_status()

        msg = ErrorMessage(text, action_label=action_label)
        await self._mount.mount_transcript_widget(msg)
        self._ui.on_status_message_mounted()

    async def add_retry(self, message: str, attempt: int, max_attempts: int, delay_seconds: int) -> None:
        """Display a retry notice, replacing any trailing status widget."""
        await self._status.remove_trailing_status()
        self._tools.end_group()
        msg = RetryMessage(message, attempt, max_attempts, delay_seconds)
        await self._mount.mount_transcript_widget(msg)
        self._ui.on_status_message_mounted()

    async def prepare_retry(self) -> None:
        """Finalize pending widgets before a main-agent stream retry."""
        self._tools.cancel_running_and_end_group()
        self._finalize_current_agent_message()

    async def cancel_running_tools(self) -> None:
        """Cancel all in-progress tool calls."""
        self._tools.cancel_running_and_end_group()
        self._finalize_dangling_compactions()
