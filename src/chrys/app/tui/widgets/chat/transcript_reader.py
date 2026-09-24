# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Read/copy helpers for mounted chat transcript widgets."""

from __future__ import annotations

from chrys.app.tui.widgets.chat.messages import AgentMessage, UserMessage
from chrys.app.tui.widgets.chat.ports import ChatWidgetQueryPort


class TranscriptReader:
    """Build copy payloads from mounted chat message widgets."""

    def __init__(self, query: ChatWidgetQueryPort) -> None:
        self._query = query

    def get_agent_responses(self) -> list[tuple[str, str]]:
        """Return (profile_name, raw_text) of all agent messages in mount order."""
        return [
            (message._profile_name or "Agent", message._text)
            for message in self._query.mounted_agent_messages()
            if message._text
        ]

    def get_user_messages(self) -> list[tuple[str, str]]:
        """Return (role_label, raw_text) of all user messages in mount order."""
        return [
            ("You", child._text)
            for child in self._query.direct_children()
            if isinstance(child, UserMessage) and child._text
        ]

    def get_all_messages(self) -> list[tuple[str, str]]:
        """Return (role_label, raw_text) of all user+agent messages in mount order."""
        result: list[tuple[str, str]] = []
        for child in self._query.direct_children():
            if isinstance(child, UserMessage) and child._text:
                result.append(("You", child._text))
            elif isinstance(child, AgentMessage) and child._text:
                result.append((child._profile_name or "Agent", child._text))
        return result
