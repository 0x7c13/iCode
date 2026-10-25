# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A persistent process header whose first-stage content remains adjacent siblings."""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual.events import Click
from textual.message import Message
from textual.widgets import Static

from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.widgets.chat.messages import ThinkingMessage
from chrys.app.tui.widgets.chat.process_visibility import ProcessMember
from chrys.app.tui.widgets.chat.tool_call import ToolGroup
from chrys.foundation.i18n import msg

_CALLS = msg(
    "tui.chat.process.tool_calls", fallback="Made {count} tool call", plural_fallback="Made {count} tool calls"
)
_ACTIVITY = msg("tui.chat.process.activity", fallback="Activity")


class ProcessBlock(Static, can_focus=False):
    """Collapse intent survives blocking members; their terminal notifications retry it."""

    @dataclass
    class ToggleRequested(Message):
        block: ProcessBlock

    DEFAULT_CSS = """
    ProcessBlock {
        height: 1;
        padding: 0 1 0 0;
        color: $text-muted;
        overflow: hidden;
    }
    """

    def __init__(
        self, members: list[ProcessMember], *, segment_index: int, tool_count: int, is_compressed: bool = False
    ) -> None:
        super().__init__()
        self.members = members
        self.segment_index = segment_index
        self.tool_count = tool_count
        self.is_compressed = is_compressed
        self.collapsed = False
        self.requested_collapsed = True

    def focus_on_click(self) -> bool:
        """Keep transcript clicks in the composer, as tool and thinking headers do."""
        return False

    @property
    def blocked(self) -> bool:
        return any(
            (isinstance(member, ToolGroup) and (not member.all_complete or member.collapse_locked))
            or (isinstance(member, ThinkingMessage) and not member.foldable)
            for member in self.members
        )

    def render(self) -> Text:
        message = _CALLS.bind(count=self.tool_count) if self.tool_count and not self.is_compressed else _ACTIVITY.bind()
        text = Text(f"{'▶' if self.collapsed else '▼'} {render_str(widget_localizer(self), message)}")
        if self.content_size.width:
            text.truncate(self.content_size.width, overflow="ellipsis")
        return text

    def try_finalize_fold(self) -> None:
        """Apply the latest intent, including after the last interactive member finishes."""
        collapsed = self.requested_collapsed and not self.blocked
        if self.is_mounted and collapsed:
            focused = self.app.focused
            if focused is not None and any(focused in member.walk_children(with_self=True) for member in self.members):
                self.screen.set_focus(None, scroll_visible=False)
        for member in self.members:
            if collapsed and isinstance(member, ThinkingMessage):
                member.collapsed = True
            member.set_process_hidden(collapsed)
        if self.collapsed != collapsed:
            self.collapsed = collapsed
            self.refresh(layout=False)

    def action_toggle_process(self) -> None:
        self.post_message(self.ToggleRequested(self))

    def on_click(self, event: Click) -> None:
        event.prevent_default()
        event.stop()
        self.action_toggle_process()
