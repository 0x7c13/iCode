# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The profile-switch notice: what the new agent is told after a profile switch.

A switch stays pending until a request whose latest switch notice is its own
returns normally; consecutive switches before that keep the original *from*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class SwitchSnapshot:
    """A pending switch as one turn announces it."""

    from_label: str
    to_label: str
    notice: str


class ProfileSwitchSource:
    """The profile switch one middleware announces, and the one a request last carried."""

    def __init__(self, tool_names: Sequence[str] | None) -> None:
        self._tool_names = _dedupe_tool_names(tool_names)
        self._pending_switch: dict[str, str] | None = None
        self._consumed_switch_to: str | None = None

    def set_profile_switch(self, from_label: str, to_label: str) -> None:
        """Set pending profile switch info."""
        self._pending_switch = {"from": from_label, "to": to_label}

    def update_profile_switch_to(self, to_label: str) -> None:
        """Update *to* for consecutive switches (keep original *from*)."""
        if self._pending_switch is not None:
            self._pending_switch["to"] = to_label

    @property
    def has_pending_switch(self) -> bool:
        return self._pending_switch is not None

    def snapshot_pending_switch(self) -> dict[str, str] | None:
        """Return a copy of pending profile-switch metadata, if any."""
        return self._pending_switch.copy() if self._pending_switch is not None else None

    @property
    def consumed_switch_to(self) -> str | None:
        """The display name we switched to, if consumed this turn."""
        return self._consumed_switch_to

    def snapshot(self) -> SwitchSnapshot | None:
        """The pending switch with its notice, None without one."""
        if self._pending_switch is None:
            return None
        ps = self._pending_switch
        return SwitchSnapshot(ps["from"], ps["to"], self._format_profile_switch_hint(ps["from"], ps["to"]))

    def begin_turn(self) -> None:
        """Forget the switch the previous turn consumed."""
        self._consumed_switch_to = None

    def carried(self, switch: SwitchSnapshot) -> None:
        """Consume *switch*: a request whose latest switch notice is its own returned normally.

        The pending switch clears only while it is still *switch*: one made
        during the request stays pending for the next turn.
        """
        self._consumed_switch_to = switch.to_label
        pending = self._pending_switch
        if pending is not None and pending.get("from") == switch.from_label and pending.get("to") == switch.to_label:
            self._pending_switch = None

    def _format_profile_switch_hint(self, from_label: str, to_label: str) -> str:
        """Format the profile-switch reminder shown to the new agent."""
        if self._tool_names:
            tools = ", ".join(self._tool_names)
            tool_hint = f"Your currently available tools are: {tools}."
        else:
            tool_hint = "Your current agent has no available tools."
        return (
            f"[Agent profile switched from '{from_label}' to '{to_label}']\n"
            "System instructions may also have changed; read and follow your current instructions carefully. "
            "Earlier conversation may include tool-call records created by the previous agent. "
            f"{tool_hint} You may only use tools that are currently available to you; "
            "treat earlier tool calls as reference and context only."
        )


def _dedupe_tool_names(tool_names: Sequence[str] | None) -> tuple[str, ...]:
    """Return non-empty tool names in caller order, without duplicates."""
    if not tool_names:
        return ()
    names: list[str] = []
    seen: set[str] = set()
    for name in tool_names:
        clean = name.strip()
        if clean and clean not in seen:
            names.append(clean)
            seen.add(clean)
    return tuple(names)
