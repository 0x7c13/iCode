# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which of a profile's sub-agent references become tools, shared by the chat and workflow builders."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from chrys.service.profiles.agents.schema import AgentProfile

ACP_SUB_AGENT_DEPTH_LIMIT = 3
"""External ACP sub-agents stop nesting at this depth; the spawn layer raises the counter per child."""

type SubAgentRegistrationState = Literal["missing", "empty_command", "depth_limit", "registered"]


def acp_sub_agent_depth() -> int:
    """How many external ACP parents this process already runs under."""
    try:
        return max(0, int(os.environ.get("CHRYS_ACP_SUBAGENT_DEPTH", "0")))
    except ValueError:
        return 0


def sub_agent_registration_state(profile: AgentProfile | None, *, acp_depth: int) -> SubAgentRegistrationState:
    """Whether a referenced sub-agent profile is registered as a tool, and if not, why."""
    if profile is None:
        return "missing"
    if profile.acp is not None and not profile.acp.command:
        return "empty_command"
    if profile.acp is not None and acp_depth >= ACP_SUB_AGENT_DEPTH_LIMIT:
        return "depth_limit"
    return "registered"


def sub_agent_skip_reason(state: SubAgentRegistrationState, *, acp_depth: int) -> str:
    """The logged reason for a non-registered state."""
    if state == "empty_command":
        return "empty ACP command"
    if state == "depth_limit":
        return f"ACP recursion depth {acp_depth} reached"
    return "profile not found"
