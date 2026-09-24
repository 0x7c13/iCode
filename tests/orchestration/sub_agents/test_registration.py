# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which sub-agent references become tools is decided once, for the chat and workflow builders alike."""

from __future__ import annotations

import pytest

from chrys.orchestration.sub_agents.registration import (
    ACP_SUB_AGENT_DEPTH_LIMIT,
    acp_sub_agent_depth,
    sub_agent_registration_state,
    sub_agent_skip_reason,
)
from chrys.service.profiles.agents.schema import AcpAgentConfig, AgentProfile


def _acp_profile(command: str) -> AgentProfile:
    return AgentProfile(name="External", acp=AcpAgentConfig(command=command))


def test_a_missing_profile_is_not_registered() -> None:
    state = sub_agent_registration_state(None, acp_depth=0)

    assert state == "missing"
    assert sub_agent_skip_reason(state, acp_depth=0) == "profile not found"


def test_a_kernel_profile_registers_at_any_depth() -> None:
    assert (
        sub_agent_registration_state(AgentProfile(name="Explore"), acp_depth=ACP_SUB_AGENT_DEPTH_LIMIT) == "registered"
    )


def test_an_acp_profile_without_a_command_is_skipped_before_the_depth_check() -> None:
    state = sub_agent_registration_state(_acp_profile(""), acp_depth=ACP_SUB_AGENT_DEPTH_LIMIT)

    assert state == "empty_command"
    assert sub_agent_skip_reason(state, acp_depth=ACP_SUB_AGENT_DEPTH_LIMIT) == "empty ACP command"


@pytest.mark.parametrize("depth", [ACP_SUB_AGENT_DEPTH_LIMIT - 1, ACP_SUB_AGENT_DEPTH_LIMIT])
def test_an_acp_profile_stops_nesting_at_the_depth_limit(depth: int) -> None:
    state = sub_agent_registration_state(_acp_profile("agent"), acp_depth=depth)

    if depth < ACP_SUB_AGENT_DEPTH_LIMIT:
        assert state == "registered"
    else:
        assert state == "depth_limit"
        assert sub_agent_skip_reason(state, acp_depth=depth) == f"ACP recursion depth {depth} reached"


@pytest.mark.parametrize(("raw", "depth"), [(None, 0), ("2", 2), ("-1", 0), ("many", 0)])
def test_the_acp_depth_comes_from_the_spawn_environment(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, depth: int
) -> None:
    if raw is None:
        monkeypatch.delenv("CHRYS_ACP_SUBAGENT_DEPTH", raising=False)
    else:
        monkeypatch.setenv("CHRYS_ACP_SUBAGENT_DEPTH", raw)

    assert acp_sub_agent_depth() == depth
