# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Focused repro tests for profile/model/tool configuration bugs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from chrys.service.approval.policy import ApprovalPolicy
from chrys.service.profiles.agents.loader import load_profile_from_yaml
from chrys.service.profiles.agents.schema import ApprovalConfig


def test_approval_loader_promotes_mcp_override_for_dotted_tool_name(tmp_path: Path) -> None:
    """MCP tools may expose dotted names, and ``mcp.<tool>`` is still a kind-qualified key."""
    path = tmp_path / "agent.yaml"
    path.write_text(
        """\
name: dotted-mcp
approval:
  default: auto
  overrides:
    mcp.server.tool: require
""",
        encoding="utf-8",
    )

    profile = load_profile_from_yaml(path)

    assert profile.approval.overrides == {"mcp.server.tool": "require"}


def test_approval_policy_matches_kind_override_for_dotted_tool_name() -> None:
    """A runtime ``kind.tool_name`` override must match even when the tool name contains dots."""
    tools = [SimpleNamespace(name="server.tool", chrys_kind="mcp")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"mcp.server.tool": "require"}),
        tools=tools,
    )

    assert policy.should_require_approval("server.tool")


def test_approval_policy_does_not_apply_absent_known_kind_override_to_other_kind() -> None:
    """A known kind-qualified override must not fall through to a same-named tool of another kind."""
    tools = [SimpleNamespace(name="server.tool", chrys_kind="filesystem.write")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="require", overrides={"mcp.server.tool": "skip"}),
        tools=tools,
    )

    assert policy.should_require_approval("server.tool")
