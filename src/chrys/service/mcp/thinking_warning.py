# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The warning that on-demand MCP tool loading can unbind an agent's preserved thinking.

A server with ``use_progressive_disclosure`` changes the request's tool set
whenever the model loads or unloads one of its tools, and a new run starts
again from the initial set. Claude Opus 5.5, Fable 5.1 and Sonnet 5.5 bind
their replayed thinking to the request prefix, tools included, so after such a
change the service may refuse a request replaying that thinking, unless the
request lets it drop the mismatched thinking (``drop_block``). Each agent
build logs at most one warning for that case, naming the agent, its model
profile and every such server.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from chrys.service.profiles.agents.schema import AgentProfile
    from chrys.service.profiles.models.schema import ModelProfile

logger = logging.getLogger(__name__)


def tool_loading_thinking_warning(
    agent: AgentProfile,
    model: ModelProfile,
    chat_options: Mapping[str, Any] | None,
    connected: Collection[str],
) -> str | None:
    """The warning for *agent* running *model* with *chat_options*, or None when there is none.

    *connected* names the agent's MCP servers that connected. The warning
    applies to an Anthropic profile whose final model binds its thinking, with
    at least one enabled, connected server loading tools on demand, unless the
    request turns thinking off or sends ``drop_block``. It names the agent and
    the model profile, which holds the settings it suggests.
    """
    if model.provider != "anthropic":
        return None
    names = [
        config.name
        for config in agent.tools.mcp
        if config.enabled and config.use_progressive_disclosure and config.name in connected
    ]
    if not names:
        return None

    # Imported only once an Anthropic profile qualifies: the package loads the Anthropic SDK.
    from chrys.service.llm.anthropic_messages.thinking_binding import resolve_thinking_binding
    from chrys.service.llm.clients import effective_model_base_url

    options = chat_options or {}
    request = dict(options)
    if not request.get("model"):
        request["model"] = model.model_id
    policy = resolve_thinking_binding(request, options, base_url=effective_model_base_url(model))
    if not policy.binding_model or policy.thinking_type == "disabled":
        return None
    if policy.mismatch_behavior_sent and policy.mismatch_behavior == "drop_block":
        return None

    servers_text = ", ".join(repr(name) for name in names)
    lead = (
        f"Agent {agent.name!r} on model profile {model.name!r}: MCP server(s) {servers_text} load tools on demand "
        "(use_progressive_disclosure). Loading or unloading a tool changes the tool set the model binds its "
        "earlier thinking to"
    )
    if policy.mismatch_behavior == "error":
        return f"{lead}, so the service can refuse a later request (the prefix mismatch behavior is error)."
    remedy = (
        'Writing the block_binding in the thinking as {"prefix_mismatch_behavior": "drop_block"}'
        if policy.block_binding_written
        else "Setting thinking_block_binding: drop_block with enabled or adaptive thinking"
    )
    return (
        f"{lead}, so the service can refuse a later request; it is then resent once without the earlier "
        f"thinking, which the model no longer sees. {remedy} lets the service drop the thinking that no longer "
        "matches instead."
    )


def warn_if_tool_loading_unbinds_thinking(
    agent: AgentProfile,
    model: ModelProfile,
    chat_options: Mapping[str, Any] | None,
    connected: Collection[str],
) -> None:
    """Log :func:`tool_loading_thinking_warning`'s warning, if there is one."""
    if (message := tool_loading_thinking_warning(agent, model, chat_options, connected)) is not None:
        logger.warning("%s", message)
