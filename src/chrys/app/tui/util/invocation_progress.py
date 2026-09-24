# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared display of agent progress and reported usage."""

from __future__ import annotations

from collections.abc import Callable

from chrys.app.tui.util.formatting import format_token_count
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

_SUB_AGENT_SPEND = msg("tui.tool_card.sub_agent.spend", fallback="Spend: {spend}")
_SUB_AGENT_TOOL_CALLS = msg("tui.tool_card.sub_agent.tool_calls", fallback="Tool calls: {n}")
_SUB_AGENT_CTX_TOKENS = msg("tui.tool_card.sub_agent.ctx_tokens", fallback="Ctx: {tokens} tokens")
_SUB_AGENT_TOKENS = msg("tui.tool_card.sub_agent.tokens", fallback="{tokens} tokens")
_SUB_AGENT_ZERO_REPORTED_TOKENS = msg(
    "tui.tool_card.sub_agent.zero_reported_tokens",
    fallback="0 reported tokens",
)
_SUB_AGENT_UNREPORTED_ATTEMPTS = msg(
    "tui.tool_card.sub_agent.unreported_attempts",
    fallback="{count} unreported attempt",
    plural_fallback="{count} unreported attempts",
)
_SUB_AGENT_COMPACTIONS = msg("tui.tool_card.sub_agent.compactions", fallback="Compactions: {n}")


def invocation_progress_parts(
    *,
    tool_calls: int = 0,
    context_tokens: int = 0,
    usage_tokens: int = 0,
    unreported_attempts: int = 0,
    compactions: int = 0,
    include_zero: bool = False,
    render: Callable[[MessageRef], str] = format_message,
) -> list[str]:
    """Format counters without conflating current context size with cumulative spend."""
    parts: list[str] = []
    if tool_calls > 0 or include_zero:
        parts.append(render(_SUB_AGENT_TOOL_CALLS.bind(n=tool_calls)))
    if context_tokens > 0:
        parts.append(render(_SUB_AGENT_CTX_TOKENS.bind(tokens=format_token_count(context_tokens))))
    if usage_tokens or unreported_attempts or include_zero:
        spend = (
            render(_SUB_AGENT_TOKENS.bind(tokens=format_token_count(usage_tokens)))
            if usage_tokens or not unreported_attempts
            else render(_SUB_AGENT_ZERO_REPORTED_TOKENS.bind())
        )
        if unreported_attempts:
            attempts = render(_SUB_AGENT_UNREPORTED_ATTEMPTS.bind(count=unreported_attempts))
            spend += f" · {attempts}"
        parts.append(render(_SUB_AGENT_SPEND.bind(spend=spend)))
    if compactions > 0:
        parts.append(render(_SUB_AGENT_COMPACTIONS.bind(n=compactions)))
    return parts
