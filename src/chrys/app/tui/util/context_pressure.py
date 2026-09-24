# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared context-pressure explanations for chat and nested agent transcripts."""

from __future__ import annotations

from chrys.foundation.i18n import MessageDef, MessageRef, msg

_CONTEXT_PRESSURE_CONVERSATION = msg(
    "tui.context_pressure.conversation.generic",
    fallback="Conversation context compaction stopped because {reason}. The active task may exceed its model window.",
)
_CONTEXT_PRESSURE_SUB_AGENT = msg(
    "tui.context_pressure.sub_agent.generic",
    fallback="Sub-agent context compaction stopped because {reason}. The active task may exceed its model window.",
)
_CONTEXT_PRESSURE_CONVERSATION_INTERNAL_LIMIT = msg(
    "tui.context_pressure.conversation.internal_limit",
    fallback=(
        "Conversation context compaction stopped because an internal safety limit was reached. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_SUB_AGENT_INTERNAL_LIMIT = msg(
    "tui.context_pressure.sub_agent.internal_limit",
    fallback=(
        "Sub-agent context compaction stopped because an internal safety limit was reached. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_CONVERSATION_DISABLED = msg(
    "tui.context_pressure.conversation.disabled",
    fallback=(
        "Conversation context compaction stopped because the compaction breaker was already disabled. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_SUB_AGENT_DISABLED = msg(
    "tui.context_pressure.sub_agent.disabled",
    fallback=(
        "Sub-agent context compaction stopped because the compaction breaker was already disabled. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_CONVERSATION_GENERATION_FAILURE = msg(
    "tui.context_pressure.conversation.generation_failure",
    fallback=(
        "Conversation context compaction stopped because progress-note generation failed repeatedly. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_SUB_AGENT_GENERATION_FAILURE = msg(
    "tui.context_pressure.sub_agent.generation_failure",
    fallback=(
        "Sub-agent context compaction stopped because progress-note generation failed repeatedly. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_CONVERSATION_NO_PROGRESS = msg(
    "tui.context_pressure.conversation.no_progress",
    fallback=(
        "Conversation context compaction stopped because repeated compaction attempts made insufficient progress. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_SUB_AGENT_NO_PROGRESS = msg(
    "tui.context_pressure.sub_agent.no_progress",
    fallback=(
        "Sub-agent context compaction stopped because repeated compaction attempts made insufficient progress. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_CONVERSATION_ROUND_LIMIT = msg(
    "tui.context_pressure.conversation.round_limit",
    fallback=(
        "Conversation context compaction stopped because the compaction attempt limit was reached. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_SUB_AGENT_ROUND_LIMIT = msg(
    "tui.context_pressure.sub_agent.round_limit",
    fallback=(
        "Sub-agent context compaction stopped because the compaction attempt limit was reached. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_CONVERSATION_SIDE_CALL_BUDGET = msg(
    "tui.context_pressure.conversation.side_call_budget",
    fallback=(
        "Conversation context compaction stopped because the progress-note token budget was exhausted. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_SUB_AGENT_SIDE_CALL_BUDGET = msg(
    "tui.context_pressure.sub_agent.side_call_budget",
    fallback=(
        "Sub-agent context compaction stopped because the progress-note token budget was exhausted. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_CONVERSATION_SPILL_ABORT = msg(
    "tui.context_pressure.conversation.spill_abort",
    fallback=(
        "Conversation context compaction stopped because dropped-call storage could not be durably checkpointed. "
        "The active task may exceed its model window."
    ),
)
_CONTEXT_PRESSURE_SUB_AGENT_SPILL_ABORT = msg(
    "tui.context_pressure.sub_agent.spill_abort",
    fallback=(
        "Sub-agent context compaction stopped because dropped-call storage could not be durably checkpointed. "
        "The active task may exceed its model window."
    ),
)

_CONTEXT_PRESSURE_MESSAGES: dict[tuple[str, str], MessageDef] = {
    ("main", "disabled"): _CONTEXT_PRESSURE_CONVERSATION_DISABLED,
    ("sub_agent", "disabled"): _CONTEXT_PRESSURE_SUB_AGENT_DISABLED,
    ("main", "generation_failure"): _CONTEXT_PRESSURE_CONVERSATION_GENERATION_FAILURE,
    ("sub_agent", "generation_failure"): _CONTEXT_PRESSURE_SUB_AGENT_GENERATION_FAILURE,
    ("main", "no_progress"): _CONTEXT_PRESSURE_CONVERSATION_NO_PROGRESS,
    ("sub_agent", "no_progress"): _CONTEXT_PRESSURE_SUB_AGENT_NO_PROGRESS,
    ("main", "round_limit"): _CONTEXT_PRESSURE_CONVERSATION_ROUND_LIMIT,
    ("sub_agent", "round_limit"): _CONTEXT_PRESSURE_SUB_AGENT_ROUND_LIMIT,
    ("main", "side_call_budget"): _CONTEXT_PRESSURE_CONVERSATION_SIDE_CALL_BUDGET,
    ("sub_agent", "side_call_budget"): _CONTEXT_PRESSURE_SUB_AGENT_SIDE_CALL_BUDGET,
    ("main", "spill_abort"): _CONTEXT_PRESSURE_CONVERSATION_SPILL_ABORT,
    ("sub_agent", "spill_abort"): _CONTEXT_PRESSURE_SUB_AGENT_SPILL_ABORT,
}


def context_pressure_message(reason: str, *, source: str = "main") -> MessageRef:
    """Keep the reason-specific warning identical across transcript surfaces."""
    source = "sub_agent" if source == "sub_agent" else "main"
    definition = _CONTEXT_PRESSURE_MESSAGES.get((source, reason))
    if definition is not None:
        return definition.bind()
    if reason:
        definition = _CONTEXT_PRESSURE_SUB_AGENT if source == "sub_agent" else _CONTEXT_PRESSURE_CONVERSATION
        return definition.bind(reason=reason.replace("_", " "))
    definition = (
        _CONTEXT_PRESSURE_SUB_AGENT_INTERNAL_LIMIT
        if source == "sub_agent"
        else _CONTEXT_PRESSURE_CONVERSATION_INTERNAL_LIMIT
    )
    return definition.bind()
