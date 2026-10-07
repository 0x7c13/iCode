# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""History marker constants for Chrys-owned session messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from chrys.foundation.i18n import msg

if TYPE_CHECKING:
    from collections.abc import Mapping

# Persisted status-marker writer audit (closed set): fixed semantic statuses are
# interruption/failure fallbacks, checkpoint closure, and reload discard; dynamic
# diagnostic detail is ``last_error``; awaiting sub-agents is dynamic semantic
# status carried by ``_invocation_ids``.
EXECUTION_INTERRUPTED_MESSAGE = msg(
    "history_markers.execution_interrupted",
    fallback="Execution interrupted",
)
EXECUTION_FAILED_MESSAGE = msg(
    "history_markers.execution_failed",
    fallback="Execution failed",
)
SESSION_CLOSED_MESSAGE = msg(
    "history_markers.session_closed",
    fallback="Session closed before the turn finished",
)
SUB_AGENT_STATE_DISCARDED_MESSAGE = msg(
    "history_markers.sub_agent_state_discarded",
    fallback="Sub-agent state was discarded on reload",
)
AWAITING_SUB_AGENTS_MESSAGE = msg(
    "history_markers.awaiting_sub_agents",
    fallback="Awaiting {count} sub-agent(s)",
    plural_fallback="Awaiting {count} sub-agent(s)",
)

ANTHROPIC_THINKING_STRIPPED_KEY: Final[str] = "_chrys_anthropic_thinking_stripped"
"""Set to True in a reasoning ``Content``'s ``additional_properties`` once it is left out for good.

Anthropic refused a request replaying it as bound to a different conversation,
and then accepted that request without any of the thinking it replayed: every
reasoning content the refused request replayed is marked, refused or not, and
never sent again.
"""


class HistoryMarkerKind:
    """String constants stored in ``Message.additional_properties``."""

    KEY: Final[str] = "_chrys_kind"
    PROFILE_SWITCH_TO_KEY: Final[str] = "_switch_to"
    STATUS_CODE_KEY: Final[str] = "_status_code"

    STATUS_EXECUTION_INTERRUPTED: Final[str] = "execution_interrupted"
    STATUS_EXECUTION_FAILED: Final[str] = "execution_failed"
    STATUS_SESSION_CLOSED: Final[str] = "session_closed"
    STATUS_SUB_AGENT_STATE_DISCARDED: Final[str] = "sub_agent_state_discarded"
    STATUS_AWAITING_SUB_AGENTS: Final[str] = "awaiting_sub_agents"
    STATUS_CODES: Final[frozenset[str]] = frozenset(
        {
            STATUS_EXECUTION_INTERRUPTED,
            STATUS_EXECUTION_FAILED,
            STATUS_SESSION_CLOSED,
            STATUS_SUB_AGENT_STATE_DISCARDED,
            STATUS_AWAITING_SUB_AGENTS,
        }
    )

    # Mid-turn user-message flags (see ``chrys.foundation.models.turns``).
    # ``_injected`` marks user-authored mid-turn input (live injections,
    # resume guidance) — permanent user content that must not open a turn.
    # ``_continuation`` marks the synthetic ``"continue"`` nudge the orchestration
    # sends on resume — never user-authored, removed post-run.  Both live in
    # ``additional_properties``: never serialized to the wire, persisted
    # wholesale in ``session.json``.
    INJECTED_KEY: Final[str] = "_injected"
    CONTINUATION_KEY: Final[str] = "_continuation"
    MID_TURN_USER_KEYS: Final[frozenset[str]] = frozenset({INJECTED_KEY, CONTINUATION_KEY})
    # Per-consumption identity for injected messages: stamped on the wire
    # copy at consumption and on the persisted/replayed history copy, so
    # crash-recovery replay dedups by identity — same-TEXT matching cannot
    # distinguish a persisted copy of this consumption from a distinct
    # earlier injection that happens to share the text.
    INJECTION_ID_KEY: Final[str] = "_injection_id"
    # System reminders a user message carried on an established provider
    # request, as ``[{"kind": ..., "text": ...}]``.  The reminder middleware
    # re-renders them on every later call so the conversation prefix stays
    # byte-identical across turns; the message's own contents never hold them.
    SYSTEM_REMINDERS_KEY: Final[str] = "_chrys_system_reminders"

    TURN: Final[str] = "turn"
    SUMMARY: Final[str] = "summary"
    INTERRUPTED: Final[str] = "interrupted"
    AWAITING_SUB_AGENTS: Final[str] = "awaiting_sub_agents"

    STATUS_MARKERS: Final[frozenset[str]] = frozenset({INTERRUPTED, AWAITING_SUB_AGENTS})
    SESSION_COUNT_EXCLUDED: Final[frozenset[str]] = frozenset({TURN, INTERRUPTED, AWAITING_SUB_AGENTS})
    LAST_WORDS_EXCLUDED: Final[frozenset[str]] = frozenset({TURN, SUMMARY, INTERRUPTED})


def copy_reminder_record(source: Mapping[str, Any] | None, target: dict[str, Any]) -> None:
    """Carry *source*'s reminder record onto *target*, another copy of the same user message.

    A rebuilt copy that lost the record would render bare on later requests
    and change the cached prefix from there on, so every site that rebuilds
    a user message calls this: the persisted injection copy, retry replay and
    its failure fallback, the crash-recovery checkpoint and a child's retry
    seed.  Entries are copied fresh: the source dict may still be live.
    """
    record = source.get(HistoryMarkerKind.SYSTEM_REMINDERS_KEY) if source is not None else None
    if isinstance(record, list) and record:
        target[HistoryMarkerKind.SYSTEM_REMINDERS_KEY] = [dict(entry) for entry in record if isinstance(entry, dict)]
