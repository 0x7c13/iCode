# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Turn identity persisted alongside the conversation (``chrys_trajectory`` state key).

``turn_id`` is minted per turn start and never reused; ``turn_number`` is the
display ordinal (``_turn``) a rollback may hand out again. Rollback and a
later runtime need the mapping between the two — ``session.rollback`` names
its ``target_turn_id`` and the first superseded sequence — so the recorder
keeps a tiny registry inside the session state: it rides along with every
save, snapshot, restore and fork exactly like the messages it describes, and
a rollback snapshot therefore restores the registry as of that turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

TRAJECTORY_STATE_KEY: Final = "chrys_trajectory"
_TURNS_KEY: Final = "turns"


@dataclass(frozen=True, slots=True)
class TurnRecord:
    turn_number: int
    turn_id: str
    started_sequence: int


def record_turn_started(state: dict[str, Any], record: TurnRecord, *, is_retry: bool = False) -> None:
    """Register *record*, replacing any earlier entry for the same turn number.

    A retry is the exception: it re-opens the ordinal its first pass already
    holds, and everything that pass wrote is part of the same turn. Keeping
    the first entry is what lets a rollback supersede the turn whole — from
    the latest attempt's sequence, the abandoned pass would stay on the
    surviving branch. A number handed out again after a rollback is not a
    retry, and its fresh pass takes the slot.
    """
    if is_retry and turn_record(state, record.turn_number) is not None:
        return
    registry = state.get(TRAJECTORY_STATE_KEY)
    if not isinstance(registry, dict):
        registry = {}
        state[TRAJECTORY_STATE_KEY] = registry
    turns = registry.get(_TURNS_KEY)
    if not isinstance(turns, dict):
        turns = {}
        registry[_TURNS_KEY] = turns
    turns[str(record.turn_number)] = {"turn_id": record.turn_id, "started_sequence": record.started_sequence}


def turn_record(state: dict[str, Any] | None, turn_number: int) -> TurnRecord | None:
    """Return the registered record for *turn_number*, if any."""
    if not state:
        return None
    registry = state.get(TRAJECTORY_STATE_KEY)
    if not isinstance(registry, dict):
        return None
    turns = registry.get(_TURNS_KEY)
    if not isinstance(turns, dict):
        return None
    entry = turns.get(str(turn_number))
    if not isinstance(entry, dict):
        return None
    turn_id = entry.get("turn_id")
    started = entry.get("started_sequence")
    if not isinstance(turn_id, str) or not isinstance(started, int) or isinstance(started, bool):
        return None
    return TurnRecord(turn_number=turn_number, turn_id=turn_id, started_sequence=started)


def forget_turns_after(state: dict[str, Any], turn_number: int) -> None:
    """Drop registry entries above *turn_number* (rollback without a snapshot registry)."""
    registry = state.get(TRAJECTORY_STATE_KEY)
    if not isinstance(registry, dict):
        return
    turns = registry.get(_TURNS_KEY)
    if not isinstance(turns, dict):
        return
    for key in list(turns):
        if not isinstance(key, str) or not key.isdigit() or int(key) > turn_number:
            del turns[key]
