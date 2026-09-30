# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Crash-recovery checkpoint shaping for in-flight session turns."""

from __future__ import annotations

import copy
from datetime import datetime
from typing import TYPE_CHECKING, Any

from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.models.history_markers import SESSION_CLOSED_MESSAGE, HistoryMarkerKind
from chrys.foundation.trajectory.metadata import read_analytics_item_id
from chrys.service.agent_middleware.reminders.archive_pointer import CATALOG_POINTER_RECORD_COUNT_STATE_KEY
from chrys.service.session.history import SessionHistoryManager, stamp_history_item_ids

if TYPE_CHECKING:
    from collections.abc import Mapping

    from chrys.foundation.models.turns import UserMessageKind
    from chrys.kernel import LoopRecorder
    from chrys.service.agent_middleware.injection import ConsumedInjection
    from chrys.service.mutations.tracker import MutationTracker
    from chrys.service.session.runtime_metadata import SessionRuntimeMetadata


class _DetachedLoopRecorder:
    """Recorder view whose mergeable messages cannot mutate live framework objects."""

    def __init__(self, loop_recorder: LoopRecorder) -> None:
        self.initial_count = loop_recorder.initial_count
        loop_messages = loop_recorder.loop_messages
        self.loop_messages = copy.deepcopy(loop_messages) if loop_messages is not None else None
        # Only used for debug logging in ``merge_loop_messages`` when no loop messages exist.
        self.captured_count = None if self.loop_messages is None else len(self.loop_messages)


def shape_checkpoint_interruption(
    manager: SessionHistoryManager,
    loop_recorder: LoopRecorder,
    *,
    source: str = "",
    consumed_injections: list[ConsumedInjection] | None = None,
    insert_index: int | None = None,
) -> None:
    """Shape history into canonical interrupted form without draining live state.

    Merge recorder messages before replaying injections: the recorder may
    already contain the clean wire copy carried across a tool iteration.
    Replay can then deduplicate against the complete attempt before trimming.
    """
    manager.merge_loop_messages(_DetachedLoopRecorder(loop_recorder), insert_index=insert_index)
    if consumed_injections:
        manager.replay_consumed_injections(consumed_injections)
    manager.trim_to_last_complete_tool_results()
    manager.remove_trailing_agent_text()
    manager.insert_interrupted_marker(
        reason=format_message(SESSION_CLOSED_MESSAGE.bind()),
        source=source,
        status_code=HistoryMarkerKind.STATUS_SESSION_CLOSED,
    )
    manager.insert_turn_marker()


def build_recovery_state(
    live_state: dict[str, Any] | None,
    loop_recorder: LoopRecorder,
    *,
    mutation_tracker: MutationTracker | None,
    runtime_meta: SessionRuntimeMetadata,
    user_text: str | None,
    user_contents: list[Any] | None,
    user_created_at: datetime | str | None,
    user_kind: UserMessageKind = "opener",
    user_reminder_source: Mapping[str, Any] | None = None,
    consumed_injections: list[ConsumedInjection] | None = None,
    insert_index: int | None = None,
    last_words: str | None = None,
    last_words_manifest: list[dict[str, Any]] | None = None,
    last_words_breaker: dict[str, Any] | None = None,
    catalog_pointer_record_count: int | None = None,
    todos: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Return an interrupted-shaped recovery state for the current turn.

    The result goes to a writer thread while the turn keeps changing live
    state, so it shares no mutable object with any input. That holds with one
    whole-history copy: ``live_state`` is deep-copied once, every value merged
    over that copy is deep-copied on its own, ``_DetachedLoopRecorder`` copies
    the recorder's loop messages, and shaping adds only objects it mints
    (markers, replayed injection messages). A new merged input needs its own
    copy; ``tests/architecture/test_copy_freshness.py`` walks the result for
    objects shared with the inputs.
    """
    if live_state is None:
        return None

    copied = copy.deepcopy(live_state)
    if mutation_tracker is not None:
        copied["chrys_mutations"] = copy.deepcopy(mutation_tracker.serialize())
    copied.update(copy.deepcopy(runtime_meta.to_state_dict()))
    # The Phase 4 LAST_WORDS note replaces the dropped tool-call history of
    # the in-flight turn — recovering without it would resume from an
    # amputated transcript.
    if last_words:
        copied["last_words"] = last_words
    else:
        copied.pop("last_words", None)
    if last_words_manifest:
        copied["last_words_manifest"] = copy.deepcopy(last_words_manifest)
    else:
        copied.pop("last_words_manifest", None)
    # Breaker truth is unconditional: failed attempts commonly have no note
    # or manifest, and must still survive a crash/retry boundary.
    if last_words_breaker:
        copied["last_words_breaker"] = copy.deepcopy(last_words_breaker)
    else:
        copied.pop("last_words_breaker", None)
    if catalog_pointer_record_count is not None:
        copied[CATALOG_POINTER_RECORD_COUNT_STATE_KEY] = catalog_pointer_record_count
    else:
        copied.pop(CATALOG_POINTER_RECORD_COUNT_STATE_KEY, None)
    if todos:
        copied["chrys_todos"] = copy.deepcopy(todos)
    else:
        copied.pop("chrys_todos", None)

    manager = SessionHistoryManager()
    stamp_history_item_ids(copied)
    manager.bind(copied)
    if user_text or user_contents:
        # The recorded prompt contents are the live turn's objects; the
        # reminder record is copied entry by entry. The live input's item id
        # is the one the trajectory announced, so the rebuilt copy keeps it.
        manager.ensure_user_message(
            user_text or "",
            created_at=user_created_at,
            contents=copy.deepcopy(user_contents),
            kind=user_kind,
            item_id=read_analytics_item_id(user_reminder_source),
            reminder_source=user_reminder_source,
        )
    # Consumed injections reach history only at finalization, so a mid-run
    # checkpoint must replay them or a hard crash loses them. Replay belongs
    # after recorder merge but before trimming and terminal markers. They need
    # no copy: replay reads their immutable text, ids and timestamps into
    # messages it mints, copies their live reminder record entry by entry,
    # and never merges an injection object itself.
    shape_checkpoint_interruption(
        manager,
        loop_recorder,
        source="",
        consumed_injections=consumed_injections,
        insert_index=insert_index,
    )
    return copied
