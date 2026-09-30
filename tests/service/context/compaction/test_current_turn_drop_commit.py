# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the Phase 4 commit protocol: spill durability, cancellation windows and catalog records."""

import ast
import asyncio
import inspect
import textwrap
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from chrys.foundation.trajectory.context import trajectory_scope
from chrys.foundation.trajectory.envelope import SEGMENT_EVENT_TYPE, EventDraft
from chrys.foundation.trajectory.event_types import EventType as TrajectoryEventType
from chrys.foundation.trajectory.writer import EmitResult
from chrys.kernel import (
    EXCLUDE_REASON_KEY,
    included_token_count,
)
from chrys.service.context import compaction as compaction_mod
from chrys.service.context.compaction import (
    CompactionInfo,
)
from chrys.service.context.compaction.current_turn_drop import CurrentTurnDropRound
from chrys.service.context.compaction.last_words_state import ManifestEntry
from chrys.service.context.compaction.spill import (
    SpillBatchResult,
    SpillManifestItem,
    SpillQuota,
    catalog_live_records,
)
from tests.service.context.compaction._compaction_helpers import (
    _async_appender,
    _build_single_turn,
    _forced_phase4,
)
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.phase4_stubs import StubLastWordsGenerator, StubReminderMiddleware
from tests.support.reminder_stack import reminder_pair
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for, wait_until


def _failed_spill_result(reason: str = "I/O failure") -> SpillBatchResult:
    return SpillBatchResult(
        entries=(
            SpillManifestItem(
                record_id="",
                group_id="failed-group",
                record_dir="compactions/dropped/turn001",
                relative_path="",
                turn=1,
                round=1,
                sequence=1,
                tool="tool",
                display_argument="",
                outcome="unknown",
                size_chars=0,
                available=False,
                no_record_reason=reason,
            ),
        ),
        unexpected_io_failure=True,
    )


# ---------------------------------------------------------------------------
# Spill durability and quota eviction
# ---------------------------------------------------------------------------


async def test_phase4_commit_marks_quota_evicted_manifest_records_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reminder = StubReminderMiddleware()
    evicted_path = "compactions/dropped/turn001/001_old_00000001.md"
    reminder.last_words.append_manifest(
        [
            ManifestEntry(
                record_id="old-record",
                group_id="old-group",
                record_dir="compactions/dropped/turn001",
                relative_path=evicted_path,
                turn=1,
                round=1,
                sequence=1,
                tool="old",
                display_argument="",
                outcome="ok",
                size_chars=10,
            )
        ]
    )
    spill_result = SpillBatchResult(
        entries=(
            SpillManifestItem(
                record_id="new-record",
                group_id="new-group",
                record_dir="compactions/dropped/turn001",
                relative_path="compactions/dropped/turn001/001_new_00000002.md",
                turn=1,
                round=1,
                sequence=1,
                tool="new",
                display_argument="",
                outcome="ok",
                size_chars=10,
            ),
        ),
        evicted_relative_paths=frozenset({evicted_path}),
    )
    monkeypatch.setattr(
        "chrys.service.context.compaction.current_turn_drop.write_spill_batch",
        lambda *_args, **_kwargs: spill_result,
    )
    messages = _build_single_turn(2, result_size=1_000)
    strategy = _forced_phase4(
        messages,
        reminder_middleware=reminder,
    )

    assert await strategy(messages)

    manifest = reminder.last_words.get_last_words_manifest()
    assert manifest[0]["relative_path"] == evicted_path
    assert manifest[0]["available"] is False
    assert manifest[1]["record_id"] == "new-record"


@pytest.mark.parametrize(
    ("durability", "commits"),
    [("true", True), ("false", False), ("exception", False), ("absent", False)],
    ids=["true", "false", "exception", "absent"],
)
async def test_phase4_unexpected_spill_io_commits_only_with_true_durability(
    durability: str,
    commits: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After an unexpected spill I/O failure the round commits only once the
    recovery persistence callback reports true durability; a false, failing or
    absent callback vetoes the round."""
    reminder = StubReminderMiddleware()
    monkeypatch.setattr(
        "chrys.service.context.compaction.current_turn_drop.write_spill_batch",
        lambda *_args, **_kwargs: _failed_spill_result(),
    )
    messages = _build_single_turn(2, result_size=1_000)
    generator = StubLastWordsGenerator()
    strategy = _forced_phase4(messages, last_words_generator=generator, reminder_middleware=reminder)
    durability_calls = 0
    if durability == "true":

        async def persist_true() -> bool:
            nonlocal durability_calls
            await asyncio.sleep(0)
            durability_calls += 1
            return True

        strategy.set_recovery_persistence_callback(persist_true)
    elif durability == "false":

        async def persist_false() -> bool:
            return False

        strategy.set_recovery_persistence_callback(persist_false)
    elif durability == "exception":

        async def persist_error() -> bool:
            raise OSError("sidecar failed")

        strategy.set_recovery_persistence_callback(persist_error)

    changed = await strategy(messages)

    dropped = any(message.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop" for message in messages)
    assert changed is commits
    assert dropped is commits
    # Only a round that recovery persistence made durable claims the committed
    # signal; an abandoned round must never claim it.
    assert generator.committed_publishes == (1 if commits else 0)
    if commits:
        assert durability_calls == 1
        assert reminder.last_words.get_last_words() is not None
        assert reminder.last_words.get_last_words_manifest()[0]["no_record_reason"] == "I/O failure"
    else:
        assert reminder.last_words.get_last_words() is None
        assert reminder.last_words.get_last_words_manifest() == []
        breaker = reminder.last_words.get_drop_round_breaker()
        assert (breaker.attempts, breaker.consecutive_no_progress, breaker.tail_override) == (1, 1, True)


# ---------------------------------------------------------------------------
# Cancellation windows and trajectory phase ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cancellation_point", ["generate", "persist_recovery"])
async def test_phase4_precommit_cancellation_retains_persisted_accounting_without_no_progress(
    cancellation_point: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ChargingGenerator(StubLastWordsGenerator):
        async def generate(self, *args, spend_side_call_tokens=None, **kwargs):  # type: ignore[no-untyped-def]
            assert spend_side_call_tokens is not None
            assert spend_side_call_tokens(123)
            if cancellation_point == "generate":
                raise asyncio.CancelledError
            return await super().generate(*args, spend_side_call_tokens=spend_side_call_tokens, **kwargs)

    reminder = StubReminderMiddleware()
    pressure_events: list[str] = []
    messages = _build_single_turn(2, result_size=1_000)
    generator = _ChargingGenerator(text="[note]")
    strategy = _forced_phase4(
        messages,
        last_words_generator=generator,
        reminder_middleware=reminder,
        spill_root=tmp_path,
        spill_quota=SpillQuota(),
        spill_session_id="precommit-cancel",
        on_context_pressure=lambda reason, _breaker, _budget: pressure_events.append(reason),
    )

    if cancellation_point == "persist_recovery":
        real_write_spill_batch = compaction_mod.write_spill_batch

        def spill_then_report_projection_failure(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = real_write_spill_batch(*args, **kwargs)
            return SpillBatchResult(
                entries=result.entries,
                unexpected_io_failure=True,
                evicted_relative_paths=result.evicted_relative_paths,
            )

        async def cancel_recovery_persistence() -> bool:
            raise asyncio.CancelledError

        monkeypatch.setattr(
            "chrys.service.context.compaction.current_turn_drop.write_spill_batch",
            spill_then_report_projection_failure,
        )
        strategy.set_recovery_persistence_callback(cancel_recovery_persistence)

    with pytest.raises(asyncio.CancelledError):
        await strategy(messages)

    breaker = reminder.last_words.get_drop_round_breaker()
    assert breaker.attempts == 1
    assert breaker.side_call_tokens == 123
    assert breaker.consecutive_no_progress == 0
    assert breaker.tail_override is False
    assert breaker.disabled is False
    assert pressure_events == []
    assert reminder.last_words.get_last_words() is None
    assert reminder.last_words.get_last_words_manifest() == []
    assert not any(message.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop" for message in messages)
    if cancellation_point == "persist_recovery":
        assert catalog_live_records(tmp_path)


async def test_phase4_cancel_during_committed_publish_leaves_commit_finalized() -> None:
    """Cancellation delivered by the committed publish cannot split the commit.

    The publish is deliberately the first await after the synchronous commit
    block — by then the exclusion anchors and token bookkeeping must already
    be in place, or an interrupt-resume that persisted no anchors would
    retain both the original tool history and the LAST_WORDS note.
    """

    class _CancelOnCommittedGenerator(StubLastWordsGenerator):
        async def publish_committed(self) -> None:
            raise asyncio.CancelledError

    generator = _CancelOnCommittedGenerator(text="[note]")
    reminder = StubReminderMiddleware()
    messages = _build_single_turn(2, result_size=1_000)
    compaction_infos: list[CompactionInfo] = []

    strategy = _forced_phase4(
        messages,
        last_words_generator=generator,
        reminder_middleware=reminder,
        on_compaction=_async_appender(compaction_infos),
    )

    with pytest.raises(asyncio.CancelledError):
        await strategy(messages)

    excluded = [m for m in messages if m.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop"]
    assert excluded
    assert reminder.last_words.get_last_words() == "[note]"
    assert len(strategy._excluded_anchors) == len(excluded)
    assert strategy._last_included_tokens == included_token_count(messages)
    # The on_compaction notification (ToolCompacted for the main agent) must
    # survive the interrupt on a detached task — the rounds it reports are
    # already durably committed.
    await wait_for(
        lambda: bool(compaction_infos),
        timeout=ENGINE_TURN_TIMEOUT,
        description="phase-4 compaction info",
    )
    assert compaction_infos[0].phase == "phase4"
    assert compaction_infos[0].compacted_groups > 0
    assert compaction_infos[0].last_words_generated is True
    assert compaction_infos[0].tokens_after == strategy._last_included_tokens
    assert not strategy._detached_deliveries


class _BlocksThePhaseAck(FakeSink):
    """Real sink shape: the line lands, then its acknowledgement keeps waiting."""

    def __init__(self) -> None:
        super().__init__()
        self.phase_ack_reached = asyncio.Event()
        self.release_phase_ack = asyncio.Event()

    async def emit(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult:
        result = await super().emit(draft, payload_factory=payload_factory)
        if draft.event_type == TrajectoryEventType.COMPACTION_PHASE_FINISHED:
            self.phase_ack_reached.set()
            await self.release_phase_ack.wait()
        return result


async def test_phase4_records_its_phase_before_the_pass_can_close_the_run() -> None:
    """Cancelling the pass closes the run from the unwind, while the delivery
    it scheduled runs on undisturbed behind its shield. The phase's lines —
    including the segments continuing its arrays — must all be on the log by
    then, or a sub-event lands behind the terminal of its own parent."""
    generator = StubLastWordsGenerator(text="[note]")
    reminder = StubReminderMiddleware()
    messages = _build_single_turn(2, result_size=1_000)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_on_compaction(info: CompactionInfo) -> None:
        entered.set()
        await release.wait()

    strategy = _forced_phase4(
        messages,
        last_words_generator=generator,
        reminder_middleware=reminder,
        on_compaction=blocking_on_compaction,
    )
    sink = _BlocksThePhaseAck()

    with trajectory_scope(make_context(sink)):
        run = asyncio.create_task(strategy(messages))
        # Whichever of the two the delivery reaches first: awaiting the ack of
        # the phase it writes, or the observer callback after it.
        await wait_for(
            lambda: sink.phase_ack_reached.is_set() or entered.is_set(),
            timeout=ENGINE_TURN_TIMEOUT,
            description="phase-4 delivery in flight",
        )
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        sink.release_phase_ack.set()
        release.set()
        await wait_for(
            lambda: not strategy._detached_deliveries,
            timeout=ENGINE_TURN_TIMEOUT,
            description="detached compaction delivery drain",
        )

    closed = sink.event_types.index(TrajectoryEventType.COMPACTION_FINISHED)
    assert sink.event_types.index(TrajectoryEventType.COMPACTION_PHASE_FINISHED) < closed
    assert [index for index, name in enumerate(sink.event_types) if name == SEGMENT_EVENT_TYPE]
    assert all(index < closed for index, name in enumerate(sink.event_types) if name == SEGMENT_EVENT_TYPE)


async def test_phase4_cancel_during_post_commit_notification_still_delivers() -> None:
    """Cancellation while the end-of-pass on_compaction await is in flight
    must not lose ToolCompacted.

    The commit is already durable by then, so the delivery runs as an
    anchored task behind a shield: cancellation still propagates to the
    caller, but the callback finishes detached.  Without the shield the
    cancellation would kill the only delivery of an already-committed
    round.
    """
    generator = StubLastWordsGenerator(text="[note]")
    reminder = StubReminderMiddleware()
    messages = _build_single_turn(2, result_size=1_000)

    entered = asyncio.Event()
    release = asyncio.Event()
    compaction_infos: list[CompactionInfo] = []

    async def blocking_on_compaction(info: CompactionInfo) -> None:
        entered.set()
        await release.wait()
        compaction_infos.append(info)

    strategy = _forced_phase4(
        messages,
        last_words_generator=generator,
        reminder_middleware=reminder,
        on_compaction=blocking_on_compaction,
    )

    run = asyncio.create_task(strategy(messages))
    await entered.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    # The delivery survived the cancellation on its detached anchor and
    # completes once the (EventBus-shaped) handler unblocks.
    assert not compaction_infos
    assert strategy._detached_deliveries
    release.set()
    await wait_for(
        lambda: bool(compaction_infos),
        timeout=ENGINE_TURN_TIMEOUT,
        description="detached phase-4 compaction info",
    )
    assert compaction_infos[0].phase == "phase4"
    assert compaction_infos[0].compacted_groups > 0
    assert compaction_infos[0].last_words_generated is True
    await wait_for(
        lambda: not strategy._detached_deliveries,
        timeout=ENGINE_TURN_TIMEOUT,
        description="detached compaction delivery drain",
    )


async def test_phase4_cancellation_drains_real_spill_worker_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()
    finished = Event()

    def blocked_spill(*_args, **_kwargs) -> SpillBatchResult:  # type: ignore[no-untyped-def]
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release spill worker")
        finished.set()
        return SpillBatchResult(entries=())

    monkeypatch.setattr("chrys.service.context.compaction.current_turn_drop.write_spill_batch", blocked_spill)
    messages = _build_single_turn(2, result_size=1_000)
    strategy = _forced_phase4(
        messages,
        reminder_middleware=StubReminderMiddleware(),
    )
    task = asyncio.create_task(strategy(messages))
    started_seen = False
    completed_while_blocked = False
    cancelled = False
    try:
        started_seen = await wait_until(started.is_set, timeout=1)
        task.cancel()
        completed_while_blocked = await wait_until(task.done, timeout=0.2, interval=0.01)
    finally:
        release.set()
        if not task.done() and not task.cancelling():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True

    assert started_seen
    assert not completed_while_blocked
    assert finished.is_set()
    assert cancelled


# ---------------------------------------------------------------------------
# Catalog and note records, retry restore
# ---------------------------------------------------------------------------


async def test_retry_state_restores_phase4_content_but_keeps_breaker_monotonic(tmp_path: Path) -> None:
    """Phase-4 note context rolls back; real per-turn safety spend does not."""
    todo = ["[TODO] baseline"]
    spill_quota = SpillQuota()
    baseline_path = "compactions/dropped/turn001/baseline.md"
    spill_quota.initialize(1, available_relative_paths=[baseline_path])
    reminder, last_words = reminder_pair(
        session_root=tmp_path,
        spill_quota=spill_quota,
        todo_state_provider=lambda: todo[0],
    )
    reminder.prepare_turn()
    last_words.set_last_words("[baseline note]")
    last_words.append_manifest(
        [
            ManifestEntry(
                record_id="baseline-record",
                group_id="baseline-group",
                record_dir="compactions/dropped/turn001",
                relative_path=baseline_path,
                turn=1,
                round=1,
                sequence=1,
                tool="baseline_tool",
                display_argument="baseline",
                outcome="completed",
                size_chars=8,
                assistant_text=False,
                available=True,
            )
        ]
    )
    messages = _build_single_turn(2, result_size=1_000)
    strategy = _forced_phase4(
        messages,
        last_words_generator=StubLastWordsGenerator(text="[attempt note]"),
        reminder_middleware=reminder,
        last_words=last_words,
        spill_root=tmp_path,
        spill_quota=spill_quota,
        spill_session_id="phase4-retry",
    )
    retry_snapshot = strategy.snapshot_retry_state()
    todo[0] = "[TODO] changed during attempt"

    assert await strategy(messages)
    attempt_breaker = last_words.get_drop_round_breaker()
    assert attempt_breaker.attempts == 1
    assert last_words.get_last_words() == "[attempt note]"
    assert len(last_words.get_last_words_manifest()) > 1
    assert last_words.claim_context_pressure_notification()
    # The provider-side catalog can evict a record while this attempt is in
    # flight; retry restore must revalidate the snapshotted row via SpillQuota.
    spill_quota.initialize(0, available_relative_paths=[])

    strategy.restore_retry_state(retry_snapshot)

    assert last_words.get_last_words() == "[baseline note]"
    restored_manifest = last_words.get_last_words_manifest()
    assert [row["record_id"] for row in restored_manifest] == ["baseline-record"]
    assert restored_manifest[0]["available"] is False
    rendered = last_words.render_last_words_reminder_text()
    assert rendered is not None
    assert "[TODO] baseline" in rendered
    assert "changed during attempt" not in rendered
    assert last_words.get_drop_round_breaker() == attempt_breaker
    assert not last_words.claim_context_pressure_notification()


async def test_phase4_cancel_after_catalog_flush_leaves_catalog_visible_and_reminder_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reminder = StubReminderMiddleware()
    messages = _build_single_turn(2, result_size=1_000)
    strategy = _forced_phase4(
        messages,
        reminder_middleware=reminder,
        spill_root=tmp_path,
        spill_quota=SpillQuota(),
        spill_session_id="cancel-session",
    )

    async def flush_then_cancel(function, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        function(*args, **kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr(compaction_mod.asyncio, "to_thread", flush_then_cancel)

    with pytest.raises(asyncio.CancelledError):
        await strategy(messages)

    assert catalog_live_records(tmp_path)
    assert reminder.last_words.get_last_words() is None
    assert reminder.last_words.get_last_words_manifest() == []
    assert not any(message.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop" for message in messages)


async def test_phase4_archives_superseded_note_record_alongside_group_records(tmp_path: Path) -> None:
    """A later round's merge archives the pre-merge note verbatim on disk."""
    generator = StubLastWordsGenerator(text="[note v2]")
    reminder = StubReminderMiddleware()
    reminder.last_words.set_last_words("[note v1]")
    messages = _build_single_turn(4, result_size=1000)
    strategy = _forced_phase4(
        messages,
        last_words_generator=generator,
        reminder_middleware=reminder,
        spill_root=tmp_path,
        spill_quota=SpillQuota(),
        spill_session_id="note-session",
    )

    assert await strategy(messages)

    manifest = reminder.last_words.get_last_words_manifest()
    note_rows = [row for row in manifest if row["tool"] == "last_words"]
    assert len(note_rows) == 1
    note_row = note_rows[0]
    assert note_row["outcome"] == "merged"
    assert note_row["sequence"] == max(row["sequence"] for row in manifest)
    record = (tmp_path / note_row["relative_path"]).read_text(encoding="utf-8")
    assert record.startswith("# Superseded LAST_WORDS note\n")
    assert "[note v1]" in record
    kinds = {record.record_id: record.kind for record in catalog_live_records(tmp_path)}
    # The state drops rows it cannot parse, so every record must still be listed
    # (regression: an empty note group_id made the note row vanish).
    assert {row["record_id"] for row in manifest} == set(kinds)
    assert kinds.pop(note_row["record_id"]) == "note"
    assert set(kinds.values()) == {"group"}


async def test_phase4_first_drop_without_previous_note_writes_no_note_record(tmp_path: Path) -> None:
    reminder = StubReminderMiddleware()
    messages = _build_single_turn(4, result_size=1000)
    strategy = _forced_phase4(
        messages,
        reminder_middleware=reminder,
        spill_root=tmp_path,
        spill_quota=SpillQuota(),
        spill_session_id="note-session",
    )

    assert await strategy(messages)

    assert reminder.last_words.get_last_words() is not None
    assert all(row["tool"] != "last_words" for row in reminder.last_words.get_last_words_manifest())
    assert all(record.kind == "group" for record in catalog_live_records(tmp_path))


# ---------------------------------------------------------------------------
# Commit window shape
# ---------------------------------------------------------------------------


def test_phase4_commit_through_exclusion_contains_no_awaits() -> None:
    source = textwrap.dedent(inspect.getsource(CurrentTurnDropRound.run))
    tree = ast.parse(source)
    lines = source.splitlines()
    start = next(index for index, line in enumerate(lines, start=1) if "Steps 4-5: synchronous commit" in line)
    # The synchronous window ends at the deliberately-post-commit
    # ``publish_committed`` await — the first await allowed after the
    # note-set/manifest/exclusion/usage-sample block completes.
    end = next(index for index, line in enumerate(lines, start=1) if index > start and "publish_committed" in line)

    awaits = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Await) and start < node.lineno < end]

    assert awaits == []
