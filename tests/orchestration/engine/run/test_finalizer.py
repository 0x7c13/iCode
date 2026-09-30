# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Regression tests for post-run finalization ordering."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.orchestration.engine.run.turn_state import TurnRuntimeState
from chrys.orchestration.engine.state.machine import Trigger
from chrys.orchestration.engine.trajectory import TrajectoryRecorder
from chrys.service.agent_middleware.injection import QueuedInjection
from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
from chrys.service.session.history import SessionHistoryManager
from chrys.service.trajectory.preparation import PreparationOutcome
from tests.support.components import make_current
from tests.support.loaded_agents import SkillRefreshLoader, install_loaded_agent, make_manifest
from tests.support.turn_services import make_turn_finalizer


class _Executor:
    def __init__(self, order: list[str]) -> None:
        self._order = order
        self.run_failed = True
        self.was_interrupted = True
        self.last_error = "interrupted"
        self.service_session_id = "service-session"
        self.history_state: dict[str, Any] = {}
        self.trajectory_context = None
        self.opening_item_ids: list[str | None] = []

    def set_opening_item_id(self, item_id: str | None) -> None:
        self.opening_item_ids.append(item_id)

    def drain_decisions(self) -> list[str]:
        self._order.append("drain_approval")
        return ["approval"]

    def drain_batch_records(self) -> list[Any]:
        self._order.append("drain_batch_records")
        return []

    @property
    def backend(self):
        return self

    @property
    def inputs(self):
        return self

    @property
    def state(self):
        return self

    @property
    def approval(self):
        return self

    @property
    def tool_events(self):
        return self


class _History:
    def __init__(self, order: list[str]) -> None:
        self._order = order
        self.messages: list[Any] = []

    def merge_loop_messages(self, _loop_recorder: object, *, insert_index: int | None = None) -> None:
        self._order.append("merge_loop_messages")

    def persist_approval_decisions(self, decisions: list[Any], *, start_index: int) -> None:
        self._order.append(f"persist_approval_decisions:{decisions}:start={start_index}")

    def trim_to_last_complete_tool_results(self) -> None:
        self._order.append("trim_to_last_complete_tool_results")

    def remove_trailing_agent_text(self) -> None:
        self._order.append("remove_trailing_agent_text")

    def persist_batch_ids(self, _batch_records: list[Any]) -> dict[int, Any]:
        self._order.append("persist_batch_ids")
        return {}

    def persist_intermediate_texts(self, _texts: dict[int, str], _batch_anchors: dict[int, Any]) -> None:
        self._order.append("persist_intermediate_texts")

    def persist_consumed_injections(self, _injections: list[Any]) -> None:
        self._order.append("persist_consumed_injections")

    def backfill_missing_created_at(self, *, start_index: int) -> None:
        self._order.append(f"backfill_missing_created_at:start={start_index}")

    def remove_awaiting_sub_agents_marker(self) -> None:
        self._order.append("remove_awaiting_sub_agents_marker")

    def insert_interrupted_marker(
        self,
        reason: str | None = None,
        source: str = "user",
        status_code: str | None = None,
    ) -> None:
        self._order.append("insert_interrupted_marker")
        history = SessionHistoryManager()
        history.bind({"messages": self.messages})
        history.insert_interrupted_marker(reason=reason, source=source, status_code=status_code)

    def remove_all_status_markers(self) -> None:
        self._order.append("remove_all_status_markers")

    def insert_turn_marker(self) -> None:
        self._order.append("insert_turn_marker")


class _Injection:
    def __init__(self, host: _Host, order: list[str]) -> None:
        self._host = host
        self._order = order
        self.pending: list[Any] = []

    def drain_pending(self) -> list[Any]:
        self._order.append("drain_pending")
        assert self._host._turn_state.lease.injection_admission_open is False
        drained, self.pending = self.pending, []
        return drained


class _Fsm:
    def __init__(self, order: list[str]) -> None:
        self._order = order

    def try_transition(self, trigger: Trigger) -> None:
        self._order.append(f"transition:{trigger.name}")


class _Host:
    def __init__(self, order: list[str], *, save_result: bool = True) -> None:
        self.current = make_current(loaded=SimpleNamespace(), manifest=make_manifest())
        self._current = self.current
        self.session = SimpleNamespace()
        self._turn_state = TurnRuntimeState(history_start_index=7)
        self._bus = EventBus()
        self.session.session_id = "session-1"
        install_loaded_agent(self, bindings=_Executor(order))
        self._history = _History(order)
        install_loaded_agent(self, loop_recorder=None)
        install_loaded_agent(self, injection=_Injection(self, order))
        self.session.shutting_down = False
        install_loaded_agent(self, intermediate_texts={})
        install_loaded_agent(self, consumed_injections=[])
        self._turn_state.paused_sub_agents: set[str] = {"child-1"}
        self._fsm = _Fsm(order)
        self.session.mutation_tracker = None
        install_loaded_agent(self, agent_profile_fingerprint="")
        install_loaded_agent(self, model_profile_fingerprint="")
        self._persistence = SimpleNamespace(checkpoint_for=lambda _session_id: None, state_store=None)
        self._trajectory_recorder = TrajectoryRecorder()
        self._workspace_change_tracker = WorkspaceChangeTracker()
        self.settings = Settings(workspace_change_notice=False)
        self.session.mutation_coordinator = None
        install_loaded_agent(self, reminder_middleware=SystemReminderMiddleware())
        self.session.hook_manager = None
        self.session.agent_profile = None
        self.session.workspace = None
        self.session.turn_number = 1
        self._on_successful_turn = lambda: order.append("successful_turn_callback")
        self.saved = False
        self.save_result = save_result
        self.writer = SimpleNamespace(save_current_session=self._record_session_save)
        self.loader = SkillRefreshLoader(self.current)

    async def _record_session_save(self) -> bool:
        self.current.loaded.bindings._order.append("save")
        self.saved = True
        return self.save_result


@pytest.mark.asyncio
async def test_failed_run_metadata_before_trim_and_injection_drain_before_transition() -> None:
    order: list[str] = []
    host = _Host(order)
    reminder_scope = host.current.loaded.reminder_middleware.create_current_run_scope()
    run_scope = host._turn_state.lease.begin_current_run_scope(
        owner_admission_id=1,
        session_generation=1,
        build_generation=1,
        reminder_scope=reminder_scope,
    )
    host._turn_state.lease.open_injection_admission(run_scope)
    host._turn_state.lease.close_injection_admission(run_scope)
    owned_task = asyncio.current_task()
    assert owned_task is not None
    host._turn_state.lease.run_task = owned_task

    outcome = await make_turn_finalizer(
        current=host.current,
        session=host.session,
        turn_state=host._turn_state,
        writer=host.writer,
        bus=host._bus,
        fsm=host._fsm,
        history=host._history,
        trajectory_recorder=host._trajectory_recorder,
        workspace_change_tracker=host._workspace_change_tracker,
        persistence=host._persistence,
        on_successful_turn=host._on_successful_turn,
        settings_handle=SimpleNamespace(settings=host.settings),
    ).finalize()

    trim_index = order.index("trim_to_last_complete_tool_results")
    assert order.index("persist_approval_decisions:['approval']:start=7") < trim_index
    assert order.index("drain_pending") < order.index("transition:RUN_INTERRUPTED")
    assert host._turn_state.lease.injection_admission_open is False
    assert host._turn_state.paused_sub_agents == set()
    assert host.current.loaded.bindings.backend.service_session_id == ""
    assert host.saved is True
    assert host._turn_state.lease.was_run_task_finally_saved(owned_task) is True
    assert outcome.failed is True
    assert outcome.interrupted is True
    assert outcome.completed_scope == run_scope


def test_interrupted_terminal_state_persists_default_semantics() -> None:
    host = _Host([])

    make_turn_finalizer(
        current=host.current,
        session=host.session,
        turn_state=host._turn_state,
        writer=host.writer,
        bus=host._bus,
        fsm=host._fsm,
        history=host._history,
        trajectory_recorder=host._trajectory_recorder,
        workspace_change_tracker=host._workspace_change_tracker,
        persistence=host._persistence,
        on_successful_turn=host._on_successful_turn,
        settings_handle=SimpleNamespace(settings=host.settings),
    )._apply_terminal_history_state(failed=True, interrupted=True)

    marker = host._history.messages[-1]
    assert marker.text.encode("utf-8") == b"Execution interrupted"
    assert marker.additional_properties["_interrupted_by"] == "user"
    assert marker.additional_properties[HistoryMarkerKind.STATUS_CODE_KEY] == (
        HistoryMarkerKind.STATUS_EXECUTION_INTERRUPTED
    )


def test_failed_terminal_state_preserves_dynamic_diagnostic_without_code() -> None:
    host = _Host([])
    host.current.loaded.bindings.state.was_interrupted = False
    host.current.loaded.bindings.state.last_error = "retryable provider failure after tool return"

    make_turn_finalizer(
        current=host.current,
        session=host.session,
        turn_state=host._turn_state,
        writer=host.writer,
        bus=host._bus,
        fsm=host._fsm,
        history=host._history,
        trajectory_recorder=host._trajectory_recorder,
        workspace_change_tracker=host._workspace_change_tracker,
        persistence=host._persistence,
        on_successful_turn=host._on_successful_turn,
        settings_handle=SimpleNamespace(settings=host.settings),
    )._apply_terminal_history_state(failed=True, interrupted=False)

    marker = host._history.messages[-1]
    assert marker.text.encode("utf-8") == b"retryable provider failure after tool return"
    assert marker.additional_properties["_interrupted_by"] == "error"
    assert HistoryMarkerKind.STATUS_CODE_KEY not in marker.additional_properties


def test_failed_terminal_state_stamps_fixed_fallback_semantics() -> None:
    host = _Host([])
    host.current.loaded.bindings.state.was_interrupted = False
    host.current.loaded.bindings.state.last_error = ""

    make_turn_finalizer(
        current=host.current,
        session=host.session,
        turn_state=host._turn_state,
        writer=host.writer,
        bus=host._bus,
        fsm=host._fsm,
        history=host._history,
        trajectory_recorder=host._trajectory_recorder,
        workspace_change_tracker=host._workspace_change_tracker,
        persistence=host._persistence,
        on_successful_turn=host._on_successful_turn,
        settings_handle=SimpleNamespace(settings=host.settings),
    )._apply_terminal_history_state(failed=True, interrupted=False)

    marker = host._history.messages[-1]
    assert marker.text.encode("utf-8") == b"Execution failed"
    assert marker.additional_properties["_interrupted_by"] == "error"
    assert marker.additional_properties[HistoryMarkerKind.STATUS_CODE_KEY] == HistoryMarkerKind.STATUS_EXECUTION_FAILED


@pytest.mark.asyncio
async def test_post_commit_failure_finalizes_approval_batch_and_intermediate_metadata() -> None:
    order: list[str] = []
    host = _Host(order)
    host.current.loaded.bindings.state.was_interrupted = False
    host.current.loaded.bindings.state.last_error = "retryable provider failure after tool return"
    install_loaded_agent(host, intermediate_texts={4: "model narration before the tool"})
    host.current.loaded.bindings.tool_events.drain_batch_records = lambda: ["batch-record"]  # type: ignore[method-assign]

    def persist_batch_ids(records: list[Any]) -> dict[int, Any]:
        order.append(f"persist_batch_ids:{records}")
        return {4: "batch-anchor"}

    def persist_intermediate_texts(texts: dict[int, str], anchors: dict[int, Any]) -> None:
        order.append(f"persist_intermediate_texts:{texts}:{anchors}")

    host._history.persist_batch_ids = persist_batch_ids  # type: ignore[method-assign]
    host._history.persist_intermediate_texts = persist_intermediate_texts  # type: ignore[method-assign]

    await make_turn_finalizer(
        current=host.current,
        session=host.session,
        turn_state=host._turn_state,
        writer=host.writer,
        bus=host._bus,
        fsm=host._fsm,
        history=host._history,
        trajectory_recorder=host._trajectory_recorder,
        workspace_change_tracker=host._workspace_change_tracker,
        persistence=host._persistence,
        on_successful_turn=host._on_successful_turn,
        settings_handle=SimpleNamespace(settings=host.settings),
    ).finalize()

    trim_index = order.index("trim_to_last_complete_tool_results")
    assert order.index("merge_loop_messages") < trim_index
    assert order.index("persist_approval_decisions:['approval']:start=7") < trim_index
    assert "persist_batch_ids:['batch-record']" in order
    assert "persist_intermediate_texts:{4: 'model narration before the tool'}:{4: 'batch-anchor'}" in order
    assert "insert_interrupted_marker" in order
    assert "transition:RUN_FAILED" in order


@pytest.mark.asyncio
async def test_failed_final_save_is_not_recorded_as_persisted() -> None:
    order: list[str] = []
    host = _Host(order, save_result=False)
    owned_task = asyncio.current_task()
    assert owned_task is not None
    host._turn_state.lease.run_task = owned_task

    await make_turn_finalizer(
        current=host.current,
        session=host.session,
        turn_state=host._turn_state,
        writer=host.writer,
        bus=host._bus,
        fsm=host._fsm,
        history=host._history,
        trajectory_recorder=host._trajectory_recorder,
        workspace_change_tracker=host._workspace_change_tracker,
        persistence=host._persistence,
        on_successful_turn=host._on_successful_turn,
        settings_handle=SimpleNamespace(settings=host.settings),
    ).finalize()

    assert host.saved is True
    assert host._turn_state.lease.was_run_task_finally_saved(owned_task) is False


@pytest.mark.asyncio
async def test_final_save_mark_stays_with_owned_task_after_successor_install() -> None:
    host = _Host([])
    owned_task = asyncio.current_task()
    assert owned_task is not None
    host._turn_state.lease.run_task = owned_task

    await make_turn_finalizer(
        current=host.current,
        session=host.session,
        turn_state=host._turn_state,
        writer=host.writer,
        bus=host._bus,
        fsm=host._fsm,
        history=host._history,
        trajectory_recorder=host._trajectory_recorder,
        workspace_change_tracker=host._workspace_change_tracker,
        persistence=host._persistence,
        on_successful_turn=host._on_successful_turn,
        settings_handle=SimpleNamespace(settings=host.settings),
    ).finalize()
    successor = asyncio.create_task(asyncio.sleep(0))
    host._turn_state.lease.run_task = successor
    await successor

    assert host._turn_state.lease.was_run_task_finally_saved(owned_task) is True
    assert host._turn_state.lease.was_run_task_finally_saved(successor) is False


@pytest.mark.asyncio
async def test_drained_abandoned_injection_leaves_no_reminders() -> None:
    """An undelivered injection never queued its hook reminders, so a retry cannot carry them."""
    order: list[str] = []
    host = _Host(order)
    reminder = host.current.loaded.reminder_middleware
    reminder_scope = reminder.create_current_run_scope()
    run_scope = host._turn_state.lease.begin_current_run_scope(
        owner_admission_id=1,
        session_generation=1,
        build_generation=1,
        reminder_scope=reminder_scope,
    )
    host._turn_state.lease.open_injection_admission(run_scope)
    host._turn_state.lease.close_injection_admission(run_scope)
    reminder.prepare_turn(reminder_scope=reminder_scope)

    class _RecordingPreparation:
        def __init__(self) -> None:
            self.terminals: list[tuple[str, str | None]] = []

        async def finished(self, *, outcome: str, target_turn_id: str | None = None) -> None:
            del outcome, target_turn_id
            raise AssertionError("finalizer cleanup must not wait for trajectory acknowledgement")

        def finished_soon(self, *, outcome: str, target_turn_id: str | None = None) -> None:
            self.terminals.append((outcome, target_turn_id))

    preparation = _RecordingPreparation()
    # Mirror an active-turn injection commit: the text is pending with its reminders.
    host.current.loaded.injection.pending = [
        QueuedInjection(
            text="never delivered",
            injection_id="inj-1",
            reminders=("hook note for withdrawn text",),
            preparation=preparation,  # type: ignore[arg-type]
            target_turn_id="turn-1",
        )
    ]
    owned_task = asyncio.current_task()
    assert owned_task is not None
    host._turn_state.lease.run_task = owned_task

    await make_turn_finalizer(
        current=host.current,
        session=host.session,
        turn_state=host._turn_state,
        writer=host.writer,
        bus=host._bus,
        fsm=host._fsm,
        history=host._history,
        trajectory_recorder=host._trajectory_recorder,
        workspace_change_tracker=host._workspace_change_tracker,
        persistence=host._persistence,
        on_successful_turn=host._on_successful_turn,
        settings_handle=SimpleNamespace(settings=host.settings),
    ).finalize()

    assert "hook note for withdrawn text" not in reminder._build_reminders()
    assert preparation.terminals == [(PreparationOutcome.TARGET_STALE, "turn-1")]
    # The retry path preserves turn reminders; the undelivered one is not among them.
    reminder.prepare_turn(reminder_scope=reminder_scope, preserve_turn_reminders=True)
    assert "hook note for withdrawn text" not in reminder._build_reminders()


@pytest.mark.asyncio
async def test_abandoned_injection_notification_cancellation_cannot_strand_later_preparations() -> None:
    """All queue-owned traces close before the first cancellable notification."""
    host = _Host([])

    class _NonBlockingPreparation:
        def __init__(self) -> None:
            self.terminals: list[tuple[str, str | None]] = []

        async def finished(self, *, outcome: str, target_turn_id: str | None = None) -> None:
            del outcome, target_turn_id
            raise AssertionError("finalizer cleanup must not wait for trajectory acknowledgement")

        def finished_soon(self, *, outcome: str, target_turn_id: str | None = None) -> None:
            self.terminals.append((outcome, target_turn_id))

    class _CancellingBus:
        def __init__(self) -> None:
            self.publish_count = 0

        async def publish(self, _event: object) -> None:
            self.publish_count += 1
            raise asyncio.CancelledError

    first = _NonBlockingPreparation()
    second = _NonBlockingPreparation()
    host.current.loaded.injection.pending = [
        QueuedInjection(text="first", preparation=first, target_turn_id="turn-1"),  # type: ignore[arg-type]
        QueuedInjection(text="second", preparation=second, target_turn_id="turn-1"),  # type: ignore[arg-type]
    ]
    bus = _CancellingBus()
    host._bus = bus  # type: ignore[assignment]

    with pytest.raises(asyncio.CancelledError):
        await make_turn_finalizer(
            current=host.current,
            session=host.session,
            turn_state=host._turn_state,
            writer=host.writer,
            bus=host._bus,
            fsm=host._fsm,
            history=host._history,
            trajectory_recorder=host._trajectory_recorder,
            workspace_change_tracker=host._workspace_change_tracker,
            persistence=host._persistence,
            on_successful_turn=host._on_successful_turn,
            settings_handle=SimpleNamespace(settings=host.settings),
        )._drain_abandoned_injections()

    expected = [(PreparationOutcome.TARGET_STALE, "turn-1")]
    assert first.terminals == expected
    assert second.terminals == expected
    assert host.current.loaded.injection.pending == []
    assert bus.publish_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("run_failed", "interrupted", "transition"),
    [
        (False, False, "transition:RUN_COMPLETED"),
        (True, False, "transition:RUN_FAILED"),
        (True, True, "transition:RUN_INTERRUPTED"),
    ],
)
async def test_workspace_capture_follows_terminal_transition_and_precedes_save(
    run_failed: bool,
    interrupted: bool,
    transition: str,
) -> None:
    order: list[str] = []
    host = _Host(order)
    host.current.loaded.bindings.state.run_failed = run_failed
    host.current.loaded.bindings.state.was_interrupted = interrupted
    host.settings = Settings(workspace_change_notice=True)

    class _OrderingTracker:
        def capture_baseline(self, turn_id: int) -> None:
            assert turn_id == 1
            order.append("capture")

        def capture_cancelled(self) -> None:
            order.append("capture_cancelled")

        def capture_timed_out(self, turn_id: int) -> None:
            order.append(f"capture_timed_out:{turn_id}")

        def invalidate(self) -> None:
            order.append("invalidate")

    host._workspace_change_tracker = _OrderingTracker()  # type: ignore[assignment]

    await make_turn_finalizer(
        current=host.current,
        session=host.session,
        turn_state=host._turn_state,
        writer=host.writer,
        bus=host._bus,
        fsm=host._fsm,
        history=host._history,
        trajectory_recorder=host._trajectory_recorder,
        workspace_change_tracker=host._workspace_change_tracker,
        persistence=host._persistence,
        on_successful_turn=host._on_successful_turn,
        settings_handle=SimpleNamespace(settings=host.settings),
    ).finalize()

    assert order.index(transition) < order.index("capture") < order.index("save")


@pytest.mark.asyncio
async def test_capture_cancellation_still_persists_final_save() -> None:
    """A shutdown cancel landing on the advisory capture must not skip the save.

    Shutdown's post-run fallback suppresses its own trailing save, so the
    finalizer is the last writer: absorb the cancellation, save, re-raise.
    """
    order: list[str] = []
    host = _Host(order)
    host.settings = Settings(workspace_change_notice=True)

    class _CancelledTracker:
        def capture_baseline(self, turn_id: int) -> None:
            order.append("capture")
            raise asyncio.CancelledError

        def capture_cancelled(self) -> None:
            order.append("capture_cancelled")

        def invalidate(self) -> None:
            order.append("invalidate")

    host._workspace_change_tracker = _CancelledTracker()  # type: ignore[assignment]

    with pytest.raises(asyncio.CancelledError):
        await make_turn_finalizer(
            current=host.current,
            session=host.session,
            turn_state=host._turn_state,
            writer=host.writer,
            bus=host._bus,
            fsm=host._fsm,
            history=host._history,
            trajectory_recorder=host._trajectory_recorder,
            workspace_change_tracker=host._workspace_change_tracker,
            persistence=host._persistence,
            on_successful_turn=host._on_successful_turn,
            settings_handle=SimpleNamespace(settings=host.settings),
        ).finalize()

    assert host.saved is True
    assert order.index("capture_cancelled") < order.index("save")


@pytest.mark.asyncio
async def test_turn_close_cancellation_still_persists_final_save() -> None:
    """A shutdown cancel landing on the turn's close acknowledgement must not skip the save.

    The terminal marker is committed by then, so the turn is finished in the
    log; only the save makes it finished in the session, and shutdown's
    post-run fallback will not save again behind this path.
    """
    order: list[str] = []
    host = _Host(order)
    host.settings = Settings(workspace_change_notice=True)

    class _CancellingRecorder:
        async def turn_finished(self, *, end_reason: object) -> None:
            order.append("turn_finished")
            raise asyncio.CancelledError

        async def checkpoint(self) -> None:
            order.append("checkpoint")

        def turn_response_settled_soon(self, **_kwargs: object) -> None:
            order.append("turn_response_settled")

    class _OrderingTracker:
        def capture_baseline(self, turn_id: int) -> None:
            order.append("capture")

        def capture_cancelled(self) -> None:
            order.append("capture_cancelled")

        def invalidate(self) -> None:
            order.append("invalidate")

    host._trajectory_recorder = _CancellingRecorder()  # type: ignore[assignment]
    host._workspace_change_tracker = _OrderingTracker()  # type: ignore[assignment]

    with pytest.raises(asyncio.CancelledError):
        await make_turn_finalizer(
            current=host.current,
            session=host.session,
            turn_state=host._turn_state,
            writer=host.writer,
            bus=host._bus,
            fsm=host._fsm,
            history=host._history,
            trajectory_recorder=host._trajectory_recorder,
            workspace_change_tracker=host._workspace_change_tracker,
            persistence=host._persistence,
            on_successful_turn=host._on_successful_turn,
            settings_handle=SimpleNamespace(settings=host.settings),
        ).finalize()

    assert host.saved is True
    # The capture behind the close is advisory and would hold shutdown for its
    # own timeout on an already-cancelled task: skipped, and its baseline dropped.
    assert "capture" not in order
    assert order.index("turn_finished") < order.index("capture_cancelled") < order.index("save")
    assert order.index("save") < order.index("checkpoint")


@pytest.mark.asyncio
async def test_disabled_workspace_notice_invalidates_before_save() -> None:
    order: list[str] = []
    host = _Host(order)

    class _InvalidatingTracker:
        def invalidate(self) -> None:
            order.append("invalidate")

    host._workspace_change_tracker = _InvalidatingTracker()  # type: ignore[assignment]

    await make_turn_finalizer(
        current=host.current,
        session=host.session,
        turn_state=host._turn_state,
        writer=host.writer,
        bus=host._bus,
        fsm=host._fsm,
        history=host._history,
        trajectory_recorder=host._trajectory_recorder,
        workspace_change_tracker=host._workspace_change_tracker,
        persistence=host._persistence,
        on_successful_turn=host._on_successful_turn,
        settings_handle=SimpleNamespace(settings=host.settings),
    ).finalize()

    assert order.index("invalidate") < order.index("save")
