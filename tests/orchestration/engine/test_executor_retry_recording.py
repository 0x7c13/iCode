# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A wire retry is recorded once, by the loop that performs it.

The executor's retry notifier drives the UI for both retry kinds, but only a
service-side whole-run retry is a ``retry.scheduled{retry_mode: run}`` of its
own.  The wire retry policy shares the notifier, so binding the recording one
there would count a single retry twice and leave a scheduled marker that no
``retry.started`` ever answers.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Event, InvocationRetryAttempt
from chrys.foundation.trajectory.event_types import EventType, ModelRunEndReason, RetryMode
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.kernel import AgentSession
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.invoker.resources import Conversation
from tests.service.trajectory._fakes import CancelAckSink, FakeSink, make_context


def _make_executor(bus: EventBus) -> tuple[TurnBindings, FakeSink]:
    executor = TurnBindings(
        conversation=Conversation(),
        agent=MagicMock(),
        session=AgentSession(),
        event_bus=bus,
        approval_middleware=MagicMock(),
        ask_user_middleware=MagicMock(),
        injection_middleware=MagicMock(),
    )
    sink = FakeSink()
    executor.trajectory_context = make_context(sink)
    executor._bound_emitter = executor._invocation_publishers.bind(executor.inputs.origin)
    return executor, sink


async def _collect(bus: EventBus) -> list[Event]:
    received: list[Event] = []

    async def _handler(event: Event) -> None:
        received.append(event)

    await bus.subscribe(InvocationRetryAttempt, _handler)
    return received


@pytest.mark.asyncio
async def test_a_run_retry_is_recorded_and_announced() -> None:
    bus = EventBus()
    received = await _collect(bus)
    executor, sink = _make_executor(bus)

    async with executor._attempt_trace.run({}, stream=True):
        pass
    await executor._attempts._publish_retry_attempt("stalled", 1, 3, 2, TimeoutError("stalled"))
    scheduled = sink.only(EventType.RETRY_SCHEDULED)
    assert sink.of_type(EventType.RETRY_STARTED) == []
    async with executor._attempt_trace.run({}, stream=True):
        pass

    started = sink.only(EventType.RETRY_STARTED)
    model_starts = sink.of_type(EventType.MODEL_RUN_STARTED)
    assert scheduled.payload["retry_mode"] == RetryMode.RUN
    assert scheduled.payload["delay_ms"] == 2000
    assert scheduled.operation_id == started.operation_id == model_starts[1].operation_id
    assert started.payload["next_operation_id"] == model_starts[1].operation_id
    assert started.monotonic_ns <= model_starts[1].monotonic_ns
    assert len(received) == 1


@pytest.mark.asyncio
async def test_first_model_run_is_caused_by_the_turn_preamble() -> None:
    executor, sink = _make_executor(EventBus())
    preamble_operation_id = new_analytics_id()
    context = executor.trajectory_context
    assert context is not None
    executor.trajectory_context = context.with_turn_preamble(preamble_operation_id)

    async with executor._attempt_trace.run({}, stream=True):
        pass

    started = sink.only(EventType.MODEL_RUN_STARTED)
    assert [(link.relation, link.target_operation_id) for link in started.links] == [
        ("caused_by", preamble_operation_id)
    ]
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_a_wire_retry_is_announced_without_a_second_scheduled_record() -> None:
    bus = EventBus()
    received = await _collect(bus)
    executor, sink = _make_executor(bus)

    await executor._publish_wire_retry_attempt("connection reset", 1, 3, 1, OSError("reset"))

    # The kernel loop already wrote this retry as a new exchange under the run.
    assert sink.of_type(EventType.RETRY_SCHEDULED) == []
    assert len(received) == 1


def test_the_wire_retry_policy_binds_the_announce_only_notifier() -> None:
    executor, _ = _make_executor(EventBus())

    policy: Any = executor._build_wire_retry_policy()

    assert policy.publish_retry == executor._publish_wire_retry_attempt


@pytest.mark.asyncio
async def test_cancellation_of_the_run_start_ack_still_settles_the_operation() -> None:
    executor, _ = _make_executor(EventBus())
    sink = CancelAckSink(at=1)
    executor.trajectory_context = make_context(sink)
    entered = False

    with pytest.raises(asyncio.CancelledError):
        async with executor._attempt_trace.run({}, stream=False):
            entered = True

    assert entered is False
    sink.assert_operations_settled()
    assert sink.event_types == [EventType.MODEL_RUN_STARTED, EventType.MODEL_RUN_FINISHED]
    assert sink.only(EventType.MODEL_RUN_FINISHED).payload["outcome"] == ModelRunEndReason.INTERRUPTED


@pytest.mark.asyncio
async def test_the_run_terminal_never_holds_the_answer_behind_its_acknowledgement() -> None:
    """The model has answered by then, and its task is gone: waiting here can lose the answer.

    A Stop pressed while the terminal waits out a slow writer finds no task to
    cancel, so it only raises the interrupt flag — and the caller then drops
    the finished response because the turn reads as interrupted.
    """
    executor, _ = _make_executor(EventBus())

    class _StuckAckSink(FakeSink):
        async def emit(self, draft: Any, *, payload_factory: Any = None, **kwargs: Any) -> Any:
            result = self._record(draft, payload_factory)
            if draft.event_type == EventType.MODEL_RUN_FINISHED:
                await asyncio.Event().wait()  # an acknowledgement that never comes
            return result

    sink = _StuckAckSink()
    executor.trajectory_context = make_context(sink)

    async def _run_span() -> None:
        async with executor._attempt_trace.run({}, stream=False):
            pass

    await asyncio.wait_for(_run_span(), timeout=5.0)

    # Queued, not awaited: the terminal has taken its place in the sequence
    # and the answer behind it publishes now rather than when the writer
    # catches up.
    assert sink.event_types == [EventType.MODEL_RUN_STARTED, EventType.MODEL_RUN_FINISHED]
    assert sink.only(EventType.MODEL_RUN_FINISHED).payload["outcome"] == ModelRunEndReason.COMPLETED
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_cancellation_of_the_retry_started_ack_never_leaves_an_orphaned_retry_started() -> None:
    """A committed ``retry.started`` must always be answered by the run it names.

    The whole-run retry commits ``retry.scheduled`` and then re-enters
    ``ModelRunTrace.run``, which commits ``retry.started`` before the retried
    ``model.run.started``. The writer commits the line before its shielded ack
    can be cancelled, so a Stop landing on the retry.started acknowledgement
    must still leave the run it announced accounted for: every
    ``retry.started{retry_mode: run}`` names the run that will happen
    (``next_operation_id == operation_id``), so the log owes it a same-id
    ``model.run.started``. The operation-settlement oracle has no retry
    lifecycle, so the pairing is asserted directly here.
    """
    executor, _ = _make_executor(EventBus())
    # Drafts 1-2: the first run's started/finished. Draft 3: retry.scheduled.
    # Draft 4: retry.started — cancel exactly on its acknowledgement.
    sink = CancelAckSink(at=4)
    executor.trajectory_context = make_context(sink)

    async with executor._attempt_trace.run({}, stream=True):
        pass
    await executor._attempts._publish_retry_attempt("stalled", 1, 3, 2, TimeoutError("stalled"))
    entered = False
    with pytest.raises(asyncio.CancelledError):
        async with executor._attempt_trace.run({}, stream=True):
            entered = True

    assert entered is False
    assert sink.drafts[3].event_type == EventType.RETRY_STARTED, "the cancel must land on retry.started"
    model_run_starts = sink.of_type(EventType.MODEL_RUN_STARTED)
    for started in sink.of_type(EventType.RETRY_STARTED):
        assert started.payload["retry_mode"] == RetryMode.RUN
        assert any(run.operation_id == started.operation_id for run in model_run_starts), (
            f"orphaned retry.started {started.operation_id}: the run it names never produced a "
            f"model.run.started (model.run.started ids: {[run.operation_id for run in model_run_starts]})"
        )
    # The pairing above is the retry contract; the oracle below still has to
    # hold for every operation it does know about.
    sink.assert_operations_settled()
    assert sink.drafts[-2].event_type == EventType.MODEL_RUN_STARTED
    assert sink.drafts[-2].payload["attempt_index"] == 1
    assert sink.drafts[3].monotonic_ns <= sink.drafts[-2].monotonic_ns
    assert sink.drafts[-1].event_type == EventType.MODEL_RUN_FINISHED
    assert sink.drafts[-1].payload["outcome"] == ModelRunEndReason.INTERRUPTED
