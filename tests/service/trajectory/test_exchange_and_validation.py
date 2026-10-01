# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``ExchangeTrace`` markers and the validation middleware's recording on top of them."""

from __future__ import annotations

import pytest

from chrys.foundation.trajectory.context import (
    TRAJECTORY_EXCHANGE_KWARG,
    ExchangeTrace,
    TrajectoryContext,
)
from chrys.foundation.trajectory.envelope import LinkRelation, MeasurementSource
from chrys.foundation.trajectory.event_types import (
    EventType,
    ExchangeOutcome,
    RetryMode,
    RetryReason,
    ValidationOutcome,
    ValidationReason,
)
from chrys.foundation.trajectory.ids import is_valid_analytics_id, new_analytics_id
from chrys.service.llm.observer import ExchangeRecorder
from chrys.service.trajectory.validation import ValidationTrace
from tests.service.trajectory._fakes import FakeSink, make_context

MONOTONIC = {"source": MeasurementSource.MONOTONIC_CLOCK, "method_version": 1}


def _exchange_context(sink: FakeSink, **facts: str) -> TrajectoryContext:
    context = make_context(sink).with_cycle(new_analytics_id()).with_exchange(new_analytics_id())
    return context.with_exchange_facts(facts) if facts else context


# ------------------------------------------------------------------------ ExchangeTrace


def test_exchange_trace_requires_an_exchange_operation() -> None:
    with pytest.raises(ValueError, match="exchange operation"):
        ExchangeTrace(make_context().with_cycle(new_analytics_id()))


def test_exchange_started_merges_facts_and_hangs_under_the_cycle() -> None:
    sink = FakeSink()
    context = _exchange_context(sink, model_profile_id="gpt", agent_profile_id="main")
    trace = ExchangeTrace(context)
    assert trace.operation_id == context.exchange_operation_id
    assert trace.generation == 0
    assert not trace.is_started and not trace.is_finished
    trace.started(payload={"continuation_mode": "none", "model_profile_id": "override"})
    trace.started(payload={"ignored": True})
    assert trace.is_started
    event = sink.only(EventType.MODEL_EXCHANGE_STARTED)
    assert event.operation_id == context.exchange_operation_id
    assert event.parent_operation_id == context.cycle_operation_id
    assert event.actor == context.actor
    assert event.turn_id == context.turn_id
    assert event.payload["model_profile_id"] == "override"
    assert event.payload["agent_profile_id"] == "main"
    assert event.payload["continuation_mode"] == "none"
    assert event.payload["started_at"] == event.occurred_at
    assert "context_revision_id" not in event.payload


def test_exchange_finished_payload_and_measurements_with_chunks() -> None:
    sink = FakeSink()
    context = _exchange_context(sink, model_profile_id="gpt")
    trace = ExchangeTrace(context)
    trace.set_context_revision(new_analytics_id())
    trace.started()
    trace.chunk_observed(visible=False)
    trace.chunk_observed(visible=True)
    trace.chunk_observed(visible=True)
    trace.stall_observed()
    trace.finished(
        outcome=ExchangeOutcome.SUCCESS,
        payload={"response_id": "resp_1", "usage": {"input_tokens": 3}},
        measurements={"/payload/usage": {"source": MeasurementSource.PROVIDER}},
    )
    trace.finished(outcome=ExchangeOutcome.ERROR)
    assert trace.is_finished
    started = sink.only(EventType.MODEL_EXCHANGE_STARTED)
    finished = sink.only(EventType.MODEL_EXCHANGE_FINISHED)
    assert started.payload["context_revision_id"] == trace.context_revision_id
    assert finished.operation_id == context.exchange_operation_id
    assert finished.parent_operation_id == context.cycle_operation_id
    payload = finished.payload
    assert payload["outcome"] == ExchangeOutcome.SUCCESS
    assert payload["model_profile_id"] == "gpt"
    assert payload["context_revision_id"] == trace.context_revision_id
    assert payload["response_id"] == "resp_1"
    assert payload["usage"] == {"input_tokens": 3}
    assert payload["started_at"] == started.payload["started_at"]
    assert payload["ended_at"] == finished.occurred_at
    assert payload["chunk_count"] == 3
    assert payload["stall_count"] == 1
    for key in ("duration_ms", "ttfc_ms", "ttfv_ms", "inter_chunk_p50_ms", "inter_chunk_p95_ms", "max_chunk_gap_ms"):
        assert payload[key] >= 0, key
    assert payload["ttfc_ms"] <= payload["ttfv_ms"] <= payload["duration_ms"]
    assert finished.measurements == {
        "/payload/duration_ms": MONOTONIC,
        "/payload/ttfc_ms": MONOTONIC,
        "/payload/ttfv_ms": MONOTONIC,
        "/payload/inter_chunk_p50_ms": MONOTONIC,
        "/payload/inter_chunk_p95_ms": MONOTONIC,
        "/payload/max_chunk_gap_ms": MONOTONIC,
        "/payload/usage": {"source": MeasurementSource.PROVIDER},
    }


def test_exchange_finished_without_chunks_omits_chunk_timings() -> None:
    sink = FakeSink()
    trace = ExchangeTrace(_exchange_context(sink))
    trace.started()
    trace.chunk_observed(visible=False)
    trace.finished(outcome=ExchangeOutcome.ERROR)
    payload = sink.only(EventType.MODEL_EXCHANGE_FINISHED).payload
    assert payload["chunk_count"] == 1
    assert "ttfc_ms" in payload
    assert "ttfv_ms" not in payload
    assert "inter_chunk_p50_ms" not in payload
    assert "max_chunk_gap_ms" not in payload


def test_an_exchange_that_never_started_records_nothing_when_it_closes() -> None:
    """The wire client reports the start immediately before the call.

    A trace closed without one never reached the provider — a cancel during
    request preparation — so there is no acquisition to report, and writing
    the terminal alone would leave a span whose start no reader can find.
    """
    sink = FakeSink()
    trace = ExchangeTrace(_exchange_context(sink))
    trace.finished(outcome=ExchangeOutcome.CANCELLED)
    assert not trace.is_started
    assert trace.is_finished
    assert sink.event_types == []


def test_exchange_mark_outcome_overrides_the_client_outcome() -> None:
    sink = FakeSink()
    trace = ExchangeTrace(_exchange_context(sink))
    trace.started()
    trace.mark_outcome(ExchangeOutcome.STALLED)
    trace.finished(outcome=ExchangeOutcome.SUCCESS)
    assert sink.only(EventType.MODEL_EXCHANGE_FINISHED).payload["outcome"] == ExchangeOutcome.STALLED


def test_exchange_abandon_closes_with_abandoned_once() -> None:
    sink = FakeSink()
    trace = ExchangeTrace(_exchange_context(sink))
    trace.started()
    trace.abandon()
    trace.abandon()
    finished = sink.of_type(EventType.MODEL_EXCHANGE_FINISHED)
    assert len(finished) == 1
    assert finished[0].payload["outcome"] == ExchangeOutcome.ABANDONED


@pytest.mark.asyncio
async def test_a_forwarded_exchange_is_not_closed_by_the_client_that_failed() -> None:
    """The loop knows which failure this was; the client only knows that it was one."""
    sink = FakeSink()
    trace = ExchangeTrace(_exchange_context(sink))
    trace.started()

    async def _boom() -> None:
        raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError):
        await ExchangeRecorder(trace, owned=False).wrap_awaitable(_boom())

    assert sink.of_type(EventType.MODEL_EXCHANGE_FINISHED) == []
    # The layer above unwinds next, and its verdict — which carries the retry
    # classification — is the one that lands.
    trace.finished(outcome=ExchangeOutcome.ERROR, payload={"error_code": "RuntimeError", "retryable": True})
    assert sink.only(EventType.MODEL_EXCHANGE_FINISHED).payload["retryable"] is True


@pytest.mark.asyncio
async def test_a_client_owned_exchange_closes_itself_when_it_fails() -> None:
    """A side call below the kernel has nobody above it to close its exchange."""
    sink = FakeSink()
    trace = ExchangeTrace(_exchange_context(sink))
    trace.started()

    async def _boom() -> None:
        raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError):
        await ExchangeRecorder(trace, owned=True).wrap_awaitable(_boom())

    finished = sink.only(EventType.MODEL_EXCHANGE_FINISHED)
    assert finished.payload["outcome"] == ExchangeOutcome.ERROR
    assert finished.payload["error_code"] == "RuntimeError"


def test_exchange_reissue_rolls_onto_a_new_operation_and_resets_state() -> None:
    sink = FakeSink()
    context = _exchange_context(sink, model_profile_id="gpt")
    trace = ExchangeTrace(context)
    original_id = trace.operation_id
    trace.set_context_revision(new_analytics_id())
    trace.started()
    trace.chunk_observed(visible=True)
    trace.stall_observed()
    trace.mark_outcome(ExchangeOutcome.STALLED)

    new_id = trace.reissue()
    assert is_valid_analytics_id(new_id) and new_id != original_id
    assert trace.operation_id == new_id
    assert trace.context.exchange_operation_id == new_id
    assert trace.context.cycle_operation_id == context.cycle_operation_id
    assert trace.context.exchange_facts == {"model_profile_id": "gpt"}
    assert trace.generation == 1
    assert not trace.is_started and not trace.is_finished
    assert trace.context_revision_id is None
    # The abandoned acquisition is closed under its own (original) operation id.
    abandoned = sink.only(EventType.MODEL_EXCHANGE_FINISHED)
    assert abandoned.operation_id == original_id
    assert abandoned.payload["outcome"] == ExchangeOutcome.STALLED
    assert abandoned.payload["chunk_count"] == 1

    trace.started()
    trace.finished(outcome=ExchangeOutcome.SUCCESS)
    started = sink.of_type(EventType.MODEL_EXCHANGE_STARTED)
    finished = sink.of_type(EventType.MODEL_EXCHANGE_FINISHED)
    assert [event.operation_id for event in started] == [original_id, new_id]
    assert finished[-1].operation_id == new_id
    assert finished[-1].payload["outcome"] == ExchangeOutcome.SUCCESS
    assert finished[-1].payload["chunk_count"] == 0
    assert finished[-1].payload["stall_count"] == 0
    assert "context_revision_id" not in finished[-1].payload


def test_exchange_reissue_accepts_a_caller_chosen_id_and_does_not_abandon_an_unstarted_trace() -> None:
    sink = FakeSink()
    trace = ExchangeTrace(_exchange_context(sink))
    chosen = new_analytics_id()
    assert trace.reissue(chosen) == chosen
    assert trace.operation_id == chosen
    assert trace.generation == 1
    assert sink.drafts == []
    trace.started()
    trace.finished(outcome=ExchangeOutcome.SUCCESS)
    # Already finished: a second reissue closes nothing again.
    trace.reissue()
    assert trace.generation == 2
    assert len(sink.of_type(EventType.MODEL_EXCHANGE_FINISHED)) == 1


# ---------------------------------------------------------------------- ValidationTrace


def test_validation_open_reads_the_exchange_trace_from_client_kwargs() -> None:
    trace = ExchangeTrace(_exchange_context(FakeSink()))
    opened = ValidationTrace.open({"client_kwargs": {TRAJECTORY_EXCHANGE_KWARG: trace}})
    assert opened is not None
    assert opened.exchange_operation_id == trace.operation_id
    assert ValidationTrace.open({}) is None
    assert ValidationTrace.open({"client_kwargs": {}}) is None
    assert ValidationTrace.open({"client_kwargs": {TRAJECTORY_EXCHANGE_KWARG: object()}}) is None
    assert ValidationTrace.open({"client_kwargs": "nope"}) is None


@pytest.mark.asyncio
async def test_validation_finished_rejected_links_to_the_exchange() -> None:
    sink = FakeSink()
    context = _exchange_context(sink)
    exchange = ExchangeTrace(context)
    validation = ValidationTrace(exchange)
    await validation.finished(
        accepted=False, reason_code=ValidationReason.EMPTY_CONTENTS, retryable=True, gave_up=False
    )
    event = sink.only(EventType.MODEL_VALIDATION_FINISHED)
    assert is_valid_analytics_id(event.operation_id)
    assert event.operation_id != exchange.operation_id
    assert event.parent_operation_id == exchange.operation_id
    assert event.links == (
        type(event.links[0])(relation=LinkRelation.VALIDATES, target_operation_id=exchange.operation_id),
    )
    assert event.payload == {
        "outcome": ValidationOutcome.REJECTED,
        "exchange_operation_id": exchange.operation_id,
        "attempt_index": 0,
        "reason_code": ValidationReason.EMPTY_CONTENTS,
        "retryable": True,
    }


@pytest.mark.asyncio
async def test_validation_finished_accepted_and_gave_up_shapes() -> None:
    sink = FakeSink()
    exchange = ExchangeTrace(_exchange_context(sink))
    validation = ValidationTrace(exchange)
    await validation.finished(accepted=True)
    await validation.finished(
        accepted=False, reason_code=ValidationReason.REASONING_ONLY, retryable=False, gave_up=True
    )
    accepted, rejected = sink.of_type(EventType.MODEL_VALIDATION_FINISHED)
    assert accepted.payload == {
        "outcome": ValidationOutcome.ACCEPTED,
        "exchange_operation_id": exchange.operation_id,
        "attempt_index": 0,
    }
    assert rejected.payload["gave_up"] is True
    assert rejected.payload["retryable"] is False
    assert rejected.payload["reason_code"] == ValidationReason.REASONING_ONLY


@pytest.mark.asyncio
async def test_validation_retry_rolls_the_exchange_and_abandons_the_started_one() -> None:
    sink = FakeSink()
    context = _exchange_context(sink)
    exchange = ExchangeTrace(context)
    validation = ValidationTrace(exchange)
    first_id = exchange.operation_id
    exchange.started()
    await validation.finished(accepted=False, reason_code=ValidationReason.WHITESPACE_ONLY, retryable=True)
    await validation.retry_scheduled(delay_seconds=1.5)
    await validation.retry_started()

    second_id = exchange.operation_id
    assert second_id != first_id and is_valid_analytics_id(second_id)
    assert exchange.generation == 1
    assert validation.exchange_operation_id == second_id

    scheduled = sink.only(EventType.RETRY_SCHEDULED)
    started = sink.only(EventType.RETRY_STARTED)
    assert scheduled.operation_id == second_id
    assert started.operation_id == second_id
    assert scheduled.parent_operation_id == context.cycle_operation_id
    assert started.parent_operation_id == context.cycle_operation_id
    assert scheduled.payload == {
        "reason_code": RetryReason.VALIDATION_REJECTED,
        "delay_ms": 1500,
        "retry_mode": RetryMode.VALIDATION,
        "previous_operation_id": first_id,
        "committed_work_present": False,
        "fallback_to_blocking": False,
    }
    assert started.payload == {
        "retry_mode": RetryMode.VALIDATION,
        "next_operation_id": second_id,
        "previous_operation_id": first_id,
    }
    abandoned = sink.only(EventType.MODEL_EXCHANGE_FINISHED)
    assert abandoned.operation_id == first_id
    assert abandoned.payload["outcome"] == ExchangeOutcome.ABANDONED
    assert sink.event_types == [
        EventType.MODEL_EXCHANGE_STARTED,
        EventType.MODEL_VALIDATION_FINISHED,
        EventType.RETRY_SCHEDULED,
        EventType.RETRY_STARTED,
        EventType.MODEL_EXCHANGE_FINISHED,
    ]

    # The next verdict is attempt 1 against the re-issued exchange.
    exchange.started()
    exchange.finished(outcome=ExchangeOutcome.SUCCESS)
    await validation.finished(accepted=True)
    verdicts = sink.of_type(EventType.MODEL_VALIDATION_FINISHED)
    assert verdicts[-1].payload["attempt_index"] == 1
    assert verdicts[-1].payload["exchange_operation_id"] == second_id
    assert verdicts[-1].parent_operation_id == second_id


@pytest.mark.asyncio
async def test_validation_retry_started_without_schedule_mints_its_own_id_and_clamps_delay() -> None:
    sink = FakeSink()
    exchange = ExchangeTrace(_exchange_context(sink))
    validation = ValidationTrace(exchange)
    first_id = exchange.operation_id
    await validation.retry_started()
    started = sink.only(EventType.RETRY_STARTED)
    assert started.operation_id == exchange.operation_id != first_id
    assert started.payload["next_operation_id"] == exchange.operation_id
    # No exchange had started, so nothing is abandoned.
    assert sink.of_type(EventType.MODEL_EXCHANGE_FINISHED) == []

    await validation.retry_scheduled(delay_seconds=-2.0)
    assert sink.only(EventType.RETRY_SCHEDULED).payload["delay_ms"] == 0


@pytest.mark.asyncio
async def test_validation_emit_failures_are_swallowed_and_the_exchange_still_rolls() -> None:
    sink = FakeSink()
    exchange = ExchangeTrace(_exchange_context(sink))
    validation = ValidationTrace(exchange)
    first_id = exchange.operation_id
    sink.fail_next = True
    await validation.finished(accepted=False, reason_code=ValidationReason.UNKNOWN)
    sink.fail_next = True
    await validation.retry_scheduled(delay_seconds=0)
    sink.fail_next = True
    await validation.retry_started()
    assert sink.drafts == []
    assert exchange.operation_id != first_id
    assert exchange.generation == 1
