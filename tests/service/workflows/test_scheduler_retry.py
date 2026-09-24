# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The §6.3 error table, the activation state machine, and manual-retry idempotency."""

from __future__ import annotations

import pytest

from chrys.service.workflows.outcomes import RunOutcome
from chrys.service.workflows.scheduler import (
    Activate,
    ActivationState,
    AttemptPhase,
    AttemptRef,
    CancelActivation,
    ErrorClass,
    FailureReport,
    NodeStateChanged,
    PersistRetryKey,
    RetryRejected,
    RetryRejection,
    RunFinished,
    RunMode,
    RunOutput,
    ScheduleRetry,
    WorkflowScheduler,
    activation_id,
    auto_retry_permitted,
)
from chrys.service.workflows.sdk import Retry, WorkflowBuilder, WorkflowValue
from tests.service.workflows.driver import SimulatedRunner, body_fn, build, complete_body, first_case, yes

A1 = activation_id("a", 1)
_BOOM = FailureReport(ErrorClass.PYTHON_EXCEPTION, "boom")

# (error class, backend approved, auto-retry while attempts remain) — the auto-retry column of the table.
_TABLE = [
    (ErrorClass.AGENT_TRANSIENT, True, True),
    (ErrorClass.AGENT_TRANSIENT, False, False),
    (ErrorClass.AGENT_NON_TRANSIENT, True, False),
    (ErrorClass.AGENT_TIMEOUT, True, True),
    (ErrorClass.AGENT_TIMEOUT, False, False),
    (ErrorClass.PYTHON_EXCEPTION, False, True),
    (ErrorClass.EVALUATION_ERROR, True, False),
    (ErrorClass.PYTHON_TIMEOUT, False, True),
    (ErrorClass.ASK_UNAVAILABLE, True, False),
    (ErrorClass.VALUE_TOO_LARGE, True, False),
    (ErrorClass.PROTOCOL_LIMIT, True, False),
    (ErrorClass.VALUE_NOT_SERIALIZABLE, True, False),
    (ErrorClass.LOOP_NO_VALUE, True, False),
]


@pytest.mark.parametrize(("error_class", "backend_approved", "expected"), _TABLE)
def test_error_table_auto_retry_column(error_class: ErrorClass, backend_approved: bool, expected: bool) -> None:
    failure = FailureReport(error_class, "x", backend_approved=backend_approved)
    assert auto_retry_permitted(failure, attempt=1, max_attempts=3) is expected
    assert auto_retry_permitted(failure, attempt=2, max_attempts=3) is expected
    assert auto_retry_permitted(failure, attempt=3, max_attempts=3) is False


def test_every_error_class_has_a_table_row() -> None:
    assert {row[0] for row in _TABLE} == set(ErrorClass)


def _single(wf: WorkflowBuilder) -> None:
    node = wf.python("a", body_fn, retry=Retry(max_attempts=3, backoff=2.5))
    wf.start(node)
    wf.output(node)


def _ref(attempt: int, node: str = "a") -> AttemptRef:
    return AttemptRef("run", node, activation_id(node, 1), attempt)


def test_auto_retry_walks_retrying_then_running_until_exhausted() -> None:
    runner = SimulatedRunner(build(_single), body=lambda work: _BOOM)
    runner.start()
    assert runner.decisions == [
        NodeStateChanged(_ref(1), ActivationState.RUNNING),
        Activate(_ref(1), "python", WorkflowValue("input"), ()),
    ]

    runner.step()  # attempt 1 fails
    assert runner.decisions[2:] == [
        NodeStateChanged(_ref(1), ActivationState.RETRYING, error=_BOOM, phase=AttemptPhase.BODY),
        ScheduleRetry(_ref(2), 2.5),
    ]
    runner.step()  # backoff elapsed
    assert runner.decisions[4:] == [
        NodeStateChanged(_ref(2), ActivationState.RUNNING),
        Activate(_ref(2), "python", WorkflowValue("input"), ()),
    ]
    runner.step()  # attempt 2 fails
    runner.step()  # backoff elapsed
    runner.step()  # attempt 3 fails: exhausted

    assert runner.decisions[-1] == NodeStateChanged(
        _ref(3), ActivationState.AWAITING_RETRY, error=_BOOM, phase=AttemptPhase.BODY
    )
    assert runner.finished is None
    assert runner.outstanding == []
    assert runner.states(A1) == [
        ActivationState.RUNNING,
        ActivationState.RETRYING,
        ActivationState.RUNNING,
        ActivationState.RETRYING,
        ActivationState.RUNNING,
        ActivationState.AWAITING_RETRY,
    ]


def test_backend_gate_decides_agent_transient_retries() -> None:
    def configure(wf: WorkflowBuilder) -> None:
        node = wf.agent("a", profile="A")  # default Retry(3)
        wf.start(node)
        wf.output(node)

    denied = FailureReport(ErrorClass.AGENT_TRANSIENT, "429", backend_approved=False)
    runner = SimulatedRunner(build(configure), body=lambda work: denied)
    runner.start()
    runner.step()
    assert runner.state(A1) is ActivationState.AWAITING_RETRY
    assert runner.of(ScheduleRetry) == []

    approved = FailureReport(ErrorClass.AGENT_TRANSIENT, "429", backend_approved=True)
    runner = SimulatedRunner(build(configure), body=lambda work: approved)
    runner.start()
    runner.step()
    assert runner.state(A1) is ActivationState.RETRYING
    assert runner.of(ScheduleRetry) == [ScheduleRetry(_ref(2), 0.0)]


@pytest.mark.parametrize(
    "error_class",
    [
        ErrorClass.AGENT_NON_TRANSIENT,
        ErrorClass.ASK_UNAVAILABLE,
        ErrorClass.VALUE_TOO_LARGE,
        ErrorClass.PROTOCOL_LIMIT,
        ErrorClass.VALUE_NOT_SERIALIZABLE,
    ],
)
def test_deterministic_failures_go_straight_to_awaiting_retry(error_class: ErrorClass) -> None:
    failure = FailureReport(error_class, "x", backend_approved=True)
    runner = SimulatedRunner(build(_single), body=lambda work: failure)
    runner.start()
    runner.step()
    assert runner.states(A1) == [ActivationState.RUNNING, ActivationState.AWAITING_RETRY]
    assert runner.of(ScheduleRetry) == []


def _fan_out(wf: WorkflowBuilder) -> None:
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    c = wf.agent("c", profile="A")
    wf.start(a)
    wf.edge(a, b)
    wf.edge(a, c)
    wf.output(b)
    wf.output(c)


def _fail_b(work: Activate) -> WorkflowValue | FailureReport:
    return _BOOM if work.ref.node_id == "b" else complete_body(work)


def test_headless_failure_fails_the_run_and_cancels_in_flight_branches() -> None:
    runner = SimulatedRunner(build(_fan_out), body=_fail_b, mode=RunMode.HEADLESS)
    runner.start()
    runner.step()  # a
    runner.step()  # b fails, c is in flight

    assert runner.decisions[-4:] == [
        NodeStateChanged(_ref(1, "b"), ActivationState.FAILED, error=_BOOM, phase=AttemptPhase.BODY),
        CancelActivation(_ref(1, "c")),
        NodeStateChanged(_ref(1, "c"), ActivationState.CANCELLED),
        RunFinished(RunOutcome.NODE_FAILED, (), node_id="b", error=_BOOM),
    ]
    assert runner.outstanding == []


def test_interactive_failure_keeps_independent_branches_running() -> None:
    runner = SimulatedRunner(build(_fan_out), body=_fail_b)
    runner.start()
    assert runner.run() is None
    assert runner.state(activation_id("b", 1)) is ActivationState.AWAITING_RETRY
    assert runner.state(activation_id("c", 1)) is ActivationState.COMPLETED

    runner.body = complete_body
    runner.retry(activation_id("b", 1), "req", 1)
    finished = runner.run()
    assert finished == RunFinished(
        RunOutcome.COMPLETED,
        (
            RunOutput("b", activation_id("b", 1), WorkflowValue("b#2")),
            RunOutput("c", activation_id("c", 1), WorkflowValue("c#1")),
        ),
    )


def test_evaluation_error_on_a_normal_node_reruns_the_whole_activation() -> None:
    def configure(wf: WorkflowBuilder) -> None:
        a = wf.python("a", body_fn)
        b = wf.python("b", body_fn)
        c = wf.python("c", body_fn)
        wf.start(a)
        wf.switch(a, cases=[(yes, b)], default=c)
        wf.output(b)
        wf.output(c)

    runner = SimulatedRunner(build(configure), outgoing=lambda work: "pred boom")
    runner.start()
    assert runner.run() is None
    change = runner.decisions[-1]
    assert change == NodeStateChanged(
        _ref(1),
        ActivationState.AWAITING_RETRY,
        error=FailureReport(ErrorClass.EVALUATION_ERROR, "pred boom"),
        phase=AttemptPhase.OUTGOING,
    )

    runner.outgoing = first_case
    out = runner.retry(A1, "req", 1)
    assert out[2] == Activate(_ref(2), "python", WorkflowValue("input"), ())
    finished = runner.run()
    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    assert [output.node_id for output in finished.outputs] == ["b"]


def test_late_inputs_are_dropped() -> None:
    runner = SimulatedRunner(build(_single))
    runner.start()
    scheduler = runner.scheduler
    live = _ref(1)

    assert scheduler.activation_completed(AttemptRef("run", "a", A1, 0), WorkflowValue("stale")) == ()
    assert scheduler.activation_completed(AttemptRef("other", "a", A1, 1), WorkflowValue("stale")) == ()
    assert scheduler.activation_failed(AttemptRef("run", "a", A1, 2), _BOOM) == ()
    assert scheduler.backoff_elapsed(_ref(2)) == ()
    assert scheduler.snapshot(A1) is not None and scheduler.snapshot(A1).state is ActivationState.RUNNING

    runner.step()
    assert runner.finished is not None
    assert scheduler.activation_completed(live, WorkflowValue("again")) == ()
    assert scheduler.cancel() == ()
    assert scheduler.worker_lost() == ()
    assert scheduler.manual_retry("a", A1, "r", 1) == (RetryRejected("a", A1, "r", RetryRejection.RUN_FINISHED),)


def test_backoff_for_another_attempt_is_ignored() -> None:
    runner = SimulatedRunner(build(_single), body=lambda work: _BOOM)
    runner.start()
    runner.step()
    assert runner.state(A1) is ActivationState.RETRYING
    assert runner.scheduler.backoff_elapsed(_ref(3)) == ()
    assert runner.scheduler.backoff_elapsed(_ref(1)) == ()
    assert runner.scheduler.backoff_elapsed(AttemptRef("other", "a", A1, 2)) == ()
    assert runner.scheduler.backoff_elapsed(_ref(2))[0] == NodeStateChanged(_ref(2), ActivationState.RUNNING)


def _awaiting_runner() -> SimulatedRunner:
    def configure(wf: WorkflowBuilder) -> None:
        node = wf.python("a", body_fn)  # python default: no auto retry
        wf.start(node)
        wf.output(node)

    runner = SimulatedRunner(build(configure), body=lambda work: _BOOM)
    runner.start()
    assert runner.run() is None
    assert runner.state(A1) is ActivationState.AWAITING_RETRY
    return runner


def test_manual_retry_persists_the_key_before_any_side_effect() -> None:
    runner = _awaiting_runner()
    runner.body = complete_body
    assert runner.retry(A1, "req", 1) == (
        PersistRetryKey("req", _ref(2)),
        NodeStateChanged(_ref(2), ActivationState.RUNNING),
        Activate(_ref(2), "python", WorkflowValue("input"), ()),
    )
    finished = runner.run()
    assert finished is not None and finished.outcome is RunOutcome.COMPLETED


def test_double_click_starts_exactly_one_attempt() -> None:
    runner = _awaiting_runner()
    runner.body = complete_body
    assert isinstance(runner.retry(A1, "click-1", 1)[0], PersistRetryKey)
    assert runner.retry(A1, "click-2", 1) == (RetryRejected("a", A1, "click-2", RetryRejection.NOT_AWAITING),)
    assert len(runner.activates("a")) == 2


def test_lost_ack_resend_with_the_same_request_id_is_a_duplicate() -> None:
    runner = _awaiting_runner()
    assert isinstance(runner.retry(A1, "req", 1)[0], PersistRetryKey)
    assert runner.retry(A1, "req", 1) == (RetryRejected("a", A1, "req", RetryRejection.DUPLICATE_REQUEST),)

    runner.run()  # the retried attempt fails again and waits with attempt 2
    assert runner.state(A1) is ActivationState.AWAITING_RETRY
    assert runner.retry(A1, "req", 2) == (RetryRejected("a", A1, "req", RetryRejection.DUPLICATE_REQUEST),)
    assert len(runner.activates("a")) == 2


def test_stale_replay_after_a_failed_retry_is_rejected() -> None:
    runner = _awaiting_runner()
    assert isinstance(runner.retry(A1, "r1", 1)[0], PersistRetryKey)
    runner.run()
    assert runner.state(A1) is ActivationState.AWAITING_RETRY

    assert runner.retry(A1, "r2", 1) == (RetryRejected("a", A1, "r2", RetryRejection.STALE_ATTEMPT),)
    assert runner.retry(A1, "r3", 2)[0] == PersistRetryKey("r3", _ref(3))
    assert len(runner.activates("a")) == 3


@pytest.mark.parametrize(
    ("fault", "outcome"),
    [
        ("cancel", RunOutcome.CANCELLED),
        ("worker_lost", RunOutcome.WORKER_LOST),
        ("storage_failed", RunOutcome.STORAGE_FAILED),
    ],
)
def test_run_faults_while_awaiting_cancel_the_activation_and_close_the_run(fault: str, outcome: RunOutcome) -> None:
    runner = _awaiting_runner()
    trigger = {
        "cancel": runner.scheduler.cancel,
        "worker_lost": runner.scheduler.worker_lost,
        "storage_failed": runner.scheduler.storage_failed,
    }[fault]

    assert trigger() == (NodeStateChanged(_ref(1), ActivationState.CANCELLED), RunFinished(outcome, ()))
    assert runner.scheduler.manual_retry("a", A1, "late", 1) == (
        RetryRejected("a", A1, "late", RetryRejection.RUN_FINISHED),
    )
    assert trigger() == ()


def _lone_node(wf: WorkflowBuilder) -> None:
    node = wf.python("a", body_fn)
    wf.start(node)
    wf.output(node)


@pytest.mark.parametrize("fault", ["cancel", "worker_lost", "storage_failed"])
def test_a_run_faulted_before_it_starts_issues_no_work(fault: str) -> None:
    scheduler = WorkflowScheduler(build(_lone_node), run_id="run", mode=RunMode.HEADLESS)
    (finished,) = getattr(scheduler, fault)()
    assert isinstance(finished, RunFinished)

    assert scheduler.start(WorkflowValue(text="input")) == ()
    assert scheduler.finished is finished
    assert scheduler.activation_ids() == ()


def test_run_fault_cancels_in_flight_and_backing_off_work() -> None:
    def configure(wf: WorkflowBuilder) -> None:
        a = wf.python("a", body_fn)
        b = wf.python("b", body_fn, retry=Retry(max_attempts=2, backoff=1.0))
        c = wf.agent("c", profile="A")
        wf.start(a)
        wf.edge(a, b)
        wf.edge(a, c)
        wf.output(c)

    runner = SimulatedRunner(build(configure), body=_fail_b)
    runner.start()
    runner.step()  # a
    runner.step()  # b fails and backs off; c stays in flight
    assert runner.state(activation_id("b", 1)) is ActivationState.RETRYING

    out = runner.scheduler.cancel()
    assert out == (
        NodeStateChanged(_ref(1, "b"), ActivationState.CANCELLED),
        CancelActivation(_ref(1, "c")),
        NodeStateChanged(_ref(1, "c"), ActivationState.CANCELLED),
        RunFinished(RunOutcome.CANCELLED, ()),
    )
    assert runner.scheduler.backoff_elapsed(_ref(2, "b")) == ()


def test_unknown_or_mismatched_activations_are_rejected() -> None:
    runner = _awaiting_runner()
    assert runner.scheduler.manual_retry("a", "nope@iter#1", "r", 1) == (
        RetryRejected("a", "nope@iter#1", "r", RetryRejection.UNKNOWN_ACTIVATION),
    )
    assert runner.scheduler.manual_retry("b", A1, "r", 1) == (
        RetryRejected("b", A1, "r", RetryRejection.UNKNOWN_ACTIVATION),
    )
