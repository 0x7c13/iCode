# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Loop boundary cases: completion barrier, closed exit, nesting, phase-resumed retries, exhaustion."""

from __future__ import annotations

import pytest

from chrys.service.workflows.graph import GraphSpec, ManifestError
from chrys.service.workflows.outcomes import RunOutcome
from chrys.service.workflows.scheduler import (
    Activate,
    ActivationState,
    AttemptPhase,
    AttemptRef,
    ErrorClass,
    EvaluateLoopUntil,
    EvaluateOutgoing,
    LoopActivated,
    LoopIteration,
    LoopVerdict,
    NodeStateChanged,
    PersistRetryKey,
    RetryRejected,
    RetryRejection,
    RunMode,
    RunOutput,
    activation_id,
)
from chrys.service.workflows.sdk import (
    BuilderScope,
    NodeHandle,
    WorkflowBuilder,
    WorkflowValidationError,
    WorkflowValue,
)
from tests.service.workflows.driver import (
    SimulatedRunner,
    body_fn,
    build,
    epoch_of,
    exit_now,
    first_case,
    yes,
)

LOOP_ID = activation_id("L", 1)


def _last_change(runner: SimulatedRunner, activation: str) -> NodeStateChanged:
    return [change for change in runner.of(NodeStateChanged) if change.ref.activation_id == activation][-1]


def _fast_exit_slow_side(wf: WorkflowBuilder) -> None:
    def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        entry = scope.python("e", body_fn)
        exit_ = scope.python("x", body_fn)
        side = scope.python("side", body_fn)
        scope.edge(entry, exit_)
        scope.edge(entry, side)
        return entry, exit_

    start = wf.python("s", body_fn)
    loop = wf.loop("L", body=body, until=yes, max_iterations=3)
    out = wf.python("out", body_fn)
    wf.start(start)
    wf.chain(start, loop, out)
    wf.output(out)


def test_iteration_ends_only_after_the_slow_side_branch_settles() -> None:
    runner = SimulatedRunner(build(_fast_exit_slow_side))
    runner.start()
    runner.step()  # s
    runner.resolve(runner.pending("e"))
    runner.resolve(runner.pending("x"))  # the exit value exists, but the side branch is still running

    assert runner.of(EvaluateLoopUntil) == []
    assert runner.state(activation_id("x", 1)) is ActivationState.COMPLETED

    runner.resolve(runner.pending("side"))
    (until,) = runner.of(EvaluateLoopUntil)
    assert until == EvaluateLoopUntil(AttemptRef("run", "L", LOOP_ID, 1), 1, WorkflowValue("x#1"))

    finished = runner.run()
    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    assert runner.of(LoopIteration) == [LoopIteration(until.ref, 1, LoopVerdict.EXIT, WorkflowValue("x#1"))]
    (out,) = runner.activates("out")
    assert out.value == WorkflowValue("x#1")
    assert runner.states(LOOP_ID) == [ActivationState.RUNNING, ActivationState.COMPLETED]


def _closed_exit(wf: WorkflowBuilder) -> None:
    def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        entry = scope.python("e", body_fn)
        exit_ = scope.python("x", body_fn)
        scope.edge(entry, exit_, when=yes)
        return entry, exit_

    loop = wf.loop("L", body=body, until=yes, max_iterations=3)
    wf.start(loop)
    wf.output(loop)


def _closed(work: EvaluateOutgoing) -> dict[str, bool]:
    return dict.fromkeys(work.edge_ids, False)


def test_closed_exit_warns_at_build_time() -> None:
    wf = WorkflowBuilder("closed")
    _closed_exit(wf)
    assert [warning["code"] for warning in wf.build().manifest()["warnings"]] == ["loop_exit_all_conditional"]


def test_closed_exit_fails_the_loop_activation_with_loop_no_value() -> None:
    runner = SimulatedRunner(build(_closed_exit), outgoing=_closed)
    runner.start()

    assert runner.run() is None
    assert runner.state(activation_id("x", 1)) is ActivationState.SKIPPED
    change = _last_change(runner, LOOP_ID)
    assert change.state is ActivationState.AWAITING_RETRY
    assert change.phase is AttemptPhase.BODY
    assert change.error is not None and change.error.error_class is ErrorClass.LOOP_NO_VALUE
    assert runner.of(EvaluateLoopUntil) == []


def test_closed_exit_is_node_failed_when_headless() -> None:
    runner = SimulatedRunner(build(_closed_exit), outgoing=_closed, mode=RunMode.HEADLESS)
    runner.start()
    finished = runner.run()

    assert finished is not None
    assert finished.outcome is RunOutcome.NODE_FAILED
    assert finished.node_id == "L"
    assert finished.error is not None and finished.error.error_class is ErrorClass.LOOP_NO_VALUE
    assert runner.state(LOOP_ID) is ActivationState.FAILED


def test_manual_retry_after_loop_no_value_restarts_the_same_iteration() -> None:
    runner = SimulatedRunner(build(_closed_exit), outgoing=_closed)
    runner.start()
    runner.run()

    out = runner.retry(LOOP_ID, "req-1", 1)
    ref2 = AttemptRef("run", "L", LOOP_ID, 2)
    assert out[:2] == (PersistRetryKey("req-1", ref2), NodeStateChanged(ref2, ActivationState.RUNNING))
    (entry2,) = [work for work in runner.activates("e") if epoch_of(work.ref.activation_id) == 2]
    assert entry2.value == WorkflowValue("input")
    assert runner.state(activation_id("x", 1)) is ActivationState.SKIPPED

    runner.outgoing = lambda work: dict.fromkeys(work.edge_ids, True)
    finished = runner.run()
    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    assert finished.outputs == (RunOutput("L", LOOP_ID, WorkflowValue("x#1")),)
    assert [item.iteration for item in runner.of(LoopIteration)] == [1]


def test_nested_loops_are_rejected_by_the_builder() -> None:
    wf = WorkflowBuilder("nested")

    def inner(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        node = scope.python("inner_node", body_fn)
        return node, node

    def outer(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        wf.loop("inner", body=inner, until=yes, max_iterations=1)
        node = scope.python("outer_node", body_fn)
        return node, node

    with pytest.raises(WorkflowValidationError, match="Nested"):
        wf.loop("outer", body=outer, until=yes, max_iterations=1)


def test_nested_loop_manifests_are_rejected_by_the_graph_reader() -> None:
    wf = WorkflowBuilder("nested")
    _closed_exit(wf)
    manifest = wf.build().manifest()
    loop_node = next(node for node in manifest["nodes"] if node["id"] == "L")
    loop_node["parent_loop"] = "L"
    with pytest.raises(ManifestError, match="nested"):
        GraphSpec.from_manifest(manifest)


def _simple_loop(
    wf: WorkflowBuilder, *, max_iterations: int = 3, on_exhausted: str = "continue", switch: bool = False
) -> None:
    def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        entry = scope.python("e", body_fn)
        exit_ = scope.python("x", body_fn)
        scope.edge(entry, exit_)
        return entry, exit_

    loop = wf.loop("L", body=body, until=yes, max_iterations=max_iterations, on_exhausted=on_exhausted)
    wf.start(loop)
    if switch:
        a = wf.python("a", body_fn)
        b = wf.python("b", body_fn)
        wf.switch(loop, cases=[(yes, a)], default=b)
        wf.output(a)
        wf.output(b)
    else:
        wf.output(loop)


def test_until_failure_retries_only_the_until_evaluation() -> None:
    runner = SimulatedRunner(build(_simple_loop), until=lambda work: "until boom")
    runner.start()
    assert runner.run() is None

    change = _last_change(runner, LOOP_ID)
    assert change.state is ActivationState.AWAITING_RETRY
    assert change.phase is AttemptPhase.UNTIL
    assert change.error is not None
    assert change.error.error_class is ErrorClass.EVALUATION_ERROR
    assert change.error.message == "until boom"
    activations_before = runner.scheduler.activation_ids()

    runner.until = exit_now
    ref2 = AttemptRef("run", "L", LOOP_ID, 2)
    assert runner.retry(LOOP_ID, "req", 1) == (
        PersistRetryKey("req", ref2),
        NodeStateChanged(ref2, ActivationState.RUNNING),
        LoopActivated(ref2, WorkflowValue("input")),
        EvaluateLoopUntil(ref2, 1, WorkflowValue("x#1")),
    )
    finished = runner.run()
    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    assert runner.scheduler.activation_ids() == activations_before
    assert finished.outputs == (RunOutput("L", LOOP_ID, WorkflowValue("x#1")),)


def test_outer_predicate_failure_is_the_loop_activations_evaluation_error() -> None:
    runner = SimulatedRunner(build(lambda wf: _simple_loop(wf, switch=True)), outgoing=lambda work: "pred boom")
    runner.start()
    assert runner.run() is None

    ref1 = AttemptRef("run", "L", LOOP_ID, 1)
    assert LoopIteration(ref1, 1, LoopVerdict.EXIT, WorkflowValue("x#1")) in runner.decisions
    change = _last_change(runner, LOOP_ID)
    assert change.state is ActivationState.AWAITING_RETRY
    assert change.phase is AttemptPhase.OUTGOING
    assert change.error is not None and change.error.error_class is ErrorClass.EVALUATION_ERROR


def test_manual_retry_after_outer_predicate_failure_reevaluates_with_the_cached_exit_value() -> None:
    runner = SimulatedRunner(build(lambda wf: _simple_loop(wf, switch=True)), outgoing=lambda work: "pred boom")
    runner.start()
    runner.run()
    body_activations = len(runner.of(Activate))

    runner.outgoing = first_case
    ref2 = AttemptRef("run", "L", LOOP_ID, 2)
    assert runner.retry(LOOP_ID, "req", 1) == (
        PersistRetryKey("req", ref2),
        NodeStateChanged(ref2, ActivationState.RUNNING),
        LoopActivated(ref2, WorkflowValue("input")),
        EvaluateOutgoing(ref2, WorkflowValue("x#1"), ("L->a", "L->b")),
    )
    finished = runner.run()
    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    assert len(runner.of(Activate)) == body_activations + 1  # only the taken switch branch ran
    (taken,) = runner.activates("a")
    assert taken.value == WorkflowValue("x#1")
    assert [output.node_id for output in finished.outputs] == ["a"]


def test_next_iteration_entry_receives_the_previous_exit_value() -> None:
    runner = SimulatedRunner(build(_simple_loop), until=lambda work: work.iteration >= 2)
    runner.start()
    finished = runner.run()

    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    entries = runner.activates("e")
    assert [work.value for work in entries] == [WorkflowValue("input"), WorkflowValue("x#1")]
    assert [(item.iteration, item.verdict) for item in runner.of(LoopIteration)] == [
        (1, LoopVerdict.CONTINUE),
        (2, LoopVerdict.EXIT),
    ]


def test_exhausted_loop_continues_downstream_with_the_last_exit_value() -> None:
    runner = SimulatedRunner(build(lambda wf: _simple_loop(wf, max_iterations=2)), until=lambda work: False)
    runner.start()
    finished = runner.run()

    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    assert [(item.iteration, item.verdict) for item in runner.of(LoopIteration)] == [
        (1, LoopVerdict.CONTINUE),
        (2, LoopVerdict.EXHAUSTED),
    ]
    assert finished.outputs == (RunOutput("L", LOOP_ID, WorkflowValue("x#1")),)
    assert len(runner.activates("x")) == 2


def test_exhausted_loop_with_fail_finishes_the_run_as_loop_exhausted() -> None:
    runner = SimulatedRunner(
        build(lambda wf: _simple_loop(wf, max_iterations=1, on_exhausted="fail")), until=lambda work: False
    )
    runner.start()
    finished = runner.run()

    assert finished is not None
    assert finished.outcome is RunOutcome.LOOP_EXHAUSTED
    assert finished.node_id == "L"
    assert finished.outputs == ()
    assert runner.state(LOOP_ID) is ActivationState.CANCELLED
    assert runner.retry(LOOP_ID, "late", 1) == (RetryRejected("L", LOOP_ID, "late", RetryRejection.RUN_FINISHED),)


@pytest.mark.parametrize("maximum", [1, 2])
def test_repeated_body_retries_do_not_consume_iterations(maximum: int) -> None:
    builder = WorkflowBuilder("retry iterations")
    _closed_exit(builder)
    manifest = builder.build().manifest()
    next(node for node in manifest["nodes"] if node["id"] == "L")["loop"]["max_iterations"] = maximum
    runner = SimulatedRunner(GraphSpec.from_manifest(manifest), outgoing=_closed, until=lambda work: False)
    runner.start("original input")
    for attempt in (1, 2):
        assert runner.run() is None
        runner.retry(LOOP_ID, f"retry-{attempt}", attempt)
    runner.outgoing = lambda work: dict.fromkeys(work.edge_ids, True)
    finished = runner.run()
    assert finished is not None
    assert [work.iteration for work in runner.of(EvaluateLoopUntil)] == list(range(1, maximum + 1))
    assert [entry.value.text for entry in runner.activates("e")[:3]] == ["original input"] * 3
    changes = [change for change in runner.of(NodeStateChanged) if change.ref.node_id in {"e", "x"}]
    assert all(change.iteration == max(1, epoch_of(change.ref.activation_id) - 2) for change in changes)
    assert runner.scheduler._live_count == 0
    assert runner.scheduler._unsettled_top_level == 0
