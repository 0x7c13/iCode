# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Random-graph properties of the scheduler, plus its fail-closed invariants."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from chrys.service.workflows.graph import EdgeSpec, GraphSpec, NodeSpec, RetrySpec
from chrys.service.workflows.outcomes import RunOutcome
from chrys.service.workflows.scheduler import (
    Activate,
    ActivationState,
    AttemptRef,
    ErrorClass,
    EvaluateLoopUntil,
    EvaluateOutgoing,
    FailureReport,
    NodeStateChanged,
    RunFinished,
    RunMode,
    SchedulerInvariantError,
    WorkflowScheduler,
    activation_id,
)
from chrys.service.workflows.sdk import BuilderScope, NodeHandle, Retry, WorkflowBuilder, WorkflowValue
from chrys.service.workflows.values import default_combine
from tests.service.workflows.driver import (
    SimulatedRunner,
    body_fn,
    build,
    epoch_of,
    every_order,
    graph_of,
    yes,
)

_SETTLED = {ActivationState.COMPLETED, ActivationState.SKIPPED}


# ---------------------------------------------------------------------------
# Random workflows


def _random_scope(scope: BuilderScope, rng: random.Random, prefix: str, size: int) -> list[NodeHandle]:
    nodes: list[NodeHandle] = []
    for index in range(size):
        name = f"{prefix}{index}"
        if rng.random() < 0.6:
            nodes.append(scope.python(name, body_fn, retry=Retry(max_attempts=rng.randint(1, 3))))
        else:
            nodes.append(scope.agent(name, profile="A"))
    for target in range(1, size):
        count = min(target, rng.choice((1, 1, 2)))
        for source in sorted(rng.sample(range(target), count)):
            if rng.random() < 0.3:
                scope.edge(nodes[source], nodes[target], when=yes)
            else:
                scope.edge(nodes[source], nodes[target])
    return nodes


def random_graph(seed: int, *, small: bool = False) -> GraphSpec:
    rng = random.Random(seed)
    wf = WorkflowBuilder(f"random-{seed}")
    top = _random_scope(wf, rng, "t", rng.randint(1, 2 if small else 5))
    wf.start(top[0])
    candidates = list(top)
    tail = top[-1]
    if rng.random() < 0.5:
        case = wf.python("c1", body_fn)
        default = wf.python("c2", body_fn)
        merge = wf.python("m", body_fn)
        wf.switch(tail, cases=[(yes, case)], default=default)
        wf.edge(case, merge)
        wf.edge(default, merge)
        candidates += [case, default, merge]
        tail = merge
    if rng.random() < 0.6:
        size = rng.randint(1, 2 if small else 4)

        def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
            nodes = _random_scope(scope, rng, "b", size)
            return nodes[0], nodes[rng.randrange(size)]

        loop = wf.loop("L", body=body, until=yes, max_iterations=rng.randint(1, 3))
        after = wf.python("after", body_fn)
        wf.edge(tail, loop)
        wf.edge(loop, after)
        candidates += [loop, after]
    for handle in rng.sample(candidates, rng.randint(1, len(candidates))):
        wf.output(handle)
    return graph_of(wf.build())


def _into_loop_exit(graph: GraphSpec, edge_id: str) -> bool:
    dst = graph.nodes[graph.edges[edge_id].dst]
    if dst.parent_loop is None:
        return False
    loop = graph.nodes[dst.parent_loop].loop
    return loop is not None and loop.exit == dst.node_id


def scripted_runner(graph: GraphSpec, seed: int) -> SimulatedRunner:
    """Policies keyed by attempt identity, so results never depend on arrival order."""

    def key_rng(*parts: object) -> random.Random:
        return random.Random(f"{seed}:" + ":".join(str(part) for part in parts))

    def body(work: Activate) -> WorkflowValue | FailureReport:
        rng = key_rng("body", work.ref.activation_id, work.ref.attempt)
        if work.ref.attempt < 3 and rng.random() < 0.25:
            if work.kind == "agent":
                return FailureReport(ErrorClass.AGENT_TRANSIENT, "429", backend_approved=rng.random() < 0.5)
            return FailureReport(ErrorClass.PYTHON_EXCEPTION, "scripted")
        return WorkflowValue(text=f"{work.ref.activation_id}/{work.ref.attempt}")

    def outgoing(work: EvaluateOutgoing) -> dict[str, bool]:
        rng = key_rng("edges", work.ref.activation_id, work.ref.attempt)
        return {edge_id: _into_loop_exit(graph, edge_id) or rng.random() < 0.6 for edge_id in work.edge_ids}

    def until(work: EvaluateLoopUntil) -> bool:
        return key_rng("until", work.ref.activation_id, work.iteration).random() < 0.5

    return SimulatedRunner(graph, body=body, outgoing=outgoing, until=until)


def drive_to_completion(runner: SimulatedRunner, rng: random.Random | None) -> RunFinished:
    """Run, and play the human: retry every awaiting activation until the run finishes."""
    runner.start()
    for _round in range(64):
        finished = runner.run(rng)
        if finished is not None:
            return finished
        awaiting = [
            activation
            for activation in runner.scheduler.activation_ids()
            if runner.state(activation) is ActivationState.AWAITING_RETRY
        ]
        assert awaiting, "quiescent without a finished run or an awaiting activation"
        for activation in awaiting:
            snapshot = runner.scheduler.snapshot(activation)
            assert snapshot is not None
            runner.retry(activation, f"req:{activation}:{snapshot.attempt}", snapshot.attempt)
    raise AssertionError("run did not converge")


def _summary(runner: SimulatedRunner) -> dict[str, tuple[ActivationState, int]]:
    result: dict[str, tuple[ActivationState, int]] = {}
    for activation in runner.scheduler.activation_ids():
        snapshot = runner.scheduler.snapshot(activation)
        assert snapshot is not None
        result[activation] = (snapshot.state, snapshot.attempt)
    return result


def check_resolution(runner: SimulatedRunner) -> None:
    graph = runner.graph
    first: dict[str, ActivationState] = {}
    for change in runner.of(NodeStateChanged):
        first.setdefault(change.ref.activation_id, change.state)
    assert set(first.values()) <= {ActivationState.RUNNING, ActivationState.SKIPPED}

    expected = 0
    for node_id in graph.top_level:
        assert activation_id(node_id, 1) in first
        expected += 1
    for node in graph.nodes.values():
        if node.loop is None:
            continue
        snapshot = runner.scheduler.snapshot(activation_id(node.node_id, 1))
        assert snapshot is not None
        epochs = {epoch_of(work.ref.activation_id) for work in runner.activates(node.loop.entry)}
        for epoch in epochs:
            for member in node.loop.body:
                assert activation_id(member, epoch) in first
            expected += len(node.loop.body)
        _check_barrier(runner, node.node_id, node.loop.body, len(epochs))
    assert len(first) == expected

    for work in runner.of(Activate):
        epoch = epoch_of(work.ref.activation_id)
        assert all(epoch_of(source.activation_id) == epoch for source in work.sources)
        if work.sources:
            expected_value = work.sources[0].value if len(work.sources) == 1 else default_combine(work.sources)
            assert work.value == expected_value

    finished = runner.finished
    assert finished is not None
    completed = [node for node in graph.outputs if runner.state(activation_id(node, 1)) is ActivationState.COMPLETED]
    assert [output.node_id for output in finished.outputs] == completed


def _check_barrier(runner: SimulatedRunner, loop_id: str, body: tuple[str, ...], iterations: int) -> None:
    for iteration in range(1, iterations):
        settled = [
            runner.index_of(change)
            for change in runner.of(NodeStateChanged)
            if change.ref.node_id in body
            and epoch_of(change.ref.activation_id) == iteration
            and change.state in _SETTLED
        ]
        assert len(settled) == len(body)
        loop = runner.graph.nodes[loop_id].loop
        assert loop is not None
        (entry_next,) = [
            work
            for work in runner.activates(loop.entry)
            if epoch_of(work.ref.activation_id) == iteration + 1 and work.ref.attempt == 1
        ]
        assert max(settled) < runner.index_of(entry_next)


@pytest.mark.parametrize("seed", range(150))
def test_random_graphs_resolve_every_node_exactly_once_per_epoch(seed: int) -> None:
    graph = random_graph(seed)
    summaries = []
    for order_seed in range(3):
        runner = scripted_runner(graph, seed)
        finished = drive_to_completion(runner, random.Random(order_seed))
        assert finished.outcome is RunOutcome.COMPLETED
        check_resolution(runner)
        summaries.append(_summary(runner))
    assert all(summary == summaries[0] for summary in summaries)


@pytest.mark.parametrize("seed", range(40))
def test_small_graphs_agree_across_every_arrival_order(seed: int) -> None:
    graph = random_graph(seed, small=True)
    summaries = []
    for _order, runner in every_order(lambda: scripted_runner(graph, seed)):
        if runner.finished is None:
            # An awaiting_retry stall is a valid quiescent point; finish it the FIFO way.
            drive_to_completion_from(runner)
        assert runner.finished is not None and runner.finished.outcome is RunOutcome.COMPLETED
        check_resolution(runner)
        summaries.append(_summary(runner))
    assert summaries
    assert all(summary == summaries[0] for summary in summaries)


def drive_to_completion_from(runner: SimulatedRunner) -> None:
    for _round in range(64):
        if runner.run() is not None:
            return
        for activation in runner.scheduler.activation_ids():
            if runner.state(activation) is ActivationState.AWAITING_RETRY:
                snapshot = runner.scheduler.snapshot(activation)
                assert snapshot is not None
                runner.retry(activation, f"req:{activation}:{snapshot.attempt}", snapshot.attempt)
    raise AssertionError("run did not converge")


def test_same_seed_and_order_replay_identically() -> None:
    graph = random_graph(7)
    first = scripted_runner(graph, 7)
    drive_to_completion(first, random.Random(3))
    second = scripted_runner(graph, 7)
    drive_to_completion(second, random.Random(3))
    assert first.decisions == second.decisions


# ---------------------------------------------------------------------------
# Fail-closed invariants


def _single_graph() -> GraphSpec:
    def configure(wf: WorkflowBuilder) -> None:
        node = wf.python("a", body_fn)
        wf.start(node)
        wf.output(node)

    return build(configure)


def test_start_twice_is_an_invariant_error() -> None:
    runner = SimulatedRunner(_single_graph())
    runner.start()
    with pytest.raises(SchedulerInvariantError, match="already started"):
        runner.start()


def test_missing_loop_specification_is_a_scheduler_invariant_error() -> None:
    graph = _single_graph()
    invalid_node = replace(graph.nodes[graph.start], kind="loop", loop=None)
    runner = SimulatedRunner(replace(graph, nodes={graph.start: invalid_node}))
    with pytest.raises(SchedulerInvariantError, match="has no loop specification"):
        runner.start()


def test_outgoing_decisions_must_match_the_conditional_edges() -> None:
    def configure(wf: WorkflowBuilder) -> None:
        a = wf.python("a", body_fn)
        b = wf.python("b", body_fn)
        c = wf.python("c", body_fn)
        wf.start(a)
        wf.edge(a, b, when=yes)
        wf.edge(a, c, when=yes)
        wf.output(b)
        wf.output(c)

    runner = SimulatedRunner(build(configure))
    runner.start()
    runner.step()
    (evaluation,) = runner.of(EvaluateOutgoing)
    with pytest.raises(SchedulerInvariantError, match="conditional edges"):
        runner.scheduler.outgoing_evaluated(evaluation.ref, {"a->b": True})
    with pytest.raises(SchedulerInvariantError, match="phase"):
        runner.scheduler.activation_completed(evaluation.ref, WorkflowValue("again"))


def test_loop_activations_never_accept_body_results_or_foreign_iterations() -> None:
    def configure(wf: WorkflowBuilder) -> None:
        def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
            node = scope.python("e", body_fn)
            return node, node

        loop = wf.loop("L", body=body, until=yes, max_iterations=2)
        wf.start(loop)
        wf.output(loop)

    runner = SimulatedRunner(build(configure))
    runner.start()
    loop_ref = AttemptRef("run", "L", activation_id("L", 1), 1)
    with pytest.raises(SchedulerInvariantError, match="no body"):
        runner.scheduler.activation_completed(loop_ref, WorkflowValue("x"))
    runner.step()  # e completes: the barrier asks for the until verdict
    with pytest.raises(SchedulerInvariantError, match="phase"):
        runner.scheduler.activation_completed(loop_ref, WorkflowValue("x"))
    with pytest.raises(SchedulerInvariantError, match="iteration"):
        runner.scheduler.loop_until_evaluated(loop_ref, 2, True)


def _node(node_id: str, *, in_edges: tuple[str, ...] = (), out_edges: tuple[str, ...] = ()) -> NodeSpec:
    return NodeSpec(
        node_id=node_id,
        kind="python",
        parent_loop=None,
        retry=RetrySpec(1, 0.0),
        timeout=None,
        in_edges=in_edges,
        out_edges=out_edges,
        has_combine=False,
        loop=None,
    )


def _edge(edge_id: str, src: str, dst: str) -> EdgeSpec:
    return EdgeSpec(edge_id, src, dst, conditional=False, switch_group=None, switch_position=None, switch_default=False)


def test_a_duplicate_token_is_an_invariant_error() -> None:
    # Hand-built (unvalidated) graph: node a lists its only edge twice.
    graph = GraphSpec(
        title="dup",
        start="a",
        outputs=("b",),
        nodes={"a": _node("a", out_edges=("a->b", "a->b")), "b": _node("b", in_edges=("a->b",))},
        node_order=("a", "b"),
        edges={"a->b": _edge("a->b", "a", "b")},
    )
    scheduler = WorkflowScheduler(graph, run_id="run", mode=RunMode.INTERACTIVE)
    scheduler.start(WorkflowValue("in"))
    with pytest.raises(SchedulerInvariantError, match="already carries a token"):
        scheduler.activation_completed(AttemptRef("run", "a", activation_id("a", 1), 1), WorkflowValue("v"))


def test_a_quiescent_unfinished_run_is_an_invariant_error() -> None:
    # Hand-built (unvalidated) graph: b is top-level but nothing ever resolves it.
    graph = GraphSpec(
        title="stuck",
        start="a",
        outputs=("b",),
        nodes={"a": _node("a"), "b": _node("b")},
        node_order=("a", "b"),
        edges={},
    )
    scheduler = WorkflowScheduler(graph, run_id="run", mode=RunMode.INTERACTIVE)
    scheduler.start(WorkflowValue("in"))
    with pytest.raises(SchedulerInvariantError, match="quiescent"):
        scheduler.activation_completed(AttemptRef("run", "a", activation_id("a", 1), 1), WorkflowValue("v"))
