# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Known counterexamples of naive DAG execution: skip deadlock, OR-merge racing, closed cycles, cross-epoch joins."""

from __future__ import annotations

import pytest

from chrys.service.workflows.graph import GraphSpec, ManifestError
from chrys.service.workflows.outcomes import RunOutcome
from chrys.service.workflows.scheduler import (
    Activate,
    ActivationState,
    EvaluateLoopUntil,
    NodeStateChanged,
    activation_id,
)
from chrys.service.workflows.sdk import (
    BuilderScope,
    NodeHandle,
    SourceValue,
    WorkflowBuilder,
    WorkflowValidationError,
    WorkflowValue,
)
from chrys.service.workflows.values import default_combine
from tests.service.workflows.driver import (
    SimulatedRunner,
    body_fn,
    build,
    epoch_of,
    every_order,
    first_case,
    yes,
)

_SETTLED = {ActivationState.COMPLETED, ActivationState.SKIPPED}
_BODY = {"e", "b1", "b2", "join:s", "s"}


def _switch_merge(wf: WorkflowBuilder) -> None:
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    c = wf.python("c", body_fn)
    d = wf.python("d", body_fn)
    wf.start(a)
    wf.switch(a, cases=[(yes, b)], default=c)
    wf.edge(b, d)
    wf.edge(c, d)
    wf.output(d)


def test_switch_then_merge_activates_the_merge_once_on_the_taken_branch() -> None:
    runner = SimulatedRunner(build(_switch_merge), outgoing=first_case)
    runner.start()
    finished = runner.run()

    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    assert runner.state(activation_id("c", 1)) is ActivationState.SKIPPED
    (merge,) = runner.activates("d")
    assert [source.node_id for source in merge.sources] == ["b"]
    assert merge.value == WorkflowValue("b#1")
    assert [output.node_id for output in finished.outputs] == ["d"]


def test_switch_takes_the_default_when_no_case_matches() -> None:
    runner = SimulatedRunner(build(_switch_merge), outgoing=lambda work: dict.fromkeys(work.edge_ids, False))
    runner.start()
    finished = runner.run()

    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    assert runner.state(activation_id("b", 1)) is ActivationState.SKIPPED
    (merge,) = runner.activates("d")
    assert [source.node_id for source in merge.sources] == ["c"]


def _skipped_chain(wf: WorkflowBuilder) -> None:
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    c = wf.python("c", body_fn)
    d = wf.python("d", body_fn)
    e = wf.python("e", body_fn)
    f = wf.python("f", body_fn)
    wf.start(a)
    wf.switch(a, cases=[(yes, b)], default=c)
    wf.chain(b, d, e)
    wf.edge(c, f)
    wf.output(e)
    wf.output(f)


def test_closed_tokens_propagate_through_skipped_chains_to_the_outputs() -> None:
    runner = SimulatedRunner(build(_skipped_chain), outgoing=lambda work: dict.fromkeys(work.edge_ids, False))
    runner.start()
    finished = runner.run()

    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    assert [runner.state(activation_id(node, 1)) for node in ("b", "d", "e")] == [ActivationState.SKIPPED] * 3
    assert [output.node_id for output in finished.outputs] == ["f"]


def _long_skipped_chain(wf: WorkflowBuilder) -> None:
    nodes = [wf.python(f"n{index}", body_fn) for index in range(400)]
    wf.start(nodes[0])
    wf.edge(nodes[0], nodes[1], when=yes)
    wf.chain(*nodes[1:])
    wf.output(nodes[-1])


def test_closed_tokens_walk_a_long_chain_without_recursion() -> None:
    runner = SimulatedRunner(build(_long_skipped_chain), outgoing=lambda work: dict.fromkeys(work.edge_ids, False))
    runner.start()
    finished = runner.run()

    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    assert runner.state(activation_id("n399", 1)) is ActivationState.SKIPPED
    assert not finished.outputs


def _closed_vs_slow(wf: WorkflowBuilder) -> None:
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    c = wf.python("c", body_fn)
    d = wf.python("d", body_fn)
    wf.start(a)
    wf.edge(a, b, when=yes)
    wf.edge(a, c)
    wf.edge(b, d)
    wf.edge(c, d)
    wf.output(d)


def test_closed_edge_never_closes_a_node_with_an_unresolved_value_edge() -> None:
    runner = SimulatedRunner(build(_closed_vs_slow), outgoing=lambda work: dict.fromkeys(work.edge_ids, False))
    runner.start()
    runner.step()  # a body
    runner.step()  # a outgoing: b closed and skipped while c is still running

    assert runner.state(activation_id("b", 1)) is ActivationState.SKIPPED
    assert runner.scheduler.snapshot(activation_id("d", 1)) is None

    finished = runner.run()
    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    (merge,) = runner.activates("d")
    assert [source.node_id for source in merge.sources] == ["c"]


def _fan_in(wf: WorkflowBuilder) -> None:
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    c = wf.python("c", body_fn)
    d = wf.python("d", body_fn)
    wf.start(a)
    wf.edge(a, b)
    wf.edge(a, c)
    wf.edge(b, d)
    wf.edge(c, d)
    wf.output(d)


def test_fan_in_waits_for_both_branches_in_every_arrival_order() -> None:
    orders = 0
    for _order, runner in every_order(lambda: SimulatedRunner(build(_fan_in))):
        orders += 1
        assert runner.finished is not None and runner.finished.outcome is RunOutcome.COMPLETED
        (merge,) = runner.activates("d")
        assert [source.node_id for source in merge.sources] == ["b", "c"]
        assert merge.value == default_combine(merge.sources)
        branch_done = [
            runner.index_of(change)
            for change in runner.of(NodeStateChanged)
            if change.state is ActivationState.COMPLETED and change.ref.node_id in {"b", "c"}
        ]
        assert max(branch_done) < runner.index_of(merge)
    assert orders == 2


def test_self_edges_and_cycles_are_rejected_by_the_builder() -> None:
    wf = WorkflowBuilder("cycle")
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    with pytest.raises(WorkflowValidationError):
        wf.edge(a, a)
    wf.start(a)
    wf.edge(a, b)
    wf.edge(b, a)
    wf.output(b)
    with pytest.raises(WorkflowValidationError, match=r"wf\.loop"):
        wf.build()


def test_manifest_cycle_is_rejected_before_scheduling() -> None:
    wf = WorkflowBuilder("dag")
    a = wf.python("a", body_fn)
    b = wf.python("b", body_fn)
    c = wf.python("c", body_fn)
    wf.start(a)
    wf.chain(a, b, c)
    wf.output(c)
    manifest = wf.build().manifest()
    manifest["edges"].append({"id": "c->b", "src": "c", "dst": "b", "conditional": False, "switch": None})
    with pytest.raises(ManifestError, match="cycle"):
        GraphSpec.from_manifest(manifest)


def test_manifest_loop_body_must_list_every_child() -> None:
    """The iteration barrier trusts loop.body: a side branch left out of it would never hold the loop back."""
    wf = WorkflowBuilder("loop")
    _fan_in_loop(wf)
    manifest = wf.build().manifest()
    (loop,) = [node for node in manifest["nodes"] if node["id"] == "L"]
    loop["loop"]["body"].remove("b2")
    with pytest.raises(ManifestError, match="omits its child 'b2'"):
        GraphSpec.from_manifest(manifest)


def _combine(sources: list[SourceValue]) -> WorkflowValue:
    return WorkflowValue("merged")


def _fan_in_loop(wf: WorkflowBuilder, *, combine: bool = False) -> None:
    def body(scope: BuilderScope) -> tuple[NodeHandle, NodeHandle]:
        entry = scope.python("e", body_fn)
        b1 = scope.python("b1", body_fn)
        b2 = scope.python("b2", body_fn)
        summary = scope.python("s", body_fn)
        scope.edge(entry, b1)
        scope.edge(entry, b2)
        scope.join([b1, b2], summary, combine=_combine if combine else None)
        return entry, summary

    start = wf.python("start", body_fn)
    loop = wf.loop("L", body=body, until=yes, max_iterations=2)
    wf.start(start)
    wf.edge(start, loop)
    wf.output(loop)


def _exit_after_two(work: EvaluateLoopUntil) -> bool:
    return work.iteration >= 2


def test_join_never_pairs_values_across_loop_epochs_in_any_arrival_order() -> None:
    orders = 0
    for _order, runner in every_order(lambda: SimulatedRunner(build(_fan_in_loop), until=_exit_after_two)):
        orders += 1
        assert runner.finished is not None and runner.finished.outcome is RunOutcome.COMPLETED
        joins = runner.activates("join:s")
        assert [epoch_of(join.ref.activation_id) for join in joins] == [1, 2]
        for join in joins:
            assert join.kind == "join"
            assert {epoch_of(source.activation_id) for source in join.sources} == {epoch_of(join.ref.activation_id)}
        (entry2,) = [work for work in runner.activates("e") if epoch_of(work.ref.activation_id) == 2]
        first_epoch_terminal = [
            runner.index_of(change)
            for change in runner.of(NodeStateChanged)
            if change.state in _SETTLED and epoch_of(change.ref.activation_id) == 1 and change.ref.node_id in _BODY
        ]
        assert len(first_epoch_terminal) == 5
        assert max(first_epoch_terminal) < runner.index_of(entry2)
    assert orders == 4


def test_join_with_a_user_combine_hands_the_sources_to_the_runner() -> None:
    runner = SimulatedRunner(build(lambda wf: _fan_in_loop(wf, combine=True)))
    runner.start()
    finished = runner.run()

    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    (join,) = runner.activates("join:s")
    assert join.value is None
    assert [source.node_id for source in join.sources] == ["b1", "b2"]
    (summary,) = runner.activates("s")
    assert summary.value == WorkflowValue("join:s#1")


def test_skipped_join_source_is_absent_from_the_combine_input() -> None:
    def configure(wf: WorkflowBuilder) -> None:
        a = wf.python("a", body_fn)
        b = wf.python("b", body_fn)
        c = wf.python("c", body_fn)
        d = wf.python("d", body_fn)
        wf.start(a)
        wf.edge(a, b, when=yes)
        wf.edge(a, c)
        wf.join([b, c], d)
        wf.output(d)

    runner = SimulatedRunner(build(configure), outgoing=lambda work: dict.fromkeys(work.edge_ids, False))
    runner.start()
    finished = runner.run()

    assert finished is not None and finished.outcome is RunOutcome.COMPLETED
    (join,) = runner.activates("join:d")
    assert [source.node_id for source in join.sources] == ["c"]
    assert join.value == WorkflowValue("c#1")
    assert isinstance(runner.activates("d")[0], Activate)
