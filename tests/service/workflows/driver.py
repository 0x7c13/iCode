# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Simulated runner for scheduler tests.

The scheduler is pure, so a test "runs" a workflow by executing its
decisions with scripted results in whatever arrival order the test picks:
first-in-first-out, a seeded random order, or every possible order.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping
from typing import TypeVar

from chrys.service.workflows.graph import GraphSpec
from chrys.service.workflows.scheduler import (
    Activate,
    ActivationState,
    CancelActivation,
    Decision,
    EvaluateLoopUntil,
    EvaluateOutgoing,
    FailureReport,
    NodeStateChanged,
    RunFinished,
    RunMode,
    ScheduleRetry,
    WorkflowScheduler,
)
from chrys.service.workflows.sdk import Workflow, WorkflowBuilder, WorkflowValue

Work = Activate | EvaluateOutgoing | EvaluateLoopUntil | ScheduleRetry
BodyPolicy = Callable[[Activate], WorkflowValue | FailureReport]
OutgoingPolicy = Callable[[EvaluateOutgoing], Mapping[str, bool] | str]
UntilPolicy = Callable[[EvaluateLoopUntil], bool | str]

D = TypeVar("D")


def yes(value: WorkflowValue) -> bool:
    return True


def body_fn(text: str) -> str:
    return text


def graph_of(workflow: Workflow) -> GraphSpec:
    return GraphSpec.from_manifest(workflow.manifest())


def build(configure: Callable[[WorkflowBuilder], None], *, title: str = "test") -> GraphSpec:
    wf = WorkflowBuilder(title)
    configure(wf)
    return graph_of(wf.build())


def complete_body(work: Activate) -> WorkflowValue:
    return WorkflowValue(text=f"{work.ref.node_id}#{work.ref.attempt}")


def open_all(work: EvaluateOutgoing) -> dict[str, bool]:
    return dict.fromkeys(work.edge_ids, True)


def first_case(work: EvaluateOutgoing) -> dict[str, bool]:
    return {edge_id: index == 0 for index, edge_id in enumerate(work.edge_ids)}


def exit_now(work: EvaluateLoopUntil) -> bool:
    return True


def epoch_of(activation_id: str) -> int:
    return int(activation_id.rsplit("#", 1)[1])


class SimulatedRunner:
    def __init__(
        self,
        graph: GraphSpec,
        *,
        mode: RunMode = RunMode.INTERACTIVE,
        run_id: str = "run",
        body: BodyPolicy = complete_body,
        outgoing: OutgoingPolicy = open_all,
        until: UntilPolicy = exit_now,
    ) -> None:
        self.graph = graph
        self.scheduler = WorkflowScheduler(graph, run_id=run_id, mode=mode)
        self.body = body
        self.outgoing = outgoing
        self.until = until
        self.decisions: list[Decision] = []
        self.outstanding: list[Work] = []

    @property
    def finished(self) -> RunFinished | None:
        return self.scheduler.finished

    def start(self, text: str = "input") -> None:
        self.absorb(self.scheduler.start(WorkflowValue(text=text)))

    def absorb(self, decisions: tuple[Decision, ...]) -> None:
        for decision in decisions:
            self.decisions.append(decision)
            if isinstance(decision, Activate | EvaluateOutgoing | EvaluateLoopUntil | ScheduleRetry):
                self.outstanding.append(decision)
            elif isinstance(decision, CancelActivation):
                self.outstanding = [work for work in self.outstanding if work.ref != decision.ref]

    def resolve(self, work: Work) -> None:
        """Feed the scripted result of one outstanding item back into the scheduler."""
        self.outstanding.remove(work)
        scheduler = self.scheduler
        if isinstance(work, Activate):
            result = self.body(work)
            if isinstance(result, FailureReport):
                out = scheduler.activation_failed(work.ref, result)
            else:
                out = scheduler.activation_completed(work.ref, result)
        elif isinstance(work, EvaluateOutgoing):
            decided = self.outgoing(work)
            if isinstance(decided, str):
                out = scheduler.outgoing_failed(work.ref, decided)
            else:
                out = scheduler.outgoing_evaluated(work.ref, decided)
        elif isinstance(work, EvaluateLoopUntil):
            verdict = self.until(work)
            if isinstance(verdict, str):
                out = scheduler.loop_until_failed(work.ref, work.iteration, verdict)
            else:
                out = scheduler.loop_until_evaluated(work.ref, work.iteration, verdict)
        else:
            out = scheduler.backoff_elapsed(work.ref)
        self.absorb(out)

    def step(self, index: int = 0) -> Work:
        work = self.outstanding[index]
        self.resolve(work)
        return work

    def run(self, rng: random.Random | None = None) -> RunFinished | None:
        """Drain outstanding work FIFO (or in seeded random order) until quiescent."""
        while self.outstanding and self.finished is None:
            self.step(rng.randrange(len(self.outstanding)) if rng is not None else 0)
        return self.finished

    def retry(self, activation_id: str, request_id: str, expected_failed_attempt: int) -> tuple[Decision, ...]:
        node_id = activation_id.rsplit("@", 1)[0]
        out = self.scheduler.manual_retry(node_id, activation_id, request_id, expected_failed_attempt)
        self.absorb(out)
        return out

    # -- observation helpers -------------------------------------------------

    def pending(self, node_id: str, *, epoch: int = 1) -> Work:
        for work in self.outstanding:
            if work.ref.node_id == node_id and epoch_of(work.ref.activation_id) == epoch:
                return work
        raise AssertionError(f"no outstanding work for {node_id}@iter#{epoch}: {self.outstanding!r}")

    def of(self, kind: type[D]) -> list[D]:
        return [decision for decision in self.decisions if isinstance(decision, kind)]

    def activates(self, node_id: str) -> list[Activate]:
        return [work for work in self.of(Activate) if work.ref.node_id == node_id]

    def states(self, activation_id: str) -> list[ActivationState]:
        return [decision.state for decision in self.of(NodeStateChanged) if decision.ref.activation_id == activation_id]

    def state(self, activation_id: str) -> ActivationState:
        snapshot = self.scheduler.snapshot(activation_id)
        assert snapshot is not None, activation_id
        return snapshot.state

    def index_of(self, decision: Decision) -> int:
        return self.decisions.index(decision)


def every_order(make_runner: Callable[[], SimulatedRunner]) -> Iterator[tuple[tuple[int, ...], SimulatedRunner]]:
    """Replay a fresh runner along every possible arrival order (small graphs only)."""
    stack: list[tuple[int, ...]] = [()]
    while stack:
        prefix = stack.pop()
        runner = make_runner()
        runner.start()
        for index in prefix:
            runner.step(index)
        if runner.outstanding and runner.finished is None:
            stack.extend((*prefix, index) for index in range(len(runner.outstanding)))
        else:
            yield prefix, runner
