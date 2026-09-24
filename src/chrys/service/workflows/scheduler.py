# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pure, deterministic workflow scheduler: arrival events in, decisions out.

No I/O, no clock, no threads. The runner feeds every event (activation
results, evaluation results, elapsed backoffs, retry requests, run-level
faults) into one scheduler and executes the returned decisions in order.
The design's error-classification table lives here once, as
:func:`auto_retry_permitted`, together with the activation state machine:

``pending → running(1) | skipped(attempt 0)``;
``running(n) → completed | failed | cancelled``;
``failed ∧ auto-retryable → retrying → running(n+1)``;
``failed ∧ exhausted/non-retryable ∧ interactive → awaiting_retry → running(n+k) | cancelled``;
``failed ∧ headless → failed`` (terminal, run ``node_failed``).

A run-level terminal (cancelled / worker_lost / storage_failed / loop_exhausted)
cancels every non-terminal activation. Inputs for terminal activations, stale
attempts, or a finished run are dropped and yield no decisions.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from chrys.service.workflows.graph import ON_EXHAUSTED_CONTINUE, GraphSpec, NodeSpec
from chrys.service.workflows.outcomes import RunOutcome
from chrys.service.workflows.sdk import SourceValue, WorkflowValue
from chrys.service.workflows.values import default_combine


class SchedulerInvariantError(RuntimeError):
    """A token was delivered twice, an input contradicts the state machine, or the run deadlocked."""


class RunMode(Enum):
    INTERACTIVE = "interactive"
    HEADLESS = "headless"


class ErrorClass(Enum):
    """Attempt failures classified for automatic retry and interactive recovery."""

    AGENT_TRANSIENT = "agent_transient"
    AGENT_NON_TRANSIENT = "agent_non_transient"
    AGENT_TIMEOUT = "agent_timeout"
    PYTHON_EXCEPTION = "python_exception"
    EVALUATION_ERROR = "evaluation_error"
    PYTHON_TIMEOUT = "python_timeout"
    ASK_UNAVAILABLE = "ask_unavailable"
    VALUE_TOO_LARGE = "value_too_large"
    PROTOCOL_LIMIT = "protocol_limit"
    VALUE_NOT_SERIALIZABLE = "value_not_serializable"
    LOOP_NO_VALUE = "loop_no_value"


_BACKEND_GATED = frozenset({ErrorClass.AGENT_TRANSIENT, ErrorClass.AGENT_TIMEOUT})
_RETRY_POLICY = frozenset({ErrorClass.PYTHON_EXCEPTION, ErrorClass.PYTHON_TIMEOUT})


@dataclass(frozen=True, slots=True)
class FailureReport:
    """Why an attempt failed. ``backend_approved`` is the backend's positive
    "a new pass for this activation is retryable" conclusion; the scheduler
    never derives it from counts."""

    error_class: ErrorClass
    message: str
    backend_approved: bool = False


def auto_retry_permitted(failure: FailureReport, *, attempt: int, max_attempts: int) -> bool:
    """Retry within budget; transient agent failures also require backend approval."""
    if attempt >= max_attempts:
        return False
    if failure.error_class in _BACKEND_GATED:
        return failure.backend_approved
    return failure.error_class in _RETRY_POLICY


class ActivationState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    AWAITING_RETRY = "awaiting_retry"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


_LIVE_STATES = frozenset({ActivationState.RUNNING, ActivationState.RETRYING, ActivationState.AWAITING_RETRY})
_SETTLED_STATES = frozenset({ActivationState.COMPLETED, ActivationState.SKIPPED})


class AttemptPhase(Enum):
    BODY = "body"
    UNTIL = "until"
    OUTGOING = "outgoing"


class LoopVerdict(Enum):
    CONTINUE = "continue"
    EXIT = "exit"
    EXHAUSTED = "exhausted"


class RetryRejection(Enum):
    RUN_FINISHED = "run_finished"
    DUPLICATE_REQUEST = "duplicate_request"
    UNKNOWN_ACTIVATION = "unknown_activation"
    NOT_AWAITING = "not_awaiting"
    STALE_ATTEMPT = "stale_attempt"


@dataclass(frozen=True, slots=True)
class AttemptRef:
    run_id: str
    node_id: str
    activation_id: str
    attempt: int


def activation_id(node_id: str, epoch: int) -> str:
    return f"{node_id}@iter#{epoch}"


# ---------------------------------------------------------------------------
# Decisions


@dataclass(frozen=True, slots=True)
class NodeStateChanged:
    ref: AttemptRef
    state: ActivationState
    error: FailureReport | None = None
    phase: AttemptPhase | None = None
    iteration: int = 0


@dataclass(frozen=True, slots=True)
class Activate:
    """Run the node body. ``value`` is the resolved input (already default-combined
    for multi-source nodes); it is ``None`` only for a join with a user combine,
    which the runner must evaluate from ``sources``."""

    ref: AttemptRef
    kind: str
    value: WorkflowValue | None
    sources: tuple[SourceValue, ...]


@dataclass(frozen=True, slots=True)
class EvaluateOutgoing:
    ref: AttemptRef
    value: WorkflowValue
    edge_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluateLoopUntil:
    ref: AttemptRef
    iteration: int
    value: WorkflowValue


@dataclass(frozen=True, slots=True)
class ScheduleRetry:
    """Wait ``backoff`` seconds then feed ``backoff_elapsed(ref)``; ``ref`` is the next attempt."""

    ref: AttemptRef
    backoff: float


@dataclass(frozen=True, slots=True)
class PersistRetryKey:
    """Emitted before the retried attempt's side effects; the runner writes it to the store first."""

    request_id: str
    ref: AttemptRef


@dataclass(frozen=True, slots=True)
class RetryRejected:
    node_id: str
    activation_id: str
    request_id: str
    reason: RetryRejection


@dataclass(frozen=True, slots=True)
class CancelActivation:
    ref: AttemptRef


@dataclass(frozen=True, slots=True)
class LoopActivated:
    """A loop activation (or a resumed attempt of it) starts with ``value`` as its input; the body is
    driven by the scheduler, so this is the runner's only sight of the loop's own input."""

    ref: AttemptRef
    value: WorkflowValue


@dataclass(frozen=True, slots=True)
class LoopIteration:
    """The ``until`` verdict on iteration ``iteration``; ``value`` is that iteration's exit value,
    which on ``exit`` (and on ``exhausted`` with ``on_exhausted="continue"``) is the loop's output."""

    ref: AttemptRef
    iteration: int
    verdict: LoopVerdict
    value: WorkflowValue


@dataclass(frozen=True, slots=True)
class RunOutput:
    node_id: str
    activation_id: str
    value: WorkflowValue


@dataclass(frozen=True, slots=True)
class RunFinished:
    outcome: RunOutcome
    outputs: tuple[RunOutput, ...]
    node_id: str | None = None
    error: FailureReport | None = None


Decision = (
    NodeStateChanged
    | Activate
    | EvaluateOutgoing
    | EvaluateLoopUntil
    | ScheduleRetry
    | PersistRetryKey
    | RetryRejected
    | CancelActivation
    | LoopActivated
    | LoopIteration
    | RunFinished
)


@dataclass(frozen=True, slots=True)
class ActivationSnapshot:
    node_id: str
    activation_id: str
    attempt: int
    state: ActivationState
    phase: AttemptPhase
    iteration: int
    output: WorkflowValue | None


# ---------------------------------------------------------------------------
# Internal records


@dataclass(frozen=True, slots=True)
class _Token:
    value: WorkflowValue | None  # None = closed
    activation_id: str


@dataclass(slots=True)
class _Activation:
    run_id: str
    node_id: str
    epoch: int
    attempt: int = 0
    state: ActivationState = ActivationState.PENDING
    phase: AttemptPhase = AttemptPhase.BODY
    input_value: WorkflowValue | None = None
    sources: tuple[SourceValue, ...] = ()
    output: WorkflowValue | None = None
    iteration: int = 0
    # Loop activations only.
    body_epoch: int = 0
    iteration_input: WorkflowValue | None = None
    unsettled: set[str] = field(default_factory=set)

    @property
    def activation_id(self) -> str:
        return activation_id(self.node_id, self.epoch)

    @property
    def ref(self) -> AttemptRef:
        return AttemptRef(self.run_id, self.node_id, self.activation_id, self.attempt)


class WorkflowScheduler:
    """One run of one graph. Every input returns the decisions it caused, in order."""

    def __init__(self, graph: GraphSpec, *, run_id: str, mode: RunMode) -> None:
        self._graph = graph
        self._run_id = run_id
        self._mode = mode
        self._activations: dict[str, _Activation] = {}
        self._tokens: dict[tuple[str, int], _Token] = {}
        self._pending: deque[tuple[str, int]] = deque()  # (node, epoch) that just received a token
        self._retry_requests: set[str] = set()
        self._live_count = 0
        self._unsettled_top_level = len(graph.top_level)
        self._started = False
        self._finished: RunFinished | None = None

    # -- observation ---------------------------------------------------------

    @property
    def finished(self) -> RunFinished | None:
        return self._finished

    def snapshot(self, activation_id: str) -> ActivationSnapshot | None:
        act = self._activations.get(activation_id)
        if act is None:
            return None
        return ActivationSnapshot(
            node_id=act.node_id,
            activation_id=act.activation_id,
            attempt=act.attempt,
            state=act.state,
            phase=act.phase,
            iteration=act.iteration,
            output=act.output,
        )

    def activation_ids(self) -> tuple[str, ...]:
        return tuple(self._activations)

    # -- inputs --------------------------------------------------------------

    def start(self, value: WorkflowValue) -> tuple[Decision, ...]:
        if self._started:
            raise SchedulerInvariantError("run already started.")
        self._started = True
        if self._finished is not None:  # a fault landed before the start: a finished run issues no work
            return ()
        out: list[Decision] = []
        self._activate(self._graph.start, 1, value, (), out)
        return self._flush(out)

    def activation_completed(self, ref: AttemptRef, value: WorkflowValue) -> tuple[Decision, ...]:
        act = self._live(ref)
        if act is None:
            return ()
        self._expect_phase(act, AttemptPhase.BODY)
        if self._node(act).is_loop:
            raise SchedulerInvariantError(f"loop activation {act.activation_id} has no body result.")
        out: list[Decision] = []
        act.output = value
        self._body_done(act, value, out)
        return self._flush(out)

    def activation_failed(self, ref: AttemptRef, failure: FailureReport) -> tuple[Decision, ...]:
        act = self._live(ref)
        if act is None:
            return ()
        self._expect_phase(act, AttemptPhase.BODY)
        if self._node(act).is_loop:
            raise SchedulerInvariantError(f"loop activation {act.activation_id} has no body result.")
        out: list[Decision] = []
        self._fail(act, failure, out)
        return self._flush(out)

    def outgoing_evaluated(self, ref: AttemptRef, decisions: Mapping[str, bool]) -> tuple[Decision, ...]:
        act = self._live(ref)
        if act is None:
            return ()
        self._expect_phase(act, AttemptPhase.OUTGOING)
        out: list[Decision] = []
        self._complete(act, decisions, out)
        return self._flush(out)

    def outgoing_failed(self, ref: AttemptRef, message: str) -> tuple[Decision, ...]:
        act = self._live(ref)
        if act is None:
            return ()
        self._expect_phase(act, AttemptPhase.OUTGOING)
        out: list[Decision] = []
        self._fail(act, FailureReport(ErrorClass.EVALUATION_ERROR, message), out)
        return self._flush(out)

    def loop_until_evaluated(self, ref: AttemptRef, iteration: int, verdict: bool) -> tuple[Decision, ...]:
        act = self._live(ref)
        if act is None:
            return ()
        self._expect_until(act, iteration)
        out: list[Decision] = []
        loop = self._node(act).loop
        assert loop is not None
        exit_value = act.output
        assert exit_value is not None
        if verdict:
            out.append(LoopIteration(act.ref, iteration, LoopVerdict.EXIT, exit_value))
            self._body_done(act, exit_value, out)
        elif iteration < loop.max_iterations:
            out.append(LoopIteration(act.ref, iteration, LoopVerdict.CONTINUE, exit_value))
            self._begin_iteration(act, iteration + 1, exit_value, out)
        else:
            out.append(LoopIteration(act.ref, iteration, LoopVerdict.EXHAUSTED, exit_value))
            if loop.on_exhausted == ON_EXHAUSTED_CONTINUE:
                self._body_done(act, exit_value, out)
            else:
                self._terminate(RunOutcome.LOOP_EXHAUSTED, out, node_id=act.node_id)
        return self._flush(out)

    def loop_until_failed(self, ref: AttemptRef, iteration: int, message: str) -> tuple[Decision, ...]:
        act = self._live(ref)
        if act is None:
            return ()
        self._expect_until(act, iteration)
        out: list[Decision] = []
        self._fail(act, FailureReport(ErrorClass.EVALUATION_ERROR, message), out)
        return self._flush(out)

    def backoff_elapsed(self, ref: AttemptRef) -> tuple[Decision, ...]:
        act = self._activations.get(ref.activation_id)
        if (
            self._finished is not None
            or ref.run_id != self._run_id
            or act is None
            or act.node_id != ref.node_id
            or act.state is not ActivationState.RETRYING
            or act.attempt + 1 != ref.attempt
        ):
            return ()
        out: list[Decision] = []
        act.attempt += 1
        self._resume(act, out)
        return self._flush(out)

    def manual_retry(
        self, node_id: str, activation_id: str, request_id: str, expected_failed_attempt: int
    ) -> tuple[Decision, ...]:
        rejection = self._retry_rejection(node_id, activation_id, request_id, expected_failed_attempt)
        if rejection is not None:
            return (RetryRejected(node_id, activation_id, request_id, rejection),)
        act = self._activations[activation_id]
        self._retry_requests.add(request_id)
        act.attempt += 1
        out: list[Decision] = [PersistRetryKey(request_id, act.ref)]
        self._resume(act, out)
        return self._flush(out)

    def cancel(self) -> tuple[Decision, ...]:
        return self._run_fault(RunOutcome.CANCELLED)

    def worker_lost(self) -> tuple[Decision, ...]:
        return self._run_fault(RunOutcome.WORKER_LOST)

    def storage_failed(self) -> tuple[Decision, ...]:
        return self._run_fault(RunOutcome.STORAGE_FAILED)

    # -- input guards --------------------------------------------------------

    def _node(self, act: _Activation) -> NodeSpec:
        return self._graph.nodes[act.node_id]

    def _live(self, ref: AttemptRef) -> _Activation | None:
        """The running activation this ref addresses, or ``None`` when the input is late."""
        if self._finished is not None or ref.run_id != self._run_id:
            return None
        act = self._activations.get(ref.activation_id)
        if act is None or act.node_id != ref.node_id or act.attempt != ref.attempt:
            return None
        if act.state is not ActivationState.RUNNING:
            return None
        return act

    def _expect_phase(self, act: _Activation, phase: AttemptPhase) -> None:
        if act.phase is not phase:
            raise SchedulerInvariantError(
                f"{act.activation_id} attempt {act.attempt} is in phase {act.phase.value}, not {phase.value}."
            )

    def _expect_until(self, act: _Activation, iteration: int) -> None:
        self._expect_phase(act, AttemptPhase.UNTIL)
        if iteration != act.iteration:
            raise SchedulerInvariantError(
                f"{act.activation_id} is at iteration {act.iteration}, got a verdict for {iteration}."
            )

    def _retry_rejection(
        self, node_id: str, activation_id: str, request_id: str, expected_failed_attempt: int
    ) -> RetryRejection | None:
        if self._finished is not None:
            return RetryRejection.RUN_FINISHED
        if request_id in self._retry_requests:
            return RetryRejection.DUPLICATE_REQUEST
        act = self._activations.get(activation_id)
        if act is None or act.node_id != node_id:
            return RetryRejection.UNKNOWN_ACTIVATION
        if act.state is not ActivationState.AWAITING_RETRY:
            return RetryRejection.NOT_AWAITING
        if act.attempt != expected_failed_attempt:
            return RetryRejection.STALE_ATTEMPT
        return None

    def _flush(self, out: list[Decision]) -> tuple[Decision, ...]:
        if self._finished is None and self._live_count == 0:
            raise SchedulerInvariantError("run is quiescent but not finished: unresolved nodes remain.")
        return tuple(out)

    # -- activation lifecycle ------------------------------------------------

    def _change_state(
        self, act: _Activation, state: ActivationState, out: list[Decision], error: FailureReport | None = None
    ) -> None:
        self._live_count += int(state in _LIVE_STATES) - int(act.state in _LIVE_STATES)
        top_level = self._node(act).parent_loop is None
        if top_level:
            self._unsettled_top_level += int(act.state in _SETTLED_STATES) - int(state in _SETTLED_STATES)
        act.state = state
        out.append(
            NodeStateChanged(
                act.ref,
                state,
                error=error,
                phase=act.phase if error is not None else None,
                iteration=0 if top_level else act.iteration,
            )
        )

    def _new_activation(self, node_id: str, epoch: int) -> _Activation:
        act = _Activation(run_id=self._run_id, node_id=node_id, epoch=epoch)
        if act.activation_id in self._activations:
            raise SchedulerInvariantError(f"{act.activation_id} resolved twice.")
        parent = self._graph.nodes[node_id].parent_loop
        if parent is not None:
            act.iteration = self._activations[activation_id(parent, 1)].iteration
        self._activations[act.activation_id] = act
        return act

    def _activate(
        self,
        node_id: str,
        epoch: int,
        value: WorkflowValue | None,
        sources: tuple[SourceValue, ...],
        out: list[Decision],
    ) -> None:
        act = self._new_activation(node_id, epoch)
        act.attempt = 1
        act.input_value = value
        act.sources = sources
        self._change_state(act, ActivationState.RUNNING, out)
        self._dispatch(act, out)

    def _dispatch(self, act: _Activation, out: list[Decision]) -> None:
        """Issue the work for a fresh attempt at phase ``body``."""
        node = self._node(act)
        act.phase = AttemptPhase.BODY
        if node.is_loop:
            assert act.input_value is not None
            out.append(LoopActivated(act.ref, act.input_value))
            self._begin_iteration(act, 1, act.input_value, out)
        else:
            out.append(Activate(act.ref, node.kind, act.input_value, act.sources))

    def _resume(self, act: _Activation, out: list[Decision]) -> None:
        """A new attempt after retrying/awaiting_retry: loops resume at the failed
        phase, every other node reruns the whole activation."""
        self._change_state(act, ActivationState.RUNNING, out)
        node = self._node(act)
        if not node.is_loop:
            self._dispatch(act, out)
            return
        assert act.input_value is not None
        out.append(LoopActivated(act.ref, act.input_value))
        if act.phase is AttemptPhase.UNTIL:
            assert act.output is not None
            out.append(EvaluateLoopUntil(act.ref, act.iteration, act.output))
        elif act.phase is AttemptPhase.OUTGOING:
            assert act.output is not None
            out.append(EvaluateOutgoing(act.ref, act.output, self._graph.conditional_edges[node.node_id]))
        else:
            assert act.iteration_input is not None
            self._begin_iteration(act, act.iteration, act.iteration_input, out)

    def _body_done(self, act: _Activation, value: WorkflowValue, out: list[Decision]) -> None:
        """The body value is known; ``completed`` waits for the outgoing evaluation."""
        act.output = value
        conditional = self._graph.conditional_edges[act.node_id]
        if conditional:
            act.phase = AttemptPhase.OUTGOING
            out.append(EvaluateOutgoing(act.ref, value, conditional))
        else:
            self._complete(act, {}, out)

    def _complete(self, act: _Activation, decisions: Mapping[str, bool], out: list[Decision]) -> None:
        node = self._node(act)
        opened = self._opened_edges(node, decisions)
        self._change_state(act, ActivationState.COMPLETED, out)
        self._settle(act, out)
        value = act.output
        for edge_id in node.out_edges:
            self._deliver(edge_id, act.epoch, value if edge_id in opened else None, act.activation_id)
        self._resolve_pending(out)
        self._check_run_completed(out)

    def _opened_edges(self, node: NodeSpec, decisions: Mapping[str, bool]) -> set[str]:
        conditional = set(self._graph.conditional_edges[node.node_id])
        if set(decisions) != conditional:
            raise SchedulerInvariantError(
                f"outgoing decisions for {node.node_id!r} do not match its conditional edges."
            )
        opened = {edge_id for edge_id in node.out_edges if edge_id not in conditional}
        switched: set[str] = set()
        for edge_ids in self._graph.switch_groups[node.node_id].values():
            switched.update(edge_ids)
            cases = sorted(
                (edge_id for edge_id in edge_ids if not self._graph.edges[edge_id].switch_default),
                key=lambda edge_id: self._graph.edges[edge_id].switch_position or 0,
            )
            chosen = next((edge_id for edge_id in cases if decisions[edge_id]), None)
            if chosen is None:
                chosen = next(edge_id for edge_id in edge_ids if self._graph.edges[edge_id].switch_default)
            opened.add(chosen)
        opened.update(edge_id for edge_id in conditional - switched if decisions[edge_id])
        return opened

    def _skip(self, node_id: str, epoch: int, out: list[Decision]) -> None:
        act = self._new_activation(node_id, epoch)
        self._change_state(act, ActivationState.SKIPPED, out)
        self._settle(act, out)
        for edge_id in self._node(act).out_edges:
            self._deliver(edge_id, epoch, None, act.activation_id)

    def _deliver(self, edge_id: str, epoch: int, value: WorkflowValue | None, activation_id: str) -> None:
        key = (edge_id, epoch)
        if key in self._tokens:
            raise SchedulerInvariantError(f"edge {edge_id!r} epoch {epoch} already carries a token.")
        self._tokens[key] = _Token(value, activation_id)
        self._pending.append((self._graph.edges[edge_id].dst, epoch))

    def _resolve_pending(self, out: list[Decision]) -> None:
        """Resolve every node that received a token; a skipped node hands its closed tokens back here."""
        while self._pending:
            node_id, epoch = self._pending.popleft()
            self._try_resolve(node_id, epoch, out)

    def _try_resolve(self, node_id: str, epoch: int, out: list[Decision]) -> None:
        if activation_id(node_id, epoch) in self._activations:
            return
        node = self._graph.nodes[node_id]
        tokens: list[tuple[str, _Token]] = []
        for edge_id in node.in_edges:
            token = self._tokens.get((edge_id, epoch))
            if token is None:
                return
            tokens.append((edge_id, token))
        sources = tuple(
            SourceValue(self._graph.edges[edge_id].src, token.activation_id, token.value)
            for edge_id, token in tokens
            if token.value is not None
        )
        if not sources:
            self._skip(node_id, epoch, out)
            return
        if node.has_combine:
            value: WorkflowValue | None = None
        elif len(sources) == 1:
            value = sources[0].value
        else:
            value = default_combine(sources)
        self._activate(node_id, epoch, value, sources, out)

    def _fail(self, act: _Activation, failure: FailureReport, out: list[Decision]) -> None:
        node = self._node(act)
        if auto_retry_permitted(failure, attempt=act.attempt, max_attempts=node.retry.max_attempts):
            self._change_state(act, ActivationState.RETRYING, out, failure)
            next_ref = AttemptRef(self._run_id, act.node_id, act.activation_id, act.attempt + 1)
            out.append(ScheduleRetry(next_ref, node.retry.backoff))
        elif self._mode is RunMode.INTERACTIVE:
            self._change_state(act, ActivationState.AWAITING_RETRY, out, failure)
        else:
            self._change_state(act, ActivationState.FAILED, out, failure)
            self._terminate(RunOutcome.NODE_FAILED, out, node_id=act.node_id, error=failure)

    # -- loops ---------------------------------------------------------------

    def _begin_iteration(self, act: _Activation, iteration: int, value: WorkflowValue, out: list[Decision]) -> None:
        loop = self._node(act).loop
        assert loop is not None
        act.phase = AttemptPhase.BODY
        act.iteration = iteration
        act.body_epoch += 1
        act.iteration_input = value
        act.output = None
        act.unsettled = set(loop.body)
        self._activate(loop.entry, act.body_epoch, value, (), out)

    def _settle(self, act: _Activation, out: list[Decision]) -> None:
        parent = self._node(act).parent_loop
        if parent is None:
            return
        loop_act = self._activations[activation_id(parent, 1)]
        loop_act.unsettled.discard(act.node_id)
        if not loop_act.unsettled and loop_act.state is ActivationState.RUNNING:
            self._iteration_barrier(loop_act, out)

    def _iteration_barrier(self, loop_act: _Activation, out: list[Decision]) -> None:
        loop = self._node(loop_act).loop
        assert loop is not None
        exit_act = self._activations[activation_id(loop.exit, loop_act.body_epoch)]
        if exit_act.state is ActivationState.SKIPPED:
            message = f"iteration {loop_act.iteration} of {loop_act.node_id!r} closed every edge into its exit node."
            self._fail(loop_act, FailureReport(ErrorClass.LOOP_NO_VALUE, message), out)
            return
        assert exit_act.output is not None
        loop_act.output = exit_act.output
        loop_act.phase = AttemptPhase.UNTIL
        out.append(EvaluateLoopUntil(loop_act.ref, loop_act.iteration, exit_act.output))

    # -- run terminal --------------------------------------------------------

    def _check_run_completed(self, out: list[Decision]) -> None:
        if self._finished is not None:
            return
        if self._unsettled_top_level:
            return
        self._finished = RunFinished(RunOutcome.COMPLETED, self._outputs())
        out.append(self._finished)

    def _run_fault(self, outcome: RunOutcome) -> tuple[Decision, ...]:
        if self._finished is not None:
            return ()
        out: list[Decision] = []
        self._terminate(outcome, out)
        return tuple(out)

    def _terminate(
        self,
        outcome: RunOutcome,
        out: list[Decision],
        *,
        node_id: str | None = None,
        error: FailureReport | None = None,
    ) -> None:
        for act in self._activations.values():
            if act.state not in _LIVE_STATES:
                continue
            if act.state is ActivationState.RUNNING:
                out.append(CancelActivation(act.ref))
            self._change_state(act, ActivationState.CANCELLED, out)
        self._finished = RunFinished(outcome, self._outputs(), node_id=node_id, error=error)
        out.append(self._finished)

    def _outputs(self) -> tuple[RunOutput, ...]:
        outputs: list[RunOutput] = []
        for node_id in self._graph.outputs:
            act = self._activations.get(activation_id(node_id, 1))
            if act is not None and act.state is ActivationState.COMPLETED and act.output is not None:
                outputs.append(RunOutput(node_id, act.activation_id, act.output))
        return tuple(outputs)
