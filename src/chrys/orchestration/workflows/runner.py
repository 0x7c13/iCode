# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""One workflow run: the pure scheduler's decisions executed against a worker, agent shells, and the journal.

The scheduler decides, the runner acts. Every scheduler input goes through
one lock, and the decisions it returns are executed, journal writes
included, before the next input is taken: the run log's sequence order is
the scheduler's decision order. Node bodies, evaluations, and backoffs run
as tasks outside that lock and feed their results back through it.

Faults converge the same way user actions do: a lost worker or a failed
store becomes a scheduler input whose terminal decisions cancel what is
still running. The runner then drains its tasks, closes every agent shell,
waits for the worker to exit, checkpoints the session, and records the terminal
last. A failed checkpoint makes the run ``storage_failed`` even if every node
completed successfully.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from chrys.foundation.events.types import (
    WORKFLOW_NOTICE_DATA_DROPPED,
    WORKFLOW_OUTPUT_EMIT,
    WORKFLOW_OUTPUT_FINAL,
    WorkflowOutputSummary,
)
from chrys.foundation.models.ask_user import AskUserAnswer, AskUserQuestion, validate_ask_user_answers
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.util.once_close import finish_close
from chrys.orchestration.invoker.contracts import AbortCause
from chrys.orchestration.workflows.agent_archive import AgentNodeArchive
from chrys.orchestration.workflows.agent_node import WorkflowAgentShell
from chrys.orchestration.workflows.agent_node_build import AgentNodeResources
from chrys.orchestration.workflows.worker_client import (
    AskUnavailable,
    AttemptTimeout,
    CapturedOutput,
    WorkerLostError,
    WorkerRpcError,
    WorkflowWorkerClient,
)
from chrys.service.trajectory.workflow import WorkflowTrace
from chrys.service.workflows.admission import AdmittedManifest
from chrys.service.workflows.graph import KIND_AGENT, KIND_JOIN, KIND_PYTHON, ON_EXHAUSTED_CONTINUE
from chrys.service.workflows.journal import WorkflowJournal, summarize
from chrys.service.workflows.outcomes import (
    REASON_DEADLINE_EXCEEDED,
    REASON_INTERNAL_ERROR,
    REASON_SHUTDOWN,
    RunOutcome,
)
from chrys.service.workflows.protocol import ErrorCode, ProtocolError
from chrys.service.workflows.scheduler import (
    Activate,
    ActivationState,
    AttemptRef,
    CancelActivation,
    Decision,
    ErrorClass,
    EvaluateLoopUntil,
    EvaluateOutgoing,
    FailureReport,
    LoopActivated,
    LoopIteration,
    LoopVerdict,
    NodeStateChanged,
    PersistRetryKey,
    RetryRejected,
    RunFinished,
    RunMode,
    RunOutput,
    ScheduleRetry,
    WorkflowScheduler,
)
from chrys.service.workflows.sdk import SourceValue, WorkflowValue
from chrys.service.workflows.store import (
    DATA_DROPPED_KEY,
    NODE_RECORD_INPUT,
    NODE_RECORD_OUTPUT,
    WorkflowStorageFailed,
)
from chrys.service.workflows.values import ValueShapeError, source_to_wire, value_to_wire

if TYPE_CHECKING:
    from chrys.service.approval.policy import ApprovalMode

logger = logging.getLogger(__name__)

MAX_CONCURRENT_AGENT_ATTEMPTS: Final = 4
"""Concurrent agent attempts per run, including shell setup and execution but excluding retry backoff."""

_CANCEL_CAUSES: Final = {
    REASON_DEADLINE_EXCEEDED: AbortCause.CALLER_TIMEOUT,
    REASON_SHUTDOWN: AbortCause.OWNER_CLOSE,
    REASON_INTERNAL_ERROR: AbortCause.OWNER_CLOSE,
}
_DURABLE_STATES: Final = frozenset(
    {ActivationState.COMPLETED, ActivationState.FAILED, ActivationState.SKIPPED, ActivationState.CANCELLED}
)
_PYTHON_FAILURE_CLASSES: Final = {
    ErrorCode.USER_EXCEPTION: ErrorClass.PYTHON_EXCEPTION,
    ErrorCode.VALUE_TOO_LARGE: ErrorClass.VALUE_TOO_LARGE,
    ErrorCode.PROTOCOL_LIMIT: ErrorClass.PROTOCOL_LIMIT,
    ErrorCode.VALUE_NOT_SERIALIZABLE: ErrorClass.VALUE_NOT_SERIALIZABLE,
    ErrorCode.ASK_UNAVAILABLE: ErrorClass.ASK_UNAVAILABLE,
}


@dataclass(frozen=True, slots=True)
class WorkflowRunResult:
    """How a run ended; ``outputs`` follows the workflow's ``output()`` declaration order."""

    run_id: str
    outcome: RunOutcome
    outputs: tuple[RunOutput, ...] = ()
    node_id: str = ""
    error: str = ""
    reason: str = ""
    duration: float = 0.0  # seconds


class WorkerCallbacks:
    """The ask/emit handlers a worker is launched with, bound to the runner once it exists.

    The worker is launched (and the workflow file loaded) before the runner
    is built; no node body runs until the runner starts the scheduler, so
    nothing reaches these handlers unbound.
    """

    def __init__(self) -> None:
        self._runner: WorkflowRunner | None = None

    def bind(self, runner: WorkflowRunner) -> None:
        self._runner = runner

    async def ask(self, ref: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        if self._runner is None:
            raise AskUnavailable("no run is active.")
        return await self._runner.on_ask(ref, questions)

    async def emit(self, ref: AttemptRef, ordinal: int, text: str) -> None:
        if self._runner is not None:
            await self._runner.on_emit(ref, ordinal, text)


@dataclass(slots=True)
class _PendingAsk:
    ref: AttemptRef
    questions: tuple[AskUserQuestion, ...]
    answering: bool = False
    future: asyncio.Future[tuple[AskUserAnswer, ...]] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )


class WorkflowRunner:
    """Executes one run of an admitted workflow; see the module docstring for the ordering contract."""

    def __init__(
        self,
        *,
        admitted: AdmittedManifest,
        journal: WorkflowJournal,
        worker: WorkflowWorkerClient,
        resources: AgentNodeResources,
        checkpoint: Callable[[], Awaitable[None]],
        mode: RunMode,
        timeout: float | None = None,
        trace: WorkflowTrace | None = None,
        load_stdout: CapturedOutput | None = None,
    ) -> None:
        self._admitted = admitted
        self._graph = admitted.graph
        self._journal = journal
        self._worker = worker
        self._resources = resources
        self._checkpoint = checkpoint
        self._mode = mode
        self._timeout = timeout
        self._trace = trace
        self._load_stdout = load_stdout or CapturedOutput("", False)
        self._output_collected = False
        self._output_attempts: dict[str, int] = {}
        self._record_end: Callable[[WorkflowRunResult], Awaitable[None]] | None = None
        self._run_id = journal.run_id
        self._scheduler = WorkflowScheduler(admitted.graph, run_id=self._run_id, mode=mode)
        self._lock = asyncio.Lock()
        self._agent_slots = asyncio.BoundedSemaphore(MAX_CONCURRENT_AGENT_ATTEMPTS)
        self._lock_holder: asyncio.Task[Any] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._aborts: set[asyncio.Task[None]] = set()
        self._invocations: dict[str, str] = {}
        self._shells: dict[str, WorkflowAgentShell] = {}
        self._started = 0.0
        self._asks: dict[str, _PendingAsk] = {}
        self._terminal: RunFinished | None = None
        self._terminal_event = asyncio.Event()
        self._deferred_fault: Callable[[], tuple[Decision, ...]] | None = None
        self._record_error: str | None = None
        self._cause: AbortCause | None = None
        self._reason = ""
        self._cancel_task: asyncio.Task[None] | None = None
        self._worker_error = ""
        self._internal_error = ""
        self._data_dropped_noticed = False
        self._result: WorkflowRunResult | None = None
        self._deadline_task: asyncio.Task[None] | None = None
        self._worker_watch: asyncio.Task[None] | None = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def result(self) -> WorkflowRunResult | None:
        """The terminal result once :meth:`run` returned it."""
        return self._result

    @property
    def finished(self) -> bool:
        return self._terminal is not None

    def holds(self, task: asyncio.Task[Any] | None) -> bool:
        """True when *task* is one the run waits for: a subscriber to one of the run's events runs on one.

        Scheduler events go out under the runner's lock, from the run task or one of its node tasks;
        outputs and asks under the journal's lock alone, from a node's or the worker's task; an agent
        node's invocation events from the tasks of its shell's pass, which the teardown waits out.
        Waiting for the run from any of them would wait on itself.
        """
        if task is None:
            return False
        return (
            task is self._lock_holder
            or task in self._tasks
            or task in self._aborts
            or self._journal.holds(task)
            or any(shell.holds(task) for shell in self._shells.values())
        )

    # -- driving ---------------------------------------------------------------

    async def run(
        self,
        input_text: str,
        *,
        startup: tuple[Callable[[], Awaitable[None]], ...],
        record_end: Callable[[WorkflowRunResult], Awaitable[None]],
    ) -> WorkflowRunResult:
        """Start the scheduler with *input_text* and return once the run is terminal and torn down."""
        self._started = time.monotonic()
        self._record_end = record_end
        if self._timeout is not None and self._terminal is None:
            self._deadline_task = self._spawn(self._deadline(self._timeout))
        self._worker_watch = self._spawn(self._watch_worker())
        try:
            # Acceptance owns these facts even if cancellation already settled
            # the scheduler. Only hook execution and node work are cancellable.
            if self._trace is not None:
                await self._trace.started()
            await self._record(self._journal.run_started())
            for step in startup:
                await self._startup_step(step)
            if self._terminal is None:
                await self._feed(lambda: self._scheduler.start(WorkflowValue(text=input_text)))
            await self._terminal_event.wait()
            await self._teardown()
        except asyncio.CancelledError:
            await asyncio.shield(self._abandon(REASON_SHUTDOWN))
            raise
        except Exception as exc:
            # Even a failure before a scheduler input must leave a terminal record and a closed worker.
            logger.exception("workflow run %s: the runner failed", self.run_id)
            reason = REASON_INTERNAL_ERROR
            if isinstance(exc, WorkflowStorageFailed):
                self._storage_failed(str(exc))
                reason = ""
            else:
                self._unexpected_failure(f"{type(exc).__name__}: {exc}")
            result = await self._abandon(reason)
            if result is None:
                raise
            return result
        return await self._finish()

    async def _startup_step(self, step: Callable[[], Awaitable[None]]) -> None:
        """Dispatch each lifecycle hook once, cancelling its waits when the run ends."""

        async def invoke() -> None:
            task = asyncio.current_task()
            if task is None:
                raise RuntimeError("Dispatching workflow hooks requires an asyncio task.")
            self._tasks.add(task)  # inline subscribers must recognize their run owner
            await step()

        # Enter the dispatch before cancellation can prevent the coroutine from
        # ever running. Already-cancelled runs still announce start/end; blocking
        # startup work stops at its first suspension and drains before proceeding.
        task = asyncio.create_task(invoke(), eager_start=True)
        task.add_done_callback(self._forget)
        if self._terminal is not None:
            task.cancel()
        (result,) = await asyncio.gather(task, return_exceptions=True)
        if isinstance(result, asyncio.CancelledError):
            if self._terminal is None:
                raise result
        elif isinstance(result, Exception):
            cancelled = self.cancel(reason=REASON_INTERNAL_ERROR)
            if cancelled is not None:
                await cancelled
        elif isinstance(result, BaseException):
            raise result

    def cancel(self, *, reason: str = "") -> asyncio.Task[None] | None:
        """Cancel the run; ``reason`` refines the terminal and picks the abort cause agent passes see.

        The scheduler input is queued, never awaited here: a caller reacting inline to one of the run's
        own events (published under the runner lock) must not wait for that lock. The queued input is
        returned for a caller outside that lock which needs it applied; ``None`` once the run is terminal.
        """
        if self._terminal is not None:
            return None
        if self._cause is None:
            self._reason = reason
            self._cause = _CANCEL_CAUSES.get(reason, AbortCause.USER_CANCEL)
            self._cancel_task = self._spawn(self._feed(self._scheduler.cancel))
        return self._cancel_task

    def retry(self, node_id: str, activation_id: str, request_id: str, expected_failed_attempt: int) -> None:
        """A manual retry request, queued like :meth:`cancel`; rejections are logged (replays are harmless)."""
        if self._terminal is not None:
            return
        self._spawn(
            self._feed(
                lambda: self._scheduler.manual_retry(node_id, activation_id, request_id, expected_failed_attempt)
            )
        )

    def answer(self, node_id: str, activation_id: str, request_id: str, answers: tuple[AskUserAnswer, ...]) -> bool:
        """Queue one answer, including from an inline ask subscriber; reject duplicate or invalid admissions.

        An invalid answer leaves the ask open, so a corrected one can still follow.
        """
        pending = self._asks.get(request_id)
        if (
            self._terminal is not None
            or pending is None
            or pending.future.done()
            or pending.answering
            or (pending.ref.node_id, pending.ref.activation_id) != (node_id, activation_id)
        ):
            return False
        validated = validate_ask_user_answers(answers, questions=pending.questions)
        if validated is None:
            return False
        pending.answering = True
        self._spawn(self._answer(pending, request_id, validated))
        return True

    async def _answer(self, pending: _PendingAsk, request_id: str, answers: tuple[AskUserAnswer, ...]) -> None:
        async def commit() -> None:
            if (
                await self._record(self._journal.node_answer(pending.ref, request_id, pending.questions, answers))
                is None
            ):
                return
            if not pending.future.done():
                pending.future.set_result(answers)

        # Once admitted, the answer owns its durable write through cancellation.
        await finish_close(asyncio.create_task(commit()))

    # -- worker callbacks ----------------------------------------------------------

    async def on_ask(self, ref: AttemptRef, questions: tuple[AskUserQuestion, ...]) -> tuple[AskUserAnswer, ...]:
        if self._mode is RunMode.HEADLESS:
            raise AskUnavailable("nobody can answer in a headless run.")
        if self._terminal is not None:
            raise AskUnavailable("the run is over.")
        request_id = new_analytics_id()
        pending = _PendingAsk(ref, questions)
        self._asks[request_id] = pending
        try:
            if await self._record(self._journal.node_ask(ref, request_id, questions)) is None:
                raise AskUnavailable("the run record failed.")
            return await pending.future
        finally:
            self._asks.pop(request_id, None)

    async def on_emit(self, ref: AttemptRef, ordinal: int, text: str) -> None:
        if not await self._write_emit(ref, ordinal, text):
            return
        await self._record(self._journal.node_output(ref, WORKFLOW_OUTPUT_EMIT, ordinal, text))

    # -- scheduler loop ------------------------------------------------------------

    async def _feed(self, produce: Callable[[], tuple[Decision, ...]]) -> None:
        """One scheduler input and every decision it caused, executed before the next input."""
        async with self._lock:
            self._lock_holder = asyncio.current_task()
            try:
                if self._terminal is not None:
                    return
                await self._apply(produce())
                if self._terminal is None and self._scheduler.finished is not None:
                    # A prior decision publisher may have failed after the scheduler reached its terminal.
                    self._on_terminal(self._scheduler.finished)
                while self._deferred_fault is not None and self._terminal is None:
                    fault, self._deferred_fault = self._deferred_fault, None
                    await self._apply(fault())
            finally:
                self._lock_holder = None

    async def _apply(self, decisions: tuple[Decision, ...]) -> None:
        for decision in decisions:
            match decision:
                case NodeStateChanged():
                    await self._on_state(decision)
                case Activate():
                    self._spawn(self._activate(decision))
                case EvaluateOutgoing():
                    self._spawn(self._evaluate_outgoing(decision))
                case EvaluateLoopUntil():
                    self._spawn(self._evaluate_until(decision))
                case ScheduleRetry():
                    self._spawn(self._backoff(decision))
                case PersistRetryKey():
                    await self._record(self._journal.retry_key(decision.request_id, decision.ref))
                case RetryRejected():
                    logger.info(
                        "workflow run %s rejected retry %s of %s: %s",
                        self._run_id,
                        decision.request_id,
                        decision.activation_id,
                        decision.reason.value,
                    )
                case CancelActivation():
                    self._cancel_activation(decision.ref)
                case LoopActivated():
                    await self._write_input(decision.ref, {"value": value_to_wire(decision.value)})
                case LoopIteration():
                    await self._record(
                        self._journal.loop_iteration(decision.ref, decision.iteration, decision.verdict.value)
                    )
                    if self._loop_settles(decision):
                        await self._settle_loop(decision.ref, decision.value)
                case RunFinished():
                    self._on_terminal(decision)

    async def _on_state(self, decision: NodeStateChanged) -> None:
        ref = decision.ref
        invocation_id = ""
        kind = self._graph.nodes[ref.node_id].kind
        if kind == KIND_AGENT and ref.attempt > 0:
            invocation_id = self._invocation_id(ref)
        failure = decision.error
        if self._trace is not None:
            await self._trace.node_state(decision, kind=kind)
        await self._record(
            self._journal.node_state(
                ref,
                decision.state.value,
                invocation_id=invocation_id,
                error=failure.message if failure is not None else "",
                error_class=failure.error_class.value if failure is not None else "",
                durable=decision.state in _DURABLE_STATES,
                iteration=decision.iteration,
                failure_phase=decision.phase.value if decision.phase is not None else "",
            )
        )
        if decision.state is ActivationState.COMPLETED and ref.activation_id in self._shells:
            # Body success alone is not terminal: outgoing conditions can still
            # fail and retry the same activation. Release only after settlement,
            # outside the scheduler lock because close drains event publishers.
            self._spawn(self._close_shell(ref.activation_id))

    def _on_terminal(self, decision: RunFinished) -> None:
        self._terminal = decision
        if self._cause is None:
            self._cause = AbortCause.RUN_TERMINAL
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        for pending in self._asks.values():
            pending.future.cancel()
        self._terminal_event.set()

    def _cancel_activation(self, ref: AttemptRef) -> None:
        """Only ever issued by a terminal: the local task is cancelled here, the worker at teardown."""
        shell = self._shells.get(ref.activation_id)
        if shell is not None:
            cause = self._cause or AbortCause.RUN_TERMINAL
            task = asyncio.create_task(shell.abort(cause), name=f"chrys.workflow.abort.{ref.activation_id}")
            self._aborts.add(task)
            task.add_done_callback(self._aborts.discard)

    # -- attempts --------------------------------------------------------------------

    async def _activate(self, decision: Activate) -> None:
        ref = decision.ref
        try:
            if decision.kind == KIND_PYTHON:
                if decision.value is None:
                    raise RuntimeError("Activating a Python or agent node requires an input value.")
                await self._run_python(ref, decision.value)
            elif decision.kind == KIND_AGENT:
                if decision.value is None:
                    raise RuntimeError("Activating a Python or agent node requires an input value.")
                async with self._agent_slots:
                    await self._run_agent(ref, decision.value)
            elif decision.kind == KIND_JOIN:
                await self._run_join(ref, decision.value, decision.sources)
            else:
                await self._fail(
                    ref, FailureReport(ErrorClass.PYTHON_EXCEPTION, f"unknown node kind {decision.kind!r}")
                )
        except WorkerLostError as exc:
            await self._worker_lost(str(exc))
        except (ProtocolError, ValueShapeError) as exc:
            await self._worker_lost(f"worker protocol violation at {ref.activation_id}: {exc}")
        except WorkflowStorageFailed as exc:
            self._storage_failed(str(exc))

    async def _run_python(self, ref: AttemptRef, value: WorkflowValue) -> None:
        if not await self._write_input(ref, {"value": value_to_wire(value)}):
            return
        node = self._graph.nodes[ref.node_id]
        try:
            result = await self._worker.run_python(ref, value, blocking=not node.fn_is_async, timeout=node.timeout)
        except WorkerRpcError as exc:
            await self._write_diagnostics(ref, "body", exc.stdout, exc.traceback)
            if exc.code == ErrorCode.ATTEMPT_TERMINATED and self._terminal is not None:
                return
            error_class = _PYTHON_FAILURE_CLASSES.get(exc.code, ErrorClass.PYTHON_EXCEPTION)
            message = exc.message if exc.code in _PYTHON_FAILURE_CLASSES else f"{exc.code}: {exc.message}"
            await self._fail(ref, FailureReport(error_class, message))
        except AttemptTimeout as exc:
            await self._write_diagnostics(ref, "body", exc.stdout, exc.traceback)
            message = f"no result within {node.timeout:g}s"
            if exc.leaked_thread:
                message += "; its thread is still running"
            await self._fail(ref, FailureReport(ErrorClass.PYTHON_TIMEOUT, message))
        else:
            await self._write_diagnostics(ref, "body", result.stdout)
            await self._complete(ref, result.value, last_emit_ordinal=result.last_emit_ordinal)

    async def _run_agent(self, ref: AttemptRef, value: WorkflowValue) -> None:
        record: dict[str, Any] = {"value": value_to_wire(value)}
        if value.data is not None:
            record[DATA_DROPPED_KEY] = True
        if not await self._write_input(ref, record):
            return
        if value.data is not None and not self._data_dropped_noticed:
            self._data_dropped_noticed = True
            await self._record(
                self._journal.run_notice(
                    ref.node_id,
                    ref.activation_id,
                    WORKFLOW_NOTICE_DATA_DROPPED,
                    f"Node {ref.node_id!r} received structured data; an agent node consumes only the text.",
                )
            )
        shell = self._shell_for(ref)
        if not shell.is_prepared:
            try:
                await shell.open(value.text, attempt=ref.attempt)
            except WorkflowStorageFailed as exc:
                self._storage_failed(str(exc))
                return
            except Exception as exc:
                logger.warning(
                    "workflow run %s: agent node %s could not open", self._run_id, ref.node_id, exc_info=True
                )
                with contextlib.suppress(Exception):
                    await shell.close()
                del self._shells[ref.activation_id]  # the next attempt builds a fresh shell under the same identity
                await self._fail(ref, FailureReport(ErrorClass.AGENT_NON_TRANSIENT, f"{type(exc).__name__}: {exc}"))
                return
        try:
            result = await shell.run(
                value.text,
                timeout=self._graph.nodes[ref.node_id].timeout,
                trajectory_context=self._trace.node_context(ref.activation_id, ref.attempt) if self._trace else None,
                attempt=ref.attempt,
            )
        except WorkflowStorageFailed as exc:
            self._storage_failed(str(exc))
            return
        if result.kind == "completed":
            await self._complete(ref, WorkflowValue(text=result.text))
        elif result.kind == "failed":
            if result.failure is None:
                raise RuntimeError("A failed agent attempt requires a failure report.")
            await self._fail(ref, result.failure)
        elif result.kind == "cancelled" and self._terminal is None:
            await self._fail(ref, FailureReport(ErrorClass.AGENT_NON_TRANSIENT, "agent pass was cancelled"))

    async def _run_join(self, ref: AttemptRef, value: WorkflowValue | None, sources: tuple[SourceValue, ...]) -> None:
        if not await self._write_input(ref, {"sources": [source_to_wire(source) for source in sources]}):
            return
        if value is not None:  # no user combine: the scheduler already folded the sources
            await self._complete(ref, value)
            return
        timeout = self._graph.nodes[ref.node_id].timeout
        try:
            combined = await self._worker.combine(ref, sources, timeout=timeout)
        except WorkerRpcError as exc:
            await self._write_diagnostics(ref, "combine", exc.stdout, exc.traceback)
            error_class = (
                ErrorClass.VALUE_TOO_LARGE if exc.code == ErrorCode.VALUE_TOO_LARGE else ErrorClass.EVALUATION_ERROR
            )
            await self._fail(ref, FailureReport(error_class, f"combine failed: {exc.message}"))
        except AttemptTimeout as exc:
            await self._write_diagnostics(ref, "combine", exc.stdout, exc.traceback)
            await self._fail(ref, FailureReport(ErrorClass.EVALUATION_ERROR, "combine did not finish in time."))
        else:
            await self._write_diagnostics(ref, "combine", combined.stdout)
            await self._complete(ref, combined.value)

    async def _evaluate_outgoing(self, decision: EvaluateOutgoing) -> None:
        ref = decision.ref
        try:
            decided = await self._worker.eval_outgoing(ref, decision.value, decision.edge_ids)
        except WorkerLostError as exc:
            await self._worker_lost(str(exc))
            return
        except WorkerRpcError as exc:
            await self._write_diagnostics(ref, "outgoing", exc.stdout, exc.traceback)
            message = f"{exc.code}: {exc.message}"
        except AttemptTimeout as exc:
            await self._write_diagnostics(ref, "outgoing", exc.stdout, exc.traceback)
            message = "edge condition did not finish in time."
        except (ProtocolError, ValueShapeError) as exc:
            await self._worker_lost(f"edge decisions are malformed: {exc}")
            return
        else:
            await self._write_diagnostics(ref, "outgoing", decided.stdout)
            await self._feed(lambda: self._scheduler.outgoing_evaluated(ref, decided.value))
            return
        await self._feed(lambda: self._scheduler.outgoing_failed(ref, message))

    async def _evaluate_until(self, decision: EvaluateLoopUntil) -> None:
        ref, iteration = decision.ref, decision.iteration
        try:
            verdict = await self._worker.eval_loop_until(ref, iteration, decision.value)
        except WorkerLostError as exc:
            await self._worker_lost(str(exc))
            return
        except WorkerRpcError as exc:
            await self._write_diagnostics(ref, "until", exc.stdout, exc.traceback, iteration=iteration)
            message = f"{exc.code}: {exc.message}"
        except AttemptTimeout as exc:
            await self._write_diagnostics(ref, "until", exc.stdout, exc.traceback, iteration=iteration)
            message = "loop condition did not finish in time."
        except (ProtocolError, ValueShapeError) as exc:
            await self._worker_lost(f"loop verdict is malformed: {exc}")
            return
        else:
            await self._write_diagnostics(ref, "until", verdict.stdout, iteration=iteration)
            await self._feed(lambda: self._scheduler.loop_until_evaluated(ref, iteration, verdict.value))
            return
        await self._feed(lambda: self._scheduler.loop_until_failed(ref, iteration, message))

    async def _backoff(self, decision: ScheduleRetry) -> None:
        await asyncio.sleep(decision.backoff)
        await self._feed(lambda: self._scheduler.backoff_elapsed(decision.ref))

    async def _deadline(self, timeout: float) -> None:
        await asyncio.sleep(timeout)
        self.cancel(reason=REASON_DEADLINE_EXCEEDED)

    async def _watch_worker(self) -> None:
        """A worker lost while no call is outstanding (a node awaiting retry, an agent-only stretch) still ends the run."""
        error = await self._worker.wait_lost()
        await self._worker_lost(str(error))

    async def _worker_lost(self, message: str) -> None:
        if not self._worker_error:
            self._worker_error = message
        await self._feed(self._scheduler.worker_lost)

    def _loop_settles(self, decision: LoopIteration) -> bool:
        """Whether this verdict makes the iteration's exit value the loop activation's output."""
        if decision.verdict is LoopVerdict.EXIT:
            return True
        loop = self._graph.nodes[decision.ref.node_id].loop
        return (
            decision.verdict is LoopVerdict.EXHAUSTED
            and loop is not None
            and loop.on_exhausted == ON_EXHAUSTED_CONTINUE
        )

    async def _settle_loop(self, ref: AttemptRef, value: WorkflowValue) -> None:
        """The loop activation's output, recorded like any node's: the value in full, then its final event."""
        if await self._write_output(ref, value):
            await self._record(self._journal.node_output(ref, WORKFLOW_OUTPUT_FINAL, 1, value.text))

    async def _complete(self, ref: AttemptRef, value: WorkflowValue, *, last_emit_ordinal: int = 0) -> None:
        if not await self._write_output(ref, value):
            return
        if (
            await self._record(self._journal.node_output(ref, WORKFLOW_OUTPUT_FINAL, last_emit_ordinal + 1, value.text))
            is None
        ):
            return
        await self._feed(lambda: self._scheduler.activation_completed(ref, value))

    async def _fail(self, ref: AttemptRef, failure: FailureReport) -> None:
        await self._feed(lambda: self._scheduler.activation_failed(ref, failure))

    # -- records ---------------------------------------------------------------------

    async def _record(self, write: Awaitable[int]) -> int | None:
        """One journal write; a storage failure becomes the run's fault instead of an exception."""
        try:
            return await write
        except WorkflowStorageFailed:
            self._storage_failed()
            return None

    async def _write_diagnostics(
        self, ref: AttemptRef, phase: str, stdout: CapturedOutput, traceback: str = "", *, iteration: int | None = None
    ) -> None:
        if iteration is None:
            snapshot = self._scheduler.snapshot(ref.activation_id)
            iteration = snapshot.iteration if snapshot is not None else 0
        await self._journal.store.offload(
            self._journal.store.write_node_diagnostics,
            ref.activation_id,
            ref.attempt,
            phase=phase,
            iteration=iteration,
            stdout=stdout.text,
            truncated=stdout.truncated,
            traceback=traceback,
        )

    async def _write_input(self, ref: AttemptRef, record: dict[str, Any]) -> bool:
        return await self._write_node_record(ref, NODE_RECORD_INPUT, record)

    async def _write_output(self, ref: AttemptRef, value: WorkflowValue) -> bool:
        # Retrying a loop's outgoing condition advances its attempt without rewriting this value.
        self._output_attempts[ref.activation_id] = ref.attempt
        return await self._write_node_record(ref, NODE_RECORD_OUTPUT, {"value": value_to_wire(value)})

    async def _write_emit(self, ref: AttemptRef, ordinal: int, text: str) -> bool:
        return await self._store(self._journal.store.append_node_emit, ref.activation_id, ref.attempt, ordinal, text)

    async def _write_node_record(self, ref: AttemptRef, kind: str, record: dict[str, Any]) -> bool:
        return await self._store(self._journal.store.write_node_value, ref.activation_id, ref.attempt, kind, record)

    async def _store[**P](self, write: Callable[P, object], *args: P.args, **kwargs: P.kwargs) -> bool:
        try:
            await self._journal.store.offload(write, *args, **kwargs)
        except WorkflowStorageFailed:
            self._storage_failed()
            return False
        return True

    def _storage_failed(self, error: str = "A node record could not be written.") -> None:
        """Queue the fault; the lock holder applies it after the current decisions, anyone else feeds it now.

        The loss is remembered either way: a terminal settled in the same batch (or already settled) no
        longer takes the fault as a scheduler input, and the run still finishes as ``storage_failed``.
        """
        if self._record_error is None:
            self._record_error = error
        if self._terminal is not None or self._deferred_fault is not None:
            return
        if self._lock.locked():
            self._deferred_fault = self._scheduler.storage_failed
        else:
            self._spawn(self._feed(self._scheduler.storage_failed))

    # -- shells and tasks --------------------------------------------------------------

    def _invocation_id(self, ref: AttemptRef) -> str:
        """The invocation identity of an agent activation: allocated once, kept across attempts."""
        invocation_id = self._invocations.get(ref.activation_id)
        if invocation_id is None:
            invocation_id = new_analytics_id()
            self._invocations[ref.activation_id] = invocation_id
        return invocation_id

    def set_approval_mode(self, mode: ApprovalMode) -> None:
        """Update live agent shells; shells opened later read the engine's launch policy."""
        for shell in list(self._shells.values()):
            shell.set_approval_mode(mode)

    def _shell_for(self, ref: AttemptRef) -> WorkflowAgentShell:
        shell = self._shells.get(ref.activation_id)
        if shell is None:
            binding = self._admitted.binding(ref.node_id)
            invocation_id = self._invocation_id(ref)
            shell = WorkflowAgentShell(
                binding=binding,
                node_id=ref.node_id,
                invocation_id=invocation_id,
                resources=self._resources,
                archive=AgentNodeArchive(
                    self._journal.store,
                    activation_id=ref.activation_id,
                    invocation_id=invocation_id,
                    profile_name=binding.agent.display_name or binding.agent.name,
                ),
            )
            self._shells[ref.activation_id] = shell
        return shell

    async def _close_shell(self, activation_id: str) -> None:
        await self._shells[activation_id].close()
        del self._shells[activation_id]

    def _spawn(self, coro: Awaitable[None]) -> asyncio.Task[None]:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._forget)
        return task

    def _forget(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            if self._terminal is None:
                self._unexpected_failure("run task was unexpectedly cancelled")
            return
        exc = task.exception()
        if isinstance(exc, WorkflowStorageFailed):
            self._storage_failed(str(exc))
        elif exc is not None:
            logger.error("workflow run %s: task failed", self._run_id, exc_info=exc)
            self._unexpected_failure(f"{type(exc).__name__}: {exc}")

    def _unexpected_failure(self, message: str) -> None:
        if not self._internal_error:
            self._internal_error = message
        if self._terminal is None:
            if self._cancel_task is not None and self._cancel_task.done():
                # Cancellation's own publisher failed. Stop work without publishing more node decisions.
                self._on_terminal(self._scheduler.finished or RunFinished(RunOutcome.CANCELLED, ()))
            else:
                self.cancel(reason=REASON_INTERNAL_ERROR)
        elif self._terminal.outcome is RunOutcome.COMPLETED:
            # A close task can fail after the last scheduler decision, before the terminal commit.
            self._terminal = RunFinished(RunOutcome.CANCELLED, self._terminal.outputs)
            self._reason = REASON_INTERNAL_ERROR

    # -- terminal ------------------------------------------------------------------------

    async def _teardown(self) -> None:
        """Drain tasks, close shells, and wait for the worker to exit; the terminal is recorded after."""
        current = asyncio.current_task()
        for watch in (self._deadline_task, self._worker_watch):
            if watch is not None:
                watch.cancel()  # the terminal is settled: nothing waits out the timer, and the close below is no fault
        live = [task for task in [*self._tasks, *self._aborts] if task is not current]
        if live:
            await asyncio.gather(*live, return_exceptions=True)
        # Every shell, opened or interrupted while opening: close() tolerates both.
        for activation_id, shell in list(self._shells.items()):
            try:
                await shell.close()
            except WorkflowStorageFailed as exc:
                self._storage_failed(str(exc))
            except Exception as exc:
                self._unexpected_failure(f"{type(exc).__name__}: {exc}")
                logger.warning(
                    "workflow run %s: agent shell %s did not close cleanly",
                    self._run_id,
                    activation_id,
                    exc_info=True,
                )
        self._shells.clear()
        await self._collect_run_output()
        await self._worker.close()

    async def _collect_run_output(self) -> None:
        if self._output_collected:
            return
        self._output_collected = True
        payload: dict[str, Any] = {"load": {"text": self._load_stdout.text, "truncated": self._load_stdout.truncated}}
        if self._worker.lost is None:
            try:
                async with asyncio.timeout(3.0):
                    native = await self._worker.native_output()
                payload["native"] = {"text": native.text, "dropped_bytes": native.dropped_bytes}
            except Exception as exc:
                payload["error"] = f"Native output could not be collected: {exc}"
                logger.warning("workflow run %s: native output unavailable", self._run_id, exc_info=True)
        else:
            payload["error"] = "Native output unavailable because the worker was lost."
        try:
            await self._journal.store.offload(self._journal.store.write_run_output, payload)
        except OSError, ValueError, TypeError:
            logger.warning("workflow run %s: run output could not be written", self._run_id, exc_info=True)

    async def _abandon(self, reason: str) -> WorkflowRunResult | None:
        """The run cannot continue on its own terms: converge as a cancel for *reason* and still tear down."""
        if self._terminal is None:
            if self._cause is None:
                self._reason = reason
                self._cause = _CANCEL_CAUSES.get(reason, AbortCause.OWNER_CLOSE)
            with contextlib.suppress(Exception):
                await self._feed(self._scheduler.cancel)
            if self._terminal is None:
                self._on_terminal(self._scheduler.finished or RunFinished(RunOutcome.CANCELLED, ()))
        with contextlib.suppress(Exception):
            await self._teardown()
        if self._terminal is None:
            return None
        with contextlib.suppress(Exception):
            return await self._finish()
        return None

    async def _finish(self) -> WorkflowRunResult:
        if self._result is not None:
            return self._result
        terminal = self._terminal
        if terminal is None:
            raise RuntimeError("Finishing a workflow run requires a terminal decision.")
        outputs = terminal.outputs
        duration = time.monotonic() - self._started
        summaries = [
            WorkflowOutputSummary(
                item.node_id,
                item.activation_id,
                summarize(item.value.text),
                attempt=self._output_attempts[item.activation_id],
            )
            for item in outputs
        ]
        error = terminal.error.message if terminal.error is not None else ""
        node_id = terminal.node_id or ""
        outcome = terminal.outcome
        if outcome is RunOutcome.WORKER_LOST:
            error = self._worker_error or str(self._worker.lost or "worker lost")
        elif self._reason == REASON_INTERNAL_ERROR:
            error = self._internal_error or error
            if outcome is RunOutcome.COMPLETED:
                outcome = RunOutcome.CANCELLED
        if self._record_error is not None:
            # A node record that failed to write after (or in the very batch of) the terminal: the run
            # is not the completed one its outputs claim, so the record, the event and the result all say so.
            outcome = RunOutcome.STORAGE_FAILED
            error = f"{self._record_error}\n{error}" if error else self._record_error
        try:
            await self._checkpoint()
        except Exception as exc:
            logger.exception("workflow run %s: final session checkpoint failed", self._run_id)
            outcome = RunOutcome.STORAGE_FAILED
            checkpoint_error = f"Session checkpoint could not be saved: {type(exc).__name__}: {exc}"
            error = f"{checkpoint_error}\n{error}" if error else checkpoint_error
        await self._journal.finish(
            outcome,
            outputs=summaries,
            duration=duration,
            node_id=node_id,
            error=error,
            reason=self._reason,
        )
        if self._journal.storage_failed:
            outcome = RunOutcome.STORAGE_FAILED
        self._result = WorkflowRunResult(
            run_id=self._run_id,
            outcome=outcome,
            outputs=outputs,
            node_id=node_id,
            error=error,
            reason=self._reason,
            duration=duration,
        )
        try:
            if self._record_end is not None:
                await self._record_end(self._result)
        finally:
            if self._trace is not None:
                self._trace.finished(outcome.value)
        return self._result
