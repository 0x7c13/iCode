# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The workflow shell around one agent-node activation.

One shell per activation: :meth:`WorkflowAgentShell.open` builds a private
kernel or ACP conversation for the node's agent and model binding, :meth:`run`
executes one pass per scheduler attempt, :meth:`abort` converges the in-flight
pass with a caller cause, and :meth:`close` unbinds and releases everything in
reverse. The shell is the third caller of the invoker contract after the main
turn and the sub-agent tool; it owns no retry loop of its own, because the
scheduler decides retries from the :class:`FailureReport` each attempt yields.

Kernel nodes are built like the chat agent — builtin, sub-agent, MCP and skill
tools plus the profile's auto-loaded memory — with no in-wire retry policy, and
the ``ask_user`` tool is dropped whenever the run is headless. No workflow
surface shows a per-child card, so a failed child ends at once instead of
pausing; the shell routes the engine's approval mode and the run's abort to
the node's own sub-agents.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal

from chrys.foundation.errors import is_retryable
from chrys.foundation.events.types import InvocationMessage, InvocationRetryAttempt, InvocationStarted
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.foundation.trajectory.context import TRAJECTORY_CONTEXT_KWARG, TrajectoryContext, workflow_node_actor
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.metadata import ensure_analytics_item_id
from chrys.kernel import AgentResponse, Message
from chrys.orchestration.invoker.acp import AcpConversation, AcpInvocationCounters
from chrys.orchestration.invoker.acp_protocol import AcpUpdateTranslator
from chrys.orchestration.invoker.child_history import active_input_message
from chrys.orchestration.invoker.contracts import (
    AbortCause,
    Aborted,
    ContinuationTicket,
    Failed,
    FailureDisposition,
    InvocationOutcome,
    Ok,
    OverlappingRun,
    PreparedClosed,
    RunIntent,
    RunRequest,
    StaleContinuation,
    Unbind,
    UnsupportedRequest,
)
from chrys.orchestration.invoker.evidence import Completeness, Count, InvocationEvidence
from chrys.orchestration.invoker.kernel import KernelConversation
from chrys.orchestration.invoker.origin import BoundEmitter
from chrys.orchestration.invoker.resources import Conversation, PassResources, PreparedAgent, finish_close
from chrys.orchestration.workflows.agent_archive import CoalescedCheckpoint
from chrys.orchestration.workflows.agent_node_build import (
    AcpNodeParts,
    AgentNodeCallbacks,
    AgentNodeResources,
    KernelNodeParts,
    build_acp_node,
    build_kernel_node,
)
from chrys.service.agent_middleware.events.hosted_tools import FinalSegment
from chrys.service.agent_middleware.events.intermediate_text import IntermediateTextBuffer
from chrys.service.agent_middleware.response_validation import hosted_commits_from_error
from chrys.service.session.history import SessionHistoryManager, stamp_history_item_ids
from chrys.service.session.sub_agent_logs import SubAgentLogStats
from chrys.service.workflows.scheduler import ErrorClass, FailureReport

if TYPE_CHECKING:
    from chrys.foundation.retry import RetryAttemptInfo
    from chrys.orchestration.workflows.agent_archive import AgentNodeArchive
    from chrys.service.approval.policy import ApprovalMode
    from chrys.service.workflows.admission import AgentBinding

logger = logging.getLogger(__name__)

# The shell whose pass the current task runs under; every task the kernel starts for a pass inherits it.
_PASS_SHELL: ContextVar[WorkflowAgentShell | None] = ContextVar("chrys.workflow.pass_shell", default=None)

ABORT_GRACE: Final = 10.0
"""Seconds a timed-out pass may take to converge after its abort before the run task is cancelled."""
_TERMINAL_CAUSES: Final = frozenset({AbortCause.USER_CANCEL, AbortCause.RUN_TERMINAL, AbortCause.OWNER_CLOSE})

AttemptKind = Literal["completed", "failed", "cancelled"]


@dataclass(frozen=True, slots=True)
class AgentAttemptResult:
    """How one pass ended: ``completed`` with the response text, ``failed`` with a report, or ``cancelled``."""

    kind: AttemptKind
    text: str = ""
    failure: FailureReport | None = None


class WorkflowAgentShell:
    """One agent activation: a private conversation, one pass per attempt, and the caller's abort latch.

    The shell is both the conversation's operation binding (``request_close`` /
    ``drained``) and the kernel's pass observer; both interfaces are small and
    keep the cancel story in one place.
    """

    def __init__(
        self,
        *,
        binding: AgentBinding,
        node_id: str,
        invocation_id: str,
        resources: AgentNodeResources,
        archive: AgentNodeArchive,
    ) -> None:
        self._binding = binding
        self._node_id = node_id
        self._invocation_id = invocation_id
        self._resources = resources
        self._archive = archive
        self._archive_attempt = 1
        self._checkpoint: CoalescedCheckpoint | None = None
        self._stats = SubAgentLogStats()
        self._intermediate_buffer = IntermediateTextBuffer()
        self._acp_translators: list[AcpUpdateTranslator] = []
        self._acp_counters = AcpInvocationCounters()
        self._display_name = binding.agent.display_name or binding.agent.name
        self.origin = InvocationOrigin("workflow_node", resources.session_id, invocation_id, None)
        self._emitter = BoundEmitter(resources.bus, self.origin)
        self._prepared = PreparedAgent()
        self._unbind: Unbind | None = None
        self._parts: KernelNodeParts | AcpNodeParts | None = None
        self._is_prepared = False
        self._abort_cause: AbortCause | None = None
        self._run_task: asyncio.Task[InvocationOutcome] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._ticket: ContinuationTicket | None = None
        self._evidence = InvocationEvidence(invocation_id)
        self._passes = 0
        self._usage_total = 0
        self._seed_item_id = new_analytics_id()
        self._prompt = ""
        self._active_run_input: list[Any] = []
        self._pass_start_index = 0

    # ------------------------------------------------------------------ identity

    @property
    def is_prepared(self) -> bool:
        """Whether this activation opened, even if its backend retired before retry."""
        return self._is_prepared

    @property
    def _backend(self) -> KernelConversation | AcpConversation | None:
        return self._parts.backend if self._parts is not None else None

    @property
    def invocation_id(self) -> str:
        return self._invocation_id

    def abort_cause(self) -> AbortCause | None:
        """The latched caller cause; also the kernel observer's abort-cause port."""
        return self._abort_cause

    @property
    def evidence(self) -> InvocationEvidence:
        """Every converged pass of this activation, accumulated."""
        return self._evidence

    # ------------------------------------------------------------------ lifecycle

    async def open(self, prompt: str) -> None:
        """Build the conversation, bind this shell as its operation, and announce the invocation."""
        await self._open_backend()
        self._is_prepared = True
        await self._emitter.publish(
            InvocationStarted(
                origin=self.origin,
                agent_name=self._display_name,
                tool_name=self._node_id,
                session_id=self._session_id,
                opening_prompt=self._user_prompt(prompt),
            )
        )

    async def _open_backend(self) -> None:
        conversation = await self._prepared.open(self._open_conversation)
        self._unbind = conversation.bind_operation(self)

    async def _retire_backend(self) -> None:
        """End a backend lifetime without ending the activation or its accounting."""
        unbind, self._unbind = self._unbind, None
        if unbind is not None:
            unbind()
        await self._prepared.aclose()
        self._prepared = PreparedAgent()
        self._parts = None
        self._ticket = None

    async def close(self) -> None:
        """Release the conversation, then drain the session's usage tail (including other publishers).

        Safe to call more than once. The shared drain keeps session persistence behind usage delivery.
        """
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(), name="chrys.workflow.agent.close")
        await finish_close(self._close_task)

    async def _close(self) -> None:
        _PASS_SHELL.set(self)
        unbind, self._unbind = self._unbind, None
        if unbind is not None:
            unbind()
        await self._flush_progress()
        try:
            await self._prepared.aclose()
            if self._checkpoint is not None:
                await self._checkpoint.close()
        finally:
            await self._resources.usage_publisher.drain()

    async def _flush_progress(self) -> None:
        if isinstance(self._parts, KernelNodeParts):
            try:
                await self._parts.events.flush_progress()
            except Exception:
                logger.debug("workflow node %s: progress flush failed", self._node_id, exc_info=True)

    async def run(
        self,
        prompt: str,
        *,
        timeout: float | None,
        trajectory_context: TrajectoryContext | None,
        attempt: int = 1,
    ) -> AgentAttemptResult:
        """Archive this attempt after its pass has drained, including cancellation."""
        self._checkpoint = CoalescedCheckpoint(self._save_transcript)
        self._archive_attempt = attempt
        self._acp_translators.clear()
        result: AgentAttemptResult | None = None
        error = ""
        try:
            if self._backend is None:
                await self._open_backend()
            result = await self._run(prompt, timeout=timeout, trajectory_context=trajectory_context)
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            await finish_close(asyncio.create_task(self._finish_attempt(result, error)))

    async def _finish_attempt(self, result: AgentAttemptResult | None, error: str) -> None:
        # Delivery belongs to the pass boundary, even while the activation is
        # retained for outgoing-condition evaluation and possible retries.
        _PASS_SHELL.set(self)
        await self._flush_progress()
        await self._resources.usage_publisher.drain()
        assert self._checkpoint is not None
        await self._checkpoint.close()
        await self._save_transcript(
            status=result.kind if result is not None else "failed" if error else "cancelled",
            error=result.failure.message if result is not None and result.failure is not None else error,
        )
        if (
            isinstance(self._backend, AcpConversation)
            and result is not None
            and result.failure is not None
            and result.failure.error_class is ErrorClass.AGENT_TIMEOUT
        ):
            await self._retire_backend()

    async def _run(
        self, prompt: str, *, timeout: float | None, trajectory_context: TrajectoryContext | None
    ) -> AgentAttemptResult:
        """Execute one pass for *prompt*; a retry pass continues the failed pass's history when it can."""
        backend = self._backend
        if backend is None:
            raise RuntimeError("WorkflowAgentShell.run before open")
        if self._run_task is not None and not self._run_task.done():
            raise RuntimeError("an agent pass is already running for this activation")
        if self._abort_cause is not None:
            return AgentAttemptResult("cancelled")
        self._prompt = prompt
        if trajectory_context is not None:
            trajectory_context = trajectory_context.with_actor(
                workflow_node_actor(self._resources.session_id, self._invocation_id)
            )
        if isinstance(self._parts, KernelNodeParts):
            self._parts.trace.context = trajectory_context
        if isinstance(backend, AcpConversation):
            backend.bind_trajectory(
                trajectory_context,
                boundary_operation_id=trajectory_context.innermost_model_operation_id
                if trajectory_context is not None
                else None,
            )
        if trajectory_context is not None and isinstance(self._parts, KernelNodeParts):
            self._parts.run_kwargs["client_kwargs"] = {
                **self._parts.run_kwargs.get("client_kwargs", {}),
                TRAJECTORY_CONTEXT_KWARG: trajectory_context,
            }
        request = self._next_request(backend)
        self._passes += 1
        run_task = asyncio.create_task(self._pass(backend, request))
        self._run_task = run_task
        timed_out = False
        try:
            if timeout is None:
                outcome = await run_task
            else:
                done, _pending = await asyncio.wait({run_task}, timeout=timeout)
                if run_task not in done:
                    timed_out = True
                    await self.abort(AbortCause.CALLER_TIMEOUT)
                    done, _pending = await asyncio.wait({run_task}, timeout=ABORT_GRACE)
                    if run_task not in done:
                        run_task.cancel()
                outcome = await run_task
        except asyncio.CancelledError:
            # The runner's attempt task was cancelled around us: converge the pass before leaving.
            if self._abort_cause is None:
                self._abort_cause = AbortCause.RUN_TERMINAL
            backend.latch_abort(self._abort_cause)
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
            raise
        except (StaleContinuation, UnsupportedRequest, OverlappingRun, PreparedClosed) as exc:
            self._ticket = None
            return AgentAttemptResult(
                "failed", failure=FailureReport(ErrorClass.AGENT_NON_TRANSIENT, f"{type(exc).__name__}: {exc}")
            )
        finally:
            if timed_out and self._abort_cause is AbortCause.CALLER_TIMEOUT:
                self._abort_cause = None  # a timeout ends the pass, not the activation
        return await self._project(outcome, timeout=timeout)

    async def _pass(self, backend: KernelConversation | AcpConversation, request: RunRequest) -> InvocationOutcome:
        """Run the pass marked as this shell's: the kernel's tasks for it inherit the mark with the context."""
        _PASS_SHELL.set(self)
        return await backend.run(request)

    def holds(self, task: asyncio.Task[Any] | None) -> bool:
        """True when *task* is the live pass or one started under it: the run waits for all of them.

        The kernel runs the pass's attempt and each of its tool calls on tasks of its own, and the usage
        callback schedules the progress publishes on others; an event published from any of them reaches
        its subscribers on that task.
        """
        if task is None:
            return False
        return task is self._run_task or task.get_context().get(_PASS_SHELL) is self

    def set_approval_mode(self, mode: ApprovalMode) -> None:
        """Apply the engine's launch policy to the live pass's subsequent tool calls, sub-agents included."""
        if isinstance(self._parts, KernelNodeParts):
            self._parts.approval.set_approval_mode(mode)
            if self._parts.sub_agent_tools is not None:
                self._parts.sub_agent_tools.set_approval_mode(mode)

    def _latch(self, cause: AbortCause) -> None:
        if self._abort_cause is None or self._abort_cause not in _TERMINAL_CAUSES:
            self._abort_cause = cause

    async def abort(self, cause: AbortCause) -> None:
        """Latch *cause* and converge the in-flight pass, if any, tearing its live sub-agents down first."""
        self._latch(cause)
        backend = self._backend
        if backend is None:
            return
        if isinstance(self._parts, KernelNodeParts) and self._parts.sub_agent_tools is not None:
            # Running children cancel their attempt, so long inner tool calls stop before the pass
            # task itself is cancelled.
            await self._parts.sub_agent_tools.cascade_abort_all()
        backend.latch_abort(cause)
        handle = backend.active_handle
        if handle is not None:
            await backend.abort(handle, cause)

    # ------------------------------------------------------------------ OperationBinding

    def request_close(self, cause: AbortCause) -> None:
        self._latch(cause)
        if self._backend is not None:
            self._backend.latch_abort(cause)
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()

    @property
    def drained(self) -> Awaitable[None]:
        return self._drain()

    async def _drain(self) -> None:
        task = self._run_task
        if task is not None and not task.done():
            await asyncio.gather(task, return_exceptions=True)

    # ------------------------------------------------------------------ KernelPassObserver

    async def begin(self, request: RunRequest, resources: PassResources) -> None:
        self._intermediate_buffer.drain()
        assert isinstance(self._parts, KernelNodeParts)
        self._pass_start_index = len(self._parts.history.messages())
        resources.begin()
        self._active_run_input = list(request.messages)
        await self._save_transcript()

    async def succeeded(self, response: AgentResponse[Any], /) -> None:
        if isinstance(self._parts, KernelNodeParts):
            await self._parts.events.reconcile_hosted_response(response.messages)

    async def failed(self, error: Exception, /) -> None:
        assert isinstance(self._parts, KernelNodeParts)
        self._parts.history.repair_after_failure(self._active_run_input, self._pass_start_index)

    async def finished(self) -> None:
        # A failed or interrupted pass publishes no outcome; its buffered text
        # belongs to this pass's transcript, not the next one.
        if isinstance(self._parts, KernelNodeParts):
            await self._parts.events.finish_intermediate_text()

    async def interrupt(self) -> None:
        pass

    def cancelled(self) -> None:
        pass

    # ------------------------------------------------------------------ construction

    async def _open_conversation(self, conversation: Conversation) -> Conversation:
        callbacks = AgentNodeCallbacks(
            observer=self,
            publish_intermediate=self._publish_intermediate,
            checkpoint=self._mark_dirty,
            usage=self._on_usage,
            side_call_usage=self._on_side_call_usage,
            validation_retry=self._publish_validation_retry,
            service_retry=self._publish_service_retry,
            interruptible_sleep=self._interruptible_sleep,
            acp_usage=self._on_acp_usage,
            adopt_translator=self._adopt_acp_translator,
            acp_counters=self._acp_counters,
        )
        if self._binding.agent.acp is not None:
            self._parts = build_acp_node(
                conversation,
                binding=self._binding,
                node_id=self._node_id,
                invocation_id=self._invocation_id,
                res=self._resources,
                emitter=self._emitter,
                archive=self._archive,
                callbacks=callbacks,
            )
        else:
            self._parts = await build_kernel_node(
                conversation,
                binding=self._binding,
                node_id=self._node_id,
                invocation_id=self._invocation_id,
                res=self._resources,
                emitter=self._emitter,
                archive=self._archive,
                callbacks=callbacks,
                intermediate_buffer=self._intermediate_buffer,
                stats=self._stats,
            )
        # Construction awaits sub-agent registration, MCP connection and skill discovery; a mode
        # change that arrived meanwhile found no parts to apply to, so the launch policy is re-read.
        self.set_approval_mode(self._resources.approval_mode())
        return conversation

    def _on_acp_usage(
        self,
        total: int,
        input_tokens: int,
        output_tokens: int,
        local_tokens: int,
        calibration_ratio: float,
        system_overhead_tokens: int,
        cache_hit_tokens: int | None,
        max_context_tokens: int | None,
        agent_profile: str,
        usage_source_id: str,
        total_usage_tokens: int,
    ) -> None:
        # ACP reports each transport once; the archive and graph summarize the whole activation.
        self._usage_total += total_usage_tokens
        self._stats.record_usage(total, self._usage_total)
        self._resources.usage_publisher.accumulate_invocation_usage(
            total,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            local_tokens=local_tokens,
            calibration_ratio=calibration_ratio,
            system_overhead_tokens=system_overhead_tokens,
            cache_hit_tokens=cache_hit_tokens,
            max_context_tokens=max_context_tokens,
            agent_profile=agent_profile,
            usage_source_id=usage_source_id,
            authoritative_total=total_usage_tokens,
        )

    # ------------------------------------------------------------------ passes

    def _next_request(self, backend: KernelConversation | AcpConversation) -> RunRequest:
        ticket = self._ticket
        if isinstance(backend, AcpConversation):
            self._ticket = None
            return RunRequest(
                self._seed_input(), RunIntent.RETRY if ticket is not None else RunIntent.FRESH, self.origin, ticket
            )
        if ticket is not None and backend.continuation_is_live(ticket, self.origin):
            self._ticket = None
            assert isinstance(self._parts, KernelNodeParts)
            return RunRequest(self._parts.history.retry_input(self._seed_input), RunIntent.RETRY, self.origin, ticket)
        self._ticket = None
        if self._passes:
            backend.history_state = {}  # a fresh pass starts from the prompt, not from a stale transcript
        return RunRequest(self._seed_input(), RunIntent.FRESH, self.origin, None)

    async def _project(self, outcome: InvocationOutcome, *, timeout: float | None) -> AgentAttemptResult:
        self._evidence = self._evidence.add(outcome.effects)
        if isinstance(outcome, Ok):
            self._ticket = None
            if isinstance(outcome.backend_payload, AgentResponse):
                # The node's value is the final segment, as a sub-agent's result is; what the agent wrote
                # between tool calls is already in the transcript.
                segment = FinalSegment.of(outcome.backend_payload)
                value, transcript = segment.result, segment.transcript
            else:
                # An ACP result is left alone, since the profile's result mode already chose its extent.
                value = transcript = _response_text(outcome)
                if isinstance(self._backend, AcpConversation) and self._backend.transcript_final_text is not None:
                    transcript = self._backend.transcript_final_text
            await self._emitter.publish(
                InvocationMessage(
                    origin=self.origin,
                    agent_name=self._display_name,
                    text=transcript,
                    is_final=True,
                    session_id=self._session_id,
                )
            )
            return AgentAttemptResult("completed", text=value)
        if isinstance(outcome, Failed):
            self._ticket = outcome.continuation
            transient = outcome.exception is not None and is_retryable(outcome.exception)
            # The kernel's whole-run gate, kept at this boundary: provider-hosted tool calls the failed
            # exchange already executed (named by the validation error, or by the middleware's probe when the
            # exchange dropped mid-stream) would run a second time under a new pass, so none is approved.
            hosted = outcome.effects.hosted_observed.observed > 0 or (
                outcome.exception is not None and bool(hosted_commits_from_error(outcome.exception))
            )
            approved = (
                transient
                and not hosted
                and outcome.continuation is not None
                and outcome.disposition is FailureDisposition.CALLER_DECISION
            )
            error_class = ErrorClass.AGENT_TRANSIENT if transient else ErrorClass.AGENT_NON_TRANSIENT
            return AgentAttemptResult(
                "failed", failure=FailureReport(error_class, outcome.error, backend_approved=approved)
            )
        assert isinstance(outcome, Aborted)
        self._ticket = outcome.continuation
        if outcome.cause is AbortCause.CALLER_TIMEOUT:
            limit = f"{timeout:g}s" if timeout is not None else "its deadline"
            return AgentAttemptResult(
                "failed",
                failure=FailureReport(
                    ErrorClass.AGENT_TIMEOUT,
                    f"no response within {limit}",
                    backend_approved=self._nothing_executed(),
                ),
            )
        return AgentAttemptResult("cancelled")

    def _nothing_executed(self) -> bool:
        """Whether every converged pass answered no local tool and observed no hosted commit, exactly."""
        evidence = self._evidence
        return (
            _exact_zero(evidence.local_answered)
            and _exact_zero(evidence.hosted_observed)
            and evidence.external_stateful is False
        )

    async def _interruptible_sleep(self, seconds: int) -> bool:
        for _ in range(max(0, seconds)):
            if self._abort_cause is not None:
                return True
            await asyncio.sleep(1)
        return self._abort_cause is not None

    # ------------------------------------------------------------------ history

    async def _mark_dirty(self) -> None:
        assert self._checkpoint is not None
        self._checkpoint.changed()

    async def _adopt_acp_translator(self, translator: AcpUpdateTranslator) -> None:
        self._acp_translators.append(translator)
        if self._checkpoint is not None:
            translator.on_audit_changed = self._checkpoint.changed
        await self._save_transcript()

    async def _save_transcript(self, *, status: str = "running", error: str = "") -> None:
        state = None
        acp_state = None
        if isinstance(self._backend, AcpConversation):
            acp_state = {
                "prompt": self._user_prompt(self._prompt),
                "translated_updates": [
                    {**item, "attempt": translator.attempt}
                    for translator in self._acp_translators
                    for item in translator.translated_updates
                ],
                "successful_attempt": self._backend.transport_ordinal if status == "completed" else None,
            }
            self._stats.tool_call_count = max(
                (self._backend.completed_tool_calls, *(t.completed_count for t in self._acp_translators))
            )
            self._stats.usage_unreported_attempts = self._backend.usage_unreported_attempts
        elif isinstance(self._parts, KernelNodeParts):
            # Keep message identities while merging into a detached list. This
            # reuses the same recovery merge as Chat without mutating the
            # active model history or deduplicating repeated provider call IDs.
            state = dict(self._parts.history.state())
            state["messages"] = list(state.get("messages", []))
            input_message = active_input_message(self._active_run_input)
            if not state["messages"] and input_message is not None:
                state["messages"].append(input_message)
            manager = SessionHistoryManager()
            manager.bind(state)
            manager.merge_loop_messages(self._parts.history.recorder, insert_index=self._pass_start_index)
            stamp_history_item_ids(state)
        await self._archive.write(
            attempt=self._archive_attempt,
            status=status,
            error=error,
            state=state,
            acp_state=acp_state,
            stats=self._stats,
        )
        if status != "running" and isinstance(self._parts, KernelNodeParts) and self._parts.sub_agent_tools is not None:
            # Children's control records go once the attempt's terminal transcript is on disk, as after
            # a chat turn's save; a running checkpoint is best-effort and may predate a child's result.
            self._parts.sub_agent_tools.finalize_pending_cleanups()

    def _user_prompt(self, prompt: str) -> str:
        """The same user text feeds the conversation and its live presentation."""
        if self._binding.agent.acp is not None and self._binding.instructions_suffix:
            return f"{prompt}\n\n{self._binding.instructions_suffix}"
        return prompt

    def _seed_input(self) -> list[Any]:
        message = Message("user", [self._user_prompt(self._prompt)])
        ensure_analytics_item_id(message.additional_properties, item_id=self._seed_item_id)
        return [message]

    # ------------------------------------------------------------------ events

    @property
    def _session_id(self) -> str | None:
        return self._resources.session_id or None

    async def _publish_intermediate(self, text: str) -> None:
        self._intermediate_buffer.new_batch()
        await self._emitter.publish(
            InvocationMessage(
                origin=self.origin,
                agent_name=self._display_name,
                text=text,
                is_final=False,
                is_intermediate=True,
                session_id=self._session_id,
            )
        )

    async def _publish_validation_retry(self, info: RetryAttemptInfo) -> None:
        await self._emitter.publish(
            InvocationRetryAttempt(
                scope="wire",
                origin=self.origin,
                agent_name=self._display_name,
                message=f"Invalid response: {info.reason}",
                attempt=info.attempt,
                max_attempts=info.max_attempts,
                delay_seconds=int(info.delay_seconds),
                session_id=self._session_id,
            )
        )

    async def _publish_service_retry(
        self, message: str, attempt: int, max_attempts: int, delay_seconds: int, exc: BaseException
    ) -> None:
        await self._emitter.publish(
            InvocationRetryAttempt(
                scope="run",
                origin=self.origin,
                agent_name=self._display_name,
                message=message,
                attempt=attempt,
                max_attempts=max_attempts,
                delay_seconds=delay_seconds,
                session_id=self._session_id,
            )
        )

    def _on_side_call_usage(self, usage_details: Mapping[str, Any]) -> None:
        """Charge summary generation to this node without replacing its context-window reading."""
        self._resources.usage_publisher.accumulate_side_call_usage(
            usage_details,
            agent_profile=self._binding.agent.name,
            usage_source_id=f"{self._invocation_id}:last_words",
            max_context_tokens=self._binding.model.max_context_tokens if self._binding.model is not None else None,
        )
        self._usage_total += int(usage_details.get("total_token_count") or 0)
        if isinstance(self._parts, KernelNodeParts):
            self._parts.events.record_usage(self._stats.total_tokens, self._usage_total)

    # Positional args mirror UsageTrackingMiddleware._fire_callback. New fields must be named and
    # forwarded here as well: *_rest tolerates extra arguments but does not account for their values.
    def _on_usage(
        self,
        total: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        local_tokens: int = 0,
        calibration_ratio: float = 1.0,
        system_overhead_tokens: int = 0,
        cache_hit_tokens: int | None = None,
        _calibration_initialized: bool = False,
        use_local_context_estimate: bool = False,
        *_rest: Any,
    ) -> None:
        self._usage_total += total
        if isinstance(self._parts, KernelNodeParts):
            self._parts.events.record_usage(total, self._usage_total)
        self._resources.usage_publisher.accumulate_invocation_usage(
            total,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            local_tokens=local_tokens,
            calibration_ratio=calibration_ratio,
            system_overhead_tokens=system_overhead_tokens,
            cache_hit_tokens=cache_hit_tokens,
            max_context_tokens=self._binding.model.max_context_tokens if self._binding.model is not None else None,
            agent_profile=self._binding.agent.name,
            usage_source_id=self._invocation_id,
            use_local_context_estimate=use_local_context_estimate,
        )


def _exact_zero(count: Count) -> bool:
    return count.completeness is Completeness.EXACT and count.observed == 0


def _response_text(outcome: Ok) -> str:
    return "".join(
        segment.text for segment in outcome.segments if segment.type == "text" and isinstance(segment.text, str)
    ).strip()
