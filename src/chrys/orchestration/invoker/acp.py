# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ACP transport attempts, pass outcomes, and cancellation-safe teardown."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from chrys.foundation.events.types import (
    InvocationRetryAttempt,
)
from chrys.foundation.models.invocations import InvocationOrigin, PassHandle
from chrys.foundation.trajectory.context import TrajectoryContext
from chrys.foundation.trajectory.event_types import RetryMode, RetryReason
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.kernel import Content
from chrys.orchestration.invoker.contracts import (
    AbortCause,
    Aborted,
    AbortResult,
    ContinuationCapability,
    ContinuationTicket,
    Failed,
    FailureCategory,
    FailureDisposition,
    InvocationOutcome,
    Ok,
    OverlappingRun,
    PreparedClosed,
    RunRequest,
    StaleContinuation,
    StopCause,
    UnsupportedRequest,
    UsageDelta,
    validate_request,
)
from chrys.orchestration.invoker.evidence import UNKNOWN_COUNT, PassEvidence
from chrys.orchestration.invoker.origin import BoundEmitter
from chrys.service.acp_client import AcpAgentClient
from chrys.service.acp_client.errors import (
    AcpAuthRequiredError,
    AcpConfigError,
    AcpConnectError,
    AcpIdleTimeoutError,
    AcpRefusalError,
    AcpSpawnError,
    AcpTransportError,
    is_remote_error,
)
from chrys.service.acp_client.spec import AcpAgentSpec, AcpPromptUsage
from chrys.service.trajectory.retries import RetryBackoffTrace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Coroutine

    from chrys.foundation.events.bus import EventBus

from chrys.foundation.platform.files import surrogate_safe_text
from chrys.foundation.util.once_close import finish_close

from .acp_protocol import AcpPermissionBroker, AcpUpdateTranslator, drain_acp_task, preview_text

logger = logging.getLogger(__name__)
_BACKOFF = (3, 7, 15, 30, 60)


# Terminal verdicts a new transport session can still overturn: the connect budget ran out.
_TRANSPORT_FAILURE_KINDS = frozenset({"sub_agent_acp_setup"})


@dataclass(frozen=True, slots=True)
class AcpExecutionResult:
    """Explicit transport verdict; the caller owns its terminal projection."""

    text: str
    succeeded: bool
    error_kind: str = ""
    exception: Exception | None = None

    @classmethod
    def failure(cls, kind: str, message: str, exception: Exception | None = None) -> AcpExecutionResult:
        safe = surrogate_safe_text(message)
        text = safe if safe.startswith("Error:") else f"Error: {safe}"
        return cls(text, succeeded=False, error_kind=kind, exception=exception)

    @property
    def category(self) -> FailureCategory:
        if self.error_kind in _TRANSPORT_FAILURE_KINDS:
            return FailureCategory.TRANSPORT
        return FailureCategory.DEFINITIVE


@dataclass(slots=True)
class AcpInvocationCounters:
    """Cumulative invocation facts retained when a caller replaces an aborted conversation."""

    pass_ordinal: int = 0
    input_spend: int = 0
    output_spend: int = 0
    transport_ordinal: int = 0
    retry_attempts_total: int = 0
    completed_calls: int = 0
    spend: int = 0
    usage_unreported: int = 0
    last_accounted_attempt: int = 0
    latest_context_tokens: int = 0


class AcpConversation:
    """One ACP logical invocation with fresh transport sessions on manual retry."""

    def __init__(
        self,
        *,
        tool_name: str,
        agent_name: str,
        prompt: str,
        origin: InvocationOrigin,
        spec_factory: Callable[[int], AcpAgentSpec],
        broker: AcpPermissionBroker,
        event_bus: EventBus | None,
        terminal_projection: Callable[[AcpExecutionResult], None],
        pass_started: Callable[[], None],
        session_id: str | None = None,
        result_mode: Literal["last_segment", "transcript"] = "last_segment",
        usage_callback: Callable[..., None] | None = None,
        attempt_callback: Callable[[int, Any, AcpUpdateTranslator], Awaitable[None]] | None = None,
        translator_callback: Callable[[AcpUpdateTranslator], Awaitable[None]] | None = None,
        counters: AcpInvocationCounters | None = None,
        max_connect_retries: int = 5,
        backoff_schedule: tuple[int, ...] = _BACKOFF,
        trajectory_context: TrajectoryContext | None = None,
        trajectory_boundary_operation_id: str | None = None,
    ) -> None:
        self._counters = counters if counters is not None else AcpInvocationCounters()
        self._invocation_id = origin.invocation_id
        self.origin = origin
        # The admitted request's origin: a later workflow attempt of this invocation publishes as itself.
        self._pass_origin = origin
        self._emitter = BoundEmitter(event_bus, self.origin)
        self._conversation_id = new_analytics_id()
        self._state_generation = 0
        self._active_handle: PassHandle | None = None
        self._ticket: ContinuationTicket | None = None
        self._pass_stateful: bool | None = False
        self._pass_cause: AbortCause | None = None
        self._tool_name = tool_name
        self._agent_name = agent_name
        self._prompt = prompt
        self._spec_factory = spec_factory
        self._broker = broker
        self._bus = event_bus
        self._session_id = session_id
        self._result_mode = result_mode
        self._usage_callback = usage_callback
        self._attempt_callback = attempt_callback
        self._translator_callback = translator_callback
        self._max_connect_retries = max_connect_retries
        self._backoff = backoff_schedule
        self._owner_close_cause: AbortCause | None = None
        self._cascade_requested = False
        self._cascade_event = asyncio.Event()
        self._active_client: AcpAgentClient | None = None
        self._cancel_watchdog: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._last_error = ""
        self._last_exception: Exception | None = None
        self._last_category = FailureCategory.TRANSPORT
        self._diagnostic_path: str | None = None
        self._stop_reason = ""
        self._transcript_final_text: str | None = None
        self._last_spec: AcpAgentSpec | None = None
        self._trajectory_context = trajectory_context
        self._trajectory_boundary_operation_id = trajectory_boundary_operation_id
        self._terminal_projection = terminal_projection
        self._pass_started = pass_started
        self._pass_done: asyncio.Event | None = None
        self._close_task: asyncio.Task[None] | None = None

    def _terminal(self, result: AcpExecutionResult) -> AcpExecutionResult:
        self._terminal_projection(result)
        return result

    def bind_trajectory(self, context: TrajectoryContext | None, *, boundary_operation_id: str | None) -> None:
        """Rebind an idle conversation and its permission waits to the caller's next attempt."""
        if self._active_handle is not None:
            raise OverlappingRun("Cannot rebind trajectory during an active ACP pass")
        self._trajectory_context = context
        self._trajectory_boundary_operation_id = boundary_operation_id
        self._broker.bind_trajectory(context, boundary_operation_id=boundary_operation_id)

    @property
    def prompt_preview(self) -> str:
        return preview_text(self._prompt)

    @property
    def retry_attempts(self) -> int:
        return self._counters.retry_attempts_total

    @property
    def diagnostic_path(self) -> str | None:
        return self._diagnostic_path

    def latch_abort(self, cause: AbortCause) -> None:
        if cause is AbortCause.OWNER_CLOSE:
            self._owner_close_cause = cause
        self._pass_cause = cause
        self._cascade_requested = True
        self._cascade_event.set()
        self._broker.mark_aborted()

    async def cancel_transport(self, cause: AbortCause, *, after_permissions: Callable[[], None] | None = None) -> None:
        """Latch permission cancellation before scheduling the remote cancel."""
        self.latch_abort(cause)
        self._start_background(self._broker.cancel_pending_waits())
        if after_permissions is not None:
            after_permissions()
        client = self._active_client
        if client is not None:
            self._start_background(self._cancel_client(client))
            if self._cancel_watchdog is None or self._cancel_watchdog.done():
                self._cancel_watchdog = asyncio.create_task(self._force_close_after_grace(client))

    async def abort(self, handle: PassHandle, cause: AbortCause) -> AbortResult:
        if self._active_handle is not handle:
            return AbortResult.ALREADY_CONVERGED
        await self.cancel_transport(cause)
        return AbortResult.REQUESTED

    async def drain_cancellation(self) -> None:
        pending = tuple(self._background_tasks)
        if pending:
            await drain_acp_task(asyncio.gather(*pending, return_exceptions=True))

    async def finish_transport_watchdog(self) -> None:
        watchdog = self._cancel_watchdog
        if watchdog is not None and not watchdog.done():
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        await finish_close(self._close_task)

    async def _close(self) -> None:
        self.latch_abort(AbortCause.OWNER_CLOSE)
        done = self._pass_done
        if done is not None:
            await self.cancel_transport(AbortCause.OWNER_CLOSE)
            # The caller may itself await aclose() after run() returns.
            await done.wait()
        await self.finish_transport_watchdog()
        await self.drain_cancellation()
        await self._broker.close()

    @property
    def total_usage_tokens(self) -> int:
        return self._counters.spend

    @property
    def usage_unreported_attempts(self) -> int:
        return self._counters.usage_unreported

    @property
    def latest_context_tokens(self) -> int:
        return self._counters.latest_context_tokens

    @property
    def completed_tool_calls(self) -> int:
        return self._counters.completed_calls

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def last_spec(self) -> AcpAgentSpec | None:
        return self._last_spec

    @property
    def transport_ordinal(self) -> int:
        """Current monotonically increasing ACP execution attempt."""
        return self._counters.transport_ordinal

    @property
    def transcript_final_text(self) -> str | None:
        """Final assistant segment not already published through ``InvocationMessage``."""
        return self._transcript_final_text

    def _start_background(self, awaitable: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(awaitable)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _cancel_client(self, client: AcpAgentClient) -> None:
        with contextlib.suppress(Exception):
            await client.cancel()

    async def _force_close_after_grace(self, client: AcpAgentClient) -> None:
        await asyncio.sleep(5)
        await asyncio.shield(client.force_close())

    @staticmethod
    async def _force_close_cancellation_safe(client: AcpAgentClient) -> None:
        """Finish teardown before propagating any cancellation."""
        await drain_acp_task(asyncio.create_task(client.force_close()))

    @staticmethod
    async def _flush_interrupted_cancellation_safe(translator: AcpUpdateTranslator) -> None:
        """Run the terminal-flush sweep to completion before propagating cancel.

        The post-teardown sweep is the last guarantee that every started tool
        call gets a terminal result. A bare ``await`` would let a second
        interrupt landing mid-sweep tear it, re-stranding a call — so shield the
        sweep exactly like ``_force_close_cancellation_safe`` shields teardown.
        """
        await drain_acp_task(asyncio.create_task(translator.flush_interrupted()))

    async def _wait_backoff(self, delay: int, trace: RetryBackoffTrace | None = None) -> None:
        sleep_task = asyncio.create_task(asyncio.sleep(delay))
        cascade_task = asyncio.create_task(self._cascade_event.wait())
        tasks = {sleep_task, cascade_task}
        try:
            try:
                done, _pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                raise
        finally:
            pending = {task for task in tasks if not task.done()}
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if cascade_task in done and self._cascade_requested:
            raise asyncio.CancelledError
        sleep_task.result()
        if trace is not None:
            await trace.started()

    @property
    def active_handle(self) -> PassHandle | None:
        return self._active_handle

    def export_audit(self) -> Mapping[str, object]:
        """Export observed transport facts; remote history is not a kernel state."""
        return {
            "backend": "acp",
            "invocation_id": self._invocation_id,
            "transport_ordinal": self._counters.transport_ordinal,
            "total_usage_tokens": self._counters.spend,
            "usage_unreported_attempts": self._counters.usage_unreported,
        }

    async def _run_fresh_execution(self) -> AcpExecutionResult | None:
        self._pass_started()
        self._transcript_final_text = None
        connect_failure = ""
        for connect_index in range(self._max_connect_retries + 1):
            if self._cascade_requested:
                raise asyncio.CancelledError
            self._counters.transport_ordinal += 1
            translator = AcpUpdateTranslator(
                origin=self._pass_origin,
                event_bus=self._bus,
                session_id=self._session_id,
                agent_name=self._agent_name,
                invocation_id=self._invocation_id,
                attempt=self._counters.transport_ordinal,
                result_mode=self._result_mode,
                completed_before=self._counters.completed_calls,
                spend_before=self._counters.spend,
                unreported_before=self._counters.usage_unreported,
            )
            self._broker.set_translator(translator)
            if self._translator_callback is not None:
                # Adopt the translator into the durable audit trail BEFORE the
                # transport exists: ACP permits a permission request the moment
                # session/new is announced (inside open_session), and a failure
                # in that window must not lose the recorded decision.
                await self._translator_callback(translator)
            if self._cascade_requested:
                raise asyncio.CancelledError
            try:
                spec = self._spec_factory(self._counters.transport_ordinal)
            except (AcpConfigError, ValueError, TypeError) as exc:
                detail = exc.detail if isinstance(exc, AcpConfigError) else preview_text(exc, limit=500)
                return self._terminal(AcpExecutionResult.failure("sub_agent_acp_config", detail, exc))
            self._last_spec = spec
            self._diagnostic_path = str(spec.stderr_log_path)
            client = AcpAgentClient(spec, self._broker, update_sink=translator, wait_controller=self._broker)
            self._active_client = client
            prompt_started = False
            usage: AcpPromptUsage | None = None
            retry_delay: int | None = None
            try:
                if self._cascade_requested:
                    raise asyncio.CancelledError
                await client.connect()
                handshake = await client.open_session()
                if self._attempt_callback is not None:
                    await self._attempt_callback(self._counters.transport_ordinal, handshake, translator)
                prompt_started = True
                outcome = await client.prompt(self._prompt)
                self._stop_reason = outcome.stop_reason
                usage = outcome.usage
                await self._account_usage(self._counters.transport_ordinal, usage, translator)
                await translator.flush_interrupted()
                self._counters.completed_calls = translator.completed_count
                if self._cascade_requested:
                    # A cascade abort must win even when the remote finishes
                    # the prompt (end_turn/refusal/truncation) before our
                    # cancel reaches it: the parent interrupt is already
                    # tearing the turn down, and the cascade terminal event
                    # may already be published — returning a success here
                    # would contradict both.
                    raise asyncio.CancelledError
                if outcome.stop_reason == "end_turn":
                    text = translator.result_text()
                    if not text:
                        return self._terminal(
                            AcpExecutionResult.failure(
                                "sub_agent_empty_output",
                                f"sub-agent '{self._tool_name}' returned no output",
                            )
                        )
                    # Segment publication is independent of the configured
                    # parent-result shape. Carry the UI delta for every mode
                    # so a segment sealed before the terminal tool is not
                    # appended to the live transcript a second time.
                    self._transcript_final_text = translator.unpublished_final_text()
                    return self._terminal(AcpExecutionResult(text, succeeded=True))
                if outcome.stop_reason == "cancelled":
                    self._last_error = "The ACP agent cancelled the prompt unexpectedly."
                    self._last_exception = AcpTransportError(self._last_error)
                    self._last_category = FailureCategory.TRANSPORT
                    return None
                if outcome.stop_reason == "refusal":
                    return self._terminal(
                        AcpExecutionResult.failure("sub_agent_refusal", "The ACP agent refused the request.")
                    )
                partial = translator.partial_text()
                message = f"The ACP agent stopped with {outcome.stop_reason}."
                if partial:
                    message = f"{message} Partial output: {partial}"
                return self._terminal(AcpExecutionResult.failure("sub_agent_truncated", message))
            except AcpConnectError as exc:
                connect_failure = exc.detail
                # The client, not a local flag, is the authority on retryability:
                # open_session deliberately classifies failures that settled
                # before session/new was sent as stateless AcpConnectError, and
                # the retry policy (§6.2) covers the whole spawn/initialize
                # window — pausing there would demand user intervention for a
                # provably side-effect-free respawn.
                if client.stateful_phase_started:
                    self._last_error = connect_failure
                    self._last_exception = exc
                    self._last_category = FailureCategory.TRANSPORT
                    await translator.flush_interrupted()
                    self._counters.completed_calls = translator.completed_count
                    return None
                if connect_index >= self._max_connect_retries:
                    return self._terminal(AcpExecutionResult.failure("sub_agent_acp_setup", connect_failure, exc))
                retry_delay = self._backoff[min(connect_index, len(self._backoff) - 1)]
                self._counters.retry_attempts_total += 1
                if self._bus is not None:
                    await self._emitter.publish(
                        InvocationRetryAttempt(
                            scope="connection",
                            origin=self._pass_origin,
                            agent_name=self._agent_name,
                            message=connect_failure,
                            attempt=connect_index + 1,
                            max_attempts=self._max_connect_retries,
                            delay_seconds=retry_delay,
                            session_id=self._session_id,
                        )
                    )
            except AcpSpawnError as exc:
                return self._terminal(AcpExecutionResult.failure("sub_agent_acp_spawn", exc.detail, exc))
            except AcpConfigError as exc:
                # Terminal-flush invariant: any exit that had a translator must
                # finalize started-but-unresolved tool calls, or a tool update
                # the remote streamed before rejecting leaves a InvocationToolCallStart
                # with no result and stale running ownership. flush_interrupted is
                # a no-op when nothing started (the pre-prompt config/auth cases).
                await translator.flush_interrupted()
                self._counters.completed_calls = translator.completed_count
                return self._terminal(AcpExecutionResult.failure("sub_agent_acp_config", exc.detail, exc))
            except AcpAuthRequiredError as exc:
                await translator.flush_interrupted()
                self._counters.completed_calls = translator.completed_count
                methods = ", ".join(exc.method_names)
                suffix = f" Advertised methods: {methods}." if methods else ""
                return self._terminal(
                    AcpExecutionResult.failure(
                        "sub_agent_acp_auth",
                        f"{exc.detail}{suffix} Authenticate via the agent's own CLI.",
                        exc,
                    )
                )
            except AcpRefusalError as exc:
                await translator.flush_interrupted()
                self._counters.completed_calls = translator.completed_count
                return self._terminal(AcpExecutionResult.failure("sub_agent_refusal", exc.detail, exc))
            except AcpIdleTimeoutError as exc:
                usage = exc.usage
                self._last_error = exc.detail
                self._last_exception = exc
                self._last_category = FailureCategory.TRANSPORT
                if prompt_started:
                    await self._account_usage(self._counters.transport_ordinal, usage, translator)
                await translator.flush_interrupted()
                self._counters.completed_calls = translator.completed_count
                if self._cascade_requested:
                    raise asyncio.CancelledError from exc
                return None
            except AcpTransportError as exc:
                usage = exc.usage
                self._last_error = exc.detail
                self._last_exception = exc
                self._last_category = (
                    FailureCategory.REMOTE_ERROR if is_remote_error(exc) else FailureCategory.TRANSPORT
                )
                if prompt_started:
                    await self._account_usage(self._counters.transport_ordinal, usage, translator)
                await translator.flush_interrupted()
                self._counters.completed_calls = translator.completed_count
                if self._cascade_requested:
                    raise asyncio.CancelledError from exc
                return None
            except asyncio.CancelledError:
                if prompt_started:
                    await self._account_usage(self._counters.transport_ordinal, usage, translator)
                # Flush unconditionally, like every sibling handler: a tool call
                # the remote streamed after session/new but before the prompt
                # (prompt_started is still False) must not be left unresolved.
                await translator.flush_interrupted()
                raise
            finally:
                try:
                    stateful = client.stateful_phase_started
                except AttributeError:
                    stateful = None
                if stateful is True or self._pass_stateful is True:
                    self._pass_stateful = True
                elif stateful is None:
                    self._pass_stateful = None
                self._active_client = None
                try:
                    await self._force_close_cancellation_safe(client)
                finally:
                    # Terminal-flush invariant, enforced AFTER teardown. The
                    # update consumer stays alive through the whole force-close
                    # ladder (the client cancels it only at the very end) and
                    # ACP updates are valid from session/new onward — before the
                    # prompt and during close. A tool-call start streamed in
                    # either window would otherwise leave a InvocationToolCallStart
                    # with no result and stale running ownership. flush_interrupted
                    # is idempotent and a no-op when nothing is unresolved, so the
                    # per-handler flushes above still drive result/usage timing
                    # while this guarantees the invariant on every exit path. The
                    # sweep is cancellation-safe so a late interrupt can't tear
                    # it; the count sync (mirroring every sibling handler) then
                    # reflects anything the sweep terminalized into
                    # completed_tool_calls, which the audit log persists.
                    try:
                        await self._flush_interrupted_cancellation_safe(translator)
                    finally:
                        self._counters.completed_calls = translator.completed_count
            if retry_delay is not None:
                retry_trace = RetryBackoffTrace.open(
                    context=self._trajectory_context,
                    parent_operation_id=self._trajectory_boundary_operation_id,
                    retry_mode=RetryMode.CONNECTION,
                )
                if retry_trace is not None:
                    await retry_trace.scheduled(reason_code=RetryReason.CONNECTION, delay_seconds=retry_delay)
                await self._wait_backoff(retry_delay, retry_trace)
                continue
        return self._terminal(AcpExecutionResult.failure("sub_agent_acp_setup", connect_failure))

    async def _account_usage(
        self,
        attempt: int,
        usage: AcpPromptUsage | None,
        translator: AcpUpdateTranslator,
    ) -> None:
        if attempt <= self._counters.last_accounted_attempt:
            return
        self._counters.last_accounted_attempt = attempt
        self._counters.latest_context_tokens = translator.latest_context_used
        if usage is None:
            self._counters.usage_unreported += 1
        else:
            self._counters.spend += usage.total_tokens
            self._counters.input_spend += usage.input_tokens
            self._counters.output_spend += usage.output_tokens
            if self._usage_callback is not None:
                self._usage_callback(
                    (
                        translator.latest_context_used
                        if translator.latest_context_size is not None
                        else usage.total_tokens
                    ),
                    usage.input_tokens,
                    usage.output_tokens,
                    0,
                    1.0,
                    0,
                    usage.cached_read_tokens,
                    translator.latest_context_size,
                    self._agent_name,
                    f"{self._invocation_id}:a{attempt}",
                    usage.total_tokens,
                )
        await translator.finalize_usage(
            spend=self._counters.spend,
            unreported_attempts=self._counters.usage_unreported,
        )

    async def run(self, request: RunRequest) -> InvocationOutcome:
        """Admit one pass, then adjudicate only after transport and late updates drain."""
        if self._cascade_requested or self._owner_close_cause is not None or self._close_task is not None:
            if request.continuation is not None:
                raise StaleContinuation("ACP continuation owner is closed")
            raise PreparedClosed("ACP caller operation is closing")
        if self._active_handle is not None:
            raise OverlappingRun("ACP operation already has an active pass")
        validate_request(request, ContinuationCapability.FRESH_SESSION)
        if not self.origin.same_invocation(request.origin):
            if request.continuation is not None:
                raise StaleContinuation("Request belongs to another caller operation")
            raise UnsupportedRequest("Request belongs to another caller operation")
        if request.continuation is not None and request.continuation is not self._ticket:
            raise StaleContinuation("ACP retry ticket is stale")
        self._state_generation += 1
        self._ticket = None
        self._counters.pass_ordinal += 1
        handle = PassHandle(self._invocation_id, f"{self._invocation_id}:{self._counters.pass_ordinal}")
        self._active_handle = handle
        done = asyncio.Event()
        self._pass_done = done
        self._pass_stateful = False
        self._pass_cause = None
        self._pass_origin = request.origin
        self._emitter = BoundEmitter(self._bus, request.origin)
        self._last_exception = None
        self._last_category = FailureCategory.TRANSPORT
        self._prompt = "\n".join(message.text for message in request.messages)
        before = (
            self._counters.input_spend,
            self._counters.output_spend,
            self._counters.spend,
            self._counters.usage_unreported,
        )
        result: AcpExecutionResult | None = None
        cancellation: asyncio.CancelledError | None = None
        try:
            try:
                result = await self._run_fresh_execution()
            except asyncio.CancelledError as exc:
                cancellation = exc
            cause = (
                self._owner_close_cause or self._pass_cause or (AbortCause.CASCADE if self._cascade_requested else None)
            )
            missing = self._counters.usage_unreported - before[3]
            usage = UsageDelta(
                self._counters.input_spend - before[0],
                self._counters.output_spend - before[1],
                self._counters.spend - before[2],
                complete=missing == 0
                and self._counters.last_accounted_attempt == self._counters.transport_ordinal
                and self._counters.transport_ordinal > 0,
                unreported=missing,
            )
            effects = PassEvidence(
                self._invocation_id, handle.pass_id, UNKNOWN_COUNT, UNKNOWN_COUNT, UNKNOWN_COUNT, self._pass_stateful
            )
            if cause is not None:
                return Aborted(
                    handle=handle,
                    usage=usage,
                    effects=effects,
                    stop=StopCause.ABORTED,
                    continuation=None,
                    cause=cause,
                )
            if cancellation is not None:
                raise cancellation
            if result is None or not result.succeeded:
                if result is None:
                    self._ticket = ContinuationTicket(
                        self._conversation_id,
                        handle.pass_id,
                        self._state_generation,
                        ContinuationCapability.FRESH_SESSION,
                    )
                return Failed(
                    handle=handle,
                    usage=usage,
                    effects=effects,
                    stop=StopCause.FAILED,
                    continuation=self._ticket,
                    error=self._last_error if result is None else result.text,
                    disposition=FailureDisposition.CALLER_DECISION if result is None else FailureDisposition.TERMINAL,
                    exception=self._last_exception if result is None else result.exception,
                    category=self._last_category if result is None else result.category,
                )
            return Ok(
                handle=handle,
                usage=usage,
                effects=effects,
                stop=StopCause.COMPLETED,
                continuation=None,
                segments=(Content.from_text(result.text),),
            )
        finally:
            self._active_handle = None
            self._pass_done = None
            done.set()
