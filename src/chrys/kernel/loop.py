# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Model/tool loop over a chat middleware layer and instrumented wire client.

``ToolLoopLayer`` owns per-run middleware pipelines, dispatch, result assembly,
continuation state and interrupt-recovery recording. Approval middleware
controls tool admission; ``MiddlewareTermination`` stops execution.
A tool that raises answers the model with a fixed ``Error: Function failed.``
unless it raised ``ModelVisibleToolError``, whose message the model reads;
argument-validation errors include safe schema guidance without echoing the
full argument payload. Every failed result also records the exception tree in
its ``exception`` field, which never goes on the wire.

The loop owns invocation logs and per-tool spans. Raw arguments and results
are logged only when ``TELEMETRY_GATE.sensitive_data`` is set; spans require
``TELEMETRY_GATE.enabled``. Direct ``FunctionTool.invoke`` calls remain silent.

Tool containers become fresh run-local lists and are normalized again before
each batch, since direct callers and progressive exposure can add tools after
agent preparation. Inner stream result hooks run once; final assembly must
not replay them.

Kernel dependencies stay within this package, allowed third-party packages
and foundation. Intra-package imports are relative.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import decimal
import functools
import hashlib
import inspect
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from enum import Enum
from time import monotonic_ns, perf_counter, time_ns
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, cast

from pydantic import BaseModel, ValidationError

from chrys.foundation.errors import clean_error_message, invalidates_continuation_token
from chrys.foundation.observability.gate import TELEMETRY_GATE
from chrys.foundation.recovery import RecoveryPersistOutcome
from chrys.foundation.retry import StreamStall
from chrys.foundation.tool_call_context import (
    TOOL_CALL_CONTEXT_METADATA_KEY,
    get_tool_context,
    merge_tool_call_context_property,
)
from chrys.foundation.tool_execution_stamp import EXECUTION_STAMP_KEY, execution_stamp_from_metadata
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY, read_tool_invocation_order
from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY, get_tool_kind
from chrys.foundation.tool_result_metadata import (
    TOOL_ERROR_KIND_METADATA_KEY,
    TOOL_ERROR_MESSAGE_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
    TOOL_INTERRUPTED_METADATA_KEY,
    TOOL_POST_PROCESSING_INTERRUPTED_METADATA_KEY,
    TOOL_RESULT_METADATA_KEY,
)
from chrys.foundation.trajectory.context import (
    TRAJECTORY_CONTEXT_KWARG,
    TRAJECTORY_EXCHANGE_KWARG,
    ExchangeTrace,
    TrajectoryContext,
    current_trajectory,
    trajectory_scope,
)
from chrys.foundation.trajectory.envelope import MeasurementSource, measurement
from chrys.foundation.trajectory.event_types import (
    EventType as TrajectoryEventType,
)
from chrys.foundation.trajectory.event_types import (
    ExchangeOutcome,
    RetryMode,
    RetryReason,
    ToolOutcome,
)
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.metadata import (
    ANALYTICS_ITEM_ID_KEY,
    OPERATION_ID_KEY,
    TOOL_RESULT_CARRIER_ITEM_ID_METADATA_KEY,
    TOOL_RESULT_ITEM_ID_METADATA_KEY,
    read_analytics_item_id,
    read_operation_id,
)
from chrys.foundation.trajectory.tools import tool_operation_finished_draft, tool_operation_started_draft
from chrys.foundation.trajectory.writer import EmitResult
from chrys.foundation.trajectory_timing import (
    TRAJECTORY_TIMING_KEY,
    build_instant_trajectory_timing,
    trajectory_timing_from_metadata,
)
from chrys.foundation.util.sub_agent_context import (
    SUB_AGENT_RESULT_COMMIT_CALLBACK_KEY,
    sub_agent_parent_result_metadata,
)

from ._content import Content, add_usage_details, normalize_stream_usage
from ._result_ceiling import apply_result_ceiling
from ._tool_arg_errors import (
    _accepts_arbitrary_argument_names,
    _argument_validation_message,
    _rejects_unexpected_arguments,
    _unexpected_argument_names,
)
from ._types import (
    ChatResponse,
    ChatResponseUpdate,
    Message,
    ResponseStream,
)
from .client import _wire_message_view, resolve_storage_mode_and_handles, start_with_wire_progress
from .exceptions import ChrysException, tool_error_result_text
from .exchanges import TOOL_CALL_CONTENT_TYPES
from .identity import WeakIdentityRegistry
from .instrumentation import (
    FUNCTION_SPAN_EXCLUDED_KWARGS,
    OtelAttr,
    capture_exception,
    get_function_duration_histogram,
    get_function_span,
    get_function_span_attributes,
)
from .middleware import FunctionMiddlewarePipeline, MiddlewareTermination, _as_middleware_list, split_middleware
from .sessions import AgentSession, is_local_history_conversation_id
from .tools import (
    FunctionTool,
    SyncToolCancelledAfterCompletion,
    _is_skip_parsing_sentinel,
    _validate_arguments_against_schema,
    normalize_tools,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Awaitable, Callable, Container, Coroutine

    from ._content import UsageDetails
    from .middleware import ChatMiddleware, ChatMiddlewareLayer, FunctionInvocationContext, FunctionMiddleware

logger = logging.getLogger(__name__)

# Default bounds on model/tool iterations and consecutive failed tool batches.
DEFAULT_MAX_ITERATIONS = 40
DEFAULT_MAX_CONSECUTIVE_ERRORS = 3

# Pause between continuation polls of a still-running background response.
# Retrieval returns immediately while the response is queued/in_progress, so
# an unpaced loop would hammer the provider straight into rate limits.
CONTINUATION_POLL_INTERVAL_SECONDS: float = 2.0


class StallExhaustedAction(Enum):
    """Action taken when the stream-idle retry budget is exhausted."""

    BLOCKING_FALLBACK = "blocking_fallback"
    RAISE = "raise"


class WireRetryPolicy(Protocol):
    """Per-run policy for retrying one logical provider call."""

    max_retries: int
    stall_timeout_seconds: float | None
    stall_max_retries: int
    stall_exhausted_action: StallExhaustedAction

    def backoff_seconds(self, attempt: int) -> int: ...

    def is_retryable(self, exc: BaseException) -> bool: ...

    def is_interrupted(self) -> bool: ...

    async def sleep(self, seconds: int) -> bool: ...

    async def on_retry(
        self,
        message: str,
        attempt: int,
        max_attempts: int,
        delay_seconds: int,
        exc: BaseException,
    ) -> None: ...

    def before_retry(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ConsumedInjectionMessageProbe:
    """Downward-safe seam for clean injections consumed on a wire call.

    Chat middleware may append transient reminders after injection.  The
    service-owned probe exposes only the clean user messages so a later tool
    iteration can retain the user's instruction without persisting wire-only
    enrichment.
    """

    drain_consumed_injection_messages: Callable[[], list[Message]]
    commit_consumed_injections: Callable[[], None] | None = None

    def take_consumed_messages(self) -> list[Message]:
        """Take clean user messages consumed by the latest wire call."""
        return self.drain_consumed_injection_messages()

    def commit(self) -> None:
        """Commit the latest logical call's injection transaction."""
        if self.commit_consumed_injections is not None:
            self.commit_consumed_injections()


# --------------------------------------------------------------------------- #
# LoopRecorder — loop-message recording for interrupt recovery
# --------------------------------------------------------------------------- #


def _contains_identity(messages: list[Message], target: Message) -> bool:
    return any(message is target for message in messages)


def _is_actionable_function_call(content: Content) -> bool:
    """Return whether a function call must be executed by the local tool loop."""
    return content.type == "function_call" and not content.informational_only


def _has_function_call(message: Message) -> bool:
    return any(_is_actionable_function_call(content) for content in message.contents)


_MAX_ITERATIONS_FALLBACK_TEXT = "Maximum iterations reached before a final answer could be produced."

_USER_VISIBLE_CONTENT_TYPES = frozenset({"data", "uri", "error", "hosted_file", "hosted_vector_store"})


def _strip_unexecutable_calls_from_response(response: ChatResponse) -> bool:
    """Drop actionable function calls from an exhaustion-tail final response.

    The tail's request carries ``tool_choice="none"``; a provider that ignores
    it returns calls the loop will never execute, and a dangling call recorded
    on this success path has no later repair. Messages the strip empties are
    dropped, and a ``"tool_calls"`` finish reason is normalized to ``"stop"``
    — the stripped response holds no remaining tool work, so advertising it
    would route a plain-text final as unfinished. Returns whether anything
    was removed.
    """
    removed = False
    kept_messages: list[Message] = []
    for message in response.messages:
        if any(_is_actionable_function_call(content) for content in message.contents):
            removed = True
            remaining = [content for content in message.contents if not _is_actionable_function_call(content)]
            if not remaining:
                continue
            message.contents = remaining
        kept_messages.append(message)
    if removed:
        response.messages = kept_messages
        if response.finish_reason == "tool_calls":
            response.finish_reason = "stop"
    return removed


def _strip_unexecutable_calls_from_update(
    update: ChatResponseUpdate, *, suppress_finish_reason: bool = False
) -> ChatResponseUpdate | None:
    """Copy-on-write strip of actionable calls from an exhaustion-tail update.

    Runs in the tail's own yield loop — above the client filter seam, so the
    ResponseStream filter contract ("must return an update") is not involved
    and a fully emptied update can be dropped by returning ``None``. The
    incoming update is never mutated: the retry stream's recorded updates
    must keep the provider's original contents so the post-assembly response
    strip still observes what came back (that observation drives the
    service-storage desync guard). An emptied update still passes through
    when it carries meaningful metadata — any semantic field ``from_updates``
    or a streaming consumer reads (continuation handles, model/timestamps/
    ids, provider metadata); usage survives as a retained content.
    ``raw_representation`` deliberately does NOT exempt: every wire-parsed
    update carries its SDK event, so exempting it would make the drop
    unreachable.

    ``finish_reason`` never exempts and is cleared from every stripped copy:
    the provider concluded the turn on calls this strip is withholding, and
    a consumer keying on the first terminal reason would treat the run as
    complete before the tail's corrective final update re-emits the real
    one. ``suppress_finish_reason`` extends that verdict to updates with no
    call of their own — providers commonly emit the call delta and the
    finish-reason chunk separately, so once the tail has stripped a call the
    caller must suppress every later provider reason too, or the reason-only
    chunk crosses untouched ahead of the corrective update.
    """
    has_actionable_call = any(_is_actionable_function_call(content) for content in update.contents)
    if not has_actionable_call and not (suppress_finish_reason and update.finish_reason is not None):
        return update
    stripped = copy(update)
    stripped.contents = [content for content in update.contents if not _is_actionable_function_call(content)]
    stripped.finish_reason = None
    if (
        stripped.contents
        or any(
            value is not None
            for value in (
                stripped.response_id,
                stripped.conversation_id,
                stripped.continuation_token,
                stripped.model,
                stripped.created_at,
                stripped.message_id,
                stripped.author_name,
            )
        )
        or stripped.additional_properties
    ):
        return stripped
    return None


def _response_has_visible_content(response: ChatResponse) -> bool:
    for message in response.messages:
        for content in message.contents:
            if content.type == "text":
                if content.text and content.text.strip():
                    return True
            elif content.type in _USER_VISIBLE_CONTENT_TYPES:
                return True
    return False


def _ensure_exhaustion_fallback_response(response: ChatResponse) -> bool:
    """Synthesize fallback text when the final response has nothing visible.

    Runs unconditionally after the tail strip (and, in streaming, after echo
    shell cleanup): a blank or reasoning-only final response would otherwise
    return as an empty success — the exact shape response validation exists to
    prevent — regardless of whether the strip removed anything.
    """
    if _response_has_visible_content(response):
        return False
    fallback_content = Content.from_text(_MAX_ITERATIONS_FALLBACK_TEXT)
    if response.messages and not response.messages[-1].contents:
        response.messages[-1].role = "assistant"
        response.messages[-1].contents = [fallback_content]
    else:
        response.messages.append(Message(role="assistant", contents=[fallback_content]))
    return True


def _invalidate_service_continuation_state(
    response: ChatResponse,
    session: AgentSession | None,
) -> None:
    """Withhold a stripped, service-stored response's continuation handles.

    The service retains the unstripped transcript (calls with no outputs)
    behind them; propagating a handle would reproduce next turn the 400 the
    strip prevents. Clearing forces authoritative local-history replay. The
    withheld handles — the response's own id included, since a stateful
    provider accepts it back as ``previous_response_id`` — are recorded on
    the session so a caller repeating one under any handle spelling gets
    suppressed instead of re-sent alongside that replay.
    """
    if session is not None:
        for handle in (response.conversation_id, response.response_id, session.service_session_id):
            if isinstance(handle, str) and handle and not is_local_history_conversation_id(handle):
                session.invalidated_service_session_ids.add(handle)
    response.conversation_id = None
    response.response_id = None
    response._chrys_service_state_invalidated = True
    if session is not None and session.service_session_id is not None:
        session.service_session_id = None


@dataclass(slots=True)
class _PendingExchangeSlot:
    ordinal: int
    function_call: Content
    call_id: str | None
    result: Content | None = None
    fill_kind: str = ""


@dataclass(slots=True)
class _PendingExchange:
    response_messages: tuple[Message, ...]
    slots: tuple[_PendingExchangeSlot, ...]
    result_carrier_item_id: str
    projection_messages: tuple[Message, ...] | None = None

    @property
    def answered_count(self) -> int:
        return sum(slot.result is not None for slot in self.slots)


@dataclass(frozen=True, slots=True)
class _SealedExchange:
    messages: tuple[Message, ...]
    answered_count: int


class _ResultCommit:
    """Synchronous, invocation-bound writer for one pending exchange slot."""

    def __init__(self, recorder: LoopRecorder, exchange: _PendingExchange, slot: _PendingExchangeSlot) -> None:
        self._recorder = recorder
        self._exchange = exchange
        self._slot = slot

    @property
    def has_raw_result(self) -> bool:
        return self._slot.fill_kind == "raw"

    def commit_raw(self, result: Content) -> None:
        self._recorder._fill_slot(self._exchange, self._slot, result, fill_kind="raw")

    def commit_final(self, result: Content) -> None:
        self._recorder._fill_slot(self._exchange, self._slot, result, fill_kind="final", upgrade_raw=True)

    def commit_interrupted(
        self,
        function_call: Content,
        invocation_context: FunctionInvocationContext | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._recorder._interrupt_slot(
            self._exchange,
            self._slot,
            function_call,
            invocation_context,
            metadata,
        )


@dataclass(frozen=True, slots=True)
class LoopRecorderSnapshot:
    """Retry snapshot of non-journal :class:`LoopRecorder` capture state."""

    initial_count: int | None
    captured: tuple[Message, ...] | None
    service_loop_messages: tuple[Message, ...]
    last_checkpoint_key: tuple[int, str] | None


class LoopRecorder:
    """Records tool-loop messages for interrupt recovery.

    Successor to the chat-middleware-based loop-capture middleware: the loop
    feeds it directly (:meth:`record_pre_call` before each wire call,
    :meth:`record_response` after each parsed response) instead of observing
    the per-call middleware pipeline. The consumer surface
    (:attr:`loop_messages`, :meth:`reset`, :attr:`on_checkpoint`) and the
    capture semantics are unchanged.

    On early termination (e.g. hard-cancel ``CancelledError``) only the last
    iteration's messages appear in the run result — previous iterations live
    only in the loop's growing ``prepped_messages`` list. The pre-call
    snapshot preserves them so the engine can merge completed iterations into
    session state.

    ``message_hasher`` turns the newest captured message into a stable payload
    string for checkpoint suppression (chrys injects a ``serialize_message``
    JSON wrapper; the kernel cannot import chrys). Defaults to ``repr``.

    **Reset contract**: :meth:`reset` must be called before each fresh run.

    **Delivery**: per-run via ``client_kwargs["loop_recorder"]`` — parallel
    sub-agent invocations share one client stack, so the recorder must never
    be attached to the layer itself.
    """

    def __init__(
        self,
        *,
        capture_service_loop_messages: bool = False,
        on_checkpoint: Callable[[], Coroutine[Any, Any, None]] | None = None,
        on_pre_wire_barrier: Callable[[], Awaitable[RecoveryPersistOutcome]] | None = None,
        on_result_checkpoint: Callable[[], Coroutine[Any, Any, None]] | None = None,
        message_hasher: Callable[[Message], str] | None = None,
    ) -> None:
        self._capture_service_loop_messages = capture_service_loop_messages
        self._on_result_checkpoint = on_result_checkpoint if on_result_checkpoint is not None else on_checkpoint
        self._on_pre_wire_barrier = on_pre_wire_barrier
        self._message_hasher: Callable[[Message], str] = message_hasher if message_hasher is not None else repr
        self._initial_count: int | None = None
        self._captured: list[Message] | None = None
        self._service_loop_messages: list[Message] = []
        self._last_checkpoint_key: tuple[int, str] | None = None
        self._sealed_exchanges: list[_SealedExchange] = []
        self._pending_exchange: _PendingExchange | None = None
        self._committed_count = 0
        self._barrier_degraded = False
        self._barrier_unconfigured = False
        self._barrier_warned = False

    def reset(self) -> None:
        """Clear captured state for a new run."""
        self._initial_count = None
        self._captured = None
        self._service_loop_messages = []
        self._last_checkpoint_key = None
        self._sealed_exchanges = []
        self._pending_exchange = None
        self._committed_count = 0
        self._barrier_degraded = False
        self._barrier_warned = False

    def snapshot(self) -> LoopRecorderSnapshot:
        """Capture recorder state for an outer provider retry."""
        return LoopRecorderSnapshot(
            initial_count=self._initial_count,
            captured=None if self._captured is None else tuple(self._captured),
            service_loop_messages=tuple(self._service_loop_messages),
            last_checkpoint_key=self._last_checkpoint_key,
        )

    def restore(self, snapshot: LoopRecorderSnapshot) -> None:
        """Restore capture state while preserving every answered journal slot."""
        self._initial_count = snapshot.initial_count
        self._captured = None if snapshot.captured is None else list(snapshot.captured)
        self._service_loop_messages = list(snapshot.service_loop_messages)
        self._last_checkpoint_key = snapshot.last_checkpoint_key
        pending = self._pending_exchange
        if pending is not None and not any(slot.fill_kind in ("raw", "final") for slot in pending.slots):
            self._pending_exchange = None

    @property
    def on_checkpoint(self) -> Callable[[], Coroutine[Any, Any, None]] | None:
        """Compatibility alias for the best-effort checkpoint callback."""
        return self._on_result_checkpoint

    @on_checkpoint.setter
    def on_checkpoint(self, callback: Callable[[], Coroutine[Any, Any, None]] | None) -> None:
        self._on_result_checkpoint = callback

    @property
    def on_pre_wire_barrier(self) -> Callable[[], Awaitable[RecoveryPersistOutcome]] | None:
        """Strict recovery barrier invoked before a post-commit provider call."""
        return self._on_pre_wire_barrier

    @on_pre_wire_barrier.setter
    def on_pre_wire_barrier(
        self,
        callback: Callable[[], Awaitable[RecoveryPersistOutcome]] | None,
    ) -> None:
        self._on_pre_wire_barrier = callback
        self._barrier_unconfigured = False

    @property
    def on_result_checkpoint(self) -> Callable[[], Coroutine[Any, Any, None]] | None:
        """Best-effort checkpoint callback kicked after slot fills and snapshots."""
        return self._on_result_checkpoint

    @on_result_checkpoint.setter
    def on_result_checkpoint(self, callback: Callable[[], Coroutine[Any, Any, None]] | None) -> None:
        self._on_result_checkpoint = callback

    @property
    def committed_count(self) -> int:
        """Number of answered ordinal slots in the current outer pass."""
        return self._committed_count

    @property
    def initial_count(self) -> int | None:
        """Message count of the first wire call, or ``None`` before any call."""
        return self._initial_count

    @property
    def captured_count(self) -> int | None:
        """Size of the newest pre-call snapshot, or ``None`` before any call."""
        return None if self._captured is None else len(self._captured)

    @property
    def loop_messages(self) -> list[Message] | None:
        """Messages from completed tool loop iterations, or ``None``.

        Returns the assistant + tool messages that the tool loop accumulated
        from **previous** iterations.  Returns ``None`` if no multi-iteration
        loop occurred (single iteration or no tool calls).
        """
        captured_delta: list[Message] = []
        if self._captured is not None and self._initial_count is not None and len(self._captured) > self._initial_count:
            captured_delta = self._captured[self._initial_count :]

        base_messages = self._service_loop_messages or captured_delta
        journal: list[tuple[set[int], tuple[Message, ...]]] = [
            (
                {id(content) for message in exchange.messages for content in message.contents},
                exchange.messages,
            )
            for exchange in self._sealed_exchanges
        ]
        if self._pending_exchange is not None and self._pending_exchange.answered_count:
            pending_projection = tuple(self._project_pending_exchange(self._pending_exchange))
            pending_owned = {
                id(content) for message in self._pending_exchange.response_messages for content in message.contents
            }
            pending_owned.update(id(slot.result) for slot in self._pending_exchange.slots if slot.result is not None)
            journal.append((pending_owned, pending_projection))

        candidates: list[Message] = []
        inserted: set[int] = set()
        all_owned = set().union(*(owned for owned, _projection in journal)) if journal else set()
        for message in base_messages:
            message_ids = {id(content) for content in message.contents}
            for index, (owned, projection) in enumerate(journal):
                if index not in inserted and message_ids.intersection(owned):
                    candidates.extend(projection)
                    inserted.add(index)
            remaining = [content for content in message.contents if id(content) not in all_owned]
            if remaining:
                candidates.append(
                    message if len(remaining) == len(message.contents) else _message_snapshot(message, remaining)
                )
        for index, (_owned, projection) in enumerate(journal):
            if index not in inserted:
                candidates.extend(projection)
        deduped = self._dedupe_projected_messages(candidates)
        return deduped or None

    async def record_pre_call(self, messages: list[Message]) -> None:
        """Snapshot the prepped message list before a wire call."""
        if self._initial_count is None:
            self._initial_count = len(messages)
        prev_len = len(self._captured) if self._captured else 0
        # Copy to avoid aliasing mutations of the loop's growing list.
        self._captured = list(messages)
        # Responses store=true service-side continuations replace the growing
        # local prepped list with compact assistant/tool payloads. Preserve
        # those too so pause-time recovery can replay locally after dropping
        # the service id. The mode is configured by chrys from the model
        # profile/options; ``ChatResponse.conversation_id`` alone is provider
        # metadata and is not a reliable discriminator.
        if self._capture_service_loop_messages and self._service_loop_messages:
            for msg in messages:
                if not _contains_identity(self._service_loop_messages, msg) and msg.role in ("assistant", "tool"):
                    self._service_loop_messages.append(msg)
        logger.debug(
            "LoopRecorder: initial=%d prev=%d now=%d delta=%d",
            self._initial_count,
            prev_len,
            len(self._captured),
            len(self._captured) - self._initial_count,
        )
        barrier_persisted = await self._run_pre_wire_barrier()
        if self._on_result_checkpoint is None:
            return
        # A persisted barrier already made this pre-call state durable; the
        # best-effort checkpoint would only build and write it again. Suppression
        # still runs first: it records this prefix as checkpointed, so a retry of
        # the same request stays suppressed after a later barrier failure.
        if self._should_suppress_checkpoint() or barrier_persisted:
            return
        await self._on_result_checkpoint()

    def record_response(self, response: ChatResponse) -> None:
        """Record assistant function-call messages from a parsed response.

        The loop owns the parsed ``ChatResponse`` for both stream modes, so the
        stream ``result_hook`` side channel of the middleware era is gone.
        """
        if not self._capture_service_loop_messages:
            return
        for msg in response.messages:
            if not _contains_identity(self._service_loop_messages, msg) and _has_function_call(msg):
                self._service_loop_messages.append(msg)

    def stage_exchange(
        self,
        response_messages: Sequence[Message],
        function_calls: Sequence[Content],
        *,
        result_carrier_item_id: str,
    ) -> tuple[_ResultCommit, ...]:
        """Reserve ordinal slots for one landed call batch."""
        slots: list[_PendingExchangeSlot] = []
        for function_call in function_calls:
            ordinal = read_tool_invocation_order(function_call.additional_properties)
            if ordinal is None:
                raise ValueError("Landed function call is missing its invocation ordinal.")
            slots.append(
                _PendingExchangeSlot(
                    ordinal=ordinal,
                    function_call=function_call,
                    call_id=function_call.call_id,
                )
            )
        exchange = _PendingExchange(
            response_messages=tuple(response_messages),
            slots=tuple(slots),
            result_carrier_item_id=result_carrier_item_id,
        )
        self._pending_exchange = exchange
        return tuple(_ResultCommit(self, exchange, slot) for slot in exchange.slots)

    def seal_exchange(self, result_message: Message) -> None:
        """Replace provisional results with their canonical carrier message."""
        exchange = self._pending_exchange
        if exchange is None:
            return
        result_identities = {id(content) for content in result_message.contents}
        if any(slot.result is not None and id(slot.result) not in result_identities for slot in exchange.slots):
            return
        self._sealed_exchanges.append(
            _SealedExchange(
                messages=(*exchange.response_messages, result_message),
                answered_count=exchange.answered_count,
            )
        )
        self._pending_exchange = None

    def _fill_slot(
        self,
        exchange: _PendingExchange,
        slot: _PendingExchangeSlot,
        result: Content,
        *,
        fill_kind: str,
        upgrade_raw: bool = False,
    ) -> None:
        if exchange is not self._pending_exchange:
            return
        if slot.result is None:
            slot.result = result
            slot.fill_kind = fill_kind
            exchange.projection_messages = None
            # Interrupted fills mark cancellation before the side-effect
            # boundary was observed: they persist through finalization but are
            # not commits, so a zero-commit retry whose own stream teardown
            # cancelled in-flight tools may still re-dispatch them.
            if fill_kind != "interrupted":
                self._committed_count += 1
            self._kick_result_checkpoint()
            return
        if upgrade_raw and slot.fill_kind == "raw":
            slot.result = result
            slot.fill_kind = fill_kind
            exchange.projection_messages = None
            self._kick_result_checkpoint()

    def _interrupt_slot(
        self,
        exchange: _PendingExchange,
        slot: _PendingExchangeSlot,
        function_call: Content,
        invocation_context: FunctionInvocationContext | None,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        result_metadata = dict(metadata or {})
        result_metadata[TOOL_INTERRUPTED_METADATA_KEY] = True
        if slot.fill_kind == "interrupted" and slot.result is not None:
            # Terminal audit persistence can finish after the early
            # interruption fill, including after the exchange was sealed.
            # The slot retains the canonical result object, so merge the late
            # replay references in place and checkpoint the upgraded history.
            existing = slot.result.additional_properties.get(TOOL_RESULT_METADATA_KEY)
            existing_metadata = dict(existing) if isinstance(existing, Mapping) else {}
            slot.result.additional_properties[TOOL_RESULT_METADATA_KEY] = {
                **existing_metadata,
                **result_metadata,
            }
            if invocation_context is not None:
                timing = _tool_trajectory_timing(function_call, invocation_context)
                slot.result.additional_properties[TRAJECTORY_TIMING_KEY] = timing
            self._kick_result_checkpoint()
            return
        if exchange is not self._pending_exchange:
            return
        if slot.fill_kind == "raw" and slot.result is not None:
            result_metadata[TOOL_POST_PROCESSING_INTERRUPTED_METADATA_KEY] = True
            existing = slot.result.additional_properties.get(TOOL_RESULT_METADATA_KEY)
            if isinstance(existing, Mapping):
                result_metadata = {**existing, **result_metadata}
            slot.result.additional_properties[TOOL_RESULT_METADATA_KEY] = result_metadata
            timing = _tool_trajectory_timing(function_call, invocation_context)
            slot.result.additional_properties[TRAJECTORY_TIMING_KEY] = timing
            self._kick_result_checkpoint()
            return
        if slot.result is not None:
            # Final results are immutable; raw and interrupted fills took
            # their dedicated merge paths above.
            return
        result_metadata[TOOL_FAILED_METADATA_KEY] = True
        additional = _result_additional_properties(function_call, invocation_context)
        additional[TOOL_RESULT_METADATA_KEY] = result_metadata
        result = Content.from_function_result(
            call_id=function_call.call_id,  # type: ignore[arg-type]
            result=(
                "Error: Tool execution was interrupted. The operation may have completed; "
                "inspect current state before retrying."
            ),
            additional_properties=additional,
        )
        self._fill_slot(exchange, slot, result, fill_kind="interrupted")

    async def _run_pre_wire_barrier(self) -> bool:
        """Strictly persist committed tool work; True only when the barrier reports it persisted."""
        if self._committed_count == 0 or self._barrier_degraded or self._barrier_unconfigured:
            return False
        callback = self._on_pre_wire_barrier
        if callback is None:
            self._barrier_unconfigured = True
            return False
        for _attempt in range(2):
            try:
                outcome = await callback()
            except Exception:
                outcome = RecoveryPersistOutcome.FAILED
            if outcome is RecoveryPersistOutcome.PERSISTED:
                return True
            if outcome is RecoveryPersistOutcome.UNCONFIGURED:
                self._barrier_unconfigured = True
                return False
        self._barrier_degraded = True
        if not self._barrier_warned:
            self._barrier_warned = True
            logger.error(
                "Recovery sidecar persistence failed twice after committed tool work; "
                "continuing with the in-memory journal."
            )
        return False

    def _kick_result_checkpoint(self) -> None:
        callback = self._on_result_checkpoint
        if callback is None:
            return
        task = asyncio.get_running_loop().create_task(callback())
        task.add_done_callback(self._observe_checkpoint_task)

    @staticmethod
    def _observe_checkpoint_task(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            exception = task.exception()
        except Exception:
            logger.debug("LoopRecorder result checkpoint failed", exc_info=True)
            return
        if exception is not None:
            logger.debug(
                "LoopRecorder result checkpoint failed",
                exc_info=(type(exception), exception, exception.__traceback__),
            )

    @staticmethod
    def _project_pending_exchange(exchange: _PendingExchange) -> list[Message]:
        if exchange.projection_messages is not None:
            return list(exchange.projection_messages)
        answered_calls = {id(slot.function_call) for slot in exchange.slots if slot.result is not None}
        staged_calls = {id(slot.function_call) for slot in exchange.slots}
        projected: list[Message] = []
        for message in exchange.response_messages:
            message_staged_calls = [content for content in message.contents if id(content) in staged_calls]
            if not message_staged_calls:
                projected.append(message)
                continue
            answered_in_message = [content for content in message_staged_calls if id(content) in answered_calls]
            informational = any(
                content.type == "function_call" and content.informational_only for content in message.contents
            )
            if not answered_in_message and not informational:
                continue
            contents = [
                content
                for content in message.contents
                if id(content) not in staged_calls or id(content) in answered_calls
            ]
            projected.append(_message_snapshot(message, contents))
        results = [slot.result for slot in exchange.slots if slot.result is not None]
        if results:
            carrier = Message(role="tool", contents=results)
            carrier.additional_properties[ANALYTICS_ITEM_ID_KEY] = exchange.result_carrier_item_id
            projected.append(carrier)
        exchange.projection_messages = tuple(projected)
        return projected

    @staticmethod
    def _dedupe_projected_messages(messages: Sequence[Message]) -> list[Message]:
        seen_contents: set[int] = set()
        projected: list[Message] = []
        for message in messages:
            contents = [content for content in message.contents if id(content) not in seen_contents]
            if not contents:
                continue
            seen_contents.update(id(content) for content in contents)
            projected.append(
                message if len(contents) == len(message.contents) else _message_snapshot(message, contents)
            )
        return projected

    def _checkpoint_messages(self) -> list[Message]:
        """Return the effective message prefix used for checkpoint suppression."""
        if self._capture_service_loop_messages and self._service_loop_messages:
            return self._service_loop_messages
        return self._captured or []

    def _should_suppress_checkpoint(self) -> bool:
        """Return True when the captured prefix has not changed since the last checkpoint."""
        messages = self._checkpoint_messages()
        key = (len(messages), self._hash_last_message(messages))
        if key == self._last_checkpoint_key:
            return True
        self._last_checkpoint_key = key
        return False

    def _hash_last_message(self, messages: list[Message]) -> str:
        if not messages:
            return ""
        try:
            payload = self._message_hasher(messages[-1])
        except Exception:
            payload = repr(messages[-1])
        # Hashing is identity-bearing: surrogatepass keeps the operation total
        # without aliasing a lone surrogate to the literal escape that spells it.
        return hashlib.sha256(payload.encode("utf-8", errors="surrogatepass")).hexdigest()


# --------------------------------------------------------------------------- #
# Loop helpers (response shaping)
# --------------------------------------------------------------------------- #


def _extract_function_calls(response: ChatResponse) -> list[Content]:
    """Collect unresolved, deduplicated function calls from a response.

    Skips calls that already carry a result
    in the same response and duplicate ``call_id``\\ s. Echoed calls never
    reach this filter — ``_land_response_contents`` removes them from
    the response before anything else observes it.
    """
    function_results = {
        item.call_id
        for message in response.messages
        for item in message.contents
        if item.type == "function_result" and item.call_id
    }
    seen_call_ids: set[str] = set()
    function_calls: list[Content] = []
    for message in response.messages:
        for item in message.contents:
            if not _is_actionable_function_call(item):
                continue
            if item.call_id and item.call_id in function_results:
                continue
            if item.call_id and item.call_id in seen_call_ids:
                continue
            if item.call_id:
                seen_call_ids.add(item.call_id)
            function_calls.append(item)
    return function_calls


def _message_snapshot(message: Message, contents: list[Content]) -> Message:
    """Loop-owned shallow copy of a message wrapper holding *contents*.

    The wrapper and its two containers (``contents`` list,
    ``additional_properties`` dict) are copied; the Content objects are NOT —
    their identity is the echo memo's currency and must stay shared. Landing
    retains these snapshots instead of client-minted wrappers
    (``_land_response_contents``); the request direction of wrapper aliasing
    is covered by ``_wire_message_view``.
    """
    snapshot = copy(message)
    snapshot.contents = contents
    snapshot.additional_properties = dict(message.additional_properties)
    # Echo-strip provenance is assembly-local. A mixed echo+fresh message is
    # retained, but its snapshot must not carry the marker into later calls.
    snapshot._chrys_echo_content_stripped = False
    return snapshot


_HOSTED_CALL_CONTENT_TYPES = frozenset(TOOL_CALL_CONTENT_TYPES - {"function_call"})
"""Provider-hosted tool invocations a landed response may carry (observed, never executed here)."""


def _hosted_call_kinds(response: ChatResponse) -> list[str]:
    """Content types of the hosted calls in *response*, in emission order."""
    return [
        item.type
        for message in response.messages
        for item in message.contents
        if item.type in _HOSTED_CALL_CONTENT_TYPES
    ]


def _land_response_contents(
    response: ChatResponse,
    next_ordinal: int,
    echo_registry: WeakIdentityRegistry,
    *,
    exchange_operation_id: str | None = None,
) -> int:
    """Normalize a landed response's contents and stamp fresh function calls.

    This loop is the single authoritative producer of ``_chrys_tool_invocation_order``.
    Landing happens at response arrival — before recording, dispatch
    extraction, tool lookup, and argument validation — over every
    non-informational ``function_call`` content in model emission order.
    Calls the dispatch filter drops (already-answered, duplicate ids,
    no-tools batches) and calls that fail pre-pipeline validation still
    consume ordinals, so persisted numbering can never diverge from pipeline
    numbering. Tool-operation ids are assigned later, once response-scoped
    pairing has distinguished work this loop owns from calls the provider
    already answered. Returns the next unassigned ordinal.

    Sole-authority consequences: a pre-existing stamp on a content this run
    has NOT numbered is foreign data (a bridged or cached response replaying
    another run's history) and is overwritten — preserving it while the
    counter advances would let a stale value collide with a fresh assignment.

    ``echo_registry`` holds the identity of every content the conversation
    already holds — the loop seeds it from caller history and response
    contents land here, batch results register in
    ``_handle_function_call_results``; usage stays out via the registry's
    ``ignore_usage`` construction policy. A content OBJECT reappearing
    later (a client echoing prior messages back, or the same object
    repeated within one response) is an echo of history, not new work, and
    is removed before the recorder, the dispatch filter, and final
    transcript assembly observe it: an echoed call would otherwise
    re-execute or persist as a dangling duplicate, and an echoed result
    would duplicate an answer the transcript already holds. Removal is by
    object identity only — a DIFFERENT content reusing an echoed call id
    is fresh work (providers mint per-response counter ids) and lands
    normally; a dead member's recycled address can never match, because
    membership requires the registered object itself to still be alive.
    Identity removal presumes assembly cannot launder an echoed object into
    a new one: ``ChatResponseUpdate`` assembly MERGES consecutive same-call
    function-call fragments (and coalesces adjacent text) into NEW objects,
    so the streaming loop registers ``_strip_echoed_update`` as a per-turn
    update filter on the wire stream — conversation-held objects are
    removed at the innermost stream BEFORE accumulation, so a merge copy of
    an echo never exists, a fresh call reusing an echoed id assembles alone
    with its own arguments, and the provider finalizer plus every result
    hook observe the same echo-free sequence (no re-assembly ever
    overwrites a hook's rewrite of the response). Call-id-level provenance
    cannot do this job: an id carrying both a laundered echo merge and a
    fresh call is ambiguous per id, and either verdict misfires
    (re-executing the echo or swallowing the fresh call via duplicate-id
    dispatch). The reverse direction of the same laundering is covered by
    ``_record_stream_fragment_identities`` after landing: merges mint NEW
    assembled objects, so the raw fragments a client just streamed would
    otherwise stay unknown to the memo and could be replayed next turn.

    Message wrappers are never shared with the client, in either direction.
    Every retained message is REPLACED in ``response.messages`` by a
    loop-owned shallow copy (own contents list, copied
    ``additional_properties``) — or dropped when nothing fresh remains.
    That isolation cuts both ways: echo removal never mutates the incoming
    message (the same object may sit in the accumulated transcript, and a
    client-held alias must keep its view), and a stateful client that later
    mutates a message object it returned — appending a fresh call between
    iterations — cannot rewrite the history this run already landed.
    The request direction is covered too: ``_wire_call`` sends EVERY
    outgoing message — caller history included — as a per-call view
    (``_wire_message_view``), so mutating an INPUT message can corrupt
    neither landed history nor the caller's session-state objects; message
    metadata still writes through because views share the wrapper's
    ``additional_properties`` dict. ``messages`` is rebound, not mutated,
    so caller-held aliases of the original list keep their view. The
    CONTENT objects stay the originals: the registry references them
    weakly, so an entry lives exactly as long as its object and a
    collected content simply stops being a member.

    Landing is also where the trajectory identities are minted, under the
    same sole-authority rule as the ordinal: every retained message gets a
    fresh analytics item id and is bound to ``exchange_operation_id`` (the
    wire exchange that produced it); every fresh function call gets its own
    item id. A pre-existing operation stamp is foreign data (a replayed or
    cached response) and is removed, never trusted. The dispatch collector
    later assigns one operation id per unresolved call id; duplicate call
    contents share it, while a provider-answered call has no Chrys tool
    operation to name.
    """
    normalized_messages: list[Message] = []
    for message in response.messages:
        fresh_contents: list[Content] = []
        dropped_echo = False
        for item in message.contents:
            if item in echo_registry:
                dropped_echo = True
                continue
            echo_registry.register(item)
            if _is_actionable_function_call(item):
                item.additional_properties[TOOL_INVOCATION_ORDER_KEY] = next_ordinal
                next_ordinal += 1
                item.additional_properties.pop(OPERATION_ID_KEY, None)
            if item.type == "function_call":
                item.additional_properties[ANALYTICS_ITEM_ID_KEY] = new_analytics_id()
            fresh_contents.append(item)
        if dropped_echo and not fresh_contents:
            continue
        # Snapshot the wrapper, keep the contents: the client holds its own
        # reference to this message and may mutate it between iterations.
        snapshot = _message_snapshot(message, fresh_contents)
        snapshot.additional_properties[ANALYTICS_ITEM_ID_KEY] = new_analytics_id()
        if exchange_operation_id is not None:
            snapshot.additional_properties[OPERATION_ID_KEY] = exchange_operation_id
        normalized_messages.append(snapshot)
    response.messages = normalized_messages
    return next_ordinal


def _stamp_function_call_operations(response: ChatResponse, function_calls: Sequence[Content]) -> None:
    """Assign one tool operation to each unresolved call the loop may dispatch.

    ``_extract_function_calls`` is the response-scoped pairing authority: a
    call already answered in the same response is provider-owned history, not
    a Chrys tool operation. Repeated unresolved ``call_id`` contents describe
    the same dispatch (the extractor keeps the first), so they share that
    first call's operation id rather than opening operations nobody executes.
    Empty ids are not deduplicated and therefore keep one operation per call.
    """
    operation_by_identity: dict[int, str] = {}
    operation_by_call_id: dict[str, str] = {}
    for function_call in function_calls:
        operation_id = new_analytics_id()
        function_call.additional_properties[OPERATION_ID_KEY] = operation_id
        operation_by_identity[id(function_call)] = operation_id
        if function_call.call_id:
            operation_by_call_id[function_call.call_id] = operation_id

    for message in response.messages:
        for item in message.contents:
            if not _is_actionable_function_call(item) or id(item) in operation_by_identity:
                continue
            operation_id = operation_by_call_id.get(item.call_id or "")
            if operation_id is not None:
                item.additional_properties[OPERATION_ID_KEY] = operation_id


def _strip_echoed_update(
    update: ChatResponseUpdate,
    echo_registry: WeakIdentityRegistry,
) -> ChatResponseUpdate:
    """The update with conversation-held content objects removed.

    Registered per turn as an update filter on the wire stream
    (:meth:`ResponseStream.with_update_filter`), so it runs at the innermost
    stream BEFORE accumulation: update assembly merges consecutive same-call
    function-call fragments (and coalesces adjacent text) into NEW objects,
    so an echoed historical object that participates in a merge would
    launder its identity past the echo memo. Filtering ahead of assembly
    means a laundered merge copy never exists, a fresh call reusing an
    echoed call id assembles alone with its own arguments, and — because
    the provider finalizer and every result hook see the already-filtered
    updates — no re-assembly ever overwrites a hook's rewrite of the
    response ("hooks run before tool extraction" stays true). The incoming
    update is never mutated: one carrying an echo is shallow-copied with a
    filtered contents list, so a producer-held alias keeps its shape.
    The copy carries private, non-serialized provenance through stream
    proxies and assembly, letting the loop drop only the logical message
    shell this update actually emptied. The marker is attached to the
    accepted update object rather than keyed by provider message ID, which
    can collide across validation retries.
    """
    if any(item in echo_registry for item in update.contents):
        update = copy(update)
        update._chrys_echo_content_stripped = True
        update.contents = [item for item in update.contents if item not in echo_registry]
    return update


def _remove_echo_emptied_message_shells(response: ChatResponse) -> None:
    """Remove only accepted-stream messages left empty by echo stripping.

    A call-wide marker cannot distinguish an unrelated legitimate empty
    message from an echo-only one. Per-update provenance is folded into the
    assembled Message, so this cleanup remains per message and per accepted
    validation attempt without trusting non-unique provider message IDs.
    """
    response.messages = [
        message for message in response.messages if message.contents or not message._chrys_echo_content_stripped
    ]


def _record_stream_fragment_identities(
    stream: ResponseStream[ChatResponseUpdate, ChatResponse],
    echo_registry: WeakIdentityRegistry,
) -> None:
    """Record the identity of every accepted stream content.

    Landing only sees the ASSEMBLED response, and assembly merges
    multi-fragment calls (and coalesces adjacent text) into NEW objects —
    so the raw fragment objects the client actually yielded would never
    enter the echo registry, and a stateful client replaying a buffered
    fragment on a later turn would sail past ``_strip_echoed_update`` and
    re-execute the call. The consumed stream's accumulated updates are the
    accepted (post-strip) sequence; under the response-validation proxy
    they are the SELECTED attempt's replay, so failed attempts' fragments
    are not recorded and can be legitimately re-sent on a retry.

    Must run AFTER ``_land_response_contents``: a single-fragment call's
    assembled content IS the fragment object, and pre-registering it
    would make landing drop the fresh call as its own echo.

    No retention side: a replay requires the client to still HOLD the
    fragment, which keeps the weak entry alive; a collected fragment's
    entry evicts, so its recycled id can never strip a genuinely fresh
    content. Usage is request-scoped accounting rather than conversation
    content (a client may reuse a mutable usage object across model
    calls), so the registry's ``ignore_usage`` policy keeps it out.
    """
    for update in stream.updates:
        for item in update.contents:
            echo_registry.register(item)


def _record_wire_request_content_identities(
    messages: Sequence[Message],
    echo_registry: WeakIdentityRegistry,
) -> None:
    """Register the post-middleware contents of one actual wire request.

    Called at the chat pipeline's final-handler boundary, after every
    middleware has transformed the request and immediately before the inner
    client sees it. This closes the identity-registry gap for wire-only
    messages such as injections and reminders, including repeated
    ``call_next`` retries.

    No retention side: a weak entry lives exactly as long as its object,
    so a collected wire-only enrichment's entry evicts and its recycled id
    can never strip a genuinely fresh content. Usage is request-scoped
    accounting; the registry's ``ignore_usage`` policy keeps it out.
    """
    for message in messages:
        for item in message.contents:
            echo_registry.register(item)


def _stream_usage_chunks(update: ChatResponseUpdate) -> list[Mapping[str, Any]]:
    """Return usage snapshots carried by one streaming update."""
    return [
        content.usage_details
        for content in update.contents
        if content.type == "usage" and content.usage_details is not None
    ]


def _truncated_final_function_call_ids(response: ChatResponse, function_calls: Sequence[Content]) -> set[int]:
    """Return the object id for the final content block when it is a function call."""
    if response.finish_reason != "length" or not function_calls:
        return set()
    final_content = next(
        (item for message in reversed(response.messages) for item in reversed(message.contents)),
        None,
    )
    if final_content is None or not _is_actionable_function_call(final_content):
        return set()
    function_call_ids = {id(item) for item in function_calls}
    final_id = id(final_content)
    return {final_id} if final_id in function_call_ids else set()


def _prepend_fcc_messages(response: ChatResponse, fcc_messages: list[Message]) -> None:
    """Insert accumulated loop messages before the final response."""
    if not fcc_messages:
        return
    for msg in reversed(fcc_messages):
        response.messages.insert(0, msg)


def _update_conversation_id(
    kwargs: dict[str, Any],
    conversation_id: str | None,
    options: dict[str, Any] | None = None,
) -> None:
    """Write a service conversation id back into kwargs/options."""
    if conversation_id is None:
        return
    if "chat_options" in kwargs:
        kwargs["chat_options"]["conversation_id"] = conversation_id
    else:
        kwargs["conversation_id"] = conversation_id
    if options is not None:
        options["conversation_id"] = conversation_id


def _update_continuation_state(
    kwargs: dict[str, Any],
    response: ChatResponse,
    *,
    session: AgentSession | None,
    options: dict[str, Any] | None = None,
) -> None:
    """Update in-flight and persisted continuation state from a response."""
    conversation_id = response.conversation_id
    if conversation_id is None:
        return

    _update_conversation_id(kwargs, conversation_id, options)
    if (
        session is not None
        and not response.has_internal_conversation_id()
        and session.service_session_id != conversation_id
    ):
        session.service_session_id = conversation_id


def _clear_internal_conversation_id(response: ChatResponse) -> ChatResponse:
    """Strip a client-internal conversation id before returning."""
    if response.has_internal_conversation_id():
        response.conversation_id = None
        response.clear_internal_conversation_id()
    return response


# --------------------------------------------------------------------------- #
# Tool execution
# --------------------------------------------------------------------------- #


def _describe_arguments(arguments: Mapping[str, Any] | BaseModel, schema_names: Container[str]) -> str:
    """Render tool arguments for the DEBUG log line.

    Raw argument text is gated on ``TELEMETRY_GATE.sensitive_data``. With the
    gate off, only names listed in the tool schema's ``properties`` are
    disclosed: the model controls the actual keys (schemas rarely pin
    ``additionalProperties: false``, so unexpected keys survive validation)
    and could smuggle sensitive text through a key name — such keys are
    reported as an ``+N unrecognized`` count only.
    """
    if _is_argument_mapping(arguments):
        argument_mapping: Mapping[str, Any] = arguments
    else:
        # The middleware contract admits ``BaseModel`` rewrites of
        # ``context.arguments``; pydantic models iterate as (key, value)
        # pairs, so ``dict()`` recovers the mapping form. ``tool.invoke``
        # normalizes the same way internally — a describe-only crash here
        # would otherwise be converted into a tool error result.
        argument_mapping = dict(arguments)
    if TELEMETRY_GATE.sensitive_data:
        return str(dict(argument_mapping))
    if not argument_mapping:
        return "0 key(s)"
    known = [key for key in argument_mapping if key in schema_names]
    parts = (
        known
        if len(known) == len(argument_mapping)
        else [*known, f"+{len(argument_mapping) - len(known)} unrecognized"]
    )
    return f"{len(argument_mapping)} key(s) ({', '.join(parts)})"


def _is_argument_mapping(value: Mapping[str, Any] | BaseModel) -> TypeGuard[Mapping[str, Any]]:
    """Narrow middleware arguments after the runtime mapping check."""
    return isinstance(value, Mapping)


def _describe_result(result: Any, *, skip_parsing: bool) -> str:
    """Render a tool result for DEBUG logging under the sensitive-data gate.

    Skip-parsing tools return arbitrary values, including non-Content lists:
    render those with ``str`` when sensitive data is enabled, otherwise use
    the type name. Parsed results are ``list[Content]`` and can be summarized
    by item type or, when permitted, joined as text.
    """
    if skip_parsing or not isinstance(result, list):
        return str(result) if TELEMETRY_GATE.sensitive_data else type(result).__name__
    if TELEMETRY_GATE.sensitive_data:
        text = "\n".join(item.text or "" for item in result if item.type == "text")
        return text or str(result)
    if not result:
        return "None"
    return f"{len(result)} item(s) ({', '.join(item.type for item in result)})"


_FUNCTION_DURATION_HISTOGRAM: Any = None


def _function_duration_histogram() -> Any:
    """Cache the function-duration histogram on its first gated use.

    Lazy construction lets provider setup finish before obtaining the meter.
    """
    global _FUNCTION_DURATION_HISTOGRAM
    if _FUNCTION_DURATION_HISTOGRAM is None:
        _FUNCTION_DURATION_HISTOGRAM = get_function_duration_histogram()
    return _FUNCTION_DURATION_HISTOGRAM


async def _invoke_with_function_span(
    tool: FunctionTool,
    context: Any,
    call_id: str | None,
    *,
    arguments: BaseModel | Mapping[str, Any],
    arguments_in_callable_keyspace: bool = False,
    tool_result_ceiling_tokens: int | None = None,
) -> Any:
    """Run ``tool.invoke`` inside an ``execute_tool {name}`` span.

    Capture arguments and results only under ``TELEMETRY_GATE.sensitive_enabled``;
    record exceptions and duration on failure as well as success. The caller
    owns logging. Skip-parsing results use ``str(result)``; parsed Content
    lists contribute their text, selected by the same ``SKIP_PARSING`` sentinel
    that ``invoke`` uses.

    Captured arguments are the middleware-final values. Validation happens
    inside ``invoke``, so the span does not capture its normalized argument dict.
    """
    attributes = get_function_span_attributes(tool, tool_call_id=call_id)
    # The middleware contract admits ``BaseModel`` rewrites of
    # ``context.arguments`` (same shape ``_describe_arguments`` / ``invoke``
    # handle); normalize it to a dict before filtering so sensitive capture
    # doesn't silently record ``"None"`` for a model-shaped argument set.
    if isinstance(arguments, BaseModel):
        argument_items: dict[str, Any] = arguments.model_dump(exclude_none=True)
    elif isinstance(arguments, Mapping):
        argument_items = dict(arguments)
    else:
        argument_items = {}
    serializable_kwargs = {k: v for k, v in argument_items.items() if k not in FUNCTION_SPAN_EXCLUDED_KWARGS}
    if TELEMETRY_GATE.sensitive_enabled:
        attributes[OtelAttr.TOOL_ARGUMENTS] = (
            json.dumps(serializable_kwargs, default=str, ensure_ascii=False) if serializable_kwargs else "None"
        )
    with get_function_span(attributes=attributes) as span:
        attributes[OtelAttr.MEASUREMENT_FUNCTION_TAG_NAME] = tool.name
        start_time_stamp = perf_counter()
        end_time_stamp: float | None = None
        try:
            result = await tool.invoke(
                arguments=arguments,
                context=context,
                tool_call_id=call_id,
                _arguments_in_callable_keyspace=arguments_in_callable_keyspace,
            )
            result = apply_result_ceiling(result, tool_result_ceiling_tokens)
            end_time_stamp = perf_counter()
        except Exception as exception:
            end_time_stamp = perf_counter()
            attributes[OtelAttr.ERROR_TYPE] = type(exception).__name__
            capture_exception(span=span, exception=exception, timestamp=time_ns())
            raise
        else:
            if TELEMETRY_GATE.sensitive_enabled:
                if _is_skip_parsing_sentinel(tool.result_parser):
                    result_str = str(result)
                else:
                    result_str = "\n".join(c.text or "" for c in result if c.type == "text") or str(result)
                span.set_attribute(OtelAttr.TOOL_RESULT, result_str)
            return result
        finally:
            duration = (end_time_stamp or perf_counter()) - start_time_stamp
            span.set_attribute(OtelAttr.MEASUREMENT_FUNCTION_INVOCATION_DURATION, duration)
            _function_duration_histogram().record(duration, attributes=attributes)


def _arguments_unparseable(arguments: Any) -> bool:
    """True when raw tool-call arguments are an empty or invalid JSON string.

    This is the fingerprint of a truncated / incomplete argument payload — the
    model was cut off before or during JSON emission, so ``Content.parse_arguments``
    either returns ``{}`` for an empty string or falls back to wrapping the raw
    blob under ``{"raw": ...}``. It deliberately does NOT flag arguments that
    parsed cleanly (a dict, or a pre-parsed mapping) but failed schema validation
    — those are real argument errors, not a token-limit cutoff.
    """
    if not isinstance(arguments, str):
        return False
    if not arguments:
        return True
    try:
        json.loads(arguments)
    except json.JSONDecodeError:
        return True
    return False


def _middleware_arguments_equal(left: Any, right: Any) -> bool:
    """Type-strict structural equality for the trusted middleware snapshot."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        if len(left) != len(right):
            return False
        unmatched = list(right.items())
        for left_key, left_value in left.items():
            for index, (right_key, right_value) in enumerate(unmatched):
                if _middleware_arguments_equal(left_key, right_key):
                    if not _middleware_arguments_equal(left_value, right_value):
                        return False
                    unmatched.pop(index)
                    break
            else:
                return False
        return True
    if type(left) in {list, tuple}:
        return len(left) == len(right) and all(
            _middleware_arguments_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if type(left) in {set, frozenset}:
        if len(left) != len(right):
            return False
        unmatched = list(right)
        for left_item in left:
            for index, right_item in enumerate(unmatched):
                if _middleware_arguments_equal(left_item, right_item):
                    unmatched.pop(index)
                    break
            else:
                return False
        return True
    if type(left) in {float, complex, decimal.Decimal}:
        return bool(left == right and repr(left) == repr(right))
    if type(left) in {datetime.datetime, datetime.time}:
        return bool(
            left == right
            and left.utcoffset() == right.utcoffset()
            and left.tzname() == right.tzname()
            and left.fold == right.fold
        )
    return bool(left == right)


def _arguments_look_like_truncated_mapping(arguments: Any, schema: Mapping[str, Any]) -> bool:
    """True for parsed Anthropic tool args that look cut off, not just invalid."""
    if not isinstance(arguments, Mapping):
        return False
    if not arguments:
        return True

    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        return False
    if not isinstance(properties, Mapping):
        return False

    argument_keys = {key for key in arguments if isinstance(key, str)}
    if len(argument_keys) != len(arguments):
        return False
    if not argument_keys.issubset(properties.keys()):
        return False
    return any(isinstance(key, str) and key not in arguments for key in required)


def _stamp_call_provenance(function_call: Content, tool: FunctionTool) -> None:
    """Stamp tool kind + static context onto the call content, first-write-wins.

    Runs right after tool lookup, BEFORE argument validation, so calls that
    fail pre-pipeline validation — which return without entering the
    middleware pipeline — still persist their provenance. Static context
    only: the per-call builder needs validated args and belongs to the
    record pipeline.
    """
    props = function_call.additional_properties
    kind = get_tool_kind(tool)
    if kind and TOOL_CALL_KIND_METADATA_KEY not in props:
        props[TOOL_CALL_KIND_METADATA_KEY] = kind
    static = get_tool_context(tool)
    if static and TOOL_CALL_CONTEXT_METADATA_KEY not in props:
        # Already sanitized at set_tool_context() — single sanitization
        # authority; the stamp only copies, never re-sanitizes.
        props[TOOL_CALL_CONTEXT_METADATA_KEY] = dict(static)


# The failure record is saved with the session and exported to telemetry: a long
# cause chain, a large exception group or a message that embeds a response body
# must not bloat either. Each message is clipped before the record is joined, the
# exception cap also bounds the record's recursion, and the total cap is the backstop.
_FAILURE_RECORD_MAX_EXCEPTIONS = 16
_FAILURE_RECORD_MAX_MESSAGE_CHARS = 1000
_FAILURE_RECORD_MAX_CHARS = 4000


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _exception_message(exc: BaseException) -> str:
    """Return *exc*'s own message, clipped for the failure record.

    ``str()`` of a ChrysException built with an ``inner_exception`` is the repr
    of its args tuple, and a KeyError's is quoted; both hold the message itself
    as their first argument. A ``__str__`` that raises must not stop the call
    from becoming a failed result, so it records what a traceback would.
    """
    if isinstance(exc, ChrysException | KeyError) and exc.args and isinstance(exc.args[0], str):
        message = exc.args[0]
    else:
        try:
            message = str(exc)
        except Exception:
            message = "<exception str() failed>"
    return _clip(message.strip(), _FAILURE_RECORD_MAX_MESSAGE_CHARS)


def _exception_causes(exc: BaseException) -> list[BaseException]:
    """Return what *exc* was explicitly raised from and, when it differs, the ``inner_exception`` a ChrysException wraps."""
    causes = [] if exc.__cause__ is None else [exc.__cause__]
    if isinstance(exc, ChrysException) and len(exc.args) > 1:
        inner = exc.args[1]
        if isinstance(inner, BaseException) and inner is not exc.__cause__:
            causes.append(inner)
    return causes


def _failure_record_text(exc: BaseException) -> str:
    """Return the ``exception`` a failed result records: ``Type: message`` for *exc* and everything it came from.

    The record is a tree, ``Type: message [member; …] (caused by cause; …)``:
    an exception group lists its members in brackets (its ``str()`` only
    counts them), and the causes are ``raise … from`` plus a ChrysException's
    ``inner_exception`` when that is a different exception. It follows only
    these explicit links, never the implicit ``__context__`` of whatever was
    being handled when the tool raised — unlike ``foundation.errors``' linear
    chain walk, which reads that context to classify retryable errors.

    The record is for people reading a saved session, never for the model. It
    is never empty: readers take a non-empty record to mean the call failed.
    """
    seen: set[int] = set()

    def describe(current: BaseException) -> str:
        seen.add(id(current))
        if isinstance(current, BaseExceptionGroup):
            message = _clip(current.message.strip(), _FAILURE_RECORD_MAX_MESSAGE_CHARS)
            members = describe_each(current.exceptions)
        else:
            message = _exception_message(current)
            members = ""
        text = f"{type(current).__name__}: {message}" if message else type(current).__name__
        if members:
            text = f"{text} [{members}]"
        causes = describe_each(_exception_causes(current))
        return f"{text} (caused by {causes})" if causes else text

    def describe_each(exceptions: Sequence[BaseException]) -> str:
        parts: list[str] = []
        for member in exceptions:
            if id(member) in seen:
                continue
            if len(seen) >= _FAILURE_RECORD_MAX_EXCEPTIONS:
                parts.append("…")
                break
            parts.append(describe(member))
        return "; ".join(parts)

    return _clip(describe(exc), _FAILURE_RECORD_MAX_CHARS)


def _result_additional_properties(
    function_call: Content,
    invocation_context: FunctionInvocationContext | None = None,
) -> dict[str, Any]:
    """Return call properties for a function result, minus call provenance.

    Every result-construction path propagates the call's properties; this
    filters call-only provenance (kind, context, invocation ordinal) and any
    stale call-side execution stamp or result metadata. Without the provenance
    filter, the nested context dict would also be shared by reference between
    call and result (the Content constructor's copy is shallow). A fresh stamp
    and fresh result metadata come only from this invocation's context — this
    fold is the single attachment point for ``_chrys_tool_result_metadata``,
    at result construction, when the result's identity is unambiguous.
    """
    result = {
        key: value
        for key, value in function_call.additional_properties.items()
        if key
        not in (
            TOOL_CALL_KIND_METADATA_KEY,
            TOOL_CALL_CONTEXT_METADATA_KEY,
            TOOL_INVOCATION_ORDER_KEY,
            EXECUTION_STAMP_KEY,
            TOOL_RESULT_METADATA_KEY,
            TRAJECTORY_TIMING_KEY,
            ANALYTICS_ITEM_ID_KEY,
        )
    }
    timing = _tool_trajectory_timing(function_call, invocation_context)
    result[TRAJECTORY_TIMING_KEY] = timing
    # The result is its own item (the call's operation id is inherited above
    # so call and result share one tool operation). A pre-minted id on the
    # invocation context keeps the raw-commit checkpoint and the terminal
    # result naming the same item.
    result[ANALYTICS_ITEM_ID_KEY] = _result_item_id(invocation_context)
    if invocation_context is not None:
        stamp = execution_stamp_from_metadata(invocation_context.metadata)
        if stamp is not None:
            result[EXECUTION_STAMP_KEY] = stamp
        carried = invocation_context.metadata.get(TOOL_RESULT_METADATA_KEY)
        if isinstance(carried, Mapping) and carried:
            result[TOOL_RESULT_METADATA_KEY] = dict(carried)
    return result


def _result_item_id(invocation_context: FunctionInvocationContext | None) -> str:
    if invocation_context is not None:
        pre_minted = invocation_context.metadata.get(TOOL_RESULT_ITEM_ID_METADATA_KEY)
        if isinstance(pre_minted, str) and pre_minted:
            return pre_minted
    return new_analytics_id()


def _stamped_tool_context(props: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The static provenance segment stamped on a call, when there is one.

    A call that never enters the pipeline has no builder output — only what
    landing copied off the tool — so an MCP call still names its server.
    """
    context = props.get(TOOL_CALL_CONTEXT_METADATA_KEY)
    return context if isinstance(context, Mapping) and context else None


async def _record_unexecuted_tool_operation(
    function_call: Content,
    *,
    outcome: str,
    result_carrier_item_id: str | None = None,
    result: Content | None = None,
    queued: bool = False,
) -> None:
    """Emit the ``tool.operation`` pair for a call that never entered the pipeline.

    Unknown-tool and invalid-argument calls still own a tool operation id
    (stamped by response dispatch collection) and a result item, so the
    trajectory closes them with their terminal outcome instead of leaving
    dangling operations. A filtered call owns the operation but never gets a
    result, and the pair closes without naming one.
    """
    context = current_trajectory()
    if context is None:
        return
    operation_id = read_operation_id(function_call.additional_properties)
    if operation_id is None:
        return
    props = function_call.additional_properties
    kind = props.get(TOOL_CALL_KIND_METADATA_KEY)
    try:
        started = tool_operation_started_draft(
            context,
            operation_id=operation_id,
            tool_name=function_call.name or "",
            tool_kind=kind if isinstance(kind, str) else "",
            invocation_order=read_tool_invocation_order(props),
            batch_index=None,
            arguments=function_call.arguments,
            call_item_id=read_analytics_item_id(props),
            tool_context=_stamped_tool_context(props),
        )
        finished = tool_operation_finished_draft(
            context,
            operation_id=operation_id,
            outcome=outcome,
            duration_ms=0,
            result_item_id=read_analytics_item_id(result.additional_properties) if result is not None else None,
            result_carrier_item_id=result_carrier_item_id if result is not None else None,
            error_kind=outcome,
        )
    except Exception:
        logger.debug("Trajectory tool operation emit failed", exc_info=True)
        return
    if queued:
        # The caller is unwinding from a cancellation: waiting for the write
        # acknowledgement here would hold a Stop for as long as the writer
        # takes to answer. Both lines take their sequence now and are
        # acknowledged in the background — and a start the sink refuses
        # outright still leaves nothing to close.
        try:
            if context.sink.emit_soon(started) is None:
                context.sink.emit_soon(finished)
        except Exception:
            logger.debug("Trajectory tool operation emit failed", exc_info=True)
        return
    try:
        opened = await context.sink.emit(started)
    except Exception:
        logger.debug("Trajectory tool operation emit failed", exc_info=True)
        return
    except BaseException:
        # Cancelled while the opening line was in flight: the write is
        # committed regardless, so the terminal has to follow it. Only this
        # wait needs the rescue — the one below can first be cancelled once
        # its own line is committed too.
        with contextlib.suppress(Exception):
            context.sink.emit_soon(finished)
        raise
    if opened is not EmitResult.WRITTEN:
        # An unknown tool's name is whatever the model asked for, and one past
        # the line budget makes the whole opening event unwritable: its slot
        # becomes a gap. A terminal behind that gap closes an operation the
        # log never opened, which is not a shape readers handle; an operation
        # that opens and never closes is one they already do.
        return
    try:
        await context.sink.emit(finished)
    except Exception:
        logger.debug("Trajectory tool operation emit failed", exc_info=True)


def _tool_trajectory_timing(
    function_call: Content,
    invocation_context: FunctionInvocationContext | None,
) -> dict[str, Any]:
    """Resolve one tool span and mirror it onto its persisted call content."""
    timing = trajectory_timing_from_metadata(invocation_context.metadata) if invocation_context is not None else None
    if timing is None:
        timing = trajectory_timing_from_metadata(function_call.additional_properties)
    if timing is None:
        timing = build_instant_trajectory_timing()
    function_call.additional_properties[TRAJECTORY_TIMING_KEY] = dict(timing)
    return timing


async def _invoke_function_call(
    function_call: Content,
    *,
    result_commit: _ResultCommit | None,
    tool_map: dict[str, FunctionTool],
    custom_args: dict[str, Any],
    invocation_session: AgentSession | None,
    pipeline: FunctionMiddlewarePipeline,
    live_tools: list[Any] | None,
    response_truncated: bool = False,
    function_call_may_be_truncated: bool = False,
    same_tool_calls_in_batch: int = 1,
    tool_result_ceiling_tokens: int | None = None,
    result_carrier_item_id: str,
) -> Content:
    """Invoke one model-requested function call through the middleware pipeline.

    A raised exception answers the model with ``tool_error_result_text``: a
    fixed line, or a ``ModelVisibleToolError``'s own message. Argument-validation
    failures include safe schema guidance so the model can repair its next call.
    Each failed result records ``_failure_record_text`` as its ``exception``.
    """
    from .middleware import FunctionInvocationContext

    tool = tool_map.get(function_call.name)  # type: ignore[arg-type]
    if tool is None:
        message = f'Requested function "{function_call.name}" not found.'
        exc = KeyError(f'Function "{function_call.name}" not found.')
        additional = _result_additional_properties(function_call)
        additional.update(
            {
                TOOL_FAILED_METADATA_KEY: True,
                TOOL_ERROR_KIND_METADATA_KEY: "tool_not_found",
                TOOL_ERROR_MESSAGE_METADATA_KEY: message,
            }
        )
        result = Content.from_function_result(
            call_id=function_call.call_id,  # type: ignore[arg-type]
            result=f"Error: {message}",
            exception=_failure_record_text(exc),
            additional_properties=additional,
        )
        await _record_unexecuted_tool_operation(
            function_call,
            outcome=ToolOutcome.UNKNOWN_TOOL,
            result_carrier_item_id=result_carrier_item_id,
            result=result,
        )
        return result

    _stamp_call_provenance(function_call, tool)

    parsed_args: dict[str, Any] = dict(function_call.parse_arguments() or {})

    # Filter out internal kwargs before passing to tools; conversation_id is an
    # internal tracking id that must not be forwarded.
    runtime_kwargs: dict[str, Any] = {
        key: value
        for key, value in custom_args.items()
        if key not in {"_function_middleware_pipeline", "middleware", "conversation_id"}
    }
    if invocation_session is not None:
        runtime_kwargs["session"] = invocation_session

    # Pre-pipeline validation: bad arguments return an error
    # result without entering the middleware pipeline (no approval dialog for
    # a call that could never run).
    argument_schema = tool.parameters()
    typed_model_dump = not tool._schema_supplied and tool.input_model is not None
    validation_input_model = tool.input_model if typed_model_dump else None
    reject_unexpected = _rejects_unexpected_arguments(
        argument_schema, typed_model_dump=typed_model_dump
    ) and not _accepts_arbitrary_argument_names(validation_input_model)
    try:
        unexpected_arguments = _unexpected_argument_names(
            parsed_args,
            argument_schema,
            reject_unexpected=reject_unexpected,
            input_model=validation_input_model,
        )
        if unexpected_arguments:
            raise TypeError(f"Unexpected argument(s) for '{tool.name}'.")
        if typed_model_dump:
            try:
                validation_args = deepcopy(parsed_args)
            except Exception:
                validation_args = parsed_args
            validated = tool.input_model.model_validate(validation_args)
            transformed_args = tool._dump_arguments(validated)
            _validate_arguments_against_schema(
                arguments=transformed_args,
                schema=tool._serialization_schema,
                tool_name=tool.name,
                tuples_as_arrays=True,
                values_prevalidated=True,
            )
            pipeline_args = transformed_args
        else:
            _validate_arguments_against_schema(arguments=parsed_args, schema=argument_schema, tool_name=tool.name)
            pipeline_args = parsed_args
    except (TypeError, ValidationError) as exc:
        # ``response_truncated`` (finish_reason == "length") is a response-wide
        # signal, so it alone doesn't prove THIS call's arguments were cut off —
        # a fully-parsed but schema-invalid call, or one bad call among several,
        # can ride along in a truncated response. Only report truncation when
        # this call's own argument payload is actually incomplete/unparseable
        # (raw JSON string), or when the final content block is a parsed mapping
        # that looks incomplete (Anthropic blocking path). Otherwise it's a real
        # argument error the model should fix by correcting the arguments, not by
        # raising max_tokens.
        arguments_unparseable = _arguments_unparseable(function_call.arguments)
        arguments_cut_off = arguments_unparseable or (
            function_call_may_be_truncated
            and _arguments_look_like_truncated_mapping(function_call.arguments, argument_schema)
        )
        if response_truncated and arguments_cut_off:
            message = (
                "The tool call was cut off at the model's output token limit "
                "(max_tokens) before its arguments were complete, so they could not "
                "be parsed. Raise max_tokens for this model, or split the work into "
                "smaller calls (e.g. write the file in parts)."
            )
            error_kind = "argument_truncated"
        else:
            message = _argument_validation_message(
                tool_name=tool.name,
                arguments=parsed_args,
                arguments_unparseable=arguments_unparseable,
                schema=argument_schema,
                exception=exc,
                reject_unexpected=reject_unexpected,
                input_model=validation_input_model,
            )
            error_kind = "argument_parsing"
        additional = _result_additional_properties(function_call)
        additional.update(
            {
                TOOL_FAILED_METADATA_KEY: True,
                TOOL_ERROR_KIND_METADATA_KEY: error_kind,
                TOOL_ERROR_MESSAGE_METADATA_KEY: message,
            }
        )
        result = Content.from_function_result(
            call_id=function_call.call_id,  # type: ignore[arg-type]
            result=f"Error: {message}",
            exception=_failure_record_text(exc),
            additional_properties=additional,
        )
        await _record_unexecuted_tool_operation(
            function_call,
            outcome=ToolOutcome.INVALID_ARGUMENTS,
            result_carrier_item_id=result_carrier_item_id,
            result=result,
        )
        return result

    args = dict(pipeline_args)
    try:
        pipeline_args_snapshot = deepcopy(args)
    except Exception:
        pipeline_args_snapshot = None

    call_id = function_call.call_id
    if call_id is None:
        # Landing stamped this call an operation id and the batch already
        # counted it as dispatched, so the terminal is owed here — the same
        # debt the unknown-tool and invalid-argument exits above settle.
        await _record_unexecuted_tool_operation(function_call, outcome=ToolOutcome.FILTERED)
        raise KeyError(f'Function "{function_call.name}" is missing call_id.')

    context = FunctionInvocationContext(
        function=tool,
        arguments=args,
        session=invocation_session,
        kwargs=runtime_kwargs.copy(),
        tools=live_tools,
    )
    context.same_tool_calls_in_batch = same_tool_calls_in_batch
    # Always pass call_id to middleware.
    context.metadata["call_id"] = call_id
    # Seed the kernel-stamped invocation ordinal for middleware readers
    # (approval, event persistence). The call content is the single source;
    # the middleware never renumbers a seeded context.
    ordinal = read_tool_invocation_order(function_call.additional_properties)
    if ordinal is not None:
        context.metadata[TOOL_INVOCATION_ORDER_KEY] = ordinal
    # Trajectory identities for middleware readers: the call's tool operation
    # id (landing stamped it on the content) and the item id every result
    # construction path below will give this call's result.
    operation_id = read_operation_id(function_call.additional_properties)
    if operation_id is not None:
        context.metadata[OPERATION_ID_KEY] = operation_id
    # The call's own item id travels the same way, so a call that runs names
    # its request item exactly like one that never reaches the pipeline.
    call_item_id = read_analytics_item_id(function_call.additional_properties)
    if call_item_id is not None:
        context.metadata[ANALYTICS_ITEM_ID_KEY] = call_item_id
    context.metadata[TOOL_RESULT_ITEM_ID_METADATA_KEY] = new_analytics_id()
    context.metadata[TOOL_RESULT_CARRIER_ITEM_ID_METADATA_KEY] = result_carrier_item_id
    if result_commit is not None:

        def _commit_sub_agent_interruption(metadata: Mapping[str, Any]) -> None:
            result_commit.commit_interrupted(function_call, context, metadata)

        context.metadata[SUB_AGENT_RESULT_COMMIT_CALLBACK_KEY] = _commit_sub_agent_interruption

    async def final_function_handler(context_obj: Any) -> Any:
        # Middleware sees normalized callable-keyspace values. An untouched
        # context goes back through invoke in the original validation keyspace;
        # a trusted mapping rewrite bypasses validation-keyspace conversion.
        logger.info("Function name: %s", tool.name)
        if logger.isEnabledFor(logging.DEBUG):
            schema_names = tool.parameters().get("properties") or {}
            logger.debug("Function arguments: %s", _describe_arguments(context_obj.arguments, schema_names))
        # No failure log line here: the outer except below already reports
        # failures (tool name + exception + "returning an error result") — a
        # second line would be duplication (invoke itself is chrys-owned and
        # telemetry/logging-free; the span records the exception).
        start = perf_counter()
        try:
            arguments_unchanged = (
                pipeline_args_snapshot is not None
                and context_obj.arguments is args
                and _middleware_arguments_equal(context_obj.arguments, pipeline_args_snapshot)
            )
        except Exception:
            arguments_unchanged = False
        invocation_arguments = parsed_args if arguments_unchanged else context_obj.arguments
        arguments_in_callable_keyspace = (
            typed_model_dump and not arguments_unchanged and isinstance(invocation_arguments, Mapping)
        )

        def _commit_raw_result(value: Any) -> None:
            if result_commit is None:
                return
            # This callback runs before FunctionMiddleware ``finally`` blocks
            # publish their measured span. The instant timing makes the raw
            # crash checkpoint complete; terminal construction overwrites it
            # from ``context.metadata`` after middleware unwinds.
            additional = _result_additional_properties(function_call, context)
            parent_metadata = sub_agent_parent_result_metadata.get()
            if parent_metadata is not None:
                carried: dict[str, Any] = {}
                if parent_metadata.sub_agent_invocation_id:
                    carried["sub_agent_invocation_id"] = parent_metadata.sub_agent_invocation_id
                if parent_metadata.sub_agent_log_file:
                    carried["sub_agent_log_file"] = parent_metadata.sub_agent_log_file
                if parent_metadata.sub_agent_audit_complete:
                    carried["sub_agent_audit_complete"] = True
                if carried:
                    additional[TOOL_RESULT_METADATA_KEY] = carried
            result_commit.commit_raw(
                Content.from_function_result(
                    call_id=call_id,
                    result=value,
                    additional_properties=additional,
                )
            )

        try:
            if TELEMETRY_GATE.enabled:
                result = await _invoke_with_function_span(
                    tool,
                    context_obj,
                    call_id,
                    arguments=invocation_arguments,
                    arguments_in_callable_keyspace=arguments_in_callable_keyspace,
                    tool_result_ceiling_tokens=tool_result_ceiling_tokens,
                )
            else:
                result = await tool.invoke(
                    arguments=invocation_arguments,
                    context=context_obj,
                    tool_call_id=call_id,
                    _arguments_in_callable_keyspace=arguments_in_callable_keyspace,
                )
                result = apply_result_ceiling(result, tool_result_ceiling_tokens)
        except SyncToolCancelledAfterCompletion as exc:
            # The worker finished before cancellation landed: journal the
            # completed value as this slot's raw result so the interrupt
            # handler preserves it instead of recording a fabricated
            # interruption for work that actually ran.
            exc.completed_result = apply_result_ceiling(exc.completed_result, tool_result_ceiling_tokens)
            _commit_raw_result(exc.completed_result)
            raise
        _commit_raw_result(result)
        logger.info("Function %s succeeded in %.3fs.", tool.name, perf_counter() - start)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Function result: %s",
                _describe_result(result, skip_parsing=_is_skip_parsing_sentinel(tool.result_parser)),
            )
        return result

    try:
        function_result = await pipeline.execute(
            context=context,
            final_handler=final_function_handler,
        )
        return Content.from_function_result(
            call_id=call_id,
            result=function_result,
            additional_properties=_result_additional_properties(function_call, context),
        )
    except asyncio.CancelledError:
        # Function middleware ``finally`` blocks have completed at this
        # boundary, so their measured timing and result metadata are now on
        # the invocation context. Commit with that context before the outer
        # batch-level cancellation fallback loses it.
        if result_commit is not None:
            result_commit.commit_interrupted(function_call, context)
        raise
    except MiddlewareTermination as term_exc:
        # Re-raise to signal loop termination, capturing any middleware-set
        # result first.
        if context.result is not None:
            term_exc.result = Content.from_function_result(
                call_id=call_id,
                result=context.result,
                additional_properties=_result_additional_properties(function_call, context),
            )
        elif isinstance(term_exc.result, Content):
            # Middleware may cache and reuse a prebuilt result across concurrent
            # calls or later runs.  Own a wrapper per invocation before binding
            # call-scoped identity and metadata; its payload may contain
            # external objects that are deliberately unsafe to deep-copy.
            owned_result = copy(term_exc.result)
            owned_result.call_id = call_id
            prebuilt_properties = term_exc.result.additional_properties
            # This call's properties go on top of whatever the prebuilt result
            # carried, the same fold every other result path uses: a result
            # cached across calls arrives wearing another invocation's identity,
            # and the tool operation and pre-minted item id have to be this
            # one's or the trajectory's terminal names an item nothing saved.
            properties = {**prebuilt_properties, **_result_additional_properties(function_call, context)}
            owned = prebuilt_properties.get(TOOL_RESULT_METADATA_KEY)
            if isinstance(owned, Mapping) and owned:
                # A middleware that built its own result Content owns whatever
                # metadata it chose to attach — fold first-write-wins only.
                properties[TOOL_RESULT_METADATA_KEY] = dict(owned)
            owned_result.additional_properties = properties
            term_exc.result = owned_result
        else:
            term_exc.result = Content.from_function_result(
                call_id=call_id,
                result=term_exc.result,
                additional_properties=_result_additional_properties(function_call, context),
            )
        raise
    except Exception as exc:
        logger.warning(
            "Function '%s' raised an exception; returning an error result to the model. Exception: %r",
            tool.name,
            exc,
        )
        return Content.from_function_result(
            call_id=function_call.call_id,  # type: ignore[arg-type]
            result=tool_error_result_text(exc),
            # Never None: it marks the result failed for the consecutive-error
            # count and the providers' error flag, whatever the model reads.
            exception=_failure_record_text(exc),
            additional_properties=_result_additional_properties(function_call, context),
        )
    finally:
        # Backfill per-call provenance subkeys (builder context resolved from
        # the final middleware args) onto the persisted call content. Runs on
        # every pipeline exit; on cancellation the middleware wrote no
        # carriage, so there is nothing to merge.
        carried_context = context.metadata.get(TOOL_CALL_CONTEXT_METADATA_KEY)
        if isinstance(carried_context, Mapping):
            merge_tool_call_context_property(function_call.additional_properties, carried_context)


async def _execute_function_calls(
    *,
    function_calls: Sequence[Content],
    result_commits: Sequence[_ResultCommit] | None,
    tool_options: dict[str, Any],
    custom_args: dict[str, Any],
    invocation_session: AgentSession | None,
    pipeline: FunctionMiddlewarePipeline,
    result_carrier_item_id: str,
    response_truncated: bool = False,
    truncated_final_function_call_ids: set[int] | None = None,
    tool_result_ceiling_tokens: int | None = None,
    dispatch_observer: Callable[[Sequence[Content]], None] | None = None,
) -> tuple[list[Content], bool, bool]:
    """Execute a batch concurrently and collect results, termination and error flags."""
    raw_tools = tool_options.get("tools")
    if not raw_tools:
        return [], False, False

    # Normalize per batch: direct callers and progressive exposure can add
    # tools after agent preparation. Every executed tool must use FunctionTool.invoke.
    tools = normalize_tools(raw_tools)
    tool_options["tools"] = tools
    if not tools:
        return [], False, False
    # Rebuild from the live list so progressive exposure takes effect next batch.
    # Tool admission has already enforced unique names.
    tool_map: dict[str, FunctionTool] = {t.name: t for t in tools if isinstance(t, FunctionTool)}
    live_tools: list[Any] | None = tools

    # Per-name call counts for this batch: middleware with singleton semantics
    # (e.g. whole-list-replacement tools) reads the stamped count to reject
    # same-batch duplicates deterministically instead of racing the gather.
    same_name_counts = Counter(fc.name for fc in function_calls)

    commits: Sequence[_ResultCommit | None]
    if result_commits is None:
        commits = (None,) * len(function_calls)
    else:
        if len(result_commits) != len(function_calls):
            raise ValueError("Tool-result commit bindings do not match the extracted call batch.")
        commits = result_commits

    async def invoke_with_termination_handling(
        function_call: Content,
        result_commit: _ResultCommit | None,
    ) -> tuple[Content, bool]:
        """Catch MiddlewareTermination, returning ``(result, should_terminate)``."""
        # The handover is per call and happens on this task's own first step,
        # not once for the batch: a task cancelled before it ever runs never
        # executes a line of this body, so its operation has to stay in the
        # loop's ledger for the reconciliation pass to close.
        if dispatch_observer is not None:
            dispatch_observer((function_call,))
        try:
            result = await _invoke_function_call(
                function_call,
                result_commit=result_commit,
                tool_map=tool_map,
                custom_args=custom_args,
                invocation_session=invocation_session,
                pipeline=pipeline,
                live_tools=live_tools,
                response_truncated=response_truncated,
                function_call_may_be_truncated=(
                    response_truncated
                    and truncated_final_function_call_ids is not None
                    and id(function_call) in truncated_final_function_call_ids
                ),
                same_tool_calls_in_batch=same_name_counts[function_call.name],
                tool_result_ceiling_tokens=tool_result_ceiling_tokens,
                result_carrier_item_id=result_carrier_item_id,
            )
            result = apply_result_ceiling(result, tool_result_ceiling_tokens)
            if result_commit is not None:
                result_commit.commit_final(result)
            return (result, False)
        except MiddlewareTermination as exc:
            # exc.result may already be a Content (set by _invoke_function_call)
            # or a raw value.
            if isinstance(exc.result, Content):
                bounded_result = apply_result_ceiling(exc.result, tool_result_ceiling_tokens)
                if result_commit is not None:
                    result_commit.commit_final(bounded_result)
                return (bounded_result, True)
            result_content = Content.from_function_result(
                call_id=function_call.call_id,  # type: ignore[arg-type]
                result=exc.result,
                additional_properties=_result_additional_properties(function_call),
            )
            result_content = apply_result_ceiling(result_content, tool_result_ceiling_tokens)
            if result_commit is not None:
                result_commit.commit_final(result_content)
            return (result_content, True)
        except asyncio.CancelledError:
            if result_commit is not None:
                result_commit.commit_interrupted(function_call, None)
            raise

    tasks = [
        asyncio.create_task(invoke_with_termination_handling(function_call, result_commit))
        for function_call, result_commit in zip(function_calls, commits, strict=True)
    ]
    try:
        settled = await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        for task in tasks:
            if not task.done():
                task.cancel()
        settling = asyncio.gather(*tasks, return_exceptions=True)
        while not settling.done():
            try:
                await asyncio.shield(settling)
            except asyncio.CancelledError:
                continue
        raise

    for outcome in settled:
        if isinstance(outcome, BaseException):
            raise outcome
    execution_results = [outcome for outcome in settled if not isinstance(outcome, BaseException)]

    contents: list[Content] = [result[0] for result in execution_results]
    should_terminate = any(result[1] for result in execution_results)
    had_errors = any(fcr.exception is not None for fcr in contents if fcr.type == "function_result")
    return contents, should_terminate, had_errors


def _handle_function_call_results(
    *,
    response: ChatResponse,
    function_call_results: list[Content],
    fcc_messages: list[Message],
    echo_registry: WeakIdentityRegistry,
    errors_in_a_row: int,
    had_errors: bool,
    max_errors: int,
    result_carrier_item_id: str,
    recorder: LoopRecorder | None = None,
) -> tuple[str, int]:
    """Fold a batch of tool results back into the response.

    Results are ``function_result`` contents. Returns
    ``(action, errors_in_a_row)`` where action is
    ``"continue"`` or ``"stop"`` (consecutive-error cap reached: submit the
    collected results once more with tools disabled).

    New results register in the echo registry —
    they are part of the run's transcript from here on, so a client echoing
    them back in a later response must see them removed as history rather
    than landing a duplicate answer (``_land_response_contents``).
    """
    for result in function_call_results:
        echo_registry.register(result)
    if had_errors:
        errors_in_a_row += 1
        reached_error_limit = errors_in_a_row >= max_errors
        if reached_error_limit:
            logger.warning(
                "Maximum consecutive function call errors reached (%d). "
                "Stopping further function calls for this request.",
                max_errors,
            )
    else:
        errors_in_a_row = 0
        reached_error_limit = False

    result_message = Message(role="tool", contents=function_call_results)
    # The carrier is a persisted item of its own. Its batch-wide id was minted
    # at dispatch, before any tool terminal named it, so the next context
    # revision and every tool.operation.finished agree on this same item.
    result_message.additional_properties[ANALYTICS_ITEM_ID_KEY] = result_carrier_item_id
    response.messages.append(result_message)
    if recorder is not None:
        recorder.seal_exchange(result_message)
    fcc_messages.extend(response.messages)
    return ("stop" if reached_error_limit else "continue", errors_in_a_row)


# --------------------------------------------------------------------------- #
# ToolLoopLayer
# --------------------------------------------------------------------------- #


class ToolLoopLayer:
    """Composition-style tool-invocation loop over an inner chat layer.

    Constructor settings bound iteration and tool execution. Clients that
    need a single model call can use the inner chat layer directly.

    Per-run state arrives via ``client_kwargs``: ``session`` (continuation
    bookkeeping target) and ``loop_recorder`` (chrys interrupt-recovery
    recorder) are popped here; everything else flows to the inner layer
    untouched, with continuation ids written back into the same dict between
    iterations.
    """

    def __init__(
        self,
        inner: ChatMiddlewareLayer,
        *,
        middleware: FunctionMiddleware | Sequence[FunctionMiddleware] | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
        max_function_calls: int | None = None,
        tool_result_ceiling_tokens: int | None = None,
    ) -> None:
        self.inner = inner
        split = split_middleware(middleware)
        if split.chat:
            raise TypeError(
                "ToolLoopLayer constructor middleware must be FunctionMiddleware; "
                f"chat middleware belongs to the inner ChatMiddlewareLayer, got {split.chat!r}"
            )
        self.function_middleware: list[FunctionMiddleware] = split.function
        self.max_iterations = max_iterations
        self.max_consecutive_errors = max_consecutive_errors
        self.max_function_calls = max_function_calls
        self.tool_result_ceiling_tokens = tool_result_ceiling_tokens

    def __getattr__(self, name: str) -> Any:
        # Delegate unknown attributes to the inner layer so agent-facing client
        # surface (model, STORES_BY_DEFAULT, ...) resolves through the stack.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    async def aclose(self) -> None:
        await self.inner.aclose()

    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
        middleware: Sequence[ChatMiddleware | FunctionMiddleware] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        """Build a lazy model/tool run.

        ``stream=False`` returns an un-awaited coroutine; ``stream=True`` returns
        a ``ResponseStream`` synchronously. Execution begins when the result is driven.
        """
        # Merge the middleware kwarg with the
        # client_kwargs channel, split chat vs function, pop per-run state.
        effective_client_kwargs = dict(client_kwargs) if client_kwargs is not None else {}
        if middleware is not None:
            existing = effective_client_kwargs.get("middleware")
            effective_client_kwargs["middleware"] = [
                *_as_middleware_list(existing),
                *middleware,
            ]
        runtime_split = split_middleware(effective_client_kwargs.pop("middleware", None))
        pipeline = FunctionMiddlewarePipeline(*self.function_middleware, *runtime_split.function)
        per_call_chat: list[ChatMiddleware] = runtime_split.chat

        raw_session = effective_client_kwargs.pop("session", None)
        invocation_session: AgentSession | None = raw_session if isinstance(raw_session, AgentSession) else None
        raw_recorder = effective_client_kwargs.pop("loop_recorder", None)
        recorder: LoopRecorder | None = raw_recorder if isinstance(raw_recorder, LoopRecorder) else None
        raw_injection_probe = effective_client_kwargs.pop("consumed_injection_message_probe", None)
        injection_probe = (
            raw_injection_probe if isinstance(raw_injection_probe, ConsumedInjectionMessageProbe) else None
        )
        raw_wire_retry_policy = effective_client_kwargs.pop("wire_retry_policy", None)
        raw_token_observer = effective_client_kwargs.pop("continuation_token_observer", None)
        continuation_token_observer: Callable[[Any], None] | None = (
            raw_token_observer if callable(raw_token_observer) else None
        )
        raw_trajectory = effective_client_kwargs.pop(TRAJECTORY_CONTEXT_KWARG, None)
        trajectory: TrajectoryContext | None = raw_trajectory if isinstance(raw_trajectory, TrajectoryContext) else None

        # Sentinel start: the logical call may begin with a retry-owned token
        # already in its options (whole-run retry resuming a background
        # response), so the first tracked value — including a terminal
        # ``None`` — must always be applied, never swallowed by dedupe.
        untracked = object()
        last_tracked_continuation_token: Any = untracked

        def _track_continuation_token(token: Any) -> None:
            # Mirror the live token into the retry owner's request state: a
            # transient poll failure must resume THIS response on the next
            # attempt (outer whole-run retry included), never re-issue the
            # original create request. Providers re-announce the id on every
            # progress event, so identical repeats are dropped.
            nonlocal last_tracked_continuation_token
            if token == last_tracked_continuation_token:
                return
            last_tracked_continuation_token = token
            if token is None:
                mutable_options.pop("continuation_token", None)
            else:
                mutable_options["continuation_token"] = token
            if continuation_token_observer is not None:
                continuation_token_observer(token)

        filtered_kwargs = effective_client_kwargs

        additional_function_arguments = dict(function_invocation_kwargs) if function_invocation_kwargs else {}
        if options and (additional_opts := options.get("additional_function_arguments")):
            additional_function_arguments.update(additional_opts)

        mutable_options: dict[str, Any] = dict(options) if options else {}
        # Tool-invocation-only key; not recognized by chat service APIs.
        mutable_options.pop("additional_function_arguments", None)
        # Run-local mutable tools list (progressive tool exposure): fresh list so
        # the caller's container is never mutated, same object shared with the
        # model and the per-batch tool map.
        if mutable_options.get("tools"):
            mutable_options["tools"] = normalize_tools(mutable_options["tools"])

        try:
            stores_by_default = bool(self.STORES_BY_DEFAULT)
        except AttributeError:
            stores_by_default = False
        try:
            force_stateless = bool(self.FORCES_STATELESS)
        except AttributeError:
            force_stateless = False
        storage = resolve_storage_mode_and_handles(
            mutable_options,
            stores_by_default=stores_by_default,
            client_kwargs=filtered_kwargs,
            force_stateless=force_stateless,
        )
        wire_retry_policy = (
            cast(WireRetryPolicy, raw_wire_retry_policy)
            if raw_wire_retry_policy is not None and not storage.service_side
            else None
        )
        max_errors = self.max_consecutive_errors

        def _wire_request_observer(echo_registry: WeakIdentityRegistry) -> Callable[[Sequence[Message]], None]:
            internal_observer = functools.partial(
                _record_wire_request_content_identities,
                echo_registry=echo_registry,
            )
            if request_message_observer is None:
                return internal_observer

            def observe(messages: Sequence[Message]) -> None:
                # Echo tracking remains the loop's invariant even if a caller
                # observer raises; callers then see the exact same provider
                # views immediately after the internal identity recorder.
                internal_observer(messages)
                request_message_observer(messages)

            return observe

        # Trajectory bookkeeping (all no-ops without a bound context): one
        # ``model.cycle`` per loop acquisition, one ``model.exchange`` per
        # wire attempt, ``retry.*`` around every wire retry. The exchange
        # trace rides to the wire client in ``client_kwargs`` — the loop
        # cannot bind it ambiently because the provider stream is lazy and
        # resolves outside this call's context.
        cycle_trajectory: TrajectoryContext | None = None
        cycle_started_ns = 0
        cycle_exchange_count = 0
        active_exchange: ExchangeTrace | None = None
        pending_exchange_id: str | None = None
        last_exchange_id: str | None = None
        landed_exchange_trajectory: TrajectoryContext | None = None
        undispatched_tool_operations: dict[int, tuple[Content, TrajectoryContext | None]] = {}

        def _register_landed_tool_operations(
            response: ChatResponse,
            function_calls: Sequence[Content],
        ) -> None:
            """Mint and remember every operation until the one dispatch point takes it."""
            _stamp_function_call_operations(response, function_calls)
            operation_context = landed_exchange_trajectory or current_trajectory()
            for function_call in function_calls:
                undispatched_tool_operations[id(function_call)] = (function_call, operation_context)

        def _mark_tool_operations_dispatched(function_calls: Sequence[Content]) -> None:
            for function_call in function_calls:
                undispatched_tool_operations.pop(id(function_call), None)

        async def _settle_undispatched_tool_operations(*, queued: bool = False) -> None:
            """Close every operation minted but never handed to the execution batch.

            This is the single reconciliation point for no-tools responses,
            exhaustion tails, future early returns, and failures between
            landing and dispatch. Every entry keeps the exchange context it
            landed under rather than consulting the loop's later current one.
            """
            pending = list(undispatched_tool_operations.values())
            undispatched_tool_operations.clear()
            first_failure: BaseException | None = None
            for function_call, operation_context in pending:
                try:
                    with trajectory_scope(operation_context):
                        await _record_unexecuted_tool_operation(
                            function_call,
                            outcome=ToolOutcome.FILTERED,
                            queued=queued,
                        )
                except BaseException as exc:
                    # The helper settles the current pair before propagating a
                    # cancelled ack. Continue so one cancellation cannot strand
                    # the rest of a parallel batch, then preserve the caller's
                    # original control-flow signal.
                    if first_failure is None:
                        first_failure = exc
            if first_failure is not None:
                raise first_failure

        async def _trajectory_emit(draft: Any) -> None:
            if trajectory is None:
                return
            try:
                await trajectory.sink.emit(draft)
            except Exception:
                logger.debug("Trajectory emit failed", exc_info=True)

        async def _trajectory_cycle_started(cycle_index: int) -> None:
            nonlocal cycle_trajectory, cycle_started_ns, cycle_exchange_count, last_exchange_id
            if trajectory is None:
                return
            cycle_id = new_analytics_id()
            cycle_trajectory = trajectory.with_cycle(cycle_id)
            cycle_started_ns = monotonic_ns()
            cycle_exchange_count = 0
            last_exchange_id = None
            await _trajectory_emit(
                cycle_trajectory.draft(
                    TrajectoryEventType.MODEL_CYCLE_STARTED,
                    operation_id=cycle_id,
                    parent_operation_id=trajectory.run_operation_id,
                    payload={"cycle_index": cycle_index, "tools_offered": bool(mutable_options.get("tools"))},
                )
            )

        def _trajectory_cycle_finished_draft(*, outcome: str, function_call_count: int = 0) -> Any:
            nonlocal landed_exchange_trajectory
            cycle = cycle_trajectory
            if cycle is None or cycle.cycle_operation_id is None:
                return None
            landed_exchange_trajectory = cycle.with_exchange(last_exchange_id)
            return cycle.draft(
                TrajectoryEventType.MODEL_CYCLE_FINISHED,
                operation_id=cycle.cycle_operation_id,
                parent_operation_id=cycle.run_operation_id,
                payload={
                    "outcome": outcome,
                    "exchange_count": cycle_exchange_count,
                    "function_call_count": function_call_count,
                    # Monotonic at both ends: a wall clock that steps mid-cycle
                    # would otherwise stretch or flatten the span it measures.
                    "duration_ms": max(0, (monotonic_ns() - cycle_started_ns) // 1_000_000),
                    "final_exchange_operation_id": last_exchange_id,
                },
                measurements={"/payload/duration_ms": measurement(MeasurementSource.MONOTONIC_CLOCK, method_version=1)},
            )

        async def _trajectory_cycle_finished(
            *,
            outcome: str,
            function_call_count: int = 0,
            response: ChatResponse | None = None,
            function_calls: Sequence[Content] = (),
        ) -> None:
            nonlocal cycle_trajectory
            draft = _trajectory_cycle_finished_draft(outcome=outcome, function_call_count=function_call_count)
            if response is not None:
                _register_landed_tool_operations(response, function_calls)
            if draft is None:
                return
            if response is not None:
                await _trajectory_hosted_calls(response)
            # Given up only here: an interrupt during the hosted-call records
            # above still leaves the cycle for the abort path to close.
            cycle_trajectory = None
            await _trajectory_emit(draft)

        async def _trajectory_hosted_calls(response: ChatResponse) -> None:
            # Hosted calls never become tool operations (the provider ran
            # them); one fact per call keeps the exchange's fan-out visible
            # without an unbounded array on the exchange event.
            exchange = landed_exchange_trajectory
            if exchange is None or exchange.exchange_operation_id is None:
                return
            for ordinal, hosted_kind in enumerate(_hosted_call_kinds(response)):
                await _trajectory_emit(
                    exchange.draft(
                        TrajectoryEventType.HOSTED_CALL_OBSERVED,
                        operation_id=new_analytics_id(),
                        parent_operation_id=exchange.exchange_operation_id,
                        payload={
                            "parent_exchange_operation_id": exchange.exchange_operation_id,
                            "hosted_kind": hosted_kind,
                            "ordinal": ordinal,
                        },
                    )
                )

        def _trajectory_abort(outcome: str) -> None:
            # Synchronous close for exits that cannot await (cancellation,
            # generator close): the open exchange and cycle are queued in
            # order without waiting for the ack.
            nonlocal cycle_trajectory
            _trajectory_close_exchange(outcome)
            draft = _trajectory_cycle_finished_draft(outcome=outcome)
            cycle_trajectory = None
            if draft is not None and trajectory is not None:
                try:
                    trajectory.sink.emit_soon(draft)
                except Exception:
                    logger.debug("Trajectory cycle close failed", exc_info=True)

        def _trajectory_exchange_scope() -> trajectory_scope:
            # Ambient context for the tool batch a landed response spawns:
            # tool middleware hangs its operations under the producing
            # exchange. Without a loop-owned context the ambient one (if any)
            # is simply re-bound.
            if landed_exchange_trajectory is None:
                return trajectory_scope(current_trajectory())
            return trajectory_scope(landed_exchange_trajectory)

        def _trajectory_begin_exchange() -> dict[str, Any]:
            nonlocal active_exchange, pending_exchange_id, last_exchange_id, cycle_exchange_count
            active_exchange = None
            if cycle_trajectory is None:
                return filtered_kwargs
            exchange_id = pending_exchange_id or new_analytics_id()
            pending_exchange_id = None
            last_exchange_id = exchange_id
            cycle_exchange_count += 1
            active_exchange = ExchangeTrace(cycle_trajectory.with_exchange(exchange_id))
            return {**filtered_kwargs, TRAJECTORY_EXCHANGE_KWARG: active_exchange}

        def _trajectory_close_exchange(outcome: str, *, exc: BaseException | None = None) -> None:
            # Idempotent: a wire client that already reported its own terminal
            # marker wins; this closes the abandoned/failed remainder.
            nonlocal active_exchange, last_exchange_id, cycle_exchange_count
            trace = active_exchange
            if trace is None:
                return
            active_exchange = None
            # A middleware beneath the loop may have re-issued the request in
            # place (validation retry): the handle then names the final
            # exchange and every re-issue was one more acquisition.
            last_exchange_id = trace.operation_id
            cycle_exchange_count += trace.generation
            payload: dict[str, Any] = {}
            if exc is not None:
                payload["error_code"] = type(exc).__name__
                policy = wire_retry_policy
                payload["retryable"] = bool(policy is not None and policy.is_retryable(exc))
            try:
                trace.finished(outcome=outcome, payload=payload)
            except Exception:
                logger.debug("Trajectory exchange close failed", exc_info=True)

        async def _trajectory_retry(
            *, reason_code: str, retry_mode: str, delay_seconds: int, fallback_to_blocking: bool
        ) -> None:
            nonlocal pending_exchange_id
            if cycle_trajectory is None:
                return
            previous = last_exchange_id
            next_id = new_analytics_id()
            pending_exchange_id = next_id
            await _trajectory_emit(
                cycle_trajectory.draft(
                    TrajectoryEventType.RETRY_SCHEDULED,
                    operation_id=next_id,
                    parent_operation_id=cycle_trajectory.cycle_operation_id,
                    payload={
                        "reason_code": reason_code,
                        "delay_ms": max(0, int(delay_seconds * 1000)),
                        "retry_mode": retry_mode,
                        "previous_operation_id": previous,
                        "committed_work_present": bool(recorder is not None and recorder.committed_count > 0),
                        "fallback_to_blocking": fallback_to_blocking,
                    },
                )
            )

        async def _trajectory_retry_started(*, retry_mode: str) -> None:
            if cycle_trajectory is None or pending_exchange_id is None:
                return
            await _trajectory_emit(
                cycle_trajectory.draft(
                    TrajectoryEventType.RETRY_STARTED,
                    operation_id=pending_exchange_id,
                    parent_operation_id=cycle_trajectory.cycle_operation_id,
                    payload={
                        "retry_mode": retry_mode,
                        "next_operation_id": pending_exchange_id,
                        "previous_operation_id": last_exchange_id,
                    },
                )
            )

        def _cancel_outcome() -> str:
            policy = wire_retry_policy
            return (
                ExchangeOutcome.INTERRUPTED
                if policy is not None and policy.is_interrupted()
                else ExchangeOutcome.CANCELLED
            )

        async def _wire_call(
            prepped: list[Message],
            *,
            as_stream: bool,
            stream_update_filter: Callable[[ChatResponseUpdate], ChatResponseUpdate] | None = None,
            request_message_observer: Callable[[Sequence[Message]], None],
        ) -> Any:
            # The recorder keeps the loop's canonical objects (its identity
            # dedup depends on re-recorded history being the SAME objects).
            if recorder is not None:
                await recorder.record_pre_call(prepped)
            # EVERY outgoing message is a per-call view — caller history
            # included — so a client mutating a received message in place can
            # corrupt neither the loop's transcript nor the caller's
            # session-state objects. Message-metadata write-through survives:
            # views share the wrapper's additional_properties dict, which is
            # how compaction exclusion flags reach stored history.
            wire_view = [_wire_message_view(m) for m in prepped]
            return self.inner.get_response(
                wire_view,
                stream=as_stream,
                stream_update_filter=stream_update_filter,
                request_message_observer=request_message_observer,
                options=mutable_options,
                middleware=per_call_chat,
                compaction_strategy=compaction_strategy,
                tokenizer=tokenizer,
                client_kwargs=_trajectory_begin_exchange(),
            )

        async def _schedule_wire_retry(
            policy: WireRetryPolicy,
            exc: BaseException,
            *,
            message: str,
            attempt: int,
            max_attempts: int,
            delay_seconds: int | None = None,
            fallback_to_blocking: bool = False,
        ) -> None:
            if mutable_options.get("continuation_token") is None:
                # A retry that resumes an already-created response via its
                # continuation token must NOT replay consumed injections: the
                # create that consumed them succeeded (the provider holds
                # them), and a poll never consumes a replay — the batch would
                # sit stranded until commit destroys it.
                policy.before_retry()
            delay = policy.backoff_seconds(attempt - 1) if delay_seconds is None else delay_seconds
            stalled = isinstance(exc, StreamStall)
            retry_mode = RetryMode.STALL_FALLBACK if fallback_to_blocking else RetryMode.WIRE
            await _trajectory_retry(
                reason_code=RetryReason.STREAM_STALL if stalled else RetryReason.TRANSIENT_ERROR,
                retry_mode=retry_mode,
                delay_seconds=delay,
                fallback_to_blocking=fallback_to_blocking,
            )
            await policy.on_retry(message, attempt, max_attempts, delay, exc)
            if policy.is_interrupted() or await policy.sleep(delay):
                raise asyncio.CancelledError
            await _trajectory_retry_started(retry_mode=retry_mode)

        async def _pause_between_continuation_polls() -> None:
            # Task cancellation is the interrupt channel for both service and
            # local runs; the policy flag check covers soft interrupts that
            # only set state.
            policy = wire_retry_policy
            if policy is not None and policy.is_interrupted():
                raise asyncio.CancelledError
            if CONTINUATION_POLL_INTERVAL_SECONDS > 0:
                await asyncio.sleep(CONTINUATION_POLL_INTERVAL_SECONDS)

        def _hosted_commits_vetoing_replay(policy: Any) -> tuple[str, ...]:
            # Upward-safe seam: a policy may expose the hosted tool calls the
            # failed attempt already executed server-side (hosted MCP / hosted
            # shell). Without a live continuation token a retry re-creates the
            # request and re-runs those side effects; with one it merely
            # resumes the same response, which is safe.
            hosted_probe = getattr(policy, "hosted_commits_in_flight", None)
            hosted_commits = tuple(hosted_probe()) if callable(hosted_probe) else ()
            if hosted_commits and mutable_options.get("continuation_token") is None:
                return hosted_commits
            return ()

        async def _blocking_response_with_retry(
            prepped: list[Message],
            *,
            request_message_observer: Callable[[Sequence[Message]], None],
        ) -> ChatResponse:
            retry_attempt = 0
            while True:
                policy = wire_retry_policy
                if policy is not None and policy.is_interrupted():
                    raise asyncio.CancelledError
                try:
                    while True:
                        response = await _resolve_response(
                            await _wire_call(
                                prepped,
                                as_stream=False,
                                request_message_observer=request_message_observer,
                            )
                        )
                        _trajectory_close_exchange(ExchangeOutcome.SUCCESS)
                        if response.continuation_token is None:
                            _track_continuation_token(None)
                            if injection_probe is not None:
                                injection_probe.commit()
                            return response
                        _track_continuation_token(response.continuation_token)
                        await _pause_between_continuation_polls()
                except asyncio.CancelledError:
                    _trajectory_close_exchange(_cancel_outcome())
                    raise
                except Exception as exc:
                    _trajectory_close_exchange(ExchangeOutcome.ERROR, exc=exc)
                    if invalidates_continuation_token(exc):
                        # The failure judged a terminal response: a retry must
                        # issue a fresh request, never re-poll the completed
                        # (immutable) one.
                        _track_continuation_token(None)
                    if policy is None or not policy.is_retryable(exc) or retry_attempt >= policy.max_retries:
                        raise
                    hosted_commits = _hosted_commits_vetoing_replay(policy)
                    if hosted_commits:
                        logger.warning(
                            "Not retrying wire call: provider-hosted tool call(s) already "
                            "executed in the failed attempt (%s)",
                            ", ".join(str(label) for label in hosted_commits),
                        )
                        raise
                    retry_attempt += 1
                    await _schedule_wire_retry(
                        policy,
                        exc,
                        message=clean_error_message(exc),
                        attempt=retry_attempt,
                        max_attempts=policy.max_retries,
                    )

        async def _watchdog_await(awaitable: Awaitable[Any], timeout: float | None, label: str) -> Any:
            # Idle timing: a pull whose first byte waits on compaction (and
            # its LAST_WORDS side call) stays alive while that work reports
            # progress, and stalls after *timeout* without any.
            if timeout is None:
                return await awaitable
            event_loop = asyncio.get_running_loop()
            last_progress = event_loop.time()

            def _on_progress() -> None:
                nonlocal last_progress
                last_progress = event_loop.time()

            task = start_with_wire_progress(awaitable, _on_progress)
            try:
                while not task.done():
                    idle_budget = last_progress + timeout - event_loop.time()
                    if idle_budget <= 0:
                        break
                    await asyncio.wait((task,), timeout=idle_budget)
            except asyncio.CancelledError:
                # The pull task may still be running INSIDE the stream's
                # generator; closing that stream before the task settles
                # would raise "asynchronous generator is already running".
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
            if task.done():
                return task.result()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise StreamStall(f"{label} produced no progress for {timeout:g}s")

        def _streaming_response_with_retry(
            prepped: list[Message],
            *,
            stream_update_filter: Callable[[ChatResponseUpdate], ChatResponseUpdate],
            request_message_observer: Callable[[Sequence[Message]], None],
        ) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
            final_response: ChatResponse | None = None
            logical_stream: ResponseStream[ChatResponseUpdate, ChatResponse]

            async def _updates() -> AsyncIterable[ChatResponseUpdate]:
                nonlocal final_response
                retry_attempt = 0
                stall_retry_attempt = 0
                while True:
                    policy = wire_retry_policy
                    if policy is not None and policy.is_interrupted():
                        raise asyncio.CancelledError
                    inner_stream: ResponseStream[ChatResponseUpdate, ChatResponse] | None = None
                    try:
                        while True:
                            raw_stream = await _wire_call(
                                prepped,
                                as_stream=True,
                                stream_update_filter=stream_update_filter,
                                request_message_observer=request_message_observer,
                            )
                            if not isinstance(raw_stream, ResponseStream):
                                raise TypeError("Streaming wire call did not return a ResponseStream.")
                            inner_stream = raw_stream
                            inner_stream.with_update_filter(stream_update_filter)
                            stream_usage_chunks: list[Mapping[str, Any]] = []
                            iterator = inner_stream.__aiter__()
                            while True:
                                try:
                                    update = await _watchdog_await(
                                        iterator.__anext__(),
                                        policy.stall_timeout_seconds if policy is not None else None,
                                        "Streaming response",
                                    )
                                except StopAsyncIteration:
                                    break
                                if update.continuation_token is not None:
                                    # Providers announce the background
                                    # response id mid-stream; mirror it
                                    # immediately so a disconnect before
                                    # finalization retries by retrieval, not
                                    # by a duplicate create.
                                    _track_continuation_token(update.continuation_token)
                                stream_usage_chunks.extend(_stream_usage_chunks(update))
                                yield update
                            response = await _watchdog_await(
                                inner_stream.get_final_response(),
                                policy.stall_timeout_seconds if policy is not None else None,
                                "Streaming response finalization",
                            )
                            inner_stream = None
                            _trajectory_close_exchange(ExchangeOutcome.SUCCESS)
                            response.latest_usage_details = normalize_stream_usage(stream_usage_chunks)
                            if response.latest_usage_details is None:
                                response.latest_usage_details = response.usage_details
                            if response.continuation_token is None:
                                _track_continuation_token(None)
                                final_response = response
                                if injection_probe is not None:
                                    injection_probe.commit()
                                return
                            _track_continuation_token(response.continuation_token)
                            await _pause_between_continuation_polls()
                    except asyncio.CancelledError:
                        _trajectory_close_exchange(_cancel_outcome())
                        if inner_stream is not None:
                            await inner_stream.aclose()
                        raise
                    except Exception as exc:
                        if isinstance(exc, StreamStall):
                            if active_exchange is not None:
                                active_exchange.stall_observed()
                            _trajectory_close_exchange(ExchangeOutcome.STALLED, exc=exc)
                        else:
                            _trajectory_close_exchange(ExchangeOutcome.ERROR, exc=exc)
                        if inner_stream is not None:
                            try:
                                await inner_stream.aclose()
                            except Exception:
                                logger.debug("Failed to close abandoned provider stream", exc_info=True)
                        if invalidates_continuation_token(exc):
                            # The failure judged a terminal response: a retry
                            # must issue a fresh request, never re-poll the
                            # completed (immutable) one.
                            _track_continuation_token(None)
                        if policy is None:
                            raise
                        hosted_commits = _hosted_commits_vetoing_replay(policy)
                        if hosted_commits:
                            logger.warning(
                                "Not replaying wire call: provider-hosted tool call(s) already "
                                "executed in the aborted stream (%s)",
                                ", ".join(str(label) for label in hosted_commits),
                            )
                            raise
                        if isinstance(exc, StreamStall):
                            if stall_retry_attempt >= policy.stall_max_retries:
                                if policy.stall_exhausted_action is StallExhaustedAction.RAISE:
                                    raise
                                await _schedule_wire_retry(
                                    policy,
                                    exc,
                                    message="Stream stalled; retrying with a blocking response",
                                    attempt=stall_retry_attempt + 1,
                                    max_attempts=policy.stall_max_retries + 1,
                                    delay_seconds=0,
                                    fallback_to_blocking=True,
                                )
                                logical_stream._updates.clear()
                                yield ChatResponseUpdate.retry_boundary()
                                final_response = await _blocking_response_with_retry(
                                    prepped,
                                    request_message_observer=request_message_observer,
                                )
                                return
                            stall_retry_attempt += 1
                            await _schedule_wire_retry(
                                policy,
                                exc,
                                message="Stream stalled",
                                attempt=stall_retry_attempt,
                                max_attempts=policy.stall_max_retries,
                            )
                            logical_stream._updates.clear()
                            yield ChatResponseUpdate.retry_boundary()
                            continue
                        if not policy.is_retryable(exc) or retry_attempt >= policy.max_retries:
                            raise
                        retry_attempt += 1
                        await _schedule_wire_retry(
                            policy,
                            exc,
                            message=clean_error_message(exc),
                            attempt=retry_attempt,
                            max_attempts=policy.max_retries,
                        )
                        logical_stream._updates.clear()
                        yield ChatResponseUpdate.retry_boundary()

            def _finalizer(_updates: Sequence[ChatResponseUpdate]) -> ChatResponse:
                if final_response is None:
                    raise RuntimeError("Logical streaming call ended without a final response.")
                return final_response

            logical_stream = ResponseStream(_updates(), finalizer=_finalizer)
            return logical_stream

        async def execute_batch(
            function_calls: list[Content],
            *,
            result_carrier_item_id: str,
            result_commits: Sequence[_ResultCommit] | None = None,
            response_truncated: bool = False,
            truncated_final_function_call_ids: set[int] | None = None,
        ) -> tuple[list[Content], bool, bool]:
            return await _execute_function_calls(
                function_calls=function_calls,
                result_commits=result_commits,
                tool_options=mutable_options,
                custom_args=additional_function_arguments,
                invocation_session=invocation_session,
                pipeline=pipeline,
                result_carrier_item_id=result_carrier_item_id,
                response_truncated=response_truncated,
                truncated_final_function_call_ids=truncated_final_function_call_ids,
                tool_result_ceiling_tokens=self.tool_result_ceiling_tokens,
                dispatch_observer=_mark_tool_operations_dispatched,
            )

        if not stream:

            async def _get_response() -> ChatResponse:
                # Non-streaming model/tool iterations.
                errors_in_a_row = 0
                total_function_calls = 0
                next_tool_ordinal = 0
                # Per-run identity registry of every content the conversation
                # holds — seeded from caller history so a client echoing a
                # historical object cannot re-execute a past side-effecting
                # call; _land_response_contents removes reappearances as
                # echoes. ignore_usage: echo filtering never strips usage.
                echo_registry = WeakIdentityRegistry(ignore_usage=True)
                for m in messages:
                    for c in m.contents:
                        echo_registry.register(c)
                wire_request_observer = _wire_request_observer(echo_registry)
                prepped_messages = list(messages)
                fcc_messages: list[Message] = []
                response: ChatResponse | None = None
                aggregated_usage: UsageDetails | None = None

                for _attempt_idx in range(self.max_iterations):
                    await _trajectory_cycle_started(_attempt_idx)
                    response = await _blocking_response_with_retry(
                        prepped_messages,
                        request_message_observer=wire_request_observer,
                    )
                    next_tool_ordinal = _land_response_contents(
                        response, next_tool_ordinal, echo_registry, exchange_operation_id=last_exchange_id
                    )
                    consumed_injection_messages = injection_probe.take_consumed_messages() if injection_probe else []
                    response.latest_usage_details = response.usage_details
                    aggregated_usage = add_usage_details(aggregated_usage, response.usage_details)
                    _update_continuation_state(
                        filtered_kwargs,
                        response,
                        session=invocation_session,
                        options=mutable_options,
                    )
                    if recorder is not None:
                        recorder.record_response(response)

                    if response.conversation_id is not None:
                        prepped_messages = []
                    else:
                        prepped_messages.extend(consumed_injection_messages)

                    function_calls = _extract_function_calls(response)
                    await _trajectory_cycle_finished(
                        outcome=ExchangeOutcome.SUCCESS,
                        function_call_count=len(function_calls),
                        response=response,
                        function_calls=function_calls,
                    )
                    if not (function_calls and mutable_options.get("tools")):
                        _prepend_fcc_messages(response, fcc_messages)
                        response.usage_details = aggregated_usage
                        return _clear_internal_conversation_id(response)

                    result_carrier_item_id = new_analytics_id()
                    result_commits = (
                        recorder.stage_exchange(
                            response.messages,
                            function_calls,
                            result_carrier_item_id=result_carrier_item_id,
                        )
                        if recorder is not None
                        else None
                    )
                    response_truncated = response.finish_reason == "length"
                    with _trajectory_exchange_scope():
                        results, should_terminate, had_errors = await execute_batch(
                            function_calls,
                            result_carrier_item_id=result_carrier_item_id,
                            result_commits=result_commits,
                            response_truncated=response_truncated,
                            truncated_final_function_call_ids=(
                                _truncated_final_function_call_ids(response, function_calls)
                                if response_truncated
                                else None
                            ),
                        )
                    action, errors_in_a_row = _handle_function_call_results(
                        response=response,
                        function_call_results=results,
                        fcc_messages=fcc_messages,
                        echo_registry=echo_registry,
                        errors_in_a_row=errors_in_a_row,
                        had_errors=had_errors,
                        max_errors=max_errors,
                        result_carrier_item_id=result_carrier_item_id,
                        recorder=recorder,
                    )
                    total_function_calls += sum(1 for r in results if r.type == "function_result")
                    if should_terminate:
                        # Middleware termination: return the current response
                        # (tool results already appended) without an fcc
                        # prepend. Streaming termination instead returns the
                        # accumulated transcript because earlier updates were delivered.
                        if storage.service_side:
                            # No further request ever posts this batch's
                            # results, so the service transcript behind the
                            # mirrored handle keeps its calls unanswered.
                            # Record and withhold the handles like the
                            # exhaustion strip does; the marker lets the
                            # agent post-hook install the local fallback.
                            _invalidate_service_continuation_state(response, invocation_session)
                        response.usage_details = aggregated_usage
                        return _clear_internal_conversation_id(response)
                    if action == "stop":
                        # Error threshold reached: force a final non-tool turn so
                        # function results are submitted before exit.
                        mutable_options["tool_choice"] = "none"
                    elif self.max_function_calls is not None and total_function_calls >= self.max_function_calls:
                        # Best-effort limit, checked after each parallel batch.
                        logger.info(
                            "Maximum function calls reached (%d/%d). Stopping further function calls for this request.",
                            total_function_calls,
                            self.max_function_calls,
                        )
                        mutable_options["tool_choice"] = "none"

                    # 'required' tool_choice resets after one iteration.
                    if mutable_options.get("tool_choice") == "required" or (
                        isinstance(mutable_options.get("tool_choice"), dict)
                        and mutable_options.get("tool_choice", {}).get("mode") == "required"
                    ):
                        mutable_options["tool_choice"] = None

                    if response.conversation_id is not None:
                        # Conversation APIs already hold the function-call message;
                        # send only the new result message.
                        prepped_messages.clear()
                        if response.messages:
                            prepped_messages.append(response.messages[-1])
                    else:
                        prepped_messages.extend(response.messages)

                # Loop exhausted: final model call with tool_choice="none" so the
                # model produces plain text instead of orphaned function calls.
                if response is not None:
                    logger.info(
                        "Maximum iterations reached (%d). Requesting final response without tools.",
                        self.max_iterations,
                    )
                mutable_options["tool_choice"] = "none"
                await _trajectory_cycle_started(self.max_iterations)
                response = await _blocking_response_with_retry(
                    prepped_messages,
                    request_message_observer=wire_request_observer,
                )
                next_tool_ordinal = _land_response_contents(
                    response, next_tool_ordinal, echo_registry, exchange_operation_id=last_exchange_id
                )
                # Counted before the strip below: a provider that ignored
                # ``tool_choice="none"`` did ask for calls, and a tail that
                # reported none would read as a compliant one.
                tail_calls = _extract_function_calls(response)
                await _trajectory_cycle_finished(
                    outcome=ExchangeOutcome.SUCCESS,
                    function_call_count=len(tail_calls),
                    response=response,
                    function_calls=tail_calls,
                )
                stripped = _strip_unexecutable_calls_from_response(response)
                _ensure_exhaustion_fallback_response(response)
                # Gate on the resolved storage mode, not response metadata: under
                # conversation storage the service holds the stripped calls even
                # when the parsed response failed to carry the handle, and a
                # client-side-storage response's metadata must survive untouched.
                if stripped and storage.service_side:
                    _invalidate_service_continuation_state(response, invocation_session)
                if injection_probe is not None:
                    injection_probe.take_consumed_messages()
                response.latest_usage_details = response.usage_details
                aggregated_usage = add_usage_details(aggregated_usage, response.usage_details)
                _update_continuation_state(
                    filtered_kwargs,
                    response,
                    session=invocation_session,
                    options=mutable_options,
                )
                if recorder is not None:
                    recorder.record_response(response)
                response.usage_details = aggregated_usage
                _prepend_fcc_messages(response, fcc_messages)
                return _clear_internal_conversation_id(response)

            async def _guarded_get_response() -> ChatResponse:
                interrupted = False
                try:
                    return await _get_response()
                except asyncio.CancelledError:
                    # Nothing below may wait on the writer once the consumer
                    # is gone: the settlement queues its lines instead.
                    interrupted = True
                    _trajectory_abort(_cancel_outcome())
                    raise
                except Exception:
                    _trajectory_abort(ExchangeOutcome.ERROR)
                    raise
                finally:
                    await _settle_undispatched_tool_operations(queued=interrupted)

            return _guarded_get_response()

        response_format = mutable_options.get("response_format")
        aggregated_usage: UsageDetails | None = None
        latest_usage: UsageDetails | None = None
        # Authoritative streamed transcript, recorded by each loop exit: the final response
        # reuses these already-assembled Message objects instead of re-merging
        # raw updates, so kernel-stamped call provenance survives streaming.
        # ``None`` means no loop exit completed — _finalize keeps the
        # ``from_updates`` merge as a defensive fallback.
        final_messages: list[Message] | None = None
        # Set by the exhaustion tail when its strip fired on a service-stored
        # response: _finalize rebuilds the response from raw updates, which
        # still carry the continuation ids the tail withheld.
        service_state_invalidated = False

        async def _stream() -> AsyncIterable[ChatResponseUpdate]:
            # Every exit but the normal return closes the trajectory bookkeeping
            # this generator opened: a consumer abandoning the stream lands here
            # as GeneratorExit, an interrupt as CancelledError.
            interrupted = False
            try:
                # Streaming model/tool iterations.
                nonlocal aggregated_usage, latest_usage, final_messages, service_state_invalidated
                errors_in_a_row = 0
                total_function_calls = 0
                next_tool_ordinal = 0
                # Per-run identity registry of every content the conversation
                # holds — seeded from caller history so a client echoing a
                # historical object cannot re-execute a past side-effecting
                # call; _land_response_contents removes reappearances as
                # echoes. ignore_usage: echo filtering never strips usage.
                echo_registry = WeakIdentityRegistry(ignore_usage=True)
                for m in messages:
                    for c in m.contents:
                        echo_registry.register(c)
                wire_request_observer = _wire_request_observer(echo_registry)
                prepped_messages = list(messages)
                fcc_messages: list[Message] = []
                response: ChatResponse | None = None

                for _attempt_idx in range(self.max_iterations):
                    await _trajectory_cycle_started(_attempt_idx)
                    # Echoed conversation-held objects must never reach update
                    # assembly (merges would launder their identity past the
                    # memo). Delivered on the REQUEST path so the middleware
                    # pipeline attaches it to the stream its final handler
                    # resolves — beneath every middleware, which is the only
                    # placement that crosses semantic stream proxies (response
                    # validation drains the inner stream and replays it) and
                    # re-attaches on every validation retry. The provider
                    # finalizer, every result hook, and the yielded updates all
                    # observe one echo-free sequence, so nothing is re-assembled
                    # behind a hook's back.
                    echo_filter = functools.partial(
                        _strip_echoed_update,
                        echo_registry=echo_registry,
                    )
                    logical_stream = _streaming_response_with_retry(
                        prepped_messages,
                        stream_update_filter=echo_filter,
                        request_message_observer=wire_request_observer,
                    )
                    async for update in logical_stream:
                        yield update
                    # Triggers the inner stream's finalizer and result hooks (the
                    # instrumented intermediate-text hook runs before tool
                    # extraction below — "hook before tool detection" ordering).
                    response = await logical_stream.get_final_response()
                    _remove_echo_emptied_message_shells(response)
                    next_tool_ordinal = _land_response_contents(
                        response, next_tool_ordinal, echo_registry, exchange_operation_id=last_exchange_id
                    )
                    _record_stream_fragment_identities(logical_stream, echo_registry)
                    latest_usage = response.latest_usage_details
                    if latest_usage is None:
                        latest_usage = response.usage_details
                    response.latest_usage_details = latest_usage
                    if latest_usage is not None:
                        aggregated_usage = add_usage_details(aggregated_usage, latest_usage)
                    consumed_injection_messages = injection_probe.take_consumed_messages() if injection_probe else []
                    _update_continuation_state(
                        filtered_kwargs,
                        response,
                        session=invocation_session,
                        options=mutable_options,
                    )
                    if recorder is not None:
                        recorder.record_response(response)

                    if response.conversation_id is not None:
                        prepped_messages = []
                    else:
                        prepped_messages.extend(consumed_injection_messages)

                    # Continue only for unresolved local function calls.
                    function_calls = _extract_function_calls(response)
                    await _trajectory_cycle_finished(
                        outcome=ExchangeOutcome.SUCCESS,
                        function_call_count=len(function_calls),
                        response=response,
                        function_calls=function_calls,
                    )
                    if not function_calls:
                        final_messages = [*fcc_messages, *response.messages]
                        return

                    if not (function_calls and mutable_options.get("tools")):
                        final_messages = [*fcc_messages, *response.messages]
                        return

                    result_carrier_item_id = new_analytics_id()
                    result_commits = (
                        recorder.stage_exchange(
                            response.messages,
                            function_calls,
                            result_carrier_item_id=result_carrier_item_id,
                        )
                        if recorder is not None
                        else None
                    )
                    response_truncated = response.finish_reason == "length"
                    with _trajectory_exchange_scope():
                        results, should_terminate, had_errors = await execute_batch(
                            function_calls,
                            result_carrier_item_id=result_carrier_item_id,
                            result_commits=result_commits,
                            response_truncated=response_truncated,
                            truncated_final_function_call_ids=(
                                _truncated_final_function_call_ids(response, function_calls)
                                if response_truncated
                                else None
                            ),
                        )
                    action, errors_in_a_row = _handle_function_call_results(
                        response=response,
                        function_call_results=results,
                        fcc_messages=fcc_messages,
                        echo_registry=echo_registry,
                        errors_in_a_row=errors_in_a_row,
                        had_errors=had_errors,
                        max_errors=max_errors,
                        result_carrier_item_id=result_carrier_item_id,
                        recorder=recorder,
                    )
                    total_function_calls += sum(1 for r in results if r.type == "function_result")
                    if should_terminate and storage.service_side:
                        # No further request ever posts this batch's results, so
                        # the service transcript behind the mirrored handle keeps
                        # its calls unanswered. Record and withhold the handles
                        # like the exhaustion strip does — ahead of the
                        # last-iteration clear below, which would drop the
                        # session's copy unrecorded — and raise the closure flag
                        # so ``_finalize`` stamps the verdict on the assembled
                        # response for the agent post-hook.
                        service_state_invalidated = True
                        _invalidate_service_continuation_state(response, invocation_session)
                    elif (
                        _attempt_idx + 1 == self.max_iterations
                        and storage.service_side
                        and invocation_session is not None
                    ):
                        # Last iteration: the yield below is the final suspension
                        # point before the exhaustion tail, and these results are
                        # not posted to the service until the tail request lands.
                        # A consumer closing at that yield must not inherit the
                        # mirrored handle — the service transcript behind it still
                        # holds this batch's unanswered calls. The finalized tail
                        # restores the fresh handle via
                        # ``_update_continuation_state``.
                        invocation_session.service_session_id = None
                    # Synthesized tool-result update so from_updates can rebuild the
                    # full transcript.
                    yield ChatResponseUpdate(contents=results, role="tool")
                    if should_terminate:
                        # This batch (calls + results) was already folded into
                        # fcc_messages by _handle_function_call_results, so the
                        # assembly is the accumulated transcript with no tail.
                        # Deliberate asymmetry with the non-streaming termination
                        # return (only the terminating response, no prepend):
                        # each path keeps its existing transcript shape.
                        final_messages = list(fcc_messages)
                        return
                    if action == "stop":
                        mutable_options["tool_choice"] = "none"
                    elif self.max_function_calls is not None and total_function_calls >= self.max_function_calls:
                        logger.info(
                            "Maximum function calls reached (%d/%d). Stopping further function calls for this request.",
                            total_function_calls,
                            self.max_function_calls,
                        )
                        mutable_options["tool_choice"] = "none"

                    if mutable_options.get("tool_choice") == "required" or (
                        isinstance(mutable_options.get("tool_choice"), dict)
                        and mutable_options.get("tool_choice", {}).get("mode") == "required"
                    ):
                        mutable_options["tool_choice"] = None

                    if response.conversation_id is not None:
                        prepped_messages.clear()
                        if response.messages:
                            prepped_messages.append(response.messages[-1])
                    else:
                        prepped_messages.extend(response.messages)

                # Loop exhausted: final non-tool streaming turn.
                if response is not None:
                    logger.info(
                        "Maximum iterations reached (%d). Requesting final response without tools.",
                        self.max_iterations,
                    )
                mutable_options["tool_choice"] = "none"
                if storage.service_side and invocation_session is not None:
                    # The tail request consumes the mirrored handle: the moment it
                    # lands, the service transcript behind that handle holds this
                    # run's still-unanswered calls, so a consumer abandoning the
                    # tail must not inherit it (the tail updates below are marked
                    # against eager re-mirroring for the same reason). Successful
                    # finalization restores the fresh handle via
                    # ``_update_continuation_state`` — or withholds it when the
                    # strip invalidates the service state.
                    invocation_session.service_session_id = None
                echo_filter = functools.partial(
                    _strip_echoed_update,
                    echo_registry=echo_registry,
                )
                await _trajectory_cycle_started(self.max_iterations)
                final_logical_stream = _streaming_response_with_retry(
                    prepped_messages,
                    stream_update_filter=echo_filter,
                    request_message_observer=wire_request_observer,
                )
                tail_stripped_call = False
                async for update in final_logical_stream:
                    kept = _strip_unexecutable_calls_from_update(update, suppress_finish_reason=tail_stripped_call)
                    if kept is not update:
                        # A call was stripped: from here on no provider
                        # finish_reason may cross the stream — providers commonly
                        # emit the call delta and the finish-reason chunk
                        # separately, and a first-terminal consumer would stop at
                        # that later chunk before the corrective final update
                        # re-emits the corrected reason.
                        tail_stripped_call = True
                    if kept is not None:
                        # Ephemeral marker (never serialized): eager continuation
                        # mirrors must skip tail updates. The run is ending, so a
                        # mid-stream mirror has no disconnect-recovery value left,
                        # and a consumer abandoning the stream mid-tail would
                        # otherwise keep the session pointed at a service
                        # transcript whose stripped calls will never be answered;
                        # the tail's own verdict (post-hook restore or
                        # invalidation) is the sole writer once the stream ends.
                        kept.__dict__["_chrys_exhaustion_tail_update"] = True
                        yield kept
                final_response = await final_logical_stream.get_final_response()
                _remove_echo_emptied_message_shells(final_response)
                next_tool_ordinal = _land_response_contents(
                    final_response, next_tool_ordinal, echo_registry, exchange_operation_id=last_exchange_id
                )
                tail_calls = _extract_function_calls(final_response)
                await _trajectory_cycle_finished(
                    outcome=ExchangeOutcome.SUCCESS,
                    function_call_count=len(tail_calls),
                    response=final_response,
                    function_calls=tail_calls,
                )
                _record_stream_fragment_identities(final_logical_stream, echo_registry)
                stripped = _strip_unexecutable_calls_from_response(final_response)
                fallback_added = _ensure_exhaustion_fallback_response(final_response)
                # Invalidate BEFORE the fallback yield: a yield suspends the
                # generator, and a consumer that stops right after the fallback
                # would otherwise leave the session pointing at the stale service
                # transcript (the eager transform already wrote it mid-stream).
                # Gate on the resolved storage mode, not response metadata: under
                # conversation storage the service holds the stripped calls even
                # when the parsed response failed to carry the handle, and a
                # client-side-storage response's metadata must survive untouched.
                if stripped and storage.service_side:
                    service_state_invalidated = True
                    _invalidate_service_continuation_state(final_response, invocation_session)
                if fallback_added:
                    # Streamed consumers saw no visible content either (the yield
                    # loop strips the same calls); surface the fallback text there
                    # too, not only on the assembled response. The stripped tail
                    # updates carry no terminal reason (the update strip clears
                    # it), so this update supplies the stream's sole terminal
                    # "stop" — the fallback ends the run, there is no tool work
                    # left to route.
                    yield ChatResponseUpdate(
                        role="assistant",
                        contents=[Content.from_text(_MAX_ITERATIONS_FALLBACK_TEXT)],
                        finish_reason="stop",
                    )
                elif stripped and final_response.finish_reason:
                    # No fallback (the final kept visible text), but the strip
                    # suppressed every provider reason after the first stripped
                    # call and normalized the wire's "tool_calls" to "stop":
                    # streamed consumers need the corrected signal — whatever
                    # reason the assembled response settled on, "stop" or a
                    # non-stop reason like "length" — and it must arrive only
                    # after every stripped update.
                    yield ChatResponseUpdate(role="assistant", contents=[], finish_reason=final_response.finish_reason)
                latest_usage = final_response.latest_usage_details
                if latest_usage is None:
                    latest_usage = final_response.usage_details
                final_response.latest_usage_details = latest_usage
                if latest_usage is not None:
                    aggregated_usage = add_usage_details(aggregated_usage, latest_usage)
                if injection_probe is not None:
                    injection_probe.take_consumed_messages()
                _update_continuation_state(
                    filtered_kwargs,
                    final_response,
                    session=invocation_session,
                    options=mutable_options,
                )
                if recorder is not None:
                    recorder.record_response(final_response)
                final_messages = [*fcc_messages, *final_response.messages]
            except asyncio.CancelledError:
                # Nothing below may wait on the writer once the consumer is
                # gone: the settlement queues its lines instead.
                interrupted = True
                _trajectory_abort(_cancel_outcome())
                raise
            except GeneratorExit:
                interrupted = True
                _trajectory_abort(ExchangeOutcome.ABANDONED)
                raise
            except Exception:
                _trajectory_abort(ExchangeOutcome.ERROR)
                raise
            finally:
                await _settle_undispatched_tool_operations(queued=interrupted)

        def _finalize(updates: Sequence[ChatResponseUpdate]) -> ChatResponse:
            # Inner result hooks already ran via get_final_response; do not run them again.
            response = ChatResponse.from_updates(updates, output_format_type=response_format)
            if final_messages is not None:
                # The loop is the sole reconstruction authority: keep
                # ``from_updates`` for response-level fields, but reuse the
                # loop's assembled Message objects. Re-merging raw fragments
                # here would rebuild multi-fragment function calls as new
                # Content objects, dropping the kernel-stamped invocation
                # ordinal, tool kind/context, and folded result metadata —
                # and could glue a boundary-shaped delta onto the synthesized
                # tool message. Structured-output ``.value`` parses lazily
                # from these swapped messages; their text is identical.
                response.messages = list(final_messages)
            response.usage_details = aggregated_usage
            response.latest_usage_details = latest_usage
            if service_state_invalidated:
                # ``from_updates`` restored the handles from raw updates; the
                # tail withheld them because the service-side transcript still
                # holds the stripped calls unanswered.
                response.conversation_id = None
                response.response_id = None
                response._chrys_service_state_invalidated = True
            return response

        return ResponseStream(_stream(), finalizer=_finalize)


async def _resolve_response(value: Any) -> ChatResponse:
    """Resolve the inner layer's non-streaming return to a ``ChatResponse``.

    The inner :class:`~.middleware.ChatMiddlewareLayer` returns an un-awaited
    coroutine for ``stream=False``; a plain client returning a response object
    directly is accepted too.
    """
    if inspect.isawaitable(value):
        return await value
    return value
