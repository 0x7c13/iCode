# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ResponseValidationMiddleware — retries on malformed agent responses.

Some LLM / inference backends (vLLM, custom routers, quantized model
weights) occasionally produce structurally invalid final responses:
empty content arrays, whitespace-only text, or text channels that leaked
tool-call markers like ``minimax:tool_call ... </minimax:tool_call>``.
These terminate the agent tool loop with a useless final message.

This :class:`ChatMiddleware` sits at the **innermost** position of the
chat middleware chain (closest to ``_inner_get_response``) so its
retries are invisible to the outer instrumentation — ``UsageTracking``,
``SystemReminder``, and the loop-level ``LoopRecorder`` all fire exactly
once per turn regardless of how many retries happen here.

Strategy
--------
* Run the chat call via ``call_next()``.
* For streaming: return a lazy validating proxy.  It emits one raw,
  content-free heartbeat for every provider update while draining the inner
  :class:`ResponseStream`, then finalises and validates.  Transport errors
  propagate unchanged so the outer
  :class:`~chrys.foundation.retry.StreamRetryLoop` handles them.
* Validate the finalised :class:`ChatResponse` with a
  :class:`ResponseValidator`.
* On a valid response, the proxy replays the buffered model-content
  updates.  Its finalizer returns the already-finalised response verbatim
  so the inner provider's ``result_hooks`` (telemetry, usage) do NOT fire
  twice.
* On a terminal, non-retryable invalid response, raise
  :class:`TerminalResponseValidationError` so the TurnBindings observer records a
  normal turn error and persists an error marker for replay.  Retryable
  verdicts flagged ``terminal_on_giveup`` (reasoning-only responses with
  no visible answer) take the same raise once retries are exhausted —
  their scrubbed response would be blank, and returning it would read as
  a successful empty answer instead of the failure it is.
* In service-storage mode failures are never replayed in place: retryable
  verdicts raise :class:`RetryableResponseValidationError` to the whole-run
  retry boundary, with the same budget and identical-reason short-circuit
  applied across those outer attempts (give-up is always terminal there).
* On an invalid local response, warn, sleep with exponential-ish backoff,
  and ``call_next()`` again — up to :data:`MAX_RETRIES` additional raw
  requests inside the same logical wire call.
* On the **final** attempt we return the exhausted response — but with
  ``function_call`` contents stripped first.  Without stripping, an
  exhausted-but-still-malformed response carrying real ``function_call``
  items would be executed by the tool loop, the loop would iterate, and
  a model that's stuck in a bad state would re-trigger the full
  validation retry cycle on every iteration (5 retries after 5
  retries, indefinitely).  Stripping forces ``_extract_function_calls``
  to return ``[]`` so the tool loop exits cleanly with whatever (still-
  bad) text we got.
* On **both** the valid and give-up paths we additionally drop
  structurally-empty assistant outputs (``contents=[]`` or
  whitespace-only-text messages, plus their corresponding updates).
  ``DefaultResponseValidator`` only inspects the *final* assistant
  message, so a response shaped like ``[empty, valid]`` would otherwise
  pass validation with a stray ``content: []`` leading message that
  lands in persisted ``session.json`` next to the engine's turn marker.
  The update-level pass is group-aware (mirroring
  ``_process_update``'s role / ``message_id`` boundary rules) so a
  metadata-bearing terminal update sharing a non-empty group survives
  and the rebuild keeps ``finish_reason`` / ``response_id``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from chrys.foundation.errors import ContinuationVerdictError, ProviderResponseError
from chrys.foundation.hosted_tools import (
    HeldHostedEvidence,
    HostedRetrySafety,
    HostedToolPhase,
    HostedToolStatus,
    normalize_hosted_tool_status,
)
from chrys.foundation.retry import RetryAttemptInfo
from chrys.foundation.trajectory.event_types import ValidationReason
from chrys.kernel import (
    ChatResponse,
    ChatResponseUpdate,
    ContextOverflowSink,
    Message,
    ResponseStream,
    in_internal_side_call,
    normalize_stream_usage,
    resolve_storage_mode_and_handles,
)
from chrys.kernel.exchanges import (
    TOOL_CALL_CONTENT_TYPES,
    TOOL_RESULT_CONTENT_TYPES,
    EmptyIdPolicy,
    LiveAccessor,
    NoneIdPolicy,
    PairingKey,
    PairingPolicy,
    iter_exchanges,
    namespaced_pairing_key,
    pair_results,
)
from chrys.kernel.middleware import ChatMiddleware
from chrys.service.agent_middleware.events.hosted_tools import (
    HostedToolArgsOp,
    HostedToolProgressOp,
    HostedToolResultOp,
    HostedToolStartOp,
    HostedToolStatusOp,
    ResponsePresentationPlan,
    adapt_hosted_tool,
)
from chrys.service.agent_middleware.validators import DefaultResponseValidator, ValidationResult
from chrys.service.trajectory.validation import ValidationTrace

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence

    from chrys.kernel.middleware import ChatContext
    from chrys.service.agent_middleware.validators import ResponseValidator

logger = logging.getLogger(__name__)


# Maximum number of retries on top of the initial attempt.  Three retries
# means up to four total LLM calls per turn in the worst case.  On the
# final (fourth) attempt we return whatever we got, valid or not.
MAX_RETRIES = 3

# Backoff schedule (seconds) — indexed by attempt number.  A missing
# slot reuses the last value.  Deliberately short since the dominant
# cost is the LLM call itself, not the sleep.
DEFAULT_BACKOFF_SCHEDULE: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)


class ResponseValidationError(Exception):
    """Base for validation failures whose rejected attempt consumed billed tokens.

    ``usage_details`` carries the failed attempt's provider usage so the
    usage-tracking middleware can bill it before the exception continues to
    its owner (the pass observer error path or the whole-run retry boundary).
    """

    def __init__(self, reason: str, *, usage_details: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.usage_details = usage_details


class TerminalResponseValidationError(ResponseValidationError, RuntimeError):
    """Raised when response validation fails in a non-retryable way."""


@dataclass(frozen=True, slots=True)
class ValidationRetryExemption:
    """Independent stored-mode validation retry budget metadata."""

    attempt: int
    max_attempts: int
    delay_seconds: int


class RetryableResponseValidationError(ResponseValidationError, ConnectionError, ContinuationVerdictError):
    """A stored response failed validation and must retry at the outer boundary.

    ``hosted_commits`` names the provider-hosted tool calls (hosted MCP,
    hosted shell) the rejected response already executed server-side.  Those
    exchanges never reach the loop recorder — the invalid response is
    swallowed here before the kernel loop sees it — so this field is the
    only carrier by which the whole-run retry owner can honour them as
    commit points instead of re-creating the request and running the side
    effects a second time.
    """

    # Read by the kernel's retry wrappers through the classifier: this failure
    # judged a TERMINAL response, so a live continuation token now refers to a
    # completed (immutable) response — a retry must issue a fresh request,
    # never re-poll the same invalid completion.
    invalidates_continuation_token = True

    def __init__(
        self,
        reason: str,
        *,
        exemption: ValidationRetryExemption,
        usage_details: dict[str, Any] | None = None,
        hosted_commits: tuple[str, ...] = (),
    ) -> None:
        super().__init__(reason, usage_details=usage_details)
        self.exemption = exemption
        self.hosted_commits = hosted_commits


def hosted_commits_from_error(exc: BaseException) -> tuple[str, ...]:
    """Hosted tool calls the failed attempt behind *exc* already executed."""
    if isinstance(exc, RetryableResponseValidationError):
        return exc.hosted_commits
    return ()


class ResponseValidationObservationHook(Protocol):
    """Optional observer for raw content and validation-attempt boundaries."""

    async def begin_response(self, *, response_index: int | None = None, batch_id: int | None = None) -> None: ...

    async def observe_contents(self, contents: Sequence[Any], *, is_final: bool = False) -> None: ...

    async def attempt_started(self, *, continuation: bool = False) -> None: ...

    async def attempt_rejected(self, reason: str = "") -> None: ...

    async def attempt_accepted(self, messages: Sequence[Message]) -> None: ...


_LIVE_ACCESSOR = LiveAccessor()


def _hosted_commit_labels(response: ChatResponse) -> tuple[str, ...]:
    """Deduplicated display names of side-effectful hosted calls in *response*."""
    labels = _hosted_commit_labels_from_plan(ResponsePresentationPlan.from_messages(response.messages or []))
    if labels:
        return labels
    return _hosted_commit_labels_from_contents(
        content for message in response.messages or [] for content in message.contents
    )


def _hosted_commit_labels_from_contents(
    candidates: Iterable[Any],
    call_labels: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Deduplicated display names of side-effectful hosted calls in *candidates*."""
    labels: list[str] = []
    for label, _key in _hosted_commits_from_contents(candidates, call_labels):
        if label not in labels:
            labels.append(label)
    return tuple(labels)


def _hosted_commits_from_contents(
    candidates: Iterable[Any],
    call_labels: Mapping[str, str] | None = None,
) -> list[tuple[str, PairingKey | None]]:
    """Display name and pairing key of each side-effectful hosted call content in *candidates*.

    Result contents carry no tool name of their own, so each is labelled
    via its paired call when one is present in the same batch.
    """
    contents = [content for content in candidates if content.provider_hosted]
    names_by_call = dict(call_labels or {})
    names_by_call.update(
        {
            content.call_id: content.tool_name
            for content in contents
            if content.type in TOOL_CALL_CONTENT_TYPES and content.call_id and content.tool_name
        }
    )
    commits: list[tuple[str, PairingKey | None]] = []
    for content in contents:
        if content.type in TOOL_CALL_CONTENT_TYPES:
            view = adapt_hosted_tool(content)
        elif content.type in TOOL_RESULT_CONTENT_TYPES:
            view = adapt_hosted_tool(None, content)
        else:
            continue
        if not _view_is_hosted_commit(view):
            continue
        label = view.tool_name
        if content.type in TOOL_RESULT_CONTENT_TYPES:
            label = names_by_call.get(content.call_id) or content.tool_name or content.type
        commits.append((label, namespaced_pairing_key(content.type, _LIVE_ACCESSOR.raw_id(content))))
    return commits


# Recoverable hosted work: a call counts once a result with its id answers it; an id-less occurrence
# answers nothing, so its call is never counted as kept.
_KEPT_HOSTED_PAIRING_POLICY = PairingPolicy(
    call_types=TOOL_CALL_CONTENT_TYPES,
    include_informational_calls=True,
    result_types=TOOL_RESULT_CONTENT_TYPES,
    none_id=NoneIdPolicy.IGNORE,
    empty_id=EmptyIdPolicy.IGNORE,
    malformed_id="treat_as_none",
)


def _landed_exchange_answered_keys(messages: Sequence[Message], landed: Sequence[Message]) -> set[PairingKey]:
    """Pairing keys answered by the exchange of *messages* whose response holds *landed*'s Content objects.

    Identity, not ids, finds the exchange: an earlier exchange reusing the same call id is other work.
    """
    landed_contents = {id(content) for message in landed for content in message.contents}
    for exchange in reversed(list(iter_exchanges(messages, _LIVE_ACCESSOR))):
        if any(
            id(content) in landed_contents
            for index in exchange.response_indices
            for content in messages[index].contents
        ):
            return pair_results(messages, exchange, _LIVE_ACCESSOR, _KEPT_HOSTED_PAIRING_POLICY).answered_keys
    return set()


def _view_is_hosted_commit(view: Any) -> bool:
    """Return whether one adapted view proves retry-blocking execution."""
    if view.retry_safety not in {HostedRetrySafety.SIDE_EFFECTFUL, HostedRetrySafety.UNKNOWN}:
        return False
    status = normalize_hosted_tool_status(view.status)
    return view.phase in {
        HostedToolPhase.START,
        HostedToolPhase.DELTA,
        HostedToolPhase.SNAPSHOT,
        HostedToolPhase.TERMINAL,
    } or status in {
        HostedToolStatus.RUNNING,
        HostedToolStatus.COMPLETED,
        HostedToolStatus.FAILED,
        HostedToolStatus.INTERRUPTED,
    }


def _hosted_commit_labels_from_plan(plan: ResponsePresentationPlan) -> tuple[str, ...]:
    """Return retry-blocking labels from exchange-paired presentation views."""
    labels: list[str] = []
    for operation in plan.operations:
        if not isinstance(
            operation,
            HostedToolStartOp | HostedToolArgsOp | HostedToolProgressOp | HostedToolStatusOp | HostedToolResultOp,
        ):
            continue
        view = operation.view
        if not _view_is_hosted_commit(view):
            continue
        label = view.tool_name or view.family or "hosted_tool"
        if label not in labels:
            labels.append(label)
    return tuple(labels)


def _note_output_truncated(context: ChatContext, verdict: ValidationResult, response: ChatResponse) -> None:
    """Tell the compaction strategy when the input left the reply no room.

    A length stop with nothing generated stays terminal; the next request
    compacts first. A reasoning-only length stop is not this: the model did
    generate, so the window was not full. Neither is an empty reply whose
    usage counts reasoning tokens the API did not return.
    """
    if verdict.code != ValidationReason.OUTPUT_TRUNCATED:
        return
    if (response.usage_details or {}).get("reasoning_output_token_count"):
        return
    strategy = context.kwargs.get("compaction_strategy")
    if isinstance(strategy, ContextOverflowSink):
        strategy.note_context_overflow()


class ResponseValidationMiddleware(ChatMiddleware):
    """Retry the LLM call on structurally invalid final responses.

    Retries assume the failure is *transient* — random sampling produced
    a one-off bad output that a re-roll will fix.  When the producer is
    *deterministically* stuck (chat-template / KV-cache mismatch, fine-
    tuning quirk, prefix-cache hit at temperature 0), every retry sees
    the same input and returns the same bad output, so retrying just
    burns latency before the inevitable give-up.  To avoid that, this
    middleware fails fast: when attempt N+1's
    :attr:`ValidationResult.reason` exactly matches attempt N's, the
    loop short-circuits to the give-up path immediately instead of
    completing the remaining retries.

    In service-storage mode a failed attempt is never replayed in place —
    it raises to the whole-run retry boundary instead — so the retry
    budget and the identical-reason short-circuit are carried on the
    instance across those outer attempts; once either trips, the failure
    is raised terminally.

    Class attributes :attr:`_MAX_RETRIES` and :attr:`_BACKOFF_SCHEDULE`
    hold defaults that can be monkeypatched in tests (mirrors the pattern
    used by :class:`~chrys.orchestration.invoker.attempts.AttemptRunner`).  Constructor arguments
    override these defaults per-instance.

    Parameters
    ----------
    validator:
        Component that inspects each :class:`ChatResponse` and returns
        :class:`~chrys.service.agent_middleware.validators.ValidationResult`.
        Defaults to :class:`DefaultResponseValidator` with the three
        built-in rules (empty contents, whitespace text, leaked tool
        calls).
    max_retries:
        Maximum number of retries on top of the initial attempt.
        ``None`` → use :attr:`_MAX_RETRIES` (default :data:`MAX_RETRIES`).
    backoff_schedule:
        Sleep duration (seconds) between attempts, indexed by attempt
        number.  Past the last entry the last value is reused.  ``None``
        → use :attr:`_BACKOFF_SCHEDULE` (default
        :data:`DEFAULT_BACKOFF_SCHEDULE`).
    publish_retry:
        Optional observer fired before each validation replay sleep. Production
        orchestration leaves this unset: logical wire-retry events have one
        publisher, the kernel policy's ``on_retry`` callback.
    observation_hook:
        Optional raw-content and attempt-lifecycle observer. Callback failures
        are isolated from validation and hosted commit-point tracking.
    """

    # Class-level defaults so tests can monkeypatch via
    # ``monkeypatch.setattr(ResponseValidationMiddleware, "_BACKOFF_SCHEDULE", (0,))``
    # without having to plumb constructor kwargs through every test fixture.
    _MAX_RETRIES: int = MAX_RETRIES
    _BACKOFF_SCHEDULE: tuple[float, ...] = DEFAULT_BACKOFF_SCHEDULE

    def __init__(
        self,
        *,
        validator: ResponseValidator | None = None,
        max_retries: int | None = None,
        backoff_schedule: Sequence[float] | None = None,
        publish_retry: Callable[[RetryAttemptInfo], Awaitable[None]] | None = None,
        observation_hook: ResponseValidationObservationHook | None = None,
    ) -> None:
        self._validator: ResponseValidator = validator or DefaultResponseValidator()
        self._max_retries = max(0, int(max_retries if max_retries is not None else type(self)._MAX_RETRIES))
        schedule = backoff_schedule if backoff_schedule is not None else type(self)._BACKOFF_SCHEDULE
        self._backoff = tuple(schedule) or (0.0,)
        self._publish_retry = publish_retry
        self._observation_hook = observation_hook
        # Service-storage failures retry at the whole-run boundary, so each
        # outer attempt re-enters :meth:`process` with fresh loop state.  The
        # instance outlives those attempts; carrying the count and last reason
        # here keeps the retry budget and the deterministic-failure
        # short-circuit effective in that mode.
        self._service_retry_count = 0
        self._service_retry_reason: str | None = None
        # Provider-hosted tool calls observed executing (hosted MCP, hosted
        # shell) at two scopes matched to what each retry owner discards:
        # the run scope backs the whole-run gates (a whole-run retry restores
        # pre-run history, so ANY hosted work this run would re-execute); the
        # wire-attempt scope backs the kernel's per-wire replay veto (an
        # in-place replay only re-sends the current request, so only hosted
        # work from the aborted attempt itself is at risk — an attempt being
        # the request that started a response plus the polls that resume it).
        # The attempt's commits also keep their pairing keys, so a pass
        # resumed from repaired history can tell which ones that history
        # already answers.
        self._hosted_commits_seen: list[str] = []
        self._hosted_commits_in_flight: list[str] = []
        self._hosted_commit_keys_in_flight: list[tuple[str, PairingKey | None]] = []
        self._hosted_call_labels_in_flight: dict[str, str] = {}

    def set_observation_hook(self, hook: ResponseValidationObservationHook | None) -> None:
        """Attach the run-scoped hosted-presentation observer."""
        self._observation_hook = hook

    # ------------------------------------------------------------------
    # Hosted-commit observation (retry-owner probes)
    # ------------------------------------------------------------------

    def hosted_commits_observed(self) -> tuple[str, ...]:
        """Hosted tool calls executed since the retry baseline (run/pass start)."""
        return tuple(self._hosted_commits_seen)

    def hosted_commits_in_flight(self) -> tuple[str, ...]:
        """Hosted tool calls executed within the current wire attempt, polls that resumed it included."""
        return tuple(self._hosted_commits_in_flight)

    def hosted_commits_in_flight_unkept(
        self, kept: Sequence[Message], landed: Sequence[Message] | None
    ) -> tuple[str, ...]:
        """Hosted tool calls of the current wire attempt that *kept* does not answer.

        *kept* is the history a resumed pass sends; *landed* is the attempt's response as the tool loop
        landed it, ``None`` while the request is still in flight. Only the exchange holding that response
        counts: its answered hosted calls show the model their results instead of running again. A call
        without a pairing id never counts as kept.
        """
        answered = set() if landed is None else _landed_exchange_answered_keys(kept, landed)
        labels: list[str] = []
        for label, key in self._hosted_commit_keys_in_flight:
            if (key is None or key not in answered) and label not in labels:
                labels.append(label)
        return tuple(labels)

    def reset_hosted_commit_observations(self) -> None:
        """Forget observed hosted work — fired when the retry baseline resets."""
        self._hosted_commits_seen.clear()
        self._hosted_commits_in_flight.clear()
        self._hosted_commit_keys_in_flight.clear()
        self._hosted_call_labels_in_flight.clear()

    def begin_pass_hosted_baseline(self, *, resumes_background_response: bool) -> None:
        """Start a pass's hosted-work baseline.

        A pass that polls a background response an earlier pass left running keeps the observations: that
        response's hosted calls may already have run, and if the poll ends without the response, a fresh
        request would run them again.
        """
        if not resumes_background_response:
            self.reset_hosted_commit_observations()

    def _begin_wire_attempt(self, *, continuation: bool) -> None:
        # A poll resumes the response an earlier attempt started: the hosted
        # work that response ran stays at risk until it ends, since a fresh
        # request would run it again.
        if continuation:
            return
        self._hosted_commits_in_flight.clear()
        self._hosted_commit_keys_in_flight.clear()
        self._hosted_call_labels_in_flight.clear()

    def _observe_hosted_contents(self, contents: Iterable[Any]) -> None:
        observed = list(contents)
        self._hosted_call_labels_in_flight.update(
            {
                content.call_id: content.tool_name
                for content in observed
                if content.provider_hosted
                and content.type in TOOL_CALL_CONTENT_TYPES
                and content.call_id
                and content.tool_name
            }
        )
        for commit in _hosted_commits_from_contents(observed, self._hosted_call_labels_in_flight):
            label = commit[0]
            if label not in self._hosted_commits_seen:
                self._hosted_commits_seen.append(label)
            if label not in self._hosted_commits_in_flight:
                self._hosted_commits_in_flight.append(label)
            if commit not in self._hosted_commit_keys_in_flight:
                self._hosted_commit_keys_in_flight.append(commit)

    async def _observe_failed_response(self, error: ProviderResponseError) -> None:
        """Count the hosted work a failed response showed but never yielded.

        The adapter raised before the response landed, so this is the only
        place the retry gates learn that the work already ran.
        """
        if contents := list(error.observed_contents):
            self._observe_hosted_contents(contents)
            await self._notify_contents_observed(contents, is_final=True)

    async def _notify_contents_observed(self, contents: Sequence[Any], *, is_final: bool) -> None:
        if self._observation_hook is None or in_internal_side_call():
            return
        try:
            await self._observation_hook.observe_contents(contents, is_final=is_final)
        except Exception:
            logger.exception("response validation observation hook raised while observing contents")

    async def _notify_response_started(self, *, continuation: bool) -> None:
        if self._observation_hook is None or in_internal_side_call() or continuation:
            return
        try:
            await self._observation_hook.begin_response()
        except Exception:
            logger.exception("response validation observation hook raised at response start")

    async def _notify_attempt_started(self, *, continuation: bool = False) -> None:
        if self._observation_hook is None or in_internal_side_call():
            return
        try:
            await self._observation_hook.attempt_started(continuation=continuation)
        except Exception:
            logger.exception("response validation observation hook raised at attempt start")

    async def _notify_attempt_rejected(self, reason: str) -> None:
        if self._observation_hook is None or in_internal_side_call():
            return
        try:
            await self._observation_hook.attempt_rejected(reason)
        except Exception:
            logger.exception("response validation observation hook raised at attempt rejection")

    async def _notify_attempt_accepted(self, messages: Sequence[Message]) -> None:
        if self._observation_hook is None or in_internal_side_call():
            return
        try:
            await self._observation_hook.attempt_accepted(messages)
        except Exception:
            logger.exception("response validation observation hook raised at attempt acceptance")

    # ------------------------------------------------------------------
    # ChatMiddleware interface
    # ------------------------------------------------------------------

    async def process(
        self,
        context: ChatContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        try:
            stores_by_default = bool(context.client.STORES_BY_DEFAULT)
        except AttributeError:
            stores_by_default = False
        try:
            force_stateless = bool(context.client.FORCES_STATELESS)
        except AttributeError:
            force_stateless = False
        nested_client_kwargs = context.kwargs.get("client_kwargs")
        storage_kwargs = nested_client_kwargs if isinstance(nested_client_kwargs, Mapping) else context.kwargs
        service_side = resolve_storage_mode_and_handles(
            context.options,
            stores_by_default=stores_by_default,
            client_kwargs=storage_kwargs,
            force_stateless=force_stateless,
        ).service_side
        continuation = isinstance(context.options, Mapping) and context.options.get("continuation_token") is not None
        await self._notify_response_started(continuation=continuation)
        validation = ValidationTrace.open(context.kwargs)
        if context.stream:
            # Return a lazy validating proxy immediately so raw, content-free provider
            # heartbeats can reach the outer stream-idle watchdog during the inner drain.
            # All model content remains buffered until its finalized response validates.
            context.result = self._validated_response_stream(
                context, call_next, service_side=service_side, validation=validation
            )
            return

        # Total attempts = 1 initial + max_retries retries.  The loop
        # variable 0..max_retries tracks the retry count; attempt=0 is
        # the initial call.
        #
        # ``previous_reason`` carries the prior attempt's
        # :attr:`ValidationResult.reason` so :meth:`_should_give_up` can
        # detect a deterministic-stuck producer (same failure mode twice
        # in a row) and short-circuit to give-up instead of burning the
        # remaining retries on what will almost certainly be the same
        # bad output.
        previous_reason: str | None = None
        for attempt in range(self._max_retries + 1):
            continuation = (
                isinstance(context.options, Mapping) and context.options.get("continuation_token") is not None
            )
            self._begin_wire_attempt(continuation=continuation)
            await self._notify_attempt_started(continuation=continuation)
            try:
                await call_next()
            except ProviderResponseError as error:
                await self._observe_failed_response(error)
                raise
            result = context.result
            if not isinstance(result, ChatResponse):
                return
            observed_contents = [content for message in result.messages or [] for content in message.contents]
            self._observe_hosted_contents(observed_contents)
            await self._notify_contents_observed(observed_contents, is_final=True)
            if result.continuation_token is not None:
                await self._notify_attempt_accepted(result.messages or [])
                return

            verdict = self._validator.validate(result)
            _note_output_truncated(context, verdict, result)
            hosted_commits = () if verdict.ok else _hosted_commit_labels(result)
            give_up = not verdict.ok and self._gives_up(
                verdict,
                attempt=attempt,
                previous_reason=previous_reason,
                hosted_committed=bool(hosted_commits),
                service_side=service_side,
            )
            if validation is not None:
                await validation.finished(
                    accepted=verdict.ok,
                    reason_code=None if verdict.ok else verdict.code,
                    retryable=None if verdict.ok else verdict.retryable,
                    gave_up=give_up,
                )
            if verdict.ok:
                self.reset_service_retry_state()
                # Even on a valid response, drop structurally-empty
                # assistant messages.  The default validator only
                # inspects the FINAL assistant message, so a response
                # shaped like ``[empty, valid]`` passes validation
                # while still carrying a stray ``content: []`` message
                # that would otherwise land in persisted history.
                # Streaming providers/layers can also create empty
                # assistant placeholders at message_id boundaries that
                # never get populated — same fix.
                _drop_empty_assistant_outputs(result, [])
                await self._notify_attempt_accepted(result.messages or [])
                return
            if service_side:
                await self._notify_attempt_rejected(verdict.reason)
                self._service_side_failure(verdict.reason, result, code=verdict.code, give_up=give_up)
            if hosted_commits:
                # The rejected response already executed provider-hosted tool
                # calls server-side; replaying the request in place would run
                # those side effects a second time. Give up now — the scrubbed
                # response keeps the hosted transcript in history.
                logger.warning(
                    "Response validation would retry, but provider-hosted tool call(s) "
                    "already executed (%s) — not re-sending the request: %s",
                    ", ".join(hosted_commits),
                    verdict.reason,
                )
            if give_up:
                # A concluded cycle — terminal raise or scrubbed acceptance —
                # must not leave stale service-side carry-over behind a later
                # storage-mode flip.
                self.reset_service_retry_state()
                if not verdict.retryable or verdict.terminal_on_giveup:
                    await self._notify_attempt_rejected(verdict.reason)
                    self._raise_terminal(verdict.reason, result, code=verdict.code)
                # Give-up path: scrub the exhausted response so neither
                # the tool loop NOR persisted history is
                # polluted by the malformed output:
                #
                # * function_call contents are stripped so
                #   ``_extract_function_calls`` returns ``[]`` and the
                #   loop exits cleanly (without this the model could
                #   re-trigger the full validation cycle every iteration).
                # * empty / whitespace-only assistant messages are
                #   dropped (function_call stripping may itself empty a
                #   message — drop those too) so they don't land in
                #   ``session.json`` next to the engine's turn marker
                #   as a stray ``content: []`` message.
                _strip_function_calls(result, [])
                _drop_empty_assistant_outputs(result, [])
                await self._notify_attempt_accepted(result.messages or [])
                return

            # Invalid + retries remaining: emit retry event + sleep.
            previous_reason = verdict.reason
            await self._notify_attempt_rejected(verdict.reason)
            await self._note_retry(verdict.reason, attempt, validation=validation)

    def _validated_response_stream(
        self,
        context: ChatContext,
        call_next: Callable[[], Awaitable[None]],
        *,
        service_side: bool,
        validation: ValidationTrace | None = None,
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
        """Validate lazily while emitting safe transport progress for every update."""
        final_response: ChatResponse | None = None
        # The inner stream of the attempt being read; None once that attempt
        # was accepted or closed for a retry.
        active_result: ResponseStream[ChatResponseUpdate, ChatResponse] | None = None
        proxy: ResponseStream[ChatResponseUpdate, ChatResponse]

        def _select_response(
            response: ChatResponse,
            result_hooks: Sequence[Callable[[ChatResponse], Any]],
        ) -> None:
            nonlocal final_response
            final_response = response
            context.result = proxy
            # The provider/inner hooks originally finalize before hooks installed
            # by outer middleware. Outer hooks are already present on the lazy
            # proxy by the time this attempt succeeds, so prepend the inner phase.
            proxy._result_hooks[:0] = result_hooks

        async def _updates() -> AsyncIterator[ChatResponseUpdate]:
            nonlocal active_result
            previous_reason: str | None = None
            for attempt in range(self._max_retries + 1):
                continuation = (
                    isinstance(context.options, Mapping) and context.options.get("continuation_token") is not None
                )
                self._begin_wire_attempt(continuation=continuation)
                await self._notify_attempt_started(continuation=continuation)
                updates: list[ChatResponseUpdate] = []
                try:
                    await call_next()
                    result = context.result
                    if not isinstance(result, ResponseStream):
                        raise ValueError("Streaming response validation requires a ResponseStream result.")
                    active_result = result

                    await result
                    inner_hooks = _collect_and_clear_hooks(result)
                    async for update in result:
                        if isinstance(evidence := update.raw_representation, HeldHostedEvidence):
                            # Hosted work the adapter holds behind an
                            # unfinished call already ran: the retry gates
                            # count it now. The update that releases it in
                            # order presents it and keeps it for the replay.
                            self._observe_hosted_contents(evidence.contents)
                            yield _stream_heartbeat_for(update)
                            continue
                        # Record hosted executions the moment their updates
                        # land — a stall or transport drop after this point
                        # must reach the retry owners as commit evidence even
                        # though the buffered content never validated.
                        observed_contents = list(update.contents or [])
                        self._observe_hosted_contents(observed_contents)
                        await self._notify_contents_observed(observed_contents, is_final=False)
                        updates.append(update)
                        yield _stream_heartbeat_for(update)
                except ProviderResponseError as error:
                    await self._observe_failed_response(error)
                    raise
                final = await _preview_finalize(result, updates)
                final_contents = [content for message in final.messages or [] for content in message.contents]
                self._observe_hosted_contents(final_contents)
                await self._notify_contents_observed(final_contents, is_final=True)

                if final.continuation_token is not None:
                    _select_response(final, inner_hooks)
                    await self._notify_attempt_accepted(final.messages or [])
                    active_result = None
                    return

                verdict = self._validator.validate(final)
                _note_output_truncated(context, verdict, final)
                hosted_commits = () if verdict.ok else _hosted_commit_labels(final)
                give_up = not verdict.ok and self._gives_up(
                    verdict,
                    attempt=attempt,
                    previous_reason=previous_reason,
                    hosted_committed=bool(hosted_commits),
                    service_side=service_side,
                )
                if validation is not None:
                    await validation.finished(
                        accepted=verdict.ok,
                        reason_code=None if verdict.ok else verdict.code,
                        retryable=None if verdict.ok else verdict.retryable,
                        gave_up=give_up,
                    )
                if verdict.ok:
                    self.reset_service_retry_state()
                    _drop_empty_assistant_outputs(final, updates)
                    _select_response(final, inner_hooks)
                    await self._notify_attempt_accepted(final.messages or [])
                    for update in updates:
                        if not _is_stream_heartbeat(update):
                            yield update
                    active_result = None
                    return

                if service_side:
                    await self._notify_attempt_rejected(verdict.reason)
                    self._service_side_failure(
                        verdict.reason,
                        final,
                        code=verdict.code,
                        give_up=give_up,
                        stream_usage_chunks=_usage_chunks_from_updates(updates),
                    )
                if hosted_commits:
                    # Same rule as the blocking path: hosted side effects
                    # in the rejected response forbid an in-place re-send.
                    logger.warning(
                        "Response validation would retry, but provider-hosted tool call(s) "
                        "already executed (%s) — not re-sending the request: %s",
                        ", ".join(hosted_commits),
                        verdict.reason,
                    )
                if give_up:
                    # A concluded cycle — terminal raise or scrubbed
                    # acceptance — must not leave stale service-side
                    # carry-over behind a later storage-mode flip.
                    self.reset_service_retry_state()
                    if not verdict.retryable or verdict.terminal_on_giveup:
                        await self._notify_attempt_rejected(verdict.reason)
                        self._raise_terminal(
                            verdict.reason,
                            final,
                            code=verdict.code,
                            stream_usage_chunks=_usage_chunks_from_updates(updates),
                        )
                    _strip_function_calls(final, updates)
                    _drop_empty_assistant_outputs(final, updates)
                    _select_response(final, inner_hooks)
                    await self._notify_attempt_accepted(final.messages or [])
                    for update in updates:
                        if not _is_stream_heartbeat(update):
                            yield update
                    active_result = None
                    return

                previous_reason = verdict.reason
                await self._notify_attempt_rejected(verdict.reason)
                await result.aclose()
                active_result = None
                await self._note_retry(verdict.reason, attempt, validation=validation)

        async def _close_active_result() -> None:
            # A proxy left mid-attempt closes the inner stream that attempt
            # was reading: after a failed or cancelled pull, and on an
            # explicit close. It is the proxy's cleanup hook, never the
            # generator's exit: finalization closes each abandoned generator
            # in a task of its own, and an exit closing another generator
            # would collide with that generator's own close.
            nonlocal active_result
            reading, active_result = active_result, None
            if reading is not None:
                await reading.aclose()

        def _finalizer(_updates: Sequence[ChatResponseUpdate]) -> ChatResponse:
            if final_response is None:
                raise RuntimeError("Streaming response validation ended without a final response.")
            return final_response

        proxy = ResponseStream(_updates(), finalizer=_finalizer).with_cleanup_hook(_close_active_result)
        return proxy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_final_attempt(self, attempt: int) -> bool:
        return attempt >= self._max_retries

    def _gives_up(
        self,
        verdict: ValidationResult,
        *,
        attempt: int,
        previous_reason: str | None,
        hosted_committed: bool,
        service_side: bool,
    ) -> bool:
        """Whether this rejected verdict ends the cycle instead of retrying again.

        Service-storage mode cannot replay in place — the whole run retries at
        the pass boundary — so its budget is the instance-carried count and
        reason rather than this loop's. It is the same decision either way, and
        taking it here is what lets the recorded verdict state it.
        """
        if service_side:
            return self._should_give_up(
                self._service_retry_count, verdict.reason, self._service_retry_reason, retryable=verdict.retryable
            )
        return hosted_committed or self._should_give_up(
            attempt, verdict.reason, previous_reason, retryable=verdict.retryable
        )

    def _should_give_up(
        self, attempt: int, reason: str, previous_reason: str | None, *, retryable: bool = True
    ) -> bool:
        """Stop retrying — non-retryable, out of attempts, or the producer is stuck.

        Three short-circuit signals:

        * **Non-retryable** — the validator marked the failure terminal
          (``ValidationResult.retryable is False``, e.g. the response hit
          the output token limit with no content).  Re-issuing the same
          request cannot help, so give up on the spot.
        * **Final attempt** — :meth:`_is_final_attempt` (out of retries).
        * **Repeat failure** — the new ``reason`` exactly matches the
          prior attempt's.  Same input + same reason almost always means
          the producer is deterministic-stuck, so further retries are
          pure latency tax.  Logs a distinct warning so operators can
          tell a fail-fast give-up apart from an exhausted-retry one.
        """
        if not retryable:
            return True
        if self._is_final_attempt(attempt):
            return True
        if previous_reason is not None and reason == previous_reason:
            logger.warning(
                "Response validation failed identically on attempt %d/%d; "
                "giving up early (deterministic failure suspected): %s",
                attempt + 1,
                self._max_retries + 1,
                reason,
            )
            return True
        return False

    def _delay_for(self, attempt: int) -> float:
        idx = min(attempt, len(self._backoff) - 1)
        return max(0.0, float(self._backoff[idx]))

    async def _note_retry(self, reason: str, attempt: int, *, validation: ValidationTrace | None = None) -> None:
        """Emit the retry event + log, then sleep the backoff delay."""
        delay = self._delay_for(attempt)
        if validation is not None:
            await validation.retry_scheduled(delay_seconds=delay)
        next_attempt = attempt + 1  # 1-based for the event / log
        logger.warning(
            "Response validation failed (attempt %d/%d), retrying in %.1fs: %s",
            next_attempt,
            self._max_retries,
            delay,
            reason,
        )
        if self._publish_retry is not None:
            try:
                await self._publish_retry(
                    RetryAttemptInfo(
                        reason=reason,
                        attempt=next_attempt,
                        max_attempts=self._max_retries,
                        delay_seconds=delay,
                    )
                )
            except Exception:
                # Telemetry must never break the retry loop.
                logger.exception("publish_retry callback raised")
        if delay > 0:
            await asyncio.sleep(delay)
        if validation is not None:
            await validation.retry_started()

    def reset_service_retry_state(self) -> None:
        """Discard carried service-side retry state.

        Called internally whenever a validation cycle concludes (success,
        give-up, terminal), and by the orchestration pass observer at the start of
        every user-initiated run cycle — an aborted cycle (interrupt during
        outer backoff, unrelated exception) exits through none of the internal
        reset points, and its leftover count/reason must not judge the next
        independent run.  Whole-run retry attempts WITHIN one cycle share the
        state; that is the point of carrying it.
        """
        self._service_retry_count = 0
        self._service_retry_reason = None

    def _service_side_failure(
        self,
        reason: str,
        response: ChatResponse,
        *,
        code: str,
        give_up: bool,
        stream_usage_chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Route a service-side validation failure through the retry policy.

        ``give_up`` is the decision :meth:`_gives_up` already took against the
        instance-carried count/reason.  Give-up is always terminal here: the
        service holds the unscrubbed conversation state, so returning a locally
        scrubbed response could desynchronize a stored continuation.  The
        terminal raise resets the carried state so a later user-initiated run
        starts with a fresh budget.
        """
        if give_up:
            self.reset_service_retry_state()
            self._raise_terminal(reason, response, code=code, stream_usage_chunks=stream_usage_chunks)
        self._service_retry_count += 1
        self._service_retry_reason = reason
        self._raise_service_retry(reason, response, stream_usage_chunks=stream_usage_chunks)

    def _raise_terminal(
        self,
        reason: str,
        response: ChatResponse,
        *,
        code: str,
        stream_usage_chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Raise a terminal validation failure into the pass observer error path.

        A filtered response is raised as the provider's own content-filter
        failure: it is classified as filtered, and a live continuation token
        names a finished response that a retry must not poll.
        """
        logger.warning("Response validation failed terminally (giving up): %s", reason)
        usage_details = None
        if stream_usage_chunks is not None:
            normalized = normalize_stream_usage(stream_usage_chunks)
            usage_details = dict(normalized) if normalized else None
        if usage_details is None and response.usage_details:
            usage_details = dict(response.usage_details)
        error = TerminalResponseValidationError(reason, usage_details=usage_details)
        if code == ValidationReason.CONTENT_FILTERED:
            raise error from ProviderResponseError(
                "content_filter", reason, retryable=False, invalidates_continuation_token=True
            )
        raise error

    def _raise_service_retry(
        self,
        reason: str,
        response: ChatResponse,
        *,
        stream_usage_chunks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Raise a stored validation failure to the whole-run retry owner."""
        usage_details = None
        if stream_usage_chunks is not None:
            normalized = normalize_stream_usage(stream_usage_chunks)
            usage_details = dict(normalized) if normalized else None
        if usage_details is None and response.usage_details:
            usage_details = dict(response.usage_details)
        raise RetryableResponseValidationError(
            reason,
            exemption=ValidationRetryExemption(
                attempt=self._service_retry_count,
                max_attempts=self._max_retries,
                delay_seconds=math.ceil(self._delay_for(self._service_retry_count - 1)),
            ),
            usage_details=usage_details,
            hosted_commits=_hosted_commit_labels(response),
        )


# ---------------------------------------------------------------------------
# Inner-stream finaliser + replay helpers
# ---------------------------------------------------------------------------


def _collect_and_clear_hooks(
    stream: ResponseStream[ChatResponseUpdate, ChatResponse],
) -> list[Callable[[ChatResponse], Any]]:
    """Walk a (possibly wrapped) ResponseStream chain; collect + clear hooks.

    ``BaseChatClient.get_response`` may wrap a provider's stream in
    ``ResponseStream.from_awaitable`` (e.g. when a compaction override is
    active), in which case the hook installed by ``_inner_get_response``
    sits on the *inner* stream, not the outer wrapper.  Walk the
    ``_inner_stream`` chain, collecting hooks from each layer and
    clearing them so neither the layer's own ``get_final_response`` nor
    the auto-finalise path inside ``__anext__`` fires them during our
    preview drain.

    The caller should ``await stream`` first so ``_inner_stream`` is
    populated on any ``from_awaitable`` wrappers.
    """
    hooks_by_layer: list[list[Callable[[ChatResponse], Any]]] = []
    seen: set[int] = set()
    cur: ResponseStream[Any, Any] | None = stream
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        hooks = cur._result_hooks
        if hooks:
            hooks_by_layer.append(list(hooks))
            cur._result_hooks = []
        cur = getattr(cur, "_inner_stream", None)
    # ResponseStream finalizes wrapped streams from the innermost layer outward.
    # Recreate that phase order when all hooks move onto the validating proxy.
    return [hook for hooks in reversed(hooks_by_layer) for hook in hooks]


async def _preview_finalize(
    inner: ResponseStream[ChatResponseUpdate, ChatResponse],
    updates: Sequence[ChatResponseUpdate],
) -> ChatResponse:
    """Produce a :class:`ChatResponse` from *updates* for validation only.

    Uses the inner stream's ``_finalizer`` if set (so provider-specific
    finalisation logic, e.g. ``output_format_type`` parsing, still runs)
    but deliberately does **not** invoke the inner's ``_result_hooks``.
    Those hooks are reserved for the caller-visible finalisation that
    runs once we've decided to return the response.
    """
    finalizer = getattr(inner, "_finalizer", None)
    if finalizer is not None:
        result = finalizer(updates)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ChatResponse):
            return result
    # Fallback: build the response directly.  Keeps the middleware
    # robust against custom ResponseStream subclasses that stash the
    # finalizer elsewhere.
    return ChatResponse.from_updates(list(updates))


def _is_stream_heartbeat(update: ChatResponseUpdate) -> bool:
    """Return whether *update* is the explicit opaque transport heartbeat."""
    return update.is_transport_heartbeat


def _stream_heartbeat_for(update: ChatResponseUpdate) -> ChatResponseUpdate:
    """Return one semantics-free progress update for every provider update.

    The continuation token rides along: it is transport bookkeeping, not
    model content, and the retry owner must learn a background response id
    as soon as the provider announces it — content buffering must not delay
    that past a mid-stream disconnect.
    """
    heartbeat = ChatResponseUpdate.transport_heartbeat()
    heartbeat.continuation_token = update.continuation_token
    return heartbeat


def _strip_function_calls(
    response: ChatResponse,
    updates: list[ChatResponseUpdate],
) -> None:
    """Remove actionable ``function_call`` contents from *response* and *updates*.

    Used on the validation **give-up** path only — never on a valid
    response, since stripping would suppress real tool calls the model
    legitimately wants to execute.  Without this, an exhausted-but-
    still-malformed response carrying a real ``function_call`` would be
    handed to the tool loop, the loop would iterate, and a model stuck
    in a bad state would re-trigger the full validation cycle every
    iteration. Stripping forces ``_extract_function_calls`` to return
    ``[]`` so the loop exits cleanly.

    Mutates ``Message.contents`` and ``ChatResponseUpdate.contents`` in
    place. Informational hosted calls and other content types (text,
    function_result, image, …) are preserved.
    """
    for msg in response.messages:
        if any(c.type == "function_call" and not c.informational_only for c in msg.contents):
            msg.contents = [c for c in msg.contents if c.type != "function_call" or c.informational_only]
    for upd in updates:
        if upd.contents and any(c.type == "function_call" and not c.informational_only for c in upd.contents):
            upd.contents = [c for c in upd.contents if c.type != "function_call" or c.informational_only]


def _usage_chunks_from_updates(updates: Sequence[ChatResponseUpdate]) -> list[dict[str, Any]]:
    """Return provider usage payloads from streamed updates."""
    return [
        dict(content.usage_details)
        for update in updates
        for content in update.contents
        if content.type == "usage" and content.usage_details
    ]


def _drop_empty_assistant_outputs(
    response: ChatResponse,
    updates: list[ChatResponseUpdate],
) -> None:
    """Drop structurally-empty assistant messages and their updates.

    Run on **every** response (valid + give-up).  Two reasons:

    1. ``DefaultResponseValidator`` only inspects the *final* assistant
       message, so a response shaped like ``[empty, valid]`` passes
       validation while still carrying a stray ``content: []`` leading
       message that would land in persisted history.
    2. The kernel streaming finalizer rebuilds the final
       persisted response with ``ChatResponse.from_updates(updates)``
       from stream updates. If we only scrub
       ``response.messages`` the streaming path would re-create the
       empty message from the leftover updates and we'd be back where
       we started.  Mirroring the drop on *updates* prevents that.

    "Empty" means: ``contents`` is ``[]`` OR every content item is
    text and the concatenated text is empty / whitespace-only.  Any
    non-text content (image, function_result, audio, …) is treated as
    a useful payload and the message/update is kept.

    The update-level pass is **group-aware**: updates are grouped by
    ``(role, message_id)`` using the same boundary rules as the kernel
    stream grouping logic, and an entire group is
    dropped only when every update in it contributes empty content.
    A real provider's terminal "stop" update (empty contents +
    ``finish_reason``) typically shares the previous text update's
    group, so it survives the drop and its metadata reaches the
    rebuild — only fully-empty placeholder groups (e.g. an unfilled
    ``message_id`` boundary) are removed.

    Mutates ``response.messages`` and *updates* in place.
    """
    response.messages = [m for m in response.messages if not _is_empty_assistant_message(m)]

    if not updates:
        return
    keep_mask = _compute_update_keep_mask(updates)
    cleaned: list[ChatResponseUpdate] = [u for u, keep in zip(updates, keep_mask, strict=True) if keep]
    updates.clear()
    updates.extend(cleaned)


def _compute_update_keep_mask(updates: list[ChatResponseUpdate]) -> list[bool]:
    """Group *updates* by their effective ``(role, message_id)`` and decide
    which updates to keep.

    Mirrors ``_process_update``'s message-boundary rules: a new group
    starts when the update has a ``message_id`` that differs from the
    current group's ``message_id``, OR when the update has a ``role``
    that differs from the current group's role.  Otherwise the update
    appends to the current group.

    A group is "empty assistant" if its effective role resolves to
    assistant (or unset, since ``_process_update`` defaults to
    assistant) AND every update in the group has empty / whitespace-
    only-text contents.  Such groups are dropped wholesale — including
    any metadata-bearing terminal updates inside them, since with no
    text/non-text content there is nothing for the metadata to attach
    to in the rebuild.

    Returns a boolean mask of the same length as *updates*.
    """
    if not updates:
        return []

    # Pass 1: assign each update to a group index, mirroring
    # ``_process_update`` exactly. Two key invariants from the kernel
    # stream rebuild logic:
    #
    # * Every new message begins life as ``Message("assistant", [])`` —
    #   default role assistant, no message_id.  The current update's
    #   ``role`` / ``message_id`` then OVERRIDE that placeholder state.
    #   So whenever we open a new group we MUST reset ``cur_role`` to
    #   ``"assistant"`` and ``cur_message_id`` to ``None`` *before*
    #   applying the new update — otherwise a role-less new-message_id
    #   update inherits the previous group's role, and a same-id
    #   follow-up update can split incorrectly.
    # * Boundary checks reference the previous message's *final* state
    #   (after all its updates have been applied).  ``cur_role`` /
    #   ``cur_message_id`` track that state for the current group.
    group_idx: list[int] = []
    group_role: list[str] = []
    cur_role: str = "assistant"
    cur_message_id: str | None = None
    cur_idx = -1
    for upd in updates:
        new_group = (
            cur_idx < 0
            or (upd.message_id and cur_message_id and upd.message_id != cur_message_id)
            or (upd.role and upd.role != cur_role)
        )
        if new_group:
            cur_idx += 1
            # New placeholder: kernel default role + no inherited id.
            cur_role = "assistant"
            cur_message_id = None
            group_role.append(cur_role)
        group_idx.append(cur_idx)
        if upd.role:
            cur_role = upd.role
            # Update the group's recorded role so subsequent comparisons
            # use the latest known role for this group.
            group_role[cur_idx] = cur_role
        if upd.message_id:
            cur_message_id = upd.message_id

    # Pass 2: per-group, check if all updates contribute empty content.
    group_count = cur_idx + 1
    group_has_content: list[bool] = [False] * group_count
    for upd, gi in zip(updates, group_idx, strict=True):
        if _update_has_useful_content(upd):
            group_has_content[gi] = True

    # Pass 3: build keep mask.  Only assistant groups are scrubbed —
    # tool / user / system groups are kept regardless of content so we
    # never drop kernel- or caller-injected non-assistant turns.
    keep_mask: list[bool] = []
    for gi in group_idx:
        is_assistant_group = group_role[gi] == "assistant"
        keep_mask.append((not is_assistant_group) or group_has_content[gi])
    return keep_mask


def _update_has_useful_content(upd: ChatResponseUpdate) -> bool:
    """Return ``True`` if *upd*'s contents include anything non-empty.

    Non-text content (image, function_call, function_result, usage, …)
    counts as useful.  Text content counts only if the concatenated text
    has at least one non-whitespace character.  Empty contents → not
    useful.
    """
    contents = upd.contents or []
    if not contents:
        return False
    text_acc: list[str] = []
    for c in contents:
        ctype = c.type
        if ctype != "text":
            return True
        text_acc.append(c.text or "")
    return bool("".join(text_acc).strip())


def _is_empty_assistant_message(msg: Message) -> bool:
    """Return ``True`` if *msg* is an assistant message with no useful payload.

    "No useful payload" means:

    * ``contents`` is empty (``[]``), OR
    * every content item is text and the concatenated text is empty or
      whitespace-only.

    A message carrying any non-text content (image, function_result,
    audio, …) is NOT considered empty even if the text fragments are
    blank — preserving multimodal payloads that may still be useful.
    """
    if msg.role != "assistant":
        return False
    if not msg.contents:
        return True
    for c in msg.contents:
        if c.type != "text":
            return False
    return not "".join((c.text or "") for c in msg.contents).strip()
