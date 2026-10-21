# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ResponseValidationMiddleware — judges each response as it lands, without holding it back.

Some LLM / inference backends occasionally end a response without an
answer: empty content arrays, whitespace-only text, reasoning with no
visible reply, hosted evidence gathering with no final text, or a content
filter stop. Those would terminate the agent tool loop with a useless final
message, so this middleware fails the turn with the reason instead
(:class:`TerminalResponseValidationError`); the user continues from there.

This :class:`ChatMiddleware` sits innermost but for the usage middleware,
which counts every landed response before this one judges it. A stream passes
through it unchanged, every update as it arrives, so its text is shown
while the model writes it; the finished response is judged by a result
hook before the kernel loop lands it. What a stream showed cannot be
taken back, so only a response that showed nothing — empty, blank or
reasoning-only — is marked ``blank_response``, which the kernel's wire caller
resends once; every other rejection ends the turn.

Beside the verdict it observes what the provider ran: every hosted tool
call the response shows, as it arrives, is recorded for the retry owners'
commit-point gates (:meth:`hosted_commits_observed`,
:meth:`hosted_commits_in_flight`), and handed to the run's hosted
presentation hook. A response that passes keeps only its non-empty
assistant messages.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from chrys.foundation.errors import ContinuationVerdictError, ProviderResponseError
from chrys.foundation.hosted_tools import (
    HeldHostedEvidence,
    HostedRetrySafety,
    HostedToolPhase,
    HostedToolStatus,
    normalize_hosted_tool_status,
)
from chrys.foundation.trajectory.event_types import ValidationReason
from chrys.kernel import (
    ChatResponse,
    ChatResponseUpdate,
    ContextOverflowSink,
    Message,
    ResponseStream,
    in_internal_side_call,
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
from chrys.kernel.instrumentation import _stream_error_of
from chrys.kernel.middleware import ChatMiddleware
from chrys.service.agent_middleware.events.hosted_tools import adapt_hosted_tool
from chrys.service.agent_middleware.validators import DefaultResponseValidator, ValidationResult
from chrys.service.trajectory.validation import ValidationTrace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Sequence

    from chrys.kernel.middleware import ChatContext
    from chrys.service.agent_middleware.validators import ResponseValidator

logger = logging.getLogger(__name__)

# Verdicts on a response that showed nothing: no answer text, no call. A
# length stop with nothing generated is not one: the input filled the window.
_BLANK_VERDICTS = frozenset(
    {
        ValidationReason.NO_ASSISTANT_MESSAGE,
        ValidationReason.EMPTY_CONTENTS,
        ValidationReason.WHITESPACE_ONLY,
        ValidationReason.REASONING_ONLY,
    }
)


class ResponseValidationError(ContinuationVerdictError):
    """Base for a response validation rejected; its usage was already counted below this middleware.

    Only a completed response is judged, so a live continuation token now
    names an immutable rejected response: a retry must send a fresh request.
    """

    invalidates_continuation_token = True


class TerminalResponseValidationError(ResponseValidationError, RuntimeError):
    """Raised when response validation rejects a response; no retry lane retries it, but for one blank resend."""


class ResponseValidationObservationHook(Protocol):
    """Optional observer for raw content and validation-attempt boundaries."""

    async def begin_response(self, *, response_index: int | None = None) -> None: ...

    async def observe_contents(
        self, contents: Sequence[Any], *, is_final: bool = False, failed: bool = False
    ) -> None: ...

    async def attempt_started(self, *, continuation: bool = False) -> None: ...

    async def attempt_rejected(self, reason: str = "") -> None: ...

    async def attempt_accepted(self, messages: Sequence[Message]) -> None: ...


_LIVE_ACCESSOR = LiveAccessor()


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
    """Fail the turn on a response that ends without an answer; see the module docstring.

    Parameters
    ----------
    validator:
        Component that inspects each :class:`ChatResponse` and returns
        :class:`~chrys.service.agent_middleware.validators.ValidationResult`.
        Defaults to :class:`DefaultResponseValidator`.
    observation_hook:
        Optional raw-content and attempt-lifecycle observer. Callback failures
        are isolated from validation and hosted commit-point tracking.
    """

    def __init__(
        self,
        *,
        validator: ResponseValidator | None = None,
        observation_hook: ResponseValidationObservationHook | None = None,
    ) -> None:
        self._validator: ResponseValidator = validator or DefaultResponseValidator()
        self._observation_hook = observation_hook
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
            await self._notify_contents_observed(contents, is_final=True, failed=True)

    async def _notify_contents_observed(self, contents: Sequence[Any], *, is_final: bool, failed: bool = False) -> None:
        if self._observation_hook is None or in_internal_side_call():
            return
        try:
            await self._observation_hook.observe_contents(contents, is_final=is_final, failed=failed)
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
        continuation = isinstance(context.options, Mapping) and context.options.get("continuation_token") is not None
        await self._notify_response_started(continuation=continuation)
        self._begin_wire_attempt(continuation=continuation)
        await self._notify_attempt_started(continuation=continuation)
        validation = ValidationTrace.open(context.kwargs)
        try:
            await call_next()
        except ProviderResponseError as error:
            await self._observe_failed_response(error)
            raise
        result = context.result
        if context.stream:
            if not isinstance(result, ResponseStream):
                raise ValueError("Streaming response validation requires a ResponseStream result.")
            stream = result

            async def _observe_failure() -> None:
                # A provider stream that failed after showing hosted work:
                # the error carries what never reached an update.
                if isinstance(error := _stream_error_of(stream), ProviderResponseError):
                    await self._observe_failed_response(error)

            async def _conclude(response: ChatResponse) -> None:
                await self._conclude(context, response, validation)

            stream.with_transform_hook(self._observe_update)
            stream.with_cleanup_hook(_observe_failure)
            stream.with_result_hook(_conclude)
            return
        if isinstance(result, ChatResponse):
            await self._conclude(context, result, validation)

    async def _observe_update(self, update: ChatResponseUpdate) -> ChatResponseUpdate | None:
        """Record the hosted work one streamed update shows, as it arrives."""
        if isinstance(evidence := update.raw_representation, HeldHostedEvidence):
            # Hosted work the adapter holds behind an unfinished call already
            # ran: the retry gates count it now. The update that releases it
            # in order presents it.
            self._observe_hosted_contents(evidence.contents)
            return ChatResponseUpdate.transport_heartbeat()
        contents = list(update.contents or [])
        self._observe_hosted_contents(contents)
        await self._notify_contents_observed(contents, is_final=False)
        return None

    async def _conclude(self, context: ChatContext, response: ChatResponse, validation: ValidationTrace | None) -> None:
        """Judge a landed response: keep it, or fail the turn with the reason."""
        contents = [content for message in response.messages or [] for content in message.contents]
        self._observe_hosted_contents(contents)
        await self._notify_contents_observed(contents, is_final=True)
        if response.continuation_token is not None:
            # A background response still running: the poll that finishes it is judged.
            await self._notify_attempt_accepted(response.messages or [])
            return
        verdict = self._validator.validate(response)
        _note_output_truncated(context, verdict, response)
        if validation is not None:
            await validation.finished(accepted=verdict.ok, reason_code=None if verdict.ok else verdict.code)
        if not verdict.ok:
            await self._notify_attempt_rejected(verdict.reason)
            _raise_terminal(verdict)
        # The validator reads only the final assistant message, so a response
        # shaped ``[empty, valid]`` passes with a stray ``content: []`` message
        # that would otherwise land in persisted history.
        response.messages = [message for message in response.messages if not _is_empty_assistant_message(message)]
        await self._notify_attempt_accepted(response.messages)


def _raise_terminal(verdict: ValidationResult) -> None:
    """Raise a rejected response into the pass observer error path.

    A filtered response is raised as the provider's own content-filter
    failure: it is classified as filtered, and a live continuation token
    names a finished response that a retry must not poll.
    """
    logger.warning("Response validation failed: %s", verdict.reason)
    error = TerminalResponseValidationError(verdict.reason)
    error.blank_response = verdict.code in _BLANK_VERDICTS
    if verdict.code == ValidationReason.CONTENT_FILTERED:
        raise error from ProviderResponseError(
            "content_filter", verdict.reason, retryable=False, invalidates_continuation_token=True
        )
    raise error


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
