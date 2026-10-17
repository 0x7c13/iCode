# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
"""Read-only projection of one session's trajectory log into per-turn request data.

Every fact here comes from the log: which pass belongs to which turn, what each
model call spent and how it ended, and which retries were scheduled. The turn
registry (``chrys_trajectory``) is deliberately not read — an offline export has
to work on a log whose session file is gone, and the turn a retry belongs to is
already recorded on its own ``turn.started``. The caller merges the registry in,
which is the only place the two views of a turn meet.

One forward pass, nothing retained between events: the only state that outlives
an event is the branch cursor and one open record per pass, call and retry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from chrys.foundation.trajectory.envelope import TrajectoryEvent
from chrys.foundation.trajectory.event_types import EventType
from chrys.service.analytics.reader import ScanBatch, ScanCursor, scan_trajectory_batch
from chrys.service.trajectory.state import TurnRecord

Coverage = Literal["exact", "partial"]

_USAGE_FIELDS: Final = (
    "input_tokens",
    "output_tokens",
    "cached_read_tokens",
    "cached_write_tokens",
    "thought_tokens",
)
_USAGE_KEYS: Final = {
    "input_total": "input_tokens",
    "output_total": "output_tokens",
    "cache_read": "cached_read_tokens",
    "cache_creation": "cached_write_tokens",
    "reasoning": "thought_tokens",
}
_USAGE_WIRE_NAMES: Final = {
    "input_tokens": "inputTokens",
    "output_tokens": "outputTokens",
    "cached_read_tokens": "cachedReadTokens",
    "cached_write_tokens": "cachedWriteTokens",
    "thought_tokens": "thoughtTokens",
}


@dataclass(slots=True)
class TraceUsage:
    """Provider-reported tokens for one call or one turn; each field is optional on its own."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_read_tokens: int | None = None
    cached_write_tokens: int | None = None
    thought_tokens: int | None = None

    def add(self, other: TraceUsage) -> None:
        for field_name in _USAGE_FIELDS:
            value = getattr(other, field_name)
            if value is None:
                continue
            current = getattr(self, field_name)
            setattr(self, field_name, value if current is None else current + value)

    def to_payload(self) -> dict[str, int]:
        # Wire names are camelCase, like every other field here: a client reads
        # these with the same schema it reads a prompt's usage with.
        return {
            _USAGE_WIRE_NAMES[field]: value for field in _USAGE_FIELDS if (value := getattr(self, field)) is not None
        }


@dataclass(slots=True)
class TraceRequestAttempt:
    """One HTTP attempt of a model call, keyed by the writer's request attempt id."""

    request_attempt_id: str
    http_status: int | None = None
    provider_request_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "requestAttemptId": self.request_attempt_id,
            "httpStatus": self.http_status,
            "providerRequestId": self.provider_request_id,
        }


@dataclass(slots=True)
class TraceModelCall:
    """One model exchange — one provider round trip — and how it ended."""

    model_call_id: str
    pass_id: str | None
    actor_role: str
    actor_kind: str
    invocation_id: str | None
    agent_profile: str | None
    provider: str | None
    request_model: str | None
    response_model: str | None
    stream: bool | None
    started_at: str | None
    ended_at: str | None
    duration_ms: int | None
    ttfc_ms: int | None
    ttfv_ms: int | None
    outcome: str | None
    finish_reason: str | None
    error_kind: str | None
    error_code: str | None
    retryable: bool | None
    response_id: str | None
    usage_reported: bool
    usage: TraceUsage
    attempts: list[TraceRequestAttempt] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "modelCallId": self.model_call_id,
            "passId": self.pass_id,
            "actor": {"role": self.actor_role, "kind": self.actor_kind, "invocationId": self.invocation_id},
            "agentProfile": self.agent_profile,
            "provider": self.provider,
            "requestModel": self.request_model,
            "responseModel": self.response_model,
            "stream": self.stream,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "durationMs": self.duration_ms,
            "ttfcMs": self.ttfc_ms,
            "ttfvMs": self.ttfv_ms,
            "outcome": self.outcome,
            "finishReason": self.finish_reason,
            "errorKind": self.error_kind,
            "errorCode": self.error_code,
            "retryable": self.retryable,
            "responseId": self.response_id,
            "usageReported": self.usage_reported,
            "usage": self.usage.to_payload(),
            "attempts": [attempt.to_payload() for attempt in self.attempts],
        }


@dataclass(slots=True)
class TraceRetry:
    """One retry of a model call, counted at its ``retry.started``.

    A ``retry.scheduled`` whose ``started`` never arrived is kept with
    ``started`` unset: the backoff was announced but the retry did not run, so
    it is reported without being counted.
    """

    retry_id: str
    pass_id: str | None
    reason_code: str | None
    delay_ms: int | None
    retry_mode: str | None
    started: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "retryId": self.retry_id,
            "passId": self.pass_id,
            "reasonCode": self.reason_code,
            "delayMs": self.delay_ms,
            "retryMode": self.retry_mode,
            "started": self.started,
        }


@dataclass(slots=True)
class TracePass:
    """One execution of a logical turn (a retry opens another one)."""

    pass_id: str
    started_at: str | None
    ended_at: str | None
    end_reason: str | None
    ended_sequence: int | None = None
    """Where the pass's ``turn.finished`` landed; ``None`` while it is unfinished."""
    model_calls: list[TraceModelCall] = field(default_factory=list)
    retries: list[TraceRetry] = field(default_factory=list)

    @property
    def retry_count(self) -> int:
        return sum(1 for retry in self.retries if retry.started)

    def to_payload(self) -> dict[str, Any]:
        return {
            "passId": self.pass_id,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "endReason": self.end_reason,
            "modelCallCount": len(self.model_calls),
            "retryCount": self.retry_count,
            "modelCalls": [call.to_payload() for call in self.model_calls],
            "retries": [retry.to_payload() for retry in self.retries],
        }


@dataclass(slots=True)
class TraceTurn:
    """One logical turn: its passes, its calls, and what they spent and retried."""

    turn_id: str
    turn_number: int | None = None
    coverage: Coverage = "exact"
    started_sequence: int | None = None
    passes: list[TracePass] = field(default_factory=list)
    model_calls: list[TraceModelCall] = field(default_factory=list)
    retries: list[TraceRetry] = field(default_factory=list)
    usage: TraceUsage = field(default_factory=TraceUsage)
    usage_unreported_calls: int = 0
    status: str = "unfinished"

    @property
    def request_attempt_count(self) -> int:
        return sum(len(call.attempts) for call in self.model_calls)

    @property
    def retry_count(self) -> int:
        return sum(1 for retry in self.retries if retry.started)

    @property
    def failed_model_call_count(self) -> int:
        return sum(1 for call in self.model_calls if call.outcome not in (None, "success"))

    @property
    def started_at(self) -> str | None:
        return next((pass_.started_at for pass_ in self.passes if pass_.started_at is not None), None)

    @property
    def ended_at(self) -> str | None:
        return self.passes[-1].ended_at if self.passes else None

    def to_payload(self) -> dict[str, Any]:
        return {
            "turnId": self.turn_id,
            "turnNumber": self.turn_number,
            "coverage": self.coverage,
            "status": self.status,
            "startedAt": self.started_at,
            "endedAt": self.ended_at,
            "usage": self.usage.to_payload(),
            "usageUnreportedCalls": self.usage_unreported_calls,
            "modelCallCount": len(self.model_calls),
            "failedModelCallCount": self.failed_model_call_count,
            "requestAttemptCount": self.request_attempt_count,
            "retryCount": self.retry_count,
            "passes": [pass_.to_payload() for pass_ in self.passes],
            "modelCalls": [call.to_payload() for call in self.model_calls],
            "retries": [retry.to_payload() for retry in self.retries],
        }


@dataclass(slots=True)
class TraceUnassignedPass:
    """A pass the log recorded without a ``turn.started`` that could place it."""

    pass_id: str
    model_calls: list[TraceModelCall] = field(default_factory=list)
    retries: list[TraceRetry] = field(default_factory=list)

    @property
    def retry_count(self) -> int:
        return sum(1 for retry in self.retries if retry.started)

    def to_payload(self) -> dict[str, Any]:
        return {
            "passId": self.pass_id,
            "coverage": "partial",
            "modelCallCount": len(self.model_calls),
            "requestAttemptCount": sum(len(call.attempts) for call in self.model_calls),
            "retryCount": self.retry_count,
            "modelCalls": [call.to_payload() for call in self.model_calls],
            "retries": [retry.to_payload() for retry in self.retries],
        }


@dataclass(slots=True)
class TraceSessionLevel:
    """Calls the log recorded outside any pass — a title generated after its turn."""

    model_calls: list[TraceModelCall] = field(default_factory=list)
    usage: TraceUsage = field(default_factory=TraceUsage)

    def to_payload(self) -> dict[str, Any]:
        return {
            "modelCallCount": len(self.model_calls),
            "requestAttemptCount": sum(len(call.attempts) for call in self.model_calls),
            "usage": self.usage.to_payload(),
            "modelCalls": [call.to_payload() for call in self.model_calls],
        }


@dataclass(slots=True)
class TurnTrace:
    """Everything one trajectory log says about a session's turns and requests."""

    session_id: str
    coverage: Coverage
    gap_count: int
    turns: list[TraceTurn]
    superseded_turns: list[TraceTurn]
    unassigned_passes: list[TraceUnassignedPass]
    session_level: TraceSessionLevel

    def to_payload(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "coverage": self.coverage,
            "gapCount": self.gap_count,
            "turns": [turn.to_payload() for turn in self.turns],
            "supersededTurns": [turn.to_payload() for turn in self.superseded_turns],
            "unassignedPasses": [pass_.to_payload() for pass_ in self.unassigned_passes],
            "sessionLevel": self.session_level.to_payload(),
        }


@dataclass(slots=True)
class _PassState:
    """One open pass, folded into a :class:`TracePass` when the scan ends."""

    pass_id: str
    turn_id: str
    started_sequence: int
    started_at: str | None
    ended_sequence: int | None = None
    next_started_sequence: int | None = None
    ended_at: str | None = None
    end_reason: str | None = None
    calls: list[TraceModelCall] = field(default_factory=list)
    retries: list[TraceRetry] = field(default_factory=list)

    def finished(self) -> TracePass:
        return TracePass(
            pass_id=self.pass_id,
            started_at=self.started_at,
            ended_at=self.ended_at,
            end_reason=self.end_reason,
            ended_sequence=self.ended_sequence,
            model_calls=self.calls,
            retries=self.retries,
        )


@dataclass(slots=True)
class _PendingRetry:
    """A ``retry.scheduled`` waiting for the ``retry.started`` that counts it."""

    retry_id: str
    pass_id: str | None
    branch_turn_id: str | None
    """Scheduling-time fallback for a retry with no pass; never follows later branch changes."""
    reason_code: str | None
    delay_ms: int | None
    retry_mode: str | None


@dataclass(slots=True)
class _Outstanding:
    """What the scan ended still holding open: unvouched turns, and the session itself."""

    turns: set[str] = field(default_factory=set)
    session: bool = False


@dataclass(slots=True)
class _TurnState:
    """A turn as the log describes it, before it is folded into a payload."""

    turn_id: str
    started_sequence: int
    turn_number: int | None = None
    passes: list[_PassState] = field(default_factory=list)
    loose_calls: list[TraceModelCall] = field(default_factory=list)
    loose_retries: list[TraceRetry] = field(default_factory=list)


class _Projection:
    """The scan's single consumer: it holds the open state and takes one event at a time."""

    def __init__(self) -> None:
        self.session_id = ""
        self.passes: dict[str, _PassState] = {}
        self.last_pass: _PassState | None = None
        self.turns: list[_TurnState] = []
        self.turns_by_id: dict[str, _TurnState] = {}
        self.branch_turn_id: str | None = None
        self.calls_by_operation: dict[str, TraceModelCall] = {}
        self.attempts_by_operation: dict[str, dict[str, TraceRequestAttempt]] = {}
        self.pending_retries: dict[str, _PendingRetry] = {}
        self.loose_calls: dict[str, list[TraceModelCall]] = {}
        self.loose_retries: dict[str, list[TraceRetry]] = {}
        self.session_calls: list[TraceModelCall] = []
        self.rollback_ranges: list[tuple[int, int]] = []
        self.gap_count = 0
        self.gap_ranges: list[tuple[int, int]] = []
        self.finished_operations: set[str] = set()
        self.missing_start_pass_ids: set[str | None] = set()

    def consume(self, event: TrajectoryEvent, _offset: int, _byte_length: int, _line_number: int) -> None:
        self.session_id = self.session_id or event.session_id
        event_type = event.event_type
        if event_type == EventType.TURN_STARTED:
            self._turn_started(event)
        elif event_type == EventType.TURN_FINISHED:
            self._turn_finished(event)
        elif event_type == EventType.MODEL_EXCHANGE_STARTED:
            self._exchange_started(event)
        elif event_type == EventType.MODEL_EXCHANGE_FINISHED:
            self._exchange_finished(event)
        elif event_type == EventType.MODEL_REQUEST_PREPARED:
            self._request_prepared(event)
        elif event_type == EventType.MODEL_REQUEST_HEADERS_RECEIVED:
            self._request_identified(event)
        elif event_type == EventType.RETRY_SCHEDULED:
            self._retry_scheduled(event)
        elif event_type == EventType.RETRY_STARTED:
            self._retry_started(event)
        elif event_type == EventType.SESSION_ROLLBACK:
            self._rollback(event)
        elif event_type == EventType.GAP:
            self.gap_count += 1
            first = _int_at(event.payload, "first_sequence")
            last = _int_at(event.payload, "last_sequence")
            if first is not None:
                self.gap_ranges.append((first, last if last is not None and last >= first else first))

    # ------------------------------------------------------------------ turns

    def _turn_started(self, event: TrajectoryEvent) -> None:
        pass_id = event.turn_id
        if pass_id is None:
            return
        logical = event.payload.get("logical_turn_id")
        if isinstance(logical, str) and logical:
            turn_id = logical
        elif _bool_at(event.payload, "is_retry") is True:
            # A pass recorded before ``logical_turn_id`` existed: it is a retry
            # (the writer mints a fresh pass id for every retry and always says
            # so) and belongs to the turn the current branch last opened. With
            # the cursor cleared — a rollback that named no target — it stands
            # alone instead of being guessed into a neighbouring turn.
            turn_id = self.branch_turn_id or pass_id
        else:
            turn_id = pass_id
        state = _PassState(
            pass_id=pass_id, turn_id=turn_id, started_sequence=event.sequence, started_at=event.occurred_at
        )
        if self.last_pass is not None:
            self.last_pass.next_started_sequence = event.sequence
        self.last_pass = state
        self.passes[pass_id] = state
        owner = self.turns_by_id.get(turn_id)
        if owner is None:
            owner = _TurnState(turn_id=turn_id, started_sequence=event.sequence)
            self.turns_by_id[turn_id] = owner
            self.turns.append(owner)
        if owner.turn_number is None:
            number = _int_at(event.payload, "turn_number")
            owner.turn_number = number if number is not None and number >= 0 else None
        owner.passes.append(state)
        # Only a fresh pass (its own id is the turn's) moves the branch cursor.
        # A retry pass — with or without ``logical_turn_id`` — resumes a turn
        # that is already open, so the cursor stays where that turn put it.
        if pass_id == turn_id:
            self.branch_turn_id = turn_id

    def _turn_finished(self, event: TrajectoryEvent) -> None:
        state = self.passes.get(event.turn_id or "")
        if state is None:
            return
        state.ended_sequence = event.sequence
        state.ended_at = event.occurred_at
        state.end_reason = _str_at(event.payload, "end_reason")

    def _rollback(self, event: TrajectoryEvent) -> None:
        first = _int_at(event.payload, "superseded_from_sequence")
        last = _int_at(event.payload, "superseded_to_sequence")
        if first is not None and last is not None and first <= last:
            self.rollback_ranges.append((first, last))
        # A rollback that names no target (back to the welcome screen, or
        # written before the field existed) leaves no current branch turn.
        self.branch_turn_id = _str_at(event.payload, "target_turn_id")

    # ------------------------------------------------------------ model calls

    def _exchange_started(self, event: TrajectoryEvent) -> None:
        operation_id = event.operation_id
        if operation_id is None or operation_id in self.finished_operations:
            return
        self.calls_by_operation[operation_id] = TraceModelCall(
            model_call_id=operation_id,
            pass_id=event.turn_id,
            actor_role=event.actor.role,
            actor_kind=event.actor.kind,
            invocation_id=event.actor.invocation_id,
            agent_profile=_str_at(event.payload, "agent_profile_id"),
            provider=_str_at(event.payload, "provider"),
            request_model=_str_at(event.payload, "request_model"),
            response_model=None,
            stream=_bool_at(event.payload, "stream"),
            started_at=event.occurred_at,
            ended_at=None,
            duration_ms=None,
            ttfc_ms=None,
            ttfv_ms=None,
            outcome=None,
            finish_reason=None,
            error_kind=None,
            error_code=None,
            retryable=None,
            response_id=None,
            usage_reported=False,
            usage=TraceUsage(),
        )
        self.attempts_by_operation.setdefault(operation_id, {})

    def _exchange_finished(self, event: TrajectoryEvent) -> None:
        operation_id = event.operation_id
        if operation_id is not None:
            # A terminal is the operation's end, once. A second one for the
            # same operation — a replayed tail, a writer that emitted twice —
            # would otherwise enter as a fresh call with a lost start and
            # count the call and its usage twice.
            if operation_id in self.finished_operations:
                return
            self.finished_operations.add(operation_id)
        call = self.calls_by_operation.pop(operation_id, None) if operation_id is not None else None
        if call is None and operation_id is not None:
            # The log lost the ``started`` (a torn head, a purged range), but
            # this record is readable work: it keeps the call — and the
            # attempts that join it to the wire — with an unknown start.
            call = self._call_without_start(event, operation_id)
        if call is not None:
            if call.started_at is None:
                self.missing_start_pass_ids.add(call.pass_id)
            # Request records can create the call before a surviving terminal
            # supplies its exchange facts. Preserve facts from a readable start,
            # especially its per-request model override over the profile model.
            call.agent_profile = call.agent_profile or _str_at(event.payload, "agent_profile_id")
            call.provider = call.provider or _str_at(event.payload, "provider")
            call.request_model = call.request_model or _str_at(event.payload, "request_model")
            call.ended_at = event.occurred_at
            call.duration_ms = _int_at(event.payload, "duration_ms")
            call.ttfc_ms = _int_at(event.payload, "ttfc_ms")
            call.ttfv_ms = _int_at(event.payload, "ttfv_ms")
            call.outcome = _str_at(event.payload, "outcome")
            call.finish_reason = _str_at(event.payload, "finish_reason")
            call.error_kind = _str_at(event.payload, "error_kind")
            call.error_code = _str_at(event.payload, "error_code")
            call.retryable = _bool_at(event.payload, "retryable")
            call.response_id = _str_at(event.payload, "response_id")
            call.response_model = _str_at(event.payload, "response_model")
            call.usage = _usage_at(event.payload)
            call.usage_reported = bool(call.usage.to_payload())
            call.attempts = list(self.attempts_by_operation.pop(operation_id or "", {}).values())
        self._place(call, event.turn_id)

    def _request_prepared(self, event: TrajectoryEvent) -> None:
        self._request_attempt(event)

    def _request_identified(self, event: TrajectoryEvent) -> None:
        attempt = self._request_attempt(event)
        if attempt is not None:
            attempt.http_status = _int_at(event.payload, "status_code")
            attempt.provider_request_id = _str_at(event.payload, "provider_request_id")

    def _request_attempt(self, event: TrajectoryEvent) -> TraceRequestAttempt | None:
        """Retain the request's exchange and pass even when both exchange markers are lost."""
        attempt_id = _str_at(event.payload, "request_attempt_id")
        operation_id = event.operation_id
        if attempt_id is None or operation_id is None or operation_id in self.finished_operations:
            return None
        if operation_id not in self.calls_by_operation:
            self.calls_by_operation[operation_id] = self._call_without_start(event, operation_id)
        return self.attempts_by_operation.setdefault(operation_id, {}).setdefault(
            attempt_id, TraceRequestAttempt(request_attempt_id=attempt_id)
        )

    def _place(self, call: TraceModelCall | None, pass_id: str | None) -> None:
        """File a finished call under its pass, or keep it unassigned.

        The envelope's ``turn_id`` is the call's own pass id, so a call from a
        pass that recorded no ``turn.started`` keeps its identity instead of
        being guessed into a neighbouring turn. A call the log gives no pass at
        all is a side call outside every pass — a title generated after its
        turn — and belongs to the session.
        """
        if call is None:
            return
        state = self.passes.get(pass_id or "")
        if state is not None:
            state.calls.append(call)
            return
        if pass_id is None:
            self.session_calls.append(call)
            return
        self.loose_calls.setdefault(pass_id, []).append(call)

    # ---------------------------------------------------------------- retries

    def _retry_scheduled(self, event: TrajectoryEvent) -> None:
        operation_id = event.operation_id
        if operation_id is None:
            return
        self.pending_retries[operation_id] = _PendingRetry(
            retry_id=operation_id,
            pass_id=event.turn_id,
            branch_turn_id=self.branch_turn_id,
            reason_code=_str_at(event.payload, "reason_code"),
            delay_ms=_int_at(event.payload, "delay_ms"),
            retry_mode=_str_at(event.payload, "retry_mode"),
        )

    def _retry_started(self, event: TrajectoryEvent) -> None:
        operation_id = event.operation_id
        if operation_id is None:
            return
        pending = self.pending_retries.pop(operation_id, None)
        retry = TraceRetry(
            retry_id=operation_id,
            pass_id=pending.pass_id if pending is not None else event.turn_id,
            reason_code=pending.reason_code if pending is not None else None,
            delay_ms=pending.delay_ms if pending is not None else None,
            retry_mode=_str_at(event.payload, "retry_mode") or (pending.retry_mode if pending is not None else None),
        )
        self._place_retry(retry, branch_turn_id=pending.branch_turn_id if pending is not None else self.branch_turn_id)

    def _place_retry(self, retry: TraceRetry, *, branch_turn_id: str | None) -> str:
        """File *retry* under its pass or turn and return the turn that owns it.

        When the retry has no pass, use the branch turn captured at scheduling,
        or at its start when the schedule is missing. A later turn or rollback
        cannot change that ownership, including when no owner was known.
        """
        state = self.passes.get(retry.pass_id or "")
        if state is not None:
            state.retries.append(retry)
            return state.turn_id
        owner = self.turns_by_id.get(branch_turn_id or "")
        if owner is not None and retry.pass_id is None:
            owner.loose_retries.append(retry)
            return owner.turn_id
        if retry.pass_id is None:
            return ""
        self.loose_retries.setdefault(retry.pass_id, []).append(retry)
        return self._turn_of_pass(retry.pass_id)

    # ----------------------------------------------------------------- finish

    def finish(self, batch: ScanBatch) -> TurnTrace:
        outstanding = self._close_outstanding()
        superseded_ids = self._superseded_turn_ids()
        turns: list[TraceTurn] = []
        superseded: list[TraceTurn] = []
        for state in self.turns:
            turn = self._build_turn(state)
            (superseded if turn.turn_id in superseded_ids else turns).append(turn)
        unassigned = self._unassigned_passes(superseded_ids)
        session_level = TraceSessionLevel(model_calls=self.session_calls)
        for call in self.session_calls:
            session_level.usage.add(call.usage)
        coverage = self._mark_coverage(batch, turns, superseded, outstanding, bool(unassigned))
        return TurnTrace(
            session_id=self.session_id,
            coverage=coverage,
            gap_count=self.gap_count,
            turns=turns,
            superseded_turns=superseded,
            unassigned_passes=unassigned,
            session_level=session_level,
        )

    def _close_outstanding(self) -> _Outstanding:
        """Fold the calls and backoffs the log never finished into what they touch.

        An unfinished call is still what a frontend sees: a request in flight
        while the session is open, or the last thing a crashed process wrote.
        Dropping it would hide readable work — and the attempt ids that join it
        to the wire — so it is reported with ``outcome: null``. What it also
        does is say the scan cannot vouch for the session: the log simply ends
        mid-request, so ``session`` is set even for a call that belongs to no
        turn this log opened.

        A ``retry.scheduled`` with no ``retry.started`` is a *cancelled*
        backoff, not a retry: the interrupt can land in the backoff sleep
        itself, and the retry never ran. It is kept as a pending record and
        never counted — but the turn that was waiting out the backoff is not
        vouched for either, so it joins *turns*.
        """
        outstanding = _Outstanding()
        for call in self.calls_by_operation.values():
            call.attempts = list(self.attempts_by_operation.get(call.model_call_id, {}).values())
            self._place(call, call.pass_id)
            outstanding.session = True
            owner = self._turn_of_pass(call.pass_id)
            if owner:
                outstanding.turns.add(owner)
        for pass_id in self.missing_start_pass_ids:
            outstanding.session = True
            owner = self._turn_of_pass(pass_id)
            if owner:
                outstanding.turns.add(owner)
        for pending in self.pending_retries.values():
            outstanding.session = True
            owner = self._place_retry(
                TraceRetry(
                    retry_id=pending.retry_id,
                    pass_id=pending.pass_id,
                    reason_code=pending.reason_code,
                    delay_ms=pending.delay_ms,
                    retry_mode=pending.retry_mode,
                    started=False,
                ),
                branch_turn_id=pending.branch_turn_id,
            )
            if owner:
                outstanding.turns.add(owner)
        self.calls_by_operation.clear()
        self.attempts_by_operation.clear()
        self.pending_retries.clear()
        return outstanding

    def _turn_of_pass(self, pass_id: str | None) -> str:
        """The turn a pass belongs to; a pass with no ``turn.started`` is not a turn."""
        if pass_id is None:
            return ""
        state = self.passes.get(pass_id)
        return state.turn_id if state is not None else pass_id

    def _call_without_start(self, event: TrajectoryEvent, operation_id: str) -> TraceModelCall:
        """A readable request or terminal whose exchange start the log lost."""
        return TraceModelCall(
            model_call_id=operation_id,
            pass_id=event.turn_id,
            actor_role=event.actor.role,
            actor_kind=event.actor.kind,
            invocation_id=event.actor.invocation_id,
            agent_profile=_str_at(event.payload, "agent_profile_id"),
            provider=_str_at(event.payload, "provider"),
            request_model=_str_at(event.payload, "request_model"),
            response_model=None,
            stream=None,
            started_at=None,
            ended_at=None,
            duration_ms=None,
            ttfc_ms=None,
            ttfv_ms=None,
            outcome=None,
            finish_reason=None,
            error_kind=None,
            error_code=None,
            retryable=None,
            response_id=None,
            usage_reported=False,
            usage=TraceUsage(),
        )

    def _superseded_turn_ids(self) -> set[str]:
        """A turn is superseded when a rollback's recorded range covers its first pass.

        Only turns this scan saw can be named: every one of them has the
        sequence its ``turn.started`` landed at, and a turn whose ordinal was
        reused is still decided by that sequence, not by the number. A turn the
        registry alone knows — inherited from a fork, no local ``turn.started``
        — never enters this set; a rollback that removed it appears in neither
        list (see ``frontend-api.md``).
        """
        superseded: set[str] = set()
        for state in self.turns:
            sequence = state.started_sequence
            if any(first <= sequence <= last for first, last in self.rollback_ranges):
                superseded.add(state.turn_id)
        return superseded

    def _build_turn(self, state: _TurnState) -> TraceTurn:
        turn = TraceTurn(turn_id=state.turn_id, turn_number=state.turn_number, started_sequence=state.started_sequence)
        for pass_state in state.passes:
            finished = pass_state.finished()
            turn.passes.append(finished)
            turn.model_calls.extend(finished.model_calls)
            turn.retries.extend(finished.retries)
        turn.model_calls.extend(state.loose_calls)
        turn.retries.extend(state.loose_retries)
        for call in turn.model_calls:
            turn.usage.add(call.usage)
            if not call.usage_reported:
                turn.usage_unreported_calls += 1
        last = turn.passes[-1] if turn.passes else None
        if last is not None and last.ended_sequence is not None:
            # The end reason is the backend's own word for how the pass ended,
            # and it can be absent even on a finished pass: report that as
            # unknown rather than inventing "unfinished".
            turn.status = last.end_reason or "unknown"
        else:
            turn.status = "unfinished"
        return turn

    def _unassigned_passes(self, superseded_ids: set[str]) -> list[TraceUnassignedPass]:
        """Passes with calls or retries but no ``turn.started``, kept rather than guessed at.

        A pass a rollback superseded is left out: identity says its calls were
        discarded, and listing them as unassigned would put them back on the
        current branch the rollback removed them from.
        """
        pass_ids = [*self.loose_calls, *(pass_id for pass_id in self.loose_retries if pass_id not in self.loose_calls)]
        return [
            TraceUnassignedPass(
                pass_id=pass_id,
                model_calls=list(self.loose_calls.get(pass_id, ())),
                retries=list(self.loose_retries.get(pass_id, ())),
            )
            for pass_id in pass_ids
            if pass_id not in superseded_ids
        ]

    def _mark_coverage(
        self,
        batch: ScanBatch,
        turns: list[TraceTurn],
        superseded: list[TraceTurn],
        outstanding: _Outstanding,
        unreadable: bool,
    ) -> Coverage:
        """Session coverage, and a ``partial`` mark on the turns the damage can reach.

        Damage is positioned by sequence — a corrupt line names the last good
        sequence before it, a gap names the range it covers, a torn tail is
        everything after the last decoded record — so it can only affect a
        pass whose own span meets it. Every damage interval is tested against
        every pass: an early gap inside one turn says nothing about a later
        turn, and gaps between finished passes and their retries touch neither
        pass. A turn holding a call with a missing boundary is partial too.
        """
        damage = self._damage(batch)
        for turn in (*turns, *superseded):
            passes = self.turns_by_id[turn.turn_id].passes
            if turn.turn_id in outstanding.turns or any(
                _pass_reaches(pass_, span) for pass_ in passes for span in damage
            ):
                turn.coverage = "partial"
        incomplete = bool(damage) or outstanding.session or unreadable
        return "partial" if incomplete else "exact"

    def _damage(self, batch: ScanBatch) -> list[tuple[int, int | None]]:
        """Every sequence range the scan could not read, inclusive.

        A corrupt line names the last good sequence before it, so what it cost
        starts one slot later and runs to the end of the log. A gap names the
        missing range exactly, and so does a prefix violation. A torn tail —
        bytes with no terminating newline — is everything after the last
        decoded record.
        """
        spans: list[tuple[int, int | None]] = [(corrupt.previous_sequence + 1, None) for corrupt in batch.corrupt_lines]
        spans.extend((sequence, sequence) for sequence in batch.unsupported_event_sequences)
        spans.extend((violation.first_sequence, violation.last_sequence) for violation in batch.prefix_violations)
        for first, last in self.gap_ranges:
            spans.append((first, last))
        if batch.torn_tail_bytes:
            spans.append((batch.cursor.previous_sequence + 1, None))
        return spans


def _pass_reaches(pass_: _PassState, damage: tuple[int, int | None]) -> bool:
    """Whether damage overlaps this pass's inclusive sequence interval.

    An unfinished pass extends to the next pass's start, or the scan's tail
    when there is no successor. Each retry has its own interval: an earlier
    pass's finish cannot bound it, and gaps between passes touch neither one.
    """
    first, last = damage
    if last is not None and pass_.started_sequence > last:
        return False
    end = pass_.ended_sequence
    if end is None and pass_.next_started_sequence is not None:
        end = pass_.next_started_sequence - 1
    return end is None or end >= first


def _usage_at(payload: Any) -> TraceUsage:
    usage = payload.get("usage")
    normalized = usage.get("normalized") if isinstance(usage, dict) else None
    if not isinstance(normalized, dict):
        return TraceUsage()
    values: dict[str, int | None] = {}
    for source_key, field_name in _USAGE_KEYS.items():
        value = normalized.get(source_key)
        values[field_name] = value if isinstance(value, int) and not isinstance(value, bool) else None
    return TraceUsage(**values)


def _str_at(payload: Any, key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _int_at(payload: Any, key: str) -> int | None:
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bool_at(payload: Any, key: str) -> bool | None:
    return value if isinstance(value := payload.get(key), bool) else None


def scan_turn_trace(path: Path) -> TurnTrace:
    """Project one trajectory log into per-turn request data."""
    projection = _Projection()
    batch = scan_trajectory_batch(path, projection.consume, cursor=ScanCursor())
    return projection.finish(batch)


def trace_payload_with_registry(
    trace: TurnTrace,
    records: list[TurnRecord],
    *,
    turn_id: str | None = None,
) -> dict[str, Any] | None:
    """Fold the turn registry into a projection, or ``None`` when *turn_id* is unknown.

    The registry is what names a turn's opening user message and where a fork
    inherited it from, and it is the only record of a turn this session never
    ran at all. Neither side is trusted over the other: a turn the log ran but
    the registry has not saved yet (the turn in flight) keeps the log's data
    with no ``userMessageId``, and a turn the registry knows but this log never
    ran is reported with ``traceAvailable: false`` and no calls. Statistics stay
    per-session: the source session's log is never read. With *turn_id*, only
    that turn is returned — a superseded turn included, and a fork-inherited
    one too, since its registry record is all it has — and the branch lists
    never mix: a turn stays in the list the log put it in. Unassigned passes
    are omitted from a filtered response; session coverage still covers the
    whole scan.
    """
    payload = trace.to_payload()
    by_id = {record.turn_id: record for record in records}
    runs = payload["turns"] + payload["supersededTurns"]
    for turn in runs:
        turn["traceAvailable"] = True
        record = by_id.get(turn["turnId"])
        if record is None:
            turn["userMessageId"] = None
            turn["inheritedFrom"] = None
            continue
        turn["turnNumber"] = record.turn_number
        turn["userMessageId"] = record.user_message_id
        turn["inheritedFrom"] = record.origin_session_id
    known = {turn["turnId"] for turn in runs}
    inherited = [_registry_only_turn(record) for record in records if record.turn_id not in known]
    if turn_id is not None:
        # Every turn this session knows is fair game for a filter, superseded
        # and fork-inherited ones included: the id is the identity, not the
        # list a turn happens to live in. The response then names only that
        # turn, and keeps it in the list it belongs to.
        matched = next(
            (
                turn
                for turn in (*payload["turns"], *payload["supersededTurns"], *inherited)
                if turn["turnId"] == turn_id
            ),
            None,
        )
        if matched is None:
            return None
        superseded = any(turn["turnId"] == turn_id for turn in payload["supersededTurns"])
        payload["turns"] = [] if superseded else [matched]
        payload["supersededTurns"] = [matched] if superseded else []
        payload["unassignedPasses"] = []
        return payload
    payload["turns"] = _in_start_order(payload["turns"], inherited, records)
    return payload


def _in_start_order(
    turns: list[dict[str, Any]], inherited: list[dict[str, Any]], records: list[TurnRecord]
) -> list[dict[str, Any]]:
    """Merge registry-only turns around the log's chronological anchors.

    Logged turns keep their relative order, even when the registry disagrees
    or omits an unsaved turn. Shared turns provide anchors for placing
    registry-only turns (including fork ancestors) in registry append order.
    The log wins ordering conflicts; display ordinals cannot order either
    list because they can repeat.
    """
    log_index = {turn["turnId"]: index for index, turn in enumerate(turns)}
    inherited_by_id = {turn["turnId"]: turn for turn in inherited}
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor = 0
    for record in records:
        index = log_index.get(record.turn_id)
        if index is not None:
            if index >= cursor:
                ordered.extend(turns[cursor : index + 1])
                cursor = index + 1
        elif record.turn_id in inherited_by_id and record.turn_id not in seen:
            ordered.append(inherited_by_id[record.turn_id])
        seen.add(record.turn_id)
    ordered.extend(turns[cursor:])
    return ordered


def _registry_only_turn(record: TurnRecord) -> dict[str, Any]:
    """A turn this session never ran — inherited by a fork, or recorded without its trajectory.

    Nothing here can be vouched for or contradicted: this session holds no
    record of the turn, which ``traceAvailable: false`` says plainly. Its
    ``coverage`` describes the absence of a local record, not damage to one.
    """
    return {
        "turnId": record.turn_id,
        "turnNumber": record.turn_number,
        "userMessageId": record.user_message_id,
        "inheritedFrom": record.origin_session_id,
        "coverage": "exact",
        "traceAvailable": False,
        "status": "unavailable",
        "startedAt": None,
        "endedAt": None,
        "usage": {},
        "usageUnreportedCalls": 0,
        "modelCallCount": 0,
        "failedModelCallCount": 0,
        "requestAttemptCount": 0,
        "retryCount": 0,
        "passes": [],
        "modelCalls": [],
        "retries": [],
    }


__all__ = [
    "Coverage",
    "TraceModelCall",
    "TracePass",
    "TraceRequestAttempt",
    "TraceRetry",
    "TraceSessionLevel",
    "TraceTurn",
    "TraceUnassignedPass",
    "TraceUsage",
    "TurnTrace",
    "scan_turn_trace",
    "trace_payload_with_registry",
]
