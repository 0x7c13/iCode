# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The session's one place where a hook's ``ask`` waits for the user.

Every ask, from any entry (new input, injected message, retry note, a tool
call of the main agent, a sub-agent or a workflow node), goes through
:meth:`HookConfirmationService.ask` the same way:

1. Refused at once (``unavailable``) when the session cannot ask the user,
   and ``invalidated`` when the operation already ended.
2. Registered in the pending table before the request is published: the
   response subscription exists from ``start``, so an answer published while
   the request is still being published is not lost.
3. Published as ``HookConfirmationRequest``; a frontend handler claims it
   synchronously and shows it on a task of its own. Unclaimed once the publish
   returns, it ends ``unavailable``: nothing would ever answer.
4. Ended by the first of: the claiming frontend's answer, the operation's
   invalidation, the configured timeout, the waiting task's cancellation, or
   the service closing.
5. Settled once, in ``_finish``: removed from the table and its claim closed
   before anything is awaited; a frontend still showing it is told with
   ``HookConfirmationWithdrawn``.

Nothing here knows which entry asked; the entry's lifetime reaches the wait
only through ``HookOperation.invalidate``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from chrys.foundation.events.types import HookConfirmationRequest, HookConfirmationResponse, HookConfirmationWithdrawn
from chrys.foundation.models.hook_confirmations import HookArgumentChange, HookConfirmationOutcome
from chrys.foundation.tool_execution_stamp import encode_json_like
from chrys.foundation.trajectory.event_types import WaitCategory
from chrys.foundation.util.once_close import OnceClose, finish_close
from chrys.service.trajectory.waits import WaitOutcome, WaitTrace

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.trajectory.context import TrajectoryContext
    from chrys.service.hooks.operation import HookOperation
    from chrys.service.trajectory.hooks import TrajectoryContextProvider

logger = logging.getLogger(__name__)

_WAIT_OUTCOMES: dict[HookConfirmationOutcome, str] = {
    "approved": WaitOutcome.COMPLETED,
    "declined": WaitOutcome.COMPLETED,
    "timed_out": WaitOutcome.TIMED_OUT,
    "interrupted": WaitOutcome.CANCELLED,
    "unavailable": WaitOutcome.FAILED,
    "invalidated": WaitOutcome.FAILED,
}


@dataclass(frozen=True, slots=True)
class HookConfirmationResult:
    """How one ask ended, and why (a code: ``user``, ``dismissed``, ``unclaimed``, an invalidation reason, …)."""

    outcome: HookConfirmationOutcome
    end_reason: str
    answered: bool = False
    """Ended by the claiming frontend's own answer: it put the question away itself, so it is not withdrawn.

    Set where the answer is taken, never read off ``end_reason`` (display text), so a new way to end
    an ask cannot change who is told.
    """


@dataclass(slots=True)
class _Pending:
    request: HookConfirmationRequest
    operation: HookOperation
    revision: int
    answer: asyncio.Future[HookConfirmationResult]
    wait: WaitTrace | None
    lifecycle: asyncio.Task[None] | None = None


CHANGE_CONTEXT_CHARS = 40
"""How much unchanged text an argument change shows on each side of what changed."""
CHANGE_SPAN_MAX_CHARS = 200
"""How much of the changed text itself an argument change shows."""


def argument_changes(before: dict[str, Any], after: dict[str, Any]) -> tuple[HookArgumentChange, ...]:
    """Each top-level argument whose value differs between *before* and *after*, in *after*'s order then removed ones.

    A long value is cut to the part that changed, with a little text around it, so
    every frontend shows where it changed rather than a beginning both share.
    """
    changes: list[HookArgumentChange] = []
    for name in [*after, *(name for name in before if name not in after)]:
        old = _json_text(before[name]) if name in before else None
        new = _json_text(after[name]) if name in after else None
        if old != new:
            if old is not None and new is not None:
                old, new = _changed_excerpts(old, new)
            elif old is not None:
                old = _excerpt(old, 0, len(old))
            elif new is not None:
                new = _excerpt(new, 0, len(new))
            changes.append(HookArgumentChange(name=str(name), before=old, after=new))
    return tuple(changes)


def _changed_excerpts(old: str, new: str) -> tuple[str, str]:
    """*old* and *new* cut to where they differ: past their common start and end."""
    limit = min(len(old), len(new))
    start = 0
    while start < limit and old[start] == new[start]:
        start += 1
    end = 0
    while end < limit - start and old[-1 - end] == new[-1 - end]:
        end += 1
    return _excerpt(old, start, len(old) - end), _excerpt(new, start, len(new) - end)


def _excerpt(text: str, start: int, end: int) -> str:
    """``text[start:end]`` (the changed part, up to its first ``CHANGE_SPAN_MAX_CHARS``) with
    ``CHANGE_CONTEXT_CHARS`` around it; ``…`` marks what is left out."""
    span_end = min(end, start + CHANGE_SPAN_MAX_CHARS)
    left = max(0, start - CHANGE_CONTEXT_CHARS)
    right = min(len(text), span_end + CHANGE_CONTEXT_CHARS)
    cut_inside = span_end < end
    head = "…" if left > 0 else ""
    if cut_inside:
        return f"{head}{text[left:span_end]}…"
    return f"{head}{text[left:right]}{'…' if right < len(text) else ''}"


def _json_text(value: Any) -> str:
    """*value* as the payload writes it; a placeholder naming its type when it has no JSON form."""
    try:
        return encode_json_like(value)
    except Exception:
        return object.__repr__(value)


class HookConfirmationService:
    """An engine's confirmations: what is pending, and the one way each ends.

    One per engine (one event bus), attached to every hook manager the
    engine builds, so the main agent, its sub-agents and its workflow nodes
    share it. Each request carries its operation's session id, and only an
    answer from that session counts. With ``enabled=False`` (a headless run)
    every ask is ``unavailable``; it is the same service either way.

    Each ask is owned by one lifecycle task (``_lifecycle``): it shows the
    request, watches for the end, and tells a frontend still showing it. The
    asking task only awaits it, and ``close`` drains every one, so nothing an
    ask started outlives it.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        enabled: bool,
        timeout_seconds: float | None = None,
    ) -> None:
        self._bus = bus
        self._enabled = enabled
        self._timeout_seconds = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
        self._pending: dict[str, _Pending] = {}
        self._lifecycles: set[asyncio.Task[None]] = set()
        self._started = False
        self._closed = False
        self._close_once = OnceClose(self._close)

    @property
    def timeout_seconds(self) -> float | None:
        """How long an ask waits for an answer; None for no limit."""
        return self._timeout_seconds

    def available(self) -> bool:
        """Whether an ask can reach a user now: what a hook's payload says as ``confirmation_available``."""
        return self._enabled and self._started and not self._closed

    async def start(self) -> None:
        """Subscribe to responses. Asks before this are ``unavailable``."""
        if self._started or self._closed:
            return
        await self._bus.subscribe(HookConfirmationResponse, self._on_response)
        self._started = True

    async def close(self) -> None:
        """End every pending ask ``unavailable``, wait until each is wound down, and stop taking answers.

        Asks after this are ``unavailable``. A frontend still showing a request
        is told it ended before this returns. Runs once, as one task every
        caller drains: a caller's cancellation never cuts it short.
        """
        await self._close_once()

    async def _close(self) -> None:
        self._closed = True
        for pending in list(self._pending.values()):
            self._settle(pending, HookConfirmationResult("unavailable", "closed"))
        if self._lifecycles:
            await asyncio.gather(*self._lifecycles, return_exceptions=True)
        if self._started:
            await self._bus.unsubscribe(HookConfirmationResponse, self._on_response)

    async def ask(
        self,
        operation: HookOperation,
        *,
        run_id: str,
        hook_id: str,
        hook_source: str,
        reason: str,
        trajectory_context: TrajectoryContext | None = None,
        trajectory_provider: TrajectoryContextProvider | None = None,
    ) -> HookConfirmationResult:
        """Ask the user about *operation*'s current version for hook *hook_id*; how the ask ended.

        Raises ``CancelledError`` when the waiting task is cancelled, after
        the ask is settled ``interrupted`` and wound down.
        """
        if not operation.pending:
            return HookConfirmationResult("invalidated", operation.end_reason or operation.state)
        if not self.available():
            return HookConfirmationResult("unavailable", "cannot_ask")
        shown = operation.arguments()
        confirmed = operation.confirmed_arguments
        changes = argument_changes(confirmed, shown) if confirmed is not None else ()
        request = HookConfirmationRequest(
            session_id=operation.session_id,
            request_id=uuid4().hex,
            operation_id=operation.operation_id,
            revision=operation.revision,
            run_id=run_id,
            origin=operation.origin,
            caller_name=operation.caller_name,
            hook_id=hook_id,
            hook_source=hook_source,
            reason=reason,
            subject=operation.subject,
            tool_name=operation.tool_name,
            tool_kind=operation.tool_kind,
            arguments=operation.arguments(),
            asked_again=bool(changes),
            changed_arguments=changes,
            input_text=operation.input_text,
        )
        pending = _Pending(
            request=request,
            operation=operation,
            revision=operation.revision,
            answer=asyncio.get_running_loop().create_future(),
            wait=WaitTrace.open(
                WaitCategory.HOOK_CONFIRMATION,
                target_operation_id=run_id,
                context=trajectory_context,
                provider=trajectory_provider,
            ),
        )
        # Registered before anything is published: an answer published during the publish finds it.
        self._pending[request.request_id] = pending
        lifecycle = asyncio.create_task(self._lifecycle(pending), name=f"chrys.hooks.confirmation.{request.request_id}")
        pending.lifecycle = lifecycle
        self._lifecycles.add(lifecycle)
        lifecycle.add_done_callback(self._lifecycles.discard)
        try:
            await asyncio.shield(lifecycle)
        except asyncio.CancelledError:
            self._settle(pending, HookConfirmationResult("interrupted", "cancelled"))
            # Wound down before the cancel goes on, however often the waiter is cancelled meanwhile:
            # the frontend is told, the span closed.
            await finish_close(lifecycle)
            raise
        result = pending.answer.result()
        if result.outcome == "approved":
            operation.note_confirmed(shown)
        return result

    async def _lifecycle(self, pending: _Pending) -> None:
        """Show *pending*, end it on the first of answer / invalidation / timeout / nobody claiming, wind it down."""
        started = time.monotonic()
        present = asyncio.create_task(self._present(pending))
        ended = asyncio.ensure_future(pending.operation.wait_invalidated())
        deadline = started + self._timeout_seconds if self._timeout_seconds is not None else None
        try:
            while not pending.answer.done():
                watched: set[asyncio.Future[Any]] = {pending.answer, ended}
                if not present.done():
                    watched.add(present)
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                done, _ = await asyncio.wait(watched, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                if pending.answer.done():
                    break
                if ended in done:
                    self._settle(pending, HookConfirmationResult("invalidated", pending.operation.end_reason))
                elif not done:
                    self._settle(pending, HookConfirmationResult("timed_out", "timeout"))
                elif present in done and not pending.request.claim.claimed:
                    # Published, and no frontend took it: nothing would ever answer.
                    self._settle(pending, HookConfirmationResult("unavailable", "unclaimed"))
        except BaseException:
            self._settle(pending, HookConfirmationResult("unavailable", "error"))
            raise
        finally:
            ended.cancel()
            if not present.done():
                # Still publishing (a handler holding it): the request is over, its showing is moot.
                present.cancel()
            with contextlib.suppress(BaseException):
                await present
        await self._wind_down(pending, pending.answer.result(), started)

    async def _present(self, pending: _Pending) -> None:
        """Open the wait span and publish the request: the step a slow subscriber can hold up."""
        if pending.wait is not None:
            await pending.wait.started()
        await self._bus.publish(pending.request)

    def _settle(self, pending: _Pending, result: HookConfirmationResult) -> bool:
        """End *pending* with *result* unless it already ended; synchronous, so it ends once."""
        if pending.answer.done():
            return False
        pending.request.claim.end()
        self._pending.pop(pending.request.request_id, None)
        pending.answer.set_result(result)
        return True

    async def _wind_down(self, pending: _Pending, result: HookConfirmationResult, started: float) -> None:
        """After *pending* settled: tell a frontend still showing it, and close its wait span."""
        if pending.request.claim.claimed and not result.answered:
            withdrawn = HookConfirmationWithdrawn(
                session_id=pending.request.session_id, request_id=pending.request.request_id, outcome=result.outcome
            )
            try:
                await self._bus.publish(withdrawn)
            except Exception:
                logger.exception("Publishing a withdrawn hook confirmation failed")
        if pending.wait is not None:
            with contextlib.suppress(Exception):
                await pending.wait.finished(
                    outcome=_WAIT_OUTCOMES[result.outcome], result=result.outcome, end_reason=result.end_reason
                )
        logger.debug(
            "Hook confirmation %s ended %s (%s) after %.1fs",
            pending.request.request_id,
            result.outcome,
            result.end_reason,
            time.monotonic() - started,
        )

    async def _on_response(self, event: HookConfirmationResponse) -> None:
        pending = self._pending.get(event.request_id)
        if pending is None or event.session_id != pending.request.session_id:
            return
        self._settle(pending, replace(self._result_of(pending, event), answered=True))

    @staticmethod
    def _result_of(pending: _Pending, event: HookConfirmationResponse) -> HookConfirmationResult:
        match event.answer:
            case "approved":
                if not pending.operation.pending:
                    return HookConfirmationResult("invalidated", pending.operation.end_reason)
                if pending.operation.revision != pending.revision:
                    return HookConfirmationResult("invalidated", "revision_changed")
                return HookConfirmationResult("approved", "user")
            case "declined":
                return HookConfirmationResult("declined", "user")
            case "dismissed":
                return HookConfirmationResult("declined", "dismissed")
            case _:
                return HookConfirmationResult("unavailable", "frontend_unavailable")
