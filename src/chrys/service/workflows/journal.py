# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Ordered lifecycle publisher: a record is written and acked, then its live event goes out with that ``seq``.

One lock serialises append + publish, so the bus sees events in exactly the
sequence order of ``events.jsonl``. Store payloads are bounded (the writer's
line budget) and use ``node``/``activation`` keys; the live events carry the
rich, unbounded fields (manifest, resolution snapshot) that ``run.json`` holds.

A failed append freezes the journal: every later record raises the same
:class:`WorkflowStorageFailed`, and :meth:`finish` then publishes the one
out-of-band terminal — after the lock has drained every already-published
prefix — with ``seq=None``, ``last_written_seq`` and ``degraded=True``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, Final

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    Event,
    WorkflowLoopIteration,
    WorkflowNodeAnswered,
    WorkflowNodeAskUser,
    WorkflowNodeOutput,
    WorkflowNodeStateChanged,
    WorkflowOutputSummary,
    WorkflowRunFinished,
    WorkflowRunNotice,
)
from chrys.foundation.models.ask_user import AskUserAnswer, AskUserQuestion
from chrys.service.workflows.asks import answer_summary, ask_summary
from chrys.service.workflows.outcomes import RunOutcome
from chrys.service.workflows.records import run_started_event
from chrys.service.workflows.scheduler import AttemptRef
from chrys.service.workflows.store import RunRecord, WorkflowRunStore, WorkflowStorageFailed

SUMMARY_MAX_CHARS: Final = 512
"""Display summaries and error messages are cut here; every full text lives in ``nodes/``."""

TERMINAL_RECORD_BUDGET_BYTES: Final = 3072
"""The terminal record's payload stays under this so the envelope around it fits the line budget."""


def summarize(text: str, limit: int = SUMMARY_MAX_CHARS) -> str:
    """The leading part of *text* that fits the line budget, marked when cut."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


class WorkflowJournal:
    """Write-then-publish adapter over one run store."""

    def __init__(self, store: WorkflowRunStore, bus: EventBus | None, *, session_id: str | None) -> None:
        self._store = store
        self._bus = bus
        self._session_id = session_id
        self._run_id = store.header.run_id
        self._lock = asyncio.Lock()
        self._holder: asyncio.Task[Any] | None = None
        self._failure: WorkflowStorageFailed | None = None
        self._finished = False

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def store(self) -> WorkflowRunStore:
        return self._store

    @property
    def storage_failed(self) -> bool:
        return self._failure is not None

    def holds(self, task: asyncio.Task[Any] | None) -> bool:
        """True when *task* is publishing under the journal's lock.

        Every run event goes out from there, so a subscriber to one is on the task holding it; waiting for
        the run from that task would wait for the lock the run needs back.
        """
        return task is not None and task is self._holder

    # -- lifecycle records ---------------------------------------------------

    async def run_started(self) -> int:
        return await self._record(
            RunRecord.RUN_STARTED,
            {},
            lambda seq: run_started_event(
                self._store.header.to_dict(), self._store.spec.to_dict(), self._store.input_text, seq=seq
            ),
        )

    async def node_state(
        self,
        ref: AttemptRef,
        state: str,
        *,
        invocation_id: str = "",
        error: str = "",
        error_class: str = "",
        durable: bool = False,
        iteration: int = 0,
        failure_phase: str = "",
    ) -> int:
        message = summarize(error)
        record: dict[str, Any] = {
            **_ref_record(ref),
            "state": state,
            "iteration": iteration,
            "failure_phase": failure_phase,
        }
        if invocation_id:
            record["invocation"] = invocation_id
        if message:
            record["error"] = message
        if error_class:
            record["error_class"] = error_class
        return await self._record(
            RunRecord.NODE_STATE,
            record,
            lambda seq: WorkflowNodeStateChanged(
                seq=seq,
                state=state,
                iteration=iteration,
                failure_phase=failure_phase,
                invocation_id=invocation_id,
                error=message,
                error_class=error_class,
                session_id=self._session_id,
                **_ref_fields(ref),
            ),
            durable=durable,
        )

    async def node_output(self, ref: AttemptRef, kind: str, ordinal: int, text: str) -> int:
        summary = summarize(text)
        return await self._record(
            RunRecord.NODE_OUTPUT,
            {**_ref_record(ref), "kind": kind, "ordinal": ordinal, "summary": summary},
            lambda seq: WorkflowNodeOutput(
                seq=seq,
                kind=kind,
                ordinal=ordinal,
                summary_text=summary,
                session_id=self._session_id,
                **_ref_fields(ref),
            ),
        )

    async def node_ask(self, ref: AttemptRef, request_id: str, questions: tuple[AskUserQuestion, ...]) -> int:
        summary = summarize(ask_summary(questions))
        return await self._record(
            RunRecord.NODE_ASK,
            {**_ref_record(ref), "request": request_id, "prompt": summary},
            lambda seq: WorkflowNodeAskUser(
                seq=seq, request_id=request_id, questions=questions, session_id=self._session_id, **_ref_fields(ref)
            ),
        )

    async def node_answer(
        self,
        ref: AttemptRef,
        request_id: str,
        questions: tuple[AskUserQuestion, ...],
        answers: tuple[AskUserAnswer, ...],
    ) -> int:
        summary = summarize(answer_summary(questions, answers))
        return await self._record(
            RunRecord.NODE_ANSWER,
            {**_ref_record(ref), "request": request_id, "answer": summary},
            lambda seq: WorkflowNodeAnswered(
                seq=seq, request_id=request_id, answer=summary, session_id=self._session_id, **_ref_fields(ref)
            ),
            durable=True,
        )

    async def loop_iteration(self, ref: AttemptRef, iteration: int, verdict: str) -> int:
        return await self._record(
            RunRecord.LOOP_ITERATION,
            {**_ref_record(ref), "iteration": iteration, "verdict": verdict},
            lambda seq: WorkflowLoopIteration(
                seq=seq,
                run_id=ref.run_id,
                loop_id=ref.node_id,
                activation_id=ref.activation_id,
                attempt=ref.attempt,
                iteration=iteration,
                verdict=verdict,
                session_id=self._session_id,
            ),
        )

    async def run_notice(self, node_id: str, activation_id: str, code: str, message: str) -> int:
        summary = summarize(message)
        return await self._record(
            RunRecord.RUN_NOTICE,
            {"node": node_id, "activation": activation_id, "attempt": 0, "code": code, "message": summary},
            lambda seq: WorkflowRunNotice(
                seq=seq,
                run_id=self._run_id,
                node_id=node_id,
                activation_id=activation_id,
                attempt=0,
                code=code,
                message=summary,
                session_id=self._session_id,
            ),
        )

    async def retry_key(self, request_id: str, ref: AttemptRef) -> int:
        """The manual-retry dedupe key, written before the retried attempt's side effects; no live event."""
        return await self._record(RunRecord.RETRY_KEY, {**_ref_record(ref), "request": request_id}, None)

    async def finish(
        self,
        outcome: RunOutcome,
        *,
        outputs: Sequence[WorkflowOutputSummary] = (),
        duration: float = 0.0,
        node_id: str = "",
        error: str = "",
        reason: str = "",
    ) -> int | None:
        """Record and publish the terminal exactly once; later calls are no-ops returning ``None``.

        After a storage failure (earlier, or in this very write) the outcome
        becomes ``storage_failed`` and the terminal is published out of band.
        """
        async with self._held():
            if self._finished:
                return None
            self._finished = True
            message = summarize(error)
            live: dict[str, Any] = {
                "run_id": self._run_id,
                "outputs": list(outputs),
                "duration": duration,
                "node_id": node_id,
                "error": message,
                "reason": reason,
                "session_id": self._session_id,
            }
            if self._failure is None:
                record: dict[str, Any] = {
                    "outputs": [
                        {"node": item.node_id, "activation": item.activation_id, "attempt": item.attempt}
                        for item in outputs
                    ],
                    "duration": duration,
                }
                if node_id:
                    record["node"] = node_id
                if message:
                    record["error"] = message
                if reason:
                    record["reason"] = reason
                output_index = record["outputs"]
                if len(json.dumps(record, ensure_ascii=False).encode("utf-8")) > TERMINAL_RECORD_BUDGET_BYTES:
                    # A wide fan-out of outputs would push the commit record over the line budget and
                    # turn a completed run into storage_failed; exact identities live in the output index.
                    record["outputs"] = []
                    record["outputs_omitted"] = len(outputs)
                try:
                    await self._store.write_outputs(output_index)
                    seq = await self._store.finish(outcome.value, record)
                except WorkflowStorageFailed as exc:
                    self._failure = exc
                else:
                    await self._publish(WorkflowRunFinished(seq=seq, outcome=outcome.value, **live))
                    return seq
            await self._publish(
                WorkflowRunFinished(
                    seq=None,
                    outcome=RunOutcome.STORAGE_FAILED.value,
                    last_written_seq=self._store.last_written_seq,
                    degraded=True,
                    **live,
                )
            )
            return None

    # -- internals -----------------------------------------------------------

    async def _record(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        live: Callable[[int], Event] | None,
        *,
        durable: bool = False,
    ) -> int:
        async with self._held():
            if self._failure is not None:
                raise self._failure
            if self._finished:
                raise RuntimeError(f"{event_type} after the run terminal")
            try:
                seq = await self._store.append(event_type, payload, durable=durable)
            except WorkflowStorageFailed as exc:
                self._failure = exc
                raise
            if live is not None:
                await self._publish(live(seq))
            return seq

    @contextlib.asynccontextmanager
    async def _held(self) -> AsyncIterator[None]:
        """The lock, with the task holding it on record for :meth:`holds`."""
        async with self._lock:
            self._holder = asyncio.current_task()
            try:
                yield
            finally:
                self._holder = None

    async def _publish(self, event: Event) -> None:
        if self._bus is not None:
            await self._bus.publish(event)


def _ref_record(ref: AttemptRef) -> dict[str, Any]:
    return {"node": ref.node_id, "activation": ref.activation_id, "attempt": ref.attempt}


def _ref_fields(ref: AttemptRef) -> dict[str, Any]:
    return {"run_id": ref.run_id, "node_id": ref.node_id, "activation_id": ref.activation_id, "attempt": ref.attempt}
