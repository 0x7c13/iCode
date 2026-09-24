# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Hook orchestrator — single entry point for every wire-up site.

The manager owns:

* The parsed :class:`HooksFile` (matched + filtered per dispatch).
* The :class:`HookRunner` (subprocess execution).
* The :class:`Outbox` (durable jobs).
* A set of in-flight non-detached tasks split into "this turn" /
  "this session" / "background" so drains can target only awaited work.

Callers fire events through :meth:`fire`; the return value is a
:class:`HookDecision` that aggregates the event-supported outputs from
blocking hooks.  Pure observer events can ignore the return value.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chrys.foundation.trajectory.context import TrajectoryContext
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.matcher import matches
from chrys.service.hooks.outbox import Outbox, OutboxJob
from chrys.service.hooks.runner import HookResult, HookRunner
from chrys.service.hooks.schema import HookConfig, HookDecision, HooksFile, MergedHooksFile
from chrys.service.trajectory.hooks import HookOperationTrace, HookOutcome, TrajectoryContextProvider

logger = logging.getLogger(__name__)
_GATED_EVENTS = frozenset({HookEvent.BEFORE_TOOL_CALL, HookEvent.USER_PROMPT_SUBMIT})
_ARGS_OVERRIDE_EVENTS = frozenset({HookEvent.BEFORE_TOOL_CALL})
_SYSTEM_REMINDER_EVENTS = frozenset({HookEvent.USER_PROMPT_SUBMIT})
_EXTRA_CONTEXT_EVENTS = frozenset({HookEvent.AFTER_TOOL_CALL, HookEvent.TOOL_ERROR})


# ---------------------------------------------------------------------------
# Tracked in-flight task
# ---------------------------------------------------------------------------


class _Tracked:
    """Bookkeeping wrapper around an in-flight hook task.

    ``scope`` lets :meth:`HookManager.drain_turn` filter to just turn-
    scoped tasks without touching background or session-scoped ones.
    """

    __slots__ = ("hook", "job", "operation_id", "scope", "task")

    def __init__(
        self,
        *,
        hook: HookConfig,
        task: asyncio.Task[HookResult],
        scope: str,
        job: OutboxJob | None,
        operation_id: str | None,
    ) -> None:
        self.hook = hook
        self.task = task
        self.scope = scope  # "turn" | "session" | "background"
        self.job = job
        self.operation_id = operation_id


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class HookManager:
    """Async orchestrator for all hook activity.

    Construct once per session when a hook config file exists.  The
    manager is dispatch-no-op when the loaded :class:`HooksFile` carries
    no hooks, but construction still creates runtime directories for the
    durable outbox.
    """

    def __init__(
        self,
        *,
        file: HooksFile | MergedHooksFile,
        hooks_dir: Path,
    ) -> None:
        if isinstance(file, HooksFile):
            file = MergedHooksFile.from_single(file)
        self._file = file
        self._hook_sources = _hook_sources(file)
        self._hook_source_paths = _hook_source_paths(file)
        self._hooks_by_id = _hooks_by_id(file.hooks)
        self._hooks_by_recovery_key = _hooks_by_recovery_key(file.hooks, self._hook_sources, self._hook_source_paths)
        self._hooks_dir = hooks_dir
        self._runner = HookRunner(hooks_dir=hooks_dir)
        self._outbox = Outbox(hooks_dir / "outbox")
        # Cap concurrency across non-detached subprocesses.
        self._sem = asyncio.Semaphore(max(1, file.settings.max_parallel_hooks))
        # Open non-detached tasks by scope.  Detached hooks are handed
        # off to a worker process and are not tracked here.
        self._inflight: list[_Tracked] = []
        self._active_turn_drain_operation_ids: tuple[str, ...] = ()
        # Listeners registered by middleware for "out-of-band" results
        # (e.g. ``after_tool_call`` decisions that arrive after the
        # tool already returned to the model).  Indexed by event for
        # cheap dispatch.
        self._closed = False
        self._session_drain_generation = 0
        # Where hook runs are recorded when no model run binds an ambient
        # trajectory scope (session / turn lifecycle hooks); the engine
        # installs its recorder's resolver.
        self.trajectory_context_provider: TrajectoryContextProvider | None = None

    @property
    def hooks_dir(self) -> Path:
        return self._hooks_dir

    @property
    def file(self) -> MergedHooksFile:
        return self._file

    def sources(self) -> list[str]:
        """List of source paths included in the manager's config.

        Empty when no source had a path (e.g. a synthetic empty
        config). Populated by :class:`MergedHooksFile` with both
        project and global paths in load order. Used for log lines
        and the future ``/hooks`` TUI screen.
        """
        return list(self._file.sources)

    def has_hooks_for(self, event: HookEvent) -> bool:
        """Cheap fast-path check.

        Wire-up sites guard with this so we don't even build payloads
        when no hooks would match.  Saves a dict allocation per tool
        call when hooks are unused — and "no hooks at all" is the
        common case.
        """
        return any(h.enabled and h.event == event for h in self._file.hooks)

    # ── dispatch ─────────────────────────────────────────────────────

    async def fire(
        self,
        event: HookEvent,
        payload: dict[str, Any],
        *,
        scope: str = "turn",
        target_operation_id: str | None = None,
        trajectory_context: TrajectoryContext | None = None,
    ) -> HookDecision:
        """Fire every matching hook for *event*.

        Args:
            event: The lifecycle point.
            payload: Event-specific data; the manager adds ``event``,
                ``schema``, and ``timestamp`` envelope fields.
            scope: ``"turn"`` (drained at end of turn), ``"session"``
                (drained at shutdown), or ``"detached"`` (never
                awaited).  Caller picks the appropriate scope for the
                event.
            target_operation_id: The trajectory operation the hooks act
                on (the tool operation for tool events), recorded on
                every ``hook.operation.*`` this dispatch emits.
            trajectory_context: Explicit recording owner for blocking hooks.
                Non-blocking turn-scoped hooks deliberately retain the normal
                provider context so response-fence membership remains resolvable.

        Returns:
            Aggregated :class:`HookDecision`.  For events with no
            matching hooks, an empty (allow / no-op) decision.
        """
        session_drain_generation = self._session_drain_generation
        if self._closed:
            return HookDecision()
        candidates = [h for h in self._file.hooks if h.event == event and h.enabled]
        if not candidates:
            return HookDecision()
        envelope = _build_envelope(event, payload)
        decision = HookDecision()

        # Two passes: blocking hooks run sequentially so the first
        # ``block`` short-circuits; non-blocking hooks then spawn in
        # parallel.  Sequential blocking is intentional — users
        # ordering hooks in the file should be able to reason about
        # which fires first.  Blocking ``modify`` decisions update the
        # envelope immediately so later blocking hooks can validate the
        # rewritten arguments.
        blocking: list[HookConfig] = []
        nonblocking: list[HookConfig] = []
        for hook in candidates:
            if hook.execution.mode == "blocking":
                blocking.append(hook)
            else:
                nonblocking.append(hook)

        for hook in blocking:
            if self._closed or self._session_drain_generation != session_drain_generation:
                return decision
            if not matches(hook, envelope):
                continue
            trace = self._open_trace(target_operation_id, context=trajectory_context)
            try:
                # Inside the block that closes the span: the start marker
                # awaits its write ack, and an interrupt landing there would
                # otherwise leave the hook operation open forever.
                if trace is not None:
                    await trace.started(
                        hook_id=hook.id,
                        hook_event=str(event),
                        execution_mode=hook.execution.mode,
                        detach=hook.execution.detach,
                        delivery=hook.execution.delivery,
                        scope=scope,
                        drain_scope=scope,
                    )
                if self._closed or self._session_drain_generation != session_drain_generation:
                    if trace is not None:
                        trace.finished_soon(outcome=HookOutcome.ABANDONED)
                    return decision
                result = await self._run_blocking(hook, envelope)
            except asyncio.CancelledError:
                if trace is not None:
                    trace.finished_soon(outcome=HookOutcome.CANCELLED)
                raise
            except Exception as exc:
                logger.exception("Blocking hook %r failed before producing a result", hook.id)
                result = HookResult(hook_id=hook.id, exit_code=None, stderr=str(exc))
            blocked_before = decision.blocked
            override_before = decision.args_override
            _merge_blocking_result(decision, hook, result, event=event)
            if trace is not None:
                await trace.finished(
                    outcome=_hook_outcome(result, blocked=decision.blocked and not blocked_before),
                    arguments_modified=decision.args_override != override_before,
                    exit_code=result.exit_code,
                    timed_out=result.timed_out,
                )
            if decision.blocked:
                # Cancel any planned non-blocking hooks for this event —
                # nothing else will execute on this dispatch.
                return decision
            if decision.args_override is not None:
                _apply_args_override_to_envelope(envelope, decision.args_override)

        if self._closed or self._session_drain_generation != session_drain_generation:
            return decision
        for hook in nonblocking:
            if not matches(hook, envelope):
                continue
            try:
                await self._spawn_nonblocking(
                    hook,
                    envelope,
                    scope=scope,
                    session_drain_generation=session_drain_generation,
                    target_operation_id=target_operation_id,
                )
            except Exception as exc:
                _log_nonblocking_spawn_failure(hook, exc)

        return decision

    def _open_trace(
        self,
        target_operation_id: str | None,
        *,
        context: TrajectoryContext | None = None,
    ) -> HookOperationTrace | None:
        return HookOperationTrace.open(
            target_operation_id=target_operation_id,
            context=context,
            provider=self.trajectory_context_provider,
        )

    # ── drains ───────────────────────────────────────────────────────

    @property
    def active_turn_drain_operation_ids(self) -> tuple[str, ...]:
        """Hook operations captured by the latest turn drain fence."""
        return self._active_turn_drain_operation_ids

    async def drain_turn(self) -> tuple[str, ...]:
        """Await every ``async`` hook spawned in this turn.

        ``fire_and_forget`` tasks are observed for logging/outbox updates
        but skipped by drains.  ``detached`` tasks are already independent
        of chrys.  Errors are logged at WARN, not raised, so a flaky hook
        cannot fail the turn that just succeeded.
        """
        operation_ids = tuple(
            tracked.operation_id
            for tracked in self._inflight
            if tracked.scope == "turn" and not tracked.task.done() and tracked.operation_id is not None
        )
        self._active_turn_drain_operation_ids = operation_ids
        await self._drain(scopes={"turn"}, timeout=None)
        return operation_ids

    async def drain_session(self, *, close: bool = True) -> None:
        """Await every still-running ``async`` hook (turn + session scopes) up to
        ``shutdown_grace_seconds``.

        On grace expiry surviving tasks are cancelled (their
        subprocesses get killed by ``managed_subprocess``).  Detached
        subprocesses are never touched. Tasks outside the drained scopes keep
        running and reporting results. The recorder settles unfinished traces
        when its log closes; replacing this manager does not end observation.

        ``close=False`` waits the same way but leaves the manager open:
        used when the live session ends before the engine shuts down
        (deleting the active session), so ``session_end`` hooks finish
        while the session files still exist and later events still fire.
        """
        self._session_drain_generation += 1
        await self._drain(
            scopes={"turn", "session"},
            timeout=self._file.settings.shutdown_grace_seconds,
        )
        if close:
            self._closed = True

    async def _drain(self, *, scopes: set[str], timeout: float | None) -> None:
        tracked = [t for t in self._inflight if t.scope in scopes]
        targets = [t for t in tracked if not t.task.done()]
        if targets:
            tasks = [t.task for t in targets]
            try:
                if timeout is None:
                    await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=timeout,
                    )
            except TimeoutError:
                logger.warning(
                    "Hook drain timed out after %.1fs; cancelling %d task(s)",
                    timeout,
                    sum(1 for t in tasks if not t.done()),
                )
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Best-effort await of cancellation so subprocess cleanup runs.
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.gather(*tasks, return_exceptions=True)

        for tracked_task in tracked:
            if not tracked_task.task.done():
                continue
            try:
                tracked_task.task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Hook task %r failed outside normal result handling", tracked_task.hook.id)

        # Reap done tasks of these scopes, including hooks that completed
        # before this drain call started.
        self._inflight = [t for t in self._inflight if not (t.scope in scopes and t.task.done())]

    # ── outbox recovery ──────────────────────────────────────────────

    async def recover_outbox(self) -> int:
        """Scan ``outbox/pending/`` and retry stale jobs.

        Returns the number of jobs queued for retry.  Called once at
        engine startup.  Durable detached jobs are recovered through the
        same detached worker path as fresh dispatches; semantics remain
        at-least-once.
        """
        stale = self._outbox.list_stale_pending(
            retry_age_seconds=self._file.settings.outbox_retry_age_seconds,
        )
        if not stale:
            return 0

        max_retries = self._file.settings.outbox_max_retries
        queued = 0
        for job in stale:
            if job.retries >= max_retries:
                logger.warning(
                    "Outbox job %s (hook=%s) exceeded max_retries=%d; moving to failed/",
                    job.job_id,
                    job.hook_id,
                    max_retries,
                )
                with contextlib.suppress(OSError):
                    self._outbox.force_fail(job, reason=f"retries>={max_retries}")
                continue
            hook, missing_reason = self._resolve_recovery_hook(job)
            if hook is None:
                logger.warning(
                    "Outbox job %s references %s; moving to failed/",
                    job.job_id,
                    missing_reason,
                )
                with contextlib.suppress(OSError):
                    self._outbox.force_fail(job, reason=missing_reason)
                continue
            if not hook.enabled:
                disabled_reason = f"disabled hook {job.hook_id!r}"
                logger.warning(
                    "Outbox job %s references %s; moving to failed/",
                    job.job_id,
                    disabled_reason,
                )
                with contextlib.suppress(OSError):
                    self._outbox.force_fail(job, reason=disabled_reason)
                continue
            # Schedule a retry with the original payload.  We reuse the
            # same job_id so done/failed can be correlated back to the
            # original incident if anyone goes looking.
            await self._spawn_nonblocking(
                hook,
                job.payload,
                scope="session",
                session_drain_generation=self._session_drain_generation,
                existing_job=job,
            )
            queued += 1
        return queued

    def _resolve_recovery_hook(self, job: OutboxJob) -> tuple[HookConfig | None, str]:
        """Return the hook a pending outbox job should retry, plus a failure label."""
        if job.hook_source:
            hook = self._hooks_by_recovery_key.get((job.hook_id, job.hook_source, job.hook_source_path))
            if hook is None and not job.hook_source_path:
                hook = self._resolve_legacy_recovery_hook_by_source(job)
            if hook is None:
                source_label = job.hook_source
                if job.hook_source_path:
                    source_label = f"{job.hook_source!r} source at {job.hook_source_path!r}"
                else:
                    source_label = f"{job.hook_source!r} source"
                return None, f"missing hook {job.hook_id!r} from {source_label}"
            return hook, ""

        matches = self._hooks_by_id.get(job.hook_id, [])
        if len(matches) > 1:
            return None, f"ambiguous legacy hook id {job.hook_id!r} with no source marker"
        if not matches:
            return None, f"missing hook {job.hook_id!r}"
        return matches[0], ""

    def _resolve_legacy_recovery_hook_by_source(self, job: OutboxJob) -> HookConfig | None:
        """Resolve pre-source-path jobs that only recorded project/global."""
        matches = [
            hook
            for hook in self._hooks_by_id.get(job.hook_id, [])
            if self._hook_sources.get(id(hook), "") == job.hook_source
        ]
        if len(matches) != 1:
            return None
        return matches[0]

    # ── internals ────────────────────────────────────────────────────

    async def _run_blocking(self, hook: HookConfig, payload: dict[str, Any]) -> HookResult:
        async with self._sem:
            return await self._runner.run_and_wait(hook, payload)

    async def _spawn_nonblocking(
        self,
        hook: HookConfig,
        payload: dict[str, Any],
        *,
        scope: str,
        session_drain_generation: int,
        existing_job: OutboxJob | None = None,
        target_operation_id: str | None = None,
    ) -> None:
        if self._closed or self._session_drain_generation != session_drain_generation:
            return
        job = existing_job
        if hook.execution.delivery == "durable" and job is None:
            job = self._outbox.write_pending(
                hook_id=hook.id,
                event=str(hook.event),
                payload=payload,
                hook_source=self._hook_sources.get(id(hook), ""),
                hook_source_path=self._hook_source_paths.get(id(hook), ""),
            )
        drain_scope: str | None
        if hook.execution.detach:
            drain_scope = None
        elif hook.execution.mode == "fire_and_forget":
            drain_scope = "background"
        else:
            drain_scope = scope
        # Async turn-scope hooks must stay in the active turn: drain_turn()
        # publishes their ids in the response fence, whose analyzer lookup is
        # intentionally confined to that turn's node membership.
        trace = self._open_trace(target_operation_id)
        if trace is not None:
            try:
                await trace.started(
                    hook_id=hook.id,
                    hook_event=str(hook.event),
                    execution_mode=hook.execution.mode,
                    detach=hook.execution.detach,
                    delivery=hook.execution.delivery,
                    scope=scope,
                    drain_scope=drain_scope,
                )
            except BaseException:
                # Interrupted in the start marker's ack wait: no task exists
                # yet that would close the span, so it is closed here — and
                # the trace drops the terminal by itself if that marker never
                # took a sequence.
                trace.finished_soon(outcome=HookOutcome.CANCELLED)
                raise
        if self._closed or self._session_drain_generation != session_drain_generation:
            if trace is not None:
                trace.finished_soon(outcome=HookOutcome.ABANDONED)
            return

        # async / fire_and_forget — wrap in our internal task so we can
        # observe completion and move outbox entries.
        if hook.execution.mode == "async":
            coro = self._run_async_tracked(hook, payload, job, trace)
            task = asyncio.create_task(coro)
            operation_id = trace.operation_id if trace is not None and trace.start_committed else None
            tracked = _Tracked(hook=hook, task=task, scope=scope, job=job, operation_id=operation_id)
            self._inflight.append(tracked)
            if scope == "detached":
                task.add_done_callback(lambda _task, tracked=tracked: self._reap_background(tracked))
            return

        # fire_and_forget
        if hook.execution.detach:
            try:
                if job is not None:
                    self._outbox.mark_started(job)
                await self._runner.spawn_detached(hook, payload, job=job)
            except asyncio.CancelledError:
                if trace is not None:
                    trace.finished_soon(outcome=HookOutcome.CANCELLED)
                raise
            except Exception as exc:
                _log_nonblocking_spawn_failure(hook, exc)
                if job is not None:
                    with contextlib.suppress(OSError):
                        self._outbox.mark_failed(job, exit_code=None, error=str(exc))
                if trace is not None:
                    await trace.finished(outcome=HookOutcome.SPAWN_FAILED)
                return
            if trace is not None:
                # Handed to the detached worker: the parent never observes
                # its exit, so the recorded span ends at the hand-off.
                await trace.finished(outcome=HookOutcome.DETACHED)
            return

        coro = self._run_async_tracked(hook, payload, job, trace)
        task = asyncio.create_task(coro)
        operation_id = trace.operation_id if trace is not None and trace.start_committed else None
        tracked = _Tracked(hook=hook, task=task, scope="background", job=job, operation_id=operation_id)
        self._inflight.append(tracked)
        task.add_done_callback(lambda _task, tracked=tracked: self._reap_background(tracked))

    def _reap_background(self, tracked: _Tracked) -> None:
        with contextlib.suppress(ValueError):
            self._inflight.remove(tracked)
        try:
            tracked.task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Background hook task %r failed outside normal result handling", tracked.hook.id)

    async def _run_async_tracked(
        self,
        hook: HookConfig,
        payload: dict[str, Any],
        job: OutboxJob | None,
        trace: HookOperationTrace | None = None,
    ) -> HookResult:
        """Wrap ``runner.run_and_wait`` with outbox + semaphore + error logging."""
        try:
            # The slot is taken inside the ``try``: a hook cancelled while it
            # queues behind ``max_parallel_hooks`` others never ran, but its
            # span was opened before the task was created and only this
            # handler ever closes it.
            async with self._sem:
                if job is not None:
                    self._outbox.mark_started(job)
                result = await self._runner.run_and_wait(hook, payload)
        except asyncio.CancelledError:
            if job is not None and hook.execution.delivery == "durable":
                logger.info(
                    "Durable hook %r was cancelled; leaving outbox job %s pending for retry",
                    hook.id,
                    job.job_id,
                )
            if trace is not None:
                trace.finished_soon(outcome=HookOutcome.CANCELLED)
            raise
        except Exception as exc:
            logger.exception("Hook %r raised", hook.id)
            if job is not None:
                with contextlib.suppress(OSError):
                    self._outbox.mark_failed(job, exit_code=None, error=str(exc))
            if trace is not None:
                await trace.finished(outcome=HookOutcome.LAUNCH_ERROR)
            return HookResult(hook_id=hook.id, exit_code=None, stderr=str(exc))
        # Before the span is closed: the hook has already run, and the outbox
        # is what decides whether it runs again. An interrupt landing on the
        # awaited marker below must not leave a finished durable job pending
        # for the next runtime to replay.
        if job is not None:
            if _result_failed(result):
                with contextlib.suppress(OSError):
                    self._outbox.mark_failed(
                        job,
                        exit_code=result.exit_code,
                        error=_result_failure_label(result),
                        stderr_tail=result.stderr,
                    )
            else:
                with contextlib.suppress(OSError):
                    self._outbox.mark_done(job, exit_code=result.exit_code or 0)

        if trace is not None:
            await trace.finished(
                outcome=_hook_outcome(result, blocked=False),
                exit_code=result.exit_code,
                timed_out=result.timed_out,
            )

        if _result_failed(result):
            _log_hook_failure(hook, result)
        return result


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_envelope(event: HookEvent, payload: dict[str, Any]) -> dict[str, Any]:
    """Combine *payload* with the standard envelope fields."""
    envelope = {
        **copy.deepcopy(payload),
        "schema": 1,
        "event": str(event),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return envelope


def _hook_sources(file: MergedHooksFile) -> dict[int, str]:
    """Map hook object identity to its merged configuration layer."""
    sources: dict[int, str] = {}
    if file.project is not None:
        for hook in file.project.hooks:
            sources[id(hook)] = "project"
    if file.global_ is not None:
        for hook in file.global_.hooks:
            sources[id(hook)] = "global"
    return sources


def _hook_source_paths(file: MergedHooksFile) -> dict[int, str]:
    """Map hook object identity to the hooks config file that supplied it."""
    sources: dict[int, str] = {}
    if file.project is not None and file.project.source:
        for hook in file.project.hooks:
            sources[id(hook)] = file.project.source
    if file.global_ is not None and file.global_.source:
        for hook in file.global_.hooks:
            sources[id(hook)] = file.global_.source
    return sources


def _hooks_by_recovery_key(
    hooks: list[HookConfig],
    hook_sources: dict[int, str],
    hook_source_paths: dict[int, str],
) -> dict[tuple[str, str, str], HookConfig]:
    """Index hooks by id plus exact source layer/path for durable outbox retry."""
    keyed: dict[tuple[str, str, str], HookConfig] = {}
    for hook in hooks:
        source = hook_sources.get(id(hook), "")
        source_path = hook_source_paths.get(id(hook), "")
        if not source and not source_path:
            continue
        keyed[(hook.id, source, source_path)] = hook
    return keyed


def _hooks_by_id(hooks: list[HookConfig]) -> dict[str, list[HookConfig]]:
    """Group hooks by id while preserving merged order within each id."""
    grouped: dict[str, list[HookConfig]] = {}
    for hook in hooks:
        grouped.setdefault(hook.id, []).append(hook)
    return grouped


def _apply_args_override_to_envelope(envelope: dict[str, Any], args_override: dict[str, Any]) -> None:
    """Merge blocking hook argument overrides into the current tool payload."""
    tool = envelope.get("tool")
    if not isinstance(tool, dict):
        return
    args = tool.get("args")
    if not isinstance(args, dict):
        args = {}
        tool["args"] = args
    args.update(args_override)


def _merge_blocking_result(
    decision: HookDecision,
    hook: HookConfig,
    result: HookResult,
    *,
    event: HookEvent,
) -> None:
    """Fold one blocking hook's outcome into the running :class:`HookDecision`."""
    on_error = hook.execution.on_error
    failed = _result_failed(result)

    if failed:
        msg_kind = _result_failure_label(result)
        if on_error == "block" and event in _GATED_EVENTS:
            decision.blocked = True
            decision.block_reason = f"hook '{hook.id}' failed ({msg_kind}): {result.stderr.strip()[:200]}"
            return
        if on_error == "warn":
            logger.warning("Hook %r failed (%s) but on_error=warn; continuing", hook.id, msg_kind)
        elif on_error == "block":
            logger.warning(
                "Hook %r failed (%s) but event %s is not gated; treating on_error=block as warn",
                hook.id,
                msg_kind,
                event,
            )
        # A failed hook's decision file may be partial or stale.  Honour
        # only the configured error policy and ignore any action payload.
        return

    action = str(result.decision.get("action") or "allow").lower()
    if action == "block":
        if event in _GATED_EVENTS:
            decision.blocked = True
            reason = str(result.decision.get("reason") or "")
            decision.block_reason = f"hook '{hook.id}' blocked: {reason}" if reason else f"hook '{hook.id}' blocked"
            return
        logger.warning("Hook %r returned action=block for non-gated event %s; ignoring block", hook.id, event)
    if action == "modify" and event in _ARGS_OVERRIDE_EVENTS:
        args_override = result.decision.get("args_override")
        if isinstance(args_override, dict):
            decision.args_override = {**(decision.args_override or {}), **args_override}

    reminder = result.decision.get("system_reminder")
    if event in _SYSTEM_REMINDER_EVENTS and isinstance(reminder, str) and reminder:
        decision.system_reminders.append(reminder)
    extra = result.decision.get("extra_context")
    if event in _EXTRA_CONTEXT_EVENTS and isinstance(extra, str) and extra:
        decision.extra_context.append(extra)


def _log_hook_failure(hook: HookConfig, result: HookResult) -> None:
    if hook.execution.on_error == "ignore":
        logger.debug("Hook %r failed (on_error=ignore)", hook.id)
        return
    detail = _result_failure_label(result)
    logger.warning("Hook %r failed (%s); stderr=%s", hook.id, detail, result.stderr.strip()[:500])


def _log_nonblocking_spawn_failure(hook: HookConfig, exc: Exception) -> None:
    if hook.execution.on_error == "ignore":
        logger.debug("Non-blocking hook %r could not be spawned", hook.id, exc_info=True)
        return
    logger.warning("Non-blocking hook %r could not be spawned: %s", hook.id, exc, exc_info=True)


def _result_failed(result: HookResult) -> bool:
    return result.timed_out or result.exit_code is None or result.exit_code != 0


def _hook_outcome(result: HookResult, *, blocked: bool) -> str:
    """Closed ``hook.operation.finished.outcome`` for one subprocess result."""
    if result.timed_out:
        return HookOutcome.TIMED_OUT
    if result.exit_code is None:
        return HookOutcome.LAUNCH_ERROR
    if result.exit_code != 0:
        return HookOutcome.FAILED
    if blocked:
        return HookOutcome.BLOCKED
    return HookOutcome.SUCCESS


def _result_failure_label(result: HookResult) -> str:
    if result.timed_out:
        return "timeout"
    if result.exit_code is None:
        return "launch_error"
    return f"exit_code={result.exit_code}"
