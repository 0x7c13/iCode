# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session-wide usage accounting shared by Chat and Workflow execution owners."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from chrys.foundation.events.types import UsageUpdate
from chrys.service.profiles.models.schema import DEFAULT_MAX_CONTEXT_TOKENS

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.service.session.runtime_metadata import SessionUsageMetadata


class UsageSession(Protocol):
    @property
    def session_id(self) -> str | None: ...
    @property
    def runtime_meta(self) -> SessionUsageMetadata: ...


class SessionUsagePublisher:
    """Accumulate invocation spend without assuming a main agent or a context window."""

    def __init__(self, *, bus: EventBus, session: UsageSession) -> None:
        self._bus = bus
        self._session = session
        self._usage_tasks: set[asyncio.Task[None]] = set()
        self._usage_publish_tail: asyncio.Task[None] | None = None

    def make_usage_event(self, *, session_id: str | None = None) -> UsageUpdate:
        identity = session_id or self._session.session_id
        meta = self._session.runtime_meta
        return UsageUpdate(
            session_id=identity,
            usage_source_id=identity or "",
            total_session_tokens=meta.total_session_tokens,
            total_session_input_tokens=meta.total_session_input_tokens,
            total_session_output_tokens=meta.total_session_output_tokens,
            total_session_cache_hit_tokens=meta.total_session_cache_hit_tokens,
        )

    @property
    def tasks(self) -> set[asyncio.Task[None]]:
        """The currently retained usage publications."""
        return self._usage_tasks

    @property
    def tail(self) -> asyncio.Task[None] | None:
        """The last publication in the ordered chain."""
        return self._usage_publish_tail

    def accumulate_invocation_usage(
        self,
        total_tokens: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        local_tokens: int = 0,
        calibration_ratio: float = 1.0,
        system_overhead_tokens: int = 0,
        cache_hit_tokens: int | None = None,
        max_context_tokens: int | None = None,
        agent_profile: str = "",
        usage_source_id: str = "",
        authoritative_total: int | None = None,
        use_local_context_estimate: bool = False,
    ) -> None:
        """Accumulate invocation spend and publish its live context reading.

        ``local_tokens``, ``calibration_ratio``, ``system_overhead_tokens``, and
        ``max_context_tokens`` come from this invocation's ContextManager/model.
        They are copied only onto the live UsageUpdate. The session stores
        cumulative spend; Chat's main-invocation context remains independent.
        """
        self._session.runtime_meta.accumulate_invocation_usage(
            total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=cache_hit_tokens,
            authoritative_total=authoritative_total,
        )
        # Each invocation may use a different model. A context percentage must
        # use that invocation's window, including when it runs alongside a
        # Chat agent with a different model. Use the caller's value or a
        # neutral default when the backend has not reported its window.
        max_ctx = DEFAULT_MAX_CONTEXT_TOKENS if max_context_tokens is None else max_context_tokens
        window_input_tokens = local_tokens if use_local_context_estimate else input_tokens
        window_total_tokens = window_input_tokens + output_tokens if use_local_context_estimate else total_tokens
        pct = round(window_total_tokens / max_ctx * 100, 1) if max_ctx else 0.0
        self.enqueue_usage_event(
            UsageUpdate(
                agent_profile=agent_profile,
                usage_source_id=usage_source_id,
                input_tokens=window_input_tokens,
                output_tokens=output_tokens,
                total_tokens=window_total_tokens,
                pct=pct,
                max_context_tokens=max_ctx,
                total_session_tokens=self._session.runtime_meta.total_session_tokens,
                total_session_input_tokens=self._session.runtime_meta.total_session_input_tokens,
                total_session_output_tokens=self._session.runtime_meta.total_session_output_tokens,
                total_session_cache_hit_tokens=self._session.runtime_meta.total_session_cache_hit_tokens,
                cache_hit_tokens=cache_hit_tokens,
                local_tokens=local_tokens,
                calibration_ratio=calibration_ratio,
                system_overhead_tokens=system_overhead_tokens,
                session_id=self._session.session_id,
            ),
        )

    def accumulate_side_call_usage(
        self,
        usage_details: Mapping[str, Any],
        *,
        agent_profile: str = "",
        usage_source_id: str = "",
        max_context_tokens: int | None = None,
    ) -> None:
        """Callback from Phase-4 LAST_WORDS generators — fold side-call spend into session totals.

        Side-call responses are consumed inside the client and never reach the
        middleware chain, so their provider-reported usage is fed here directly
        — every attempt, including responses later rejected by validation or
        the tool-call guard.  Accumulate-only: the parent's ``last_usage_details``
        (context meter) stays untouched. A child source publishes its own
        side-call window; unscoped callers refresh the main window unchanged.
        """
        from chrys.service.context.middleware.usage import extract_cache_hit_tokens

        input_tokens = int(usage_details.get("input_token_count") or 0)
        output_tokens = int(usage_details.get("output_token_count") or 0)
        total_tokens = int(usage_details.get("total_token_count") or 0)
        if not (input_tokens or output_tokens or total_tokens):
            return
        # Response usage_details carry the kernel-normalized cache field; raw
        # provider keys (what extract_cache_hit_tokens reads) only as a fallback.
        cache_hit_tokens = usage_details.get("cache_read_input_token_count")
        if cache_hit_tokens is None:
            cache_hit_tokens = extract_cache_hit_tokens(dict(usage_details))
        if usage_source_id:
            self.accumulate_invocation_usage(
                total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_hit_tokens=int(cache_hit_tokens) if cache_hit_tokens is not None else None,
                agent_profile=agent_profile,
                usage_source_id=usage_source_id,
                max_context_tokens=max_context_tokens,
            )
            return
        self._session.runtime_meta.accumulate_out_of_band_usage(
            total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=int(cache_hit_tokens) if cache_hit_tokens is not None else None,
        )
        self.enqueue_usage_event(self.make_usage_event())

    def enqueue_usage_event(self, event: UsageUpdate) -> None:
        """Keep usage publish tasks alive and deliver them in creation order."""
        previous = self._usage_publish_tail

        async def _publish_after_previous() -> None:
            if previous is not None:
                # ``gather(return_exceptions=True)`` waits for ``previous`` without
                # inheriting its failure or cancellation, so a cancelled earlier
                # publish cannot cascade-cancel every queued UsageUpdate behind it.
                # ``contextlib.suppress(Exception)`` would not catch CancelledError
                # (BaseException since 3.8), so the chain would otherwise unravel
                # silently on shutdown.
                await asyncio.gather(previous, return_exceptions=True)
            await self._bus.publish(event)

        task = asyncio.create_task(_publish_after_previous())
        self._usage_publish_tail = task
        self.track(task)

    def clear_tail(self, task: asyncio.Task[None]) -> None:
        if self._usage_publish_tail is task:
            self._usage_publish_tail = None

    def track(self, task: asyncio.Task[None]) -> None:
        """Retain a usage publish and observe its completion."""
        self._usage_tasks.add(task)
        task.add_done_callback(self._usage_tasks.discard)
        task.add_done_callback(self.clear_tail)
        task.add_done_callback(_observe_task_exception)

    async def drain(self) -> None:
        """Await all pending UsageUpdate publish tasks.

        Sub-agent ``_invoke`` calls this in its ``finally`` so the sub-agent's
        trailing UsageUpdate is delivered before the parent's ToolCallResult.
        Without it a slow subscriber can let the parent tool result overtake
        the sub-agent's last usage event.
        """
        tail = self._usage_publish_tail
        if tail is None:
            return
        await asyncio.gather(tail, return_exceptions=True)

    async def settle(self) -> None:
        """Finish retained publishes and clear their bookkeeping at shutdown."""
        if self._usage_tasks:
            await asyncio.gather(*tuple(self._usage_tasks), return_exceptions=True)
        self._usage_tasks.clear()
        self._usage_publish_tail = None


def _observe_task_exception(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    with contextlib.suppress(Exception):
        task.exception()
